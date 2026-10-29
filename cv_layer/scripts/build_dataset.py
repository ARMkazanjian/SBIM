#!/usr/bin/env python
"""Index a flat image folder into the COCO file every other script reads (data/annotations/all.json).

Steps: (1) index images, derive session/domain from filename rules; (2) attach image-level species labels from
labels_csv / label_rules / --default-species (never guessed); (3) optionally --auto-annotate: Grounding DINO proposes
boxes for labelled images. Auto boxes are flagged auto_generated=true, human_reviewed=false (spec §11.4) — review them
(e.g. import into Roboflow) and set human_reviewed=true before treating any metric as evidence.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.cli import bootstrap  # noqa: E402
from src.common.config import add_common_args, load_config, load_yaml, resolve  # noqa: E402


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--image-dir", default=None, help="default: dataset.image_dir in pipeline.yaml")
    ap.add_argument("--out", default=None, help="default: dataset.coco_path")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--classifier-config", default="configs/classifier.yaml")
    ap.add_argument("--detector-config", default="configs/detector.yaml")
    ap.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    ap.add_argument("--default-species", default=None, help="label EVERY unlabelled image with this species (only if true)")
    ap.add_argument("--auto-annotate", action="store_true", help="propose boxes with Grounding DINO (GPU recommended)")
    ap.add_argument("--limit", type=int, default=None, help="only auto-annotate the first N images (smoke test)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    bootstrap(a.seed if a.seed is not None else cfg["seed"], cfg["logging"]["level"])
    from src.data_prep.folder import image_species, index_folder

    d = cfg["dataset"]
    prep = load_yaml(a.preprocessing_config)
    species = load_yaml(a.classifier_config)["classes"]["species"]
    image_dir = a.image_dir or d["image_dir"]
    coco, rejected = index_folder(image_dir, prep["ingestion"]["extensions"], d["folder"],
                                  prep["ingestion"]["min_side_px"], species)
    labels = image_species(coco, d["folder"], a.default_species)
    cat_id = {c["name"]: c["id"] for c in coco["categories"]}
    for im in coco["images"]:
        im["species_label"] = labels.get(im["id"], {}).get("species")

    if a.auto_annotate:
        from src.detection.grounding_dino import GroundingDinoDetector
        from src.detection.run import detect_original_coords
        from src.ingestion.loader import load_image

        dcfg = load_config(a.detector_config)
        det = GroundingDinoDetector.from_config(dcfg)
        th, rev = dcfg["thresholds"], (dcfg["model"]["revision"] or "unpinned")[:8]
        todo = [im for im in coco["images"] if im["id"] in labels][: a.limit]
        if not todo:
            sys.exit("--auto-annotate needs species labels (labels_csv, label_rules or --default-species); "
                     "none of the images has one")
        for n, im in enumerate(todo, 1):
            li = load_image(Path(image_dir) / im["file_name"], prep["ingestion"]["min_side_px"])
            b, s, *_ = detect_original_coords(det, li, dcfg["prompt"]["selected"], th["box_threshold"],
                                              th["text_threshold"], prep, dcfg["postprocess"])
            lab = labels[im["id"]]
            for box, score in zip(b, s):
                coco["annotations"].append({
                    "id": len(coco["annotations"]) + 1, "image_id": im["id"], "category_id": cat_id[lab["species"]],
                    "bbox": [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])],
                    "life_stage": lab["life_stage"], "score": float(score), "annotator_id": f"groundingdino:{rev}",
                    "auto_generated": True, "human_reviewed": False, "species_verified_by_entomologist": False})
            if n % 100 == 0:
                print(f"auto-annotated {n}/{len(todo)}", file=sys.stderr)

    out = resolve(a.out or d["coco_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(coco))
    dom = {}
    for im in coco["images"]:
        dom[im["domain"]] = dom.get(im["domain"], 0) + 1
    print(json.dumps({"coco": str(out), "images": len(coco["images"]), "rejected": len(rejected), "domains": dom,
                      "sessions": len({i["session_id"] for i in coco["images"]}),
                      "images_with_species_label": len(labels), "annotations": len(coco["annotations"])}, indent=2))
    if not labels:
        print("NOTE: no species labels attached (set dataset.folder.labels_csv / label_rules or pass --default-species)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
