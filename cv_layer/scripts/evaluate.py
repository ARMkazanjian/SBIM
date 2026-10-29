#!/usr/bin/env python
"""Full evaluation (spec §12): detection, classification, end-to-end, adversarial FPR, acceptance check.

Ablation switches: --prompt (§5.2), --tiling {auto,on,off} (§12.5 #6), --set <key>=<val> for thresholds / backbone,
--baseline-predictions for the single-stage YOLOv8m+CBAM+ASFF comparison (§12.5 #7; COCO-results JSON produced by
the baseline's own run on the same split: [{"image_id", "bbox":[x,y,w,h], "score"}]).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.cli import bootstrap  # noqa: E402
from src.common.config import DEFAULT_COCO, DEFAULT_IMAGES, add_common_args, load_config, resolve  # noqa: E402


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--coco", default=DEFAULT_COCO, help="COCO JSON (default: dataset.coco_path)")
    ap.add_argument("--images", default=DEFAULT_IMAGES, help="image root (default: dataset.image_dir)")
    ap.add_argument("--splits", default="data/splits/splits.json")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--domain", default="field", choices=["field", "lab", "all"],
                    help="acceptance criteria apply to the field-only test set")
    ap.add_argument("--adversarial-dir", default=None, help="empty-background images (canopy, soil, litter, other insects)")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--tiling", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--baseline-predictions", default=None)
    ap.add_argument("--out", default="runs/evaluation.json")
    a = ap.parse_args(argv)
    pcfg = load_config(a.config, a.overrides)
    bootstrap(a.seed if a.seed is not None else pcfg["seed"], pcfg["logging"]["level"])
    import numpy as np
    from src.data_prep.detection_data import load_gt
    from src.data_prep.splits import load_coco
    from src.evaluation import (check_acceptance, classification_report, count_metrics, empty_image_fpr,
                                evaluate_detection, species_count_mae, uncertain_rate)
    from src.ingestion.loader import iter_directory, load_image
    from src.pipeline import CVPipeline

    pipe = CVPipeline.from_configs(a.config, a.overrides)
    if a.prompt:
        from src.detection.prompts import validate_prompt
        pipe.prompt = validate_prompt(a.prompt)
    if a.tiling != "auto":
        pipe.prep["tiling"]["enabled"] = a.tiling == "on"
    ccfg = pipe.ccfg
    classes = ccfg["classes"]["species"]
    coco = load_coco(a.coco)
    ids = set(json.loads(resolve(a.splits).read_text())[a.split])
    gts = [g for g in load_gt(coco, ids, a.images) if a.domain == "all" or g.domain == a.domain]
    if not gts:
        sys.exit(f"no {a.domain} images in split {a.split}")

    det_gt, det_pred, lat, packets = {}, {}, [], {}
    for g in gts:
        pk = pipe.run(load_image(g.path, pipe.prep["ingestion"]["min_side_px"]))
        packets[g.image_id] = pk
        det_gt[g.image_id] = g.boxes
        det_pred[g.image_id] = (np.array([d["bbox"] for d in pk["detections"]], float).reshape(-1, 4),
                                np.array([d["detection_confidence"] for d in pk["detections"]]))
        lat.append(pk["processing"]["inference_ms"]["total"])
    iou = pipe.dcfg["decision_gate"]["eval_iou"]
    det = evaluate_detection(det_gt, det_pred, iou, lat)

    # classification: classify GT crops directly (isolates Stage 2 from Stage 1 errors)
    from src.detection.postprocess import extract_crop
    pad, size = pipe.dcfg["postprocess"]["crop_padding"], ccfg["model"]["input_size"]
    ys, probs = [], []
    for g in gts:
        if not len(g.boxes):
            continue
        img = load_image(g.path, 0).image
        keep = [i for i, s in enumerate(g.species) if s in classes]
        if not keep:
            continue
        crops = [extract_crop(img, tuple(g.boxes[i]), pad, size) for i in keep]
        p, _ = pipe.classifier.classify(crops)
        probs.append(p)
        ys += [classes.index(g.species[i]) for i in keep]
    cls_rep = None
    if ys:
        P = np.concatenate(probs)
        cls_rep = classification_report(np.array(ys), P, classes, ccfg["classes"]["confusion_pairs"], ["uncertain"])
        from src.classification.calibration import expected_calibration_error
        cls_rep["ece"] = expected_calibration_error(P, np.array(ys), ccfg["calibration"]["n_bins"])

    # end-to-end counts
    pred_counts = [packets[g.image_id]["aggregates"]["total_count"] for g in gts]
    true_counts = [len(g.boxes) for g in gts]
    e2e = count_metrics(pred_counts, true_counts)
    e2e["species_count_mae"] = species_count_mae(
        [packets[g.image_id]["aggregates"]["count_by_species"] for g in gts],
        [{s: g.species.count(s) for s in set(g.species)} for g in gts], classes)
    e2e["uncertain_rate"] = uncertain_rate([d["species"] for pk in packets.values() for d in pk["detections"]])

    adv = None
    if a.adversarial_dir:
        counts = []
        for it in iter_directory(a.adversarial_dir, pipe.prep["ingestion"]["extensions"],
                                 pipe.prep["ingestion"]["min_side_px"], recursive=True):
            if not isinstance(it, tuple):
                counts.append(pipe.run(it)["aggregates"]["total_count"])
        adv = empty_image_fpr(counts)

    baseline = None
    if a.baseline_predictions:
        raw = json.loads(Path(a.baseline_predictions).read_text())
        bp: dict = {g.image_id: ([], []) for g in gts}
        for r in raw:
            if r["image_id"] in bp:
                x, y, w, h = r["bbox"]
                bp[r["image_id"]][0].append([x, y, x + w, y + h])
                bp[r["image_id"]][1].append(r["score"])
        baseline = evaluate_detection(det_gt, {k: (np.array(v[0], float).reshape(-1, 4), np.array(v[1])) for k, v in bp.items()}, iou)

    pair_acc = {k: v["accuracy"] for k, v in (cls_rep or {}).get("confusion_pairs", {}).items()}
    acceptance = check_acceptance({
        "detection_recall": det["recall"], "detection_precision": det["precision"], "count_mae": e2e["count_mae"],
        "macro_f1": (cls_rep or {}).get("macro_f1"), "pair_accuracy": pair_acc,
        "empty_fpr": adv["fpr"] if adv else None, "ece": (cls_rep or {}).get("ece"),
        "total_inference_ms": float(np.max(lat)) if lat else None}, pcfg["acceptance"])
    report = {"split": a.split, "domain": a.domain, "n_images": len(gts), "detection": det, "classification": cls_rep,
              "end_to_end": e2e, "adversarial": adv, "baseline_detection": baseline, "acceptance": acceptance,
              "prompt": pipe.prompt}
    out = resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({"acceptance": acceptance, "detection": det}, indent=2, default=float))


if __name__ == "__main__":
    main()
