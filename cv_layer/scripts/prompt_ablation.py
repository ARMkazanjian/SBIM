#!/usr/bin/env python
"""Evaluate every candidate Grounding DINO prompt on the validation set; select by recall (spec §5.2)."""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.cli import bootstrap  # noqa: E402
from src.common.config import DEFAULT_COCO, DEFAULT_IMAGES, add_common_args, load_config, resolve  # noqa: E402


def run_ablation(backend, gts, dcfg, prep, prompts=None, box_thr=None, text_thr=None):
    """gts: list[GTImage]. Returns list of result dicts (one per prompt) sorted by recall, best first."""
    from src.detection.prompts import validate_prompt
    from src.detection.run import detect_original_coords
    from src.evaluation.detection import evaluate_detection
    from src.ingestion.loader import load_image

    th = dcfg["thresholds"]
    box_thr, text_thr = box_thr or th["box_threshold"], text_thr or th["text_threshold"]
    images = {g.image_id: load_image(g.path, prep["ingestion"]["min_side_px"]) for g in gts}
    results = []
    for p in prompts or dcfg["prompt"]["candidates"]:
        validate_prompt(p)
        preds, lat = {}, []
        for g in gts:
            b, s, _, ms, _ = detect_original_coords(backend, images[g.image_id], p, box_thr, text_thr, prep,
                                                    dcfg["postprocess"])
            preds[g.image_id] = (b, s)
            lat.append(ms)
        m = evaluate_detection({g.image_id: g.boxes for g in gts}, preds, dcfg["decision_gate"]["eval_iou"], lat)
        results.append({"prompt": p, "recall": m["recall"], "precision": m["precision"], "mean_iou": m["mean_iou"],
                        "f1": m["f1"], "latency_ms_mean": m.get("latency_ms_mean")})
    return sorted(results, key=lambda r: (-r["recall"], -r["precision"]))


def write_selection(config_path: Path, winner: str, scores: dict) -> None:
    """Update ``selected`` and ``ablation_scores`` in detector.yaml in place (comments preserved)."""
    txt = config_path.read_text()
    txt = re.sub(r'(?m)^(  selected: ).*$', lambda m: m.group(1) + json.dumps(winner), txt)
    txt = re.sub(r'(?m)^(  ablation_scores: ).*$', lambda m: m.group(1) + json.dumps(scores), txt)
    config_path.write_text(txt)


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--coco", default=DEFAULT_COCO, help="COCO JSON (default: dataset.coco_path)")
    ap.add_argument("--images", default=DEFAULT_IMAGES, help="image root (default: dataset.image_dir)")
    ap.add_argument("--splits", default="data/splits/splits.json")
    ap.add_argument("--split", default="val", choices=["train", "val"], help="never 'test' (spec §5.4)")
    ap.add_argument("--detector-config", default="configs/detector.yaml")
    ap.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    ap.add_argument("--out", default="runs/prompt_ablation.json")
    ap.add_argument("--write-config", action="store_true", help="record the winner in detector.yaml")
    a = ap.parse_args(argv)
    bootstrap(a.seed if a.seed is not None else 42)
    from src.data_prep.detection_data import load_gt
    from src.data_prep.splits import load_coco
    from src.detection.grounding_dino import GroundingDinoDetector

    dcfg = load_config(a.detector_config, a.overrides)
    prep = load_config(a.preprocessing_config)
    coco = load_coco(a.coco)
    ids = set(json.loads(resolve(a.splits).read_text())[a.split])
    gts = load_gt(coco, ids, a.images)
    res = run_ablation(GroundingDinoDetector.from_config(dcfg), gts, dcfg, prep)
    out = resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": a.split, "selected_by": "recall", "results": res}, indent=2))
    for r in res:
        print(f"recall={r['recall']:.3f} precision={r['precision']:.3f} iou={r['mean_iou']:.3f}  {r['prompt']!r}")
    if a.write_config:
        write_selection(resolve(a.detector_config), res[0]["prompt"], {r["prompt"]: round(r["recall"], 4) for r in res})


if __name__ == "__main__":
    main()
