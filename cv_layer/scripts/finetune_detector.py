#!/usr/bin/env python
"""Spec §5.3 decision gate (explicit checkpoint) followed, only if warranted, by head-only fine-tuning.

1. Zero-shot evaluation on the FIELD validation split -> recall.
2. Branch: >=0.85 ship zero-shot | 0.70-0.85 fine-tune head | <0.70 fall back to YOLOv8m+CBAM+ASFF.
3. The metrics and chosen branch are logged and written to the decision record before anything else happens.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.cli import bootstrap  # noqa: E402
from src.common.config import DEFAULT_COCO, DEFAULT_IMAGES, add_common_args, load_config, resolve  # noqa: E402


def eval_recall(backend, gts, dcfg, prep):
    from src.detection.run import detect_original_coords
    from src.evaluation.detection import evaluate_detection
    from src.ingestion.loader import load_image

    th = dcfg["thresholds"]
    preds = {}
    for g in gts:
        b, s, *_ = detect_original_coords(backend, load_image(g.path, prep["ingestion"]["min_side_px"]),
                                          dcfg["prompt"]["selected"], th["box_threshold"], th["text_threshold"],
                                          prep, dcfg["postprocess"])
        preds[g.image_id] = (b, s)
    return evaluate_detection({g.image_id: g.boxes for g in gts}, preds, dcfg["decision_gate"]["eval_iou"])


def finetune_head(detector, train_gts, val_gts, dcfg, prep, out_dir: Path):
    """Freeze Swin backbone + BERT text encoder; train the rest (decoder + detection heads) at lr 1e-5 with early
    stopping on validation recall. Targets follow the HF GroundingDino loss format: one class (0) per box,
    boxes as normalized cxcywh."""
    import torch
    from PIL import Image
    from src.ingestion.loader import load_image

    fcfg = dcfg["finetune"]
    model, proc, dev = detector.model, detector.processor, detector.device
    for n, p in model.named_parameters():
        p.requires_grad = not any(n.startswith(f"model.{k}") for k in ("backbone", "text_backbone"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=fcfg["lr"])
    best, bad = -1.0, 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(fcfg["epochs"]):
        model.train()
        for g in train_gts:
            li = load_image(g.path, prep["ingestion"]["min_side_px"])
            w, h = li.image.size
            if not len(g.boxes):
                continue
            bx = g.boxes
            cxcywh = torch.tensor([[(b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h, (b[2] - b[0]) / w, (b[3] - b[1]) / h]
                                   for b in bx], dtype=torch.float32)
            inputs = proc(images=li.image, text=dcfg["prompt"]["selected"], return_tensors="pt").to(dev)
            labels = [{"class_labels": torch.zeros(len(bx), dtype=torch.long, device=dev), "boxes": cxcywh.to(dev)}]
            loss = model(**inputs, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
        model.eval()
        rec = eval_recall(detector, val_gts, dcfg, prep)["recall"]
        print(json.dumps({"epoch": epoch, "val_recall": rec}))
        if rec > best:
            best, bad = rec, 0
            model.save_pretrained(out_dir / "best")
        else:
            bad += 1
            if bad >= fcfg["patience"]:
                break
    return best


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--coco", default=DEFAULT_COCO, help="COCO JSON (default: dataset.coco_path)")
    ap.add_argument("--images", default=DEFAULT_IMAGES, help="image root (default: dataset.image_dir)")
    ap.add_argument("--splits", default="data/splits/splits.json")
    ap.add_argument("--detector-config", default="configs/detector.yaml")
    ap.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    ap.add_argument("--decision-out", default="runs/detector_decision.json")
    ap.add_argument("--evaluate-only", action="store_true", help="stop after recording the decision")
    a = ap.parse_args(argv)
    bootstrap(a.seed if a.seed is not None else 42)
    from src.data_prep.detection_data import load_gt
    from src.data_prep.splits import load_coco
    from src.detection.grounding_dino import GroundingDinoDetector
    from src.evaluation.detection import decision_gate, record_decision

    dcfg = load_config(a.detector_config, a.overrides)
    prep = load_config(a.preprocessing_config)
    coco = load_coco(a.coco)
    splits = json.loads(resolve(a.splits).read_text())
    det = GroundingDinoDetector.from_config(dcfg)
    val_field = [g for g in load_gt(coco, set(splits["val"]), a.images) if g.domain == "field"]
    if not val_field:
        sys.exit("no field-domain images in the validation split; the §5.3 gate must be evaluated on field imagery")
    metrics = eval_recall(det, val_field, dcfg, prep)
    decision = decision_gate(metrics["recall"], dcfg["decision_gate"])
    path = record_decision(decision, resolve(a.decision_out), {"zero_shot_metrics": metrics,
                                                               "prompt": dcfg["prompt"]["selected"],
                                                               "thresholds": dcfg["thresholds"]["box_threshold"]})
    print(json.dumps({"DECISION": decision, "metrics": metrics, "record": str(path)}, indent=2))
    if a.evaluate_only or decision["branch"] != "finetune_head":
        return
    train = [g for g in load_gt(coco, set(splits["train"]), a.images)]
    best = finetune_head(det, train, val_field, dcfg, prep, resolve(dcfg["finetune"]["output_dir"]))
    record_decision({**decision, "finetuned_val_recall": best}, resolve(a.decision_out),
                    {"zero_shot_metrics": metrics, "note": "fine-tune completed"})


if __name__ == "__main__":
    main()
