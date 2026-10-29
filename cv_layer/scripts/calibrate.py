#!/usr/bin/env python
"""Fit temperature scaling on the validation set; report ECE before/after (spec §8.2)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.cli import bootstrap  # noqa: E402
from src.common.config import DEFAULT_COCO, DEFAULT_IMAGES, add_common_args, apply_overrides, load_yaml, resolve  # noqa: E402


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--coco", default=DEFAULT_COCO, help="COCO JSON (default: dataset.coco_path)")
    ap.add_argument("--images", default=DEFAULT_IMAGES, help="image root (default: dataset.image_dir)")
    ap.add_argument("--splits", default="data/splits/splits.json")
    ap.add_argument("--checkpoint", default="runs/classifier/best.pt")
    ap.add_argument("--classifier-config", default="configs/classifier.yaml")
    ap.add_argument("--out", default="runs/classifier/calibration.json")
    a = ap.parse_args(argv)
    ccfg = apply_overrides(load_yaml(a.classifier_config), a.overrides)
    bootstrap(a.seed if a.seed is not None else ccfg["train"]["seed"])
    import torch
    from torch.utils.data import DataLoader
    from src.classification.calibration import calibrate, save_calibration
    from src.classification.dataset import CropDataset, load_samples
    from src.classification.model import StinkBugClassifier, load_checkpoint
    from src.classification.train import predict

    ck = load_checkpoint(resolve(a.checkpoint))
    cfg = ck["config"]
    cfg["model"]["life_stage_head"] = ck["state_dict"].get("stage_head.weight") is not None
    model = StinkBugClassifier.from_config(cfg, pretrained=False)
    model.load_state_dict(ck["state_dict"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    splits = json.loads(resolve(a.splits).read_text())
    pad = load_yaml("configs/detector.yaml")["postprocess"]["crop_padding"]
    va = load_samples(a.coco, set(splits["val"]), cfg["classes"]["species"], cfg["classes"]["life_stages"], a.images)
    dl = DataLoader(CropDataset(va, cfg["model"]["input_size"], pad), batch_size=cfg["train"]["batch_size"],
                    num_workers=cfg["train"]["num_workers"])
    pred = predict(model, dl, dev)
    rec = calibrate(pred["species_logits"], pred["y"], ccfg["calibration"]["n_bins"], ccfg["calibration"]["max_iter"])
    rec["target_met"] = rec["ece_after"] <= ccfg["calibration"]["ece_target"]
    save_calibration(rec, resolve(a.out))
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
