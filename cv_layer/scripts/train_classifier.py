#!/usr/bin/env python
"""Train the species classifier (spec §7): two-phase, hard-negative rounds, backbone / resolution benchmarks.

Hard-negative workflow (max 2 rounds, spec §7.4):
  round 0:  train_classifier.py ...                       -> best.pt, then mines confusion-pair errors on train
                                                            and writes <out>/hard_negative_review_round1.csv
  (manual): reviewer fills the `verdict` column (label_correct | label_wrong)
  round 1:  train_classifier.py ... --round 1 --verified-csv <that csv>   (verified cases oversampled 3x)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.cli import bootstrap  # noqa: E402
from src.common.config import DEFAULT_COCO, DEFAULT_IMAGES, add_common_args, apply_overrides, load_yaml, resolve  # noqa: E402


def run_one(ccfg, pcfg_prep, coco_path, splits, images, out_dir, args, tag):
    import torch
    from torch.utils.data import DataLoader
    from src.classification.dataset import CropDataset, load_samples
    from src.classification.hard_negatives import (label_noise_rate, mine_confusions, oversample, read_verified,
                                                   write_review_sample)
    from src.classification.model import StinkBugClassifier, load_checkpoint
    from src.classification.train import predict, train_classifier
    from src.common.tracking import Tracker
    from src.preprocessing.augment import build_train_augmentation

    cls, stg = ccfg["classes"]["species"], ccfg["classes"]["life_stages"]
    pad = load_yaml("configs/detector.yaml")["postprocess"]["crop_padding"]
    tr = load_samples(coco_path, set(splits["train"]), cls, stg, images)
    va = load_samples(coco_path, set(splits["val"]), cls, stg, images)
    if not tr or not va:
        sys.exit("empty train or validation crop set — check COCO categories against classifier.yaml")
    if args.verified_csv:
        n = oversample(tr, read_verified(args.verified_csv), ccfg["hard_negative"]["oversample_factor"])
        print(f"oversampled {n} verified hard cases x{ccfg['hard_negative']['oversample_factor']}; "
              f"label-noise rate in reviewed sample: {label_noise_rate(args.verified_csv)}")
    aug = build_train_augmentation(pcfg_prep["augmentation"], ccfg["model"]["input_size"])
    tracker = Tracker(ccfg["tracking"]["backend"], out_dir, ccfg["tracking"]["project"], ccfg, tag)
    res = train_classifier(ccfg, pcfg_prep, tr, va, tracker, out_dir, pad, aug, full_config=ccfg)
    tracker.close()

    if args.mine:   # mine on the TRAIN set with the best checkpoint
        ck = load_checkpoint(out_dir / "best.pt")
        cfg2 = json.loads(json.dumps(ccfg))
        cfg2["model"]["life_stage_head"] = ck["state_dict"].get("stage_head.weight") is not None
        model = StinkBugClassifier.from_config(cfg2, pretrained=False)
        model.load_state_dict(ck["state_dict"])
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(dev)
        dl = DataLoader(CropDataset(tr, ccfg["model"]["input_size"], pad), batch_size=ccfg["train"]["batch_size"],
                        num_workers=ccfg["train"]["num_workers"])
        pred = predict(model, dl, dev)
        mined = mine_confusions(tr, pred["species_logits"], cls, ccfg["classes"]["confusion_pairs"])
        rnd = args.round + 1
        if rnd > ccfg["hard_negative"]["max_rounds"]:
            print("hard-negative round limit reached; not mining further (overfits the hard set)")
        else:
            csv_path = out_dir / f"hard_negative_review_round{rnd}.csv"
            write_review_sample(mined, csv_path, ccfg["hard_negative"]["review_sample_size"], ccfg["train"]["seed"])
            print(f"{len(mined)} confusion-pair errors mined; review sample -> {csv_path}")
    return res


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--coco", default=DEFAULT_COCO, help="COCO JSON (default: dataset.coco_path)")
    ap.add_argument("--images", default=DEFAULT_IMAGES, help="image root (default: dataset.image_dir)")
    ap.add_argument("--splits", default="data/splits/splits.json")
    ap.add_argument("--classifier-config", default="configs/classifier.yaml")
    ap.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    ap.add_argument("--out", default=None, help="output dir (default: train.output_dir)")
    ap.add_argument("--round", type=int, default=0, help="hard-negative round index (0 = first full run)")
    ap.add_argument("--verified-csv", default=None, help="reviewed hard-negative CSV from the previous round")
    ap.add_argument("--mine", action="store_true", help="after training, mine confusion-pair errors on train")
    ap.add_argument("--benchmark-backbones", action="store_true", help="train each backbone; select on val macro-F1")
    ap.add_argument("--resolution-ablation", action="store_true", help="train at 224/320/384; compare val macro-F1")
    a = ap.parse_args(argv)
    ccfg = apply_overrides(load_yaml(a.classifier_config), a.overrides)
    bootstrap(a.seed if a.seed is not None else ccfg["train"]["seed"])
    prep = load_yaml(a.preprocessing_config)
    splits = json.loads(resolve(a.splits).read_text())
    out = resolve(a.out or ccfg["train"]["output_dir"])
    if a.round > ccfg["hard_negative"]["max_rounds"]:
        sys.exit(f"--round {a.round} exceeds hard_negative.max_rounds")

    variants = [(None, ccfg)]
    if a.benchmark_backbones:
        variants = []
        for b in ccfg["benchmark_backbones"]:
            c = json.loads(json.dumps(ccfg))
            c["model"].update(backbone=b["backbone"], name=b["name"], input_size=b["input_size"])
            variants.append((b["name"], c))
    elif a.resolution_ablation:
        variants = []
        for s in ccfg["resolution_ablation"]:
            c = json.loads(json.dumps(ccfg))
            c["model"]["input_size"] = s
            c["model"]["name"] = f"{ccfg['model']['backbone']}_{s}"
            variants.append((c["model"]["name"], c))

    summary = {}
    for tag, c in variants:
        d = out / tag if tag else out
        r = run_one(c, prep, a.coco, splits, a.images, d, a, tag or "run")
        summary[tag or "run"] = {"best_val_macro_f1": r["best_val_macro_f1"], "best_epoch": r["best_epoch"],
                                 "temperature": r["calibration"]["temperature"],
                                 "life_stage_enabled": r["life_stage_enabled"]}
    if len(summary) > 1:
        best = max(summary, key=lambda k: summary[k]["best_val_macro_f1"])   # macro-F1, not accuracy
        summary["_selected"] = best
        (out / "benchmark.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
