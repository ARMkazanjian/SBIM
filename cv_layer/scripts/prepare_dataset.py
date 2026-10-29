#!/usr/bin/env python
"""Validate COCO annotations, split by capture session (stratified by species x domain), verify no leakage."""
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
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--pipeline-config", default="configs/pipeline.yaml")
    ap.add_argument("--classifier-config", default="configs/classifier.yaml")
    ap.add_argument("--allow-unreviewed", action="store_true",
                    help="downgrade annotation-review problems (annotator id, human review, entomologist) to warnings")
    ap.add_argument("--strict", action="store_true", help="exit non-zero on field-fraction warnings too")
    a = ap.parse_args(argv)
    cfg = apply_overrides(load_yaml(a.pipeline_config), a.overrides)
    bootstrap(a.seed if a.seed is not None else cfg["dataset"]["seed"])
    from src.data_prep.coco_check import validate_coco
    from src.data_prep.splits import (annotator_agreement, domain_report, field_only_test, load_coco,
                                      split_dataset, verify_no_leakage)

    coco = load_coco(a.coco)
    species = load_yaml(a.classifier_config)["classes"]["species"]
    problems = validate_coco(coco, species)
    if problems:
        print(f"{len(problems)} annotation problems (first 20):", *problems[:20], sep="\n  ")
        if not a.allow_unreviewed:
            sys.exit(2)
    d = cfg["dataset"]
    splits = split_dataset(coco, d["split"], a.seed if a.seed is not None else d["seed"],
                           d["min_field_fraction_test"], d["min_field_fraction_train"])
    leaks = verify_no_leakage(coco, splits)
    if leaks:
        print("LEAKAGE:", *leaks, sep="\n  ")
        sys.exit(3)
    rep = domain_report(coco, splits, d["min_field_fraction_train"], d["min_field_fraction_test"])
    rep["annotator_agreement"] = annotator_agreement(coco)
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "splits.json").write_text(json.dumps(splits))
    (out / "field_test.json").write_text(json.dumps(field_only_test(coco, splits)))
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    if rep["warnings"] and a.strict:
        sys.exit(4)


if __name__ == "__main__":
    main()
