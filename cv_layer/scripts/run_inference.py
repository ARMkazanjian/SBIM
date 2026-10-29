#!/usr/bin/env python
"""Run the full CV pipeline on an image or directory; write one validated JSON packet per image."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.cli import bootstrap  # noqa: E402
from src.common.config import add_common_args, load_config, resolve  # noqa: E402


def main(argv=None):
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("input", help="image file or directory")
    ap.add_argument("--out", default="runs/packets")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--pixels-per-cm", type=float, default=None, help="reference scale; enables density")
    ap.add_argument("--recursive", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.overrides)
    bootstrap(a.seed if a.seed is not None else cfg["seed"], cfg["logging"]["level"])
    from src.ingestion.loader import IngestionError, iter_directory
    from src.output.serialize import write_packet
    from src.pipeline import CVPipeline

    pipe = CVPipeline.from_configs(a.config, a.overrides)
    src = Path(a.input)
    out = resolve(a.out)
    failures = []
    if src.is_dir():
        items = iter_directory(src, pipe.prep["ingestion"]["extensions"], pipe.prep["ingestion"]["min_side_px"],
                               a.recursive)
    else:
        items = []
        try:
            from src.ingestion.loader import load_image
            items = [load_image(src, pipe.prep["ingestion"]["min_side_px"])]
        except IngestionError as e:
            failures.append((str(src), str(e)))
    n = 0
    for it in items:
        if isinstance(it, tuple):
            failures.append((str(it[0]), it[1]))
            continue
        packet = pipe.run(it, a.pixels_per_cm)
        write_packet(packet, out / f"{it.image_id}.json")
        n += 1
    print(json.dumps({"processed": n, "rejected": [{"path": p, "reason": r} for p, r in failures], "out": str(out)}))
    if failures and n == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
