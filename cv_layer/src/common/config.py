"""YAML config loading with dotted CLI overrides. All tunables live in YAML (spec §13.2)."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_yaml(path: str | Path) -> dict:
    with open(resolve(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_value(raw: str) -> Any:
    val = yaml.safe_load(raw)
    if isinstance(val, str):          # PyYAML (YAML 1.1) reads "3e-3" as a string; accept it as a number
        try:
            return float(val)
        except ValueError:
            pass
    return val


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply ``a.b.c=value`` overrides. Values are parsed as YAML scalars/lists."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override must look like key.sub=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    return apply_overrides(load_yaml(path), overrides)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override a config value (dotted key), repeatable")
    parser.add_argument("--seed", type=int, default=None, help="override the global seed")
    return parser


def _dataset_defaults() -> tuple[str, str]:
    try:
        d = load_yaml("configs/pipeline.yaml")["dataset"]
        return str(resolve(d["coco_path"])), d["image_dir"]
    except (OSError, KeyError):
        return str(ROOT / "data/annotations/all.json"), str(ROOT / "data/raw")


DEFAULT_COCO, DEFAULT_IMAGES = _dataset_defaults()
