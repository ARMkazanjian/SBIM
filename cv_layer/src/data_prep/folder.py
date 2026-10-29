"""Build a COCO skeleton from a flat folder of images (Roboflow-style names, any mix of formats)."""
from __future__ import annotations

import csv
import re
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from ..common.logging import get_logger

log = get_logger(__name__)
_ROBOFLOW_SUFFIX = re.compile(r"_(?:jpe?g|png|webp|heic)\.rf\..*$", re.IGNORECASE)


def source_stem(file_name: str) -> str:
    """Original file stem with the Roboflow ``_jpg.rf.<hash>`` export suffix removed."""
    name = Path(file_name).name
    stripped = _ROBOFLOW_SUFFIX.sub("", name)
    return stripped if stripped != name else Path(name).stem


def first_rule(rules: list[dict], text: str, default=None):
    """First ``{pattern, value}`` rule whose regex matches ``text``; ``value`` may use \\1-style groups."""
    for r in rules or []:
        m = re.search(r["pattern"], text)
        if m:
            return m.expand(str(r["value"]))
    return default


def index_folder(directory: str | Path, extensions: list[str], folder_cfg: dict, min_side: int,
                 species: list[str]) -> tuple[dict, list[tuple[str, str]]]:
    """Returns (coco with images only, rejected [(file, reason)]). Domain/session come from config rules and are
    recorded with ``domain_source`` so a reviewer can tell inferred metadata from verified metadata."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"image directory not found: {directory}")
    exts = {e.lower() for e in extensions}
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts)
    images, rejected = [], []
    for p in files:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except (UnidentifiedImageError, OSError, ValueError) as e:
            rejected.append((p.name, f"unreadable: {e}"))
            continue
        if min(w, h) < min_side:
            rejected.append((p.name, f"{w}x{h} under minimum {min_side}px"))
            continue
        stem = source_stem(p.name)
        images.append({
            "id": len(images) + 1, "file_name": p.name, "width": w, "height": h, "source_id": stem,
            "session_id": first_rule(folder_cfg.get("session_rules"), stem, stem),
            "domain": first_rule(folder_cfg.get("domain_rules"), stem, folder_cfg.get("default_domain", "unknown")),
            "domain_source": "filename_rule"})
    for f, why in rejected:
        log.error("image_rejected", extra={"path": f, "reason": why})
    cats = [{"id": i + 1, "name": s} for i, s in enumerate(species)]
    return {"images": images, "annotations": [], "categories": cats}, rejected


def image_species(coco: dict, folder_cfg: dict, default_species: str | None = None) -> dict[int, dict]:
    """Image-level labels from labels_csv, then label_rules, then ``default_species``. Never guessed otherwise.
    Returns {image_id: {"species":..., "life_stage":...}} only for labelled images."""
    csv_rows: dict[str, dict] = {}
    if folder_cfg.get("labels_csv"):
        with open(folder_cfg["labels_csv"], newline="") as f:
            csv_rows = {r["file_name"]: r for r in csv.DictReader(f)}
    names = {c["name"] for c in coco["categories"]}
    out = {}
    for im in coco["images"]:
        row = csv_rows.get(im["file_name"])
        sp = (row or {}).get("species") or first_rule(folder_cfg.get("label_rules"), im["file_name"]) or default_species
        if not sp:
            continue
        if sp not in names:
            raise ValueError(f"{im['file_name']}: species {sp!r} is not one of {sorted(names)}")
        out[im["id"]] = {"species": sp, "life_stage": (row or {}).get("life_stage") or None}
    return out
