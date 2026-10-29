"""COCO -> per-image ground truth for detection / end-to-end evaluation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class GTImage:
    image_id: int
    path: str
    domain: str
    boxes: np.ndarray                       # (N,4) xyxy, coordinates of the EXIF-corrected image
    species: list[str] = field(default_factory=list)
    empty_background: bool = False          # adversarial set member with no stink bugs


def load_gt(coco: dict, image_ids: set | list | None, root: str | Path) -> list[GTImage]:
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    per = defaultdict(list)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        per[a["image_id"]].append(([x, y, x + w, y + h], cats[a["category_id"]]))
    keep = set(image_ids) if image_ids is not None else None
    out = []
    for im in coco["images"]:
        if keep is not None and im["id"] not in keep:
            continue
        ann = per.get(im["id"], [])
        out.append(GTImage(im["id"], str(Path(root) / im["file_name"]), im.get("domain", "unknown"),
                           np.array([a[0] for a in ann], dtype=float).reshape(-1, 4), [a[1] for a in ann],
                           bool(im.get("empty_background", False)) and not ann))
    return out
