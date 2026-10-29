"""Stage 1 orchestration: whole-image or tiled detection over any DetectorBackend (spec §4.3)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from ..preprocessing.tiling import merge_tile_boxes, offset_boxes, should_tile, tile_windows
from .types import DetectorBackend


def detect_whole(backend: DetectorBackend, image: Image.Image, prompt: str, box_thr: float, text_thr: float):
    return backend.detect(image, prompt, box_thr, text_thr)


def detect_tiled(backend: DetectorBackend, image: Image.Image, prompt: str, box_thr: float, text_thr: float,
                 tile_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    boxes_all, scores_all = [], []
    for x0, y0, x1, y1 in tile_windows(*image.size, tile_cfg["tile_size"], tile_cfg["overlap_ratio"]):
        b, s = backend.detect(image.crop((x0, y0, x1, y1)), prompt, box_thr, text_thr)
        if len(b):
            boxes_all.append(offset_boxes(b, x0, y0))
            scores_all.append(np.asarray(s, dtype=float))
    if not boxes_all:
        return np.zeros((0, 4)), np.zeros(0)
    boxes, scores = np.concatenate(boxes_all), np.concatenate(scores_all)
    keep = merge_tile_boxes(boxes, scores, tile_cfg["merge_nms_iou"])
    return boxes[keep], scores[keep]


def run_detection(backend: DetectorBackend, image: Image.Image, prompt: str, box_thr: float, text_thr: float,
                  tile_cfg: dict, force_tiling: bool | None = None) -> tuple[np.ndarray, np.ndarray, bool]:
    """Whole-image first pass; re-run tiled if triggered. Returns (boxes, scores, tiled)."""
    w, h = image.size
    tiling_on = tile_cfg.get("enabled", True) if force_tiling is None else force_tiling
    if tiling_on and max(w, h) > tile_cfg["trigger_long_edge"]:
        b, s = detect_tiled(backend, image, prompt, box_thr, text_thr, tile_cfg)
        return b, s, True
    b, s = detect_whole(backend, image, prompt, box_thr, text_thr)
    if tiling_on and should_tile(w, h, tile_cfg["trigger_long_edge"], len(b), tile_cfg["trigger_box_count"]):
        b2, s2 = detect_tiled(backend, image, prompt, box_thr, text_thr, tile_cfg)
        return b2, s2, True
    return np.asarray(b).reshape(-1, 4), np.asarray(s), False
