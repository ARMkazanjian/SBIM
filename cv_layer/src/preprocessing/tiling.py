"""Sliced inference helpers for dense / high-res images (spec §4.3). Implemented manually (no sahi dependency
at runtime) so Grounding DINO support does not depend on a third-party adapter."""
from __future__ import annotations

import numpy as np

from ..common.boxes import nms


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def tile_windows(width: int, height: int, tile_size: int, overlap_ratio: float) -> list[tuple[int, int, int, int]]:
    """Return (x0, y0, x1, y1) windows covering the image with the given overlap."""
    stride = max(1, int(round(tile_size * (1.0 - overlap_ratio))))
    return [(x, y, min(x + tile_size, width), min(y + tile_size, height))
            for y in _starts(height, tile_size, stride) for x in _starts(width, tile_size, stride)]


def should_tile(width: int, height: int, trigger_long_edge: int, first_pass_boxes: int | None,
                trigger_box_count: int) -> bool:
    if max(width, height) > trigger_long_edge:
        return True
    return first_pass_boxes is not None and first_pass_boxes > trigger_box_count


def merge_tile_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """NMS across tile boundaries; returns kept indices."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    return nms(boxes, scores, iou_thr)


def offset_boxes(boxes: np.ndarray, x0: int, y0: int) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
    boxes[:, [0, 2]] += x0
    boxes[:, [1, 3]] += y0
    return boxes
