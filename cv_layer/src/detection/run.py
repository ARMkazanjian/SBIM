"""Detect on a loaded image and return boxes in ORIGINAL-image coordinates (shared by pipeline-adjacent scripts)."""
from __future__ import annotations

import time

import numpy as np

from ..ingestion.loader import LoadedImage
from ..preprocessing.resize import downscale_long_edge
from .detector import run_detection
from .postprocess import postprocess_boxes


def detect_original_coords(backend, loaded: LoadedImage, prompt: str, box_thr: float, text_thr: float,
                           prep_cfg: dict, post_cfg: dict, force_tiling: bool | None = None):
    t0 = time.perf_counter()
    work, scale = downscale_long_edge(loaded.image, prep_cfg["resize"]["max_long_edge"])
    boxes, scores, tiled = run_detection(backend, work, prompt, box_thr, text_thr, prep_cfg["tiling"], force_tiling)
    post = postprocess_boxes(boxes, scores, work.size, post_cfg)
    b = np.array([d.bbox for d in post.detections], dtype=float).reshape(-1, 4) / scale
    s = np.array([d.score for d in post.detections], dtype=float)
    return b, s, tiled, (time.perf_counter() - t0) * 1000, post
