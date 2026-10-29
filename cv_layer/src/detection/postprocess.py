"""Stage 1.5 — box post-processing and crop extraction (spec §6)."""
from __future__ import annotations

import numpy as np
from PIL import Image

from ..common.boxes import box_area, nms as np_nms
from ..common.logging import get_logger
from .types import Detection, PostprocessResult

log = get_logger(__name__)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    try:
        import torch
        import torchvision

        keep = torchvision.ops.nms(torch.as_tensor(boxes, dtype=torch.float32),
                                   torch.as_tensor(scores, dtype=torch.float32), iou)
        return keep.numpy()
    except ImportError:
        return np_nms(boxes, scores, iou)


def postprocess_boxes(boxes: np.ndarray, scores: np.ndarray, image_size: tuple[int, int], cfg: dict,
                      skip_nms: bool = False) -> PostprocessResult:
    """cfg is detector.yaml ``postprocess``. image_size = (width, height) of the image the boxes live in."""
    w, h = image_size
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    dropped = {"nms": 0, "size_small": 0, "size_large": 0, "aspect": 0}
    # clip to the image first
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

    if len(boxes) and not skip_nms:
        keep = _nms(boxes, scores, cfg["nms_iou"])
        dropped["nms"] = len(boxes) - len(keep)
        boxes, scores = boxes[keep], scores[keep]

    out: list[Detection] = []
    img_area = float(w * h)
    margin = cfg["edge_margin_px"]
    for b, s in zip(boxes, scores):
        bw, bh = b[2] - b[0], b[3] - b[1]
        area_frac = (bw * bh) / img_area if img_area else 0.0
        if bw <= 0 or bh <= 0 or area_frac < cfg["min_area_frac"]:
            dropped["size_small"] += 1
            log.info("box_dropped", extra={"reason": "size_small", "bbox": b.tolist(), "area_frac": area_frac})
            continue
        if area_frac > cfg["max_area_frac"]:
            dropped["size_large"] += 1
            log.info("box_dropped", extra={"reason": "size_large", "bbox": b.tolist(), "area_frac": area_frac})
            continue
        # Boxes are axis-aligned and bugs sit at arbitrary rotation, so the 0.5-3.0 window (spec §6.3) is applied
        # symmetrically: 1/max_aspect <= w/h <= max_aspect, i.e. orientation-free length:width <= max_aspect.
        aspect = max(bw, bh) / min(bw, bh)
        if aspect > cfg["max_aspect"]:
            dropped["aspect"] += 1
            log.info("box_dropped", extra={"reason": "aspect", "bbox": b.tolist(), "aspect": aspect})
            continue
        partial = bool(b[0] <= margin or b[1] <= margin or b[2] >= w - margin or b[3] >= h - margin)
        out.append(Detection(tuple(float(v) for v in b), float(s), partial))

    warnings = []
    labels = {"nms": "merged by NMS", "size_small": "dropped by min-size filter", "size_large": "dropped by max-size filter",
              "aspect": "dropped by aspect ratio filter"}
    for k, n in dropped.items():
        if n and k != "nms":
            warnings.append(f"{n} detection{'s' if n != 1 else ''} {labels[k]}")
    return PostprocessResult(out, dropped, warnings)


def expand_box(bbox: tuple[float, float, float, float], padding: float,
               image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    pw, ph = (x1 - x0) * padding, (y1 - y0) * padding
    w, h = image_size
    return (int(max(0, np.floor(x0 - pw))), int(max(0, np.floor(y0 - ph))),
            int(min(w, np.ceil(x1 + pw))), int(min(h, np.ceil(y1 + ph))))


def pad_to_square(crop: Image.Image, size: int, fill: tuple[int, int, int] = (124, 116, 104)) -> Image.Image:
    """Aspect-preserving resize + pad (never stretch)."""
    w, h = crop.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    resized = crop.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def extract_crop(image: Image.Image, bbox: tuple[float, float, float, float], padding: float,
                 out_size: int) -> Image.Image:
    box = expand_box(bbox, padding, image.size)
    return pad_to_square(image.crop(box), out_size)
