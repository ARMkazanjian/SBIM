from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]   # xyxy, absolute px in the (<=2048) working image
    score: float
    partial: bool = False
    label: str = ""


@dataclass
class PostprocessResult:
    detections: list[Detection]
    dropped: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class DetectorBackend(Protocol):
    """Anything that maps (image, prompt, thresholds) -> boxes (N,4 xyxy abs) and scores (N,)."""
    name: str

    def detect(self, image: Image.Image, prompt: str, box_threshold: float,
               text_threshold: float) -> tuple[np.ndarray, np.ndarray]: ...
