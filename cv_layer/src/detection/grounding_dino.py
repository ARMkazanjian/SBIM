"""Grounding DINO wrapper (spec §5.1). Revision is pinned in config; main is never tracked."""
from __future__ import annotations

import numpy as np
from PIL import Image

from ..common.logging import get_logger
from .prompts import validate_prompt

log = get_logger(__name__)


class GroundingDinoDetector:
    def __init__(self, hf_id: str, revision: str | None, device: str = "auto", name: str = "grounding-dino-base",
                 allow_unpinned_revision: bool = False):
        if not revision and not allow_unpinned_revision:
            raise ValueError("detector.model.revision must be pinned to an exact commit hash (spec §5.1). "
                             "Set it in configs/detector.yaml, or set allow_unpinned_revision=true for local tests.")
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.name = name
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else ("cpu" if device == "auto" else device))
        self.processor = AutoProcessor.from_pretrained(hf_id, revision=revision)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(hf_id, revision=revision)
        self.model.to(self.device).eval()

    @classmethod
    def from_config(cls, cfg: dict) -> "GroundingDinoDetector":
        m = cfg["model"]
        return cls(m["hf_id"], m.get("revision"), m.get("device", "auto"), m["name"],
                   m.get("allow_unpinned_revision", False))

    def detect(self, image: Image.Image, prompt: str, box_threshold: float,
               text_threshold: float) -> tuple[np.ndarray, np.ndarray]:
        import torch

        validate_prompt(prompt)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        target = [image.size[::-1]]
        kwargs = dict(input_ids=inputs.input_ids, text_threshold=text_threshold, target_sizes=target)
        try:
            res = self.processor.post_process_grounded_object_detection(outputs, threshold=box_threshold, **kwargs)[0]
        except TypeError:  # older transformers name
            res = self.processor.post_process_grounded_object_detection(outputs, box_threshold=box_threshold,
                                                                         **kwargs)[0]
        boxes = res["boxes"].detach().cpu().numpy().reshape(-1, 4).astype(float)
        scores = res["scores"].detach().cpu().numpy().astype(float)
        return boxes, scores
