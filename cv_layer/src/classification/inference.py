"""Batch crop inference with temperature-scaled probabilities."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .calibration import softmax_np
from .dataset import to_tensor
from .model import StinkBugClassifier, load_checkpoint


class CropClassifier:
    def __init__(self, checkpoint: str, device: str | None = None, temperature: float = 1.0,
                 classifier_cfg: dict | None = None):
        ck = load_checkpoint(checkpoint)
        cfg = ck["config"] if classifier_cfg is None else classifier_cfg
        cfg = cfg.get("classifier", cfg)
        self.cfg = cfg
        self.classes = cfg["classes"]["species"]
        self.stages = cfg["classes"]["life_stages"]
        self.size = cfg["model"]["input_size"]
        self.name = cfg["model"]["name"]
        self.has_stage_head = ck["state_dict"].get("stage_head.weight") is not None
        cfg["model"]["life_stage_head"] = self.has_stage_head
        self.model = StinkBugClassifier.from_config(cfg, pretrained=False)
        self.model.load_state_dict(ck["state_dict"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        self.temperature = temperature

    @torch.inference_mode()
    def classify(self, crops: list[Image.Image], batch_size: int = 32):
        """Returns (species_probs (N,C) calibrated, stage_probs (N,2) or None)."""
        sp, st = [], []
        for i in range(0, len(crops), batch_size):
            x = torch.stack([to_tensor(c) for c in crops[i:i + batch_size]]).to(self.device)
            a, b = self.model(x)
            sp.append(a.float().cpu().numpy())
            if b is not None:
                st.append(b.float().cpu().numpy())
        if not sp:
            return np.zeros((0, len(self.classes))), None
        probs = softmax_np(np.concatenate(sp), self.temperature)
        return probs, (softmax_np(np.concatenate(st)) if st else None)
