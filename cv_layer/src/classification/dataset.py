"""Crop dataset built from COCO annotations, plus imbalance handling (spec §7.2)."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from ..detection.postprocess import extract_crop

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Sample:
    image_path: str
    bbox: tuple[float, float, float, float]   # xyxy in image px
    species: int
    life_stage: int                            # -1 = unlabeled
    key: str = ""                              # stable id for hard-negative bookkeeping
    weight: float = 1.0


def load_samples(coco_path: str | Path, image_ids: set | None, species: list[str], life_stages: list[str],
                 image_root: str | Path | None = None) -> list[Sample]:
    coco = json.loads(Path(coco_path).read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i for i in coco["images"]}
    sidx = {s: i for i, s in enumerate(species)}
    out = []
    for a in coco["annotations"]:
        if image_ids is not None and a["image_id"] not in image_ids:
            continue
        name = cats[a["category_id"]]
        if name not in sidx:
            continue
        x, y, w, h = a["bbox"]
        stage = a.get("life_stage")
        root = Path(image_root) if image_root else Path(coco_path).parent
        out.append(Sample(str(root / images[a["image_id"]]["file_name"]), (x, y, x + w, y + h), sidx[name],
                          life_stages.index(stage) if stage in life_stages else -1,
                          key=f"{a['image_id']}:{a['id']}"))
    return out


def class_counts(samples: list[Sample], n: int) -> np.ndarray:
    c = Counter(s.species for s in samples)
    return np.array([c.get(i, 0) for i in range(n)], dtype=float)


def make_sampler(samples: list[Sample], n_classes: int) -> WeightedRandomSampler:
    """Inverse-frequency sampler; per-sample ``weight`` multiplies in hard-case oversampling."""
    counts = np.maximum(class_counts(samples, n_classes), 1)
    w = np.array([s.weight / counts[s.species] for s in samples], dtype=np.float64)
    return WeightedRandomSampler(torch.as_tensor(w), num_samples=len(samples), replacement=True)


def class_loss_weights(samples: list[Sample], n_classes: int) -> torch.Tensor:
    counts = np.maximum(class_counts(samples, n_classes), 1)
    w = counts.sum() / (n_classes * counts)
    return torch.as_tensor(w, dtype=torch.float32)


def to_tensor(img: Image.Image) -> torch.Tensor:
    a = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(a.transpose(2, 0, 1).copy())


class CropDataset(Dataset):
    def __init__(self, samples: list[Sample], input_size: int, padding: float, transform=None):
        self.samples, self.size, self.padding, self.transform = samples, input_size, padding, transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = Image.open(s.image_path).convert("RGB")
        crop = extract_crop(img, s.bbox, self.padding, self.size)
        if self.transform is not None:
            crop = Image.fromarray(self.transform(image=np.asarray(crop))["image"])
        return to_tensor(crop), s.species, s.life_stage, i
