"""Hard-negative mining over the confusion pairs (spec §7.4)."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

from .dataset import Sample


def mine_confusions(samples: list[Sample], logits: np.ndarray, classes: list[str],
                    confusion_pairs: list[list[str]]) -> list[dict]:
    """Every misclassification where (true, pred) both lie inside the same confusion pair."""
    idx = {c: i for i, c in enumerate(classes)}
    pairs = {frozenset((idx[a], idx[b])) for a, b in confusion_pairs}
    pred = logits.argmax(1)
    out = []
    for s, p in zip(samples, pred):
        if p != s.species and frozenset((s.species, int(p))) in pairs:
            out.append({"key": s.key, "image_path": s.image_path, "bbox": list(s.bbox),
                        "true": classes[s.species], "pred": classes[int(p)]})
    return out


def write_review_sample(mined: list[dict], path: str | Path, sample_size: int, seed: int = 0) -> list[dict]:
    """Write a random sample for manual label-noise review; reviewer fills the `verdict` column
    (`label_correct` | `label_wrong`)."""
    rng = random.Random(seed)
    sample = rng.sample(mined, min(sample_size, len(mined)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "image_path", "bbox", "true", "pred", "verdict"])
        w.writeheader()
        for r in sample:
            w.writerow({**r, "bbox": ",".join(f"{v:.1f}" for v in r["bbox"]), "verdict": ""})
    return sample


def read_verified(path: str | Path) -> set[str]:
    """Keys the reviewer marked label_correct (the model was genuinely wrong => a real hard case)."""
    with open(path, newline="") as f:
        return {r["key"] for r in csv.DictReader(f) if r["verdict"].strip() == "label_correct"}


def label_noise_rate(path: str | Path) -> float | None:
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["verdict"].strip()]
    return sum(r["verdict"].strip() == "label_wrong" for r in rows) / len(rows) if rows else None


def oversample(samples: list[Sample], verified_keys: set[str], factor: int) -> int:
    """Set weight = factor on verified hard cases (used by the weighted sampler). Returns count changed."""
    n = 0
    for s in samples:
        s.weight = float(factor) if s.key in verified_keys else 1.0
        n += s.key in verified_keys
    return n
