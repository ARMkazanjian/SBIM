"""Human-in-the-loop review queue: structured directory + manifest CSV (spec §8.3)."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

MANIFEST_FIELDS = ["crop_path", "meta_path", "image_id", "source", "detection_id", "bbox", "predicted_species",
                   "detection_confidence", "classification_confidence", "fused_confidence", "queued_at",
                   "reviewed_species", "reviewer_id"]


class ReviewQueue:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.crops = self.root / "crops"
        self.crops.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifest.csv"
        if not self.manifest.exists():
            with open(self.manifest, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=MANIFEST_FIELDS).writeheader()

    def add(self, crop: Image.Image, image_id: str, source: str, detection: dict, predicted_species: str,
            extra: dict | None = None) -> Path:
        stem = f"{image_id}_{detection['id']:03d}"
        crop_path, meta_path = self.crops / f"{stem}.png", self.crops / f"{stem}.json"
        crop.save(crop_path)
        meta = {"image_id": image_id, "source": source, "detection": detection,
                "predicted_species": predicted_species, **(extra or {})}
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        with open(self.manifest, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=MANIFEST_FIELDS).writerow({
                "crop_path": str(crop_path.relative_to(self.root)), "meta_path": str(meta_path.relative_to(self.root)),
                "image_id": image_id, "source": source, "detection_id": detection["id"],
                "bbox": ",".join(f"{v:.1f}" for v in detection["bbox"]), "predicted_species": predicted_species,
                "detection_confidence": detection["detection_confidence"],
                "classification_confidence": detection["classification_confidence"],
                "fused_confidence": detection["fused_confidence"],
                "queued_at": datetime.now(timezone.utc).isoformat(), "reviewed_species": "", "reviewer_id": ""})
        return crop_path
