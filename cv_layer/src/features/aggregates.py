"""Per-image feature extraction (spec §9)."""
from __future__ import annotations

import numpy as np


def nearest_neighbor_index(centroids: np.ndarray, area: float) -> float | None:
    """Clark-Evans R = observed mean NN distance / expected under complete spatial randomness.
    R<1 clustered, R~1 random, R>1 dispersed. Uses the image area as the study region."""
    n = len(centroids)
    if n < 2 or area <= 0:
        return None
    d = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    observed = d.min(axis=1).mean()
    expected = 0.5 / np.sqrt(n / area)
    return float(observed / expected)


def spatial_distribution(bboxes: list[list[float]], image_size: tuple[int, int], cfg: dict) -> str | None:
    if len(bboxes) < cfg["min_points_for_distribution"]:
        return None
    b = np.asarray(bboxes, dtype=float)
    cent = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)
    r = nearest_neighbor_index(cent, float(image_size[0] * image_size[1]))
    if r is None:
        return None
    return "clustered" if r < cfg["nn_index_split"] else "dispersed"


def compute_aggregates(dets: list[dict], image_size: tuple[int, int], cfg: dict,
                       pixels_per_cm: float | None = None) -> dict:
    """dets: dicts with bbox, species, detection_confidence, classification_confidence, partial.
    image_size is the (w, h) in the same coordinate system as the boxes."""
    n = len(dets)
    counts: dict[str, int] = {}
    for d in dets:
        counts[d["species"]] = counts.get(d["species"], 0) + 1
    counts.setdefault("uncertain", 0)
    scale = pixels_per_cm if pixels_per_cm is not None else cfg.get("scale_pixels_per_cm")
    scale_known = bool(scale)
    density = None
    if scale_known:                                  # never fabricate: only with a real reference scale
        area_cm2 = (image_size[0] / scale) * (image_size[1] / scale)
        density = round(n / area_cm2, 6)
    mean = lambda k: round(float(np.mean([d[k] for d in dets])), 4) if dets else None
    return {
        "total_count": n,
        "count_by_species": counts,
        "density": density,
        "scale_known": scale_known,
        "spatial_distribution": spatial_distribution([d["bbox"] for d in dets], image_size, cfg),
        "mean_detection_confidence": mean("detection_confidence"),
        "mean_classification_confidence": mean("classification_confidence"),
        "partial_detections": sum(bool(d["partial"]) for d in dets),
    }
