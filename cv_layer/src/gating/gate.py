"""Confidence fusion and uncertainty routing (spec §8.1)."""
from __future__ import annotations

from dataclasses import dataclass


def fused_confidence(det_conf: float, cls_conf: float, alpha: float) -> float:
    """fused = det^alpha * cls^(1-alpha). alpha=0.4 weights classification higher."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    d, c = min(max(det_conf, 0.0), 1.0), min(max(cls_conf, 0.0), 1.0)
    return round(float((d ** alpha) * (c ** (1.0 - alpha))), 9)   # rounding keeps band edges exact


@dataclass(frozen=True)
class GateDecision:
    species: str                  # predicted label, or "uncertain"
    fused_confidence: float
    flag: str | None              # None | "low_confidence"
    queue_for_review: bool


def route(species: str, det_conf: float, cls_conf: float, cfg: dict) -> GateDecision:
    """cfg is pipeline.yaml ``gating``. The detection always survives; only the species label can become 'uncertain'."""
    f = fused_confidence(det_conf, cls_conf, cfg["alpha"])
    if f >= cfg["accept_threshold"]:
        return GateDecision(species, f, None, False)
    if f >= cfg["low_conf_threshold"]:
        return GateDecision(species, f, "low_confidence", False)
    return GateDecision("uncertain", f, None, True)
