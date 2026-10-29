"""Temperature scaling and ECE (spec §8.2)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def softmax_np(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Minimize NLL over a single scalar T (parametrized as log T so T>0)."""
    lg = torch.as_tensor(logits, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def calibrate(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15, max_iter: int = 200) -> dict:
    T = fit_temperature(logits, labels, max_iter)
    return {"temperature": T,
            "ece_before": expected_calibration_error(softmax_np(logits), labels, n_bins),
            "ece_after": expected_calibration_error(softmax_np(logits, T), labels, n_bins),
            "n": int(len(labels)), "n_bins": n_bins}


def save_calibration(rec: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec, indent=2))


def load_temperature(path: str | Path | None) -> float:
    if path and Path(path).is_file():
        return float(json.loads(Path(path).read_text())["temperature"])
    return 1.0
