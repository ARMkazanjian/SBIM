"""End-to-end metrics (spec §12.3) and acceptance check (§12.4)."""
from __future__ import annotations

import numpy as np


def count_metrics(pred_counts: list[int], true_counts: list[int]) -> dict:
    p, t = np.asarray(pred_counts, float), np.asarray(true_counts, float)
    err = np.abs(p - t)
    nz = t > 0
    return {"count_mae": float(err.mean()) if len(err) else float("nan"),
            "count_mape": float((err[nz] / t[nz]).mean() * 100) if nz.any() else None,
            "zero_error_images": int((err == 0).sum()), "n_images": int(len(err)),
            "zero_error_fraction": float((err == 0).mean()) if len(err) else float("nan")}


def species_count_mae(pred: list[dict], true: list[dict], species: list[str]) -> dict:
    """Per-species absolute count error; `uncertain` predictions are excluded from per-species error."""
    out = {}
    for s in species:
        e = [abs(p.get(s, 0) - t.get(s, 0)) for p, t in zip(pred, true)]
        out[s] = float(np.mean(e)) if e else float("nan")
    return out


def empty_image_fpr(pred_counts_on_empty: list[int]) -> dict:
    n = len(pred_counts_on_empty)
    fp_imgs = sum(1 for c in pred_counts_on_empty if c > 0)
    return {"n_empty_images": n, "images_with_false_positive": fp_imgs,
            "fpr": fp_imgs / n if n else None, "total_false_detections": int(sum(pred_counts_on_empty))}


def uncertain_rate(species_per_detection: list[str]) -> float | None:
    return (sum(s == "uncertain" for s in species_per_detection) / len(species_per_detection)
            if species_per_detection else None)


def check_acceptance(metrics: dict, acc: dict) -> dict:
    """metrics keys: detection_recall, detection_precision, count_mae, macro_f1, pair_accuracy (dict),
    empty_fpr, ece, total_inference_ms. Missing metrics are reported as 'not_evaluated' (never silently passed)."""
    checks = {
        "detection_recall": (metrics.get("detection_recall"), ">=", acc["detection_recall"]),
        "detection_precision": (metrics.get("detection_precision"), ">=", acc["detection_precision"]),
        "count_mae": (metrics.get("count_mae"), "<=", acc["count_mae"]),
        "macro_f1": (metrics.get("macro_f1"), ">=", acc["macro_f1"]),
        "empty_fpr": (metrics.get("empty_fpr"), "<=", acc["empty_fpr"]),
        "ece": (metrics.get("ece"), "<=", acc["ece"]),
        "total_inference_ms": (metrics.get("total_inference_ms"), "<=", acc["total_inference_ms"]),
    }
    for pair, v in (metrics.get("pair_accuracy") or {}).items():
        checks[f"pair_accuracy[{pair}]"] = (v, ">=", acc["confusion_pair_accuracy"])
    res = {}
    for k, (v, op, thr) in checks.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            res[k] = {"value": None, "threshold": thr, "status": "not_evaluated"}
        else:
            ok = v >= thr if op == ">=" else v <= thr
            res[k] = {"value": v, "threshold": thr, "op": op, "status": "pass" if ok else "fail"}
    res["_overall"] = ("pass" if all(r["status"] == "pass" for r in res.values()) else
                       "fail" if any(r["status"] == "fail" for r in res.values()) else "incomplete")
    return res
