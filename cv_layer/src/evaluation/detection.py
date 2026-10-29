"""Stage 1 metrics (spec §12.1) and the §5.3 decision gate."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..common.boxes import iou_matrix


def match_image(gt: np.ndarray, pred: np.ndarray, scores: np.ndarray, iou_thr: float):
    """Greedy score-ordered matching. Returns (tp flags aligned to score-sorted preds, order, matched ious)."""
    gt = np.asarray(gt, dtype=float).reshape(-1, 4)
    pred = np.asarray(pred, dtype=float).reshape(-1, 4)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    ious = iou_matrix(pred[order], gt)
    used = np.zeros(len(gt), bool)
    tp = np.zeros(len(order), bool)
    matched_iou = []
    for i in range(len(order)):
        if ious.shape[1] == 0:
            break
        cand = np.where(~used, ious[i], -1.0)
        j = int(cand.argmax())
        if cand[j] >= iou_thr:
            used[j] = True
            tp[i] = True
            matched_iou.append(float(cand[j]))
    return tp, order, matched_iou


def _ap(tp_all: np.ndarray, scores_all: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if len(tp_all) == 0:
        return 0.0
    o = np.argsort(-scores_all, kind="stable")
    tp = tp_all[o].astype(float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]       # monotone envelope, all-point interpolation
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_detection(gts: dict[str, np.ndarray], preds: dict[str, tuple[np.ndarray, np.ndarray]],
                       iou_thr: float = 0.5, latency_ms: list[float] | None = None) -> dict:
    """gts: image_id -> (N,4); preds: image_id -> (boxes (M,4), scores (M,)). Single class (stink bug)."""
    n_gt = sum(len(np.asarray(g).reshape(-1, 4)) for g in gts.values())
    n_pred = tp_count = 0
    ious: list[float] = []
    thr_grid = np.arange(0.5, 0.96, 0.05)
    ap_by_thr = []
    all_tp = {round(t, 2): ([], []) for t in thr_grid}
    for img, g in gts.items():
        b, s = preds.get(img, (np.zeros((0, 4)), np.zeros(0)))
        b, s = np.asarray(b, dtype=float).reshape(-1, 4), np.asarray(s, dtype=float)
        n_pred += len(b)
        tp, order, m_iou = match_image(g, b, s, iou_thr)
        tp_count += int(tp.sum())
        ious += m_iou
        for t in thr_grid:
            tpt, o, _ = match_image(g, b, s, float(t))
            all_tp[round(t, 2)][0].append(tpt)
            all_tp[round(t, 2)][1].append(s[o])
    for t in thr_grid:
        tps, scs = all_tp[round(t, 2)]
        ap_by_thr.append(_ap(np.concatenate(tps) if tps else np.zeros(0, bool),
                             np.concatenate(scs) if scs else np.zeros(0), n_gt))
    precision = tp_count / n_pred if n_pred else 0.0
    recall = tp_count / n_gt if n_gt else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    out = {"n_images": len(gts), "n_gt": n_gt, "n_pred": n_pred, "tp": tp_count, "precision": precision,
           "recall": recall, "f1": f1, "mean_iou": float(np.mean(ious)) if ious else 0.0,
           "map50": ap_by_thr[0], "map50_95": float(np.nanmean(ap_by_thr))}
    if latency_ms:
        out["latency_ms_mean"] = float(np.mean(latency_ms))
        out["latency_ms_p95"] = float(np.percentile(latency_ms, 95))
    return out


def decision_gate(zero_shot_recall: float, cfg: dict) -> dict:
    """§5.3 checkpoint. Returns the branch record; callers must persist it via record_decision()."""
    if zero_shot_recall >= cfg["ship_zero_shot_recall"]:
        branch, action = "ship_zero_shot", "Ship Grounding DINO zero-shot."
    elif zero_shot_recall >= cfg["finetune_recall_floor"]:
        branch, action = "finetune_head", ("Fine-tune detection head only (freeze Swin backbone + text encoder), "
                                           "lr 1e-5, few epochs, early stop on validation recall.")
    else:
        branch, action = "fallback_yolov8m_cbam_asff", ("Grounding DINO is the wrong tool for this imagery. Fall back to "
                                                       "YOLOv8m + CBAM + ASFF (Upadhyaya et al. 2026).")
    return {"branch": branch, "action": action, "zero_shot_recall": zero_shot_recall,
            "thresholds": {"ship": cfg["ship_zero_shot_recall"], "floor": cfg["finetune_recall_floor"]}}


def record_decision(decision: dict, path: str | Path, evidence: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {**decision, "evidence": evidence or {}, "recorded_at": datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(rec, indent=2, default=str))
    return path
