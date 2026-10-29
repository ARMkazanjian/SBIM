"""Stage 2 metrics (spec §12.2)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def classification_report(y_true: np.ndarray, probs: np.ndarray, classes: list[str],
                          confusion_pairs: list[list[str]], extra_labels: list[str] | None = None) -> dict:
    """y_true: int indices into ``classes``; probs: (N, len(classes)).

    The confusion matrix is reported over ``classes + extra_labels`` (spec: full 7x7 incl. 'uncertain'); rows for
    labels never present as ground truth are zero.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    y_pred = probs.argmax(1)
    labels_all = list(classes) + list(extra_labels or [])
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(classes))), zero_division=0)
    present = sup > 0
    top2 = np.argsort(-probs, axis=1)[:, :2]
    out = {
        "n": int(len(y_true)),
        "top1": float((y_pred == y_true).mean()) if len(y_true) else float("nan"),
        "top2": float(np.mean([t in row for t, row in zip(y_true, top2)])) if len(y_true) else float("nan"),
        "macro_f1": float(f1[present].mean()) if present.any() else float("nan"),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]), "support": int(sup[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": {"labels": labels_all,
                             "matrix": _pad(confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))),
                                            len(labels_all)).tolist()},
    }
    idx = {c: i for i, c in enumerate(classes)}
    pairs = {}
    for a, b in confusion_pairs:
        m = np.isin(y_true, [idx[a], idx[b]])
        key = f"{a}<->{b}"
        pairs[key] = {"n": int(m.sum()), "accuracy": float((y_pred[m] == y_true[m]).mean()) if m.any() else None,
                      "a_as_b": int(((y_true == idx[a]) & (y_pred == idx[b])).sum()),
                      "b_as_a": int(((y_true == idx[b]) & (y_pred == idx[a])).sum())}
    out["confusion_pairs"] = pairs
    return out


def _pad(cm: np.ndarray, size: int) -> np.ndarray:
    full = np.zeros((size, size), dtype=int)
    full[:cm.shape[0], :cm.shape[1]] = cm
    return full
