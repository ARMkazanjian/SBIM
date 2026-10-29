import numpy as np
import pytest

from src.classification.calibration import calibrate, expected_calibration_error, softmax_np
from src.evaluation import (check_acceptance, classification_report, count_metrics, decision_gate, empty_image_fpr,
                            evaluate_detection, record_decision)


def test_detection_metrics_perfect_and_partial():
    gt = {"a": np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float), "b": np.array([[5, 5, 15, 15]], float)}
    perfect = {k: (v.copy(), np.array([0.9] * len(v))) for k, v in gt.items()}
    m = evaluate_detection(gt, perfect)
    assert m["precision"] == m["recall"] == m["map50"] == m["map50_95"] == 1.0 and m["mean_iou"] == 1.0
    miss = dict(perfect)
    miss["a"] = (gt["a"][:1], np.array([0.9]))
    miss["b"] = (np.vstack([gt["b"], [[100, 100, 110, 110]]]), np.array([0.9, 0.8]))
    m = evaluate_detection(gt, miss)
    assert m["tp"] == 2 and m["n_pred"] == 3 and m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)


def test_each_gt_matched_once():
    gt = {"a": np.array([[0, 0, 10, 10]], float)}
    pred = {"a": (np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float), np.array([0.9, 0.8]))}
    m = evaluate_detection(gt, pred)
    assert m["tp"] == 1 and m["precision"] == 0.5


def test_decision_gate_branches(cfgs, tmp_path):
    g = cfgs["detector"]["decision_gate"]
    assert decision_gate(0.90, g)["branch"] == "ship_zero_shot"
    assert decision_gate(0.85, g)["branch"] == "ship_zero_shot"
    assert decision_gate(0.80, g)["branch"] == "finetune_head"
    assert decision_gate(0.70, g)["branch"] == "finetune_head"
    assert decision_gate(0.69, g)["branch"] == "fallback_yolov8m_cbam_asff"
    p = record_decision(decision_gate(0.5, g), tmp_path / "d.json", {"x": 1})
    assert "fallback" in p.read_text() and "recorded_at" in p.read_text()


def test_classification_report_7x7_and_pairs():
    classes = ["rbsb", "sgsb", "gsb", "bmsb", "bsb", "other_pentatomid"]
    y = np.array([0, 0, 1, 2, 3, 4, 5, 1])
    probs = np.eye(6)[[0, 0, 2, 2, 3, 3, 5, 1]] * 0.8 + 0.04
    r = classification_report(y, probs, classes, [["sgsb", "gsb"], ["bmsb", "bsb"], ["rbsb", "other_pentatomid"]],
                              ["uncertain"])
    cm = np.array(r["confusion_matrix"]["matrix"])
    assert cm.shape == (7, 7) and cm[1, 2] == 1 and cm[4, 3] == 1 and cm[6].sum() == 0
    assert r["confusion_pairs"]["sgsb<->gsb"]["accuracy"] == pytest.approx(2 / 3)
    assert r["confusion_pairs"]["bmsb<->bsb"]["b_as_a"] == 1
    assert 0 < r["macro_f1"] < 1 and r["top2"] >= r["top1"]


def test_count_metrics():
    m = count_metrics([3, 5, 0], [3, 7, 0])
    assert m["count_mae"] == pytest.approx(2 / 3) and m["zero_error_images"] == 2
    assert m["count_mape"] == pytest.approx(((0 + 2 / 7) / 2) * 100)
    assert empty_image_fpr([0, 0, 2, 0])["fpr"] == 0.25


def test_acceptance_never_silently_passes_missing(cfgs):
    acc = cfgs["pipeline"]["acceptance"]
    r = check_acceptance({"detection_recall": 0.9, "detection_precision": 0.95, "count_mae": 2.0, "macro_f1": 0.9,
                          "empty_fpr": 0.0, "ece": 0.03, "total_inference_ms": 900,
                          "pair_accuracy": {"a<->b": 0.8}}, acc)
    assert r["_overall"] == "pass"
    r = check_acceptance({"detection_recall": 0.9}, acc)
    assert r["_overall"] == "incomplete" and r["ece"]["status"] == "not_evaluated"
    r = check_acceptance({"detection_recall": 0.5}, acc)
    assert r["_overall"] == "fail"


def test_temperature_scaling_reduces_ece():
    rng = np.random.default_rng(0)
    n, k = 3000, 4
    true_logits = rng.normal(size=(n, k)) * 1.5
    labels = np.array([rng.choice(k, p=p) for p in softmax_np(true_logits)])
    overconfident = true_logits * 3.0
    rec = calibrate(overconfident, labels)
    assert rec["temperature"] == pytest.approx(3.0, rel=0.25)
    assert rec["ece_after"] < rec["ece_before"] and rec["ece_after"] < 0.05
    assert expected_calibration_error(np.eye(3)[[0, 1]], np.array([0, 1])) == 0.0
