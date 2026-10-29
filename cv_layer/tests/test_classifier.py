import copy
import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.classification.dataset import (Sample, class_loss_weights, load_samples, make_sampler)
from src.classification.hard_negatives import (label_noise_rate, mine_confusions, oversample, read_verified,
                                               write_review_sample)
from src.classification.model import EMA, StinkBugClassifier, load_checkpoint, save_checkpoint
from src.classification.train import lr_factor, train_classifier
from src.common.tracking import Tracker

CLASSES = ["rbsb", "sgsb", "gsb", "bmsb", "bsb", "other_pentatomid"]


def small_cfg(cfgs, **train):
    c = copy.deepcopy(cfgs["classifier"])
    c["model"].update(backbone="resnet18", input_size=64, pretrained=False, name="resnet18_64")
    c["train"].update(batch_size=8, epochs=3, patience=5, phase1_head_only_epochs=1, warmup_epochs=1, num_workers=0,
                      amp=False, min_labeled_nymphs=1, **train)
    return c


@pytest.fixture
def tiny_dataset(tmp_path):
    """Class-colour images so a tiny model can learn something; 6 species, 3 crops per image."""
    rng = np.random.default_rng(0)
    images, anns = [], []
    for i in range(36):
        arr = np.full((128, 128, 3), 40, np.uint8)
        sp = i % 6
        arr[10:60, 10:60] = (sp * 40, 255 - sp * 40, 100)
        Image.fromarray(arr).save(tmp_path / f"{i}.png")
        images.append({"id": i, "file_name": f"{i}.png"})
        anns.append({"id": i, "image_id": i, "category_id": sp + 1, "bbox": [10, 10, 50, 50],
                     "life_stage": "adult" if i % 3 else "nymph"})
    coco = {"images": images, "annotations": anns, "categories": [{"id": k + 1, "name": c} for k, c in enumerate(CLASSES)]}
    (tmp_path / "coco.json").write_text(json.dumps(coco))
    return tmp_path


def test_model_forward_heads_and_freezing(cfgs):
    c = small_cfg(cfgs)
    m = StinkBugClassifier.from_config(c, pretrained=False)
    sp, st = m(torch.randn(2, 3, 64, 64))
    assert sp.shape == (2, 6) and st.shape == (2, 2)
    m.set_backbone_trainable(False)
    assert not any(p.requires_grad for p in m.backbone.parameters())
    assert all(p.requires_grad for p in m.head_parameters())
    c["model"]["life_stage_head"] = False
    sp, st = StinkBugClassifier.from_config(c, pretrained=False)(torch.randn(1, 3, 64, 64))
    assert st is None


def test_default_backbone_builds(cfgs):
    m = StinkBugClassifier.from_config(cfgs["classifier"], pretrained=False)
    sp, _ = m(torch.randn(1, 3, 384, 384))
    assert sp.shape == (1, 6)


def test_lr_schedule_warmup_cosine():
    assert lr_factor(0, 3, 100) < 0.01
    assert lr_factor(1.5, 3, 100) == pytest.approx(0.5, abs=1e-3)
    assert lr_factor(3, 3, 100) == pytest.approx(1.0)
    assert lr_factor(100, 3, 100) == pytest.approx(0.0, abs=1e-6)


def test_ema_tracks_weights():
    m = torch.nn.Linear(2, 2)
    ema = EMA(m, 0.9)
    before = ema.state_dict()["weight"].clone()
    with torch.no_grad():
        m.weight += 1
    ema.update(m)
    d = min(0.9, 2 / 11)                       # warm-up decay on the first update
    assert torch.allclose(ema.state_dict()["weight"], before * d + (before + 1) * (1 - d))
    for _ in range(200):
        ema.update(m)
    assert ema.decay == 0.9 and torch.allclose(ema.state_dict()["weight"], m.weight, atol=1e-3)


def test_imbalance_handling():
    s = [Sample("p", (0, 0, 1, 1), 0, -1) for _ in range(90)] + [Sample("p", (0, 0, 1, 1), 1, -1) for _ in range(10)]
    w = class_loss_weights(s, 2)
    assert w[1] > w[0]
    smp = make_sampler(s, 2)
    draws = np.array([s[i].species for i in list(smp)])
    assert 0.35 < draws.mean() < 0.65          # roughly balanced
    s[0].weight = 3
    assert oversample(s, {"k"}, 3) == 0 and s[0].weight == 1.0


def test_train_end_to_end_smoke(cfgs, tiny_dataset, tmp_path):
    c = small_cfg(cfgs)
    tr = load_samples(tiny_dataset / "coco.json", set(range(0, 30)), CLASSES, ["adult", "nymph"], tiny_dataset)
    va = load_samples(tiny_dataset / "coco.json", set(range(30, 36)), CLASSES, ["adult", "nymph"], tiny_dataset)
    tracker = Tracker("jsonl", tmp_path / "run")
    res = train_classifier(c, cfgs["prep"], tr, va, tracker, tmp_path / "run", 0.15, device="cpu", pretrained=False)
    tracker.close()
    assert (tmp_path / "run" / "best.pt").exists() and (tmp_path / "run" / "calibration.json").exists()
    assert len((tmp_path / "run" / "metrics.jsonl").read_text().splitlines()) >= 1
    ck = load_checkpoint(tmp_path / "run" / "best.pt")
    assert ck["config"]["model"]["name"] == "resnet18_64"          # checkpoint carries its config
    assert res["life_stage_enabled"] is True and res["calibration"]["temperature"] > 0


def test_life_stage_head_disabled_when_labels_scarce(cfgs, tiny_dataset, tmp_path):
    c = small_cfg(cfgs)
    c["train"]["min_labeled_nymphs"] = 10_000
    tr = load_samples(tiny_dataset / "coco.json", set(range(0, 30)), CLASSES, ["adult", "nymph"], tiny_dataset)
    va = load_samples(tiny_dataset / "coco.json", set(range(30, 36)), CLASSES, ["adult", "nymph"], tiny_dataset)
    tracker = Tracker("jsonl", tmp_path / "r2")
    res = train_classifier(c, cfgs["prep"], tr, va, tracker, tmp_path / "r2", 0.15, device="cpu", pretrained=False)
    tracker.close()
    assert res["life_stage_enabled"] is False
    assert "stage_head.weight" not in load_checkpoint(tmp_path / "r2" / "best.pt")["state_dict"]


def test_both_imbalance_modes_rejected(cfgs, tmp_path):
    c = small_cfg(cfgs, imbalance="both")
    with pytest.raises(ValueError):
        train_classifier(c, cfgs["prep"], [], [], None, tmp_path, 0.15)


def test_hard_negative_mining_workflow(tmp_path):
    samples = [Sample("p", (0, 0, 1, 1), s, -1, key=f"k{i}") for i, s in enumerate([1, 2, 0, 3, 4, 0, 1])]
    logits = np.eye(6)[[2, 1, 5, 3, 3, 0, 1]] * 5      # k0 sgsb->gsb, k1 gsb->sgsb (pair), k2 rbsb->other (pair), k4 bsb->bmsb (pair)
    mined = mine_confusions(samples, logits, CLASSES, [["sgsb", "gsb"], ["bmsb", "bsb"], ["rbsb", "other_pentatomid"]])
    assert {m["key"] for m in mined} == {"k0", "k1", "k2", "k4"}
    # error outside a confusion pair is not mined
    assert mine_confusions([Sample("p", (0, 0, 1, 1), 0, -1, key="x")], np.eye(6)[[3]], CLASSES,
                           [["sgsb", "gsb"]]) == []
    p = tmp_path / "review.csv"
    write_review_sample(mined, p, 10)
    verdicts = {"k0": "label_correct", "k1": "label_correct", "k2": "label_wrong", "k4": "label_correct"}
    import csv
    with open(p) as f:
        data = list(csv.DictReader(f))
    for d in data:
        d["verdict"] = verdicts[d["key"]]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=data[0].keys())
        w.writeheader()
        w.writerows(data)
    assert read_verified(p) == {"k0", "k1", "k4"} and label_noise_rate(p) == pytest.approx(1 / 4)
    assert oversample(samples, read_verified(p), 3) == 3
    assert [s.weight for s in samples][0] == 3.0
