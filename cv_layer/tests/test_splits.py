import json

import numpy as np
import pytest

from src.data_prep.coco_check import validate_coco
from src.data_prep.splits import (annotator_agreement, domain_report, field_only_test, split_dataset,
                                  verify_no_leakage)

SPECIES = ["rbsb", "sgsb", "gsb", "bmsb", "bsb", "other_pentatomid"]


def make_coco(n_sessions=60, per=4, seed=0):
    rng = np.random.default_rng(seed)
    cats = [{"id": i + 1, "name": s} for i, s in enumerate(SPECIES)]
    images, anns = [], []
    for s in range(n_sessions):
        domain = "field" if s % 2 else "lab"
        sp = int(rng.integers(0, 6))
        for j in range(per):
            iid = len(images) + 1
            images.append({"id": iid, "file_name": f"{iid}.jpg", "domain": domain, "session_id": f"s{s}"})
            for _ in range(2):
                anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": sp + 1, "bbox": [1, 1, 20, 30],
                             "annotator_id": "A1", "human_reviewed": True, "species_verified_by_entomologist": True})
    return {"images": images, "annotations": anns, "categories": cats}


def test_split_has_no_leakage_and_right_sizes():
    coco = make_coco()
    sp = split_dataset(coco, {"train": .7, "val": .15, "test": .15}, 1)
    assert verify_no_leakage(coco, sp) == []
    n = len(coco["images"])
    assert sum(len(v) for v in sp.values()) == n
    assert 0.6 < len(sp["train"]) / n < 0.8 and len(sp["val"]) > 0 and len(sp["test"]) > 0
    # deterministic given seed
    assert sp == split_dataset(coco, {"train": .7, "val": .15, "test": .15}, 1)


def test_leakage_detector_catches_session_split():
    coco = make_coco(10)
    sp = split_dataset(coco, {"train": .7, "val": .15, "test": .15}, 1)
    moved = sp["train"].pop(0)
    sp["test"].append(moved)
    # moved image's session-mates remain in train -> leak
    problems = verify_no_leakage(coco, sp)
    assert any("spans splits" in p for p in problems)


def test_stratified_by_domain_and_field_test():
    coco = make_coco()
    sp = split_dataset(coco, {"train": .7, "val": .15, "test": .15}, 3)
    rep = domain_report(coco, sp, 0.40)
    assert all(0.2 < v["field_fraction"] < 0.8 for v in rep["splits"].values())
    ft = field_only_test(coco, sp)
    assert ft and set(ft) <= set(sp["test"])


def test_test_split_predominantly_field_without_leakage():
    coco = make_coco()
    sp = split_dataset(coco, {"train": .7, "val": .15, "test": .15}, 3, min_test_field=0.6, min_train_field=0.4)
    assert verify_no_leakage(coco, sp) == []
    rep = domain_report(coco, sp, 0.40, 0.6)
    assert rep["splits"]["test"]["field_fraction"] >= 0.6 and rep["warnings"] == []
    assert sum(len(v) for v in sp.values()) == len(coco["images"])


def test_domain_report_warns_on_lab_heavy_data():
    coco = make_coco()
    for im in coco["images"]:
        im["domain"] = "lab"
    sp = split_dataset(coco, {"train": .7, "val": .15, "test": .15}, 3)
    rep = domain_report(coco, sp, 0.40)
    assert len(rep["warnings"]) == 2


def test_validate_coco():
    coco = make_coco(4)
    assert validate_coco(coco, SPECIES) == []
    coco["annotations"][0]["annotator_id"] = None
    coco["annotations"][1]["auto_generated"] = True
    coco["annotations"][1]["human_reviewed"] = False
    coco["categories"].append({"id": 99, "name": "uncertain"})
    probs = validate_coco(coco, SPECIES)
    assert any("annotator_id" in p for p in probs) and any("not human reviewed" in p for p in probs)
    assert any("uncertain" in p for p in probs)


def test_annotator_agreement():
    coco = make_coco(1, 1)
    base = coco["annotations"][0]
    coco["annotations"].append({**base, "id": 99, "annotator_id": "A2"})
    coco["annotations"].append({**base, "id": 100, "annotator_id": "A2", "bbox": [500, 500, 10, 10]})
    r = annotator_agreement(coco)
    assert r["label_agreement"] == 1.0 and r["matched_pairs"] >= 1


def test_scarce_field_data_does_not_starve_train():
    coco = make_coco(60)
    for im in coco["images"]:
        im["domain"] = "field" if int(im["session_id"][1:]) % 10 == 0 else "lab"      # ~10% field
    sp = split_dataset(coco, {"train": .7, "val": .15, "test": .15}, 3, min_test_field=0.6, min_train_field=0.4)
    assert verify_no_leakage(coco, sp) == []
    rep = domain_report(coco, sp, 0.40, 0.6)
    assert rep["splits"]["train"]["field"] > 0 and len(rep["warnings"]) == 2    # reported, not hidden
