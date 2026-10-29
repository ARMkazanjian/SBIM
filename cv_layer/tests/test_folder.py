import csv

import pytest

from src.data_prep.folder import first_rule, image_species, index_folder, source_stem
from tests.conftest import make_image

FOLDER = {"session_rules": [{"pattern": r"^(AdltStnkBug_\d+)_", "value": r"\1"}],
          "domain_rules": [{"pattern": r"^(AdltStnkBug_|IMG_)", "value": "lab"}], "default_domain": "field",
          "labels_csv": None, "label_rules": []}
SPECIES = ["rbsb", "sgsb", "gsb", "bmsb", "bsb", "other_pentatomid"]


@pytest.mark.parametrize("name,stem", [
    ("AdltStnkBug_001_3500_jpg.rf.e62d994437094281244bbba152a68c14.jpg", "AdltStnkBug_001_3500"),
    ("original_jpeg.rf.bceb.jpg", "original"),
    ("large-12-_jpeg.rf.fNeNaeNaNaNfNeNd(1).jpg", "large-12-"),
    ("plain_name.png", "plain_name"),
])
def test_source_stem(name, stem):
    assert source_stem(name) == stem


def test_index_folder_mixed_formats(tmp_path):
    for n in ["AdltStnkBug_003_3500_jpg.rf.aa.jpg", "AdltStnkBug_003_4500_jpg.rf.bb.jpg", "IMG_1_jpg.rf.cc.png",
              "large-1-_jpeg.rf.dd.jpg"]:
        make_image(300, 300).save(tmp_path / n)
    make_image(100, 100).save(tmp_path / "tiny.png")
    (tmp_path / "bad.jpg").write_bytes(b"x")
    (tmp_path / "readme.txt").write_text("x")
    coco, rej = index_folder(tmp_path, [".jpg", ".png"], FOLDER, 224, SPECIES)
    assert len(coco["images"]) == 4 and {r[0] for r in rej} == {"tiny.png", "bad.jpg"}
    by = {i["file_name"]: i for i in coco["images"]}
    a, b = by["AdltStnkBug_003_3500_jpg.rf.aa.jpg"], by["AdltStnkBug_003_4500_jpg.rf.bb.jpg"]
    assert a["session_id"] == b["session_id"] == "AdltStnkBug_003"      # frames of one specimen share a split
    assert a["domain"] == "lab" and by["IMG_1_jpg.rf.cc.png"]["domain"] == "lab"
    assert by["large-1-_jpeg.rf.dd.jpg"]["domain"] == "field"
    assert by["large-1-_jpeg.rf.dd.jpg"]["session_id"] == "large-1-"


def test_missing_dir():
    with pytest.raises(FileNotFoundError):
        index_folder("/nonexistent/xyz", [".jpg"], FOLDER, 224, SPECIES)


def test_labels_never_guessed(tmp_path):
    coco = {"images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"}],
            "categories": [{"id": 1, "name": "rbsb"}, {"id": 2, "name": "gsb"}]}
    assert image_species(coco, FOLDER) == {}
    p = tmp_path / "l.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows([["file_name", "species", "life_stage"], ["a.jpg", "gsb", "adult"]])
    cfg = {**FOLDER, "labels_csv": str(p), "label_rules": [{"pattern": "^b", "value": "rbsb"}]}
    assert image_species(coco, cfg) == {1: {"species": "gsb", "life_stage": "adult"},
                                        2: {"species": "rbsb", "life_stage": None}}
    assert image_species(coco, FOLDER, "rbsb")[1]["species"] == "rbsb"
    with pytest.raises(ValueError):
        image_species(coco, FOLDER, "not_a_species")
    assert first_rule([], "x", "d") == "d"
