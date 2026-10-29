import json

import numpy as np
import pytest

from src.ingestion import load_image
from src.output import ReviewQueue, validate_packet
from src.pipeline import CVPipeline
from tests.conftest import StubClassifier, StubDetector, jpeg_bytes, make_image


def build(cfgs, boxes, scores, rows, tmp_path, overrides=None):
    p, d, c, pre = (cfgs[k] for k in ("pipeline", "detector", "classifier", "prep"))
    if overrides:
        p = {**p, **overrides}
    return CVPipeline(StubDetector(boxes, scores), StubClassifier(rows), p, d, pre, c,
                      ReviewQueue(tmp_path / "rq"))


CONF = [0.95, 0.01, 0.01, 0.01, 0.01, 0.01]      # confident rbsb
MID = [0.62, 0.30, 0.02, 0.02, 0.02, 0.02]
LOW = [0.20, 0.19, 0.19, 0.15, 0.15, 0.12]


def test_end_to_end_packet(cfgs, tmp_path):
    pipe = build(cfgs, [[50, 50, 130, 110], [200, 200, 280, 260], [400, 300, 480, 360]], [0.9, 0.9, 0.9],
                 [CONF, MID, LOW], tmp_path)
    pk = pipe.run(load_image_from(tmp_path))
    validate_packet(pk)
    d = pk["detections"]
    assert [x["species"] for x in d] == ["rbsb", "rbsb", "uncertain"]
    assert [x["flag"] for x in d] == [None, "low_confidence", None]
    assert pk["aggregates"]["total_count"] == 3                      # uncertain still counted
    assert pk["aggregates"]["count_by_species"] == {"rbsb": 2, "uncertain": 1}
    assert d[0]["species_scientific"] == "Piezodorus guildinii" and d[2]["species_scientific"] is None
    assert d[0]["life_stage"] == "adult" and 0 <= d[0]["life_stage_confidence"] <= 1
    assert pk["aggregates"]["density"] is None and pk["aggregates"]["scale_known"] is False
    assert pk["warnings"] == [] and pk["processing"]["tiled_inference"] is False
    assert pk["processing"]["detector_prompt"] == cfgs["detector"]["prompt"]["selected"]
    # exactly the uncertain detection reached the review queue, with manifest row + crop
    rows = (tmp_path / "rq" / "manifest.csv").read_text().strip().splitlines()
    assert len(rows) == 2 and ",rbsb," in rows[1]          # queued with the model's best guess
    assert list((tmp_path / "rq" / "crops").glob("*.png"))


def load_image_from(tmp_path):
    from src.ingestion import load_bytes
    return load_bytes(jpeg_bytes(make_image(640, 480)), "img1")


def test_empty_image(cfgs, tmp_path):
    pk = build(cfgs, np.zeros((0, 4)), [], [CONF], tmp_path).run(load_image_from(tmp_path))
    assert pk["detections"] == [] and pk["aggregates"]["total_count"] == 0
    assert pk["aggregates"]["mean_detection_confidence"] is None


def test_boxes_reported_in_original_coordinates(cfgs, tmp_path):
    from src.ingestion import load_bytes
    big = load_bytes(jpeg_bytes(make_image(4096, 3072)), "big")
    # detector sees the 2048x1536 working image
    pipe = build(cfgs, [[100, 100, 200, 180]], [0.9], [CONF], tmp_path)
    pk = pipe.run(big)
    assert pk["source_metadata"]["original_dimensions"] == [4096, 3072]
    assert pk["detections"][0]["bbox"] == [200.0, 200.0, 400.0, 360.0]
    assert pk["processing"]["tiled_inference"] is True                # long edge > 1536


def test_partial_box_downweighted_and_counted(cfgs, tmp_path):
    pk = build(cfgs, [[0, 100, 60, 160]], [0.9], [CONF], tmp_path).run(load_image_from(tmp_path))
    d = pk["detections"][0]
    assert d["partial"] and d["detection_confidence"] == pytest.approx(0.9 * 0.85, abs=1e-3)
    assert pk["aggregates"]["partial_detections"] == 1


def test_filter_warning_surfaces(cfgs, tmp_path):
    pk = build(cfgs, [[100, 100, 400, 120]], [0.9], [CONF], tmp_path).run(load_image_from(tmp_path))
    assert pk["detections"] == [] and pk["warnings"] == ["1 detection dropped by aspect ratio filter"]


def test_run_bytes_and_path(cfgs, tmp_path):
    pipe = build(cfgs, [[50, 50, 130, 110]], [0.9], [CONF], tmp_path)
    assert pipe.run_bytes(jpeg_bytes(make_image()), "b")["image_id"] == "b"
    p = tmp_path / "x.jpg"
    p.write_bytes(jpeg_bytes(make_image()))
    assert pipe.run_path(p)["image_id"] == "x"


def test_life_stage_unknown_when_head_absent(cfgs, tmp_path):
    pipe = build(cfgs, [[50, 50, 130, 110]], [0.9], [CONF], tmp_path)
    pipe.classifier.has_stage_head = False
    pipe.use_stage = False
    d = pipe.run(load_image_from(tmp_path))["detections"][0]
    assert d["life_stage"] == "unknown" and d["life_stage_confidence"] is None


def test_unpinned_detector_revision_refused():
    from src.detection.grounding_dino import GroundingDinoDetector
    with pytest.raises(ValueError, match="pinned"):
        GroundingDinoDetector("IDEA-Research/grounding-dino-base", None)


def test_config_overrides():
    from src.common.config import apply_overrides
    c = apply_overrides({"a": {"b": 1}}, ["a.b=3e-3", "a.c=[1,2]", "x.y=true", "name=stink bug ."])
    assert c == {"a": {"b": 0.003, "c": [1, 2]}, "x": {"y": True}, "name": "stink bug ."}
