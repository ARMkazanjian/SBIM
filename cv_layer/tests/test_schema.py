import copy
import json

import pytest

from src.output import SchemaViolation, to_json, validate_packet

EXAMPLE = {
    "schema_version": "1.0", "image_id": "img1",
    "source_metadata": {"original_dimensions": [4032, 3024], "capture_timestamp": "2026-10-27T14:32:11Z",
                        "gps": {"lat": 30.41, "lon": -91.18}, "device": "iPhone 15 Pro Max"},
    "processing": {"detector": "grounding-dino-base", "detector_prompt": "stink bug . shield bug .",
                   "classifier": "efficientnetv2_s_384", "tiled_inference": False,
                   "inference_ms": {"detection": 412, "classification": 88, "total": 500}},
    "detections": [{"id": 0, "bbox": [120, 340, 210, 455], "bbox_format": "xyxy_absolute",
                    "detection_confidence": 0.91, "species": "rbsb", "species_scientific": "Piezodorus guildinii",
                    "classification_confidence": 0.87, "fused_confidence": 0.88, "life_stage": "adult",
                    "life_stage_confidence": 0.79, "partial": False, "flag": None}],
    "aggregates": {"total_count": 1, "count_by_species": {"rbsb": 1, "uncertain": 0}, "density": None,
                   "scale_known": False, "spatial_distribution": None, "mean_detection_confidence": 0.91,
                   "mean_classification_confidence": 0.87, "partial_detections": 0},
    "warnings": [],
}


def mut(fn):
    d = copy.deepcopy(EXAMPLE)
    fn(d)
    return d


def test_example_is_valid_and_roundtrips():
    assert json.loads(to_json(EXAMPLE))["detections"][0]["species"] == "rbsb"


@pytest.mark.parametrize("fn", [
    lambda d: d["detections"][0].__setitem__("detection_confidence", 1.2),
    lambda d: d["detections"][0].__setitem__("bbox", [10, 10, 5, 20]),
    lambda d: d["detections"][0].__setitem__("bbox_format", "xywh"),
    lambda d: d["detections"][0].__setitem__("extra", 1),
    lambda d: d.__setitem__("schema_version", "2.0"),
    lambda d: d.pop("warnings"),
    lambda d: d["aggregates"].__setitem__("total_count", 2),
    lambda d: d["aggregates"].__setitem__("density", 0.5),            # density without scale
    lambda d: d["aggregates"]["count_by_species"].pop("uncertain"),
    lambda d: d["detections"][0].__setitem__("life_stage", "unknown"),  # unknown + confidence
    lambda d: d["aggregates"].__setitem__("count_by_species", {"sgsb": 1, "uncertain": 0}),
    lambda d: d["detections"][0].__setitem__("species", "uncertain"),
    lambda d: d["source_metadata"]["gps"].__setitem__("lat", 123),
])
def test_violations_fail_loudly(fn):
    with pytest.raises(SchemaViolation):
        validate_packet(mut(fn))


def test_uncertain_detection_valid():
    def f(d):
        det = d["detections"][0]
        det.update(species="uncertain", species_scientific=None, fused_confidence=0.2)
        d["aggregates"]["count_by_species"] = {"uncertain": 1}
    assert validate_packet(mut(f))
