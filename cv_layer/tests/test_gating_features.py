import numpy as np
import pytest

from src.features import compute_aggregates, nearest_neighbor_index, spatial_distribution
from src.gating import fused_confidence, route


def test_fused_formula():
    assert fused_confidence(1.0, 1.0, 0.4) == 1.0
    assert fused_confidence(0.9, 0.5, 0.4) == pytest.approx(0.9 ** 0.4 * 0.5 ** 0.6)
    # alpha=0.4 weights classification more heavily
    assert fused_confidence(0.5, 0.9, 0.4) > fused_confidence(0.9, 0.5, 0.4)
    with pytest.raises(ValueError):
        fused_confidence(.5, .5, 1.5)


@pytest.mark.parametrize("det,cls,species,flag", [
    (0.95, 0.9, "rbsb", None),
    (0.8, 0.6, "rbsb", "low_confidence"),
    (0.5, 0.3, "uncertain", None),
])
def test_routing_bands(cfgs, det, cls, species, flag):
    d = route("rbsb", det, cls, cfgs["pipeline"]["gating"])
    assert d.species == species and d.flag == flag
    assert d.queue_for_review == (species == "uncertain")


def test_band_edges(cfgs):
    g = dict(cfgs["pipeline"]["gating"], alpha=0.5)
    assert route("x", 0.75, 0.75, g).flag is None            # exactly 0.75 -> normal
    assert route("x", 0.45, 0.45, g).species == "x"          # exactly 0.45 -> low_confidence
    assert route("x", 0.45, 0.45, g).flag == "low_confidence"
    assert route("x", 0.44, 0.44, g).species == "uncertain"


def _dets(boxes, species=None):
    return [{"bbox": b, "species": (species or ["rbsb"] * len(boxes))[i], "detection_confidence": 0.8,
             "classification_confidence": 0.6, "partial": i == 0} for i, b in enumerate(boxes)]


def test_aggregates_density_null_without_scale(cfgs):
    f = cfgs["pipeline"]["features"]
    agg = compute_aggregates(_dets([[0, 0, 10, 10], [50, 50, 60, 60]], ["rbsb", "uncertain"]), (1000, 1000), f)
    assert agg["density"] is None and agg["scale_known"] is False
    assert agg["total_count"] == 2 and agg["count_by_species"] == {"rbsb": 1, "uncertain": 1}
    assert agg["partial_detections"] == 1 and agg["mean_detection_confidence"] == 0.8
    assert agg["spatial_distribution"] is None            # below min points


def test_density_only_with_scale(cfgs):
    agg = compute_aggregates(_dets([[0, 0, 10, 10]] * 4), (1000, 1000), cfgs["pipeline"]["features"], 10.0)
    assert agg["scale_known"] and agg["density"] == pytest.approx(4 / (100 * 100))


def test_empty_image_aggregates(cfgs):
    agg = compute_aggregates([], (500, 500), cfgs["pipeline"]["features"])
    assert agg["total_count"] == 0 and agg["count_by_species"] == {"uncertain": 0}
    assert agg["mean_detection_confidence"] is None


def test_spatial_distribution(cfgs):
    f = cfgs["pipeline"]["features"]
    clustered = [[100 + i * 3, 100 + j * 3, 104 + i * 3, 104 + j * 3] for i in range(3) for j in range(3)]
    grid = [[x, y, x + 10, y + 10] for x in range(100, 1000, 300) for y in range(100, 1000, 300)]
    assert spatial_distribution(clustered, (1000, 1000), f) == "clustered"
    assert spatial_distribution(grid, (1000, 1000), f) == "dispersed"
    assert nearest_neighbor_index(np.array([[0., 0.]]), 100) is None
