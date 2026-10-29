import pytest

from src.preprocessing import downscale_long_edge, should_tile, tile_windows
from src.preprocessing.augment import build_train_augmentation
from tests.conftest import make_image
import numpy as np


def test_downscale_only_above_cap():
    im, s = downscale_long_edge(make_image(4000, 3000), 2048)
    assert im.size == (2048, 1536) and s == pytest.approx(0.512)
    im, s = downscale_long_edge(make_image(1000, 800), 2048)
    assert im.size == (1000, 800) and s == 1.0


def test_tiles_cover_image_with_overlap():
    w, h = 1300, 900
    wins = tile_windows(w, h, 512, 0.2)
    cover = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in wins:
        assert x1 - x0 <= 512 and y1 - y0 <= 512
        cover[y0:y1, x0:x1] = True
    assert cover.all()
    xs = sorted({x0 for x0, *_ in wins})
    assert xs[1] - xs[0] == 410          # 512 * 0.8


def test_small_image_single_tile():
    assert tile_windows(300, 200, 512, 0.2) == [(0, 0, 300, 200)]


def test_tiling_triggers():
    assert should_tile(2000, 1000, 1536, 3, 15)
    assert should_tile(1000, 800, 1536, 16, 15)
    assert not should_tile(1000, 800, 1536, 15, 15)
    assert not should_tile(1000, 800, 1536, None, 15)


def test_augmentation_runs_and_caps_hue(cfgs):
    aug = build_train_augmentation(cfgs["prep"]["augmentation"], 64)
    out = aug(image=np.full((64, 64, 3), 128, np.uint8))["image"]
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8
    names = {type(t).__name__ for t in aug.transforms}
    assert not names & {"ChannelShuffle", "ToGray", "InvertImg", "MixUp", "CutMix"}
    bad = dict(cfgs["prep"]["augmentation"], hue_shift_limit=30)
    with pytest.raises(ValueError):
        build_train_augmentation(bad, 64)
