import numpy as np
import pytest
from PIL import Image

from src.detection import (expand_box, extract_crop, format_prompt, pad_to_square, postprocess_boxes,
                           run_detection, validate_prompt)
from tests.conftest import StubDetector, make_image


def test_prompt_rules():
    assert format_prompt("Stink Bug.Shield  bug") == "stink bug . shield bug ."
    assert validate_prompt("stink bug . shield bug .") == "stink bug . shield bug ."
    for bad in ["Stink bug .", "stink bug", "stink bug.", "stink bug . . ", "a . b"]:
        with pytest.raises(ValueError):
            validate_prompt(bad)


def test_config_prompts_all_valid(cfgs):
    for p in cfgs["detector"]["prompt"]["candidates"] + [cfgs["detector"]["prompt"]["selected"]]:
        validate_prompt(p)


def test_postprocess_filters_and_flags(cfgs):
    cfg = cfgs["detector"]["postprocess"]
    W, H = 1000, 800
    boxes = np.array([
        [100, 100, 180, 160],     # ok
        [102, 101, 182, 161],     # duplicate -> NMS
        [0, 300, 60, 360],        # touches left edge -> partial
        [500, 500, 503, 502],     # too small (<0.02%)
        [0, 0, 700, 500],         # too large (>25%)
        [400, 400, 500, 420],     # aspect 5 -> dropped
        [600, 100, 620, 160],     # aspect 3.0 vertical -> kept (orientation free)
    ], float)
    scores = np.array([0.9, 0.8, 0.7, 0.9, 0.9, 0.9, 0.6])
    r = postprocess_boxes(boxes, scores, (W, H), cfg)
    kept = [tuple(d.bbox) for d in r.detections]
    assert (100, 100, 180, 160) in kept and (102, 101, 182, 161) not in kept
    assert r.dropped == {"nms": 1, "size_small": 1, "size_large": 1, "aspect": 1}
    assert [d.partial for d in r.detections if d.bbox[0] == 0] == [True]
    assert any("aspect ratio" in w for w in r.warnings)
    assert len(r.detections) == 3


def test_nms_conservative_keeps_touching_bugs(cfgs):
    boxes = np.array([[100, 100, 200, 160], [180, 100, 280, 160]], float)   # overlap IoU ~0.1
    r = postprocess_boxes(boxes, np.array([0.9, 0.8]), (1000, 800), cfgs["detector"]["postprocess"])
    assert len(r.detections) == 2


def test_crop_padding_clip_and_aspect_preserved():
    assert expand_box((100, 100, 200, 200), 0.15, (1000, 1000)) == (85, 85, 215, 215)
    assert expand_box((0, 0, 50, 50), 0.15, (60, 60)) == (0, 0, 58, 58)
    im = make_image(500, 300)
    sq = pad_to_square(im, 128)
    assert sq.size == (128, 128)
    # bands above/below the resized image are pad colour, i.e. not stretched
    assert sq.getpixel((64, 2)) == sq.getpixel((2, 2)) == (124, 116, 104)
    assert extract_crop(im, (100, 100, 200, 160), 0.15, 96).size == (96, 96)


def test_tiled_detection_merges_across_boundaries():
    # one bug straddles a tile boundary, detected by both overlapping tiles -> counted once
    W, H = 1800, 600
    img = Image.new("RGB", (W, H))
    bug = np.array([400, 100, 470, 160], float)      # lies in overlap of tile0 (0-512) and tile1 (410-922)
    # Emulate per-tile responses: each overlapping tile that wholly contains the bug reports it in tile coordinates
    from src.detection import detect_tiled
    from src.preprocessing.tiling import tile_windows
    wins = tile_windows(W, H, 512, 0.2)
    seq = []
    for x0, y0, x1, y1 in wins:
        if x0 <= bug[0] and bug[2] <= x1 and y0 <= bug[1] and bug[3] <= y1:
            seq.append(np.array([[bug[0] - x0, bug[1] - y0, bug[2] - x0, bug[3] - y0]]))
        else:
            seq.append(np.zeros((0, 4)))
    assert sum(len(s) for s in seq) == 2           # detected twice
    it = iter(seq)

    class Det2:
        name = "d"

        def detect(self, image, prompt, bt, tt):
            b = next(it)
            return b, np.full(len(b), 0.9)

    b, s = detect_tiled(Det2(), img, "stink bug .", 0.25, 0.25,
                        {"tile_size": 512, "overlap_ratio": 0.2, "merge_nms_iou": 0.45})
    assert len(b) == 1 and np.allclose(b[0], bug)


def test_run_detection_triggers(cfgs):
    tcfg = cfgs["prep"]["tiling"]
    d = StubDetector([[10, 10, 50, 40]], [0.9])
    b, s, tiled = run_detection(d, make_image(800, 600), "stink bug .", .25, .25, tcfg)
    assert not tiled and d.calls == 1
    d = StubDetector([[10, 10, 50, 40]], [0.9])
    _, _, tiled = run_detection(d, make_image(1700, 900), "stink bug .", .25, .25, tcfg)
    assert tiled and d.calls > 1
    many = [[i * 30, 10, i * 30 + 20, 30] for i in range(16)]
    d = StubDetector(many, [0.9] * 16)
    _, _, tiled = run_detection(d, make_image(800, 600), "stink bug .", .25, .25, tcfg)
    assert tiled
    d = StubDetector(many, [0.9] * 16)
    _, _, tiled = run_detection(d, make_image(800, 600), "stink bug .", .25, .25, tcfg, force_tiling=False)
    assert not tiled
