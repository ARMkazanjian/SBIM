import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfgs():
    return {"pipeline": load_config("configs/pipeline.yaml"), "detector": load_config("configs/detector.yaml"),
            "classifier": load_config("configs/classifier.yaml"),
            "prep": load_config("configs/preprocessing.yaml")}


def make_image(w=640, h=480, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


@pytest.fixture
def img():
    return make_image()


def jpeg_bytes(im: Image.Image, exif=None) -> bytes:
    b = io.BytesIO()
    im.save(b, "JPEG", exif=exif) if exif is not None else im.save(b, "JPEG")
    return b.getvalue()


class StubDetector:
    """Returns fixed boxes; for tiles returns those boxes falling in the window (shifted)."""
    name = "stub-detector"

    def __init__(self, boxes, scores):
        self.boxes, self.scores = np.array(boxes, float).reshape(-1, 4), np.array(scores, float)
        self.calls = 0

    def detect(self, image, prompt, box_threshold, text_threshold):
        self.calls += 1
        return self.boxes.copy(), self.scores.copy()


class StubClassifier:
    classes = ["rbsb", "sgsb", "gsb", "bmsb", "bsb", "other_pentatomid"]
    stages = ["adult", "nymph"]
    name = "stub-classifier"
    has_stage_head = True

    def __init__(self, probs_rows):
        self.rows = np.array(probs_rows, float)

    def classify(self, crops):
        n = len(crops)
        p = np.stack([self.rows[i % len(self.rows)] for i in range(n)])
        st = np.tile([0.8, 0.2], (n, 1))
        return p, st
