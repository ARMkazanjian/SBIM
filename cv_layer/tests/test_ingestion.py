import io

import pytest
from PIL import Image

from src.ingestion import IngestionError, iter_directory, load_bytes, load_image
from tests.conftest import jpeg_bytes, make_image


def test_exif_orientation_applied(tmp_path):
    im = make_image(400, 300)
    exif = Image.Exif()
    exif[0x0112] = 6          # rotate 90 CW on display
    exif[0x0110] = "iPhone 15 Pro Max"
    exif[0x010F] = "Apple"
    p = tmp_path / "a.jpg"
    p.write_bytes(jpeg_bytes(im, exif.tobytes()))
    li = load_image(p)
    assert li.image.size == (300, 400)
    assert li.metadata.original_dimensions == (300, 400)
    assert li.metadata.device == "Apple iPhone 15 Pro Max"
    assert li.metadata.gps is None and li.metadata.capture_timestamp is None


def test_missing_metadata_is_null_not_fabricated():
    li = load_bytes(jpeg_bytes(make_image()))
    assert li.metadata.gps is None and li.metadata.device is None


def test_small_image_rejected(tmp_path):
    p = tmp_path / "s.png"
    make_image(200, 500).save(p)
    with pytest.raises(IngestionError):
        load_image(p)


def test_corrupt_rejected(tmp_path):
    p = tmp_path / "c.jpg"
    p.write_bytes(b"not an image")
    with pytest.raises(IngestionError):
        load_image(p)
    with pytest.raises(IngestionError):
        load_bytes(b"\x00\x01")


def test_rgba_and_palette_to_rgb():
    b = io.BytesIO()
    Image.new("RGBA", (300, 300), (255, 0, 0, 128)).save(b, "PNG")
    li = load_bytes(b.getvalue())
    assert li.image.mode == "RGB"
    b = io.BytesIO()
    make_image(300, 300).convert("L").save(b, "PNG")
    assert load_bytes(b.getvalue()).image.mode == "RGB"


def test_directory_batch_logs_rejections(tmp_path):
    make_image().save(tmp_path / "ok.png")
    (tmp_path / "bad.jpg").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("skip")
    out = list(iter_directory(tmp_path, [".jpg", ".png"]))
    assert sum(isinstance(o, tuple) for o in out) == 1
    assert sum(not isinstance(o, tuple) for o in out) == 1
