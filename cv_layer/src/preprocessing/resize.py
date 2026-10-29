"""Resolution handling (spec §4.2 step 3). Never downscale to 640; cap long edge only."""
from __future__ import annotations

from PIL import Image


def downscale_long_edge(img: Image.Image, max_long_edge: int) -> tuple[Image.Image, float]:
    """Lanczos downscale so the long edge <= max_long_edge. Returns (image, scale) with scale <= 1."""
    w, h = img.size
    long_edge = max(w, h)
    if long_edge <= max_long_edge:
        return img, 1.0
    scale = max_long_edge / long_edge
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS), scale
