"""Stage 0 ingestion: load, validate, EXIF-normalize, capture metadata (spec §4.1, §4.2)."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageOps, UnidentifiedImageError

from ..common.logging import get_logger

log = get_logger(__name__)

_EXIF_IFD, _GPS_IFD = 0x8769, 0x8825
_TAG_MODEL, _TAG_DATETIME, _TAG_DT_ORIG = 0x0110, 0x0132, 0x9003
_TAG_OFFSET_ORIG, _TAG_MAKE = 0x9011, 0x010F


class IngestionError(Exception):
    """Raised for inputs that are rejected (corrupt, too small, unconvertible)."""


@dataclass
class SourceMetadata:
    original_dimensions: tuple[int, int]            # (width, height) after EXIF orientation
    capture_timestamp: str | None = None
    gps: dict | None = None
    device: str | None = None

    def to_dict(self) -> dict:
        return {"original_dimensions": list(self.original_dimensions),
                "capture_timestamp": self.capture_timestamp, "gps": self.gps, "device": self.device}


@dataclass
class LoadedImage:
    image_id: str
    image: Image.Image                              # RGB, EXIF-transposed, NOT yet resized
    metadata: SourceMetadata
    source: str = ""
    warnings: list[str] = field(default_factory=list)


def _register_heif() -> None:
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        log.warning("pillow_heif_missing", extra={"detail": "HEIC/HEIF files cannot be opened"})


_register_heif()


def _dms_to_deg(dms, ref) -> float | None:
    try:
        d, m, s = (float(x) for x in dms)
        deg = d + m / 60 + s / 3600
        return -deg if str(ref).upper() in ("S", "W") else deg
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def extract_metadata(img: Image.Image, dims: tuple[int, int]) -> SourceMetadata:
    """Pull timestamp / GPS / device from EXIF. Missing fields stay None (never fabricated)."""
    ts = gps = device = None
    try:
        exif = img.getexif()
    except Exception:
        exif = None
    if exif:
        sub = {}
        try:
            sub = dict(exif.get_ifd(_EXIF_IFD))
        except Exception:
            pass
        raw_ts = sub.get(_TAG_DT_ORIG) or exif.get(_TAG_DATETIME)
        if raw_ts:
            try:
                dt = datetime.strptime(str(raw_ts).strip(), "%Y:%m:%d %H:%M:%S")
                off = sub.get(_TAG_OFFSET_ORIG)
                if off and len(str(off)) >= 6:   # e.g. "-05:00"
                    s = str(off)
                    sign = -1 if s[0] == "-" else 1
                    h, m = int(s[1:3]), int(s[4:6])
                    from datetime import timedelta
                    dt = (dt - sign * timedelta(hours=h, minutes=m)).replace(tzinfo=timezone.utc)
                ts = dt.isoformat().replace("+00:00", "Z") if dt.tzinfo else dt.isoformat()
            except ValueError:
                pass
        model, make = exif.get(_TAG_MODEL), exif.get(_TAG_MAKE)
        if model:
            model, make = str(model).strip("\x00 "), str(make or "").strip("\x00 ")
            device = model if (not make or model.lower().startswith(make.lower())) else f"{make} {model}"
        try:
            g = dict(exif.get_ifd(_GPS_IFD))
        except Exception:
            g = {}
        if g.get(2) and g.get(4):
            lat, lon = _dms_to_deg(g[2], g.get(1)), _dms_to_deg(g[4], g.get(3))
            if lat is not None and lon is not None:
                gps = {"lat": round(lat, 6), "lon": round(lon, 6)}
    return SourceMetadata(original_dimensions=dims, capture_timestamp=ts, gps=gps, device=device)


def _load_from_fp(fp, image_id: str, source: str, min_side: int) -> LoadedImage:
    try:
        img = Image.open(fp)
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise IngestionError(f"corrupt or unreadable image {source!r}: {e}") from e
    metadata_src = img
    img = ImageOps.exif_transpose(img)                      # step 1: EXIF orientation, before anything else
    # exif_transpose strips orientation but keeps other EXIF on the result; fall back to the original handle.
    dims = img.size
    meta = extract_metadata(metadata_src, dims)
    if min(dims) < min_side:
        raise IngestionError(f"image {source!r} is {dims[0]}x{dims[1]}, under minimum {min_side}px per side")
    try:                                                    # step 2: RGB, no alpha
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.getchannel("A"))
            img = bg
        else:
            img = img.convert("RGB")
    except (ValueError, OSError) as e:
        raise IngestionError(f"cannot convert {source!r} (mode {img.mode}) to RGB: {e}") from e
    return LoadedImage(image_id=image_id, image=img, metadata=meta, source=source)


def load_image(path: str | Path, min_side: int = 224) -> LoadedImage:
    path = Path(path)
    if not path.is_file():
        raise IngestionError(f"not a file: {path}")
    with open(path, "rb") as fp:
        return _load_from_fp(fp, path.stem, str(path), min_side)


def load_bytes(data: bytes, image_id: str = "buffer", min_side: int = 224) -> LoadedImage:
    return _load_from_fp(io.BytesIO(data), image_id, f"<bytes:{image_id}>", min_side)


def iter_directory(directory: str | Path, extensions: list[str], min_side: int = 224,
                   recursive: bool = False) -> Iterator[LoadedImage | tuple[Path, str]]:
    """Batch mode. Yields LoadedImage, or ``(path, reason)`` for rejected files (rejections are logged)."""
    directory = Path(directory)
    exts = {e.lower() for e in extensions}
    files = sorted(p for p in (directory.rglob("*") if recursive else directory.iterdir())
                   if p.is_file() and p.suffix.lower() in exts)
    for p in files:
        try:
            yield load_image(p, min_side)
        except IngestionError as e:
            log.error("image_rejected", extra={"path": str(p), "reason": str(e)})
            yield (p, str(e))
