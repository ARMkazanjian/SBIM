"""Shared bootstrap for scripts: make the package importable, seed, configure logging."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PARENT = str(ROOT)  # scripts import the package as `src`


def bootstrap(seed: int | None, level: str = "INFO") -> None:
    if PARENT not in sys.path:
        sys.path.insert(0, PARENT)
    from .logging import configure_logging
    from .seed import seed_everything
    configure_logging(level, True)
    if seed is not None:
        seed_everything(seed)
