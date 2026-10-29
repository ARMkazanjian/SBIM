from __future__ import annotations

import os
import random

import numpy as np

from .logging import get_logger

log = get_logger(__name__)


def seed_everything(seed: int) -> int:
    """Seed python, numpy and (if present) torch; log the seed (spec §13.2)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch-free environments (e.g. data prep only)
        pass
    log.info("seed_set", extra={"seed": seed})
    return seed
