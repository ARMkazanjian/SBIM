"""Classifier-only training augmentation (spec §4.4). Forbidden transforms are enforced, not just avoided."""
from __future__ import annotations

import albumentations as A

FORBIDDEN = ("ChannelShuffle", "ToGray", "InvertImg", "MixUp", "CutMix", "Solarize", "RandomGamma_extreme")
MAX_HUE_SHIFT = 10


def build_train_augmentation(cfg: dict, input_size: int) -> A.Compose:
    """cfg is the ``augmentation`` section of preprocessing.yaml."""
    if cfg["hue_shift_limit"] > MAX_HUE_SHIFT:
        raise ValueError(f"hue_shift_limit {cfg['hue_shift_limit']} exceeds the +/-{MAX_HUE_SHIFT} cap (spec §4.4)")
    ph, pg = cfg["p_photometric"], cfg["p_geometric"]
    nv = cfg["noise_var_limit"]
    noise_std = (max(0.01, nv[0] ** 0.5 / 255), max(0.02, nv[1] ** 0.5 / 255))
    drop_frac = cfg["dropout_max_fraction"]
    tfms = [
        A.RandomRotate90(p=pg),
        A.Rotate(limit=cfg["rotate_limit"], border_mode=0, p=pg),
        A.RandomBrightnessContrast(brightness_limit=cfg["brightness_limit"],
                                   contrast_limit=cfg["contrast_limit"], p=ph),
        A.HueSaturationValue(hue_shift_limit=cfg["hue_shift_limit"], sat_shift_limit=cfg["sat_shift_limit"],
                             val_shift_limit=cfg["val_shift_limit"], p=ph),
        A.OneOf([A.GaussNoise(std_range=noise_std, p=1.0), A.ISONoise(p=1.0)], p=0.3),
        A.MotionBlur(blur_limit=tuple(cfg["motion_blur_limit"]), p=0.2),
        A.RandomShadow(p=cfg["shadow_p"]),
        A.RandomSunFlare(src_radius=input_size // 4, p=cfg["sun_flare_p"]),
        A.CoarseDropout(num_holes_range=(1, cfg["dropout_max_holes"]),
                        hole_height_range=(0.02, drop_frac), hole_width_range=(0.02, drop_frac),
                        fill=0, p=0.3),
    ]
    names = {type(t).__name__ for t in tfms}
    assert not names.intersection(FORBIDDEN), f"forbidden augmentation present: {names & set(FORBIDDEN)}"
    return A.Compose(tfms)
