"""timm classifier with a species head and an optional shared-backbone life-stage head (spec §7.1, §7.5)."""
from __future__ import annotations

import torch
from torch import nn

# Config uses stable short names; map to timm identifiers (pretrained tag variants differ across timm versions).
TIMM_ALIASES = {
    "efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "vit_base_patch16_384": "vit_base_patch16_384.augreg_in21k_ft_in1k",
    "efficientnet_b3": "efficientnet_b3.ra2_in1k",
}


class StinkBugClassifier(nn.Module):
    def __init__(self, backbone: str, num_species: int, pretrained: bool = True, drop_rate: float = 0.2,
                 life_stage_head: bool = True, num_stages: int = 2, input_size: int | None = None):
        super().__init__()
        import timm

        kwargs = {"img_size": input_size} if backbone.startswith("vit") and input_size else {}
        self.backbone = timm.create_model(TIMM_ALIASES.get(backbone, backbone), pretrained=pretrained,
                                          num_classes=0, drop_rate=drop_rate, **kwargs)
        feat = self.backbone.num_features
        self.species_head = nn.Linear(feat, num_species)
        self.stage_head = nn.Linear(feat, num_stages) if life_stage_head else None

    def forward(self, x: torch.Tensor):
        f = self.backbone(x)
        sp = self.species_head(f)
        return sp, (self.stage_head(f) if self.stage_head is not None else None)

    def head_parameters(self):
        ps = list(self.species_head.parameters())
        if self.stage_head is not None:
            ps += list(self.stage_head.parameters())
        return ps

    def set_backbone_trainable(self, flag: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = flag

    @classmethod
    def from_config(cls, cfg: dict, pretrained: bool | None = None) -> "StinkBugClassifier":
        m = cfg["model"]
        return cls(m["backbone"], len(cfg["classes"]["species"]),
                   m["pretrained"] if pretrained is None else pretrained, m["drop_rate"],
                   m["life_stage_head"], len(cfg["classes"]["life_stages"]), m["input_size"])


class EMA:
    """Exponential moving average of weights (spec §7.2, decay 0.9998)."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.updates = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        # warm-up so early-training EMA is not pinned to the random init: d_t = min(decay, (1+t)/(10+t))
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self) -> dict:
        return self.shadow

    def apply_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


def save_checkpoint(path, model: nn.Module, config: dict, extra: dict | None = None) -> None:
    """Checkpoints always embed the config that produced them (spec §13.2)."""
    torch.save({"state_dict": model.state_dict(), "config": config, **(extra or {})}, path)


def load_checkpoint(path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)
