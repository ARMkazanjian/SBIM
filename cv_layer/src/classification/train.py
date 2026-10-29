"""Classifier training loop (spec §7.2, §7.3): two-phase, discriminative LR, warmup+cosine, EMA, AMP, early stopping."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ..common.logging import get_logger
from .calibration import calibrate, save_calibration
from .dataset import CropDataset, Sample, class_loss_weights, make_sampler
from .model import EMA, StinkBugClassifier, save_checkpoint

log = get_logger(__name__)


def lr_factor(epoch_float: float, warmup: int, total: int) -> float:
    if epoch_float < warmup:
        return (epoch_float + 1e-8) / warmup if warmup else 1.0
    prog = (epoch_float - warmup) / max(1e-8, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


def build_optimizer(model: StinkBugClassifier, tcfg: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW([
        {"params": model.head_parameters(), "lr": tcfg["lr_head"], "base_lr": tcfg["lr_head"]},
        {"params": model.backbone.parameters(), "lr": tcfg["lr_backbone"], "base_lr": tcfg["lr_backbone"]},
    ], weight_decay=tcfg["weight_decay"])


@torch.no_grad()
def predict(model: StinkBugClassifier, loader: DataLoader, device) -> dict:
    model.eval()
    sp, st, ys, ystage, idx = [], [], [], [], []
    for x, y, stage, i in loader:
        a, b = model(x.to(device))
        sp.append(a.float().cpu())
        st.append(b.float().cpu() if b is not None else torch.zeros(len(y), 2))
        ys.append(y)
        ystage.append(stage)
        idx.append(i)
    return {"species_logits": torch.cat(sp).numpy(), "stage_logits": torch.cat(st).numpy(),
            "y": torch.cat(ys).numpy(), "stage_y": torch.cat(ystage).numpy(), "idx": torch.cat(idx).numpy()}


def macro_f1(y: np.ndarray, logits: np.ndarray, n_classes: int) -> float:
    present = sorted(set(y.tolist()))
    return float(f1_score(y, logits.argmax(1), labels=present, average="macro", zero_division=0)) if len(y) else 0.0


def train_classifier(cfg: dict, pre_cfg: dict, train_samples: list[Sample], val_samples: list[Sample],
                     tracker, out_dir: str | Path, padding: float, augment=None, device: str | None = None,
                     pretrained: bool | None = None, full_config: dict | None = None) -> dict:
    tcfg = cfg["train"]
    if tcfg["imbalance"] not in ("weighted_sampler", "class_weighted_loss"):
        raise ValueError("train.imbalance must be exactly one of weighted_sampler | class_weighted_loss")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n_cls = len(cfg["classes"]["species"])
    size = cfg["model"]["input_size"]

    n_stage_labels = sum(s.life_stage >= 0 for s in train_samples)
    use_stage = cfg["model"]["life_stage_head"] and n_stage_labels >= tcfg["min_labeled_nymphs"]
    if cfg["model"]["life_stage_head"] and not use_stage:
        log.warning("life_stage_head_disabled", extra={"labeled": n_stage_labels, "needed": tcfg["min_labeled_nymphs"]})
    cfg = copy.deepcopy(cfg)
    cfg["model"]["life_stage_head"] = bool(use_stage)

    model = StinkBugClassifier.from_config(cfg, pretrained).to(device)
    ema = EMA(model, tcfg["ema_decay"])
    train_ds = CropDataset(train_samples, size, padding, augment)
    val_ds = CropDataset(val_samples, size, padding, None)
    sampler = make_sampler(train_samples, n_cls) if tcfg["imbalance"] == "weighted_sampler" else None
    bs = tcfg["batch_size"]
    train_dl = DataLoader(train_ds, batch_size=bs, sampler=sampler, shuffle=sampler is None,
                          num_workers=tcfg["num_workers"], drop_last=len(train_ds) > bs)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=tcfg["num_workers"])
    cw = class_loss_weights(train_samples, n_cls).to(device) if tcfg["imbalance"] == "class_weighted_loss" else None
    opt = build_optimizer(model, tcfg)
    scaler = torch.amp.GradScaler(enabled=tcfg["amp"] and device.type == "cuda")
    best_f1, best_epoch, bad = -1.0, -1, 0
    accum = tcfg["grad_accum_steps"]
    steps_per_epoch = max(1, len(train_dl))
    history = []

    for epoch in range(tcfg["epochs"]):
        phase = 1 if epoch < tcfg["phase1_head_only_epochs"] else 2
        model.set_backbone_trainable(phase == 2)
        model.train()
        if phase == 1:
            model.backbone.eval()      # keep BN statistics of the frozen backbone stable
        run_loss = 0.0
        opt.zero_grad(set_to_none=True)
        for step, (x, y, stage, _) in enumerate(train_dl):
            f = lr_factor(epoch + step / steps_per_epoch, tcfg["warmup_epochs"], tcfg["epochs"])
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            x, y, stage = x.to(device), y.to(device), stage.to(device)
            with torch.autocast(device_type=device.type, enabled=tcfg["amp"] and device.type == "cuda"):
                sp, st = model(x)
                loss = F.cross_entropy(sp.float(), y, weight=cw, label_smoothing=tcfg["label_smoothing"])
                if st is not None and (stage >= 0).any():
                    m = stage >= 0
                    loss = loss + tcfg["life_stage_loss_weight"] * F.cross_entropy(st[m].float(), stage[m])
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0 or step + 1 == len(train_dl):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update(model)
            run_loss += float(loss.item())

        # validate with EMA weights
        live = copy.deepcopy(model.state_dict())
        ema.apply_to(model)
        pred = predict(model, val_dl, device)
        vf1 = macro_f1(pred["y"], pred["species_logits"], n_cls)
        rec = {"epoch": epoch, "phase": phase, "train_loss": run_loss / steps_per_epoch, "val_macro_f1": vf1,
               "val_acc": float((pred["species_logits"].argmax(1) == pred["y"]).mean()) if len(pred["y"]) else None}
        tracker.log(rec, step=epoch)
        history.append(rec)
        if vf1 > best_f1:
            best_f1, best_epoch, bad = vf1, epoch, 0
            save_checkpoint(out_dir / "best.pt", model, full_config or cfg,
                            {"epoch": epoch, "val_macro_f1": vf1, "life_stage_enabled": bool(use_stage)})
        else:
            bad += 1
        model.load_state_dict(live)       # resume raw weights
        if bad >= tcfg["patience"]:
            log.info("early_stop", extra={"epoch": epoch, "best_epoch": best_epoch})
            break

    # calibration on val with best weights
    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    pred = predict(model, val_dl, device)
    cal = calibrate(pred["species_logits"], pred["y"], cfg["calibration"]["n_bins"], cfg["calibration"]["max_iter"]) \
        if len(pred["y"]) else {"temperature": 1.0}
    save_calibration(cal, out_dir / "calibration.json")
    return {"best_val_macro_f1": best_f1, "best_epoch": best_epoch, "history": history, "calibration": cal,
            "life_stage_enabled": bool(use_stage)}
