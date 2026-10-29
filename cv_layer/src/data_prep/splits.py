"""Leak-free dataset splitting (spec §11.3): split by capture session / source image, never by crop,
stratified by species and source domain."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def load_coco(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def group_key(img: dict) -> str:
    """Capture session if known, else the source image itself."""
    return str(img.get("session_id") or img.get("source_id") or img["id"])


def build_groups(coco: dict) -> dict[str, dict]:
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    per_img = defaultdict(list)
    for a in coco["annotations"]:
        per_img[a["image_id"]].append(cats[a["category_id"]])
    groups: dict[str, dict] = {}
    for img in coco["images"]:
        g = groups.setdefault(group_key(img), {"image_ids": [], "species": defaultdict(int), "domains": set()})
        g["image_ids"].append(img["id"])
        g["domains"].add(img.get("domain", "unknown"))
        for s in per_img.get(img["id"], []):
            g["species"][s] += 1
    return groups


def _stratum(g: dict) -> tuple[str, str]:
    dom = "field" if "field" in g["domains"] else sorted(g["domains"])[0]
    sp = max(g["species"], key=g["species"].get) if g["species"] else "__none__"
    return dom, sp


def split_dataset(coco: dict, ratios: dict[str, float], seed: int, min_test_field: float | None = None,
                  min_train_field: float = 0.0) -> dict[str, list]:
    """Returns {'train': [image_id...], 'val': [...], 'test': [...]}. Groups are never divided.

    If ``min_test_field`` is given, lab groups in the test split are swapped with field groups from train (same
    image count preferred) until the test set is at least that field fraction (spec §11.2: test must be
    predominantly field imagery) — but never if that would push the train field fraction below ``min_train_field``.
    When field data is too scarce for both, the stratified split is kept and the shortfall is reported by
    ``domain_report`` rather than silently robbing train.
    """
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1")
    groups = build_groups(coco)
    strata: dict[tuple, list[str]] = defaultdict(list)
    for k, g in groups.items():
        strata[_stratum(g)].append(k)
    rng = random.Random(seed)
    names = ["train", "val", "test"]
    out: dict[str, list] = {n: [] for n in names}
    assigned = {n: 0 for n in names}
    total_imgs = sum(len(g["image_ids"]) for g in groups.values())
    for key in sorted(strata, key=str):
        members = sorted(strata[key])
        rng.shuffle(members)
        n = len(members)
        # per-stratum quotas by largest remainder; tiny strata go to train, then fill val/test by global deficit
        quota = {s: ratios[s] * n for s in names}
        take = {s: int(quota[s]) for s in names}
        for s in sorted(names, key=lambda s: quota[s] - take[s], reverse=True)[: n - sum(take.values())]:
            take[s] += 1
        if n >= 3:                      # keep representation of rare strata in eval splits where possible
            for s in ("val", "test"):
                if take[s] == 0:
                    donor = max(names, key=lambda x: take[x])
                    take[donor] -= 1
                    take[s] += 1
        i = 0
        for s in names:
            for gk in members[i:i + take[s]]:
                out[s] += groups[gk]["image_ids"]
                assigned[s] += len(groups[gk]["image_ids"])
            i += take[s]
    if min_test_field is not None:
        _enforce_field_test(coco, groups, out, min_test_field, min_train_field, rng)
    for s in out:
        out[s].sort()
    return out


def _enforce_field_test(coco, groups, out, min_frac, min_train_field, rng) -> None:
    imgs = {i["id"]: i for i in coco["images"]}
    where = {gk: ("test" if g["image_ids"][0] in set(out["test"]) else "train" if g["image_ids"][0] in set(out["train"])
                  else "val") for gk, g in groups.items()}
    is_field = lambda gk: "field" in groups[gk]["domains"]
    n = lambda gk: len(groups[gk]["image_ids"])

    def frac(split):
        t = [i for gk in groups if where[gk] == split for i in groups[gk]["image_ids"]]
        return sum(imgs[i].get("domain") == "field" for i in t) / len(t) if t else 1.0

    lab_test = sorted((gk for gk in groups if where[gk] == "test" and not is_field(gk)), key=lambda g: (n(g), g))
    field_train = sorted((gk for gk in groups if where[gk] == "train" and is_field(gk)), key=lambda g: (n(g), g))
    rng.shuffle(lab_test)
    while frac("test") < min_frac and lab_test and field_train:
        lt = lab_test.pop()
        ft = min(field_train, key=lambda g: (abs(n(g) - n(lt)), g))
        where[lt], where[ft] = "train", "test"
        if frac("train") < min_train_field:                 # would starve train: undo and stop
            where[lt], where[ft] = "test", "train"
            break
        field_train.remove(ft)
    for s in ("train", "test"):
        out[s] = [i for gk in groups if where[gk] == s for i in groups[gk]["image_ids"]]


def verify_no_leakage(coco: dict, splits: dict[str, list]) -> list[str]:
    """Return a list of violations (empty == clean)."""
    imgs = {i["id"]: i for i in coco["images"]}
    problems = []
    seen: dict = {}
    for s, ids in splits.items():
        for i in ids:
            if i in seen and seen[i] != s:
                problems.append(f"image {i} appears in {seen[i]} and {s}")
            seen[i] = s
    group_split: dict[str, set] = defaultdict(set)
    for i, s in seen.items():
        group_split[group_key(imgs[i])].add(s)
    problems += [f"group {g!r} spans splits {sorted(v)}" for g, v in group_split.items() if len(v) > 1]
    missing = set(imgs) - set(seen)
    if missing:
        problems.append(f"{len(missing)} images unassigned")
    return problems


def domain_report(coco: dict, splits: dict[str, list], min_train_field: float, min_test_field: float = 0.5) -> dict:
    imgs = {i["id"]: i for i in coco["images"]}
    rep, warns = {}, []
    for s, ids in splits.items():
        n = len(ids)
        f = sum(imgs[i].get("domain") == "field" for i in ids)
        rep[s] = {"images": n, "field": f, "field_fraction": f / n if n else None}
    if rep["train"]["field_fraction"] is None or rep["train"]["field_fraction"] < min_train_field:
        warns.append(f"train field fraction {rep['train']['field_fraction']} < required {min_train_field} (spec §11.2)")
    if rep["test"]["field_fraction"] is None or rep["test"]["field_fraction"] < min_test_field:
        warns.append(f"test set is not predominantly field imagery ({rep['test']['field_fraction']})")
    return {"splits": rep, "warnings": warns}


def field_only_test(coco: dict, splits: dict[str, list]) -> list:
    imgs = {i["id"]: i for i in coco["images"]}
    return [i for i in splits["test"] if imgs[i].get("domain") == "field"]


def annotator_agreement(coco: dict) -> dict:
    """Simple inter-annotator agreement over boxes annotated by >=2 annotators (same image, IoU>=0.5, category match)."""
    from ..common.boxes import iou_matrix
    import numpy as np

    by_img = defaultdict(lambda: defaultdict(list))
    for a in coco["annotations"]:
        if a.get("annotator_id") is not None:
            x, y, w, h = a["bbox"]
            by_img[a["image_id"]][a["annotator_id"]].append(([x, y, x + w, y + h], a["category_id"]))
    pairs = agree = 0
    for per in by_img.values():
        ann = sorted(per)
        for ai in range(len(ann)):
            for bi in range(ai + 1, len(ann)):
                A, B = per[ann[ai]], per[ann[bi]]
                m = iou_matrix(np.array([a[0] for a in A]), np.array([b[0] for b in B]))
                for r in range(len(A)):
                    c = int(m[r].argmax()) if m.shape[1] else -1
                    if c >= 0 and m[r, c] >= 0.5:
                        pairs += 1
                        agree += A[r][1] == B[c][1]
    return {"matched_pairs": pairs, "label_agreement": agree / pairs if pairs else None}
