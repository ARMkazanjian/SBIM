"""COCO validation under the expanded 7-class label schema (spec §11.4)."""
from __future__ import annotations


def validate_coco(coco: dict, species: list[str], require_annotator: bool = True) -> list[str]:
    problems = []
    cat_names = {c["name"] for c in coco.get("categories", [])}
    for s in species:
        if s not in cat_names:
            problems.append(f"missing category {s!r}")
    if "uncertain" in cat_names:
        problems.append("'uncertain' is an output state, not a trainable class; remove it from annotations")
    imgs = {i["id"]: i for i in coco["images"]}
    for i in imgs.values():
        if i.get("domain") not in ("lab", "field"):
            problems.append(f"image {i['id']}: domain must be 'lab' or 'field'")
    ids = {c["id"]: c["name"] for c in coco["categories"]}
    for a in coco["annotations"]:
        if a["image_id"] not in imgs:
            problems.append(f"annotation {a['id']}: unknown image")
        if a["category_id"] not in ids:
            problems.append(f"annotation {a['id']}: unknown category")
        if len(a["bbox"]) != 4 or a["bbox"][2] <= 0 or a["bbox"][3] <= 0:
            problems.append(f"annotation {a['id']}: bad bbox {a['bbox']}")
        if require_annotator and not a.get("annotator_id"):
            problems.append(f"annotation {a['id']}: missing annotator_id")
        if a.get("auto_generated") and not a.get("human_reviewed"):
            problems.append(f"annotation {a['id']}: auto-generated and not human reviewed (spec §11.4)")
        img = imgs.get(a["image_id"], {})
        if img.get("domain") == "field" and ids.get(a["category_id"]) and not a.get("species_verified_by_entomologist"):
            problems.append(f"annotation {a['id']}: field species label not entomologist-verified")
    return problems
