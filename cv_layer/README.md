# CV layer — stink bug detection + species classification

Implements `../CV_LAYER_SPEC.md`: image → Grounding DINO → box post-processing → crop → EfficientNetV2-S species
classifier → calibrated confidence gating → aggregate features → **schema-validated JSON packet (v1.0)**.

```
python scripts/run_inference.py photo.jpg --out runs/packets
```

## Layout (spec §13)

| Path | Contents |
|---|---|
| `configs/` | Every threshold, path and hyperparameter. `src/` contains none. Override any value with `--set a.b=c`. |
| `src/ingestion` | load / validate / EXIF-transpose / RGB / metadata (HEIC via `pillow-heif`) |
| `src/preprocessing` | 2048-px Lanczos cap, manual tiling (512 px, 20 % overlap), classifier-only albumentations (hue cap ±10 enforced, forbidden transforms asserted) |
| `src/detection` | Grounding DINO wrapper (pinned revision, prompt-format validation), whole/tiled detection, NMS + size/aspect/edge filters, padded aspect-preserving crops |
| `src/classification` | timm model with species head + optional life-stage head, two-phase training, EMA, AMP, temperature scaling/ECE, hard-negative mining, inference |
| `src/gating` | `fused = det^α · cls^(1-α)`, 0.75 / 0.45 bands |
| `src/features` | count, density (null without scale), Clark–Evans spatial distribution, means, partial count |
| `src/output` | pydantic schema v1.0, fail-loud serialization, review queue (crops + `manifest.csv`) |
| `src/evaluation` | detection / classification (7×7 CM, pair accuracy) / end-to-end metrics, §5.3 decision gate, §12.4 acceptance check |
| `src/data_prep` | COCO validation, session-grouped stratified splits, leakage verification |
| `scripts/` | `prepare_dataset`, `prompt_ablation`, `finetune_detector` (gate + head fine-tune), `train_classifier`, `calibrate`, `evaluate`, `run_inference` |
| `docs/output_schema_v1.0.json` | JSON Schema of the frozen output contract |

## Dataset from a flat image folder

`dataset.image_dir` (pipeline.yaml) points at the folder (currently `/home/garik/Downloads/all_images`: 2,273 Roboflow-exported
jpg/png files, no annotation files). `scripts/build_dataset.py` indexes it into `data/annotations/all.json`, which every
other script now reads by default (no `--coco/--images` needed). Session grouping strips Roboflow `_jpg.rf.<hash>` suffixes
and groups `AdltStnkBug_NNN_*` frames together; `domain` is inferred from filename rules (`AdltStnkBug_`/`IMG_` = lab, the
rest = field) — edit `dataset.folder.*` if that is wrong. Species labels are **never guessed**: provide
`dataset.folder.labels_csv` (`file_name,species[,life_stage]`), `label_rules`, or `--default-species`. `--auto-annotate` adds
Grounding DINO boxes flagged `auto_generated`/`human_reviewed=false`; review them before trusting any metric (§11.4).
On this folder only 11.7 % of images are field, so `prepare_dataset.py` correctly warns that the §11.2 40 % field minimum is unmet.

## Workflow (spec §14)

```
python scripts/prepare_dataset.py   --coco data/annotations/all.json
python scripts/prompt_ablation.py   --coco ... --images data/raw --write-config      # selects by recall, val split only
python scripts/finetune_detector.py --coco ... --images data/raw                     # §5.3 GATE: records runs/detector_decision.json
#   STOP here and read the decision. 'fallback_yolov8m_cbam_asff' means do not proceed with Grounding DINO.
python scripts/train_classifier.py  --coco ... --images data/raw --mine              # round 0 + mining
#   review runs/classifier/hard_negative_review_round1.csv (fill `verdict`), then:
python scripts/train_classifier.py  --coco ... --images data/raw --round 1 --verified-csv <csv> --mine
python scripts/calibrate.py         --coco ... --images data/raw
python scripts/evaluate.py          --coco ... --images data/raw --domain field --adversarial-dir data/raw/empty
```
Ablations: `--prompt`, `--tiling on|off`, `train_classifier.py --benchmark-backbones | --resolution-ablation`,
`evaluate.py --baseline-predictions` (YOLOv8m+CBAM+ASFF results). Run tests with `python -m pytest`.

Annotation conventions expected by `prepare_dataset.py`: COCO `images[].domain` (`lab`|`field`), `session_id`,
optional `empty_background`; `annotations[].annotator_id`, `life_stage`, `human_reviewed`,
`species_verified_by_entomologist` (required for field images); categories = the six trained classes
(never `uncertain`). Box coordinates are in the EXIF-corrected image frame.

## Interpretation choices worth knowing

- **Output boxes are in original-image pixels** (`bbox_format: xyxy_absolute`), even though detection and crops use the ≤2048-px copy.
- **Aspect filter** (§6.3) is applied orientation-free (`max(w,h)/min(w,h) ≤ 3.0`): boxes are axis-aligned and bugs sit at any rotation, so the literal 0.5–3.0 window on w/h would reject vertical bugs.
- **Partial boxes** (touching the border within 2 px) have `detection_confidence` multiplied by 0.85 *in the emitted value*, so fusion and the packet agree.
- `uncertain` detections keep their box/count, emit `species_scientific: null`, `flag: null`, and are written to the review queue with the model's best guess in the manifest.
- `count_by_species` always contains an `uncertain` key. `spatial_distribution` is `clustered` (Clark–Evans R<1), `dispersed`, or `null` for <3 detections (schema has two states).
- Test split is rebalanced by whole-group swaps to ≥60 % field images (`dataset.min_field_fraction_test`); `prepare_dataset.py` warns if train <40 % field.
- EMA decay is warmed up (`min(0.9998, (1+t)/(10+t))`); a flat 0.9998 leaves the EMA pinned near initialization on small datasets.
- Life-stage head is trained only if ≥ `train.min_labeled_nymphs` labeled crops exist; otherwise the checkpoint has no head and packets say `life_stage: "unknown"`.

## Status / what has and has not been verified

Verified here (CPU, `pytest`, 74 tests): ingestion, tiling, post-processing, gating, features, schema violations, splits & leakage, metrics, temperature scaling, training loop + checkpoint/config round-trip, hard-negative workflow, pipeline end-to-end with stubs. Also run for real: the pinned Grounding DINO base weights through `GroundingDinoDetector`, and `run_inference.py` through the full pipeline on a synthetic image with a toy classifier trained by `train_classifier.py`.

**Not done, because it needs inputs that do not exist in this repo (spec §16):** no stink-bug data, so no trained species classifier, no prompt-ablation winner (`prompt.selected` is the spec's second candidate as a placeholder, `ablation_scores` empty), no §5.3 decision, no hard-negative rounds, no acceptance numbers, no YOLOv8m+CBAM+ASFF baseline (the fallback architecture is referenced but not implemented — only needed if the gate lands below 0.70). `finetune_detector.py`'s head fine-tuning loop follows the HF Grounding DINO loss API but has not been run. Latency/Jetson profiling needs the target hardware.

`requirements.lock` was frozen from the CPU test environment (`torch+cpu`); regenerate with `scripts/lock.sh` in the CUDA training environment. `sahi` is listed as optional — tiling is implemented in-repo.


## Implementation and Execution
`python -m venv .venv && source .venv/bin/activate`
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
`pip install -r requirements.txt`
`python -m pytest -q`

`python scripts/build_dataset.py --default-species rbsb --auto-annotate   # or omit --default-species and set dataset.folder.labels_csv`
`python scripts/prepare_dataset.py --allow-unreviewed`
`python scripts/prompt_ablation.py --write-config`
`python scripts/finetune_detector.py --evaluate-only`                      # §5.3 decision gate: read runs/detector_decision.json before continuing

`python scripts/train_classifier.py --mine --set train.num_workers=8`
`python scripts/calibrate.py`
`python scripts/evaluate.py --domain all`
