# Computer Vision Layer — Implementation Specification

**Project:** Real-Time Stink Bug Identification and Management (DSFAS)
**Scope of this document:** CV layer only. The LLM reasoning layer is out of scope and is specified separately. This layer's sole responsibility is to turn an image into a validated, structured JSON detection packet.
**Audience:** Claude Code, for implementation, training, and evaluation.

---

## 1. Objective

Build a two-stage computer vision pipeline that, given a field or lab image of soybean stink bugs, produces:

1. Bounding-box localization of every stink bug present
2. Species-level classification of each detection
3. Per-image aggregate features (count, density, life stage where inferable)
4. A structured JSON packet consumed downstream by the LLM layer

The pipeline must handle both controlled imagery (top-down, flat surface, sweep-net samples) and uncontrolled field imagery (variable lighting, cluttered foliage backgrounds, occlusion).

### 1.1 Design rationale — why two stages

A single closed-vocabulary detector couples detection and classification. That coupling is a liability here for two reasons:

- **New species require full retraining.** The target species list will expand over the project lifetime, and the framework is intended to generalize to other pests.
- **Species discrimination is a finer-grained problem than localization.** Stink bug species share near-identical silhouettes; the discriminative signal is coloration, banding, scutellum shape, and connexivum pattern — features better learned by a dedicated classifier operating on a tight crop than by a detection head optimizing a joint loss.

The architecture therefore separates **"is there a stink bug here"** (open-vocabulary detector) from **"which species is it"** (fine-grained classifier).

### 1.2 Prior art this builds on

Two reference points constrain the design:

| Source | Contribution | What we take |
|---|---|---|
| Upadhyaya et al. 2026 (Front. Artif. Intell.) | YOLOv8m + CBAM + ASFF on 2,281 RBSB images: 96.30% precision, 76.40% recall, 82.00% mAP50, 75.30% mAP50-95, MAE 3.15, 15.9 ms inference | Baseline to beat. Their metrics are the acceptance floor. Their CBAM/ASFF enhancement is a validated fallback architecture. |
| Project proposal preliminary data | YOLOv8m on 1,008 lab RBSB images: 99.4% mAP50, 99.6% precision, 98.7% counting accuracy | Demonstrates ceiling under controlled conditions. The lab-to-field gap is the central technical risk. |

Both prior results are **single-species, RBSB only**. This spec extends to five species.

---

## 2. Target species

| Common name | Scientific name | Label | Notes |
|---|---|---|---|
| Redbanded stink bug | *Piezodorus guildinii* | `rbsb` | Primary target. Lowest action threshold, most economically damaging. |
| Southern green stink bug | *Nezara viridula* | `sgsb` | Visually confusable with green stink bug. |
| Green stink bug | *Chinavia hilaris* | `gsb` | Visually confusable with southern green. |
| Brown marmorated stink bug | *Halyomorpha halys* | `bmsb` | Distinctive antennal banding. |
| Brown stink bug | *Euschistus servus* | `bsb` | Confusable with BMSB at low resolution. |

**Two additional classes required:**

- `other_pentatomid` — a stink bug that is not one of the five targets (e.g. *Thyanta custator*, red-shouldered stink bug, a known RBSB confusion case in the literature)
- `uncertain` — reserved output state when classifier confidence falls below threshold; not a trained class, assigned by post-processing

The `other_pentatomid` class is **not optional**. Without it, the classifier is forced to assign every detection to one of five species, which converts an out-of-distribution input into a confident wrong answer. That failure mode is exactly what causes misapplied pesticide — the problem this project exists to solve.

### 2.1 Confusion pairs requiring explicit attention

These pairs must be reported separately in evaluation and must have dedicated hard-negative mining:

- `sgsb` ↔ `gsb` (both uniformly green, differ in connexivum spotting and scutellum tip)
- `bmsb` ↔ `bsb` (both mottled brown, differ in antennal banding and abdominal edge pattern)
- `rbsb` ↔ `other_pentatomid` (specifically *T. custator*, which also has a reddish transverse band)

---

## 3. Pipeline architecture

```
Image ingestion
      ↓
Preprocessing & EXIF normalization
      ↓
Stage 1: Grounding DINO (open-vocabulary detection)
      ↓
Box post-processing (NMS, size/aspect filtering)
      ↓
Crop extraction (with context padding)
      ↓
Stage 2: Species classifier (fine-grained CNN)
      ↓
Feature extraction (count, density, life stage)
      ↓
Confidence gating & uncertainty routing
      ↓
JSON output packet
```

Each stage is implemented as an independently testable module with a defined input/output contract. No stage may reach into another stage's internals.

---

## 4. Stage 0 — Ingestion and preprocessing

### 4.1 Supported inputs

- Single image file (JPEG, PNG, HEIC — HEIC requires `pillow-heif`)
- Directory of images (batch mode)
- Byte stream / in-memory buffer (for the eventual API endpoint)

Reject and log: corrupt files, images under 224×224 px, non-RGB images that cannot be converted.

### 4.2 Preprocessing steps, in order

1. **EXIF orientation correction.** Phone images frequently carry rotation flags. Apply `ImageOps.exif_transpose` before anything else. Silently skipping this is a common and hard-to-diagnose source of accuracy loss.
2. **Color space normalization.** Convert to RGB. Strip alpha channel.
3. **Resolution handling.** Do *not* naively downscale to 640×640 as the Upadhyaya pipeline did. Stink bugs in field imagery are frequently small objects; aggressive downscaling destroys the discriminative texture the classifier depends on. Instead:
   - If the long edge exceeds 2048 px, downscale to 2048 with Lanczos resampling
   - Pass the resized image to Grounding DINO at its native expected resolution (handled by its own processor)
   - **Crop extraction for the classifier happens from the 2048-px version, not the detector's input resolution** — this preserves crop fidelity
4. **Metadata capture.** Record original dimensions, EXIF capture timestamp, GPS if present, device model. These pass through to the output packet untouched; the LLM layer uses them for context.

### 4.3 Tiling for high-density images

Images containing many small, densely packed specimens (the 23-bug case in the Upadhyaya figures) degrade badly under whole-image inference. Implement optional sliced inference:

- Trigger when the image long edge exceeds 1536 px **or** when a first-pass whole-image detection returns more than 15 boxes
- Tile with 512×512 windows at 20% overlap
- Run detection per tile, map boxes back to global coordinates, merge with NMS across tile boundaries
- `sahi` is the appropriate library; implement manually if its Grounding DINO support is inadequate

### 4.4 Training-time augmentation

Applied to the **classifier** training set only. The detector is used zero-shot or lightly tuned (see §5.3).

Required augmentations (use `albumentations`):

| Augmentation | Parameters | Rationale |
|---|---|---|
| `RandomBrightnessContrast` | ±0.3 | Field lighting variability, the paper's stated primary limitation |
| `HueSaturationValue` | hue ±10, sat ±20 | Must be **conservative** — hue is a discriminative feature for green-vs-brown species |
| `RandomRotate90` + `Rotate` | ±30° | Specimens appear at arbitrary orientation on sweep-net samples |
| `GaussNoise` / `ISONoise` | moderate | Sensor noise across the Nikon / iPhone / drone sensor mix |
| `MotionBlur` | kernel 3–7 | Handheld capture |
| `RandomShadow` / `RandomSunFlare` | sparse | Field illumination |
| `CoarseDropout` | ≤2 holes, ≤15% | Simulates partial occlusion by foliage |

**Explicitly forbidden:** aggressive hue shift beyond ±10, channel shuffle, grayscale conversion, inversion. Each of these destroys the exact color signal that separates `sgsb` from `bsb`.

**Do not apply** Mixup or CutMix to the fine-grained classifier. They blur class boundaries between already-confusable species.

---

## 5. Stage 1 — Grounding DINO detection

### 5.1 Model and loading

- Model: `IDEA-Research/grounding-dino-base` via HuggingFace `transformers` (`AutoProcessor`, `AutoModelForZeroShotObjectDetection`)
- Pin the exact revision hash in config. Do not track `main`.
- Swin-B backbone. The `-tiny` variant is insufficient for small-object insect detection; benchmark it anyway for the edge-deployment comparison but do not make it the default.

### 5.2 Text prompting

Grounding DINO's output is highly prompt-sensitive. This is the single highest-leverage tuning knob in Stage 1 and must be treated as a hyperparameter, not a constant.

**Prompt format rules (these are hard requirements of the model, not style preferences):**
- Lowercase
- Phrases separated by ` . ` (space-period-space)
- Terminated with a trailing ` .`

**Candidate prompts to benchmark systematically:**

```
"stink bug ."
"stink bug . shield bug ."
"insect . bug ."
"stink bug . beetle . insect ."
"small green insect . small brown insect ."
"shield-shaped insect on leaf ."
```

Build a `prompt_ablation.py` that evaluates every candidate against the validation set and reports recall, precision, and mean box IoU. **Select by recall.** Stage 1's job is to not miss specimens; Stage 2 and the confidence gate handle false positives. A missed bug cannot be recovered downstream; a false positive can be filtered.

Document the winning prompt in config with its ablation scores. Re-run the ablation whenever the dataset composition changes materially.

### 5.3 Fine-tuning decision

Run zero-shot first and record the metrics. Then decide:

- **Zero-shot recall ≥ 0.85 on the field validation split** → ship zero-shot. The generalization benefit is worth more than marginal precision.
- **Recall between 0.70 and 0.85** → fine-tune the detection head only, freezing the Swin backbone and BERT text encoder. Low learning rate (1e-5), few epochs, early stopping on validation recall.
- **Recall below 0.70** → Grounding DINO is the wrong tool for this imagery. Fall back to the YOLOv8m + CBAM + ASFF architecture from Upadhyaya et al., which has published metrics on near-identical data. Treat this as an expected branch, not a failure. Document the decision and the evidence.

This decision gate must be an explicit checkpoint in the training script, with the metrics logged and the chosen branch recorded.

### 5.4 Thresholds

- `box_threshold`: start 0.25, sweep 0.15–0.45 on validation
- `text_threshold`: start 0.25, sweep 0.15–0.40

Tune jointly — they interact. Sweep on validation, never on test.

### 5.5 Known failure modes to test for

- Clustered specimens merged into a single box (the dense-image case)
- Background green foliage triggering detections on leaves
- Shadows and soil clods detected as specimens (the Upadhyaya paper attributes its 8.5% misdetection rate to exactly this: color/size/shape similarity to background)
- Partial specimens at image edges

Build a dedicated adversarial test set containing: empty soybean canopy images, soil-only images, non-pentatomid insects, and leaf-litter images. **False positive rate on this set is a required reported metric.** A detector that fires on empty foliage is worse than useless in a decision-support tool.

---

## 6. Stage 1.5 — Box post-processing

Applied between detection and classification.

1. **NMS.** `torchvision.ops.nms`, IoU threshold 0.45. Tune 0.35–0.55. Stink bugs can legitimately touch or overlap in sweep-net samples, so an aggressive threshold destroys true counts — be conservative.
2. **Size filtering.** Drop boxes whose area falls below 0.02% or above 25% of image area. A box covering a quarter of the frame is a detection failure, not a large bug.
3. **Aspect ratio filtering.** Stink bug aspect ratios cluster roughly 1.2–2.0 (length:width). Drop boxes outside 0.5–3.0. Log every drop — a high drop rate signals a prompt problem.
4. **Edge handling.** Flag boxes touching the image border as `partial: true`. Classify them but down-weight their confidence; a partially visible specimen has less diagnostic signal.
5. **Crop extraction.** Expand each box by 15% context padding before cropping. The classifier benefits from seeing some surrounding substrate for scale reference. Clip to image bounds. Resize crops to classifier input size with aspect-preserving pad, not stretch.

---

## 7. Stage 2 — Species classifier

### 7.1 Architecture

Primary: **EfficientNetV2-S** via `timm`, pretrained on ImageNet, input 384×384.

Rationale: fine-grained insect classification rewards higher input resolution more than it rewards parameter count. EfficientNetV2-S at 384 px outperforms heavier backbones at 224 px on this class of problem, and the inference budget stays acceptable.

Benchmark alternatives:
- `convnext_tiny` — stronger texture modeling, relevant for connexivum patterning
- `vit_base_patch16_384` — if the dataset reaches sufficient size (>10k crops/class)
- `efficientnet_b3` — the lighter option, for the edge-deployment comparison

Select on validation macro-F1, not accuracy. Class imbalance is guaranteed (RBSB will dominate the dataset given the existing colonies and prior collection effort).

### 7.2 Training configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW, weight decay 0.01 |
| LR schedule | Cosine with 3-epoch linear warmup |
| Base LR | 3e-4 (head), 3e-5 (backbone) — discriminative LR |
| Batch size | 32 (adjust to VRAM; use gradient accumulation if constrained) |
| Epochs | 100 with early stopping, patience 15 on val macro-F1 |
| Loss | Cross-entropy with label smoothing 0.1 |
| Class imbalance | Weighted sampler **or** class-weighted loss — use one, not both |
| Mixed precision | Enabled (`torch.amp`) |
| EMA | Enabled, decay 0.9998 |

### 7.3 Two-phase fine-tuning

1. **Phase 1 — head only.** Freeze backbone, train classifier head 5 epochs. Stabilizes the head before backbone gradients destabilize pretrained features.
2. **Phase 2 — full network.** Unfreeze, train with discriminative learning rates.

### 7.4 Hard-negative mining

After the first full training run, mine the confusion pairs from §2.1:

1. Run inference on the full training set
2. Extract every misclassification within a known confusion pair
3. Review a sample manually for label noise — on fine-grained species data, a meaningful fraction of apparent model errors are annotation errors
4. Oversample verified-correct hard cases 3× in the next training round
5. Repeat for two rounds maximum; beyond that it overfits the hard set

### 7.5 Life stage inference

Nymph vs. adult. Treat as a **separate binary head** on the shared backbone, not as additional species classes — crossing 6 species × 2 stages creates 12 sparse classes and fragments the data.

Nymphs differ substantially in appearance from conspecific adults (different coloration, no fully developed wings, distinct banding). Expect lower accuracy here than on species. If labeled nymph data is insufficient at training time, ship the head disabled and emit `life_stage: "unknown"` rather than guessing. Note that both prior studies address **adults only** — nymph data will likely need to be collected specifically.

---

## 8. Confidence gating and uncertainty routing

This section is the most important part of the spec for downstream safety.

A confident wrong species identification causes a wrong pesticide application. The system must be able to say "I don't know."

### 8.1 Gating logic

Compute a fused confidence per detection:

```
fused_conf = detection_conf ^ alpha  *  classification_conf ^ (1 - alpha)
```

Start `alpha = 0.4` (weighting classification higher), tune on validation.

Route by band:

| Fused confidence | Action |
|---|---|
| ≥ 0.75 | Emit species label normally |
| 0.45 – 0.75 | Emit label with `flag: "low_confidence"` — the LLM layer must surface uncertainty to the user |
| < 0.45 | Emit `species: "uncertain"`, retain the box and count, queue the crop for expert review |

**A detection below the classification threshold still counts toward the total.** Count and species ID are independent outputs. Knowing there are 14 stink bugs but being unsure of species is useful; silently dropping 14 bugs because species is uncertain is a counting error with direct economic consequence.

### 8.2 Calibration

Raw softmax outputs are overconfident and unsuitable as the basis for a pesticide decision. Apply **temperature scaling** fitted on the validation set. Report Expected Calibration Error (ECE) before and after. Target ECE below 0.05.

### 8.3 Review queue

Every `uncertain` crop is written to a review directory with its metadata, original image reference, and both confidence scores. This queue is the human-in-the-loop retraining feedback path. Build it as a simple structured directory plus a manifest CSV — no database needed at this stage.

---

## 9. Feature extraction

Per image, compute:

- `total_count` — all detections passing post-processing
- `count_by_species` — dict, includes `uncertain` as a key
- `density` — detections per unit area, normalized; only meaningful with known scale, so emit `null` plus a `scale_known: false` flag when no reference scale is available. **Do not fabricate a density value.**
- `spatial_distribution` — clustered vs. dispersed, via nearest-neighbor index on box centroids. Ecologically relevant: RBSB adults are documented as more dispersed than the clustered pests other detection papers target.
- `mean_detection_confidence`, `mean_classification_confidence`
- `partial_detections` — count of edge-touching boxes

---

## 10. Output contract

The JSON packet is the interface to the LLM layer. **Treat this schema as frozen** — the LLM layer is built against it. Version it explicitly.

```json
{
  "schema_version": "1.0",
  "image_id": "string",
  "source_metadata": {
    "original_dimensions": [4032, 3024],
    "capture_timestamp": "2026-10-27T14:32:11Z",
    "gps": {"lat": 30.41, "lon": -91.18},
    "device": "iPhone 15 Pro Max"
  },
  "processing": {
    "detector": "grounding-dino-base",
    "detector_prompt": "stink bug . shield bug .",
    "classifier": "efficientnetv2_s_384",
    "tiled_inference": false,
    "inference_ms": {"detection": 412, "classification": 88, "total": 500}
  },
  "detections": [
    {
      "id": 0,
      "bbox": [120, 340, 210, 455],
      "bbox_format": "xyxy_absolute",
      "detection_confidence": 0.91,
      "species": "rbsb",
      "species_scientific": "Piezodorus guildinii",
      "classification_confidence": 0.87,
      "fused_confidence": 0.88,
      "life_stage": "adult",
      "life_stage_confidence": 0.79,
      "partial": false,
      "flag": null
    }
  ],
  "aggregates": {
    "total_count": 14,
    "count_by_species": {"rbsb": 11, "sgsb": 2, "uncertain": 1},
    "density": null,
    "scale_known": false,
    "spatial_distribution": "dispersed",
    "mean_detection_confidence": 0.84,
    "mean_classification_confidence": 0.81,
    "partial_detections": 2
  },
  "warnings": ["3 detections dropped by aspect ratio filter"]
}
```

Notes:
- `bbox_format` is explicit so the LLM layer never has to guess coordinate convention
- `warnings` is an array, always present, empty if none
- Every confidence is a float in [0, 1] **after calibration**
- `null` is used honestly — never substitute a placeholder number

---

## 11. Dataset

### 11.1 Assembly

| Source | Est. volume | Character |
|---|---|---|
| Existing RBSB lab dataset (prior work) | ~1,000–2,300 images | Controlled, top-down, flat surface |
| LSU rearing colonies, remaining 4 species | To be collected | Controlled, matched protocol |
| iNaturalist | Variable | Field, high variability, research-grade only |
| GBIF | Variable | Field, verify license per record |
| BugGuide | Variable | Field; **verify usage rights before ingestion** |
| New field collection (sweep-net) | To be collected | Field, the highest-value data |

### 11.2 The central data risk

Both prior studies are dominated by controlled lab imagery, and both authors identify field generalization as the primary limitation. **Do not repeat this.** Enforce a minimum field-image proportion of 40% in the training set, and make the test set **predominantly field imagery**. A model evaluated on lab images will report numbers that do not survive contact with a soybean canopy.

### 11.3 Splitting

Split **by capture session / source image**, never by crop. Multiple crops from one photograph sharing a split boundary leaks information and inflates every reported metric.

- Train 70% / Val 15% / Test 15%
- Stratify by species **and** by source domain (lab vs. field)
- Maintain a **separate held-out field-only test set** reported independently. This is the number that predicts deployment performance.

### 11.4 Annotation

- Platform: Roboflow with SAM-assisted labeling (consistent with prior work)
- Format: COCO JSON, bounding boxes plus species label
- **Every auto-generated annotation requires human review.** Prior work generated ~29k annotations this way; auto-labels are a starting point, not ground truth.
- Species labels on field-sourced images must be verified by someone with entomological training. Crowd-sourced platform labels contain species errors, particularly within the confusion pairs.
- Record annotator ID per label to enable inter-annotator agreement measurement.

---

## 12. Evaluation

### 12.1 Detection metrics (Stage 1)

Precision, Recall, F1, mAP50, mAP50-95, mean IoU, inference latency.

### 12.2 Classification metrics (Stage 2)

Per-class precision/recall/F1, macro-F1, **full 7×7 confusion matrix**, top-1 and top-2 accuracy, per-confusion-pair accuracy reported separately.

### 12.3 End-to-end metrics

These are the metrics that matter.

| Metric | Definition |
|---|---|
| Count MAE | Mean absolute error vs. manual count |
| Count MAPE | Percentage error, more interpretable for growers |
| Species-aware count MAE | Per-species count error |
| Images with zero count error | Prior work: 76/100 — this is the comparable benchmark |
| False positive rate on empty images | From the adversarial set (§5.5) |
| ECE | Calibration quality |
| Uncertain rate | Proportion routed to `uncertain` — track; a rising rate signals distribution shift |

### 12.4 Acceptance criteria

Must meet or exceed on the **field-only test set**:

- Detection recall ≥ 0.80
- Detection precision ≥ 0.90
- Count MAE ≤ 3.15 (the published baseline)
- Species macro-F1 ≥ 0.85
- Confusion-pair accuracy ≥ 0.75 for each pair
- False positive rate on empty-background images ≤ 0.05
- ECE ≤ 0.05
- Total inference ≤ 2000 ms on target hardware

If Grounding DINO cannot meet detection recall, invoke the §5.3 fallback.

### 12.5 Ablations to run

1. Grounding DINO zero-shot vs. fine-tuned
2. Prompt variants (§5.2)
3. Classifier backbone comparison
4. Input resolution: 224 vs. 320 vs. 384
5. With vs. without hard-negative mining
6. Tiled vs. whole-image on dense subsets
7. **Two-stage pipeline vs. single-stage YOLOv8m+CBAM+ASFF baseline** — required for publication; the reviewers will ask

---

## 13. Repository structure

```
cv_layer/
├── configs/
│   ├── detector.yaml
│   ├── classifier.yaml
│   ├── preprocessing.yaml
│   └── pipeline.yaml
├── src/
│   ├── ingestion/        # loading, EXIF, validation
│   ├── preprocessing/    # resize, tile, augment
│   ├── detection/        # Grounding DINO wrapper, prompts, NMS
│   ├── classification/   # timm wrapper, training loop, calibration
│   ├── features/         # count, density, spatial stats
│   ├── output/           # schema, serialization, validation
│   └── pipeline.py       # orchestration
├── scripts/
│   ├── prepare_dataset.py
│   ├── prompt_ablation.py
│   ├── train_classifier.py
│   ├── finetune_detector.py
│   ├── calibrate.py
│   ├── evaluate.py
│   └── run_inference.py
├── tests/
├── notebooks/
└── data/                 # gitignored
    ├── raw/ processed/ annotations/ splits/
```

### 13.1 Environment

Python 3.11+, PyTorch 2.x + CUDA. Core: `transformers`, `timm`, `torchvision`, `albumentations`, `opencv-python`, `pillow` + `pillow-heif`, `numpy`, `pandas`, `scikit-learn`, `pydantic` (schema validation), `pyyaml`, `sahi` (tiling), `wandb` or `mlflow` (tracking).

Pin every version. Commit a lockfile.

### 13.2 Engineering requirements

- Every config value in YAML. **No hardcoded paths, thresholds, or hyperparameters anywhere in `src/`.**
- Seed everything; log the seed.
- Validate output JSON against a `pydantic` schema before emission. Fail loudly on violation.
- Structured logging (`structlog` or stdlib `logging` with JSON formatter).
- Every stage independently unit-testable with fixture images.
- Experiment tracking from the first run, not retrofitted.
- Checkpoints saved with the config that produced them.
- Each script runnable standalone with CLI args overriding config.

---

## 14. Build order

1. Data ingestion + EXIF + validation, with tests
2. Dataset assembly and split scripts; verify no leakage
3. Grounding DINO zero-shot inference wrapper
4. Prompt ablation → select prompt
5. Box post-processing + crop extraction
6. **Detection evaluation → §5.3 decision gate** (stop and decide here)
7. Classifier training pipeline
8. Hard-negative mining round
9. Calibration + confidence gating
10. Feature extraction + JSON serialization with schema validation
11. End-to-end pipeline orchestration
12. Full evaluation + ablations
13. Adversarial/empty-background testing
14. Latency profiling and optimization

Do not proceed past step 6 without recording the decision and its evidence.

---

## 15. Out of scope

Explicitly **not** part of this layer: LLM reasoning, RAG, knowledge graphs, economic injury level logic, pesticide recommendation, mobile app, API serving, user auth, text-based identification. The CV layer's contract ends at the JSON packet.

---

## 16. Open questions to resolve before or during implementation

1. Current volume of labeled non-RBSB species imagery — determines whether the 5-species classifier is trainable now or needs a data-collection phase first
2. Availability of nymph-stage labeled data (§7.5)
3. Target deployment hardware — cloud GPU, Jetson, or on-device — this determines whether the 384-px EfficientNetV2-S budget is viable
4. BugGuide licensing terms for model training use
5. Whether a scale reference (ruler, fiducial marker, known-size object) can be included in the field capture protocol — this is the only way `density` becomes a real number rather than `null`
6. Whether existing RBSB annotations can be reused directly or need re-review under the expanded label schema
