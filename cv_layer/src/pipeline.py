"""End-to-end orchestration: image -> validated JSON packet (spec §3). Stages communicate only through the
contracts in detection.types / classifier.classify / output.schema."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .common.config import load_config
from .common.logging import get_logger
from .detection.detector import run_detection
from .detection.postprocess import extract_crop, postprocess_boxes
from .detection.prompts import validate_prompt
from .features.aggregates import compute_aggregates
from .gating.gate import route
from .ingestion.loader import LoadedImage, load_bytes, load_image
from .output.review_queue import ReviewQueue
from .output.serialize import validate_packet
from .preprocessing.resize import downscale_long_edge

log = get_logger(__name__)


class CropClassifierLike(Protocol):
    classes: list[str]
    stages: list[str]
    name: str
    has_stage_head: bool

    def classify(self, crops: list[Image.Image]) -> tuple[np.ndarray, np.ndarray | None]: ...


class CVPipeline:
    def __init__(self, detector, classifier: CropClassifierLike, pipeline_cfg: dict, detector_cfg: dict,
                 preprocessing_cfg: dict, classifier_cfg: dict, review_queue: ReviewQueue | None = None):
        self.detector, self.classifier = detector, classifier
        self.pcfg, self.dcfg, self.prep, self.ccfg = pipeline_cfg, detector_cfg, preprocessing_cfg, classifier_cfg
        self.prompt = validate_prompt(detector_cfg["prompt"]["selected"])
        self.review_queue = review_queue
        self.sci = classifier_cfg["classes"]["scientific_names"]
        self.use_stage = bool(pipeline_cfg["life_stage"]["enabled"] and getattr(classifier, "has_stage_head", False))

    @classmethod
    def from_configs(cls, config_path: str = "configs/pipeline.yaml", overrides: list[str] | None = None,
                     detector=None, classifier=None) -> "CVPipeline":
        pcfg = load_config(config_path, overrides)
        dcfg = load_config(pcfg["paths"]["detector_config"])
        ccfg = load_config(pcfg["paths"]["classifier_config"])
        prep = load_config(pcfg["paths"]["preprocessing_config"])
        if detector is None:
            from .detection.grounding_dino import GroundingDinoDetector
            detector = GroundingDinoDetector.from_config(dcfg)
        if classifier is None:
            from .classification.calibration import load_temperature
            from .classification.inference import CropClassifier
            temp = load_temperature(pcfg["paths"]["calibration_file"]) if pcfg["gating"]["calibrate"] else 1.0
            classifier = CropClassifier(str(pcfg["paths"]["classifier_checkpoint"]), temperature=temp)
        rq = ReviewQueue(pcfg["paths"]["review_dir"])
        return cls(detector, classifier, pcfg, dcfg, prep, ccfg, rq)

    # ---- entry points -------------------------------------------------
    def run_path(self, path: str | Path, pixels_per_cm: float | None = None) -> dict:
        return self.run(load_image(path, self.prep["ingestion"]["min_side_px"]), pixels_per_cm)

    def run_bytes(self, data: bytes, image_id: str = "buffer", pixels_per_cm: float | None = None) -> dict:
        return self.run(load_bytes(data, image_id, self.prep["ingestion"]["min_side_px"]), pixels_per_cm)

    def run(self, loaded: LoadedImage, pixels_per_cm: float | None = None) -> dict:
        t0 = time.perf_counter()
        warnings: list[str] = list(loaded.warnings)
        work, scale = downscale_long_edge(loaded.image, self.prep["resize"]["max_long_edge"])
        th = self.dcfg["thresholds"]

        # Stage 1 (+ tiling) and 1.5
        boxes, scores, tiled = run_detection(self.detector, work, self.prompt, th["box_threshold"],
                                             th["text_threshold"], self.prep["tiling"])
        post = postprocess_boxes(boxes, scores, work.size, self.dcfg["postprocess"], skip_nms=False)
        warnings += post.warnings
        t_det = time.perf_counter()

        # Crops come from the <=2048 image, not the detector input (spec §4.2)
        pad = self.dcfg["postprocess"]["crop_padding"]
        size = getattr(self.classifier, "size", self.ccfg["model"]["input_size"])   # checkpoint is the source of truth
        crops = [extract_crop(work, d.bbox, pad, size) for d in post.detections]
        probs, stage_probs = self.classifier.classify(crops) if crops else (np.zeros((0, 0)), None)
        t_cls = time.perf_counter()

        pw = self.dcfg["postprocess"]["partial_conf_weight"]
        out_dets: list[dict] = []
        for i, d in enumerate(post.detections):
            k = int(np.argmax(probs[i]))
            cls_conf = float(probs[i][k])
            det_conf = d.score * (pw if d.partial else 1.0)
            gd = route(self.classifier.classes[k], det_conf, cls_conf, self.pcfg["gating"])
            life, life_conf = "unknown", None
            if self.use_stage and stage_probs is not None:
                j = int(np.argmax(stage_probs[i]))
                life, life_conf = self.classifier.stages[j], round(float(stage_probs[i][j]), 4)
            bbox = [round(v / scale, 1) for v in d.bbox]
            out_dets.append({
                "id": i, "bbox": bbox, "bbox_format": "xyxy_absolute",
                "detection_confidence": round(min(det_conf, 1.0), 4),
                "species": gd.species,
                "species_scientific": self.sci.get(gd.species) if gd.species != "uncertain" else None,
                "classification_confidence": round(cls_conf, 4),
                "fused_confidence": round(min(gd.fused_confidence, 1.0), 4),
                "life_stage": life, "life_stage_confidence": life_conf,
                "partial": d.partial, "flag": gd.flag,
            })
            if gd.queue_for_review and self.review_queue is not None:
                self.review_queue.add(crops[i], loaded.image_id, loaded.source, out_dets[-1], self.classifier.classes[k])

        aggregates = compute_aggregates(out_dets, loaded.image.size, self.pcfg["features"], pixels_per_cm)
        t_end = time.perf_counter()
        packet = {
            "schema_version": self.pcfg["schema_version"],
            "image_id": loaded.image_id,
            "source_metadata": loaded.metadata.to_dict(),
            "processing": {
                "detector": getattr(self.detector, "name", "unknown"), "detector_prompt": self.prompt,
                "classifier": self.classifier.name, "tiled_inference": tiled,
                "inference_ms": {"detection": round((t_det - t0) * 1000, 1),
                                 "classification": round((t_cls - t_det) * 1000, 1),
                                 "total": round((t_end - t0) * 1000, 1)}},
            "detections": out_dets, "aggregates": aggregates, "warnings": warnings,
        }
        validate_packet(packet)           # fail loudly on violation
        log.info("image_processed", extra={"image_id": loaded.image_id, "count": len(out_dets), "tiled": tiled})
        return packet
