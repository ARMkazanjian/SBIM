"""Frozen v1.0 output contract (spec §10). Bump SCHEMA_VERSION for any change."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"
Conf = Field(ge=0.0, le=1.0)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GPS(_Strict):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class SourceMetadata(_Strict):
    original_dimensions: tuple[int, int]
    capture_timestamp: Optional[str] = None
    gps: Optional[GPS] = None
    device: Optional[str] = None


class InferenceMs(_Strict):
    detection: float = Field(ge=0)
    classification: float = Field(ge=0)
    total: float = Field(ge=0)


class Processing(_Strict):
    detector: str
    detector_prompt: str
    classifier: str
    tiled_inference: bool
    inference_ms: InferenceMs


class DetectionOut(_Strict):
    id: int = Field(ge=0)
    bbox: tuple[float, float, float, float]
    bbox_format: Literal["xyxy_absolute"] = "xyxy_absolute"
    detection_confidence: float = Conf
    species: str
    species_scientific: Optional[str] = None
    classification_confidence: float = Conf
    fused_confidence: float = Conf
    life_stage: Literal["adult", "nymph", "unknown"] = "unknown"
    life_stage_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    partial: bool
    flag: Optional[Literal["low_confidence"]] = None

    @model_validator(mode="after")
    def _check(self):
        x0, y0, x1, y1 = self.bbox
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"degenerate bbox {self.bbox}")
        if self.life_stage == "unknown" and self.life_stage_confidence is not None:
            raise ValueError("life_stage 'unknown' must have null life_stage_confidence")
        if self.species == "uncertain" and self.flag is not None:
            raise ValueError("'uncertain' detections carry no flag")
        return self


class Aggregates(_Strict):
    total_count: int = Field(ge=0)
    count_by_species: dict[str, int]
    density: Optional[float] = None
    scale_known: bool
    spatial_distribution: Optional[Literal["clustered", "dispersed"]] = None
    mean_detection_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    mean_classification_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    partial_detections: int = Field(ge=0)

    @model_validator(mode="after")
    def _check(self):
        if "uncertain" not in self.count_by_species:
            raise ValueError("count_by_species must include 'uncertain'")
        if sum(self.count_by_species.values()) != self.total_count:
            raise ValueError("count_by_species does not sum to total_count")
        if self.density is not None and not self.scale_known:
            raise ValueError("density must be null when scale_known is false")
        if self.total_count == 0 and (self.mean_detection_confidence is not None
                                      or self.mean_classification_confidence is not None):
            raise ValueError("mean confidences must be null when there are no detections")
        return self


class Packet(_Strict):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    image_id: str
    source_metadata: SourceMetadata
    processing: Processing
    detections: list[DetectionOut]
    aggregates: Aggregates
    warnings: list[str]

    @field_validator("warnings")
    @classmethod
    def _warn(cls, v):
        return list(v)

    @model_validator(mode="after")
    def _consistent(self):
        if self.aggregates.total_count != len(self.detections):
            raise ValueError("aggregates.total_count != len(detections)")
        ids = [d.id for d in self.detections]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate detection ids")
        counted: dict[str, int] = {}
        for d in self.detections:
            counted[d.species] = counted.get(d.species, 0) + 1
        for k, v in self.aggregates.count_by_species.items():
            if counted.get(k, 0) != v:
                raise ValueError(f"count_by_species[{k}]={v} disagrees with detections ({counted.get(k, 0)})")
        return self
