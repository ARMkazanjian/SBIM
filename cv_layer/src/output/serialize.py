from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .schema import Packet


class SchemaViolation(RuntimeError):
    """Raised when a packet would violate the output contract. Never swallowed (spec §13.2)."""


def validate_packet(raw: dict) -> Packet:
    try:
        return Packet.model_validate(raw)
    except ValidationError as e:
        raise SchemaViolation(f"output packet violates schema v1.0:\n{e}") from e


def to_json(raw: dict, indent: int | None = 2) -> str:
    return validate_packet(raw).model_dump_json(indent=indent)


def write_packet(raw: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(raw) + "\n")
    return path


def export_json_schema(path: str | Path) -> None:
    Path(path).write_text(json.dumps(Packet.model_json_schema(), indent=2))
