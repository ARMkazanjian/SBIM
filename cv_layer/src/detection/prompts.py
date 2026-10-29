"""Grounding DINO prompt rules (spec §5.2): lowercase, phrases separated by ' . ', trailing ' .'."""
from __future__ import annotations

import re

_VALID = re.compile(r"^[a-z0-9\- ]+( \. [a-z0-9\- ]+)* \.$")


def format_prompt(phrases: list[str] | str) -> str:
    if isinstance(phrases, str):
        phrases = [p for p in phrases.split(".")]
    cleaned = [re.sub(r"\s+", " ", p.strip().lower()) for p in phrases]
    cleaned = [p for p in cleaned if p]
    if not cleaned:
        raise ValueError("prompt has no phrases")
    return " . ".join(cleaned) + " ."


def validate_prompt(prompt: str) -> str:
    if not _VALID.match(prompt):
        raise ValueError(f"invalid Grounding DINO prompt {prompt!r}; expected lowercase 'a . b .' format "
                         f"(try format_prompt)")
    return prompt
