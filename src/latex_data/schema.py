"""Corpus record schema helpers."""

from __future__ import annotations

from typing import Any

BUCKETS = (
    "multilingual",
    "code",
    "enterprise",
    "scientific",
    "synthetic_structure",
)

REQUIRED = ("id", "text", "bucket")

# Dialogue / persona coaching only — listing employee names (Anna, …) is allowed.
PERSONA_BAN_MARKERS = (
    "anna said:",
    "adeline said:",
    "karen said:",
    "anna said “",
    'anna said "',
    "i am your digital employee",
    "my personality is",
    "speaking in a warm tone",
    "as your hr agent i feel",
)


def validate_record(obj: dict[str, Any], min_chars: int = 32) -> list[str]:
    errs: list[str] = []
    for k in REQUIRED:
        if k not in obj:
            errs.append(f"missing:{k}")
    if "bucket" in obj and obj["bucket"] not in BUCKETS:
        errs.append(f"bad_bucket:{obj.get('bucket')}")
    text = obj.get("text", "")
    if not isinstance(text, str) or len(text.strip()) < min_chars:
        errs.append("text_too_short")
    low = text.lower() if isinstance(text, str) else ""
    for m in PERSONA_BAN_MARKERS:
        if m in low:
            errs.append(f"ban:{m}")
            break
    return errs
