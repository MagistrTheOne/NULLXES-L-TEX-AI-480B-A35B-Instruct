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
# Markers must be LINE-START (after \n or BOS) so Wikipedia "Anna said:" mid-prose is OK.
PERSONA_BAN_LINE_PREFIXES = (
    "anna said:",
    "adeline said:",
    "karen said:",
    "anna said “",
    'anna said "',
)
PERSONA_BAN_SUBSTRINGS = (
    "i am your digital employee",
    "my personality is",
    "speaking in a warm tone",
    "as your hr agent i feel",
)


def _persona_ban_hit(low: str) -> str | None:
    for m in PERSONA_BAN_SUBSTRINGS:
        if m in low:
            return m
    for m in PERSONA_BAN_LINE_PREFIXES:
        if low.startswith(m) or f"\n{m}" in low:
            return m
    return None


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
    hit = _persona_ban_hit(low)
    if hit:
        errs.append(f"ban:{hit}")
    return errs
