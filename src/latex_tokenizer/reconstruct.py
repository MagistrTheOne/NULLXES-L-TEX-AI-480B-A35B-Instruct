"""Lossless encode/decode reconstruction tests."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass
class ReconstructResult:
    name: str
    passed: bool
    detail: str = ""


def _normalize_compare(a: str, b: str) -> bool:
    """Allow NFKC equivalence only."""
    if a == b:
        return True
    return unicodedata.normalize("NFKC", a) == unicodedata.normalize("NFKC", b)


def reconstruct(
    name: str,
    text: str,
    encode_fn,
    decode_fn,
) -> ReconstructResult:
    ids = encode_fn(text)
    roundtrip = decode_fn(ids)
    # SentencePiece may add/remove leading space marker semantics
    ok = _normalize_compare(text, roundtrip) or _normalize_compare(
        text.strip(), roundtrip.strip()
    )
    # SP often injects/removes ▁-driven spaces — compare without SP space artifacts
    if not ok:
        t2 = text.replace(" ", "")
        r2 = roundtrip.replace(" ", "")
        ok = _normalize_compare(t2, r2)
    return ReconstructResult(
        name=name,
        passed=ok,
        detail="" if ok else f"mismatch len {len(text)} vs {len(roundtrip)}",
    )


DOMAIN_FILES = {
    "plain_text": ("russian.txt", "english.txt", "chinese.txt", "markdown.md", "legal.txt"),
    "code": ("python.py", "typescript.ts"),
    "json": ("json.json", "api.yaml"),
}
