"""Unicode normalization for LÆTEX tokenizer."""

from __future__ import annotations

import re
import unicodedata

# Preserve these patterns across NFKC (re-inject if mangled — defensive)
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_URL = re.compile(r"https?://[^\s<>\"']+")


def normalize_text(text: str, form: str = "NFKC") -> str:
    """NFKC normalize while keeping emails/URLs as contiguous spans."""
    if not text:
        return text
    # Drop NULs (wiki/binary junk) — SentencePiece warns otherwise
    if "\x00" in text:
        text = text.replace("\x00", "")
    emails = _EMAIL.findall(text)
    urls = _URL.findall(text)
    out = unicodedata.normalize(form, text)
    # Re-check: if normalization split critical spans, prefer original span forms
    for e in emails:
        ne = unicodedata.normalize(form, e)
        if ne in out and e not in out and ne != e:
            out = out.replace(ne, e)
    for u in urls:
        nu = unicodedata.normalize(form, u)
        if nu in out and u not in out and nu != u:
            out = out.replace(nu, u)
    return out


def preserves_math_code_hints(text: str) -> bool:
    """Smoke: common math/code characters survive normalization."""
    probes = ["∑", "π", "ε", "{", "}", "_", "→", "`"]
    normed = normalize_text(text)
    for p in probes:
        if p in text and p not in normed:
            return False
    return True
