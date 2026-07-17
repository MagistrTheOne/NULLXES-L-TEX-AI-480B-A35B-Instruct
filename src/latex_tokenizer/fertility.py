"""Fertility, inflation, and fragmentation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FertilityResult:
    name: str
    chars: int
    tokens: int
    chars_per_token: float
    passed: bool
    detail: str = ""


@dataclass
class FragmentResult:
    text: str
    n_tokens: int
    passed: bool
    pieces: list[str] = field(default_factory=list)
    detail: str = ""


def chars_per_token(text: str, n_tokens: int) -> float:
    if n_tokens <= 0:
        return 0.0
    return len(text) / n_tokens


def evaluate_fertility_file(
    name: str,
    text: str,
    n_tokens: int,
    cfg: dict[str, Any],
) -> FertilityResult:
    cpt = chars_per_token(text, n_tokens)
    fert = cfg.get("fertility", {})
    passed = True
    detail = ""

    if name.startswith("russian") or name == "legal.txt":
        lo, hi = fert.get("ru_chars_per_token", [2.5, 4.0])
        # technical floor
        tmin = fert.get("ru_technical_min", 2.2)
        passed = cpt >= tmin
        # soft band for general RU when enough text
        if len(text) > 80 and not (lo <= cpt <= hi * 1.5):
            detail = f"RU band soft-warn target [{lo},{hi}], got {cpt:.3f}"
        if cpt < tmin:
            passed = False
            detail = f"RU technical min {tmin}, got {cpt:.3f}"
    elif name.startswith("english") or name == "markdown.md":
        lo, hi = fert.get("en_chars_per_token", [3.0, 5.0])
        tmin = fert.get("en_technical_min", 3.0)
        passed = cpt >= min(lo, tmin) * 0.85  # slight slack on tiny samples
        if cpt < tmin * 0.85:
            passed = False
            detail = f"EN technical min ~{tmin}, got {cpt:.3f}"
        elif not (lo * 0.7 <= cpt <= hi * 2):
            detail = f"EN band soft-warn [{lo},{hi}], got {cpt:.3f}"
    elif name.endswith(".py") or name.endswith(".ts"):
        # identifier preservation checked separately; fertility: not insane inflation
        passed = cpt >= 1.5
        detail = "code fertility floor"
    elif name.endswith(".json") or name.endswith(".yaml"):
        # JSON: flag extreme punctuation shred (cpt very low)
        passed = cpt >= 1.2
        detail = "json/yaml punctuation fragmentation floor"
    else:
        passed = cpt >= 1.0

    return FertilityResult(name, len(text), n_tokens, cpt, passed, detail)


def is_char_shredded(text: str, pieces: list[str]) -> bool:
    """True if mostly single-char pieces relative to identifier length."""
    if not pieces:
        return True
    # Strip SP underline marker
    clean = [p[1:] if p.startswith("▁") else p for p in pieces]
    if len(text) <= 4:
        return False
    single = sum(1 for p in clean if len(p) == 1)
    # shredded if >70% single-char and many pieces
    return len(clean) >= max(6, len(text) * 0.6) and single / max(len(clean), 1) > 0.7


def evaluate_fragmentation(
    text: str,
    pieces: list[str],
) -> FragmentResult:
    shredded = is_char_shredded(text, pieces)
    return FragmentResult(
        text=text,
        n_tokens=len(pieces),
        passed=not shredded,
        pieces=pieces,
        detail="char-shred" if shredded else "ok",
    )


def load_samples(samples_dir: Path) -> dict[str, str]:
    out = {}
    if not samples_dir.is_dir():
        return out
    for p in sorted(samples_dir.iterdir()):
        if p.is_file():
            out[p.name] = p.read_text(encoding="utf-8", errors="replace")
    return out
