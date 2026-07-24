"""Document filters and near-duplicate detection for the LÆTEX V1 corpus.

The corpus is deliberately small, so every accepted document has to earn its
place: web boilerplate, SEO copy, and near-duplicates cost proportionally far
more here than in a web-scale run.

Near-duplicate detection uses SimHash over word 5-grams with banded lookup.
MinHash with 64 permutations would hash every shingle 64 times; SimHash hashes
each shingle once, which is what makes a pure-Python pass over hundreds of
thousands of documents practical.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from hashlib import blake2b

CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
LATIN = re.compile(r"[a-zA-Z]")
CJK = re.compile(r"[\u4e00-\u9fff]")
WORD = re.compile(r"\w+", re.UNICODE)
WHITESPACE = re.compile(r"\s+")

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"),
    re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|password|passwd|token)\s*[:=]\s*['\"][^'\"]{8,}"),
)

MARKETING_MARKERS = (
    "click here",
    "buy now",
    "limited time offer",
    "subscribe to our newsletter",
    "best price guaranteed",
    "seo optimization services",
    "подпишитесь на наш",
    "подписывайтесь на канал",
    "успей купить",
    "по промокоду",
    "заказать со скидкой",
    "лучшие цены",
)

BOILERPLATE_MARKERS = (
    "this website uses cookies",
    "all rights reserved",
    "terms of service apply",
    "мы используем файлы cookie",
    "все права защищены",
)


@dataclass
class FilterConfig:
    min_chars: int = 200
    max_chars: int = 40_000
    code_max_chars: int = 100_000
    min_words: int = 30
    languages: tuple[str, ...] = ("ru", "en", "code", "zh")
    max_symbol_ratio: float = 0.25
    max_digit_ratio: float = 0.30
    max_duplicate_line_ratio: float = 0.35
    max_top_word_ratio: float = 0.18
    drop_marketing: bool = True
    drop_boilerplate: bool = True
    drop_secrets: bool = True

    @classmethod
    def from_dict(cls, data: dict | None) -> "FilterConfig":
        data = data or {}
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        if "languages" in known:
            known["languages"] = tuple(known["languages"])
        return cls(**known)


def detect_language(text: str) -> str:
    """Coarse script-based language label. Enough to keep the mix honest."""
    cyr = len(CYRILLIC.findall(text))
    lat = len(LATIN.findall(text))
    cjk = len(CJK.findall(text))
    total = cyr + lat + cjk
    if total == 0:
        return "other"
    if cjk / total > 0.15:
        return "zh"
    if cyr / total > 0.35:
        return "ru"
    if lat / total > 0.5:
        return "en"
    return "other"


def _duplicate_line_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return 0.0
    unique = len(set(lines))
    return 1.0 - unique / len(lines)


def _top_word_ratio(words: list[str]) -> float:
    if not words:
        return 1.0
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return max(counts.values()) / len(words)


def quality_reject(text: str, cfg: FilterConfig, *, bucket: str = "") -> str | None:
    """Return a rejection reason, or None when the document is acceptable."""
    stripped = text.strip()
    n = len(stripped)
    is_code = bucket == "code"
    if n < cfg.min_chars:
        return "too_short"
    if n > (cfg.code_max_chars if is_code else cfg.max_chars):
        return "too_long"

    low = stripped.lower()
    if cfg.drop_secrets and any(p.search(stripped) for p in SECRET_PATTERNS):
        return "secret"
    # Marketing and boilerplate are prose defects. In code, "All rights
    # reserved" is a license header, not junk to throw the file away over.
    if not is_code:
        if cfg.drop_marketing and any(m in low for m in MARKETING_MARKERS):
            return "marketing"
        if cfg.drop_boilerplate and any(m in low for m in BOILERPLATE_MARKERS):
            return "boilerplate"

    words = WORD.findall(stripped)
    if not is_code and len(words) < cfg.min_words:
        return "too_few_words"

    lang = detect_language(stripped)
    if not is_code and lang not in cfg.languages:
        return f"lang:{lang}"

    # Code legitimately carries punctuation and digits; prose does not.
    if not is_code:
        symbols = sum(
            1
            for ch in stripped
            if not ch.isalnum() and not ch.isspace() and unicodedata.category(ch)[0] in "PS"
        )
        if symbols / n > cfg.max_symbol_ratio:
            return "symbol_ratio"
        digits = sum(1 for ch in stripped if ch.isdigit())
        if digits / n > cfg.max_digit_ratio:
            return "digit_ratio"
        if _top_word_ratio([w.lower() for w in words]) > cfg.max_top_word_ratio:
            return "word_repetition"

    if _duplicate_line_ratio(stripped) > cfg.max_duplicate_line_ratio:
        return "line_repetition"

    return None


def normalize_for_dedup(text: str) -> str:
    return WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text).lower()).strip()


def exact_hash(text: str) -> str:
    return blake2b(normalize_for_dedup(text).encode("utf-8"), digest_size=16).hexdigest()


def _shingles(text: str, size: int = 5) -> list[str]:
    words = WORD.findall(normalize_for_dedup(text))
    if len(words) <= size:
        return [" ".join(words)] if words else []
    return [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]


def simhash(text: str, *, shingle_size: int = 5) -> int:
    """64-bit SimHash over word shingles."""
    vector = [0] * 64
    shingles = _shingles(text, shingle_size)
    if not shingles:
        return 0
    for shingle in shingles:
        h = int.from_bytes(blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if (h >> bit) & 1 else -1
    out = 0
    for bit in range(64):
        if vector[bit] > 0:
            out |= 1 << bit
    return out


@dataclass
class DedupIndex:
    """Exact + near-duplicate index.

    Banding splits the 64-bit signature into four 16-bit bands. Two signatures
    within `hamming_threshold` bits must agree on at least one band, so only
    same-band candidates are compared.
    """

    hamming_threshold: int = 3
    _exact: set[str] = field(default_factory=set)
    _bands: dict[tuple[int, int], list[int]] = field(default_factory=dict)

    def add_or_reject(self, text: str) -> str | None:
        digest = exact_hash(text)
        if digest in self._exact:
            return "dup_exact"
        signature = simhash(text)
        keys = [(band, (signature >> (16 * band)) & 0xFFFF) for band in range(4)]
        for key in keys:
            for other in self._bands.get(key, ()):
                if bin(other ^ signature).count("1") <= self.hamming_threshold:
                    return "dup_near"
        self._exact.add(digest)
        for key in keys:
            self._bands.setdefault(key, []).append(signature)
        return None

    def __len__(self) -> int:
        return len(self._exact)
