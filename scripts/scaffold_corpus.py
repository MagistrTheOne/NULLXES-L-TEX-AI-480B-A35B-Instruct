#!/usr/bin/env python3
"""Create datasets/ directory skeleton (raw/processed empty, seed/manifests ready)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BUCKETS = (
    "multilingual",
    "code",
    "enterprise",
    "scientific",
    "synthetic_structure",
)


def main() -> int:
    for b in BUCKETS:
        (ROOT / "datasets" / "seed" / b).mkdir(parents=True, exist_ok=True)
        raw = ROOT / "datasets" / "raw" / "shards" / b
        raw.mkdir(parents=True, exist_ok=True)
        (raw / ".gitkeep").write_text("", encoding="utf-8")
    (ROOT / "datasets" / "processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "datasets" / "processed" / ".gitkeep").write_text("", encoding="utf-8")
    (ROOT / "datasets" / "manifests").mkdir(parents=True, exist_ok=True)
    print(f"[ok] scaffolded under {ROOT / 'datasets'}")
    print("Next: python scripts/build_seed_corpus.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
