#!/usr/bin/env python3
"""Build committed Gate0 seed JSONL + manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latex_data.seed import build_seed  # noqa: E402


def main() -> int:
    man = build_seed(ROOT)
    print(json.dumps({"totals": man["totals"], "mix": man["mix"]}, indent=2, ensure_ascii=False))
    print("[ok] wrote datasets/seed/** and datasets/manifests/gate0_tokenizer.json")
    print("Next: python scripts/validate_corpus.py --manifest datasets/manifests/gate0_tokenizer.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
