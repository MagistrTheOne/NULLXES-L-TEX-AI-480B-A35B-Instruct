#!/usr/bin/env python3
"""Write the LÆTEX V1 SFT set to datasets/sft/latex_v1.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latex_data.sft_v1 import build_sft_v1  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="datasets/sft/latex_v1.jsonl")
    args = p.parse_args()

    records = build_sft_v1()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    by_task = Counter(r["task"] for r in records)
    by_lang = Counter(r["lang"] for r in records)
    print(
        json.dumps(
            {
                "file": str(out.relative_to(ROOT)),
                "examples": len(records),
                "by_task": dict(by_task),
                "by_lang": dict(by_lang),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    refusal_like = sum(v for k, v in by_task.items() if "refusal" in k or k == "criticism")
    if refusal_like == 0:
        print("[fail] no refusal/criticism examples — the answer protocol will not hold")
        return 1
    print(f"[ok] refusal/criticism examples: {refusal_like}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
