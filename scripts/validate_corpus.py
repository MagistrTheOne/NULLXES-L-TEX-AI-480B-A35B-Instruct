#!/usr/bin/env python3
"""Validate corpus manifest — must PASS before full tokenizer Gate0 / pretrain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latex_data.validate import validate_corpus  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest",
        default="datasets/manifests/gate0_tokenizer.json",
    )
    p.add_argument("--min-docs", type=int, default=20)
    args = p.parse_args()
    man = ROOT / args.manifest
    if not man.is_file():
        print(f"[fail] missing manifest {man}", file=sys.stderr)
        print("Run: python scripts/build_seed_corpus.py", file=sys.stderr)
        return 2
    report = validate_corpus(man, repo_root=ROOT, min_docs_per_bucket=args.min_docs)
    stem = man.stem.replace("/", "_")
    out = ROOT / "datasets" / "manifests" / f"{stem}_validate_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "buckets": report["buckets"], "error_count": report["error_count"]}, indent=2))
    if not report["passed"]:
        for e in report["errors"][:20]:
            print(f"  ERR {e}", file=sys.stderr)
    print(f"report: {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
