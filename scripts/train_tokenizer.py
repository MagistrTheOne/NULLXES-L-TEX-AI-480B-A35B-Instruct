#!/usr/bin/env python3
"""Train NULLXES-LÆTEX Tokenizer v0.1 (Gate 0)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latex_tokenizer.trainer import train_tokenizer  # noqa: E402


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    p = argparse.ArgumentParser(description="Train LÆTEX tokenizer v0.1")
    p.add_argument("--config", default="configs/tokenizer_stage0.yaml")
    p.add_argument("--runtime", default="configs/runtime.yaml")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Small vocab pipeline test (does NOT count as Gate0 PASS)",
    )
    args = p.parse_args()

    cfg = load_yaml(ROOT / args.config)
    runtime = load_yaml(ROOT / args.runtime)
    # Resolve relative paths from repo root
    for key in ("samples_dir", "artifact_dir"):
        cfg["paths"][key] = str((ROOT / cfg["paths"][key]).resolve())

    out = train_tokenizer(cfg, runtime, smoke=args.smoke)
    print(f"[ok] artifacts written to {out}")
    if args.smoke:
        print("[note] smoke run — Gate0 PASS requires full vocab 131072 + corpus mix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
