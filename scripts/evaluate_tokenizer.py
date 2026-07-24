#!/usr/bin/env python3
"""Evaluate Research Gate 0 tokenizer suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latex_tokenizer.evaluator import run_gate0_eval  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Gate 0 tokenizer evaluation")
    p.add_argument("--config", default="configs/tokenizer_latex_v1.yaml")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Relax vocab/fertility critical set for smoke artifacts",
    )
    args = p.parse_args()

    cfg_path = ROOT / args.config
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("samples_dir", "artifact_dir"):
        cfg["paths"][key] = str((ROOT / cfg["paths"][key]).resolve())

    report = run_gate0_eval(cfg, smoke=args.smoke)
    print(json.dumps({k: report[k] for k in ("passed", "critical_checks", "report_path")}, indent=2))
    print(f"full report: {report['report_path']}")
    for name, check in report["checks"].items():
        status = "PASS" if check.get("passed") else "FAIL"
        print(f"  [{status}] {name}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
