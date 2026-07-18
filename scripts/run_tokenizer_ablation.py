#!/usr/bin/env python3
"""
Gate B — tokenizer vocab ablation on the same corpus sample.

  python scripts/run_tokenizer_ablation.py
  python scripts/run_tokenizer_ablation.py --sizes 32000 64000 96000 131072
  python scripts/run_tokenizer_ablation.py --resume   # skip sizes that already have tokenizer.model

Writes under tokenizer/ablation/v{size}/ and a summary JSON.
Winner freeze is manual → copy best to tokenizer/latex-v0.2/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latex_tokenizer.evaluator import run_gate0_eval  # noqa: E402
from latex_tokenizer.trainer import train_tokenizer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="LÆTEX tokenizer vocab ablation")
    p.add_argument("--config", default="configs/tokenizer_latex_v0.2.yaml")
    p.add_argument("--runtime", default="configs/runtime_runpod_rtx_pro_6000.yaml")
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[32000, 64000, 96000, 131072],
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Only evaluate existing ablation dirs",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="If tokenizer.model exists for a size, skip retrain (eval only that size)",
    )
    args = p.parse_args()

    base_cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    runtime = yaml.safe_load((ROOT / args.runtime).read_text(encoding="utf-8"))

    man = ROOT / base_cfg["paths"]["corpus_manifest"]
    if not man.is_file():
        print(f"[fail] missing corpus manifest: {man}", flush=True)
        print("  run Gate A proxy download first", flush=True)
        return 1

    summary: list[dict] = []
    for size in args.sizes:
        cfg = json.loads(json.dumps(base_cfg))  # deep copy
        cfg["vocab_size"] = int(size)
        cfg["name"] = f"latex-tokenizer-v0.2-ablate-{size}"
        cfg["version"] = f"0.2-ablate-{size}"
        out_dir = ROOT / "tokenizer" / "ablation" / f"v{size}"
        cfg["paths"]["samples_dir"] = str((ROOT / cfg["paths"]["samples_dir"]).resolve())
        cfg["paths"]["artifact_dir"] = str(out_dir.resolve())

        model_path = out_dir / "tokenizer.model"
        have_model = model_path.is_file()

        print(f"\n=== vocab={size} → {out_dir} ===", flush=True)
        if args.skip_train or (args.resume and have_model):
            if not have_model:
                print(f"[fail] missing {model_path} (cannot skip-train/resume)", flush=True)
                return 1
            print(f"[skip-train] using existing {model_path}", flush=True)
        else:
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            train_tokenizer(cfg, runtime, smoke=False)

        report = run_gate0_eval(cfg, smoke=False)
        row = {
            "vocab_size": size,
            "passed": report.get("passed"),
            "report_path": report.get("report_path"),
            "critical_checks": report.get("critical_checks"),
        }
        checks = report.get("checks") or {}
        for k in ("vocab_size", "fertility", "special_tokens", "compression"):
            if k in checks:
                row[k] = checks[k]
        summary.append(row)
        print(f"[ablate] size={size} passed={row['passed']}", flush=True)

    out = ROOT / "tokenizer" / "ablation" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n[ok] summary → {out}", flush=True)
    print("Freeze winner manually:", flush=True)
    print("  cp -r tokenizer/ablation/vBEST/* tokenizer/latex-v0.2/", flush=True)
    return 0 if any(r.get("passed") for r in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
