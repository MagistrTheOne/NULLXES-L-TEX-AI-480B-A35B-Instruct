#!/usr/bin/env python3
"""
Acceptance smoke for Transformers surface (requires Stage1 deps).

  python scripts/smoke_hf_causal.py --checkpoint checkpoints/nullxes-latex-7b
  python scripts/smoke_hf_causal.py --checkpoint checkpoints/nullxes-latex-7b --device cuda

Or one-shot: bash scripts/run_genesis_smoke.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_CKPT = "checkpoints/nullxes-latex-7b"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--device", default="cpu", help="cpu | cuda")
    args = p.parse_args()
    ckpt = ROOT / args.checkpoint

    has_weights = (ckpt / "model.safetensors").is_file() or (
        ckpt / "model.safetensors.index.json"
    ).is_file()
    if not ckpt.is_dir() or not has_weights:
        print(
            f"[fail] missing checkpoint {ckpt}\n"
            f"  Run: bash scripts/run_genesis_smoke.sh\n"
            f"  Or:  python scripts/init_model.py --config configs/nullxes_latex_7b.yaml "
            f"--dtype bfloat16 --smoke-device cuda",
            file=sys.stderr,
        )
        return 2

    try:
        import latex  # noqa: F401 — Auto registration
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as e:
        print(f"[fail] {e}\nInstall: pip install -r requirements-stage1.txt", file=sys.stderr)
        return 2

    print(f"[load] {ckpt}", flush=True)
    config = AutoConfig.from_pretrained(ckpt, trust_remote_code=False)
    assert config.model_type == "latex", config.model_type
    try:
        model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16)
    if args.device == "cuda":
        model = model.to("cuda")
    print("config.model_type =", model.config.model_type, flush=True)
    print("class =", model.__class__.__name__, flush=True)

    device = next(model.parameters()).device
    ids = torch.tensor([[2, 10, 11, 3]], device=device)
    out = model.generate(ids, max_new_tokens=4, use_cache=True)
    assert out.shape[-1] == ids.shape[-1] + 4, out.shape
    print("generate shape =", tuple(out.shape), flush=True)
    print("[PASS] HF CausalLM smoke", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
