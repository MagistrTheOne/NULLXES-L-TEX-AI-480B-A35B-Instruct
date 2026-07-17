"""
Acceptance smoke for Transformers surface (requires Stage1 deps).

  python scripts/smoke_hf_causal.py --checkpoint checkpoints/nullxes-latex-7b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/nullxes-latex-7b")
    args = p.parse_args()
    ckpt = ROOT / args.checkpoint
    if not ckpt.is_dir():
        print(f"[fail] missing checkpoint {ckpt} — run scripts/init_model.py first", file=sys.stderr)
        return 2

    try:
        import latex  # noqa: F401 — Auto registration
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as e:
        print(f"[fail] {e}\nInstall: pip install -r requirements-stage1.txt", file=sys.stderr)
        return 2

    config = AutoConfig.from_pretrained(ckpt, trust_remote_code=False)
    assert config.model_type == "latex", config.model_type
    model = AutoModelForCausalLM.from_pretrained(ckpt)
    print("config.model_type =", model.config.model_type)
    print("class =", model.__class__.__name__)

    ids = torch.tensor([[2, 10, 11, 3]])  # bos, user, assistant, eos-ish
    out = model.generate(ids, max_new_tokens=4)
    print("generate shape =", tuple(out.shape))
    print("[PASS] HF CausalLM smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
