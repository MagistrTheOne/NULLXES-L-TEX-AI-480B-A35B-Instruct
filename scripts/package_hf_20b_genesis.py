#!/usr/bin/env python3
"""
Package 20B Weight Genesis for Hugging Face Hub (honest research brick).

  python scripts/package_hf_20b_genesis.py \
    --checkpoint checkpoints/nullxes-latex-20b \
    --repo-id MagistrTheOne/NULLXES-L-TEX-20B-Genesis-v0.1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

README = """---
language:
- en
- ru
license: other
library_name: transformers
tags:
- nullxes
- latex
- causal-lm
- genesis
- research
- moe-roadmap
pipeline_tag: text-generation
---

# NULLXES-LÆTEX-20B-Genesis-v0.1

**Weight Genesis** scaffold of the NULLXES-LÆTEX Stage2 dense proxy (~18.8B params, A35B-compatible width).

Developed by **NULLXES** · [nullxesdai.online](https://www.nullxesdai.online/) · Hub: [MagistrTheOne](https://huggingface.co/MagistrTheOne)

> **Status:** research brick / architectural checkpoint.  
> **Not** a chat model. Random muP-init weights — expect nonsense generations until Stage2 train.  
> **Next rework / first trained weights target:** **August 2026**.

## What this is

- Own architecture: `LatexForCausalLM` (`model_type=latex`), NHAT hybrid attention
- Shape: L=24, d_model=8192, GQA 64/8, d_ff=22016 → **~18.757B**
- Tokenizer: [NULLXES-L-TEX-Tokenizer-v0.2](https://huggingface.co/MagistrTheOne/NULLXES-L-TEX-Tokenizer-v0.2) (131072 Unigram, full fill)
- Init: muP + DeepNorm residual scaling, bf16 sharded safetensors
- Intended path: Stage2 pretrain → A35B depth expand → future 480B-A35B MoE

## What this is NOT

- Not instruct / not SFT / not a Digital Employee personality
- Not trained language competence (identity QA will fail on purpose)
- Not distilled from Qwen / Llama / Mistral / DeepSeek / GLM
- Not a replacement for [100M Stage0a](https://huggingface.co/MagistrTheOne/NULLXES-L-TEX-100M-Stage0a-v0.1) (that one *is* identity-trained)

## Load

```python
import torch
from transformers import AutoModelForCausalLM, AutoConfig

repo = "REPO_ID"
config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
)
```

## Roadmap

| When | Milestone |
|------|-----------|
| now | Genesis v0.1 (this repo) |
| next week | Stage2 ZeRO-3 smoke + ~100M tokens mid-eval on RTX PRO 6000 |
| **Aug 2026** | First trained 20B weights / card refresh |
| later | A35B dense → 480B-A35B MoE (cluster) |

## Contact

NULLXES · @MagistrTheOne · ceo@nullxes.com
"""


def _has_weights(ckpt: Path) -> bool:
    return (ckpt / "model.safetensors").is_file() or (
        ckpt / "model.safetensors.index.json"
    ).is_file() or any(ckpt.glob("model-*-of-*.safetensors"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/nullxes-latex-20b")
    p.add_argument("--repo-id", default="MagistrTheOne/NULLXES-L-TEX-20B-Genesis-v0.1")
    args = p.parse_args()
    ckpt = ROOT / args.checkpoint
    if not _has_weights(ckpt):
        print(f"[fail] missing weights in {ckpt}", file=sys.stderr)
        return 2
    if not (ckpt / "init_report.json").is_file():
        print(f"[warn] no init_report.json in {ckpt}", file=sys.stderr)

    src_pkg = ROOT / "src" / "latex"
    dst_pkg = ckpt / "latex"
    if dst_pkg.exists():
        shutil.rmtree(dst_pkg)
    shutil.copytree(
        src_pkg,
        dst_pkg,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "nhat_dense.py"),
    )

    cfg_path = ckpt / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["model_type"] = "latex"
    cfg["architectures"] = ["LatexForCausalLM"]
    cfg["auto_map"] = {
        "AutoConfig": "latex.configuration_latex.LatexConfig",
        "AutoModel": "latex.modeling_latex.LatexModel",
        "AutoModelForCausalLM": "latex.modeling_latex.LatexForCausalLM",
        "AutoTokenizer": "latex.tokenization_latex.LatexTokenizer",
    }
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    (ckpt / "README.md").write_text(
        README.replace("REPO_ID", args.repo_id),
        encoding="utf-8",
    )

    for name in ("vocab.json", "tokenizer.model", "special_tokens.json", "meta.json"):
        src = ROOT / "tokenizer" / "latex-v0.2" / name
        if src.is_file() and not (ckpt / name).is_file():
            shutil.copy2(src, ckpt / name)

    print(
        json.dumps(
            {
                "checkpoint": str(ckpt),
                "repo_id": args.repo_id,
                "sharded": (ckpt / "model.safetensors.index.json").is_file()
                or any(ckpt.glob("model-*-of-*.safetensors")),
                "upload": (
                    f"hf upload {args.repo_id} {ckpt.as_posix()} . "
                    f"--commit-message 'NULLXES-LÆTEX-20B-Genesis-v0.1'"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
