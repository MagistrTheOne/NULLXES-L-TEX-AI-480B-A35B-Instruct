#!/usr/bin/env python3
"""
Package 20B V1 Weight Genesis for Hugging Face Hub.

  python scripts/package_hf_20b_genesis.py \
    --checkpoint checkpoints/nullxes-latex-20b-v1 \
    --repo-id MagistrTheOne/NULLXES-L-TEX-20B-V1-Genesis
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
- foundation
pipeline_tag: text-generation
---

# NULLXES-LÆTEX-20B-V1-Genesis

**Weight Genesis** of the LÆTEX V1 dense foundation model (~18.8B params, A35B-compatible width).

Developed by **NULLXES** · [nullxesdai.online](https://www.nullxesdai.online/)

This is **foundation bootstrapping**, not Chinchilla pretraining. Tokenizer: `latex-v1` (131072).

## Shape

- `LatexForCausalLM` / NHAT hybrid attention
- L=24, d_model=8192, GQA 64/8, d_ff=22016 → ~18.757B
- Init: muP + DeepNorm residual scaling

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

20B dense V1 → A35B dense → 200B MoE → 480B-A35B MoE

## Contact

NULLXES · @MagistrTheOne · ceo@nullxes.com
"""


def _has_weights(ckpt: Path) -> bool:
    return (ckpt / "model.safetensors").is_file() or (
        ckpt / "model.safetensors.index.json"
    ).is_file() or any(ckpt.glob("model-*-of-*.safetensors"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/nullxes-latex-20b-v1")
    p.add_argument("--repo-id", default="MagistrTheOne/NULLXES-L-TEX-20B-V1-Genesis")
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
        src = ROOT / "tokenizer" / "latex-v1" / name
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
                    f"--commit-message 'NULLXES-LÆTEX-20B-V1-Genesis'"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
