#!/usr/bin/env python3
"""
Package Stage0a checkpoint for Hugging Face Hub upload.

Copies `src/latex/` into the checkpoint, sets auto_map, writes README.

  python scripts/package_hf_release.py \
    --checkpoint checkpoints/nullxes-latex-100m-stage0a-v0.1 \
    --repo-id MagistrTheOne/NULLXES-L-TEX-100M-Stage0a-v0
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
- stage0a
- research
pipeline_tag: text-generation
---

# NULLXES-LÆTEX-100M-Stage0a-v0

**First trained brain** of the NULLXES-LÆTEX family (~102M params, dense NHAT).

Developed by **NULLXES** · [nullxesdai.online](https://www.nullxesdai.online/) · org on Hub via [MagistrTheOne](https://huggingface.co/MagistrTheOne)

## What this is

- Own architecture (`LatexForCausalLM`, `model_type=latex`)
- Own tokenizer (NULLXES-LÆTEX v0.1, vocab export 131072, ~4k real Unigram pieces + unused pad)
- Bootstrap pretrain on NULLXES identity + repo code corpus (~50M tokens)
- Intended to answer as **LÆTEX / NULLXES-LÆTEX**, not as ChatGPT/Claude/Llama

## What this is NOT

- Not a general-purpose LLM
- Not Chinchilla-scale pretrain (tiny corpus → heavy memorization / overfit)
- Not the 7B genesis (separate architectural checkpoint)
- Not distilled from Qwen/Llama/Mistral/DeepSeek

## Load

```python
import torch
from transformers import AutoModelForCausalLM, AutoConfig

# registers custom classes
repo = "MagistrTheOne/NULLXES-L-TEX-100M-Stage0a-v0"
config = AutoConfig.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo, trust_remote_code=True, torch_dtype=torch.bfloat16
)
```

Tokenizer artifacts ship in-repo (`tokenizer.model`, `special_tokens.json`). Prefer loading via the research package `LatexTokenizer` from the [NULLXES-LÆTEX GitHub](https://github.com/MagistrTheOne/NULLXES-L-TEX-AI-480B-A35B-Instruct) when doing local QA.

## Identity

Correct self-name: **NULLXES-LÆTEX** (short: **LÆTEX**), built by NULLXES for Digital Employees.

## Training note

Stage0a bootstrap: ~50M tokens, final train loss ~0.01 on a small identity/code mix.
Treat as research brick #1, not production intelligence.
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/nullxes-latex-100m-stage0a-v0.1")
    p.add_argument("--repo-id", default="MagistrTheOne/NULLXES-L-TEX-100M-Stage0a-v0")
    args = p.parse_args()
    ckpt = ROOT / args.checkpoint
    if not (ckpt / "model.safetensors").is_file():
        print(f"[fail] missing safetensors in {ckpt}", file=sys.stderr)
        return 2

    # Ship modeling code for trust_remote_code
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
    cfg["transformers_version"] = cfg.get("transformers_version", "4.46.0")
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    (ckpt / "README.md").write_text(
        README.replace("MagistrTheOne/NULLXES-L-TEX-100M-Stage0a-v0", args.repo_id),
        encoding="utf-8",
    )

    # Manifest snapshot for Hub
    man = ROOT / "datasets" / "manifests" / "pretrain_stage0.json"
    if man.is_file():
        shutil.copy2(man, ckpt / "dataset_manifest.json")

    # Ensure vocab.json present (padded)
    if not (ckpt / "vocab.json").is_file():
        src_v = ROOT / "tokenizer" / "latex-v0.1" / "vocab.json"
        if src_v.is_file():
            shutil.copy2(src_v, ckpt / "vocab.json")

    print(
        json.dumps(
            {
                "checkpoint": str(ckpt),
                "repo_id": args.repo_id,
                "has_latex_pkg": (ckpt / "latex" / "modeling_latex.py").is_file(),
                "has_readme": (ckpt / "README.md").is_file(),
                "has_safetensors": (ckpt / "model.safetensors").is_file(),
                "upload": (
                    f"huggingface-cli upload {args.repo_id} {ckpt.as_posix()} . "
                    f"--commit-message 'NULLXES-LÆTEX-100M-Stage0a-v0 bootstrap'"
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
