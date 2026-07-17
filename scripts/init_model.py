#!/usr/bin/env python3
"""
NULLXES-LÆTEX Weight Genesis — Transformers-compatible Causal LM.

Creates LatexForCausalLM (not an island nn.Module), applies muP init,
writes HF checkpoint via save_pretrained().

  python scripts/init_model.py --config configs/nullxes_latex_7b.yaml

Requires: pip install -r requirements-stage1.txt
Does NOT train.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description="NULLXES-LÆTEX Weight Genesis (HF CausalLM)")
    p.add_argument("--config", default="configs/nullxes_latex_7b.yaml")
    p.add_argument("--device", default="cpu", help="cpu | cuda")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    args = p.parse_args()

    try:
        import torch
        import yaml
        from transformers import GenerationConfig
    except ImportError as e:
        print(
            "Missing Stage1 deps.\n  pip install -r requirements-stage1.txt\n",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    import latex  # registers Auto*
    from latex import LatexConfig, LatexForCausalLM, LatexTokenizer
    from latex.models.init_weights import apply_mup_init

    cfg_path = ROOT / args.config
    with cfg_path.open(encoding="utf-8") as f:
        full = yaml.safe_load(f)

    config = LatexConfig.from_yaml_model_section(full["model"])
    config.architectures = ["LatexForCausalLM"]
    config.model_type = "latex"

    init_cfg = full.get("init") or {}
    tok_cfg = full.get("tokenizer") or {}
    ckpt_cfg = full.get("checkpoint") or {}
    out_dir = ROOT / ckpt_cfg.get("output_dir", "checkpoints/nullxes-latex-7b")
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    device = torch.device(args.device)

    model = LatexForCausalLM(config)
    model = model.to(device=device, dtype=dtype)
    mup_meta = apply_mup_init(model, init_cfg)

    # Diagnostics
    nan_detected = False
    weight_stds = []
    dead = 0
    with torch.no_grad():
        for _, p in model.named_parameters():
            if not torch.isfinite(p).all():
                nan_detected = True
            if p.ndim >= 2:
                s = float(p.float().std().item())
                weight_stds.append(s)
                if s < 1e-8:
                    dead += 1
    mean_std = sum(weight_stds) / max(len(weight_stds), 1)
    n_params = sum(p.numel() for p in model.parameters())

    smoke_ok, smoke_err = True, None
    try:
        ids = torch.randint(0, min(1024, config.vocab_size), (1, 8), device=device)
        out = model(input_ids=ids, labels=ids)
        if out.logits is None or not torch.isfinite(out.logits).all():
            smoke_ok, smoke_err = False, "non-finite logits"
        if out.loss is not None and not torch.isfinite(out.loss):
            smoke_ok, smoke_err = False, "non-finite loss"
    except Exception as e:  # noqa: BLE001
        smoke_ok, smoke_err = False, str(e)

    # HF save_pretrained
    model.save_pretrained(out_dir, safe_serialization=True)
    config.save_pretrained(out_dir)

    gen_cfg = GenerationConfig(
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
    )
    gen_cfg.save_pretrained(out_dir)

    # Tokenizer bind
    specials = ROOT / tok_cfg.get(
        "special_tokens_file", "tokenizer/latex-v0.1/special_tokens.json"
    )
    art = ROOT / tok_cfg.get("artifact_dir", "tokenizer/latex-v0.1")
    sp_model = art / "tokenizer.model"
    tok = LatexTokenizer(
        vocab_file=str(sp_model) if sp_model.is_file() else None,
        special_tokens_map_file=str(specials) if specials.is_file() else None,
    )
    tok.save_pretrained(out_dir)
    if specials.is_file():
        shutil.copy2(specials, out_dir / "special_tokens.json")
    if sp_model.is_file():
        shutil.copy2(sp_model, out_dir / "tokenizer.model")

    emb = model.get_input_embeddings().weight
    report = {
        "model_class": "LatexForCausalLM",
        "model_type": "latex",
        "transformers_compatible": True,
        "parameters": f"{n_params / 1e9:.3f}B",
        "parameters_exact": n_params,
        "mean_weight_std": round(mean_std, 6),
        "dead_layers": dead,
        "nan_detected": nan_detected,
        "embedding_shape": list(emb.shape),
        "lm_head_shape": list(model.lm_head.weight.shape),
        "smoke_forward_ok": smoke_ok,
        "smoke_error": smoke_err,
        "mup": mup_meta,
        "dtype": args.dtype,
        "device": args.device,
        "config_source": str(cfg_path),
        "checkpoint_dir": str(out_dir.relative_to(ROOT)),
        "save_pretrained": True,
        "passed": (not nan_detected) and smoke_ok and dead == 0,
    }
    (out_dir / "init_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"[Weight Genesis] HF checkpoint → {out_dir}")
    print("Load test: AutoModelForCausalLM.from_pretrained(path) after `import latex`")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
