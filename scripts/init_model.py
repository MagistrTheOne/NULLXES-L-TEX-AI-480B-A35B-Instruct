#!/usr/bin/env python3
"""
Stage1 Weight Genesis — create first NULLXES-LÆTEX dense weights + diagnostics.

  python scripts/init_model.py --config configs/nullxes_latex_7b.yaml

Does NOT train. Emits safetensors + init_report.json.
Requires: pip install -r requirements-stage1.txt
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    p = argparse.ArgumentParser(description="NULLXES-LÆTEX Weight Genesis")
    p.add_argument("--config", default="configs/nullxes_latex_7b.yaml")
    p.add_argument("--device", default="cpu", help="cpu | cuda")
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
    )
    args = p.parse_args()

    try:
        import torch
        import yaml
        from safetensors.torch import save_file
    except ImportError as e:
        print(
            "Missing Stage1 deps. Install:\n  pip install -r requirements-stage1.txt\n",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    from latex.models.init_weights import apply_mup_init
    from latex.models.nhat_dense import NHATConfig, NHATDense

    cfg_path = ROOT / args.config
    with cfg_path.open(encoding="utf-8") as f:
        full = yaml.safe_load(f)

    model_cfg = NHATConfig.from_dict(full["model"])
    init_cfg = full.get("init") or {}
    tok_cfg = full.get("tokenizer") or {}
    ckpt_cfg = full.get("checkpoint") or {}
    out_dir = ROOT / ckpt_cfg.get("output_dir", "checkpoints/nullxes-latex-7b")
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        args.dtype
    ]
    device = torch.device(args.device)

    model = NHATDense(model_cfg).to(device=device, dtype=dtype)
    mup_meta = apply_mup_init(model, init_cfg)

    # Diagnostics
    nan_detected = False
    weight_stds = []
    dead_layers = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not torch.isfinite(p).all():
                nan_detected = True
            if p.ndim >= 2:
                s = p.float().std().item()
                weight_stds.append(s)
                if s < 1e-8:
                    dead_layers += 1

    mean_std = sum(weight_stds) / max(len(weight_stds), 1)
    n_params = model.count_parameters()

    # Smoke forward (tiny) — genesis health
    smoke_ok = True
    smoke_err = None
    try:
        ids = torch.randint(0, min(1024, model_cfg.vocab_size), (1, 8), device=device)
        logits = model(ids)
        if not torch.isfinite(logits).all():
            smoke_ok = False
            smoke_err = "non-finite logits"
    except Exception as e:  # noqa: BLE001
        smoke_ok = False
        smoke_err = str(e)

    # Save weights
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    weights_path = out_dir / "model.safetensors"
    save_file(state, str(weights_path))

    # config.json — portable model card for checkpoint
    config_json = {
        "model_type": "nullxes_nhat_dense",
        "architectures": ["NHATDense"],
        **{k: getattr(model_cfg, k) for k in model_cfg.__dataclass_fields__},
        "tokenizer": tok_cfg,
        "init": mup_meta,
        "stage": full.get("stage", "weight_genesis"),
    }
    (out_dir / "config.json").write_text(
        json.dumps(config_json, indent=2) + "\n", encoding="utf-8"
    )

    # Bind tokenizer specials into checkpoint tree
    specials_src = ROOT / tok_cfg.get(
        "special_tokens_file", "tokenizer/latex-v0.1/special_tokens.json"
    )
    if specials_src.is_file():
        shutil.copy2(specials_src, out_dir / "special_tokens.json")
    # tokenizer.json stub — full SP model copied when Gate0 artifact exists
    tok_json = {
        "name": tok_cfg.get("name", "nullxes-latex-v0.1"),
        "vocab_size": tok_cfg.get("vocab_size", model_cfg.vocab_size),
        "special_tokens_file": "special_tokens.json",
        "model_file": "tokenizer.model",
        "note": "Copy tokenizer.model from tokenizer/latex-v0.1/ after Gate0 PASS",
    }
    (out_dir / "tokenizer.json").write_text(
        json.dumps(tok_json, indent=2) + "\n", encoding="utf-8"
    )
    model_src = ROOT / tok_cfg.get("artifact_dir", "tokenizer/latex-v0.1") / "tokenizer.model"
    if model_src.is_file():
        shutil.copy2(model_src, out_dir / "tokenizer.model")

    report = {
        "parameters": f"{n_params / 1e9:.3f}B",
        "parameters_exact": n_params,
        "mean_weight_std": round(mean_std, 6),
        "dead_layers": dead_layers,
        "nan_detected": nan_detected,
        "embedding_shape": list(model.tok_emb.weight.shape),
        "lm_head_shape": list(model.lm_head.weight.shape),
        "smoke_forward_ok": smoke_ok,
        "smoke_error": smoke_err,
        "mup": mup_meta,
        "dtype": args.dtype,
        "device": args.device,
        "config": str(cfg_path),
        "weights": str(weights_path.relative_to(ROOT)),
        "passed": (not nan_detected) and smoke_ok and dead_layers == 0,
    }
    report_path = out_dir / "init_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"[Weight Genesis] wrote {out_dir}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
