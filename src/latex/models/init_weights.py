"""muP / deepnorm weight initialization for Weight Genesis."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from latex.models.nhat_dense import NHATDense, SwiGLUFFN, Attention


def _parse_residual_scale(init_cfg: dict[str, Any], n_layers: int) -> float:
    rs = init_cfg.get("residual_scale", {})
    if isinstance(rs, dict):
        expr = rs.get("value", "0.02 / sqrt(2*n_layers)")
    else:
        expr = str(rs)
    # Safe eval of documented formula only
    expr = expr.replace("n_layers", str(n_layers))
    allowed = {"sqrt": math.sqrt, "__builtins__": {}}
    return float(eval(expr, allowed, {}))  # noqa: S307 — controlled formula


def apply_mup_init(model: NHATDense, init_cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Width-aware init:
      std_base = init.std (default 0.02)
      input/hidden ~ N(0, std^2)
      residual out-projections scaled by residual_scale (deepnorm-style)
      muP: scale LR-sensitive widths vs base_width (recorded in report)
    """
    std = float(init_cfg.get("std", 0.02))
    base_width = float(init_cfg.get("base_width", model.cfg.d_model))
    width_mult = model.cfg.d_model / base_width
    residual_scale = _parse_residual_scale(init_cfg, model.cfg.n_layers)

    def trunc_normal_(w: torch.Tensor, s: float):
        nn.init.trunc_normal_(w, mean=0.0, std=s, a=-2 * s, b=2 * s)

    with torch.no_grad():
        trunc_normal_(model.tok_emb.weight, std)
        if not model.cfg.tie_embeddings:
            trunc_normal_(model.lm_head.weight, std / math.sqrt(max(width_mult, 1e-8)))

        for layer in model.layers:
            attn: Attention = layer.attn
            ffn: SwiGLUFFN = layer.ffn
            for lin in (attn.wq, attn.wk, attn.wv):
                trunc_normal_(lin.weight, std)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
            # residual out
            trunc_normal_(attn.wo.weight, std * residual_scale)
            if attn.wo.bias is not None:
                nn.init.zeros_(attn.wo.bias)
            trunc_normal_(ffn.w1.weight, std)
            trunc_normal_(ffn.w3.weight, std)
            trunc_normal_(ffn.w2.weight, std * residual_scale)
            for lin in (ffn.w1, ffn.w2, ffn.w3):
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
            if attn.q_norm is not None:
                nn.init.ones_(attn.q_norm.weight)
                nn.init.ones_(attn.k_norm.weight)
            nn.init.ones_(layer.attn_norm.weight)
            nn.init.ones_(layer.ffn_norm.weight)
        nn.init.ones_(model.out_norm.weight)

    return {
        "scheme": "mup",
        "std": std,
        "base_width": base_width,
        "base_depth": init_cfg.get("base_depth"),
        "width_mult": width_mult,
        "residual_scale": residual_scale,
        "residual_scale_type": (init_cfg.get("residual_scale") or {}).get("type", "deepnorm")
        if isinstance(init_cfg.get("residual_scale"), dict)
        else "legacy",
    }
