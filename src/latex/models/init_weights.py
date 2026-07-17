"""muP / deepnorm weight initialization for LÆTEX Weight Genesis."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from latex.models.attention import NHATAttention
from latex.models.ffn import DenseSwiGLUFFN


def _parse_residual_scale(init_cfg: dict[str, Any], n_layers: int) -> float:
    rs = init_cfg.get("residual_scale", {})
    if isinstance(rs, dict):
        expr = rs.get("value", "0.02 / sqrt(2*n_layers)")
    else:
        expr = str(rs) if rs else "0.02 / sqrt(2*n_layers)"
    expr = expr.replace("n_layers", str(n_layers))
    return float(eval(expr, {"sqrt": math.sqrt, "__builtins__": {}}, {}))  # noqa: S307


def apply_mup_init(model: nn.Module, init_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Initialize LatexForCausalLM / LatexModel weights.
    Targets: embed_tokens, layers.*.self_attn, layers.*.mlp, lm_head, norms.
    """
    init_cfg = init_cfg or {}
    std = float(init_cfg.get("std", getattr(model.config, "initializer_range", 0.02)))
    n_layers = int(model.config.num_hidden_layers)
    base_width = float(init_cfg.get("base_width", model.config.hidden_size))
    width_mult = model.config.hidden_size / max(base_width, 1e-8)
    residual_scale = _parse_residual_scale(init_cfg, n_layers)

    def trunc_normal_(w: torch.Tensor, s: float):
        nn.init.trunc_normal_(w, mean=0.0, std=s, a=-2 * s, b=2 * s)

    # Resolve backbone
    backbone = model.model if hasattr(model, "model") and hasattr(model.model, "layers") else model

    with torch.no_grad():
        if hasattr(backbone, "embed_tokens"):
            trunc_normal_(backbone.embed_tokens.weight, std)

        if hasattr(model, "lm_head") and model.lm_head is not None:
            if not model.config.tie_word_embeddings:
                trunc_normal_(
                    model.lm_head.weight, std / math.sqrt(max(width_mult, 1e-8))
                )

        for layer in backbone.layers:
            attn: NHATAttention = layer.self_attn
            for lin in (attn.q_proj, attn.k_proj, attn.v_proj):
                trunc_normal_(lin.weight, std)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
            trunc_normal_(attn.o_proj.weight, std * residual_scale)
            if attn.o_proj.bias is not None:
                nn.init.zeros_(attn.o_proj.bias)
            if attn.q_norm is not None:
                nn.init.ones_(attn.q_norm.weight)
                nn.init.ones_(attn.k_norm.weight)

            mlp = layer.mlp
            if isinstance(mlp, DenseSwiGLUFFN):
                trunc_normal_(mlp.gate_proj.weight, std)
                trunc_normal_(mlp.up_proj.weight, std)
                trunc_normal_(mlp.down_proj.weight, std * residual_scale)
                for lin in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
                    if lin.bias is not None:
                        nn.init.zeros_(lin.bias)

            nn.init.ones_(layer.input_layernorm.weight)
            nn.init.ones_(layer.post_attention_layernorm.weight)

        if hasattr(backbone, "norm"):
            nn.init.ones_(backbone.norm.weight)

    return {
        "scheme": init_cfg.get("scheme", "mup"),
        "std": std,
        "base_width": base_width,
        "base_depth": init_cfg.get("base_depth"),
        "width_mult": width_mult,
        "residual_scale": residual_scale,
        "residual_scale_type": (init_cfg.get("residual_scale") or {}).get("type", "deepnorm")
        if isinstance(init_cfg.get("residual_scale"), dict)
        else "legacy",
    }
