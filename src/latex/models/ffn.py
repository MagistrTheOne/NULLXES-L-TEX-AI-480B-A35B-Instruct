"""FFN modules — Dense SwiGLU now; MoE slot later (ffn_type)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from latex.configuration_latex import LatexConfig


class DenseSwiGLUFFN(nn.Module):
    def __init__(self, config: LatexConfig):
        super().__init__()
        bias = config.attention_bias
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def build_ffn(config: LatexConfig) -> nn.Module:
    """Factory — dense today; `moe` reserved for future LÆTEX MoE."""
    if config.ffn_type == "dense":
        return DenseSwiGLUFFN(config)
    if config.ffn_type == "moe":
        raise NotImplementedError(
            "MoE FFN not implemented yet — architecture slot reserved (NHAT MoE expansion)."
        )
    raise ValueError(f"Unknown ffn_type={config.ffn_type}")
