"""NHAT decoder block — dual residual, FFN swappable for MoE."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from latex.configuration_latex import LatexConfig
from latex.models.attention import NHATAttention
from latex.models.embeddings import LatexRMSNorm
from latex.models.ffn import build_ffn


class NHATDecoderLayer(nn.Module):
    def __init__(self, config: LatexConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NHATAttention(config, layer_idx)
        self.mlp = build_ffn(config)
        self.input_layernorm = LatexRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LatexRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs
