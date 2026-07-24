"""NHAT causal attention (GQA + RoPE/NoPE + KV cache)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from latex.configuration_latex import LatexConfig
from latex.models.embeddings import LatexRMSNorm, apply_rotary_pos_emb


class NHATAttention(nn.Module):
    """Decoder-only causal attention — engine for LÆTEX."""

    def __init__(self, config: LatexConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        if self.hidden_size != self.num_heads * self.head_dim:
            # allow head_dim override
            pass

        bias = config.attention_bias
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=bias)

        self.q_norm = LatexRMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None
        self.k_norm = LatexRMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None

        is_full = config.hybrid_attention and (
            layer_idx % config.full_every == config.full_every - 1
        )
        top = layer_idx >= int(config.num_hidden_layers * (1.0 - config.depth_nope_ratio))
        self.use_rope = not (is_full or top)
        self.is_local = config.hybrid_attention and not is_full
        self.local_window = config.local_window
        self.attention_dropout = config.attention_dropout

        # Plain attributes, not buffers: never enter state_dict.
        self._bias_key: tuple | None = None
        self._bias_cache: Optional[torch.Tensor] = None

    def _causal_bias(
        self,
        q_len: int,
        kv_len: int,
        past_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Additive [1, 1, q_len, kv_len] causal mask, windowed on local layers.

        Query i of the current chunk sits at absolute position `past_len + i`,
        so the same construction covers prefill and cached decode.
        """
        key = (q_len, kv_len, past_len, dtype, device, self.is_local, self.local_window)
        if self._bias_key == key and self._bias_cache is not None:
            return self._bias_cache

        q_pos = torch.arange(past_len, past_len + q_len, device=device).unsqueeze(1)
        k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
        allowed = k_pos <= q_pos
        if self.is_local:
            allowed = allowed & ((q_pos - k_pos) < self.local_window)

        bias = torch.zeros((1, 1, q_len, kv_len), device=device, dtype=dtype)
        bias.masked_fill_(~allowed.view(1, 1, q_len, kv_len), torch.finfo(dtype).min)
        self._bias_key = key
        self._bias_cache = bias
        return bias

    @staticmethod
    def _align_key_mask(attention_mask: torch.Tensor, kv_len: int) -> torch.Tensor:
        """Align an additive mask's key axis to `kv_len`.

        Cached decode often hands a mask whose last dim covers only the current
        chunk (or a short suffix), while `bias` spans the full KV cache. A raw
        `mask[..., -kv_len:]` does not left-pad: if the mask is shorter than
        `kv_len` the slice is a no-op and the later add either crashes or
        silently broadcasts a singleton over every past key.
        """
        mask_k = attention_mask.shape[-1]
        if mask_k == kv_len:
            return attention_mask
        if mask_k > kv_len:
            return attention_mask[..., -kv_len:]
        # Left-pad with 0 (additive "keep") so past keys stay visible and the
        # provided columns align to the newest positions on the right.
        pad = attention_mask.new_zeros(*attention_mask.shape[:-1], kv_len - mask_k)
        return torch.cat([pad, attention_mask], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        if self.use_rope and position_embeddings is not None:
            cos, sin = position_embeddings
            # cos/sin: [1,1,S,D] — slice to current length
            cos = cos[:, :, :q_len, :]
            sin = sin[:, :, :q_len, :]
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            key_states = torch.cat([past_k, key_states], dim=2)
            value_states = torch.cat([past_v, value_states], dim=2)

        past_key_value_out = (key_states, value_states) if use_cache else None

        # GQA expand
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Causal SDPA
        # attention_mask: expected additive [B,1,Q,K] or None
        dropout_p = self.attention_dropout if self.training else 0.0

        kv_len = key_states.shape[2]
        past_len = kv_len - q_len
        needs_window = self.is_local and kv_len > self.local_window
        # `is_causal` only masks correctly when queries and keys are aligned and
        # nothing else has to be masked; anything past that needs an explicit bias.
        aligned_prefill = past_len == 0 and attention_mask is None and not needs_window

        if aligned_prefill:
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                dropout_p=dropout_p,
                is_causal=True,
            )
        else:
            bias = self._causal_bias(
                q_len, kv_len, past_len, query_states.dtype, hidden_states.device
            )
            if attention_mask is not None:
                bias = bias + self._align_key_mask(attention_mask, kv_len).to(bias.dtype)
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=bias,
                dropout_p=dropout_p,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        else:
            attn_weights = None  # SDPA fused path — weights not returned

        return attn_output, attn_weights, past_key_value_out
