"""NULLXES Hybrid Attention Transformer — dense (Weight Genesis)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NHATConfig:
    n_layers: int = 32
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    d_head: int = 128
    d_ff: int = 11008
    vocab_size: int = 131072
    max_seq_len: int = 8192
    rope_theta: float = 500000.0
    tie_embeddings: bool = False
    qk_norm: bool = True
    no_bias: bool = True
    hybrid_enabled: bool = True
    local_window: int = 4096
    full_every: int = 4
    depth_nope_ratio: float = 0.25
    z_loss_coeff: float = 1.0e-5

    @classmethod
    def from_dict(cls, m: dict[str, Any]) -> "NHATConfig":
        hy = m.get("hybrid_attention") or {}
        return cls(
            n_layers=int(m["n_layers"]),
            d_model=int(m["d_model"]),
            n_heads=int(m["n_heads"]),
            n_kv_heads=int(m["n_kv_heads"]),
            d_head=int(m.get("d_head", m["d_model"] // m["n_heads"])),
            d_ff=int(m["d_ff"]),
            vocab_size=int(m["vocab_size"]),
            max_seq_len=int(m.get("max_seq_len", m.get("max_seq_len_train", 8192))),
            rope_theta=float(m.get("rope_theta", 500000.0)),
            tie_embeddings=bool(m.get("tie_embeddings", False)),
            qk_norm=bool(m.get("qk_norm", False)),
            no_bias=bool(m.get("no_bias", True)),
            hybrid_enabled=bool(hy.get("enabled", False)),
            local_window=int(hy.get("local_window", 4096)),
            full_every=int(hy.get("full_every", 4)),
            depth_nope_ratio=float(m.get("depth_nope_ratio", 0.25)),
            z_loss_coeff=float(m.get("z_loss_coeff", 1.0e-5)),
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q,k: [B, H, T, D]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


class Attention(nn.Module):
    def __init__(self, cfg: NHATConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.d_head = cfg.d_head
        bias = not cfg.no_bias
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=bias)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=bias)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=bias)
        self.wo = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=bias)
        self.q_norm = RMSNorm(cfg.d_head) if cfg.qk_norm else None
        self.k_norm = RMSNorm(cfg.d_head) if cfg.qk_norm else None
        # use NoPE on full layers or top depth fraction
        is_full = cfg.hybrid_enabled and (layer_idx % cfg.full_every == cfg.full_every - 1)
        top = layer_idx >= int(cfg.n_layers * (1.0 - cfg.depth_nope_ratio))
        self.use_rope = not (is_full or top)
        self.is_local = cfg.hybrid_enabled and not is_full
        self.local_window = cfg.local_window

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.wk(x).view(b, t, self.n_kv, self.d_head).transpose(1, 2)
        v = self.wv(x).view(b, t, self.n_kv, self.d_head).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if self.use_rope:
            q, k = apply_rope(q, k, cos[:t, : self.d_head], sin[:t, : self.d_head])
        # GQA expand
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # SDPA causal; local window approximated via attn mask when needed
        if self.is_local and t > self.local_window:
            # Build banded causal mask [T, T]
            idx = torch.arange(t, device=x.device)
            mask = (idx[None, :] <= idx[:, None]) & (
                idx[:, None] - idx[None, :] < self.local_window
            )
            mask = mask.float()
            mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        return self.wo(y)


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: NHATConfig):
        super().__init__()
        bias = not cfg.no_bias
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=bias)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=bias)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class NHATBlock(nn.Module):
    def __init__(self, cfg: NHATConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, layer_idx)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class NHATDense(nn.Module):
    def __init__(self, cfg: NHATConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([NHATBlock(cfg, i) for i in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.rope = RotaryEmbedding(cfg.d_head, cfg.max_seq_len, cfg.rope_theta)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        x = self.tok_emb(input_ids)
        cos, sin = self.rope(t)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.out_norm(x)
        return self.lm_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
