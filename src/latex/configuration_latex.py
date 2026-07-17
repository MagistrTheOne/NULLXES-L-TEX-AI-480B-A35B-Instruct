"""LÆTEXConfig — HuggingFace PretrainedConfig for NULLXES-LÆTEX Causal LM."""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig


class LatexConfig(PretrainedConfig):
    """
    Configuration for NULLXES-LÆTEX dense (and future MoE) Causal LM.

    Python identifier: LatexConfig (ASCII).
    Product name: NULLXES-LÆTEX / model_type = \"latex\".
    """

    model_type = "latex"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = 8,
        head_dim: int | None = None,
        max_position_embeddings: int = 8192,
        rope_theta: float = 500_000.0,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        qk_norm: bool = True,
        # NHAT hybrid attention
        hybrid_attention: bool = True,
        local_window: int = 4096,
        full_every: int = 4,
        depth_nope_ratio: float = 0.25,
        # Future MoE (dense default)
        ffn_type: str = "dense",  # "dense" | "moe" (moe not implemented yet)
        z_loss_coeff: float = 1.0e-5,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        **kwargs: Any,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        )
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.qk_norm = qk_norm
        self.hybrid_attention = hybrid_attention
        self.local_window = local_window
        self.full_every = full_every
        self.depth_nope_ratio = depth_nope_ratio
        self.ffn_type = ffn_type
        self.z_loss_coeff = z_loss_coeff

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_yaml_model_section(cls, model: dict[str, Any], **kwargs: Any) -> "LatexConfig":
        """Map NULLXES research YAML `model:` block → LatexConfig."""
        hy = model.get("hybrid_attention") or {}
        return cls(
            vocab_size=int(model.get("vocab_size", 131072)),
            hidden_size=int(model["d_model"]),
            intermediate_size=int(model["d_ff"]),
            num_hidden_layers=int(model["n_layers"]),
            num_attention_heads=int(model["n_heads"]),
            num_key_value_heads=int(model.get("n_kv_heads", 8)),
            head_dim=int(model.get("d_head", model["d_model"] // model["n_heads"])),
            max_position_embeddings=int(
                model.get("max_seq_len", model.get("max_seq_len_train", 8192))
            ),
            rope_theta=float(model.get("rope_theta", 500000.0)),
            hidden_act="silu",
            tie_word_embeddings=bool(model.get("tie_embeddings", False)),
            attention_bias=not bool(model.get("no_bias", True)),
            qk_norm=bool(model.get("qk_norm", True)),
            hybrid_attention=bool(hy.get("enabled", False)),
            local_window=int(hy.get("local_window", 4096)),
            full_every=int(hy.get("full_every", 4)),
            depth_nope_ratio=float(model.get("depth_nope_ratio", 0.25)),
            z_loss_coeff=float(model.get("z_loss_coeff", 1.0e-5)),
            architectures=["LatexForCausalLM"],
            **kwargs,
        )


# Branding alias
LATEXConfig = LatexConfig
