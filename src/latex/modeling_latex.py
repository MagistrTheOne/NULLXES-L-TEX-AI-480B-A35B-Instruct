"""
NULLXES-LÆTEX Causal LM — Transformers-compatible surface.

Public API:
  LatexPreTrainedModel
  LatexModel              — decoder backbone
  LatexForCausalLM        — + LM head + GenerationMixin

Internal engine: NHAT decoder layers (not a public island).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel

from latex.configuration_latex import LatexConfig
from latex.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from latex.models.embeddings import LatexRMSNorm, LatexRotaryEmbedding
from latex.models.nhat_block import NHATDecoderLayer


def _normalize_past_key_values(past_key_values) -> Tuple[Optional[Tuple], int]:
    """Convert HF Cache / legacy tuple → (legacy_or_None, past_seq_len).

    transformers≥4.46 generate() may pass an empty DynamicCache whose layers
    are (None, None) — treat that as no past.
    """
    if past_key_values is None:
        return None, 0

    if hasattr(past_key_values, "get_seq_length"):
        try:
            seq_len = int(past_key_values.get_seq_length())
        except Exception:  # noqa: BLE001
            seq_len = 0
        if seq_len == 0:
            return None, 0
        if hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()

    if not past_key_values:
        return None, 0

    layer0 = past_key_values[0]
    if layer0 is None:
        return None, 0
    if isinstance(layer0, (tuple, list)):
        key = layer0[0] if len(layer0) > 0 else None
        if key is None or not hasattr(key, "shape"):
            return None, 0
        return past_key_values, int(key.shape[2])

    return None, 0


def _layer_past(past_key_values, idx: int):
    if past_key_values is None:
        return None
    try:
        pkv = past_key_values[idx]
    except (IndexError, KeyError, TypeError):
        return None
    if pkv is None:
        return None
    if isinstance(pkv, (tuple, list)) and (len(pkv) < 2 or pkv[0] is None or pkv[1] is None):
        return None
    return pkv


class LatexPreTrainedModel(PreTrainedModel):
    config_class = LatexConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NHATDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_cache_class = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LatexModel(LatexPreTrainedModel):
    """Decoder-only NHAT backbone."""

    def __init__(self, config: LatexConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [NHATDecoderLayer(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = LatexRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LatexRotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds")
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You must specify input_ids or inputs_embeds")

        past_key_values, past_key_values_length = _normalize_past_key_values(past_key_values)

        if position_ids is None:
            device = inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds
        # RoPE cache for full sequence length including past
        total_len = seq_length + past_key_values_length
        cos, sin = self.rotary_emb(hidden_states, seq_len=total_len)
        # For current chunk, offset into cos/sin when past exists
        if past_key_values_length > 0:
            cos = cos[:, :, past_key_values_length : past_key_values_length + seq_length, :]
            sin = sin[:, :, past_key_values_length : past_key_values_length + seq_length, :]
        else:
            cos = cos[:, :, :seq_length, :]
            sin = sin[:, :, :seq_length, :]
        position_embeddings = (cos, sin)

        # Build additive attention mask if provided (2D → 4D)
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: [B, S] with 1=keep
            # For SDPA with is_causal, often better leave None when no padding
            if attention_mask.dim() == 2:
                # If all ones and no past — rely on is_causal
                if not torch.all(attention_mask == 1) or past_key_values_length > 0:
                    # Expand to [B,1,Q,K]
                    k_len = seq_length + past_key_values_length
                    # pad mask for keys if past
                    if attention_mask.shape[-1] != k_len:
                        # assume mask covers full sequence including past when provided shorter
                        pass
                    expanded = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                    attn_mask = (1.0 - expanded) * torch.finfo(hidden_states.dtype).min
            else:
                attn_mask = attention_mask

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = _layer_past(past_key_values, idx)

            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_states,
                    attn_mask,
                    position_embeddings,
                    past_key_value,
                    output_attentions,
                    False,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attn_mask,
                    position_embeddings=position_embeddings,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LatexForCausalLM(LatexPreTrainedModel, GenerationMixin):
    """
    NULLXES-LÆTEX Causal LM — public Transformers entrypoint.

    Supports: forward(labels=...), generate(), save_pretrained / from_pretrained.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LatexConfig):
        super().__init__(config)
        self.model = LatexModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_decoder(self):
        return self.model

    def set_decoder(self, decoder):
        self.model = decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits,) + (
                outputs.past_key_values,
                outputs.hidden_states,
                outputs.attentions,
            )
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        _, past_len = _normalize_past_key_values(past_key_values)
        if past_len > 0:
            input_ids = input_ids[:, -1:]
        elif past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
            # Empty Cache from generate() — drop it so first step is cache-less.
            if past_key_values.get_seq_length() == 0:
                past_key_values = None

        model_inputs = (
            {"inputs_embeds": inputs_embeds}
            if inputs_embeds is not None and past_key_values is None
            else {"input_ids": input_ids}
        )
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


# Branding aliases
LATEXModel = LatexModel
LATEXForCausalLM = LatexForCausalLM
LATEXPreTrainedModel = LatexPreTrainedModel
