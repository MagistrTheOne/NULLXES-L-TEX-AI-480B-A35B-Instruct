"""Re-export standard Transformers modeling outputs for LÆTEX."""

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

__all__ = ["BaseModelOutputWithPast", "CausalLMOutputWithPast"]
