# NULLXES-LÆTEX Causal LM — Transformers-compatible architecture
#
# Public surface (Hugging Face ecosystem):
#   LatexConfig / LatexModel / LatexForCausalLM / LatexTokenizer
# Internal engine:
#   NHAT decoder blocks (GQA, RoPE/NoPE, SwiGLU) — swappable FFN for future MoE
#
# We are NOT shipping an isolated nn.Module island.
# NHAT is the engine; LatexForCausalLM is the chassis + standard connector.

from latex.configuration_latex import LatexConfig
from latex.modeling_latex import LatexForCausalLM, LatexModel, LatexPreTrainedModel
from latex.tokenization_latex import LatexTokenizer

__all__ = [
    "LatexConfig",
    "LatexPreTrainedModel",
    "LatexModel",
    "LatexForCausalLM",
    "LatexTokenizer",
]

# Aliases (LÆTEX branding — ASCII class names for Python)
LATEXConfig = LatexConfig
LATEXModel = LatexModel
LATEXForCausalLM = LatexForCausalLM
LATEXTokenizer = LatexTokenizer


def register_with_auto():
    """Register with transformers Auto* classes (idempotent)."""
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    AutoConfig.register("latex", LatexConfig)
    AutoModel.register(LatexConfig, LatexModel)
    AutoModelForCausalLM.register(LatexConfig, LatexForCausalLM)
    AutoTokenizer.register(LatexConfig, LatexTokenizer, LatexTokenizer)


try:
    register_with_auto()
except Exception:
    # transformers not installed yet (Gate0-only env)
    pass
