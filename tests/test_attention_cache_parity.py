"""Cached decode must match full-sequence forward, including on local layers.

Before the windowed-bias fix, local layers applied their sliding window only
when `past_key_value is None`, so generation attended over the whole KV cache
while training attended over `local_window`. The two paths then disagreed and
QA measured a model that training never produced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latex import LatexConfig, LatexForCausalLM  # noqa: E402


def _tiny_model(local_window: int) -> LatexForCausalLM:
    config = LatexConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        hybrid_attention=True,
        local_window=local_window,
        full_attention_every=4,
    )
    torch.manual_seed(0)
    return LatexForCausalLM(config).eval()


@pytest.mark.parametrize("local_window", [4, 8])
def test_cached_decode_matches_full_forward(local_window: int) -> None:
    model = _tiny_model(local_window)
    ids = torch.randint(0, model.config.vocab_size, (1, 24))

    with torch.no_grad():
        full = model(input_ids=ids).logits

        prefill = model(input_ids=ids[:, :-1], use_cache=True)
        stepwise = model(
            input_ids=ids[:, -1:],
            past_key_values=prefill.past_key_values,
            use_cache=True,
        ).logits

    torch.testing.assert_close(stepwise[:, -1], full[:, -1], rtol=1e-4, atol=1e-4)


def test_local_layer_ignores_tokens_outside_window() -> None:
    """A local layer must not react to a token older than its window."""
    model = _tiny_model(local_window=4)
    attn = model.model.layers[0].self_attn
    assert attn.is_local, "layer 0 is expected to be a local layer in this pattern"

    ids = torch.randint(0, model.config.vocab_size, (1, 16))
    far_past = ids.clone()
    far_past[0, 0] = (far_past[0, 0] + 7) % model.config.vocab_size

    hidden = model.model.embed_tokens(ids)
    hidden_changed = model.model.embed_tokens(far_past)
    cos, sin = model.model.rotary_emb(hidden, seq_len=ids.shape[1])

    with torch.no_grad():
        out_a = attn(hidden, position_embeddings=(cos, sin))[0]
        out_b = attn(hidden_changed, position_embeddings=(cos, sin))[0]

    # Last query is 15 positions away from the edited token, window is 4.
    torch.testing.assert_close(out_a[:, -1], out_b[:, -1], rtol=1e-5, atol=1e-5)
    assert not torch.allclose(out_a[:, 0], out_b[:, 0]), "edited position must change"
