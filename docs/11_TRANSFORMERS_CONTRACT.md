# NULLXES-LÆTEX Causal LM — Transformers Contract

> We are not writing a "PyTorch toy module".  
> We create the first **NULLXES-LÆTEX class in the Hugging Face Transformers ecosystem**.  
> **NHAT is the internal engine**; outside, the model must behave as a full **CausalLM**.

## Public surface

| Class | Role |
|-------|------|
| `LatexConfig` | `PretrainedConfig`, `model_type="latex"` |
| `LatexModel` | Decoder backbone |
| `LatexForCausalLM` | Causal LM + `GenerationMixin` |
| `LatexTokenizer` | HF tokenizer API |

```python
import latex  # registers Auto*
from transformers import AutoModelForCausalLM, AutoConfig

model = AutoModelForCausalLM.from_pretrained("./checkpoints/nullxes-latex-7b")
print(model.config.model_type)  # latex
out = model.generate(input_ids, max_new_tokens=32)
```

## Engine (internal)

```
Token Embedding → NHATDecoderLayer×N → RMSNorm → LM Head
```

FFN factory `build_ffn(config)` — `ffn_type="dense"` now; `"moe"` reserved.

## Files

```
src/latex/
  configuration_latex.py
  modeling_latex.py
  modeling_outputs.py
  tokenization_latex.py
  __init__.py          # AutoConfig / AutoModel / AutoModelForCausalLM register
  models/
    attention.py       # NHATAttention (causal, GQA, KV cache)
    embeddings.py
    ffn.py             # DenseSwiGLU + MoE slot
    nhat_block.py
    init_weights.py
```

## Weight Genesis

```bash
pip install -r requirements-stage1.txt
python scripts/init_model.py --config configs/nullxes_latex_7b.yaml
python scripts/smoke_hf_causal.py --checkpoint checkpoints/nullxes-latex-7b
```

Checkpoint must contain: `config.json`, `model.safetensors`, `generation_config.json`, `init_report.json`, tokenizer files.

## Forbidden

- Isolated `nn.Module` without HF API  
- Foreign checkpoints / copying Llama-Qwen weights  
- Custom trainer framework replacing Trainer/Accelerate  

## Definition of Done (7B Genesis)

- [ ] Own `LatexConfig`  
- [ ] Own `LatexForCausalLM`  
- [ ] `save_pretrained` / `from_pretrained`  
- [ ] AutoModelForCausalLM registration  
- [ ] `generate()`  
- [ ] Tokenizer bind  
- [ ] `init_report.json`  
- [ ] NHAT ready for MoE FFN swap later  
