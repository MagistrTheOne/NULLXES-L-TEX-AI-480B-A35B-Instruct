# Stage1 Weight Genesis (Transformers)

**Not training.** Birth of **LatexForCausalLM** + HF checkpoint.

See also [`docs/11_TRANSFORMERS_CONTRACT.md`](11_TRANSFORMERS_CONTRACT.md).

```bash
pip install -r requirements-stage1.txt
python scripts/init_model.py --config configs/nullxes_latex_7b.yaml
python scripts/smoke_hf_causal.py --checkpoint checkpoints/nullxes-latex-7b
```

## Outputs

```
checkpoints/nullxes-latex-7b/
  config.json              # model_type: latex
  model.safetensors
  generation_config.json
  init_report.json
  tokenizer files…
```

## Load

```python
import latex
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("checkpoints/nullxes-latex-7b")
```
