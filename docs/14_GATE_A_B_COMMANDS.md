# Gate A (proxy) + Gate B (tokenizer) — pod commands

**Honest status**

| Gate | Full PASS (docs/13) | Runnable now |
|------|---------------------|--------------|
| A | `datasets/clean/corpus-v0.2/` ≥10 GB + license/dedup manifests | **Proxy**: HF samples + identity |
| B | Ablation 32→131k, freeze `latex-v0.2`, eval without `--smoke` | **Yes** via ablation script |

Real Gate A (10 GB clean tree) still needs `build_clean_manifest.py` / `gate_corpus_release.py` — later.

---

## 0) Sync + hf_transfer (required, no fallback)

```bash
cd /workspace/NULLXES-L-TEX-AI-480B-A35B-Instruct
git pull

# REQUIRED — do not unset / do not disable transfer
pip install -U hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
# token already in env on pod if set; else:
# export HF_TOKEN=hf_...
```

---

## Gate A — proxy corpus (run step-by-step; wait for download)

```bash
python scripts/build_identity_corpus.py

# resume-safe: skips complete shards (e.g. pes2o already done)
python scripts/download_local_corpus.py --config configs/datasets_gate_a_proxy.yaml

# ONLY after download prints [done]:
python scripts/validate_corpus.py --manifest datasets/manifests/corpus_gate_a_proxy.json
du -sh datasets/raw/shards/hf_gate_a
ls -la datasets/raw/shards/hf_gate_a/
```

Proxy PASS signal: shards exist, validate OK, wiki/ru+en+science+code+identity present.  
Full Gate A PASS (10 GB + `corpus-v0.2/` tree) = not this script yet.

---

## Gate B — tokenizer ablation → freeze v0.2

```bash
# trains 32k / 64k / 96k / 131k on same Gate A proxy corpus
python scripts/run_tokenizer_ablation.py \
  --config configs/tokenizer_latex_v0.2.yaml \
  --runtime configs/runtime_runpod_rtx_pro_6000.yaml

# report
cat tokenizer/ablation/summary.json
```

### Ablation result (2026-07-19 Gate A proxy)

| vocab | Gate0 passed | note |
|------:|:------------:|------|
| 32k | FAIL | markdown fertility |
| 64k | PASS | smallest PASS |
| 96k | PASS | better EN/RU |
| **131072** | **PASS** | **WINNER** — full Unigram fill, no unused_* pad; matches 20B/A35B width |

Freeze winner:

```bash
rm -rf tokenizer/latex-v0.2
mkdir -p tokenizer/latex-v0.2
cp -a tokenizer/ablation/v131072/. tokenizer/latex-v0.2/
# strip bulky train tmp from freeze
rm -rf tokenizer/latex-v0.2/tmp
python3 - <<'PY'
import json
from pathlib import Path
p = Path("tokenizer/latex-v0.2/meta.json")
m = json.loads(p.read_text())
m["frozen_as"] = "latex-v0.2"
m["ablation_winner"] = 131072
m["note"] = "Gate B winner on corpus_gate_a_proxy; full 131072 Unigram; Stage0a 100M stays on v0.1"
p.write_text(json.dumps(m, indent=2) + "\n")
PY

python scripts/evaluate_tokenizer.py --config configs/tokenizer_latex_v0.2.yaml
```

### Hub upload (do NOT overwrite 100M Stage0a tokenizer)

100M model [NULLXES-L-TEX-100M-Stage0a-v0.1](https://huggingface.co/MagistrTheOne/NULLXES-L-TEX-100M-Stage0a-v0.1) was trained with **v0.1** — replacing its `tokenizer.model` breaks the brain.

Upload v0.2 as its own Hub repo:

```bash
# create once: huggingface-cli repo create NULLXES-L-TEX-Tokenizer-v0.2 --type model
huggingface-cli upload MagistrTheOne/NULLXES-L-TEX-Tokenizer-v0.2 \
  tokenizer/latex-v0.2 . \
  --commit-message "NULLXES-LÆTEX Tokenizer v0.2 (131072 Unigram, Gate B winner)"

# optional: also stash ablation summary next to it
huggingface-cli upload MagistrTheOne/NULLXES-L-TEX-Tokenizer-v0.2 \
  tokenizer/ablation/summary.json ablation_summary.json
```

Single-size train (no ablation):

```bash
python scripts/train_tokenizer.py \
  --config configs/tokenizer_latex_v0.2.yaml \
  --runtime configs/runtime_runpod_rtx_pro_6000.yaml
python scripts/evaluate_tokenizer.py --config configs/tokenizer_latex_v0.2.yaml
```

Do **not** use `--smoke` for Gate B PASS.  
Stage0a 100M stays on `tokenizer/latex-v0.1`. 20B uses **v0.2 only**.

---

## After B PASS

→ Gate C broader 100M eval → Gate D 20B genesis on 1× RTX PRO 6000.
