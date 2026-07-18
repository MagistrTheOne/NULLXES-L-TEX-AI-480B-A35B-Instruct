# Gate A (proxy) + Gate B (tokenizer) — pod commands

**Honest status**

| Gate | Full PASS (docs/13) | Runnable now |
|------|---------------------|--------------|
| A | `datasets/clean/corpus-v0.2/` ≥10 GB + license/dedup manifests | **Proxy**: HF samples + identity |
| B | Ablation 32→131k, freeze `latex-v0.2`, eval without `--smoke` | **Yes** via ablation script |

Real Gate A (10 GB clean tree) still needs `build_clean_manifest.py` / `gate_corpus_release.py` — later.

---

## 0) Sync

```bash
cd /workspace/NULLXES-L-TEX-AI-480B-A35B-Instruct
git pull
# HF token if gated datasets later
export HF_TOKEN=hf_...   # optional for public wiki/peS2o
```

---

## Gate A — proxy corpus

```bash
# identity + rules (already done if v0.1.2 fresh)
python scripts/build_identity_corpus.py

# download licensed HF samples → merge with identity
python scripts/download_local_corpus.py --config configs/datasets_gate_a_proxy.yaml

# validate manifests
python scripts/validate_corpus.py --manifest datasets/manifests/pretrain_stage0.json
python scripts/validate_corpus.py --manifest datasets/manifests/corpus_gate_a_proxy.json

# size check (proxy bar — grow toward 10GB)
du -sh datasets/raw/shards/hf_gate_a datasets/raw/shards/identity
wc -l datasets/manifests/corpus_gate_a_proxy.json
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

Freeze winner (example 64k — pick by summary, not habit):

```bash
mkdir -p tokenizer/latex-v0.2
cp -r tokenizer/ablation/v64000/* tokenizer/latex-v0.2/
# edit meta.json: set vocab_size_chosen + note why

# final eval (no --smoke)
python scripts/evaluate_tokenizer.py --config configs/tokenizer_latex_v0.2.yaml
# if artifact_dir in yaml still points to latex-v0.2 — OK after freeze
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
