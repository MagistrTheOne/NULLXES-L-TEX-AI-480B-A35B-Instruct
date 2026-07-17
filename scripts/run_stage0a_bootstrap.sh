#!/usr/bin/env bash
# Corpus PASS → full tokenizer (pad to 131k) → Stage0a 100M train → ready for HF
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. Corpus ==="
python scripts/build_identity_corpus.py
python scripts/validate_corpus.py --manifest datasets/manifests/pretrain_stage0.json

echo "=== 2. Tokenizer full (soft Unigram + pad to 131072) ==="
python scripts/train_tokenizer.py --config configs/tokenizer_stage0.yaml
python scripts/evaluate_tokenizer.py --config configs/tokenizer_stage0.yaml || true
cat tokenizer/latex-v0.1/meta.json

echo "=== 3. Stage0a ~100M bootstrap pretrain ==="
python scripts/train_stage0a.py --config configs/stage0a_100m_bootstrap.yaml --device cuda

echo "=== DONE ==="
echo "Checkpoint: checkpoints/nullxes-latex-100m-stage0a-v0.1"
echo "Next: upload to HF as MagistrTheOne/NULLXES-LÆTEX-100M-Stage0a-v0.1"
ls -lah checkpoints/nullxes-latex-100m-stage0a-v0.1 | head -30
