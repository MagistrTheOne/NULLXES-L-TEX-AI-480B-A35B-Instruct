#!/usr/bin/env bash
# Train / freeze tokenizer v0.3 under 20B-G/tokenizer/latex-v0.3/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
SMOKE="${1:-}"
if [[ "$SMOKE" == "--smoke" ]]; then
  echo "[20B-G] tokenizer v0.3 SMOKE (not freeze)"
  python scripts/train_tokenizer.py \
    --config 20B-G/configs/tokenizer_latex_v0.3.yaml \
    --runtime configs/runtime_runpod_rtx_pro_6000.yaml \
    --smoke
else
  echo "[20B-G] tokenizer v0.3 FULL train → 20B-G/tokenizer/latex-v0.3/"
  python scripts/train_tokenizer.py \
    --config 20B-G/configs/tokenizer_latex_v0.3.yaml \
    --runtime configs/runtime_runpod_rtx_pro_6000.yaml
fi
echo "[20B-G] freeze check: ls 20B-G/tokenizer/latex-v0.3/"
ls -la 20B-G/tokenizer/latex-v0.3/ || true
