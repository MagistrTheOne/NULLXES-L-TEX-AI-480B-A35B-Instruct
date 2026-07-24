#!/usr/bin/env bash
# Weight Genesis ~2B → 20B-G/checkpoints/latex-2b-genesis/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
TOK="20B-G/tokenizer/latex-v0.3/tokenizer.model"
if [[ ! -f "$TOK" ]]; then
  echo "[fail] missing $TOK — run train_tokenizer_v03.sh first"
  exit 2
fi
echo "[20B-G] init 2B genesis"
python scripts/init_model.py \
  --config 20B-G/configs/nullxes_latex_2b.yaml \
  --dtype bfloat16 \
  --smoke-device cpu
echo "[20B-G] genesis at 20B-G/checkpoints/latex-2b-genesis/"
