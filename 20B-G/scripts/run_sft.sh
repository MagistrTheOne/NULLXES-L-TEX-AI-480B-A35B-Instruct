#!/usr/bin/env bash
# Short SFT 250 steps after iter5. DPO remains stub.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
ITER5="20B-G/checkpoints/latex-2b-iter5"
if [[ ! -d "$ITER5" ]]; then
  echo "[fail] need $ITER5 — finish run_iter_train.sh first"
  exit 2
fi
python 20B-G/scripts/build_agent_seed.py
python scripts/train_stage0a.py \
  --config 20B-G/configs/stage0_2b_sft.yaml \
  --device cuda \
  --max-steps 250 \
  --resume "$ITER5"
echo "[ok] SFT → 20B-G/checkpoints/latex-2b-sft/"
echo "[note] DPO stub only: 20B-G/configs/stage0_2b_dpo_stub.yaml (enabled: false)"
