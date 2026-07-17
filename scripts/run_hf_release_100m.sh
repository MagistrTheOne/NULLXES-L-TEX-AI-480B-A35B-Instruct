#!/usr/bin/env bash
# QA → package → upload Stage0a 100M to Hugging Face
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="checkpoints/nullxes-latex-100m-stage0a-v0.1"
REPO_ID="${HF_REPO_ID:-MagistrTheOne/NULLXES-L-TEX-100M-Stage0a-v0}"

echo "=== QA ==="
python scripts/qa_stage0a.py --checkpoint "$CKPT" --device cuda

echo "=== Package for Hub ==="
python scripts/package_hf_release.py --checkpoint "$CKPT" --repo-id "$REPO_ID"

echo "=== Upload ==="
# Requires: huggingface-cli login  (use a NEW token — revoke any leaked ones)
huggingface-cli upload "$REPO_ID" "$CKPT" . \
  --commit-message "NULLXES-LÆTEX-100M-Stage0a-v0 identity bootstrap"

echo "=== LIVE ==="
echo "https://huggingface.co/$REPO_ID"
