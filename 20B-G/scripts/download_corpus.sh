#!/usr/bin/env bash
# Download agent+code corpus into 20B-G/datasets/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "[20B-G] download → configs/datasets_agent_code_v03.yaml"
# Seed first so local_identity merge has a manifest
python 20B-G/scripts/build_agent_seed.py
python scripts/download_local_corpus.py --config 20B-G/configs/datasets_agent_code_v03.yaml
echo "[20B-G] next: bash 20B-G/scripts/train_tokenizer_v03.sh"
