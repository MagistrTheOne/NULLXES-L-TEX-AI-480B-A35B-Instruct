#!/usr/bin/env bash
# NULLXES-LÆTEX — bootstrap on RunPod 1× RTX PRO 6000
# Image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 0. Torch / CUDA sanity (image torch preferred) ==="
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available"
print("device", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print(f"vram_gb≈{props.total_memory/1024**3:.1f}")
# Expect 2.8.x + cu12.8 from image
v = torch.__version__
assert v.startswith("2.8"), f"expected torch 2.8.x from image, got {v}"
print("OK")
PY

echo "=== 1. Python deps (do NOT reinstall torch from PyPI) ==="
pip install -r requirements-stage1.txt

echo "=== 2. Identity corpus + validate ==="
python scripts/build_identity_corpus.py
python scripts/validate_corpus.py --manifest datasets/manifests/pretrain_stage0.json

echo "=== 3. CPU Weight Genesis (100M) ==="
python scripts/init_model.py \
  --config configs/stage0a_100m_rtx_pro_6000.yaml \
  --dtype bfloat16 \
  --smoke-device cpu

echo "=== 4. GPU Stage0a train (identity + RU/EN) ==="
python scripts/train_stage0a.py \
  --config configs/stage0a_100m_rtx_pro_6000.yaml \
  --device cuda

echo "=== DONE ==="
echo "Checkpoint: checkpoints/nullxes-latex-100m-stage0a-rtxpro6000"
ls -lah checkpoints/nullxes-latex-100m-stage0a-rtxpro6000 | head -30
