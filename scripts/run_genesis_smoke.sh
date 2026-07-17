#!/usr/bin/env bash
# NULLXES-LÆTEX — one-shot Weight Genesis + HF smoke on RunPod H200.
# Canonical checkpoint: checkpoints/nullxes-latex-7b
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
CKPT="checkpoints/nullxes-latex-7b"

echo "=== CUDA check ==="
python -c "import torch; assert torch.cuda.is_available(), 'CUDA down — reinstall cu124 torch'; print(torch.__version__, torch.cuda.get_device_name(0))"

echo "=== Weight Genesis (CPU save, CUDA smoke) ==="
python scripts/init_model.py \
  --config configs/nullxes_latex_7b.yaml \
  --dtype bfloat16 \
  --smoke-device cuda

echo "=== HF CausalLM smoke ==="
python scripts/smoke_hf_causal.py --checkpoint "$CKPT"

echo "=== DONE ==="
ls -lah "$CKPT" | head -40
test -f "$CKPT/init_report.json" && echo "init_report: $ROOT/$CKPT/init_report.json"
