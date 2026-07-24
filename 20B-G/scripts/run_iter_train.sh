#!/usr/bin/env bash
# 5 × 250-step iterations with mid-eval between rounds.
# Ckpts: 20B-G/checkpoints/latex-2b-iter{1..5}/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

ITERS="${ITERS:-5}"
STEPS="${STEPS:-250}"
BASE_CFG="20B-G/configs/stage0_2b_iter.yaml"
GENESIS="20B-G/checkpoints/latex-2b-genesis"
EVAL_PROMPTS="20B-G/evals/mid_eval_prompts.txt"

if [[ ! -d "$GENESIS" ]]; then
  echo "[fail] missing genesis $GENESIS — run init_2b.sh"
  exit 2
fi

mkdir -p 20B-G/evals 20B-G/checkpoints
if [[ ! -f "$EVAL_PROMPTS" ]]; then
  cat > "$EVAL_PROMPTS" <<'EOF'
Who are you?
What is NULLXES?
Write a Python function fib(n).
Use a tool to list files in /tmp.
Кто ты?
EOF
fi

prev="$GENESIS"
for k in $(seq 1 "$ITERS"); do
  out="20B-G/checkpoints/latex-2b-iter${k}"
  work_cfg="20B-G/configs/_stage0_2b_iter${k}_runtime.yaml"
  echo "========== ITER $k / $ITERS =========="
  echo "[resume] $prev → [out] $out  steps=$STEPS"

  python - <<PY
from pathlib import Path
import yaml
root = Path(".")
cfg = yaml.safe_load((root / "$BASE_CFG").read_text(encoding="utf-8"))
cfg["checkpoint"]["output_dir"] = "$out"
cfg["training"]["resume_from"] = "$prev"
cfg["training"]["tokens_target"] = $STEPS * 1024 * 8 * 4
(root / "$work_cfg").write_text(yaml.dump(cfg, sort_keys=False), encoding="utf-8")
print("[ok] wrote $work_cfg")
PY

  python scripts/train_stage0a.py \
    --config "$work_cfg" \
    --device cuda \
    --max-steps "$STEPS" \
    --resume "$prev"

  echo "[mid-eval] iter $k — loss/ckpt smoke"
  if [[ -f scripts/qa_stage0a.py ]]; then
    python scripts/qa_stage0a.py --checkpoint "$out" --device cuda || true
  fi
  echo "[mid-eval] prompts:"
  cat "$EVAL_PROMPTS"
  echo "[ok] iter $k ckpt → $out"
  prev="$out"
done

echo "[done] $ITERS iters. Final: $prev"
echo "next: bash 20B-G/scripts/run_sft.sh"
