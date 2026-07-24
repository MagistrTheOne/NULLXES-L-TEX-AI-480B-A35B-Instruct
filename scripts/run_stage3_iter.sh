#!/usr/bin/env bash
# LÆTEX V1 staged foundation bootstrapping.
#
#   bash scripts/run_stage3_iter.sh            # 4 stages x 250 steps
#   STAGES=8 STEPS=250 GPUS=4 bash scripts/run_stage3_iter.sh
#
# Each stage: train -> holdout eval -> QA. The run stops when holdout loss stops
# falling, because that is the point where further steps only buy memorization.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STAGES="${STAGES:-4}"
STEPS="${STEPS:-250}"
GPUS="${GPUS:-4}"
BASE_CFG="${BASE_CFG:-configs/stage3_20b_iter.yaml}"
GENESIS="${GENESIS:-checkpoints/nullxes-latex-20b-v1}"
HOLDOUT="${HOLDOUT:-datasets/manifests/corpus_latex_v1_holdout.json}"
TOKENIZER="${TOKENIZER:-tokenizer/latex-v1}"
# Holdout must improve by at least this much or the stage loop stops
MIN_DELTA="${MIN_DELTA:-0.01}"

if [[ ! -d "$GENESIS" ]]; then
  echo "[fail] missing genesis $GENESIS"
  echo "       python scripts/init_model.py --config configs/nullxes_latex_20b_v1.yaml --dtype bfloat16"
  exit 2
fi
if [[ ! -f "$HOLDOUT" ]]; then
  echo "[fail] missing holdout manifest $HOLDOUT — run scripts/build_corpus_v1.py"
  exit 2
fi

prev="$GENESIS"
prev_holdout=""
for k in $(seq 1 "$STAGES"); do
  out="checkpoints/latex-20b-v1-iter${k}"
  work_cfg="configs/_stage3_iter${k}_runtime.yaml"
  stage_name="LÆTEX V1 Stage $(printf '%02d' "$k")"
  echo "========== STAGE $k / $STAGES =========="
  echo "[resume] $prev -> [out] $out  steps=$STEPS gpus=$GPUS"

  python - "$BASE_CFG" "$work_cfg" "$out" "$prev" "$STEPS" "$GPUS" "$stage_name" <<'PY'
import sys
from pathlib import Path
import yaml

base, work, out, prev, steps, gpus, stage_name = sys.argv[1:8]
cfg = yaml.safe_load(Path(base).read_text(encoding="utf-8"))
t = cfg["training"]
cfg["checkpoint"]["output_dir"] = out
t["resume_from"] = prev
t["stage_name"] = stage_name
tokens_per_step = int(t["seq_len"]) * int(t["micro_batch_size"]) * int(t["grad_accum_steps"]) * int(gpus)
t["tokens_target"] = int(steps) * tokens_per_step
t["release_name"] = f"NULLXES-LÆTEX-20B-V1-Stage{stage_name.split()[-1]}"
Path(work).write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"[cfg] {work} tokens_target={t['tokens_target']} ({tokens_per_step}/step)")
PY

  deepspeed --num_gpus "$GPUS" scripts/train_stage2_20b.py --config "$work_cfg"

  echo "[eval] holdout for stage $k"
  python scripts/eval_holdout.py \
    --checkpoint "$out" \
    --manifest "$HOLDOUT" \
    --tokenizer "$TOKENIZER" \
    --device cuda

  echo "[qa] identity / protocol gates for stage $k"
  python scripts/qa_stage0a.py --checkpoint "$out" --device cuda || echo "[warn] QA gate failed at stage $k"

  holdout=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['holdout_loss'])" "$out/holdout_report.json")
  echo "[gate] stage $k holdout_loss=$holdout (previous: ${prev_holdout:-none})"
  if [[ -n "$prev_holdout" ]]; then
    stop=$(python -c "import sys; print(int(float(sys.argv[1]) > float(sys.argv[2]) - float(sys.argv[3])))" \
      "$holdout" "$prev_holdout" "$MIN_DELTA")
    if [[ "$stop" == "1" ]]; then
      echo "[stop] holdout did not improve by $MIN_DELTA — further steps buy memorization, not language"
      prev="$out"
      break
    fi
  fi
  prev_holdout="$holdout"
  prev="$out"
done

echo "[done] final checkpoint: $prev"
echo "next: record the stage card in docs/MODEL_HISTORY.md, then scripts/train_sft_v1.py"
