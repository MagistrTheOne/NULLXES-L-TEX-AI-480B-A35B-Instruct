# Next week — 20B Genesis Hub + ZeRO-3 100M train

## Tonight / now — publish Genesis (honest card)

```bash
cd /workspace/NULLXES-L-TEX-AI-480B-A35B-Instruct
git pull

python scripts/package_hf_20b_genesis.py \
  --checkpoint checkpoints/nullxes-latex-20b \
  --repo-id MagistrTheOne/NULLXES-L-TEX-20B-Genesis-v0.1

hf repo create MagistrTheOne/NULLXES-L-TEX-20B-Genesis-v0.1 --repo-type model
# ~35GB upload — use tmux/screen
hf upload MagistrTheOne/NULLXES-L-TEX-20B-Genesis-v0.1 \
  checkpoints/nullxes-latex-20b . \
  --commit-message "NULLXES-LÆTEX-20B-Genesis-v0.1 (muP scaffold; train Aug 2026)"
```

Card already states: nonsense until train; next rework **August 2026**.

---

## Next week — first train (~100M tok)

Prereqs on pod:

```bash
cd /workspace/NULLXES-L-TEX-AI-480B-A35B-Instruct
git pull
pip install -U deepspeed
export HF_HUB_ENABLE_HF_TRANSFER=1

# corpus still present?
ls datasets/manifests/corpus_gate_a_proxy.json
ls checkpoints/nullxes-latex-20b/init_report.json
ls tokenizer/latex-v0.2/tokenizer.model
```

Train (tmux recommended):

```bash
tmux new -s latex20b
cd /workspace/NULLXES-L-TEX-AI-480B-A35B-Instruct

deepspeed --num_gpus=1 scripts/train_stage2_20b.py \
  --config configs/stage2_20b_rtx_pro_6000_100m.yaml

# after DONE:
python scripts/qa_stage0a.py \
  --checkpoint checkpoints/nullxes-latex-20b-train-100m \
  --device cuda || true
cat checkpoints/nullxes-latex-20b-train-100m/train_report.json
```

Expect: loss drops from ~12 → lower; identity may start sticking (low mix).  
Not production intelligence — mid-eval brick for Aug 2026 refresh.

---

## Do not

- Blind 50B on proxy corpus  
- Soft-identity 90%  
- Replace 100M Stage0a tokenizer with v0.2  
- Schedule 480B
