# SFT / DPO under 20B-G

| Step | Config | Script |
|------|--------|--------|
| Agent seed JSONL | manifests in `datasets/` | `scripts/build_agent_seed.py` |
| SFT 250 steps | `configs/stage0_2b_sft.yaml` | `scripts/run_sft.sh` |
| DPO | `configs/stage0_2b_dpo_stub.yaml` | **stub** (`enabled: false`) |

SFT resumes from `checkpoints/latex-2b-iter5/`.
DPO pairs stub: `datasets/sft/dpo_pairs_stub.jsonl`.
