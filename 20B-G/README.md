# 20B-G — LÆTEX Genesis (baby iterations)

Isolated line for **agent + coding** short iterations. Does **not** pollute root Stage0a / `tokenizer/latex-v0.2` / `checkpoints/nullxes-latex-20b`.

| Piece | Path |
|-------|------|
| Tokenizer | `tokenizer/latex-v0.3/` (vocab **131072**) |
| Baby model | ~2B dense NHAT → `checkpoints/latex-2b-genesis/` |
| Train | **5 × 250 steps** → `checkpoints/latex-2b-iter{k}/` |
| Mix | code **0.60** / chat+agent **0.35** / NULLXES canon **0.05** |
| Style | Magistr × Grok spice ≤2%; no Digital Employees coaching |

Root Hub **20B Genesis** stays trunk for later continue; this folder only links in docs.

## Pod order

```bash
cd /workspace/NULLXES-L-TEX-AI-480B-A35B-Instruct
bash 20B-G/scripts/download_corpus.sh
bash 20B-G/scripts/train_tokenizer_v03.sh   # or smoke first
bash 20B-G/scripts/build_agent_seed.sh
bash 20B-G/scripts/init_2b.sh
bash 20B-G/scripts/run_iter_train.sh        # 5×250
bash 20B-G/scripts/run_sft.sh               # after iter5
```

Wrappers call root `scripts/*.py` with paths under `20B-G/`.

See `docs/16_LATEX_BABY_ITER.md`.
