# LÆTEX Baby Iterations (20B-G)

Date: 2026-07-24  
Line: **20B-G** (isolated folder)  
Contact: @MagistrTheOne · ceo@nullxes.com

## Goal

Prove agent+code mix on a **~2B** dense NHAT before continuing the 20B Hub genesis trunk.

## Fixed decisions

| Item | Choice |
|------|--------|
| Isolation | All artifacts under `20B-G/` |
| Model | ~2B dense NHAT (`configs/nullxes_latex_2b.yaml`) |
| Tokenizer | v0.3 vocab **131072** → `tokenizer/latex-v0.3/` |
| Shape | **5 × 250 steps** + mid-eval (not one blind ~1200) |
| Mix | code **0.60** / chat+agent **0.35** / canon **0.05** |
| Style | Magistr × Grok ≤2%; **no** Digital Employees |
| 20B Genesis | Trunk only; continue later via docs links |

## Token math

- `seq_len=1024`, `micro=8`, `accum=4` → **32768 tok/step**
- 250 steps ≈ **8.2M** tokens / iter
- 5 iters ≈ **41M** tokens

## Protocol

1. `bash 20B-G/scripts/download_corpus.sh` (builds agent seed, then HF download)
2. `bash 20B-G/scripts/train_tokenizer_v03.sh`
3. `bash 20B-G/scripts/init_2b.sh`
4. `bash 20B-G/scripts/run_iter_train.sh` → `checkpoints/latex-2b-iter{1..5}/`
5. `bash 20B-G/scripts/run_sft.sh` → SFT; DPO stub only
6. Hub from `20B-G` artifacts (`scripts/hub_upload_notes.sh`)

## Antipatterns

- Putting baby ckpt / v0.3 into root `checkpoints/` or `tokenizer/latex-v0.2`
- Soft-identity ~90%
- Training full 20B before mix proof on 2B

## Hub mapping

| Artifact | Suggested Hub id |
|----------|------------------|
| Tokenizer v0.3 | `MagistrTheOne/NULLXES-L-TEX-Tokenizer-v0.3` |
| Baby iter5 | `MagistrTheOne/NULLXES-L-TEX-2B-Baby-v0.3` |
| 20B Genesis (unchanged) | `MagistrTheOne/NULLXES-L-TEX-20B-Genesis-v0.1` |

## Pointers

- Root history: `docs/MODEL_HISTORY.md` (entry **20B-G baby**)
- Folder README: `20B-G/README.md`
