# NULLXES-LÆTEX — RunPod Deployment Plan

**Phase 1 cloud:** RunPod only · later replicate to other clouds / on-prem.

## Active pod (Stage0 proxy — now)

| Item | Spec |
|------|------|
| Pod | **1× RTX PRO 6000** (96 GB VRAM) |
| Host | 140 GB RAM · 16 vCPU · **~5750 GB** disk |
| Price (verify) | ~$1.99/hr |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Torch | **2.8.0 + CUDA 12.8** (image-bundled) |
| Runtime | `configs/runtime_runpod_rtx_pro_6000.yaml` |
| Train cfg | `configs/stage0a_100m_rtx_pro_6000.yaml` |
| Setup | `scripts/runpod_setup_rtx_pro_6000.sh` |

### Stack rules

1. **Do not** `pip install torch` from default PyPI (wrong CUDA).  
2. Prefer image torch. Restore only via `requirements-torch-cu128.txt` + cu128 index.  
3. **Init on CPU** (`init_model.py --smoke-device cpu`) → **train on CUDA**.  
4. Corpus = identity + RU/EN semantics — not FineWeb / not 5 TB junk.

```bash
# on pod, repo at /workspace/nullxes-latex (or clone)
bash scripts/runpod_setup_rtx_pro_6000.sh
```

Legacy local Win11 2080: `local_2080/` + `requirements-torch-cu124.txt`.

---

## Scale map (later)

| Stage | Product | Cluster |
|------:|---------|---------|
| 0 proxy | 50–100M | **1× RTX PRO 6000** (active) |
| 0 | 1.6B | 1–8× H200 |
| 1 | 7B | 8–32× H200 Instant Cluster |
| 2 | A35B | **32–64× B300** (or 64–128× H200) |
| 3 | 480B-A35B | **256–512× B300** |

## Pricing check (verify in console)

- RTX PRO 6000: ~$1.99/hr  
- H100 SXM: ~$2.99/hr  
- H200 SXM: ~$4.39/hr Secure  
- B300: ~$7.39/hr Secure  

## Parallelism cheat-sheet

| Stage | Suggested |
|------:|-----------|
| 0 proxy (this pod) | single GPU |
| 0 | DP only |
| 1 | TP=2, DP |
| 2 | TP=4, PP=4, DP |
| 3 | TP=4, PP=8, EP=16, CP=2, DP=auto · FP8 |

## Artifacts on disk (~5.7 TB)

- tokenizer model + vocab  
- clean corpus shards + manifests  
- checkpoints (rolling + eval)  
- identity shards  

## Later clouds

Export same container + Megatron dist checkpoints; no architecture fork.
