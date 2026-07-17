# NULLXES-LÆTEX — RunPod Deployment Plan

**Phase 1 cloud:** RunPod only · later replicate to other clouds / on-prem.

## Immediate (this week)

| Item | Spec |
|------|------|
| Pod | **1× H200 SXM** Secure Cloud |
| Storage | Network Volume **2–5 TB** |
| Image | PyTorch 2.x + CUDA 12.x + Transformer Engine |
| Work | Tokenizer corpus sample, Stage0 code, `param_count` sanity |

## Stage map on RunPod

| Stage | Product | Cluster |
|------:|---------|---------|
| 0 | 1.6B | 1–8× H200 pod / Instant Cluster |
| 1 | 7B | 8–32× H200 Instant Cluster |
| 2 | A35B | **32–64× B300** (or 64–128× H200) |
| 3 | 480B-A35B | **256–512× B300** reserved Instant Clusters |

## Pricing check (verify in console)

- H200: ~$3.59/hr Community · ~$4.39/hr Secure  
- B300: ~$6.94/hr Community · ~$7.39/hr Secure  

Prefer **Secure Cloud** for enterprise data / checkpoints.

## Parallelism cheat-sheet

| Stage | Suggested |
|------:|-----------|
| 0 | DP only |
| 1 | TP=2, DP |
| 2 | TP=4, PP=4, DP |
| 3 | TP=4, PP=8, EP=16, CP=2, DP=auto · FP8 |

## Artifacts to keep on Network Volume

- tokenizer model + vocab  
- data shards + mix manifests  
- checkpoints (3 rolling + eval milestones)  
- eval reports (NX suite)  
- IEL identity table  

## Later clouds

Export same container + Megatron dist checkpoints; no architecture fork. On-prem = same serving image (vLLM/SGLang) + IEL + memory services.
