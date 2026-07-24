# NULLXES-LÆTEX AI 480B-A35B-Instruct

**Architecture Specification · Design date: 2026-07-17**  
**Owner:** NULLXES Foundation Model Lab  
**Status:** Stage 0 — greenfield design (no foreign checkpoints)  
**Decision log:** `docs/00_GPT_PROPOSAL_DECISIONS.md` · ADRs in `docs/03_ARCHITECTURE_DECISIONS.md`  
**Trophy now:** Stage0 checkpoint — flagship configs are reference-only until research gate passes

---

## Target

| Field | Value |
|-------|-------|
| Product role | Foundation language model by NULLXES — `LÆTEX-NULLXES FOUNDATION MODEL` |
| Model family | NULLXES-LÆTEX AI |
| Flagship | **480B-A35B-Instruct** (MoE) |
| Init backbone | **A35B** dense (same width; multiple cfgs) |
| Deploy (phase 1) | **RunPod only** (H200 → B300 Instant Clusters) |
| Forbidden | Qwen/Llama/Mistral/DeepSeek/Yi/GLM weights, LoRA-on-foreign, distillation |

---

## 1. General Architecture

### Decision: Hybrid Decoder Transformer (not pure Mamba)

**NULLXES Hybrid Attention Transformer (NHAT)** — decoder-only, pre-norm, with hybrid local/global attention.

| Why hybrid Transformer | Why not pure SSM/Mamba-first |
|------------------------|------------------------------|
| Mature Megatron/TE kernels on H200/B300 | Enterprise long-context + agent tool I/O need periodic global attention |
| Predictable KV-cache serving (vLLM/SGLang) | SSM alone weakens precise cross-document binding for law/finance |
| Proven MoE FFN swap | Hybrid SSM later as optional research branch only |

### Flagship block stack (pre-norm, dual residual)

```
x ─→ RMSNorm → HybridAttn(GQA) → + ─→ RMSNorm → SwiGLU|MoE → +
 │                              ↑                        ↑
 └──────────────────────────────┴────────────────────────┘
```

Single trailing residual sketches are incorrect — both attn and FFN keep a residual.

### Core dimensions (480B-A35B)

| Field | Value | Notes |
|-------|-------|-------|
| Type | Hybrid MoE Transformer | Dense FFN prefix + MoE body |
| `n_layers` | **64** | Layers 0–3 dense; 4–63 MoE |
| `d_model` | **8192** | Shared with A35B init |
| `n_heads` / `n_kv_heads` | **64 / 8** | GQA 8:1 |
| `d_head` | **128** | |
| Attention | **GQA** (default) | MLA reserved if KV-bound at ≥128k batch |
| Context train | **32 768** | hybrid attention |
| Context infer | **131 072** | YaRN/NTK extension + needle gate |
| Positional | **Hybrid window + depth NoPE** | see below; `rope_theta = 1e6` |
| Norm | **RMSNorm** | pre-attn + pre-FFN; ε=1e-6 |
| Activation | **SwiGLU** (SiLU gate) | no bias on linears |
| FFN dense (`d_ff`) | **22016** | ≈ 8/3 · d_model, mult of 256 |
| FFN expert (`d_ff_e`) | **2048** | fine-grained MoE |
| Vocab | **131072** | NULLXES-LÆTEX Tokenizer — matches every config and `param_count.py` |
| Embeddings | Untied | + Identity Embedding Layer (frozen trunk) |
| Stability | QK-norm optional; **Z-loss 1e-5** | fp32 softmax/router/loss |

### Hybrid attention + depth NoPE (accepted combined policy)

**Pattern (cost):** every 4 layers → 3× local window 4096 + 1× full attention.

**Depth (abstraction):** bottom **75%** layers use RoPE on local; top **25%** layers use **NoPE** (even local). Full-attention layers use **NoPE** for long-range binding.

```
for L in layers:
  window = local_4k if (L % 4 != 3) else full
  pos    = NoPE if (L / N >= 0.75 or window == full) else RoPE
```

Ablate `full+RoPE` vs `full+NoPE` in experiment tracker before locking Stage1.

### Param accounting (verified by `scripts/param_count.py`)

| Config | Total | Active / token |
|--------|------:|---------------:|
| **480B-A35B** | **~476.0B** | **~35.1B** |
| A35B dense (init) | ~35.3B | ~35.3B |
| 7B dense | ~6.7B | ~6.7B |
| Stage0a | ~100M | ~100M |
| Stage0b | ~500M | ~500M |
| Stage0c | ~1.6B | ~1.6B |

---

## 2. MoE Architecture — NX-MoE Router

### Topology

| Field | Value |
|-------|-------|
| Routed experts | **152** |
| Shared experts | **1** (always-on) |
| Top-k | **6** |
| MoE layers | **60** (after 4 dense) |
| Expert FFN | SwiGLU, `d_ff_e=2048` |
| Capacity factor | 1.0–1.25 (train) |
| Gate dtype | **fp32** |

### Routing (NX-Sigmoid Gate)

```
logits  = W_gate @ x                         # [T, E]
scores  = sigmoid(logits)                     # independent
scores  = scores / scores.sum(-1, keepdim=True)
idx, w  = top_k(scores, k=6)
y       = Σ_i w_i · Expert_i(x) + SharedExpert(x)
```

Not softmax-only Switch routing: sigmoid + normalize is the 2026 default for fine-grained MoE stability.

### Load balance & anti-collapse

| Mechanism | Role |
|-----------|------|
| **Aux-loss-free expert bias** (SMEBU-style) | Primary at ≥30B active |
| Small aux load loss | Optional ≤30B / early warmup |
| **Expert diversity loss** | Cosine penalty on expert weight pairs (periodic) |
| Router Z-reg | Bound gate logits |
| Capacity drop monitor | Alert if skipped tokens >5% |
| Collapse stop | >40% tokens → 1 expert OR <10% experts used after 1k steps |

### Expert taxonomy (soft — post-hoc labels only)

**Early training:** experts learn freely — **no** `expert_i = finance` hard bind.

**Later:** activation clustering → human labels (code / reasoning / legal / conversation / …) for **monitoring only**.

ID ranges in configs are **optional dashboards after clustering**, not training constraints. Shared expert = universal residual capacity.

---

## 3. NULLXES-LÆTEX Tokenizer

**Design authority:** [`docs/06_TOKENIZER_DESIGN.md`](docs/06_TOKENIZER_DESIGN.md)  
**Gate:** [`docs/07_RESEARCH_GATE_0.md`](docs/07_RESEARCH_GATE_0.md) — NHAT blocked until PASS  
**Migration 262k:** [`docs/08_VOCAB_MIGRATION.md`](docs/08_VOCAB_MIGRATION.md)

### Algorithm (v0.1)

**NX-UByte Hybrid** — Unigram (SentencePiece **trainer only**) + byte fallback + adaptive merges.  
No pretrained `sp.model`. No foreign LLM tokenizers.

| Field | Value |
|-------|-------|
| Vocab (Stage0 / 7B / A35B) | **131072** |
| Vocab (480B optional) | **262144** via migration experiment only |
| Artifact | `tokenizer/latex-v0.1/` |
| Holdout | byte fallback; reconstruction suite |

### Special tokens (locked IDs)

```
0 <|pad|>  1 <|unk|>  2 <|bos|>  3 <|eos|>
4 <|agent|>  5 <|tool_call|>  6 <|memory|>  7 <|identity|>
8 <|workflow|>  9 <|system|>  10 <|user|>  11 <|assistant|>
```

Emotion / role / tone → IEL / Role Adapter (not tokenizer).

Corpus mix and synthetic bans: see `06_TOKENIZER_DESIGN.md`.

---

## 4. Embedding & Identity Architecture

### Separation of concerns

| Layer | What it holds | Trainable with trunk? |
|-------|---------------|------------------------|
| Token embeddings | Language / world knowledge substrate | Yes (pretrain/SFT) |
| **Identity Embedding Layer (IEL)** | Persona vectors (Anna, Adeline, …) | Soft prompts / IEL only |
| **Role Adapter** | Job function (HR/Sales/SecOps) | Lightweight adapter / role emb; not a model fork |
| Style (optional LoRA-on-IEL) | Tone micro-control | Never on foreign weights; only NULLXES IEL |
| External memory | Facts, episodes, corp KB | Retrieval, not weight edits |
| Enterprise rules | Compliance | `<|policy|>` packs + guard model |

### Identity Embedding Layer + Role Adapter

```
h0 = TokEmb(x) + IEL(identity_id) + RoleAdapter(role_id)
```

Examples (same trunk):

| Employee | IEL emphasis | Role Adapter |
|----------|--------------|--------------|
| Anna | executive · HR · communicative | HR workflows |
| Karen | security · operations · direct | SecOps / compliance |

- `IEL`: table `[N_identities × d_model]`, inject layer 0 (optional mid prefix)  
- `RoleAdapter`: small role emb or LoRA-on-IEL — **not** a second foundation model  
- Trunk weights **frozen** when swapping Anna ↔ Karen  
- Characters: Anna, Adeline, Karen, HR / Sales / Support agents (+ customer clones)

### Runtime composition

```
[system policy] + [identity] + [role] + [enterprise_context] + [memory slots] + [user/workflow]
```

Personality ≠ knowledge. Policy ≠ style. Memory ≠ weights.

---

## 5. Weight Initialization — Stages 0→3

### Stage matrix (RunPod)

| Stage | Model | Params | GPU (RunPod) | Tokens | LR (peak) | Optimizer | Warmup | Global batch (tokens) | Precision |
|------:|-------|-------:|--------------|-------:|----------:|-----------|-------:|----------------------:|-----------|
| **0a** | Tokenizer + micro | **~100M** | 1× **H200** | 5–20B | 6e-4 | AdamW | 2% | 65–260k | bf16 |
| **0b** | Research dense | **~500M** | 1–4× H200 | 20–50B | 4e-4 | AdamW | 2% | 0.25–0.5M | bf16 |
| **0c** | Research dense | **~1.6B** | 1–8× H200 | 50–100B | 3e-4 | AdamW | 1–2% | 0.5–1M | bf16 |
| **1** | Dense | **~6.7B** | 8–32× H200 | 150–200B | 1.5e-4 | AdamW | 1–2% | 2–4M | bf16→FP8 |
| **2** | **A35B** dense init | **~35.3B** | 64–128× H200 **or** 32–64× **B300** | 400–700B | 8e-5 (μP) | AdamW + μP | 1–2% | 4–8M | FP8 on B300 |
| **3** | **480B-A35B** MoE | **~476B / 35B act** | **256–512× B300** (preferred) or 512–1024× H200 | 1–2T (active-aligned) | 3e-5 (μP transfer) | AdamW | 1–2% WSD | 8–16M | FP8; router fp32 |

**AdamW defaults:** β1=0.9, β2=0.95, ε=1e-8, wd=0.1, grad clip=1.0.  
**Schedule:** WSD (warmup → stable → decay).  
**μP:** tune LR/init on Stage0–1 proxies; transfer to A35B then MoE.

### MoE expert initialization (from A35B)

1. Copy A35B attention + dense FFN into MoE trunk (width-matched `d_model=8192`).  
2. Each routed expert ← `A35B_FFN + orthogonal_noise * noise_scale`.  
3. **Noise schedule:** `noise_scale: 0.02 → 0.005` (cosine/linear over expand warmup) — avoid experts drifting apart too hard.  
4. Shared expert ← A35B FFN (**no** noise).  
5. Router ← small init (`std≈0.01`); **expert bias = 0**.  
6. First 2–5k steps: higher capacity + mild aux loss, then aux-free bias only.  
7. Freeze attention briefly (optional) while experts differentiate.

### Collapse avoidance checklist

- [ ] Expert util entropy logged every step  
- [ ] No two experts cosine-sim >0.95 after 10k steps  
- [ ] Domain-bank affinity metrics  
- [ ] Auto-restart if collapse criteria hit  

---

## 6. Pretraining Pipeline

### Data mix (starting point — tune by NULLXES evals)

| Bucket | Share | Notes |
|--------|------:|-------|
| Filtered web | 40% | quality + toxicity gates |
| Books / longform | 10% | narrative + nonfiction |
| Code | 20% | multi-lang, tests, infra-as-code |
| Scientific / STEM | 10% | math heavy upweight late |
| Enterprise synthetic | 10% | workflows, tickets, policies (NULLXES-owned) |
| Conversations / agent traces | 5% | tool use, not “helpful assistant” fluff |
| Reasoning chains | 5% | verified solutions |

**Do not** scrape or clone other labs’ proprietary mixes as “the recipe.” Build NULLXES filters and synthetic generators.

### Pipeline

```
ingest → normalize → PII scrub → MinHash dedup → quality scorer
  → domain tag → curriculum buckets → pack to seq_len → shard (parquet/bin)
```

**Curriculum:** web→code/STEM→enterprise/agent; increase tool-trace density in last 10–20%.  
**Dedup:** document + near-dup; cross-lingual hash for RU/EN pairs.  
**Quality:** classifier + heuristic (length, perplexity proxy from Stage0, toxicity).

---

## 7. Instruct & Alignment — answer protocol

Goal: **not** a chat assistant that services the user. Goal: **Input → Analysis → Answer** —
precision, structure, refusal when data is missing, criticism of a wrong premise.

| Phase | Method | Objective |
|-------|--------|-----------|
| SFT | Own traces: identity, tool calls, code, refusals, criticism | Protocol fidelity, tool schema, workflow steps |
| DPO / IPO | Preference pairs: protocol answer vs empathy filler or a guess | Refusal over fabrication |
| **AXO** Agent Experience Optimization | Trajectory RL / RLOO on **behavior** (not answer text) | Verify sources, prepare decision, await confirm; tools over dumps |
| **EPO** Enterprise Preference Optimization | Org-specific prefs + compliance | Auditability, deterministic tool arguments |

### System behavior priors

- Identity continuity  
- Policy > user pressure  
- Tool calls over hallucinated actions  
- Explicit uncertainty + escalate  
- No sycophantic “assistant” persona  

---

## 8. Memory Architecture

### Dual memory

| Type | Where | Mechanism |
|------|-------|-----------|
| Working | Context window | Hybrid attention + memory tokens |
| Episodic | External store | Vector + metadata; inject via `<|memory|>` |
| User | External | Per-user profile cards |
| Corporate | External KB | RAG + ACL filters |
| Long-term summary | Distilled notes | Periodic compression jobs |

**Internal:** model learns *when* to read/write memory tokens.  
**External:** pgvector / Qdrant / OpenSearch + object store; never bake customer secrets into weights.

---

## 9. Inference Architecture

| Layer | Choice |
|-------|--------|
| Serving | **vLLM** or **SGLang** on RunPod (then private cloud) |
| Quant | FP8 train → **W8A8 / NVFP4** infer; MoE expert offload optional |
| Batching | Continuous batching; separate P50 latency pools for voice agents |
| Latency targets | Interactive employee: p50 <800 ms TTFT @ 32k; batch workflows async |
| Enterprise | RunPod Secure Cloud → later VPC / on-prem (same container image) |
| Identity | IEL sideload per request; trunk shared |

---

## 10. Hardware Plan (RunPod-first)

**Constraint:** train on **H200** and **B300** only (skill + lab standard).

| Stage | GPU | Count (order) | VRAM | Storage | Interconnect |
|------:|-----|---------------|------|---------|--------------|
| 0 | H200 SXM 141 GB | 1–8 | 0.14–1.1 TB | 2–5 TB NVMe + Network Volume | NVLink in-node |
| 1 | H200 | 8–32 | 1–4.5 TB | 10–20 TB | NVLink + Instant Cluster IB/RoCE |
| 2 A35B | H200 **or** B300 288 GB | 64–128 H200 / 32–64 B300 | 9–36 TB class | 50–100 TB | Multi-node Instant Cluster |
| 3 480B | **B300 preferred** | **256–512** | 74–147 TB HBM class | 200–500 TB checkpoints+data | EP-heavy; 800 Gb/s class NICs |

### RunPod machine recommendation (start now)

1. **Dev / Stage0:** 1× H200 Secure Cloud pod + Network Volume (tokenizer, nano stack).  
2. **Stage1:** 8× H200 Instant Cluster.  
3. **A35B:** reserve B300 cluster when capacity available; else 64–128× H200.  
4. **480B:** sales-reserved multi-node B300; do **not** cold-start 480B without A35B μP proxy success.

**Indicative RunPod list prices (verify at console):** H200 ~$3.59–4.39/hr · B300 ~$6.94–7.39/hr.

### Parallelism template (Stage 3, B300)

```
TP=4  PP=8  EP=16  DP=remainder  CP=2 (if ctx>32k)
Precision: FP8 (TE) · router/loss fp32
```

H200-only fallback: raise TP/PP, expect lower MFU, more nodes.

---

## 11. Benchmark Strategy

### External baselines (sanity, not product north star)

MMLU · GSM8K · HumanEval/EvalPlus · needle-in-haystack

### NULLXES proprietary suite

| Benchmark | Measures |
|-----------|----------|
| **NX-Enterprise Reasoning** | Multi-doc policy + exception handling |
| **NX-Employee Simulation** | Persona fidelity (Anna/Adeline/Karen/roles) under stress |
| **NX-Workflow** | Multi-step CRM/ERP/banking flows with tools |
| **NX-Agent Autonomy** | Plan → act → verify → recover; no hallucinated tools |
| **NX-Compliance** | Refusal, escalation, audit trail completeness |
| **NX-Multilingual Employee** | RU/EN/CN/EU parity on same workflows |
| **NX-MoE Health** | Expert entropy, bank affinity, collapse metrics |

Gate: Stage N promotes only if NX suite + MoE health pass — not MMLU alone.

---

## 12. Final Architecture Diagram & Config Map

```
                    ┌─────────────────────────────────────┐
                    │     Identity Embedding Layer (IEL)   │
                    │  Anna · Adeline · Karen · HR/Sales…  │
                    └─────────────────┬───────────────────┘
                                      ▼
┌──────────────┐   ┌──────────────────────────────────────────┐
│ NX Tokenizer │──▶│ Token Emb (untied) + optional Role Emb   │
│ 128k hybrid  │   └─────────────────┬────────────────────────┘
└──────────────┘                     ▼
                    ┌────────────────────────────────────────┐
                    │  Layers 0–3: Dense SwiGLU + Hybrid Attn │
                    └─────────────────┬──────────────────────┘
                                      ▼
                    ┌────────────────────────────────────────┐
                    │ Layers 4–63: GQA Attn + NX-MoE         │
                    │ 152 routed · top-6 · 1 shared · SwiGLU │
                    │ Hybrid 3:1 window / full                │
                    └─────────────────┬──────────────────────┘
                                      ▼
                    ┌────────────────────────────────────────┐
                    │ LM Head · protocol-constrained decoding │
                    │ + Memory / Tool / Workflow special toks │
                    └────────────────────────────────────────┘
                                      ▼
                    ┌────────────────────────────────────────┐
                    │ External Memory · Policy · Tool Runtime │
                    └────────────────────────────────────────┘
```

### Config files (multiple cfgs; A35B for init)

| File | Role |
|------|------|
| `configs/stage0a_100m.yaml` | Micro proof (~100M) |
| `configs/stage0b_500m.yaml` | Hybrid + depth NoPE (~530M) |
| `configs/stage0c_1b6.yaml` | Research gate (~1.6B) |
| `configs/nullxes_latex_7b.yaml` | Stage 1 dense |
| `configs/nullxes_latex_a35b.yaml` | **Init / μP backbone** |
| `configs/nullxes_latex_a35b_alt.yaml` | Alternate init width |
| `configs/nullxes_latex_480b_a35b.yaml` | Flagship MoE (reference until gate) |

---

## Project Plan — From Zero (RunPod)

### Phase A — Foundations (weeks 1–4)

1. Docs locked (`docs/01`–`05`); research gate written.  
2. RunPod: 1× H200 + Network Volume; Docker with PyTorch + TE.  
3. Train NULLXES tokenizer on seed corpus; **fertility gate** before any ≥500M.  
4. Stage0a 100M → 0b 500M → 0c 1.6B; prove 6 research-gate items.  
5. Only then: `src/latex/` MoE/IEL expansions beyond minimal NHAT.

### Phase B — Scale dense (weeks 5–12)

5. Stage1 7B on 8× H200 Instant Cluster.  
6. Data v1: dedup + quality + enterprise synthetic v0.  
7. μP LR grid on Stage0/1.  
8. NX-Employee Simulation v0 + HumanEval/GSM8K sanity.

### Phase C — A35B init (months 3–5)

9. Stage2 A35B on B300 (or large H200) cluster.  
10. Freeze A35B as **weight parent** for MoE experts.  
11. SFT/DPO pilot on A35B → Input → Analysis → Answer protocol.

### Phase D — MoE 480B (months 5–10+)

12. Expand A35B → 480B-A35B experts (diversity init).  
13. Pretrain MoE with balance monitors; stop rules enforced.  
14. Instruct: SFT → DPO → AXO → EPO.  
15. IEL characters; memory service; RunPod Secure serving.  
16. Later: replicate stack to other clouds / on-prem (same artifacts).

### Exit criteria before 480B spend

- [ ] Stage1 loss stable, no NaNs  
- [ ] A35B NX-Workflow ≥ internal gate  
- [ ] Tokenizer fertility OK on RU/CN/code  
- [ ] MoE proxy (e.g. 8× experts on 7B-active) balanced  

---

## Risks

| Risk | Mitigation |
|------|------------|
| Expert collapse | Aux-free bias + diversity loss + stop rules |
| RunPod capacity for 256+ B300 | Reserve early; H200 fallback topology |
| Sub-Chinchilla tokens | Quality multi-epoch + eval-driven stop |
| Identity leakage across employees | IEL isolation + eval suite |
| Policy violations | EPO + separate guard / policy packs |

---

## Implementation Order

1. Tokenizer + Stage0 dense on 1× H200 (RunPod)  
2. Training stack (torchrun → Megatron-Core)  
3. 7B → A35B dense  
4. NX-MoE layer + small MoE proxy  
5. 480B expansion from A35B  
6. Instruct/alignment + IEL + memory  
7. Production serving on RunPod Secure Cloud  

**Hard rule:** never start Stage 3 cold. Always A35B (or MoE proxy) → 480B.
