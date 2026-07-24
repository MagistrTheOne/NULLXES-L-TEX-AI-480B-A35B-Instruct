# NULLXES-LÆTEX — active configs (2026)

## Model family (keep these)

| Config | Role | Status |
|--------|------|--------|
| [`nullxes_latex_20b_v1.yaml`](nullxes_latex_20b_v1.yaml) | **20B dense** V1 — current train target | active |
| [`nullxes_latex_200b_moe.yaml`](nullxes_latex_200b_moe.yaml) | **200B-A28B MoE** midscale | reference |
| [`nullxes_latex_480b_a35b.yaml`](nullxes_latex_480b_a35b.yaml) | **480B-A35B MoE** flagship | reference |
| [`nullxes_latex_a35b.yaml`](nullxes_latex_a35b.yaml) | A35B dense ancestor (MoE expand parent) | reference |

## V1 pipeline

| Config | Role |
|--------|------|
| [`datasets_latex_v1.yaml`](datasets_latex_v1.yaml) | Corpus download + filter/dedup |
| [`tokenizer_latex_v1.yaml`](tokenizer_latex_v1.yaml) | Tokenizer train / eval (vocab 131072) |
| [`stage3_20b_iter.yaml`](stage3_20b_iter.yaml) | Staged foundation bootstrapping |
| [`sft_20b_v1.yaml`](sft_20b_v1.yaml) | SFT after stages |
| [`runtime.yaml`](runtime.yaml) | Storage / precision defaults |
| [`deepspeed/zero2_20b_h200.json`](deepspeed/zero2_20b_h200.json) | 4× H200 train |
| [`deepspeed/zero3_cpu_offload_20b.json`](deepspeed/zero3_cpu_offload_20b.json) | 2× H100 / tight VRAM |

## Scale path

```
20B dense V1  →  A35B dense  →  200B MoE  →  480B-A35B MoE
   (L=24)          (L=48)        (L=48 MoE)     (L=64 MoE)
   d=8192          d=8192        d=8192         d=8192
```

MoE is not scheduled until dense path ≥50B / V1 gates pass. Tokenizer for all: `tokenizer/latex-v1`.

## Hardware

Primary: **H200** (proxy / mid) and **B300** (30B+ MoE, FP8). See skill llm-from-scratch-2026.
