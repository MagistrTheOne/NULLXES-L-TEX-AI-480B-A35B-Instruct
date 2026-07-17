# GPT Proposal Review — Decision Log

**Date:** 2026-07-17  
**Source:** GPT critique of NULLXES-LÆTEX v1  
**Rule:** Analyze → select → only then code Stage0

| # | Proposal | Verdict | Action |
|---|----------|---------|--------|
| 1 | NHAT block diagram | **MODIFY** | Keep **two** residuals (attn + FFN). GPT diagram collapsed them — incorrect for pre-norm Transformer. |
| 2 | GQA 64/8 | **KEEP** | Already locked. |
| 3 | RoPE 0–75% / NoPE 75–100% | **MODIFY → COMBINE** | Depth-NoPE **and** hybrid window pattern. Not replace. |
| 4 | Soft expert taxonomy + post-hoc clustering | **TAKE** | Banks become monitoring labels after activation clustering, not hard assignment. |
| 5 | A35B → expert expand | **KEEP** | Core genetic strategy. |
| 6 | Noise schedule 0.02 → 0.005 | **TAKE** | Replace fixed random/ortho-only init. |
| 7 | IEL + Role Adapter + Memory | **TAKE** | Role Adapter explicit (was implicit in role tokens). |
| 8 | AXO = Agent Experience Optimization | **TAKE** | Rename/clarify behavior-over-answer. |
| 9 | Tokenizer → 100M → 500M → 1.6B | **TAKE** | Split Stage0 into 0a/0b/0c. Do not jump to 1.6B on raw tokenizer. |
| 10 | Five docs before code | **TAKE** | Create `docs/01`…`05`. |
| 11 | `src/latex/` skeleton | **TAKE (later)** | After docs + tokenizer proof; not before. |
| 12 | Six scientific proofs before 480B valid | **TAKE** | Hard research gate. |
| 13 | Touch flagship now | **REJECT** | Flagship configs stay frozen reference only. Trophy = Stage0 checkpoint. |

## Combined positional policy (accepted)

```
For each layer L in [0, N):
  window = local_4k if (L % 4 != 3) else full
  rope   = RoPE if (L / N) < 0.75 else NoPE
  # Global (full) layers below 75% may still use NoPE for long-range binding
  if window == full and (L / N) < 0.75:
      rope = NoPE   # optional ablation: RoPE-full vs NoPE-full
```

Default v1: **local→RoPE**, **full→NoPE**, **top 25% layers→NoPE even if local**.

## Research gate (must pass on ≤1.6B)

1. Tokenizer works (fertility, byte-fallback, specials)  
2. NHAT block trains  
3. Loss decreases  
4. Gradients stable (no NaN, grad_norm bounded)  
5. MoE expansion possible (tiny MoE proxy from dense)  
6. IEL separates from trunk (swap identity, frozen trunk, behavior shift)

If unmet → 480B design is **non-operational**.
