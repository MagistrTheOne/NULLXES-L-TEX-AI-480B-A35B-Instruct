# Failure Modes

Stop rules apply from Stage0 upward. Log `F00x` id in trainer.

## F001 — Router collapse

**Signal:** >40% tokens → one expert; gate entropy collapse.  
**Stop:** yes after 1k steps sustained.  
**Mitigation:** aux-loss-free bias, capacity factor, router fp32, short aux warmup.

## F002 — Dead experts

**Signal:** <10% experts with >1% utilization.  
**Stop:** yes if persists past warmup.  
**Mitigation:** diversity loss, re-init dead experts from shared + noise, lower top-k temporarily.

## F003 — Tokenizer fragmentation

**Signal:** fertility blow-up on RU/CN/code; special-token misuse; UNK >0.1% without byte fallback.  
**Stop:** block Stage0c+.  
**Mitigation:** rebuild vocab; freeze specials; fertility suite before any ≥500M run.

## F004 — Identity leakage

**Signal:** Anna answers as Karen with frozen IEL swap; or policy of one role bleeds.  
**Stop:** block character release.  
**Mitigation:** IEL/Role isolation eval; no persona facts in trunk SFT without tags.

## F005 — Long context degradation

**Signal:** needle fail at 32k/128k; attention sink pathologies.  
**Stop:** block context marketing claims.  
**Mitigation:** hybrid window, NoPE on full/top layers, YaRN ablation, needle gate.

## F006 — Gradient explosion / NaN

**Signal:** grad_norm >10× baseline or NaN loss.  
**Stop:** immediate.  
**Mitigation:** grad clip 1.0, Z-loss, bf16/FP8 TE checks, lower LR.

## F007 — Expert twin collapse (post-expand)

**Signal:** cosine(expert_i, expert_j) >0.95 for many pairs after expand.  
**Stop:** re-expand with higher noise floor (then decay).  
**Mitigation:** noise schedule 0.02→0.005; diversity loss.

## F008 — Assistant regression

**Signal:** model reverts to generic “helpful chatbot” under AXO/EPO.  
**Stop:** fail AXO eval.  
**Mitigation:** employee trajectory data; punish instruction-dump without tools/verify.
