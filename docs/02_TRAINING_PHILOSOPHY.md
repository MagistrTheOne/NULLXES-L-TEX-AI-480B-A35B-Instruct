# Training Philosophy

## North star

Prove science on a **small real model**, then scale. The strongest idea is not 480B — it is:

**20B dense V1 → larger dense ancestor → MoE expansion.**

## Principles

1. **No foreign foundation weights.** Own tokenizer, init, router, alignment.  
2. **Reality over ambition.** Stage0 trophy before flagship spend.  
3. **μP transfer.** LR/init tuned on proxies; transferred up.  
4. **Behavior > answer.** Alignment optimizes employee trajectories (AXO), not chatty replies.  
5. **Identity ≠ knowledge.** Trunk holds competence; IEL/Role/Memory hold persona and state.  
6. **Experts earn labels.** Free routing early; cluster → label later.  
7. **Fail closed.** Documented failure modes with stop rules (see `04_FAILURE_MODES.md`).

## AXO — Agent Experience Optimization

Optimize **behavior**, not surface text.

| Bad (assistant) | Good (NULLXES employee) |
|-----------------|-------------------------|
| «Вот инструкция» | «Проверил 3 источника, подготовил решение, жду подтверждения.» |
| Hallucinated action | Real tool_call + tool_result loop |
| Endless chatter | Escalate / stop when policy requires |

Pipeline: **SFT → DPO → AXO → EPO** (Enterprise Preference Optimization).

## Token budget mindset

MoE: budget vs **active** params; quality multi-epoch beats fake Chinchilla on total params.

## What we refuse to optimize early

- 480B kernel micro-opts before Stage0 loss is boringly stable  
- Fancy MLA before GQA baseline works  
- Hard expert taxonomy before activation clustering exists  
