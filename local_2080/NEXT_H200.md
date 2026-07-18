# Next month — H200 industrial (outline)

Не смешивать с `local_2080/`.

1. Corpus v0.2 (no FineWeb/CC): peS2o sample + StarCoder py/ts + PG19 + identity  
2. `pretrain_stage0b.json` + validate  
3. Retrain tokenizer (`max_sentence_length` ↑)  
4. `configs/stage0a_100m.yaml` → tokens_target **2B–5B**  
5. QA + HF `…-Stage0a-v0.2`  
6. Keep `NULLXES-LÆTEX-7B-Genesis` as architecture brick  

Local 2080 остаётся для отладки кода и коротких smoke.
