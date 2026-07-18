# NULLXES-LÆTEX datasets

See [`docs/10_CORPUS_PLAN.md`](../docs/10_CORPUS_PLAN.md) (gates) and  
[`docs/12_STAGE0_CORPUS_BLUEPRINT.md`](../docs/12_STAGE0_CORPUS_BLUEPRINT.md) (production Stage0 blueprint).

| Path | Git | Role |
|------|-----|------|
| `seed/` | committed | Bootstrap corpus for Gate0 |
| `manifests/` | committed | Mix + shard inventory |
| `schema/` | committed | JSONL record schema |
| `raw/` | ignored | Large production shards |
| `processed/` | ignored | Packed tokens |

```bash
python scripts/scaffold_corpus.py
python scripts/build_seed_corpus.py
python scripts/build_identity_corpus.py
python scripts/validate_corpus.py --manifest datasets/manifests/gate0_tokenizer.json
python scripts/validate_corpus.py --manifest datasets/manifests/pretrain_stage0.json
```

Identity rule: the brain name is **LÆTEX / NULLXES-LÆTEX** (not ChatGPT/Claude/Llama cosplay).

Product site (real): https://www.nullxesdai.online/
