# NULLXES-LÆTEX datasets

See [`docs/10_CORPUS_PLAN.md`](../docs/10_CORPUS_PLAN.md).

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
python scripts/validate_corpus.py --manifest datasets/manifests/gate0_tokenizer.json
```

Product site (real): https://www.nullxesdai.online/
