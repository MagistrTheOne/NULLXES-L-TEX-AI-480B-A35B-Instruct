# Tokenizer v0.3 freeze status

| State | Meaning |
|-------|---------|
| **smoke** (current local) | Pipeline OK; vocab ≠ 131072 — **not** Gate freeze |
| **freeze** | Full Unigram 131072 on `corpus_agent_code_v03` after `download_corpus.sh` |

```bash
bash 20B-G/scripts/download_corpus.sh
bash 20B-G/scripts/train_tokenizer_v03.sh   # no --smoke
# then tag freeze: meta.json vocab_padded=false, vocab_size_trained≈131072
```

Do not upload smoke artifacts as Hub Tokenizer-v0.3.
