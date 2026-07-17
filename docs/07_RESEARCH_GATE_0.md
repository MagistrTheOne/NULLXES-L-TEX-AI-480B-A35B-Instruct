# NULLXES-LÆTEX Research Gate 0

## Official name

**NULLXES-LÆTEX Research Gate 0: Tokenizer Fertility & Representation Gate**

Validates **how information is represented**, not only tokenize speed.

| Field | Value |
|-------|-------|
| Status | **ACTIVE** — NHAT / Stage0a **BLOCKED** until PASS |
| Target | NULLXES-LÆTEX Tokenizer v0.1 |
| Vocab | 131072 (LATEX-Vocab-131k) |
| Compute | 1× H200 (via `configs/runtime.yaml`) |
| Experiment | E001 |

---

## Pass criteria

- [ ] Vocab trains to **131072** with locked special IDs **0–11**
- [ ] Encode/decode **deterministic**
- [ ] Fertility + inflation thresholds pass on `tests/tokenizer_samples/`
- [ ] Fragmentation suite passes (enterprise IDs)
- [ ] Reconstruction **100%** on plain / code / json (NFKC exceptions documented)
- [ ] No Unicode destruction on math / code / URL / email
- [ ] Versioned `tokenizer/latex-v0.1/` + `checksum.sha256`
- [ ] **No** pretrained SentencePiece / foreign tokenizer used

## Fail → stop

Do **not** implement NHAT, MoE, or Stage0a training until every box is checked and logged in `05_EXPERIMENT_TRACKER.md`.

## After PASS

Only then: Stage0a ~100M dense (`configs/stage0a_100m.yaml`).

## Forbidden during Gate 0

- `src/latex/models/` NHAT code
- Loading foreign foundation tokenizers
- Automatic vocab expand 131k → 262k
- Synthetic persona / tone / employee-behavior corpus for tokenizer training
