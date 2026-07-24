#!/usr/bin/env bash
# Optional: package baby ckpt card text for Hub (no upload).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT="${1:-20B-G/checkpoints/latex-2b-iter5}"
OUT="20B-G/docs/HUB_CARD_2B_BABY.md"
cat > "$ROOT/$OUT" <<EOF
---
language: [en, ru]
tags: [nullxes, latex, moe-prep, agent, code]
license: other
---

# NULLXES-LÆTEX ~2B Baby (20B-G)

Dense NHAT baby for **agent + coding** mix proof. Not Digital Employees.

- Params: ~1.85B
- Tokenizer: latex-v0.3 (131072) — only after full freeze, not smoke
- Train: 5×250 steps then optional SFT
- Parent trunk: 20B Genesis (separate Hub repo)

Contact: @MagistrTheOne · ceo@nullxes.com
EOF
echo "[ok] wrote $OUT (source ckpt hint: $CKPT)"
echo "Upload: see 20B-G/scripts/hub_upload_notes.sh"
