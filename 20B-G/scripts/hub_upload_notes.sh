#!/usr/bin/env bash
# Package / upload notes for Hub from 20B-G artifacts (run on pod after freeze).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "=== Hub upload checklist (20B-G) ==="
echo "Tokenizer: 20B-G/tokenizer/latex-v0.3/"
echo "  → MagistrTheOne/NULLXES-L-TEX-Tokenizer-v0.3"
echo "Baby after iter5: 20B-G/checkpoints/latex-2b-iter5/"
echo "  → MagistrTheOne/NULLXES-L-TEX-2B-Baby-v0.3 (optional)"
echo "Do NOT overwrite MagistrTheOne/NULLXES-L-TEX-20B-Genesis-v0.1 with baby weights."
echo ""
echo "Example (after HF login):"
echo "  huggingface-cli upload MagistrTheOne/NULLXES-L-TEX-Tokenizer-v0.3 20B-G/tokenizer/latex-v0.3 ."
echo "  huggingface-cli upload MagistrTheOne/NULLXES-L-TEX-2B-Baby-v0.3 20B-G/checkpoints/latex-2b-iter5 ."
