# Full tokenizer like H200: soft Unigram + pad to 131072 (NO --smoke)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (Test-Path ".\.venv-2080\Scripts\Activate.ps1") {
  .\.venv-2080\Scripts\Activate.ps1
}

Write-Host "=== 1. Corpus (identity + gate0) ===" -ForegroundColor Cyan
python scripts/build_identity_corpus.py
python scripts/validate_corpus.py --manifest datasets/manifests/gate0_tokenizer.json

Write-Host "=== 2. Tokenizer FULL (pad 131072) ===" -ForegroundColor Cyan
python scripts/train_tokenizer.py `
  --config local_2080/configs/tokenizer_stage0.yaml `
  --runtime local_2080/configs/runtime_2080.yaml

Write-Host "=== 3. Evaluate ===" -ForegroundColor Cyan
python scripts/evaluate_tokenizer.py `
  --config local_2080/configs/tokenizer_stage0.yaml

Write-Host "=== meta.json ===" -ForegroundColor Green
Get-Content tokenizer\latex-v0.1\meta.json
