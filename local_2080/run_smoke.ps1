# RTX 2080 SUPER — 50-step smoke (VRAM + loss)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (Test-Path ".\.venv-2080\Scripts\Activate.ps1") {
  .\.venv-2080\Scripts\Activate.ps1
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
  .\.venv\Scripts\Activate.ps1
}

Write-Host "=== GPU ===" -ForegroundColor Cyan
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

if (-not (Test-Path "tokenizer\latex-v0.1\tokenizer.model")) {
  Write-Host "[fail] missing tokenizer/latex-v0.1/tokenizer.model" -ForegroundColor Red
  Write-Host "Copy from RunPod OR: python scripts/train_tokenizer.py --config configs/tokenizer_stage0.yaml"
  exit 2
}

if (-not (Test-Path "datasets\manifests\pretrain_stage0.json")) {
  python scripts/build_identity_corpus.py
}

python scripts/train_stage0a.py `
  --config local_2080/configs/latex_50m_2080_smoke.yaml `
  --device cuda `
  --max-steps 50

Write-Host "=== smoke checkpoint ===" -ForegroundColor Green
Get-ChildItem checkpoints\local_2080\latex-50m-smoke | Select-Object Name, Length
