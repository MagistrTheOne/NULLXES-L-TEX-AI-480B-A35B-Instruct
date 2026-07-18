# RTX 2080 SUPER — evening train (~100M tokens, ~50M model)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (Test-Path ".\.venv-2080\Scripts\Activate.ps1") {
  .\.venv-2080\Scripts\Activate.ps1
} elseif (Test-Path ".\.venv\Scripts\Activate.ps1") {
  .\.venv\Scripts\Activate.ps1
}

Write-Host "=== GPU ===" -ForegroundColor Cyan
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Close Chrome/Broadcast if OOM — free VRAM
python scripts/train_stage0a.py `
  --config local_2080/configs/latex_50m_2080.yaml `
  --device cuda

Write-Host "=== QA ===" -ForegroundColor Cyan
python scripts/qa_stage0a.py `
  --checkpoint checkpoints/local_2080/latex-50m `
  --device cuda

Write-Host "Done. Next month: H200 industrial — configs/stage0a_*.yaml" -ForegroundColor Green
