# Download mini HF corpus for local 2080 (token via env only — never commit)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (Test-Path ".\.venv-2080\Scripts\Activate.ps1") {
  .\.venv-2080\Scripts\Activate.ps1
}

if (-not $env:HF_TOKEN) {
  Write-Host "Set HF_TOKEN first, e.g.:" -ForegroundColor Yellow
  Write-Host '  $env:HF_TOKEN = "hf_..."'
  exit 2
}

python scripts/download_local_corpus.py --config local_2080/configs/datasets_mini.yaml
python scripts/validate_corpus.py --manifest datasets/manifests/pretrain_local_2080.json
