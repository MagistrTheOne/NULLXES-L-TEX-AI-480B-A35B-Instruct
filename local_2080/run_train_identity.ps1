# 2080 continue: hard identity mantra + loss weight (~9h / 100M tokens)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (Test-Path ".\.venv-2080\Scripts\Activate.ps1") {
  .\.venv-2080\Scripts\Activate.ps1
}

Write-Host "=== rebuild identity mantra ===" -ForegroundColor Cyan
python scripts/build_identity_corpus.py

# Patch local manifest to include mantra without re-downloading HF
python -c @"
import json
from pathlib import Path
p = Path('datasets/manifests/pretrain_local_2080.json')
man = json.loads(p.read_text(encoding='utf-8'))
mantra = 'datasets/raw/shards/identity/identity_mantra.jsonl'
for b in ('synthetic_structure', 'multilingual'):
    sh = man.setdefault('shards', {}).setdefault(b, {'files': [], 'docs': 0})
    if mantra not in sh['files']:
        sh['files'].append(mantra)
p.write_text(json.dumps(man, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('manifest patched:', mantra)
"@

Write-Host "=== GPU ===" -ForegroundColor Cyan
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

python scripts/train_stage0a.py `
  --config local_2080/configs/latex_50m_2080_identity.yaml `
  --device cuda

Write-Host "=== QA ===" -ForegroundColor Cyan
python scripts/qa_stage0a.py `
  --checkpoint checkpoints/local_2080/latex-50m-idv1 `
  --device cuda
