#!/usr/bin/env bash
# Alias — build seed JSONL
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python 20B-G/scripts/build_agent_seed.py
