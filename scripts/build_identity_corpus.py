#!/usr/bin/env python3
"""
Build NULLXES Seed Corpus (identity + architecture + repo code).

Writes:
  datasets/raw/shards/identity/*.jsonl
  datasets/raw/shards/code/nullxes_repo_clean.jsonl
  datasets/raw/shards/docs/*.jsonl
  datasets/raw/shards/reasoning/*.jsonl
  datasets/sft/identity_v0.1.jsonl

Updates manifests:
  datasets/manifests/gate0_tokenizer.json  (seed + raw)
  datasets/manifests/pretrain_stage0.json

Usage:
  python scripts/build_identity_corpus.py
  python scripts/validate_corpus.py --manifest datasets/manifests/gate0_tokenizer.json
  python scripts/validate_corpus.py --manifest datasets/manifests/pretrain_stage0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latex_data.identity_corpus import (  # noqa: E402
    architecture_docs,
    digital_employee_docs,
    enterprise_ai_docs,
    identity_docs,
    sft_identity_examples,
    technical_reasoning_docs,
)
from latex_data.mix import save_manifest  # noqa: E402
from latex_data.schema import validate_record  # noqa: E402

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".cursor",
    "checkpoints",
    "tokenizer",
    "logs",
    ".mypy_cache",
    ".pytest_cache",
    "agent-transcripts",
}
SKIP_FILE_PARTS = {".env", "secrets", "credentials", ".pem", ".key"}
CODE_SUFFIXES = {".py", ".ts", ".tsx", ".md", ".yaml", ".yml", ".json", ".toml"}
MAX_FILE_BYTES = 80_000
MAX_CODE_FILES = 120


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def pack_repo_code(repo: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    idx = 0
    for path in sorted(repo.rglob("*")):
        if not path.is_file():
            continue
        if any(p in SKIP_DIR_NAMES for p in path.parts):
            continue
        if path.suffix.lower() not in CODE_SUFFIXES:
            continue
        name_low = path.name.lower()
        if any(s in name_low for s in SKIP_FILE_PARTS):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == 0 or size > MAX_FILE_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if len(text.strip()) < 40:
            continue
        rel = path.relative_to(repo).as_posix()
        idx += 1
        desc = _describe(rel)
        body = (
            f"File: {rel}\n\n"
            f"Description:\n{desc}\n\n"
            f"Code:\n\n{text.strip()}\n"
        )
        rec = {
            "id": f"nlx-code-repo-{idx:04d}",
            "text": body,
            "lang": "code",
            "bucket": "code",
            "source": "nullxes_repo_clean_v0.1",
            "license": "nullxes_internal",
            "split": "train",
            "path": rel,
        }
        if validate_record(rec, min_chars=32):
            continue
        records.append(rec)
        if len(records) >= MAX_CODE_FILES:
            break
    return records


def _describe(rel: str) -> str:
    if rel.startswith("src/latex"):
        return "NULLXES-LÆTEX model code (NHAT / HF CausalLM surface)."
    if rel.startswith("src/latex_tokenizer"):
        return "NULLXES-LÆTEX tokenizer training and evaluation."
    if rel.startswith("src/latex_data"):
        return "NULLXES corpus / identity data plane."
    if rel.startswith("scripts/"):
        return "NULLXES research lab operational script."
    if rel.startswith("configs/"):
        return "NULLXES-LÆTEX stage configuration."
    if rel.startswith("docs/"):
        return "NULLXES-LÆTEX research documentation."
    return "NULLXES-LÆTEX repository file."


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for _ in path.open(encoding="utf-8"))


def build_manifests(repo: Path) -> None:
    seed_root = repo / "datasets" / "seed"
    raw = repo / "datasets" / "raw" / "shards"

    def seed_file(bucket: str) -> str:
        return f"datasets/seed/{bucket}/seed_0001.jsonl"

    raw_files: dict[str, list[str]] = {
        "multilingual": [
            "datasets/raw/shards/identity/nullxes_identity.jsonl",
        ],
        "code": [
            "datasets/raw/shards/code/nullxes_repo_clean.jsonl",
            "datasets/raw/shards/code/frontend.jsonl",
            "datasets/raw/shards/code/backend.jsonl",
            "datasets/raw/shards/code/infrastructure.jsonl",
        ],
        "enterprise": [
            "datasets/raw/shards/docs/digital_employee.jsonl",
            "datasets/raw/shards/docs/enterprise_ai.jsonl",
        ],
        "scientific": [
            "datasets/raw/shards/identity/latex_architecture.jsonl",
            "datasets/raw/shards/reasoning/technical_reasoning.jsonl",
        ],
        "synthetic_structure": [
            "datasets/sft/identity_v0.1.jsonl",
        ],
    }

    def assemble(name: str, include_seed: bool) -> dict[str, Any]:
        mix = {
            "multilingual": 0.40,
            "code": 0.25,
            "enterprise": 0.20,
            "scientific": 0.10,
            "synthetic_structure": 0.05,
        }
        shards: dict[str, Any] = {}
        totals: dict[str, int] = {}
        for bucket in mix:
            files: list[str] = []
            docs = 0
            if include_seed:
                sf = seed_file(bucket)
                sp = repo / sf
                if sp.is_file():
                    files.append(sf)
                    docs += _count_jsonl(sp)
            for rel in raw_files.get(bucket, []):
                p = repo / rel
                if p.is_file() and _count_jsonl(p) > 0:
                    files.append(rel)
                    docs += _count_jsonl(p)
            shards[bucket] = {"files": files, "docs": docs}
            totals[bucket] = docs
        return {
            "name": name,
            "version": "0.2-identity",
            "mix": mix,
            "shards": shards,
            "totals": totals,
            "notes": (
                "Identity + repo code + seed. "
                "Goal: LÆTEX knows its name; Stage0a pretrain next — not 7B random SFT."
            ),
        }

    gate0 = assemble("gate0_tokenizer", include_seed=True)
    pretrain = assemble("pretrain_stage0", include_seed=True)
    pretrain["status"] = "READY_FOR_STAGE0A"
    save_manifest(repo / "datasets/manifests/gate0_tokenizer.json", gate0)
    save_manifest(repo / "datasets/manifests/pretrain_stage0.json", pretrain)
    print(json.dumps({"gate0_totals": gate0["totals"], "pretrain_totals": pretrain["totals"]}, indent=2))


def main() -> int:
    p = argparse.ArgumentParser(description="Build NULLXES identity seed corpus")
    p.add_argument("--repo", type=Path, default=ROOT)
    args = p.parse_args()
    repo = args.repo.resolve()

    raw = repo / "datasets" / "raw" / "shards"
    n_id = write_jsonl(raw / "identity" / "nullxes_identity.jsonl", identity_docs())
    n_arch = write_jsonl(raw / "identity" / "latex_architecture.jsonl", architecture_docs())
    n_de = write_jsonl(raw / "docs" / "digital_employee.jsonl", digital_employee_docs())
    n_ent = write_jsonl(raw / "docs" / "enterprise_ai.jsonl", enterprise_ai_docs())
    n_rea = write_jsonl(raw / "reasoning" / "technical_reasoning.jsonl", technical_reasoning_docs())

    # Split repo pack into frontend/backend/infra + full clean
    code_all = pack_repo_code(repo)
    frontend = [r for r in code_all if r["path"].endswith((".ts", ".tsx", ".css", ".html"))]
    backend = [
        r
        for r in code_all
        if r["path"].endswith(".py")
        or "/src/latex" in r["path"]
        or r["path"].startswith("scripts/")
    ]
    infra = [
        r
        for r in code_all
        if r["path"].endswith((".yaml", ".yml", ".toml", ".json", ".md"))
        or r["path"].startswith("configs/")
        or r["path"].startswith("docs/")
    ]
    # If no frontend in this repo, write placeholder factual stub (not empty file)
    if not frontend:
        frontend = [
            {
                "id": "nlx-code-frontend-0001",
                "text": (
                    "File: (none in this research repo)\n\n"
                    "Description:\nNULLXES Digital Employee UI lives in product repos; "
                    "this LÆTEX research repository is model/tokenizer/training focused.\n\n"
                    "Code:\n\n// LÆTEX research monorepo has no .tsx surface yet.\n"
                ),
                "lang": "code",
                "bucket": "code",
                "source": "nullxes_repo_clean_v0.1",
                "license": "nullxes_internal",
                "split": "train",
                "path": "_meta/no_frontend.md",
            }
        ]

    n_code = write_jsonl(raw / "code" / "nullxes_repo_clean.jsonl", code_all)
    n_fe = write_jsonl(raw / "code" / "frontend.jsonl", frontend)
    n_be = write_jsonl(raw / "code" / "backend.jsonl", backend or code_all[:5])
    n_inf = write_jsonl(raw / "code" / "infrastructure.jsonl", infra or code_all[:5])

    n_sft = write_jsonl(repo / "datasets" / "sft" / "identity_v0.1.jsonl", sft_identity_examples())

    build_manifests(repo)

    print(
        json.dumps(
            {
                "written": {
                    "identity": n_id,
                    "architecture": n_arch,
                    "digital_employee": n_de,
                    "enterprise_ai": n_ent,
                    "reasoning": n_rea,
                    "repo_code": n_code,
                    "frontend": n_fe,
                    "backend": n_be,
                    "infrastructure": n_inf,
                    "sft_identity": n_sft,
                },
                "next": [
                    "python scripts/validate_corpus.py --manifest datasets/manifests/gate0_tokenizer.json",
                    "python scripts/validate_corpus.py --manifest datasets/manifests/pretrain_stage0.json",
                    "python scripts/train_tokenizer.py --config configs/tokenizer_stage0.yaml",
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
