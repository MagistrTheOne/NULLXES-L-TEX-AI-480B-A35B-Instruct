"""Guard: full tokenizer train must not silently fall back to 9 fertility samples."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latex_tokenizer.corpus import manifest_shard_paths  # noqa: E402
from latex_tokenizer.trainer import prepare_corpus  # noqa: E402


def test_manifest_shard_paths_reports_missing(tmp_path: Path):
    man = {
        "mix": {"code": 1.0},
        "shards": {"code": {"files": ["datasets/latex_v1/train/code/missing.jsonl"]}},
    }
    existing, missing = manifest_shard_paths(man, tmp_path)
    assert existing == []
    assert missing == ["datasets/latex_v1/train/code/missing.jsonl"]


def test_prepare_corpus_rejects_not_built_stub(tmp_path: Path):
    art = tmp_path / "tok"
    art.mkdir()
    man = tmp_path / "manifest.json"
    man.write_text(
        json.dumps({"status": "not_built", "shards": {}, "mix": {}}),
        encoding="utf-8",
    )
    samples = ROOT / "tests" / "tokenizer_samples"
    cfg = {
        "paths": {
            "artifact_dir": str(art),
            "corpus_manifest": str(man),
            "samples_dir": str(samples),
        }
    }
    runtime = {"storage": {"dataset_path": str(tmp_path / "datasets")}}
    with pytest.raises(FileNotFoundError, match="not_built"):
        prepare_corpus(cfg, runtime, smoke=False)


def test_prepare_corpus_rejects_tiny_corpus(tmp_path: Path):
    art = tmp_path / "tok"
    art.mkdir()
    shard = tmp_path / "shard.jsonl"
    shard.write_text(
        json.dumps({"text": "hello world"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    man = tmp_path / "manifest.json"
    man.write_text(
        json.dumps(
            {
                "mix": {"code": 1.0},
                "shards": {"code": {"files": [str(shard)]}},
            }
        ),
        encoding="utf-8",
    )
    samples = ROOT / "tests" / "tokenizer_samples"
    cfg = {
        "paths": {
            "artifact_dir": str(art),
            "corpus_manifest": str(man),
            "samples_dir": str(samples),
        }
    }
    runtime = {"storage": {"dataset_path": str(tmp_path / "datasets")}}
    with pytest.raises(RuntimeError, match="corpus too small"):
        prepare_corpus(cfg, runtime, smoke=False)
