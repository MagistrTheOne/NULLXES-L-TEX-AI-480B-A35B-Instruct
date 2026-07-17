"""Streaming corpus helpers — mmap/shard friendly, no full RAM load."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


MIX_KEYS = (
    "multilingual",
    "code",
    "enterprise",
    "scientific",
    "synthetic_structure",
)


def iter_jsonl_shard(path: Path) -> Iterator[str]:
    """Yield text fields from a JSONL shard (streaming)."""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
            if isinstance(obj, str):
                yield obj
            elif isinstance(obj, dict):
                text = obj.get("text") or obj.get("content") or obj.get("body")
                if text:
                    yield str(text)


def iter_text_file(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        yield f.read()


def discover_shards(dataset_path: Path) -> list[Path]:
    raw = dataset_path / "raw" / "shards"
    if raw.is_dir():
        shards = sorted(raw.glob("**/*.jsonl"))
        if shards:
            return shards
    seed = dataset_path / "seed"
    if seed.is_dir():
        shards = sorted(seed.glob("**/*.jsonl"))
        if shards:
            return shards
    if dataset_path.is_dir():
        return sorted(dataset_path.glob("**/*.jsonl"))
    return []


def iter_manifest_texts(manifest: dict[str, Any], repo_root: Path) -> Iterator[str]:
    """Yield texts following manifest shard list (all docs, bucket by bucket)."""
    shards = manifest.get("shards") or {}
    for bucket in (manifest.get("mix") or shards).keys():
        meta = shards.get(bucket) or {}
        for rel in meta.get("files") or []:
            path = repo_root / rel
            if path.is_file():
                yield from iter_jsonl_shard(path)


def iter_sample_dir(samples_dir: Path) -> Iterator[tuple[str, str]]:
    """Yield (name, text) from fixed Gate 0 benchmark samples."""
    if not samples_dir.is_dir():
        return
    for path in sorted(samples_dir.iterdir()):
        if path.is_file():
            yield path.name, path.read_text(encoding="utf-8", errors="replace")


def write_train_corpus(
    texts: Iterator[str],
    out_path: Path,
    max_chars: int | None = None,
) -> int:
    """Write newline-delimited training corpus for SentencePiece (streaming)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    total_chars = 0
    with out_path.open("w", encoding="utf-8") as out:
        for text in texts:
            if not text.strip():
                continue
            flat = " ".join(text.splitlines())
            out.write(flat + "\n")
            n += 1
            total_chars += len(flat)
            if max_chars is not None and total_chars >= max_chars:
                break
    return n
