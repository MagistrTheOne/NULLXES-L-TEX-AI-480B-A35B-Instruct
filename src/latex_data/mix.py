"""Manifest load/save and mix weight checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mix_sum_ok(mix: dict[str, float], tol: float = 1e-6) -> bool:
    return abs(sum(mix.values()) - 1.0) <= tol


def iter_manifest_shards(manifest: dict[str, Any], root: Path) -> list[tuple[str, Path]]:
    """Return (bucket, absolute_path) for every shard in manifest."""
    out: list[tuple[str, Path]] = []
    for bucket, meta in (manifest.get("shards") or {}).items():
        for rel in meta.get("files") or []:
            out.append((bucket, (root / rel).resolve()))
    return out
