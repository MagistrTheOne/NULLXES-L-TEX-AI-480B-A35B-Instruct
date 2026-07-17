"""Validate corpus manifests + JSONL shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from latex_data.mix import iter_manifest_shards, load_manifest, mix_sum_ok
from latex_data.schema import validate_record


def validate_corpus(
    manifest_path: Path,
    *,
    repo_root: Path | None = None,
    min_docs_per_bucket: int = 20,
    min_text_chars: int = 32,
) -> dict[str, Any]:
    root = repo_root or manifest_path.resolve().parents[2]
    # manifests live at datasets/manifests/ → parents[1]=datasets, parents[2]=repo
    if (manifest_path.parent.name == "manifests") and (manifest_path.parents[1].name == "datasets"):
        root = manifest_path.parents[2]

    man = load_manifest(manifest_path)
    report: dict[str, Any] = {
        "manifest": str(manifest_path),
        "passed": True,
        "errors": [],
        "buckets": {},
    }

    mix = man.get("mix") or {}
    if not mix_sum_ok({k: float(v) for k, v in mix.items()}):
        report["passed"] = False
        report["errors"].append(f"mix_sum={sum(mix.values())} expected 1.0")

    shard_pairs = iter_manifest_shards(man, root)
    if not shard_pairs:
        report["passed"] = False
        report["errors"].append("no_shards_in_manifest")

    counts: dict[str, int] = {b: 0 for b in mix}
    for bucket, path in shard_pairs:
        if not path.is_file():
            report["passed"] = False
            report["errors"].append(f"missing_file:{path}")
            continue
        with path.open(encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    report["passed"] = False
                    report["errors"].append(f"json:{path.name}:{lineno}")
                    continue
                if not isinstance(obj, dict):
                    report["passed"] = False
                    report["errors"].append(f"not_object:{path.name}:{lineno}")
                    continue
                errs = validate_record(obj, min_chars=min_text_chars)
                if errs:
                    report["passed"] = False
                    report["errors"].append(f"{path.name}:{lineno}:{','.join(errs)}")
                b = obj.get("bucket", bucket)
                counts[b] = counts.get(b, 0) + 1

    for b, need in ((b, min_docs_per_bucket) for b in mix):
        n = counts.get(b, 0)
        report["buckets"][b] = {"docs": n, "min": need, "ok": n >= need}
        if n < need:
            report["passed"] = False
            report["errors"].append(f"bucket_underfilled:{b}:{n}<{need}")

    report["mix"] = mix
    report["error_count"] = len(report["errors"])
    # cap error list for readability
    if len(report["errors"]) > 50:
        report["errors"] = report["errors"][:50] + [f"... +{report['error_count']-50} more"]
    return report
