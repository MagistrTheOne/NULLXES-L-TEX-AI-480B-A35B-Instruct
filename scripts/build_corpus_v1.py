#!/usr/bin/env python3
"""
Build the LÆTEX V1 corpus: subsample -> filter -> dedup -> holdout split.

Input is whatever `download_local_corpus.py` and `build_identity_corpus.py`
already wrote; this script never touches the network.

  python scripts/build_corpus_v1.py --config configs/datasets_latex_v1.yaml
  python scripts/build_corpus_v1.py --config configs/datasets_latex_v1.yaml \
      --tokenizer tokenizer/latex-v1        # measured token counts

The holdout shard is the only honest signal between training stages, so it is
split off here, before any trainer can see the documents.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterator

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latex_data.filters import DedupIndex, FilterConfig, detect_language, quality_reject  # noqa: E402
from latex_data.mix import iter_manifest_shards, save_manifest  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def iter_manifest_docs(manifest_path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    if not manifest_path.is_file():
        _log(f"[warn] manifest not found, skipped: {manifest_path}")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seen_files: set[Path] = set()
    for bucket, path in iter_manifest_shards(manifest, ROOT):
        if path in seen_files or not path.is_file():
            if not path.is_file():
                _log(f"[warn] shard missing: {path}")
            continue
        seen_files.add(path)
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict) and (obj.get("text") or "").strip():
                    yield obj.get("bucket", bucket), obj


class ShardWriter:
    """Buffers per-bucket documents and flushes them to JSONL shards."""

    def __init__(self, out_dir: Path, docs_per_shard: int = 20000):
        self.out_dir = out_dir
        self.docs_per_shard = docs_per_shard
        self._buffers: dict[str, list[dict[str, Any]]] = {}
        self._counts: dict[str, int] = {}
        self._files: dict[str, list[str]] = {}
        self._chars: dict[str, int] = {}

    def add(self, bucket: str, rec: dict[str, Any]) -> None:
        buf = self._buffers.setdefault(bucket, [])
        buf.append(rec)
        self._counts[bucket] = self._counts.get(bucket, 0) + 1
        self._chars[bucket] = self._chars.get(bucket, 0) + len(rec["text"])
        if len(buf) >= self.docs_per_shard:
            self._flush(bucket)

    def _flush(self, bucket: str) -> None:
        buf = self._buffers.get(bucket)
        if not buf:
            return
        index = len(self._files.get(bucket, []))
        path = self.out_dir / bucket / f"shard_{index:04d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for rec in buf:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._files.setdefault(bucket, []).append(path.relative_to(ROOT).as_posix())
        buf.clear()

    def close(self) -> dict[str, Any]:
        for bucket in list(self._buffers):
            self._flush(bucket)
        return {
            bucket: {
                "files": files,
                "docs": self._counts.get(bucket, 0),
                "chars": self._chars.get(bucket, 0),
            }
            for bucket, files in self._files.items()
        }


def count_tokens(shards: dict[str, Any], tokenizer_dir: Path | None, chars_per_token: float):
    """Measured token counts when a frozen tokenizer exists, estimates otherwise."""
    totals = {b: int(meta["chars"] / chars_per_token) for b, meta in shards.items()}
    if tokenizer_dir is None:
        return totals, "estimated_from_chars"

    from latex import LatexTokenizer

    tok = LatexTokenizer(
        vocab_file=str(tokenizer_dir / "tokenizer.model"),
        special_tokens_map_file=str(tokenizer_dir / "special_tokens.json"),
    )
    measured: dict[str, int] = {}
    for bucket, meta in shards.items():
        n = 0
        for rel in meta["files"]:
            with (ROOT / rel).open(encoding="utf-8") as f:
                for line in f:
                    text = json.loads(line).get("text") or ""
                    n += len(tok.encode(text, add_special_tokens=False))
        measured[bucket] = n
    return measured, "measured"


def main() -> int:
    p = argparse.ArgumentParser(description="Build LÆTEX V1 corpus")
    p.add_argument("--config", default="configs/datasets_latex_v1.yaml")
    p.add_argument("--tokenizer", default="", help="Frozen tokenizer dir for exact token counts")
    p.add_argument("--limit", type=int, default=0, help="Debug: stop after N input docs")
    args = p.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    build = cfg["build"]
    filter_cfg = FilterConfig.from_dict(cfg.get("filters"))
    dedup = DedupIndex(hamming_threshold=int((cfg.get("dedup") or {}).get("hamming_threshold", 3)))
    rng = random.Random(int(build.get("seed", 42)))
    holdout_ratio = float(build["holdout_ratio"])
    protected_prefixes = tuple(build.get("protected_source_prefixes") or ["nullxes_"])

    train_dir = ROOT / build["train_dir"]
    holdout_dir = ROOT / build["holdout_dir"]
    for d in (train_dir, holdout_dir):
        if d.exists():
            for old in d.rglob("*.jsonl"):
                old.unlink()

    train = ShardWriter(train_dir)
    holdout = ShardWriter(holdout_dir)

    sources = [ROOT / cfg["manifest"]]
    local = cfg.get("local_identity") or {}
    if local.get("enabled"):
        sources.append(ROOT / local["manifest"])

    seen = 0
    kept = 0
    rejects: dict[str, int] = {}
    for manifest_path in sources:
        _log(f"[read] {manifest_path.relative_to(ROOT)}")
        for bucket, obj in iter_manifest_docs(manifest_path):
            seen += 1
            if args.limit and seen > args.limit:
                break
            text = obj["text"].strip()
            # Canon is short on purpose (identity mantras, protocol rules) and
            # would be cut by the prose length floor. It still has to be unique.
            protected = str(obj.get("source", "")).startswith(protected_prefixes)
            reason = None if protected else quality_reject(text, filter_cfg, bucket=bucket)
            if reason is None:
                reason = dedup.add_or_reject(text)
            if reason is not None:
                rejects[reason] = rejects.get(reason, 0) + 1
                continue
            rec = {
                "id": obj.get("id") or f"latex-v1-{kept:08d}",
                "text": text,
                "lang": obj.get("lang") or detect_language(text),
                "bucket": bucket,
                "source": obj.get("source", "latex_v1"),
                "license": obj.get("license", "permissive"),
                "split": "train",
                "protected": protected,
            }
            # Canon never goes to holdout: it is meant to be learned, so holding
            # it out would both weaken identity and make holdout loss unreadable.
            if not protected and rng.random() < holdout_ratio:
                rec["split"] = "holdout"
                holdout.add(bucket, rec)
            else:
                train.add(bucket, rec)
            kept += 1
            if kept % 20000 == 0:
                _log(f"  kept={kept} seen={seen} unique={len(dedup)}")

    train_shards = train.close()
    holdout_shards = holdout.close()

    tok_dir = (ROOT / args.tokenizer) if args.tokenizer else None
    cpt = float(build.get("chars_per_token_estimate", 3.6))
    train_tokens, token_mode = count_tokens(train_shards, tok_dir, cpt)
    holdout_tokens, _ = count_tokens(holdout_shards, tok_dir, cpt)
    total_tokens = sum(train_tokens.values())

    mix = cfg["mix"]
    common = {
        "name": cfg["name"],
        "version": cfg["version"],
        "mix": mix,
        "token_mode": token_mode,
        "stage": "foundation_bootstrapping",
        "notes": (
            "Filtered + deduplicated LÆTEX V1. Small by design: the goal is a "
            "proven pipeline, not world knowledge. Not a pretraining corpus."
        ),
    }
    save_manifest(
        ROOT / build["train_manifest"],
        {
            **common,
            "split": "train",
            "shards": train_shards,
            "tokens_by_bucket": train_tokens,
            "tokens_total": total_tokens,
        },
    )
    save_manifest(
        ROOT / build["holdout_manifest"],
        {
            **common,
            "split": "holdout",
            "holdout_ratio": holdout_ratio,
            "shards": holdout_shards,
            "tokens_by_bucket": holdout_tokens,
            "tokens_total": sum(holdout_tokens.values()),
        },
    )

    lo = int(build.get("target_tokens_min", 0))
    hi = int(build.get("target_tokens_max", 0))
    in_band = lo <= total_tokens <= hi if hi else True
    summary = {
        "seen": seen,
        "kept": kept,
        "unique": len(dedup),
        "rejects": dict(sorted(rejects.items(), key=lambda kv: -kv[1])),
        "train_tokens": total_tokens,
        "holdout_tokens": sum(holdout_tokens.values()),
        "token_mode": token_mode,
        "target_band": [lo, hi],
        "in_target_band": in_band,
    }
    _log(json.dumps(summary, ensure_ascii=False, indent=2))
    if not in_band:
        _log(
            f"[warn] {total_tokens/1e6:.1f}M tokens outside the {lo/1e6:.0f}-{hi/1e6:.0f}M band — "
            "adjust max_docs per source before freezing the tokenizer"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
