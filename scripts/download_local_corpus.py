#!/usr/bin/env python3
"""
Download licensed HF samples for Gate A proxy / local_2080 mini pretrain.

  export HF_HUB_ENABLE_HF_TRANSFER=1   # required — no fallback
  pip install hf_transfer
  export HF_TOKEN=...                  # optional for public; needed for gated
  python scripts/download_local_corpus.py --config configs/datasets_gate_a_proxy.yaml

Resume: existing complete shards are skipped. Manifest rewritten at end.
Never writes tokens to disk. No FineWeb/CC full dumps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _log(msg: str) -> None:
    print(msg, flush=True)


def require_hf_transfer() -> None:
    """HF_HUB_ENABLE_HF_TRANSFER=1 with no silent disable / no fallback."""
    flag = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "").strip()
    if flag not in ("1", "true", "True", "yes"):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        _log("[env] set HF_HUB_ENABLE_HF_TRANSFER=1 (required, no fallback)")
    try:
        import hf_transfer  # noqa: F401
    except ImportError as e:
        _log("[fail] hf_transfer required but not installed.")
        _log("  pip install -U hf_transfer")
        raise SystemExit(1) from e
    _log("[ok] hf_transfer ready (HF_HUB_ENABLE_HF_TRANSFER=1)")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def truncate(text: str, max_chars: int) -> str:
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rsplit(" ", 1)[0] + "…"


def download_source(src: dict[str, Any], out_dir: Path, defaults: dict[str, Any]) -> dict[str, Any]:
    from datasets import load_dataset

    max_docs = int(src.get("max_docs", defaults.get("max_docs_per_source", 2000)))
    max_chars = int(src.get("max_chars_per_doc", defaults.get("max_chars_per_doc", 8000)))
    text_field = src["text_field"]
    prefix_field = src.get("text_prefix_field")
    hub = src["hub"]
    split = src.get("split", "train")
    config = src.get("config")
    revision = src.get("revision")
    streaming = bool(src.get("streaming", True))
    allow_types = set(src.get("filter_open_type") or [])
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    out = out_dir / f"{src['id']}.jsonl"

    # Resume: skip complete shards
    existing = count_jsonl_lines(out)
    if existing >= max_docs:
        _log(f"[skip] {src['id']}: already have {existing} docs (≥{max_docs})")
        return {
            "id": src["id"],
            "hub": hub,
            "bucket": src["bucket"],
            "file": str(out.relative_to(ROOT).as_posix()),
            "docs": existing,
            "resumed": True,
        }

    _log(f"[dl] {src['id']}: {hub}" + (f"/{config}" if config else "") + f" n≤{max_docs}")

    kwargs: dict[str, Any] = {
        "split": split,
        "streaming": streaming,
        "trust_remote_code": False,
        "token": token,
    }
    if revision:
        kwargs["revision"] = revision
    if config:
        ds = load_dataset(hub, config, **kwargs)
    else:
        ds = load_dataset(hub, **kwargs)

    rows: list[dict[str, Any]] = []
    scanned = 0
    for ex in ds:
        scanned += 1
        if scanned > max_docs * 40 and len(rows) < max_docs // 4:
            _log(f"[warn] {src['id']}: sparse filter, scanned={scanned} kept={len(rows)}")
            break
        if allow_types:
            ot = ex.get("open_type") or ex.get("collection") or ""
            if ot not in allow_types:
                continue
        raw = ex.get(text_field) or ""
        if prefix_field and isinstance(ex.get(prefix_field), str):
            raw = f"{ex[prefix_field].strip()}\n\n{raw}" if raw else ex[prefix_field]
        if not isinstance(raw, str) or len(raw.strip()) < 64:
            continue
        text = truncate(raw, max_chars)
        if len(text) < 64:
            continue
        rows.append(
            {
                "id": f"{src['id']}_{len(rows):06d}",
                "text": text,
                "source": hub,
                "bucket": src["bucket"],
            }
        )
        if len(rows) >= max_docs:
            break

    n = write_jsonl(out, rows)
    _log(f"[ok] {out.relative_to(ROOT)} docs={n} scanned={scanned}")
    return {
        "id": src["id"],
        "hub": hub,
        "bucket": src["bucket"],
        "file": str(out.relative_to(ROOT).as_posix()),
        "docs": n,
        "resumed": False,
    }


def merge_manifest(cfg: dict[str, Any], shard_infos: list[dict[str, Any]]) -> Path:
    buckets: dict[str, dict[str, Any]] = {}
    for info in shard_infos:
        b = info["bucket"]
        buckets.setdefault(b, {"files": [], "docs": 0})
        buckets[b]["files"].append(info["file"])
        buckets[b]["docs"] += info["docs"]

    local = cfg.get("local_identity") or {}
    if local.get("enabled"):
        man_path = ROOT / local["manifest"]
        if man_path.is_file():
            old = json.loads(man_path.read_text(encoding="utf-8"))
            for b, shard in (old.get("shards") or {}).items():
                buckets.setdefault(b, {"files": [], "docs": 0})
                for f in shard.get("files") or []:
                    if f not in buckets[b]["files"]:
                        buckets[b]["files"].append(f)
                buckets[b]["docs"] += int(shard.get("docs") or 0)

    man = {
        "name": cfg["name"],
        "version": cfg.get("version", "0.1"),
        "hardware": "runpod_or_local",
        "mix_note": "HF Gate A proxy + local identity; no FineWeb; hf_transfer required",
        "shards": buckets,
        "sources_ok": [i["id"] for i in shard_infos],
    }
    out = ROOT / cfg["manifest"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _log(f"[manifest] {out.relative_to(ROOT)}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="local_2080/configs/datasets_mini.yaml")
    args = p.parse_args()

    require_hf_transfer()

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        _log("[warn] HF_TOKEN not set — public datasets may still work; gated ones will fail")

    cfg_path = ROOT / args.config
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    defaults = cfg.get("defaults") or {}

    infos: list[dict[str, Any]] = []
    failed: list[str] = []
    for src in cfg["sources"]:
        try:
            infos.append(download_source(src, out_dir, defaults))
            # rewrite manifest after each source so a crash mid-run is recoverable
            merge_manifest(cfg, infos)
        except Exception as e:
            _log(f"[fail] {src['id']}: {e}")
            failed.append(src["id"])
            if infos:
                merge_manifest(cfg, infos)
            return 1

    merge_manifest(cfg, infos)
    _log(f"[done] sources={len(infos)} failed={failed}")
    _log("next: python scripts/validate_corpus.py --manifest " + cfg["manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
