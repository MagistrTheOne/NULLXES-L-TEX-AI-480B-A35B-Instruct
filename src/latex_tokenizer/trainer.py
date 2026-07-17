"""Train NULLXES-LÆTEX tokenizer via SentencePiece trainer library only."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from latex_tokenizer.corpus import (
    discover_shards,
    iter_jsonl_shard,
    iter_manifest_texts,
    iter_sample_dir,
    write_train_corpus,
)
from latex_tokenizer.normalize import normalize_text


class PretrainedLoadForbidden(RuntimeError):
    """Raised if code attempts to load a foreign/pretrained SP model for training init."""


def _special_list(cfg: dict[str, Any]) -> list[tuple[str, int]]:
    specs = cfg["special_tokens"]
    items = [(v["token"], int(v["id"])) for v in specs.values()]
    items.sort(key=lambda x: x[1])
    # verify contiguous 0..n-1
    for i, (_, tid) in enumerate(items):
        if tid != i:
            raise ValueError(f"Special token IDs must be contiguous from 0; got id={tid} at {i}")
    return items


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_user_defined_symbols(cfg: dict[str, Any]) -> list[str]:
    return [tok for tok, _ in _special_list(cfg)]


def prepare_corpus(cfg: dict[str, Any], runtime: dict[str, Any], smoke: bool) -> Path:
    storage = runtime.get("storage", {})
    dataset_path = Path(storage.get("dataset_path", "datasets"))
    # Prefer repo-relative datasets/ when runtime points at /workspace
    repo_guess = Path(__file__).resolve().parents[2]
    if not dataset_path.is_dir() or str(dataset_path).startswith("/workspace"):
        local = repo_guess / "datasets"
        if local.is_dir():
            dataset_path = local

    artifact_dir = Path(cfg["paths"]["artifact_dir"])
    tmp = artifact_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    corpus_path = tmp / ("corpus_smoke.txt" if smoke else "corpus.txt")

    manifest_path = repo_guess / "datasets" / "manifests" / "gate0_tokenizer.json"
    if cfg.get("paths", {}).get("corpus_manifest"):
        manifest_path = Path(cfg["paths"]["corpus_manifest"])
        if not manifest_path.is_absolute():
            manifest_path = repo_guess / manifest_path

    def gen():
        if manifest_path.is_file():
            man = json.loads(manifest_path.read_text(encoding="utf-8"))
            for text in iter_manifest_texts(man, repo_guess):
                yield normalize_text(text)
        else:
            shards = discover_shards(dataset_path)
            for shard in shards:
                for text in iter_jsonl_shard(shard):
                    yield normalize_text(text)
        # Always include fixed fertility probes
        samples = Path(cfg["paths"]["samples_dir"])
        for _, text in iter_sample_dir(samples):
            yield normalize_text(text)

    max_chars = 5_000_000 if smoke else None
    n = write_train_corpus(gen(), corpus_path, max_chars=max_chars)
    if n == 0:
        raise FileNotFoundError(
            f"No training text. Run: python scripts/build_seed_corpus.py "
            f"(looked for {manifest_path} and {dataset_path})"
        )
    return corpus_path


def train_tokenizer(
    cfg: dict[str, Any],
    runtime: dict[str, Any],
    *,
    smoke: bool = False,
) -> Path:
    """
    Train Unigram model with byte fallback.
    Emits versioned artifacts under tokenizer/latex-v0.1/.
    """
    sp_cfg = cfg.get("sentencepiece", {})
    if sp_cfg.get("allow_pretrained_load", False):
        raise PretrainedLoadForbidden(
            "allow_pretrained_load must remain false — NULLXES owns vocab artifacts"
        )
    if not sp_cfg.get("use_trainer_library", True):
        raise RuntimeError("Gate 0 requires sentencepiece trainer library")

    try:
        import sentencepiece as spm
    except ImportError as e:
        raise ImportError("Install sentencepiece: pip install sentencepiece") from e

    artifact_dir = Path(cfg["paths"]["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Always write locked specials first
    specials_path = artifact_dir / "special_tokens.json"
    specials_path.write_text(
        json.dumps(cfg["special_tokens"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    corpus_path = prepare_corpus(cfg, runtime, smoke=smoke)
    if smoke:
        # Tiny sample corpus cannot fill 131k / 1024 — use safe smoke size
        vocab_size = 450
    else:
        vocab_size = int(cfg["vocab_size"])
    # Reserve room: SP vocab_size includes specials when user_defined_symbols set
    symbols = build_user_defined_symbols(cfg)
    # First four are pad/unk/bos/eos — pass as control pieces; rest as user_defined
    control = {i: tok for tok, i in _special_list(cfg) if i <= 3}
    extra_symbols = [tok for tok, i in _special_list(cfg) if i > 3]

    model_prefix = artifact_dir / "tmp" / ("spm_smoke" if smoke else "spm")
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    train_kwargs = dict(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=float(sp_cfg.get("character_coverage", 0.9995)),
        byte_fallback=True,
        normalization_rule_name="identity",  # we pre-normalize NFKC ourselves
        add_dummy_prefix=True,
        remove_extra_whitespaces=False,
        user_defined_symbols=extra_symbols,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=control[0],
        unk_piece=control[1],
        bos_piece=control[2],
        eos_piece=control[3],
        train_extremely_large_corpus=not smoke,
        seed_sentencepiece_size=int(cfg.get("seed", 42)),
    )
    nthreads = int(sp_cfg.get("num_threads", 0))
    if nthreads > 0:
        train_kwargs["num_threads"] = nthreads

    spm.SentencePieceTrainer.train(**train_kwargs)

    model_src = Path(str(model_prefix) + ".model")
    vocab_src = Path(str(model_prefix) + ".vocab")
    model_dst = artifact_dir / "tokenizer.model"
    vocab_dst = artifact_dir / "vocab.json"

    shutil.copy2(model_src, model_dst)

    # Export vocab.json as id -> piece (NULLXES-owned artifact)
    sp = spm.SentencePieceProcessor()
    # Loading OUR freshly trained model is required — not a foreign pretrained
    sp.load(str(model_dst))
    vocab = {i: sp.id_to_piece(i) for i in range(sp.get_piece_size())}
    vocab_dst.write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checksum_path = artifact_dir / "checksum.sha256"
    lines = [
        f"{_sha256_file(model_dst)}  tokenizer.model",
        f"{_sha256_file(vocab_dst)}  vocab.json",
        f"{_sha256_file(specials_path)}  special_tokens.json",
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "name": cfg.get("name"),
        "version": cfg.get("version"),
        "vocab_size_config": cfg["vocab_size"],
        "vocab_size_trained": sp.get_piece_size(),
        "smoke": smoke,
        "byte_fallback": True,
        "pretrained_load": False,
        "seed": cfg.get("seed", 42),
    }
    (artifact_dir / "meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return artifact_dir


def load_owned_processor(artifact_dir: Path):
    """Load NULLXES-owned tokenizer.model from versioned directory only."""
    import sentencepiece as spm

    model_path = artifact_dir / "tokenizer.model"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing {model_path} — run train_tokenizer.py first")
    # Refuse paths that look like foreign dumps
    if "latex-v" not in artifact_dir.as_posix():
        raise PretrainedLoadForbidden(
            f"Refusing to load tokenizer outside versioned latex-v* dir: {artifact_dir}"
        )
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp
