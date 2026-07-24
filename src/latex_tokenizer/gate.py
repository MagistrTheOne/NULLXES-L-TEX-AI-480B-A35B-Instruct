"""Blocking artifact gate for tokenizers.

A smoke tokenizer (a few hundred pieces) loads without error and silently
produces a model whose embedding table is mostly unused. Every entry point that
initializes or trains weights calls `require_trainable` first so that failure
mode becomes a startup error instead of a wasted run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LOCKED_SPECIAL_IDS: dict[str, int] = {
    "<|pad|>": 0,
    "<|unk|>": 1,
    "<|bos|>": 2,
    "<|eos|>": 3,
    "<|agent|>": 4,
    "<|tool_call|>": 5,
    "<|memory|>": 6,
    "<|identity|>": 7,
    "<|workflow|>": 8,
    "<|system|>": 9,
    "<|user|>": 10,
    "<|assistant|>": 11,
}


class TokenizerGateError(RuntimeError):
    """Raised when a tokenizer artifact must not be used for weights."""


def read_meta(artifact_dir: Path) -> dict[str, Any]:
    path = Path(artifact_dir) / "meta.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _special_id_errors(artifact_dir: Path) -> list[str]:
    path = Path(artifact_dir) / "special_tokens.json"
    if not path.is_file():
        return ["special_tokens.json missing"]
    data = json.loads(path.read_text(encoding="utf-8"))
    by_token = {
        meta["token"]: int(meta["id"])
        for meta in data.values()
        if isinstance(meta, dict) and "token" in meta and "id" in meta
    }
    errors = []
    for token, expected in LOCKED_SPECIAL_IDS.items():
        got = by_token.get(token)
        if got is None:
            errors.append(f"{token} absent")
        elif got != expected:
            errors.append(f"{token} id={got} expected {expected}")
    return errors


def check_artifact(artifact_dir: str | Path, expected_vocab: int) -> dict[str, Any]:
    """Report whether this artifact may back a weight init or a train run."""
    artifact_dir = Path(artifact_dir)
    meta = read_meta(artifact_dir)
    errors: list[str] = []

    if not (artifact_dir / "tokenizer.model").is_file():
        errors.append(f"tokenizer.model missing under {artifact_dir}")
    if not (artifact_dir / "vocab.json").is_file():
        errors.append(f"vocab.json missing under {artifact_dir}")
    if not meta:
        errors.append("meta.json missing — artifact provenance unknown")
    if meta.get("smoke"):
        errors.append("meta.smoke=true — pipeline test artifact, not a frozen tokenizer")

    export = meta.get("vocab_size_export")
    if export is not None and int(export) != int(expected_vocab):
        errors.append(f"vocab_size_export={export} expected {expected_vocab}")
    if meta.get("vocab_padded"):
        trained = meta.get("vocab_size_trained")
        errors.append(
            f"vocab_padded=true — only {trained} real pieces, rest are <|unused_*|>; "
            "the corpus was too small to fill the vocabulary"
        )

    errors.extend(_special_id_errors(artifact_dir))

    return {
        "passed": not errors,
        "artifact_dir": str(artifact_dir),
        "expected_vocab": int(expected_vocab),
        "vocab_size_export": export,
        "vocab_size_trained": meta.get("vocab_size_trained"),
        "vocab_padded": meta.get("vocab_padded"),
        "smoke": bool(meta.get("smoke")),
        "errors": errors,
    }


def require_trainable(artifact_dir: str | Path, expected_vocab: int) -> dict[str, Any]:
    report = check_artifact(artifact_dir, expected_vocab)
    if not report["passed"]:
        raise TokenizerGateError(
            "tokenizer artifact rejected:\n  - " + "\n  - ".join(report["errors"])
        )
    return report
