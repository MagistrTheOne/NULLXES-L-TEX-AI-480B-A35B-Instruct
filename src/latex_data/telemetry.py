"""Run telemetry: checkpoint provenance, EMA weights, stage cards.

Six months from now the only question that matters about a checkpoint is
"which corpus, which tokenizer, which commit". A manifest written at save time
is the only way to answer it, so every save goes through here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_files(paths: Iterable[Path]) -> str | None:
    """Order-independent digest over several files."""
    digests = sorted(d for d in (sha256_file(p) for p in paths) if d)
    if not digests:
        return None
    return hashlib.sha256("".join(digests).encode("ascii")).hexdigest()


def git_info(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                args, cwd=str(root), capture_output=True, text=True, timeout=15, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
        "git_branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
    }


def write_checkpoint_manifest(
    out_dir: Path,
    *,
    stage: str,
    step: int,
    tokens_seen: int,
    train_loss: float | None,
    holdout_loss: float | None,
    grad_norm_p50: float | None,
    tokenizer_dir: Path,
    config_path: Path,
    dataset_manifest: Path | None,
    hardware: str,
    ema: bool,
    root: Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = Path(tokenizer_dir)
    manifest = {
        "stage": stage,
        "step": step,
        "tokens_seen": tokens_seen,
        "train_loss": train_loss,
        "holdout_loss": holdout_loss,
        "grad_norm_p50": grad_norm_p50,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tokenizer": {
            "dir": str(tokenizer_dir),
            "sha256": sha256_files(
                [
                    tokenizer_dir / "tokenizer.model",
                    tokenizer_dir / "vocab.json",
                    tokenizer_dir / "special_tokens.json",
                ]
            ),
        },
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "dataset_manifest": {
            "path": str(dataset_manifest) if dataset_manifest else None,
            "sha256": sha256_file(dataset_manifest) if dataset_manifest else None,
        },
        "hardware": hardware,
        "ema": ema,
        **git_info(root),
        **(extra or {}),
    }
    path = out_dir / "checkpoint_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


class CpuEma:
    """Shadow copy of the weights kept in bf16 on CPU.

    A 20B shadow costs ~37.5 GB of host RAM; on GPU it would eat the activation
    budget instead. Updates are periodic rather than per-step because the copy
    back over PCIe is what costs, not the arithmetic.

    Requires unsharded parameters, i.e. ZeRO stage <= 2.
    """

    def __init__(self, model, *, decay: float = 0.999):
        import torch

        self.decay = float(decay)
        self.updates = 0
        self.enabled = True
        self._shadow: dict[str, Any] = {}
        for name, param in model.named_parameters():
            if param.numel() == 0:
                self.enabled = False
                self._shadow.clear()
                return
            self._shadow[name] = param.detach().to("cpu", torch.bfloat16).clone()

    def update(self, model) -> bool:
        if not self.enabled:
            return False
        import torch

        with torch.no_grad():
            for name, param in model.named_parameters():
                shadow = self._shadow.get(name)
                if shadow is None or param.numel() == 0:
                    continue
                current = param.detach().to("cpu", torch.bfloat16)
                shadow.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
        self.updates += 1
        return True

    def state_dict(self) -> dict[str, Any]:
        return self._shadow

    def save(self, out_dir: Path) -> Path | None:
        if not self.enabled:
            return None
        import torch

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "ema_state.pt"
        torch.save({"decay": self.decay, "updates": self.updates, "weights": self._shadow}, path)
        return path


STAGE_CARD_FIELDS = (
    "stage",
    "tokens",
    "train_loss",
    "holdout_loss",
    "tokenizer",
    "dataset_hash",
    "git",
    "hardware",
    "failures",
)


def render_stage_card(card: dict[str, Any]) -> str:
    """Fixed-shape stage record. `failures` is mandatory and must not be empty.

    An empty failures field means nobody looked, so a clean stage records what
    was actually checked instead.
    """
    missing = [f for f in STAGE_CARD_FIELDS if not str(card.get(f, "")).strip()]
    if missing:
        raise ValueError(f"stage card missing required fields: {missing}")
    lines = [f"### {card['stage']}", ""]
    for field in STAGE_CARD_FIELDS[1:]:
        label = field.replace("_", " ").title()
        lines.append(f"- {label}: {card[field]}")
    lines.append("")
    return "\n".join(lines)


def append_stage_card(history_path: Path, card: dict[str, Any]) -> None:
    history_path = Path(history_path)
    body = render_stage_card(card)
    with history_path.open("a", encoding="utf-8") as f:
        f.write("\n" + body)
