"""Load train texts from a LÆTEX corpus manifest (streaming-friendly shards)."""

from __future__ import annotations

import json
from pathlib import Path


def is_soft_identity(text: str) -> bool:
    low = text.lower()
    return any(
        k in low
        for k in (
            "nullxes-lætex",
            "nullxes-latex",
            "lætex-nullxes foundation model",
            "short name of the model is",
            "краткое имя модели",
            "when asked who it is",
            "на вопрос «кто ты?»",
            "i am nullxes",
            "я nullxes",
            "developed by nullxes",
            "разработан nullxes",
            "разработала компания nullxes",
        )
    )


def is_mantra(text: str, source_hint: str = "") -> bool:
    if "identity_mantra" in source_hint or "sft_identity" in source_hint:
        return True
    low = text.lower()
    if "<|assistant|>" in low and (
        "who are you" in low or "как тебя зовут" in low or "кто ты?" in low
    ):
        return True
    if text.startswith("Q:") and "\nA:" in text and ("lætex" in low or "nullxes" in low):
        return True
    return False


def load_corpus(
    manifest_path: Path,
    repo: Path,
    identity_upsample: int = 1,
) -> tuple[list[str], list[str], list[str]]:
    """Return (base_texts, soft_identity_texts, mantra_texts)."""
    from latex_tokenizer.corpus import iter_jsonl_shard

    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    if man.get("status") == "not_built":
        raise FileNotFoundError(
            f"{manifest_path} is a stub — run scripts/build_corpus_v1.py first"
        )

    base: list[str] = []
    soft: list[str] = []
    mantra: list[str] = []
    seen_mantra: set[str] = set()

    shards = man.get("shards") or {}
    for _bucket, meta in shards.items():
        for rel in meta.get("files") or []:
            path = Path(rel)
            if not path.is_absolute():
                path = repo / rel
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    t = (obj.get("text") or obj.get("content") or "").strip()
                    if len(t) < 32:
                        continue
                    src = str(obj.get("source") or "")
                    is_m = bool(obj.get("identity_mantra")) or is_mantra(t, src)
                    if is_m:
                        if t not in seen_mantra:
                            seen_mantra.add(t)
                            mantra.append(t)
                        continue
                    if is_soft_identity(t):
                        soft.append(t)
                        for _ in range(max(0, identity_upsample - 1)):
                            soft.append(t)
                    else:
                        base.append(t)

    extra = repo / "datasets/raw/shards/identity/identity_mantra.jsonl"
    if extra.is_file():
        for t in iter_jsonl_shard(extra):
            t = t.strip()
            if len(t) >= 32 and t not in seen_mantra and is_mantra(t, "identity_mantra"):
                seen_mantra.add(t)
                mantra.append(t)

    if not base and not soft and not mantra:
        raise FileNotFoundError(f"No texts in {manifest_path}")
    return base, soft, mantra
