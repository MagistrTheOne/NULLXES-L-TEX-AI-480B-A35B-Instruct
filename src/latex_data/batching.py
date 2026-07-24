"""Document packing and label masking — shared data plane for LÆTEX trainers.

Documents are concatenated through EOS into continuous windows of exactly
`seq_len` tokens. Padding therefore only ever appears in a trailing window,
and every counted token is a token the model actually learned from.

Both Stage0a and the ZeRO trainer import this so token accounting and loss
masking cannot drift apart between them.
"""

from __future__ import annotations

import random
from typing import Callable, Iterable

IGNORE_INDEX = -100

BASE = "base"
SOFT = "soft"
MANTRA = "mantra"


class DocSampler:
    """Picks documents according to the identity mix probabilities.

    `mantra_mix` and `soft_mix` are probabilities of drawing from the hard
    identity Q/A set and the soft identity prose set respectively; everything
    else comes from the general pool.
    """

    def __init__(
        self,
        base: list[str],
        soft: list[str],
        mantra: list[str],
        *,
        mantra_mix: float,
        soft_mix: float,
        rng: random.Random,
        is_soft: Callable[[str], bool] | None = None,
    ):
        self._base = base
        self._soft = soft
        self._mantra = mantra
        self._mantra_mix = float(mantra_mix)
        self._soft_mix = float(soft_mix)
        self._rng = rng
        self._is_soft = is_soft or (lambda _t: False)
        self._pool = base + soft or list(mantra)
        if not self._pool:
            raise ValueError("empty document pool")

    def next(self) -> tuple[str, str]:
        r = self._rng.random()
        if self._mantra and r < self._mantra_mix:
            return self._rng.choice(self._mantra), MANTRA
        if self._soft and r < self._mantra_mix + self._soft_mix:
            return self._rng.choice(self._soft), SOFT
        text = self._rng.choice(self._pool)
        return text, (SOFT if self._is_soft(text) else BASE)


class SequencePacker:
    """Streams packed `seq_len` windows out of a document sampler.

    Leftover tokens of a document carry over into the next window instead of
    being padded away, so short identity docs cost their real length rather
    than a full sequence.
    """

    def __init__(
        self,
        sampler: DocSampler,
        encode: Callable[[str], list[int]],
        *,
        seq_len: int,
        eos_id: int,
        kind_weights: dict[str, float] | None = None,
        min_doc_tokens: int = 8,
    ):
        self._sampler = sampler
        self._encode = encode
        self._seq_len = int(seq_len)
        self._eos_id = int(eos_id)
        self._weights = kind_weights or {}
        self._min_doc_tokens = int(min_doc_tokens)
        self._ids: list[int] = []
        self._token_weights: list[float] = []
        self.tokens_by_kind: dict[str, int] = {BASE: 0, SOFT: 0, MANTRA: 0}

    def _fill(self) -> None:
        while len(self._ids) < self._seq_len:
            text, kind = self._sampler.next()
            ids = self._encode(text)
            if len(ids) < self._min_doc_tokens:
                continue
            ids = ids + [self._eos_id]
            weight = float(self._weights.get(kind, 1.0))
            self._ids.extend(ids)
            self._token_weights.extend([weight] * len(ids))
            self.tokens_by_kind[kind] = self.tokens_by_kind.get(kind, 0) + len(ids)

    def window(self) -> tuple[list[int], float]:
        """Return one full window and its token-count-weighted loss weight."""
        self._fill()
        ids = self._ids[: self._seq_len]
        weights = self._token_weights[: self._seq_len]
        self._ids = self._ids[self._seq_len :]
        self._token_weights = self._token_weights[self._seq_len :]
        return ids, sum(weights) / len(weights)

    def batch(self, micro_batch_size: int) -> tuple[list[list[int]], float]:
        """Return `micro_batch_size` windows and their mean loss weight."""
        rows: list[list[int]] = []
        weights: list[float] = []
        for _ in range(micro_batch_size):
            ids, w = self.window()
            rows.append(ids)
            weights.append(w)
        return rows, sum(weights) / len(weights)

    def kind_rates(self) -> dict[str, float]:
        total = sum(self.tokens_by_kind.values()) or 1
        return {k: v / total for k, v in self.tokens_by_kind.items()}


def mask_labels(input_ids, pad_id: int):
    """Clone `input_ids` into labels with pad positions set to IGNORE_INDEX.

    Packed windows are pad-free, but a trailing window or an unpacked caller
    still must not train the model to predict padding.
    """
    labels = input_ids.clone()
    labels[input_ids == pad_id] = IGNORE_INDEX
    return labels


def assistant_only_labels(input_ids, assistant_id: int, pad_id: int, eos_id: int | None = None):
    """Mask everything up to and including `<|assistant|>` in each row.

    Without this the model is trained to produce the system and user turns as
    well, i.e. to be the user.
    """
    labels = mask_labels(input_ids, pad_id)
    for row in range(input_ids.shape[0]):
        seq = input_ids[row]
        positions = (seq == assistant_id).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            labels[row, :] = IGNORE_INDEX
            continue
        start = int(positions[0].item()) + 1
        labels[row, :start] = IGNORE_INDEX
        if eos_id is not None:
            after = (seq[start:] == eos_id).nonzero(as_tuple=True)[0]
            if after.numel() > 0:
                stop = start + int(after[0].item()) + 1
                labels[row, stop:] = IGNORE_INDEX
    return labels


def iter_texts(records: Iterable[dict]) -> list[str]:
    out: list[str] = []
    for obj in records:
        text = (obj.get("text") or obj.get("content") or "").strip()
        if text:
            out.append(text)
    return out
