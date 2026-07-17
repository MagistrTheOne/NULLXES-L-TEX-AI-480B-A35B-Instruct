"""
NULLXES-LÆTEX Tokenizer — HuggingFace PreTrainedTokenizer interface.

Wraps SentencePiece artifacts under tokenizer/latex-v0.1/.
Does NOT load foreign LLM tokenizers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transformers.tokenization_utils import PreTrainedTokenizer


class LatexTokenizer(PreTrainedTokenizer):
    """HF-compatible tokenizer for NULLXES-LÆTEX v0.1."""

    vocab_files_names = {
        "vocab_file": "tokenizer.model",
        "special_tokens_map_file": "special_tokens.json",
    }
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: Optional[str] = None,
        special_tokens_map_file: Optional[str] = None,
        unk_token: str = "<|unk|>",
        bos_token: str = "<|bos|>",
        eos_token: str = "<|eos|>",
        pad_token: str = "<|pad|>",
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self._sp = None
        self._id_to_piece: Dict[int, str] = {}
        self._piece_to_id: Dict[str, int] = {}

        # Locked specials (may be overridden by special_tokens.json)
        self._specials = {
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
        if special_tokens_map_file and Path(special_tokens_map_file).is_file():
            data = json.loads(Path(special_tokens_map_file).read_text(encoding="utf-8"))
            for _k, meta in data.items():
                if isinstance(meta, dict) and "token" in meta and "id" in meta:
                    self._specials[meta["token"]] = int(meta["id"])

        self._vocab_size_override: Optional[int] = None
        if vocab_file and Path(vocab_file).is_file():
            self._load_sp(vocab_file)
            # Prefer padded vocab.json next to tokenizer.model (131072 export)
            vocab_json = Path(vocab_file).with_name("vocab.json")
            if vocab_json.is_file():
                self._load_vocab_json(vocab_json)
        else:
            for tok, tid in self._specials.items():
                self._piece_to_id[tok] = tid
                self._id_to_piece[tid] = tok

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    def _load_sp(self, vocab_file: str):
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(vocab_file)
        self._sp = sp
        for i in range(sp.get_piece_size()):
            piece = sp.id_to_piece(i)
            self._id_to_piece[i] = piece
            self._piece_to_id[piece] = i

    def _load_vocab_json(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        for k, piece in data.items():
            tid = int(k)
            self._id_to_piece[tid] = piece
            self._piece_to_id[piece] = tid
        if data:
            self._vocab_size_override = max(int(k) for k in data.keys()) + 1

    @property
    def vocab_size(self) -> int:
        if self._vocab_size_override is not None:
            return int(self._vocab_size_override)
        if self._sp is not None:
            return int(self._sp.get_piece_size())
        return max(self._id_to_piece.keys(), default=11) + 1

    def get_vocab(self) -> Dict[str, int]:
        return dict(self._piece_to_id)

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        if self._sp is not None:
            return self._sp.encode(text, out_type=str)
        # Fallback: character-level for missing SP (should not be used for Gate0 PASS)
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._piece_to_id.get(token, self._specials.get(self.unk_token, 1))

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_piece.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        if self._sp is not None:
            return self._sp.decode(tokens)
        return "".join(tokens).replace("▁", " ").strip()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        import shutil

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        out_model = save_directory / f"{prefix}tokenizer.model"
        out_specials = save_directory / f"{prefix}special_tokens.json"

        if self.vocab_file and Path(self.vocab_file).is_file():
            shutil.copy2(self.vocab_file, out_model)
        elif not out_model.exists():
            out_model.write_bytes(b"")

        names = [
            "pad",
            "unk",
            "bos",
            "eos",
            "agent",
            "tool_call",
            "memory",
            "identity",
            "workflow",
            "system",
            "user",
            "assistant",
        ]
        specials_payload = {}
        for name in names:
            tok = f"<|{name}|>"
            if tok in self._specials:
                specials_payload[name] = {"token": tok, "id": self._specials[tok]}
        out_specials.write_text(
            json.dumps(specials_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return (str(out_model), str(out_specials))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        path = Path(pretrained_model_name_or_path)
        vocab = path / "tokenizer.model"
        specials = path / "special_tokens.json"
        # Also accept latex-v0.1 layout
        if not vocab.is_file() and (path / "tokenizer" / "latex-v0.1" / "tokenizer.model").is_file():
            vocab = path / "tokenizer" / "latex-v0.1" / "tokenizer.model"
            specials = path / "tokenizer" / "latex-v0.1" / "special_tokens.json"
        return cls(
            vocab_file=str(vocab) if vocab.is_file() else None,
            special_tokens_map_file=str(specials) if specials.is_file() else None,
            **kwargs,
        )


LATEXTokenizer = LatexTokenizer
