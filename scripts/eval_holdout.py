#!/usr/bin/env python3
"""
Holdout loss for a LÆTEX checkpoint.

  python scripts/eval_holdout.py \
      --checkpoint checkpoints/latex-20b-v1-iter1 \
      --manifest datasets/manifests/corpus_latex_v1_holdout.json \
      --device cuda

Train loss falls whether the model is learning language or memorizing the
corpus; holdout is what separates the two. Windows are packed exactly like
training windows so the two numbers are comparable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_holdout_texts(manifest_path: Path, limit: int = 0) -> list[str]:
    from latex_data.mix import iter_manifest_shards

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    texts: list[str] = []
    for _bucket, path in iter_manifest_shards(manifest, ROOT):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = (json.loads(line).get("text") or "").strip()
                except json.JSONDecodeError:
                    continue
                if text:
                    texts.append(text)
                if limit and len(texts) >= limit:
                    return texts
    return texts


def main() -> int:
    p = argparse.ArgumentParser(description="LÆTEX holdout evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", default="datasets/manifests/corpus_latex_v1_holdout.json")
    p.add_argument("--tokenizer", default="", help="Defaults to the checkpoint dir")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-windows", type=int, default=64)
    p.add_argument("--max-docs", type=int, default=4000)
    p.add_argument("--out", default="", help="Report path (default: <checkpoint>/holdout_report.json)")
    args = p.parse_args()

    ckpt = ROOT / args.checkpoint
    manifest_path = ROOT / args.manifest
    if not manifest_path.is_file():
        _log(f"[fail] holdout manifest not found: {manifest_path}")
        return 2

    texts = load_holdout_texts(manifest_path, limit=args.max_docs)
    if not texts:
        _log("[fail] holdout is empty — rebuild with scripts/build_corpus_v1.py")
        return 2

    import torch

    import latex  # noqa: F401
    from latex import LatexForCausalLM, LatexTokenizer
    from latex_data.batching import IGNORE_INDEX

    tok_dir = ROOT / args.tokenizer if args.tokenizer else ckpt
    tokenizer = LatexTokenizer(
        vocab_file=str(tok_dir / "tokenizer.model"),
        special_tokens_map_file=str(tok_dir / "special_tokens.json"),
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _log(f"[load] {ckpt} → {device}")
    model = LatexForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to(device).eval()

    eos_id = tokenizer.eos_token_id or 3
    stream: list[int] = []
    windows: list[list[int]] = []
    for text in texts:
        stream.extend(tokenizer.encode(text, add_special_tokens=False) + [eos_id])
        while len(stream) >= args.seq_len and len(windows) < args.max_windows:
            windows.append(stream[: args.seq_len])
            stream = stream[args.seq_len :]
        if len(windows) >= args.max_windows:
            break

    if not windows:
        _log(
            f"[fail] holdout has fewer than one full window of {args.seq_len} tokens "
            f"({len(stream)} tokens total)"
        )
        return 2

    # Token-weighted mean: windows are equal length, but the last one may not be.
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for window in windows:
            ids = torch.tensor([window], dtype=torch.long, device=device)
            logits = model(input_ids=ids).logits.float()
            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = ids[:, 1:].reshape(-1)
            loss = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="sum"
            )
            total_loss += float(loss)
            total_tokens += int(shift_labels.numel())

    mean_loss = total_loss / max(total_tokens, 1)
    report = {
        "checkpoint": str(ckpt),
        "manifest": str(manifest_path),
        "windows": len(windows),
        "tokens": total_tokens,
        "holdout_loss": mean_loss,
        "holdout_ppl": math.exp(min(mean_loss, 20.0)),
    }
    out_path = Path(args.out) if args.out else ckpt / "holdout_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _log(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
