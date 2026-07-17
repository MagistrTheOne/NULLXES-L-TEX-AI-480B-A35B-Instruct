#!/usr/bin/env python3
"""
NULLXES-LÆTEX Stage0a bootstrap pretrain (~100M).

First TRAINED brain — not 7B SFT on random weights.

  python scripts/train_stage0a.py --config configs/stage0a_100m_bootstrap.yaml

Requires: tokenizer/latex-v0.1/ (full train, may be padded to 131072)
          datasets via pretrain_stage0.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_texts(manifest_path: Path, repo: Path, identity_upsample: int) -> list[str]:
    from latex_tokenizer.corpus import iter_manifest_texts

    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    texts: list[str] = []
    for t in iter_manifest_texts(man, repo):
        t = t.strip()
        if len(t) < 32:
            continue
        texts.append(t)
        # Upsample explicit identity docs only
        low = t.lower()
        if (
            "nullxes-lætex" in low
            or "nullxes-latex" in low
            or "short name of the model is" in low
            or "краткое имя модели" in low
            or "when asked who it is" in low
            or "на вопрос «кто ты?»" in low
        ):
            for _ in range(max(0, identity_upsample - 1)):
                texts.append(t)
    if not texts:
        raise FileNotFoundError(f"No texts in {manifest_path} — run build_identity_corpus.py")
    return texts


def text_stream(texts: list[str], rng: random.Random) -> Iterator[str]:
    while True:
        order = list(range(len(texts)))
        rng.shuffle(order)
        for i in order:
            yield texts[i]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage0a_100m_bootstrap.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps", type=int, default=0, help="0 = derive from tokens_target")
    args = p.parse_args()

    import torch
    import yaml
    from torch.optim import AdamW

    import latex  # noqa: F401
    from latex import LatexConfig, LatexForCausalLM, LatexTokenizer
    from latex.models.init_weights import apply_mup_init

    cfg_path = ROOT / args.config
    with cfg_path.open(encoding="utf-8") as f:
        full = yaml.safe_load(f)

    train_cfg = full["training"]
    if not train_cfg.get("enabled", True):
        _log("[fail] training.enabled is false in config")
        return 2

    tok_dir = ROOT / full["tokenizer"]["artifact_dir"]
    if not (tok_dir / "tokenizer.model").is_file():
        _log("[fail] missing tokenizer — run:\n  python scripts/train_tokenizer.py --config configs/tokenizer_stage0.yaml")
        return 2

    man_rel = train_cfg.get("corpus_manifest", "datasets/manifests/pretrain_stage0.json")
    texts = load_texts(ROOT / man_rel, ROOT, int(train_cfg.get("identity_upsample", 4)))
    _log(f"[data] {len(texts)} docs (with identity upsample)")

    config = LatexConfig.from_yaml_model_section(full["model"])
    config.architectures = ["LatexForCausalLM"]
    config.model_type = "latex"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if train_cfg.get("precision") == "bf16" and device.type == "cuda" else torch.float32
    _log(f"[model] construct Stage0a on {device} dtype={dtype} …")
    model = LatexForCausalLM(config)
    apply_mup_init(model, full.get("init") or {})
    model = model.to(device=device, dtype=dtype)
    model.train()

    tokenizer = LatexTokenizer(
        vocab_file=str(tok_dir / "tokenizer.model"),
        special_tokens_map_file=str(tok_dir / "special_tokens.json"),
    )
    _log(f"[tok] vocab_size={tokenizer.vocab_size} (export may be padded)")

    seq_len = int(train_cfg.get("seq_len", 512))
    micro = int(train_cfg.get("micro_batch_size", 4))
    accum = int(train_cfg.get("grad_accum_steps", 8))
    tokens_per_step = seq_len * micro * accum
    tokens_target = int(train_cfg["tokens_target"])
    max_steps = args.max_steps or max(1, tokens_target // tokens_per_step)
    lr = float(train_cfg["lr"])
    warmup = max(1, int(max_steps * float(train_cfg.get("warmup_ratio", 0.05))))
    log_every = int(train_cfg.get("log_every", 20))
    clip = float(train_cfg.get("grad_clip", 1.0))

    opt = AdamW(
        model.parameters(),
        lr=lr,
        betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.95))),
        eps=float(train_cfg.get("eps", 1e-8)),
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
    )

    out_dir = ROOT / full["checkpoint"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    stream = text_stream(texts, rng)

    def encode_batch() -> torch.Tensor:
        batch_ids = []
        while len(batch_ids) < micro:
            text = next(stream)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 8:
                continue
            # pack / truncate
            if len(ids) < seq_len:
                ids = ids + [tokenizer.pad_token_id or 0] * (seq_len - len(ids))
            else:
                start = rng.randint(0, max(0, len(ids) - seq_len))
                ids = ids[start : start + seq_len]
            batch_ids.append(ids[:seq_len])
        return torch.tensor(batch_ids, dtype=torch.long, device=device)

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        # stable then linear decay last 20%
        decay_start = int(max_steps * 0.8)
        if step < decay_start:
            return lr
        t = (step - decay_start) / max(1, max_steps - decay_start)
        return lr * (1.0 - 0.9 * t)

    _log(
        f"[train] steps={max_steps} tokens/step≈{tokens_per_step} "
        f"target≈{max_steps * tokens_per_step / 1e6:.1f}M tokens"
    )
    t0 = time.time()
    tokens_seen = 0
    opt.zero_grad(set_to_none=True)
    last_loss = None

    for step in range(1, max_steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss_acc = 0.0
        for _ in range(accum):
            input_ids = encode_batch()
            out = model(input_ids=input_ids, labels=input_ids)
            loss = out.loss / accum
            if not torch.isfinite(loss):
                _log(f"[fail] non-finite loss at step {step}")
                return 1
            loss.backward()
            loss_acc += float(loss.detach().item()) * accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens_seen += tokens_per_step
        last_loss = loss_acc / accum

        if step % log_every == 0 or step == 1 or step == max_steps:
            elapsed = time.time() - t0
            tps = tokens_seen / max(elapsed, 1e-6)
            _log(
                f"step {step}/{max_steps} loss={last_loss:.4f} "
                f"lr={lr_at(step):.2e} tokens={tokens_seen/1e6:.2f}M tps={tps:.0f}"
            )

        if step % int(full["checkpoint"].get("save_every_steps", 200)) == 0:
            model.save_pretrained(out_dir, safe_serialization=True)
            config.save_pretrained(out_dir)

    # final save + tokenizer bind
    model.cpu()
    model.save_pretrained(out_dir, safe_serialization=True)
    config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    for name in ("tokenizer.model", "special_tokens.json", "vocab.json", "meta.json"):
        src = tok_dir / name
        if src.is_file():
            (out_dir / name).write_bytes(src.read_bytes())

    report = {
        "name": "NULLXES-LÆTEX-100M-Stage0a-v0.1",
        "model_class": "LatexForCausalLM",
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps": max_steps,
        "tokens_seen": tokens_seen,
        "final_loss": last_loss,
        "checkpoint_dir": str(out_dir.relative_to(ROOT)),
        "identity_note": "Knows name LÆTEX / NULLXES-LÆTEX via upsampled identity corpus",
        "passed": last_loss is not None and math.isfinite(last_loss),
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _log(json.dumps(report, indent=2))
    _log(f"[done] HF checkpoint → {out_dir}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
