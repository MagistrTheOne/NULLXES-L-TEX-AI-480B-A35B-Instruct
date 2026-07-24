#!/usr/bin/env python3
"""
LÆTEX V1 SFT — loss on the assistant span only.

  deepspeed --num_gpus 4 scripts/train_sft_v1.py --config configs/sft_20b_v1.yaml

Training on the whole sequence teaches the model to produce the system and user
turns as well, i.e. to be the user. Here everything up to and including the
first <|assistant|> token is masked out, and so is padding.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _log(msg: str) -> None:
    print(msg, flush=True)


def _p50(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def load_sft(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="LÆTEX V1 SFT")
    p.add_argument("--config", default="configs/sft_20b_v1.yaml")
    p.add_argument("--local_rank", type=int, default=-1)
    args = p.parse_args()

    import deepspeed
    import torch
    import yaml
    from torch.optim import AdamW

    import latex  # noqa: F401
    from latex import LatexForCausalLM, LatexTokenizer
    from latex_data.batching import assistant_only_labels
    from latex_data.telemetry import CpuEma, write_checkpoint_manifest
    from latex_tokenizer.gate import TokenizerGateError, require_trainable

    cfg_path = ROOT / args.config
    full = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    train_cfg = full["training"]
    if not train_cfg.get("enabled", False):
        _log("[fail] training.enabled is false")
        return 2

    tok_dir = ROOT / full["tokenizer"]["artifact_dir"]
    try:
        require_trainable(tok_dir, int(full["tokenizer"].get("vocab_size", 131072)))
    except TokenizerGateError as e:
        _log(f"[fail] {e}")
        return 2

    data_path = ROOT / train_cfg["sft_dataset"]
    if not data_path.is_file():
        _log(f"[fail] missing SFT set {data_path} — run scripts/build_sft_v1.py")
        return 2
    rows = load_sft(data_path)
    _log(f"[data] {len(rows)} SFT examples from {data_path.relative_to(ROOT)}")

    resume = ROOT / train_cfg["resume_from"]
    if not resume.is_dir():
        _log(f"[fail] missing resume_from={resume}")
        return 2

    tokenizer = LatexTokenizer(
        vocab_file=str(tok_dir / "tokenizer.model"),
        special_tokens_map_file=str(tok_dir / "special_tokens.json"),
    )
    assistant_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    pad_id = tokenizer.pad_token_id or 0
    eos_id = tokenizer.eos_token_id or 3
    if assistant_id is None or assistant_id < 0:
        _log("[fail] tokenizer has no <|assistant|> id — cannot mask the assistant span")
        return 2

    seq_len = int(train_cfg.get("seq_len", 1024))
    micro = int(train_cfg.get("micro_batch_size", 1))
    accum = int(train_cfg.get("grad_accum_steps", 8))
    epochs = int(train_cfg.get("epochs", 0))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

    encoded: list[list[int]] = []
    dropped = 0
    for row in rows:
        ids = tokenizer.encode(row["text"], add_special_tokens=False)
        if assistant_id not in ids:
            dropped += 1
            continue
        ids = ids[:seq_len] + [eos_id]
        if assistant_id not in ids[:seq_len]:
            # Truncation cut the answer marker off; the example is unusable.
            dropped += 1
            continue
        encoded.append(ids[:seq_len] + [pad_id] * max(0, seq_len - len(ids)))
    if dropped:
        _log(f"[data] dropped {dropped} examples without a usable <|assistant|> span")
    if not encoded:
        _log("[fail] no usable SFT examples")
        return 2

    steps_cfg = int(train_cfg.get("max_steps", 0))
    per_step = micro * accum * world
    if epochs:
        max_steps = max(1, epochs * len(encoded) // per_step)
    else:
        max_steps = max(1, steps_cfg or 250)
    lr = float(train_cfg["lr"])
    warmup = max(1, int(max_steps * float(train_cfg.get("warmup_ratio", 0.1))))
    clip = float(train_cfg.get("grad_clip", 1.0))
    log_every = int(train_cfg.get("log_every", 5))

    _log(f"[model] load {resume} …")
    model = LatexForCausalLM.from_pretrained(resume, torch_dtype=torch.bfloat16)
    if train_cfg.get("gradient_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    ds_config = json.loads((ROOT / train_cfg["deepspeed_config"]).read_text(encoding="utf-8"))
    ds_config["train_micro_batch_size_per_gpu"] = micro
    ds_config["gradient_accumulation_steps"] = accum
    ds_config["train_batch_size"] = micro * accum * world
    ds_config["gradient_clipping"] = clip

    opt = AdamW(
        model.parameters(),
        lr=lr,
        betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.95))),
        eps=float(train_cfg.get("eps", 1e-8)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    model_engine, opt, _, _ = deepspeed.initialize(model=model, optimizer=opt, config=ds_config)

    out_dir = ROOT / full["checkpoint"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(train_cfg.get("seed", 42)) + 1000 * rank)

    def batch() -> "torch.Tensor":
        rows_ = [rng.choice(encoded) for _ in range(micro)]
        return torch.tensor(rows_, dtype=torch.long, device=model_engine.device)

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * step / warmup
        t = (step - warmup) / max(1, max_steps - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t)))

    ema_cfg = train_cfg.get("ema") or {}
    ema = CpuEma(model, decay=float(ema_cfg.get("decay", 0.999))) if ema_cfg.get("enabled") else None
    if ema is not None and not ema.enabled:
        ema = None

    _log(f"[sft] steps={max_steps} examples={len(encoded)} seq_len={seq_len} world={world}")
    t0 = time.time()
    grad_norms: list[float] = []
    supervised_tokens = 0
    total_tokens = 0
    last_loss = 0.0

    for step in range(1, max_steps + 1):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        step_loss = 0.0
        for _ in range(accum):
            ids = batch()
            labels = assistant_only_labels(ids, assistant_id, pad_id, eos_id)
            supervised_tokens += int((labels != -100).sum())
            total_tokens += int(labels.numel())
            out = model_engine(
                input_ids=ids,
                attention_mask=(ids != pad_id).long(),
                labels=labels,
            )
            model_engine.backward(out.loss)
            model_engine.step()
            step_loss += float(out.loss.detach().float())
        last_loss = step_loss / accum
        if not math.isfinite(last_loss):
            _log(f"[fail] non-finite loss at step {step}")
            return 1
        gn = model_engine.get_global_grad_norm()
        if gn is not None and math.isfinite(float(gn)):
            grad_norms.append(float(gn))

        if ema is not None and step % int(ema_cfg.get("every_steps", 100)) == 0:
            ema.update(model)

        if step % log_every == 0 or step == 1 or step == max_steps:
            share = supervised_tokens / max(total_tokens, 1)
            _log(
                f"step {step}/{max_steps} loss={last_loss:.4f} lr={lr_at(step):.2e} "
                f"gnorm={grad_norms[-1] if grad_norms else float('nan'):.3f} "
                f"assistant_tokens={100*share:.1f}%"
            )

    try:
        model_engine.save_16bit_model(str(out_dir), "pytorch_model")
    except Exception as e:  # noqa: BLE001
        _log(f"[warn] save_16bit_model failed: {e}")
        model_engine.save_checkpoint(str(out_dir), tag="final")
    if ema is not None:
        ema.update(model)
        ema.save(out_dir)

    write_checkpoint_manifest(
        out_dir,
        stage=train_cfg.get("stage_name", "LÆTEX V1 SFT"),
        step=max_steps,
        tokens_seen=total_tokens,
        train_loss=last_loss,
        holdout_loss=None,
        grad_norm_p50=_p50(grad_norms),
        tokenizer_dir=tok_dir,
        config_path=cfg_path,
        dataset_manifest=data_path,
        hardware=f"{world}x gpu",
        ema=ema is not None,
        root=ROOT,
        extra={"phase": "sft", "assistant_token_share": supervised_tokens / max(total_tokens, 1)},
    )

    report = {
        "name": train_cfg.get("release_name"),
        "steps": max_steps,
        "examples": len(encoded),
        "final_loss": last_loss,
        "grad_norm_p50": _p50(grad_norms),
        "assistant_token_share": supervised_tokens / max(total_tokens, 1),
        "loss_masking": "assistant span only; system/user/pad set to -100",
        "elapsed_sec": time.time() - t0,
        "passed": math.isfinite(last_loss),
    }
    (out_dir / "sft_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _log(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
