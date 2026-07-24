#!/usr/bin/env python3
"""
Stage2 20B train — DeepSpeed ZeRO-3 + CPU offload (1× RTX PRO 6000).

  deepspeed --num_gpus=1 scripts/train_stage2_20b.py \
    --config configs/stage2_20b_rtx_pro_6000_100m.yaml

Expect ~100M tokens then stop. Identity mix stays low (≤~1.5%).
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage2_20b_rtx_pro_6000_100m.yaml")
    p.add_argument("--local_rank", type=int, default=-1)
    args = p.parse_args()

    import torch
    import yaml
    import deepspeed
    from torch.optim import AdamW

    import latex  # noqa: F401
    from latex import LatexForCausalLM, LatexTokenizer
    from latex_data.batching import (
        BASE,
        MANTRA,
        SOFT,
        DocSampler,
        SequencePacker,
        mask_labels,
    )
    from latex_data.telemetry import CpuEma, write_checkpoint_manifest
    from train_stage0a import load_corpus, _is_soft_identity  # reuse Stage0a data plane

    cfg_path = ROOT / args.config
    with cfg_path.open(encoding="utf-8") as f:
        full = yaml.safe_load(f)
    train_cfg = full["training"]
    if not train_cfg.get("enabled", False):
        _log("[fail] training.enabled is false")
        return 2

    tok_dir = ROOT / full["tokenizer"]["artifact_dir"]
    from latex_tokenizer.gate import TokenizerGateError, require_trainable

    try:
        require_trainable(tok_dir, int(full["tokenizer"].get("vocab_size", 131072)))
    except TokenizerGateError as e:
        _log(f"[fail] {e}")
        return 2

    man_rel = train_cfg["corpus_manifest"]
    identity_upsample = int(train_cfg.get("identity_upsample", 1))
    mantra_mix = float(train_cfg.get("mantra_mix", 0.005))
    soft_id_mix = float(train_cfg.get("soft_identity_mix", 0.01))
    id_loss_w = float(train_cfg.get("identity_loss_weight", 1.2))
    mantra_loss_w = float(train_cfg.get("mantra_loss_weight", 1.5))

    base, soft, mantra = load_corpus(ROOT / man_rel, ROOT, identity_upsample)
    _log(
        f"[data] base={len(base)} soft={len(soft)} mantra={len(mantra)} "
        f"mantra_mix={mantra_mix} soft_mix={soft_id_mix}"
    )
    if not base and not soft and not mantra:
        _log("[fail] empty corpus — build identity + Gate A proxy first")
        return 2

    resume = ROOT / train_cfg["resume_from"]
    if not resume.is_dir():
        _log(f"[fail] missing resume_from={resume}")
        return 2

    _log(f"[model] load genesis {resume} …")
    model = LatexForCausalLM.from_pretrained(resume, torch_dtype=torch.bfloat16)
    if train_cfg.get("gradient_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        _log("[model] gradient_checkpointing=ON")

    tokenizer = LatexTokenizer(
        vocab_file=str(tok_dir / "tokenizer.model"),
        special_tokens_map_file=str(tok_dir / "special_tokens.json"),
    )

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

    seq_len = int(train_cfg.get("seq_len", 2048))
    micro = int(train_cfg.get("micro_batch_size", 1))
    accum = int(train_cfg.get("grad_accum_steps", 16))
    # Global counts: every data-parallel rank consumes its own micro-batch.
    tokens_per_micro = seq_len * micro * world
    tokens_per_step = tokens_per_micro * accum
    tokens_target = int(train_cfg["tokens_target"])
    max_steps = max(1, tokens_target // tokens_per_step)
    lr = float(train_cfg["lr"])
    warmup = max(1, int(max_steps * float(train_cfg.get("warmup_ratio", 0.05))))
    log_every = int(train_cfg.get("log_every", 5))
    clip = float(train_cfg.get("grad_clip", 1.0))

    ds_path = ROOT / train_cfg["deepspeed_config"]
    ds_config = json.loads(ds_path.read_text(encoding="utf-8"))
    ds_config["train_micro_batch_size_per_gpu"] = micro
    ds_config["gradient_accumulation_steps"] = accum
    # DeepSpeed requires train_batch_size == micro * gas * data_parallel_world
    ds_config["train_batch_size"] = micro * accum * world
    ds_config["gradient_clipping"] = clip

    opt = AdamW(
        model.parameters(),
        lr=lr,
        betas=(float(train_cfg.get("beta1", 0.9)), float(train_cfg.get("beta2", 0.95))),
        eps=float(train_cfg.get("eps", 1e-8)),
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
    )

    model_engine, opt, _, _ = deepspeed.initialize(
        model=model,
        optimizer=opt,
        config=ds_config,
    )
    _log(f"[ds] zero={ds_config['zero_optimization']['stage']} steps={max_steps} tok/step≈{tokens_per_step}")

    out_dir = ROOT / full["checkpoint"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_name = train_cfg.get("stage_name") or full.get("stage") or "latex-v1-stage"
    hw = full.get("hardware") or {}
    cluster = hw.get("cluster") or {}
    hardware = f"{cluster.get('nodes', 1)} node x {cluster.get('gpus', world)}x {hw.get('gpu', 'unknown')}"
    # Per-rank offset: with a shared seed every rank samples the same documents,
    # so a 4-GPU run would train on one batch four times instead of four batches.
    rng = random.Random(int(train_cfg.get("seed", 42)) + 1000 * rank)
    pad_id = tokenizer.pad_token_id or 0
    sampler = DocSampler(
        base,
        soft,
        mantra,
        mantra_mix=mantra_mix,
        soft_mix=soft_id_mix,
        rng=rng,
        is_soft=_is_soft_identity,
    )
    packer = SequencePacker(
        sampler,
        lambda t: tokenizer.encode(t, add_special_tokens=False),
        seq_len=seq_len,
        eos_id=tokenizer.eos_token_id or 3,
        kind_weights={BASE: 1.0, SOFT: id_loss_w, MANTRA: mantra_loss_w},
    )

    def encode_batch() -> tuple:
        rows, weight = packer.batch(micro)
        t = torch.tensor(rows, dtype=torch.long, device=model_engine.device)
        return t, weight

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * step / warmup
        # simple cosine to 10%
        t = (step - warmup) / max(1, max_steps - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t)))

    ema_cfg = train_cfg.get("ema") or {}
    ema = None
    if ema_cfg.get("enabled"):
        ema = CpuEma(model, decay=float(ema_cfg.get("decay", 0.999)))
        if ema.enabled:
            _log(f"[ema] cpu shadow ready decay={ema.decay} every={ema_cfg.get('every_steps', 100)}")
        else:
            _log("[ema] disabled — parameters are sharded (ZeRO-3), shadow would be empty")
            ema = None
    ema_every = int(ema_cfg.get("every_steps", 100))

    t0 = time.time()
    tokens_seen = 0
    last_loss = 0.0
    grad_norms: list[float] = []

    for step in range(1, max_steps + 1):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        # One optimizer step = `accum` micro-batches. DeepSpeed applies the
        # update only on the accumulation boundary and scales the loss itself.
        step_loss = 0.0
        for _ in range(accum):
            ids, w = encode_batch()
            labels = mask_labels(ids, pad_id)
            out = model_engine(input_ids=ids, labels=labels)
            loss = out.loss * w
            model_engine.backward(loss)
            model_engine.step()
            step_loss += float(loss.detach().float().item())
            tokens_seen += tokens_per_micro
        last_loss = step_loss / accum
        grad_norm = model_engine.get_global_grad_norm()
        grad_norm = float(grad_norm) if grad_norm is not None else float("nan")
        if math.isfinite(grad_norm):
            grad_norms.append(grad_norm)

        if not math.isfinite(last_loss):
            _log(f"[fail] non-finite loss at step {step}")
            return 1

        if step % log_every == 0 or step == 1 or step == max_steps:
            elapsed = max(1e-6, time.time() - t0)
            tps = tokens_seen / elapsed
            rates = packer.kind_rates()
            _log(
                f"step {step}/{max_steps} loss={last_loss:.4f} "
                f"lr={lr_at(step):.2e} tokens={tokens_seen/1e6:.2f}M "
                f"tps={tps:.0f} gnorm={grad_norm:.3f} "
                f"mantra%={100*rates.get(MANTRA, 0.0):.1f} "
                f"soft%={100*rates.get(SOFT, 0.0):.1f}"
            )
            if math.isfinite(grad_norm) and grad_norm >= clip * 10:
                _log(f"[warn] grad_norm {grad_norm:.2f} >= 10x clip — instability ahead")

        if ema is not None and step % ema_every == 0:
            ema.update(model)

        if step % int(full["checkpoint"].get("save_every_steps", 50)) == 0:
            tag = f"step{step}"
            model_engine.save_checkpoint(str(out_dir), tag=tag)
            _log(f"[ckpt] deepspeed → {out_dir}/{tag}")
            write_checkpoint_manifest(
                out_dir,
                stage=stage_name,
                step=step,
                tokens_seen=tokens_seen,
                train_loss=last_loss,
                holdout_loss=None,
                grad_norm_p50=_p50(grad_norms),
                tokenizer_dir=tok_dir,
                config_path=cfg_path,
                dataset_manifest=ROOT / man_rel,
                hardware=hardware,
                ema=ema is not None,
                root=ROOT,
                extra={"tag": tag, "world_size": world},
            )

    # Final HF-ish export (16-bit gather when ZeRO-3)
    try:
        model_engine.save_16bit_model(str(out_dir), "pytorch_model")
        _log(f"[save] 16bit weights → {out_dir}")
    except Exception as e:  # noqa: BLE001
        _log(f"[warn] save_16bit_model failed: {e} — ds checkpoint still on disk")
        model_engine.save_checkpoint(str(out_dir), tag="final")

    if ema is not None:
        ema.update(model)
        ema_path = ema.save(out_dir)
        _log(f"[ema] saved → {ema_path} updates={ema.updates}")

    write_checkpoint_manifest(
        out_dir,
        stage=stage_name,
        step=max_steps,
        tokens_seen=tokens_seen,
        train_loss=last_loss,
        holdout_loss=None,
        grad_norm_p50=_p50(grad_norms),
        tokenizer_dir=tok_dir,
        config_path=cfg_path,
        dataset_manifest=ROOT / man_rel,
        hardware=hardware,
        ema=ema is not None,
        root=ROOT,
        extra={"tag": "final", "world_size": world},
    )

    ordered = sorted(grad_norms)
    report = {
        "name": train_cfg.get("release_name"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps": max_steps,
        "tokens_seen": tokens_seen,
        "tokens_per_step": tokens_per_step,
        "final_loss": last_loss,
        "grad_norm_p50": ordered[len(ordered) // 2] if ordered else None,
        "grad_norm_max": ordered[-1] if ordered else None,
        "token_rates_by_kind": packer.kind_rates(),
        "checkpoint_dir": str(out_dir.relative_to(ROOT)),
        "resume_from": str(resume.relative_to(ROOT)),
        "deepspeed": str(ds_path.relative_to(ROOT)),
        "world_size": world,
        "packing": "documents joined through eos; pad excluded from loss",
        "passed": math.isfinite(last_loss),
    }
    (out_dir / "train_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _log(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
