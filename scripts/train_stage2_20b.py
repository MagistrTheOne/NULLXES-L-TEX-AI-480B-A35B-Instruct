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
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _log(msg: str) -> None:
    print(msg, flush=True)


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
    from latex import LatexConfig, LatexForCausalLM, LatexTokenizer
    from train_stage0a import load_corpus, _is_soft_identity  # reuse Stage0a data plane

    cfg_path = ROOT / args.config
    with cfg_path.open(encoding="utf-8") as f:
        full = yaml.safe_load(f)
    train_cfg = full["training"]
    if not train_cfg.get("enabled", False):
        _log("[fail] training.enabled is false")
        return 2

    tok_dir = ROOT / full["tokenizer"]["artifact_dir"]
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

    seq_len = int(train_cfg.get("seq_len", 2048))
    micro = int(train_cfg.get("micro_batch_size", 1))
    accum = int(train_cfg.get("grad_accum_steps", 16))
    tokens_per_step = seq_len * micro * accum
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
    ds_config["train_batch_size"] = micro * accum
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
    rng = random.Random(42)
    pool = (base + soft) if (base or soft) else list(mantra)

    def encode_batch() -> tuple:
        batch_ids = []
        kinds = []
        while len(batch_ids) < micro:
            r = rng.random()
            if mantra and r < mantra_mix:
                text, kind = rng.choice(mantra), "mantra"
            elif soft and r < mantra_mix + soft_id_mix:
                text, kind = rng.choice(soft), "soft"
            else:
                text = rng.choice(pool)
                kind = "soft" if _is_soft_identity(text) else "base"
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < 8:
                continue
            if len(ids) < seq_len:
                ids = ids + [0] * (seq_len - len(ids))
            else:
                start = rng.randint(0, len(ids) - seq_len)
                ids = ids[start : start + seq_len]
            batch_ids.append(ids)
            kinds.append(kind)
        t = torch.tensor(batch_ids, dtype=torch.long, device=model_engine.device)
        return t, kinds

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * step / warmup
        # simple cosine to 10%
        t = (step - warmup) / max(1, max_steps - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t)))

    t0 = time.time()
    tokens_seen = 0
    mantra_n = soft_n = 0
    last_loss = 0.0

    for step in range(1, max_steps + 1):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        ids, kinds = encode_batch()
        mantra_n += sum(1 for k in kinds if k == "mantra")
        soft_n += sum(1 for k in kinds if k == "soft")
        w = mantra_loss_w if "mantra" in kinds else (id_loss_w if "soft" in kinds else 1.0)
        out = model_engine(input_ids=ids, labels=ids)
        loss = out.loss * w
        model_engine.backward(loss)
        model_engine.step()
        last_loss = float(loss.detach().float().item())
        tokens_seen += tokens_per_step

        if step % log_every == 0 or step == 1 or step == max_steps:
            elapsed = max(1e-6, time.time() - t0)
            tps = tokens_seen / elapsed
            _log(
                f"step {step}/{max_steps} loss={last_loss:.4f} "
                f"lr={lr_at(step):.2e} tokens={tokens_seen/1e6:.2f}M "
                f"tps={tps:.0f} mantra%={100*mantra_n/(step*micro):.1f} "
                f"soft%={100*soft_n/(step*micro):.1f}"
            )

        if step % int(full["checkpoint"].get("save_every_steps", 50)) == 0:
            tag = f"step{step}"
            model_engine.save_checkpoint(str(out_dir), tag=tag)
            _log(f"[ckpt] deepspeed → {out_dir}/{tag}")

    # Final HF-ish export (16-bit gather when ZeRO-3)
    try:
        model_engine.save_16bit_model(str(out_dir), "pytorch_model")
        _log(f"[save] 16bit weights → {out_dir}")
    except Exception as e:  # noqa: BLE001
        _log(f"[warn] save_16bit_model failed: {e} — ds checkpoint still on disk")
        model_engine.save_checkpoint(str(out_dir), tag="final")

    report = {
        "name": train_cfg.get("release_name"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps": max_steps,
        "tokens_seen": tokens_seen,
        "final_loss": last_loss,
        "checkpoint_dir": str(out_dir.relative_to(ROOT)),
        "resume_from": str(resume.relative_to(ROOT)),
        "deepspeed": str(ds_path.relative_to(ROOT)),
        "passed": math.isfinite(last_loss),
    }
    (out_dir / "train_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _log(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
