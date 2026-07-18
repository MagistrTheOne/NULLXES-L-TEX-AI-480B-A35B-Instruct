#!/usr/bin/env python3
"""
NULLXES-LÆTEX Stage0a bootstrap pretrain (~100M).

First TRAINED brain — not 7B SFT on random weights.

  python scripts/train_stage0a.py --config configs/stage0a_100m_bootstrap.yaml
  python scripts/train_stage0a.py --config local_2080/configs/latex_50m_2080_identity.yaml
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


def _is_soft_identity(text: str) -> bool:
    low = text.lower()
    return any(
        k in low
        for k in (
            "nullxes-lætex",
            "nullxes-latex",
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


def _is_mantra(text: str, source_hint: str = "") -> bool:
    if "identity_mantra" in source_hint or "sft_identity" in source_hint:
        return True
    low = text.lower()
    # chat or Q/A mantra shapes
    if "<|assistant|>" in low and ("who are you" in low or "как тебя зовут" in low or "кто ты?" in low):
        return True
    if text.startswith("Q:") and "\nA:" in text and ("lætex" in low or "nullxes" in low):
        return True
    return False


def load_corpus(
    manifest_path: Path,
    repo: Path,
    identity_upsample: int,
) -> tuple[list[str], list[str], list[str]]:
    """Return (base_texts, soft_identity_texts, mantra_texts)."""
    from latex_tokenizer.corpus import iter_jsonl_shard

    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    base: list[str] = []
    soft: list[str] = []
    mantra: list[str] = []
    seen_mantra: set[str] = set()

    shards = man.get("shards") or {}
    for bucket, meta in shards.items():
        for rel in meta.get("files") or []:
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
                    is_m = bool(obj.get("identity_mantra")) or _is_mantra(t, src)
                    if is_m:
                        if t not in seen_mantra:
                            seen_mantra.add(t)
                            mantra.append(t)
                        continue
                    if _is_soft_identity(t):
                        soft.append(t)
                        for _ in range(max(0, identity_upsample - 1)):
                            soft.append(t)
                    else:
                        base.append(t)

    # Also pull dedicated mantra file if present but not in manifest yet
    extra = repo / "datasets/raw/shards/identity/identity_mantra.jsonl"
    if extra.is_file():
        for t in iter_jsonl_shard(extra):
            t = t.strip()
            if len(t) >= 32 and t not in seen_mantra and _is_mantra(t, "identity_mantra"):
                seen_mantra.add(t)
                mantra.append(t)

    if not base and not soft and not mantra:
        raise FileNotFoundError(f"No texts in {manifest_path}")
    return base, soft, mantra


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage0a_100m_bootstrap.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-steps", type=int, default=0, help="0 = derive from tokens_target")
    p.add_argument("--resume", default="", help="override training.resume_from")
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
        _log("[fail] missing tokenizer — run train_tokenizer.py")
        return 2

    man_rel = train_cfg.get("corpus_manifest", "datasets/manifests/pretrain_stage0.json")
    identity_upsample = int(train_cfg.get("identity_upsample", 4))
    mantra_mix = float(train_cfg.get("mantra_mix", 0.0))  # fraction of microbatches
    soft_id_mix = float(train_cfg.get("soft_identity_mix", 0.0))  # extra soft id chance
    id_loss_w = float(train_cfg.get("identity_loss_weight", 1.0))
    mantra_loss_w = float(train_cfg.get("mantra_loss_weight", 1.5))

    base, soft, mantra = load_corpus(ROOT / man_rel, ROOT, identity_upsample)
    _log(
        f"[data] base={len(base)} soft_id={len(soft)} mantra_unique={len(mantra)} "
        f"mantra_mix={mantra_mix:.3f} soft_mix={soft_id_mix:.3f} "
        f"loss_w id={id_loss_w} mantra={mantra_loss_w}"
    )
    if mantra_mix > 0 and not mantra:
        _log("[warn] mantra_mix>0 but no mantra docs — build_identity_corpus.py first")

    config = LatexConfig.from_yaml_model_section(full["model"])
    config.architectures = ["LatexForCausalLM"]
    config.model_type = "latex"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    precision = str(train_cfg.get("precision", "fp32")).lower()
    use_amp = False
    amp_dtype = torch.float16
    if device.type == "cuda":
        if precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float32
            use_amp = True
            amp_dtype = torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = torch.float32

    resume = args.resume or train_cfg.get("resume_from") or ""
    resume_path = ROOT / resume if resume else None

    if resume_path and (resume_path / "model.safetensors").is_file():
        _log(f"[model] resume from {resume_path} …")
        model = LatexForCausalLM.from_pretrained(resume_path)
        model = model.to(device=device, dtype=dtype)
    else:
        _log(f"[model] construct on {device} dtype={dtype} precision={precision} amp={use_amp} …")
        model = LatexForCausalLM(config)
        apply_mup_init(model, full.get("init") or {})
        model = model.to(device=device, dtype=dtype)

    if train_cfg.get("gradient_checkpointing"):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            _log("[model] gradient_checkpointing=ON")
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = ROOT / full["checkpoint"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    pool = base + soft
    if not pool:
        pool = mantra[:]

    def encode_batch() -> tuple[torch.Tensor, list[str]]:
        batch_ids = []
        kinds: list[str] = []
        while len(batch_ids) < micro:
            # re-pick with proper kind tagging
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
                ids = ids + [tokenizer.pad_token_id or 0] * (seq_len - len(ids))
            else:
                # for short mantras prefer start; for long docs random crop
                if kind == "mantra" or len(ids) <= seq_len:
                    ids = ids[:seq_len]
                else:
                    start = rng.randint(0, max(0, len(ids) - seq_len))
                    ids = ids[start : start + seq_len]
            batch_ids.append(ids[:seq_len])
            kinds.append(kind)
        return torch.tensor(batch_ids, dtype=torch.long, device=device), kinds

    def lr_at(step: int) -> float:
        if step < warmup:
            return lr * (step + 1) / warmup
        decay_start = int(max_steps * 0.8)
        if step < decay_start:
            return lr
        t = (step - decay_start) / max(1, max_steps - decay_start)
        return lr * (1.0 - 0.9 * t)

    def kind_weight(kinds: list[str]) -> float:
        # micro_batch usually 1; average if >1
        ws = []
        for k in kinds:
            if k == "mantra":
                ws.append(mantra_loss_w)
            elif k == "soft":
                ws.append(id_loss_w)
            else:
                ws.append(1.0)
        return sum(ws) / max(len(ws), 1)

    _log(
        f"[train] steps={max_steps} tokens/step≈{tokens_per_step} "
        f"target≈{max_steps * tokens_per_step / 1e6:.1f}M tokens"
    )
    t0 = time.time()
    tokens_seen = 0
    opt.zero_grad(set_to_none=True)
    last_loss = None
    mantra_hits = 0
    soft_hits = 0
    micro_total = 0

    for step in range(1, max_steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss_acc = 0.0
        for _ in range(accum):
            input_ids, kinds = encode_batch()
            micro_total += len(kinds)
            mantra_hits += sum(1 for k in kinds if k == "mantra")
            soft_hits += sum(1 for k in kinds if k == "soft")
            w = kind_weight(kinds)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(input_ids=input_ids, labels=input_ids)
                loss = (out.loss * w) / accum
            if not torch.isfinite(loss):
                _log(f"[fail] non-finite loss at step {step}")
                return 1
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_acc += float(loss.detach().item()) * accum
        if use_amp:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if use_amp:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
        tokens_seen += tokens_per_step
        last_loss = loss_acc / accum

        if step % log_every == 0 or step == 1 or step == max_steps:
            elapsed = time.time() - t0
            tps = tokens_seen / max(elapsed, 1e-6)
            m_rate = mantra_hits / max(micro_total, 1)
            s_rate = soft_hits / max(micro_total, 1)
            _log(
                f"step {step}/{max_steps} loss={last_loss:.4f} "
                f"lr={lr_at(step):.2e} tokens={tokens_seen/1e6:.2f}M tps={tps:.0f} "
                f"mantra%={100*m_rate:.1f} soft%={100*s_rate:.1f}"
            )

        if step % int(full["checkpoint"].get("save_every_steps", 200)) == 0:
            model.save_pretrained(out_dir, safe_serialization=True)
            config.save_pretrained(out_dir)

    model.cpu()
    model.save_pretrained(out_dir, safe_serialization=True)
    config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    for name in ("tokenizer.model", "special_tokens.json", "vocab.json", "meta.json"):
        src = tok_dir / name
        if src.is_file():
            (out_dir / name).write_bytes(src.read_bytes())

    report = {
        "name": train_cfg.get("release_name") or full["model"].get("name") or "NULLXES-LÆTEX-Stage0a",
        "model_class": "LatexForCausalLM",
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps": max_steps,
        "tokens_seen": tokens_seen,
        "final_loss": last_loss,
        "checkpoint_dir": str(out_dir.relative_to(ROOT)),
        "precision": precision,
        "resume_from": resume or None,
        "mantra_mix": mantra_mix,
        "mantra_loss_weight": mantra_loss_w,
        "identity_loss_weight": id_loss_w,
        "mantra_rate_observed": mantra_hits / max(micro_total, 1),
        "identity_note": "Hard mantra Q/A + soft identity + loss weighting",
        "passed": last_loss is not None and math.isfinite(last_loss),
    }
    (out_dir / "train_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _log(json.dumps(report, indent=2))
    _log(f"[done] HF checkpoint → {out_dir}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
