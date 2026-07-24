#!/usr/bin/env python3
"""
NULLXES-LÆTEX Weight Genesis — Transformers-compatible Causal LM.

Creates LatexForCausalLM, applies muP init, writes HF checkpoint.

  python scripts/init_model.py --config configs/nullxes_latex_20b_v1.yaml \\
      --dtype bfloat16 --smoke-device cpu \\
      --holdout-jsonl datasets/latex_v1/holdout/multilingual/shard_0000.jsonl

Does NOT train. Train stack: requirements-train.txt + image torch (cu124/cu128).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Defaults align with configs/nullxes_latex_20b_v1.yaml
DEFAULT_CKPT = "checkpoints/nullxes-latex-7b"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _measure_ce(model, ids) -> float:
    """Next-token cross entropy in fp32.

    The model computes its loss in the parameter dtype; over a 131k vocab that
    is too coarse to compare against ln(V) with a 0.15 band.
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        logits = model(input_ids=ids).logits.float()
    shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1])
    shift_labels = ids[..., 1:].reshape(-1)
    return float(F.cross_entropy(shift_logits, shift_labels))


# Margins are relative to ln(V) so the gate holds for proxy configs too.
# At the production vocab 131072 (ln = 11.78) they reproduce the 9 / 15 anchors.
LEAKAGE_MARGIN = 2.8
BROKEN_MARGIN = 3.2


def _classify_init_loss(measurements: dict, vocab_size: int, tolerance: float) -> dict:
    """Untrained weights must sit at ln(V); anything else is a defect, not noise.

    Far below the head already prefers some tokens: label leakage, tied weights
    where untied were expected, or a shape bug. Far above, the logit scale or
    the initialization itself is broken.
    """
    expected = math.log(vocab_size)
    verdicts = {}
    for name, value in measurements.items():
        if abs(value - expected) <= tolerance:
            verdicts[name] = "pass"
        elif value < expected - LEAKAGE_MARGIN:
            verdicts[name] = "leakage_suspected"
        elif value > expected + BROKEN_MARGIN:
            verdicts[name] = "broken_init"
        else:
            verdicts[name] = "off_band"
    passed = bool(verdicts) and all(v == "pass" for v in verdicts.values())
    worst = "pass"
    for v in verdicts.values():
        if v != "pass":
            worst = v
            break
    return {
        "passed": passed,
        "expected": expected,
        "tolerance": tolerance,
        "verdict": worst,
        "verdicts": verdicts,
        "measurements": measurements,
    }


def _holdout_batch(path, root, tok_artifact, vocab_size, device, torch, rows: int = 2, seq: int = 64):
    """Encode a couple of real holdout documents, or None when unavailable."""
    if path is None:
        return None
    p = path if path.is_absolute() else root / path
    if not p.is_file():
        _log(f"[warn] holdout shard not found: {p}")
        return None

    from latex import LatexTokenizer

    sp_model = tok_artifact / "tokenizer.model"
    if not sp_model.is_file():
        return None
    tok = LatexTokenizer(
        vocab_file=str(sp_model),
        special_tokens_map_file=str(tok_artifact / "special_tokens.json"),
    )
    batch = []
    with p.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                text = (json.loads(line).get("text") or "").strip()
            except json.JSONDecodeError:
                continue
            ids = [i for i in tok.encode(text, add_special_tokens=False) if 0 <= i < vocab_size]
            if len(ids) >= seq:
                batch.append(ids[:seq])
            if len(batch) >= rows:
                break
    if not batch:
        _log("[warn] holdout shard had no document long enough to measure")
        return None
    return torch.tensor(batch, dtype=torch.long, device=device)


def main() -> int:
    p = argparse.ArgumentParser(description="NULLXES-LÆTEX Weight Genesis (HF CausalLM)")
    p.add_argument("--config", default="configs/nullxes_latex_20b_v1.yaml")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="Parameter dtype for checkpoint (default: bfloat16)",
    )
    p.add_argument(
        "--smoke-device",
        default="cpu",
        help="cpu | cuda — only forward smoke; weights always saved from CPU",
    )
    # Back-compat: --device maps to smoke-device
    p.add_argument("--device", default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--allow-smoke-tokenizer",
        action="store_true",
        help="Init against a smoke tokenizer artifact (pipeline test only)",
    )
    p.add_argument(
        "--holdout-jsonl",
        default="",
        help="Optional holdout shard — init loss is also measured on real tokens",
    )
    args = p.parse_args()
    if args.device is not None:
        args.smoke_device = args.device

    try:
        import torch
        import yaml
        from transformers import GenerationConfig
    except ImportError as e:
        print(
            "Missing Stage1 deps.\n"
            "  # RunPod image torch 2.8+cu128 preferred; restore only if needed:\n"
            "  pip install -r requirements-torch-cu128.txt "
            "--index-url https://download.pytorch.org/whl/cu128\n"
            "  pip install -r requirements-train.txt\n",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    import latex  # registers Auto*
    from latex import LatexConfig, LatexForCausalLM, LatexTokenizer
    from latex.models.init_weights import apply_mup_init

    cfg_path = ROOT / args.config
    with cfg_path.open(encoding="utf-8") as f:
        full = yaml.safe_load(f)

    config = LatexConfig.from_yaml_model_section(full["model"])
    config.architectures = ["LatexForCausalLM"]
    config.model_type = "latex"

    init_cfg = full.get("init") or {}
    tok_cfg = full.get("tokenizer") or {}
    ckpt_cfg = full.get("checkpoint") or {}
    out_dir = ROOT / ckpt_cfg.get("output_dir", DEFAULT_CKPT)
    out_dir.mkdir(parents=True, exist_ok=True)

    from latex_tokenizer.gate import TokenizerGateError, check_artifact, require_trainable

    tok_artifact = ROOT / tok_cfg.get("artifact_dir", "tokenizer/latex-v1")
    expected_vocab = int(tok_cfg.get("vocab_size", config.vocab_size))
    if args.allow_smoke_tokenizer:
        tok_gate = check_artifact(tok_artifact, expected_vocab)
        _log(f"[tok-gate] bypassed (smoke allowed): errors={tok_gate['errors']}")
    else:
        try:
            tok_gate = require_trainable(tok_artifact, expected_vocab)
        except TokenizerGateError as e:
            print(f"[fail] {e}", file=sys.stderr)
            return 2

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    smoke_device = torch.device(args.smoke_device)

    if smoke_device.type == "cuda":
        if not torch.cuda.is_available():
            print(
                "[fail] CUDA smoke requested but torch.cuda.is_available() is False.\n"
                "  pip uninstall -y torch torchvision torchaudio\n"
                "  pip install -r requirements-torch-cu128.txt "
                "--index-url https://download.pytorch.org/whl/cu128\n",
                file=sys.stderr,
            )
            return 2
        _log(f"[cuda] {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

    # Build + init on CPU so save_pretrained does not stall on GPU→CPU copies.
    _log("[1/5] construct LatexForCausalLM on CPU …")
    model = LatexForCausalLM(config)
    _log(f"[2/5] cast to {args.dtype} on CPU …")
    model = model.to(device="cpu", dtype=dtype)
    _log("[3/5] apply muP init …")
    mup_meta = apply_mup_init(model, init_cfg)

    nan_detected = False
    weight_stds = []
    dead = 0
    with torch.no_grad():
        for _, p in model.named_parameters():
            if not torch.isfinite(p).all():
                nan_detected = True
            if p.ndim >= 2:
                s = float(p.float().std().item())
                weight_stds.append(s)
                if s < 1e-8:
                    dead += 1
    mean_std = sum(weight_stds) / max(len(weight_stds), 1)
    n_params = sum(p.numel() for p in model.parameters())
    _log(f"    params={n_params / 1e9:.3f}B mean_std={mean_std:.6f}")

    smoke_ok, smoke_err = True, None
    loss_gate = {"passed": False, "measurements": {}}
    _log(f"[4/5] forward smoke + init-loss gate on {smoke_device} …")
    try:
        if smoke_device.type == "cuda":
            model = model.to(smoke_device)
        ids = torch.randint(0, config.vocab_size, (2, 32), device=smoke_device)
        out = model(input_ids=ids, labels=ids)
        if out.logits is None or not torch.isfinite(out.logits).all():
            smoke_ok, smoke_err = False, "non-finite logits"
        if out.loss is not None and not torch.isfinite(out.loss):
            smoke_ok, smoke_err = False, "non-finite loss"

        if smoke_ok:
            tolerance = float(init_cfg.get("loss_tolerance", 0.15))
            measurements = {
                "random_ids": _measure_ce(model, ids),
            }
            holdout_ids = _holdout_batch(
                Path(args.holdout_jsonl) if args.holdout_jsonl else None,
                ROOT,
                tok_artifact,
                config.vocab_size,
                smoke_device,
                torch,
            )
            if holdout_ids is not None:
                measurements["holdout"] = _measure_ce(model, holdout_ids)
            loss_gate = _classify_init_loss(measurements, config.vocab_size, tolerance)
            for name, value in measurements.items():
                _log(f"    init loss [{name}] = {value:.4f}")
            _log(
                f"    expected ln({config.vocab_size}) = {loss_gate['expected']:.4f} "
                f"→ {loss_gate['verdict']}"
            )
    except Exception as e:  # noqa: BLE001
        smoke_ok, smoke_err = False, str(e)
    finally:
        if next(model.parameters()).device.type == "cuda":
            _log("    moving weights back to CPU for save …")
            model = model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _log(f"[5/5] save_pretrained → {out_dir.relative_to(ROOT)} (CPU, sharded) …")
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")
    config.save_pretrained(out_dir)

    gen_cfg = GenerationConfig(
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
    )
    gen_cfg.save_pretrained(out_dir)

    specials = ROOT / tok_cfg.get(
        "special_tokens_file", "tokenizer/latex-v1/special_tokens.json"
    )
    art = ROOT / tok_cfg.get("artifact_dir", "tokenizer/latex-v1")
    sp_model = art / "tokenizer.model"
    tok = LatexTokenizer(
        vocab_file=str(sp_model) if sp_model.is_file() else None,
        special_tokens_map_file=str(specials) if specials.is_file() else None,
    )
    tok.save_pretrained(out_dir)
    if specials.is_file():
        shutil.copy2(specials, out_dir / "special_tokens.json")
    if sp_model.is_file():
        shutil.copy2(sp_model, out_dir / "tokenizer.model")

    emb = model.get_input_embeddings().weight
    report = {
        "model_class": "LatexForCausalLM",
        "model_type": "latex",
        "transformers_compatible": True,
        "parameters": f"{n_params / 1e9:.3f}B",
        "parameters_exact": n_params,
        "mean_weight_std": round(mean_std, 6),
        "dead_layers": dead,
        "nan_detected": nan_detected,
        "embedding_shape": list(emb.shape),
        "lm_head_shape": list(model.lm_head.weight.shape),
        "smoke_forward_ok": smoke_ok,
        "smoke_error": smoke_err,
        "init_loss_gate": loss_gate,
        "tokenizer_gate": tok_gate,
        "mup": mup_meta,
        "dtype": args.dtype,
        "smoke_device": args.smoke_device,
        "save_device": "cpu",
        "config_source": str(cfg_path),
        "checkpoint_dir": str(out_dir.relative_to(ROOT)),
        "save_pretrained": True,
        "passed": (not nan_detected) and smoke_ok and dead == 0 and loss_gate["passed"],
    }
    (out_dir / "init_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _log(json.dumps(report, indent=2))
    _log(f"[Weight Genesis] HF checkpoint → {out_dir}")
    _log(f"Next: python scripts/smoke_hf_causal.py --checkpoint {out_dir.relative_to(ROOT)}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
