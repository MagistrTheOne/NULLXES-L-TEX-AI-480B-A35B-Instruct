#!/usr/bin/env python3
"""
QA for NULLXES-LÆTEX-100M-Stage0a checkpoint before HF upload.

  python scripts/qa_stage0a.py --checkpoint checkpoints/nullxes-latex-100m-stage0a-v0.1 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="checkpoints/nullxes-latex-100m-stage0a-v0.1",
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    ckpt = ROOT / args.checkpoint
    if not (ckpt / "model.safetensors").is_file():
        print(f"[fail] missing {ckpt / 'model.safetensors'}", file=sys.stderr)
        return 2

    import torch
    import latex  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM

    from latex import LatexTokenizer

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[load] {ckpt} → {device}", flush=True)
    config = AutoConfig.from_pretrained(ckpt, trust_remote_code=False)
    assert config.model_type == "latex", config.model_type
    model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16)
    model = model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"[ok] params={n/1e6:.1f}M class={model.__class__.__name__}", flush=True)

    tok = LatexTokenizer(
        vocab_file=str(ckpt / "tokenizer.model"),
        special_tokens_map_file=str(ckpt / "special_tokens.json"),
    )

    report = {
        "checkpoint": str(ckpt),
        "parameters": n,
        "model_type": config.model_type,
        "checks": {},
    }

    # 1) safetensors load + forward
    ids = torch.tensor([[2, 10, 11, 3]], device=device)
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)
    fin = bool(torch.isfinite(out.loss))
    report["checks"]["forward_finite"] = {"passed": fin, "loss": float(out.loss)}
    print(f"[qa] forward_finite={fin} loss={float(out.loss):.4f}", flush=True)

    # 2) generate shape
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=8, use_cache=True)
    shape_ok = gen.shape[-1] == ids.shape[-1] + 8
    report["checks"]["generate"] = {"passed": shape_ok, "shape": list(gen.shape)}
    print(f"[qa] generate shape={tuple(gen.shape)} ok={shape_ok}", flush=True)

    # 3) identity prompts (bootstrap overfit — expect LÆTEX / NULLXES)
    prompts = [
        "NULLXES-LÆTEX AI is a foundation",
        "The short name of the model is",
        "Краткое имя модели:",
        "<|system|> You are NULLXES-LÆTEX (LÆTEX), developed by NULLXES. <|user|> Who are you? <|assistant|>",
        "Who are you?",
        "What is your name?",
        "Who developed you?",
        "Кто ты?",
        "Как тебя зовут?",
    ]
    identity_hits = 0
    leak_hits = 0
    samples = []
    # Open social prompts — schema pipes here = INTERNAL leak
    leak_watch = {"Who are you?", "What is your name?", "Кто ты?", "Как тебя зовут?"}
    for prompt in prompts:
        enc = tok.encode(prompt, add_special_tokens=False)
        if not enc:
            continue
        t = torch.tensor([enc[-64:]], device=device)
        with torch.no_grad():
            g = model.generate(t, max_new_tokens=48, do_sample=False)
        text = tok.decode(g[0].tolist(), skip_special_tokens=False)
        # Score only the continuation (avoid prompt echo false positives)
        cont = text[len(prompt) :] if text.startswith(prompt) else text
        low = cont.lower()
        hit = any(
            k in low
            for k in (
                "lætex",
                "latex",
                "nullxes",
                "digital employee",
                "foundation",
            )
        )
        leak = False
        if prompt in leak_watch:
            leak = (
                "| schema" in low
                or "d_model" in low
                or "hybrid moe" in low
                or cont.count("|") >= 3
                or "<<<internal>>>" in low
            )
            leak_hits += int(leak)
        identity_hits += int(hit)
        samples.append(
            {"prompt": prompt[:80], "out": cont[-200:], "hit": hit, "leak": leak}
        )
        print(
            f"[qa] identity hit={hit} leak={leak} | {cont[-140:].replace(chr(10), ' ')}",
            flush=True,
        )

    id_ok = identity_hits >= 2
    # Informational until v0.1.2 patch; do not fail old ckpts on leak alone
    no_leak = leak_hits == 0
    report["checks"]["identity"] = {
        "passed": id_ok,
        "hits": identity_hits,
        "n_prompts": len(prompts),
        "samples": samples,
    }
    report["checks"]["output_control"] = {
        "passed": no_leak,
        "leak_hits": leak_hits,
        "note": "PUBLIC answers must not emit schema tables / INTERNAL fences",
    }

    # 4) files present
    need = [
        "config.json",
        "model.safetensors",
        "tokenizer.model",
        "special_tokens.json",
        "train_report.json",
    ]
    files_ok = all((ckpt / n).is_file() for n in need)
    report["checks"]["artifacts"] = {"passed": files_ok, "need": need}

    critical = ["forward_finite", "generate", "artifacts", "identity"]
    passed = all(report["checks"][k]["passed"] for k in critical)
    report["passed"] = passed
    report["critical"] = critical
    out_path = ckpt / "qa_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": passed,
                "identity_hits": identity_hits,
                "output_control_leaks": leak_hits,
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"report: {out_path}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
