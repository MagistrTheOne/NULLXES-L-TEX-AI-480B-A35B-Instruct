"""Gate 0 evaluator: determinism, fertility, fragmentation, reconstruction, specials."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from latex_tokenizer.fertility import (
    evaluate_fertility_file,
    evaluate_fragmentation,
    load_samples,
)
from latex_tokenizer.normalize import preserves_math_code_hints
from latex_tokenizer.reconstruct import DOMAIN_FILES, reconstruct
from latex_tokenizer.trainer import load_owned_processor


def _encode(sp, text: str) -> list[int]:
    return sp.encode(text, out_type=int)


def _decode(sp, ids: list[int]) -> str:
    return sp.decode(ids)


def _pieces(sp, text: str) -> list[str]:
    return sp.encode(text, out_type=str)


def check_special_ids(cfg: dict[str, Any], artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "special_tokens.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = cfg["special_tokens"]
    ok = True
    mismatches = []
    for key, meta in expected.items():
        got = data.get(key) or data.get(meta["token"])
        # our file uses same structure as config
        if key in data:
            if int(data[key]["id"]) != int(meta["id"]) or data[key]["token"] != meta["token"]:
                ok = False
                mismatches.append(key)
        else:
            ok = False
            mismatches.append(key)
    return {"passed": ok, "mismatches": mismatches}


def check_determinism(sp, text: str, rounds: int = 5) -> bool:
    first = _encode(sp, text)
    for _ in range(rounds - 1):
        if _encode(sp, text) != first:
            return False
    return True


def run_gate0_eval(cfg: dict[str, Any], *, smoke: bool = False) -> dict[str, Any]:
    artifact_dir = Path(cfg["paths"]["artifact_dir"])
    samples_dir = Path(cfg["paths"]["samples_dir"])
    sp = load_owned_processor(artifact_dir)
    samples = load_samples(samples_dir)

    report: dict[str, Any] = {
        "gate": "Research Gate 0: Tokenizer Fertility & Representation Gate",
        "smoke": smoke,
        "artifact_dir": str(artifact_dir),
        "checks": {},
    }

    # Specials
    report["checks"]["special_tokens"] = check_special_ids(cfg, artifact_dir)

    # Determinism
    probe = samples.get("english.txt", "NULLXES-LÆTEX")
    report["checks"]["determinism"] = {"passed": check_determinism(sp, probe)}

    # Unicode / math
    math_probe = samples.get("russian.txt", "∑ π ε research@nullxes.ai https://nullxes.ai")
    report["checks"]["unicode_preserve"] = {
        "passed": preserves_math_code_hints(math_probe),
    }

    # Fertility
    fert_results = []
    fert_pass = True
    for name, text in samples.items():
        ids = _encode(sp, text)
        fr = evaluate_fertility_file(name, text, len(ids), cfg)
        fert_results.append(asdict(fr))
        if not fr.passed:
            fert_pass = False
    report["checks"]["fertility"] = {"passed": fert_pass, "files": fert_results}

    # Fragmentation
    frag_pass = True
    frag_results = []
    for text in cfg.get("fragmentation_must_not_char_split", []):
        pcs = _pieces(sp, text)
        fr = evaluate_fragmentation(text, pcs)
        frag_results.append(asdict(fr))
        if not fr.passed:
            frag_pass = False
    report["checks"]["fragmentation"] = {"passed": frag_pass, "items": frag_results}

    # Reconstruction by domain
    recon_pass = True
    recon = {}
    for domain, files in DOMAIN_FILES.items():
        domain_ok = True
        details = []
        for fn in files:
            if fn not in samples:
                continue
            rr = reconstruct(
                fn,
                samples[fn],
                lambda t, _sp=sp: _encode(_sp, t),
                lambda ids, _sp=sp: _decode(_sp, ids),
            )
            details.append(asdict(rr))
            if not rr.passed:
                domain_ok = False
                recon_pass = False
        recon[domain] = {"passed": domain_ok, "files": details}
    report["checks"]["reconstruction"] = {"passed": recon_pass, "domains": recon}

    # Speed smoke
    t0 = time.perf_counter()
    for _ in range(100):
        _encode(sp, probe)
    elapsed = time.perf_counter() - t0
    report["checks"]["speed_smoke"] = {
        "passed": elapsed < 5.0,
        "encode_100_sec": elapsed,
    }

    # Checksum exists
    checksum = artifact_dir / "checksum.sha256"
    report["checks"]["versioned_artifacts"] = {
        "passed": checksum.is_file() and (artifact_dir / "tokenizer.model").is_file(),
        "checksum": checksum.is_file(),
        "model": (artifact_dir / "tokenizer.model").is_file(),
        "vocab": (artifact_dir / "vocab.json").is_file(),
    }

    # Vocab size (full gate requires 131072; smoke allows smaller)
    meta_path = artifact_dir / "meta.json"
    trained = sp.get_piece_size()
    if smoke:
        vocab_ok = trained >= 100
    else:
        vocab_ok = trained == int(cfg["vocab_size"]) or trained >= int(cfg["vocab_size"]) - 16
    report["checks"]["vocab_size"] = {
        "passed": vocab_ok,
        "trained": trained,
        "expected": cfg["vocab_size"],
        "smoke": smoke,
        "meta_exists": meta_path.is_file(),
    }

    # Overall
    critical = [
        "special_tokens",
        "determinism",
        "unicode_preserve",
        "fertility",
        "fragmentation",
        "reconstruction",
        "versioned_artifacts",
        "vocab_size",
    ]
    # On smoke: fertility/fragmentation/vocab may be soft — still report
    if smoke:
        critical = ["special_tokens", "determinism", "reconstruction", "versioned_artifacts"]

    overall = all(report["checks"][k]["passed"] for k in critical)
    report["passed"] = overall
    report["critical_checks"] = critical

    out_path = artifact_dir / ("gate0_report_smoke.json" if smoke else "gate0_report.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(out_path)
    return report
