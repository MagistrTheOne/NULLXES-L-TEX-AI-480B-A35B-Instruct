"""Build committed seed JSONL corpus + Gate0 manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from latex_data.mix import save_manifest
from latex_data.schema import BUCKETS

SITE = "https://www.nullxesdai.online/"
EMAIL = "ceo@nullxes.com"
INN = "ИНН2311391270"


def _rec(i: int, bucket: str, lang: str, text: str) -> dict[str, Any]:
    return {
        "id": f"nlx-seed-{bucket[:3]}-{lang}-{i:04d}",
        "text": text.strip(),
        "lang": lang,
        "bucket": bucket,
        "source": "nullxes_seed_v0.1",
        "license": "nullxes_internal",
        "split": "train",
    }


def multilingual_docs() -> list[dict[str, Any]]:
    docs = []
    ru = [
        f"NULLXES строит цифровых сотрудников на собственной платформе. Сайт: {SITE} Контакт: {EMAIL}. Реквизиты организации включают {INN}.",
        "Корпоративный процесс: заявка → проверка политики → эскалация человеку при риске. Workflow id: customer_support.workflow.",
        "Русский технический абзац: KubernetesOperator масштабирует сервисы, RFC-9457 описывает problem details для HTTP API.",
        "Математическая вставка в тексте: сумма x_i^2 ограничена константой; email и URL не должны ломаться нормализатором.",
        "Юридический фрагмент: стороны соглашаются с условиями поставки ПО NULLXES-LÆTEX без передачи персональных данных в веса модели.",
    ]
    en = [
        f"NULLXES Digital Employees platform. Canonical site {SITE}. Contact {EMAIL}. Tax id probe {INN}.",
        "Enterprise reasoning requires multi-document policy binding and explicit tool_call schemas.",
        "Identifier preservation probes: NULLXES-LÆTEX-AI-480B-A35B, gpt-4.1-mini, OpenAIRealtimeAPI, RFC-9457.",
        "Code and API docs must survive tokenization without punctuation shredding.",
        "Long-context workflows pack memory slots separately from trunk weights.",
    ]
    zh = [
        f"NULLXES 数字员工平台。官网 {SITE}。联系 {EMAIL}。标识符 KubernetesOperator 与 customer_support.workflow 需完整保留。",
        "中文技术文本：KubernetesOperator 与 customer_support.workflow 标识符需完整保留，避免按字符切碎。",
        "分词评估需要稳定的中英俄混合文档，而不是虚构的域名；只使用正式站点 nullxesdai.online。",
        "企业文档包含合同条款、流程编号与 API 字段名，以及合规审计所需的结构化日志字段。",
        "科学研究文本可包含公式符号与引用标记，同时保留邮箱与 URL 的连续跨度以便 reconstruction。",
    ]
    de = [
        f"NULLXES baut digitale Mitarbeiter. Website {SITE}. Kontakt {EMAIL}.",
        "Deutsche Unternehmensdokumente enthalten Workflow-IDs und Compliance-Hinweise.",
        "Tokenisierung muss Umlaute und Fachbegriffe effizient abbilden.",
        "RFC-9457 und KubernetesOperator sind Fragmentierungsproben.",
        "Keine Persona-Dialoge im Tokenizer-Korpus.",
    ]
    fr = [
        f"NULLXES construit des employés numériques. Site {SITE}. Contact {EMAIL}.",
        "Les documents d'entreprise contiennent des identifiants de workflow stables.",
        "Le tokenizer doit respecter le français et les mélanges code/texte.",
        "Les schémas JSON et OpenAPI font partie du mix code/enterprise.",
        "Interdiction des dialogues de personnalité dans le corpus seed.",
    ]
    i = 0
    for lang, block in [("ru", ru), ("en", en), ("zh", zh), ("de", de), ("fr", fr)]:
        for t in block:
            i += 1
            docs.append(_rec(i, "multilingual", lang, t))
    # pad to >=20
    while len(docs) < 20:
        i += 1
        docs.append(
            _rec(
                i,
                "multilingual",
                "en",
                f"NULLXES-LÆTEX seed multilingual pad {i}. Site {SITE}. Contact {EMAIL}. Bucket multilingual.",
            )
        )
    return docs


def code_docs() -> list[dict[str, Any]]:
    docs = []
    snippets = [
        (
            "code",
            '''def KubernetesOperator(name: str, replicas: int = 3) -> dict:
    return {"name": name, "replicas": replicas, "workflow": "customer_support.workflow"}
''',
        ),
        (
            "code",
            '''export function ping(): string {
  // NULLXES endpoint must stay intact for fertility/reconstruction tests
  return "https://www.nullxesdai.online/";
}
''',
        ),
        (
            "code",
            json.dumps(
                {
                    "tool_call": {
                        "name": "create_ticket",
                        "arguments": {
                            "workflow": "customer_support.workflow",
                            "inn": INN,
                            "site": SITE,
                        },
                    }
                },
                ensure_ascii=False,
            ),
        ),
        (
            "code",
            "openapi: 3.1.0\ninfo:\n  title: NULLXES Tool API\n  version: 0.1.0\nservers:\n  - url: https://www.nullxesdai.online/\n",
        ),
        (
            "code",
            "SELECT workflow_id, status FROM tickets WHERE workflow_id = 'customer_support.workflow' AND owner IS NOT NULL;",
        ),
    ]
    for i, (lang, t) in enumerate(snippets, 1):
        docs.append(_rec(i, "code", lang, t + f"\n# probe NULLXES-LÆTEX RFC-9457 line {i}"))
    while len(docs) < 20:
        n = len(docs) + 1
        docs.append(
            _rec(
                n,
                "code",
                "code",
                f"// seed code {n}\nconst ID = 'NULLXES-LÆTEX-AI-480B-A35B';\nconst W = 'customer_support.workflow';\n",
            )
        )
    return docs


def enterprise_docs() -> list[dict[str, Any]]:
    docs = []
    base = [
        f"ДОГОВОР поставки ПО NULLXES-LÆTEX. {INN}. Контакт {EMAIL}. Сайт {SITE}.",
        "Policy pack: human escalation required for KYC exceptions; log tool_call and tool_result.",
        "Workflow customer_support.workflow: intake → classify → resolve or escalate.",
        "Enterprise memory is external; trunk weights must not store customer secrets.",
        "Compliance note: audit trail must record identity_id and role_id separately from IEL.",
    ]
    for i, t in enumerate(base, 1):
        docs.append(_rec(i, "enterprise", "ru" if i % 2 else "en", t))
    while len(docs) < 20:
        n = len(docs) + 1
        docs.append(
            _rec(
                n,
                "enterprise",
                "en",
                f"Enterprise seed doc {n}: workflow customer_support.workflow, operator KubernetesOperator, site {SITE}.",
            )
        )
    return docs


def scientific_docs() -> list[dict[str, Any]]:
    docs = []
    base = [
        "Let S = sum_{i=1}^n x_i^2. Bound S by a constant C under Gaussian assumptions.",
        "Transformer pre-norm with RMSNorm stabilizes deep residual streams for language modeling.",
        "RoPE encodes relative position; NoPE on global layers can help abstraction at depth.",
        "Mixture-of-Experts routing uses independent sigmoid scores then top-k normalization.",
        "Мультилингальная модель требует сбалансированной fertility по алфавитам и скриптам.",
    ]
    for i, t in enumerate(base, 1):
        docs.append(_rec(i, "scientific", "en" if i < 5 else "ru", t))
    while len(docs) < 20:
        n = len(docs) + 1
        docs.append(
            _rec(
                n,
                "scientific",
                "en",
                f"Scientific seed {n}: optimize loss L(theta) with AdamW; monitor grad_norm and MFU. Symbol pi appears as π in some docs.",
            )
        )
    return docs


def synthetic_structure_docs() -> list[dict[str, Any]]:
    docs = []
    for i in range(1, 21):
        payload = {
            "tool_call": {
                "name": "lookup",
                "arguments": {
                    "workflow": "customer_support.workflow",
                    "fields": ["status", "owner"],
                    "i": i,
                },
            }
        }
        docs.append(
            _rec(
                i,
                "synthetic_structure",
                "struct",
                json.dumps(payload, ensure_ascii=False)
                + f" schema_only seed {i} site={SITE}",
            )
        )
    return docs


GENERATORS = {
    "multilingual": multilingual_docs,
    "code": code_docs,
    "enterprise": enterprise_docs,
    "scientific": scientific_docs,
    "synthetic_structure": synthetic_structure_docs,
}


def build_seed(repo_root: Path) -> dict[str, Any]:
    seed_root = repo_root / "datasets" / "seed"
    mix = {
        "multilingual": 0.40,
        "code": 0.25,
        "enterprise": 0.20,
        "scientific": 0.10,
        "synthetic_structure": 0.05,
    }
    shards: dict[str, Any] = {}
    totals = {}
    for bucket in BUCKETS:
        docs = GENERATORS[bucket]()
        bdir = seed_root / bucket
        bdir.mkdir(parents=True, exist_ok=True)
        path = bdir / "seed_0001.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        rel = path.relative_to(repo_root).as_posix()
        shards[bucket] = {"files": [rel], "docs": len(docs)}
        totals[bucket] = len(docs)

    manifest = {
        "name": "gate0_tokenizer",
        "version": "0.1",
        "mix": mix,
        "shards": shards,
        "totals": totals,
        "notes": "Committed seed bootstrap. Expand via datasets/raw/shards for production Gate0.",
    }
    man_path = repo_root / "datasets" / "manifests" / "gate0_tokenizer.json"
    save_manifest(man_path, manifest)
    return manifest
