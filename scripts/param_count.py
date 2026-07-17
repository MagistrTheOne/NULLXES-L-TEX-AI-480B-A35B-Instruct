"""NULLXES-LÆTEX parameter estimator (no external deps)."""


def attn(d: int, n_kv: int, d_h: int) -> int:
    return d * d + 2 * d * (n_kv * d_h) + d * d


def swiglu(d: int, d_ff: int) -> int:
    return 3 * d * d_ff


def moe_count(
    n_layers: int,
    d: int,
    n_kv: int,
    d_h: int,
    vocab: int,
    dense_prefix: int,
    n_routed: int,
    n_shared: int,
    topk: int,
    d_ff_e: int,
    d_ff_dense: int,
) -> dict:
    emb = 2 * vocab * d
    a = attn(d, n_kv, d_h)
    moe_layers = n_layers - dense_prefix
    per_expert = swiglu(d, d_ff_e)
    router = d * n_routed
    total_attn = n_layers * a
    total_dense_ffn = dense_prefix * swiglu(d, d_ff_dense)
    total_experts = moe_layers * (n_routed + n_shared) * per_expert
    total_router = moe_layers * router
    total = emb + total_attn + total_dense_ffn + total_experts + total_router
    active_moe = moe_layers * (topk + n_shared) * per_expert
    active = emb + total_attn + total_dense_ffn + active_moe
    return {
        "total_B": total / 1e9,
        "active_B": active / 1e9,
        "per_expert_M": per_expert / 1e6,
        "moe_layers": moe_layers,
    }


def dense_count(
    n_layers: int, d: int, n_kv: int, d_h: int, vocab: int, d_ff: int
) -> float:
    emb = 2 * vocab * d
    a = attn(d, n_kv, d_h)
    f = swiglu(d, d_ff)
    return (emb + n_layers * (a + f)) / 1e9


def main() -> None:
    cands = []
    for n_routed in [128, 144, 152, 160, 168]:
        for d_ff_e in [1792, 1920, 2048, 2176]:
            for topk in [4, 6, 8]:
                for n_shared in [1, 2]:
                    r = moe_count(
                        64, 8192, 8, 128, 131072, 4, n_routed, n_shared, topk, d_ff_e, 22016
                    )
                    if 460 <= r["total_B"] <= 500 and 32 <= r["active_B"] <= 38:
                        cands.append((r["total_B"], r["active_B"], n_routed, n_shared, topk, d_ff_e))

    cands.sort(key=lambda x: abs(x[0] - 480) + abs(x[1] - 35))
    print("=== MoE candidates (~480B / ~35B) ===")
    for c in cands[:15]:
        print(
            f"total={c[0]:.1f}B active={c[1]:.1f}B "
            f"routed={c[2]} shared={c[3]} topk={c[4]} d_ff={c[5]}"
        )

    print("\n=== Dense A35B / 7B / Stage0 ===")
    specs = [
        ("A35B", 48, 8192, 8, 22016),
        ("A35B-alt", 60, 7168, 8, 19456),
        ("7B", 32, 4096, 8, 11008),
        ("Stage0", 24, 2048, 4, 5504),
    ]
    for name, nl, d, nkv, dff in specs:
        print(f"{name}: {dense_count(nl, d, nkv, 128, 131072, dff):.2f}B (L={nl} d={d})")


if __name__ == "__main__":
    main()
