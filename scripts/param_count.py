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
    print("=== Active family (2026) ===")
    print(f"20B dense V1: {dense_count(24, 8192, 8, 128, 131072, 22016):.3f}B")
    print(f"A35B dense:   {dense_count(48, 8192, 8, 128, 131072, 22016):.3f}B")
    m200 = moe_count(48, 8192, 8, 128, 131072, 4, 80, 1, 6, 2176, 22016)
    print(f"200B MoE:     total={m200['total_B']:.1f}B active={m200['active_B']:.1f}B")
    m480 = moe_count(64, 8192, 8, 128, 131072, 4, 152, 1, 6, 2048, 22016)
    print(f"480B MoE:     total={m480['total_B']:.1f}B active={m480['active_B']:.1f}B")


if __name__ == "__main__":
    main()
