"""TokenSpeed MLA implementation and direct compatibility exports."""

from tokenspeed_kernel.ops.attention.mla.tokenspeed_mla._bindings import (
    AVAILABLE,
    get_num_sm,
    mla_kv_pack_quantize_fp8,
    tokenspeed_mla_decode,
    tokenspeed_mla_prefill,
    warmup_compile_prefill,
)

if AVAILABLE:
    import tokenspeed_kernel.ops.attention.mla.tokenspeed_mla.decode  # noqa: F401
    import tokenspeed_kernel.ops.attention.mla.tokenspeed_mla.prefill  # noqa: F401

__all__ = [
    "get_num_sm",
    "mla_kv_pack_quantize_fp8",
    "tokenspeed_mla_decode",
    "tokenspeed_mla_prefill",
    "warmup_compile_prefill",
]
