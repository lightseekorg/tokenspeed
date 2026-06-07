from tokenspeed_kernel._vendor import export_vendor_symbols, false_fn

__all__ = ["is_supported", "lm_head_gemm", "should_use_fused"]

globals().update(
    export_vendor_symbols(
        "nvidia",
        "tokenspeed_kernel.ops.gemm.lm_head",
        __all__,
        fallback_by_name={
            "is_supported": false_fn,
            "should_use_fused": false_fn,
        },
    )
)
