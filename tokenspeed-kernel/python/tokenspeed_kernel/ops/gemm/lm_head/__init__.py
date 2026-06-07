from tokenspeed_kernel.registrations._vendor import export_vendor_symbols, false_fn

__all__ = ["is_supported", "lm_head_gemm", "should_use_fused"]

globals().update(
    export_vendor_symbols(
        "nvidia",
        "tokenspeed_kernel_nvidia.thirdparty.cuda.lm_head_gemm",
        __all__,
        fallback_by_name={
            "is_supported": false_fn,
            "should_use_fused": false_fn,
        },
    )
)
