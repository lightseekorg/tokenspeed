from tokenspeed_kernel.registrations._vendor import export_vendor_symbols, false_fn

__all__ = [
    "fused_qnorm_rope_kv_insert",
    "has_fused_qnorm_rope_kv_insert",
    "has_indexer_mxfp4_paged_gather",
    "has_indexer_topk_prefill",
    "has_persistent_topk",
    "indexer_mxfp4_paged_gather",
    "indexer_topk_prefill",
    "persistent_topk",
]

globals().update(
    export_vendor_symbols(
        "nvidia",
        "tokenspeed_kernel_nvidia.attention.cuda.deepseek_v4",
        __all__,
        fallback_by_name={
            "has_fused_qnorm_rope_kv_insert": false_fn,
            "has_indexer_mxfp4_paged_gather": false_fn,
            "has_indexer_topk_prefill": false_fn,
            "has_persistent_topk": false_fn,
        },
    )
)
