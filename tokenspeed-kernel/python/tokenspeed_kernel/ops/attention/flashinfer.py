from tokenspeed_kernel.registrations._vendor import (
    export_vendor_symbols,
    import_vendor_module,
)
from tokenspeed_kernel.registry import error_fn

__all__ = [
    "BatchDecodeWithPagedKVCacheWrapper",
    "BatchMLAPagedAttentionWrapper",
    "BatchPrefillWithPagedKVCacheWrapper",
    "BatchPrefillWithRaggedKVCacheWrapper",
    "cudnn_batch_prefill_with_kv_cache",
    "gated_delta_rule",
    "trtllm_batch_context_with_kv_cache",
    "trtllm_batch_decode_with_kv_cache",
    "trtllm_batch_decode_with_kv_cache_mla",
    "trtllm_ragged_attention_deepseek",
]

globals().update(
    export_vendor_symbols(
        "nvidia",
        "tokenspeed_kernel_nvidia.attention.flashinfer",
        [name for name in __all__ if name != "gated_delta_rule"],
    )
)
gated_delta_rule = import_vendor_module(
    "nvidia",
    "tokenspeed_kernel_nvidia.attention.flashinfer.gated_delta_rule",
    fallback=error_fn,
)
