"""FlashInfer direct MLA APIs."""

from __future__ import annotations

from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import ErrorClass, error_fn

BatchMLAPagedAttentionWrapper = ErrorClass
trtllm_batch_decode_with_kv_cache_mla = error_fn
trtllm_ragged_attention_deepseek = error_fn

platform = current_platform()
if platform.is_nvidia:
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla
    from flashinfer.prefill import trtllm_ragged_attention_deepseek

if platform.is_blackwell or platform.is_hopper:
    from flashinfer.mla import (
        BatchMLAPagedAttentionWrapper,
        trtllm_batch_decode_with_kv_cache_mla,
    )

__all__ = [
    "BatchMLAPagedAttentionWrapper",
    "trtllm_batch_decode_with_kv_cache_mla",
    "trtllm_ragged_attention_deepseek",
]
