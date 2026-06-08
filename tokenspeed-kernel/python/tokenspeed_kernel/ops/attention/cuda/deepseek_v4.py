# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

"""CUDA DeepSeek V4 attention kernels."""

from tokenspeed_kernel.registry import error_fn

try:
    from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
        fused_qnorm_rope_kv_insert,
        has_fused_qnorm_rope_kv_insert,
        has_indexer_mxfp4_paged_gather,
        has_indexer_topk_prefill,
        has_persistent_topk,
        indexer_mxfp4_paged_gather,
        indexer_topk_prefill,
        persistent_topk,
    )
except ImportError:

    def has_fused_qnorm_rope_kv_insert() -> bool:
        return False

    def has_indexer_topk_prefill() -> bool:
        return False

    def has_indexer_mxfp4_paged_gather() -> bool:
        return False

    def has_persistent_topk() -> bool:
        return False

    fused_qnorm_rope_kv_insert = error_fn
    indexer_mxfp4_paged_gather = error_fn
    indexer_topk_prefill = error_fn
    persistent_topk = error_fn

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
