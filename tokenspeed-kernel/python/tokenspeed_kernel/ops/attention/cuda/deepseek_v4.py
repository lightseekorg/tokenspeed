# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CUDA DeepSeek V4 attention kernels."""

import torch
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

try:
    from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
        fused_qnorm_rope_kv_insert as _fused_qnorm_rope_kv_insert,
    )
    from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
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

    _fused_qnorm_rope_kv_insert = error_fn
    indexer_mxfp4_paged_gather = error_fn
    indexer_topk_prefill = error_fn
    persistent_topk = error_fn


if current_platform().is_nvidia and has_fused_qnorm_rope_kv_insert():

    @register_kernel(
        "attention",
        "deepseek_v4_swa_cache_insert",
        name="cuda_deepseek_v4_swa_cache_insert",
        solution="cuda",
        capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
        signatures=frozenset(
            format_signature(
                q=dense_tensor_format(dtype),
                kv=dense_tensor_format(dtype),
                swa_kv_cache=dense_tensor_format(torch.uint8),
            )
            for dtype in (torch.float16, torch.bfloat16)
        ),
        traits={
            "head_dim": frozenset({512}),
            "rope_dim": frozenset({64}),
            "quant_block_size": frozenset({64}),
            "cache_layout": frozenset({"fp8_swa_page_planar"}),
            "has_q_out": frozenset({True, False}),
        },
        priority=Priority.SPECIALIZED,
        tags={"nvidia", "cache_insert", "latency"},
    )
    def cuda_deepseek_v4_swa_cache_insert(
        q: torch.Tensor,
        kv: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        rms_norm_eps: float,
        page_size: int,
        q_out: torch.Tensor | None = None,
    ) -> None:
        q_destination = q
        if q_out is not None:
            q_out.copy_(q)
            q_destination = q_out
        _fused_qnorm_rope_kv_insert(
            q_destination,
            kv,
            swa_kv_cache,
            slot_mapping,
            positions,
            cos_sin_cache,
            rms_norm_eps,
            page_size,
        )


fused_qnorm_rope_kv_insert = _fused_qnorm_rope_kv_insert

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
