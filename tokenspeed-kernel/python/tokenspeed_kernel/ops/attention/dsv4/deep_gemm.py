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

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import current_platform, prepare_cuda_toolkit_env

_IS_HOPPER_PLUS = current_platform().is_hopper_plus

if _IS_HOPPER_PLUS:
    prepare_cuda_toolkit_env()
    import deep_gemm


def _cache_row_bytes(cache_2d: torch.Tensor, page_size: int) -> int:
    if cache_2d.ndim != 2 or cache_2d.dtype != torch.uint8:
        raise ValueError(
            "index_k_cache must be a 2-D uint8 page matrix, got "
            f"{tuple(cache_2d.shape)} {cache_2d.dtype}"
        )
    if page_size <= 0 or cache_2d.shape[1] % page_size != 0:
        raise ValueError(
            "index-K cache row size must be divisible by page_size, got "
            f"shape={tuple(cache_2d.shape)}, page_size={page_size}"
        )
    return cache_2d.shape[1] // page_size


def _mxfp4_cache_view(cache_2d: torch.Tensor, page_size: int) -> torch.Tensor:
    row_bytes = _cache_row_bytes(cache_2d, page_size)
    return torch.as_strided(
        cache_2d,
        (cache_2d.shape[0], page_size, 1, row_bytes),
        (cache_2d.stride(0), row_bytes, row_bytes, 1),
    )


def _warmup_fp8_fp4_mqa_logits(
    *,
    num_heads: int,
    index_head_dim: int,
    device: torch.device,
    max_kv_len: int,
) -> None:
    """Pre-compile the ragged prefill sparse-indexer kernel."""
    head_dim_bytes = index_head_dim // 2
    for num_tokens in (1, 256):
        q_vals = torch.zeros(
            num_tokens, num_heads, head_dim_bytes, dtype=torch.uint8, device=device
        ).view(torch.int8)
        q_scales = torch.zeros(num_tokens, num_heads, dtype=torch.int32, device=device)
        k_vals = torch.zeros(
            max_kv_len, head_dim_bytes, dtype=torch.uint8, device=device
        ).view(torch.int8)
        k_scales = torch.zeros(max_kv_len, dtype=torch.int32, device=device)
        weights = torch.ones(num_tokens, num_heads, dtype=torch.float32, device=device)
        cu_start = torch.zeros(num_tokens, dtype=torch.int32, device=device)
        cu_end = torch.full((num_tokens,), max_kv_len, dtype=torch.int32, device=device)

        deep_gemm.fp8_fp4_mqa_logits(
            q=(q_vals, q_scales),
            kv=(k_vals, k_scales),
            weights=weights,
            cu_seq_len_k_start=cu_start,
            cu_seq_len_k_end=cu_end,
            clean_logits=False,
            max_seqlen_k=max_kv_len,
            logits_dtype=torch.float32,
        )


def _warmup_fp8_fp4_paged_mqa_logits(
    *,
    num_heads: int,
    index_head_dim: int,
    cache_block_size: int,
    max_decode_tokens: int,
    device: torch.device,
) -> None:
    """Pre-compile paged decode indexer and schedule-metadata kernels."""
    head_dim_bytes = index_head_dim // 2
    row_bytes = head_dim_bytes + 4
    num_sms = deep_gemm.get_num_sms()
    top_bucket = max(32, ((max_decode_tokens + 31) // 32) * 32)

    for num_tokens in range(32, top_bucket + 1, 32):
        num_blocks = max(1, num_tokens)
        q_values = torch.zeros(
            num_tokens, num_heads, head_dim_bytes, dtype=torch.uint8, device=device
        )
        q_scales = torch.zeros(num_tokens, num_heads, dtype=torch.int32, device=device)
        cache_2d = torch.zeros(
            num_blocks, cache_block_size * row_bytes, dtype=torch.uint8, device=device
        )
        kv_cache = torch.as_strided(
            cache_2d,
            (num_blocks, cache_block_size, 1, row_bytes),
            (cache_2d.stride(0), row_bytes, row_bytes, 1),
        )
        weights = torch.ones(num_tokens, num_heads, dtype=torch.float32, device=device)
        context_lens = torch.full(
            (num_tokens, 1), cache_block_size, dtype=torch.int32, device=device
        )
        block_table = torch.arange(num_tokens, dtype=torch.int32, device=device).view(
            num_tokens, 1
        )
        schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens, cache_block_size, num_sms
        )
        deep_gemm.fp8_fp4_paged_mqa_logits(
            q=(q_values.view(torch.int8).unsqueeze(1), q_scales.unsqueeze(1)),
            kv_cache=kv_cache,
            weights=weights,
            context_lens=context_lens,
            block_table=block_table,
            schedule_meta=schedule_meta,
            max_context_len=cache_block_size,
            clean_logits=False,
            logits_dtype=torch.float32,
        )
    torch.cuda.synchronize()


def warmup_mqa_logits(
    *,
    num_heads: int,
    index_head_dim: int,
    cache_block_size: int,
    max_decode_tokens: int,
    device: torch.device,
) -> None:
    """Compile packed MQA scoring and paged metadata kernels before capture."""
    _warmup_fp8_fp4_mqa_logits(
        num_heads=num_heads,
        index_head_dim=index_head_dim,
        device=device,
        max_kv_len=4096,
    )
    _warmup_fp8_fp4_paged_mqa_logits(
        num_heads=num_heads,
        index_head_dim=index_head_dim,
        cache_block_size=cache_block_size,
        max_decode_tokens=max_decode_tokens,
        device=device,
    )
