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

"""Loader for the vendored TRT-inspired DeepSeek-V4 C128 prefill op."""

from __future__ import annotations

import functools
from pathlib import Path

import torch
import tvm_ffi


@functools.cache
def _load_module():
    so_path = (
        Path(__file__).resolve().parent
        / "objs"
        / "trtllm_deepseek_v4_c128_prefill"
        / "trtllm_deepseek_v4_c128_prefill.so"
    )
    if not so_path.exists():
        raise RuntimeError(
            "tokenspeed_kernel TRT-inspired DeepSeek-V4 C128 prefill library "
            f"not found at {so_path}. Run `pip install -e tokenspeed-kernel/python/` "
            "to build."
        )
    return tvm_ffi.load_module(str(so_path))


def has_trtllm_deepseek_v4_c128_prefill_kernels() -> bool:
    """Return whether the C128 pure-prefill compressor op is built."""

    try:
        module = _load_module()
    except Exception:
        return False
    return hasattr(module, "trtllm_deepseek_v4_c128_prefill_compress_cache")


def trtllm_deepseek_v4_c128_prefill_compress_cache_raw(
    *,
    state_cache: torch.Tensor,
    scratch: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    block_table_base_offsets: torch.Tensor | None,
    state_block_size: int,
    kv_block_size: int,
    max_outputs: int,
    rms_norm_eps: float,
) -> None:
    """Launch the allocation-free C128 prefill reduction and cache scatter.

    All tensors must already satisfy the native op contract. In particular,
    this wrapper performs no dtype/device conversion or contiguous copy.

    Args:
        state_cache: FP32 combined ``[kv | score+APE]`` state pages shaped
            ``[blocks, state_block_size, 1024]``.
        scratch: Mutable contiguous FP32 projection buffer shaped
            ``[tokens, >=512]``. Boundary rows are overwritten in ``:512``.
        positions: Contiguous INT64 absolute token positions, ``[tokens]``.
        compressor_slot_mapping: Contiguous INT64 compressor-state slots,
            ``[tokens]``. A negative boundary slot suppresses that output.
        query_start_loc: Contiguous INT32 packed query offsets, ``[batch+1]``.
        seq_lens: Contiguous INT32 final sequence lengths, ``[batch]``.
        block_table: Contiguous INT32 compact state page table, ``[batch,width]``.
        rms_norm_weight: Contiguous BF16 or FP32 RMSNorm weights, ``[512]``.
        cos_sin_cache: Contiguous FP32 interleaved-RoPE table, ``[maxpos,64]``.
            It must cover every 128-aligned compressed position referenced by
            the batch; this device-resident bound is a caller precondition.
        kv_cache: Mutable contiguous UINT8 fp8_ds_mla pages, ``[pages,stride]``.
        kv_slot_mapping: Contiguous INT64 output slots, ``[tokens]``.
        block_table_base_offsets: Optional contiguous INT32 logical-page bases,
            ``[batch]``.
        state_block_size: Number of token rows in each state page.
        kv_block_size: Number of compressed rows in each KV page.
        max_outputs: Positive launch bound, at most 65535, that must cover
            every compressed output per request; an undersized bound skips
            outputs.
        rms_norm_eps: RMSNorm epsilon.

    Returns:
        ``None``. ``scratch`` and ``kv_cache`` are mutated in place.
    """

    has_base_offsets = block_table_base_offsets is not None
    # TVM-FFI requires a concrete TensorView even when the optional argument is
    # disabled. seq_lens is a same-device contiguous INT32 placeholder; the
    # native op never dereferences it when has_base_offsets is false.
    base_offsets_arg = block_table_base_offsets if has_base_offsets else seq_lens
    _load_module().trtllm_deepseek_v4_c128_prefill_compress_cache(
        state_cache,
        scratch,
        positions,
        compressor_slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        base_offsets_arg,
        has_base_offsets,
        rms_norm_weight,
        cos_sin_cache,
        kv_cache,
        kv_slot_mapping,
        int(state_block_size),
        int(kv_block_size),
        int(max_outputs),
        float(rms_norm_eps),
    )
