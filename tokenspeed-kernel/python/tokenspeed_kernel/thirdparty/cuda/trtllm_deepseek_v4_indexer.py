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

"""Wrappers for vendored TRT-LLM DeepSeek-V4 indexer Q kernels."""

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
        / "trtllm_deepseek_v4_indexer_q"
        / "trtllm_deepseek_v4_indexer_q.so"
    )
    if not so_path.exists():
        raise RuntimeError(
            "tokenspeed_kernel TRT-LLM DeepSeek-V4 indexer library not found at "
            f"{so_path}. Run `pip install -e tokenspeed-kernel/python/` to build."
        )
    return tvm_ffi.load_module(str(so_path))


def has_trtllm_indexer_q_kernels() -> bool:
    """Return whether the vendored TRT-LLM RoPE and FP4 pack ops are built."""

    try:
        module = _load_module()
    except Exception:
        return False
    return all(
        hasattr(module, name)
        for name in (
            "trtllm_deepseek_v4_mla_rope_inplace",
            "trtllm_deepseek_v4_fused_cat_fp4",
        )
    )


def trtllm_mla_rope_inplace(
    data: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    enable_pdl: bool = False,
) -> None:
    """Apply TRT-LLM's interleaved DeepSeek-V4 MLA RoPE in place.

    Args:
        data: Contiguous BF16 tensor shaped ``[tokens, heads, 128]``.
        positions: Contiguous INT32/INT64 absolute positions shaped ``[tokens]``.
        cos_sin_cache: Contiguous FP32 table shaped ``[max_position, 64]``.
        enable_pdl: Whether to enable programmatic dependent launch.

    Returns:
        ``None``. The final 64 elements of each head in ``data`` are updated.
    """

    if data.dtype != torch.bfloat16:
        raise TypeError(f"data must be bfloat16, got {data.dtype}")
    if data.ndim != 3 or data.shape[-1] != 128:
        raise ValueError(f"data must be [tokens, heads, 128], got {tuple(data.shape)}")
    if not data.is_contiguous():
        raise ValueError("data must be contiguous")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"positions must be int32 or int64, got {positions.dtype}")
    if positions.ndim != 1 or positions.shape[0] != data.shape[0]:
        raise ValueError(
            f"positions must be [{data.shape[0]}], got {tuple(positions.shape)}"
        )
    if cos_sin_cache.dtype != torch.float32:
        raise TypeError(f"cos_sin_cache must be float32, got {cos_sin_cache.dtype}")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[1] != 64:
        raise ValueError(
            "cos_sin_cache must be [max_position, 64], "
            f"got {tuple(cos_sin_cache.shape)}"
        )
    _load_module().trtllm_deepseek_v4_mla_rope_inplace(
        data,
        positions.contiguous(),
        cos_sin_cache.contiguous(),
        bool(enable_pdl),
    )


def trtllm_fused_cat_fp4(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run TRT-LLM's fused concat plus per-block MXFP4 pack kernel.

    Args:
        first: BF16 tensor with a contiguous innermost dimension and
            row-linear leading dimensions.
        second: BF16 tensor with the same flattened row count and layout
            contract as ``first``.

    Returns:
        A pair ``(packed, scales)``. ``packed`` is UINT8 ``[rows, 64]`` and
        ``scales`` is INT32 ``[rows, 1]`` containing four packed UE8M0 bytes.
    """

    if first.dtype != torch.bfloat16 or second.dtype != torch.bfloat16:
        raise TypeError(
            f"first and second must be bfloat16, got {first.dtype} and {second.dtype}"
        )
    if first.ndim < 2 or second.ndim < 2:
        raise ValueError("first and second must be at least 2-D")
    if first.shape[-1] + second.shape[-1] != 128:
        raise ValueError(
            "first and second innermost dimensions must sum to 128, got "
            f"{first.shape[-1]} + {second.shape[-1]}"
        )
    if first.stride(-1) != 1 or second.stride(-1) != 1:
        raise ValueError("first and second innermost dimensions must be contiguous")
    for name, tensor in (("first", first), ("second", second)):
        if tensor.numel() == 0:
            continue
        expected_stride = tensor.stride(-2)
        for size, stride in zip(
            reversed(tensor.shape[:-1]),
            reversed(tensor.stride()[:-1]),
        ):
            if size > 1 and stride != expected_stride:
                raise ValueError(
                    f"{name} leading dimensions must be row-linear, got "
                    f"shape {tuple(tensor.shape)} and strides {tensor.stride()}"
                )
            expected_stride *= size
    first_rows = first.numel() // first.shape[-1]
    second_rows = second.numel() // second.shape[-1]
    if first_rows != second_rows:
        raise ValueError(
            f"first and second row counts must match, got {first_rows} and {second_rows}"
        )
    packed = torch.empty((first_rows, 64), dtype=torch.uint8, device=first.device)
    scales = torch.empty((first_rows, 1), dtype=torch.int32, device=first.device)
    _load_module().trtllm_deepseek_v4_fused_cat_fp4(
        first,
        second,
        packed,
        scales,
    )
    return packed, scales
