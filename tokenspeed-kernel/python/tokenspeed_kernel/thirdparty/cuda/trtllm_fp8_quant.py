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

"""Wrapper for the vendored TRT-LLM packed UE8M0 FP8 quantizer."""

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
        / "trtllm_fp8_quant_packed"
        / "trtllm_fp8_quant_packed.so"
    )
    if not so_path.exists():
        raise RuntimeError(
            "tokenspeed_kernel TRT-LLM packed FP8 quantization library not "
            f"found at {so_path}. Run `pip install -e tokenspeed-kernel/python/` "
            "to build."
        )
    return tvm_ffi.load_module(str(so_path))


@functools.cache
def has_trtllm_fp8_quant_packed() -> bool:
    """Return whether the vendored TRT-LLM packed quantizer is built.

    Returns:
        ``True`` when the shared library can be loaded and exports the op.
    """

    try:
        module = _load_module()
    except Exception:
        return False
    return hasattr(module, "trtllm_fp8_quantize_1x128_packed_ue8m0")


@functools.cache
def _is_supported_device(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (10, 0)


def supports_trtllm_fp8_quant_packed(x: torch.Tensor) -> bool:
    """Check whether an input can use the vendored SM100 quantizer.

    Args:
        x: Candidate activation tensor.

    Returns:
        ``True`` when the op is present and ``x`` has the required device,
        dtype, shape, contiguity, and alignment.
    """

    if (
        not has_trtllm_fp8_quant_packed()
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.ndim != 2
        or not x.is_contiguous()
        or x.shape[0] == 0
        or x.shape[1] == 0
        or x.shape[1] % 128 != 0
        or x.data_ptr() % 32 != 0
    ):
        return False
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _is_supported_device(device_index)


def trtllm_fp8_quantize_1x128_packed_ue8m0(
    x: torch.Tensor,
    *,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 rows and emit DeepGEMM packed UE8M0 scales.

    Args:
        x: Contiguous, 32-byte-aligned CUDA BF16 matrix shaped
            ``[rows, columns]``. The column count must be divisible by 128.
        enable_pdl: Whether to allow programmatic dependent launch.

    Returns:
        A pair ``(values, scales)``. ``values`` is row-major E4M3 with the same
        shape as ``x``. ``scales`` is an INT32 view shaped
        ``[rows, ceil(columns / 512)]`` with stride ``(1, align(rows, 4))``.
    """

    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"x must be bfloat16, got {x.dtype}")
    if x.ndim != 2:
        raise ValueError(f"x must be a matrix, got shape {tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.data_ptr() % 32 != 0:
        raise ValueError("x must be 32-byte aligned")
    rows, columns = x.shape
    if rows == 0:
        raise ValueError("x must contain at least one row")
    if columns == 0 or columns % 128 != 0:
        raise ValueError(f"x columns must be divisible by 128, got {columns}")

    aligned_rows = (rows + 3) // 4 * 4
    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if not _is_supported_device(device_index):
        raise RuntimeError("TRT-LLM packed UE8M0 quantization requires SM100")

    value_storage = torch.empty(
        (aligned_rows, columns),
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )
    values = value_storage[:rows]
    packed_scale_columns = (columns // 128 + 3) // 4
    scale_storage = torch.empty(
        (packed_scale_columns, aligned_rows),
        dtype=torch.int32,
        device=x.device,
    )
    _load_module().trtllm_fp8_quantize_1x128_packed_ue8m0(
        x,
        values,
        scale_storage,
        bool(enable_pdl),
    )
    scales = scale_storage.transpose(0, 1)[:rows, :]
    return values, scales


__all__ = [
    "has_trtllm_fp8_quant_packed",
    "supports_trtllm_fp8_quant_packed",
    "trtllm_fp8_quantize_1x128_packed_ue8m0",
]
