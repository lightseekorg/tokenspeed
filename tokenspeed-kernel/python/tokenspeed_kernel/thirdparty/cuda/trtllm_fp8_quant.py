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

"""FP8 E4M3 quantizers adapted from TensorRT-LLM.

Both return the same tensors as the TensorRT-LLM ops they replace:
``fp8_quantize_1x128`` as ``torch.ops.trtllm.fp8_quantize_1x128`` and
``quantize_e4m3_activation`` as ``torch.ops.tensorrt_llm.quantize_e4m3_activation``.
"""

import functools
from pathlib import Path

import torch


def _objs_dir() -> Path:
    return Path(__file__).resolve().parent / "objs"


@functools.cache
def _load_trtllm_fp8_quant_module():
    """Load the pre-compiled trtllm_fp8_quant shared library via TVM FFI."""
    import tvm_ffi

    so_path = _objs_dir() / "trtllm_fp8_quant" / "trtllm_fp8_quant.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel trtllm_fp8_quant library not found at {so_path}. "
            "Run `pip install -e tokenspeed_kernel/python/` to build."
        )
    return tvm_ffi.load_module(str(so_path))


@functools.cache
def _is_sm100_family(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index)[0] == 10


def fp8_quantize_1x128(
    x: torch.Tensor, use_ue8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 ``[M, N]`` to FP8 E4M3 with one FP32 scale per 1x128 group.

    Args:
        x: Contiguous BF16 CUDA ``[M, N]`` with ``N % 16 == 0``.
        use_ue8m0: Round each dequant scale up to a power of two.

    Returns:
        FP8 ``[M, N]`` codes (a row slice of an ``[ceil(M / 4) * 4, N]``
        buffer) and the FP32 dequant scales. On SM10x the scales are a
        contiguous ``[ceil(N / 128), M]`` tensor; elsewhere they are the flat
        MN-major buffer with a row stride of ``ceil(M / 4) * 4``, padded to a
        128-byte size.
    """
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("fp8_quantize_1x128 requires a contiguous CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise ValueError("Input matrix dtype must be BF16.")
    if x.dim() != 2:
        raise ValueError("input must be a matrix")
    m, n = x.shape
    if n % 16 != 0:
        raise ValueError(f"x.shape[1] must be a multiple of 16, but got {n}")
    m_padded = (m + 3) // 4 * 4
    groups = (n + 127) // 128
    # getActScaleSize: m_padded * groups FP32 scales, rounded up to 128 bytes.
    scale_numel = (m_padded * groups * 4 + 127) // 128 * 128 // 4
    values = torch.empty((m_padded, n), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((scale_numel,), dtype=torch.float32, device=x.device)
    _load_trtllm_fp8_quant_module().fp8_quantize_1x128(
        values, scales, x, bool(use_ue8m0)
    )
    if _is_sm100_family(x.device.index):
        # The SM10x GEMMs take the scales without the 4-row alignment.
        scales = scales[: groups * m_padded].view(groups, m_padded)[:, :m].contiguous()
    return values[:m], scales


def quantize_e4m3_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token dynamic FP8 E4M3 quantization.

    Each row's scale is ``max(amax / 448, 1 / (448 * 512))`` stored in the
    input dtype; the codes are ``x / scale`` with that stored scale.

    Args:
        x: Contiguous FP32, FP16 or BF16 CUDA tensor with at least two dims.

    Returns:
        FP8 E4M3 codes shaped like ``x`` and ``[..., 1]`` scales in ``x.dtype``.
    """
    if not x.is_cuda or not x.is_contiguous():
        raise ValueError("quantize_e4m3_activation requires a contiguous CUDA tensor")
    if x.numel() == 0:
        raise ValueError("input should not be empty tensor")
    if x.dim() < 2:
        raise ValueError("input must have at least two dimensions")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("input must be FP16 or BF16 or FP32")
    values = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((*x.shape[:-1], 1), dtype=x.dtype, device=x.device)
    _load_trtllm_fp8_quant_module().quantize_e4m3_per_token(values, scales, x)
    return values, scales
