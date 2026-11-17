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

"""NVFP4 cuBLASLt GEMM adapted from TensorRT-LLM's ``CublasLtFP4GemmRunner``.

Loads the pre-compiled ``cublaslt_fp4_gemm.so`` via TVM FFI.
"""

import functools
from pathlib import Path

import torch

# TensorRT-LLM's CUBLAS_WORKSPACE_SIZE.
_WORKSPACE_BYTES = 33554432


def _objs_dir() -> Path:
    return Path(__file__).resolve().parent / "objs"


@functools.cache
def _load_cublaslt_fp4_gemm_module():
    """Load the pre-compiled cublaslt_fp4_gemm shared library via TVM FFI."""
    import tvm_ffi

    so_path = _objs_dir() / "cublaslt_fp4_gemm" / "cublaslt_fp4_gemm.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel cublaslt_fp4_gemm library not found at {so_path}. "
            "Run `pip install -e tokenspeed_kernel/python/` to build."
        )
    return tvm_ffi.load_module(str(so_path))


@functools.cache
def _device_buffers(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The per-device cuBLASLt workspace and the zero beta it reads on device."""
    workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    beta = torch.zeros(1, dtype=torch.float32, device=device)
    return workspace, beta


def _as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float4_e2m1fn_x2:
        return tensor.view(torch.uint8)
    return tensor


def cublaslt_fp4_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``alpha * a @ b.T`` for NVFP4 operands with 16-element UE4M3 block scales.

    Matches TensorRT-LLM's ``CublasLtFP4GemmRunner(out_dtype).run_gemm(a, b,
    a_sf, b_sf, alpha, False, 0)``: the same descriptors, heuristic, and first
    successful algorithm, so the result is bit-identical.

    Args:
        a: Packed FP4 activations ``[M, K // 2]`` (uint8 or float4_e2m1fn_x2).
        b: Packed FP4 weight ``[N, K // 2]``, row-major.
        a_sf: Activation block scales in the 128x4 swizzled layout; only its
            data pointer is read.
        b_sf: Weight block scales in the 128x4 swizzled layout; only its data
            pointer is read.
        alpha: Float32 global scale on the GPU (its first element is read).
        out_dtype: ``torch.bfloat16``, ``torch.float16`` or ``torch.float32``.
        out: Optional ``[M, N]`` output; written directly when contiguous.

    Returns:
        The ``[M, N]`` product, in ``out`` when supplied.
    """
    m, n = a.shape[0], b.shape[0]
    if out is not None and out.is_contiguous() and out.dtype == out_dtype:
        result = out
    else:
        result = torch.empty((m, n), dtype=out_dtype, device=a.device)
    if m > 0 and n > 0:
        workspace, beta = _device_buffers(a.device)
        _load_cublaslt_fp4_gemm_module().cublaslt_fp4_gemm(
            result, _as_bytes(a), _as_bytes(b), a_sf, b_sf, alpha, beta, workspace
        )
    if out is not None and result is not out:
        return out.copy_(result)
    return result
