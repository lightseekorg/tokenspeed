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

"""CuTe DSL based sampling kernels.

Wraps the upstream CuTe DSL ``ArgmaxKernel`` (derived from the Quack library and
ported through TensorRT-LLM) so the runtime can call it without touching the
third-party module directly.

Exports two entry points:

* :func:`argmax_pair`: row-wise ``(max_value, argmax_index)`` in a single
  ``(M, 2)`` float32 tensor — direct passthrough to the kernel.
* :func:`argmax`: drop-in replacement for ``torch.argmax(logits, dim=-1)``.
  Returns int64 indices and transparently falls back to ``torch.argmax`` when
  the CuTe DSL kernel is unavailable or its preconditions are not met
  (dtype/N/alignment/SM-version).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn

__all__ = [
    "argmax",
    "argmax_pair",
    "is_available",
]

_argmax_kernel_impl = error_fn
_compile_cache: dict[tuple, object] = {}

# Minimum vocab size for the CuTe tiled kernel.
#
# The kernel hangs on B200 (sm_100) when ``_calculate_threads_per_row()``
# returns 32 AND ``tiler_mn[1] == N`` (i.e. ``is_even_N`` skips ``fill_oob``).
# Empirically that happens for N ∈ {256, 512, 1024, 2048, 3072} — every clean
# multiple in the upstream ``128 < N <= 3072`` band. Bumping the floor above
# 3072 sidesteps the bad band entirely; every real LLM vocab (≥ 30K) is far
# above this, so we never lose the kernel in practice.
_MIN_VOCAB_SIZE = 4096

# The async copy requires 128-byte alignment.
_VOCAB_SIZE_ALIGNMENT = 32


def _ts_supported_arch() -> bool:
    """Gate: only NVIDIA Hopper/Blackwell run the CuTe DSL kernel.

    * Vendor must be NVIDIA — AMD ROCm and any future vendor get the torch
      fallback (CuTe DSL has no ROCm backend).
    * SM range ``[9.0, 12.0)``: ``redux.sync.max.f32`` exists from Blackwell
      (sm_100/sm_103); we run on Hopper too via the shuffle path. ``sm_120+``
      is excluded — upstream TRT-LLM reports CUTLASS DSL JIT instability there.
    * If platform detection itself raises (e.g. CPU-only host with no GPU),
      treat it as unsupported and let callers fall back transparently.
    """
    try:
        p = current_platform()
    except Exception:
        return False
    if not p.is_nvidia:
        return False
    sm = p.arch_version.major * 10 + p.arch_version.minor
    return 90 <= sm < 120


_ARCH_SUPPORTED = _ts_supported_arch()


# Only import the third-party CuTe DSL module on supported NVIDIA hardware.
# On AMD / CPU-only / unsupported SM, leave ``_CUTE_AVAILABLE = False`` so every
# entry point in this module routes through ``torch.argmax``.
_CUTE_AVAILABLE = False
if _ARCH_SUPPORTED:
    try:
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from tokenspeed_kernel.thirdparty.cute_dsl.argmax import (
            ArgmaxKernel,
            CUDAGraphCompatibleWrapper,
            torch2cute_dtype_map,
        )

        _CUTE_AVAILABLE = True
    except ImportError:
        _CUTE_AVAILABLE = False


def is_available() -> bool:
    """Whether the CuTe DSL argmax kernel can run on the current platform."""
    return _CUTE_AVAILABLE


def _supports_cute(N: int, dtype: torch.dtype) -> bool:
    if not _CUTE_AVAILABLE:
        return False
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    # The current upstream wrapper only ships a float32 path. Honor that here so
    # we don't surprise callers with reduced-precision argmax on bf16/fp16.
    if dtype is not torch.float32:
        return False
    if N < _MIN_VOCAB_SIZE:
        return False
    if N % _VOCAB_SIZE_ALIGNMENT != 0:
        return False
    return True


def argmax_pair(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Row-wise ``(max_value, argmax_index)`` in a single ``(M, 2)`` float32 tensor.

    The index is stored as ``float32`` — exact for any vocab below ``2**24``
    (16,777,216), which covers every shipped LLM vocab size.

    Args:
        logits: 2D tensor of shape ``(M, N)`` on CUDA.
        out: Optional pre-allocated ``(M, 2)`` float32 tensor. When supplied,
            the kernel writes into it (useful inside CUDA graphs).

    Returns:
        ``(M, 2)`` float32 tensor: ``out[:, 0]`` is the max value,
        ``out[:, 1]`` is the argmax index.
    """
    if logits.dim() != 2:
        raise ValueError(f"argmax_pair expects 2D input, got {logits.dim()}D")
    if not logits.is_cuda:
        raise ValueError("argmax_pair requires a CUDA tensor")

    M, N = logits.shape

    if out is None:
        out = torch.empty((M, 2), dtype=torch.float32, device=logits.device)
    else:
        if out.shape != (M, 2):
            raise ValueError(
                f"out must have shape (M, 2)={M, 2}, got {tuple(out.shape)}"
            )
        if out.dtype != torch.float32 or out.device != logits.device:
            raise ValueError("out must be float32 on the same device as logits")

    if not _supports_cute(N, logits.dtype):
        # Fallback: emulate the same (max, idx) packing using torch ops so the
        # downstream extraction path is uniform.
        max_vals, max_indices = torch.max(logits, dim=-1, keepdim=True)
        out[:, 0:1].copy_(max_vals.to(torch.float32))
        out[:, 1:2].copy_(max_indices.to(torch.float32))
        return out

    dtype = torch2cute_dtype_map[logits.dtype]

    def _convert(t: torch.Tensor):
        return from_dlpack(
            CUDAGraphCompatibleWrapper(t.detach()), assumed_align=16
        ).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

    x_tensor = _convert(logits)
    out_tensor = _convert(out)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Blackwell (sm_100/103) supports redux.sync.max.f32; Hopper falls back to
    # warp shuffles.
    p = current_platform()
    sm = p.arch_version.major * 10 + p.arch_version.minor
    use_redux = 100 <= sm < 120

    compile_key = (dtype, N, use_redux)
    compiled = _compile_cache.get(compile_key)
    if compiled is None:
        kernel = ArgmaxKernel(dtype, N, use_redux=use_redux)
        compiled = cute.compile(kernel, x_tensor, out_tensor, stream)
        _compile_cache[compile_key] = compiled

    compiled(x_tensor, out_tensor, stream)
    return out


def argmax(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in replacement for ``torch.argmax(logits, dim=-1)``.

    Returns an int64 tensor of shape ``(M,)``. The CuTe DSL kernel is used
    when ``logits`` is a 2D CUDA float32 tensor with N >= 256 and N % 32 == 0
    on a supported Blackwell-class SM; otherwise this falls back to
    ``torch.argmax`` for transparent compatibility.

    Args:
        logits: 2D tensor of shape ``(M, N)``. Only the last dim is reduced.
        out: Optional pre-allocated int64 buffer of shape ``(M,)``.

    Returns:
        int64 tensor of shape ``(M,)``.
    """
    if (
        logits.dim() != 2
        or not logits.is_cuda
        or not _supports_cute(logits.shape[1], logits.dtype)
    ):
        result = torch.argmax(logits, dim=-1)
        if out is not None:
            out.copy_(result)
            return out
        return result

    pair = argmax_pair(logits)
    indices = pair[:, 1].to(torch.int64)
    if out is not None:
        out.copy_(indices)
        return out
    return indices


# Expose the underlying compiled-kernel entry as the registered impl so other
# code can probe `is _argmax_kernel_impl` style availability checks if needed.
if _CUTE_AVAILABLE:
    _argmax_kernel_impl = argmax_pair
