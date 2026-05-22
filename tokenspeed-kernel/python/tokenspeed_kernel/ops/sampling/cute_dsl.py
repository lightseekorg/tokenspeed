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

* :func:`argmax`: drop-in replacement for ``torch.argmax(logits, dim=-1)``.
  Returns int64 indices written by the kernel directly — no post-kernel cast
  on the hot path. Transparently falls back to ``torch.argmax`` when the CuTe
  DSL kernel is unavailable or its preconditions are not met
  (dtype/N/alignment/SM-version).
* :func:`argmax_pair`: row-wise ``(max_value, argmax_index)`` packed as a
  single ``(M, 2)`` float32 tensor. The kernel writes the max value and index
  into two separate tensors; this entry point assembles them back into the
  legacy ``(M, 2)`` layout (one extra elementwise copy off the hot path). The
  runtime no longer uses this layout — kept for tests / future logprob users.
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


def _convert_to_cute(t: torch.Tensor):
    """Wrap a torch tensor as a CuTe DSL tensor with a CUDA-graph-safe view."""
    return from_dlpack(
        CUDAGraphCompatibleWrapper(t.detach()), assumed_align=16
    ).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))


def _convert_to_cute_1d(t: torch.Tensor):
    """1D-tensor variant of :func:`_convert_to_cute`."""
    return from_dlpack(
        CUDAGraphCompatibleWrapper(t.detach()), assumed_align=16
    ).mark_compact_shape_dynamic(mode=0, stride_order=(0,))


def _invoke_kernel(
    logits: torch.Tensor, out_max: torch.Tensor, out_idx: torch.Tensor
) -> None:
    """Launch ArgmaxKernel with separate ``(M,)`` max and idx output tensors.

    Caller is responsible for shape/dtype checks; this helper assumes inputs
    are already validated by :func:`_supports_cute`.
    """
    dtype = torch2cute_dtype_map[logits.dtype]
    x_tensor = _convert_to_cute(logits)
    max_tensor = _convert_to_cute_1d(out_max)
    idx_tensor = _convert_to_cute_1d(out_idx)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Blackwell (sm_100/103) supports redux.sync.max.f32; Hopper falls back to
    # warp shuffles.
    p = current_platform()
    sm = p.arch_version.major * 10 + p.arch_version.minor
    use_redux = 100 <= sm < 120

    N = logits.shape[1]
    # Cache by index dtype too: the kernel writes the index with the output
    # tensor's element type, so int64 vs int32 produce distinct compiled units.
    compile_key = (dtype, N, use_redux, out_idx.dtype)
    compiled = _compile_cache.get(compile_key)
    if compiled is None:
        kernel = ArgmaxKernel(dtype, N, use_redux=use_redux)
        compiled = cute.compile(kernel, x_tensor, max_tensor, idx_tensor, stream)
        _compile_cache[compile_key] = compiled

    compiled(x_tensor, max_tensor, idx_tensor, stream)


_SUPPORTED_OUT_DTYPES = (torch.int32, torch.int64)


def argmax(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in replacement for ``torch.argmax(logits, dim=-1)``.

    Returns an int64 tensor of shape ``(M,)`` by default. The CuTe DSL kernel
    is used when ``logits`` is a 2D CUDA float32 tensor with N >= 4096 and
    N % 32 == 0 on a supported Blackwell-class SM; otherwise this falls back
    to ``torch.argmax`` for transparent compatibility.

    The kernel writes the indices into ``out`` (or into a freshly allocated
    int64 tensor) directly, so no post-kernel elementwise cast is launched on
    the hot path. Callers that need int32 token ids can pass an int32 ``out``
    to skip a downstream ``.to(torch.int32)`` cast — the kernel will write
    int32 values straight into the buffer.

    Args:
        logits: 2D tensor of shape ``(M, N)``. Only the last dim is reduced.
        out: Optional pre-allocated buffer of shape ``(M,)``. Must be int32
            or int64 on the same device as ``logits``. Determines the dtype
            of the returned tensor; defaults to int64 when omitted.

    Returns:
        Integer tensor of shape ``(M,)`` with dtype matching ``out`` (or
        int64 when ``out`` was not supplied).
    """
    if out is not None:
        if out.shape != (logits.shape[0],):
            raise ValueError(
                f"out must have shape (M,)={(logits.shape[0],)}, "
                f"got {tuple(out.shape)}"
            )
        if out.dtype not in _SUPPORTED_OUT_DTYPES:
            raise ValueError(f"out must be int32 or int64; got {out.dtype}")
        if out.device != logits.device:
            raise ValueError("out must be on the same device as logits")

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

    M = logits.shape[0]
    device = logits.device

    if out is None:
        out_idx = torch.empty((M,), dtype=torch.int64, device=device)
    else:
        out_idx = out

    # The max value is needed only inside the kernel reduction; the caller
    # never sees it. Allocate a scratch buffer so the kernel has somewhere to
    # write it.
    scratch_max = torch.empty((M,), dtype=torch.float32, device=device)
    _invoke_kernel(logits, scratch_max, out_idx)
    return out_idx


def argmax_pair(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Row-wise ``(max_value, argmax_index)`` packed as a ``(M, 2)`` float32 tensor.

    The index is stored as ``float32`` — exact for any vocab below ``2**24``
    (16,777,216), which covers every shipped LLM vocab size. This is a legacy
    layout kept for tests / future logprob users; the runtime hot path uses
    :func:`argmax` instead and avoids the extra packing entirely.

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
    device = logits.device

    if out is None:
        out = torch.empty((M, 2), dtype=torch.float32, device=device)
    else:
        if out.shape != (M, 2):
            raise ValueError(
                f"out must have shape (M, 2)={M, 2}, got {tuple(out.shape)}"
            )
        if out.dtype != torch.float32 or out.device != device:
            raise ValueError("out must be float32 on the same device as logits")

    if not _supports_cute(N, logits.dtype):
        # Fallback: emulate the same (max, idx) packing using torch ops so the
        # downstream extraction path is uniform.
        max_vals, max_indices = torch.max(logits, dim=-1, keepdim=True)
        out[:, 0:1].copy_(max_vals.to(torch.float32))
        out[:, 1:2].copy_(max_indices.to(torch.float32))
        return out

    # Kernel writes into separate (M,) tensors; assemble into the legacy
    # (M, 2) layout for backward compatibility. This is off the runtime hot
    # path (callers use :func:`argmax` instead), so the extra copy/cast is OK.
    tmp_max = torch.empty((M,), dtype=torch.float32, device=device)
    tmp_idx = torch.empty((M,), dtype=torch.int64, device=device)
    _invoke_kernel(logits, tmp_max, tmp_idx)
    out[:, 0].copy_(tmp_max)
    out[:, 1].copy_(tmp_idx.to(torch.float32))
    return out


# Expose the underlying compiled-kernel entry as the registered impl so other
# code can probe `is _argmax_kernel_impl` style availability checks if needed.
if _CUTE_AVAILABLE:
    _argmax_kernel_impl = _invoke_kernel
