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

Platform support:

The CuTe DSL kernel ships only for NVIDIA Hopper/Blackwell (sm_90..<sm_120).
On every other target — AMD ROCm, CPU-only, unsupported SM, missing
``nvidia-cutlass-dsl`` — the public ``argmax`` / ``argmax_pair`` names are
bound at import time to pure-torch fallback implementations, so callers don't
need to test for kernel availability. The cute-DSL imports are gated behind
``_ARCH_SUPPORTED`` and never executed on non-NVIDIA hosts.
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


def _validate_argmax_out(logits: torch.Tensor, out: torch.Tensor) -> None:
    if out.shape != (logits.shape[0],):
        raise ValueError(
            f"out must have shape (M,)={(logits.shape[0],)}, got {tuple(out.shape)}"
        )
    if out.dtype not in _SUPPORTED_OUT_DTYPES:
        raise ValueError(f"out must be int32 or int64; got {out.dtype}")
    if out.device != logits.device:
        raise ValueError("out must be on the same device as logits")


def _validate_argmax_pair_out(logits: torch.Tensor, out: torch.Tensor) -> None:
    M = logits.shape[0]
    if out.shape != (M, 2):
        raise ValueError(f"out must have shape (M, 2)={M, 2}, got {tuple(out.shape)}")
    if out.dtype != torch.float32 or out.device != logits.device:
        raise ValueError("out must be float32 on the same device as logits")


def _argmax_torch_fallback(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-torch implementation of :func:`argmax`.

    Selected at import time on non-NVIDIA / unsupported-SM hosts (AMD ROCm,
    CPU-only, sm_80, sm_120+, missing ``nvidia-cutlass-dsl``). Also reached
    per-call from the cute path when the input fails the kernel's
    preconditions (1D / non-CUDA / fp16 / bf16 / small N / unaligned N).
    """
    if out is not None:
        _validate_argmax_out(logits, out)
    result = torch.argmax(logits, dim=-1)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _argmax_pair_torch_fallback(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure-torch implementation of :func:`argmax_pair`.

    Selected at import time on non-NVIDIA / unsupported-SM hosts, and reached
    per-call from the cute path when the input fails the kernel's
    preconditions.
    """
    if logits.dim() != 2:
        raise ValueError(f"argmax_pair expects 2D input, got {logits.dim()}D")
    M = logits.shape[0]
    device = logits.device
    if out is None:
        out = torch.empty((M, 2), dtype=torch.float32, device=device)
    else:
        _validate_argmax_pair_out(logits, out)

    max_vals, max_indices = torch.max(logits, dim=-1, keepdim=True)
    out[:, 0:1].copy_(max_vals.to(torch.float32))
    out[:, 1:2].copy_(max_indices.to(torch.float32))
    return out


def _argmax_cute(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """CuTe DSL fast path for argmax.

    Falls back per-call to :func:`_argmax_torch_fallback` when the input
    isn't kernel-eligible (1D / non-CUDA / unsupported dtype / small N /
    unaligned N). Only ever bound to the public ``argmax`` name on NVIDIA
    hosts with the cute DSL Python packages available — see the module-level
    dispatch below.
    """
    if out is not None:
        _validate_argmax_out(logits, out)

    if (
        logits.dim() != 2
        or not logits.is_cuda
        or not _supports_cute(logits.shape[1], logits.dtype)
    ):
        return _argmax_torch_fallback(logits, out=out)

    M = logits.shape[0]
    device = logits.device
    out_idx = (
        out if out is not None else torch.empty((M,), dtype=torch.int64, device=device)
    )

    # The max value is needed only inside the kernel reduction; the caller
    # never sees it. Allocate a scratch buffer so the kernel has somewhere to
    # write it.
    scratch_max = torch.empty((M,), dtype=torch.float32, device=device)
    _invoke_kernel(logits, scratch_max, out_idx)
    return out_idx


def _argmax_pair_cute(
    logits: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """CuTe DSL fast path for argmax_pair. Falls back per-call when needed."""
    if logits.dim() != 2:
        raise ValueError(f"argmax_pair expects 2D input, got {logits.dim()}D")

    M, N = logits.shape
    device = logits.device

    if out is None:
        out = torch.empty((M, 2), dtype=torch.float32, device=device)
    else:
        _validate_argmax_pair_out(logits, out)

    if not logits.is_cuda or not _supports_cute(N, logits.dtype):
        # Reuse the pure-torch packing path; pass our pre-allocated buffer so
        # the caller-supplied ``out`` is honored.
        return _argmax_pair_torch_fallback(logits, out=out)

    # Kernel writes into separate (M,) tensors; assemble into the legacy
    # (M, 2) layout for backward compatibility. This is off the runtime hot
    # path (callers use :func:`argmax` instead), so the extra copy/cast is OK.
    tmp_max = torch.empty((M,), dtype=torch.float32, device=device)
    tmp_idx = torch.empty((M,), dtype=torch.int64, device=device)
    _invoke_kernel(logits, tmp_max, tmp_idx)
    out[:, 0].copy_(tmp_max)
    out[:, 1].copy_(tmp_idx.to(torch.float32))
    return out


# Public API binding. On NVIDIA + cute-DSL-installed hosts the fast path is
# selected; everywhere else the public names refer to pure-torch
# implementations so callers don't need to test for availability. Mirrors the
# pattern used in ``tokenspeed_kernel.ops.sampling.cuda``.
if _CUTE_AVAILABLE:
    argmax = _argmax_cute
    argmax_pair = _argmax_pair_cute
    _argmax_kernel_impl = _invoke_kernel
else:
    argmax = _argmax_torch_fallback
    argmax_pair = _argmax_pair_torch_fallback
