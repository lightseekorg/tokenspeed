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

"""BF16 projection eligibility and the independent fused add3 epilogue."""

from __future__ import annotations

import functools
import threading
from types import MappingProxyType

import torch
from tokenspeed_kernel.ops.gemm.flashinfer import (
    BF16_GEMM_MAX_M,
    flashinfer_joint_bf16_supported,
)
from tokenspeed_kernel.ops.gemm.triton_gemv import _select, torch_decode_gemv
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)

# Shapes an eager call has already compiled and allocated for. Capture must
# not JIT or allocate, so an unwarmed shape falls back to torch.mm there.
_warmed: set[tuple[str, int, int, int, int]] = set()
_warmed_lock = threading.Lock()


def _usable_in_capture(backend: str, dev: int, m: int, n: int, k: int) -> bool:
    # Device-keyed: warmth on one GPU says nothing about another's modules.
    return (
        not torch.cuda.is_current_stream_capturing()
        or (backend, dev, m, n, k) in _warmed
    )


def _mark_warmed(backend: str, dev: int, m: int, n: int, k: int) -> None:
    # Only a successful eager call earns capture trust.
    if not torch.cuda.is_current_stream_capturing():
        with _warmed_lock:
            _warmed.add((backend, dev, m, n, k))


# Fused ``a + x @ W.T + c`` (K3 MoE latent up-proj epilogue): (m, n, k) ->
# (block_size, outputs_per_block, k_unroll). Cold-L2 vs the incumbent: M == 1
# 8.86us vs rowcta_gemv_add3 9.42, M == 2 10.05 vs composed 12.81. M == 4 was
# 1.04x, under the margin, so it keeps the composed path.
ADD3_ROUTE: MappingProxyType[tuple[int, int, int], tuple[int, int, int]] = (
    MappingProxyType(
        {
            (1, 7168, 3584): (64, 4, 2),
            (2, 7168, 3584): (64, 7, 2),
        }
    )
)


def decode_gemv_routed(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Use joint FI on its supported range, or the registered CDNA5 path."""
    if (
        flashinfer_joint_bf16_supported(x, weight, None)
        and x.shape[0] <= BF16_GEMM_MAX_M
    ):
        return True
    if (
        not x.is_cuda
        or x.ndim != 2
        or weight.ndim != 2
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        return False
    m, k = x.shape
    if not current_platform().is_cdna5 or k < 256:
        return False
    return _select(m, weight.shape[0], k, True) is not torch_decode_gemv


@functools.lru_cache(maxsize=8)
def _is_measured_arch(device_index: int) -> bool:
    """ADD3_ROUTE's arch floor: its configs were only swept on sm103."""

    platform = current_platform()
    if platform.vendor != "nvidia":
        return False
    return torch.cuda.get_device_capability(device_index) >= (10, 3)


def skinny_add3_supported(m: int, n: int, k: int, device: torch.device) -> bool:
    """Whether :func:`skinny_gemv_add3` has a measured config for this call.

    Args:
        m/n/k: the projection extents (``x[M, K] @ W[N, K].T``).
        device: the CUDA device the call would run on.

    Returns:
        True only for table shapes on the measured architecture.
    """
    return (m, n, k) in ADD3_ROUTE and _is_measured_arch(device.index or 0)


def _composed_add3(
    x: torch.Tensor,
    weight: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    out: torch.Tensor | None,
) -> torch.Tensor:
    result = torch.addmm(c, x, weight.t())
    result += a
    if out is not None:
        out.copy_(result)
        return out
    return result


def skinny_gemv_add3(
    x: torch.Tensor,
    weight: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``a + x @ weight.T + c`` via the skinny GEMM's dual-residual epilogue.

    Args:
        x: ``[M, K]`` contiguous bf16 activations.
        weight: ``[N, K]`` contiguous bf16 weight.
        a/c: ``[M, N]`` addends with unit inner stride (row stride free, so a
            column slice of a wider tensor is accepted).
        out: optional ``[M, N]`` destination.

    Returns:
        ``[M, N]`` result in ``x``'s dtype.
    """

    m, k = x.shape
    n = weight.shape[0]
    dev = x.device.index or 0
    tuned = ADD3_ROUTE.get((m, n, k))
    if (
        tuned is None
        or x.dtype != torch.bfloat16
        or not _is_measured_arch(dev)
        or not _usable_in_capture("skinny_add3", dev, m, n, k)
    ):
        return _composed_add3(x, weight, a, c, out)
    config = SkinnyGemmConfig(m, *tuned)
    if not shape_dynamic_skinny_gemm.supports(config, m, n, k):
        return _composed_add3(x, weight, a, c, out)
    result = shape_dynamic_skinny_gemm(
        x.detach(),
        weight.detach(),
        config,
        residual=a.detach(),
        residual2=c.detach(),
        out=out,
    )
    _mark_warmed("skinny_add3", dev, m, n, k)
    return result
