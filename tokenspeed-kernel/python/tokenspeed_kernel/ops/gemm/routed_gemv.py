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

"""Measured decode-GEMV routing for Kimi-K3's projection shapes.

Every entry in ``MEASURED_ROUTE`` was measured on GB300 (sm103) at the exact
(N, K) the TP8 decode path hands ``decode_gemv`` -- extracted from an nsys
trace of serving, not assumed -- with the tuner cycling eight weight copies
so the L2 never holds the operand between calls, the way serving streams a
different layer's weight each launch. Hot-cache numbers ranked backends
wrongly (1.9x off serving at 6288x7168); cold-L2 reproduces serving per-shape
times within ~5%. ``test/gemm_tuning/tune_route.py`` reproduces the sweep.

A backend earns an entry only by beating the incumbent selection by at least
4% -- above measurement noise, so a noise-level lead does not become a
maintenance obligation. Shapes not listed keep the selection they had
(rowcta at M == 1, torch.mm otherwise). The table is data, not policy:
re-run the sweep on new hardware or after a kernel change and replace the
literals wholesale.
"""

from __future__ import annotations

import functools
import os
import threading
from types import MappingProxyType

import torch
from tokenspeed_kernel.ops.gemm.triton_gemv import _torch_decode_gemv
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

# (m, n, k) -> backend; immutable so the registry's import-time view and the
# wrappers' per-call view cannot diverge. Measured; see module docstring.
MEASURED_ROUTE: MappingProxyType[tuple[int, int, int], str] = MappingProxyType(
    {
        (1, 3584, 7168): "skinny",  # MoE latent down-proj, 92 calls/step
        # The three shard families below are GB300-measured (2026-08-21);
        # GB200 owes the same tune_route.py re-sweep the other entries got.
        # KDA o_proj shard, 69 calls/step: tgv 2.83 vs cublas 5.98 (2.11x)
        (1, 7168, 1536): "tgv",
        (2, 7168, 1536): "tgv",
        (4, 7168, 1536): "tgv",
        (8, 7168, 1536): "tgv",
        # Shared gate_up shard, 92 calls/step: skinny 5.56 vs cublas 9.13 (1.64x)
        (1, 1536, 7168): "skinny",
        (2, 1536, 7168): "skinny",
        (4, 1536, 7168): "skinny",
        # Shared down shard, 92 calls/step: tgv 2.52 vs cublas 2.99 (1.19x)
        (1, 7168, 768): "tgv",
        (2, 7168, 768): "tgv",
        (4, 7168, 768): "tgv",
        (8, 7168, 768): "tgv",
        (1, 6288, 7168): "skinny",  # KDA in_proj (qkvgfab), 69 calls/step
        (1, 3648, 7168): "skinny",  # MLA fused qkv_a + gate, 24 calls/step
        # (1, 2304, 1536) mla_q_b stays on rowcta: 2.48 vs skinny 2.53.
        # M > 1 (small batches, speculative verify) vs the cublas incumbent.
        (2, 3584, 7168): "skinny",  # 9.20 vs 11.67 (1.27x)
        (4, 3584, 7168): "skinny",  # 10.38 vs 11.21 (1.08x)
        (2, 6288, 7168): "tgv",  # 15.29 vs 17.07 (1.12x)
        (4, 6288, 7168): "tgv",  # 15.25 vs 17.26 (1.13x)
        (8, 6288, 7168): "tgv",  # 15.38 vs 16.75 (1.09x)
        (2, 3648, 7168): "skinny",  # 9.33 vs 11.58 (1.24x)
        (4, 3648, 7168): "skinny",  # 10.40 vs 11.51 (1.11x)
        (8, 3648, 7168): "tgv",  # 10.65 vs 11.47 (1.08x)
        (2, 2304, 1536): "skinny",  # 2.52 vs 4.97 (1.97x)
        (4, 2304, 1536): "tgv",  # 2.86 vs 4.58 (1.60x)
        (8, 2304, 1536): "tgv",  # 2.86 vs 4.45 (1.56x)
        # TP16 (GB200, sm100). The shapes were read off the decode_gemv
        # dispatch during a live bs=1 run rather than scaled from TP8: only
        # 1152x1536 halves, 3584x7168 is not TP-sharded at all, 2880x7168 has
        # no TP8 analogue, and TP8's largest entry (6288x7168) never reaches
        # decode_gemv here. Same >= 4% margin, same cold-L2 tuner.
        (3, 3584, 7168): "skinny",  # 9.54 vs 11.23 (1.18x)
        (1, 2880, 7168): "skinny",  # 7.13 vs rowcta 8.61 (1.21x)
        (2, 2880, 7168): "skinny",  # 7.95 vs 9.80 (1.23x)
        (3, 2880, 7168): "skinny",  # 8.24 vs 9.77 (1.19x)
        (4, 2880, 7168): "skinny",  # 9.32 vs 9.78 (1.05x)
        # 3584 and 2880 stay on the incumbent at M >= 5: skinny inverts there
        # (15.85 vs cublas 11.06 at M=8, 3584x7168), so the win is not simply
        # "skinny for small M" and the boundary has to be measured per width.
        (2, 1152, 1536): "skinny",  # 1.99 vs 4.85 (2.44x)
        (3, 1152, 1536): "skinny",  # 2.10 vs 4.47 (2.13x)
        (4, 1152, 1536): "skinny",  # 2.24 vs 4.40 (1.97x)
        (5, 1152, 1536): "skinny",  # 2.23 vs 4.40 (1.98x)
        (6, 1152, 1536): "skinny",  # 2.35 vs 4.42 (1.89x)
        (7, 1152, 1536): "skinny",  # 2.59 vs 4.45 (1.72x)
        (8, 1152, 1536): "skinny",  # 2.72 vs 4.46 (1.64x)
        # (1, 1152, 1536) stays on rowcta: 1.907 vs 1.943, inside the margin.
    }
)

_BF16_SIG = frozenset(
    {
        format_signature(
            x=dense_tensor_format(torch.bfloat16),
            weight=dense_tensor_format(torch.bfloat16),
        )
    }
)
# sm100 (GB200/B200) and sm103 (GB300) both run these kernels -- the skinny
# GEMM uses only generic CuTe primitives, and the sm103 gate recorded where the
# sweep had been run, not what the kernels require. Re-swept on GB200: the
# sm103 winners reproduce entry for entry at the TP8 shapes.
_CAPABILITY = CapabilityRequirement(
    min_arch_version=ArchVersion(10, 0),
    vendors=frozenset({"nvidia"}),
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


@functools.lru_cache(maxsize=32)
def _skinny_config(m: int, n: int, k: int):
    from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm import (
        shape_dynamic_skinny_gemm,
    )

    return shape_dynamic_skinny_gemm.default_config(m, n, k)


def skinny_gemv(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T`` via the vendored CuTe skinny GEMM.

    Caller-owned invariant, as for every ``decode_gemv`` backend: ``out`` must
    not overlap ``x`` or ``weight`` storage.

    Args:
        x: ``[M, K]`` contiguous bf16 activations.
        weight: ``[N, K]`` contiguous bf16 weight.
        out: optional ``[M, N]`` destination.

    Returns:
        ``[M, N]`` output in ``x``'s dtype.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm import (
        shape_dynamic_skinny_gemm,
    )

    m, k = x.shape
    n = weight.shape[0]
    dev = x.device.index or 0
    # Table shapes only; anything else would grow the compile cache unbounded.
    if (
        MEASURED_ROUTE.get((m, n, k)) != "skinny"
        or x.dtype != torch.bfloat16
        or not _usable_in_capture("skinny", dev, m, n, k)
    ):
        return _torch_decode_gemv(x, weight, out)
    config = _skinny_config(m, n, k)
    # default_config can emit a config supports() rejects; fall back, don't raise.
    if not shape_dynamic_skinny_gemm.supports(config, m, n, k):
        return _torch_decode_gemv(x, weight, out)
    # DLPack refuses requires_grad tensors; detach is a zero-copy view.
    result = shape_dynamic_skinny_gemm(x.detach(), weight.detach(), config, out=out)
    _mark_warmed("skinny", dev, m, n, k)
    return result


# TGV requires a bias; the routed GEMVs have none. Never an evicting cache: a
# captured graph replays against these exact tensors.
_tgv_biases: dict[tuple[int, int], torch.Tensor] = {}
_tgv_bias_lock = threading.Lock()


def _tgv_bias(n: int, device_index: int) -> torch.Tensor:
    key = (n, device_index)
    bias = _tgv_biases.get(key)
    if bias is None:
        with _tgv_bias_lock:
            # Re-check: a racing thread must not replace a graph-held entry.
            bias = _tgv_biases.get(key)
            if bias is None:
                bias = torch.zeros(
                    n, device=f"cuda:{device_index}", dtype=torch.bfloat16
                )
                _tgv_biases[key] = bias
    return bias


def tgv_gemv(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """``x @ weight.T`` via FlashInfer's TGV low-latency GEMM.

    Args:
        x: ``[M, K]`` contiguous bf16 activations.
        weight: ``[N, K]`` contiguous bf16 weight; its transpose is the
            column-major ``(K, N)`` layout TGV wants, with no copy.
        out: optional ``[M, N]`` destination.

    Returns:
        ``[M, N]`` output in ``x``'s dtype.
    """
    from flashinfer import mm_bf16

    m, k = x.shape
    n = weight.shape[0]
    dev = x.device.index or 0
    if (
        MEASURED_ROUTE.get((m, n, k)) != "tgv"
        or x.dtype != torch.bfloat16
        or not _usable_in_capture("tgv", dev, m, n, k)
    ):
        return _torch_decode_gemv(x, weight, out)
    bias = _tgv_bias(n, dev)
    # TGV is CuTe DSL inside FlashInfer: same DLPack no-grad rule.
    pdl = os.environ.get("TOKENSPEED_DISABLE_PDL") != "1"
    result = mm_bf16(
        x.detach(), weight.detach().t(), bias=bias, pdl=pdl, backend="tgv", out=out
    )
    _mark_warmed("tgv", dev, m, n, k)
    return result


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


@functools.lru_cache(maxsize=8)
def decode_gemv_routed(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether :data:`MEASURED_ROUTE` covers this call on this platform.

    Args:
        x: ``[M, K]`` activation.
        weight: ``[N, K]`` weight.

    Returns:
        True when ``decode_gemv`` would reach a measured backend rather than
        the portable fallback.
    """
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or x.ndim != 2
    ):
        return False
    m, k = x.shape
    return (m, weight.shape[0], k) in MEASURED_ROUTE and _is_measured_arch(
        x.device.index or 0
    )


def _is_measured_arch(device_index: int) -> bool:
    from tokenspeed_kernel.platform import current_platform

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
    from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm import (
        SkinnyGemmConfig,
        shape_dynamic_skinny_gemm,
    )

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


def _register_route() -> None:
    impls = {"skinny": skinny_gemv, "tgv": tgv_gemv}
    for (m, n, k), backend in MEASURED_ROUTE.items():
        register_kernel(
            "gemm",
            "decode_gemv",
            name=f"{backend}_gemv_m{m}_n{n}_k{k}",
            solution="cute_dsl" if backend == "skinny" else "flashinfer",
            capability=_CAPABILITY,
            signatures=_BF16_SIG,
            traits={
                "m": frozenset({m}),
                "n": frozenset({n}),
                "k": frozenset({k}),
            },
            # Above the M == 1 rowcta spec so a measured win takes the shape.
            priority=Priority.SPECIALIZED + 2,
        )(impls[backend])


_register_route()
