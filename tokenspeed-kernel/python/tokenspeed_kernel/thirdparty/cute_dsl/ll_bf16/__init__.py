# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Ported from vLLM's LLBf16Gemm; cutovers are measured here, not inherited.

"""Low-latency BF16 router GEMM: ``a @ b.T`` in FP32 for decode-sized M.

Two vendored CuTe kernels serve K3's router shape (``[M, 7168] x [896, 7168]``).
Measured on GB300, cold L2, us/call, against the cublas path they displace::

    M          1     2     4     8    16    24    32    64   128   512
    dot-prod  3.39  3.83  4.99  8.21 12.27     -     -     -     -     -
    split-K      -     -     -  4.93  4.94  7.55  7.44 14.98 25.75 90.1
    cublas    9.51  9.69  9.82  9.89  9.94  9.26  9.56  9.62 10.98  14.7

Hence ``MAX_M_DOTPROD = 4`` (split-K wins from 8; the dot product only carries
past 4 when split-K is absent) and ``MAX_M = 32`` (both lose to cublas above it,
a bound vLLM does not apply). ``(split_k, num_stages)`` is tuned here too:
vLLM's tables cover ``(K, N)`` from ``(4096, 256)`` to ``(7168, 384)`` and never
``(7168, 896)``, so they run K3 on a ``(6, 4)`` default that loses at every M
measured -- 5.75 / 8.74 / 9.19 us at M = 16 / 24 / 32 against the picks below.
"""

from __future__ import annotations

import threading
from typing import Any

import torch
from tokenspeed_kernel.platform import _pdl_enabled

MAX_M_DOTPROD = 4
MAX_M = 32
_BLOCK_SIZE_BY_M: dict[int, int] = {1: 256, 2: 256, 4: 256, 8: 128}
_DEFAULT_BLOCK_SIZE = 128
_SPLITK_CONFIG_BY_M: tuple[tuple[int, tuple[int, int]], ...] = (
    (16, (4, 4)),  # 4.94 us at M = 16, against 6.04 for (4, 2)
    (MAX_M, (4, 2)),  # 7.55 / 7.44 at M = 24 / 32, against 8.28 / 8.31
)
_SPLITK_TILE_N = 16
_SPLITK_TILE_K = 256
_SPLITK_DMA_WARPS = 4


def block_size_for(m: int) -> int:
    """Threads per output column that measured fastest at this token count."""
    return _BLOCK_SIZE_BY_M.get(m, _DEFAULT_BLOCK_SIZE)


def splitk_config_for(m: int) -> tuple[int, int]:
    """``(split_k, num_stages)`` that measured fastest at this token count."""
    for upper, config in _SPLITK_CONFIG_BY_M:
        if m <= upper:
            return config
    return _SPLITK_CONFIG_BY_M[-1][1]


def _cutedsl_available() -> bool:
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
        import quack.compile_utils  # noqa: F401
    except ImportError:
        return False
    return True


class LLBf16Router:
    """Compile-once driver for the two vendored CuTe router GEMM kernels."""

    def __init__(self) -> None:
        # Device-keyed: a callable compiled on one GPU must not run on another.
        self._dotprod: dict[tuple[int, int, int, int], Any] = {}
        self._splitk: dict[tuple[int, int, int], Any] = {}
        self._compile_lock = threading.Lock()
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Whether CuTe DSL is importable, memoized."""
        if self._available is None:
            self._available = _cutedsl_available()
        return self._available

    @staticmethod
    def _stream(device: torch.device):
        from cuda.bindings.driver import CUstream

        return CUstream(torch.cuda.current_stream(device).cuda_stream)

    @staticmethod
    def _use_pdl(device: torch.device) -> bool:
        return _pdl_enabled(torch.cuda.get_device_capability(device)[0] >= 9)

    def supports(self, a: torch.Tensor, b: torch.Tensor, m: int) -> bool:
        """Whether either backend can serve the given operands.

        Args:
            a: ``[M, K]`` activation; b: ``[N, K]`` weight.
            m: Token count.

        Returns:
            True when a vendored kernel is compilable and applicable here.
        """
        return (
            self.is_available()
            and m <= MAX_M
            and a.dtype is torch.bfloat16
            and b.dtype is torch.bfloat16
            and a.is_contiguous()
            and b.is_contiguous()
            # 128-bit vectorized bf16 loads need 16-byte aligned rows.
            and a.shape[1] % 8 == 0
            and a.shape[1] == b.shape[1]
            and a.device == b.device
            and a.device.type == "cuda"
            # Split-K reduces through DSMEM inside a thread block cluster.
            and (
                m <= MAX_M_DOTPROD or torch.cuda.get_device_capability(a.device)[0] >= 9
            )
        )

    def _compile_dotprod(
        self, m: int, k: int, block_size: int, device: torch.device
    ) -> None:
        import cutlass.cute as cute
        from cutlass import BFloat16, Float32
        from quack.compile_utils import make_fake_tensor
        from tokenspeed_kernel.thirdparty.cute_dsl.ll_bf16._kernel import (
            LLBf16Dotprod as _Kernel,
        )

        n = cute.sym_int()
        divisibility = 8 if k % 8 == 0 else 1
        a = make_fake_tensor(BFloat16, (m, k), divisibility=divisibility)
        b = make_fake_tensor(BFloat16, (n, k), divisibility=divisibility)
        c = make_fake_tensor(Float32, (m, n), divisibility=1)
        gemm = _Kernel(k=k, bs=block_size, use_pdl=self._use_pdl(device))
        # cute.compile targets the current device, not the operands'.
        with torch.cuda.device(device):
            self._dotprod[(device.index or 0, m, k, block_size)] = cute.compile(
                gemm,
                a,
                b,
                c,
                m,
                k,
                1,  # runtime N placeholder for the fake-tensor compile
                self._stream(device),
                options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
            )

    def _compile_splitk(self, config: tuple[int, int], device: torch.device) -> None:
        import cutlass.cute as cute
        from cutlass import BFloat16, Float32
        from quack.compile_utils import make_fake_tensor
        from tokenspeed_kernel.thirdparty.cute_dsl.ll_bf16._splitk_kernel import (
            LLBf16SplitK as _Kernel,
        )

        split_k, num_stages = config
        m, k, n = cute.sym_int(), cute.sym_int(), cute.sym_int()
        a = make_fake_tensor(BFloat16, (m, k), divisibility=8)
        b = make_fake_tensor(BFloat16, (n, k), divisibility=8)
        c = make_fake_tensor(Float32, (m, n), divisibility=1)
        gemm = _Kernel(
            tile_n=_SPLITK_TILE_N,
            tile_k=_SPLITK_TILE_K,
            num_stages=num_stages,
            num_dma_warps=_SPLITK_DMA_WARPS,
            split_k=split_k,
            use_pdl=self._use_pdl(device),
        )
        with torch.cuda.device(device):
            self._splitk[(device.index or 0, split_k, num_stages)] = cute.compile(
                gemm, a, b, c, self._stream(device), options="--enable-tvm-ffi"
            )

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        block_size: int | None = None,
    ) -> torch.Tensor:
        """Compute ``a @ b.T`` with FP32 accumulation and an FP32 result.

        The dot-product kernel serves ``M <= MAX_M_DOTPROD`` and the split-K
        kernel the rest up to ``MAX_M``.

        Args:
            a: ``[M, K]`` contiguous BF16 activation.
            b: ``[N, K]`` contiguous BF16 weight.
            out: Optional ``[M, N]`` FP32 destination; allocated when omitted.
            block_size: Dot-product threads per output column; the measured
                pick for ``M`` when omitted. Ignored by split-K.

        Returns:
            ``[M, N]`` FP32 tensor, ``out`` when it was given.
        """
        m, k = a.shape
        n = b.shape[0]
        if not self.supports(a, b, m):
            raise ValueError(
                f"ll_bf16 cannot serve M={m} K={k} dtypes={a.dtype}/{b.dtype}"
            )
        if out is None:
            out = torch.empty(m, n, dtype=torch.float32, device=a.device)
        elif out.shape != (m, n) or out.dtype is not torch.float32:
            raise ValueError(f"out must be a [{m}, {n}] float32 tensor")

        device = a.device
        stream = self._stream(device)
        if m <= MAX_M_DOTPROD:
            if block_size is None:
                block_size = block_size_for(m)
            key = (device.index or 0, m, k, block_size)
            if key not in self._dotprod:
                # Double-checked: cute.compile is expensive and not thread-safe.
                with self._compile_lock:
                    if key not in self._dotprod:
                        self._compile_dotprod(m, k, block_size, device)
            self._dotprod[key](a, b, out, n, stream)
            return out

        config = splitk_config_for(m)
        key = (device.index or 0, *config)
        if key not in self._splitk:
            with self._compile_lock:
                if key not in self._splitk:
                    self._compile_splitk(config, device)
        self._splitk[key](a, b, out, stream, 1.0)
        return out


ll_bf16_router = LLBf16Router()

__all__ = [
    "MAX_M",
    "MAX_M_DOTPROD",
    "LLBf16Router",
    "block_size_for",
    "splitk_config_for",
    "ll_bf16_router",
]
