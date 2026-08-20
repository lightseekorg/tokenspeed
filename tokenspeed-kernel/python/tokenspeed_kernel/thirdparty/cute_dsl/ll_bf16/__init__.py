# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Ported from vLLM's LLBf16Gemm: the dot-product driver only, with tokenspeed's
# stream/PDL helpers. The split-K backend vLLM uses above M == 4 is not ported.

"""Low-latency BF16 router GEMM: ``a @ b.T`` in FP32 for small-M decode."""

from __future__ import annotations

import threading
from typing import Any

import torch

# Measured on GB300 at K3's router shape ([M, 7168] x [896, 7168]), cold L2,
# against the cublas path this displaces (us/call): M=1 3.39 vs 5.52, M=2 3.83
# vs 7.06, M=4 4.99 vs 8.12, M=8 8.21 vs 10.04, M=16 12.27 vs 10.09. vLLM caps
# their dot-product backend at M == 4 because their split-K kernel takes over
# there; without that kernel the dot product is still the better of ours to 8.
MAX_M = 8
_BLOCK_SIZE_BY_M: dict[int, int] = {1: 256, 2: 256, 4: 256, 8: 128}
_DEFAULT_BLOCK_SIZE = 128


def block_size_for(m: int) -> int:
    """Threads per output column that measured fastest at this token count."""
    return _BLOCK_SIZE_BY_M.get(m, _DEFAULT_BLOCK_SIZE)


def _cutedsl_available() -> bool:
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
        import quack.compile_utils  # noqa: F401
    except ImportError:
        return False
    return True


class LLBf16Dotprod:
    """Compile-once-per-(M, K, block_size) driver for the vendored kernel."""

    def __init__(self) -> None:
        # Device-keyed: a callable compiled on one GPU must not run on another.
        self._compiled: dict[tuple[int, int, int, int], Any] = {}
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
        return torch.cuda.get_device_capability(device)[0] >= 9

    def supports(self, a: torch.Tensor, b: torch.Tensor, m: int) -> bool:
        """Whether this driver can serve the given operands.

        Args:
            a: ``[M, K]`` activation; b: ``[N, K]`` weight.
            m: Token count, which the kernel bakes in as a constant.

        Returns:
            True when the kernel is compilable and applicable here.
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
        )

    def _compile(self, m: int, k: int, block_size: int, device: torch.device) -> None:
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
        self._compiled[(device.index or 0, m, k, block_size)] = cute.compile(
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

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        block_size: int | None = None,
    ) -> torch.Tensor:
        """Compute ``a @ b.T`` with FP32 accumulation and an FP32 result.

        Args:
            a: ``[M, K]`` contiguous BF16 activation.
            b: ``[N, K]`` contiguous BF16 weight.
            out: Optional ``[M, N]`` FP32 destination; allocated when omitted.
            block_size: Threads cooperating on one output column; the measured
                pick for ``M`` when omitted.

        Returns:
            ``[M, N]`` FP32 tensor, ``out`` when it was given.
        """
        m, k = a.shape
        n = b.shape[0]
        if block_size is None:
            block_size = block_size_for(m)
        if not self.supports(a, b, m):
            raise ValueError(
                f"ll_bf16 dotprod cannot serve M={m} K={k} "
                f"dtypes={a.dtype}/{b.dtype}"
            )
        if out is None:
            out = torch.empty(m, n, dtype=torch.float32, device=a.device)
        elif out.shape != (m, n) or out.dtype is not torch.float32:
            raise ValueError(f"out must be a [{m}, {n}] float32 tensor")

        key = (a.device.index or 0, m, k, block_size)
        if key not in self._compiled:
            # Double-checked: cute.compile is expensive and not thread-safe.
            with self._compile_lock:
                if key not in self._compiled:
                    self._compile(m, k, block_size, a.device)
        self._compiled[key](a, b, out, n, self._stream(a.device))
        return out


ll_bf16_dotprod = LLBf16Dotprod()

__all__ = ["MAX_M", "LLBf16Dotprod", "block_size_for", "ll_bf16_dotprod"]
