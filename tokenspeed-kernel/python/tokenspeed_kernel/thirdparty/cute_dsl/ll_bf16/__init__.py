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
from tokenspeed_kernel.platform import pdl_enabled

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
OUT_DTYPES: tuple[torch.dtype, ...] = (torch.float32, torch.bfloat16)
_DotprodCacheKey = tuple[int, int, int, int, torch.dtype, bool, bool]
_SplitKCacheKey = tuple[int, int, int, torch.dtype, bool, bool]


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


def _cutlass_dtype(out_dtype: torch.dtype):
    from cutlass import BFloat16, Float32

    return BFloat16 if out_dtype is torch.bfloat16 else Float32


class LLBf16Router:
    """Compile-once driver for the two vendored CuTe router GEMM kernels."""

    def __init__(self) -> None:
        # Device-keyed: a callable compiled on one GPU must not run on another.
        # PDL is part of the compiled kernel, so the process-wide policy belongs
        # in the cache key as well.
        self._dotprod: dict[_DotprodCacheKey, Any] = {}
        self._splitk: dict[_SplitKCacheKey, Any] = {}
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
        self,
        m: int,
        k: int,
        block_size: int,
        device: torch.device,
        out_dtype: torch.dtype,
        has_bias: bool,
        enable_pdl: bool,
    ) -> None:
        import cutlass.cute as cute
        from cutlass import BFloat16
        from quack.compile_utils import make_fake_tensor
        from tokenspeed_kernel.thirdparty.cute_dsl.ll_bf16._kernel import (
            LLBf16Dotprod as _Kernel,
        )

        n = cute.sym_int()
        divisibility = 8 if k % 8 == 0 else 1
        cut_out_dtype = _cutlass_dtype(out_dtype)
        a = make_fake_tensor(BFloat16, (m, k), divisibility=divisibility)
        b = make_fake_tensor(BFloat16, (n, k), divisibility=divisibility)
        c = make_fake_tensor(cut_out_dtype, (m, n), divisibility=1)
        bias = make_fake_tensor(cut_out_dtype, (n,), divisibility=1)
        gemm = _Kernel(
            k=k,
            bs=block_size,
            use_pdl=enable_pdl,
            out_dtype=cut_out_dtype,
            has_bias=has_bias,
        )
        # cute.compile targets the current device, not the operands'.
        with torch.cuda.device(device):
            key = (
                device.index or 0,
                m,
                k,
                block_size,
                out_dtype,
                has_bias,
                enable_pdl,
            )
            self._dotprod[key] = cute.compile(
                gemm,
                a,
                b,
                c,
                bias,
                m,
                k,
                1,  # runtime N placeholder for the fake-tensor compile
                self._stream(device),
                options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
            )

    def _compile_splitk(
        self,
        config: tuple[int, int],
        device: torch.device,
        out_dtype: torch.dtype,
        has_bias: bool,
        enable_pdl: bool,
    ) -> None:
        import cutlass.cute as cute
        from cutlass import BFloat16
        from quack.compile_utils import make_fake_tensor
        from tokenspeed_kernel.thirdparty.cute_dsl.ll_bf16._splitk_kernel import (
            LLBf16SplitK as _Kernel,
        )

        split_k, num_stages = config
        m, k, n = cute.sym_int(), cute.sym_int(), cute.sym_int()
        cut_out_dtype = _cutlass_dtype(out_dtype)
        a = make_fake_tensor(BFloat16, (m, k), divisibility=8)
        b = make_fake_tensor(BFloat16, (n, k), divisibility=8)
        c = make_fake_tensor(cut_out_dtype, (m, n), divisibility=1)
        bias = make_fake_tensor(cut_out_dtype, (n,), divisibility=1)
        gemm = _Kernel(
            tile_n=_SPLITK_TILE_N,
            tile_k=_SPLITK_TILE_K,
            num_stages=num_stages,
            num_dma_warps=_SPLITK_DMA_WARPS,
            split_k=split_k,
            use_pdl=enable_pdl,
            out_dtype=cut_out_dtype,
            has_bias=has_bias,
        )
        with torch.cuda.device(device):
            key = (
                device.index or 0,
                split_k,
                num_stages,
                out_dtype,
                has_bias,
                enable_pdl,
            )
            self._splitk[key] = cute.compile(
                gemm, a, b, c, bias, self._stream(device), options="--enable-tvm-ffi"
            )

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        bias: torch.Tensor | None = None,
        out_dtype: torch.dtype | None = None,
        block_size: int | None = None,
    ) -> torch.Tensor:
        """Compute ``a @ b.T (+ bias)`` with FP32 accumulation.

        The dot-product kernel serves ``M <= MAX_M_DOTPROD`` and the split-K
        kernel the rest up to ``MAX_M``.

        Args:
            a: ``[M, K]`` contiguous BF16 activation.
            b: ``[N, K]`` contiguous BF16 weight.
            out: Optional ``[M, N]`` destination in ``out_dtype``; allocated
                when omitted.
            bias: Optional contiguous ``[N]`` bias in ``out_dtype``, added in
                the epilogue.
            out_dtype: Output element type, one of :data:`OUT_DTYPES`. Defaults
                to ``out``'s dtype, or float32 when ``out`` is omitted too.
            block_size: Dot-product threads per output column; the measured
                pick for ``M`` when omitted. Ignored by split-K.

        Returns:
            ``[M, N]`` tensor in ``out_dtype``, ``out`` when it was given.
        """
        m, k = a.shape
        n = b.shape[0]
        if not self.supports(a, b, m):
            raise ValueError(
                f"ll_bf16 cannot serve M={m} K={k} dtypes={a.dtype}/{b.dtype}"
            )
        if out_dtype is None:
            out_dtype = torch.float32 if out is None else out.dtype
        if out_dtype not in OUT_DTYPES:
            raise ValueError(f"out_dtype must be one of {OUT_DTYPES}, got {out_dtype}")
        if out is None:
            out = torch.empty(m, n, dtype=out_dtype, device=a.device)
        elif out.shape != (m, n) or out.dtype is not out_dtype:
            raise ValueError(f"out must be a [{m}, {n}] {out_dtype} tensor")
        if bias is not None and (
            bias.dtype is not out_dtype
            or bias.shape != (n,)
            or not bias.is_contiguous()
            or bias.device != a.device
        ):
            raise ValueError(f"bias must be a contiguous [{n}] {out_dtype} tensor")
        # The kernels always take a bias operand and read it only under has_bias.
        bias_arg = out[0] if bias is None else bias
        has_bias = bias is not None

        device = a.device
        stream = self._stream(device)
        enable_pdl = pdl_enabled() and torch.cuda.get_device_capability(device)[0] >= 9
        # Every split-K cluster rank must own at least one K tile. Router
        # projections are normally wide enough, but low-rank consumers (for
        # example a padded rank-320 hyperconnection up projection) are not.
        # Route those shapes through the CTA-local dot-product kernel instead
        # of launching a cluster whose idle ranks can never complete.
        config = splitk_config_for(m)
        split_k, _ = config
        use_dotprod = m <= MAX_M_DOTPROD or k < split_k * _SPLITK_TILE_K
        if use_dotprod:
            if block_size is None:
                block_size = block_size_for(m)
            key = (
                device.index or 0,
                m,
                k,
                block_size,
                out_dtype,
                has_bias,
                enable_pdl,
            )
            if key not in self._dotprod:
                # Double-checked: cute.compile is expensive and not thread-safe.
                with self._compile_lock:
                    if key not in self._dotprod:
                        self._compile_dotprod(
                            m,
                            k,
                            block_size,
                            device,
                            out_dtype,
                            has_bias,
                            enable_pdl,
                        )
            self._dotprod[key](a, b, out, bias_arg, n, stream)
            return out

        key = (device.index or 0, *config, out_dtype, has_bias, enable_pdl)
        if key not in self._splitk:
            with self._compile_lock:
                if key not in self._splitk:
                    self._compile_splitk(
                        config, device, out_dtype, has_bias, enable_pdl
                    )
        self._splitk[key](a, b, out, bias_arg, stream, 1.0)
        return out


ll_bf16_router = LLBf16Router()

__all__ = [
    "MAX_M",
    "MAX_M_DOTPROD",
    "OUT_DTYPES",
    "LLBf16Router",
    "block_size_for",
    "splitk_config_for",
    "ll_bf16_router",
]
