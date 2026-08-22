# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Ported from vLLM's ShapeDynamicSkinnyGemm: same compile cache and argument
# checks, tokenspeed's stream/PDL utilities, no warmup-provider registration.

"""Shape-dynamic skinny GEMM: ``a @ b.T`` for small M decode activations."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import torch

MAX_M = 16


@dataclass(frozen=True, slots=True)
class SkinnyGemmConfig:
    """One compiled specialization of the skinny GEMM.

    Args:
        num_rows: M the kernel is specialized for; must equal the call's M.
        block_size: threads cooperating on one output column.
        outputs_per_block: output columns per block; N must divide by it.
        k_unroll: K-loop unroll factor.
        vector_width: elements per vectorized load.
        static_k: bake K in as a constant when set, which the widest
            configurations need to keep their unrolled K loop.
    """

    num_rows: int
    block_size: int
    outputs_per_block: int
    k_unroll: int = 1
    vector_width: int = 8
    static_k: int | None = None


def _cutedsl_available() -> bool:
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
    except ImportError:
        return False
    return True


class ShapeDynamicSkinnyGemm:
    """Compile-once-per-(dtype, config) driver for the vendored CuTe kernel."""

    def __init__(self) -> None:
        # Device-keyed: a callable compiled on one GPU must not run on another.
        self._compiled: dict[
            tuple[int, torch.dtype, SkinnyGemmConfig, bool, bool], Any
        ] = {}
        self._compile_lock = threading.Lock()
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Whether CuTe DSL is importable, memoized."""
        if self._available is None:
            self._available = _cutedsl_available()
        return self._available

    @staticmethod
    def _cutlass_dtype(dtype: torch.dtype):
        from cutlass import BFloat16, Float16

        return BFloat16 if dtype == torch.bfloat16 else Float16

    @staticmethod
    def _stream(device: torch.device):
        from cuda.bindings.driver import CUstream

        return CUstream(torch.cuda.current_stream(device).cuda_stream)

    @staticmethod
    def _use_pdl(device: torch.device) -> bool:
        return torch.cuda.get_device_capability(device)[0] >= 9

    def _compile(
        self,
        dtype: torch.dtype,
        config: SkinnyGemmConfig,
        has_residual: bool,
        has_residual2: bool,
        device: torch.device,
    ) -> None:
        import cutlass.cute as cute
        from quack.compile_utils import make_fake_tensor
        from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm._kernel import (
            CuteSkinnyGemm,
        )

        element_type = self._cutlass_dtype(dtype)
        n = cute.sym_int(divisibility=config.outputs_per_block)
        k = (
            config.static_k
            if config.static_k is not None
            else cute.sym_int(divisibility=config.block_size * config.vector_width)
        )
        a = make_fake_tensor(
            element_type, (config.num_rows, k), divisibility=config.vector_width
        )
        b = make_fake_tensor(element_type, (n, k), divisibility=config.vector_width)
        c = make_fake_tensor(element_type, (config.num_rows, n), divisibility=1)
        residual = make_fake_tensor(element_type, (config.num_rows, n), divisibility=1)
        residual2 = make_fake_tensor(element_type, (config.num_rows, n), divisibility=1)
        kernel = CuteSkinnyGemm(
            element_type=element_type,
            num_rows=config.num_rows,
            block_size=config.block_size,
            outputs_per_block=config.outputs_per_block,
            vector_width=config.vector_width,
            k_unroll=config.k_unroll,
            has_residual=has_residual,
            has_residual2=has_residual2,
            use_pdl=self._use_pdl(device),
            static_k=config.static_k,
        )
        with torch.cuda.device(device):
            compiled = cute.compile(
                kernel,
                a,
                b,
                residual,
                residual2,
                c,
                self._stream(device),
                options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
            )
        self._compiled[
            (device.index or 0, dtype, config, has_residual, has_residual2)
        ] = compiled

    @staticmethod
    def default_config(m: int, n: int, k: int) -> SkinnyGemmConfig:
        """vLLM's shape heuristic, used when no measured config is on file."""
        wide_block = 224
        if m == 1 and k >= 7168 and k % (wide_block * 8) == 0:
            if n % 3 == 0:
                return SkinnyGemmConfig(m, wide_block, 3, 2 if n <= 2304 else 4)
            if 2304 < n < 4096 and n % 2 == 0:
                return SkinnyGemmConfig(m, wide_block, 2, k_unroll=4)

        if k <= 2048 or k % (128 * 8) != 0:
            outputs_per_block = 2 if k <= 2048 else 4
            if n % outputs_per_block:
                outputs_per_block = 1
            if k % (64 * 8) == 0:
                return SkinnyGemmConfig(m, 64, outputs_per_block, 2)
            if k % (32 * 8) == 0:
                return SkinnyGemmConfig(m, 32, outputs_per_block, 2)
            return SkinnyGemmConfig(m, 32, outputs_per_block, 2, vector_width=4)

        block_size = 64 if 4096 <= n < 8192 else 128
        outputs_per_block = 1 if m == 1 and n <= 2304 else 2
        if n % outputs_per_block:
            outputs_per_block = 1
        k_unroll = 2 if n <= 2304 or n >= 16384 else 1
        return SkinnyGemmConfig(m, block_size, outputs_per_block, k_unroll=k_unroll)

    def supports(self, config: SkinnyGemmConfig, m: int, n: int, k: int) -> bool:
        """Whether ``config`` can run this shape, without compiling anything."""
        return (
            config.num_rows == m
            and 1 <= m <= MAX_M
            and n % config.outputs_per_block == 0
            and k % (config.block_size * config.vector_width) == 0
            and (config.static_k is None or config.static_k == k)
        )

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        config: SkinnyGemmConfig,
        residual: torch.Tensor | None = None,
        residual2: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ``a @ b.T`` (plus residual addends) with the given config.

        Args:
            a: ``[M, K]`` contiguous bf16/fp16 activations, ``1 <= M <= 16``.
            b: ``[N, K]`` contiguous weight in the same dtype.
            config: the specialization to run; ``num_rows`` must equal M.
            residual: optional ``[M, N]`` addend folded into the epilogue.
            residual2: optional second ``[M, N]`` epilogue addend; requires
                ``residual``.
            out: optional ``[M, N]`` destination.

        Returns:
            ``[M, N]`` result in ``a``'s dtype.
        """
        m, k = a.shape
        n = b.shape[0]
        if not self.supports(config, m, n, k):
            raise ValueError(f"config {config} cannot run M={m} N={n} K={k}")
        if a.dtype != b.dtype or a.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("a and b must share one BF16 or FP16 dtype")
        if not a.is_contiguous() or not b.is_contiguous():
            raise ValueError("a and b must be contiguous")

        if residual2 is not None and residual is None:
            raise ValueError("residual2 requires residual")
        has_residual = residual is not None
        has_residual2 = residual2 is not None
        cache_key = (a.device.index or 0, a.dtype, config, has_residual, has_residual2)
        if cache_key not in self._compiled:
            # Double-checked: cute.compile is expensive and not thread-safe.
            with self._compile_lock:
                if cache_key not in self._compiled:
                    self._compile(
                        a.dtype, config, has_residual, has_residual2, a.device
                    )
        if out is None:
            out = torch.empty((m, n), dtype=a.dtype, device=a.device)
        self._compiled[cache_key](
            a,
            b,
            out if residual is None else residual,
            out if residual2 is None else residual2,
            out,
            self._stream(a.device),
        )
        return out


shape_dynamic_skinny_gemm = ShapeDynamicSkinnyGemm()

__all__ = [
    "MAX_M",
    "ShapeDynamicSkinnyGemm",
    "SkinnyGemmConfig",
    "shape_dynamic_skinny_gemm",
]
