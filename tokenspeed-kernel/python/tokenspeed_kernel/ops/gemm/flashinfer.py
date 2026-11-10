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

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple, get_args

import torch
from tokenspeed_kernel.ops.tuning import is_autotuning, is_warming_up
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    format_signatures,
    tensor_format,
)

logger = logging.getLogger(__name__)

platform = current_platform()
_fp8_dtype = torch.float8_e4m3fn

_fp4_dtypes: frozenset[torch.dtype] = frozenset({torch.uint8, torch.float4_e2m1fn_x2})
_MXFP8_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="block",
    block_shape=(128, 128),
)
_NVFP4_SCALE_DTYPES: frozenset[torch.dtype] = frozenset(
    {torch.float32, torch.uint8, torch.float8_e4m3fn}
)
_MXFP8_FORMAT_SIGNATURES = format_signatures(
    ("a", "b"), "mxfp8", {_fp8_dtype}, scale=_MXFP8_SCALE
)
_NVFP4_FORMAT_SIGNATURES = frozenset(
    format_signature(
        a=tensor_format(
            "nvfp4",
            storage_dtype,
            scale=ScaleFormat(
                storage_dtype=a_scale_dtype, granularity="block", block_shape=(16,)
            ),
        ),
        b=tensor_format(
            "nvfp4",
            storage_dtype,
            scale=ScaleFormat(
                storage_dtype=b_scale_dtype, granularity="block", block_shape=(16,)
            ),
        ),
    )
    for storage_dtype in _fp4_dtypes
    for a_scale_dtype in _NVFP4_SCALE_DTYPES
    for b_scale_dtype in _NVFP4_SCALE_DTYPES
)

# ===-----------------------------------------------------------------------===#
# FP8 block-scale: FlashInfer (NVIDIA SM100+)
# ===-----------------------------------------------------------------------===#


gemm_fp8_nt_groupwise = error_fn
tinygemm_bf16 = error_fn

if platform.is_hopper_plus:
    from flashinfer.gemm import (
        gemm_fp8_nt_groupwise,
    )
    from flashinfer.gemm import tinygemm_bf16 as _tinygemm_bf16

    def tinygemm_bf16(
        input: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor,
        bias: torch.Tensor | None = None,
        use_pdl: bool | None = None,
    ) -> None:
        """Run FlashInfer tiny GEMM using the platform PDL default.

        Args:
            input: Contiguous BF16 input matrix.
            weight: Contiguous BF16 weight matrix.
            out: Preallocated contiguous BF16 output matrix.
            bias: Optional contiguous BF16 bias.
            use_pdl: Whether to use PDL. Uses the platform default when omitted.

        Returns:
            None; ``out`` is updated in place.
        """
        _tinygemm_bf16(
            input,
            weight,
            out,
            bias,
            use_pdl=pdl_enabled() if use_pdl is None else use_pdl,
        )


def has_flashinfer_fp8_blockscale() -> bool:
    """Return whether the native FlashInfer FP8 block-scale GEMM is usable."""
    # Every Blackwell datacenter part runs this kernel; GB300 reports 10.3.
    return gemm_fp8_nt_groupwise is not error_fn and platform.is_blackwell


def _supports_flashinfer_fp8_blockscale(m: int, _n: int, _k: int) -> bool:
    return not 17 <= m <= 32


if gemm_fp8_nt_groupwise is not error_fn:

    @register_kernel(
        "gemm",
        "mm",
        name="flashinfer_mm_fp8_blockscale",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_MXFP8_FORMAT_SIGNATURES,
        traits={
            "n_align": frozenset({128}),
            "k_align": frozenset({128}),
            "mnk_problem_filter": frozenset({_supports_flashinfer_fp8_blockscale}),
            "a_contiguous": frozenset({True}),
            "b_contiguous": frozenset({True}),
            "block_scale_layout": frozenset({"canonical", "canonical_blackwell"}),
            "operands_aligned_16": frozenset({True}),
        },
        priority=Priority.SPECIALIZED + 2,
    )
    def flashinfer_mm_fp8_blockscale(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor | None = None,
        block_size: list[int] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run FlashInfer FP8 GEMM with canonical scales."""
        assert (
            A_scales is not None
        ), "A_scales is required; online quantization should be done by the caller"
        assert B_scales is not None, "B_scales is required for FP8 blockscale GEMM"
        orig_m = A.shape[0]
        if not _supports_flashinfer_fp8_blockscale(orig_m, B.shape[0], A.shape[1]):
            raise ValueError(
                "FlashInfer FP8 block-scale GEMM does not support 17 <= M <= 32"
            )
        # K-major mode reads the quant kernel's native (m, k//128) activation
        # scales and the checkpoint's native (n//128, k//128) weight scales,
        # so no padding, transposes, or scale copies are needed per call.
        # FlashInfer defect: SM10x mis-reads these scales for 17 <= M <= 32.
        if A_scales.shape[0] != orig_m:
            A_scales = A_scales[:orig_m]
        # The kernel reads raw row-major storage; normalize strided views
        # (a no-op on the hot path, where quant output is contiguous).
        if not A_scales.is_contiguous():
            A_scales = A_scales.contiguous()
        if not B_scales.is_contiguous():
            B_scales = B_scales.contiguous()
        direct_out = (
            out is not None
            and out.is_contiguous()
            and out.shape == (orig_m, B.shape[0])
            and out.data_ptr() % 16 == 0
        )
        output = gemm_fp8_nt_groupwise(
            A,
            B,
            A_scales,
            B_scales,
            scale_major_mode="K",
            out=out if direct_out else None,
            out_dtype=out_dtype,
        )
        if out is not None and not direct_out:
            out.copy_(output)
            return out
        return output


# ===-----------------------------------------------------------------------===#
# MXFP8 (1x32 UE8M0): FlashInfer CuTe DSL (NVIDIA SM100)
# ===-----------------------------------------------------------------------===#


mm_mxfp8 = error_fn

if platform.is_nvidia and platform.is_blackwell:
    from tokenspeed_kernel.thirdparty.flashinfer.mxfp8 import mm_mxfp8

_MXFP8_UE8M0_1X32_SCALE = ScaleFormat(
    storage_dtype=torch.uint8,
    granularity="block",
    block_shape=(1, 32),
)
_MXFP8_FLOAT_1X32_SCALE = ScaleFormat(
    storage_dtype=torch.float32,
    granularity="block",
    block_shape=(1, 32),
)
_MXFP8_1X32_FORMAT_SIGNATURES = frozenset(
    format_signature(
        a=tensor_format("mxfp8", _fp8_dtype, scale=a_scale),
        b=tensor_format("mxfp8", _fp8_dtype, scale=_MXFP8_UE8M0_1X32_SCALE),
    )
    for a_scale in (_MXFP8_FLOAT_1X32_SCALE, _MXFP8_UE8M0_1X32_SCALE)
)


def has_flashinfer_mxfp8() -> bool:
    """Whether the flashinfer cute-dsl MXFP8 (1,32) GEMM is usable here.

    Returns:
        True when running on an NVIDIA Blackwell (SM10x) GPU with a
        flashinfer build that provides ``mm_mxfp8``.
    """
    return mm_mxfp8 is not error_fn


def swizzle_mxfp8_scale(sf: torch.Tensor, m: int, k: int) -> torch.Tensor:
    m_tiles = (m + 127) // 128
    k_tiles = (k + 127) // 128
    padded = torch.zeros((m_tiles * 128, k_tiles * 4), dtype=sf.dtype, device=sf.device)
    padded[:m, : k // 32] = sf
    return padded.view(m_tiles, 4, 32, k_tiles, 4).transpose(1, 3).contiguous().view(-1)


if mm_mxfp8 is not error_fn:

    @register_kernel(
        "gemm",
        "mm",
        name="flashinfer_mm_mxfp8",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 7),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_MXFP8_1X32_FORMAT_SIGNATURES,
        traits={
            "k_align": frozenset({32}),
            "n_min": frozenset({128}),
            "k_min": frozenset({128}),
            "pdl_enabled": frozenset({True}),
        },
        priority=Priority.SPECIALIZED + 2,
    )
    def flashinfer_mm_mxfp8(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor | None = None,
        block_size: list[int] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MXFP8 (1,32)-block ue8m0 GEMM via flashinfer's cute-dsl backend.

        Args:
            A: ``[M, K]`` float8_e4m3fn activations.
            B: ``[N, K]`` (or ``[K, N]`` column-major) float8_e4m3fn weight.
            A_scales: uint8 e8m0 activation scales, either 1D in the
                F8_128x4 swizzled layout or ``[M, K // 32]`` row-major
                (re-swizzled per call; prefer pre-swizzled).
            B_scales: uint8 e8m0 weight scales, same layout options with
                ``[N, K // 32]`` row-major.
            out_dtype: Output dtype (bf16/fp16).
            alpha: Unused.
            block_size: Must be ``[1, 32]``.
            out: Optional output buffer.

        Returns:
            ``[M, N]`` tensor of ``out_dtype``.
        """
        assert (
            A_scales is not None
        ), "A_scales is required; online quantization should be done by the caller"
        assert B_scales is not None, "B_scales is required for MXFP8 GEMM"
        assert block_size == [1, 32], f"expected block_size [1, 32], got {block_size}"
        k = A.shape[1]
        # B follows the dispatch convention of a [N, K] weight (row-major,
        # like the Triton kernel assumes); mm_mxfp8 wants the [K, N]
        # column-major view. Shape alone cannot disambiguate square weights,
        # so decide by memory layout.
        if B.shape[0] == k and B.stride(0) == 1:
            b = B
        else:
            b = B.t()
        n = b.shape[1]
        if k < 128 or k % 32 != 0 or n < 128:
            raise ValueError(
                f"flashinfer_mm_mxfp8 requires K >= 128, K % 32 == 0 and "
                f"N >= 128, got K={k}, N={n}"
            )
        if A_scales.dtype != torch.uint8 or B_scales.dtype != torch.uint8:
            raise ValueError(
                "flashinfer_mm_mxfp8 requires uint8 e8m0 scales, got "
                f"A_scales={A_scales.dtype}, B_scales={B_scales.dtype}"
            )
        if A_scales.dim() != 1:
            A_scales = swizzle_mxfp8_scale(A_scales.contiguous(), A.shape[0], k)
        if B_scales.dim() != 1:
            B_scales = swizzle_mxfp8_scale(B_scales.contiguous(), n, k)
        output = mm_mxfp8(
            A,
            b,
            A_scales,
            B_scales,
            out_dtype=out_dtype,
            backend="cute-dsl",
        )
        if out is not None:
            out.copy_(output)
            return out
        return output


# ===-----------------------------------------------------------------------===#
# Per-tensor FP8: FlashInfer cuBLASLt (NVIDIA SM100)
# ===-----------------------------------------------------------------------===#


_FP8_TENSOR_SCALE = ScaleFormat(storage_dtype=torch.float32, granularity="tensor")
cublas_fp8_gemm = error_fn

if platform.is_nvidia and platform.is_blackwell:
    from tokenspeed_kernel.thirdparty.flashinfer.fp8_gemm import cublas_fp8_gemm

if cublas_fp8_gemm is not error_fn:

    @register_kernel(
        "gemm",
        "mm",
        name="flashinfer_mm_fp8_tensor_scaled",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures(
            ("a", "b"), "scaled-fp8", {_fp8_dtype}, scale=_FP8_TENSOR_SCALE
        ),
        # cuBLASLt reads B column-major: a transposed [N, K] weight.
        traits={
            "a_inner_stride_one": frozenset({True}),
            "b_inner_stride_one": frozenset({False}),
        },
        priority=Priority.PERFORMANT + 3,
    )
    def flashinfer_mm_fp8_tensor_scaled(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor | None = None,
        block_size: list[int] | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-tensor scaled FP8 GEMM on FlashInfer's cuBLASLt backend.

        Args:
            A: ``[M, K]`` row-major FP8 activations.
            B: ``[K, N]`` column-major FP8 weights (a transposed ``[N, K]``).
            A_scales: One-element FP32 activation dequant scale.
            B_scales: One-element FP32 weight dequant scale.
            out_dtype: BF16 or FP16 output dtype.
            alpha: Must be None; the per-tensor scales carry the dequant.
            block_size: Must be None; the scales are per tensor.
            out: Optional ``[M, N]`` output buffer.

        Returns:
            ``[M, N]`` output, ``out`` when given.
        """
        if alpha is not None or block_size is not None:
            raise ValueError("per-tensor FP8 GEMM takes no alpha or block_size")
        if out_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"per-tensor FP8 GEMM writes BF16 or FP16, not {out_dtype}"
            )
        # cuBLASLt reads dense operands: row-major A and column-major B.
        A = A.contiguous()
        B = B.t().contiguous().t()
        direct = out is not None and out.is_contiguous()
        result = (
            out
            if direct
            else torch.empty(A.shape[0], B.shape[1], dtype=out_dtype, device=A.device)
        )
        cublas_fp8_gemm(
            A.unsqueeze(0),
            B.unsqueeze(0),
            A_scales,
            B_scales,
            result.unsqueeze(0),
        )
        if out is None or direct:
            return result
        # cuBLASLt writes dense rows; a strided view gets a copy.
        return out.copy_(result)


# ===-----------------------------------------------------------------------===#
# NVFP4: FlashInfer (NVIDIA SM100)
# ===-----------------------------------------------------------------------===#


mm_fp4 = error_fn

if platform.is_nvidia and platform.is_blackwell:
    from flashinfer import mm_fp4

if mm_fp4 is not error_fn:

    @register_kernel(
        "gemm",
        "mm",
        name="flashinfer_mm_nvfp4",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_NVFP4_FORMAT_SIGNATURES,
        traits={},
        priority=Priority.SPECIALIZED + 2,
    )
    def flashinfer_mm_nvfp4(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor | None = None,
        block_size: list[int] | None = None,
        enable_pdl: bool = False,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # backend="cutlass" (not "auto") to skip flashinfer's cuDNN-graph plan compile.
        output = mm_fp4(
            A,
            B,
            A_scales,
            B_scales,
            alpha,
            out_dtype,
            backend="cutlass",
            enable_pdl=enable_pdl,
        )
        if out is not None:
            out.copy_(output)
            return out
        return output


# ===-----------------------------------------------------------------------===#
# NVFP4 decode-sized M: FlashInfer CuTe DSL (NVIDIA SM100)
# ===-----------------------------------------------------------------------===#


# Up to this M the CuTe-DSL kernel beat cuBLASLt on every measured SM100 shape; larger M stays on cuBLASLt.
NVFP4_CUTE_DSL_MAX_M = 128
# From K = 26624 on GB200, cuBLASLt splits K at small M and its bits stop matching this kernel's in-order sum.
NVFP4_CUTE_DSL_MAX_K = 18432

if mm_fp4 is not error_fn:
    from flashinfer.gemm.gemm_base import (
        _cute_dsl_gemm_fp4_runner,
        _select_sm100_mm_fp4_cute_dsl_tactic,
    )
    from flashinfer.utils import get_device_sm_count

    _nvfp4_cute_dsl_runner = functools.cache(_cute_dsl_gemm_fp4_runner)

    def _aligned_copy(tensor: torch.Tensor, alignment: int) -> torch.Tensor:
        """``tensor`` itself, or a same-strided copy when its data is not ``alignment``-byte aligned."""
        if tensor.data_ptr() % alignment == 0:
            return tensor
        copy = torch.empty_strided(
            tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device
        )
        return copy.copy_(tensor)

    @register_kernel(
        "gemm",
        "mm",
        name="flashinfer_cute_dsl_mm_nvfp4",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 7),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_NVFP4_FORMAT_SIGNATURES,
        # mm's k for NVFP4 is the packed width, K // 2.
        traits={
            "m_max": frozenset({NVFP4_CUTE_DSL_MAX_M}),
            "n_align": frozenset({8}),
            "k_align": frozenset({16}),
            "k_max": frozenset({NVFP4_CUTE_DSL_MAX_K // 2}),
            "out_dtype": frozenset({torch.bfloat16, torch.float16}),
        },
        priority=Priority.SPECIALIZED + 4,
    )
    def flashinfer_cute_dsl_mm_nvfp4(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor,
        B_scales: torch.Tensor,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor,
        block_size: list[int] | None = None,
        enable_pdl: bool,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """NVFP4 GEMM for decode-sized M, bit-identical to ``cublaslt_mm_nvfp4``.

        cuBLASLt launches too few CTAs at small M to stream the weight at full
        bandwidth; FlashInfer's persistent CuTe-DSL kernel does, summing K in
        the same order up to :data:`NVFP4_CUTE_DSL_MAX_K`.

        Args:
            A: Packed FP4 activations ``[M, K // 2]``.
            B: Packed FP4 weight as a ``[K // 2, N]`` view of ``[N, K // 2]``.
            A_scales: Activation block-16 scales in the 128x4 swizzled layout.
            B_scales: Weight block-16 scales, transposed like ``B``.
            out_dtype: ``torch.bfloat16`` or ``torch.float16``.
            alpha: One-element float32 global scale.
            block_size: Scale block shape; only ``[16]`` is supported.
            enable_pdl: Whether to enable Programmatic Dependent Launch.
            out: Optional ``[M, N]`` output buffer.

        Returns:
            The ``[M, N]`` product, in ``out`` when supplied.
        """
        if block_size is not None and tuple(block_size) != (16,):
            raise ValueError(f"NVFP4 scales use 16-element blocks, got {block_size}")
        if alpha is None or alpha.numel() != 1:
            raise ValueError("NVFP4 GEMM takes a one-element global alpha")
        m, n, k = A.shape[0], B.shape[1], A.shape[1] * 2
        if m == 0:
            # The kernel cannot launch an empty grid, as on an idle data-parallel rank.
            return out if out is not None else A.new_empty((0, n), dtype=out_dtype)
        direct = out is not None and out.is_contiguous() and out.data_ptr() % 16 == 0
        result = (
            out if direct else torch.empty((m, n), dtype=out_dtype, device=A.device)
        )
        # mm_fp4's own selector may pick split-K, which reorders the K sum; this one never does.
        tactic = _select_sm100_mm_fp4_cute_dsl_tactic(
            m, n, k, get_device_sm_count(A.device), 16
        )
        runner = _nvfp4_cute_dsl_runner(
            platform.arch_version.major,
            platform.arch_version.minor,
            enable_pdl,
            out_dtype,
            True,
        )
        # mm_fp4's input order with uint8 FP4 storage; the runner never reads the workspace slot.
        inputs = [
            _aligned_copy(A.view(torch.uint8), 32),
            _aligned_copy(B.view(torch.uint8), 32),
            A_scales,
            B_scales,
            alpha,
            out_dtype,
            result,
            16,
            True,
            None,
        ]
        runner(inputs=inputs, tactic=tactic)
        if out is not None and not direct:
            return out.copy_(result)
        return result


_CUTE_DSL_BACKEND = "cute-dsl"
_CUTE_DSL_SM100_ARCHS = frozenset({ArchVersion(10, 0), ArchVersion(10, 3)})


# ===-----------------------------------------------------------------------===#
# BF16 x NVFP4: FlashInfer CuTe DSL (NVIDIA SM100, SM103)
# ===-----------------------------------------------------------------------===#


_mm_bf16_fp4 = error_fn
_prepare_bf16_fp4_weights = error_fn

if platform.is_nvidia and platform.arch_version in _CUTE_DSL_SM100_ARCHS:
    from flashinfer.gemm import mm_bf16_fp4 as _mm_bf16_fp4
    from flashinfer.gemm import prepare_bf16_fp4_weights as _prepare_bf16_fp4_weights

_NVFP4_A16_FORMAT_SIGNATURES = frozenset(
    format_signature(
        a=dense_tensor_format(torch.bfloat16),
        b=tensor_format(
            "nvfp4",
            torch.uint8,
            scale=ScaleFormat(
                storage_dtype=scale_dtype,
                granularity="block",
                block_shape=(16,),
            ),
        ),
    )
    for scale_dtype in (torch.float8_e4m3fn, torch.uint8)
)


def has_flashinfer_cute_dsl_nvfp4_a16() -> bool:
    """Whether FlashInfer's CuTe-DSL BF16 x NVFP4 GEMM is usable here.

    Returns:
        True on SM100 or SM103 when both the preparation and GEMM entry points
        are available.
    """
    return _mm_bf16_fp4 is not error_fn and _prepare_bf16_fp4_weights is not error_fn


def prepare_nvfp4_a16_weights(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Prepare canonical NVFP4 weights for FlashInfer's CuTe-DSL W4A16 GEMM.

    Args:
        weight: Packed uint8 weight shaped ``[N, K / 2]``.
        weight_scale: Runtime 128x4-swizzled block-16 scales.
        alpha: Optional float32 global scale. A scalar tensor is normalized to
            shape ``(1,)`` before preparation.

    Returns:
        The prepared ``(weight, weight_scale, alpha)`` tuple accepted by
        :func:`mm` with ``quant="nvfp4_a16"``.
    """
    if alpha is not None and alpha.ndim == 0:
        alpha = alpha.reshape(1).to(dtype=torch.float32)
    return _prepare_bf16_fp4_weights(
        weight,
        weight_scale,
        alpha,
        backend=_CUTE_DSL_BACKEND,
        block_size=16,
    )


if has_flashinfer_cute_dsl_nvfp4_a16():

    @register_kernel(
        "gemm",
        "mm",
        name="flashinfer_cute_dsl_mm_nvfp4_a16",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_NVFP4_A16_FORMAT_SIGNATURES,
        traits={"k_align": frozenset({16})},
        priority=Priority.SPECIALIZED + 2,
    )
    def flashinfer_cute_dsl_mm_nvfp4_a16(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scales: torch.Tensor | None,
        B_scales: torch.Tensor | None,
        out_dtype: torch.dtype,
        *,
        alpha: torch.Tensor | None = None,
        block_size: list[int] | None = None,
        enable_pdl: bool = False,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run prepared NVFP4 weights against dense BF16 activations.

        Args:
            A: Dense BF16 activation shaped ``[M, K]``.
            B: Packed uint8 weight returned by
                :func:`prepare_nvfp4_a16_weights`.
            A_scales: Must be None because activations are dense.
            B_scales: Six-dimensional scale view returned by
                :func:`prepare_nvfp4_a16_weights`.
            out_dtype: Requested output dtype.
            alpha: Prepared optional float32 global scale.
            block_size: Optional logical block size; only 16 is supported.
            enable_pdl: Whether to enable Programmatic Dependent Launch.
            out: Optional preallocated output tensor.

        Returns:
            The ``[M, N]`` output, using ``out`` directly when supplied.
        """
        if A_scales is not None:
            raise ValueError(
                "nvfp4_a16 uses dense BF16 activations and requires A_scales=None"
            )
        n, k = B.shape[0], B.shape[1] * 2
        n_tiles = (n + 127) // 128
        k_tiles = (k // 16 + 3) // 4
        expected_scale_shape = (32, 4, n_tiles, 4, k_tiles, 1)
        expected_scale_stride = (
            16,
            4,
            k_tiles * 512,
            1,
            512,
            n_tiles * k_tiles * 512,
        )
        if (
            B_scales is None
            or tuple(B_scales.shape) != expected_scale_shape
            or tuple(B_scales.stride()) != expected_scale_stride
        ):
            shape = None if B_scales is None else tuple(B_scales.shape)
            stride = None if B_scales is None else tuple(B_scales.stride())
            raise ValueError(
                "nvfp4_a16 B_scales must be the 6-D view returned by "
                "prepare_nvfp4_a16_weights; expected "
                f"shape={expected_scale_shape}, stride={expected_scale_stride}, "
                f"got shape={shape}, stride={stride}"
            )
        if block_size is not None and tuple(block_size) != (16,):
            raise ValueError(f"nvfp4_a16 requires block_size=[16], got {block_size}")
        direct_out = out is None or out.is_contiguous()
        output = _mm_bf16_fp4(
            A,
            B,
            B_scales,
            alpha,
            backend=_CUTE_DSL_BACKEND,
            out_dtype=out_dtype,
            out=out if direct_out else None,
            block_size=16,
            enable_pdl=enable_pdl,
        )
        if out is not None and not direct_out:
            out.copy_(output)
            return out
        return output


# ===-----------------------------------------------------------------------===#
# BF16 low-latency GEMM: FlashInfer TGV and CuTe DSL (NVIDIA SM100/103/107)
# ===-----------------------------------------------------------------------===#


_mm_bf16 = error_fn
_fi_gemm = None
# Automatic dispatch scope, not a TGV capability limit.
BF16_GEMM_MAX_M = 4 if platform.arch_version == ArchVersion(10, 7) else 32

if platform.is_nvidia and platform.arch_version in (
    *_CUTE_DSL_SM100_ARCHS,
    ArchVersion(10, 7),
):
    from flashinfer import mm_bf16 as _mm_bf16
    from flashinfer.gemm import gemm_base as _fi_gemm


def _declares_cute_dsl_backend(mm_bf16: Callable[..., object]) -> bool:
    """Whether this ``mm_bf16`` lists :data:`_CUTE_DSL_BACKEND`.

    Args:
        mm_bf16: FlashInfer's entry point, whose ``backend`` annotation is the
            ``Literal`` of the backends that build it.

    Returns:
        True on wheels carrying the upstreamed kernels, False on earlier ones,
        which name every other backend but not this one.
    """
    try:
        # eval_str resolves the Literal even if FlashInfer postpones annotations.
        backend = inspect.signature(mm_bf16, eval_str=True).parameters["backend"]
    except (KeyError, NameError, TypeError, ValueError):
        return False
    return _CUTE_DSL_BACKEND in get_args(backend.annotation)


@functools.lru_cache(maxsize=1)
def has_flashinfer_cute_dsl_bf16() -> bool:
    """Whether the flashinfer cute-dsl BF16 low-latency GEMM is usable here.

    Returns:
        True when running on an SM100, SM103, or SM107 GPU with a flashinfer build
        whose ``mm_bf16`` declares the backend.
    """
    return _mm_bf16 is not error_fn and _declares_cute_dsl_backend(_mm_bf16)


def _bf16_gemm_runner_names(k: int) -> list[str]:
    """Admit each backend by its own K contract, not their intersection."""
    if k <= 0:
        return []
    # TGV handles partial K tiles; its TMA rows need 16-byte (8 BF16) strides.
    runners = ["tgv"] if k % 8 == 0 else []
    # Only cute-dsl requires whole 128-element K tiles. Its native runners
    # filter N/tactic constraints independently (e.g. warp Split-K's N % 16).
    if k % 128 == 0:
        runners.append("cute-dsl")
    return runners


def flashinfer_joint_bf16_supported(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None
) -> bool:
    """Check the common contract and whether at least one backend is eligible.

    Discovery may use large M; execution enforces the separate small-M bound.
    """
    return (
        _fi_gemm is not None
        and has_flashinfer_cute_dsl_bf16()
        and x.is_cuda
        and x.device == weight.device
        and x.ndim == weight.ndim == 2
        # Limit SM107 dispatch to the measured attention output projection.
        # Other dense projections keep their existing implementation.
        and (
            platform.arch_version != ArchVersion(10, 7) or weight.shape == (2560, 6144)
        )
        and x.dtype == weight.dtype == torch.bfloat16
        and x.shape[0] > 0
        and weight.shape[0] > 0
        and x.shape[1] == weight.shape[1]
        and weight.shape[1] > 0
        and bool(_bf16_gemm_runner_names(weight.shape[1]))
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.data_ptr() % 32 == weight.data_ptr() % 32 == 0
        and (
            out is None
            or (
                out.shape == (x.shape[0], weight.shape[0])
                and out.dtype == x.dtype
                and out.device == x.device
                and out.is_contiguous()
                and out.data_ptr() % 32 == 0
            )
        )
    )


def _canonical_bf16_view(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize singleton strides of an already-contiguous matrix without copying."""
    strides = (tensor.shape[1], 1)
    if tensor.stride() != strides:
        return tensor.as_strided(tensor.shape, strides)
    return tensor


def flashinfer_bf16_gemm(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None
) -> torch.Tensor:
    """Compute BF16 x[M,K] @ weight[N,K].T using FI's joint runner/tactic search.

    The caller checks the contract and warms the actual shape before capture.
    Only rows within the platform's small-M bound enter this search. Larger
    calls keep the original GEMM.
    No TokenSpeed backend choice or second cache is maintained.
    """
    if (
        not flashinfer_joint_bf16_supported(x, weight, out)
        or x.shape[0] > BF16_GEMM_MAX_M
    ):
        raise ValueError("Unsupported input to joint FlashInfer BF16 GEMM")
    if out is None:
        out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    workspace = _fi_gemm._get_cache_buf(
        "mm_bf16_workspace", _fi_gemm.DEFAULT_WORKSPACE_SIZE, x.device
    )
    # WAR: the public auto heuristic excludes cute-dsl, and its SM100a check
    # rejects SM107 even though the native runners support it. Reuse the FI
    # dispatcher so eligible families enter one choose_one, including cache
    # lookup. A backend that cannot handle K must not exclude the other one.
    # Contiguous singleton rows can retain a sliced tensor's larger row stride;
    # FI's dynamic-M kernels require the canonical compact stride even at M=1.
    _fi_gemm.bf16_gemm_sm100(
        a=_canonical_bf16_view(x.detach()),
        b=weight.detach().t(),
        bias=None,
        pdl=pdl_enabled(),
        out=_canonical_bf16_view(out),
        workspace_buffer=workspace,
        runner_names=_bf16_gemm_runner_names(weight.shape[1]),
    )
    return out


def autotune_bf16_gemm(x: torch.Tensor, weight: torch.Tensor) -> None:
    """Expose native small-M profiles from any encountered projection's N/K.

    FI skips cached profiles. Scratch inputs/output never alias the model output;
    this function does nothing outside the startup autotune window. The scratch
    row count exposes native profiles up to the dispatch bound: 1/2/4 on
    SM107, and 1/2/4/8/16/32 on SM100/103.
    """
    if is_autotuning() and flashinfer_joint_bf16_supported(x, weight, None):
        with torch.no_grad():
            sample = x.new_zeros((BF16_GEMM_MAX_M, weight.shape[1]))
            flashinfer_bf16_gemm(sample, weight, None)


# ===-----------------------------------------------------------------------===#
# BF16 decode GEMM on small weights: pinned FlashInfer TGV tiles (SM100/103)
# ===-----------------------------------------------------------------------===#
#
# With PDL and before serving, mm() runs a bias-free BF16 GEMM with
# TGV_DECODE_MIN_M < M <= TGV_DECODE_MAX_M rows and
# TGV_DECODE_MIN_K <= K <= TGV_DECODE_MAX_K on a weight of at most
# TGV_DECODE_MAX_WEIGHT_BYTES with one TGV tile instead of cuBLASLt. Under
# PDL, TGV's weight-load warp fills its shared-memory stages while the
# producer kernel still runs; only its activation-load warp waits for the
# producer. cuBLASLt loads nothing before the producer finishes. FI's tactic
# profiling times each GEMM alone and cannot see this overlap, so the tile is
# chosen here rather than by FI's tuner: the deepest-staged tile whose grid
# fits in one wave. Without PDL there is no overlap, and cuBLASLt stays.
#
# FI's tuner keeps the 2-CTA TGV tactics 16 to 28, these tiles among them,
# blocklisted (autotune() in ops/tuning.py) after replay failures of FI
# 0.6.18's C++ TGV kernel. FI 0.7.2.post1 runs TGV as a CuTe DSL rewrite
# (tgv_gemm_cute_ext), which these tiles call directly, outside the tuner.
# The GPU tests capture and replay each tile, alone and in a chain of PDL
# kernels.

# The policy routes rows above TGV_DECODE_MIN_M, the joint FlashInfer path's
# bound on SM100 and SM103 (BF16_GEMM_MAX_M there; the GPU tests check that
# they agree). BF16_GEMM_MAX_M itself follows the host GPU, 4 on SM107, so the
# policy keeps its own bound and depends only on its arguments.
TGV_DECODE_MIN_M = 32
TGV_DECODE_MAX_M = 512
TGV_DECODE_MAX_WEIGHT_BYTES = 16 << 20
# Measured on GB300 against cuBLASLt in a captured PDL chain (an RMSNorm, then
# this GEMM on a weight cold in L2): with K in this window TGV was faster on
# most shapes; outside it, it lost on many shapes, down to 0.73x cuBLASLt's
# speed: at K of 512 to 1024, at K >= 4096 with few weight rows, and on some
# shapes at K = 1536.
TGV_DECODE_MIN_K = 2048
TGV_DECODE_MAX_K = 3072


class TgvTile(NamedTuple):
    """One FlashInfer TGV tile, in the form of an entry of FI's tactic table.

    FI swaps the GEMM operands for TGV: the kernel's M runs over weight rows
    and its N over activation rows.
    """

    cta_m: int  # weight rows per CTA
    cta_n: int  # activation rows per tile
    num_ab_stage: int  # shared-memory pipeline stages
    use_2cta: bool  # a pair of CTAs computes a tile of 2 * cta_m weight rows


# 2-CTA tiles of 128 weight rows by 16, 32 and 64 activation rows, each with
# the most stages that fit in shared memory, in the order the policy tries them.
TGV_DECODE_TILES = (
    TgvTile(cta_m=64, cta_n=16, num_ab_stage=12, use_2cta=True),
    TgvTile(cta_m=64, cta_n=32, num_ab_stage=11, use_2cta=True),
    TgvTile(cta_m=64, cta_n=64, num_ab_stage=9, use_2cta=True),
)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def select_tgv_decode_tile(
    *,
    m: int,
    n: int,
    k: int,
    weight_bytes: int,
    arch: ArchVersion,
    sm_count: int,
    pdl: bool,
) -> TgvTile | None:
    """Choose the pinned TGV tile for a bias-free BF16 GEMM, or None.

    The choice depends only on the arguments, and it can only be one of
    :data:`TGV_DECODE_TILES`, so compiling those tiles at startup covers
    every row count.

    Args:
        m: Activation rows.
        n: Weight rows, i.e. output columns.
        k: Weight columns, the reduction length.
        weight_bytes: Size of the weight in bytes.
        arch: GPU architecture.
        sm_count: Number of streaming multiprocessors of the GPU.
        pdl: Whether kernels launch with programmatic dependent launch
            (:func:`~tokenspeed_kernel.platform.pdl_enabled`). The tiles are
            pinned for the overlap PDL gives them, so without it no tile is
            chosen.

    Returns:
        With PDL on SM100 and SM103, for
        ``TGV_DECODE_MIN_M < m <= TGV_DECODE_MAX_M``,
        ``TGV_DECODE_MIN_K <= k <= TGV_DECODE_MAX_K`` and
        ``weight_bytes <= TGV_DECODE_MAX_WEIGHT_BYTES``, the first tile of
        :data:`TGV_DECODE_TILES` whose grid has at most ``sm_count`` CTAs. TGV
        gives each CTA a whole SM's shared memory and has no split-K, so a
        larger grid runs in more than one wave. None otherwise.
    """
    if (
        not pdl
        or arch not in _CUTE_DSL_SM100_ARCHS
        or not TGV_DECODE_MIN_M < m <= TGV_DECODE_MAX_M
        or not TGV_DECODE_MIN_K <= k <= TGV_DECODE_MAX_K
        or weight_bytes > TGV_DECODE_MAX_WEIGHT_BYTES
    ):
        return None
    for tile in TGV_DECODE_TILES:
        # TGV's grid: weight-row tiles rounded up to whole clusters, times
        # activation-row tiles.
        cluster = 2 if tile.use_2cta else 1
        weight_ctas = _ceil_div(_ceil_div(n, tile.cta_m), cluster) * cluster
        if weight_ctas * _ceil_div(m, tile.cta_n) <= sm_count:
            return tile
    return None


def resolve_tgv_tactics(
    tactic_table: Sequence[tuple[int, int, int, bool]],
) -> dict[TgvTile, int] | None:
    """Look up the tactic id of each pinned tile in a FlashInfer TGV table.

    A tactic id is a position in FI's table, which a FlashInfer release may
    reorder or extend, so the tiles are matched by shape, not by fixed ids.

    Args:
        tactic_table: FI's TGV tactic table. Entry ``i`` describes tactic
            ``i`` as ``(cta_m, cta_n, num_ab_stage, use_2cta)``.

    Returns:
        The tactic id of every tile of :data:`TGV_DECODE_TILES`, or None when
        the table lacks one of them.
    """
    tactics = {}
    for tile in TGV_DECODE_TILES:
        tactic = next(
            (i for i, entry in enumerate(tactic_table) if entry == tile), None
        )
        if tactic is None:
            return None
        tactics[tile] = tactic
    return tactics


@dataclass(frozen=True)
class _FlashInferTgv:
    """FlashInfer's TGV launcher and the tactic id of each pinned tile."""

    run: Callable[..., torch.Tensor]
    tactics: Mapping[TgvTile, int]


@functools.cache
def _flashinfer_tgv() -> _FlashInferTgv | None:
    """Bind the installed FlashInfer's TGV launcher and pinned tactics.

    Returns:
        The binding, or None, with one warning, when FlashInfer lacks the TGV
        module, a launcher this file can call, or one of the pinned tiles. The
        GEMMs then keep cuBLASLt.
    """
    try:
        from flashinfer.gemm.kernels.tgv_gemm_cute_ext import (
            _TGV_CUTE_EXT_TACTIC_CONFIGS,
            run_tgv_cute_ext,
        )
    except ImportError as error:
        logger.warning(
            f"FlashInfer TGV is unavailable ({error}); decode-sized BF16 GEMMs "
            "on small weights keep cuBLASLt"
        )
        return None
    try:
        # The arguments flashinfer_tgv_decode_gemm passes, by keyword.
        inspect.signature(run_tgv_cute_ext).bind(
            a=None, b=None, bias=None, out=None, pdl=False, tactic=0
        )
    except TypeError as error:
        logger.warning(
            f"FlashInfer's TGV launcher changed ({error}); decode-sized BF16 "
            "GEMMs on small weights keep cuBLASLt"
        )
        return None
    tactics = resolve_tgv_tactics(_TGV_CUTE_EXT_TACTIC_CONFIGS)
    if tactics is None:
        logger.warning(
            f"FlashInfer's TGV tactic table lacks one of {TGV_DECODE_TILES}; "
            "decode-sized BF16 GEMMs on small weights keep cuBLASLt"
        )
        return None
    return _FlashInferTgv(run=run_tgv_cute_ext, tactics=tactics)


def flashinfer_tgv_decode_tactic(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None
) -> int | None:
    """Choose the FlashInfer TGV tactic for a bias-free BF16 GEMM, or None.

    Args:
        x: ``[M, K]`` activations.
        weight: ``[N, K]`` weight.
        out: Optional ``[M, N]`` destination.

    Returns:
        The tactic id of the :func:`select_tgv_decode_tile` tile for this GPU
        and :func:`~tokenspeed_kernel.platform.pdl_enabled` when the operands
        meet :func:`flashinfer_joint_bf16_supported` and the installed
        FlashInfer lists the pinned tiles. None otherwise, and the caller
        keeps its GEMM.
    """
    if not flashinfer_joint_bf16_supported(x, weight, out):
        return None
    tile = select_tgv_decode_tile(
        m=x.shape[0],
        n=weight.shape[0],
        k=weight.shape[1],
        weight_bytes=weight.numel() * weight.element_size(),
        arch=platform.arch_version,
        sm_count=platform.sm_count,
        pdl=pdl_enabled(),
    )
    if tile is None:
        return None
    tgv = _flashinfer_tgv()
    return None if tgv is None else tgv.tactics[tile]


def flashinfer_tgv_decode_gemm(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None, tactic: int
) -> torch.Tensor:
    """Compute BF16 ``x @ weight.T`` with one pinned FlashInfer TGV tactic.

    TGV accumulates in FP32 and rounds once to BF16, in another reduction
    order than cuBLASLt, so the result is not bitwise identical to it. Each
    output element comes from one CTA in a fixed order, so repeated calls,
    eager or replayed from a CUDA graph, agree bit for bit.

    Args:
        x: ``[M, K]`` activations.
        weight: ``[N, K]`` weight.
        out: Optional ``[M, N]`` destination.
        tactic: A tactic id from :func:`flashinfer_tgv_decode_tactic`.

    Returns:
        The ``[M, N]`` product, written to ``out`` when given.

    Raises:
        ValueError: If the operands do not meet
            :func:`flashinfer_joint_bf16_supported` or ``tactic`` is not a
            pinned tile of the installed FlashInfer.
    """
    tgv = _flashinfer_tgv()
    if not flashinfer_joint_bf16_supported(x, weight, out) or (
        tgv is None or tactic not in tgv.tactics.values()
    ):
        raise ValueError("Unsupported input to the pinned FlashInfer TGV GEMM")
    if out is None:
        out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    # TGV compiles once per tactic, PDL setting, and operand layout, never
    # per shape. Canonical strides keep every call on one layout.
    tgv.run(
        a=_canonical_bf16_view(x.detach()),
        b=weight.detach().t(),
        bias=None,
        out=_canonical_bf16_view(out),
        pdl=pdl_enabled(),
        tactic=tactic,
    )
    return out


def warm_tgv_decode_gemm(x: torch.Tensor, weight: torch.Tensor) -> None:
    """Compile every pinned TGV tactic during startup kernel warmup.

    TGV compiles per tactic, PDL setting, and operand layout, not per shape,
    so warming the pinned tactics on any weight the policy routes, at the
    process's PDL setting, keeps compilation out of CUDA graph capture.
    Does nothing outside :func:`~tokenspeed_kernel.ops.tuning.kernel_warmup`,
    while the current stream is capturing, or for a weight no row count routes.

    Args:
        x: Activations of the encountered GEMM; only their dtype and device
            are used.
        weight: ``[N, K]`` weight of the encountered GEMM.
    """
    if not is_warming_up():
        return
    # The fewest rows routed. Fewer rows never need a larger grid, so if
    # they get no tile, no row count does.
    m = TGV_DECODE_MIN_M + 1
    if (
        select_tgv_decode_tile(
            m=m,
            n=weight.shape[0],
            k=weight.shape[1],
            weight_bytes=weight.numel() * weight.element_size(),
            arch=platform.arch_version,
            sm_count=platform.sm_count,
            pdl=pdl_enabled(),
        )
        is None
        # Asked only on SM100/SM103, where CUDA is present.
        or torch.cuda.is_current_stream_capturing()
    ):
        return
    with torch.no_grad():
        sample = x.new_zeros((m, weight.shape[1]))
        if not flashinfer_joint_bf16_supported(sample, weight, None):
            return
        tgv = _flashinfer_tgv()
        if tgv is None:
            return
        for tactic in tgv.tactics.values():
            flashinfer_tgv_decode_gemm(sample, weight, None, tactic)


# ===-----------------------------------------------------------------------===#
# FP8 block-scale prepacked layout: FlashInfer
# ===-----------------------------------------------------------------------===#


# Padding rows beyond this many tokens costs more than the prepacked path
# saves; aligned token counts always take the prepacked path.
_PREPACKED_PAD_THRESHOLD = 256


def use_flashinfer_fp8_blockscale_prepacked(num_tokens: int) -> bool:
    """Return whether the prepacked (MN-major scales) FP8 block-scale path
    should be used for ``num_tokens`` tokens."""
    return num_tokens <= _PREPACKED_PAD_THRESHOLD or num_tokens % 4 == 0


def prepare_flashinfer_fp8_blockscale_weight_scales(
    weight_scales: torch.Tensor,
) -> torch.Tensor:
    """Convert canonical ``[N // 128, K // 128]`` weight scales to the
    prepacked MN-major layout ``[K // 128, N // 128]`` consumed by
    ``gemm_fp8_nt_groupwise`` with ``scale_major_mode="MN"``."""
    return weight_scales.t().contiguous()
