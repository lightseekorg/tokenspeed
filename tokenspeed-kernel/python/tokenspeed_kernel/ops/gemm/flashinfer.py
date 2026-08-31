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
from collections.abc import Callable
from typing import get_args

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import (
    ScaleFormat,
    format_signature,
    format_signatures,
    tensor_format,
)

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

# ---- FlashInfer block-scaled FP8 ----------------------------------------

gemm_fp8_nt_groupwise = error_fn
tinygemm_bf16 = error_fn

if platform.is_hopper_plus:
    try:
        from flashinfer.gemm import (
            gemm_fp8_nt_groupwise,
        )
        from flashinfer.gemm import tinygemm_bf16 as _tinygemm_bf16
    except ImportError:
        pass
    else:

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


# Past ~224 rows (GB300, K=7168) padding M costs more than the transpose it saves.
_PREPACKED_PAD_TOKEN_LIMIT = 256


def use_flashinfer_fp8_blockscale_prepacked(num_tokens: int) -> bool:
    """Whether MN-major prepacked scales beat canonical scales for this M.

    Args:
        num_tokens: Row count ``M`` of the activation matrix.

    Returns:
        True when the prepared MN-major path avoids more work than it adds.
        Row counts that are already a multiple of four need no padding at all,
        so the quantizer's native output is used as-is.
    """
    return num_tokens % 4 == 0 or num_tokens <= _PREPACKED_PAD_TOKEN_LIMIT


def prepare_flashinfer_fp8_blockscale_weight_scales(
    scales: torch.Tensor,
) -> torch.Tensor:
    """Pack canonical weight scales into FlashInfer's MN-major layout.

    Args:
        scales: Contiguous canonical scales shaped ``[N / 128, K / 128]``.

    Returns:
        A contiguous tensor shaped ``[K / 128, N / 128]``. This conversion is
        intended to run once after weight loading rather than in every GEMM.
    """
    if scales.ndim != 2:
        raise ValueError(f"weight scales must be 2-D, got shape {tuple(scales.shape)}")
    if scales.dtype != torch.float32:
        raise ValueError(
            "FlashInfer FP8 block-scale weight scales must use float32, "
            f"got {scales.dtype}"
        )
    return scales.transpose(0, 1).contiguous()


def _validate_flashinfer_fp8_blockscale_prepacked(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor,
    B_scales: torch.Tensor,
    original_m: int,
    block_size: list[int] | None,
) -> None:
    """Validate the prepared-layout contract without modifying its inputs."""
    if block_size is not None and tuple(block_size) != (128, 128):
        raise ValueError(
            "prepacked FlashInfer scales require block_size=[128, 128], "
            f"got {block_size}"
        )
    if not 0 < original_m <= A.shape[0]:
        raise ValueError(f"original_m must be in [1, {A.shape[0]}], got {original_m}")
    if A.shape[0] % 4:
        raise ValueError(
            "prepacked FlashInfer activations must have an M dimension "
            f"divisible by four, got {A.shape[0]}"
        )

    expected_a_scales = (A.shape[1] // 128, A.shape[0])
    expected_b_scales = (B.shape[1] // 128, B.shape[0] // 128)
    if tuple(A_scales.shape) != expected_a_scales or not A_scales.is_contiguous():
        raise ValueError(
            "prepacked activation scales must be contiguous with shape "
            f"{expected_a_scales}, got shape={tuple(A_scales.shape)} "
            f"stride={tuple(A_scales.stride())}"
        )
    if tuple(B_scales.shape) != expected_b_scales or not B_scales.is_contiguous():
        raise ValueError(
            "prepacked weight scales must be contiguous with shape "
            f"{expected_b_scales}, got shape={tuple(B_scales.shape)} "
            f"stride={tuple(B_scales.stride())}"
        )


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
            "n_align_128": frozenset({True}),
            "k_align_128": frozenset({True}),
            "block_scale_layout": frozenset(
                {"canonical", "canonical_blackwell", "flashinfer_mn"}
            ),
        },
        priority=Priority.SPECIALIZED + 3,
        tags={"throughput"},
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
        prepacked_scales: bool = False,
        original_m: int | None = None,
    ) -> torch.Tensor:
        """Run FlashInfer FP8 GEMM with canonical or prepared scales.

        Set ``prepacked_scales`` only when ``A_scales`` and ``B_scales`` already
        use FlashInfer's contiguous MN-major layout. The default canonical path
        passes the native K-major layouts through without copies.
        """
        assert (
            A_scales is not None
        ), "A_scales is required; online quantization should be done by the caller"
        assert B_scales is not None, "B_scales is required for FP8 blockscale GEMM"
        orig_m = A.shape[0] if original_m is None else int(original_m)
        if prepacked_scales:
            _validate_flashinfer_fp8_blockscale_prepacked(
                A,
                B,
                A_scales,
                B_scales,
                orig_m,
                block_size,
            )
            output = gemm_fp8_nt_groupwise(
                A,
                B,
                A_scales,
                B_scales,
                scale_major_mode="MN",
                out_dtype=out_dtype,
            )
            output = output[:orig_m] if output.shape[0] != orig_m else output
            if out is not None:
                out.copy_(output)
                return out
            return output

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


# ---- FlashInfer MXFP8 (1,32) ue8m0, cute-dsl backend ---------------------

mm_mxfp8 = error_fn

if platform.is_nvidia and platform.is_blackwell:
    try:
        from flashinfer.gemm import mm_mxfp8
    except ImportError:
        pass

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


if mm_mxfp8 is not error_fn:
    from tokenspeed_kernel.ops.gemm.fp8_utils import swizzle_mxfp8_scale

    @register_kernel(
        "gemm",
        "mm",
        name="flashinfer_mm_mxfp8",
        solution="flashinfer",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=_MXFP8_1X32_FORMAT_SIGNATURES,
        traits={
            "k_align_32": frozenset({True}),
            "n_min_128": frozenset({True}),
            "k_min_128": frozenset({True}),
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


# ---- FlashInfer FP4 -----------------------------------------------------

mm_fp4 = error_fn

if platform.is_nvidia and platform.is_blackwell:
    try:
        from flashinfer import mm_fp4
    except ImportError:
        pass

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


# ---- FlashInfer BF16 low-latency GEMM, cute-dsl backend ------------------

_CUTE_DSL_BF16_BACKEND = "cute-dsl"
_CUTE_DSL_BF16_ARCHS = frozenset({ArchVersion(10, 0), ArchVersion(10, 3)})

_mm_bf16 = error_fn

if platform.is_nvidia and platform.arch_version in _CUTE_DSL_BF16_ARCHS:
    try:
        from flashinfer import mm_bf16 as _mm_bf16
    except ImportError:
        pass


def _declares_cute_dsl_backend(mm_bf16: Callable[..., object]) -> bool:
    """Whether this ``mm_bf16`` lists :data:`_CUTE_DSL_BF16_BACKEND`.

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
    return _CUTE_DSL_BF16_BACKEND in get_args(backend.annotation)


@functools.lru_cache(maxsize=1)
def has_flashinfer_cute_dsl_bf16() -> bool:
    """Whether the flashinfer cute-dsl BF16 low-latency GEMM is usable here.

    Returns:
        True when running on an SM100 or SM103 GPU with a flashinfer build
        whose ``mm_bf16`` declares the backend.
    """
    return _mm_bf16 is not error_fn and _declares_cute_dsl_backend(_mm_bf16)


def flashinfer_cute_dsl_mm_bf16(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ weight.T (+ bias)`` through the cute-dsl ``mm_bf16`` backend.

    Args:
        x: ``[M, K]`` contiguous BF16 activation.
        weight: ``[N, K]`` contiguous BF16 weight; its transpose is the
            column-major ``(K, N)`` operand the backend wants, with no copy.
        bias: Optional contiguous ``[N]`` BF16 bias, fused into the epilogue.
        out: Optional ``[M, N]`` BF16 destination; allocated when omitted.

    Returns:
        ``[M, N]`` BF16 output, ``out`` when it was given.
    """
    return _mm_bf16(
        x,
        weight.t(),
        bias=bias,
        pdl=pdl_enabled(),
        out=out,
        backend=_CUTE_DSL_BF16_BACKEND,
    )
