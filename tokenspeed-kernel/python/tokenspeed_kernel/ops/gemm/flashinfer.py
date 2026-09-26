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
    dense_tensor_format,
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
            "block_scale_layout": frozenset({"canonical", "canonical_blackwell"}),
        },
        priority=Priority.SPECIALIZED + 3,
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


_CUTE_DSL_BACKEND = "cute-dsl"
_CUTE_DSL_SM100_ARCHS = frozenset({ArchVersion(10, 0), ArchVersion(10, 3)})


# ---- FlashInfer BF16 x NVFP4 GEMM, cute-dsl backend -----------------------

_mm_bf16_fp4 = error_fn
_prepare_bf16_fp4_weights = error_fn

if platform.is_nvidia and platform.arch_version in _CUTE_DSL_SM100_ARCHS:
    try:
        from flashinfer.gemm import mm_bf16_fp4 as _mm_bf16_fp4
        from flashinfer.gemm import (
            prepare_bf16_fp4_weights as _prepare_bf16_fp4_weights,
        )
    except ImportError:
        pass

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


# ---- FlashInfer BF16 low-latency GEMM, cute-dsl backend ------------------

_mm_bf16 = error_fn

if platform.is_nvidia and platform.arch_version in _CUTE_DSL_SM100_ARCHS:
    try:
        from flashinfer import mm_bf16 as _mm_bf16
    except ImportError:
        pass


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
        backend=_CUTE_DSL_BACKEND,
    )
