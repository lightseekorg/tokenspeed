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

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial
from math import prod
from types import MappingProxyType

import torch
from tokenspeed_kernel.compile_monitor import is_serving
from tokenspeed_kernel.ops.gemm._flashinfer.fp8_quant import (
    flashinfer_fp8_blockscale_quantize_prepacked,
)
from tokenspeed_kernel.ops.gemm._triton.bf16x3 import (
    BF16X3_MIN_M,
    split_fp32_weight_bf16x3,
    triton_bf16x3_gemm_fp32,
)
from tokenspeed_kernel.ops.gemm._triton.gemv import rowcta_gemv_add3
from tokenspeed_kernel.ops.gemm._triton.kimi3 import (
    kimi3_projection_gemv,
    kimi3_shared_situ_projection_gemv,
)
from tokenspeed_kernel.ops.gemm.triton_router_gemv import fp32_router_gemv_counters
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.ops.quantization.triton import _quantize_fp8_group32_swizzled
from tokenspeed_kernel.platform import (
    Platform,
    current_platform,
    pdl_enabled,
)
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import (
    KernelPlan,
    NoKernelFoundError,
    SelectedKernel,
    resolve_kernel_override,
    select_kernel,
    spec_matches_shape_traits,
    spec_matches_traits,
)
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_format_signature,
    dense_tensor_format,
    format_signature,
    tensor_format,
)
from tokenspeed_kernel.thirdparty.cute_dsl.skinny_gemm import (
    SkinnyGemmConfig,
    shape_dynamic_skinny_gemm,
)
from tokenspeed_kernel.weights import get_weight_broker
from tokenspeed_kernel.workspace import zeroed_state

logger = logging.getLogger(__name__)

__all__ = [
    # GEMM
    "bmm",
    "mm",
    "plan_dense_gemm",
    # NVFP4 x BF16 weight preparation
    "has_flashinfer_cute_dsl_nvfp4_a16",
    "prepare_nvfp4_a16_weights",
    # DeepSeek V4 projections
    "dsv4_grouped_output_projection",
    "dsv4_grouped_output_projection_plan",
    "dsv4_grouped_output_projection_preprocessor",
    "dsv4_grouped_output_projection_warmup",
    "dsv4_grouped_output_projection_warmup_model",
    "dsv4_linear_fp32",
    "grouped_bf16_projection",
    "grouped_bf16_projection_quantize_fp8",
    # MV: small-M dense projection
    "mv",
    "mv_weightless_rmsnorm_gated_sigmoid",
    "mv_weightless_rmsnorm_gated_sigmoid_supported",
    # Linear AttnRes partials
    "fp32_router_gemv",
    "fp32_router_gemv_counters",
    "fp32_router_gemv_execute",
    "fp32_router_gemv_plan",
    "fp32_router_gemv_process_weight",
    "has_flashinfer_cute_dsl_nvfp4_a16",
    "linear_attnres_partials",
    "linear_attnres_partials_available",
    # Kimi K3 projections
    "KIMI3_HIDDEN_SIZE",
    "KIMI3_LATENT_SIZE",
    "KIMI3_QKVFAB_SIZE",
    "KIMI3_ROUTER_SIZE",
    "KIMI3_SHARED_GATE_UP_LOCAL_SIZE",
    "KIMI3_SHARED_LOCAL_SIZE",
    "Kimi3MLAQKVGateProjection",
    "kimi3_latent_projection",
    "kimi3_latent_projection_add3",
    "kimi3_mla_qkv_gate_projection",
    "kimi3_qkvfab_projection",
    "kimi3_router_projection",
    "kimi3_shared_down_projection",
    "kimi3_shared_situ_projection",
    # MXFP4 linear with dynamic MXFP4 activations
    "Mxfp4LinearWeight",
    "mxfp4_linear",
    "mxfp4_swiglu_linear",
    "prepare_mxfp4_linear",
    "mm",
    "prepare_mxfp8_weight_scales",
    "prepare_nvfp4_a16_weights",
]

_platform = Platform.get()
_fp8_dtype = torch.float8_e4m3fn

# ===-----------------------------------------------------------------------===#
# Selection traits
# ===-----------------------------------------------------------------------===#
#
# The trait dicts this module passes to ``select_kernel`` and the ``traits``
# that gemm registrations declare share one vocabulary. Problem-shape traits
# describe the kernel's supported envelope:
#
#   batch, m, n, k       exact dimensions; the request carries an int, the
#                        spec a set of supported values
#   <dim>_align          spec only: set of accepted alignments for <dim>
#   <dim>_min            spec only: set of accepted minimums for <dim>
#   mnk_problem_filter   spec only: ``(m, n, k) -> bool`` predicates for rules
#                        the above cannot express
#
# ``selection.spec_matches_shape_traits`` evaluates these; a spec that
# constrains a dimension rejects a request that does not supply it. Every
# other trait (layout flags such as ``a_inner_stride_one``, plus
# ``block_scale_layout``, ``out_dtype``, ``pdl_enabled``, ...) is matched by
# set membership in ``selection.spec_matches_traits``.
#
# Trait dicts list the shape traits first, in ``batch``, ``m``, ``n``, ``k``
# order with ``_align``/``_min`` after the exact sets and
# ``mnk_problem_filter`` last, followed by the remaining traits alphabetically.


# ===-----------------------------------------------------------------------===#
# GEMM: mm and bmm
# ===-----------------------------------------------------------------------===#


# Kernels that accept and own bias application inside their GEMM wrapper.
# For any kernel not listed here, dispatch applies the bias with a post-GEMM
# add instead of passing it to the kernel.
_KERNELS_WITH_FUSED_BIAS: frozenset[str] = frozenset(
    {
        "torch_bmm",
        "torch_mm",
        "triton_mm_fp8_scaled",
    }
)

# Kernels that accept an ``enable_pdl`` kwarg for Programmatic Dependent Launch.
_KERNELS_WITH_PDL: frozenset[str] = frozenset(
    {
        "flashinfer_cute_dsl_mm_nvfp4",
        "flashinfer_cute_dsl_mm_nvfp4_a16",
        "flashinfer_mm_nvfp4",
    }
)


def _as_2d_tensor_scale(scale: torch.Tensor | None) -> torch.Tensor | None:
    """A single-element scale of rank < 2 as its ``[1, 1]`` view."""
    if scale is not None and scale.dim() < 2 and scale.numel() == 1:
        return scale.view(1, 1)
    return scale


def _infer_scale_type(
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
) -> str | None:
    """For fp8, distinguish tensor/channel/scalar scaling."""
    if A_scales is None or B_scales is None:
        return None
    if A_scales.numel() == 1 and B_scales.numel() == 1:
        return "tensor"
    return "channel"


def _scale_storage_dtype(*scales: torch.Tensor | None) -> torch.dtype:
    for scale in scales:
        if scale is not None:
            return scale.dtype
    return torch.float32


def _gemm_format_signature(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    out_dtype: torch.dtype,
    quant: str | None,
    block_size: list[int] | None,
):
    _ = out_dtype
    if quant == "mxfp8":
        if block_size is None:
            raise ValueError("mxfp8 format selection requires block_size")
        if B_scales is None:
            raise ValueError("mxfp8 format selection requires B_scales")
        # Match the scale encoding used by online activation quantization.
        online_scale_dtype = (
            torch.uint8 if B_scales.dtype == torch.uint8 else torch.float32
        )
        a_scale = ScaleFormat(
            storage_dtype=(
                A_scales.dtype if A_scales is not None else online_scale_dtype
            ),
            granularity="block",
            block_shape=tuple(block_size),
        )
        b_scale = ScaleFormat(
            storage_dtype=B_scales.dtype,
            granularity="block",
            block_shape=tuple(block_size),
        )
        a_storage_dtype = _fp8_dtype if A_scales is None else A.dtype
        return format_signature(
            a=tensor_format("mxfp8", a_storage_dtype, scale=a_scale),
            b=tensor_format("mxfp8", B.dtype, scale=b_scale),
        )
    if quant == "fp8":
        if A_scales is None and A.dtype != _fp8_dtype and B_scales is not None:
            # Online per-token quantization (see mm): the kernel receives FP8
            # activations with FP32 per-token scales, or quantizes them itself.
            scale = ScaleFormat(storage_dtype=torch.float32, granularity="channel")
            return format_signature(
                a=tensor_format("scaled-fp8", _fp8_dtype, scale=scale),
                b=tensor_format("scaled-fp8", B.dtype, scale=scale),
            )
        scale = ScaleFormat(
            storage_dtype=_scale_storage_dtype(A_scales, B_scales),
            granularity=_infer_scale_type(A_scales, B_scales) or "unknown",
        )
        return format_signature(
            a=tensor_format("scaled-fp8", A.dtype, scale=scale),
            b=tensor_format("scaled-fp8", B.dtype, scale=scale),
        )
    if quant == "nvfp4_a16":
        if A_scales is not None:
            raise ValueError("nvfp4_a16 requires A_scales=None for dense activations")
        if B_scales is None:
            raise ValueError("nvfp4_a16 format selection requires B_scales")
        b_scale = ScaleFormat(
            storage_dtype=B_scales.dtype,
            granularity="block",
            block_shape=(16,),
        )
        return format_signature(
            a=dense_tensor_format(A.dtype),
            b=tensor_format("nvfp4", B.dtype, scale=b_scale),
        )
    if quant == "nvfp4":
        a_scale = ScaleFormat(
            storage_dtype=_scale_storage_dtype(A_scales),
            granularity="block",
            block_shape=(16,),
        )
        b_scale = ScaleFormat(
            storage_dtype=_scale_storage_dtype(B_scales),
            granularity="block",
            block_shape=(16,),
        )
        return format_signature(
            a=tensor_format("nvfp4", A.dtype, scale=a_scale),
            b=tensor_format("nvfp4", B.dtype, scale=b_scale),
        )
    if quant == "mxfp4":
        a_scale = ScaleFormat(
            storage_dtype=_scale_storage_dtype(A_scales),
            granularity="block",
            block_shape=(32,),
        )
        b_scale = ScaleFormat(
            storage_dtype=_scale_storage_dtype(B_scales),
            granularity="block",
            block_shape=(32,),
        )
        return format_signature(
            a=tensor_format("mxfp4", A.dtype, scale=a_scale),
            b=tensor_format("mxfp4", B.dtype, scale=b_scale),
        )
    return format_signature(
        a=dense_tensor_format(A.dtype), b=dense_tensor_format(B.dtype)
    )


def _online_quantize_mxfp8(
    A: torch.Tensor,
    block_size: list[int],
    scale_encoding: str,
    *,
    enable_pdl: bool,
    kernel_name: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    gfx1250_gluon = kernel_name in {
        "gluon_mm_mxfp8_ue8m0_gfx1250",
        "gluon_mm_mxfp8_ue8m0_largem_gfx1250",
    }
    return quantize_fp8(
        A,
        granularity="token_group",
        group_size=block_size[1],
        scale_encoding=scale_encoding,
        enable_pdl=False if gfx1250_gluon else enable_pdl,
        solution=None if gfx1250_gluon else "triton",
    )


def _kernel_handles_online_mxfp8(kernel_name: str) -> bool:
    spec = KernelRegistry.get().get_by_name(kernel_name)
    return spec is not None and spec.solution == "torch"


def _validate_gemm_out(
    out: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    op: str,
) -> None:
    if tuple(out.shape) != shape:
        raise ValueError(f"{op} out expects shape {shape}, got {tuple(out.shape)}")
    if out.dtype != dtype:
        raise ValueError(f"{op} out expects dtype {dtype}, got {out.dtype}")
    if out.device != device:
        raise ValueError(f"{op} out expects device {device}, got {out.device}")
    if out.stride(-1) != 1:
        raise ValueError(f"{op} out must have stride(-1) == 1")


def _mxfp8_swizzled_scale_count(rows: int, k: int) -> int:
    # The F8_128x4 layout pads rows to 128 and scale columns to 4 (K to 128).
    return -(-rows // 128) * 128 * -(-k // 128) * 4


def prepare_mxfp8_weight_scales(
    weight: torch.Tensor, weight_scales: torch.Tensor
) -> torch.Tensor | None:
    """Swizzle 1x32 UE8M0 weight scales once for ``mm(prepacked_scales=True)``.

    The prepacked call skips the per-call weight-scale swizzle and quantizes
    activations straight into the swizzled layout. Whether the native kernel
    serves this weight is decided here, from the weight shape, the PDL policy
    and any forced ``gemm``/``mm`` or ``quantization``/``fp8_with_scale``
    kernel, so callers keep their canonical path when this returns None.

    Args:
        weight: E4M3 weight ``[N, K]`` with unit inner stride.
        weight_scales: Canonical uint8 UE8M0 scales ``[N, K // 32]``; left
            unchanged.

    Returns:
        New 1-D uint8 scales in FlashInfer's F8_128x4 layout, or None when
        the prepacked path does not apply.
    """
    if (
        weight.ndim != 2
        or weight.dtype != _fp8_dtype
        or weight.shape[1] % 32
        or weight_scales.dtype != torch.uint8
        or tuple(weight_scales.shape) != (weight.shape[0], weight.shape[1] // 32)
        or weight_scales.device != weight.device
    ):
        raise ValueError("1x32 FP8 weights require canonical UE8M0 [N, K/32] scales")
    if not (
        has_flashinfer_mxfp8()
        and weight.is_cuda
        and min(weight.shape) >= 128
        and weight.stride(-1) == 1
        and pdl_enabled()
        and resolve_kernel_override("gemm", "mm", None) in (None, "flashinfer_mm_mxfp8")
        and resolve_kernel_override("quantization", "fp8_with_scale", None) is None
    ):
        return None
    return swizzle_mxfp8_scale(weight_scales, *weight.shape)


def _mm_mxfp8_prepacked(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scales: torch.Tensor | None,
    B_scales: torch.Tensor | None,
    *,
    bias: torch.Tensor | None,
    out: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    n, k = B.shape
    if not has_flashinfer_mxfp8():
        raise ValueError(
            "mm(prepacked_scales=True) with block_size [1, 32] requires "
            "FlashInfer's MXFP8 GEMM"
        )
    quantized = A.dtype == _fp8_dtype
    if (
        A.ndim != 2
        or A.dtype not in (torch.bfloat16, torch.float16, _fp8_dtype)
        or A.shape[1] != k
        or A.stride(-1) != 1
        or quantized != (A_scales is not None)
    ):
        raise ValueError(
            "mm(prepacked_scales=True) with block_size [1, 32] requires 16-bit "
            "[M, K] activations, or E4M3 codes with their UE8M0 scales, with "
            "unit inner stride"
        )
    if quantized and (
        A_scales.dtype != torch.uint8 or tuple(A_scales.shape) != (A.shape[0], k // 32)
    ):
        raise ValueError(
            "mm(prepacked_scales=True) requires canonical [M, K/32] UE8M0 "
            "activation scales with E4M3 codes"
        )
    if (
        B_scales is None
        or B_scales.dtype != torch.uint8
        or tuple(B_scales.shape) != (_mxfp8_swizzled_scale_count(n, k),)
    ):
        raise ValueError(
            "mm(prepacked_scales=True) with block_size [1, 32] requires weight "
            "scales from prepare_mxfp8_weight_scales"
        )
    if out is not None:
        _validate_gemm_out(
            out, shape=(A.shape[0], n), dtype=out_dtype, device=A.device, op="mm"
        )
    if A.shape[0] == 0:
        return out if out is not None else A.new_empty((0, n), dtype=out_dtype)
    if quantized:
        # Pre-quantized codes keep their canonical scales; the native kernel
        # swizzles these per call, which is small next to the weight scales.
        values, activation_scales = A, A_scales
    else:
        # Same codes and scales as the reference 1x32 quantizer, written
        # directly in the layout the native kernel reads.
        values, activation_scales = _quantize_fp8_group32_swizzled(A)
    result = flashinfer_mm_mxfp8(
        values,
        B,
        activation_scales,
        B_scales,
        out_dtype,
        block_size=[1, 32],
        out=out if bias is None else None,
    )
    if bias is None:
        return result
    result = result + bias
    if out is not None:
        out.copy_(result)
        return out
    return result


def plan_dense_gemm(
    *,
    mode: str,
    input_size: int,
    output_size: int,
    dtype: torch.dtype,
    batch_size: int | None,
    batch_invariant: bool,
) -> KernelPlan:
    """Plan dense MM/BMM from model geometry, without allocating example tokens.

    ``batch_size`` is the fixed head/group count for BMM, and None for MM.
    The token dimension M remains unrestricted. Both inputs and output use dtype.
    """
    if mode not in ("mm", "bmm") or (mode == "bmm") != (batch_size is not None):
        raise ValueError("Dense GEMM planning requires mm or bmm with its group count")
    traits = {"n": output_size, "k": input_size, "out_dtype": dtype}
    if batch_size is not None:
        traits["batch"] = batch_size
    return KernelPlan.select(
        "gemm",
        mode,
        format_signature(a=dense_tensor_format(dtype), b=dense_tensor_format(dtype)),
        features=frozenset({"batch_invariant"}) if batch_invariant else frozenset(),
        static_traits=traits,
        variable_traits=frozenset({"m"}),
        solution=None,
        declared_traits=frozenset(),
        ignore_layout=False,
    )


def mm(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    A_scales: torch.Tensor | None = None,
    B_scales: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    quant: str | None = None,
    plan: KernelPlan | None = None,
    override: str | None = None,
    solution: str | None = None,
    prepacked_scales: bool = False,
) -> torch.Tensor:
    """Dense matrix multiply with automatic kernel selection.

    Quantization type is inferred from input dtype and the presence of
    scales, or can be set explicitly via ``quant``.  When ``A_scales``
    is ``None`` for a quantized mode (e.g. ``quant="mxfp8"``), online
    activation quantization is performed here before calling the kernel.

    Args:
        A: Activation matrix ``[M, K]``.
        B: Weight matrix.
        A_scales: Activation scales. With ``quant="fp8"``, None for BF16/FP16
            ``A`` quantizes it per token here before the call (FP8 E4M3, one
            FP32 scale per row, by ``quantize_fp8(granularity="token")``).
        B_scales: Weight scales (layout depends on quant type).
        bias: Optional bias vector of shape ``[N]`` added to the
            output.  When the selected kernel supports a fused bias
            epilogue (see ``_KERNELS_WITH_FUSED_BIAS``) it is passed
            into the kernel; otherwise it is added after the GEMM.
        out: Optional output buffer. The output may be a strided view
            but must have contiguous rows (``stride(-1) == 1``).
        out_dtype: Output dtype (defaults to ``A.dtype``).
        alpha: Global scaling factor (nvfp4 modes only).
        block_size: Block size for block-wise quantization, e.g.
            ``[128, 128]``
        quant: Explicit quant type override. One of ``"mxfp8"``, ``"fp8"``,
            ``"nvfp4"``, ``"nvfp4_a16"``, ``"mxfp4"``, ``"none"``.
            ``"nvfp4_a16"`` uses dense BF16 activations and prepared packed
            NVFP4 weights. If ``None``, inferred from input dtypes and scales.
        plan: Initialization-time kernel selection; execution never reselects.
        override: Force selection of a specific kernel by name (e.g.
            ``"cublaslt_mm_nvfp4"``). Bypasses heuristic scoring.
        solution: Restrict selection to a registered implementation family.
        prepacked_scales: ``B_scales`` are prepared once per weight and ``A``
            is quantized here into the matching layout. With block_size
            ``[128, 128]`` they come from
            ``prepare_flashinfer_fp8_blockscale_weight_scales``; with
            ``[1, 32]`` from ``prepare_mxfp8_weight_scales``, and ``A`` may
            also be E4M3 codes with canonical ``[M, K/32]`` UE8M0 ``A_scales``.
    """
    if plan is not None and (
        prepacked_scales
        or quant not in (None, "none")
        or A_scales is not None
        or B_scales is not None
    ):
        raise ValueError("A dense GEMM plan cannot use online quantization")
    override = resolve_kernel_override("gemm", "mm", override)
    enable_pdl = pdl_enabled()
    out_dtype = out_dtype or (out.dtype if out is not None else A.dtype)
    # Per-tensor scales may arrive as 0-dim or [1]; kernels take them as [1, 1].
    A_scales = _as_2d_tensor_scale(A_scales)
    B_scales = _as_2d_tensor_scale(B_scales)

    M = A.shape[0]
    b_layout = "NK"
    if quant == "mxfp4":
        K = A.shape[-1] * 2
        N = B.shape[0]
    elif quant == "nvfp4_a16":
        K = A.shape[-1]
        N = B.shape[0]
    else:
        K = A.shape[-1]
        b_layout = "KN" if B.shape[0] == K else "NK"
        N = B.shape[-1] if b_layout == "KN" else B.shape[0]

    if prepacked_scales:
        if quant != "mxfp8" or block_size is None:
            raise ValueError(
                "mm(prepacked_scales=True) requires quant='mxfp8' with block_size"
            )
        if list(block_size) == [1, 32]:
            if (
                alpha is not None
                or solution is not None
                or override
                not in (
                    None,
                    "flashinfer_mm_mxfp8",
                )
            ):
                raise ValueError(
                    "mm(prepacked_scales=True) with block_size [1, 32] runs only "
                    "flashinfer_mm_mxfp8 and takes no alpha or solution"
                )
            return _mm_mxfp8_prepacked(
                A, B, A_scales, B_scales, bias=bias, out=out, out_dtype=out_dtype
            )
        if B_scales is None or tuple(B_scales.shape) != (
            K // block_size[1],
            B.shape[0] // block_size[0],
        ):
            raise ValueError(
                "mm(prepacked_scales=True) requires prepacked weight scales in "
                "the MN-major layout from "
                "prepare_flashinfer_fp8_blockscale_weight_scales"
            )
        M = A.shape[0]
        values, activation_scales = flashinfer_fp8_blockscale_quantize_prepacked(
            A, block_size[1]
        )
        result = gemm_fp8_nt_groupwise(
            values,
            B,
            activation_scales,
            B_scales,
            scale_major_mode="MN",
            out_dtype=out_dtype,
        )[:M]
        if out is not None:
            out.copy_(result)
            return out
        return result

    if out is not None:
        _validate_gemm_out(
            out,
            shape=(M, N),
            dtype=out_dtype,
            device=A.device,
            op="mm",
        )

    # A dense layer's large-M arm reaches mm directly, while small M can
    # select mv. Expose those branches at the shared kernel boundary.
    if (
        plan is None
        and solution is None
        and override is None
        and alpha is None
        and quant in (None, "none")
        and A_scales is None
        and B_scales is None
        and out_dtype == torch.bfloat16
        and B.shape[-1] == K
    ):
        # Row-parallel layers add bias only on rank 0. All ranks must expose
        # the same bias-free tuning branch for FI's timing collectives, even
        # when this rank's actual GEMM retains the generic biased path.
        autotune_bf16_gemm(A, B)
        if (
            bias is None
            and M <= BF16_GEMM_MAX_M
            and flashinfer_joint_bf16_supported(A, B, out)
            and not is_serving()
        ):
            shape_params = {"M": M, "N": N, "K": K}
            ShapeCapture.get().record(
                "gemm", "mm", "flashinfer_bf16_gemm", A.dtype, shape_params
            )
            with kernel_scope(
                "gemm",
                "mm",
                A.dtype,
                kernel_name="flashinfer_bf16_gemm",
                **shape_params,
                has_out=out is not None,
            ):
                return flashinfer_bf16_gemm(A, B, out)
        if bias is None and _use_mv(A, B):
            return mv(A, B, out)

    block_scale_layout = (
        "canonical_blackwell" if Platform.get().is_blackwell_plus else "canonical"
    )
    # Supported online FP32/UE8M0 quantizers materialize contiguous FP8 operands.
    # Describe the layout seen by GEMM, not the original dense activation view.
    online_contiguous_a = (
        quant == "mxfp8"
        and A_scales is None
        and A.dtype in (torch.bfloat16, torch.float16)
        and B_scales is not None
        and block_size is not None
        and (
            (B_scales.dtype == torch.float32 and block_size[1] in (32, 128))
            or (
                B_scales.dtype == torch.uint8
                and block_size[1] == 32
                and A.stride(-1) == 1
            )
        )
    )
    traits: dict[str, object] = {
        "m": M,
        "n": N,
        "k": K,
        "b_layout": b_layout,
        "a_contiguous": online_contiguous_a or A.is_contiguous(),
        "a_inner_stride_one": online_contiguous_a or A.stride(-1) == 1,
        "a_scales_inner_stride_one": (A_scales is None or A_scales.stride(-1) == 1),
        "b_contiguous": B.is_contiguous(),
        "b_inner_stride_one": B.stride(-1) == 1,
        "b_scales_contiguous": B_scales is None or B_scales.is_contiguous(),
        "b_scales_inner_stride_one": (B_scales is None or B_scales.stride(-1) == 1),
        "block_scale_layout": block_scale_layout,
        "has_alpha": alpha is not None,
        "out_contiguous": out is None or out.is_contiguous(),
        "out_dtype": out_dtype,
        "out_inner_stride_one": out is None or out.stride(-1) == 1,
        "pdl_enabled": enable_pdl,
    }
    if quant == "mxfp8" and block_size is not None and tuple(block_size) == (128, 128):
        # Contiguous offset views can still violate a backend's pointer alignment.
        # Online quantization allocates aligned activation values and scales.
        traits["operands_aligned_16"] = (
            online_contiguous_a or A.data_ptr() % 16 == 0
        ) and B.data_ptr() % 16 == 0
        traits["out_aligned_16"] = out is None or out.data_ptr() % 16 == 0
        traits["scales_aligned_16"] = (
            A_scales is None or A_scales.data_ptr() % 16 == 0
        ) and (B_scales is None or B_scales.data_ptr() % 16 == 0)

    signature = _gemm_format_signature(
        A, B, A_scales, B_scales, out_dtype, quant, block_size
    )
    select_dtype = signature.storage_dtype_for("a") or A.dtype

    if plan is not None:
        plan.validate(
            "gemm",
            "mm",
            signature,
            traits=traits,
            solution=solution,
            override=override,
            features=None,
        )
        kernel = plan.kernel
    else:
        kernel = select_kernel(
            "gemm",
            "mm",
            signature,
            traits=traits,
            solution=solution,
            override=override,
        )
    # Online activation quantization
    if (
        quant == "mxfp8"
        and A_scales is None
        and not _kernel_handles_online_mxfp8(kernel.name)
    ):
        assert (
            block_size is not None
        ), "block_size is required for online activation quantization"
        A, A_scales = _online_quantize_mxfp8(
            A,
            block_size,
            "ue8m0" if B_scales.dtype == torch.uint8 else "float32",
            enable_pdl=enable_pdl,
            kernel_name=kernel.name,
        )

    # Online per-token FP8 quantization of 16-bit activations.
    if (
        quant == "fp8"
        and A_scales is None
        and A.dtype != _fp8_dtype
        and B_scales is not None
    ):
        A, A_scales = quantize_fp8(A, granularity="token", enable_pdl=enable_pdl)

    kernel_args = (A, B, A_scales, B_scales, out_dtype)
    kernel_kwargs: dict[str, object] = {
        "alpha": alpha,
        "block_size": block_size,
    }
    if out is not None:
        kernel_kwargs["out"] = out
    fused_bias = bias is not None and kernel.name in _KERNELS_WITH_FUSED_BIAS
    if fused_bias:
        kernel_kwargs["bias"] = bias

    if kernel.name in _KERNELS_WITH_PDL:
        kernel_kwargs["enable_pdl"] = enable_pdl

    shape_params = {"M": M, "N": N, "K": K}
    ShapeCapture.get().record(
        "gemm",
        "mm",
        kernel.name,
        select_dtype,
        shape_params,
    )
    with kernel_scope(
        "gemm",
        "mm",
        select_dtype,
        kernel_name=kernel.name,
        **shape_params,
        has_out=out is not None,
    ):
        output = kernel(*kernel_args, **kernel_kwargs)

    if bias is not None and not fused_bias:
        if out is not None:
            output.add_(bias.to(dtype=output.dtype))
        else:
            output = output + bias.to(dtype=output.dtype)
    return output


def bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    A_scales: torch.Tensor | None = None,
    B_scales: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    alpha: torch.Tensor | None = None,
    block_size: list[int] | None = None,
    quant: str | None = None,
    enable_pdl: bool = False,
    plan: KernelPlan | None = None,
    override: str | None = None,
) -> torch.Tensor:
    """Batched matrix multiply with automatic kernel selection.

    Mirrors :func:`mm` for batched inputs with an outer batch dimension.
    ``A`` must use ``[B, M, K]`` layout and ``B`` must use ``[B, N, K]``
    layout. The result shape is ``[B, M, N]``.
    If ``out`` is provided, kernels that support direct output write into that
    buffer. The output may be a strided view but must have contiguous rows
    (``stride(-1) == 1``).

    Args:
        A: Activation batch ``[batch, M, K]``.
        B: Weight batch ``[batch, N, K]``.
        A_scales: Activation scales.
        B_scales: Weight scales (layout depends on quant type).
        bias: Optional bias vector of shape ``[N]`` or per-batch bias matrix
            of shape ``[batch, N]`` added to the output. When the selected
            kernel supports a fused bias epilogue (see
            ``_KERNELS_WITH_FUSED_BIAS``) it is passed into the kernel;
            otherwise it is added after the BMM.
        out: Optional output buffer. The output may be a strided view
            but must have contiguous rows (``stride(-1) == 1``).
        out_dtype: Output dtype (defaults to ``A.dtype``).
        alpha: Global scaling factor (nvfp4 only).
        block_size: Block size for block-wise quantization, e.g.
            ``[128, 128]``.
        quant: Explicit quant type override. One of ``"mxfp8"``,
            ``"fp8"``, ``"nvfp4"``, ``"mxfp4"``, ``"none"``.
            If ``None``, inferred from input dtypes and scales.
        enable_pdl: Whether to request Programmatic Dependent Launch support
            from kernels that accept it.
        plan: Initialization-time kernel selection; execution never reselects.
        override: Force selection of a specific kernel by name. Bypasses
            heuristic scoring.
    """
    out_dtype = out_dtype or (out.dtype if out is not None else A.dtype)
    if A.ndim != 3:
        raise ValueError(f"bmm expects A with shape [B, M, K], got {tuple(A.shape)}")
    if B.ndim != 3:
        raise ValueError(f"bmm expects B with shape [B, N, K], got {tuple(B.shape)}")

    batch, M, A_storage_K = A.shape
    B_batch, N, B_storage_K = B.shape
    if B_batch != batch:
        raise ValueError(f"bmm batch mismatch: A batch={batch}, B batch={B_batch}")
    if B_storage_K != A_storage_K:
        raise ValueError(f"bmm K mismatch: A K={A_storage_K}, B K={B_storage_K}")
    K = A_storage_K * 2 if quant == "mxfp4" else A_storage_K

    if out is not None:
        _validate_gemm_out(
            out,
            shape=(batch, M, N),
            dtype=out_dtype,
            device=A.device,
            op="bmm",
        )

    traits: dict[str, object] = {
        "batch": batch,
        "m": M,
        "n": N,
        "k": K,
        "a_inner_stride_one": A.stride(-1) == 1,
        "b_n_stride_one": B.stride(1) == 1,
        "out_dtype": out_dtype,
        "out_inner_stride_one": out is None or out.stride(-1) == 1,
    }

    signature = _gemm_format_signature(
        A, B, A_scales, B_scales, out_dtype, quant, block_size
    )
    select_dtype = signature.storage_dtype_for("a") or A.dtype

    if plan is not None:
        plan.validate(
            "gemm",
            "bmm",
            signature,
            traits=traits,
            override=override,
            features=None,
            solution=None,
        )
        kernel = plan.kernel
    else:
        kernel = select_kernel(
            "gemm",
            "bmm",
            signature,
            traits=traits,
            override=override,
        )

    if (
        quant == "mxfp8"
        and A_scales is None
        and not _kernel_handles_online_mxfp8(kernel.name)
    ):
        assert (
            block_size is not None
        ), "block_size is required for online activation quantization"
        A, A_scales = _online_quantize_mxfp8(
            A,
            block_size,
            "ue8m0" if B_scales.dtype == torch.uint8 else "float32",
            enable_pdl=enable_pdl,
            kernel_name=kernel.name,
        )

    kernel_args = (A, B, A_scales, B_scales, out_dtype)
    kernel_kwargs: dict[str, object] = {
        "alpha": alpha,
        "block_size": block_size,
    }
    if out is not None:
        kernel_kwargs["out"] = out

    fused_bias = bias is not None and kernel.name in _KERNELS_WITH_FUSED_BIAS
    if fused_bias:
        kernel_kwargs["bias"] = bias

    if kernel.name in _KERNELS_WITH_PDL:
        kernel_kwargs["enable_pdl"] = enable_pdl

    shape_params = {"B": batch, "M": M, "N": N, "K": K}
    ShapeCapture.get().record(
        "gemm",
        "bmm",
        kernel.name,
        select_dtype,
        shape_params,
    )
    with kernel_scope(
        "gemm",
        "bmm",
        select_dtype,
        kernel_name=kernel.name,
        **shape_params,
        has_out=out is not None,
    ):
        output = kernel(*kernel_args, **kernel_kwargs)

    if bias is not None and not fused_bias:
        bias = bias.to(dtype=output.dtype)
        if bias.ndim == 1:
            bias_view = bias.view(1, 1, -1)
        elif bias.ndim == 2:
            bias_view = bias.view(bias.shape[0], 1, bias.shape[1])
        else:
            raise ValueError(f"bmm bias expects shape [N] or [B, N], got {bias.shape}")
        if out is not None:
            output.add_(bias_view)
        else:
            output = output + bias_view
    return output


# ===-----------------------------------------------------------------------===#
# MXFP4 linear with dynamic MXFP4 activations
# ===-----------------------------------------------------------------------===#


@dataclass(frozen=True)
class Mxfp4LinearWeight:
    """A dense MXFP4 weight prepared for the selected MXFP4 linear kernels.

    Attributes:
        weight: Packed E2M1 ``[N, K/2]`` uint8 in ``layout`` order.
        weight_scale: UE8M0 ``[N, K/32]`` uint8, one scale per 32 K values.
        layout: Weight layout the selected kernels read; the ``weight_layout``
            trait of their registrations.
    """

    weight: torch.Tensor
    weight_scale: torch.Tensor
    layout: str


def _mxfp4_linear_traits(n: int, k: int, layout: str | None) -> dict[str, object]:
    # ``weight_elements`` lets kernels bound the weight size (for example by
    # 32-bit buffer offsets) at selection, before the weight is prepared.
    traits: dict[str, object] = {"n": n, "k": k, "weight_elements": n * k}
    if layout is not None:
        traits["weight_layout"] = layout
    return traits


def _prepare_mxfp4_weight_layout(layout: str, weight: torch.Tensor) -> torch.Tensor:
    if layout == MXFP4_LINEAR_LINEAR_LAYOUT:
        return weight
    if layout == MXFP4_LINEAR_MFMA_LAYOUT:
        return prepare_mxfp4_linear_mfma_weight(weight)
    raise ValueError(f"unknown MXFP4 linear weight layout {layout!r}")


def prepare_mxfp4_linear(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    swiglu: bool,
    solution: str | None = None,
) -> Mxfp4LinearWeight:
    """Prepare a checkpoint MXFP4 weight for ``mxfp4_linear``.

    Selects the kernel that will run the layer and reorders the weight into
    the layout it reads, once, at load time.

    Args:
        weight: Packed E2M1 ``[N, K/2]`` uint8 in checkpoint order.
        weight_scale: UE8M0 ``[N, K/32]`` uint8.
        swiglu: Whether the layer runs ``mxfp4_swiglu_linear`` (``weight``
            stacks the gate rows over the up rows) instead of
            ``mxfp4_linear``.
        solution: Optional registered solution to restrict selection to.

    Returns:
        The prepared weight; ``weight`` itself when the layout is unchanged.
    """
    n, k = weight.shape[0], weight.shape[1] * 2
    kernel = select_kernel(
        "gemm",
        "mxfp4_swiglu_linear" if swiglu else "mxfp4_linear",
        MXFP4_LINEAR_SIGNATURE,
        traits=_mxfp4_linear_traits(n, k, None),
        solution=solution,
    )
    (layout,) = KernelRegistry.get().get_by_name(kernel.name).traits["weight_layout"]
    return Mxfp4LinearWeight(
        weight=_prepare_mxfp4_weight_layout(layout, weight),
        weight_scale=weight_scale,
        layout=layout,
    )


def _run_mxfp4_linear(
    mode: str,
    x: torch.Tensor,
    weight: Mxfp4LinearWeight,
    out: torch.Tensor | None,
) -> torch.Tensor:
    n, k = weight.weight.shape[0], weight.weight.shape[1] * 2
    if x.ndim != 2 or x.shape[1] != k or x.dtype != torch.bfloat16:
        raise ValueError(
            f"{mode} expects BF16 [M, {k}] activations, got {x.dtype} {tuple(x.shape)}"
        )
    kernel = select_kernel(
        "gemm",
        mode,
        MXFP4_LINEAR_SIGNATURE,
        traits=_mxfp4_linear_traits(n, k, weight.layout),
    )
    shape_params = {"M": x.shape[0], "N": n, "K": k}
    ShapeCapture.get().record("gemm", mode, kernel.name, x.dtype, shape_params)
    with kernel_scope(
        "gemm",
        mode,
        x.dtype,
        kernel_name=kernel.name,
        **shape_params,
        has_out=out is not None,
    ):
        return kernel(x, weight.weight, weight.weight_scale, out=out)


def mxfp4_linear(
    x: torch.Tensor,
    weight: Mxfp4LinearWeight,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense MXFP4 linear with dynamic MXFP4 activations.

    Computes ``mxfp4(x) @ W.T`` where ``mxfp4`` quantizes each 32-value group
    of ``x`` to E2M1 with a UE8M0 scale (the ``quantize_mxfp4`` recipe).

    Args:
        x: BF16 ``[M, K]`` activations with unit inner stride.
        weight: Weight from ``prepare_mxfp4_linear(..., swiglu=False)``.
        out: Optional BF16 ``[M, N]`` destination with unit inner stride.

    Returns:
        BF16 ``[M, N]``; ``out`` when provided.
    """
    return _run_mxfp4_linear("mxfp4_linear", x, weight, out)


def mxfp4_swiglu_linear(
    x: torch.Tensor,
    weight: Mxfp4LinearWeight,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense MXFP4 gate/up projection followed by ``silu(gate) * up``.

    Equals ``silu_and_mul(mxfp4_linear(x, weight))`` with the projection
    output rounded to BF16 before the activation, as the unfused path does.

    Args:
        x: BF16 ``[M, K]`` activations with unit inner stride.
        weight: Weight from ``prepare_mxfp4_linear(..., swiglu=True)``; its
            rows ``[0, N/2)`` are the gate and ``[N/2, N)`` the up projection.
        out: Optional BF16 ``[M, N/2]`` destination with unit inner stride.

    Returns:
        BF16 ``[M, N/2]``; ``out`` when provided.
    """
    return _run_mxfp4_linear("mxfp4_swiglu_linear", x, weight, out)


# ===-----------------------------------------------------------------------===#
# DeepSeek V4 projections
# ===-----------------------------------------------------------------------===#


@dataclass(frozen=True)
class _GroupedOutputProjectionPlan:
    input_dtype: torch.dtype
    weight_dtype: torch.dtype
    weight_scale_dtype: torch.dtype
    num_groups: int
    heads_per_group: int
    head_dim: int
    nope_dim: int
    rope_dim: int
    output_dim: int
    block_size: tuple[int, int]
    scale_format: str | None


def dsv4_grouped_output_projection_plan(
    *,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    weight_scale_dtype: torch.dtype,
    num_groups: int,
    heads_per_group: int,
    head_dim: int,
    nope_dim: int,
    rope_dim: int,
    output_dim: int,
    block_size: tuple[int, int] | list[int],
    scale_format: str | None = None,
) -> object:
    """Create an opaque plan for the DeepSeek V4 grouped output projection.

    The configuration records the projection geometry and source formats.

    Args:
        input_dtype: Attention output dtype before dynamic FP8 quantization.
        weight_dtype: Storage dtype of the grouped projection weight.
        weight_scale_dtype: Checkpoint storage dtype of the weight scales.
        num_groups: Number of output-projection groups local to this rank.
        heads_per_group: Number of local attention heads in each group.
        head_dim: Width of each attention head.
        nope_dim: Non-rotary width in each attention head.
        rope_dim: Rotary width in each attention head.
        output_dim: Output width of each grouped projection.
        block_size: Logical weight-scale block shape ``[block_n, block_k]``.
        scale_format: Logical checkpoint scale encoding, such as ``"ue8m0"``.

    Returns:
        An opaque plan accepted by the execution and warmup APIs.
    """
    if min(num_groups, heads_per_group, head_dim, output_dim) <= 0:
        raise ValueError("grouped output projection dimensions must be positive")
    if nope_dim < 0 or rope_dim <= 0 or nope_dim + rope_dim != head_dim:
        raise ValueError("nope_dim and rope_dim must partition head_dim")
    if len(block_size) != 2 or min(block_size) <= 0:
        raise ValueError("block_size must contain two positive dimensions")
    block_n, block_k = (int(block_size[0]), int(block_size[1]))
    if head_dim & (head_dim - 1) or block_k & (block_k - 1):
        raise ValueError("head_dim and block_k must be powers of two")
    input_dim = heads_per_group * head_dim
    if head_dim % block_k != 0 or input_dim % block_k != 0 or output_dim % block_n != 0:
        raise ValueError(
            "grouped output projection dimensions must be block aligned: "
            f"input_dim={input_dim}, output_dim={output_dim}, "
            f"block_size={(block_n, block_k)}"
        )
    if rope_dim % 2 != 0 or rope_dim > block_k:
        raise ValueError(
            "rope_dim must be even and fit in the final quantization block: "
            f"rope_dim={rope_dim}, block_k={block_k}"
        )

    return _GroupedOutputProjectionPlan(
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        weight_scale_dtype=weight_scale_dtype,
        num_groups=num_groups,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        output_dim=output_dim,
        block_size=(block_n, block_k),
        scale_format=scale_format,
    )


def _require_grouped_output_projection_plan(
    plan: object,
) -> _GroupedOutputProjectionPlan:
    if not isinstance(plan, _GroupedOutputProjectionPlan):
        raise TypeError("plan must be returned by dsv4_grouped_output_projection_plan")
    return plan


def dsv4_grouped_output_projection_preprocessor(
    plan: object,
    *,
    solution: str | None = None,
) -> tuple[tuple[str, ...], Callable] | None:
    """Select tensor preprocessing for a grouped output projection layer.

    Args:
        plan: Opaque grouped output projection plan.
        solution: Optional implementation family preference for preparation.

    Returns:
        Source attribute names and their transformation with geometry bound,
        or None when the layer should enroll canonical weights. The layer runs
        the returned preprocessor through the shared weight broker.
    """
    typed_plan = _require_grouped_output_projection_plan(plan)
    selected = select_kernel(
        "gemm",
        "dsv4_grouped_output_projection",
        format_signature(
            attention=dense_tensor_format(typed_plan.input_dtype),
            weight=dense_tensor_format(typed_plan.weight_dtype),
        ),
        traits={
            "block_size": typed_plan.block_size,
            "scale_format": typed_plan.scale_format,
            "weight_scale_dtype": typed_plan.weight_scale_dtype,
        },
        solution=solution,
        ignore_layout=True,
    )
    spec = KernelRegistry.get().get_by_name(selected.name)
    assert spec is not None
    if spec.weight_preprocessor is None:
        return None
    names, transform = spec.weight_preprocessor
    return names, partial(
        transform,
        config={
            "num_groups": typed_plan.num_groups,
            "output_dim": typed_plan.output_dim,
            "input_dim": typed_plan.heads_per_group * typed_plan.head_dim,
            "block_size": typed_plan.block_size,
        },
    )


def _select_grouped_output_projection(
    plan: _GroupedOutputProjectionPlan,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    allow_unknown_layout: bool,
    solution: str | None,
) -> SelectedKernel:
    broker = get_weight_broker()
    layouts = frozenset(
        broker.layout(tensor, allow_unknown=allow_unknown_layout)
        for tensor in (weight, weight_scale)
    )
    return select_kernel(
        "gemm",
        "dsv4_grouped_output_projection",
        format_signature(
            layouts=layouts,
            attention=dense_tensor_format(plan.input_dtype),
            weight=dense_tensor_format(weight.dtype),
        ),
        traits={
            "block_size": plan.block_size,
            "scale_format": plan.scale_format,
            "weight_scale_dtype": weight_scale.dtype,
        },
        solution=solution,
    )


def dsv4_grouped_output_projection(
    plan: object,
    attention: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    allow_unknown_layout: bool,
    solution: str | None = None,
) -> torch.Tensor:
    """Run inverse RoPE, FP8 quantization, and grouped ``wo_a`` projection.

    Args:
        plan: Opaque grouped output projection plan.
        attention: Attention output shaped ``[tokens, local_heads, head_dim]``.
        positions: Position index for each token.
        cos_sin_cache: Rotary embedding cache used to undo output RoPE.
        weight: Processed grouped FP8 projection weight.
        weight_scale: Scales prepared by the layer.
        allow_unknown_layout: Whether unenrolled storage may be treated as
            canonical. Layers that preprocess weights must pass False.
        solution: Optional implementation family preference for this call.

    Returns:
        Grouped BF16 output shaped ``[tokens, num_groups, output_dim]``.
    """
    typed_plan = _require_grouped_output_projection_plan(plan)
    expected_attention_shape = (
        typed_plan.num_groups * typed_plan.heads_per_group,
        typed_plan.head_dim,
    )
    if attention.ndim != 3 or tuple(attention.shape[1:]) != expected_attention_shape:
        raise ValueError(
            "grouped output projection attention shape mismatch: expected "
            f"[tokens, {expected_attention_shape[0]}, {expected_attention_shape[1]}], "
            f"got {tuple(attention.shape)}"
        )
    if attention.dtype != typed_plan.input_dtype:
        raise ValueError(
            f"grouped output projection expected {typed_plan.input_dtype}, "
            f"got {attention.dtype}"
        )
    if weight.dtype != typed_plan.weight_dtype:
        raise ValueError(
            f"grouped output projection expected weight dtype {typed_plan.weight_dtype}, "
            f"got {weight.dtype}"
        )

    shape_params = {
        "T": int(attention.shape[0]),
        "G": typed_plan.num_groups,
        "N": typed_plan.output_dim,
        "K": typed_plan.heads_per_group * typed_plan.head_dim,
    }
    kernel = _select_grouped_output_projection(
        typed_plan,
        weight,
        weight_scale,
        allow_unknown_layout=allow_unknown_layout,
        solution=solution,
    )
    ShapeCapture.get().record(
        "gemm",
        "dsv4_grouped_output_projection",
        kernel.name,
        attention.dtype,
        shape_params,
    )
    with kernel_scope(
        "gemm",
        "dsv4_grouped_output_projection",
        attention.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            attention=attention,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            weight=weight,
            weight_scale=weight_scale,
            num_groups=typed_plan.num_groups,
            heads_per_group=typed_plan.heads_per_group,
            output_dim=typed_plan.output_dim,
            nope_dim=typed_plan.nope_dim,
            rope_dim=typed_plan.rope_dim,
            block_size=typed_plan.block_size,
        )


def dsv4_grouped_output_projection_warmup(
    plan: object,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    max_tokens: int,
    *,
    allow_unknown_layout: bool,
    solution: str | None = None,
) -> None:
    """Warm the implementation selected for the current prepared weights.

    Args:
        plan: Opaque grouped output projection plan.
        weight: Processed grouped FP8 projection weight.
        weight_scale: Scales prepared by the layer.
        max_tokens: Largest token count to include in the warmup sweep.
        allow_unknown_layout: Whether unenrolled storage may be treated as
            canonical. Layers that preprocess weights must pass False.
        solution: Optional implementation family preference for this call.

    Returns:
        None.
    """
    typed_plan = _require_grouped_output_projection_plan(plan)
    kernel = _select_grouped_output_projection(
        typed_plan,
        weight,
        weight_scale,
        allow_unknown_layout=allow_unknown_layout,
        solution=solution,
    )
    warmup = getattr(kernel.impl, "_tokenspeed_warmup", None)
    if warmup is None:
        return
    warmup(
        weight=weight,
        weight_scale=weight_scale,
        num_groups=typed_plan.num_groups,
        output_dim=typed_plan.output_dim,
        input_dim=typed_plan.heads_per_group * typed_plan.head_dim,
        block_size=typed_plan.block_size,
        max_tokens=max_tokens,
    )


def dsv4_grouped_output_projection_warmup_model(
    model: torch.nn.Module,
    max_tokens: int,
    *,
    allow_unknown_layout: bool,
    solution: str | None = None,
) -> None:
    """Warm every distinct grouped output projection plan attached to a model.

    Args:
        model: Model containing layers with prepared grouped projection plans.
        max_tokens: Largest token count to include in each backend warmup sweep.
        allow_unknown_layout: Whether unenrolled storage may be treated as
            canonical. Layers that preprocess weights must pass False.
        solution: Optional implementation family preference for this call.

    Returns:
        None.
    """
    seen: set[tuple] = set()
    for module in model.modules():
        plan = getattr(module, "_dsv4_grouped_output_projection_plan", None)
        if plan is None:
            continue
        typed_plan = _require_grouped_output_projection_plan(plan)
        # Validate every layer even when another layer already warmed its shape.
        kernel = _select_grouped_output_projection(
            typed_plan,
            module.weight,
            module.weight_scale_inv,
            allow_unknown_layout=allow_unknown_layout,
            solution=solution,
        )
        key = (
            typed_plan,
            kernel.name,
            module.weight_scale_inv.dtype,
            module.weight.device,
        )
        if key in seen:
            continue
        seen.add(key)
        dsv4_grouped_output_projection_warmup(
            typed_plan,
            module.weight,
            module.weight_scale_inv,
            max_tokens,
            allow_unknown_layout=allow_unknown_layout,
            solution=solution,
        )


def grouped_bf16_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None,
    solution: str | None,
) -> torch.Tensor:
    """Compute ``einsum('tgd,grd->tgr', x, weight)`` from canonical BF16 weights.

    Args:
        x: BF16 ``[tokens, groups, input_dim]`` activations. Strided group and
            token views, including a slice of padded attention heads, are valid.
        weight: BF16 ``[groups, output_dim, input_dim]`` live weights. No packing,
            copying or persistent weight cache is created. Group/row strides are
            explicit; nonunit inner strides use the original torch fallback.
        out: Optional contiguous BF16 ``[tokens, groups, output_dim]`` result.
            Must not overlap either input. None allocates a fresh result.
        solution: Optional registered solution restriction ("triton" or "torch").
            None selects an implementation from the input geometry.

    Returns:
        BF16 ``[tokens, groups, output_dim]``; out when provided. Registered
        implementations declare their supported geometry; unmatched inputs use
        torch.einsum. Eager and graph execution use the same selection path.
    """
    if x.ndim != 3 or weight.ndim != 3:
        raise ValueError("Grouped projection requires two rank-3 tensors")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("Grouped projection requires BF16 operands")
    if x.device != weight.device or x.shape[1:] != (weight.shape[0], weight.shape[2]):
        raise ValueError(
            "Grouped projection operands must share device/groups/input_dim"
        )
    expected = (x.shape[0], weight.shape[0], weight.shape[1])
    if out is not None and (
        tuple(out.shape) != expected
        or out.dtype != x.dtype
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "Grouped projection out must be contiguous BF16 with matching shape/device"
        )
    kernel = select_kernel(
        "gemm",
        "grouped_bf16_projection",
        format_signature(
            x=dense_tensor_format(torch.bfloat16),
            weight=dense_tensor_format(torch.bfloat16),
        ),
        traits={
            "batch": weight.shape[0],
            "m": x.shape[0],
            "n": weight.shape[1],
            "k": weight.shape[2],
            "a_inner_stride_one": x.stride(-1) == 1,
            "b_contiguous": weight.is_contiguous(),
            "b_inner_stride_one": weight.stride(-1) == 1,
            "is_cuda": x.is_cuda,
        },
        solution=solution,
    )
    return kernel(x, weight, out)


def grouped_bf16_projection_quantize_fp8(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``grouped_bf16_projection`` flattened and quantized for an MXFP8 GEMM.

    Args:
        x: BF16 ``[tokens, groups, input_dim]`` activations (see
            ``grouped_bf16_projection``).
        weight: BF16 ``[groups, output_dim, input_dim]`` weights.

    Returns:
        E4M3 codes ``[tokens, groups * output_dim]`` and uint8 UE8M0 scales
        ``[tokens, groups * output_dim / 32]`` of the BF16 projection, as
        ``quantize_fp8(granularity="token_group", group_size=32,
        scale_encoding="ue8m0")`` produces them. A kernel that quantizes in
        its epilogue is used when one is registered for the geometry.
    """
    try:
        kernel = select_kernel(
            "gemm",
            "grouped_bf16_projection_fp8",
            format_signature(
                x=dense_tensor_format(torch.bfloat16),
                weight=dense_tensor_format(torch.bfloat16),
            ),
            traits={
                "batch": weight.shape[0],
                "n": weight.shape[1],
                "k": weight.shape[2],
                "a_inner_stride_one": x.stride(-1) == 1,
                "b_contiguous": weight.is_contiguous(),
                "is_cuda": x.is_cuda,
            },
        )
    except NoKernelFoundError:
        projected = grouped_bf16_projection(x, weight, None, None).flatten(1)
        return quantize_fp8(
            projected,
            granularity="token_group",
            group_size=32,
            scale_encoding="ue8m0",
            enable_pdl=False,
            override=None,
            solution=None,
        )
    return kernel(x, weight)


def dsv4_linear_fp32(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Project DeepSeek V4 hidden states and return FP32 output.

    Args:
        hidden_states: Floating-point activations with trailing dimension K.
        weight: Floating-point row-major weight shaped [N, K].
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        FP32 projected activations with trailing dimension N.
    """
    if hidden_states.ndim == 0:
        raise ValueError("hidden_states must have at least one dimension")
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N, K], got {tuple(weight.shape)}")
    if hidden_states.shape[-1] != weight.shape[1]:
        raise ValueError(
            "DeepSeek V4 linear K mismatch: "
            f"hidden_states K={hidden_states.shape[-1]}, weight K={weight.shape[1]}"
        )
    if not hidden_states.is_floating_point() or not weight.is_floating_point():
        raise ValueError("hidden_states and weight must be floating-point tensors")

    enable_pdl = pdl_enabled()
    k = int(weight.shape[1])
    traits = {
        "n": int(weight.shape[0]),
        "k": k,
        "has_tokens": hidden_states.numel() > 0,
        "hidden_inner_stride_one": hidden_states.stride(-1) == 1,
        "hidden_rank": hidden_states.ndim,
        "weight_contiguous": weight.is_contiguous(),
        "ll_bf16_supported": (
            hidden_states.ndim == 2
            and ll_bf16_router_supported(hidden_states, weight, hidden_states.shape[0])
        ),
    }
    signature = format_signature(
        hidden_states=dense_tensor_format(hidden_states.dtype),
        weight=dense_tensor_format(weight.dtype),
    )
    try:
        kernel = select_kernel(
            "gemm",
            "dsv4_linear_fp32",
            signature,
            traits=traits,
            override=override,
            solution=solution,
        )
    except NoKernelFoundError:
        if override is not None or solution is not None:
            raise
        flat = hidden_states.reshape(-1, k)
        if flat.is_cuda and flat.dtype == weight.dtype:
            output = torch.mm(flat, weight.t(), out_dtype=torch.float32)
        else:
            output = torch.mm(flat.float(), weight.float().t())
        return output.reshape(*hidden_states.shape[:-1], weight.shape[0])
    shape_params = {
        "M": int(prod(hidden_states.shape[:-1])),
        "N": int(weight.shape[0]),
        "K": k,
        "enable_pdl": bool(enable_pdl),
    }
    ShapeCapture.get().record(
        "gemm",
        "dsv4_linear_fp32",
        kernel.name,
        hidden_states.dtype,
        shape_params,
    )
    with kernel_scope(
        "gemm",
        "dsv4_linear_fp32",
        hidden_states.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(hidden_states, weight, enable_pdl=enable_pdl)


# ===-----------------------------------------------------------------------===#
# MV: small-M dense projection
# ===-----------------------------------------------------------------------===#


# BF16 and FP32 small-M dense projection dispatch.
#
# Eligible small-M BF16 projections use FlashInfer joint runner/tactic selection.
# The registry keeps the architecture-specific and portable fallbacks, while the
# row-CTA implementation also provides the independent fused add3 epilogue.
#
# FP32 projections have no FlashInfer route and go straight to the registry.
# Narrow ones with up to 16 rows take a row-CTA kernel that streams each weight
# row once against every activation row, with FP32 products and accumulation and
# no fused multiply-add; a single row is reduced in one block over the whole row.
# Torch serves the other FP32 shapes. The kernel also takes BF16 activations
# against an FP32 weight and widens each element as it loads it, which is exact,
# so the result equals the same call on ``x.float()`` without materializing that
# copy.
#
# BF16 activations against an FP32 weight give an FP32 result. Torch widens the
# activations first. A layer that keeps the weight split from
# :func:`mv_weight_split` passes it to :func:`mv`, and the
# registry can then run the product on BF16 tensor cores
# (``triton_bf16x3_gemm_fp32``). From 17 to 96 rows, for weights of up to 256
# rows (a multiple of 4) with K a multiple of 512 up to 8192, the registry runs
# it on CUDA cores from the FP32 weight instead, with or without the split
# (``gluon_simt_gemm_fp32``).

# Registry signature of each served (x dtype, weight dtype) pair.
_MV_SIGNATURES = {
    (x_dtype, weight_dtype): format_signature(
        x=dense_tensor_format(x_dtype),
        weight=dense_tensor_format(weight_dtype),
    )
    for x_dtype, weight_dtype in (
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.float32),
    )
}


@lru_cache(maxsize=64)
def _select_mv(
    m: int,
    n: int,
    k: int,
    on_cuda: bool,
    x_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    weight_split: bool,
):
    signature = _MV_SIGNATURES.get((x_dtype, weight_dtype))
    if not on_cuda or signature is None:
        return torch_decode_gemv

    reg = KernelRegistry.get()
    # Honor each registered implementation's architecture gate and dtype
    # signature before its shape traits, including the specialized CDNA5
    # kernels. Kernels that take a split weight declare the weight_split trait.
    traits = {"m": m, "n": n, "k": k, "weight_split": weight_split}
    for spec in reg.get_for_operator(
        "gemm",
        "mv",
        platform=current_platform(),
        format_signature=signature,
    ):
        if spec_matches_traits(spec, traits) and spec_matches_shape_traits(
            spec, traits
        ):
            return reg.get_impl(spec.name)
    return torch_decode_gemv


def _use_mv(x: torch.Tensor, weight: torch.Tensor) -> bool:
    if (
        flashinfer_joint_bf16_supported(x, weight, None)
        and x.shape[0] <= BF16_GEMM_MAX_M
    ):
        # FI compiles per row count; serving's eager rows take the caller's GEMM.
        return not is_serving()
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
    platform = current_platform()
    n = weight.shape[0]
    if platform.is_cdna4:
        impl = _select_mv(m, n, k, True, x.dtype, weight.dtype, False)
        if impl is torch_decode_gemv:
            return False
        # Single rows keep mm's own path (Torch's skinny GEMM or hipBLASLt)
        # except where a gfx950 GEMV measured faster: the row-CTA GEMV and the
        # decode GEMV each on their measured shapes.
        return (
            m >= 2
            or prefers_gluon_rowcta_bf16_gemv_gfx950(n, k)
            or impl is KernelRegistry.get().get_impl("gluon_gemv_a16w16_gfx950")
        )
    if not platform.is_cdna5 or k < 256:
        return False
    return (
        _select_mv(m, n, k, True, x.dtype, weight.dtype, False) is not torch_decode_gemv
    )


def mv(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    weight_split: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ weight.T`` through joint FI tuning or the ordinary registry fallback.

    FlashInfer owns runner/tactic selection on the supported BF16 range.
    The registry retains other architectures, unsupported input layouts and
    FP32 weights, which also take BF16 activations and return FP32.
    Noncontiguous inputs and other dtypes take Torch.

    Args:
        x: ``[M, K]`` activations.
        weight: ``[N, K]`` weight.
        out: optional ``[M, N]`` destination in the promoted dtype.
        weight_split: the weight's :func:`mv_weight_split`, for an
            FP32 weight; ``None`` keeps the kernels that read ``weight``.

    Returns:
        ``[M, N]`` output in the promoted dtype of ``x`` and ``weight``.
    """

    expected = (x.shape[0], weight.shape[0])
    if out is not None:
        if (
            tuple(out.shape) != expected
            or out.dtype != torch.promote_types(x.dtype, weight.dtype)
            or out.device != x.device
            or out.stride(-1) != 1
        ):
            raise ValueError(f"out must match x and have shape {expected}")
        if not out.is_contiguous():
            return torch_decode_gemv(x, weight, out)

    autotune_bf16_gemm(x, weight)
    if (
        flashinfer_joint_bf16_supported(x, weight, out)
        and x.shape[0] <= BF16_GEMM_MAX_M
    ):
        # Serving never compiles: FI keys runners on the row count, rowcta may be cold.
        if is_serving():
            return torch_decode_gemv(x, weight, out)
        return flashinfer_bf16_gemm(x, weight, out)
    if not x.is_contiguous() or not weight.is_contiguous():
        return torch_decode_gemv(x, weight, out)
    impl = _select_mv(
        x.shape[0],
        weight.shape[0],
        weight.shape[1],
        x.is_cuda,
        x.dtype,
        weight.dtype,
        weight_split is not None,
    )
    if impl is triton_bf16x3_gemm_fp32:
        return impl(x, weight_split, out)
    return impl(x, weight, out)


def mv_weight_split(weight: torch.Tensor) -> torch.Tensor | None:
    """Split an FP32 weight for the tensor-core path of :func:`mv`.

    The layer that owns the weight calls this once after loading it and
    passes the result as ``mv(..., weight_split=...)``. The split
    takes 1.5x the FP32 weight's memory, so it is only made where the
    registration of ``triton_bf16x3_gemm_fp32`` takes this weight on this
    device. The kernel the registry prefers at one row count does not decide
    it: a kernel registered ahead of the split kernel for some row counts
    serves those rows from ``weight``, and the split serves the others.

    Args:
        weight: ``[N, K]`` weight.

    Returns:
        ``[3, N, K]`` BF16 pieces, or ``None`` when the split kernel does not
        take ``weight`` on this device.
    """
    if not weight.is_cuda or weight.ndim != 2 or not weight.is_contiguous():
        return None
    signature = _MV_SIGNATURES.get((torch.bfloat16, weight.dtype))
    if signature is None:
        return None
    n, k = weight.shape
    # _select_mv()'s filters on the split kernel's own registration, at its
    # first row count.
    traits = {"m": BF16X3_MIN_M, "n": n, "k": k, "weight_split": True}
    reg = KernelRegistry.get()
    if not any(
        reg.get_impl(spec.name) is triton_bf16x3_gemm_fp32
        and spec_matches_traits(spec, traits)
        and spec_matches_shape_traits(spec, traits)
        for spec in reg.get_for_operator(
            "gemm",
            "mv",
            platform=current_platform(),
            format_signature=signature,
        )
    ):
        return None
    return split_fp32_weight_bf16x3(weight)


def _select_mv_weightless_rmsnorm_gated_sigmoid(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> SelectedKernel | None:
    """Select the one-launch gated GEMV for this row, or None."""
    if (
        not x.is_cuda
        or x.ndim != 2
        or x.shape[1] != num_heads * head_dim
        or not x.is_contiguous()
        or gate.shape != x.shape
        or gate.stride(1) != 1
        or weight.ndim != 2
        or weight.shape[1] != x.shape[1]
        or not weight.is_contiguous()
        or gate.device != x.device
        or weight.device != x.device
        or resolve_kernel_override("gemm", "mm", None) is not None
    ):
        return None
    m, (n, k) = x.shape[0], weight.shape
    try:
        kernel = select_kernel(
            "gemm",
            "mv_weightless_rmsnorm_gated_sigmoid",
            dense_format_signature(
                ("x", "gate", "weight"), (x.dtype, gate.dtype, weight.dtype)
            ),
            traits={"m": m, "n": n, "k": k, "head_dim": head_dim},
        )
    except NoKernelFoundError:
        return None
    # The launch reproduces the gated norm and GEMV it replaces bit for bit, so
    # it only stands in where those two calls select the kernels it copies.
    norm = select_kernel(
        "layernorm",
        "weightless_rmsnorm_gated_sigmoid",
        dense_format_signature(("x", "gate"), (x.dtype, gate.dtype)),
    )
    gemv = _select_mv(m, n, k, True, x.dtype, weight.dtype, False)
    if (
        norm.name != "triton_weightless_rmsnorm_gated_sigmoid"
        or gemv is not KernelRegistry.get().get_impl("gluon_rowcta_bf16_gemv_gfx950")
    ):
        return None
    return kernel


def mv_weightless_rmsnorm_gated_sigmoid_supported(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> bool:
    """Return whether :func:`mv_weightless_rmsnorm_gated_sigmoid` serves this row.

    A registered kernel must take the shape, and the separate gated norm and
    ``mv`` must select the kernels it reproduces.

    Args:
        x: Attention row ``[1, num_heads * head_dim]``.
        gate: Gate logits with ``x``'s shape.
        weight: Output projection weight ``[N, num_heads * head_dim]``.
        num_heads: Heads in ``x``.
        head_dim: Width of each head.

    Returns:
        ``True`` when one launch can replace the gated norm followed by ``mv``.
    """
    return (
        _select_mv_weightless_rmsnorm_gated_sigmoid(
            x, gate, weight, num_heads, head_dim
        )
        is not None
    )


def mv_weightless_rmsnorm_gated_sigmoid(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    num_heads: int,
    eps: float,
    out: torch.Tensor,
) -> torch.Tensor:
    """Normalize, gate and project one row in one launch.

    Computes ``mv(weightless_rmsnorm_gated_sigmoid(x, gate, ...), weight,
    out)`` where :func:`mv_weightless_rmsnorm_gated_sigmoid_supported` holds.

    Args:
        x: Contiguous BF16 row ``[1, num_heads * head_dim]``.
        gate: Gate logits with ``x``'s shape and a unit inner stride.
        weight: ``[N, K]`` weight.
        num_heads: Heads in ``x``.
        eps: Epsilon added to each per-head variance.
        out: ``[1, N]`` destination.

    Returns:
        ``out``.
    """
    head_dim = x.shape[-1] // max(num_heads, 1)
    kernel = _select_mv_weightless_rmsnorm_gated_sigmoid(
        x, gate, weight, num_heads, head_dim
    )
    if kernel is None:
        raise ValueError(
            "mv_weightless_rmsnorm_gated_sigmoid needs a supported gated row "
            f"(see mv_weightless_rmsnorm_gated_sigmoid_supported), got x "
            f"{tuple(x.shape)}, weight {tuple(weight.shape)}, {num_heads} heads"
        )
    return kernel(x, gate, weight, out, num_heads, eps)


# ===-----------------------------------------------------------------------===#
# Linear AttnRes partials
# ===-----------------------------------------------------------------------===#


# Linear projection with independent AttnRes block partials.


def _validate_linear_attnres_inputs(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    blocks: torch.Tensor,
    score_weight_a: torch.Tensor,
    score_weight_b: torch.Tensor,
    scratch_a: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scratch_b: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    eps: float,
    out: torch.Tensor | None,
) -> tuple[int, int, tuple[int, int]]:
    if hidden_states.ndim != 2 or hidden_states.shape[0] < 1:
        raise ValueError("hidden_states must have shape [tokens, input_size]")
    tokens, input_size = hidden_states.shape
    if weight.ndim != 2 or weight.shape[1] != input_size:
        raise ValueError("weight must have shape [output_size, input_size]")
    if (
        blocks.ndim != 3
        or blocks.shape[0] < 1
        or tuple(blocks.shape[1:]) != (tokens, input_size)
    ):
        raise ValueError("blocks must have shape [blocks, tokens, input_size]")
    tensors = (weight, blocks, score_weight_a, score_weight_b)
    if any(
        tensor.dtype != hidden_states.dtype or tensor.device != hidden_states.device
        for tensor in tensors
    ):
        raise ValueError("projection and AttnRes inputs must match dtype and device")
    if score_weight_a.shape != (input_size,) or score_weight_b.shape != (input_size,):
        raise ValueError("score weights must match input_size")

    expected_scratch_shapes = ((tokens,), (tokens,), (tokens, input_size))
    for scratch in (scratch_a, scratch_b):
        if len(scratch) != 3:
            raise ValueError("scratch must be an (max, sum, accumulator) tuple")
        for tensor, shape in zip(scratch, expected_scratch_shapes, strict=True):
            if (
                tuple(tensor.shape) != shape
                or tensor.dtype != torch.float32
                or tensor.device != hidden_states.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"scratch tensors must be contiguous FP32 {expected_scratch_shapes}"
                )
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    output_shape = (tokens, weight.shape[0])
    if out is not None and (
        tuple(out.shape) != output_shape
        or out.dtype != hidden_states.dtype
        or out.device != hidden_states.device
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous with shape {output_shape}")
    return tokens, input_size, output_shape


def _select_linear_attnres_kernel(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    blocks: torch.Tensor,
    score_weight_a: torch.Tensor,
    score_weight_b: torch.Tensor,
    scratch_a: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scratch_b: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    out: torch.Tensor | None,
    *,
    override: str | None,
    solution: str | None,
):
    tokens, input_size = hidden_states.shape
    signature = format_signature(
        hidden_states=dense_tensor_format(hidden_states.dtype),
        weight=dense_tensor_format(weight.dtype),
        blocks=dense_tensor_format(blocks.dtype),
        score_weight_a=dense_tensor_format(score_weight_a.dtype),
        score_weight_b=dense_tensor_format(score_weight_b.dtype),
        out=dense_tensor_format(hidden_states.dtype if out is None else out.dtype),
    )
    traits = {
        "tokens": tokens,
        "input_size": input_size,
        "output_size": weight.shape[0],
        "num_blocks": blocks.shape[0],
        "inputs_contiguous": all(
            tensor.is_contiguous()
            for tensor in (
                hidden_states,
                weight,
                blocks,
                score_weight_a,
                score_weight_b,
                *scratch_a,
                *scratch_b,
            )
        )
        and (out is None or out.is_contiguous()),
    }
    try:
        platform = current_platform()
    except RuntimeError:
        platform = None
    if platform is not None and platform.is_cdna5:
        traits["gfx1250_linear_attnres_enabled"] = True
    if not hidden_states.is_cuda:
        return None, traits
    try:
        kernel = select_kernel(
            "gemm",
            "linear_attnres_partials",
            signature,
            traits=traits,
            override=override,
            solution=solution,
        )
    except NoKernelFoundError:
        if override is not None or solution is not None:
            raise
        kernel = None
    return kernel, traits


def linear_attnres_partials_available(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    blocks: torch.Tensor,
    score_weight_a: torch.Tensor,
    score_weight_b: torch.Tensor,
    scratch_a: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scratch_b: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    eps: float,
) -> bool:
    """Return whether a registered fused implementation accepts these inputs.

    Args:
        hidden_states: Input shaped ``[tokens, input_size]``.
        weight: Projection weight shaped ``[output_size, input_size]``.
        blocks: Residual candidates shaped ``[blocks, tokens, input_size]``.
        score_weight_a: First AttnRes score weight shaped ``[input_size]``.
        score_weight_b: Second AttnRes score weight shaped ``[input_size]``.
        scratch_a: First ``(max, sum, accumulator)`` FP32 output tuple.
        scratch_b: Second ``(max, sum, accumulator)`` FP32 output tuple.
        eps: Positive RMSNorm epsilon used by both reductions.

    Returns:
        ``True`` when registry dispatch can run the fused operation.
    """
    try:
        _validate_linear_attnres_inputs(
            hidden_states,
            weight,
            blocks,
            score_weight_a,
            score_weight_b,
            scratch_a,
            scratch_b,
            eps,
            None,
        )
    except ValueError:
        return False
    kernel, _ = _select_linear_attnres_kernel(
        hidden_states,
        weight,
        blocks,
        score_weight_a,
        score_weight_b,
        scratch_a,
        scratch_b,
        None,
        override=None,
        solution=None,
    )
    return kernel is not None


def linear_attnres_partials(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    blocks: torch.Tensor,
    score_weight_a: torch.Tensor,
    score_weight_b: torch.Tensor,
    scratch_a: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    scratch_b: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    eps: float,
    out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Run a linear projection and two independent AttnRes reductions.

    Args:
        hidden_states: Input shaped ``[tokens, input_size]``.
        weight: Projection weight shaped ``[output_size, input_size]``.
        blocks: Residual candidates shaped ``[blocks, tokens, input_size]``.
        score_weight_a: First AttnRes score weight shaped ``[input_size]``.
        score_weight_b: Second AttnRes score weight shaped ``[input_size]``.
        scratch_a: First ``(max, sum, accumulator)`` FP32 output tuple.
        scratch_b: Second ``(max, sum, accumulator)`` FP32 output tuple.
        eps: Positive RMSNorm epsilon used by both reductions.
        out: Optional contiguous projection output.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        Projection output shaped ``[tokens, output_size]``. Both scratch tuples
        are overwritten with the corresponding AttnRes partials.
    """
    out_was_provided = out is not None
    tokens, _, output_shape = _validate_linear_attnres_inputs(
        hidden_states,
        weight,
        blocks,
        score_weight_a,
        score_weight_b,
        scratch_a,
        scratch_b,
        eps,
        out,
    )
    if not out_was_provided:
        out = hidden_states.new_empty(output_shape)
    kernel, traits = _select_linear_attnres_kernel(
        hidden_states,
        weight,
        blocks,
        score_weight_a,
        score_weight_b,
        scratch_a,
        scratch_b,
        out,
        override=override,
        solution=solution,
    )

    if kernel is not None:
        if override is None and solution is None:
            autotune_bf16_gemm(hidden_states, weight)
        ShapeCapture.get().record(
            "gemm",
            "linear_attnres_partials",
            kernel.name,
            hidden_states.dtype,
            traits,
        )
        with kernel_scope(
            "gemm",
            "linear_attnres_partials",
            hidden_states.dtype,
            kernel_name=kernel.name,
            **traits,
        ):
            return kernel(
                hidden_states=hidden_states,
                weight=weight,
                blocks=blocks,
                score_weight_a=score_weight_a,
                score_weight_b=score_weight_b,
                scratch_a=scratch_a,
                scratch_b=scratch_b,
                eps=eps,
                out=out,
            )

    # mv falls back to torch.mm for unlisted shapes, so widening this
    # gate past M == 1 only adds the measured kernels.
    if tokens <= 8 and hidden_states.is_contiguous() and weight.is_contiguous():
        projected = mv(hidden_states, weight)
        if out_was_provided:
            out.copy_(projected)
        else:
            out = projected
    else:
        if override is None and solution is None:
            autotune_bf16_gemm(hidden_states, weight)
        torch.mm(hidden_states, weight.T, out=out)
    triton_partial_eligible = (
        blocks.is_cuda
        and blocks.dtype == torch.bfloat16
        and blocks.shape[2] == KIMI3_HIDDEN_SIZE
        and blocks.stride(2) == 1
        and score_weight_a.is_contiguous()
        and score_weight_b.is_contiguous()
    )
    if triton_partial_eligible:
        from tokenspeed_kernel.ops.activation.triton import attnres_partial_dual

        attnres_partial_dual(
            blocks,
            score_weight_a,
            score_weight_b,
            eps,
            scratch_a,
            scratch_b,
        )
    else:
        values = blocks.float()
        inverse_rms = torch.rsqrt(values.square().mean(dim=-1) + eps)
        for score_weight, scratch in (
            (score_weight_a, scratch_a),
            (score_weight_b, scratch_b),
        ):
            logits = torch.einsum("bth,h->bt", values, score_weight.float())
            logits *= inverse_rms
            maxima = logits.max(dim=0).values
            unnormalized = torch.exp(logits - maxima)
            scratch[0].copy_(maxima)
            scratch[1].copy_(unnormalized.sum(dim=0))
            scratch[2].copy_(torch.einsum("bt,bth->th", unnormalized, values))
    return out


# ===-----------------------------------------------------------------------===#
# Kimi K3 projections
# ===-----------------------------------------------------------------------===#


# Dense BF16 projection kernels for Kimi K3.
#
# K3 uses two replicated dense projections around its routed expert block:
# 7168 -> 3584 and 3584 -> 7168. This code keeps their shape contract explicit
# and provides gfx950-tuned Triton decode and Gluon middle/large-M implementations
# while retaining the vendor GEMM as a selectable fallback. KDA decode also uses
# a bandwidth-oriented fused Q/K/V/output-gate projection. On gfx950 the other
# K3 projections take the Gluon large-M kernel for the prefill token counts where
# it beats hipBLASLt.

if _platform.is_amd:
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.decode_gemv import (
        prefers_gluon_rowcta_bf16_gemv_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
        supports_gluon_mm_a16w16_prefill_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
        supports_gluon_mm_a16w16_decode_add3_gfx950,
        supports_gluon_mm_a16w16_decode_gfx950,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
        use_gluon_largem_gfx1250,
        use_gluon_wmma_dense_gfx1250,
    )

else:

    def use_gluon_largem_gfx1250(m: int, k: int, n: int) -> bool:
        return False

    def use_gluon_wmma_dense_gfx1250(m: int, k: int, n: int) -> bool:
        return False

    def supports_gluon_mm_a16w16_decode_gfx950(m: int, n: int, k: int) -> bool:
        return False

    def supports_gluon_mm_a16w16_decode_add3_gfx950(m: int, n: int, k: int) -> bool:
        return False

    def supports_gluon_mm_a16w16_prefill_gfx950(m: int, n: int, k: int) -> bool:
        return False

    def prefers_gluon_rowcta_bf16_gemv_gfx950(n: int, k: int) -> bool:
        return False


# FP8 storage dtypes served by the w8a8 projection branch (matches the
# runtime quantization layers' width: e4m3fn on NVIDIA, e4m3fnuz on ROCm).
_FP8_WEIGHT_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)

KIMI3_HIDDEN_SIZE = 7168
KIMI3_LATENT_SIZE = 3584
KIMI3_QKVFAB_SIZE = 6288
KIMI3_ROUTER_SIZE = 896

KIMI3_SHARED_LOCAL_SIZE = 768


KIMI3_SHARED_GATE_UP_LOCAL_SIZE = 2 * KIMI3_SHARED_LOCAL_SIZE
_KIMI3_SHAPES = {
    (KIMI3_HIDDEN_SIZE, KIMI3_LATENT_SIZE),
    (KIMI3_LATENT_SIZE, KIMI3_HIDDEN_SIZE),
}


@dataclass(frozen=True)
class Kimi3MLAQKVGateProjection:
    """QKV-a/gate projection result plus its communication layout."""

    qkv: torch.Tensor
    gate: torch.Tensor
    packed: torch.Tensor | None


def _use_gluon_mediumm(m: int, k: int, n: int) -> bool:
    if (k, n) == (KIMI3_HIDDEN_SIZE, KIMI3_LATENT_SIZE):
        return 768 <= m <= 1024 and m % 128 == 0
    if (k, n) == (KIMI3_LATENT_SIZE, KIMI3_HIDDEN_SIZE):
        return 384 <= m <= 512 and m % 64 == 0
    return False


def _use_gluon_largem(m: int, k: int, n: int) -> bool:
    return (k, n) in _KIMI3_SHAPES and supports_gluon_mm_a16w16_prefill_gfx950(m, n, k)


def _try_gluon_largem_gfx950(
    activation: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Run the CDNA4 large-M kernel, or return None when the call is ineligible.

    Named K3 APIs use this on their automatic path for the prefill shapes the
    kernel was measured to win; BF16 ``UnquantizedLinearMethod`` layers reach
    the same kernel through the ``mm`` registry.
    """

    if (
        not Platform.get().is_cdna4
        or activation.ndim != 2
        or weight.ndim != 2
        or activation.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not activation.is_cuda
        or weight.device != activation.device
        or not activation.is_contiguous()
        or not weight.is_contiguous()
        or (
            out is not None and (out.device != activation.device or out.stride(-1) != 1)
        )
        or not supports_gluon_mm_a16w16_prefill_gfx950(
            int(activation.shape[0]),
            int(weight.shape[0]),
            int(activation.shape[1]),
        )
    ):
        return None
    from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
        launch_gluon_mm_a16w16_prefill_gfx950,
    )

    return launch_gluon_mm_a16w16_prefill_gfx950(
        activation, weight, activation.dtype, out=out
    )


def _try_gluon_largem_gfx1250(
    activation: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Run the CDNA5 large-M kernel, or return None when the call is ineligible.

    Named K3 APIs and BF16 ``UnquantizedLinearMethod`` layers use this instead
    of registering the kernel on generic ``mm()``.
    """

    if (
        not Platform.get().is_cdna5
        or activation.ndim != 2
        or weight.ndim != 2
        or activation.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not activation.is_cuda
        or not weight.is_cuda
        or weight.device != activation.device
        or not activation.is_contiguous()
        or not weight.is_contiguous()
        or (
            out is not None
            and (
                not out.is_cuda
                or out.device != activation.device
                or not out.is_contiguous()
            )
        )
        or not use_gluon_largem_gfx1250(
            int(activation.shape[0]),
            int(activation.shape[1]),
            int(weight.shape[0]),
        )
    ):
        return None
    from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
        gluon_mm_a16w16_largem_gfx1250,
    )

    return gluon_mm_a16w16_largem_gfx1250(activation, weight, out=out)


def _validate_fallback_projection(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None,
    *,
    name: str,
    out_dtype: torch.dtype | None = None,
) -> tuple[int, int, int]:
    """Validate a dense projection while leaving kernel eligibility to dispatch."""

    if hidden_states.ndim != 2 or weight.ndim != 2:
        raise ValueError(f"{name} expects [M, K] input and [N, K] weight")
    m, k = hidden_states.shape
    n, weight_k = weight.shape
    if weight_k != k:
        raise ValueError(f"{name} K mismatch: {k} != {weight_k}")
    if hidden_states.device != weight.device:
        raise ValueError(f"{name} input and weight must be colocated")
    expected_dtype = hidden_states.dtype if out_dtype is None else out_dtype
    if out is not None and (
        tuple(out.shape) != (m, n)
        or out.dtype != expected_dtype
        or out.device != hidden_states.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"{name} out must be contiguous {expected_dtype} with shape {(m, n)}"
        )
    return m, n, k


def kimi3_latent_projection(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    solution: str = "auto",
) -> torch.Tensor:
    """Apply a replicated latent projection with K3-specialized dispatch.

    ``solution='torch'`` uses the vendor BLAS selected by PyTorch;
    ``solution='triton_gemv'`` forces the decode kernel,
    ``solution='gluon_smallm'`` forces the split-K small-M gfx950 kernel,
    ``solution='gluon_mediumm'`` forces the middle-M gfx950 kernel, and
    ``solution='gluon_largem'`` forces the large-M gfx950 kernel. ``auto``
    uses their measured gfx950 crossovers for canonical K3 shapes, routes
    decode batches through the ``mv`` registry, and retains the
    vendor GEMM for other shapes and architectures.
    """

    m, n, k = _validate_fallback_projection(
        hidden_states,
        weight,
        out,
        name="Kimi K3 latent projection",
    )
    if solution not in {
        "auto",
        "torch",
        "triton_gemv",
        "gluon_smallm",
        "gluon_mediumm",
        "gluon_largem",
        "gluon_largem_gfx1250",
    }:
        raise ValueError(f"unknown Kimi K3 projection solution {solution!r}")
    specialized = (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and weight.is_contiguous()
        and (k, n) in _KIMI3_SHAPES
    )
    routed = solution == "auto"
    if solution == "auto":
        autotune_bf16_gemm(hidden_states, weight)
        if Platform.get().is_cdna4 and specialized and m == 1:
            solution = "triton_gemv"
        elif Platform.get().is_cdna4 and specialized and _use_gluon_mediumm(m, k, n):
            solution = "gluon_mediumm"
        elif Platform.get().is_cdna4 and specialized and _use_gluon_largem(m, k, n):
            solution = "gluon_largem"
        elif (
            Platform.get().is_cdna5
            and specialized
            and use_gluon_largem_gfx1250(m, k, n)
            and (out is None or out.is_contiguous())
        ):
            solution = "gluon_largem_gfx1250"
        else:
            solution = "torch"
    elif solution != "torch" and not specialized:
        raise ValueError(
            f"Kimi K3 {solution} latent projection requires a contiguous gfx950 "
            "BF16 7168<->3584 shape"
        )
    if solution == "triton_gemv":
        return kimi3_projection_gemv(
            hidden_states,
            weight,
            out=out,
            config=(
                (2, 1024, 4, 0)
                if (k, n) == (KIMI3_HIDDEN_SIZE, KIMI3_LATENT_SIZE)
                else (16, 512, 16, 1)
            ),
        )
    if solution == "gluon_smallm":
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
            launch_gluon_mm_a16w16_splitk_gfx950,
        )

        return launch_gluon_mm_a16w16_splitk_gfx950(
            hidden_states,
            weight,
            hidden_states.dtype,
            out=out,
        )
    if solution == "gluon_mediumm":
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
            launch_gluon_mm_a16w16_medium_gfx950,
        )

        return launch_gluon_mm_a16w16_medium_gfx950(
            hidden_states,
            weight,
            hidden_states.dtype,
            out=out,
        )
    if solution == "gluon_largem":
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
            launch_gluon_mm_a16w16_prefill_gfx950,
        )

        output = launch_gluon_mm_a16w16_prefill_gfx950(
            hidden_states,
            weight,
            hidden_states.dtype,
            out=out,
        )
        if output is None:
            raise ValueError("Kimi K3 Gluon latent projection rejected this shape")
        return output
    if solution == "gluon_largem_gfx1250":
        if not (
            Platform.get().is_cdna5
            and specialized
            and use_gluon_largem_gfx1250(m, k, n)
        ):
            raise ValueError(
                "Kimi K3 gfx1250 Gluon projection requires a contiguous BF16 "
                "K3 shape with M >= 512"
            )
        from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
            gluon_mm_a16w16_largem_gfx1250,
        )

        return gluon_mm_a16w16_largem_gfx1250(
            hidden_states,
            weight,
            out=out,
        )
    if routed and _use_mv(hidden_states, weight):
        return mv(hidden_states, weight, out)
    if out is None:
        return torch.nn.functional.linear(hidden_states, weight)
    return torch.mm(hidden_states, weight.T, out=out)


def kimi3_mla_qkv_gate_projection(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    qkv_width: int,
    *,
    solution: str = "auto",
) -> Kimi3MLAQKVGateProjection:
    """Project K3 MLA QKV-a and gate rows with an architecture-selected schedule.

    ``packed`` is populated when communication should retain the fused row
    layout. The split CDNA4 prefill schedule returns independent QKV and gate
    tensors so callers can communicate only the QKV rows.
    """

    m, output_width, input_width = _validate_fallback_projection(
        hidden_states,
        weight,
        None,
        name="Kimi K3 MLA QKV/gate projection",
    )
    routed = solution == "auto"
    if not 0 < qkv_width < output_width:
        raise ValueError(
            f"Kimi K3 MLA qkv_width must be within (0, {output_width}), got {qkv_width}"
        )
    if solution not in {
        "auto",
        "fused",
        "split",
        "gluon_wmma_gfx1250",
        "gluon_largem_gfx1250",
    }:
        raise ValueError(f"unknown Kimi K3 MLA projection solution {solution!r}")
    if solution == "auto":
        autotune_bf16_gemm(hidden_states, weight)
        gfx1250_tdm = (
            Platform.get().is_cdna5
            and hidden_states.is_cuda
            and weight.is_cuda
            and 2 <= m <= 64
            and input_width == KIMI3_HIDDEN_SIZE
            and qkv_width == 2112
            and output_width - qkv_width == 1536
            and hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
        )
        gfx1250_largem = (
            Platform.get().is_cdna5
            and hidden_states.is_cuda
            and weight.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
            and use_gluon_largem_gfx1250(m, hidden_states.shape[1], qkv_width)
            and use_gluon_largem_gfx1250(
                m,
                hidden_states.shape[1],
                output_width - qkv_width,
            )
        )
        solution = (
            "gluon_wmma_gfx1250"
            if gfx1250_tdm
            else (
                "gluon_largem_gfx1250"
                if gfx1250_largem
                else ("fused" if m <= 32 or _use_mv(hidden_states, weight) else "split")
            )
        )

    if solution == "gluon_wmma_gfx1250":
        if not (
            Platform.get().is_cdna5
            and 2 <= m <= 64
            and input_width == KIMI3_HIDDEN_SIZE
            and qkv_width == 2112
            and output_width - qkv_width == 1536
            and hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
        ):
            raise ValueError(
                "Kimi K3 gfx1250 MLA WMMA projection requires contiguous BF16 "
                "A [M,7168] for M in 2-64, weight [3648,7168], and qkv_width=2112"
            )
        from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
            gluon_wmma_tdm_mla_qkv_gate_gfx1250,
        )

        packed = gluon_wmma_tdm_mla_qkv_gate_gfx1250(
            hidden_states,
            weight,
        )
        qkv, gate = packed.split((qkv_width, output_width - qkv_width), dim=-1)
        return Kimi3MLAQKVGateProjection(qkv=qkv, gate=gate, packed=packed)

    if solution == "fused":
        packed = mv(hidden_states, weight)
        qkv, gate = packed.split((qkv_width, output_width - qkv_width), dim=-1)
        return Kimi3MLAQKVGateProjection(qkv=qkv, gate=gate, packed=packed)

    if solution == "gluon_largem_gfx1250":
        if not (
            Platform.get().is_cdna5
            and hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
            and use_gluon_largem_gfx1250(m, hidden_states.shape[1], qkv_width)
            and use_gluon_largem_gfx1250(
                m,
                hidden_states.shape[1],
                output_width - qkv_width,
            )
        ):
            raise ValueError(
                "Kimi K3 gfx1250 MLA Gluon projection requires contiguous "
                "BF16 K3 QKV/gate shapes with M >= 512"
            )
        from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
            gluon_mm_a16w16_largem_gfx1250,
        )

        qkv = gluon_mm_a16w16_largem_gfx1250(
            hidden_states,
            weight[:qkv_width],
        )
        gate = gluon_mm_a16w16_largem_gfx1250(
            hidden_states,
            weight[qkv_width:],
        )
        return Kimi3MLAQKVGateProjection(qkv=qkv, gate=gate, packed=None)

    qkv_weight = weight[:qkv_width]
    gate_weight = weight[qkv_width:]
    qkv = _try_gluon_largem_gfx950(hidden_states, qkv_weight) if routed else None
    if qkv is None:
        qkv = torch.nn.functional.linear(hidden_states, qkv_weight)
    gate = _try_gluon_largem_gfx950(hidden_states, gate_weight) if routed else None
    if gate is None:
        gate = torch.nn.functional.linear(hidden_states, gate_weight)
    return Kimi3MLAQKVGateProjection(qkv=qkv, gate=gate, packed=None)


# Retained fused add3 tile configurations, independent of ordinary GEMV tuning.
# These configurations were measured on sm103 for the K3 latent-up projection.
_SKINNY_ADD3_CONFIGS = MappingProxyType(
    {(1, 7168, 3584): (64, 4, 2), (2, 7168, 3584): (64, 7, 2)}
)
_skinny_add3_warmed: set[tuple[int, int, int, int]] = set()
_skinny_add3_warmed_lock = threading.Lock()


@lru_cache(maxsize=8)
def _skinny_add3_arch_supported(device_index: int) -> bool:
    """Keep the fused tile configurations on their measured architecture."""
    if Platform.get().vendor != "nvidia":
        return False
    return torch.cuda.get_device_capability(device_index) == (10, 3)


def _skinny_add3_supported(m: int, n: int, k: int, device: torch.device) -> bool:
    return (m, n, k) in _SKINNY_ADD3_CONFIGS and _skinny_add3_arch_supported(
        device.index or 0
    )


def _skinny_add3_fallback(
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


def _skinny_gemv_add3(
    x: torch.Tensor,
    weight: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    out: torch.Tensor | None,
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
    key = (dev, m, n, k)
    tuned = _SKINNY_ADD3_CONFIGS.get((m, n, k))
    if (
        tuned is None
        or x.dtype != torch.bfloat16
        or not _skinny_add3_arch_supported(dev)
        or (torch.cuda.is_current_stream_capturing() and key not in _skinny_add3_warmed)
    ):
        return _skinny_add3_fallback(x, weight, a, c, out)
    config = SkinnyGemmConfig(m, *tuned)
    if not shape_dynamic_skinny_gemm.supports(config, m, n, k):
        return _skinny_add3_fallback(x, weight, a, c, out)
    result = shape_dynamic_skinny_gemm(
        x.detach(),
        weight.detach(),
        config,
        residual=a.detach(),
        residual2=c.detach(),
        out=out,
    )
    # Only successful eager calls earn capture trust, separately for each GPU.
    if not torch.cuda.is_current_stream_capturing():
        with _skinny_add3_warmed_lock:
            _skinny_add3_warmed.add(key)
    return result


def kimi3_latent_projection_add3(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    prefix: torch.Tensor,
    shared_output: torch.Tensor,
    *,
    norm_weight: torch.Tensor | None = None,
    eps: float | None = None,
    solution: str = "auto",
) -> torch.Tensor:
    """Project K3 latent rows and add the residual and shared-expert output.

    Args:
        hidden_states: Contiguous BF16 latent rows shaped ``[M, K]``.
        weight: Contiguous BF16 projection weight shaped ``[N, K]``.
        prefix: BF16 residual rows shaped ``[M, N]``.
        shared_output: BF16 shared-expert rows shaped ``[M, N]``.
        norm_weight: Optional contiguous BF16 RMSNorm weight shaped ``[K]``.
            When provided, RMSNorm is applied to ``hidden_states`` before the
            projection.
        eps: Positive RMSNorm epsilon required with ``norm_weight``.
        solution: ``"auto"`` selects the dual-residual skinny epilogue where
            it holds a measured win (sm103, M <= 2), the fused row-CTA GEMV
            for other one-token execution, the fused MFMA epilogue for CDNA4
            decode batches (2 <= M <= 32), the fused WMMA epilogue for CDNA5
            batches of 2 to 64 rows, and otherwise composes the
            registered projection and add kernels. ``"rowcta_gemv"``, ``"skinny_add3"``,
            ``"gluon_mfma_add3"``, ``"gluon_wmma_add3"``,
            ``"triton_wmma_add3"``, and ``"composed"`` force an implementation.

    Returns:
        ``prefix + hidden_states @ weight.T + shared_output`` shaped ``[M, N]``.
    """

    m, n, k = _validate_fallback_projection(
        hidden_states,
        weight,
        None,
        name="Kimi K3 latent projection-add3",
    )
    expected_shape = (m, n)
    for name, tensor in (("prefix", prefix), ("shared_output", shared_output)):
        if tensor.shape != expected_shape:
            raise ValueError(
                f"Kimi K3 latent projection {name} must be {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != hidden_states.dtype or tensor.device != hidden_states.device:
            raise ValueError(
                f"Kimi K3 latent projection {name} must match input dtype and device"
            )
        if tensor.stride(1) != 1:
            raise ValueError(
                f"Kimi K3 latent projection {name} must have unit inner stride"
            )

    if solution not in {
        "auto",
        "rowcta_gemv",
        "skinny_add3",
        "gluon_mfma_add3",
        "gluon_wmma_add3",
        "triton_wmma_add3",
        "composed",
    }:
        raise ValueError(f"unknown Kimi K3 projection-add3 solution {solution!r}")
    specialized = (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and weight.is_contiguous()
        and (k, n) in _KIMI3_SHAPES
    )
    if norm_weight is not None:
        if (
            norm_weight.shape != (k,)
            or norm_weight.dtype != hidden_states.dtype
            or norm_weight.device != hidden_states.device
            or not norm_weight.is_contiguous()
        ):
            raise ValueError("Kimi K3 norm weight must match the latent input")
        if eps is None or eps <= 0.0:
            raise ValueError("Kimi K3 RMSNorm epsilon must be positive")
        if (
            solution == "auto"
            and Platform.get().is_cdna4
            and m <= 2
            and (k, n) == (KIMI3_LATENT_SIZE, KIMI3_HIDDEN_SIZE)
            and specialized
        ):
            from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.rmsnorm_linear_add import (
                gluon_rmsnorm_linear_add_gfx950,
            )

            return gluon_rmsnorm_linear_add_gfx950(
                hidden_states,
                norm_weight,
                weight,
                prefix,
                shared_output,
                eps=eps,
            )
        if (
            solution == "auto"
            and Platform.get().is_cdna4
            and (k, n) == (KIMI3_LATENT_SIZE, KIMI3_HIDDEN_SIZE)
            and specialized
            and supports_gluon_mm_a16w16_decode_add3_gfx950(m, n, k)
        ):
            from tokenspeed_kernel.ops.layernorm.triton import rmsnorm

            # Rows the fused norm kernel above leaves (3 <= m <= 32) normalize
            # in one Triton launch, then take the fused add3 below as without a
            # norm. Keying on the add3 predicate keeps the two ranges aligned.
            hidden_states = rmsnorm(hidden_states, norm_weight, eps)
        else:
            source = hidden_states.float()
            hidden_states = (
                source
                * torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + eps)
                * norm_weight.float()
            ).to(hidden_states.dtype)
    elif eps is not None:
        raise ValueError("Kimi K3 RMSNorm epsilon requires a norm weight")

    if solution == "auto":
        autotune_bf16_gemm(hidden_states, weight)
        if specialized and _skinny_add3_supported(m, n, k, hidden_states.device):
            # Dual-residual skinny epilogue: 1.06x over rowcta_gemv_add3.
            solution = "skinny_add3"
        elif m == 1 and specialized:
            solution = "rowcta_gemv"
        elif (
            Platform.get().is_cdna4
            and (k, n) == (KIMI3_LATENT_SIZE, KIMI3_HIDDEN_SIZE)
            and specialized
            and supports_gluon_mm_a16w16_decode_add3_gfx950(m, n, k)
        ):
            # Fusing the additions saves the add3 launch: 2.0-2.6 us per call.
            solution = "gluon_mfma_add3"
        elif (
            Platform.get().is_cdna5
            and 2 <= m <= 64
            and (k, n) == (KIMI3_LATENT_SIZE, KIMI3_HIDDEN_SIZE)
            and specialized
        ):
            solution = "gluon_wmma_add3"
        else:
            solution = "composed"
    if solution == "skinny_add3":
        if not specialized:
            raise ValueError(
                "skinny_add3 projection-add3 requires contiguous CUDA BF16 "
                "inputs at a K3 latent shape"
            )
        return _skinny_gemv_add3(
            hidden_states,
            weight,
            prefix,
            shared_output,
            None,
        )
    if solution == "rowcta_gemv":
        if m != 1:
            raise ValueError("rowcta_gemv projection-add3 requires one input row")
        if not specialized:
            raise ValueError(
                "rowcta_gemv projection-add3 requires a contiguous CUDA BF16 "
                "7168<->3584 shape"
            )
        return rowcta_gemv_add3(
            hidden_states,
            weight,
            prefix,
            shared_output,
        )
    if solution == "gluon_mfma_add3":
        if (
            not Platform.get().is_cdna4
            or (k, n) != (KIMI3_LATENT_SIZE, KIMI3_HIDDEN_SIZE)
            or not specialized
            or not supports_gluon_mm_a16w16_decode_add3_gfx950(m, n, k)
        ):
            raise ValueError(
                "gluon_mfma_add3 projection-add3 requires 2-32 contiguous CUDA "
                "BF16 rows with the K3 3584->7168 shape on CDNA4"
            )
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
            launch_gluon_mm_a16w16_decode_add3_gfx950,
        )

        return launch_gluon_mm_a16w16_decode_add3_gfx950(
            hidden_states,
            weight,
            prefix,
            shared_output,
        )
    if solution == "gluon_wmma_add3":
        if (
            not Platform.get().is_cdna5
            or not 2 <= m <= 64
            or (k, n) != (KIMI3_LATENT_SIZE, KIMI3_HIDDEN_SIZE)
            or not specialized
        ):
            raise ValueError(
                "gluon_wmma_add3 projection-add3 requires 2-64 contiguous "
                "CUDA BF16 rows with the K3 3584->7168 shape on CDNA5"
            )
        from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
            gluon_wmma_tdm_add3_gfx1250,
            gluon_wmma_tdm_add3_m16_gfx1250,
        )

        add3 = (
            gluon_wmma_tdm_add3_m16_gfx1250 if m == 16 else gluon_wmma_tdm_add3_gfx1250
        )
        return add3(
            hidden_states,
            weight,
            prefix,
            shared_output,
        )
    if solution == "triton_wmma_add3":
        if (
            not Platform.get().is_cdna5
            or m != 16
            or (k, n) != (KIMI3_LATENT_SIZE, KIMI3_HIDDEN_SIZE)
            or not specialized
        ):
            raise ValueError(
                "triton_wmma_add3 projection-add3 requires 16 contiguous CUDA "
                "BF16 rows with the K3 3584->7168 shape on CDNA5"
            )
        from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
            triton_mm_a16w16_add3_m16_gfx1250,
        )

        return triton_mm_a16w16_add3_m16_gfx1250(
            hidden_states,
            weight,
            prefix,
            shared_output,
            block_n=32,
            block_k=64,
            num_warps=2,
            waves_per_eu=1,
        )

    projected = kimi3_latent_projection(hidden_states, weight)
    from tokenspeed_kernel.ops.activation.triton import add3

    return add3(prefix, projected, shared_output)


def kimi3_shared_situ_projection(
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    *,
    beta: float = 1.0,
    linear_beta: float | None = None,
    out: torch.Tensor | None = None,
    solution: str = "auto",
) -> torch.Tensor:
    """Apply K3's TP8 local shared-expert gate/up projection and SiTU.

    Args:
        hidden_states: Contiguous BF16 activation shaped ``[M, 7168]``.
        gate_up_weight: Contiguous BF16 TP8 shard shaped ``[1536, 7168]``.
        beta: Positive SiTU gate soft-clipping scale.
        linear_beta: Optional positive SiTU up-branch soft-clipping scale.
        out: Optional contiguous BF16 output shaped ``[M, 768]``.
        solution: ``"auto"`` selects the fused gfx950 decode kernel, or the
            Gluon large-M projection for measured prefill token counts, and
            otherwise uses the portable Torch projection plus TokenSpeed SiTU
            kernel.

    Returns:
        The local activated shared-expert rows shaped ``[M, 768]``.
    """

    m, gate_up_width, hidden_width = _validate_fallback_projection(
        hidden_states,
        gate_up_weight,
        None,
        name="Kimi K3 shared SiTU projection",
    )
    if gate_up_width % 2:
        raise ValueError("Kimi K3 shared SiTU projection requires an even output width")
    output_width = gate_up_width // 2
    expected_output = (m, output_width)
    if beta <= 0.0 or (linear_beta is not None and linear_beta <= 0.0):
        raise ValueError("Kimi K3 shared SiTU beta values must be positive")
    if out is None:
        out = hidden_states.new_empty(expected_output)
    elif (
        tuple(out.shape) != expected_output
        or out.dtype != hidden_states.dtype
        or out.device != hidden_states.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "Kimi K3 shared SiTU out must match the activated projection shape"
        )
    if solution not in {"auto", "triton_gemv", "torch"}:
        raise ValueError(f"unknown Kimi K3 shared SiTU solution {solution!r}")
    routed = solution == "auto"
    specialized = (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and gate_up_weight.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and gate_up_weight.is_contiguous()
        and m == 1
        and hidden_width == KIMI3_HIDDEN_SIZE
        and gate_up_width == KIMI3_SHARED_GATE_UP_LOCAL_SIZE
    )
    if solution == "auto":
        autotune_bf16_gemm(hidden_states, gate_up_weight)
        solution = "triton_gemv" if Platform.get().is_cdna4 and specialized else "torch"
    if solution == "triton_gemv":
        if not specialized:
            raise ValueError(
                "Kimi K3 shared SiTU Triton GEMV requires contiguous gfx950 "
                "BF16 [1, 7168] input and [1536, 7168] weight"
            )
        return kimi3_shared_situ_projection_gemv(
            hidden_states, gate_up_weight, out, beta, linear_beta
        )

    if routed and _use_mv(hidden_states, gate_up_weight):
        gate_up = mv(hidden_states, gate_up_weight)
    else:
        gate_up = None
        if routed:
            gate_up = _try_gluon_largem_gfx1250(hidden_states, gate_up_weight)
            if gate_up is None:
                gate_up = _try_gluon_largem_gfx950(hidden_states, gate_up_weight)
        if gate_up is None:
            gate_up = torch.nn.functional.linear(hidden_states, gate_up_weight)
    if gate_up.is_cuda:
        from tokenspeed_kernel.ops.activation import situ_and_mul

        return situ_and_mul(
            gate_up,
            out=out,
            beta=beta,
            linear_beta=linear_beta,
        )
    gate, up = gate_up.float().chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    out.copy_((gate * up).to(out.dtype))
    return out


def kimi3_shared_down_projection(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    solution: str = "auto",
) -> torch.Tensor:
    """Apply K3's TP8 local shared-expert down projection.

    Args:
        hidden_states: Contiguous BF16 activated rows shaped ``[M, 768]``.
        weight: Contiguous BF16 TP8 shard shaped ``[7168, 768]``.
        out: Optional BF16 output shaped ``[M, 7168]``. A contiguous tensor
            keeps the existing kernel. With ``solution="auto"``, a row-strided
            tensor is written by the CDNA5 dense WMMA when that kernel accepts
            the tensor contract; otherwise Torch writes into the same destination.
            Row-strided outputs require unit inner stride and non-overlapping
            rows, and are supported only by ``"auto"`` and ``"torch"``.
        solution: ``"auto"`` selects the gfx950 decode GEMV, or the Gluon
            large-M kernel for measured prefill token counts, and otherwise
            uses the portable Torch linear operation.

    Returns:
        The local shared-expert output contribution shaped ``[M, 7168]``.
    """

    if solution not in {"auto", "triton_gemv", "gluon_largem_gfx1250", "torch"}:
        raise ValueError(f"unknown Kimi K3 shared down solution {solution!r}")
    strided_out = out is not None and not out.is_contiguous()
    m, output_width, input_width = _validate_fallback_projection(
        hidden_states,
        weight,
        None if strided_out else out,
        name="Kimi K3 shared down projection",
    )
    if strided_out:
        if (
            tuple(out.shape) != (m, output_width)
            or out.dtype != hidden_states.dtype
            or out.device != hidden_states.device
            or out.stride(-1) != 1
            or out.stride(0) < output_width
        ):
            raise ValueError(
                "Kimi K3 shared down out must have matching shape, dtype and "
                "device, unit inner stride and non-overlapping rows"
            )
        if solution not in {"auto", "torch"}:
            raise ValueError(f"{solution!r} requires a contiguous shared down out")
        if (
            solution == "auto"
            and Platform.get().is_cdna5
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
            and use_gluon_wmma_dense_gfx1250(m, input_width, output_width)
        ):
            from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
                gluon_wmma_tdm_dense_gfx1250,
            )

            return gluon_wmma_tdm_dense_gfx1250(
                hidden_states, weight, out=out, split_k=None
            )
        if solution == "auto":
            output = _try_gluon_largem_gfx950(hidden_states, weight, out=out)
            if output is not None:
                return output
        return torch.mm(hidden_states, weight.T, out=out)
    expected_output = (m, output_width)
    if out is None:
        out = hidden_states.new_empty(expected_output)
    specialized = (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and weight.is_contiguous()
        and m == 1
        and input_width == KIMI3_SHARED_LOCAL_SIZE
        and output_width == KIMI3_HIDDEN_SIZE
    )
    routed = solution == "auto"
    if solution == "auto":
        autotune_bf16_gemm(hidden_states, weight)
        if Platform.get().is_cdna4 and specialized:
            solution = "triton_gemv"
        elif (
            Platform.get().is_cdna5
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
            and out.is_contiguous()
            and use_gluon_largem_gfx1250(
                m,
                input_width,
                output_width,
            )
        ):
            solution = "gluon_largem_gfx1250"
        else:
            solution = "torch"
    if routed and _use_mv(hidden_states, weight):
        return mv(hidden_states, weight, out)
    if solution == "triton_gemv":
        if not specialized:
            raise ValueError(
                "Kimi K3 shared down Triton GEMV requires contiguous gfx950 "
                "BF16 [1, 768] input and [7168, 768] weight"
            )
        return kimi3_projection_gemv(
            hidden_states,
            weight,
            out=out,
            config=(8, 512, 8, 1),
        )
    if solution == "gluon_largem_gfx1250":
        if not (
            Platform.get().is_cdna5
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
            and use_gluon_largem_gfx1250(m, input_width, output_width)
        ):
            raise ValueError(
                "Kimi K3 gfx1250 shared down projection requires a contiguous "
                "BF16 K3 shape with M >= 512"
            )
        from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
            gluon_mm_a16w16_largem_gfx1250,
        )

        return gluon_mm_a16w16_largem_gfx1250(
            hidden_states,
            weight,
            out=out,
        )
    if routed:
        output = _try_gluon_largem_gfx950(hidden_states, weight, out=out)
        if output is not None:
            return output
    return torch.mm(hidden_states, weight.T, out=out)


def kimi3_qkvfab_projection(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    weight_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    solution: str = "auto",
) -> torch.Tensor:
    """Project all K3 KDA hidden-state consumers in one GEMM/GEMV.

    The output rows contain local Q/K/V/output-gate projections followed by the
    replicated ``f_a`` projection, local beta logits, and alignment padding.

    BF16 weights keep the original dispatch. FP8 (e4m3) weights run the w8a8
    blockscale path (TRT-LLM's FP8_PB_WO alias): per-token 1x128 online
    activation quantization + block-scale GEMM — the same kernel path
    ``Fp8LinearMethod`` uses, called through the ``mm`` dispatcher because
    the merged projection is not a LinearBase module. Rows are padded to
    ``N % 128 == 0`` at load so the flashinfer kernel stays selected; pad
    rows carry zero codes and produce exact-zero outputs, and callers only
    consume the used rows.

    Args:
        hidden_states: BF16 activation shaped ``[M, 7168]``.
        weight: Stacked weight ``[N, 7168]``: BF16, or FP8 with
            ``weight_scale``.
        weight_scale: f32 ``[N/128, 7168/128]`` block dequant multipliers
            (FP8 weights only).
        out: Optional contiguous BF16 output buffer shaped ``[M, N]``.
        solution: ``"auto"`` selects the architecture-specific BF16 route;
            ``"triton_gemv"``, ``"gluon_wmma_gfx1250"``,
            ``"gluon_largem_gfx1250"``, ``"gluon_largem"`` (gfx950 prefill),
            and ``"torch"`` force one. (BF16 path only.)

    Returns:
        The projected BF16 tensor shaped ``[M, N]``.
    """
    m, output_width, input_width = _validate_fallback_projection(
        hidden_states,
        weight,
        out,
        name="Kimi K3 QKVFAB projection",
    )
    if weight.dtype in _FP8_WEIGHT_DTYPES:
        if weight_scale is None:
            raise ValueError("FP8 Kimi K3 QKVFAB projection requires weight_scale")
        result = mm(
            hidden_states,
            weight,
            B_scales=weight_scale,
            out_dtype=hidden_states.dtype,
            quant="mxfp8",
            block_size=[128, 128],
        )
        if out is None:
            return result
        out.copy_(result)
        return out
    if weight_scale is not None:
        raise ValueError("weight_scale is only valid with FP8 weights")
    if solution not in {
        "auto",
        "mv",
        "triton_gemv",
        "gluon_wmma_gfx1250",
        "gluon_largem_gfx1250",
        "gluon_largem",
        "torch",
    }:
        raise ValueError(f"unknown Kimi K3 QKVFAB solution {solution!r}")
    specialized = (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and weight.is_contiguous()
        and m <= 8
        and input_width == KIMI3_HIDDEN_SIZE
        and output_width == KIMI3_QKVFAB_SIZE
    )
    if solution == "auto":
        autotune_bf16_gemm(hidden_states, weight)
        if (
            Platform.get().is_cdna5
            and hidden_states.is_cuda
            and weight.is_cuda
            and 2 <= m <= 64
            and input_width == KIMI3_HIDDEN_SIZE
            and output_width == KIMI3_QKVFAB_SIZE
            and hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
            and (out is None or out.is_contiguous())
        ):
            solution = "gluon_wmma_gfx1250"
        elif (
            Platform.get().is_cdna5
            and hidden_states.is_cuda
            and weight.is_cuda
            and use_gluon_largem_gfx1250(m, input_width, output_width)
            and hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
            and (out is None or out.is_contiguous())
        ):
            solution = "gluon_largem_gfx1250"
        elif Platform.get().is_cdna4 and specialized and m == 1:
            solution = "triton_gemv"
        elif (
            Platform.get().is_cdna4
            and hidden_states.is_cuda
            and hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
            and (out is None or out.is_contiguous())
            and supports_gluon_mm_a16w16_prefill_gfx950(m, output_width, input_width)
        ):
            solution = "gluon_largem"
        elif specialized or _use_mv(hidden_states, weight):
            # Let the registry pick per (M, N, K); unlisted shapes hit torch.mm.
            solution = "mv"
        else:
            solution = "torch"
    if solution == "gluon_wmma_gfx1250":
        if not (
            Platform.get().is_cdna5
            and 2 <= m <= 64
            and input_width == KIMI3_HIDDEN_SIZE
            and output_width == KIMI3_QKVFAB_SIZE
            and hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
            and (out is None or out.is_contiguous())
        ):
            raise ValueError(
                "Kimi K3 gfx1250 QKVFAB WMMA projection requires contiguous "
                "BF16 A [M,7168] for M in 2-64 and weight [6288,7168]"
            )
        from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
            gluon_wmma_tdm_kda_qkvfab_gfx1250,
        )

        return gluon_wmma_tdm_kda_qkvfab_gfx1250(
            hidden_states,
            weight,
            out=out,
        )
    if solution == "mv":
        result = mv(hidden_states, weight)
        if out is None:
            return result
        out.copy_(result)
        return out
    if solution == "triton_gemv":
        if not (specialized and m == 1):
            raise ValueError(
                "Kimi K3 QKVFAB Triton GEMV requires contiguous gfx950 "
                "BF16 [1, 7168] input and [6288, 7168] weight"
            )
        return kimi3_projection_gemv(
            hidden_states,
            weight,
            out=out,
            config=(8, 512, 8, 1),
        )
    if solution == "gluon_largem":
        if not (
            Platform.get().is_cdna4
            and hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
            and (out is None or out.is_contiguous())
        ):
            raise ValueError(
                "Kimi K3 gfx950 QKVFAB Gluon projection requires contiguous "
                "BF16 inputs and output"
            )
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
            launch_gluon_mm_a16w16_prefill_gfx950,
        )

        output = launch_gluon_mm_a16w16_prefill_gfx950(
            hidden_states, weight, hidden_states.dtype, out=out
        )
        if output is None:
            raise ValueError("Kimi K3 gfx950 QKVFAB Gluon projection rejected shape")
        return output
    if solution == "gluon_largem_gfx1250":
        if not (
            Platform.get().is_cdna5
            and use_gluon_largem_gfx1250(m, input_width, output_width)
            and hidden_states.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and weight.is_contiguous()
            and (out is None or out.is_contiguous())
        ):
            raise ValueError(
                "Kimi K3 gfx1250 QKVFAB Gluon projection requires contiguous "
                "BF16 [M,7168] input and [6288,7168] weight with M >= 512"
            )
        from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
            gluon_mm_a16w16_largem_gfx1250,
        )

        return gluon_mm_a16w16_largem_gfx1250(
            hidden_states,
            weight,
            out=out,
        )
    if out is None:
        return torch.nn.functional.linear(hidden_states, weight)
    return torch.mm(hidden_states, weight.T, out=out)


# Largest token count the hand-written router CUDA kernel still wins at; the
# tensor-core GEMM takes over above it (see kimi3_router_projection docstring).
_ROUTER_CUDA_MAX_TOKENS = 4


def _ll_bf16_usable(hidden_states: torch.Tensor, weight: torch.Tensor, m: int) -> bool:
    """Whether the vendored CuTe dot-product router GEMM can serve this call."""
    return ll_bf16_router_supported(hidden_states, weight, m)


@lru_cache(maxsize=1)
def _mm_out_dtype_supported() -> bool:
    """Whether ``torch.mm`` accepts ``out_dtype`` (BF16 in, FP32 out).

    Exposed by torch >= 2.8 as the cublasLt BF16xBF16->FP32 epilogue; probed
    once so older torch falls back to the CUDA-kernel/torch paths untouched.
    """
    try:
        a = torch.empty(1, 2, dtype=torch.bfloat16, device="cuda")
        return torch.mm(a, a.t(), out_dtype=torch.float32).dtype == torch.float32
    except (TypeError, RuntimeError):
        return False


def kimi3_router_projection(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    solution: str = "auto",
) -> torch.Tensor:
    """Compute K3's BF16-input/BF16-weight router logits directly in FP32.

    Args:
        hidden_states: BF16 activation shaped ``[M, 7168]``.
        weight: BF16 router weight shaped ``[896, 7168]``.
        out: Optional contiguous FP32 output buffer shaped ``[M, 896]``.
        solution: ``"auto"`` selects a specialized CDNA4 or Hopper kernel
            when eligible and otherwise falls back to Torch. On NVIDIA the
            vendored CuTe ``"ll_bf16"`` kernels serve ``M <= 32`` when CuTe
            DSL is installed (dot product to 8, split-K above); otherwise the
            hand-written CUDA kernel serves ``M <= 4`` and ``"cublas"``
            (``torch.mm`` with ``out_dtype``) serves larger batches: the CUDA
            kernel's per-thread token loop runs on CUDA cores, so its time
            grows linearly with M (4.0us at M=1 -> 34.8us at M=32 on B300)
            while the tensor-core GEMM stays flat (~7.3us), crossing between
            M=4 and M=8.
    Returns:
        FP32 router logits shaped ``[M, 896]``.
    """
    m, output_width, input_width = _validate_fallback_projection(
        hidden_states,
        weight,
        out,
        name="Kimi K3 router projection",
        out_dtype=torch.float32,
    )
    if solution not in {
        "auto",
        "ll_bf16",
        "cuda",
        "cublas",
        "triton_gemv",
        "gluon_decode",
        "torch",
    }:
        raise ValueError(f"unknown Kimi K3 router solution {solution!r}")
    specialized = (
        hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and weight.is_contiguous()
        and input_width == KIMI3_HIDDEN_SIZE
        and output_width == KIMI3_ROUTER_SIZE
    )
    if solution == "auto":
        platform = Platform.get()
        if platform.is_cdna4 and specialized and m == 1:
            solution = "triton_gemv"
        elif (
            platform.is_cdna4
            and specialized
            and supports_gluon_mm_a16w16_decode_gfx950(m, output_width, input_width)
        ):
            solution = "gluon_decode"
        elif platform.is_hopper_plus and specialized:
            if _ll_bf16_usable(hidden_states, weight, m):
                solution = "ll_bf16"
            elif m > _ROUTER_CUDA_MAX_TOKENS and _mm_out_dtype_supported():
                solution = "cublas"
            else:
                solution = "cuda"
        else:
            solution = "torch"
    if solution == "ll_bf16":
        return cute_dsl_ll_bf16_router(hidden_states, weight, out)
    if solution == "cublas":
        if not specialized or not _mm_out_dtype_supported():
            raise ValueError(
                "Kimi K3 router cublas path requires contiguous BF16 "
                "[M, 7168] input, [896, 7168] weight, and torch.mm out_dtype "
                "support (torch >= 2.8)"
            )
        # cublasLt BF16xBF16 with FP32 accumulate AND FP32 store: same output
        # semantics as the CUDA kernel, flat ~7.3us across M on B300.
        logits = torch.mm(hidden_states, weight.t(), out_dtype=torch.float32)
        if out is None:
            return logits
        out.copy_(logits)
        return out
    if solution == "triton_gemv":
        if not specialized or m != 1:
            raise ValueError(
                "Kimi K3 router Triton GEMV requires contiguous gfx950 "
                "BF16 [1, 7168] input and [896, 7168] weight"
            )
        if out is None:
            out = torch.empty(
                (m, output_width),
                dtype=torch.float32,
                device=hidden_states.device,
            )
        return kimi3_projection_gemv(
            hidden_states,
            weight,
            out=out,
            config=((4, 1024, 4, 1) if m == 1 else (8, 256, 8, 1)),
        )
    if solution == "gluon_decode":
        if not specialized:
            raise ValueError(
                "Kimi K3 router Gluon decode GEMM requires contiguous gfx950 "
                "BF16 [M, 7168] input and [896, 7168] weight"
            )
        from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
            gluon_mm_a16w16_decode_num_counters_gfx950,
            launch_gluon_mm_a16w16_decode_gfx950,
        )

        counters = zeroed_state(
            gluon_mm_a16w16_decode_num_counters_gfx950(
                hidden_states.shape[0], weight.shape[0], hidden_states.shape[1]
            ),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        # FP32 accumulate and FP32 store, matching the other router paths.
        return launch_gluon_mm_a16w16_decode_gfx950(
            hidden_states, weight, torch.float32, counters=counters, out=out
        )
    if solution == "cuda":
        if not specialized:
            raise ValueError(
                "Kimi K3 router CUDA kernel requires contiguous BF16 "
                "[M, 7168] input and [896, 7168] weight"
            )
        logits = dsv3_router_gemm(
            hidden_states,
            weight,
            out_dtype=torch.float32,
            enable_pdl=pdl_enabled(),
        )
        if out is None:
            return logits
        out.copy_(logits)
        return out
    logits = torch.nn.functional.linear(hidden_states.float(), weight.float())
    if out is None:
        return logits
    out.copy_(logits)
    return out


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.numerics.torch.gemm  # noqa: E402,F401
import tokenspeed_kernel.ops.gemm.cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.gemm.deepgemm  # noqa: E402,F401
import tokenspeed_kernel.ops.gemm.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.gemm.flashinfer  # noqa: E402,F401
import tokenspeed_kernel.ops.gemm.gluon  # noqa: E402,F401
import tokenspeed_kernel.ops.gemm.native  # noqa: E402,F401
import tokenspeed_kernel.ops.gemm.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.gemm.triton_fp8_channel  # noqa: E402,F401
import tokenspeed_kernel.ops.gemm.trtllm  # noqa: E402,F401
from tokenspeed_kernel.ops.gemm.cuda import dsv3_router_gemm  # noqa: E402
from tokenspeed_kernel.ops.gemm.cute_dsl import (  # noqa: E402
    cute_dsl_ll_bf16_router,
    ll_bf16_router_supported,
)
from tokenspeed_kernel.ops.gemm.flashinfer import (  # noqa: E402
    BF16_GEMM_MAX_M,
    autotune_bf16_gemm,
    flashinfer_bf16_gemm,
    flashinfer_joint_bf16_supported,
    gemm_fp8_nt_groupwise,
    has_flashinfer_cute_dsl_nvfp4_a16,
    has_flashinfer_mxfp8,
    prepare_nvfp4_a16_weights,
    swizzle_mxfp8_scale,
)

if has_flashinfer_mxfp8():
    from tokenspeed_kernel.ops.gemm.flashinfer import (  # noqa: E402
        flashinfer_mm_mxfp8,
    )
from tokenspeed_kernel.ops.gemm.gluon import (  # noqa: E402
    MXFP4_LINEAR_MFMA_LAYOUT,
    prepare_mxfp4_linear_mfma_weight,
)
from tokenspeed_kernel.ops.gemm.native import torch_decode_gemv  # noqa: E402
from tokenspeed_kernel.ops.gemm.triton import (  # noqa: E402
    MXFP4_LINEAR_LINEAR_LAYOUT,
    MXFP4_LINEAR_SIGNATURE,
)


# isort: on
def fp32_router_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    multiplier: float,
    counters: torch.Tensor,
    enable_pdl: bool,
) -> torch.Tensor:
    """Compute ``(x @ weight) * multiplier`` in FP32 for decode-sized batches.

    The weight keeps a row-major ``[K, N]`` layout (as MoE routers store it).
    Up to ``ROUTER_GEMV_MAX_ROWS`` rows run as one exact-FP32 split-K Triton
    launch whose partials are reduced in a fixed order. Where cuBLAS would use
    TF32, up to ``ROUTER_DOT_MAX_ROWS`` rows run the same split-K on TF32 tensor
    cores with cuBLAS's rounding; other batches take ``torch.matmul``. Repeat
    calls on each path are bitwise stable, but the paths round differently, so
    a batch that crosses a threshold sees differences at that path's precision.

    Args:
        x: Contiguous FP32 input ``[rows, hidden]``.
        weight: Contiguous FP32 weight ``[hidden, columns]``.
        multiplier: Scale applied to the logits.
        counters: Buffer from ``fp32_router_gemv_counters`` for ``columns``.
        enable_pdl: Enable NVIDIA programmatic dependent launch.

    Returns:
        FP32 outputs ``[rows, columns]``.
    """
    kernel = select_kernel(
        "gemm",
        "fp32_router_gemv",
        dense_format_signature(("x", "weight"), (x.dtype, weight.dtype)),
    )
    return kernel(x, weight, multiplier, counters, enable_pdl)


@dataclass(frozen=True)
class _FP32RouterGEMVPlan:
    kernel: SelectedKernel
    weight_preprocessor: Callable
    hidden_size: int
    output_size: int
    dtype: torch.dtype


def fp32_router_gemv_plan(
    *,
    hidden_size: int,
    output_size: int,
    dtype: torch.dtype,
    solution: str | None,
    batch_invariant: bool,
) -> object:
    """Select an FP32 router GEMV and its persistent weight layout.

    Args:
        hidden_size: Router input width.
        output_size: Number of router outputs.
        dtype: Input and canonical weight dtype, which must be FP32.
        solution: Optional implementation family override.
        batch_invariant: Require a provider declaring the ``batch_invariant``
            feature, as bitwise numerics envelopes do.

    Returns:
        An opaque plan accepted by the related process and execute APIs.
    """
    if hidden_size <= 0 or output_size <= 0:
        raise ValueError("router dimensions must be positive")
    if dtype != torch.float32:
        raise ValueError("router GEMV requires FP32 inputs and weights")
    signature = dense_format_signature(("x", "weight"), (dtype, dtype))
    kernel = select_kernel(
        "gemm",
        "prepared_fp32_router_gemv",
        signature,
        features=frozenset({"batch_invariant"}) if batch_invariant else frozenset(),
        solution=solution,
    )
    spec = KernelRegistry.get().get_by_name(kernel.name)
    if spec is None or spec.weight_preprocessor is None:
        raise RuntimeError(f"Router GEMV kernel {kernel.name!r} has no preprocessor")
    return _FP32RouterGEMVPlan(
        kernel=kernel,
        weight_preprocessor=spec.weight_preprocessor,
        hidden_size=hidden_size,
        output_size=output_size,
        dtype=dtype,
    )


def _require_fp32_router_gemv_plan(plan: object) -> _FP32RouterGEMVPlan:
    if not isinstance(plan, _FP32RouterGEMVPlan):
        raise TypeError("plan must be returned by fp32_router_gemv_plan")
    return plan


def fp32_router_gemv_process_weight(
    plan: object,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Convert a canonical router weight to the plan's persistent layout.

    Args:
        plan: Opaque router GEMV plan.
        weight: Contiguous canonical FP32 weight in ``[hidden, outputs]`` layout.

    Returns:
        Weight in the selected implementation's persistent layout.
    """
    typed_plan = _require_fp32_router_gemv_plan(plan)
    expected_shape = (typed_plan.hidden_size, typed_plan.output_size)
    if tuple(weight.shape) != expected_shape:
        raise ValueError(
            f"router weight must have shape {expected_shape}, got {tuple(weight.shape)}"
        )
    if weight.dtype != typed_plan.dtype or not weight.is_contiguous():
        raise ValueError("router weight must be contiguous FP32")
    return typed_plan.weight_preprocessor(weight)


def fp32_router_gemv_execute(
    plan: object,
    x: torch.Tensor,
    prepared_weight: torch.Tensor,
    multiplier: float,
    counters: torch.Tensor,
    enable_pdl: bool,
) -> torch.Tensor:
    """Execute a planned router GEMV using a preprocessed weight.

    Args:
        plan: Opaque router GEMV plan.
        x: Contiguous FP32 input rows.
        prepared_weight: Result of :func:`fp32_router_gemv_process_weight`.
        multiplier: Scale applied to the router logits.
        counters: Per-call-site synchronization storage.
        enable_pdl: Enable programmatic dependent launch when supported.

    Returns:
        FP32 router logits in ``[rows, outputs]`` layout.
    """
    typed_plan = _require_fp32_router_gemv_plan(plan)
    if (
        x.ndim != 2
        or x.shape[1] != typed_plan.hidden_size
        or x.dtype != typed_plan.dtype
        or not x.is_contiguous()
    ):
        raise ValueError(
            "router input must be contiguous FP32 " f"[rows, {typed_plan.hidden_size}]"
        )
    return typed_plan.kernel(
        x,
        prepared_weight,
        multiplier,
        counters,
        enable_pdl,
    )
