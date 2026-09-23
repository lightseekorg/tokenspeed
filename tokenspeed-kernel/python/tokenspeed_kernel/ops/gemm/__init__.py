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
from collections.abc import Callable
from dataclasses import dataclass
from math import prod

# Backend registration (side-effect imports)
import tokenspeed_kernel.numerics.reference.gemm  # noqa: F401
import tokenspeed_kernel.ops.gemm.cuda  # noqa: F401
import tokenspeed_kernel.ops.gemm.flashinfer  # noqa: F401
import tokenspeed_kernel.ops.gemm.gluon  # noqa: F401
import tokenspeed_kernel.ops.gemm.ll_bf16  # noqa: F401
import tokenspeed_kernel.ops.gemm.routed_gemv  # noqa: F401
import tokenspeed_kernel.ops.gemm.triton  # noqa: F401
import tokenspeed_kernel.ops.gemm.trtllm  # noqa: F401
import tokenspeed_kernel.ops.gemm.trtllm_cutedsl  # noqa: F401
import torch
from tokenspeed_kernel.ops.gemm.flashinfer import (
    has_flashinfer_cute_dsl_nvfp4_a16,
    prepare_nvfp4_a16_weights,
)
from tokenspeed_kernel.ops.gemm.kimi3 import (
    kimi3_latent_projection,
    kimi3_latent_projection_add3,
    kimi3_mla_qkv_gate_projection,
    kimi3_qkvfab_projection,
    kimi3_router_projection,
    kimi3_shared_down_projection,
    kimi3_shared_situ_projection,
)
from tokenspeed_kernel.ops.gemm.linear_attnres_partials import (
    linear_attnres_partials,
    linear_attnres_partials_available,
)
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.platform import Platform, pdl_enabled
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    SelectedKernel,
    select_kernel,
)
from tokenspeed_kernel.signature import (
    ScaleFormat,
    dense_tensor_format,
    format_signature,
    tensor_format,
)

logger = logging.getLogger(__name__)

__all__ = [
    "bmm",
    "dsv4_grouped_output_projection",
    "dsv4_grouped_output_projection_plan",
    "dsv4_grouped_output_projection_warmup",
    "dsv4_grouped_output_projection_warmup_model",
    "dsv4_linear_fp32",
    "has_flashinfer_cute_dsl_nvfp4_a16",
    "linear_attnres_partials",
    "linear_attnres_partials_available",
    "kimi3_latent_projection",
    "kimi3_mla_qkv_gate_projection",
    "kimi3_latent_projection_add3",
    "kimi3_qkvfab_projection",
    "kimi3_router_projection",
    "kimi3_shared_down_projection",
    "kimi3_shared_situ_projection",
    "mm",
    "prepare_nvfp4_a16_weights",
]

_platform = Platform.get()
_fp8_dtype = torch.float8_e4m3fn

# ---------------------------------------------------------------------------
# Selection traits
# ---------------------------------------------------------------------------
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


@dataclass(frozen=True)
class _GroupedOutputProjectionPlan:
    kernel: SelectedKernel
    warmup: Callable | None
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
    tma_aligned_scales: bool
    execution_recipe: tuple[int, int, int]


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
    solution: str | None = None,
) -> object:
    """Create an opaque plan for the DeepSeek V4 grouped output projection.

    The plan currently selects the portable Triton implementation, which
    consumes canonical checkpoint scales without preprocessing.

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
        solution: Optional implementation family override.

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

    signature = format_signature(
        attention=dense_tensor_format(input_dtype),
        weight=dense_tensor_format(weight_dtype),
    )
    traits = {
        "block_size": (block_n, block_k),
        "scale_format": scale_format,
        "weight_scale_dtype": weight_scale_dtype,
    }
    if solution not in {None, "triton"}:
        raise NotImplementedError(
            "DeepSeek V4 grouped output projections currently support only "
            "canonical scales through the Triton implementation"
        )
    kernel = select_kernel(
        "gemm",
        "dsv4_grouped_output_projection",
        signature,
        traits=traits,
        solution="triton",
    )

    tma_aligned_scales = False
    execution_recipe = (1, block_n, block_k)
    return _GroupedOutputProjectionPlan(
        kernel=kernel,
        warmup=None,
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
        tma_aligned_scales=tma_aligned_scales,
        execution_recipe=execution_recipe,
    )


def _require_grouped_output_projection_plan(
    plan: object,
) -> _GroupedOutputProjectionPlan:
    if not isinstance(plan, _GroupedOutputProjectionPlan):
        raise TypeError("plan must be returned by dsv4_grouped_output_projection_plan")
    return plan


def dsv4_grouped_output_projection(
    plan: object,
    attention: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Run inverse RoPE, FP8 quantization, and grouped ``wo_a`` projection.

    Args:
        plan: Opaque grouped output projection plan.
        attention: Attention output shaped ``[tokens, local_heads, head_dim]``.
        positions: Position index for each token.
        cos_sin_cache: Rotary embedding cache used to undo output RoPE.
        weight: Processed grouped FP8 projection weight.
        weight_scale: Scales returned by the plan's preprocessing API.

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
    kernel = typed_plan.kernel
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
            tma_aligned_scales=typed_plan.tma_aligned_scales,
            recipe=typed_plan.execution_recipe,
        )


def dsv4_grouped_output_projection_warmup(
    plan: object,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    max_tokens: int,
) -> None:
    """Warm the implementation pinned by a grouped output projection plan.

    Args:
        plan: Opaque grouped output projection plan.
        weight: Processed grouped FP8 projection weight.
        weight_scale: Scales returned by the plan's preprocessing API.
        max_tokens: Largest token count to include in the warmup sweep.

    Returns:
        None.
    """
    typed_plan = _require_grouped_output_projection_plan(plan)
    if typed_plan.warmup is None:
        return
    typed_plan.warmup(
        weight=weight,
        weight_scale=weight_scale,
        num_groups=typed_plan.num_groups,
        output_dim=typed_plan.output_dim,
        input_dim=typed_plan.heads_per_group * typed_plan.head_dim,
        block_size=typed_plan.block_size,
        tma_aligned_scales=typed_plan.tma_aligned_scales,
        recipe=typed_plan.execution_recipe,
        max_tokens=max_tokens,
    )


def dsv4_grouped_output_projection_warmup_model(
    model: torch.nn.Module,
    max_tokens: int,
) -> None:
    """Warm every distinct grouped output projection plan attached to a model.

    Args:
        model: Model containing layers with prepared grouped projection plans.
        max_tokens: Largest token count to include in each backend warmup sweep.

    Returns:
        None.
    """
    seen: set[_GroupedOutputProjectionPlan] = set()
    for module in model.modules():
        plan = getattr(module, "_dsv4_grouped_output_projection_plan", None)
        if plan is None:
            continue
        typed_plan = _require_grouped_output_projection_plan(plan)
        if typed_plan in seen:
            continue
        seen.add(typed_plan)
        dsv4_grouped_output_projection_warmup(
            typed_plan,
            module.weight,
            module.weight_scale_inv,
            max_tokens,
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
            "b_inner_stride_one": weight.stride(-1) == 1,
            "is_cuda": x.is_cuda,
        },
        solution=solution,
    )
    return kernel(x, weight, out)


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
    traits = {
        "has_tokens": hidden_states.numel() > 0,
        "hidden_rank": hidden_states.ndim,
    }
    signature = format_signature(
        hidden_states=dense_tensor_format(hidden_states.dtype),
        weight=dense_tensor_format(weight.dtype),
    )
    k = int(weight.shape[1])
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
        "flashinfer_cute_dsl_mm_nvfp4_a16",
        "flashinfer_mm_nvfp4",
    }
)


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
        # Kernel selection precedes online activation quantization, so the
        # signature must predict the scale storage produced afterward. Most
        # paths emit FP32 scales; on CDNA5, canonical (1, 32) uint8 weight
        # scales identify the UE8M0 contract, whose matching activation
        # quantizer also emits uint8 UE8M0 scales.
        online_scale_dtype = torch.float32
        if (
            A_scales is None
            and tuple(block_size) == (1, 32)
            and B_scales.dtype == torch.uint8
            and _platform.is_cdna5
        ):
            online_scale_dtype = torch.uint8
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
) -> tuple[torch.Tensor, torch.Tensor]:
    return quantize_fp8(
        A,
        granularity="token_group",
        group_size=block_size[1],
        scale_encoding=scale_encoding,
        enable_pdl=enable_pdl,
        solution="triton",
    )


def _kernel_handles_online_mxfp8(kernel_name: str) -> bool:
    spec = KernelRegistry.get().get_by_name(kernel_name)
    return spec is not None and spec.solution == "reference"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


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
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Dense matrix multiply with automatic kernel selection.

    Quantization type is inferred from input dtype and the presence of
    scales, or can be set explicitly via ``quant``.  When ``A_scales``
    is ``None`` for a quantized mode (e.g. ``quant="mxfp8"``), online
    activation quantization is performed here before calling the kernel.

    Args:
        A: Activation matrix ``[M, K]``.
        B: Weight matrix.
        A_scales: Activation scales.
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
        override: Force selection of a specific kernel by name (e.g.
            ``"cublaslt_mm_nvfp4"``). Bypasses heuristic scoring.
        solution: Restrict selection to a registered implementation family.
    """
    enable_pdl = pdl_enabled()
    out_dtype = out_dtype or (out.dtype if out is not None else A.dtype)

    M = A.shape[0]
    if quant == "mxfp4":
        K = A.shape[-1] * 2
        N = B.shape[0]
    elif quant == "nvfp4_a16":
        K = A.shape[-1]
        N = B.shape[0]
    else:
        K = A.shape[-1]
        N = B.shape[-1] if B.shape[0] == K else B.shape[0]

    if out is not None:
        _validate_gemm_out(
            out,
            shape=(M, N),
            dtype=out_dtype,
            device=A.device,
            op="mm",
        )

    block_scale_layout = (
        "canonical_blackwell" if Platform.get().is_blackwell_plus else "canonical"
    )
    traits: dict[str, object] = {
        "m": M,
        "n": N,
        "k": K,
        "a_inner_stride_one": A.stride(-1) == 1,
        "a_scales_inner_stride_one": (A_scales is None or A_scales.stride(-1) == 1),
        "b_inner_stride_one": B.stride(-1) == 1,
        "b_scales_inner_stride_one": (B_scales is None or B_scales.stride(-1) == 1),
        "block_scale_layout": block_scale_layout,
        "out_dtype": out_dtype,
        "out_inner_stride_one": out is None or out.stride(-1) == 1,
        "pdl_enabled": enable_pdl,
    }

    signature = _gemm_format_signature(
        A, B, A_scales, B_scales, out_dtype, quant, block_size
    )
    select_dtype = signature.storage_dtype_for("a") or A.dtype

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
