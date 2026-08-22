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

"""DeepSeek V4 dense and grouped output projection APIs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod

import torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import SelectedKernel, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@dataclass(frozen=True)
class _GroupedOutputProjectionPlan:
    kernel: SelectedKernel
    weight_preprocessor: Callable
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
    preprocess_recipe: tuple[int, int, int]
    execution_recipe: tuple[int, int, int]


def deepseek_v4_grouped_output_projection_plan(
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

    The selected implementation owns both weight-scale preprocessing and
    execution. Callers must retain the returned object without inspecting it,
    preprocess the loaded scales once, and use the same plan for execution.

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
        An opaque plan accepted by the related preprocess, execute, and warmup
        APIs.
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
    kernel = select_kernel(
        "gemm",
        "deepseek_v4_grouped_output_projection",
        signature,
        traits=traits,
        solution=solution,
    )
    spec = KernelRegistry.get().get_by_name(kernel.name)
    if spec is None or spec.weight_preprocessor is None:
        raise RuntimeError(
            f"Grouped output projection kernel {kernel.name!r} has no preprocessor"
        )

    from tokenspeed_kernel.platform import current_platform

    tma_aligned_scales = (
        spec.solution == "deep_gemm" and current_platform().is_blackwell_plus
    )
    preprocess_recipe = (1, block_n, block_k)
    execution_recipe = (1, 1, block_n) if tma_aligned_scales else preprocess_recipe
    return _GroupedOutputProjectionPlan(
        kernel=kernel,
        weight_preprocessor=spec.weight_preprocessor,
        warmup=getattr(kernel.impl, "_tokenspeed_warmup", None),
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
        preprocess_recipe=preprocess_recipe,
        execution_recipe=execution_recipe,
    )


def _require_grouped_output_projection_plan(
    plan: object,
) -> _GroupedOutputProjectionPlan:
    if not isinstance(plan, _GroupedOutputProjectionPlan):
        raise TypeError(
            "plan must be returned by deepseek_v4_grouped_output_projection_plan"
        )
    return plan


def deepseek_v4_grouped_output_projection_process_weights(
    plan: object,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Prepare grouped projection scales using the implementation pinned by plan.

    Args:
        plan: Opaque grouped output projection plan.
        weight: Loaded FP8 weight in flattened ``[groups * output_dim, input_dim]``
            layout.
        weight_scale: Loaded canonical block scales.

    Returns:
        The scales in the selected implementation's persistent layout.
    """
    typed_plan = _require_grouped_output_projection_plan(plan)
    expected_weight_shape = (
        typed_plan.num_groups * typed_plan.output_dim,
        typed_plan.heads_per_group * typed_plan.head_dim,
    )
    if tuple(weight.shape) != expected_weight_shape:
        raise ValueError(
            "grouped output projection weight shape mismatch: "
            f"expected {expected_weight_shape}, got {tuple(weight.shape)}"
        )
    return typed_plan.weight_preprocessor(
        weight=weight,
        weight_scale=weight_scale,
        num_groups=typed_plan.num_groups,
        output_dim=typed_plan.output_dim,
        input_dim=expected_weight_shape[1],
        block_size=typed_plan.block_size,
        recipe=typed_plan.preprocess_recipe,
    )


def deepseek_v4_grouped_output_projection(
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
        "deepseek_v4_grouped_output_projection",
        kernel.name,
        attention.dtype,
        shape_params,
    )
    with kernel_scope(
        "gemm",
        "deepseek_v4_grouped_output_projection",
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


def deepseek_v4_grouped_output_projection_warmup(
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


def deepseek_v4_grouped_output_projection_warmup_model(
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
        plan = getattr(module, "_deepseek_v4_grouped_output_projection_plan", None)
        if plan is None:
            continue
        typed_plan = _require_grouped_output_projection_plan(plan)
        if typed_plan in seen:
            continue
        seen.add(typed_plan)
        deepseek_v4_grouped_output_projection_warmup(
            typed_plan,
            module.weight,
            module.weight_scale_inv,
            max_tokens,
        )


def deepseek_v4_linear_fp32(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    enable_pdl: bool = False,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Project DeepSeek V4 hidden states and return FP32 output.

    Args:
        hidden_states: Floating-point activations with trailing dimension K.
        weight: Floating-point row-major weight shaped [N, K].
        enable_pdl: Request Programmatic Dependent Launch when supported.
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

    traits = {
        "hidden_rank": hidden_states.ndim,
        "weight_rank": weight.ndim,
        "has_tokens": hidden_states.numel() > 0,
        "k_match": True,
    }
    signature = format_signature(
        hidden_states=dense_tensor_format(hidden_states.dtype),
        weight=dense_tensor_format(weight.dtype),
    )
    kernel = select_kernel(
        "gemm",
        "deepseek_v4_linear_fp32",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    k = int(weight.shape[1])
    shape_params = {
        "M": int(prod(hidden_states.shape[:-1])),
        "N": int(weight.shape[0]),
        "K": k,
        "enable_pdl": bool(enable_pdl),
    }
    ShapeCapture.get().record(
        "gemm",
        "deepseek_v4_linear_fp32",
        kernel.name,
        hidden_states.dtype,
        shape_params,
    )
    with kernel_scope(
        "gemm",
        "deepseek_v4_linear_fp32",
        hidden_states.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(hidden_states, weight, enable_pdl=enable_pdl)


__all__ = [
    "deepseek_v4_grouped_output_projection",
    "deepseek_v4_grouped_output_projection_plan",
    "deepseek_v4_grouped_output_projection_process_weights",
    "deepseek_v4_grouped_output_projection_warmup",
    "deepseek_v4_grouped_output_projection_warmup_model",
    "deepseek_v4_linear_fp32",
]
