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

"""DeepGEMM implementation of the DeepSeek V4 MegaMoE boundary."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

try:
    from tokenspeed_kernel.thirdparty.deep_gemm import (
        fp8_fp4_mega_moe,
        get_pdl,
        get_symm_buffer_for_mega_moe,
        set_pdl,
        transform_sf_into_required_layout,
        transform_weights_for_mega_moe,
        warmup_mega_moe_jit,
    )
    from tokenspeed_kernel.thirdparty.triton import stage_dsv4_mega_moe_inputs
except ImportError:  # pragma: no cover - DeepGEMM and Triton are optional
    fp8_fp4_mega_moe = None
    get_pdl = None
    get_symm_buffer_for_mega_moe = None
    set_pdl = None
    transform_sf_into_required_layout = None
    transform_weights_for_mega_moe = None
    warmup_mega_moe_jit = None
    stage_dsv4_mega_moe_inputs = None


_MXFP4_BLOCK_SIZE = 32
_DISABLE_WARMUP_ENV = "TOKENSPEED_DISABLE_MEGA_MOE_WARMUP"
_symm_buffer_cache: dict[tuple[int, int, int, int, int, int, int], object] = {}
_warmed_configs: set[
    tuple[int, torch.device, int, int, int, int, int, float | None]
] = set()


@dataclass(frozen=True)
class _DeepGemmMegaMoEState:
    l1_weights: tuple[torch.Tensor, torch.Tensor]
    l2_weights: tuple[torch.Tensor, torch.Tensor]
    device: torch.device


def _ue8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.uint8:
        return (scale.to(torch.int32) << 23).view(torch.float32)
    return scale.float()


def _expected_shapes(
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    return (
        (num_local_experts, 2 * intermediate_size, hidden_size // 2),
        (
            num_local_experts,
            2 * intermediate_size,
            hidden_size // _MXFP4_BLOCK_SIZE,
        ),
        (num_local_experts, hidden_size, intermediate_size // 2),
        (
            num_local_experts,
            hidden_size,
            intermediate_size // _MXFP4_BLOCK_SIZE,
        ),
    )


def _deep_gemm_dsv4_mega_moe_process_weights(
    *,
    w13_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> object:
    tensors = (w13_weight, w13_weight_scale, w2_weight, w2_weight_scale)
    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    expected_shapes = _expected_shapes(
        num_local_experts, hidden_size, intermediate_size
    )
    for name, tensor, expected_shape in zip(names, tensors, expected_shapes):
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"MegaMoE {name} shape mismatch: expected {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.device.type != "cuda":
            raise ValueError(f"MegaMoE {name} must be loaded on CUDA")
        if tensor.device != w13_weight.device:
            raise ValueError("MegaMoE checkpoint tensors must be on the same device")
    if w13_weight.dtype != torch.uint8 or w2_weight.dtype != torch.uint8:
        raise ValueError("MegaMoE packed checkpoint weights must have dtype uint8")

    w13_scale = transform_sf_into_required_layout(
        sf=_ue8m0_to_float(w13_weight_scale).contiguous(),
        mn=2 * intermediate_size,
        k=hidden_size,
        recipe=(1, _MXFP4_BLOCK_SIZE),
        num_groups=num_local_experts,
    )
    w2_scale = transform_sf_into_required_layout(
        sf=_ue8m0_to_float(w2_weight_scale).contiguous(),
        mn=hidden_size,
        k=intermediate_size,
        recipe=(1, _MXFP4_BLOCK_SIZE),
        num_groups=num_local_experts,
    )
    l1_weights, l2_weights = transform_weights_for_mega_moe(
        (w13_weight.view(torch.int8).contiguous(), w13_scale),
        (w2_weight.view(torch.int8).contiguous(), w2_scale),
    )
    # DeepGEMM may return the contiguous L2 input itself. Break that reference
    # so callers can release all canonical checkpoint tensors after processing.
    return _DeepGemmMegaMoEState(
        l1_weights=l1_weights,
        l2_weights=(l2_weights[0].clone(), l2_weights[1]),
        device=w13_weight.device,
    )


def _resolve_process_group(process_group: object | None) -> object:
    if process_group is not None:
        return process_group
    if not dist.is_initialized():
        raise RuntimeError(
            "DeepSeek V4 MegaMoE requires an initialized process group or an "
            "explicit process_group in dsv4_mega_moe_plan"
        )
    return dist.group.WORLD


def _get_symm_buffer(
    *,
    state: _DeepGemmMegaMoEState,
    process_group: object | None,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    max_num_tokens: int,
) -> object:
    group = _resolve_process_group(process_group)
    device_index = (
        state.device.index
        if state.device.index is not None
        else torch.cuda.current_device()
    )
    key = (
        id(group),
        device_index,
        num_experts,
        max_num_tokens,
        top_k,
        hidden_size,
        intermediate_size,
    )
    buffer = _symm_buffer_cache.get(key)
    if buffer is None:
        buffer = get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            max_num_tokens,
            top_k,
            hidden_size,
            intermediate_size,
        )
        _symm_buffer_cache[key] = buffer
    return buffer


def _warmup_deep_gemm_dsv4_mega_moe(
    *,
    state: object,
    process_group: object | None,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    max_num_tokens: int,
    activation_clamp: float | None,
) -> None:
    if get_pdl() != pdl_enabled():
        set_pdl(pdl_enabled())
    if os.environ.get(_DISABLE_WARMUP_ENV) == "1":
        return
    if not isinstance(state, _DeepGemmMegaMoEState):
        raise TypeError("invalid DeepGEMM MegaMoE state")
    group = _resolve_process_group(process_group)
    warmup_key = (
        id(group),
        state.device,
        num_experts,
        max_num_tokens,
        top_k,
        hidden_size,
        intermediate_size,
        activation_clamp,
    )
    if warmup_key in _warmed_configs:
        return
    if dist.is_initialized():
        dist.barrier(group=group)
    symm_buffer = _get_symm_buffer(
        state=state,
        process_group=group,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        max_num_tokens=max_num_tokens,
    )
    warmup_mega_moe_jit(
        num_experts=num_experts,
        max_num_tokens=max_num_tokens,
        top_k=top_k,
        hidden_size=hidden_size,
        device=state.device,
        transformed_l1_weights=state.l1_weights,
        transformed_l2_weights=state.l2_weights,
        symm_buffer=symm_buffer,
        activation_clamp=activation_clamp,
    )
    _warmed_configs.add(warmup_key)


if fp8_fp4_mega_moe is not None and stage_dsv4_mega_moe_inputs is not None:

    @register_kernel(
        "moe",
        "dsv4_mega_moe",
        name="deep_gemm_dsv4_mega_moe_sm100",
        solution="deep_gemm",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 9),
            required_features=frozenset({"tensor_core:f4"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    hidden_states=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "scale_format": frozenset({"ue8m0"}),
            "scale_block_size": frozenset({_MXFP4_BLOCK_SIZE}),
            "supports_ep": frozenset({True}),
        },
        priority=Priority.SPECIALIZED,
        tags={"throughput"},
        weight_preprocessor=_deep_gemm_dsv4_mega_moe_process_weights,
    )
    def deep_gemm_dsv4_mega_moe(
        *,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        state: object,
        process_group: object | None,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        max_num_tokens: int,
        activation_clamp: float | None,
        fast_math: bool,
    ) -> torch.Tensor:
        if get_pdl() != pdl_enabled():
            set_pdl(pdl_enabled())
        if not isinstance(state, _DeepGemmMegaMoEState):
            raise TypeError("invalid DeepGEMM MegaMoE state")
        if hidden_states.device != state.device:
            raise ValueError("MegaMoE inputs and processed weights must share a device")
        symm_buffer = _get_symm_buffer(
            state=state,
            process_group=process_group,
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            max_num_tokens=max_num_tokens,
        )
        num_tokens = hidden_states.shape[0]
        topk_ids = topk_ids.to(torch.int64)
        stage_dsv4_mega_moe_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            symm_buffer.x[:num_tokens],
            symm_buffer.x_sf[:num_tokens],
            symm_buffer.topk_idx[:num_tokens],
            symm_buffer.topk_weights[:num_tokens],
        )
        output = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        fp8_fp4_mega_moe(
            output,
            state.l1_weights,
            state.l2_weights,
            symm_buffer,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
        return output

    deep_gemm_dsv4_mega_moe._tokenspeed_warmup = (  # type: ignore[attr-defined]
        _warmup_deep_gemm_dsv4_mega_moe
    )
