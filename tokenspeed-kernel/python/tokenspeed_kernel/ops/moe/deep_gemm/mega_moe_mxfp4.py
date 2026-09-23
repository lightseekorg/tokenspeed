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

"""DeepGEMM implementation of the DeepSeek V4/V4.1 MegaMoE boundary."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
    pdl_enabled,
    prepare_cuda_toolkit_env,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

platform = current_platform()
logger = logging.getLogger(__name__)

if platform.is_blackwell:
    prepare_cuda_toolkit_env()
    from deep_gemm import (
        fp8_fp4_mega_moe,
        get_pdl,
        get_symm_buffer_for_mega_moe,
        set_pdl,
    )
    from tokenspeed_kernel.ops.moe.deep_gemm._triton.mega_moe_stage import (
        stage_mxfp4_mega_moe_inputs,
    )


_MXFP4_BLOCK_SIZE = 32
_DISABLE_WARMUP_ENV = "TOKENSPEED_DISABLE_MEGA_MOE_WARMUP"
_symm_buffer_cache: dict[tuple[int, int, int, int, int, int, int], object] = {}
_warmed_configs: set[
    tuple[int, torch.device, int, int, int, int, int, float | None, bool]
] = set()


@dataclass(frozen=True)
class _DeepGemmMegaMoEState:
    l1_weights: tuple[torch.Tensor, torch.Tensor]
    l2_weights: tuple[torch.Tensor, torch.Tensor]
    device: torch.device


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


def _interleave_gate_up_(weight: torch.Tensor, granularity: int) -> torch.Tensor:
    """Interleave gate/up blocks in place with one-expert scratch space."""
    squeeze_group_dim = weight.ndim == 2
    if squeeze_group_dim:
        weight = weight.unsqueeze(0)
    groups, rows, *rest = weight.shape
    half = rows // 2
    blocks = half // granularity
    for group in range(groups):
        source = weight[group].clone().view(2, blocks, granularity, *rest)
        weight[group].view(blocks, 2, granularity, *rest).copy_(source.transpose(0, 1))
    return weight.squeeze(0) if squeeze_group_dim else weight


def _pack_ue8m0_scale_(scale: torch.Tensor) -> torch.Tensor:
    """Pack UE8M0 bytes into DeepGEMM's layout with one-expert scratch."""
    squeeze_group_dim = scale.ndim == 2
    if squeeze_group_dim:
        scale = scale.unsqueeze(0)
    groups, rows, columns = scale.shape
    if columns % 4:
        raise ValueError("MegaMoE UE8M0 scale columns must be divisible by four")
    packed_columns = columns // 4
    for group in range(groups):
        source = scale[group].clone().view(rows, packed_columns, 4)
        scale[group].view(packed_columns, rows, 4).copy_(source.permute(1, 0, 2))
    packed = scale.view(torch.int32)
    output = torch.as_strided(
        packed,
        size=(groups, rows, packed_columns),
        stride=(rows * packed_columns, 1, rows),
    )
    return output.squeeze(0) if squeeze_group_dim else output


def _reorder_scale_rows_(
    scale: torch.Tensor, interleave_gate_up: bool, granularity: int
) -> torch.Tensor:
    """Apply MegaMoE gate/up and UTCCP row permutations in place."""
    grouped = scale if scale.ndim == 3 else scale.unsqueeze(0)
    groups, rows, packed_columns = grouped.shape
    if rows % 128:
        raise ValueError("MegaMoE scale rows must be divisible by 128")
    for group in range(groups):
        source = grouped[group].contiguous()
        if interleave_gate_up:
            half = rows // 2
            blocks = half // granularity
            source = source.view(2, blocks, granularity, packed_columns)
            source = source.transpose(0, 1).reshape(rows, packed_columns)
        source = source.view(-1, 4, 32, packed_columns)
        source = source.transpose(1, 2).reshape(rows, packed_columns)
        grouped[group].copy_(source)
    return scale


def deep_gemm_mxfp4_mega_moe_process_weights(
    plan: dict,
    w: torch.nn.Module,
) -> object:
    w13_weight = w.w13_weight.data
    w13_weight_scale = w.w13_weight_scale.data
    w2_weight = w.w2_weight.data
    w2_weight_scale = w.w2_weight_scale.data
    num_local_experts = w.num_local_experts
    hidden_size = w.hidden_size
    intermediate_size = w.intermediate_size
    max_num_tokens = plan.get("persistent_max_num_tokens_per_gpu")
    if max_num_tokens is None or max_num_tokens <= 0:
        raise ValueError("DeepGEMM MegaMoE requires persistent_max_num_tokens_per_gpu")
    if w.top_k > w.num_experts:
        raise ValueError("top_k cannot exceed num_experts")
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

    w13_weight = _interleave_gate_up_(w13_weight.view(torch.int8), 8)
    w13_scale = _pack_ue8m0_scale_(w13_weight_scale)
    w13_scale = _reorder_scale_rows_(w13_scale, True, 8)
    w2_scale = _pack_ue8m0_scale_(w2_weight_scale)
    w2_scale = _reorder_scale_rows_(w2_scale, False, 8)

    state = _DeepGemmMegaMoEState(
        l1_weights=(w13_weight, w13_scale),
        l2_weights=(w2_weight.view(torch.int8), w2_scale),
        device=w13_weight.device,
    )
    w._moe_backend_state = state
    return state


def _resolve_process_group(process_group: object | None) -> object:
    if process_group is not None:
        return process_group
    if not dist.is_initialized():
        raise RuntimeError(
            "DeepGEMM MegaMoE requires an initialized process group or an "
            "explicit process_group in moe_plan"
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


def _warmup_m_values(max_tokens: int) -> list[int]:
    """Return token counts covering every DeepGEMM tile reachable at runtime."""
    dense = min(max_tokens, 2048)
    values: set[int] = set(range(1, dense + 1))
    values.update(range(dense, max_tokens + 1, 16))
    values.add(max_tokens)
    return sorted(values)


def _warmup_mega_moe_jit(
    *,
    num_experts: int,
    max_num_tokens: int,
    top_k: int,
    hidden_size: int,
    device: torch.device,
    transformed_l1_weights: tuple[torch.Tensor, torch.Tensor],
    transformed_l2_weights: tuple[torch.Tensor, torch.Tensor],
    symm_buffer: object,
    activation_clamp: float | None,
    fast_math: bool,
) -> None:
    """Pre-compile MegaMoE kernel tiles using the initialized model state."""
    token_counts = _warmup_m_values(max_num_tokens)
    logger.info(
        f"Warming up mega_moe JIT: {len(token_counts):d} token counts up to "
        f"{max_num_tokens:d}",
    )

    for num_tokens in token_counts:
        hidden_states = torch.randn(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        topk_ids = torch.randint(
            0,
            num_experts,
            (num_tokens, top_k),
            dtype=torch.int32,
            device=device,
        )
        topk_weights = torch.full(
            (num_tokens, top_k),
            1.0 / top_k,
            dtype=torch.float32,
            device=device,
        )

        output = torch.empty_like(hidden_states)
        symm_buffer.x[:num_tokens].copy_(hidden_states.to(torch.float8_e4m3fn))
        symm_buffer.x_sf[:num_tokens].fill_(1.0)
        symm_buffer.topk_idx[:num_tokens].copy_(topk_ids)
        symm_buffer.topk_weights[:num_tokens].copy_(topk_weights)
        fp8_fp4_mega_moe(
            output,
            transformed_l1_weights,
            transformed_l2_weights,
            symm_buffer,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )

    torch.cuda.synchronize()


def warmup_deep_gemm_mxfp4_mega_moe(plan: dict, w: torch.nn.Module) -> None:
    if get_pdl() != pdl_enabled():
        set_pdl(pdl_enabled())
    if os.environ.get(_DISABLE_WARMUP_ENV) == "1":
        return
    state = w._moe_backend_state
    if not isinstance(state, _DeepGemmMegaMoEState):
        raise TypeError("invalid DeepGEMM MegaMoE state")
    process_group = plan.get("process_group")
    num_experts = w.num_experts
    top_k = w.top_k
    hidden_size = w.hidden_size
    intermediate_size = w.intermediate_size
    max_num_tokens = plan["persistent_max_num_tokens_per_gpu"]
    activation_clamp = None if w.swiglu_arg is None else w.swiglu_arg.limit
    fast_math = plan["fast_math"]
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
        fast_math,
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
    _warmup_mega_moe_jit(
        num_experts=num_experts,
        max_num_tokens=max_num_tokens,
        top_k=top_k,
        hidden_size=hidden_size,
        device=state.device,
        transformed_l1_weights=state.l1_weights,
        transformed_l2_weights=state.l2_weights,
        symm_buffer=symm_buffer,
        activation_clamp=activation_clamp,
        fast_math=fast_math,
    )
    _warmed_configs.add(warmup_key)


if platform.is_blackwell:

    @register_kernel(
        "moe",
        "apply",
        name="deep_gemm_mxfp4_mega_moe_apply",
        solution="mega_moe",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 9),
            required_features=frozenset({"tensor_core:f4"}),
        ),
        signatures=frozenset(
            {
                format_signature(
                    x=dense_tensor_format(torch.bfloat16),
                )
            }
        ),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"swiglu"}),
            "routing_mode": frozenset({"precomputed_topk"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "persistent_workspace": frozenset({True}),
            "ispp_alignment": frozenset({128}),
            "hidden_alignment": frozenset({128}),
            "swiglu_form": frozenset({"standard"}),
            "activation_clamped": frozenset({False, True}),
            "internal_activation_dtype": frozenset({"input"}),
            "supports_bias": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
        weight_preprocessor=deep_gemm_mxfp4_mega_moe_process_weights,
    )
    def deep_gemm_mxfp4_mega_moe_apply(
        plan: dict,
        x: torch.Tensor,
        w: torch.nn.Module,
        router_logits: torch.Tensor | None,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_tokens_global: int | None,
        max_num_tokens_per_gpu: int | None,
        do_finalize: bool,
        enable_pdl: bool,
    ) -> torch.Tensor:
        if not do_finalize:
            raise ValueError("DeepGEMM MegaMoE requires complete finalization")
        if not isinstance(x, torch.Tensor):
            raise TypeError("DeepGEMM MegaMoE requires dense input activations")
        state = w._moe_backend_state
        if not isinstance(state, _DeepGemmMegaMoEState):
            raise TypeError("invalid DeepGEMM MegaMoE state")
        if x.device != state.device:
            raise ValueError("MegaMoE inputs and processed weights must share a device")
        if x.ndim != 2 or x.shape[1] != w.hidden_size:
            raise ValueError(
                f"DeepGEMM MegaMoE input must have shape [tokens, {w.hidden_size}]"
            )
        expected_routing_shape = (x.shape[0], w.top_k)
        if topk_weights is None or tuple(topk_weights.shape) != expected_routing_shape:
            raise ValueError(f"topk_weights must have shape {expected_routing_shape}")
        if topk_ids is None or tuple(topk_ids.shape) != expected_routing_shape:
            raise ValueError(f"topk_ids must have shape {expected_routing_shape}")
        max_num_tokens = plan["persistent_max_num_tokens_per_gpu"]
        if x.shape[0] > max_num_tokens:
            raise ValueError(
                f"DeepGEMM MegaMoE got {x.shape[0]} tokens, but its symmetric "
                f"buffer was sized for {max_num_tokens}"
            )
        if get_pdl() != enable_pdl:
            set_pdl(enable_pdl)
        symm_buffer = _get_symm_buffer(
            state=state,
            process_group=plan.get("process_group"),
            num_experts=w.num_experts,
            top_k=w.top_k,
            hidden_size=w.hidden_size,
            intermediate_size=w.intermediate_size,
            max_num_tokens=max_num_tokens,
        )
        num_tokens = x.shape[0]
        topk_ids = topk_ids.to(torch.int64)
        stage_mxfp4_mega_moe_inputs(
            x,
            topk_weights,
            topk_ids,
            symm_buffer.x[:num_tokens],
            symm_buffer.x_sf[:num_tokens],
            symm_buffer.topk_idx[:num_tokens],
            symm_buffer.topk_weights[:num_tokens],
        )
        output = torch.empty_like(x, dtype=torch.bfloat16)
        fp8_fp4_mega_moe(
            output,
            state.l1_weights,
            state.l2_weights,
            symm_buffer,
            activation_clamp=(None if w.swiglu_arg is None else w.swiglu_arg.limit),
            fast_math=plan["fast_math"],
        )
        return output

    deep_gemm_mxfp4_mega_moe_apply._tokenspeed_warmup = (  # type: ignore[attr-defined]
        warmup_deep_gemm_mxfp4_mega_moe
    )
