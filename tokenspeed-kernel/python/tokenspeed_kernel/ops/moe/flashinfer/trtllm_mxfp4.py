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

import torch
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signature, format_signatures

platform = current_platform()
next_power_of_2 = lambda value: 1 if value <= 1 else 1 << (value - 1).bit_length()

_permute_indices_cache: dict[tuple[str, torch.Size], torch.Tensor] = {}
_permute_indices_device_cache: dict[
    tuple[str, tuple[int, ...], int, int, str, int], torch.Tensor
] = {}


def _float8_view_dtype(x: torch.Tensor) -> torch.dtype | None:
    float8_dtypes = tuple(
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
            getattr(torch, "float8_e8m0fnu", None),
        )
        if dtype is not None
    )
    return x.dtype if x.dtype in float8_dtypes else None


def _pair_swap_rows(x: torch.Tensor, dim: int = -2) -> torch.Tensor:
    view_dtype = _float8_view_dtype(x)
    if view_dtype is not None:
        x = x.view(torch.uint8)
    if dim < 0:
        dim += x.dim()
    shape = list(x.shape)
    if shape[dim] % 2 != 0:
        raise ValueError(f"expected even size in dim {dim}, got {shape[dim]}")
    new_shape = shape[:dim] + [shape[dim] // 2, 2] + shape[dim + 1 :]
    out = x.reshape(new_shape).flip(dim + 1).reshape(shape).contiguous()
    return out.view(view_dtype) if view_dtype is not None else out


def _reorder_w1w3_to_w3w1(x: torch.Tensor, dim: int = -2) -> torch.Tensor:
    view_dtype = _float8_view_dtype(x)
    if view_dtype is not None:
        x = x.view(torch.uint8)
    if dim < 0:
        dim += x.dim()
    size = x.shape[dim]
    if size % 2 != 0:
        raise ValueError(f"expected even size in dim {dim}, got {size}")
    first, second = x.split(size // 2, dim=dim)
    out = torch.cat([second, first], dim=dim).contiguous()
    return out.view(view_dtype) if view_dtype is not None else out


if platform.is_nvidia:
    from flashinfer import (
        mxfp8_quantize,
        nvfp4_block_scale_interleave,
        trtllm_fp4_block_scale_moe,
    )
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices as maybe_get_cached_w3_w1_permute_indices,
    )
    from flashinfer.fused_moe.core import (
        get_w2_permute_indices_with_cache,
    )

    def _get_device_permute_indices(
        x: torch.Tensor,
        epilogue_tile_m: int,
        num_elts_per_sf: int | None = None,
        *,
        kind: str = "w2",
    ) -> torch.Tensor:
        extra_args = (
            {} if num_elts_per_sf is None else {"num_elts_per_sf": num_elts_per_sf}
        )
        if kind == "w13":
            permute_indices = maybe_get_cached_w3_w1_permute_indices(
                _permute_indices_cache,
                x,
                epilogue_tile_m,
                **extra_args,
            )
        elif kind == "w2":
            permute_indices = get_w2_permute_indices_with_cache(
                _permute_indices_cache,
                x,
                epilogue_tile_m,
                **extra_args,
            )
        else:
            raise ValueError(f"unknown FlashInfer MXFP4 permute kind: {kind}")

        device_index = -1 if x.device.index is None else x.device.index
        num_elts_per_sf_key = -1 if num_elts_per_sf is None else num_elts_per_sf
        cache_key = (
            kind,
            tuple(x.shape),
            epilogue_tile_m,
            num_elts_per_sf_key,
            x.device.type,
            device_index,
        )
        cached_device_indices = _permute_indices_device_cache.get(cache_key)
        if cached_device_indices is None:
            cached_device_indices = permute_indices.to(x.device)
            _permute_indices_device_cache[cache_key] = cached_device_indices
        return cached_device_indices

    def _param_like_weight(
        w: torch.nn.Module,
        value: float | None,
    ) -> torch.nn.Parameter | None:
        if value is None:
            return None
        return torch.nn.Parameter(
            torch.full(
                (w.w13_weight.shape[0],),
                float(value),
                dtype=torch.float32,
                device=w.w13_weight.device,
            ),
            requires_grad=False,
        )

    def _routing_value(w: torch.nn.Module, name: str, default):
        routing_config = getattr(w, "routing_config", {})
        if not isinstance(routing_config, dict):
            routing_config = {}
        return (
            routing_config[name]
            if name in routing_config
            else getattr(w, name, default)
        )

    @register_kernel(
        "moe",
        "process_weights",
        name="flashinfer_trtllm_mxfp4_moe_process_weights",
        solution="flashinfer_trtllm",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=frozenset({format_signature()}),
        traits={"weight_dtype": frozenset({"mxfp4"})},
        priority=Priority.SPECIALIZED,
    )
    def flashinfer_trtllm_mxfp4_moe_process_weights(plan: dict, w: torch.nn.Module):
        sf_block_size = 32
        num_experts = w.w13_weight.shape[0]
        ispp_padded = w.w13_weight.shape[1] // 2
        hidden_padded = w.w2_weight.shape[1]

        swiglu_arg = getattr(w, "swiglu_arg", None)
        if swiglu_arg is None:
            alpha = 1.702
            limit = 7.0
            beta = 1.0
        else:
            alpha = swiglu_arg.alpha
            limit = swiglu_arg.limit
            beta = getattr(w, "swiglu_beta", None)
        w.gemm1_alpha = _param_like_weight(w, alpha)
        w.gemm1_beta = _param_like_weight(w, beta)
        w.gemm1_clamp_limit = _param_like_weight(w, limit)

        w13_weight_scale = w.w13_weight_scale.data
        w2_weight_scale = w.w2_weight_scale.data
        w13_weight = w.w13_weight.data
        w2_weight = w.w2_weight.data
        has_bias = hasattr(w, "w13_weight_bias") and hasattr(w, "w2_weight_bias")
        w13_bias = w.w13_weight_bias.data.to(torch.float32) if has_bias else None
        w2_bias = w.w2_weight_bias.data.to(torch.float32) if has_bias else None

        w13_layout = getattr(w, "w13_input_layout", "concatenated")
        if w13_layout == "interleaved":
            w13_weight_scale = _pair_swap_rows(w13_weight_scale, -2)
            w13_weight = _pair_swap_rows(w13_weight, -2)
            if w13_bias is not None:
                w13_bias = _pair_swap_rows(w13_bias, -1)
            w13_permute_kind = "w2"
        elif w13_layout == "concatenated":
            w13_weight_scale = _reorder_w1w3_to_w3w1(w13_weight_scale, -2)
            w13_weight = _reorder_w1w3_to_w3w1(w13_weight, -2)
            if w13_bias is not None:
                w13_bias = _reorder_w1w3_to_w3w1(w13_bias, -1)
            w13_permute_kind = "w13"
        else:
            raise ValueError(f"unknown w13_input_layout: {w13_layout!r}")

        epilogue_tile_m = 128
        w13_weight_perm = _get_device_permute_indices(
            w13_weight[0].view(torch.uint8), epilogue_tile_m, kind=w13_permute_kind
        )
        w13_scale_perm = _get_device_permute_indices(
            w13_weight_scale[0].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
            kind=w13_permute_kind,
        )
        w2_weight_perm = _get_device_permute_indices(
            w2_weight[0].view(torch.uint8), epilogue_tile_m, kind="w2"
        )
        w2_scale_perm = _get_device_permute_indices(
            w2_weight_scale[0].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
            kind="w2",
        )
        if has_bias:
            w13_bias_perm = _get_device_permute_indices(
                w13_bias[0].reshape(-1, 1), epilogue_tile_m
            )
            w2_bias_perm = _get_device_permute_indices(
                w2_bias[0].reshape(-1, 1), epilogue_tile_m
            )

        gemm1_weights_shuffled = []
        gemm1_scales_shuffled = []
        gemm2_weights_shuffled = []
        gemm2_scales_shuffled = []
        gemm1_bias_shuffled = []
        gemm2_bias_shuffled = []
        for idx in range(num_experts):
            gemm1_weights_shuffled.append(
                w13_weight[idx].view(torch.uint8)[w13_weight_perm].contiguous()
            )
            gemm1_scales_shuffled.append(
                nvfp4_block_scale_interleave(
                    w13_weight_scale[idx].view(torch.uint8)[w13_scale_perm].contiguous()
                )
            )
            gemm2_weights_shuffled.append(
                w2_weight[idx].view(torch.uint8)[w2_weight_perm].contiguous()
            )
            gemm2_scales_shuffled.append(
                nvfp4_block_scale_interleave(
                    w2_weight_scale[idx].view(torch.uint8)[w2_scale_perm].contiguous()
                )
            )
            if has_bias:
                gemm1_bias_shuffled.append(
                    w13_bias[idx].reshape(-1, 1)[w13_bias_perm].contiguous()
                )
                gemm2_bias_shuffled.append(
                    w2_bias[idx].reshape(-1, 1)[w2_bias_perm].contiguous()
                )

        w.w13_weight = torch.nn.Parameter(
            torch.stack(gemm1_weights_shuffled), requires_grad=False
        )
        w.w13_weight_scale = torch.nn.Parameter(
            torch.stack(gemm1_scales_shuffled)
            .reshape(num_experts, 2 * ispp_padded, hidden_padded // sf_block_size)
            .view(torch.float8_e4m3fn),
            requires_grad=False,
        )
        w.w2_weight = torch.nn.Parameter(
            torch.stack(gemm2_weights_shuffled), requires_grad=False
        )
        w.w2_weight_scale = torch.nn.Parameter(
            torch.stack(gemm2_scales_shuffled)
            .reshape(num_experts, hidden_padded, ispp_padded // sf_block_size)
            .view(torch.float8_e4m3fn),
            requires_grad=False,
        )
        if has_bias:
            w.w13_weight_bias = torch.nn.Parameter(
                torch.stack(gemm1_bias_shuffled).reshape(num_experts, -1),
                requires_grad=False,
            )
            w.w2_weight_bias = torch.nn.Parameter(
                torch.stack(gemm2_bias_shuffled).reshape(num_experts, -1),
                requires_grad=False,
            )
        w.intermediate_size_per_partition = ispp_padded
        w.hidden_size_padded = hidden_padded
        w.hidden_size_original = getattr(w, "hidden_size", hidden_padded)
        w._flashinfer_trtllm_autotuned = False
        return None

    def _call_mxfp4_moe(
        w: torch.nn.Module,
        router_logits: torch.Tensor,
        x_quant: torch.Tensor,
        x_scale: torch.Tensor | None,
        output: torch.Tensor,
    ) -> torch.Tensor:
        routing_logits = router_logits.to(torch.float32)
        local_experts = getattr(w, "num_local_experts", w.w13_weight.shape[0])
        return trtllm_fp4_block_scale_moe(
            routing_logits=routing_logits,
            routing_bias=None,
            hidden_states=x_quant,
            hidden_states_scale=(
                None if x_scale is None else x_scale.view(torch.float8_e4m3fn)
            ),
            gemm1_weights=w.w13_weight,
            gemm1_weights_scale=w.w13_weight_scale.view(torch.float8_e4m3fn),
            gemm1_bias=getattr(w, "w13_weight_bias", None),
            gemm1_alpha=getattr(w, "gemm1_alpha", None),
            gemm1_beta=getattr(w, "gemm1_beta", None),
            gemm1_clamp_limit=getattr(w, "gemm1_clamp_limit", None),
            gemm2_weights=w.w2_weight,
            gemm2_weights_scale=w.w2_weight_scale.view(torch.float8_e4m3fn),
            gemm2_bias=getattr(w, "w2_weight_bias", None),
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=getattr(w, "num_experts"),
            top_k=getattr(w, "top_k"),
            n_group=None,
            topk_group=None,
            intermediate_size=getattr(w, "intermediate_size_per_partition"),
            local_expert_offset=getattr(w, "ep_rank", 0) * local_experts,
            local_num_experts=local_experts,
            routed_scaling_factor=None,
            routing_method_type=1,
            do_finalize=True,
            tune_max_num_tokens=next_power_of_2(x_quant.shape[0]),
            output=output,
        )[0]

    @register_kernel(
        "moe",
        "apply",
        name="flashinfer_trtllm_mxfp4_moe_apply",
        solution="flashinfer_trtllm",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(10, 0),
            max_arch_version=ArchVersion(10, 3),
        ),
        signatures=format_signatures(
            "x",
            "dense",
            {torch.float16, torch.bfloat16},
        ),
        traits={
            "weight_dtype": frozenset({"mxfp4"}),
            "activation": frozenset({"silu", "swiglu"}),
            "routing_mode": frozenset({"kernel_routing"}),
            "supports_deferred_finalize": frozenset({False}),
            "supports_ep": frozenset({True}),
            "supports_all_to_all_ep": frozenset({False}),
            "ispp_alignment": frozenset({1}),
            "supports_bias": frozenset({True}),
        },
        priority=Priority.SPECIALIZED,
    )
    def flashinfer_trtllm_mxfp4_moe_apply(
        plan: dict,
        x: torch.Tensor,
        w: torch.nn.Module,
        router_logits: torch.Tensor,
        topk_weights: torch.Tensor | None = None,
        topk_ids: torch.Tensor | None = None,
        num_tokens_global: int | None = None,
        max_num_tokens_per_gpu: int | None = None,
        do_finalize: bool = True,
        enable_pdl: bool = False,
    ):
        hidden_padded = getattr(w, "hidden_size_padded", w.w2_weight_scale.shape[1])
        hidden_original = getattr(w, "hidden_size_original", hidden_padded)
        if x.shape[0] == 0:
            return x.new_empty(0, hidden_original)

        precision = plan.get(
            "flashinfer_trtllm_moe_precision",
            getattr(w, "flashinfer_trtllm_moe_precision", "default"),
        )
        if precision == "bf16":
            if x.dtype != torch.bfloat16:
                raise TypeError("FlashInfer MXFP4 bf16 precision requires bf16 input")
            x_quant = x
            x_scale = None
            if hidden_padded != x_quant.shape[-1]:
                x_quant = torch.nn.functional.pad(
                    x_quant,
                    (0, hidden_padded - x_quant.shape[-1]),
                    mode="constant",
                    value=0.0,
                )
        elif precision == "default":
            x_quant, x_scale = mxfp8_quantize(x, False, alignment=hidden_padded)
            x_scale = x_scale.view(torch.float8_e4m3fn).reshape(*x.shape[:-1], -1)
        else:
            raise NotImplementedError(
                f"Unknown flashinfer_trtllm_moe_precision: {precision}"
            )

        if x_quant.shape[-1] != hidden_padded:
            raise RuntimeError(
                f"expected hidden size {hidden_padded}, got {x_quant.shape[-1]}"
            )

        h_dim = (
            x_quant.shape[-1] * 2 if x_quant.dtype == torch.uint8 else x_quant.shape[-1]
        )
        output = torch.empty(
            x_quant.shape[0], h_dim, dtype=torch.bfloat16, device=x_quant.device
        )

        if not getattr(w, "_flashinfer_trtllm_autotuned", False):
            with autotune():
                _call_mxfp4_moe(w, router_logits, x_quant, x_scale, output)
            w._flashinfer_trtllm_autotuned = True

        result = _call_mxfp4_moe(w, router_logits, x_quant, x_scale, output)
        if hidden_original != hidden_padded:
            result = result[:, :hidden_original].contiguous()
        return result
