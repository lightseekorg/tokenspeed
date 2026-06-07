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

import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.activation.flashinfer import (
    silu_and_mul as flashinfer_silu_and_mul,
)
from tokenspeed_kernel.ops.moe.flashinfer import moe_wna16_marlin_gemm
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend
from tokenspeed.runtime.layers.moe.backends.triton_config import (
    try_get_optimal_moe_config,
)
from tokenspeed.runtime.layers.moe.backends.wna16.weights import attach_marlin_weights
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.quantization import CompressedTensorsConfig


def get_scalar_type(num_bits: int, has_zp: bool):
    from tokenspeed.runtime.layers.quantization.compressed_tensors.scalar_type import (
        scalar_types,
    )

    if has_zp:
        assert num_bits == 4
        return scalar_types.uint4
    return scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128


def _check_shape(input: torch.Tensor, output: torch.Tensor) -> None:
    assert input.ndim == output.ndim, f"{input.ndim} != {output.ndim}"
    assert (
        input.shape[:-1] == output.shape[:-1]
    ), f"{input.shape[:-1]} != {output.shape[:-1]}"
    assert (
        input.shape[-1] == 2 * output.shape[-1]
    ), f"{input.shape[-1]} != {2 * output.shape[-1]}"


def silu_and_mul(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    if input.shape[-1] * input.dtype.itemsize % 16 != 0:
        raise ValueError("The pointers must be multiple of 16 bytes.")
    if out is not None:
        _check_shape(input, out)
    else:
        out = torch.empty(
            input.shape[:-1] + (input.shape[-1] // 2,),
            device=input.device,
            dtype=input.dtype,
        )
    flashinfer_silu_and_mul(input=input, out=out)
    return out


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    g_idx1: torch.Tensor | None = None,
    g_idx2: torch.Tensor | None = None,
    sort_indices1: torch.Tensor | None = None,
    sort_indices2: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    num_bits: int = 8,
    is_k_full: bool = True,
    inplace: bool = False,
) -> torch.Tensor:
    assert hidden_states.shape[1] == w1.shape[1] * 16, "Hidden size mismatch w1"
    assert hidden_states.shape[1] == w2.shape[2] // (
        num_bits // 2
    ), "Hidden size mismatch w2"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert num_bits in [4, 8]

    M, K = hidden_states.shape
    E = w1.shape[0]
    N = w2.shape[1] * 16
    topk = topk_ids.shape[1]

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        w2.shape,
        topk_ids.shape[1],
        None,
        is_marlin=True,
    )
    config = get_config_func(M)
    block_size_m = config["BLOCK_SIZE_M"]

    if global_num_experts == -1:
        global_num_experts = E
    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        tokenspeed_kernel.moe_dispatch(
            topk_ids,
            block_size_m,
            global_num_experts,
            dtype=torch.int32,
            expected_kernel_name="triton_moe_align_block_size",
        )
    )

    if workspace is None:
        max_workspace_size = (max(2 * N, K) // 64) * (
            sorted_token_ids.size(0) // block_size_m
        )
        device = hidden_states.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_workspace_size = min(max_workspace_size, sms * 4)
        workspace = torch.zeros(
            max_workspace_size, dtype=torch.int, device=device, requires_grad=False
        )

    scalar_type1 = get_scalar_type(num_bits, w1_zeros is not None)
    scalar_type2 = get_scalar_type(num_bits, w2_zeros is not None)

    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache13 = torch.empty(
        (M * topk_ids.shape[1] * max(2 * N, K),),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = intermediate_cache13[: M * topk_ids.shape[1] * 2 * N]
    intermediate_cache1 = intermediate_cache1.view(-1, 2 * N)
    intermediate_cache3 = intermediate_cache13[: M * topk_ids.shape[1] * K]
    intermediate_cache3 = intermediate_cache3.view(-1, K)

    use_atomic_add = (
        hidden_states.dtype == torch.half or current_platform().arch_version.major >= 9
    )

    intermediate_cache1 = moe_wna16_marlin_gemm(
        hidden_states,
        intermediate_cache1,
        w1,
        w1_scale,
        w1_zeros,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=False,
        is_ep=expert_map is not None,
        b_q_type_id=scalar_type1.id,
        size_m=M,
        size_n=2 * N,
        size_k=K,
        is_full_k=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )

    silu_and_mul(intermediate_cache1.view(-1, 2 * N), intermediate_cache2)

    if expert_map is not None:
        intermediate_cache3.zero_()

    intermediate_cache3 = moe_wna16_marlin_gemm(
        intermediate_cache2,
        intermediate_cache3,
        w2,
        w2_scale,
        w2_zeros,
        g_idx2,
        sort_indices2,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=1,
        mul_topk_weights=True,
        is_ep=expert_map is not None,
        b_q_type_id=scalar_type2.id,
        size_m=M * topk,
        size_n=K,
        size_k=N,
        is_full_k=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    ).view(-1, topk, K)

    output = hidden_states if inplace else torch.empty_like(hidden_states)
    return torch.sum(
        intermediate_cache3.view(*intermediate_cache3.shape), dim=1, out=output
    )


def finalize_marlin_weights(backend, layer: nn.Module) -> None:
    from tokenspeed.runtime.layers.quantization.compressed_tensors.gptq_marlin_moe import (
        gptq_marlin_moe_repack,
        marlin_moe_permute_scales,
    )
    from tokenspeed.runtime.layers.quantization.utils import replace_parameter

    num_experts = layer.w13_weight_g_idx.shape[0]
    device = layer.w13_weight_g_idx.device

    # when running models with grouped act order,
    # resort to g_idx values provided in checkpoint
    if backend._actorder == "group":
        w13_g_idx_sort_indices = torch.empty_like(layer.w13_weight_g_idx)
        w2_g_idx_sort_indices = torch.empty_like(layer.w2_weight_g_idx)
        w13_sorted_g_idx = torch.empty_like(layer.w13_weight_g_idx)
        w2_sorted_g_idx = torch.empty_like(layer.w2_weight_g_idx)

        for e_idx in range(num_experts):
            w13_g_idx_sort_indices[e_idx] = torch.argsort(
                layer.w13_weight_g_idx[e_idx]
            ).to(torch.int32)
            w2_g_idx_sort_indices[e_idx] = torch.argsort(
                layer.w2_weight_g_idx[e_idx]
            ).to(torch.int32)
            w13_sorted_g_idx[e_idx] = layer.w13_weight_g_idx[e_idx][
                w13_g_idx_sort_indices[e_idx]
            ]
            w2_sorted_g_idx[e_idx] = layer.w2_weight_g_idx[e_idx][
                w2_g_idx_sort_indices[e_idx]
            ]

        replace_parameter(layer, "w13_weight_g_idx", w13_sorted_g_idx)
        replace_parameter(layer, "w2_weight_g_idx", w2_sorted_g_idx)
        replace_parameter(layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)
    else:
        layer.w13_weight_g_idx = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        layer.w2_weight_g_idx = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        layer.w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )
        layer.w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty((num_experts, 0), dtype=torch.int32, device=device),
            requires_grad=False,
        )

    marlin_w13_qweight = gptq_marlin_moe_repack(
        layer.w13_weight_packed,
        layer.w13_g_idx_sort_indices,
        layer.w13_weight_packed.shape[1] * backend._packed_factor,
        layer.w13_weight_packed.shape[2],
        backend._num_bits,
    )
    replace_parameter(layer, "w13_weight_packed", marlin_w13_qweight)

    marlin_w2_qweight = gptq_marlin_moe_repack(
        layer.w2_weight_packed,
        layer.w2_g_idx_sort_indices,
        layer.w2_weight_packed.shape[1] * backend._packed_factor,
        layer.w2_weight_packed.shape[2],
        backend._num_bits,
    )
    replace_parameter(layer, "w2_weight_packed", marlin_w2_qweight)

    # The current path does not use the latest external kernel operators, so scale
    # flipping is skipped here.
    # Repack scales
    marlin_w13_scales = marlin_moe_permute_scales(
        layer.w13_weight_scale,
        layer.w13_weight_packed.shape[2],
        layer.w13_weight_scale.shape[2],
        backend._group_size,
    )
    replace_parameter(layer, "w13_weight_scale", marlin_w13_scales)

    marlin_w2_scales = marlin_moe_permute_scales(
        layer.w2_weight_scale,
        layer.w2_weight_scale.shape[1]
        * (
            backend._group_size if backend._group_size != -1 else backend._packed_factor
        ),
        layer.w2_weight_scale.shape[2],
        backend._group_size,
    )
    replace_parameter(layer, "w2_weight_scale", marlin_w2_scales)


def marlin_forward(backend, layer: nn.Module, hidden_states: torch.Tensor, topk_output):
    return fused_marlin_moe(
        hidden_states,
        layer.w13_weight_packed,
        layer.w2_weight_packed,
        layer.w13_weight_scale,
        layer.w2_weight_scale,
        topk_output.topk_weights,
        topk_output.topk_ids,
        g_idx1=layer.w13_weight_g_idx,
        g_idx2=layer.w2_weight_g_idx,
        sort_indices1=layer.w13_g_idx_sort_indices,
        sort_indices2=layer.w2_g_idx_sort_indices,
        num_bits=backend._num_bits,
        is_k_full=backend._is_k_full,
    )


class Wna16MarlinBackend(MoEBackend):
    supported_arches = frozenset({"any"})

    def __init__(
        self,
        key,
        spec: MoELayerSpec,
        quant_config: CompressedTensorsConfig,
        routing_config: dict | None = None,
    ):
        del routing_config
        self.key = key
        self.spec = spec
        self.quant_config = quant_config

        # Current limitation: refactor this to use schemes as other kernels are
        # supported and check if the layer is being ignored.
        config = quant_config.target_scheme_map["Linear"].get("weights")
        self._num_bits = config.num_bits
        self._packed_factor = 32 // config.num_bits
        self._strategy = config.strategy
        self._group_size = config.group_size
        self._actorder = config.actorder
        assert config.symmetric, "Only symmetric quantization is supported for MoE"

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        if not isinstance(quant_config, CompressedTensorsConfig):
            return False
        weight_quant = quant_config.target_scheme_map["Linear"].get("weights")
        input_quant = quant_config.target_scheme_map["Linear"].get("input_activations")
        return (
            current_platform().is_nvidia
            and spec.activation in {"silu", "swiglu"}
            and quant_config._is_wNa16_group_channel(weight_quant, input_quant)
        )

    def create_layer_weights(
        self, layer: nn.Module, *, with_bias: bool = False
    ) -> None:
        del with_bias
        attach_marlin_weights(self, layer)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        finalize_marlin_weights(self, layer)

    def forward(
        self,
        layer: nn.Module,
        hidden_states,
        topk_output: object,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ):
        del num_global_tokens, max_num_tokens_per_gpu
        return marlin_forward(self, layer, hidden_states, topk_output)

    @property
    def apply_routed_scaling_factor_on_output(self) -> bool:
        return False


__all__ = ["Wna16MarlinBackend"]
