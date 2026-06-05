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
import json
import os
from functools import partial
from typing import Any

import torch
import triton.language as tl
from tokenspeed_kernel.ops.moe.triton import (
    triton_moe_align_block_size,
    triton_moe_fused_experts,
    triton_moe_sum_reduce,
)
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.layers.activation import silu_and_mul
from tokenspeed.runtime.layers.moe.backends.base import MoEBackend
from tokenspeed.runtime.layers.moe.backends.weights import (
    create_moe_weight_pair,
)
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.quantization import Fp8Config
from tokenspeed.runtime.utils import (
    get_colorful_logger,
    get_device_name,
    set_weight_attrs,
)
from tokenspeed.runtime.utils.env import envs

logger = get_colorful_logger(__name__)
_is_amd = current_platform().is_amd


def _get_config_file_name(
    E: int,
    inter_size: int,
    hidden_size: int,
    dtype: str | None,
    block_shape: list[int] | tuple[int, int] | None = None,
    down_moe: bool = False,
) -> str:
    device_name = get_device_name().replace(" ", "_")
    dtype_selector = "" if not dtype else f",dtype={dtype}"
    block_shape_selector = (
        "" if not block_shape or not all(block_shape) else f",block_shape={block_shape}"
    )
    down_moe_selector = "_down" if down_moe else ""
    return (
        f"{E=},{inter_size=},{hidden_size=},device_name={device_name}"
        f"{dtype_selector}{block_shape_selector}{down_moe_selector}.json"
    )


def _get_config_file_path(
    E: int,
    inter_size: int,
    hidden_size: int,
    dtype: str | None,
    block_n: int | None = 0,
    block_k: int | None = 0,
    down_moe: bool = False,
) -> str:
    json_file_name = _get_config_file_name(
        E,
        inter_size,
        hidden_size,
        dtype,
        [block_n, block_k],
        down_moe=down_moe,
    )
    default_config_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config_dir = envs.TOKENSPEED_MOE_CONFIG_DIR.get_set_value_or(default_config_dir)
    return os.path.join(config_dir, json_file_name)


@functools.lru_cache
def _get_moe_configs(
    E: int,
    inter_size: int,
    hidden_size: int,
    dtype: str | None,
    block_n: int | None = 0,
    block_k: int | None = 0,
    down_moe: bool = False,
) -> dict[int, Any] | None:
    config_file_path = _get_config_file_path(
        E,
        inter_size,
        hidden_size,
        dtype,
        block_n,
        block_k,
        down_moe,
    )
    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info("Using MoE kernel config from %s.", config_file_path)
            return {int(key): val for key, val in json.load(f).items()}

    logger.warning(
        (
            "Using default MoE kernel config. Performance might be sub-optimal! "
            "Config file not found at %s."
        ),
        config_file_path,
    )
    return None


def _get_default_moe_config(
    M: int,
    E: int,
    inter_size: int,
    hidden_size: int,
    topk: int,
    dtype: str | None,
    block_shape: list[int] | tuple[int, int] | None = None,
) -> dict[str, int]:
    del hidden_size, topk
    if dtype == "fp8_w8a8":
        if block_shape is None:
            config = {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 32,
                "num_warps": 8,
                "num_stages": 2 if _is_amd else 4,
            }
            if M <= E:
                config = {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 1,
                    "num_warps": 4,
                    "num_stages": 2 if _is_amd else 4,
                }
        else:
            config = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": block_shape[0],
                "BLOCK_SIZE_K": block_shape[1],
                "GROUP_SIZE_M": 32,
                "num_warps": 4,
                "num_stages": 2 if _is_amd else 3,
            }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
        if M <= E:
            config = {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
            }
    return config


def _try_get_optimal_moe_config(
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    top_k: int,
    dtype: str | None,
    M: int,
    block_shape: list[int] | tuple[int, int] | None = None,
    return_down_config: bool = False,
):
    del w1_shape
    E, hidden_size, inter_size = w2_shape
    block_n = block_shape[0] if block_shape else 0
    block_k = block_shape[1] if block_shape else 0

    configs = _get_moe_configs(
        E, inter_size, hidden_size, dtype, block_n, block_k, down_moe=False
    )
    if configs:
        config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
    else:
        config = _get_default_moe_config(
            M, E, inter_size, hidden_size, top_k, dtype, block_shape
        )

    down_config = None
    max_block_m = None
    if return_down_config:
        down_configs = _get_moe_configs(
            E, inter_size, hidden_size, dtype, block_n, block_k, down_moe=True
        )
        if down_configs:
            down_config = down_configs[
                min(down_configs.keys(), key=lambda x: abs(x - M))
            ]
            down_config = dict(**down_config)
            max_block_m = max([cfg["BLOCK_SIZE_M"] for cfg in down_configs.values()])
        else:
            down_config = _get_default_moe_config(
                M, E, inter_size, hidden_size, top_k, dtype, block_shape
            )
            max_block_m = down_config["BLOCK_SIZE_M"]

    if return_down_config:
        assert (
            down_config is None or config["BLOCK_SIZE_M"] == down_config["BLOCK_SIZE_M"]
        )
        return config, (down_config, max_block_m)
    return config


def _attach_dense_weight_pair(
    backend: MoEBackend,
    layer: nn.Module,
    *,
    with_bias: bool = False,
    params_dtype: torch.dtype,
) -> int:
    ispp = backend.spec.intermediate_size // backend.spec.tp_size
    create_moe_weight_pair(
        backend,
        layer,
        backend.spec.num_local_experts,
        backend.spec.hidden_size,
        ispp,
        params_dtype,
        with_bias=with_bias,
    )
    return ispp


def _register_block_scale_inverses(
    backend: MoEBackend,
    layer: nn.Module,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    block_shape: tuple[int, int],
) -> None:
    block_n, block_k = block_shape
    w13_weight_scale = torch.nn.Parameter(
        torch.ones(
            num_local_experts,
            2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
            (hidden_size + block_k - 1) // block_k,
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    w2_weight_scale = torch.nn.Parameter(
        torch.ones(
            num_local_experts,
            (hidden_size + block_n - 1) // block_n,
            (intermediate_size_per_partition + block_k - 1) // block_k,
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
    layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)

    weight_loader = backend._make_weight_loader()
    set_weight_attrs(w13_weight_scale, {"weight_loader": weight_loader})
    set_weight_attrs(w2_weight_scale, {"weight_loader": weight_loader})


def _build_triton_gemms(
    layer: nn.Module,
    spec: MoELayerSpec,
    *,
    use_fp8_w8a8: bool = False,
    per_channel_quant: bool = False,
    block_shape=None,
    dtype_tag: str = "bf16",
    gate_up_B_scale=None,
    down_B_scale=None,
):
    num_local_experts, intermediate_size_x2, hidden_size = layer.w13_weight.shape
    intermediate_size = intermediate_size_x2 // 2

    common = dict(
        compute_type=tl.bfloat16,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        filter_expert=True,
    )
    experts_common = dict(
        **common,
    )
    gemm = partial(triton_moe_fused_experts, **experts_common)
    gate_up_gemm = partial(
        gemm,
        A_scale=None,
        B_scale=gate_up_B_scale,
        mul_routed_weight=False,
        top_k=spec.top_k,
    )
    down_gemm = partial(
        gemm,
        A_scale=None,
        B_scale=down_B_scale,
        mul_routed_weight=True,
        top_k=1,
    )
    get_config_func = partial(
        _try_get_optimal_moe_config,
        (num_local_experts, intermediate_size * 2, hidden_size),
        (num_local_experts, hidden_size, intermediate_size),
        spec.top_k,
        dtype_tag,
        block_shape=None,
        return_down_config=True,
    )
    return gate_up_gemm, down_gemm, get_config_func


def _triton_forward(
    gate_up_gemm,
    down_gemm,
    get_config_func,
    activation: str,
    layer: nn.Module,
    hidden_states: torch.Tensor,
    topk_output: object,
) -> torch.Tensor:
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"

    topk_ids = topk_output.topk_ids
    topk_weights = topk_output.topk_weights
    ep_size = getattr(layer, "ep_size", 1)
    if ep_size > 1:
        num_local_experts_for_ep = getattr(
            layer, "num_local_experts", layer.w13_weight.shape[0]
        )
        local_expert_start = getattr(layer, "ep_rank", 0) * num_local_experts_for_ep
        local_expert_end = local_expert_start + num_local_experts_for_ep
        local_expert_mask = (topk_ids >= local_expert_start) & (
            topk_ids < local_expert_end
        )
        topk_ids = torch.where(
            local_expert_mask,
            topk_ids - local_expert_start,
            torch.zeros_like(topk_ids),
        )
        topk_weights = torch.where(
            local_expert_mask,
            topk_weights,
            torch.zeros_like(topk_weights),
        )
    m_tokens = hidden_states.shape[0]
    num_experts, intermediate_size_x2, hidden_size = layer.w13_weight.shape
    top_k = topk_ids.shape[1]
    dtype = hidden_states.dtype
    device = hidden_states.device

    config, (down_config, _max_block_m) = get_config_func(M=m_tokens)

    gate_up_moe_use_tma = config is not None and config.pop("USE_TMA", False)
    down_moe_use_tma = down_config is not None and down_config.pop("USE_TMA", False)

    sorted_token_ids, expert_ids, num_tokens_post_padded = triton_moe_align_block_size(
        topk_ids,
        config["BLOCK_SIZE_M"],
        num_experts,
    )

    max_num_active_experts = min(m_tokens * top_k, num_experts + 1)
    padded_tokens = (
        max_num_active_experts * (config["BLOCK_SIZE_M"] - 1) if down_moe_use_tma else 0
    )
    intermediate_cache1 = torch.empty(
        (m_tokens * top_k + padded_tokens, intermediate_size_x2),
        device=device,
        dtype=dtype,
    )
    intermediate_cache2 = torch.empty(
        (m_tokens * top_k + padded_tokens, intermediate_size_x2 // 2),
        device=device,
        dtype=dtype,
    )
    intermediate_cache3 = torch.empty(
        (m_tokens, top_k, hidden_size),
        device=device,
        dtype=dtype,
    )

    gate_up_gemm(
        A=hidden_states,
        B=layer.w13_weight,
        bias=None,
        C=intermediate_cache1,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        config=config,
        a_use_tma=False,
        b_use_tma=gate_up_moe_use_tma,
        c_sorted=down_moe_use_tma,
    )

    if activation == "silu":
        silu_and_mul(
            intermediate_cache1.view(-1, intermediate_size_x2),
            intermediate_cache2,
        )
    else:
        raise ValueError(f"Unsupported activation: {activation}")

    down_gemm(
        A=intermediate_cache2,
        B=layer.w2_weight,
        bias=None,
        C=intermediate_cache3,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        config=down_config,
        a_use_tma=down_moe_use_tma,
        b_use_tma=down_moe_use_tma,
    )

    out_hidden_states = torch.empty_like(hidden_states)
    routed_scaling_factor = 1.0
    triton_moe_sum_reduce(
        intermediate_cache3,
        out_hidden_states,
        routed_scaling_factor,
    )
    return out_hidden_states


class Fp8TritonBackend(MoEBackend):
    supported_arches = frozenset({"any"})

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        return (
            isinstance(quant_config, Fp8Config)
            and quant_config.weight_block_size is not None
        )

    def create_layer_weights(self, layer, *, with_bias: bool = False) -> None:
        ispp = _attach_dense_weight_pair(
            self,
            layer,
            with_bias=with_bias,
            params_dtype=torch.float8_e4m3fn,
        )
        _register_block_scale_inverses(
            self,
            layer,
            num_local_experts=self.spec.num_local_experts,
            hidden_size=self.spec.hidden_size,
            intermediate_size_per_partition=ispp,
            block_shape=self.quant_config.weight_block_size,
        )

        self._gate_up_gemm, self._down_gemm, self._get_config_func = (
            _build_triton_gemms(
                layer,
                self.spec,
                use_fp8_w8a8=True,
                block_shape=self.quant_config.weight_block_size,
                dtype_tag="fp8_w8a8",
                gate_up_B_scale=layer.w13_weight_scale_inv,
                down_B_scale=layer.w2_weight_scale_inv,
            )
        )

    def forward(
        self,
        layer,
        hidden_states,
        topk_output,
        num_global_tokens,
        max_num_tokens_per_gpu,
    ):
        del num_global_tokens, max_num_tokens_per_gpu
        return _triton_forward(
            self._gate_up_gemm,
            self._down_gemm,
            self._get_config_func,
            layer.activation,
            layer,
            hidden_states,
            topk_output,
        )


__all__ = ["Fp8TritonBackend"]
