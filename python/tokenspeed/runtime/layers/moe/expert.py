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


import logging
from collections.abc import Callable
from dataclasses import replace

import tokenspeed_kernel
import torch

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.activation import SwigluArg
from tokenspeed.runtime.layers.moe.topk import TopKOutput, TopKOutputFormat
from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.utils import (
    RoutingMethodType,
    get_all2all_backend,
    get_deepep_mode,
    get_moe_backend,
)
from tokenspeed.runtime.layers.moe.weights import create_layer_weights
from tokenspeed.runtime.layers.moe.weights.loaders import round_up
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.utils import (
    should_exclude_quant_module,
    should_ignore_quant_layer,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)


class MoELayer(torch.nn.Module):
    def __init__(
        self,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig,
        layer_index: int,
        prefix: str = "",
        tp_rank: int | None = None,
        tp_size: int | None = None,
        ep_rank: int | None = None,
        ep_size: int | None = None,
        zero_expert_type: str = "",
        activation: str = "silu",
        activation_situ_beta: float | None = None,
        activation_situ_linear_beta: float | None = None,
        activation_alpha=None,
        swiglu_limit=None,
        swiglu_beta: float | None = None,
        w13_input_layout: str = "concatenated",
        with_bias=False,
        routing_config: dict = {},
        routing_mode: str | None = None,
        internal_activation_dtype_override: str | None = None,
    ):
        super().__init__()
        self.layer_index = layer_index
        # Deployment-level override for the activation precision (e.g. a K3
        # w4a8 flag), forcing the value the quant_config would otherwise derive.
        self._internal_activation_dtype_override = internal_activation_dtype_override
        self.prefix = prefix
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.quant_config = quant_config
        self.ep_num_redundant_experts = global_server_args_dict[
            "ep_num_redundant_experts"
        ]
        self.zero_expert_type = zero_expert_type
        self.activation = activation
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta
        if self.activation == "situ" and (
            activation_situ_beta is None
            or activation_situ_beta <= 0
            or (
                activation_situ_linear_beta is not None
                and activation_situ_linear_beta <= 0
            )
        ):
            raise ValueError("SiTU beta values must be positive")
        self.swiglu_arg = None
        if self.activation == "swiglu":
            self.swiglu_arg = SwigluArg(alpha=activation_alpha, limit=swiglu_limit)
        # Per-model knobs the MoE backend reads in process_weights_after_loading.
        # ``swiglu_beta``: gpt-oss uses silu(α·gate)·(up + 1) and sets 1.0;
        # standard SwiGLU (e.g. deepseek-v4) leaves it None.
        # ``w13_input_layout``: "interleaved" for HF gpt-oss-style row layout
        # ([w1_0, w3_0, w1_1, w3_1, ...]); "concatenated" (default) for the
        # shared MoE checkpoint loader's [w1_all | w3_all] block layout.
        self.swiglu_beta = swiglu_beta
        if w13_input_layout not in {"interleaved", "concatenated"}:
            raise ValueError(
                f"w13_input_layout must be 'interleaved' or 'concatenated', "
                f"got {w13_input_layout!r}"
            )
        self.w13_input_layout = w13_input_layout

        if tp_rank is None:
            assert tp_size is None
            tp_rank, tp_size = 0, 1
        self.tp_rank, self.tp_size = tp_rank, tp_size
        self.moe_tp_size = self.tp_size
        if ep_rank is None:
            assert ep_size is None
            ep_rank, ep_size = 0, 1
        self.ep_rank, self.ep_size = ep_rank, ep_size

        if tp_size > 1 and ep_size > 1:
            raise ValueError("Mixed TP and EP is not supported yet.")

        if num_experts % self.ep_size:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by ep_size "
                f"({self.ep_size}) for contiguous expert ownership"
            )
        num_local_experts = num_experts // self.ep_size

        self.num_local_experts = num_local_experts
        self._spec = MoELayerSpec(
            top_k=top_k,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            prefix=prefix,
            a2a_backend=get_all2all_backend().value,
        )

        # Routing config
        self.routing_config = routing_config
        self._correction_bias = routing_config.get("correction_bias", None)
        self._routing_method_type = routing_config.get(
            "routing_method_type", RoutingMethodType.DeepSeekV3
        )
        self._routing_logits_dtype = torch.bfloat16
        if self._routing_method_type in (
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.MiniMax2,
        ):
            self._routing_logits_dtype = torch.float32
        self._n_group = routing_config.get("n_group", 0)
        self._topk_group = routing_config.get("topk_group", 0)
        self._routed_scaling_factor = routing_config.get("routed_scaling_factor", 1.0)
        self._normalize_topk_weights = routing_config.get(
            "normalize_topk_weights", True
        )

        # Quantization config. ignored_layers (compressed-tensors) keys the MoE
        # block; exclude_modules (ModelOpt) keys the fused experts.
        self._quant_kind = "unquant"
        if (
            quant_config is not None
            and not should_ignore_quant_layer(self.prefix, quant_config.ignored_layers)
            and not should_exclude_quant_module(
                f"{self.prefix}.experts", quant_config.exclude_modules
            )
        ):
            self._quant_kind = quant_config.moe_weight_dtype(self.prefix)

        fp8_scale_block_shape = None
        internal_activation_dtype = "input"
        if self._quant_kind == "fp8":
            fp8_scale_block_shape = tuple(self.quant_config.weight_block_size)
            self._apply_trtllm_ispp_padding(
                fp8_scale_block_shape[0], "FP8 block scales tile it"
            )
        if self._quant_kind == "unquant":
            # The flashinfer_trtllm unquant kernel declares
            # ispp_alignment={128} (ops/moe/flashinfer/trtllm_unquant.py);
            # without padding a misaligned intermediate size silently
            # deselects it during moe_plan and the layer falls back to the
            # triton bf16 path. The padded tail rows/columns stay zero
            # (create_dense_weight_pair zero-initializes) and contribute
            # nothing to the MoE output.
            self._apply_trtllm_ispp_padding(
                128, "the flashinfer_trtllm unquant kernel accepts it"
            )
        if self._quant_kind == "mxfp4":
            if self.quant_config.is_w4a8_fp8:
                internal_activation_dtype = "fp8"
            elif getattr(self.quant_config, "use_dynamic_mxfp4_activations", False):
                internal_activation_dtype = "mxfp4"
        if self._internal_activation_dtype_override is not None:
            internal_activation_dtype = self._internal_activation_dtype_override

        input_dtype = torch.get_default_dtype()
        if input_dtype not in {torch.float16, torch.bfloat16}:
            input_dtype = torch.float16

        deepep_group = None
        deepep_mode = None
        deepep_low_latency_max_num_tokens_per_gpu = None
        if self._spec.use_deepep:
            mapping = global_server_args_dict["mapping"]
            deepep_group = pg_manager.get_process_group(
                "nccl",
                mapping.moe.tp_ep_group,
            )
            deepep_mode = get_deepep_mode().value
            # Pin the low-latency capacity from the server arg: the DeepEP
            # buffer is allocated once, on the first forward that dispatches,
            # so sizing it from that batch would make decode depend on
            # whichever batch happened to arrive first.
            deepep_low_latency_max_num_tokens_per_gpu = global_server_args_dict[
                "low_latency_max_num_tokens_per_gpu"
            ]

        # Moe Backend plan
        moe_backend = get_moe_backend().value
        moe_backend = None if moe_backend == "auto" else moe_backend
        self.plan = tokenspeed_kernel.moe_plan(
            self._quant_kind,
            input_dtype=input_dtype,
            activation=self.activation,
            # e.g. "precomputed_topk" when fused kernels cannot reproduce the routing; None = unconstrained.
            routing_mode=routing_mode,
            a2a_backend=self._spec.a2a_backend,
            ep_size=self.ep_size,
            ispp=self.intermediate_size // self.tp_size,
            fp8_scale_block_shape=fp8_scale_block_shape,
            internal_activation_dtype=internal_activation_dtype,
            with_bias=with_bias,
            deepep_group=deepep_group,
            deepep_mode=deepep_mode,
            deepep_low_latency_max_num_tokens_per_gpu=(
                deepep_low_latency_max_num_tokens_per_gpu
            ),
            solution=moe_backend,
        )

        create_layer_weights(
            self._spec,
            self,
            self._quant_kind,
            self.quant_config,
            with_bias=with_bias,
            solution=self.plan["solution"],
        )
        self._weights_processed = False

    def _apply_trtllm_ispp_padding(self, alignment: int, reason: str) -> None:
        """Round the intermediate size up when the trtllm backend needs it.

        Args:
            alignment: Required multiple along the intermediate dimension
                (the FP8 scale block size, or the kernel's declared
                ``ispp_alignment``).
            reason: Log fragment describing why the padding is required.
        """
        if get_moe_backend().value != "flashinfer_trtllm":
            return
        ispp = self.intermediate_size // self.tp_size
        if ispp % alignment == 0:
            return
        padded = round_up(ispp, alignment)
        logger.info(
            "%s: padding MoE intermediate size per partition %d -> %d so %s",
            self.prefix,
            ispp,
            padded,
            reason,
        )
        self.intermediate_size = padded * self.tp_size
        self._spec = replace(self._spec, intermediate_size=self.intermediate_size)

    def process_weights_after_loading(self, module) -> None:
        if self._weights_processed:
            return

        tokenspeed_kernel.moe_process_weights(self.plan, module)
        self._weights_processed = True

    @property
    def support_routing(self) -> bool:
        return self.plan["support_routing"]

    @property
    def supports_precomputed_topk(self) -> bool:
        # The fallback keeps lightweight out-of-tree/mock plans compatible.
        return self.plan.get("supports_precomputed_topk", not self.support_routing)

    @property
    def topk_output_format(self):
        if self.support_routing:
            return TopKOutputFormat.BYPASSED
        return TopKOutputFormat.STANDARD

    @property
    def supports_deferred_finalize(self) -> bool:
        return self.plan["supports_deferred_finalize"]

    def forward_zero_experts(self, topk_output):
        zero_expert_limit = self.num_experts
        if self.ep_num_redundant_experts is not None:
            zero_expert_limit = zero_expert_limit - self.ep_num_redundant_experts

        normal_expert_mask = topk_output.topk_ids >= zero_expert_limit
        topk_output.topk_ids[normal_expert_mask] = -1
        if self.zero_expert_type == "copy":
            topk_output.topk_weights[normal_expert_mask] = 1.0
        if self.zero_expert_type == "drop":
            topk_output.topk_weights[normal_expert_mask] = 0.0

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        do_finalize: bool = True,
        low_latency: bool | None = None,
        overlap_fn: Callable[[], None] | None = None,
        shared_input: torch.Tensor | None = None,
        shared_weight: torch.Tensor | None = None,
        shared_out: torch.Tensor | None = None,
    ):
        """Run the planned MoE kernel over this layer's weights.

        Args:
            hidden_states: ``[tokens, hidden]`` local hidden states.
            topk_output: Routing result, or the raw logits when the kernel
                routes itself.
            num_global_tokens: Token count summed over the attention DP ranks.
            max_num_tokens_per_gpu: Largest per-GPU token count this forward.
            do_finalize: Whether the kernel must produce the finalized output.
            low_latency: For all-to-all EP plans, whether to take the
                latency-optimized dispatch/combine legs. Must be identical on
                every rank of the EP group; see
                ``moe.utils.use_deepep_low_latency``.
            overlap_fn: Optional work to run inside an all-to-all EP dispatch
                window; ignored by plans that own no dispatch legs.
        """
        if not do_finalize and not self.supports_deferred_finalize:
            raise AssertionError("MoELayer does not support do_finalize=False")

        shared_tensors = (shared_input, shared_weight, shared_out)
        if any(value is not None for value in shared_tensors) and not all(
            value is not None for value in shared_tensors
        ):
            raise ValueError(
                "joint shared projection requires input, weight, and output"
            )
        shared_kwargs = (
            {
                "shared_input": shared_input,
                "shared_weight": shared_weight,
                "shared_out": shared_out,
            }
            if all(value is not None for value in shared_tensors)
            else {}
        )

        use_kernel_routing = topk_output.format.is_bypassed() or (
            self.support_routing and not self.supports_precomputed_topk
        )
        if use_kernel_routing:
            if not self.support_routing:
                raise ValueError(
                    "selected MoE kernel does not support in-kernel routing"
                )
            return tokenspeed_kernel.moe_apply(
                self.plan,
                hidden_states,
                self,
                topk_output.router_logits,
                num_tokens_global=num_global_tokens,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu,
                do_finalize=do_finalize,
                low_latency=low_latency,
                overlap_fn=overlap_fn,
                **shared_kwargs,
            )
        if not self.supports_precomputed_topk:
            raise ValueError(
                "selected MoE kernel does not support precomputed top-k routing"
            )
        return tokenspeed_kernel.moe_apply(
            self.plan,
            hidden_states,
            self,
            topk_output.router_logits,
            topk_weights=topk_output.topk_weights,
            topk_ids=topk_output.topk_ids,
            num_tokens_global=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            do_finalize=do_finalize,
            low_latency=low_latency,
            overlap_fn=overlap_fn,
            **shared_kwargs,
        )
