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
import logging

import torch
from tokenspeed_kernel.ops.gemm import dsv4_linear_fp32
from tokenspeed_kernel.ops.moe import moe_topk
from tokenspeed_kernel.ops.moe.cuda import moe_finalize_fuse_shared
from tokenspeed_kernel.platform import pdl_enabled
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_step import get_is_capture_mode
from tokenspeed.runtime.layers.mlp import GatedMLP, gated_activation
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.topk import (
    StandardTopKOutput,
    TopK,
    TopKOutput,
    simulated_router_logits,
)
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType, get_moe_backend
from tokenspeed.runtime.layers.quantization import Fp8Config, Mxfp4Config
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.nvtx import nvtx_range

logger = logging.getLogger(__name__)


def _deepseek_v4_mega_moe_max_num_tokens() -> int:
    override = int(
        global_server_args_dict.get("deepseek_v4_mega_moe_max_num_tokens", 0) or 0
    )
    if override > 0:
        return override

    candidates = [
        global_server_args_dict.get("chunked_prefill_size", 0),
        global_server_args_dict.get("prefill_graph_max_tokens", 0),
        global_server_args_dict.get("max_cudagraph_capture_size", 0),
        global_server_args_dict.get("max_num_seqs", 0),
    ]
    return max([int(value or 0) for value in candidates] + [1])


def _random_expert_ids(
    param: torch.Tensor, generator: torch.Generator, num_experts: int
) -> None:
    """Give each token distinct experts, chosen uniformly at random."""
    scores = torch.rand(
        param.shape[0], num_experts, generator=generator, device=param.device
    )
    param.data.copy_(scores.topk(param.shape[1], dim=1).indices)


class DeepseekV4MoEGate(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_index: int,
        hash_indices_dtype: torch.dtype = torch.int32,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )
        self.is_hash_moe = layer_index < config.num_hash_layers
        if self.is_hash_moe:
            self.tid2eid = nn.Parameter(
                torch.empty(
                    config.vocab_size,
                    config.num_experts_per_tok,
                    dtype=hash_indices_dtype,
                ),
                requires_grad=False,
            )
            self.tid2eid.dummy_initializer = functools.partial(
                _random_expert_ids, num_experts=config.n_routed_experts
            )
            self.e_score_correction_bias = None
        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.register_parameter("tid2eid", None)
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.register_parameter("tid2eid", None)
            self.e_score_correction_bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return dsv4_linear_fp32(
            hidden_states,
            self.weight,
        )


class DeepseekV4TopK(TopK):
    def __init__(
        self,
        top_k: int,
        renormalize: bool,
        correction_bias: torch.Tensor | None,
        routed_scaling_factor: float,
        hash_routing: bool,
        weights_dtype: torch.dtype,
    ) -> None:
        super().__init__(
            top_k=top_k,
            renormalize=renormalize,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
        )
        self.hash_routing = hash_routing
        self.weights_dtype = weights_dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        routing_correction_bias: torch.Tensor | None = None,
        hash_indices_table: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> TopKOutput:
        correction_bias = (
            self.topk_config.correction_bias
            if routing_correction_bias is None
            else routing_correction_bias
        )
        if self.simulate_routing:
            router_logits = simulated_router_logits(router_logits)
        topk_weights, topk_ids = moe_topk(
            router_logits,
            self.topk_config.top_k,
            "sqrt_softplus",
            "hash" if self.hash_routing else "topk",
            self.topk_config.renormalize,
            self.topk_config.routed_scaling_factor,
            correction_bias=correction_bias,
            hash_indices_table=hash_indices_table,
            input_ids=input_ids,
            topk_weights_dtype=self.weights_dtype,
        )
        return StandardTopKOutput(topk_weights, topk_ids, router_logits)


def _deepseek_v4_routed_expert_quant_config(
    config: PretrainedConfig,
    quant_config: QuantizationConfig,
) -> tuple[QuantizationConfig, bool]:
    # ``quant_method=fp8`` describes the model-wide quantization, but DeepSeek
    # V4 can still serialize routed experts as packed MXFP4. Prefer the explicit
    # expert contract so packed FP4 tensors are never allocated as block FP8.
    is_block_fp8 = isinstance(quant_config, Fp8Config) and (
        getattr(config, "expert_dtype", None) != "fp4"
    )
    if is_block_fp8:
        return quant_config, True
    return (
        Mxfp4Config(
            ignored_layers=quant_config.ignored_layers,
            is_checkpoint_mxfp4_serialized=True,
        ),
        False,
    )


class DeepseekV4MoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        layer_index: int,
        prefix: str,
        aux_stream: object | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_index = layer_index
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError(
                f"Unsupported DeepSeek V4 MoE scoring: {self.scoring_func}"
            )
        moe_backend = get_moe_backend()
        self.use_mega_moe = moe_backend.is_mega_moe()
        self.use_gluon_petit = moe_backend.is_gluon_petit()
        self.owns_ep_communication = self.use_mega_moe or self.use_gluon_petit
        # Petit's VMM-backed collective and auxiliary-stream shared experts do
        # not replay safely together. Keep both branches on the main stream.
        self.stream_fork = StreamFork(None if self.use_gluon_petit else aux_stream)
        if mapping.moe.ep_size > 1:
            if global_server_args_dict.get("enable_eplb", False):
                raise ValueError(
                    "DeepSeek V4 EP does not support EPLB with precomputed routing."
                )
            if global_server_args_dict.get("init_expert_location", "trivial") not in (
                None,
                "trivial",
            ):
                raise ValueError(
                    "DeepSeek V4 EP requires trivial expert placement with "
                    "precomputed routing."
                )
        if self.use_mega_moe:
            if mapping.moe.ep_size <= 1:
                raise ValueError("DeepSeek V4 MegaMoE requires expert parallelism.")
            if mapping.moe.tp_size != 1:
                raise ValueError("DeepSeek V4 MegaMoE does not support mixed TP/EP.")
            if global_server_args_dict.get("ep_num_redundant_experts", 0):
                raise ValueError(
                    "DeepSeek V4 MegaMoE does not support redundant EP experts."
                )
            if config.n_routed_experts % mapping.moe.ep_size != 0:
                raise ValueError(
                    "DeepSeek V4 MegaMoE requires n_routed_experts divisible by "
                    "EP size."
                )
        elif mapping.moe.ep_size > 1:
            if global_server_args_dict.get("ep_num_redundant_experts", 0):
                raise ValueError(
                    "DeepSeek V4 normal EP does not support redundant experts "
                    "with precomputed routing."
                )
            if mapping.attn.tp_size not in (1, mapping.moe.tp_ep_size):
                raise ValueError(
                    "DeepSeek V4 normal EP requires attention TP size 1 or "
                    "attention TP size equal to the MoE TPxEP size."
                )
            if config.hidden_size % 256:
                raise ValueError(
                    "DeepSeek V4 normal EP requires hidden_size divisible by 256."
                )
        self.hash_indices_dtype = torch.int32
        self.gate = DeepseekV4MoEGate(
            config,
            layer_index,
            hash_indices_dtype=self.hash_indices_dtype,
        )

        if config.n_shared_experts is not None:
            # MegaMoE outputs are complete, so its shared expert stays dense and local.
            if self.use_mega_moe or self.use_gluon_petit:
                shared_tp_rank = mapping.dense.tp_rank
                shared_tp_size = mapping.dense.tp_size
                shared_tp_group = mapping.dense.tp_group
            else:
                shared_tp_rank = mapping.moe.tp_ep_rank
                shared_tp_size = mapping.moe.tp_ep_size
                shared_tp_group = mapping.moe.tp_ep_group
            self.shared_experts = GatedMLP.tensor_parallel(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size
                * config.n_shared_experts,
                activation=gated_activation(
                    config.hidden_act, getattr(config, "swiglu_limit", None)
                ),
                tp_rank=shared_tp_rank,
                tp_size=shared_tp_size,
                tp_group=shared_tp_group,
                reduce_results=False,
                fusions=(),
                quant_config=quant_config,
                prefix=add_prefix("shared_experts", prefix),
            )
        else:
            self.shared_experts = None
        # Set once MegaMoE runs the shared expert inside its own launch.
        self.shared_expert_fused = False

        if self.use_mega_moe and global_server_args_dict["moe_mxfp4_fp8_activation"]:
            raise ValueError(
                "--moe-mxfp4-fp8-activation selects the FlashInfer cutlass W4A8 "
                "MoE; it does not apply to MegaMoE"
            )
        routed_quant_config, _ = _deepseek_v4_routed_expert_quant_config(
            config, quant_config
        )
        self.experts = MoELayer(
            top_k=config.num_experts_per_tok,
            num_experts=config.n_routed_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=routed_quant_config,
            layer_index=layer_index,
            prefix=prefix,
            tp_rank=mapping.moe.tp_rank,
            tp_size=mapping.moe.tp_size,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
            activation="swiglu",
            swiglu_limit=getattr(config, "swiglu_limit", None),
            with_bias=False,
            routing_mode="precomputed_topk",
            routing_config={
                "routed_scaling_factor": 1.0,
                "normalize_topk_weights": config.norm_topk_prob,
                "correction_bias": self.gate.e_score_correction_bias,
                "routing_method_type": RoutingMethodType.Renormalize,
            },
            persistent_max_num_tokens_per_gpu=(
                _deepseek_v4_mega_moe_max_num_tokens() if self.use_mega_moe else None
            ),
        )
        self.topk = DeepseekV4TopK(
            top_k=config.num_experts_per_tok,
            renormalize=self._renormalize_routing_weights(),
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            hash_routing=self.gate.tid2eid is not None,
            weights_dtype=self.experts.topk_weights_dtype,
        )

    def process_weights_after_loading(self, module: nn.Module) -> None:
        # Loaders call this once per module. The sharded-state loader calls it
        # before copying a saved state in, which then holds the fused layout,
        # as the routed experts' processing expects too.
        self.fuse_shared_expert()

    def fuse_shared_expert(self) -> None:
        """Hand the shared expert to MegaMoE when its kernel can run it.

        The shared expert's weights are then laid out in place for MegaMoE and
        its own forward is no longer used.
        """
        shared = self.shared_experts
        if not self.use_mega_moe or shared is None or self.shared_expert_fused:
            return
        gate_up, down = shared.gate_up_proj, shared.down_proj
        if (
            gate_up.weight.dtype != torch.float8_e4m3fn
            or down.weight.dtype != torch.float8_e4m3fn
        ):
            reason = (
                f"its weights are {gate_up.weight.dtype} and {down.weight.dtype}, "
                "not torch.float8_e4m3fn"
            )
        # MegaMoE clamps every expert with the routed experts' limit.
        elif shared.act_fn.swiglu_limit != self.experts.swiglu_arg.limit:
            reason = (
                f"its SwiGLU limit {shared.act_fn.swiglu_limit} differs from the "
                f"routed experts' {self.experts.swiglu_arg.limit}"
            )
        else:
            self.experts.process_weights_after_loading(self.experts)
            self.shared_expert_fused = self.experts.fuse_shared_expert(
                gate_up.weight.data,
                gate_up.weight_scale_inv.data,
                down.weight.data,
                down.weight_scale_inv.data,
            )
            if self.shared_expert_fused:
                return
            reason = (
                "the MegaMoE kernel does not take its weight shapes "
                f"{tuple(gate_up.weight.shape)} and {tuple(down.weight.shape)}"
            )
        logger.info(
            f"Layer {self.layer_index} runs its shared expert outside MegaMoE: "
            f"{reason}"
        )

    def warmup(self) -> None:
        warmup = self.experts.plan.get("warmup")
        if warmup is not None:
            if not self.experts._weights_processed:
                raise RuntimeError("MoE weights must be processed before warmup")
            warmup(self.experts.plan, self.experts)

    def _routing_inputs(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if hidden_states.shape[0] == 0:
            router_logits = hidden_states.new_empty(
                (0, self.config.n_routed_experts), dtype=torch.float32
            )
        else:
            router_logits = self.gate(hidden_states)
        hash_indices_table = self.gate.tid2eid
        return (
            router_logits,
            self.gate.e_score_correction_bias,
            hash_indices_table,
            input_ids if hash_indices_table is not None else None,
        )

    def _renormalize_routing_weights(self) -> bool:
        return self.config.norm_topk_prob

    def _compute_topk_output(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> TopKOutput:
        router_logits, correction_bias, hash_indices_table, routing_input_ids = (
            self._routing_inputs(hidden_states, input_ids)
        )
        return self.topk(
            hidden_states,
            router_logits,
            routing_correction_bias=correction_bias,
            hash_indices_table=hash_indices_table,
            input_ids=routing_input_ids,
        )

    def _forward_routed_experts(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        do_finalize: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            do_finalize=do_finalize,
        )

    def _forward_shared_experts(
        self,
        hidden_states: torch.Tensor,
        ctx: ForwardContext | None = None,
        comm_manager: CommManager | None = None,
    ) -> torch.Tensor | None:
        if self.shared_experts is None:
            return None
        if comm_manager is not None:
            hidden_states = comm_manager.pre_dense_comm(hidden_states, ctx)
        if hidden_states.shape[0] == 0:
            return None
        with nvtx_range("moe_shared_experts"):
            shared = self.shared_experts(hidden_states)
        if comm_manager is not None:
            shared, _ = comm_manager.post_dense_comm(shared, None, ctx)
        return shared

    def forward_mega_moe(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        ctx: ForwardContext,
        comm_manager: CommManager,
    ) -> torch.Tensor:
        with nvtx_range("moe_select_experts"):
            topk_output = self._compute_topk_output(hidden_states, input_ids)
        if self.shared_expert_fused:
            # The routed output already includes the shared expert.
            with nvtx_range("moe_mega_experts"):
                return self._forward_routed_experts(
                    hidden_states,
                    topk_output,
                    num_global_tokens,
                    max_num_tokens_per_gpu,
                    do_finalize=True,
                )
        shared = None
        with self.stream_fork.scope(enable=get_is_capture_mode()) as fork:
            with nvtx_range("moe_mega_experts"):
                routed = self._forward_routed_experts(
                    hidden_states,
                    topk_output,
                    num_global_tokens,
                    max_num_tokens_per_gpu,
                    do_finalize=True,
                )
            with fork.branch():
                shared = self._forward_shared_experts(
                    hidden_states,
                    ctx=ctx,
                    comm_manager=comm_manager,
                )
        return routed + shared if shared is not None else routed

    def forward_normal(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write the routed + shared sum into out when supplied.

        The following all-reduce can read this memory in place.
        """
        if hidden_states.shape[0] == 0 and not self.use_gluon_petit:
            return hidden_states if out is None else out.copy_(hidden_states)
        with nvtx_range("moe_select_experts"):
            topk_output = self._compute_topk_output(hidden_states, input_ids)
        deferred_finalize = (
            hidden_states.dtype == torch.bfloat16
            and self.experts.supports_deferred_cuda_finalize
        )
        shared = None
        with self.stream_fork.scope(enable=get_is_capture_mode()) as fork:
            with nvtx_range("moe_experts"):
                routed = self._forward_routed_experts(
                    hidden_states,
                    topk_output,
                    num_global_tokens,
                    max_num_tokens_per_gpu,
                    do_finalize=not deferred_finalize,
                )
            with fork.branch():
                shared = self._forward_shared_experts(hidden_states)
        if deferred_finalize:
            # Reduce routed rows and add the shared expert in one finalize.
            gemm2_out, expert_weights, expanded_idx = routed
            result = moe_finalize_fuse_shared(
                gemm2_out,
                expanded_idx,
                expert_weights,
                shared,
                top_k=expert_weights.shape[1],
                enable_pdl=pdl_enabled(),
                hidden_dim=hidden_states.shape[-1],
            )
            return result if out is None else out.copy_(result)
        if shared is None:
            return routed if out is None else out.copy_(routed)
        return torch.add(routed, shared, out=out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        ctx: ForwardContext | None = None,
        comm_manager: CommManager | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_mega_moe:
            if out is not None:
                raise ValueError("DeepSeek V4 MegaMoE does not support out")
            return self.forward_mega_moe(
                hidden_states,
                input_ids,
                num_global_tokens,
                max_num_tokens_per_gpu,
                ctx,
                comm_manager,
            )
        return self.forward_normal(
            hidden_states, input_ids, num_global_tokens, max_num_tokens_per_gpu, out
        )
