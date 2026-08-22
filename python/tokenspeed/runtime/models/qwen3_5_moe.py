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


"""Qwen3.5 MoE blocks shared by dense and MoE model variants."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.activation.triton import (
    fused_gate_sigmoid_mul_add,
    fused_swiglu_fp8_ue8m0,
)
from tokenspeed_kernel.ops.gemm.cute_dsl import (
    nvfp4_gemm_swiglu_nvfp4_quant,
)
from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.configs.qwen3_5_text_base_config import Qwen3_5BaseTextConfig
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.cuda_graph_wrapper import (
    get_is_capture_mode,
    get_is_cuda_graph_phase,
)
from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.dense.nvfp4 import Nvfp4LinearMethod
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.topk import TopK
from tokenspeed.runtime.layers.moe.utils import (
    RoutingMethodType,
    get_all2all_backend,
    get_moe_backend,
    use_deepep_low_latency,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config
from tokenspeed.runtime.layers.quantization.utils import should_exclude_quant_module
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.env import envs, global_server_args_dict
from tokenspeed.runtime.utils.pdl import pdl_enabled

_is_blackwell = current_platform().is_blackwell


def _is_moe_layer(layer_id: int, config) -> bool:
    """Return whether the given decoder layer should use the MoE block."""
    if layer_id < 0:
        return False
    mlp_only_layers = getattr(config, "mlp_only_layers", [])
    if layer_id in mlp_only_layers:
        return False
    return config.num_experts > 0 and (layer_id + 1) % config.decoder_sparse_step == 0


_GATE_UP_CHECKPOINT_MEMBERS = ("gate_proj", "up_proj")


def _gate_up_quant_config(
    quant_config: QuantizationConfig | None, prefix: str
) -> QuantizationConfig | None:
    """Drop the quant config when the checkpoint keeps this MLP's weights bf16.

    Args:
        quant_config: Quantization config the caller would otherwise pass.
        prefix: Full runtime prefix of the fused ``gate_up_proj`` module.

    Returns:
        ``None`` when either checkpoint member is excluded, else ``quant_config``.
    """
    if not isinstance(quant_config, Nvfp4Config) or not prefix:
        return quant_config
    parent = prefix.rpartition(".")[0]
    for member in _GATE_UP_CHECKPOINT_MEMBERS:
        member_prefix = f"{parent}.{member}" if parent else member
        if should_exclude_quant_module(member_prefix, quant_config.exclude_modules):
            return None
    return quant_config


class Qwen3_5MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
        replicate_weights: bool = False,
    ) -> None:
        """Gated MLP used for the dense and shared-expert paths.

        Args:
            hidden_size: Model hidden dimension.
            intermediate_size: Per-expert intermediate dimension (unsharded).
            hidden_act: Activation name; only "silu" is supported.
            mapping: Parallelism mapping; the dense TP group shards the weights.
            quant_config: Optional quantization config.
            reduce_results: Whether the row-parallel projection reduces across
                the dense TP group. Ignored when weights are replicated.
            prefix: Weight-name prefix.
            replicate_weights: Keep the full weights on every rank and skip TP
                sharding, making this MLP data parallel over whatever token rows
                the caller passes. Required when the caller already holds a
                distinct token shard per rank (the DeepEP layout), where a
                tensor-parallel reduction would sum partial products computed
                from different tokens.
        """
        super().__init__()
        self.mapping = mapping
        gate_up_quant_config = _gate_up_quant_config(
            quant_config, add_prefix("gate_up_proj", prefix)
        )
        if mapping.dense.has_tp and not replicate_weights:
            tp_size = mapping.dense.tp_size
            tp_rank = mapping.dense.tp_rank
            tp_group = mapping.dense.tp_group
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size,
                [intermediate_size] * 2,
                bias=False,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_group=tp_group,
                quant_config=gate_up_quant_config,
                prefix=add_prefix("gate_up_proj", prefix),
            )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_group=tp_group,
                reduce_results=reduce_results,
                quant_config=quant_config,
                prefix=add_prefix("down_proj", prefix),
            )
        else:
            self.gate_up_proj = ReplicatedLinear(
                hidden_size,
                intermediate_size * 2,
                bias=False,
                quant_config=gate_up_quant_config,
                prefix=add_prefix("gate_up_proj", prefix),
            )
            self.down_proj = ReplicatedLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("down_proj", prefix),
            )

        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

        self._use_nvfp4_gemm_swiglu_nvfp4_quant = (
            envs.TOKENSPEED_NVFP4_GEMM_SWIGLU_NVFP4_QUANT.get()
            and _is_blackwell
            and isinstance(self.gate_up_proj.quant_method, Nvfp4LinearMethod)
            and isinstance(self.down_proj.quant_method, Nvfp4LinearMethod)
        )
        self.gate_up_proj.interleave_linear_and_gate = (
            self._use_nvfp4_gemm_swiglu_nvfp4_quant
        )

    def _uses_deep_gemm_fp8_mlp(self) -> bool:
        """Whether both projections can consume packed UE8M0 activations."""
        return bool(
            _is_blackwell
            and getattr(self.gate_up_proj, "_use_deep_gemm_fp8", False)
            and getattr(self.down_proj, "_use_deep_gemm_fp8", False)
        )

    def forward(self, x):
        if x.shape[0] == 0:
            return x
        if self._use_nvfp4_gemm_swiglu_nvfp4_quant:
            x_fc1_fp4, x_fc1_scale = fp4_quantize(
                x,
                self.gate_up_proj.input_scale_inv,
                enable_pdl=pdl_enabled(),
            )
            x_fp4, x_scale = nvfp4_gemm_swiglu_nvfp4_quant(
                x_fc1_fp4,
                x_fc1_scale,
                self.gate_up_proj.weight_swiglu_interleaved,
                self.gate_up_proj.weight_scale_swiglu_interleaved,
                self.gate_up_proj.alpha,
                self.down_proj.input_scale_inv,
                enable_pdl=pdl_enabled(),
            )
            x, _ = self.down_proj((x_fp4, x_scale))
            return x
        gate_up, _ = self.gate_up_proj(x)
        if self._uses_deep_gemm_fp8_mlp():
            x_fp8, x_scale = fused_swiglu_fp8_ue8m0(
                gate_up,
                enable_pdl=pdl_enabled(),
            )
            if isinstance(self.down_proj, RowParallelLinear):
                x, _ = self.down_proj(x_fp8, x_scale)
            else:
                # Replicated shared-expert projections do not have the
                # RowParallelLinear scale shortcut, so request BF16 explicitly
                # instead of inheriting the FP8 activation dtype.
                x, _ = self.down_proj(
                    x_fp8,
                    block_scale=x_scale,
                    output_dtype=torch.bfloat16,
                )
            return x
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: Qwen3_5BaseTextConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        layer_index: int = -1,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.mapping = mapping
        self.layer_index = layer_index
        self.tp_size = mapping.world_size
        self.stream_fork = StreamFork(alt_stream)
        # DeepEP needs a MoE backend whose apply kernel drives the
        # dispatch/combine legs itself: nvfp4 via flashinfer cutedsl, or
        # block-scale fp8 via DeepGEMM masked grouped GEMMs.
        # Draft models (non-quantized) must fall back to the TP path even
        # when the target model has deep_ep configured globally.
        moe_backend = get_moe_backend()
        self.use_deepep = get_all2all_backend().is_deepep() and (
            moe_backend.is_flashinfer_cutedsl() or moe_backend.is_deep_gemm()
        )
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_index,
            is_moe=True,
            prev_is_moe=_is_moe_layer(layer_index - 1, config),
        )

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        self.experts = MoELayer(
            top_k=config.num_experts_per_tok,
            num_experts=config.num_experts
            + global_server_args_dict["ep_num_redundant_experts"],
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=layer_index,
            prefix=prefix,
            tp_rank=self.mapping.moe.tp_rank,
            tp_size=self.mapping.moe.tp_size,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
            routing_config={
                "routing_method_type": RoutingMethodType.RenormalizeNaive,
                "normalize_topk_weights": config.norm_topk_prob,
            },
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=False,
            output_format=self.experts.topk_output_format,
        )
        if getattr(config, "shared_expert_intermediate_size", 0) > 0:
            self.shared_expert = Qwen3_5MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                mapping=self.mapping,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_expert", prefix),
                # The DeepEP path runs the whole block on this rank's token
                # shard, so the shared expert is data parallel over those rows:
                # replicated weights, no cross-rank reduction. Sharding it
                # instead would reduce partial products belonging to different
                # tokens. It also lifts the dense-TP ceiling that a small
                # shared_expert_intermediate_size imposes under block-quantized
                # FP8 (a shard must stay >= the quantization block).
                replicate_weights=self.use_deepep,
            )
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)
        else:
            self.shared_expert = None
            self.shared_expert_gate = None

    def get_moe_routed_weights(self):
        """Return routed expert weights excluding auxiliary shared parameters."""
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"] and "shared_experts" not in name
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        if self.use_deepep:
            return self._forward_deepep(
                hidden_states, num_global_tokens, max_num_tokens_per_gpu, ctx
            )
        return self._forward_tp(
            hidden_states, num_global_tokens, max_num_tokens_per_gpu, ctx
        )

    def _forward_tp(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Gate on local (pre-comm) tokens
        router_logits, _ = self.gate(hidden_states)

        # All-gather hidden_states and router_logits for topk + experts
        hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
        router_logits = self.comm_manager.pre_mlp_comm(router_logits, ctx)

        shared_output = None
        with self.stream_fork.scope(
            enable=(
                self.shared_expert is not None
                and hidden_states.shape[0] > 0
                and get_is_cuda_graph_phase()
            ),
            overlap=get_is_capture_mode(),
        ) as fork:
            with fork.branch():
                if self.shared_expert is not None:
                    shared_output = self.shared_expert(hidden_states)

            if hidden_states.shape[0] > 0:
                topk_output = self.topk(hidden_states, router_logits)
            else:
                topk_output = self.topk.empty_topk_output(
                    hidden_states.device,
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                )

            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                topk_output=topk_output,
                num_global_tokens=num_global_tokens,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            )

        if shared_output is not None:
            if self.shared_expert_gate is not None and hidden_states.shape[0] > 0:
                fused_gate_sigmoid_mul_add(
                    hidden_states,
                    self.shared_expert_gate.weight.squeeze(0),
                    shared_output,
                    final_hidden_states,
                )
            else:
                final_hidden_states = final_hidden_states + shared_output

        # Reduce-scatter / all-reduce expert output back to local token count
        final_hidden_states, _ = self.comm_manager.post_mlp_fused(
            final_hidden_states, None, ctx
        )

        return final_hidden_states.view(num_tokens, hidden_dim)

    def _forward_deepep(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """DeepEP path: routing on local tokens, dispatch/combine handled by executor."""
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Gate on local tokens (no all-gather needed)
        router_logits, _ = self.gate(hidden_states)

        # Shared expert on this rank's token shard. Weights are replicated for
        # the DeepEP path (see the constructor), so the result is already
        # complete -- no tensor-parallel reduction. It only reads
        # ``hidden_states``, so it runs inside DeepEP's dispatch window below,
        # where its GEMMs cover the in-flight token transfer.
        shared_output = None
        overlap_fn = None
        if self.shared_expert is not None:

            def overlap_fn() -> None:
                nonlocal shared_output
                shared_output = self.shared_expert(hidden_states)

        # TopK on local tokens
        if hidden_states.shape[0] > 0:
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk.empty_topk_output(
                hidden_states.device,
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

        # DeepEP executor handles dispatch -> MoE GEMM -> combine internally.
        # Decode-shaped forwards take the low-latency legs; extend-shaped ones
        # exceed their fixed capacity and must take the normal legs.
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            low_latency=use_deepep_low_latency(ctx, self.mapping.attn.dp_size),
            overlap_fn=overlap_fn,
        )

        if shared_output is not None:
            if self.shared_expert_gate is not None and hidden_states.shape[0] > 0:
                fused_gate_sigmoid_mul_add(
                    hidden_states,
                    self.shared_expert_gate.weight.squeeze(0),
                    shared_output,
                    final_hidden_states,
                )
            else:
                final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states.view(num_tokens, hidden_dim)
