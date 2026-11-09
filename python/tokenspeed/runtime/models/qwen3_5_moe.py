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

from typing import Literal

import torch
from tokenspeed_kernel.ops.activation.triton import (
    fused_gate_sigmoid_mul_add,
    moe_finalize_gate_add,
)
from tokenspeed_kernel.ops.moe import moe_warmup
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.configs.qwen3_5_text_base_config import Qwen3_5BaseTextConfig
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_step import (
    get_is_capture_mode,
    get_is_cuda_graph_phase,
)
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.mlp import GatedMLP, GatedMLPFusion, gated_activation
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.topk import TopK
from tokenspeed.runtime.layers.moe.utils import (
    RoutingMethodType,
    get_all2all_backend,
    get_moe_backend,
    use_deepep_low_latency,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config
from tokenspeed.runtime.layers.quantization.utils import should_exclude_quant_module
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.env import global_server_args_dict


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


class Qwen3_5MoeMLP(GatedMLP):
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
        *,
        parallelism: Literal["dense", "moe_shared"],
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
        gate_up_quant_config = _gate_up_quant_config(
            quant_config, add_prefix("gate_up_proj", prefix)
        )
        if parallelism == "moe_shared":
            tp_size = mapping.moe.tp_ep_size
            tp_rank = mapping.moe.tp_ep_rank
            tp_group = mapping.moe.tp_ep_group
        elif parallelism == "dense":
            tp_size = mapping.dense.tp_size
            tp_rank = mapping.dense.tp_rank
            tp_group = mapping.dense.tp_group
        else:
            raise ValueError(f"Unsupported MLP parallelism: {parallelism}")
        if tp_size > 1 and not replicate_weights:
            gate_up_proj = MergedColumnParallelLinear(
                hidden_size,
                [intermediate_size] * 2,
                bias=False,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_group=tp_group,
                quant_config=gate_up_quant_config,
                prefix=add_prefix("gate_up_proj", prefix),
            )
            down_proj = RowParallelLinear(
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
            gate_up_proj = ReplicatedLinear(
                hidden_size,
                intermediate_size * 2,
                bias=False,
                quant_config=gate_up_quant_config,
                prefix=add_prefix("gate_up_proj", prefix),
            )
            down_proj = ReplicatedLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("down_proj", prefix),
            )
        super().__init__(
            gate_up_proj,
            down_proj,
            gated_activation(hidden_act, swiglu_limit=None),
            fusions=(
                GatedMLPFusion.NVFP4_GEMM_SWIGLU_QUANT,
                GatedMLPFusion.MXFP4_GEMM_SWIGLU,
            ),
        )


def _mxfp4_mega_moe_capacity(mapping: Mapping) -> int:
    """Bound prefill and speculative verify rows before allocating workspace."""
    args = global_server_args_dict
    prefill = int(args.get("chunked_prefill_size") or 0)
    if prefill <= 0:
        max_model_len = args["max_model_len"]
        if max_model_len is None or max_model_len <= 0:
            raise ValueError(
                "MXFP4 MegaMoE requires a positive --max-model-len "
                "when chunked prefill is disabled"
            )
        prefill = args["max_prefill_tokens"] + max_model_len
    verify_tokens = max(int(args.get("speculative_num_draft_tokens") or 1), 1)
    per_dp_requests = int(args.get("max_num_seqs") or 0) // mapping.attn.dp_size
    decode_batch = max(
        per_dp_requests, int(args.get("max_cudagraph_capture_size") or 0)
    )
    return max(
        prefill,
        int(args.get("prefill_graph_max_tokens") or 0),
        decode_batch * verify_tokens,
        1,
    )


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
        self.use_mega_moe = moe_backend.is_mega_moe()
        if self.use_mega_moe:
            if mapping.moe.ep_size <= 1:
                raise ValueError("Qwen3.5 MegaMoE requires EP>1")
            if not get_all2all_backend().is_none():
                raise ValueError(
                    "Qwen3.5 MegaMoE owns dispatch/combine and requires "
                    "--all2all-backend none"
                )
            if (
                global_server_args_dict.get("enable_eplb", False)
                or global_server_args_dict.get("ep_num_redundant_experts", 0)
                or global_server_args_dict.get("init_expert_location", "trivial")
                not in (None, "trivial")
            ):
                raise ValueError(
                    "Qwen3.5 MegaMoE requires trivial expert placement without "
                    "EPLB or redundant experts"
                )
            if mapping.attn.tp_size == mapping.moe.tp_ep_size:
                raise ValueError(
                    "Qwen3.5 MegaMoE requires scattered MoE rows: attention TP must "
                    "differ from MoE TP*EP; TP8/EP8 is unsupported. Use DP4/TP2/EP8 "
                    "or DP8/TP1/EP8."
                )
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_index,
            is_moe=True,
            prev_is_moe=_is_moe_layer(layer_index - 1, config),
            dense_batch_invariant=False,
            query_sharded=False,
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
            num_experts=config.num_experts,
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
            routing_mode="precomputed_topk" if self.use_mega_moe else None,
            persistent_max_num_tokens_per_gpu=(
                _mxfp4_mega_moe_capacity(mapping)
                if self.use_mega_moe and isinstance(quant_config, Mxfp4Config)
                else None
            ),
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
                # DeepEP and MegaMoE run the whole block on this rank's token
                # shard, so the shared expert is data parallel over those rows:
                # replicated weights, no cross-rank reduction. Sharding it
                # instead would reduce partial products belonging to different
                # tokens. It also lifts the dense-TP ceiling that a small
                # shared_expert_intermediate_size imposes under block-quantized
                # FP8 (a shard must stay >= the quantization block).
                replicate_weights=self.use_deepep or self.use_mega_moe,
                parallelism="moe_shared",
            )
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)
        else:
            self.shared_expert = None
            self.shared_expert_gate = None

    def warmup(self) -> None:
        """Reserve and warm the selected persistent MoE backend before capture."""
        capacity = self.experts.plan["persistent_max_num_tokens_per_gpu"]
        if capacity is not None:
            moe_warmup(self.experts.plan, self.experts, max_num_tokens=capacity)

    def nvfp4_global_scale(self) -> torch.Tensor | None:
        """None: the experts quantize their own inputs, so no producer applies a scale."""
        return None

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
        if self.use_mega_moe:
            return self._forward_mega_moe(
                hidden_states, num_global_tokens, max_num_tokens_per_gpu, ctx
            )
        if self.use_deepep:
            return self._forward_deepep(
                hidden_states, num_global_tokens, max_num_tokens_per_gpu, ctx
            )
        return self._forward_tp(
            hidden_states, num_global_tokens, max_num_tokens_per_gpu, ctx
        )

    def _forward_mega_moe(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """Route already-scattered rows, including empty collective owners."""
        num_tokens = hidden_states.shape[0]
        if num_tokens > 0:
            router_logits, _ = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
        else:
            router_logits = hidden_states.new_empty((0, self.experts.num_experts))
            topk_output = self.topk.empty_topk_output(
                hidden_states.device,
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

        shared_output = None
        # Fork after the router GEMM: shared GEMMs may reuse its dense scratch,
        # while routed MegaMoE owns a separate workspace.
        with self.stream_fork.scope(
            enable=(
                self.shared_expert is not None
                and num_tokens > 0
                and get_is_cuda_graph_phase()
            ),
            overlap=get_is_capture_mode(),
        ) as fork:
            with fork.branch():
                if self.shared_expert is not None and num_tokens > 0:
                    shared_output = self.shared_expert(hidden_states)

            # Empty owners must still enter the collective. The result is
            # complete for local rows and aliases the reusable workspace.
            output = self.experts(
                hidden_states=hidden_states,
                topk_output=topk_output,
                num_global_tokens=num_global_tokens,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu,
                do_finalize=True,
            )
        if shared_output is not None:
            fused_gate_sigmoid_mul_add(
                hidden_states,
                self.shared_expert_gate.weight.squeeze(0),
                shared_output,
                output,
            )
        return output

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

            # The shared-expert gate add absorbs the routed finalize: one
            # latency-bound gather kernel instead of the kernel's finalize and
            # the gate kernel, bitwise equal. Only plans that hand back
            # (route outputs, route weights, row map) defer, and only when no
            # post-kernel routing scale applies to the finalized output. The
            # fold is limited to NVFP4 experts, the deferred layout it was
            # checked against; other plans (e.g. the FP8 MTP draft experts)
            # keep the kernel's own finalize.
            fuse_finalize = (
                self.shared_expert is not None
                and self.shared_expert_gate is not None
                and hidden_states.shape[0] > 0
                and hidden_states.dtype == torch.bfloat16
                and self.experts.supports_deferred_finalize
                and self.experts.plan.get("weight_dtype") == "nvfp4"
                and current_platform().is_nvidia
                and not isinstance(topk_output.output_scale, torch.Tensor)
                and topk_output.output_scale == 1.0
            )
            final_hidden_states = self.experts(
                hidden_states=hidden_states,
                topk_output=topk_output,
                num_global_tokens=num_global_tokens,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu,
                do_finalize=not fuse_finalize,
            )

        if shared_output is not None:
            if fuse_finalize:
                gemm2_out, expert_weights, expanded_idx = final_hidden_states
                final_hidden_states = moe_finalize_gate_add(
                    gemm2_out,
                    expanded_idx,
                    expert_weights,
                    hidden_states,
                    self.shared_expert_gate.weight.squeeze(0),
                    shared_output,
                )
            elif self.shared_expert_gate is not None and hidden_states.shape[0] > 0:
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
