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

from collections.abc import Iterable as _Iterable

import tokenspeed_kernel
import torch
import torch.nn as nn
import torch.nn.functional as _F
from tokenspeed_kernel.platform import current_platform as _current_platform
from tokenspeed_kernel.thirdparty.cuda import dsv3_router_gemm as _dsv3_router_gemm
from tokenspeed_kernel.thirdparty.cuda import (
    moe_finalize_fuse_shared as _moe_finalize_fuse_shared,
)
from transformers import PretrainedConfig as _PretrainedConfig

from tokenspeed.runtime.configs.numerics import BITWISE_ENVELOPES
from tokenspeed.runtime.configs.utils import get_rope_theta as _get_rope_theta
from tokenspeed.runtime.distributed.comm_manager import CommManager as _CommManager
from tokenspeed.runtime.distributed.mapping import Mapping as _Mapping
from tokenspeed.runtime.execution.context import ForwardContext as _ForwardContext
from tokenspeed.runtime.execution.forward_step import (
    get_is_capture_mode as _get_is_capture_mode,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm as _RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema as _ExpertCheckpointSchema,
)
from tokenspeed.runtime.layers.moe import (
    build_moe_checkpoint_loader as _build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.expert import MoELayer as _MoELayer
from tokenspeed.runtime.layers.moe.topk import (
    ExpertLocationDispatchInfo as _ExpertLocationDispatchInfo,
)
from tokenspeed.runtime.layers.moe.topk import TopK as _TopK
from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat as _TopKOutputFormat
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType as _RoutingMethodType
from tokenspeed.runtime.layers.moe.utils import (
    get_all2all_backend as _get_all2all_backend,
)
from tokenspeed.runtime.layers.quantization.base_config import (
    QuantizationConfig as _QuantizationConfig,
)
from tokenspeed.runtime.layers.quantization.utils import block_dequant as _block_dequant
from tokenspeed.runtime.layers.quantization.utils import (
    should_ignore_quant_layer as _should_ignore_quant_layer,
)
from tokenspeed.runtime.layers.utils import get_layer_id as _get_layer_id
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as _VocabParallelEmbedding,
)
from tokenspeed.runtime.model_loader.weight_utils import (
    default_weight_loader as _default_weight_loader,
)
from tokenspeed.runtime.models.base import BaseCausalLM as _BaseCausalLM
from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3AttentionMLA as _DeepseekV3AttentionMLA,
)
from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3MLP as _DeepseekV3MLP
from tokenspeed.runtime.models.deepseek_v3 import (
    _prepare_mla_kv_b_proj_weights,
)
from tokenspeed.runtime.moe.dispatch_algorithm import (
    has_zero_expert as _has_zero_expert,
)
from tokenspeed.runtime.moe.expert_location import (
    ExpertLocationMetadata as _ExpertLocationMetadata,
)
from tokenspeed.runtime.moe.expert_location import (
    ModelConfigForExpertLocation as _ModelConfigForExpertLocation,
)
from tokenspeed.runtime.moe.expert_location import (
    get_global_expert_location_metadata as _get_global_expert_location_metadata,
)
from tokenspeed.runtime.utils import LazyValue, add_prefix, get_colorful_logger
from tokenspeed.runtime.utils.cuda_stream import StreamFork as _StreamFork
from tokenspeed.runtime.utils.env import global_server_args_dict

_longcat_logger = get_colorful_logger(__name__)
_longcat_platform = _current_platform()
_longcat_is_hopper_plus = _longcat_platform.is_hopper_plus
_LONGCAT_OPTIONAL_MISSING_WEIGHT_SUFFIXES = (
    ".k_scale",
    ".v_scale",
)


def _ensure_longcat_config(config):
    """Normalize LongCat HF config aliases used by the runtime layers."""

    if not hasattr(config, "num_hidden_layers") and hasattr(config, "num_layers"):
        config.num_hidden_layers = config.num_layers
    if not hasattr(config, "intermediate_size") and hasattr(config, "ffn_hidden_size"):
        config.intermediate_size = config.ffn_hidden_size
    if not hasattr(config, "moe_intermediate_size"):
        if hasattr(config, "expert_ffn_hidden_size"):
            config.moe_intermediate_size = config.expert_ffn_hidden_size
        else:
            config.moe_intermediate_size = config.intermediate_size
    if not hasattr(config, "num_experts_per_tok") and hasattr(config, "moe_topk"):
        config.num_experts_per_tok = config.moe_topk
    if not hasattr(config, "moe_topk") and hasattr(config, "num_experts_per_tok"):
        config.moe_topk = config.num_experts_per_tok

    if not hasattr(config, "hidden_act"):
        config.hidden_act = "silu"
    if not hasattr(config, "norm_topk_prob"):
        config.norm_topk_prob = False
    if not hasattr(config, "zero_expert_num"):
        config.zero_expert_num = 0
    if not hasattr(config, "zero_expert_type"):
        config.zero_expert_type = ""
    if not hasattr(config, "router_bias"):
        config.router_bias = False
    if not hasattr(config, "router_dtype"):
        config.router_dtype = "float32"
    if not hasattr(config, "routed_scaling_factor"):
        config.routed_scaling_factor = 1.0

    return config


def _get_longcat_moe_quant_config(
    config: _PretrainedConfig,
    quant_config: _QuantizationConfig | None,
    prefix: str,
):
    if quant_config is None:
        return None

    ignored_layers = quant_config.ignored_layers
    if not ignored_layers:
        return quant_config

    expert_proj_names = ("gate_proj", "up_proj", "down_proj")
    num_expected = config.n_routed_experts * len(expert_proj_names)
    num_ignored = 0
    for expert_id in range(config.n_routed_experts):
        expert_prefix = add_prefix(f"experts.{expert_id}", prefix)
        for proj_name in expert_proj_names:
            if _should_ignore_quant_layer(
                prefix=add_prefix(proj_name, expert_prefix),
                ignored_layers=ignored_layers,
            ):
                num_ignored += 1

    if num_ignored == 0:
        return quant_config
    if num_ignored == num_expected:
        return None

    raise ValueError(
        f"LongCat MoE layer {prefix} has partially ignored expert quantization "
        f"({num_ignored}/{num_expected} expert projections). TokenSpeed requires "
        "all experts in one fused MoE layer to use the same weight format."
    )


def _check_longcat_expert_placement(
    placement: _ExpertLocationMetadata,
    config: _PretrainedConfig,
    layer_index: int,
    mapping: _Mapping,
) -> None:
    """Refuse a placement whose geometry is not this model's."""
    if placement.num_logical_experts != config.n_routed_experts:
        raise ValueError(
            f"expert placement has {placement.num_logical_experts} logical experts, "
            f"LongCat routes {config.n_routed_experts}"
        )
    if not 0 <= layer_index < placement.num_layers:
        raise ValueError(
            f"LongCat MoE layer {layer_index} is outside the placement's "
            f"{placement.num_layers} layers; the layer index must be passed"
        )
    if placement.ep_size != mapping.moe.ep_size:
        raise ValueError(
            f"expert placement spans ep_size={placement.ep_size}, the MoE mapping "
            f"has ep_size={mapping.moe.ep_size}"
        )
    algorithm = global_server_args_dict["ep_dispatch_algorithm"]
    if config.zero_expert_num > 0 and not _has_zero_expert(algorithm):
        raise ValueError(
            f"LongCat routes {config.zero_expert_num} zero experts; use "
            "--ep-dispatch-algorithm static_with_zero_expert (or "
            f"dynamic_with_zero_expert), not {algorithm}"
        )


class _RuntimeLongcatRouter(nn.Module):
    def __init__(self, config: _PretrainedConfig, prefix: str = ""):
        super().__init__()
        if getattr(config, "router_bias", False):
            raise ValueError("LongCat router bias is not supported.")

        num_logits = config.n_routed_experts + config.zero_expert_num
        params_dtype = (
            torch.bfloat16 if config.router_dtype == "bfloat16" else torch.float32
        )
        self.classifier = ReplicatedLinear(
            config.hidden_size,
            num_logits,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=add_prefix("classifier", prefix),
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(num_logits, dtype=torch.float32)
        )

    def forward(self, hidden_states: torch.Tensor):
        if global_server_args_dict["numerics"] in BITWISE_ENVELOPES:
            # The classifier's logits feed expert selection, so they must be
            # batch-invariant or top-k flips at near-ties. cuBLAS and the
            # dsv3 router kernel tile by shape; the aok leaf does not.
            return tokenspeed_kernel.mm(
                hidden_states.float(),
                self.classifier.weight.float(),
                override="aok",
            )
        if _longcat_is_hopper_plus and hidden_states.shape[0] > 0:
            return _dsv3_router_gemm(
                hidden_states,
                self.classifier.weight,
                out_dtype=torch.float32,
            )
        return _F.linear(hidden_states.float(), self.classifier.weight.float(), None)


class _RuntimeLongcatMoE(nn.Module):
    def __init__(
        self,
        config: _PretrainedConfig,
        mapping: _Mapping,
        quant_config: _QuantizationConfig | None = None,
        layer_index: int = -1,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.mapping = mapping
        self.layer_index = layer_index
        self.n_routed_experts = config.n_routed_experts
        self.zero_expert_num = config.zero_expert_num
        self.zero_expert_type = config.zero_expert_type
        self.routed_scaling_factor = config.routed_scaling_factor
        # The routed output leaves this module as one partial per MoE TP-EP
        # rank and post_moe_comm sums the group (all-reduce or reduce-scatter),
        # so the identity zero-expert residual, which every rank could compute
        # from its replicated input, must enter exactly one partial.
        self.adds_zero_expert_residual: bool = self.mapping.moe.tp_ep_rank == 0
        self.stream_fork = _StreamFork(alt_stream)

        if self.mapping.moe.ep_size > config.n_routed_experts:
            raise ValueError(
                f"EP size {self.mapping.moe.ep_size} is greater than the number "
                f"of LongCat routed experts {config.n_routed_experts}."
            )
        if _get_all2all_backend().is_deepep():
            # The decoder layer gathers the MoE input over the MoE TP-EP group
            # and reduces the routed output through post_moe_comm, and the
            # identity zero-expert residual enters one rank's partial on that
            # assumption. DeepEP's combine already reduces inside the kernel
            # and keeps each rank's own token rows, so the two cannot compose.
            raise ValueError(
                "LongCat-Flash does not support --all2all-backend deepep: its MoE "
                "layer reduces the routed output through the host's MoE "
                "all-reduce / reduce-scatter, which DeepEP's in-kernel combine "
                "already performs; launch with --all2all-backend none"
            )
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for LongCat."
            )

        self.router = _RuntimeLongcatRouter(
            config=config,
            prefix=add_prefix("router", prefix),
        )
        # The target's expert placement (process-global while the target is
        # built; None for drafts and plain serving): P = E + R physical slots,
        # the router emits physical ids and the loader fills every replica.
        self.expert_placement: _ExpertLocationMetadata | None = (
            _get_global_expert_location_metadata()
        )
        if self.expert_placement is not None:
            _check_longcat_expert_placement(
                self.expert_placement, config, layer_index, self.mapping
            )
        self.experts = _MoELayer(
            top_k=config.moe_topk,
            num_experts=(
                config.n_routed_experts
                if self.expert_placement is None
                else self.expert_placement.num_physical_experts
            ),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=layer_index,
            prefix=prefix,
            tp_rank=self.mapping.moe.tp_rank,
            tp_size=self.mapping.moe.tp_size,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
            zero_expert_num=config.zero_expert_num,
            # LongCat applies its own zero-expert routing to gated SiLU experts.
            activation="swiglu",
            routing_mode="precomputed_topk",
            routing_config={
                "routed_scaling_factor": self.routed_scaling_factor,
                "normalize_topk_weights": config.norm_topk_prob,
                "correction_bias": self.router.e_score_correction_bias[
                    : config.n_routed_experts
                ],
                "routing_method_type": _RoutingMethodType.DeepSeekV3,
            },
        )
        if config.zero_expert_num > 0 and self.experts.topk_output_format.is_bypassed():
            raise ValueError(
                "LongCat zero experts require a MoE backend that accepts "
                "precomputed top-k ids. Launch with --moe-runner-backend triton."
            )
        # --moe-combine-order (docs/design/numerics.md): under "slot" the MoE
        # leaf folds a token's slots across the EP group itself, identity
        # zero-expert residual included, so this module hands it the raw
        # top-k and adds nothing (post_moe_comm then reduces nothing either).
        self.combine_order: str = self.experts.combine_order
        self.topk = _TopK(
            top_k=config.moe_topk,
            layer_id=layer_index,
            renormalize=config.norm_topk_prob,
            correction_bias=self.router.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            output_format=_TopKOutputFormat.STANDARD,
            zero_expert_num=config.zero_expert_num,
            # DeepEP, the one consumer of int64 ids, is refused above.
            topk_indices_dtype=torch.int32,
        )
        # This layer's view of the placement tables for the router; the
        # dispatch flavour follows the MoE kernel: all-to-all EP routes each
        # rank's own tokens to its nearest replica, replicated-input EP routes
        # every token on every rank and needs a rank-agnostic replica choice.
        self.expert_dispatch_info: _ExpertLocationDispatchInfo | None = None
        if self.expert_placement is not None:
            self.expert_dispatch_info = _ExpertLocationDispatchInfo.init_new(
                layer_id=layer_index,
                ep_dispatch_algorithm=global_server_args_dict["ep_dispatch_algorithm"],
                expert_location_metadata=self.expert_placement,
                all_to_all_ep=self.experts.supports_all_to_all_ep,
            )

    def get_moe_routed_weights(self):
        return [
            param.data
            for name, param in self.experts.named_parameters()
            if name not in ["correction_bias"] and "shared_experts" not in name
        ]

    def _apply_zero_experts(self, hidden_states: torch.Tensor, topk_output):
        """Mask the zero-expert slots out of the routing and return this rank's
        share of the identity residual (None when it adds none).

        The residual ``hidden * sum(zero-slot weights)`` is added to the routed
        partial BEFORE post_moe_comm sums the partials over the MoE TP-EP
        group, so only one rank (``adds_zero_expert_residual``) materializes
        it; the others contribute exactly 0 and the reduction counts it once.

        Under the slot-order combine the top-k stays as routed: zero-expert
        slots keep their ``-1`` / past-the-experts id and their weight, and
        the leaf folds the residual in fp32 slot order itself.
        """
        if self.zero_expert_num <= 0 or self.combine_order == "slot":
            return None

        # The router's contract: a zero expert is -1, every other id is a
        # physical slot in [0, P) (the routed experts, plus the replicas an
        # expert placement adds past E). Nothing here depends on whether a
        # placement is active; the bound is checked device-side, graph-safe.
        topk_ids = topk_output.topk_ids
        torch._assert_async(
            (topk_ids < self.experts.num_experts).all(),
            f"LongCat top-k id at or beyond the {self.experts.num_experts} "
            "physical experts; zero experts must be -1",
        )
        zero_expert_mask = topk_ids < 0
        zero_expert_weights = torch.where(
            zero_expert_mask,
            topk_output.topk_weights,
            torch.zeros_like(topk_output.topk_weights),
        )
        # Fused MoE kernels still read every selected expert id while building
        # the dispatch plan, so zero-expert slots must keep a valid id.
        topk_output.topk_ids[zero_expert_mask] = 0
        topk_output.topk_weights[zero_expert_mask] = 0.0

        if self.zero_expert_type in ("identity", "copy"):
            if not self.adds_zero_expert_residual:
                return None
            zero_weight = zero_expert_weights.sum(dim=-1, keepdim=True).to(
                hidden_states.dtype
            )
            return hidden_states * zero_weight
        if self.zero_expert_type in ("", "drop"):
            return None
        raise ValueError(
            f"Unsupported LongCat zero expert type: {self.zero_expert_type}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        with self.stream_fork.scope(enable=_get_is_capture_mode()):
            router_logits = self.router(hidden_states)
            if hidden_states.shape[0] > 0:
                topk_output = self.topk(
                    hidden_states,
                    router_logits,
                    expert_location_dispatch_info=self.expert_dispatch_info,
                )
            else:
                topk_output = self.topk.empty_topk_output(
                    hidden_states.device,
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                )

            zero_expert_output = self._apply_zero_experts(hidden_states, topk_output)
            deferred_finalize = self.experts.supports_deferred_finalize
            routed_expert_output = self.experts(
                hidden_states=hidden_states,
                topk_output=topk_output,
                num_global_tokens=num_global_tokens,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu,
                do_finalize=not deferred_finalize,
            )

        if deferred_finalize:
            gemm2_out, expert_weights, expanded_idx = routed_expert_output
            return _moe_finalize_fuse_shared(
                gemm2_out,
                expanded_idx,
                expert_weights,
                zero_expert_output,
                top_k=self.topk.topk_config.top_k,
            )

        if zero_expert_output is not None:
            # Pre-reduction add: the caller's post_moe_comm sums this partial
            # with the other MoE TP-EP ranks', which hold None here.
            routed_expert_output = routed_expert_output + zero_expert_output
        return routed_expert_output


def _lora_norm_scales(config: _PretrainedConfig) -> tuple[float | None, float | None]:
    """LongCat's ``sqrt(hidden / lora_rank)`` factors on its q and kv LoRA norms.

    Returns:
        ``(q_scale, kv_scale)``; an entry is None when the checkpoint does not
        apply that scale (``mla_scale_q_lora`` / ``mla_scale_kv_lora`` unset).
    """
    q_scale = None
    if (
        getattr(config, "mla_scale_q_lora", False)
        and getattr(config, "q_lora_rank", None) is not None
    ):
        q_scale = (config.hidden_size / config.q_lora_rank) ** 0.5
    kv_scale = None
    if getattr(config, "mla_scale_kv_lora", False):
        kv_scale = (config.hidden_size / config.kv_lora_rank) ** 0.5
    return q_scale, kv_scale


class _RuntimeLongcatDecoderLayer(nn.Module):
    """One LongCat layer: two attention/dense-MLP branches beside one MoE.

    Row layout: the residual stream runs through the dense branches
    (attention 0 -> MLP 0 -> attention 1 -> MLP 1), so the layer's rows follow
    the dense comm pattern -- all-reduce (every attention-TP rank holds every
    row of its attention DP group) or RSAG (each rank holds its scattered
    share). The MoE is a side branch fed from attention 0's output; its own
    pattern may differ (attention TP equal to the dense TP but not to the MoE
    TP-EP width, e.g. attention DP with EP), so ``_forward_moe`` bridges its
    rows into and out of the dense layout around the MoE collectives.
    """

    def __init__(
        self,
        config: _PretrainedConfig,
        layer_id: int,
        mapping: _Mapping,
        quant_config: _QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        rope_theta = _get_rope_theta(config)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling and "factor" not in rope_scaling:
            rope_scaling = None
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        # --mla-lora-scale: "runtime" hands the norm scales to the attention
        # as separate multiplies; "folded" leaves them to post_load_weights.
        q_lora_scale, kv_lora_scale = (
            _lora_norm_scales(config)
            if global_server_args_dict["mla_lora_scale"] == "runtime"
            else (None, None)
        )

        self.self_attn = nn.ModuleList(
            [
                _DeepseekV3AttentionMLA(
                    config=config,
                    hidden_size=self.hidden_size,
                    num_heads=config.num_attention_heads,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    v_head_dim=config.v_head_dim,
                    q_lora_rank=getattr(config, "q_lora_rank", None),
                    kv_lora_rank=config.kv_lora_rank,
                    rope_theta=rope_theta,
                    rope_scaling=rope_scaling,
                    max_position_embeddings=max_position_embeddings,
                    quant_config=(
                        None
                        if "self_attn" in getattr(config, "disable_quant_module", [])
                        else quant_config
                    ),
                    layer_id=layer_id * 2 + branch_id,
                    prefix=add_prefix(f"self_attn.{branch_id}", prefix),
                    reduce_attn_results=False,
                    alt_stream=alt_stream,
                    mapping=self.mapping,
                    q_lora_scale=q_lora_scale,
                    kv_lora_scale=kv_lora_scale,
                )
                for branch_id in range(2)
            ]
        )
        self.input_layernorm = nn.ModuleList(
            [_RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(2)]
        )
        self.post_attention_layernorm = nn.ModuleList(
            [_RMSNorm(config.hidden_size, eps=config.rms_norm_eps) for _ in range(2)]
        )
        dense_quant_config = (
            None
            if "mlps" in getattr(config, "disable_quant_module", [])
            else quant_config
        )
        # --tp-batch-invariant attn+dense: column-parallel down_proj and a
        # transposing dense tail (no cross-rank sum outside MoE).
        dense_batch_invariant = (
            global_server_args_dict["tp_batch_invariant"] == "attn+dense"
        )
        self.mlps = nn.ModuleList(
            [
                _DeepseekV3MLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    mapping=self.mapping,
                    quant_config=dense_quant_config,
                    prefix=add_prefix(f"mlps.{branch_id}", prefix),
                    is_shared_expert=False,
                    batch_invariant=dense_batch_invariant,
                )
                for branch_id in range(2)
            ]
        )
        self.mlp = _RuntimeLongcatMoE(
            config=config,
            mapping=self.mapping,
            quant_config=_get_longcat_moe_quant_config(
                config,
                quant_config,
                add_prefix("mlp", prefix),
            ),
            layer_index=layer_id,
            prefix=add_prefix("mlp", prefix),
            alt_stream=alt_stream,
        )

        self._init_comm()

    def _init_comm(self) -> None:
        """Build the comm managers (see the class docstring for the row layout).

        Subclasses that build their modules themselves call this after
        ``input_layernorm`` and ``post_attention_layernorm`` exist.
        """
        # Under query context parallelism the layer's rows are this rank's
        # shard of the chunk (the executor slices the inputs by
        # ctx.query_shard), so the communication managers run the sharded
        # layout: identity around attention, all-gather / reduce-scatter
        # around the dense and MoE legs, sampled rows gathered at the exit.
        query_sharded = self.mapping.attn.has_qcp
        # Attention 0 and MLP 0 share branch_comm[0]; the MoE manager only
        # drives the MoE collectives.
        self.moe_comm = _CommManager(
            mapping=self.mapping,
            layer_id=self.layer_id,
            is_moe=True,
            prev_is_moe=False,
            dense_batch_invariant=False,
            query_sharded=query_sharded,
        )
        # --tp-batch-invariant attn+dense: the dense tail transposes rows
        # instead of reduce-scattering channel partials (see the MLPs).
        dense_batch_invariant = (
            global_server_args_dict["tp_batch_invariant"] == "attn+dense"
        )
        self.branch_comm = [
            _CommManager(
                mapping=self.mapping,
                layer_id=self.layer_id * 2 + branch_id,
                is_moe=False,
                prev_is_moe=False,
                input_layernorm=self.input_layernorm[branch_id],
                post_attn_layernorm=self.post_attention_layernorm[branch_id],
                dense_batch_invariant=dense_batch_invariant,
                query_sharded=query_sharded,
            )
            for branch_id in range(2)
        ]
        self.final_norm_comm = self.branch_comm[1]
        # Without attention TP the one rank of the attention-TP group holds
        # every row of its DP group either way, so the two layouts coincide
        # and no bridge is needed (pure attention DP with EP lands here).
        self.moe_rows_differ: bool = self.mapping.has_attn_tp and (
            self.moe_comm.use_all_reduce(is_moe=True)
            != self.moe_comm.use_all_reduce(is_moe=False)
        )
        if self.moe_rows_differ and global_server_args_dict.get(
            "enable_allreduce_fusion", False
        ):
            # A fused norm reduces the un-reduced sum of both MLP outputs;
            # the bridged MoE output is already reduced in another layout.
            raise ValueError(
                "LongCat all-reduce fusion requires the MoE and dense MLPs to "
                "share one comm pattern (attention TP equal to both the dense "
                "TP and the MoE TP-EP width, or to neither)"
            )

    def _to_moe_rows(
        self, hidden_states: torch.Tensor, ctx: _ForwardContext
    ) -> torch.Tensor:
        """Re-lay dense-layout rows for the MoE collectives."""
        if not self.moe_rows_differ:
            return hidden_states
        if self.moe_comm.use_all_reduce(is_moe=False):
            return self.moe_comm.slice_scattered_rows(hidden_states, ctx)
        return self.moe_comm.gather_scattered_rows(hidden_states, ctx)

    def _to_dense_rows(
        self, hidden_states: torch.Tensor, ctx: _ForwardContext
    ) -> torch.Tensor:
        """Return the reduced MoE output to the dense layout."""
        if not self.moe_rows_differ:
            return hidden_states
        if self.moe_comm.use_all_reduce(is_moe=False):
            return self.moe_comm.gather_scattered_rows(hidden_states, ctx)
        return self.moe_comm.slice_scattered_rows(hidden_states, ctx)

    def _forward_dense_mlp(
        self,
        branch_id: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx: _ForwardContext,
    ):
        comm = self.branch_comm[branch_id]
        hidden_states = comm.pre_mlp_comm(hidden_states, ctx)
        hidden_states = self.mlps[branch_id](hidden_states)
        hidden_states, residual = comm.post_mlp_fused(hidden_states, residual, ctx)
        return hidden_states, residual

    def _forward_moe(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx: _ForwardContext,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ):
        hidden_states = self._to_moe_rows(hidden_states, ctx)
        hidden_states = self.moe_comm.pre_mlp_comm(hidden_states, ctx)
        hidden_states = self.mlp(
            hidden_states,
            num_global_tokens,
            max_num_tokens_per_gpu,
        )
        hidden_states, residual = self.moe_comm.post_mlp_fused(
            hidden_states,
            residual,
            ctx,
        )
        hidden_states = self._to_dense_rows(hidden_states, ctx)
        return hidden_states, residual

    def _forward_idle(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        ctx: _ForwardContext,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """An idle attention-DP rank joins every collective over no rows, in
        the active ranks' order.

        The MoE TP-EP group always spans the DP groups; the dense TP group
        does too when the dense TP is wider than the attention TP, so both
        dense branches run as well. Each attention runs in its place and
        decides for itself whether its layout has collectives to join (head
        TP does; a no-op otherwise).
        """
        self.self_attn[0](
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            comm_manager=self.branch_comm[0],
        )
        hidden_states, residual = self._forward_moe(
            hidden_states,
            residual,
            ctx,
            num_global_tokens,
            max_num_tokens_per_gpu,
        )
        hidden_states, residual = self._forward_dense_mlp(
            0, hidden_states, residual, ctx
        )
        self.self_attn[1](
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            comm_manager=self.branch_comm[1],
        )
        hidden_states, residual = self._forward_dense_mlp(
            1, hidden_states, residual, ctx
        )
        return hidden_states, residual

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: _ForwardContext,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_global_tokens, max_num_tokens_per_gpu = self.moe_comm.get_num_tokens(ctx)

        if ctx.forward_mode.is_idle():
            return self._forward_idle(
                positions,
                hidden_states,
                residual,
                ctx,
                num_global_tokens,
                max_num_tokens_per_gpu,
            )

        hidden_states, residual = self.branch_comm[0].input_reduce_norm(
            hidden_states,
            residual,
        )
        hidden_states = self.self_attn[0](
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            comm_manager=self.branch_comm[0],
        )
        hidden_states, residual = self.branch_comm[0].post_attn_reduce_norm(
            hidden_states,
            residual,
            ctx,
        )

        branch_input = hidden_states
        branch_residual = residual
        moe_hidden_states, _ = self._forward_moe(
            branch_input,
            branch_residual,
            ctx,
            num_global_tokens,
            max_num_tokens_per_gpu,
        )

        hidden_states, residual = self._forward_dense_mlp(
            0,
            branch_input,
            branch_residual,
            ctx,
        )
        # Mid-layer, not a layer boundary: stays fused under every
        # --layer-boundary-norm, as the trainer does.
        hidden_states, residual = self.branch_comm[1].intra_layer_add_norm(
            hidden_states,
            residual,
        )
        hidden_states = self.self_attn[1](
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            comm_manager=self.branch_comm[1],
        )
        hidden_states, residual = self.branch_comm[1].post_attn_reduce_norm(
            hidden_states,
            residual,
            ctx,
        )
        hidden_states, residual = self._forward_dense_mlp(
            1,
            hidden_states,
            residual,
            ctx,
        )

        hidden_states = hidden_states + moe_hidden_states
        return hidden_states, residual


class _RuntimeLongcatModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: _PretrainedConfig,
        mapping: _Mapping,
        quant_config: _QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        _ensure_longcat_config(config)
        self.mapping = mapping
        self.padding_id = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = _VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )
        self.alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.layers = nn.ModuleList(
            [
                _RuntimeLongcatDecoderLayer(
                    config,
                    layer_id,
                    mapping=self.mapping,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    alt_stream=self.alt_stream,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = _RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture: set[int] = set()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: _ForwardContext,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if input_embeds is not None:
            hidden_states = input_embeds
        else:
            # Under query context parallelism the ids are this rank's shard;
            # the embedding gathers them to the span for its vocab shards and
            # reduce-scatters the rows back.
            hidden_states = self.embed_tokens(input_ids, query_shard=ctx.query_shard)

        residual = None
        aux_hidden_states = [] if self.layers_to_capture else None
        layer = None
        for layer_id, layer in enumerate(self.layers):
            if aux_hidden_states is not None and layer_id in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )
            hidden_states, residual = layer(
                positions,
                hidden_states,
                ctx,
                residual,
            )

        if not ctx.forward_mode.is_idle() and layer is not None:
            hidden_states, _ = layer.final_norm_comm.final_norm(
                hidden_states,
                residual,
                ctx,
                self.norm,
            )
        return hidden_states, aux_hidden_states


class LongcatFlashForCausalLM(_BaseCausalLM):
    model_cls = _RuntimeLongcatModel
    # The MoE layers size their slots from the placement and route through
    # its tables; load_weights fills every placed replica.
    supports_expert_placement = True

    def __init__(
        self,
        config: _PretrainedConfig,
        mapping: _Mapping,
        model: _RuntimeLongcatModel | None = None,
        quant_config: _QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        _ensure_longcat_config(config)
        self._model_override = model
        super().__init__(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
        )

    def resolve_model(
        self,
        config: _PretrainedConfig,
        mapping: _Mapping,
        quant_config: _QuantizationConfig | None,
        prefix: str,
    ) -> _RuntimeLongcatModel:
        if self._model_override is not None:
            return self._model_override
        return self.model_cls(
            config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

    def post_init(self) -> None:
        # Pipeline stages hold PPMissingLayer slots for the other stages'
        # layers; only resident layers carry routed experts.
        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_routed_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer, _RuntimeLongcatDecoderLayer)
                and isinstance(layer.mlp, _RuntimeLongcatMoE)
            }
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    @property
    def expert_placement(self) -> _ExpertLocationMetadata | None:
        """The placement the MoE layers were built with; None routes trivially."""
        for layer in self.model.layers:
            if isinstance(layer.mlp, _RuntimeLongcatMoE):
                return layer.mlp.expert_placement
        return None

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None):
        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = {2, num_layers // 2, num_layers - 3}
        else:
            self.model.layers_to_capture = {val + 1 for val in layer_ids}

    def get_param(self, params_dict, name):
        if name in params_dict:
            return params_dict[name]
        if "language_model." in name:
            name = name.replace("language_model.", "")
            if name in params_dict:
                return params_dict[name]
        if ".mtp." in name or name.startswith("model.mtp."):
            return None
        if name.endswith(_LONGCAT_OPTIONAL_MISSING_WEIGHT_SUFFIXES):
            return None
        _longcat_logger.warning(f"The {name!s} is not in the model.")
        return None

    def load_weights(self, weights: _Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load a (possibly partial) checkpoint stream.

        Returns the ``named_parameters()`` names that received data (the
        ``BaseCausalLM`` weight-update contract).
        """
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        fuse_qkv_a_proj = getattr(self.config, "q_lora_rank", None) is not None
        params_dict = dict(self.named_parameters())
        # The placement the MoE layers were built with: every local slot is
        # filled from the logical expert it holds (replicas included), and RL
        # weight sync through this same path lands in every replica.
        expert_placement = self.expert_placement
        # ``get_param`` remaps checkpoint names; report the parameter's own.
        param_names = {id(param): name for name, param in params_dict.items()}
        loaded: set[str] = set()
        moe_loader = _build_moe_checkpoint_loader(
            params_dict=params_dict,
            expert_schema=_ExpertCheckpointSchema(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
            ),
            num_experts=(
                self.config.n_routed_experts
                if expert_placement is None
                else expert_placement.num_physical_experts
            ),
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
            expert_placement=expert_placement,
        )

        for name, loaded_weight in weights:
            layer_id = _get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts." in name and name not in params_dict:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    continue
                param = self.get_param(params_dict, mapped_name)
                if param is None:
                    break
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(param_names[id(param)])
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if moe_loader.matches(name):
                    loaded.add(moe_loader.load(name, loaded_weight))
                    continue

                if fuse_qkv_a_proj and (
                    "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                ):
                    quant_block_size = 1
                    if (
                        self.quant_config is not None
                        and self.quant_config.weight_block_size is not None
                    ):
                        quant_block_size = self.quant_config.weight_block_size[0]
                    begin_size_by_name = {
                        "q_a_proj": 0,
                        "kv_a_proj_with_mqa": self.config.q_lora_rank,
                    }
                    if "q_a_proj" in name:
                        param = self.get_param(
                            params_dict,
                            name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa"),
                        )
                        begin_size = begin_size_by_name["q_a_proj"]
                    else:
                        param = self.get_param(
                            params_dict,
                            name.replace(
                                "kv_a_proj_with_mqa",
                                "fused_qkv_a_proj_with_mqa",
                            ),
                        )
                        begin_size = begin_size_by_name["kv_a_proj_with_mqa"]
                    if param is None:
                        continue
                    if "scale_inv" in name:
                        begin_size //= quant_block_size
                    param.weight_loader(param, loaded_weight, begin_size=begin_size)
                    loaded.add(param_names[id(param)])
                    continue

                if "q_a_proj" in name and name not in params_dict:
                    name = name.replace("q_a_proj", "q_proj")
                param = self.get_param(params_dict, name)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(param_names[id(param)])

        self.post_load_weights()
        return loaded

    def post_load_weights(self):
        """Derive the absorbed MLA weights and fold the LoRA norm scales.

        Safe to re-run after a live update: ``w_kc``/``w_vc`` are written
        into their existing storage (captured graphs hold those addresses),
        and the ``sqrt(hidden/rank)`` fold into ``q_a_layernorm`` /
        ``kv_a_layernorm`` -- which multiplies the parameter in place and so
        must happen exactly once per loaded value -- is applied only to the
        norms this update reloaded. The initial load reloads all of them.
        Under ``--mla-lora-scale runtime`` there is no fold at all: the
        attention multiplies the scales in its forward.
        """
        reloaded = self._weight_update_loaded_names
        param_names = (
            {id(param): name for name, param in self.named_parameters()}
            if reloaded is not None
            else None
        )

        def _reloaded(param: torch.Tensor) -> bool:
            return param_names is None or param_names[id(param)] in reloaded

        for layer in self.model.layers:
            if not isinstance(layer, _RuntimeLongcatDecoderLayer):
                continue  # PPMissingLayer: another pipeline stage owns it
            for self_attn in layer.self_attn:
                if hasattr(
                    self.quant_config, "weight_block_size"
                ) and self_attn.kv_b_proj.weight.dtype in (
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fnuz,
                ):
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        if not hasattr(self_attn.kv_b_proj, "weight_scale_inv"):
                            raise RuntimeError(
                                "kv_b_proj.weight_scale_inv is required for block FP8 dequant."
                            )
                        dtype = torch.get_default_dtype()
                        w = _block_dequant(
                            self_attn.kv_b_proj.weight,
                            self_attn.kv_b_proj.weight_scale_inv,
                            weight_block_size,
                        ).to(dtype)
                    else:
                        w = self_attn.kv_b_proj.weight
                else:
                    w = self_attn.kv_b_proj.weight

                self_attn.w_kc, self_attn.w_vc = _prepare_mla_kv_b_proj_weights(
                    w, self_attn
                )
                if global_server_args_dict["mla_lora_scale"] == "folded":
                    # Under "runtime" the attention multiplies these scales in
                    # its forward and the norm weights stay as loaded (so a
                    # weight update can never fold them twice). The fold
                    # multiplies the parameter in place, so it is applied only
                    # to the norms this load (re)loaded.
                    q_scale, kv_scale = _lora_norm_scales(self.config)
                    if q_scale is not None and _reloaded(
                        self_attn.q_a_layernorm.weight
                    ):
                        self_attn.q_a_layernorm.weight.data *= q_scale
                    if kv_scale is not None and _reloaded(
                        self_attn.kv_a_layernorm.weight
                    ):
                        self_attn.kv_a_layernorm.weight.data *= kv_scale

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        _ensure_longcat_config(config)
        return _ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )


FLASHForCausalLM = LongcatFlashForCausalLM
EntryClass = LongcatFlashForCausalLM
