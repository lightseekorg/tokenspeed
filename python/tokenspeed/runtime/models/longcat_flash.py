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

import torch
import torch.nn as nn
import torch.nn.functional as _F
from tokenspeed_kernel.ops.moe.cuda import (
    moe_finalize_fuse_shared as _moe_finalize_fuse_shared,
)
from tokenspeed_kernel.ops.routing.cuda import dsv3_router_gemm as _dsv3_router_gemm
from tokenspeed_kernel.platform import current_platform as _current_platform
from transformers import PretrainedConfig as _PretrainedConfig

from tokenspeed.runtime.configs.utils import get_rope_theta as _get_rope_theta
from tokenspeed.runtime.distributed.comm_manager import CommManager as _CommManager
from tokenspeed.runtime.distributed.mapping import Mapping as _Mapping
from tokenspeed.runtime.execution.context import ForwardContext as _ForwardContext
from tokenspeed.runtime.execution.cuda_graph_wrapper import (
    get_is_capture_mode as _get_is_capture_mode,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm as _RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.moe.checkpoint import (
    ExpertCheckpointSchema as _ExpertCheckpointSchema,
)
from tokenspeed.runtime.layers.moe.checkpoint import (
    build_moe_checkpoint_loader as _build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.layer import MoELayer as _MoELayer
from tokenspeed.runtime.layers.moe.topk import TopK as _TopK
from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat as _TopKOutputFormat
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType as _RoutingMethodType
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
from tokenspeed.runtime.model_loader.weight_utils import (
    kv_cache_scales_loader as _kv_cache_scales_loader,
)
from tokenspeed.runtime.models.base import BaseCausalLM as _BaseCausalLM
from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3AttentionMLA as _DeepseekV3AttentionMLA,
)
from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3MLP as _DeepseekV3MLP
from tokenspeed.runtime.moe.distribution_recorder import (
    get_global_expert_distribution_recorder as _get_global_expert_distribution_recorder,
)
from tokenspeed.runtime.moe.expert_location import (
    ModelConfigForExpertLocation as _ModelConfigForExpertLocation,
)
from tokenspeed.runtime.utils import LazyValue, add_prefix, get_colorful_logger
from tokenspeed.runtime.utils.cuda_stream import StreamFork as _StreamFork
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.pdl import pdl_enabled as _pdl_enabled

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

    ignored_layers = getattr(quant_config, "ignored_layers", None)
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
        if _longcat_is_hopper_plus and hidden_states.shape[0] > 0:
            return _dsv3_router_gemm(
                hidden_states,
                self.classifier.weight,
                out_dtype=torch.float32,
                enable_pdl=_pdl_enabled(),
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
        self.stream_fork = _StreamFork(alt_stream)

        if self.mapping.moe.ep_size > config.n_routed_experts:
            raise ValueError(
                f"EP size {self.mapping.moe.ep_size} is greater than the number "
                f"of LongCat routed experts {config.n_routed_experts}."
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
        self.experts = _MoELayer(
            top_k=config.moe_topk,
            num_experts=(
                config.n_routed_experts
                + global_server_args_dict["ep_num_redundant_experts"]
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
            zero_expert_type=config.zero_expert_type,
            routing_config={
                "routed_scaling_factor": self.routed_scaling_factor,
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
        self.topk = _TopK(
            top_k=config.moe_topk,
            renormalize=config.norm_topk_prob,
            correction_bias=self.router.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=(
                self.experts.apply_routed_scaling_factor_on_output
            ),
            output_format=_TopKOutputFormat.STANDARD,
            zero_expert_num=config.zero_expert_num,
            topk_indices_dtype=(
                torch.int64
                if global_server_args_dict.get("enable_deep_ep", False)
                else torch.int32
            ),
        )

    def get_moe_routed_weights(self):
        return [
            param.data
            for name, param in self.experts.named_parameters()
            if name not in ["correction_bias"] and "shared_experts" not in name
        ]

    def _apply_zero_experts(self, hidden_states: torch.Tensor, topk_output):
        if self.zero_expert_num <= 0:
            return None

        zero_expert_mask = (topk_output.topk_ids < 0) | (
            topk_output.topk_ids >= self.n_routed_experts
        )
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
                topk_output = self.topk(hidden_states, router_logits)
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
                enable_pdl=_pdl_enabled(),
            )

        if not self.experts.apply_routed_scaling_factor_on_output:
            routed_expert_output *= self.routed_scaling_factor
        if zero_expert_output is not None:
            routed_expert_output = routed_expert_output + zero_expert_output
        return routed_expert_output


class _RuntimeLongcatDecoderLayer(nn.Module):
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

        self.moe_comm = _CommManager(
            mapping=self.mapping,
            layer_id=self.layer_id,
            is_moe=True,
            prev_is_moe=False,
            input_layernorm=self.input_layernorm[0],
            post_attn_layernorm=self.post_attention_layernorm[0],
        )
        self.branch_comm = [
            _CommManager(
                mapping=self.mapping,
                layer_id=self.layer_id * 2 + branch_id,
                is_moe=False,
                prev_is_moe=False,
                input_layernorm=self.input_layernorm[branch_id],
                post_attn_layernorm=self.post_attention_layernorm[branch_id],
            )
            for branch_id in range(2)
        ]
        self.final_norm_comm = self.branch_comm[1]

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
        return hidden_states, residual

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: _ForwardContext,
        out_cache_loc: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_global_tokens, max_num_tokens_per_gpu = self.moe_comm.get_num_tokens(ctx)

        if ctx.forward_mode.is_idle():
            hidden_states, residual = self._forward_moe(
                hidden_states,
                residual,
                ctx,
                num_global_tokens,
                max_num_tokens_per_gpu,
            )
            return hidden_states, residual

        hidden_states, residual = self.moe_comm.input_reduce_norm(
            hidden_states,
            residual,
        )
        hidden_states = self.self_attn[0](
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
            comm_manager=self.moe_comm,
        )
        hidden_states, residual = self.moe_comm.post_attn_reduce_norm(
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
        hidden_states, residual = self.branch_comm[1].input_reduce_norm(
            hidden_states,
            residual,
        )
        hidden_states = self.self_attn[1](
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
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
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if input_embeds is not None:
            hidden_states = input_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None
        aux_hidden_states = [] if self.layers_to_capture else None
        layer = None
        for layer_id, layer in enumerate(self.layers):
            if aux_hidden_states is not None and layer_id in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )
            with _get_global_expert_distribution_recorder().with_current_layer(
                layer_id
            ):
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    ctx,
                    out_cache_loc,
                    residual,
                )

        if not ctx.forward_mode.is_idle() and layer is not None:
            hidden_states = layer.final_norm_comm.final_norm(
                hidden_states,
                residual,
                ctx,
                self.norm,
            )
        return hidden_states, aux_hidden_states


class LongcatFlashForCausalLM(_BaseCausalLM):
    model_cls = _RuntimeLongcatModel

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
        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_routed_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, _RuntimeLongcatMoE)
            }
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

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
        _longcat_logger.warning("The %s is not in the model.", name)
        return None

    def load_weights(self, weights: _Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        fuse_qkv_a_proj = getattr(self.config, "q_lora_rank", None) is not None
        params_dict = dict(self.named_parameters())
        moe_loader = _build_moe_checkpoint_loader(
            params_dict=params_dict,
            expert_schema=_ExpertCheckpointSchema(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
            ),
            num_experts=self.config.n_routed_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
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
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if moe_loader.matches(name):
                    moe_loader.load(name, loaded_weight)
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
                    continue

                if "q_a_proj" in name and name not in params_dict:
                    name = name.replace("q_a_proj", "q_proj")
                param = self.get_param(params_dict, name)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, loaded_weight)

        self.post_load_weights()

    def post_load_weights(self):
        for layer in self.model.layers:
            for self_attn in layer.self_attn:
                if hasattr(
                    self.quant_config, "weight_block_size"
                ) and self_attn.kv_b_proj.weight.dtype in (
                    torch.float8_e4m3fn,
                    torch.float8_e4m3fnuz,
                ):
                    weight_block_size = self.quant_config.weight_block_size
                    if weight_block_size is not None:
                        assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
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

                w_kc, w_vc = w.unflatten(
                    0,
                    (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim),
                ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
                self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
                self_attn.w_vc = w_vc.contiguous().transpose(1, 2)
                if getattr(self.config, "mla_scale_q_lora", False) and hasattr(
                    self_attn,
                    "q_a_layernorm",
                ):
                    self_attn.q_a_layernorm.weight.data *= (
                        self.config.hidden_size / self.config.q_lora_rank
                    ) ** 0.5
                if getattr(self.config, "mla_scale_kv_lora", False):
                    self_attn.kv_a_layernorm.weight.data *= (
                        self.config.hidden_size / self.config.kv_lora_rank
                    ) ** 0.5

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = self.mapping.attn.tp_size
        tp_rank = self.mapping.attn.tp_rank
        for attn_idx, scaling_factor in _kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers * 2,
            self.config.__class__.model_type,
        ):
            layer_idx, branch_idx = divmod(attn_idx, 2)
            if not isinstance(self.model.layers[layer_idx], nn.Identity):
                self_attn = self.model.layers[layer_idx].self_attn[branch_idx]
                for attn in (self_attn.attn_mha, self_attn.attn_mqa):
                    if attn is not None and hasattr(attn, "k_scale"):
                        attn.k_scale = scaling_factor
                        attn.k_scale_float = scaling_factor

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

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
