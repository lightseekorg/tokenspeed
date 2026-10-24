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

"""LongCat Pro layers, packed-history embedding and optional decode engines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import PretrainedConfig as _PretrainedConfig

from tokenspeed.runtime.configs.longcat_config import (
    is_longcat_lsa_config as _is_longcat_lsa_config,
)
from tokenspeed.runtime.configs.utils import get_rope_theta as _get_rope_theta
from tokenspeed.runtime.distributed.mapping import Mapping as _Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as _pg_manager,
)
from tokenspeed.runtime.execution.context import ForwardContext as _ForwardContext
from tokenspeed.runtime.execution.forward_step import (
    get_is_capture_mode as _get_is_capture_mode,
)
from tokenspeed.runtime.execution.request_token_history import RequestTokenHistoryView
from tokenspeed.runtime.layers.layernorm import RMSNorm as _RMSNorm
from tokenspeed.runtime.layers.quantization.base_config import (
    QuantizationConfig as _QuantizationConfig,
)
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as _VocabParallelEmbedding,
)
from tokenspeed.runtime.models.base.decoder_layer import (
    capture_layer_input as _capture_layer_input,
)
from tokenspeed.runtime.models.base.transformer_model import notify_target_capture
from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3MLP as _DeepseekV3MLP
from tokenspeed.runtime.models.longcat.dsa import LongcatDsaAttention as _DSAAttention
from tokenspeed.runtime.models.longcat.over_embedding import (
    LongCatOverEmbedding as _LongCatOverEmbedding,
)
from tokenspeed.runtime.models.longcat.over_embedding import (
    resolve_longcat_oe_hyperparameters as _resolve_longcat_oe_hyperparameters,
)
from tokenspeed.runtime.models.longcat.weights import LongcatWeightState
from tokenspeed.runtime.models.longcat_flash import (
    _ensure_longcat_config,
    _get_longcat_moe_quant_config,
    _longcat_logger,
    _lora_norm_scales,
)
from tokenspeed.runtime.models.longcat_flash import (
    _RuntimeLongcatDecoderLayer as _ClassicDecoderLayer,
)
from tokenspeed.runtime.models.longcat_flash import (
    _RuntimeLongcatMoE,
)
from tokenspeed.runtime.moe.expert_location import get_global_expert_location_metadata
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.cuda_stream import StreamFork as _StreamFork
from tokenspeed.runtime.utils.env import envs, global_server_args_dict

if TYPE_CHECKING:
    from tokenspeed.runtime.models.longcat.attention0 import LongcatAttention0Engine
    from tokenspeed.runtime.models.longcat.layer import LongcatLayerEngine
    from tokenspeed.runtime.models.longcat.mlp import LongcatMlpWeights
    from tokenspeed.runtime.models.longcat.moe import LongcatExpertWeights


def engines_requested() -> bool:
    return envs.TOKENSPEED_LONGCAT_LAYER_ENGINE.get()


def _validate_pro_launch(mapping) -> None:
    if mapping.attn.has_qcp:
        raise ValueError("LongCat Pro does not support query context parallelism")
    if mapping.pp_size != 1:
        raise ValueError("LongCat Pro requires pipeline parallel size 1")
    if mapping.attn.dp_size != 1 or mapping.attn.kvp_size != 1:
        raise ValueError("LongCat Pro requires one attention DP and KV partition group")
    if engines_requested():
        if (
            global_server_args_dict["max_num_seqs"] != 1
            or global_server_args_dict["speculative_num_draft_tokens"] != 8
        ):
            raise ValueError(
                "LongCat engines require --max-num-seqs 1 and "
                "--speculative-num-draft-tokens 8"
            )
        if (
            global_server_args_dict["enable_prefix_caching"]
            or global_server_args_dict["enable_kvstore"]
        ):
            raise ValueError(
                "LongCat engines require --disable-prefix-caching and --disable-kvstore"
            )
        if (
            mapping.attn.tp_size != 8
            or mapping.attn.dp_size != 1
            or mapping.attn.kvp_size != 1
            or mapping.moe.tp_ep_size != 8
            or mapping.dense.tp_size != 8
            or mapping.moe.ep_size != 1
        ):
            raise ValueError("LongCat engines require TP8 with one attention DP group")
        if get_global_expert_location_metadata() is not None:
            raise ValueError("LongCat engines do not support expert placement updates")
        if global_server_args_dict["mla_lora_scale"] != "folded":
            raise ValueError("LongCat engines require --mla-lora-scale folded")
        if global_server_args_dict["numerics"] != "auto":
            raise ValueError("LongCat engines require --numerics auto")


def _normalize_pro_oe(config):
    """Normalize LongCat HF config aliases used by the runtime layers."""

    oe_ratio = getattr(config, "ngram_vocab_size_ratio", None)
    if oe_ratio is None:
        oe_ratio = getattr(config, "oe_vocab_size_ratio", None)
    config.use_over_embedding = bool(
        getattr(config, "use_over_embedding", False) or oe_ratio is not None
    )
    if (
        config.use_over_embedding
        and getattr(config, "over_embedding_m", None) is None
        and oe_ratio is None
    ):
        raise ValueError(
            "LongCat OE requires ngram_vocab_size_ratio or oe_vocab_size_ratio"
        )
    if config.use_over_embedding and getattr(config, "over_embedding_m", None) is None:
        config.over_embedding_m = int(config.vocab_size * oe_ratio)
    config.oe_ignore_tokens = list(
        getattr(config, "oe_ignore_tokens", None)
        or getattr(config, "oe_ignored_token_ids", None)
        or []
    )

    return config


class LongcatProDecoderLayer(_ClassicDecoderLayer):
    def __init__(
        self,
        config: _PretrainedConfig,
        layer_id: int,
        mapping: _Mapping,
        quant_config: _QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.mapping = mapping
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.stream_fork = _StreamFork(alt_stream)

        rope_theta = _get_rope_theta(config)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling and "factor" not in rope_scaling:
            rope_scaling = None
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.uses_lsa = True
        q_lora_scale, kv_lora_scale = (
            _lora_norm_scales(config)
            if global_server_args_dict["mla_lora_scale"] == "runtime"
            else (None, None)
        )

        self.self_attn = nn.ModuleList(
            [
                _DSAAttention(
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
                    computes_selection=branch_id == 0,
                    selection_owner_layer_id=layer_id * 2,
                    initial_tokens=config.index_init_tokens,
                    local_tokens=config.index_local_tokens,
                    index_k_norm_type=config.index_k_norm_type,
                    indexer_rope_is_neox_style=False,
                    lora_norm_eps=config.rms_norm_eps,
                    decode_query_length_limit=8,
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
                    batch_invariant=(
                        global_server_args_dict["tp_batch_invariant"] == "attn+dense"
                    ),
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

        self.attn0_engine: LongcatAttention0Engine | None = None
        self.layer_engine: LongcatLayerEngine | None = None
        self.mlp0_weights: LongcatMlpWeights | None = None
        self.mlp1_weights: LongcatMlpWeights | None = None
        self.expert_weights: LongcatExpertWeights | None = None
        self.output_prenormed = False

    def initialize_engines(self, next_norm) -> None:
        from tokenspeed.runtime.models.longcat.attention0 import LongcatAttention0Engine
        from tokenspeed.runtime.models.longcat.layer import LongcatLayerEngine
        from tokenspeed.runtime.models.longcat.mlp import LongcatMlpWeights
        from tokenspeed.runtime.models.longcat.moe import LongcatExpertWeights

        self.attn0_engine = LongcatAttention0Engine(
            self.self_attn[0], self.post_attention_layernorm[0], self.mapping
        )
        self.mlp0_weights = LongcatMlpWeights(
            self.mlps[0], self.input_layernorm[1], moe=self.mlp
        )
        self.mlp1_weights = LongcatMlpWeights(self.mlps[1], next_norm)
        self.expert_weights = LongcatExpertWeights(self.mlp, self.mapping)
        self.layer_engine = LongcatLayerEngine(
            self.self_attn[1],
            self.post_attention_layernorm[1],
            self.mapping,
            self.mlp0_weights,
            self.mlp1_weights,
            self.expert_weights,
        )

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
        self, hidden_states, residual, ctx, num_global_tokens, max_num_tokens_per_gpu
    ):
        hidden_states = self._to_moe_rows(hidden_states, ctx)
        hidden_states = self.moe_comm.pre_mlp_comm(hidden_states, ctx)
        hidden_states = self.mlp(
            hidden_states, num_global_tokens, max_num_tokens_per_gpu
        )
        hidden_states, residual = self.moe_comm.post_mlp_fused(
            hidden_states, residual, ctx
        )
        hidden_states = self._to_dense_rows(hidden_states, ctx)
        return hidden_states, residual

    def forward(
        self,
        positions,
        hidden_states,
        ctx,
        residual,
        pending_local_add=None,
        input_prenormed=False,
        aux_hidden_states=None,
        *,
        target_capture_idx: int | None,
    ):
        num_global_tokens, max_num_tokens_per_gpu = self.moe_comm.get_num_tokens(ctx)
        self.output_prenormed = False
        idle = ctx.forward_mode.is_idle()
        if not idle and not input_prenormed:
            if pending_local_add is not None:
                hidden_states = hidden_states + pending_local_add
            hidden_states, residual = self.branch_comm[0].input_reduce_norm(
                hidden_states, residual
            )
        _capture_layer_input(
            self.branch_comm[0],
            hidden_states if idle else residual,
            ctx,
            aux_hidden_states,
        )
        if (
            aux_hidden_states is not None
            and target_capture_idx is not None
            and ctx.target_capture_sink is not None
            and len(aux_hidden_states) == target_capture_idx + 1
        ):
            # Queue the tap's aux work before this layer launches its engines.
            ctx.target_capture_sink.on_target_capture(
                target_capture_idx, aux_hidden_states[target_capture_idx]
            )
        if idle:
            hidden_states, residual = self._forward_moe(
                hidden_states, residual, ctx, num_global_tokens, max_num_tokens_per_gpu
            )
            return hidden_states, residual, None

        if self.attn0_engine is not None and self.attn0_engine.eligible(
            self.self_attn[0], hidden_states, ctx, self.moe_comm
        ):
            scales = (
                self.mlp0_weights.input_scale_inv,
                self.expert_weights.input_scale_inv,
            )
            hidden_states, residual, selection = self.attn0_engine.forward(
                positions, hidden_states, residual, ctx, quant_scales=scales
            )
            packed = self.attn0_engine.quantized_outputs(scales)
            if any(value is None for value in packed):
                raise RuntimeError(
                    "LongCat layer requires both packed attention0 outputs"
                )
            hidden_states, residual = self.layer_engine.forward(
                hidden_states, residual, ctx, selection, packed
            )
            self.output_prenormed = True
            return hidden_states, residual, None

        hidden_states, selection = self.self_attn[0].forward_with_selection(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            comm_manager=self.branch_comm[0],
            selection=None,
        )
        hidden_states, residual = self.branch_comm[0].post_attn_reduce_norm(
            hidden_states, residual, ctx
        )
        branch_input, branch_residual = hidden_states, residual
        overlap = _get_is_capture_mode() and self.moe_comm.should_fuse(
            branch_input.shape[0]
        )
        with self.stream_fork.scope(enable=overlap) as fork:
            with fork.branch():
                moe_hidden_states, _ = self._forward_moe(
                    branch_input,
                    branch_residual,
                    ctx,
                    num_global_tokens,
                    max_num_tokens_per_gpu,
                )
            hidden_states, residual = self._forward_dense_mlp(
                0, branch_input, branch_residual, ctx
            )
            hidden_states, residual = self.branch_comm[1].intra_layer_add_norm(
                hidden_states, residual
            )
            hidden_states, _ = self.self_attn[1].forward_with_selection(
                positions=positions,
                hidden_states=hidden_states,
                ctx=ctx,
                comm_manager=self.branch_comm[1],
                selection=selection,
            )
            hidden_states, residual = self.branch_comm[1].post_attn_reduce_norm(
                hidden_states, residual, ctx
            )
            hidden_states, residual = self._forward_dense_mlp(
                1, hidden_states, residual, ctx
            )
        return hidden_states, residual, moe_hidden_states


class LongcatProModel(nn.Module):
    fall_back_to_pt_during_load = False
    decoder_layer_cls = LongcatProDecoderLayer

    def __init__(
        self,
        config: _PretrainedConfig,
        mapping: _Mapping,
        quant_config: _QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        _ensure_longcat_config(config)
        _normalize_pro_oe(config)
        _validate_pro_launch(mapping)
        self.weight_state = LongcatWeightState(quant_config)
        self.engines_initialized = False
        self.mapping = mapping
        self.uses_lsa = _is_longcat_lsa_config(config, config)
        if not self.uses_lsa:
            if engines_requested():
                raise ValueError("LongCat engines require an LSA checkpoint")
            self.decoder_layer_cls = _ClassicDecoderLayer
        self.padding_id = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        self.embed_tokens = self._build_embed_tokens(config)
        self.alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.layers = nn.ModuleList(
            [
                self.decoder_layer_cls(
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
        self._dflash_capture_idx_map: dict[int, int] = {}

    def _build_embed_tokens(self, config):
        if not config.use_over_embedding:
            return _VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                tp_rank=self.mapping.attn.tp_rank,
                tp_size=self.mapping.attn.tp_size,
                tp_group=self.mapping.attn.tp_group,
            )

        max_ngram_order, hashes_per_order = _resolve_longcat_oe_hyperparameters(config)
        return _LongCatOverEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            over_embedding_m=config.over_embedding_m,
            hashes_per_order=hashes_per_order,
            max_ngram_order=max_ngram_order,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
            ignored_token_ids=tuple(config.oe_ignore_tokens),
            eos_token_id=getattr(config, "eos_token_id", None),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: _ForwardContext,
        request_token_history: RequestTokenHistoryView | None = None,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if input_embeds is not None:
            hidden_states = input_embeds
        elif isinstance(self.embed_tokens, _LongCatOverEmbedding):
            hidden_states = self.embed_tokens(input_ids, request_token_history)
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None
        pending_local_add = None
        aux_hidden_states = [] if self.layers_to_capture else None
        # The attention0 engines share one int32 view of positions / slots /
        # block table per forward; drop the previous forward's here.
        engine = self.layers[0].attn0_engine if self.uses_lsa and self.layers else None
        if engine is not None:
            engine.begin_forward()
        layer = None
        prenormed = False
        for layer_id, layer in enumerate(self.layers):
            captures = aux_hidden_states if layer_id in self.layers_to_capture else None
            if self.uses_lsa:
                hidden_states, residual, pending_local_add = layer(
                    positions,
                    hidden_states,
                    ctx,
                    residual,
                    pending_local_add,
                    input_prenormed=prenormed,
                    aux_hidden_states=captures,
                    target_capture_idx=self._dflash_capture_idx_map.get(layer_id),
                )
            else:
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    ctx,
                    residual,
                    aux_hidden_states=captures,
                )
                if captures is not None and ctx.target_capture_sink is not None:
                    notify_target_capture(
                        ctx.target_capture_sink,
                        self._dflash_capture_idx_map,
                        layer_id,
                        captures,
                    )
            prenormed = self.uses_lsa and layer.output_prenormed

        final_capture_layer = len(self.layers)
        capture_final = final_capture_layer in self.layers_to_capture
        if not ctx.forward_mode.is_idle() and layer is not None and not prenormed:
            if pending_local_add is not None:
                hidden_states = hidden_states + pending_local_add
            hidden_states, final_residual = layer.final_norm_comm.final_norm(
                hidden_states,
                residual,
                ctx,
                self.norm,
            )
        else:
            final_residual = residual if residual is not None else hidden_states
        if capture_final and layer is not None:
            _capture_layer_input(
                layer.final_norm_comm, final_residual, ctx, aux_hidden_states
            )
            if ctx.target_capture_sink is not None:
                notify_target_capture(
                    ctx.target_capture_sink,
                    self._dflash_capture_idx_map,
                    final_capture_layer,
                    aux_hidden_states,
                )
        return hidden_states, aux_hidden_states

    def initialize_engines(self) -> None:
        if self.engines_initialized:
            return
        if not engines_requested():
            self.engines_initialized = True
            return
        layers = list(self.layers)
        error = None
        try:
            for index, layer in enumerate(layers):
                next_norm = (
                    layers[index + 1].input_layernorm[0]
                    if index + 1 < len(layers)
                    else self.norm
                )
                layer.initialize_engines(next_norm)
                if not layer.moe_comm.should_fuse(8):
                    raise ValueError(
                        "LongCat layer engine requires fused all-reduce normalization"
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        flag = torch.tensor(
            [int(error is None)], dtype=torch.int32, device=torch.cuda.current_device()
        )
        torch.distributed.all_reduce(
            flag,
            op=torch.distributed.ReduceOp.MIN,
            group=_pg_manager.get_device_process_group(self.mapping.attn.tp_group),
        )
        if int(flag.item()) == 0:
            for layer in layers:
                layer.attn0_engine = layer.layer_engine = None
                layer.mlp0_weights = layer.mlp1_weights = layer.expert_weights = None
            raise ValueError(
                error or "LongCat layer engine initialization failed on a peer rank"
            )
        self.engines_initialized = True
        _longcat_logger.info(
            f"LongCat layer engine enabled on {len(layers)} layers (TP rank {self.mapping.attn.tp_rank})"
        )
