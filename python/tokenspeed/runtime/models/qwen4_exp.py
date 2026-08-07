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

"""Inference-only Qwen4-Exp text and multimodal models."""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable

import torch
from torch import nn

from tokenspeed.runtime.configs.base_config import get_rope_parameters
from tokenspeed.runtime.configs.qwen4_exp_config import (
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
)
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.backends.qwen4_exp import (
    bind_qwen4_exp_side_state,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.attention.linear.layernorm_gated import rmsnorm_fn
from tokenspeed.runtime.layers.hyperconnection import (
    GatedResidualSimple,
    HyperConnectionConfig,
)
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm
from tokenspeed.runtime.layers.linear import QKVParallelLinear, RowParallelLinear
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.qwen4_exp_ple import (
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
    quantize_ple_embedding_rows,
)
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
)
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.models.qwen3_5 import (
    Qwen3_5AttentionDecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
    Qwen3_5LinearDecoderLayer,
    _gdn_in_proj_stacked_mapping,
)
from tokenspeed.runtime.models.qwen3_5_moe import (
    Qwen3_5MoeMLP,
    Qwen3_5MoeSparseMoeBlock,
)
from tokenspeed.runtime.models.utils import validate_attention_partition
from tokenspeed.runtime.moe.distribution_recorder import (
    get_global_expert_distribution_recorder,
)
from tokenspeed.runtime.moe.expert_location import ModelConfigForExpertLocation
from tokenspeed.runtime.utils import add_prefix

logger = logging.getLogger(__name__)


def _qwen4_exp_uses_sigmoid_output_gate(config: Qwen4ExpTextConfig) -> bool:
    """Return whether the checkpoint expects a sigmoid-only GDN output gate."""

    return getattr(config, "output_gate_type", None) == "sigmoid"


def _qwen4_exp_uses_sparse_moe(config: Qwen4ExpTextConfig) -> bool:
    """Return whether the Qwen4-Exp checkpoint has sparse experts."""

    return getattr(config, "num_experts", None) is not None


def _build_qwen4_exp_mlp(
    config: Qwen4ExpTextConfig,
    mapping: Mapping,
    layer_id: int,
    quant_config: QuantizationConfig | None,
    alt_stream: torch.cuda.Stream | None,
    prefix: str,
) -> tuple[nn.Module, bool]:
    """Build a Qwen4-Exp dense or sparse MLP without changing Qwen3.5."""

    is_moe = _qwen4_exp_uses_sparse_moe(config)
    if is_moe:
        mlp = Qwen3_5MoeSparseMoeBlock(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            layer_index=layer_id,
            alt_stream=alt_stream,
            prefix=add_prefix("mlp", prefix),
        )
    else:
        mlp = Qwen3_5MoeMLP(
            mapping=mapping,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("mlp", prefix),
        )
    return mlp, is_moe


class _Qwen4ExpRMSNormGated(nn.Module):
    """Qwen4-Exp per-head RMS normalization with a sigmoid output gate.

    CUDA runs the fused one-pass gated layer-norm kernel (single launch for
    norm + sigmoid gate + weight); the eager fallback below is the reference
    implementation kept for CPU tensors.
    """

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        if x.is_cuda:
            return rmsnorm_fn(
                x,
                self.weight,
                None,
                z=z,
                eps=self.eps,
                norm_before_gate=True,
                sigmoid_gate=True,
            )
        input_dtype = x.dtype
        value = x.float()
        variance = value.square().mean(dim=-1, keepdim=True)
        value = value * torch.rsqrt(variance + self.eps)
        if z is not None:
            value = value * torch.sigmoid(z.float())
        return (value * self.weight.float()).to(input_dtype)


class _Qwen4ExpDecoderMixin:
    """Hyper-connection and PLE behavior shared by both decoder layer kinds."""

    @staticmethod
    def _uses_sparse_moe(config: Qwen4ExpTextConfig) -> bool:
        return _qwen4_exp_uses_sparse_moe(config)

    @staticmethod
    def _uses_attention_output_gate(config: Qwen4ExpTextConfig) -> bool:
        return True

    def _init_qwen4_exp_extensions(
        self,
        config: Qwen4ExpTextConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        self.hc_count = int(config.hc_count)
        self.hidden_size = int(config.hidden_size)
        hc_config = HyperConnectionConfig(
            hc_count=self.hc_count,
            hidden_size=self.hidden_size,
            hc_lowrank=int(config.hc_lowrank),
            rms_norm_eps=float(config.rms_norm_eps),
            params_dtype=torch.get_default_dtype(),
            hc_per_branch_norm=True,
        )
        self.attn_hyper_connection = GatedResidualSimple(hc_config)
        self.mlp_hyper_connection = GatedResidualSimple(hc_config)
        self.ple = None
        one_based_layer_id = layer_id + 1
        ple_ids = sorted({int(value) for value in config.ple_layer_ids})
        if one_based_layer_id in ple_ids:
            layer_prefix = prefix.replace(".self_attn", "").replace(".linear_attn", "")
            self.ple = Qwen4ExpPLELayer(
                config=config,
                mapping=mapping,
                layer_id=layer_id,
                ple_layer_index=ple_ids.index(one_based_layer_id),
                quant_config=quant_config,
                prefix=add_prefix("ple", layer_prefix),
            )

    def _prepare_attention(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        ctx: ForwardContext,
    ):
        expected = self.hc_count * self.hidden_size
        if hidden_states.shape[-1] == self.hidden_size:
            hidden_states = hidden_states.repeat(1, self.hc_count)
        elif hidden_states.shape[-1] != expected:
            raise RuntimeError(
                f"Qwen4-Exp expected hidden width {self.hidden_size} or "
                f"{expected}, got {hidden_states.shape[-1]}"
            )
        hidden_states = self.comm_manager.pre_attn_comm(hidden_states, ctx)
        if self.ple is not None:
            # The PLE layer folds this residual add into its conv epilogue.
            hidden_states = self.ple(hidden_states, input_ids, ctx)
        return self.attn_hyper_connection.mix(hidden_states)

    def _finish_attention(
        self,
        attention_output: torch.Tensor,
        residuals,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        if ctx.forward_mode.is_idle():
            return self.attn_hyper_connection.combine(attention_output, residuals)
        attention_output, residual = self.comm_manager.post_attn_comm(
            attention_output, residuals[0], ctx
        )
        return self.attn_hyper_connection.combine(
            attention_output,
            self.attn_hyper_connection.norm_for(residual, residuals),
        )

    def _run_mlp(self, hidden_states: torch.Tensor, ctx: ForwardContext):
        mixed, residuals = self.mlp_hyper_connection.mix(hidden_states)
        num_global_tokens, max_tokens_per_gpu = self.comm_manager.get_num_tokens(ctx)
        if self.is_moe:
            deferred_reduce = (
                not self.mlp.use_deepep
                and self.mlp.comm_manager.should_fuse(mixed.shape[0])
            )
            output = self.mlp(mixed, num_global_tokens, max_tokens_per_gpu, ctx)
            if deferred_reduce:
                output, _ = self.mlp.comm_manager.post_mlp_comm(
                    output, residuals[0], ctx
                )
        else:
            mixed = self.comm_manager.pre_mlp_comm(mixed, ctx)
            output = self.mlp(mixed)
            output, _ = self.comm_manager.post_mlp_comm(output, residuals[0], ctx)
        return self.mlp_hyper_connection.combine(output, residuals)


class Qwen4ExpLinearDecoderLayer(_Qwen4ExpDecoderMixin, Qwen3_5LinearDecoderLayer):
    """Qwen4-Exp GDN layer with a hyper-connection residual stream."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.mapping = mapping
        self.layer_id = layer_id
        self.linear_attn = Qwen3_5GatedDeltaNet(
            config, mapping, layer_id, quant_config, prefix=prefix
        )
        self.mlp, self.is_moe = _build_qwen4_exp_mlp(
            config,
            mapping,
            layer_id,
            quant_config,
            alt_stream,
            prefix.replace(".linear_attn", ""),
        )
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=self.is_moe,
            prev_is_moe=self.is_moe,
        )
        if _qwen4_exp_uses_sigmoid_output_gate(config):
            self.linear_attn.norm = _Qwen4ExpRMSNormGated(
                self.linear_attn.head_v_dim,
                float(config.rms_norm_eps),
            )
        self._init_qwen4_exp_extensions(config, mapping, layer_id, quant_config, prefix)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        **kwargs,
    ):
        del residual, kwargs
        mixed, residuals = self._prepare_attention(hidden_states, input_ids, ctx)
        attention_output = (
            mixed if ctx.forward_mode.is_idle() else self.linear_attn(mixed, ctx)
        )
        hidden_states = self._finish_attention(attention_output, residuals, ctx)
        return self._run_mlp(hidden_states, ctx), None


class Qwen4ExpAttentionDecoderLayer(
    _Qwen4ExpDecoderMixin, Qwen3_5AttentionDecoderLayer
):
    """Qwen4-Exp full-attention layer with HC and optional QSA indexer."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.mapping = mapping
        self.hidden_size = config.hidden_size
        self.attn_tp_rank = mapping.attn.tp_rank
        self.attn_tp_size = mapping.attn.tp_size
        self.attn_tp_group = mapping.attn.tp_group
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        validate_attention_partition(
            self.total_num_heads,
            self.total_num_kv_heads,
            self.attn_tp_size,
        )
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.rope_scaling = get_rope_parameters(config)
        self.rope_theta = self.rope_scaling.get("rope_theta", 10000)
        self.partial_rotary_factor = self.rope_scaling.get("partial_rotary_factor", 1.0)
        self.layer_id = layer_id
        self.attn_output_gate = self._uses_attention_output_gate(config)
        if self.attn_output_gate:
            logger.warning_once("using attn output gate!")
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            rope_scaling=self.rope_scaling,
            base=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )
        attn_quant_config = (
            None
            if quant_config and quant_config.get_name() == "nvfp4"
            else quant_config
        )
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=attn_quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            tp_group=self.attn_tp_group,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=attn_quant_config,
            reduce_results=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            tp_group=self.attn_tp_group,
            prefix=add_prefix("o_proj", prefix),
        )
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            group_id=FULL_ATTENTION,
        )
        self.mlp, self.is_moe = _build_qwen4_exp_mlp(
            config,
            mapping,
            layer_id,
            quant_config,
            alt_stream,
            prefix.replace(".self_attn", ""),
        )
        self.input_layernorm = None
        self.post_attention_layernorm = None
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=self.is_moe,
            prev_is_moe=self.is_moe,
        )
        self.indexer = None
        if getattr(config, "indexer_n_heads", None) is not None:
            from tokenspeed.runtime.layers.attention.qsa import QSAIndexer

            self.indexer = QSAIndexer(
                config=config,
                mapping=mapping,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("indexer", prefix),
                rotary_emb=self.rotary_emb,
            )
        self._init_qwen4_exp_extensions(config, mapping, layer_id, quant_config, prefix)

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v, gate = self._project_qkv_rope(positions, hidden_states)
        if self.indexer is not None:
            topk_indices = self.indexer(hidden_states, positions, ctx)
            attention_output = self._qsa_attention(
                q=q,
                k=k,
                v=v,
                gate=gate,
                attention_layer=self.attn,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
                topk_indices=topk_indices,
            )
        else:
            attention_output = self._attn(q, k, v, gate, ctx, out_cache_loc)
        output, _ = self.o_proj(attention_output)
        return output

    def _qsa_attention(self, **kwargs) -> torch.Tensor:
        """Sparse-attention hook specialized by the MTP draft layer."""

        return self.indexer.sparse_attention(**kwargs)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_ids: torch.Tensor,
        **kwargs,
    ):
        del residual, kwargs
        mixed, residuals = self._prepare_attention(hidden_states, input_ids, ctx)
        attention_output = (
            mixed
            if ctx.forward_mode.is_idle()
            else self.self_attention(positions, mixed, ctx, out_cache_loc)
        )
        hidden_states = self._finish_attention(attention_output, residuals, ctx)
        return self._run_mlp(hidden_states, ctx), None


class Qwen4ExpModel(Qwen3_5ForCausalLM):
    """Qwen4-Exp decoder backbone returning logits-width and HC-width states."""

    ATTENTION_LAYER_CLS = Qwen4ExpAttentionDecoderLayer
    LINEAR_LAYER_CLS = Qwen4ExpLinearDecoderLayer

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, mapping, quant_config, prefix)
        self.norm = None
        hc_config = HyperConnectionConfig(
            hc_count=int(config.hc_count),
            hidden_size=int(config.hidden_size),
            hc_lowrank=int(config.hc_lowrank),
            rms_norm_eps=float(config.rms_norm_eps),
            params_dtype=torch.get_default_dtype(),
            hc_per_branch_norm=True,
        )
        self.hyper_connection_mixer = GatedResidualSimple(hc_config, use_combine=False)
        self.ple_layers = tuple(
            layer.ple
            for layer in self.layers
            if getattr(layer, "ple", None) is not None
        )
        self.qsa_indexers = tuple(
            layer.indexer
            for layer in self.layers
            if getattr(layer, "indexer", None) is not None
        )

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        """Load per-tensor FP8 KV scales for full-attention layers."""

        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            self.mapping.attn.tp_rank,
            self.mapping.attn.tp_size,
            self.config.num_hidden_layers,
            self.config.model_type,
        ):
            paged_attention = getattr(self.layers[layer_idx], "attn", None)
            if paged_attention is None:
                continue
            scale = float(scaling_factor)
            paged_attention.k_scale = scale
            paged_attention.v_scale = scale
            paged_attention.k_scale_float = scale
            paged_attention.v_scale_float = scale

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        pp_proxy_tensors=None,
        input_deepstack_embeds: torch.Tensor | None = None,
    ):
        del pp_proxy_tensors
        bind_qwen4_exp_side_state(
            ctx.attn_backend,
            self.ple_layers,
            self.qsa_indexers,
        )
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        residual = None
        for layer_id, layer in enumerate(self.layers):
            with get_global_expert_distribution_recorder().with_current_layer(layer_id):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    ctx=ctx,
                    out_cache_loc=out_cache_loc,
                    input_ids=input_ids,
                )
            if (
                input_deepstack_embeds is not None
                and input_deepstack_embeds.numel()
                and layer_id < 3
            ):
                start = self.hidden_size * layer_id
                deepstack = input_deepstack_embeds[
                    :, start : start + self.hidden_size
                ].repeat(1, self.config.hc_count)
                hidden_states.add_(deepstack)

        if self.layers:
            hidden_states, _ = self.layers[-1].comm_manager.post_final_norm_comm(
                hidden_states, hidden_states, ctx
            )
        hc_hidden_states = hidden_states
        hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)
        return hidden_states, [hc_hidden_states]


def _normalize_checkpoint_name(name: str) -> str:
    if "language_model" in name:
        name = name.replace("model.language_model.", "model.")
    if ".self_attn." in name:
        name = name.replace(".self_attn", "")
    if "visual" in name:
        name = name.replace("attn.qkv.", "attn.qkv_proj.")
        name = name.replace("model.visual.", "visual.")
    return name


def _copy_ple_shard(
    module: nn.Module,
    name: str,
    loaded_weight: torch.Tensor,
    split_parts: int,
) -> str | None:
    match = re.search(r"\.ngram_embedding\.shard_(\d+)\.weight$", name)
    if match is None:
        return None
    module_name = name[: name.index(".ngram_embedding.shard_")]
    ple_modules = dict(module.named_modules())
    ple_embedding = ple_modules.get(module_name)
    if not isinstance(ple_embedding, Qwen4ExpNGramEmbedding):
        return None
    embedding = ple_embedding.ngram_embedding
    shard_index = int(match.group(1))
    shard_size = (embedding.org_vocab_size + split_parts - 1) // split_parts
    row_start = shard_index * shard_size
    row_end = row_start + loaded_weight.shape[0]
    tp_start = embedding.shard_indices.org_vocab_start_index
    tp_end = embedding.shard_indices.org_vocab_end_index
    overlap_start = max(row_start, tp_start)
    overlap_end = min(row_end, tp_end)
    if overlap_start < overlap_end:
        destination = overlap_start - tp_start
        source = overlap_start - row_start
        rows = overlap_end - overlap_start
        source_rows = loaded_weight[source : source + rows]
        target_rows = embedding.weight.data[destination : destination + rows]
        scale_buffer = getattr(ple_embedding, "ngram_embedding_scale", None)
        source_is_fp8 = source_rows.dtype == torch.float8_e4m3fn
        target_is_fp8 = target_rows.dtype == torch.float8_e4m3fn
        # Matching formats are copied unchanged. In particular, FP8-to-FP8
        # preserves the checkpoint payload; its global scale is loaded by
        # _load_ple_weight_scale independently of checkpoint weight ordering.
        if target_is_fp8 and not source_is_fp8:
            if scale_buffer is None:
                raise RuntimeError("FP8 PLE embedding is missing its scale buffer")
            # Quantize compute-dtype checkpoint rows for FP8 storage and retain
            # their independently derived dequant scales.
            source_rows, scale = quantize_ple_embedding_rows(source_rows)
            scale_buffer[destination : destination + rows].copy_(
                scale.to(scale_buffer.device, scale_buffer.dtype)
            )
        elif source_is_fp8 and not target_is_fp8:
            source_rows = (
                source_rows.to(torch.float32) * ple_embedding._checkpoint_weight_scale
            )
        target_rows.copy_(source_rows.to(target_rows.device, target_rows.dtype))
    return f"{module_name}.ngram_embedding.weight"


def _load_ple_weight_scale(
    module: nn.Module,
    name: str,
    loaded_weight: torch.Tensor,
) -> str | None:
    """Apply a checkpoint-wide scale for split, pre-quantized PLE shards."""

    suffix = ".ngram_embedding.weight_scale"
    if not name.endswith(suffix):
        return None
    module_name = name[: -len(suffix)]
    ple_embedding = dict(module.named_modules()).get(module_name)
    if not isinstance(ple_embedding, Qwen4ExpNGramEmbedding):
        return None
    if loaded_weight.numel() != 1:
        raise ValueError(
            f"Qwen4-Exp PLE weight scale must be scalar, got "
            f"{tuple(loaded_weight.shape)} for {name}"
        )
    scale = float(loaded_weight.to(torch.float32).item())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"Qwen4-Exp PLE weight scale must be finite and positive, got "
            f"{scale} for {name}"
        )

    scale_buffer = getattr(ple_embedding, "ngram_embedding_scale", None)
    if scale_buffer is not None:
        # The checkpoint scale is shared by every pre-quantized row, so one
        # fill handles both scale-before-shards and scale-after-shards order.
        scale_buffer.fill_(scale)
    else:
        # Compute-dtype target: rescale any raw FP8 rows copied before the
        # scale tensor. Future shard copies multiply by the new value.
        ple_embedding.ngram_embedding.weight.data.mul_(
            scale / ple_embedding._checkpoint_weight_scale
        )
    ple_embedding._checkpoint_weight_scale = scale
    return f"{module_name}.ngram_embedding.weight"


def load_qwen4_exp_weights(
    module: nn.Module,
    config: Qwen4ExpTextConfig,
    mapping: Mapping,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    include_visual: bool,
) -> set[str]:
    """Load dense/MoE, PLE, QSA, and optional vision checkpoint tensors."""

    params = dict(module.named_parameters(remove_duplicate=False))
    buffers = dict(module.named_buffers())
    stacked = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
        # PLE fuses key_proj/value_proj into one kv_proj GEMM; route the
        # checkpoint shards to the merged parameter's row ranges.
        ("kv_proj", "key_proj", "key"),
        ("kv_proj", "value_proj", "value"),
        # The HC mix-down and inject projections share one fused GEMM; route
        # their checkpoint tensors to that parameter's row ranges.
        ("mix_inject_proj", "input_mix_weight_down", "mix"),
        ("mix_inject_proj", "block_inject_weight", "inject"),
    ]
    stacked += _gdn_in_proj_stacked_mapping(params)
    num_experts = getattr(config, "num_experts", None)
    moe_loader = None
    if num_experts is not None:
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
            ),
            fused_schema=ExpertCheckpointSchema(
                gate_up_fused_name="gate_up_proj",
                down_proj_name="down_proj",
            ),
            num_experts=int(num_experts),
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
        )
    loaded = set()
    ignored_suffixes = (
        ".bias",
        "_bias",
        ".k_scale",
        "_k_scale",
        ".v_scale",
        "_v_scale",
        ".weight_scale",
        "_weight_scale",
        ".input_scale",
        "_input_scale",
    )
    split_parts = int(getattr(config, "split_ngram_parts", 512))
    for original_name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in original_name or "mtp" in original_name:
            continue
        if "visual" in original_name and not include_visual:
            continue
        name = _normalize_checkpoint_name(original_name)
        buffer_name = name.rsplit(".", 1)[-1]
        if ".ple.ple_embedding." in name and (
            buffer_name == "token_lookup" or buffer_name.startswith("hashstats_")
        ):
            continue
        if name in buffers and buffer_name in {
            "layer_multipliers",
            "ngram_heads_offsets",
            "ngram_heads_vocab_sizes",
        }:
            buffer = buffers[name]
            if buffer.shape != loaded_weight.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: expected {tuple(buffer.shape)}, "
                    f"got {tuple(loaded_weight.shape)}"
                )
            buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
            loaded.add(name)
            continue
        scale_name = _load_ple_weight_scale(module, name, loaded_weight)
        if scale_name is not None:
            loaded.add(scale_name)
            continue
        shard_name = _copy_ple_shard(module, name, loaded_weight, split_parts)
        if shard_name is not None:
            loaded.add(shard_name)
            continue
        for param_name, weight_name, shard_id in stacked:
            if weight_name not in name or "mlp.experts" in name or "visual" in name:
                continue
            mapped = name.replace(weight_name, param_name)
            if mapped not in params:
                continue
            loader = getattr(params[mapped], "weight_loader", default_weight_loader)
            loader(params[mapped], loaded_weight, shard_id)
            loaded.add(mapped)
            break
        else:
            if moe_loader is not None and moe_loader.matches(name):
                loaded.add(moe_loader.load(name, loaded_weight))
                continue
            if moe_loader is not None and moe_loader.is_expert_checkpoint_weight(name):
                continue
            if name.endswith(ignored_suffixes) and name not in params:
                continue
            if name not in params:
                logger.warning("Qwen4-Exp parameter %s was not found", name)
                continue
            loader = getattr(params[name], "weight_loader", default_weight_loader)
            loader(params[name], loaded_weight)
            loaded.add(name)
    return loaded


class Qwen4ExpForCausalLM(BaseCausalLM):
    """Text-only Qwen4-Exp entry class."""

    model_cls = Qwen4ExpModel

    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        text_config = getattr(config, "text_config", config)
        super().__init__(text_config, mapping, quant_config, prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return load_qwen4_exp_weights(
            self, self.config, self.mapping, weights, include_visual=False
        )

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        config = getattr(config, "text_config", config)
        if getattr(config, "num_experts", None) is None:
            return None
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


class Qwen4ExpForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Multimodal Qwen4-Exp entry class using the Qwen3.5 vision tower."""

    model_cls = Qwen4ExpModel

    def __init__(
        self,
        config: Qwen4ExpConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__(
            config,
            mapping,
            quant_config,
            prefix,
            is_multimodal_active,
            mm_attention_backend,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return load_qwen4_exp_weights(
            self,
            self.config,
            self.mapping,
            weights,
            include_visual=self.is_multimodal_active,
        )

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        if self.model is not None:
            self.model.load_kv_cache_scales(quantization_param_path)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return Qwen4ExpForCausalLM.get_model_config_for_expert_location(config)


EntryClass = [Qwen4ExpForConditionalGeneration, Qwen4ExpForCausalLM]
