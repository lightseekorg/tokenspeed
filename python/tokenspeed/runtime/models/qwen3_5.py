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

"""Inference-only Qwen3.5 model and Qwen3.5 MoE model compatible with HuggingFace weights."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
import torch.nn as nn
import triton
import triton.language as tl
from tokenspeed_kernel.ops.activation.triton import sigmoid_mul
from tokenspeed_kernel.ops.layernorm.triton import (
    fused_qk_rmsnorm_rope_gate,
    qk_rmsnorm,
)

# Configs
from tokenspeed.runtime.configs.qwen3_5_config import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
)

# Distributed
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext

# Layers - Attention
from tokenspeed.runtime.layers.attention.linear.layernorm_gated import (
    RMSNorm as RMSNormGated,
)

# Layers - Others
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm

# Layers - Linear
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata, LogitsProcessor
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.parameter import (
    BlockQuantScaleParameter,
    PerTensorScaleParameter,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import (
    default_weight_loader,
    mamba_v2_sharded_weight_loader,
    sharded_weight_loader,
)
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.models.qwen3_5_moe import (
    Qwen3_5MoeMLP,
    Qwen3_5MoeSparseMoeBlock,
)
from tokenspeed.runtime.models.qwen3_vision import Qwen3VLMoeVisionModel
from tokenspeed.runtime.moe.distribution_recorder import (
    get_global_expert_distribution_recorder,
)
from tokenspeed.runtime.moe.expert_location import ModelConfigForExpertLocation
from tokenspeed.runtime.multimodal.embedder import (
    EncoderSpec,
    VisionEmbedder,
    pad_input_tokens,
)
from tokenspeed.runtime.multimodal.encoder_cudagraph import EncoderCudaGraphWrapper
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from tokenspeed.runtime.utils import (
    add_prefix,
    make_layers,
    set_weight_attrs,
)
from tokenspeed.runtime.utils.cuda_stream import StreamFork

logger = logging.getLogger(__name__)


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.attn_tp_rank = mapping.attn.tp_rank
        self.attn_tp_size = mapping.attn.tp_size
        self.attn_tp_group = mapping.attn.tp_group
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_id = layer_id
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        # Conv1d layer
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            quant_config=None,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            tp_group=self.attn_tp_group,
            prefix=add_prefix("conv1d", prefix),
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj_qkvzba = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[
                self.key_dim,
                self.key_dim,
                self.value_dim,
                self.value_dim,
                self.num_v_heads,
                self.num_v_heads,
            ],
            bias=False,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            tp_group=self.attn_tp_group,
            prefix=add_prefix("in_proj_qkvzba", prefix),
        )
        self._qkvz_dim = (self.key_dim * 2 + self.value_dim * 2) // self.attn_tp_size
        self._ba_dim = (self.num_v_heads * 2) // self.attn_tp_size

        # Override weight loaders for packed checkpoint format.
        # Important: for FP8, this must cover not only `.weight` but also
        # `weight_scale_inv` / `weight_scale` / `input_scale` if present.
        self._bind_packed_weight_loaders(self.in_proj_qkvzba)

        # Conv1d weight loader setup
        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": mamba_v2_sharded_weight_loader(
                    [
                        query_key_settings,
                        query_key_settings,
                        value_settings,
                    ],
                    self.attn_tp_size,
                    self.attn_tp_rank,
                )
            },
        )

        # State parameters
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.attn_tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(self.num_v_heads // self.attn_tp_size),
        )

        set_weight_attrs(
            self.A_log, {"weight_loader": sharded_weight_loader(0, self.attn_tp_rank)}
        )
        set_weight_attrs(
            self.dt_bias, {"weight_loader": sharded_weight_loader(0, self.attn_tp_rank)}
        )

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        self.conv_weights = conv_weights

        # Normalization layer
        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=torch.get_device_module().current_device(),
            dtype=config.torch_dtype,
        )

        # Output projection
        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            tp_group=self.attn_tp_group,
            prefix=add_prefix("out_proj", prefix),
        )

    @staticmethod
    def _override_weight_loader(param, loader):
        """Robustly override loader for:
        1) BaseWeightParameter subclasses: real storage is `_weight_loader`
        2) regular Parameters that already have mutable `weight_loader`
        3) regular Parameters without `weight_loader` yet
        """
        if hasattr(param, "_weight_loader"):
            # FP8 / quantized BaseWeightParameter path
            param._weight_loader = loader
            return

        if hasattr(param, "weight_loader"):
            # Regular parameter/tensor that already has a mutable attr.
            # Do NOT call set_weight_attrs here, because it asserts when
            # overwriting an existing attribute.
            param.weight_loader = loader
            return

        # Fresh attribute on a normal tensor/Parameter
        set_weight_attrs(param, {"weight_loader": loader})

    def _bind_packed_weight_loaders(self, module):
        """Bind packed-checkpoint-aware loaders to all relevant params of a merged module."""
        for attr_name in ("weight", "weight_scale_inv", "weight_scale", "input_scale"):
            param = getattr(module, attr_name, None)
            if param is None:
                continue
            original_loader = getattr(param, "weight_loader", None)
            if original_loader is None:
                continue
            wrapped_loader = self._make_packed_weight_loader(module, original_loader)
            self._override_weight_loader(param, wrapped_loader)

    @staticmethod
    def _get_split_sizes_for_param(module, param, loaded_shard_id):
        """Return checkpoint-side split sizes for this param type."""
        if isinstance(param, BlockQuantScaleParameter):
            # Split by output blocks, not raw output sizes.
            block_n, _ = module.quant_method.quant_config.weight_block_size
            block_n = 1 if getattr(param, "format_ue8m0", False) else block_n
            return [
                (module.output_sizes[idx] + block_n - 1) // block_n
                for idx in loaded_shard_id
            ]

        if isinstance(param, PerTensorScaleParameter):
            # One logical scale per logical shard.
            return [1 for _ in loaded_shard_id]

        # Normal weight / non-block quant tensor
        return [module.output_sizes[idx] for idx in loaded_shard_id]

    @classmethod
    def _make_packed_weight_loader(cls, module, original_weight_loader):
        """Wrap the param's original loader so split checkpoints:
          - in_proj_qkv + in_proj_z + in_proj_b + in_proj_a -> merged in_proj_qkvzba
        can load correctly for both normal and FP8 params.
        """

        def weight_loader(param, loaded_weight, loaded_shard_id=None):
            # Only intercept split-checkpoint tuple shards.
            # int shard_id and None should preserve original behavior.
            if isinstance(loaded_shard_id, tuple):
                split_sizes = cls._get_split_sizes_for_param(
                    module, param, loaded_shard_id
                )

                if len(loaded_weight.shape) == 0:
                    # Scalar only makes sense for a single logical shard.
                    assert len(split_sizes) == 1 and split_sizes[0] == 1, (
                        f"Unexpected scalar for tuple shard load: "
                        f"{loaded_shard_id=}, {split_sizes=}"
                    )
                    chunks = [loaded_weight.reshape(1)]
                else:
                    split_dim = getattr(param, "output_dim", 0)
                    chunks = loaded_weight.split(split_sizes, dim=split_dim)

                assert len(chunks) == len(loaded_shard_id), (
                    f"Chunk/shard mismatch: {len(chunks)=}, "
                    f"{len(loaded_shard_id)=}, {split_sizes=}"
                )

                for idx, chunk in zip(loaded_shard_id, chunks):
                    # Delegate each chunk to the param's original int-shard loader.
                    original_weight_loader(param, chunk, idx)
                return

            return original_weight_loader(param, loaded_weight, loaded_shard_id)

        return weight_loader

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        k_tp = self.key_dim // self.attn_tp_size
        v_tp = self.value_dim // self.attn_tp_size
        nv_tp = self.num_v_heads // self.attn_tp_size

        # Directly split, no head group reshape
        query, key, value, z = mixed_qkvz.split([k_tp, k_tp, v_tp, v_tp], dim=-1)
        b, a = mixed_ba.split([nv_tp, nv_tp], dim=-1)

        # value / z reshape to (seq, num_v_heads/tp, head_v_dim)
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)

        return query, key, value, z, b, a

    def _forward_input_proj(self, hidden_states: torch.Tensor):
        projected_all, _ = self.in_proj_qkvzba(hidden_states)
        projected_states_qkvz, projected_states_ba = projected_all.split(
            [self._qkvz_dim, self._ba_dim], dim=-1
        )
        return projected_states_qkvz, projected_states_ba

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
    ):
        seq_len, _ = hidden_states.shape

        projected_states_qkvz, projected_states_ba = self._forward_input_proj(
            hidden_states
        )

        if self.num_v_heads // self.num_k_heads in [1, 2, 4]:
            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
                projected_states_qkvz,
                projected_states_ba,
                triton.cdiv(self.num_k_heads, self.attn_tp_size),
                triton.cdiv(self.num_v_heads, self.attn_tp_size),
                self.head_k_dim,
                self.head_v_dim,
            )
        else:
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                projected_states_qkvz, projected_states_ba
            )
            query, key, value = map(
                lambda x: x.reshape(x.shape[0], -1), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)

        kwargs = {
            "mixed_qkv": mixed_qkv,
            "conv_weights": self.conv_weights,
            "bias": self.conv1d.bias,
            "activation": self.activation,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "attention_tp_size": self.attn_tp_size,
            "head_k_dim": self.head_k_dim,
            "head_v_dim": self.head_v_dim,
            "a": a,
            "b": b,
            "A_log": self.A_log,
            "dt_bias": self.dt_bias,
            "layer_id": self.layer_id,
            "seq_len": seq_len,
            "z": z,
        }

        core_attn_out = ctx.attn_backend.forward(
            q=None,
            k=None,
            v=None,
            layer=None,
            out_cache_loc=None,
            token_to_kv_pool=ctx.token_to_kv_pool,
            forward_mode=ctx.forward_mode,
            bs=ctx.bs,
            **kwargs,
        )

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)
        output, _ = self.out_proj(core_attn_out)
        return output


class Qwen3_5LinearDecoderLayer(nn.Module):
    """Qwen3.5 Decoder Layer with Linear Attention (GatedDeltaNet)."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_id = layer_id

        linear_attn_quant_config = (
            None
            if quant_config and quant_config.get_name() == "nvfp4"
            else quant_config
        )
        self.linear_attn = Qwen3_5GatedDeltaNet(
            config, mapping, layer_id, linear_attn_quant_config, prefix=prefix
        )

        #  Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3_5MoeSparseMoeBlock(
                config=config,
                mapping=self.mapping,
                quant_config=quant_config,
                layer_index=layer_id,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
            )
            is_moe = True
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3_5MoeMLP(
                mapping=self.mapping,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
            )
            is_moe = False
        else:
            raise ValueError(f"Invalid model type: {config.model_type}")

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.is_moe = is_moe
        self.comm_manager = CommManager(
            mapping=self.mapping,
            layer_id=self.layer_id,
            is_moe=is_moe,
            prev_is_moe=is_moe,
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        ctx: ForwardContext,
        **kwargs,
    ):
        num_global_tokens, max_num_tokens_per_gpu = self.comm_manager.get_num_tokens(
            ctx
        )

        if not ctx.forward_mode.is_idle():
            hidden_states, residual = self.comm_manager.input_reduce_norm(
                hidden_states, residual
            )
            hidden_states = self.comm_manager.pre_attn_comm(hidden_states, ctx)

            hidden_states = self.linear_attn(
                hidden_states,
                ctx,
            )

            hidden_states, residual = self.comm_manager.post_attn_reduce_norm(
                hidden_states, residual, ctx
            )

        hidden_states = self.forward_mlp(
            hidden_states,
            residual,
            ctx,
            num_global_tokens,
            max_num_tokens_per_gpu,
        )

        return hidden_states, residual

    def forward_mlp(
        self,
        hidden_states,
        residual,
        ctx: ForwardContext,
        num_global_tokens,
        max_num_tokens_per_gpu,
    ):
        if isinstance(self.mlp, Qwen3_5MoeSparseMoeBlock):
            hidden_states = self.mlp(
                hidden_states, num_global_tokens, max_num_tokens_per_gpu, ctx
            )
        else:
            hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
            hidden_states = self.mlp(hidden_states)
            hidden_states, residual = self.comm_manager.post_mlp_fused(
                hidden_states, residual, ctx
            )
        return hidden_states


class Qwen3_5AttentionDecoderLayer(nn.Module):
    """Qwen3.5 Decoder Layer with Full Attention."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.hidden_size = config.hidden_size
        self.attn_tp_rank = mapping.attn.tp_rank
        self.attn_tp_size = mapping.attn.tp_size
        self.attn_tp_group = mapping.attn.tp_group
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.attn_tp_size == 0
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.attn_tp_size:
            assert self.total_num_kv_heads % self.attn_tp_size == 0
        else:
            assert self.attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        if hasattr(config, "rope_parameters"):
            self.rope_scaling = getattr(config, "rope_parameters", None)
        else:
            self.rope_scaling = getattr(config, "rope_scaling", None)

        self.rope_theta = self.rope_scaling.get("rope_theta", 10000)
        self.partial_rotary_factor = self.rope_scaling.get("partial_rotary_factor", 1.0)
        self.layer_id = layer_id

        self.attn_output_gate = getattr(config, "attn_output_gate", True)
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
        )

        # Dense MLP for non-MoE variant
        if config.model_type == "qwen3_5_text":
            self.mlp = Qwen3_5MoeMLP(
                mapping=self.mapping,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
            )
            is_moe = False
        elif config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3_5MoeSparseMoeBlock(
                config=config,
                mapping=self.mapping,
                quant_config=quant_config,
                layer_index=layer_id,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
            )
            is_moe = True
        else:
            raise ValueError(f"Invalid model type: {config.model_type}")

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.is_moe = is_moe
        self.comm_manager = CommManager(
            mapping=self.mapping,
            layer_id=self.layer_id,
            is_moe=is_moe,
            prev_is_moe=is_moe,
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
        )

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # qk_rmsnorm expects GemmaRMSNorm's effective gamma.
        return qk_rmsnorm(
            q,
            k,
            self.q_norm.gemma_weight,
            self.k_norm.gemma_weight,
            self.q_norm.variance_epsilon,
        )

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        """Full attention forward pass."""
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            q, k, gate = fused_qk_rmsnorm_rope_gate(
                q_gate,
                k,
                self.q_norm.gemma_weight,
                self.k_norm.gemma_weight,
                self.rotary_emb.cos_sin_cache,
                positions,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.rotary_emb.rotary_dim,
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self._apply_qk_norm(q, k)
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, ctx, out_cache_loc)

        if self.attn_output_gate:
            sigmoid_mul(attn_output, gate)

        if ctx.draft_first_step_reduce:
            # Slice attn_output to [bs, H] so o_proj runs on live rows only.
            attn_output = attn_output.index_select(0, ctx.gather_ids)

        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        **kwargs,
    ):
        num_global_tokens, max_num_tokens_per_gpu = self.comm_manager.get_num_tokens(
            ctx
        )

        if not ctx.forward_mode.is_idle():
            hidden_states, residual = self.comm_manager.input_reduce_norm(
                hidden_states, residual
            )
            hidden_states = self.comm_manager.pre_attn_comm(hidden_states, ctx)
            hidden_states = self.self_attention(
                positions=positions,
                hidden_states=hidden_states,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
            )
            if ctx.draft_first_step_reduce:
                # Gather residual to self_attention's [bs, H].
                residual = residual.index_select(0, ctx.gather_ids)
            hidden_states, residual = self.comm_manager.post_attn_reduce_norm(
                hidden_states, residual, ctx
            )

        hidden_states = self.forward_mlp(
            hidden_states,
            residual,
            ctx,
            num_global_tokens,
            max_num_tokens_per_gpu,
        )

        return hidden_states, residual

    def forward_mlp(
        self,
        hidden_states,
        residual,
        ctx: ForwardContext,
        num_global_tokens,
        max_num_tokens_per_gpu,
    ):
        if isinstance(self.mlp, Qwen3_5MoeSparseMoeBlock):
            hidden_states = self.mlp(
                hidden_states, num_global_tokens, max_num_tokens_per_gpu, ctx
            )
        else:
            hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
            hidden_states = self.mlp(hidden_states)
            hidden_states, residual = self.comm_manager.post_mlp_fused(
                hidden_states, residual, ctx
            )
        return hidden_states


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3_5AttentionDecoderLayer,
    "linear_attention": Qwen3_5LinearDecoderLayer,
}


class Qwen3_5ForCausalLM(nn.Module):
    """Qwen3.5 Model with support for dense variant."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.hidden_size = config.hidden_size

        alt_stream = torch.cuda.Stream()

        # Embedding layer
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )

        # Decoder layers
        def get_layer(idx: int, prefix: str):
            layer_type = config.layers_block_type[idx]
            layer_class = ALL_DECODER_LAYER_TYPES[layer_type]
            if layer_type == "attention":
                prefix = add_prefix("self_attn", prefix)
            else:
                prefix = add_prefix("linear_attn", prefix)
            return layer_class(
                config=config,
                mapping=self.mapping,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            )

        self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )

        # Final normalization
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

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
    ) -> tuple[torch.Tensor, None]:
        # Initialize hidden states
        if input_embeds is None:
            # Only skip embedding allreduce when the first layer's fused
            # allreduce+residual+norm will handle it
            if self.layers[0].comm_manager.should_fuse(input_ids.shape[0]):
                hidden_states = self.embed_tokens(input_ids, reduce_results=False)
                residual = torch.zeros_like(hidden_states)
            else:
                hidden_states = self.embed_tokens(input_ids)
                residual = None
        else:
            hidden_states = input_embeds
            residual = None

        # Pass through decoder layers
        for layer_idx in range(len(self.layers)):
            layer = self.layers[layer_idx]
            with get_global_expert_distribution_recorder().with_current_layer(
                layer_idx
            ):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    ctx=ctx,
                    out_cache_loc=out_cache_loc,
                )

            # Process deepstack embeddings if provided
            if (
                input_deepstack_embeds is not None
                and input_deepstack_embeds.numel() > 0
                and layer_idx < 3
            ):
                sep = self.hidden_size * layer_idx
                hidden_states.add_(
                    input_deepstack_embeds[:, sep : sep + self.hidden_size]
                )

        # Apply final normalization with optional allreduce fusion
        hidden_states = layer.comm_manager.final_norm(
            hidden_states, residual, ctx, self.norm
        )

        return hidden_states, None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN (GatedDeltaNet) linear attention projections
            # Split checkpoint format (separate qkv/z/b/a files)
            ("in_proj_qkvzba.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvzba.", "in_proj_z.", 3),
            ("in_proj_qkvzba.", "in_proj_b.", 4),
            ("in_proj_qkvzba.", "in_proj_a.", 5),
            # Pre-packed checkpoint format (already merged qkvz and ba)
            ("in_proj_qkvzba.", "in_proj_qkvz.", (0, 1, 2, 3)),
            ("in_proj_qkvzba.", "in_proj_ba.", (4, 5)),
        ]

        loaded_params: set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning("Parameter %s not found in params_dict", name)
                    continue
                param = params_dict[name]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config, mapping=mapping, quant_config=quant_config, prefix=prefix
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN (GatedDeltaNet) linear attention projections
            # Split checkpoint format (separate qkv/z/b/a files)
            ("in_proj_qkvzba.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvzba.", "in_proj_z.", 3),
            ("in_proj_qkvzba.", "in_proj_b.", 4),
            ("in_proj_qkvzba.", "in_proj_a.", 5),
            # Pre-packed checkpoint format (already merged qkvz and ba)
            ("in_proj_qkvzba.", "in_proj_qkvz.", (0, 1, 2, 3)),
            ("in_proj_qkvzba.", "in_proj_ba.", (4, 5)),
        ]

        # Skip loading extra parameters for GPTQ/nvfp4 models.
        ignore_suffixes = (
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
        loaded_params: set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        # MoE expert weights, scales, and activation scales are handled
        # by the checkpoint loader.
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params_dict,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
            ),
            fused_schema=ExpertCheckpointSchema(
                gate_up_fused_name="gate_up_proj",
                down_proj_name="down_proj",
            ),
            num_experts=self.config.num_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/nvfp4 models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith((".bias", "_bias")) and name not in params_dict:
                    continue
                if moe_loader.matches(name):
                    mapped_name = moe_loader.load(name, loaded_weight)
                    loaded_params.add(mapped_name)
                    continue

                # Skip loading extra parameters for GPTQ/nvfp4 models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning("Parameter %s not found in params_dict", name)
            loaded_params.add(name)

        return loaded_params


class Qwen3_5ForConditionalGeneration(BaseCausalLM):
    model_cls = Qwen3_5ForCausalLM

    def __init__(
        self,
        config: Qwen3_5Config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
    ):
        super().__init__(
            config=config.text_config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
        )

        rope_config = getattr(self.config, "rope_parameters", None) or getattr(
            self.config, "rope_scaling", {}
        )
        self.is_mrope_enabled = "mrope_section" in rope_config
        self.is_multimodal_active = is_multimodal_active
        if not self.is_multimodal_active:
            self.visual = None
            self.deepstack_visual_indexes = []
            self.num_deepstack_embeddings = 0
            self.vision_embedder = None
            self.image_encoder = None
            self.video_encoder = None
        else:
            self.visual = Qwen3VLMoeVisionModel(
                config.vision_config,
                quant_config=None,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                prefix=add_prefix("model.visual", prefix),
                mapping=mapping,
                mm_attention_backend=mm_attention_backend,
            )
            self.deepstack_visual_indexes = self.visual.deepstack_visual_indexes
            self.num_deepstack_embeddings = len(self.deepstack_visual_indexes)
            # image_encoder may be swapped to a cudagraph wrapper by ModelExecutor.
            self.vision_embedder = VisionEmbedder()
            self.image_encoder = self.get_image_feature
            self.video_encoder = self.get_video_feature

    def separate_deepstack_embeds(self, embedding: torch.Tensor):
        assert embedding.shape[-1] % (1 + self.num_deepstack_embeddings) == 0, (
            f"hidden_state of {embedding.shape} should be divisible by "
            f"{1 + self.num_deepstack_embeddings}"
        )
        separate_index = self.config.hidden_size
        input_embeds = embedding[:, :separate_index]
        input_deepstack_embeds = embedding[:, separate_index:]
        return input_embeds, input_deepstack_embeds

    def pad_input_ids(self, input_ids: list[int], mm_inputs: MultimodalInputs):
        return pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        """Eager image encode via the ``pre_encode`` / ``forward_blocks`` /
        ``post_encode`` decomposition the cudagraph wrapper uses, so eager
        and captured paths share a single source of truth."""
        tokens, grid = self.pre_encode(items, grid_attr="image_grid_thw")
        metadata = self.visual.prepare_metadata(grid)
        encoded = self.visual.forward_blocks(tokens, metadata)
        return self.post_encode([encoded], grid)

    def get_video_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        """Eager video encode; videos never go through the cudagraph wrapper."""
        tokens, grid = self.pre_encode(items, grid_attr="video_grid_thw")
        metadata = self.visual.prepare_metadata(grid)
        encoded = self.visual.forward_blocks(tokens, metadata)
        return self.post_encode([encoded], grid)

    def pre_encode(
        self,
        items: list[MultimodalDataItem],
        grid_attr: str = "image_grid_thw",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eager patch-embed before the captured region; returns ``(tokens, grid)``.

        ``grid_attr`` selects which grid field on each item to read --
        ``image_grid_thw`` (default, used by the wrapper) or ``video_grid_thw``.
        """
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        grid = torch.concat([getattr(item, grid_attr) for item in items], dim=0)
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert grid.dim() == 2, grid.dim()
        x = self.visual.prepare_patch_embed(pixel_values, grid)
        return x, grid

    def post_encode(
        self, encoder_outs: list[torch.Tensor], grid: torch.Tensor
    ) -> torch.Tensor:
        """Eager step after the captured region; returns features."""
        return torch.cat(encoder_outs, dim=0)

    def make_encoder_cudagraph_wrapper(self, mapping):
        # Captured region is ``Qwen3VLMoeVisionModel.forward_blocks`` (blocks +
        # deepstack mergers + merger); the merger applies a
        # ``spatial_merge_size ** 2`` token reduction, so budgets count
        # post-merge tokens while the capture input buffer holds
        # ``spatial_merge_size ** 2 * budget`` patches.
        return EncoderCudaGraphWrapper(
            mapping=mapping,
            tower=self.visual,
            pre_encode=self.pre_encode,
            post_encode=self.post_encode,
            out_div=self.visual.spatial_merge_size**2,
            merge=self.visual.spatial_merge_size,
            budget_range=(64, 4096),
            input_feature_shape=(1, self.visual.hidden_size),
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        multimodal_context = kwargs.pop("multimodal_context", None)
        if (
            multimodal_context is None
            or not multimodal_context.has_extend_inputs()
            or ctx.forward_mode.is_decode_or_idle()
        ):
            return super().forward(
                ctx,
                input_ids,
                positions,
                out_cache_loc,
                **kwargs,
            )

        input_embeds, model_kwargs = self.vision_embedder.apply(
            input_ids=input_ids,
            text_embedding=self.model.get_input_embeddings(),
            ctx=multimodal_context,
            encoders={
                Modality.IMAGE: EncoderSpec(self.image_encoder, deepstack=True),
                Modality.VIDEO: EncoderSpec(self.video_encoder, deepstack=True),
            },
            multimodal_model=self,
            is_decode_or_idle=ctx.forward_mode.is_decode_or_idle(),
        )
        hidden_states, aux_hidden_states = self.model(
            input_ids,
            positions,
            ctx,
            out_cache_loc,
            input_embeds=input_embeds,
            **model_kwargs,
        )
        logits_metadata = LogitsMetadata.from_forward_context(ctx)
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            logits_metadata,
            aux_hidden_states,
        )

    def resolve_model(
        self,
        config: Qwen3_5TextConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ):
        return self.model_cls(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("model.language_model", prefix),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN (GatedDeltaNet) linear attention projections
            # Split checkpoint format (separate qkv/z/b/a files)
            ("in_proj_qkvzba.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvzba.", "in_proj_z.", 3),
            ("in_proj_qkvzba.", "in_proj_b.", 4),
            ("in_proj_qkvzba.", "in_proj_a.", 5),
            # Pre-packed checkpoint format (already merged qkvz and ba)
            ("in_proj_qkvzba.", "in_proj_qkvz.", (0, 1, 2, 3)),
            ("in_proj_qkvzba.", "in_proj_ba.", (4, 5)),
        ]

        loaded_params: set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if not self.is_multimodal_active and "visual" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            if "visual" in name:
                name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                name = name.replace(r"model.visual.", r"visual.")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "visual" in name or "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "visual" in name:
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                    name = name.replace(r"model.visual.", r"visual.")
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning("Parameter %s not found in params_dict", name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 MoE Vision-Language Model."""

    model_cls = Qwen3_5MoeForCausalLM

    def __init__(
        self,
        config: Qwen3_5Config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
            is_multimodal_active=is_multimodal_active,
            mm_attention_backend=mm_attention_backend,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN (GatedDeltaNet) linear attention projections
            # Split checkpoint format (separate qkv/z/b/a files)
            ("in_proj_qkvzba.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvzba.", "in_proj_z.", 3),
            ("in_proj_qkvzba.", "in_proj_b.", 4),
            ("in_proj_qkvzba.", "in_proj_a.", 5),
            # Pre-packed checkpoint format (already merged qkvz and ba)
            ("in_proj_qkvzba.", "in_proj_qkvz.", (0, 1, 2, 3)),
            ("in_proj_qkvzba.", "in_proj_ba.", (4, 5)),
        ]

        ignore_suffixes = (
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
        loaded_params: set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        # MoE expert weights, scales, and activation scales are handled
        # by the checkpoint loader.
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params_dict,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
            ),
            fused_schema=ExpertCheckpointSchema(
                gate_up_fused_name="gate_up_proj",
                down_proj_name="down_proj",
            ),
            num_experts=self.config.num_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if not self.is_multimodal_active and "visual" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            if "visual" in name:
                name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                name = name.replace(r"model.visual.", r"visual.")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "visual" in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra parameters for GPTQ/nvfp4 models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith((".bias", "_bias")) and name not in params_dict:
                    continue
                if moe_loader.matches(name):
                    mapped_name = moe_loader.load(name, loaded_weight)
                    loaded_params.add(mapped_name)
                    continue

                # Skip loading extra parameters for GPTQ/nvfp4 models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning("Parameter %s not found in params_dict", name)
            loaded_params.add(name)

        return loaded_params

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        text_config = getattr(config, "text_config", config)
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=text_config.num_experts,
            num_groups=None,
        )


@triton.jit
def fused_qkvzba_split_reshape_cat_contiguous_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    stride_qkvz,
    stride_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    i_bs, i_qk = tl.program_id(0), tl.program_id(1)

    V_PER_GROUP: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK

    # ── Input dimensions ──
    TOTAL_Q: tl.constexpr = NUM_HEADS_QK * HEAD_QK
    TOTAL_K: tl.constexpr = NUM_HEADS_QK * HEAD_QK
    TOTAL_V: tl.constexpr = NUM_HEADS_V * HEAD_V

    # ── Output dimensions ──
    QKV_DIM_T: tl.constexpr = TOTAL_Q + TOTAL_K + TOTAL_V

    # ── Read from input (supports non-contiguous stride) ──
    # q for head group i_qk: in the all_q region, offset i_qk * HEAD_QK
    blk_q_ptr = mixed_qkvz + i_bs * stride_qkvz + i_qk * HEAD_QK + tl.arange(0, HEAD_QK)
    # k for head group i_qk: in the all_k region
    blk_k_ptr = (
        mixed_qkvz
        + i_bs * stride_qkvz
        + TOTAL_Q
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    # v for head group i_qk: in the all_v region
    blk_v_ptr = (
        mixed_qkvz
        + i_bs * stride_qkvz
        + TOTAL_Q
        + TOTAL_K
        + i_qk * V_PER_GROUP * HEAD_V
        + tl.arange(0, V_PER_GROUP * HEAD_V)
    )
    # z for head group i_qk: in the all_z region
    blk_z_ptr = (
        mixed_qkvz
        + i_bs * stride_qkvz
        + TOTAL_Q
        + TOTAL_K
        + TOTAL_V
        + i_qk * V_PER_GROUP * HEAD_V
        + tl.arange(0, V_PER_GROUP * HEAD_V)
    )

    # ── Write to output (identical layout to the interleaved kernel) ──
    blk_q_st_ptr = mixed_qkv + i_bs * QKV_DIM_T + i_qk * HEAD_QK + tl.arange(0, HEAD_QK)
    blk_k_st_ptr = (
        mixed_qkv
        + i_bs * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    blk_v_st_ptr = (
        mixed_qkv
        + i_bs * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK * 2
        + i_qk * V_PER_GROUP * HEAD_V
        + tl.arange(0, V_PER_GROUP * HEAD_V)
    )
    blk_z_st_ptr = (
        z
        + i_bs * NUM_HEADS_V * HEAD_V
        + i_qk * V_PER_GROUP * HEAD_V
        + tl.arange(0, V_PER_GROUP * HEAD_V)
    )

    tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
    tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
    tl.store(blk_v_st_ptr, tl.load(blk_v_ptr))
    tl.store(blk_z_st_ptr, tl.load(blk_z_ptr))

    # ── b and a ──
    for i in tl.static_range(V_PER_GROUP):
        blk_b_ptr = mixed_ba + i_bs * stride_ba + i_qk * V_PER_GROUP + i
        blk_b_st_ptr = b + i_bs * NUM_HEADS_V + i_qk * V_PER_GROUP + i
        tl.store(blk_b_st_ptr, tl.load(blk_b_ptr))

    for i in tl.static_range(V_PER_GROUP):
        blk_a_ptr = mixed_ba + i_bs * stride_ba + NUM_HEADS_V + i_qk * V_PER_GROUP + i
        blk_a_st_ptr = a + i_bs * NUM_HEADS_V + i_qk * V_PER_GROUP + i
        tl.store(blk_a_st_ptr, tl.load(blk_a_ptr))


def fused_qkvzba_split_reshape_cat_contiguous(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    """Fused split/reshape/cat for Qwen3.5. Supports non-contiguous inputs.

    Input layout (per row):
        mixed_qkvz: [all_q | all_k | all_v | all_z]
        mixed_ba:   [all_b | all_a]

    Output layout:
        mixed_qkv: [all_q | all_k | all_v]  (z stripped)
        z: [num_v_heads, head_v]
        b: [num_v_heads]
        a: [num_v_heads]
    """
    batch, seq_len = mixed_qkvz.shape[0], 1
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = torch.empty(
        [batch * seq_len, qkv_dim_t],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    z = torch.empty(
        [batch * seq_len, num_heads_v, head_v],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        [batch * seq_len, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty_like(b)
    grid = (batch * seq_len, num_heads_qk)
    fused_qkvzba_split_reshape_cat_contiguous_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        mixed_qkvz.stride(0),
        mixed_ba.stride(0),
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        num_warps=1,
        num_stages=3,
    )
    return mixed_qkv, z, b, a


EntryClass = [Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration]
