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

"""Inference-only GLM-5.3-Flash modeling.

The checkpoint alternates Kimi Delta Attention (KDA) and sparse MLA layers,
uses multi-stream hyper-connections (mHC) around attention and FFN, and scores
the sparse MLA history through a compressed kpool index. This file owns the
vision tower and text parameter hierarchy. The DSA backend owns KPool planning,
cache updates, and sparse-history selection; modeling supplies the learned
indexer projections and consumes the resulting generic DSA metadata.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Protocol

import torch
import torch.nn.functional as F
from tokenspeed_kernel import mhc_post, mhc_pre
from tokenspeed_kernel.ops.activation.triton import rmsnorm_gated_sigmoid, silu_and_mul
from tokenspeed_kernel.ops.transform import hadamard_transform
from torch import nn

from tokenspeed.runtime.configs.glm53_flash_config import (
    Glm53FlashConfig,
    Glm53FlashTextConfig,
    Glm53FlashVisionConfig,
)
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.breakable_cuda_graph import break_point
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.dsa import DSABackend
from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
    HybridLinearAttnBackend,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.attention.mm_encoder_attention import VisionAttention
from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, LayerNorm, RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    MoELayer,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.topk import TopK
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3AttentionMLA,
    DeepseekV3MoE,
    MoEGate,
    _prepare_mla_kv_b_proj_weights,
)
from tokenspeed.runtime.models.glm5 import (
    GlmDsaDecodeTopK,
    GlmDsaIndexer,
    GlmDsaPrefillTopK,
    GlmMoeDsaAttention,
)
from tokenspeed.runtime.multimodal.embedder import (
    EncoderSpec,
    VisionEmbedder,
    pad_input_tokens,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.env import global_server_args_dict

# ===----------------------------------------------------------------------=== #
# Multimodal vision path
# ===----------------------------------------------------------------------=== #


class Glm53FlashVisionPatchEmbed(nn.Module):
    def __init__(self, config: Glm53FlashVisionConfig) -> None:
        super().__init__()
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        kernel_size = (
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        self.proj = nn.Conv3d(
            config.in_channels,
            config.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(pixel_values.to(self.proj.weight.dtype)).flatten(1)


class Glm53FlashVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


class Glm53FlashVisionMLP(nn.Module):
    def __init__(self, config: Glm53FlashVisionConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.attention_bias,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=config.attention_bias,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        if self.swiglu_limit is not None:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(F.silu(gate) * up)


class Glm53FlashVisionBlock(nn.Module):
    def __init__(
        self,
        config: Glm53FlashVisionConfig,
        mapping: Mapping,
        prefix: str,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        head_dim = config.hidden_size // config.num_heads
        self.attn = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            head_size=head_dim,
            mapping=mapping,
            quant_config=None,
            prefix=add_prefix("attn", prefix),
            qkv_bias=config.attention_bias,
            proj_bias=config.attention_bias,
            mm_attention_backend=mm_attention_backend,
        )
        self.attn.q_norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.attn.k_norm = nn.RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.mlp = Glm53FlashVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        attention_output = self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=(cos, sin),
            max_seqlen=max_seqlen,
        )
        if attention_output.dim() == 3:
            attention_output = attention_output.squeeze(0)
        hidden_states = hidden_states + attention_output
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Glm53FlashVisionPatchMerger(nn.Module):
    def __init__(self, config: Glm53FlashVisionConfig) -> None:
        super().__init__()
        dim = config.out_hidden_size
        context_dim = config.projection_intermediate_size
        self.proj = nn.Linear(dim, dim, bias=False)
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=False)
        self.up_proj = nn.Linear(dim, context_dim, bias=False)
        self.down_proj = nn.Linear(context_dim, dim, bias=False)
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.gelu(self.post_projection_norm(self.proj(hidden_states)))
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        if self.swiglu_limit is not None:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(F.silu(gate) * up)


class Glm53FlashVision(nn.Module):
    """GLM-OCR-style vision tower with the GLM-5.3-Flash merger width."""

    def __init__(
        self,
        config: Glm53FlashVisionConfig,
        mapping: Mapping,
        prefix: str = "",
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_embed = Glm53FlashVisionPatchEmbed(config)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Glm53FlashVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                Glm53FlashVisionBlock(
                    config,
                    mapping,
                    prefix=add_prefix(f"blocks.{layer_id}", prefix),
                    mm_attention_backend=mm_attention_backend,
                )
                for layer_id in range(config.depth)
            ]
        )
        self.post_layernorm = nn.RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.downsample = nn.Conv2d(
            config.hidden_size,
            config.out_hidden_size,
            kernel_size=config.spatial_merge_size,
            stride=config.spatial_merge_size,
        )
        self.merger = Glm53FlashVisionPatchMerger(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def _position_ids(self, grid_thw: torch.Tensor) -> torch.Tensor:
        position_ids = []
        merge = self.spatial_merge_size
        for temporal, height, width in grid_thw.tolist():
            if height % merge or width % merge:
                raise ValueError(
                    "GLM-5.3-Flash vision grid height and width must be divisible "
                    f"by spatial_merge_size={merge}"
                )
            h_ids, w_ids = torch.meshgrid(
                torch.arange(height, device=self.device),
                torch.arange(width, device=self.device),
                indexing="ij",
            )
            block_shape = (height // merge, merge, width // merge, merge)
            h_ids = h_ids.reshape(block_shape).transpose(1, 2).flatten()
            w_ids = w_ids.reshape(block_shape).transpose(1, 2).flatten()
            position_ids.append(torch.stack((h_ids, w_ids), dim=-1).repeat(temporal, 1))
        return torch.cat(position_ids, dim=0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        grid_thw = grid_thw.to(device=self.device)
        hidden_states = self.patch_embed(
            pixel_values.to(device=self.device, dtype=self.dtype)
        )
        expected_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        if hidden_states.shape[0] != expected_tokens:
            raise ValueError(
                "GLM-5.3-Flash vision grid describes "
                f"{expected_tokens} patches, got {hidden_states.shape[0]}"
            )
        frame_lengths = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2],
            grid_thw[:, 0],
        )
        cu_seqlens = F.pad(
            frame_lengths.cumsum(dim=0, dtype=torch.int32),
            (1, 0),
            value=0,
        )
        max_seqlen = int(frame_lengths.max().item())
        rotary = self.rotary_pos_emb(self._position_ids(grid_thw))
        rotary = torch.cat((rotary, rotary), dim=-1)
        cos = rotary.cos()
        sin = rotary.sin()
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens,
                cos,
                sin,
                max_seqlen,
            )
        hidden_states = self.post_layernorm(hidden_states)
        merge = self.spatial_merge_size
        hidden_states = hidden_states.reshape(-1, merge, merge, hidden_states.shape[-1])
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.downsample(hidden_states).flatten(1)
        return self.merger(hidden_states)

    def embed_media(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        if not items:
            return torch.empty(
                (0, self.config.out_hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
        pixel_values = torch.cat(
            [item.feature.to(self.device, non_blocking=True) for item in items],
            dim=0,
        )
        grids = []
        for item in items:
            if item.modality == Modality.VIDEO:
                grids.append(item.video_grid_thw)
            else:
                grids.append(item.image_grid_thw)
        return self(pixel_values, torch.cat(grids, dim=0))


# ===----------------------------------------------------------------------=== #
# Text model
# ===----------------------------------------------------------------------=== #


@dataclass
class Glm53FlashIndexerOutput:
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    gate: torch.Tensor


class Glm53FlashMLP(nn.Module):
    """GLM-5.3-Flash SwiGLU with the checkpoint's activation clamp."""

    def __init__(
        self,
        config: Glm53FlashTextConfig,
        intermediate_size: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        is_shared_expert: bool = False,
    ) -> None:
        super().__init__()
        if is_shared_expert:
            tp_rank = mapping.moe.tp_ep_rank
            tp_size = mapping.moe.tp_ep_size
            tp_group = mapping.moe.tp_ep_group
        else:
            tp_rank = mapping.dense.tp_rank
            tp_size = mapping.dense.tp_size
            tp_group = mapping.dense.tp_group
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            config.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("down_proj", prefix),
        )
        self.swiglu_limit = config.swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        gate_up, _ = self.gate_up_proj(hidden_states)
        activated = silu_and_mul(gate_up, limit=self.swiglu_limit)
        output, _ = self.down_proj(activated)
        return output


class Glm53FlashMoE(DeepseekV3MoE):
    """DeepSeek routing with GLM-5.3-Flash's clamped SwiGLU experts."""

    def __init__(
        self,
        config: Glm53FlashTextConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        layer_index: int = -1,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.mapping = mapping
        self.layer_index = layer_index
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.stream_fork = StreamFork(alt_stream)

        if mapping.moe.ep_size > config.n_routed_experts:
            raise ValueError(
                f"EP size {mapping.moe.ep_size} is greater than the number of "
                f"experts {config.n_routed_experts}."
            )
        self.gate = MoEGate(config=config, prefix=add_prefix("gate", prefix))
        self.experts = MoELayer(
            top_k=config.num_experts_per_tok,
            num_experts=(
                config.n_routed_experts
                + global_server_args_dict["ep_num_redundant_experts"]
            ),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=layer_index,
            prefix=prefix,
            tp_rank=mapping.moe.tp_rank,
            tp_size=mapping.moe.tp_size,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
            activation="swiglu",
            activation_alpha=1.0,
            swiglu_limit=config.swiglu_limit,
            routing_config={
                "n_group": config.n_group,
                "topk_group": config.topk_group,
                "routed_scaling_factor": config.routed_scaling_factor,
                "normalize_topk_weights": config.norm_topk_prob,
                "correction_bias": self.gate.e_score_correction_bias,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
            },
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            num_fused_shared_experts=0,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=config.routed_scaling_factor,
            output_format=self.experts.topk_output_format,
        )
        self.shared_experts = Glm53FlashMLP(
            config=config,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("shared_experts", prefix),
            is_shared_expert=True,
        )


class _Glm53FlashMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Merged TP projection with selected output shards replicated."""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        replicated_shard_ids: tuple[int, ...],
        tp_size: int,
        **kwargs,
    ) -> None:
        self.replicated_shard_ids = set(replicated_shard_ids)
        merged_output_sizes = output_sizes.copy()
        for shard_id in self.replicated_shard_ids:
            merged_output_sizes[shard_id] *= tp_size
        super().__init__(input_size, merged_output_sizes, tp_size=tp_size, **kwargs)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        if loaded_shard_id in self.replicated_shard_ids:
            self.tp_rank = 0
        try:
            super().weight_loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank

    def weight_loader_v2(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        if loaded_shard_id in self.replicated_shard_ids:
            self.tp_rank = 0
        try:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank


class Glm53FlashKDA(nn.Module):
    """Kimi Delta Attention with the GLM-5.3-Flash checkpoint parameter names."""

    def __init__(
        self,
        config: Glm53FlashTextConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.layer_id = layer_id
        linear = config.linear_attn_config
        self.num_heads = linear["num_heads"]
        self.head_dim = linear["head_dim"]
        self.conv_size = linear["short_conv_kernel_size"]
        self.lower_bound = linear["gate_lower_bound"]

        tp_rank = mapping.attn.tp_rank
        tp_size = mapping.attn.tp_size
        tp_group = mapping.attn.tp_group
        self.local_num_heads = self.num_heads // tp_size
        projection_size = self.num_heads * self.head_dim
        local_projection_size = self.local_num_heads * self.head_dim

        self.fused_qkvbfg_a_proj = _Glm53FlashMergedColumnParallelLinear(
            config.hidden_size,
            [
                projection_size,
                projection_size,
                projection_size,
                self.num_heads,
                self.head_dim,
                self.head_dim,
            ],
            replicated_shard_ids=(4, 5),
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("fused_qkvbfg_a_proj", prefix),
        )
        self.fused_qkvbfg_a_split_sizes = (
            3 * local_projection_size,
            self.local_num_heads,
            self.head_dim,
            self.head_dim,
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("f_b_proj", prefix),
        )
        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("g_b_proj", prefix),
        )

        self.q_conv1d_weight = nn.Parameter(
            torch.zeros(local_projection_size, 1, self.conv_size)
        )
        self.k_conv1d_weight = nn.Parameter(
            torch.zeros(local_projection_size, 1, self.conv_size)
        )
        self.v_conv1d_weight = nn.Parameter(
            torch.zeros(local_projection_size, 1, self.conv_size)
        )
        for weight in (
            self.q_conv1d_weight,
            self.k_conv1d_weight,
            self.v_conv1d_weight,
        ):
            weight.weight_loader = sharded_weight_loader(0, tp_rank)

        self.A_log = nn.Parameter(
            torch.zeros(self.local_num_heads, dtype=torch.float32)
        )
        head_start = tp_rank * self.local_num_heads

        def load_a_log(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
            flat = loaded_weight.reshape(-1)
            param.data.copy_(flat[head_start : head_start + self.local_num_heads])

        self.A_log.weight_loader = load_a_log
        self.dt_bias = nn.Parameter(
            torch.zeros(local_projection_size, dtype=torch.float32)
        )
        self.dt_bias.weight_loader = sharded_weight_loader(0, tp_rank)
        self.conv_weights: torch.Tensor | None = None
        self._conv_weight_versions = (-1, -1, -1)

        self.o_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            projection_size,
            config.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            prefix=add_prefix("o_proj", prefix),
        )

    def fuse_conv_weights(self) -> None:
        self.conv_weights = torch.cat(
            (
                self.q_conv1d_weight,
                self.k_conv1d_weight,
                self.v_conv1d_weight,
            ),
            dim=0,
        ).squeeze(1)
        self._conv_weight_versions = (
            self.q_conv1d_weight._version,
            self.k_conv1d_weight._version,
            self.v_conv1d_weight._version,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states

        projected = self.fused_qkvbfg_a_proj(hidden_states)[0]
        mixed_qkv, beta, f_a, g_a = projected.split(
            self.fused_qkvbfg_a_split_sizes,
            dim=-1,
        )
        output_gate = self.g_b_proj(g_a)[0]
        if not ctx.forward_mode.is_decode():
            mixed_qkv = mixed_qkv.contiguous()
        conv_versions = (
            self.q_conv1d_weight._version,
            self.k_conv1d_weight._version,
            self.v_conv1d_weight._version,
        )
        if self.conv_weights is None or conv_versions != self._conv_weight_versions:
            self.fuse_conv_weights()
        conv_weights = self.conv_weights

        num_tokens = hidden_states.shape[0]
        projection_size = self.num_heads * self.head_dim
        fuse_decode_output_norm = ctx.forward_mode.is_decode() and num_tokens == ctx.bs
        core_output = ctx.attn_backend.forward(
            q=None,
            k=None,
            v=None,
            layer=None,
            out_cache_loc=out_cache_loc,
            token_to_kv_pool=ctx.token_to_kv_pool,
            forward_mode=ctx.forward_mode,
            bs=ctx.bs,
            mixed_qkv=mixed_qkv,
            conv_weights=conv_weights,
            bias=None,
            activation="silu",
            key_dim=projection_size,
            value_dim=projection_size,
            attention_tp_size=self.mapping.attn.tp_size,
            head_k_dim=self.head_dim,
            head_v_dim=self.head_dim,
            f_a_out=f_a,
            f_b_weight=self.f_b_proj.weight,
            beta_raw=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            lower_bound=self.lower_bound,
            output_gate=output_gate if fuse_decode_output_norm else None,
            norm_weight=self.o_norm.weight if fuse_decode_output_norm else None,
            norm_eps=self.o_norm.variance_epsilon if fuse_decode_output_norm else None,
            layer_id=self.layer_id,
            seq_len=num_tokens,
        )
        core_output = core_output.reshape(
            num_tokens, self.local_num_heads * self.head_dim
        )
        if not fuse_decode_output_norm:
            core_output = rmsnorm_gated_sigmoid(
                core_output.contiguous(),
                output_gate.contiguous(),
                self.o_norm.weight,
                self.o_norm.variance_epsilon,
                self.local_num_heads,
                self.head_dim,
            )
        return self.o_proj(core_output)[0]


class Glm53FlashIndexer(GlmDsaIndexer):
    """DSA indexer that emits raw KPool compression inputs."""

    def __init__(
        self,
        config: Glm53FlashTextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.index_topk = config.index_topk
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.index_kpool = config.index_kpool
        self.rope_dim = config.qk_rope_head_dim
        self.softmax_scale = self.index_head_dim**-0.5
        self.weights_softmax_scale = self.softmax_scale * self.index_n_heads**-0.5

        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.wk = ReplicatedLinear(
            config.hidden_size,
            self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            self.index_n_heads,
            bias=False,
            quant_config=None,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.wk_weights_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [self.index_head_dim, self.index_n_heads],
            bias=False,
            quant_config=None,
            prefix=add_prefix("wk_weights_proj", prefix),
        )
        self._wk_weights_proj_loaded = False
        self.k_norm = LayerNorm(self.index_head_dim, eps=1e-6)
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(
                self.index_kpool,
                self.index_head_dim,
                dtype=torch.float32,
            )
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(
                self.index_head_dim,
                config.hidden_size,
                dtype=torch.bfloat16,
            )
        )
        self.rotary_emb = (
            get_rope(
                self.rope_dim,
                rotary_dim=self.rope_dim,
                max_position=config.max_position_embeddings,
                base=10000.0,
                rope_scaling=None,
                is_neox_style=not config.indexer_rope_interleave,
            )
            if self.rope_dim > 0
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
    ) -> Glm53FlashIndexerOutput:
        query = self.wq_b(q_lora)[0].view(-1, self.index_n_heads, self.index_head_dim)
        if self._wk_weights_proj_loaded:
            key_weights = self.wk_weights_proj(hidden_states)[0]
            key, weights = key_weights.split(
                [self.index_head_dim, self.index_n_heads], dim=-1
            )
        else:
            key = self.wk(hidden_states)[0]
            weights = self.weights_proj(hidden_states)[0]
        key = self.k_norm(key)
        if self.rotary_emb is not None:
            query_pe, query_nope = query.split(
                [self.rope_dim, self.index_head_dim - self.rope_dim],
                dim=-1,
            )
            key_pe, key_nope = key.split(
                [self.rope_dim, self.index_head_dim - self.rope_dim],
                dim=-1,
            )
            query_pe, key_pe = self.rotary_emb(
                positions,
                query_pe,
                key_pe[:, None, :],
            )
            query = torch.cat((query_pe, query_nope), dim=-1)
            key = torch.cat((key_pe.squeeze(1), key_nope), dim=-1)
        query = hadamard_transform(
            query.contiguous(),
            scale=self.index_head_dim**-0.5,
        )
        gate = F.linear(hidden_states, self.index_kpool_compress_gate)
        return Glm53FlashIndexerOutput(
            query=query,
            key=key,
            weights=weights,
            gate=gate,
        )


class Glm53FlashAttention(GlmMoeDsaAttention):
    """Sparse MLA attention with pooled index cache and FlatKV selections."""

    def __init__(
        self,
        config: Glm53FlashTextConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        DeepseekV3AttentionMLA.__init__(
            self,
            config=config,
            mapping=mapping,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=10000.0,
            rope_scaling=None,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=prefix,
            reduce_attn_results=False,
            alt_stream=alt_stream,
            skip_rope=True,
        )
        # DSA uses the same latent-history cache group as K3 MLA. The inherited
        # PagedAttention modules are created without a group id.
        self.attn_mqa.group_id = FULL_ATTENTION
        self.attn_mha.group_id = FULL_ATTENTION
        self.q_a_layernorm = RMSNorm(config.q_lora_rank, eps=1e-6)
        self.kv_a_layernorm = RMSNorm(config.kv_lora_rank, eps=1e-6)
        self.fused_qk_layernorm = FusedRMSNorm(
            self.q_a_layernorm,
            self.kv_a_layernorm,
        )
        self.index_topk = config.index_topk
        self.index_kpool = config.index_kpool
        self.is_nextn = False
        self.skip_indexer_topk = config.indexer_types[layer_id] == "shared"
        self.indexer = (
            None
            if self.skip_indexer_topk
            else Glm53FlashIndexer(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("indexer", prefix),
            )
        )
        self._decode_topk_indices_buffer: torch.Tensor | None = None
        self._decode_topk_lens_buffer: torch.Tensor | None = None
        self._absorbed_kv_b_version = -1

    def _prepare_absorbed_mla_weights(self) -> None:
        self.w_kc, self.w_vc = _prepare_mla_kv_b_proj_weights(
            self.kv_b_proj.weight,
            self,
        )
        self._absorbed_kv_b_version = self.kv_b_proj.weight._version

    @break_point
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        if isinstance(ctx.attn_backend, HybridLinearAttnBackend):
            dsa_backend = ctx.attn_backend.full_attn_backend
        elif isinstance(ctx.attn_backend, DSABackend):
            dsa_backend = ctx.attn_backend
        else:
            raise TypeError(
                "GLM-5.3-Flash sparse attention requires a DSA backend, got "
                f"{type(ctx.attn_backend).__name__}."
            )
        kpool_runtime = dsa_backend.require_kpool_runtime()
        kpool_runtime.ensure_prefill_plan(
            ctx,
            dsa_backend,
            self.attn_mqa.layer_id,
            token_capacity=hidden_states.shape[0],
        )
        ctx = replace(ctx, attn_backend=dsa_backend)
        qkv = self.fused_qkv_a_proj_with_mqa(
            hidden_states,
            block_scale,
            torch.bfloat16,
        )
        qkv_width = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
        if qkv.shape[-1] != qkv_width:
            qkv = qkv[..., :qkv_width]
        qkv = comm_manager.pre_attn_comm(qkv, ctx)
        q_a, latent_cache = qkv.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        kv_a = latent_cache[..., : self.kv_lora_rank]
        q_norm = torch.empty_like(q_a)
        if q_a.size(0) > 0:
            self.fused_qk_layernorm(
                input_q_a=q_a,
                input_kv_a=kv_a,
                output_q_a=q_norm,
            )

        logical_num_tokens = q_norm.shape[0]
        if ctx.num_extends > 0:
            prefill_plan = dsa_backend.forward_metadata.kpool_prefill_plan
            if prefill_plan is None:
                raise RuntimeError("GLM-5.3-Flash prefill requires a KPool plan")
            num_decode_reqs = self._resolve_decode_req_count(
                ctx,
                dsa_backend.forward_decode_metadata,
            )
            spec_width = int(getattr(dsa_backend, "spec_num_tokens", 1) or 1)
            logical_num_tokens = (
                prefill_plan.num_prefill_tokens + num_decode_reqs * spec_width
            )
            real_input_num_tokens = getattr(ctx, "real_input_num_tokens", None)
            if (
                real_input_num_tokens is not None
                and logical_num_tokens != real_input_num_tokens
            ):
                raise RuntimeError(
                    "GLM-5.3-Flash live row metadata disagrees with the scheduler: "
                    f"metadata={logical_num_tokens}, scheduler={real_input_num_tokens}"
                )
            if logical_num_tokens > q_norm.shape[0]:
                raise RuntimeError(
                    "GLM-5.3-Flash live rows exceed the padded input capacity: "
                    f"live={logical_num_tokens}, capacity={q_norm.shape[0]}"
                )

        decode_window = self._resolve_decode_window(
            ctx,
            dsa_backend.forward_decode_metadata,
            total_tokens=logical_num_tokens,
        )
        num_prefill_tokens = decode_window.start
        num_decode_tokens = decode_window.num_tokens
        decode_start = decode_window.start
        decode_end = decode_window.end
        if ctx.num_extends > 0:
            prefill_plan = dsa_backend.forward_metadata.kpool_prefill_plan
            if num_prefill_tokens != prefill_plan.num_prefill_tokens:
                raise RuntimeError(
                    "GLM-5.3-Flash live row split disagrees with KPool metadata: "
                    f"split={num_prefill_tokens}, "
                    f"prefill={prefill_plan.num_prefill_tokens}"
                )

        should_compute_indexer = not self.skip_indexer_topk or (
            self.is_nextn
            and (
                (num_prefill_tokens > 0 and ctx.dsa_prefill_topk is None)
                or (num_decode_tokens > 0 and ctx.dsa_decode_topk is None)
            )
        )
        if should_compute_indexer:
            hidden_states = comm_manager.pre_attn_comm(hidden_states, ctx)
            indexer_output = self.indexer(hidden_states, q_norm, positions)
            if num_prefill_tokens > 0:
                kpool_runtime.write_prefill(
                    key=indexer_output.key,
                    gate=indexer_output.gate,
                    compress_ape=self.indexer.index_kpool_compress_ape,
                    ctx=ctx,
                    backend=dsa_backend,
                    layer_id=self.attn_mqa.layer_id,
                )
            if num_decode_tokens > 0:
                kpool_runtime.write_decode(
                    key=indexer_output.key[decode_start:decode_end],
                    gate=indexer_output.gate[decode_start:decode_end],
                    compress_ape=self.indexer.index_kpool_compress_ape,
                    ctx=ctx,
                    backend=dsa_backend,
                    layer_id=self.attn_mqa.layer_id,
                    num_reqs=decode_window.num_reqs,
                    q_len_per_req=decode_window.q_len_per_req,
                )
            if ctx.num_extends > 0:
                ctx.dsa_prefill_topk = self._compute_prefill_topk_indices(
                    indexer_output,
                    ctx,
                    num_prefill_tokens,
                )
            if ctx.num_extends < ctx.bs:
                ctx.dsa_decode_topk = self._compute_decode_topk_indices(
                    indexer_output,
                    ctx,
                    logical_num_tokens=logical_num_tokens,
                )

        q = self.q_b_proj(q_norm)[0]
        attn_output = torch.empty(
            q.size(0),
            self.num_local_heads * self.v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )

        if ctx.num_extends > 0:
            prefill_ctx = replace(
                ctx,
                bs=ctx.num_extends,
                input_num_tokens=q.shape[0],
                forward_mode=ForwardMode.EXTEND,
            )
            if ctx.dsa_prefill_topk is None:
                raise RuntimeError(
                    "GLM-5.3-Flash sparse prefill requires computed top-k indices."
                )
            self.forward_dsa_sparse_prefill(
                positions,
                q,
                latent_cache,
                prefill_ctx,
                out_cache_loc,
                attn_output,
                prefill_topk=ctx.dsa_prefill_topk,
                cache_num_tokens=num_prefill_tokens,
            )

        if num_decode_tokens > 0:
            decode_ctx = replace(
                ctx,
                bs=decode_window.num_reqs,
                num_extends=0,
                input_num_tokens=num_decode_tokens,
                forward_mode=ForwardMode.DECODE,
            )
            if ctx.dsa_decode_topk is None:
                raise RuntimeError(
                    "GLM-5.3-Flash sparse decode requires computed top-k indices."
                )
            topk_indices, topk_lens = self._slice_decode_topk(
                ctx.dsa_decode_topk,
                decode_start,
                decode_end,
            )
            self.forward_absorb(
                positions[decode_start:decode_end],
                q[decode_start:decode_end],
                latent_cache[decode_start:decode_end],
                decode_ctx,
                out_cache_loc[decode_start:decode_end],
                attn_output[decode_start:decode_end],
                topk_indices=topk_indices,
                topk_lens=topk_lens,
            )

        if ctx.accept_lengths is not None:
            attn_output = attn_output.index_select(0, ctx.gather_ids)
        output, _ = self.o_proj(attn_output)
        return output

    def _compute_decode_topk_indices_portable(
        self,
        *,
        indexer_output: Glm53FlashIndexerOutput,
        ctx: ForwardContext,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        q_len_per_req: int,
        decode_start: int,
        num_tokens: int,
        num_decode_tokens: int,
        topk: int,
    ) -> GlmDsaDecodeTopK:
        del topk
        topk_indices = self._get_decode_topk_workspace(
            "_decode_topk_indices_buffer",
            num_tokens,
            self.index_topk + self.index_kpool - 1,
            indexer_output.query.device,
        )
        topk_lens = self._get_decode_topk_lens_workspace(
            num_tokens,
            indexer_output.query.device,
        )
        ctx.attn_backend.require_kpool_runtime().select_decode(
            query=indexer_output.query,
            weights=indexer_output.weights,
            softmax_scale=self.indexer.weights_softmax_scale,
            ctx=ctx,
            layer_id=self.attn_mqa.layer_id,
            seq_lens=seq_lens,
            page_table=page_table,
            q_len_per_req=q_len_per_req,
            decode_start=decode_start,
            num_decode_tokens=num_decode_tokens,
            out=topk_indices,
            lens_out=topk_lens,
        )
        return GlmDsaDecodeTopK(
            topk_indices=topk_indices,
            topk_lens=topk_lens,
        )

    def _compute_prefill_topk_indices(
        self,
        indexer_output: Glm53FlashIndexerOutput,
        ctx: ForwardContext,
        num_prefill_tokens: int,
    ) -> GlmDsaPrefillTopK | None:
        selected = ctx.attn_backend.require_kpool_runtime().select_prefill(
            query=indexer_output.query,
            weights=indexer_output.weights,
            softmax_scale=self.indexer.weights_softmax_scale,
            ctx=ctx,
            backend=ctx.attn_backend,
            layer_id=self.attn_mqa.layer_id,
            num_prefill_tokens=num_prefill_tokens,
        )
        if selected is None:
            return None
        return GlmDsaPrefillTopK(
            workspace_indices=selected.workspace_indices,
            topk_lens=selected.topk_lens,
            page_table=selected.page_table,
            seq_lens=selected.seq_lens,
            kv_seq_lens=selected.kv_seq_lens,
            max_seq_len=selected.max_seq_len,
            kv_workspace_slots=selected.kv_workspace_slots,
        )


class Glm53FlashDecoderLayer(nn.Module):
    """One mixed KDA/DSA decoder layer with native GLM-5.3-Flash mHC."""

    def __init__(
        self,
        config: Glm53FlashTextConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_id = layer_id
        self.mhc = config.mhc and not is_nextn
        self.is_kda_layer = False if is_nextn else config.is_kda_layer(layer_id)
        self.is_moe_layer = (
            True if is_nextn else config.mlp_layer_types[layer_id] == "sparse"
        )

        if self.is_kda_layer:
            self.self_attn = Glm53FlashKDA(
                config=config,
                mapping=mapping,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
            )
        else:
            self.self_attn = Glm53FlashAttention(
                config=config,
                mapping=mapping,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("self_attn", prefix),
                alt_stream=alt_stream,
            )
            self.self_attn.is_nextn = is_nextn

        if self.is_moe_layer:
            self.mlp = Glm53FlashMoE(
                config=config,
                mapping=mapping,
                quant_config=quant_config,
                layer_index=layer_id,
                prefix=add_prefix("mlp", prefix),
                alt_stream=alt_stream,
            )
        else:
            self.mlp = Glm53FlashMLP(
                config=config,
                intermediate_size=config.intermediate_size,
                mapping=mapping,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        prev_is_moe = (
            True
            if is_nextn
            else (
                config.mlp_layer_types[layer_id - 1] == "sparse" if layer_id else False
            )
        )
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=self.is_moe_layer,
            prev_is_moe=prev_is_moe,
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
        )

        if self.mhc:
            hc_dim = config.hc_mult * config.hidden_size
            mix_hc = (2 + config.hc_mult) * config.hc_mult
            self.hc_attn_fn = nn.Parameter(
                torch.empty(mix_hc, hc_dim, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_attn_base = nn.Parameter(
                torch.empty(mix_hc, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_attn_scale = nn.Parameter(
                torch.empty(3, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_ffn_fn = nn.Parameter(
                torch.empty(mix_hc, hc_dim, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_ffn_base = nn.Parameter(
                torch.empty(mix_hc, dtype=torch.float32),
                requires_grad=False,
            )
            self.hc_ffn_scale = nn.Parameter(
                torch.empty(3, dtype=torch.float32),
                requires_grad=False,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_global_tokens, max_num_tokens_per_gpu = self.comm_manager.get_num_tokens(
            ctx
        )

        if self.mhc:
            residual_streams = hidden_states
            hidden_states, post, comb = mhc_pre(
                residual_streams,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.config.rms_norm_eps,
                self.config.hc_eps,
                self.config.hc_sinkhorn_iters,
            )
            hidden_states = self.input_layernorm(hidden_states)
            if self.is_kda_layer:
                hidden_states = self.comm_manager.pre_attn_comm(hidden_states, ctx)
            else:
                if (
                    self.self_attn.w_kc is None
                    or self.self_attn._absorbed_kv_b_version
                    != self.self_attn.kv_b_proj.weight._version
                ):
                    self.self_attn._prepare_absorbed_mla_weights()
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
                comm_manager=self.comm_manager,
            )
            if ctx.accept_lengths is not None:
                residual_streams = residual_streams.index_select(0, ctx.gather_ids)
            hidden_states, residual_streams = self.comm_manager.post_attn_comm(
                hidden_states,
                residual_streams,
                ctx,
            )
            hidden_states = mhc_post(hidden_states, residual_streams, post, comb)
            residual_streams = hidden_states
            hidden_states, post, comb = mhc_pre(
                residual_streams,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                self.config.rms_norm_eps,
                self.config.hc_eps,
                self.config.hc_sinkhorn_iters,
            )
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
            if self.is_moe_layer:
                hidden_states = self.mlp(
                    hidden_states,
                    num_global_tokens,
                    max_num_tokens_per_gpu,
                )
            else:
                hidden_states = self.mlp(hidden_states)
            hidden_states, _ = self.comm_manager.post_mlp_comm(
                hidden_states,
                residual_streams,
                ctx,
            )
            hidden_states = mhc_post(hidden_states, residual_streams, post, comb)
            return hidden_states, None

        hidden_states, residual = self.comm_manager.input_reduce_norm(
            hidden_states,
            residual,
        )
        if self.is_kda_layer:
            hidden_states = self.comm_manager.pre_attn_comm(hidden_states, ctx)
        else:
            if (
                self.self_attn.w_kc is None
                or self.self_attn._absorbed_kv_b_version
                != self.self_attn.kv_b_proj.weight._version
            ):
                self.self_attn._prepare_absorbed_mla_weights()
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
            comm_manager=self.comm_manager,
        )
        if ctx.accept_lengths is not None:
            residual = residual.index_select(0, ctx.gather_ids)
        hidden_states, residual = self.comm_manager.post_attn_reduce_norm(
            hidden_states,
            residual,
            ctx,
        )
        hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
        if self.is_moe_layer:
            hidden_states = self.mlp(
                hidden_states,
                num_global_tokens,
                max_num_tokens_per_gpu,
            )
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states, residual = self.comm_manager.post_mlp_fused(
            hidden_states,
            residual,
            ctx,
        )
        return hidden_states, residual


class Glm53FlashModel(nn.Module):
    """GLM-5.3-Flash text backbone with alternating KDA and sparse MLA layers."""

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: Glm53FlashTextConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.layers = nn.ModuleList(
            [
                Glm53FlashDecoderLayer(
                    config=config,
                    layer_id=layer_id,
                    mapping=mapping,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    alt_stream=self.alt_stream,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture: list[int] = []

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        residual = None
        if self.config.mhc:
            hidden_states = hidden_states.unsqueeze(1).repeat(
                1,
                self.config.hc_mult,
                1,
            )

        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                ctx,
                out_cache_loc,
                residual,
            )
        if self.config.mhc:
            hidden_states, _ = self.layers[-1].comm_manager.post_final_norm_comm(
                hidden_states,
                hidden_states,
                ctx,
            )
            hidden_states = hidden_states.mean(dim=1)
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.layers[-1].comm_manager.final_norm(
                hidden_states,
                residual,
                ctx,
                self.norm,
            )
        aux_hidden_states = None
        if (
            ctx.capture_hidden_mode is not None
            and ctx.capture_hidden_mode.need_capture()
        ):
            aux_hidden_states = [hidden_states]
        return hidden_states, aux_hidden_states


class _Glm53FlashWeightOwner(Protocol):
    config: object
    mapping: object
    quant_config: object | None

    def named_modules(self, *args, **kwargs) -> Iterable[tuple[str, nn.Module]]: ...

    def named_parameters(
        self, *args, **kwargs
    ) -> Iterable[tuple[str, nn.Parameter]]: ...


def load_glm53_flash_text_weights(
    owner: _Glm53FlashWeightOwner,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> None:
    """Route normalized GLM-5.3-Flash text weights into target or NextN modules."""
    params_dict = dict(owner.named_parameters())
    modules_dict = dict(owner.named_modules())
    loaded_indexer_shards: dict[str, set[int]] = {}
    moe_loader = build_moe_checkpoint_loader(
        params_dict=params_dict,
        expert_schema=ExpertCheckpointSchema(),
        num_experts=owner.config.n_routed_experts,
        ep_rank=owner.mapping.moe.ep_rank,
        ep_size=owner.mapping.moe.ep_size,
    )

    for name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in name or "hc_head" in name:
            continue
        if "_conv1d.weight" in name:
            name = name.replace("_conv1d.weight", "_conv1d_weight")

        kda_input_shard = None
        for weight_name, shard_id in (
            (".self_attn.q_proj.", 0),
            (".self_attn.k_proj.", 1),
            (".self_attn.v_proj.", 2),
            (".self_attn.b_proj.", 3),
            (".self_attn.f_a_proj.", 4),
            (".self_attn.g_a_proj.", 5),
        ):
            if weight_name in name:
                kda_input_shard = (weight_name, shard_id)
                break
        if kda_input_shard is not None:
            weight_name, shard_id = kda_input_shard
            mapped = name.replace(
                weight_name,
                ".self_attn.fused_qkvbfg_a_proj.",
            )
            param = params_dict.get(mapped)
            if param is not None:
                param.weight_loader(param, loaded_weight, shard_id)
            continue

        if ".indexer.wk." in name or ".indexer.weights_proj." in name:
            direct_param = params_dict.get(name)
            if direct_param is not None:
                if hasattr(direct_param, "weight_loader"):
                    direct_param.weight_loader(direct_param, loaded_weight)
                else:
                    default_weight_loader(direct_param, loaded_weight)
            if name.endswith(".weight"):
                if ".indexer.wk." in name:
                    module_name = name.rsplit(".wk.", 1)[0]
                    shard_id = 0
                else:
                    module_name = name.rsplit(".weights_proj.", 1)[0]
                    shard_id = 1
                fused_param = params_dict.get(f"{module_name}.wk_weights_proj.weight")
                if fused_param is not None:
                    fused_param.weight_loader(
                        fused_param,
                        loaded_weight,
                        shard_id,
                    )
                    shards = loaded_indexer_shards.setdefault(module_name, set())
                    shards.add(shard_id)
                    module = modules_dict.get(module_name)
                    if shards == {0, 1} and hasattr(module, "_wk_weights_proj_loaded"):
                        module._wk_weights_proj_loaded = True
            continue

        if moe_loader.matches(name):
            moe_loader.load(name, loaded_weight)
            continue

        if ".q_a_proj." in name or ".kv_a_proj_with_mqa." in name:
            if ".q_a_proj." in name:
                mapped = name.replace(
                    ".q_a_proj.",
                    ".fused_qkv_a_proj_with_mqa.",
                )
                begin_size = 0
            else:
                mapped = name.replace(
                    ".kv_a_proj_with_mqa.",
                    ".fused_qkv_a_proj_with_mqa.",
                )
                begin_size = owner.config.q_lora_rank
            if "scale_inv" in name:
                weight_block_size = getattr(
                    owner.quant_config, "weight_block_size", None
                )
                if weight_block_size is None:
                    raise RuntimeError(
                        "FP8 scale checkpoint requires weight_block_size"
                    )
                begin_size //= weight_block_size[0]
            param = params_dict.get(mapped)
            if param is not None:
                param.weight_loader(
                    param,
                    loaded_weight,
                    begin_size=begin_size,
                )
            continue

        stacked = False
        for param_name, weight_name, shard_id in (
            (".gate_up_proj.", ".gate_proj.", 0),
            (".gate_up_proj.", ".up_proj.", 1),
        ):
            if weight_name not in name or ".experts." in name:
                continue
            mapped = name.replace(weight_name, param_name)
            param = params_dict.get(mapped)
            if param is not None:
                param.weight_loader(param, loaded_weight, shard_id)
            stacked = True
            break
        if stacked:
            continue

        param = params_dict.get(name)
        if param is None:
            continue
        if hasattr(param, "weight_loader"):
            param.weight_loader(param, loaded_weight)
        else:
            default_weight_loader(param, loaded_weight)


class Glm53FlashForCausalLM(BaseCausalLM):
    """GLM-5.3-Flash text model, LM head, and streaming checkpoint loader."""

    model_cls = Glm53FlashModel

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        def text_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, loaded_weight in weights:
                if name.startswith("model.layers."):
                    layer_name = name.split(".", 3)[2]
                    if (
                        layer_name.isdigit()
                        and int(layer_name) >= self.config.num_hidden_layers
                    ):
                        continue
                yield name, loaded_weight

        load_glm53_flash_text_weights(self, text_weights())
        self.post_load_weights()

    def post_load_weights(self) -> None:
        for layer in self.model.layers:
            if isinstance(layer.self_attn, Glm53FlashAttention):
                layer.self_attn._prepare_absorbed_mla_weights()
            else:
                layer.self_attn.fuse_conv_weights()
            if isinstance(layer.mlp, Glm53FlashMoE):
                layer.mlp.experts.process_weights_after_loading(layer.mlp.experts)


# ===----------------------------------------------------------------------=== #
# Registered multimodal wrapper
# ===----------------------------------------------------------------------=== #


class Glm53FlashForConditionalGeneration(nn.Module):
    """Top-level checkpoint hierarchy for the GLM-5.3-Flash multimodal model."""

    def __init__(
        self,
        config: Glm53FlashConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config
        self.is_multimodal_active = is_multimodal_active
        encoder_only = hasattr(config, "encoder_only") and config.encoder_only
        self.language_model = (
            None
            if encoder_only
            else Glm53FlashForCausalLM(
                config.text_config,
                mapping=mapping,
                quant_config=quant_config,
            )
        )
        if is_multimodal_active:
            self.vision = Glm53FlashVision(
                config.vision_config,
                mapping,
                prefix=add_prefix("visual", prefix),
                mm_attention_backend=mm_attention_backend,
            )
            if self.language_model is not None:
                target_dtype = self.get_input_embeddings().weight.dtype
                self.vision = self.vision.to(dtype=target_dtype)
            self.vision_embedder = VisionEmbedder(encoder_mapping=mapping.vision)
            self.image_encoder = self.vision.embed_media
            self.video_encoder = self.vision.embed_media
        else:
            self.vision = None
            self.vision_embedder = None
            self.image_encoder = None
            self.video_encoder = None

    def get_input_embeddings(self) -> nn.Module:
        if self.language_model is None:
            raise AttributeError(
                "GLM-5.3-Flash encoder-only mode does not expose text embeddings"
            )
        return self.language_model.model.get_input_embeddings()

    def get_embed_and_head(self):
        if self.language_model is None:
            raise AttributeError(
                "GLM-5.3-Flash encoder-only mode does not expose an LM head"
            )
        return self.language_model.get_embed_and_head()

    @property
    def logits_processor(self):
        if self.language_model is None:
            raise AttributeError(
                "GLM-5.3-Flash encoder-only mode does not expose a logits processor"
            )
        return self.language_model.logits_processor

    @property
    def lm_head(self):
        if self.language_model is None:
            raise AttributeError(
                "GLM-5.3-Flash encoder-only mode does not expose an LM head"
            )
        return self.language_model.lm_head

    def pad_input_ids(
        self,
        input_ids: list[int],
        mm_inputs: MultimodalInputs,
    ) -> list[int]:
        return pad_input_tokens(input_ids, mm_inputs)

    @torch.no_grad()
    def multimodal_input_embeds(
        self,
        input_ids: torch.Tensor,
        ctx: ForwardContext,
        multimodal_context,
    ) -> torch.Tensor | None:
        if (
            multimodal_context is None
            or self.vision_embedder is None
            or not multimodal_context.has_extend_inputs()
            or ctx.forward_mode.is_decode_or_idle()
        ):
            return None
        input_embeds, model_kwargs = self.vision_embedder.apply(
            input_ids=input_ids,
            text_embedding=self.get_input_embeddings(),
            ctx=multimodal_context,
            encoders={
                Modality.IMAGE: EncoderSpec(self.image_encoder),
                Modality.VIDEO: EncoderSpec(self.video_encoder),
            },
            multimodal_model=self,
        )
        if model_kwargs:
            raise RuntimeError(
                "GLM-5.3-Flash multimodal path must remain embedding-only"
            )
        return input_embeds

    @torch.no_grad()
    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if self.language_model is None:
            raise RuntimeError(
                "GLM-5.3-Flash encoder-only mode cannot run language forward"
            )
        multimodal_context = kwargs.pop("multimodal_context", None)
        input_embeds = self.multimodal_input_embeds(
            input_ids,
            ctx,
            multimodal_context,
        )
        if input_embeds is not None:
            kwargs["input_embeds"] = input_embeds
        return self.language_model.forward(
            ctx,
            input_ids,
            positions,
            out_cache_loc,
            **kwargs,
        )

    def post_load_weights(self) -> None:
        if self.language_model is not None:
            self.language_model.post_load_weights()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Route vision weights and stream the normalized text weights."""
        vision_params = (
            dict(self.vision.named_parameters(remove_duplicate=False))
            if self.vision is not None
            else None
        )

        def language_weights():
            for name, weight in weights:
                if name.startswith("model.visual."):
                    vision_name = name[len("model.visual.") :]
                elif name.startswith("visual."):
                    vision_name = name[len("visual.") :]
                else:
                    if name.startswith("model.language_model."):
                        name = name[len("model.language_model.") :]
                        if not name.startswith("model.") and not name.startswith(
                            "lm_head."
                        ):
                            name = "model." + name
                    elif name.startswith("language_model."):
                        name = name[len("language_model.") :]
                    yield name, weight
                    continue
                vision_name = vision_name.replace(".attn.qkv.", ".attn.qkv_proj.")
                if vision_params is None:
                    continue
                param = vision_params.get(vision_name)
                if param is None:
                    continue
                if hasattr(param, "weight_loader"):
                    param.weight_loader(param, weight)
                else:
                    default_weight_loader(param, weight)

        if self.language_model is not None:
            self.language_model.load_weights(language_weights())
        else:
            for _ in language_weights():
                pass


EntryClass = [Glm53FlashForConditionalGeneration]
