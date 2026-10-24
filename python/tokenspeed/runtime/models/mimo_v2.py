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

"""MiMo-V2 (Xiaomi MiMo-V2.5 / V2.6) text inference.

Every layer is GQA with 192-wide query/key heads (a 64-wide NeoX RoPE on the
leading dims) and 128-wide value heads scaled by ``attention_value_scale``.
``hybrid_layer_pattern`` marks sliding-window layers (window 128, a learned
per-head attention sink, their own RoPE base) among full-attention layers.
Layer 0 is a dense MLP; the rest route DeepSeek-V3 style (sigmoid scores plus
a correction bias, FP32 router) over experts stored as MXFP4.

The checkpoint fuses each layer's QKV projection and pre-shards it for
``num_key_value_heads`` ranks: ``[Q_0 | K_0 | V_0 | ... | Q_n | K_n | V_n]``,
with the FP8 block scales tiled per chunk. ``MiMoV2QKVProjection`` keeps the
chunks this rank owns without re-quantizing them.
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear, RowParallelLinear
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.topk import TopK, TopKOutputFormat
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType
from tokenspeed.runtime.layers.paged_attention import (
    PagedAttention,
    hf_sliding_window_to_window_left,
)
from tokenspeed.runtime.layers.quantization import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3MoE,
    get_layer_id,
)
from tokenspeed.runtime.models.utils import validate_attention_partition
from tokenspeed.runtime.utils import add_prefix

# The checkpoint's FP8 block size along the output rows of the fused QKV.
_QKV_SCALE_BLOCK = 128


def _is_sliding_layer(config, layer_id: int) -> bool:
    return config.hybrid_layer_pattern[layer_id] == 1


class MiMoV2QKVProjection(nn.Module):
    """This rank's chunks of the pre-sharded fused QKV projection.

    A checkpoint chunk holds ``[Q | K | V]`` rows for ``1 / num_chunks`` of
    the query and KV heads. The rank stores its chunks as
    ``[Q_0 .. Q_m | KV_0 .. KV_m]`` with each ``KV_j = [K_j | V_j]`` padded to
    a whole FP8 scale block, so every block keeps the checkpoint's scale and
    the query comes out of the GEMM as one contiguous column range.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int,
        num_chunks: int,
        tp_rank: int,
        tp_size: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        if num_heads % num_chunks or num_kv_heads % num_chunks:
            raise ValueError(
                f"MiMo-V2 fused QKV is pre-sharded into {num_chunks} chunks, "
                f"which must divide {num_heads} query and {num_kv_heads} KV heads"
            )
        if num_chunks % tp_size:
            raise ValueError(
                f"attention TP size {tp_size} must divide the {num_chunks} "
                "pre-sharded fused QKV chunks of the MiMo-V2 checkpoint"
            )
        self.num_chunks = num_chunks
        self.num_local_chunks = num_chunks // tp_size
        self.first_chunk = tp_rank * self.num_local_chunks
        self.q_rows = num_heads // num_chunks * head_dim
        self.k_rows = num_kv_heads // num_chunks * head_dim
        self.v_rows = num_kv_heads // num_chunks * v_head_dim
        if self.q_rows % _QKV_SCALE_BLOCK:
            raise ValueError(
                f"a fused QKV chunk's {self.q_rows} query rows must fill whole "
                f"{_QKV_SCALE_BLOCK}-row FP8 scale blocks"
            )
        kv_rows = self.k_rows + self.v_rows
        self.kv_blocks = -(-kv_rows // _QKV_SCALE_BLOCK)
        self.kv_stride = self.kv_blocks * _QKV_SCALE_BLOCK
        self.q_size = self.num_local_chunks * self.q_rows
        self.proj = ReplicatedLinear(
            hidden_size,
            self.q_size + self.num_local_chunks * self.kv_stride,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.proj(hidden_states)[0]
        q = out[:, : self.q_size]
        kv = out[:, self.q_size :].view(-1, self.num_local_chunks, self.kv_stride)
        k = kv[:, :, : self.k_rows].flatten(1)
        v = kv[:, :, self.k_rows : self.k_rows + self.v_rows].flatten(1)
        return q, k, v

    def load(self, kind: str, loaded: torch.Tensor) -> None:
        """Copy this rank's chunks of ``weight`` or ``weight_scale_inv``."""
        rows_per_chunk = self.q_rows + self.k_rows + self.v_rows
        if kind == "weight":
            unit, q_units, kv_units, kv_stride = 1, self.q_rows, None, self.kv_stride
            chunk_units = rows_per_chunk
            param = self.proj.weight
        elif kind == "weight_scale_inv":
            unit = _QKV_SCALE_BLOCK
            q_units = self.q_rows // unit
            kv_stride = self.kv_blocks
            chunk_units = q_units + self.kv_blocks
            param = self.proj.weight_scale_inv
        else:
            raise ValueError(f"Unexpected MiMo-V2 qkv_proj tensor {kind!r}")
        if loaded.shape[0] != self.num_chunks * chunk_units:
            raise ValueError(
                f"MiMo-V2 qkv_proj.{kind} has {loaded.shape[0]} rows, expected "
                f"{self.num_chunks} chunks of {chunk_units}"
            )
        chunks = loaded.view(self.num_chunks, chunk_units, *loaded.shape[1:])
        local = chunks[self.first_chunk : self.first_chunk + self.num_local_chunks]
        q_total = self.num_local_chunks * q_units
        dst = param.data
        dst[:q_total].copy_(local[:, :q_units].flatten(0, 1))
        kv_dst = dst[q_total:].view(self.num_local_chunks, kv_stride, *dst.shape[1:])
        kv_units = chunk_units - q_units
        kv_dst[:, :kv_units].copy_(local[:, q_units:])
        # Padding rows stay zero: they produce discarded output columns.
        kv_dst[:, kv_units:].zero_()


class MiMoV2Attention(nn.Module):
    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        is_sliding = _is_sliding_layer(config, layer_id)
        self.hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = (
            config.swa_num_key_value_heads if is_sliding else config.num_key_value_heads
        )
        head_dim, v_head_dim = config.head_dim, config.v_head_dim
        attn_tp_size = mapping.attn.tp_size
        validate_attention_partition(num_heads, num_kv_heads, attn_tp_size)
        self.num_local_heads = num_heads // attn_tp_size
        self.v_head_dim = v_head_dim
        self.v_scale = config.attention_value_scale

        self.qkv_proj = MiMoV2QKVProjection(
            hidden_size=config.hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            # The checkpoint pre-shards every layer by the model-level count.
            num_chunks=config.num_key_value_heads,
            tp_rank=mapping.attn.tp_rank,
            tp_size=attn_tp_size,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            num_heads * v_head_dim,
            config.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=attn_tp_size,
            tp_group=mapping.attn.tp_group,
        )
        has_sink = (
            config.add_swa_attention_sink_bias
            if is_sliding
            else config.add_full_attention_sink_bias
        )
        self.attention_sink_bias = (
            nn.Parameter(
                torch.empty(self.num_local_heads, dtype=torch.bfloat16),
                requires_grad=False,
            )
            if has_sink
            else None
        )
        self.head_start = mapping.attn.tp_rank * self.num_local_heads
        rotary_dim = int(head_dim * config.partial_rotary_factor)
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=config.swa_rope_theta if is_sliding else config.rope_theta,
            is_neox_style=True,
        )
        self.attn = PagedAttention(
            self.num_local_heads,
            head_dim,
            head_dim**-0.5,
            num_kv_heads=max(1, num_kv_heads // attn_tp_size),
            layer_id=layer_id,
            v_head_dim=v_head_dim,
            sliding_window_size=(
                hf_sliding_window_to_window_left(config.sliding_window)
                if is_sliding
                else -1
            ),
            rotary_emb=self.rotary_emb,
            qk_norm=None,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        comm_manager: CommManager,
    ) -> torch.Tensor:
        hidden_states = comm_manager.pre_attn_comm(hidden_states, ctx)
        if hidden_states.shape[0] == 0:
            return hidden_states.new_empty(0, self.hidden_size)
        q, k, v = self.qkv_proj(hidden_states)
        # The reference scales values before attention; softmax weights are
        # unchanged, so this equals scaling the attention output.
        v = v * self.v_scale
        output = self.attn(q, k, v, positions, ctx, sinks=self.attention_sink_bias)
        return self.o_proj(output)[0]


class MiMoV2Router(nn.Module):
    """FP32 router logits, as the reference computes them."""

    def __init__(self, config) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                config.n_routed_experts, config.hidden_size, dtype=torch.float32
            ),
            requires_grad=False,
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states.float(), self.weight)


class MiMoV2MoE(DeepseekV3MoE):
    """Routed experts without shared experts, top-k computed outside the kernel.

    MiMo-V2 routes DeepSeek-V3 style over a single group of 256 (Flash) or 384
    (Pro) experts; the fused kernels' DeepSeek-V3 router does not cover a group
    that wide, so routing runs in TopK and the experts take precomputed top-k.
    """

    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        layer_index: int,
        prefix: str,
    ) -> None:
        nn.Module.__init__(self)
        self.mapping = mapping
        self.layer_index = layer_index
        self.n_shared_experts = None
        self.routed_scaling_factor = 1.0
        self.gate = MiMoV2Router(config)
        self.experts = MoELayer(
            top_k=config.num_experts_per_tok,
            num_experts=config.n_routed_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=layer_index,
            prefix=prefix,
            tp_rank=mapping.moe.tp_rank,
            tp_size=mapping.moe.tp_size,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
            routing_mode="precomputed_topk",
            routing_config={
                "n_group": config.n_group,
                "topk_group": config.topk_group,
                "routed_scaling_factor": self.routed_scaling_factor,
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
            routed_scaling_factor=self.routed_scaling_factor,
            # Always route here; the experts consume the top-k ids and weights.
            output_format=TopKOutputFormat.STANDARD,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        router_logits = self.gate(hidden_states)
        if hidden_states.shape[0] > 0:
            topk_output = self.topk(hidden_states, router_logits)
        else:
            topk_output = self.topk.empty_topk_output(
                hidden_states.device,
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        return self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
        )


class MiMoV2DecoderLayer(DeepseekV3DecoderLayer):
    def __init__(
        self,
        config,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.mapping = mapping
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.self_attn = MiMoV2Attention(
            config, mapping, layer_id, quant_config, add_prefix("self_attn", prefix)
        )
        self.is_moe_layer = self._is_moe_layer(layer_id, False, config)
        if self.is_moe_layer:
            self.mlp = MiMoV2MoE(
                config,
                mapping,
                quant_config,
                layer_index=layer_id,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = DeepseekV3MLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                mapping,
                quant_config,
                prefix=add_prefix("mlp", prefix),
                batch_invariant=False,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.layernorm_epsilon
        )
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=self.is_moe_layer,
            prev_is_moe=self._is_moe_layer(layer_id - 1, False, config),
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
            dense_batch_invariant=False,
            query_sharded=False,
        )

    @staticmethod
    def _is_moe_layer(layer_id: int, is_nextn: bool, config) -> bool:
        del is_nextn  # MiMo-V2 serves no NextN layer here
        return layer_id >= 0 and bool(config.moe_layer_freq[layer_id])


class MiMoV2Model(DeepseekV3Model):
    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.mapping = mapping
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                MiMoV2DecoderLayer(
                    config,
                    layer_id,
                    mapping,
                    quant_config,
                    add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.layers_to_capture = set()
        self._dflash_capture_idx_map = {}


class MiMoV2ForCausalLM(DeepseekV3ForCausalLM):
    model_cls = MiMoV2Model

    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        is_multimodal_active: bool = False,
        mm_attention_backend: str | None = None,
    ) -> None:
        del mm_attention_backend
        if is_multimodal_active:
            raise NotImplementedError(
                "MiMo-V2 vision and audio encoders are not implemented"
            )
        if config.attention_projection_layout != "fused_qkv":
            raise ValueError(
                "MiMo-V2 serving expects the fused_qkv attention projection, got "
                f"{config.attention_projection_layout!r}"
            )
        if (
            config.hidden_act != "silu"
            or config.scoring_func != "sigmoid"
            or config.topk_method != "noaux_tc"
            or config.n_shared_experts is not None
            or config.attention_bias
        ):
            raise ValueError("Unsupported MiMo-V2 configuration")
        if config.routed_scaling_factor not in (None, 1.0):
            raise ValueError("MiMo-V2 routes without an expert scaling factor")
        super().__init__(config, mapping, quant_config=quant_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
            ),
            num_experts=self.config.n_routed_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )
        loaded = set()
        for name, tensor in weights:
            # Text-only: the checkpoint also carries the vision and audio
            # encoders, speech embeddings and the MTP layers.
            if not name.startswith(("model.layers.", "model.embed_tokens.")) and (
                name not in ("model.norm.weight", "lm_head.weight")
            ):
                continue
            if moe_loader.matches(name):
                loaded.add(moe_loader.load(name, tensor))
                continue
            if moe_loader.is_expert_checkpoint_weight(name):
                continue
            prefix, _, kind = name.rpartition(".")
            if prefix.endswith(".self_attn.qkv_proj"):
                layer_id = get_layer_id(name)
                self.model.layers[layer_id].self_attn.qkv_proj.load(kind, tensor)
                loaded.add(f"{prefix}.proj.{kind}")
                continue
            if name.endswith(".self_attn.attention_sink_bias"):
                attn = self.model.layers[get_layer_id(name)].self_attn
                attn.attention_sink_bias.data.copy_(
                    tensor[attn.head_start : attn.head_start + attn.num_local_heads]
                )
                loaded.add(name)
                continue
            shard = None
            for source, index in (("gate_proj", 0), ("up_proj", 1)):
                if f".mlp.{source}." in name:
                    name = name.replace(f".mlp.{source}.", ".mlp.gate_up_proj.")
                    shard = index
                    break
            if name not in params:
                raise ValueError(f"Unexpected MiMo-V2 text weight: {name}")
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            if shard is None:
                loader(param, tensor)
            else:
                loader(param, tensor, shard)
            loaded.add(name)
        self.post_load_weights()
        return loaded

    def post_load_weights(self) -> None:
        """No derived weights: MiMo-V2 has no absorbed MLA projections."""


EntryClass = [MiMoV2ForCausalLM]
