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

"""Inference-only DeepseekV3 model."""

# ruff: noqa: E402

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention import attn_merge_state
from tokenspeed_kernel.ops.attention.mla import (
    mla_project_value,
    mla_project_value_prefers_contiguous_weight,
)
from tokenspeed_kernel.ops.attention.mla.tokenspeed_mla import mla_kv_pack_quantize_fp8
from tokenspeed_kernel.ops.attention.prologue import MLAExpandedKV, MLAPrologueOutput
from tokenspeed_kernel.ops.gemm import bmm
from tokenspeed_kernel.ops.gemm.cuda import dsv3_router_gemm
from tokenspeed_kernel.ops.gemm.cute_dsl import (
    nvfp4_gemm_swiglu_nvfp4_quant,
)
from tokenspeed_kernel.ops.gemm.trtllm import dsv3_fused_a_gemm
from tokenspeed_kernel.ops.moe.cuda import moe_finalize_fuse_shared
from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize
from tokenspeed_kernel.platform import current_platform
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.configs.numerics import BITWISE_ENVELOPES
from tokenspeed.runtime.configs.utils import get_rope_theta
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.utils import get_layer_id

_platform = current_platform()
_is_blackwell = _platform.is_blackwell
_is_hopper_plus = _platform.is_hopper_plus
_device_sm = _platform.arch_version.major * 10 + _platform.arch_version.minor
_FUSED_A_MAX_M = 16  # measured cliff: wins to M=16, flat ~1.35x loss from 18 to 64


from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_manager import (
    CommManager,
    head_tp_row_counts,
)
from tokenspeed.runtime.distributed.comm_ops import (
    all_gather,
    all_reduce,
    all_to_all_head_scatter,
    all_to_all_transpose,
    token_all_gather,
    token_reduce_scatter,
)
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
)
from tokenspeed.runtime.execution.context import (
    ForwardContext,
    report_collective_sizing,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.forward_step import (
    get_is_capture_mode,
    get_is_cuda_graph_phase,
)
from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.attention.dcp.cache import gather_mla_history
from tokenspeed.runtime.layers.dense.nvfp4 import Nvfp4LinearMethod
from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.topk import TopK
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType
from tokenspeed.runtime.layers.paged_attention import (
    PagedAttention,
    QueryShardGather,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.nvfp4 import Nvfp4Config
from tokenspeed.runtime.layers.quantization.utils import (
    block_dequant,
    should_exclude_quant_module,
)
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.model_loader.weight_utils import (
    bind_or_copy,
    default_weight_loader,
)
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.moe.expert_location import ModelConfigForExpertLocation
from tokenspeed.runtime.utils import (
    LazyValue,
    add_prefix,
    get_colorful_logger,
)
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.env import envs, global_server_args_dict

logger = get_colorful_logger(__name__)

_OPTIONAL_MISSING_WEIGHT_SUFFIXES = (
    ".k_scale",
    ".v_scale",
)


def _prepare_mla_kv_b_proj_weights(
    w: torch.Tensor, self_attn
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``kv_b_proj`` into the absorbed ``w_kc``/``w_vc`` pair.

    When ``self_attn`` already holds a pair of the same geometry (a live
    weight update re-running ``post_load_weights``), the new values are
    copied into it so captured CUDA graphs keep valid addresses.
    """
    w_kc, w_vc = w.unflatten(
        0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
    ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
    if mla_project_value_prefers_contiguous_weight(
        dtype=w.dtype,
        heads=w_vc.shape[0],
        latent_dim=w_vc.shape[2],
        value_dim=w_vc.shape[1],
    ):
        w_kc, w_vc = w_kc.contiguous(), w_vc.transpose(1, 2).contiguous()
    else:
        w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
        w_vc = w_vc.contiguous().transpose(1, 2)
    return (
        bind_or_copy(self_attn.w_kc, w_kc),
        bind_or_copy(self_attn.w_vc, w_vc),
    )


def _reject_query_shard(ctx: ForwardContext, where: str) -> None:
    """Fail loud when a sharded forward reaches an MLA path that attends every
    row of the span (the dense, expanded prologue with head-sharded weights);
    only the absorbed sparse path can take a query shard."""
    if ctx.query_shard is not None and ctx.query_shard.size > 1:
        raise RuntimeError(
            f"{where} cannot take a query shard: query context parallelism "
            "runs the absorbed sparse DSA prefill, whose prologue gathers the "
            "rotated latent across the group"
        )


class DeepseekV3MLP(nn.Module):
    """Dense SwiGLU MLP sharded over the dense (or shared-expert) TP group.

    ``batch_invariant`` selects the TP-batch-invariant layout: ``down_proj``
    is column-parallel on hidden and consumes the whole intermediate
    activation, all-gathered along its channels, so every output element is
    one full-K GEMM result and the layer's tail moves rows instead of
    summing partials (``CommManager(dense_batch_invariant=True)``). The
    layout needs a dense TP group and unquantized weights.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_shared_expert: bool = False,
        *,
        batch_invariant: bool,
    ) -> None:
        super().__init__()
        self.mapping = mapping
        if is_shared_expert:
            tp_rank = self.mapping.moe.tp_ep_rank
            tp_size = self.mapping.moe.tp_ep_size
            tp_group = self.mapping.moe.tp_ep_group
        else:
            tp_rank = self.mapping.dense.tp_rank
            tp_size = self.mapping.dense.tp_size
            tp_group = self.mapping.dense.tp_group
        self.batch_invariant = batch_invariant
        self.tp_group = tp_group
        if batch_invariant:
            if is_shared_expert:
                raise ValueError(
                    "the batch-invariant dense layout applies to dense layers, "
                    "not shared experts"
                )
            if tp_size == 1:
                raise ValueError(
                    "the batch-invariant dense layout needs a dense TP group"
                )
            if quant_config is not None:
                raise ValueError(
                    "the batch-invariant dense layout needs an unquantized down_proj"
                )

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        if batch_invariant:
            # Full K (the whole intermediate dim) on every rank; the output
            # is this rank's hidden shard of every gathered token.
            self.down_proj = ColumnParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_group=tp_group,
                quant_config=None,
                prefix=add_prefix("down_proj", prefix),
            )
        else:
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                reduce_results=False,  # Communication is handled externally and manually controlled
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_group=tp_group,
                quant_config=quant_config,
                prefix=add_prefix("down_proj", prefix),
            )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
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

    def forward(self, x):
        if self.batch_invariant:
            if x.size(0) == 0:
                # Keep the [T_full, H / W] shape the transposing tail expects.
                return x.new_empty(0, self.down_proj.output_size_per_partition)
            gate_up, _ = self.gate_up_proj(x)
            x = self.act_fn(gate_up)
            # Rank order along the channels matches the column shards of
            # gate_up_proj, so the gathered activation is in global order.
            x = all_gather(x, self.tp_group, dim=-1)
            x, _ = self.down_proj(x)
            return x

        if x.size(0) == 0:
            return x

        if self._use_nvfp4_gemm_swiglu_nvfp4_quant:
            x_fc1_fp4, x_fc1_scale = fp4_quantize(
                x,
                self.gate_up_proj.input_scale_inv,
            )
            x_fp4, x_scale = nvfp4_gemm_swiglu_nvfp4_quant(
                x_fc1_fp4,
                x_fc1_scale,
                self.gate_up_proj.weight_swiglu_interleaved,
                self.gate_up_proj.weight_scale_swiglu_interleaved,
                self.gate_up_proj.alpha,
                self.down_proj.input_scale_inv,
            )
            x, _ = self.down_proj((x_fp4, x_scale))
            return x

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MoEGate(nn.Module):
    _DSV3_ROUTER_GEMM_HIDDEN = (3072, 6144, 7168)

    def __init__(self, config, prefix: str = ""):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )
        if config.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((config.n_routed_experts), dtype=torch.float32)
            )
        else:
            self.e_score_correction_bias = None

        self.use_dsv3_router_gemm = (
            _is_hopper_plus
            and self.weight.dtype in (torch.bfloat16, torch.float32)
            and config.hidden_size in self._DSV3_ROUTER_GEMM_HIDDEN
        )

    def forward(self, hidden_states):
        if self.use_dsv3_router_gemm and hidden_states.size(0) > 0:
            logits = dsv3_router_gemm(
                hidden_states,
                self.weight,
                out_dtype=torch.float32,
            )
        else:
            logits = F.linear(hidden_states, self.weight, None)
        return logits


class DeepseekV3MoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        layer_index: int = -1,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.mapping = mapping
        self.layer_index = layer_index
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.stream_fork = StreamFork(alt_stream)

        if self.mapping.moe.ep_size > config.n_routed_experts:
            raise ValueError(
                f"EP size {self.mapping.moe.ep_size} is greater than the number of experts {config.n_routed_experts}."
            )
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. Only silu is supported for now."
            )

        self.gate = MoEGate(config=config, prefix=add_prefix("gate", prefix))

        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                mapping=self.mapping,
                quant_config=quant_config,
                prefix=add_prefix("shared_experts", prefix),
                is_shared_expert=True,
                batch_invariant=False,
            )

        self.experts = MoELayer(
            top_k=config.num_experts_per_tok,
            num_experts=config.n_routed_experts,
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
                "n_group": getattr(config, "n_group", 0),
                "topk_group": getattr(config, "topk_group", 0),
                "routed_scaling_factor": getattr(config, "routed_scaling_factor", 1.0),
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
            output_format=self.experts.topk_output_format,
        )

    def get_moe_routed_weights(self):
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
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)

        # Warm the shared-expert branch serially on its capture stream, then
        # overlap it with routed experts during capture.
        with self.stream_fork.scope(
            enable=get_is_cuda_graph_phase(), overlap=get_is_capture_mode()
        ) as fork:
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            if num_tokens > 0:
                topk_output = self.topk(hidden_states, router_logits)
            else:
                topk_output = self.topk.empty_topk_output(
                    hidden_states.device,
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                )

            deferred_finalize = self.experts.supports_deferred_finalize
            routed_expert_output = self.experts(
                hidden_states=hidden_states,
                topk_output=topk_output,
                num_global_tokens=num_global_tokens,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu,
                do_finalize=not deferred_finalize,
            )

            shared_output = None
            with fork.branch():
                if self.n_shared_experts is not None and num_tokens > 0:
                    shared_output = self.shared_experts(hidden_states)

        if deferred_finalize:
            gemm2_out, expert_weights, expanded_idx = routed_expert_output
            final_hidden_states = moe_finalize_fuse_shared(
                gemm2_out,
                expanded_idx,
                expert_weights,
                shared_output,
                top_k=self.topk.topk_config.top_k,
            )
        else:
            final_hidden_states = (
                routed_expert_output + shared_output
                if shared_output is not None
                else routed_expert_output
            )
        return final_hidden_states


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV3FusedQkvAProjWithMqa(ReplicatedLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        # ModelOpt NVFP4 checkpoints (e.g. DeepSeek-R1-0528-NVFP4-v2) keep the
        # q_a_proj / kv_a_proj_with_mqa weights as bf16 via exclude_modules.
        # exclude_modules matches by component name, not by the fused parent
        # prefix, so the fused layer would otherwise allocate an NVFP4-packed
        # buffer and crash when bf16 weights are copied in.
        if isinstance(quant_config, Nvfp4Config) and prefix:
            q_a_prefix = prefix.replace("fused_qkv_a_proj_with_mqa", "q_a_proj")
            kv_a_prefix = prefix.replace(
                "fused_qkv_a_proj_with_mqa", "kv_a_proj_with_mqa"
            )
            if should_exclude_quant_module(
                q_a_prefix, quant_config.exclude_modules
            ) or should_exclude_quant_module(kv_a_prefix, quant_config.exclude_modules):
                quant_config = None
        super().__init__(
            input_size,
            output_size,
            bias=bias,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.use_min_latency = (
            self.bias is None
            and self.weight.dtype == torch.bfloat16
            and self.weight.size() == (2112, 7168)
            and current_platform().is_nvidia
            and _device_sm >= 90
            and _device_sm not in (120, 121)
        )

    def forward(
        self, x: torch.Tensor, block_scale=None, output_dtype=None
    ) -> torch.Tensor:
        if (
            self.use_min_latency
            and x.size(0) > 0
            and x.size(0) <= _FUSED_A_MAX_M
            and block_scale is None
            and (output_dtype is None or output_dtype == torch.bfloat16)
        ):
            return dsv3_fused_a_gemm(x, self.weight.T)

        return super().forward(x, block_scale=block_scale, output_dtype=output_dtype)[0]


class DeepseekV3AttentionMLA(nn.Module):
    # Backends that support chunked ragged prefill with prefix replay.
    _RAGGED_PREFILL_BACKENDS = ("mla", "trtllm_mla", "tokenspeed_mla")

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        layer_id=None,
        prefix: str = "",
        reduce_attn_results=True,
        alt_stream: torch.cuda.Stream | None = None,
        skip_rope: bool = False,
        q_lora_scale: float | None = None,
        kv_lora_scale: float | None = None,
    ) -> None:
        """
        Args:
            q_lora_scale: Runtime multiplier applied to ``q`` after
                ``q_b_proj`` (LongCat's ``sqrt(hidden / q_lora_rank)`` under
                ``--mla-lora-scale runtime``). None when the checkpoint has no
                such scale or it is folded into ``q_a_layernorm``'s weight.
            kv_lora_scale: Runtime multiplier applied to the latent after
                ``kv_a_layernorm``; None as above.
        """
        super().__init__()
        self.mapping = mapping
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        if q_lora_scale is not None and q_lora_rank is None:
            raise ValueError("q_lora_scale needs a q_lora_rank to scale")
        self.q_lora_scale = q_lora_scale
        self.kv_lora_scale = kv_lora_scale
        self.num_heads = num_heads
        # The head projections (q_b/kv_b/o_proj) shard over the head-TP
        # group: the attention TP group today (replicated under query context
        # parallelism, whose shards hold different rows), or, under head TP,
        # a group of ranks holding different rows -- attention-DP ranks, or
        # the query shards -- where the forward exchanges heads for tokens
        # around core attention.
        self.has_head_tp = self.mapping.attn.has_head_tp
        self.head_tp_size = self.mapping.attn.head_tp_size
        self.head_tp_rank = self.mapping.attn.head_tp_rank
        self.head_tp_group = self.mapping.attn.head_tp_group
        if num_heads % self.head_tp_size != 0:
            raise ValueError(
                f"num_heads={num_heads} must be divisible by the head TP size "
                f"{self.head_tp_size} (attn_tp_size={self.mapping.attn.tp_size})."
            )
        self.num_local_heads = num_heads // self.head_tp_size
        # TP batch invariance: o_proj is column-parallel on hidden over the
        # head group (full K on every rank), so the attention tail moves rows
        # instead of reduce-scattering head partials.
        self.o_proj_batch_invariant = (
            global_server_args_dict["tp_batch_invariant"] != "none"
        )
        if self.o_proj_batch_invariant and not self.has_head_tp:
            raise ValueError(
                "--tp-batch-invariant needs the head-sharded o_proj of "
                "--attn-head-tp-size > 1"
            )
        if self.o_proj_batch_invariant and quant_config is not None:
            raise ValueError(
                "--tp-batch-invariant needs an unquantized o_proj; this layer's "
                "o_proj is quantized"
            )
        if self.has_head_tp and not self.supports_head_tp:
            raise NotImplementedError(
                f"{type(self).__name__} overrides the attention forward without "
                "the head-TP exchange; --attn-head-tp-size > 1 is not supported "
                "for this model"
            )
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.config = config
        self.alt_stream = alt_stream
        self.cli_factor = getattr(config, "cli_factor", 1)
        self.prefix = prefix

        # modification to rope_scaling must be done early enough, b/c e.g. Indexer needs it
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj_with_mqa = DeepseekV3FusedQkvAProjWithMqa(
                self.hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("fused_qkv_a_proj_with_mqa", prefix),
            )

            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_b_proj", prefix),
                tp_rank=self.head_tp_rank,
                tp_size=self.head_tp_size,
                tp_group=self.head_tp_group,
            )
        else:
            if self.has_head_tp:
                raise ValueError(
                    "attention head TP gathers the normalized q latent over the "
                    "head group and needs q_lora_rank"
                )
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
                tp_rank=self.head_tp_rank,
                tp_size=self.head_tp_size,
                tp_group=self.head_tp_group,
            )

            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            )

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=self.head_tp_rank,
            tp_size=self.head_tp_size,
            tp_group=self.head_tp_group,
        )
        # O projection.
        if self.o_proj_batch_invariant:
            # Full K (every head's values, all-gathered) on every rank; the
            # output is this rank's hidden shard of every gathered token.
            self.o_proj = ColumnParallelLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                quant_config=None,
                prefix=add_prefix("o_proj", prefix),
                tp_rank=self.head_tp_rank,
                tp_size=self.head_tp_size,
                tp_group=self.head_tp_group,
            )
        else:
            self.o_proj = RowParallelLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=False,
                # Under head TP the attention's own tail (project_output)
                # reduces the partials: a reduce-scatter to each rank's rows,
                # or an all-reduce on a replicated-row forward.
                reduce_results=reduce_attn_results and not self.has_head_tp,
                quant_config=quant_config,
                prefix=add_prefix("o_proj", prefix),
                tp_rank=self.head_tp_rank,
                tp_size=self.head_tp_size,
                tp_group=self.head_tp_group,
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        # Fusion layer
        if self.q_lora_rank is not None:
            self.fused_qk_layernorm = FusedRMSNorm(
                self.q_a_layernorm,
                self.kv_a_layernorm,
            )

        if not skip_rope:
            self.rotary_emb = get_rope(
                qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
                is_neox_style=False,
            )

            if rope_scaling:
                mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
                scaling_factor = rope_scaling["factor"]
                mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
                self.scaling = self.scaling * mscale * mscale
        else:
            self.rotary_emb = None

        # The one absorbed core layer. Under head TP it declares every head:
        # the exchange delivers every head of this rank's own rows to the
        # core. A replicated-row forward on a query-sharding engine (the
        # drafter's decode steps) exchanges nothing and hands the same layer
        # the attention-TP head slice; the DSA core -- the only backend a
        # query-sharding engine runs -- takes its head count from the query,
        # so one layer serves both forms (``docs/design/unified_path.md``,
        # "Head TP over the query shards").
        self.attn_mqa = PagedAttention(
            self.num_heads if self.has_head_tp else self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            rotary_emb=self.rotary_emb,
            qk_norm=None,
        )

        self.attn_mha = PagedAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            rotary_emb=self.rotary_emb,
            qk_norm=None,
        )

        self.w_kc = None
        self.w_vc = None

    @property
    def supports_head_tp(self) -> bool:
        """Whether this class's ``forward`` runs the head-TP exchange.

        The base forward and break do, and so does the draft's break; a
        subclass that overrides ``forward`` or ``_attn`` and threads the
        head-TP hooks (``head_tp_exchanges``, ``head_tp_gather_tokens``,
        ``forward_absorb_qkv_proj``, ``forward_absorb_attn_v_proj`` or
        ``sparse_prefill_attn_v_proj``, ``project_output``) declares it with
        ``supports_head_tp = True``.
        """
        cls = type(self)
        return cls.forward is DeepseekV3AttentionMLA.forward and cls._attn in (
            DeepseekV3AttentionMLA._attn,
            DeepseekV3DraftAttentionMLA._attn,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MLA attention with a NARROW prefill-graph break.

        The token-shaped input/output projections (q/kv-down, layernorm,
        q_b_proj, o_proj) and, outside a decode round, the expanded prefill
        prologue with its KV write stay in the captured prefill graph; only
        the data-dependent attention -- varlen prefill / absorb decode kernels
        + the live prefill/decode split -- runs as the eager break
        (``_attn``). This keeps the big projection GEMMs graphed instead of
        dispatch-bound eager, collapsing the inter-segment bubbles a coarse
        whole-attention break leaves. Outside capture the ``@break_point`` is
        a direct call, so the eager path is unchanged.

        This dense MLA path (expanded prefill prologue, head-sharded weights)
        attends every row of the span, so it refuses a query shard outright
        -- before the empty-row return, since a rank whose shard is empty
        would otherwise skip collectives the other ranks join.

        Every decoder layer calls this on an idle forward too, with its empty
        input rows. Without head TP that is a no-op; under head TP the rank
        still owns a head shard of its group's tokens, so it computes that
        shard here and joins every exchange (they are collectives) -- the one
        place idle participation lives, so no layer branches on the layout.
        """
        _reject_query_shard(ctx, "DeepseekV3AttentionMLA.forward")
        if hidden_states.shape[0] == 0 and not self.has_head_tp:
            # The o_proj output shape (the Eagle3 input is twice as wide).
            return hidden_states.new_empty(0, self.hidden_size)
        if self.has_head_tp and ctx.num_extends > 0:
            # Unreachable by configuration (see _validate_decode_tp_layouts
            # and validate_qcp): a guard against a prefill row reaching the
            # expanded prologue on a head-sharded layout. Head TP serves
            # decode rows, or extend rows under query context parallelism
            # through the absorbed sparse prefill (the shard was refused
            # above); the expanded path never.
            raise RuntimeError(
                "attention head TP serves decode rows, or extend rows under query "
                "context parallelism with an absorbed sparse prefill: the "
                "head-sharded kv_b_proj cannot expand every head's K/V for the "
                f"expanded prefill; this forward carries {ctx.num_extends} "
                "extending requests"
            )
        q, latent_cache = self._project_q_latent(
            hidden_states, ctx, comm_manager, block_scale
        )
        expanded = self._prefill_prologue_before_break(positions, q, latent_cache, ctx)
        attn_output = self._attn(positions, q, latent_cache, ctx, expanded=expanded)
        return self.project_output(
            attn_output, ctx, self.attention_output_rows(hidden_states, ctx)
        )

    # ---- Head-TP hooks -------------------------------------------------
    #
    # Under head TP the per-layer data flow is, per rank with T_own rows of
    # a head group of W ranks holding T_full rows together:
    #   q latent [T_own, q_lora]  --token all-gather-->  [T_full, q_lora]
    #   q_b_proj + absorb         -->  Q [T_full, H/W, kv_lora + rope]
    #   all_to_all_transpose      -->  Q [T_own, H, kv_lora + rope]
    #   latent prologue (RoPE, KV write) and core attention on own KV
    #   all_to_all_head_scatter   -->  [T_full, H/W, kv_lora]
    #   value projection (w_vc)   -->  [T_full, H/W * v]
    #   o_proj tail               -->  [T_own, hidden]
    # The exchange precedes the prologue so the prologue sees one row count
    # for the query, the latent and the write slots, and RoPE (a per-row
    # rotation of each head) commutes with the head permutation; the local
    # rows' positions are the rows' own, so no positions collective exists.
    #
    # The head group's ranks hold different rows: attention-DP ranks (the
    # decode role), or the query shards of a query-context-parallel prefill
    # engine, where the exchange serves the sharded extend forwards and the
    # KV write inside the prologue gathers the rotated latent to the span
    # as every QCP forward does. Row counts come from one resolver
    # (``comm_manager.head_tp_row_counts``: the forward's DP tables, or the
    # shard plan), by leg: the legs up to core attention move the forward's
    # input rows; the legs after it move its collective rows, which a
    # narrowing drafter has reduced to the live rows. Every rank of the
    # group, including one with no rows of its own, picks the table the same
    # way.
    #
    # The drafter's decode steps on a query-sharding engine hold every row
    # on every rank: there is nothing to exchange (``head_tp_exchanges`` is
    # False), so they attend this rank's head slice of every row as
    # attention TP does -- the same core layer, whose DSA backend takes the
    # head count from the query -- and the o_proj tail reduces over the head
    # group instead of returning rows. That predicate is the one fork of the
    # head-TP path: every leg below asks it, and the legs it switches off
    # are the exchanges and nothing else.

    def head_tp_exchanges(self, ctx: ForwardContext) -> bool:
        """Whether this forward exchanges heads for tokens over the head group.

        Every forward on the attention-DP layout does; on the query-sharding
        layout the sharded extend forwards do, while a forward without a
        shard (the drafter's decode steps) holds every row on every rank and
        runs the attention-TP form on this rank's head slice.
        """
        if not self.has_head_tp:
            return False
        if not self.mapping.attn.has_qcp:
            return True
        return ctx.query_shard is not None and ctx.query_shard.size > 1

    def head_tp_leg_row_counts(
        self, ctx: ForwardContext, num_rows: int, *, collective: bool
    ) -> list[int]:
        """Rows every head-group rank holds in one leg of the exchange;
        ``num_rows`` is this rank's. ``collective=False`` is the forward's
        input rows (the legs up to core attention), ``True`` its collective
        rows (the legs after it; see ``comm_manager.head_tp_row_counts``)."""
        return head_tp_row_counts(ctx, self.mapping, num_rows, collective=collective)

    def head_tp_gather_tokens(
        self, x: torch.Tensor, ctx: ForwardContext
    ) -> torch.Tensor:
        """Token all-gather of this rank's input rows ``[T_own, F]`` over the
        head group."""
        counts = self.head_tp_leg_row_counts(ctx, x.shape[0], collective=False)
        if sum(counts) == 0:
            # The whole head group is idle this forward: nothing to gather,
            # and every rank reads the same table so every rank skips.
            return x
        return token_all_gather(
            x, group=self.head_tp_group, scattered_num_tokens=counts
        )

    def attention_output_rows(
        self, hidden_states: torch.Tensor, ctx: ForwardContext
    ) -> int:
        """Rows this rank's attention output has; a narrowing draft overrides."""
        return hidden_states.shape[0]

    def project_output(
        self, attn_output: torch.Tensor, ctx: ForwardContext, num_rows: int
    ) -> torch.Tensor:
        """``o_proj`` and, under head TP, the return to this rank's own rows.

        ``attn_output`` is ``[T, H_local * v]``; ``num_rows`` is the rows this
        rank owns (``T`` itself without head TP, where the output is the
        plain ``o_proj`` result). Under head TP the batch-invariant tail
        all-gathers the heads and runs the column-parallel ``o_proj``; the
        plain tail runs the row-parallel ``o_proj`` on the head partials.
        What follows is the one fork, on whether the forward exchanged: an
        exchanging forward returns to its own rows (the hidden shards
        transposed back, or the partials reduce-scattered); a replicated-row
        forward (``T == num_rows`` on every rank, nothing exchanged) keeps
        every row on every rank (the hidden shards all-gathered, or the
        partials all-reduced -- the attention-TP form).
        """
        if not self.has_head_tp:
            return self.o_proj(attn_output)[0]
        exchanges = self.head_tp_exchanges(ctx)
        if exchanges:
            counts = self.head_tp_leg_row_counts(ctx, num_rows, collective=True)
            if sum(counts) == 0:
                # The whole head group is idle: no rows anywhere, no collective.
                return attn_output.new_empty(0, self.hidden_size)
        else:
            if attn_output.shape[0] != num_rows:
                raise ValueError(
                    f"a replicated-row forward projects its own {num_rows} rows, "
                    f"got {attn_output.shape[0]}"
                )
            if num_rows == 0:
                # Every rank holds the same (no) rows: no collective.
                return attn_output.new_empty(0, self.hidden_size)
        if self.o_proj_batch_invariant:
            attn_output = all_gather(attn_output, self.head_tp_group, dim=-1)
            partial_hidden, _ = self.o_proj(attn_output)
            if exchanges:
                return all_to_all_transpose(
                    partial_hidden, self.head_tp_group, input_split_sizes=counts
                )
            return all_gather(partial_hidden, self.head_tp_group, dim=-1)
        partial, _ = self.o_proj(attn_output)
        if exchanges:
            return token_reduce_scatter(
                partial, group=self.head_tp_group, scattered_num_tokens=counts
            )
        return all_reduce(partial, self.head_tp_group)

    def _prefill_prologue_before_break(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        ctx: ForwardContext,
    ) -> MLAPrologueOutput | None:
        """Assemble the expanded prefill inputs and write every row's latent in
        the captured segment. A decode round, or a narrowed draft step, keeps
        its one-launch prologue inside the break instead (see :meth:`_attn`).
        An idle rank only reaches here under head TP, with no rows of its own."""
        if ctx.forward_mode.is_decode_or_idle() or ctx.draft_narrowing is not None:
            return None
        slots = ctx.attn_backend.padded_write_locations(
            self.attn_mha, ctx.forward_mode, q.shape[0]
        )
        return self.forward_normal_chunked_kv_prepare(
            positions, q, latent_cache, ctx, slots
        )

    def _project_q_latent(
        self,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """QKV projection producing the ``q_b_proj`` output ``q`` and this
        rank's raw ``latent_cache``. Under head TP ``q`` covers the head
        group's gathered tokens (this rank's head shard of each).

        The LoRA norm scales, when applied at runtime (``--mla-lora-scale
        runtime``), multiply exactly where the trainer does: the latent right
        after ``kv_a_layernorm`` (in place, so the cache sees the scaled
        latent) and ``q`` right after ``q_b_proj``. ``q_norm`` itself stays
        unscaled, which is the ``q_lora`` a DSA indexer must read.
        """
        if self.q_lora_rank is not None:
            if hidden_states.shape[0] == 0:
                # An idle rank under head TP: no rows to project (the
                # quantized GEMMs do not take empty inputs), but the head
                # group's gathered rows below still need this rank's shard.
                qkv = hidden_states.new_empty(
                    0, self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
                )
            else:
                qkv = self.fused_qkv_a_proj_with_mqa(
                    hidden_states, block_scale, torch.bfloat16
                )
            qkv = comm_manager.pre_attn_comm(qkv, ctx)
            q_a, latent_cache = qkv.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
            kv_a = latent_cache[..., : self.kv_lora_rank]
            q_norm = torch.empty_like(q_a)
            if q_a.size(0) > 0:
                self.fused_qk_layernorm(
                    input_q_a=q_a, input_kv_a=kv_a, output_q_a=q_norm
                )
                if self.kv_lora_scale is not None:
                    kv_a.mul_(self.kv_lora_scale)
            if self.head_tp_exchanges(ctx):
                q_norm = self.head_tp_gather_tokens(q_norm, ctx)
            q = self.q_b_proj(q_norm)[0]
            if self.q_lora_scale is not None:
                q = q * self.q_lora_scale
        else:
            hidden_states = comm_manager.pre_attn_comm(hidden_states, ctx)
            q = self.q_proj(hidden_states)[0]
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            kv_a = latent_cache[..., : self.kv_lora_rank]
            self.kv_a_layernorm(kv_a, out=kv_a)
            if self.kv_lora_scale is not None:
                kv_a.mul_(self.kv_lora_scale)
        return q, latent_cache

    @break_point
    def _attn(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        ctx: ForwardContext,
        *,
        expanded: MLAPrologueOutput | None,
        output_gate: torch.Tensor | None = None,
        absorbed_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The eager break: varlen prefill / absorb decode attention.

        Prefill/decode dispatch over the full rows; subclasses override (the
        draft variant narrows to live rows, see ``DeepseekV3DraftAttentionMLA``).
        The split is recovered from LIVE state -- correct both in eager and
        under a prefill-graph replay, where ``ctx`` is the live ambient context
        but ``q`` is padded to the graph bucket (``q.size(0)`` is NOT the real
        token count). The decode token count comes from the live ctx; the real
        prefill token count from the live attention metadata (the same source
        the padding scrub uses). Padded tail rows produce discarded garbage.
        Outside a decode round the captured segment already ran the expanded
        prologue over every row and stored the latent
        (:meth:`_prefill_prologue_before_break`); the prefill half attends the
        leading rows of that output, and the decode half of a MIXED round
        assembles its absorbed query here and rewrites its own rows through
        the backend's DECODE window.
        """
        spec = ctx.attn_backend.spec_num_tokens or 1
        num_decodes = max(ctx.bs - ctx.num_extends, 0)
        num_decode_tokens = num_decodes * spec
        if ctx.num_extends > 0:
            cmeta = ctx.attn_backend.chunked_prefill_metadata
            num_prefill_tokens = int(sum(cmeta.extend_seq_lens_cpu))
        else:
            num_prefill_tokens = 0
        real_total = num_prefill_tokens + num_decode_tokens

        if self.head_tp_exchanges(ctx):
            # ``forward`` refused extending rows before the projections. A
            # replicated-row forward under head TP (the drafter's decode
            # steps on a query-sharding engine) takes the plain path below:
            # this rank's head slice over every row, the all-reduce tail.
            if output_gate is not None or absorbed_query is not None:
                raise NotImplementedError(
                    "attention head TP does not support an output gate or a "
                    "pre-absorbed query"
                )
            # ``q`` holds the head group's gathered input rows (this rank's
            # head shard); the output holds the group's collective rows,
            # which a narrowing drafter on a peer rank may have reduced to
            # its live rows (this rank, idle in that step, has none either
            # way). The latent, positions and slots are this rank's own rows.
            attn_output = q.new_empty(
                sum(self.head_tp_leg_row_counts(ctx, real_total, collective=True)),
                self.num_local_heads * self.v_head_dim,
            )
            # Every rank of the head group takes part, with or without rows
            # of its own, so an idle rank runs the exchanges too; it writes no
            # KV and asks the backend for no slots.
            decode_ctx = replace(
                ctx,
                bs=num_decodes,
                num_extends=0,
                input_num_tokens=num_decode_tokens,
                forward_mode=ForwardMode.DECODE,
            )
            slots = (
                ctx.attn_backend.write_locations(self.attn_mha, ForwardMode.DECODE)
                if num_decode_tokens > 0
                else positions.new_empty(0, dtype=torch.int64)
            )
            self.forward_absorb(
                positions[:real_total],
                q,
                latent_cache[:real_total],
                decode_ctx,
                slots,
                attn_output,
            )
            return attn_output

        attn_output = torch.empty(
            q.size(0),
            self.num_local_heads * self.v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )

        if num_prefill_tokens > 0:
            prefill_ctx = replace(
                ctx,
                bs=max(ctx.bs - num_decodes, 1),
                num_extends=max(ctx.bs - num_decodes, 1),
                input_num_tokens=num_prefill_tokens,
                forward_mode=ForwardMode.EXTEND,
            )
            if expanded is None:
                raise RuntimeError("prefill rows reached the break without a prologue")
            if getattr(cmeta, "use_absorbed_cached_extend", False):
                # Absorbed cached extend (gluon) rebuilds its query here; its rows are stored.
                self.forward_absorb(
                    positions[:num_prefill_tokens],
                    q[:num_prefill_tokens],
                    latent_cache[:num_prefill_tokens],
                    prefill_ctx,
                    ctx.attn_backend.write_locations(self.attn_mha, ForwardMode.EXTEND)[
                        :0
                    ],
                    attn_output[:num_prefill_tokens],
                )
            else:
                self.forward_normal_chunked_kv_core(
                    expanded.query[:num_prefill_tokens],
                    expanded.key[:num_prefill_tokens],
                    expanded.value[:num_prefill_tokens],
                    prefill_ctx,
                    attn_output[:num_prefill_tokens],
                )

        if num_decode_tokens > 0:
            decode_ctx = replace(
                ctx,
                bs=num_decodes,
                num_extends=0,
                input_num_tokens=num_decode_tokens,
                forward_mode=ForwardMode.DECODE,
            )
            self.forward_absorb(
                positions[num_prefill_tokens:real_total],
                q[num_prefill_tokens:real_total],
                latent_cache[num_prefill_tokens:real_total],
                decode_ctx,
                ctx.attn_backend.write_locations(self.attn_mha, ForwardMode.DECODE),
                attn_output[num_prefill_tokens:real_total],
                output_gate=(
                    None
                    if output_gate is None
                    else output_gate[num_prefill_tokens:real_total]
                ),
                absorbed_query=(
                    None
                    if absorbed_query is None
                    else absorbed_query[num_prefill_tokens:real_total]
                ),
            )

        return attn_output

    def forward_absorb(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        output: torch.Tensor,
        output_gate: torch.Tensor | None = None,
        absorbed_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        Q = self.forward_absorb_qkv_proj(
            q,
            latent_cache,
            positions,
            ctx,
            out_cache_loc,
            absorbed_query=absorbed_query,
        )
        return self.forward_absorb_attn_v_proj(
            Q,
            ctx,
            output,
            output_gate=output_gate,
        )

    def absorb_query(
        self,
        q: torch.Tensor,
        absorbed_query: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Absorb the ``q_b_proj`` output into latent space.

        ``q`` is ``[T, H_local * qk_head_dim]`` (or the non-RoPE part alone
        when ``absorbed_query`` supplies the pre-allocated
        ``[T, H_local, kv_lora_rank + rope]`` query whose RoPE channels are
        already filled). Returns the absorbed query, whose leading channels
        are ``q_nope @ w_kc``, and the unrotated RoPE part: a view of ``q``
        (the prologue rotates it into the query) or of ``absorbed_query``.
        """
        if absorbed_query is None:
            q = q.view(-1, self.num_local_heads, self.qk_head_dim)
            q_nope, q_pe = q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            Q = torch.empty(
                q_nope.size(0),
                self.num_local_heads,
                self.kv_lora_rank + self.qk_rope_head_dim,
                dtype=q_nope.dtype,
                device=q_nope.device,
            )
        else:
            q_nope = q
            Q = absorbed_query
            q_pe = Q[..., self.kv_lora_rank :]
        # The absorption projection must be per-row batch-invariant under
        # rl-bitwise: the cuBLAS batched GEMM retiles by the token count. A
        # rank whose query shard is empty has nothing to absorb but still
        # runs the prologue for its collectives.
        if q_nope.shape[0] > 0:
            bmm(
                q_nope.transpose(0, 1),
                self.w_kc.transpose(1, 2),
                out=Q[..., : self.kv_lora_rank].transpose(0, 1),
                override=(
                    "aok"
                    if global_server_args_dict["numerics"] in BITWISE_ENVELOPES
                    else None
                ),
            )
        return Q, q_pe

    def head_tp_scatter_query(
        self, Q: torch.Tensor, ctx: ForwardContext, num_rows: int
    ) -> torch.Tensor:
        """Heads-to-tokens leg: ``[T_full, H_local, D]`` of this rank's head
        shard becomes ``[num_rows, H, D]`` of its own input rows with every head."""
        rows_full, heads_local, dim = Q.shape
        # all_to_all_transpose checks the counts against the rows.
        return all_to_all_transpose(
            Q.reshape(rows_full, heads_local * dim),
            self.head_tp_group,
            input_split_sizes=self.head_tp_leg_row_counts(
                ctx, num_rows, collective=False
            ),
        ).view(-1, self.head_tp_size * heads_local, dim)

    def head_tp_gather_heads(
        self, attn_output: torch.Tensor, ctx: ForwardContext
    ) -> torch.Tensor:
        """Tokens-to-heads leg: ``[T_own, H, D]`` of this rank's tokens becomes
        ``[T_full, H_local, D]`` of its head shard over the group's tokens."""
        return all_to_all_head_scatter(
            attn_output,
            self.head_tp_group,
            output_split_sizes=self.head_tp_leg_row_counts(
                ctx, attn_output.shape[0], collective=True
            ),
        )

    def latent_prologue(
        self,
        Q: torch.Tensor,
        q_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        slots: torch.Tensor,
        *,
        key_rows: QueryShardGather | None,
    ) -> torch.Tensor:
        """The absorbed MLA prologue: rotate the query and the latent key
        part, write the latent rows to ``slots`` and return the attention
        query. One row count across the query, latent and positions; under a
        query shard ``key_rows`` gathers the rotated latent to the whole span
        before the owner-masked store. The head count is the query's (the
        prologue rotates whatever heads it is handed)."""
        return self.attn_mqa.latent_prologue(
            Q,
            q_pe,
            latent_cache,
            positions,
            ctx,
            slots=slots,
            expanded=None,
            key_rows=key_rows,
        ).query

    def forward_absorb_qkv_proj(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        absorbed_query: torch.Tensor | None = None,
        cache_num_tokens: int | None = None,
    ) -> torch.Tensor:
        """Absorb ``q`` and run the latent prologue over this rank's rows.

        Under head TP ``q`` carries the head group's gathered rows and the
        exchange to this rank's own rows happens between the absorption and
        the prologue, so the prologue sees one row count; a rank with no rows
        of its own skips the prologue and returns an empty query -- unless
        the forward is a query shard, whose prologue gathers the rotated
        latent to the whole span (``out_cache_loc``) and stores it
        owner-masked, so an empty shard still runs it. Without an exchange
        (no head TP, or the replicated decode rows of a query-sharding
        engine) the rows are this rank's own from the start.
        """
        Q, q_pe = self.absorb_query(q, absorbed_query)
        if self.head_tp_exchanges(ctx):
            # The RoPE part travels inside the query through the exchange;
            # the prologue then rotates the query's own RoPE channels.
            Q[..., self.kv_lora_rank :] = q_pe
            Q = self.head_tp_scatter_query(Q, ctx, latent_cache.shape[0])
            q_pe = Q[..., self.kv_lora_rank :]
            if Q.shape[0] == 0 and ctx.query_shard is None:
                # An idle attention-DP rank: nothing to rotate or write. An
                # empty query shard goes on: its prologue joins the gather.
                return Q
        # GLM's sparse prefill runs more rows than it commits: write the leading rows.
        query_tokens = Q.shape[0]
        key_rows = None
        if ctx.query_shard is not None:
            # A query shard rotates its own rows; the prologue gathers the
            # rotated latent to the whole span (out_cache_loc) before the
            # owner-masked store.
            if cache_num_tokens is not None:
                raise RuntimeError(
                    "a query shard writes every row of the span; a partial write "
                    "count cannot be combined with it"
                )
            key_rows = QueryShardGather(ctx.query_shard, self.mapping.attn.qcp_group)
            cache_num_tokens = ctx.query_shard.total_rows
        elif cache_num_tokens is None:
            cache_num_tokens = query_tokens
        if cache_num_tokens < 0 or (
            key_rows is None and cache_num_tokens > query_tokens
        ):
            raise RuntimeError(
                "MLA cache write count is outside the query capacity: "
                f"writes={cache_num_tokens}, queries={query_tokens}"
            )
        return self.latent_prologue(
            Q,
            q_pe,
            latent_cache,
            positions,
            ctx,
            slots=out_cache_loc[:cache_num_tokens],
            key_rows=key_rows,
        )

    def forward_absorb_attn_v_proj(
        self,
        Q,
        ctx: ForwardContext,
        output: torch.Tensor,
        record_kv_cache: bool | None = None,
        output_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Core absorbed attention over this rank's KV through the backend's
        dispatch (``attn_mqa``: the decode kernels, or a backend's own
        extend), then :meth:`project_attended_heads`. A sparse-attention
        model's extend rows take :meth:`sparse_prefill_attn_v_proj` instead,
        since the DSA backend has no ``forward_extend``.

        The core runs on every rank that holds rows or a query shard: an
        empty shard's sparse core still joins its group's history gathers.
        Only an idle attention-DP rank under head TP (no rows, no shard)
        skips it; its dense core has no collective to join, and it still
        takes the exchange legs around it.
        """
        exchanges = self.head_tp_exchanges(ctx)
        if exchanges and output_gate is not None:
            raise NotImplementedError(
                "attention head TP does not support an output gate"
            )
        use_projected_value_decode = (
            output_gate is not None
            and ctx.num_extends == 0
            and ctx.attn_backend.supports_mla_projected_value_decode
        )
        if Q.shape[0] == 0 and ctx.query_shard is None:
            attn_output = Q.new_empty(0, Q.shape[1] * self.kv_lora_rank)
        else:
            attn_output = self.attn_mqa(
                Q,
                k=None,
                v=None,
                positions=None,
                ctx=ctx,
                record_kv_cache=record_kv_cache,
                value_weight=self.w_vc if use_projected_value_decode else None,
                output_gate=output_gate if use_projected_value_decode else None,
                projected_output=output if use_projected_value_decode else None,
            )
        if use_projected_value_decode:
            return attn_output
        return self.project_attended_heads(attn_output, ctx, output, gate=output_gate)

    def sparse_prefill_attn_v_proj(
        self,
        Q: torch.Tensor,
        ctx: ForwardContext,
        output: torch.Tensor,
        *,
        kv_seq_lens: torch.Tensor | None,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        """The sparse prefill core over this rank's extend rows, then
        :meth:`project_attended_heads`: the sequence a DSA model's extend
        runs after :meth:`forward_absorb_qkv_proj` (the backend's
        ``forward_sparse_prefill`` with the model's selection, which under a
        query shard attends the gathered history and is joined by an empty
        shard too; under head TP the tokens-to-heads exchange; the local
        ``w_vc``). ``topk_slots`` / ``topk_lens`` / ``kv_seq_lens`` are the
        backend's sparse-prefill arguments for ``Q``'s rows."""
        attn_output = ctx.attn_backend.forward_sparse_prefill(
            q=Q,
            layer=self.attn_mqa,
            token_to_kv_pool=ctx.token_to_kv_pool,
            kv_seq_lens=kv_seq_lens,
            topk_slots=topk_slots,
            topk_lens=topk_lens,
            max_seq_len=max_seq_len,
        )
        return self.project_attended_heads(attn_output, ctx, output, gate=None)

    def project_attended_heads(
        self,
        attn_output: torch.Tensor,
        ctx: ForwardContext,
        output: torch.Tensor,
        *,
        gate: torch.Tensor | None,
    ) -> torch.Tensor:
        """The value projection of core attention's output into ``output``.

        ``attn_output`` is ``[T_own, heads * kv_lora_rank]`` with the heads
        the core attended: every head after an exchange, which the
        tokens-to-heads leg turns back into this rank's head shard of the
        group's collective rows before the local ``w_vc``; the attention-TP
        slice otherwise, projected in place. A forward with no rows to
        project (an empty query shard, a wholly idle head group) returns
        ``output`` untouched.
        """
        if self.head_tp_exchanges(ctx):
            attn_output = self.head_tp_gather_heads(
                attn_output.view(-1, self.num_heads, self.kv_lora_rank), ctx
            )
        else:
            attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
        if attn_output.shape[0] == 0:
            return output
        return mla_project_value(attn_output, self.w_vc, gate=gate, out=output)

    def forward_normal_chunked_kv_prepare(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        ctx: ForwardContext,
        slots: torch.Tensor,
    ) -> MLAPrologueOutput:
        """The expanded prefill prologue over every row of ``q``: per-head keys
        and values up-projected from the latent, the rotated query, and the
        latent rows stored at ``slots``; the inputs are left as given. The
        expanded form cannot gather per-head keys across a query shard and
        ``slots`` would be the shard's rows at the span's head, so a sharded
        forward is refused here rather than writing the wrong rows."""
        _reject_query_shard(ctx, "the expanded MLA prefill prologue")
        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        # kv_b_proj's fp8 online-quant GEMM needs a contiguous latent, not this strided slice.
        kv = self.kv_b_proj(latent_cache[..., : self.kv_lora_rank].contiguous())[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return self.attn_mha.latent_prologue(
            q,
            q[..., self.qk_nope_head_dim :],
            latent_cache,
            positions,
            ctx,
            slots=slots,
            expanded=MLAExpandedKV(k_nope=k_nope, value=v),
            key_rows=None,
        )

    def forward_normal_chunked_kv_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: ForwardContext,
        output: torch.Tensor,
    ) -> torch.Tensor:
        attn_backend = ctx.attn_backend
        chunk_meta = attn_backend.chunked_prefill_metadata

        scaling = self.attn_mha.scaling

        # Causal self-attention over the new chunk tokens. q_lens == kv_lens ==
        # extend_seq_lens, so cum_seq_lens_q and cum_seq_lens_kv alias the same
        # cum_extend_seq_lens. Causal pass writes directly into output; each
        # chunk's merge accumulates in place via attn_merge_state(inplace=True).
        num_extends = chunk_meta.extend_seq_lens.size(0)
        output_view = output.view(-1, self.num_local_heads, self.v_head_dim)
        _, accum_lse = attn_backend.forward_extend_chunked(
            q,
            k,
            v,
            scaling,
            self.attn_mha.logit_cap,
            cum_seq_lens_q=chunk_meta.cum_extend_seq_lens,
            cum_seq_lens_kv=chunk_meta.cum_extend_seq_lens,
            max_q_len=chunk_meta.max_extend_seq_len,
            max_kv_len=chunk_meta.max_extend_seq_len,
            seq_lens=chunk_meta.extend_seq_lens,
            batch_size=num_extends,
            causal=True,
            out=output_view,
        )

        # Always read KV cache as BF16 for kv_b_proj (weight is BF16), even if Q is FP8.
        read_dtype = (
            q.dtype
            if q.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
            else torch.bfloat16
        )

        placement = attn_backend.cache_placement(self.attn_mha)
        for loop_idx in range(chunk_meta.chunked_loop_num):
            chunk_kv_indices = chunk_meta.chunk_kv_indices_list[loop_idx]

            if placement is None:
                kv_a_normed, k_pe = ctx.token_to_kv_pool.get_mla_kv_buffer(
                    self.attn_mha, chunk_kv_indices, read_dtype
                )
            else:
                kv_a_normed, k_pe = gather_mla_history(
                    ctx.token_to_kv_pool,
                    self.attn_mha,
                    chunk_kv_indices,
                    dst_dtype=read_dtype,
                    placement=placement,
                )

            kv_a_normed = kv_a_normed.squeeze(1)
            kv = self.kv_b_proj(kv_a_normed)[0]
            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            v = kv[..., self.qk_nope_head_dim :]
            k_nope = kv[..., : self.qk_nope_head_dim]

            if q.dtype == torch.float8_e4m3fn:
                # FP8 Attention
                k, v = mla_kv_pack_quantize_fp8(k_nope, k_pe, v)
            else:
                # BF16 Attention
                k = torch.cat(
                    [k_nope, k_pe.expand(-1, self.num_local_heads, -1)], dim=-1
                )

            chunk_output, lse = attn_backend.forward_extend_chunked(
                q,
                k,
                v,
                scaling,
                self.attn_mha.logit_cap,
                cum_seq_lens_q=chunk_meta.cum_extend_seq_lens,
                cum_seq_lens_kv=chunk_meta.cu_chunked_seq_len[loop_idx],
                max_q_len=chunk_meta.max_extend_seq_len,
                max_kv_len=chunk_meta.max_chunk_len_per_loop[loop_idx],
                seq_lens=chunk_meta.chunked_seq_len[loop_idx],
                batch_size=num_extends,
                causal=False,
            )

            attn_merge_state(
                output_view,
                accum_lse,
                chunk_output,
                lse,
                inplace=True,
            )

        return output


class DeepseekV3DraftAttentionMLA(DeepseekV3AttentionMLA):
    """Draft variant of MLA shared by the NextN and Eagle3 MLA drafters.

    On the active first draft step the full ``latent_cache`` (N rows) is
    projected so every KV cache entry is written, but only the live query rows
    (``ctx.gather_ids``) run the absorbed decode attention, narrowing the output
    to ``[bs, H]``.  Multi-step decode and target paths delegate to the base.
    Single-layer only, so dropping the dead rows has no downstream consumer.
    """

    def _attn(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        ctx: ForwardContext,
        *,
        expanded: MLAPrologueOutput | None,
        absorbed_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if ctx.draft_narrowing is None:
            return super()._attn(
                positions,
                q,
                latent_cache,
                ctx,
                expanded=expanded,
                absorbed_query=absorbed_query,
            )

        # The live rows attend over the accepted prefix, not the verify window.
        ctx.draft_narrowing.publish_accepted_prefix()

        # Every input row's KV is written: the extend rows, then the verify window.
        out_cache_loc = ctx.attn_backend.forward_write_locations(
            self.attn_mqa, ForwardMode.DECODE
        )

        # Every row's KV is written; only the live rows attend, so the output is [bs, H].
        decode_ctx = replace(ctx, forward_mode=ForwardMode.DECODE)
        Q = self.forward_absorb_qkv_proj(
            q,
            latent_cache,
            positions,
            decode_ctx,
            out_cache_loc,
            absorbed_query=absorbed_query,
        )
        Q = Q.index_select(0, ctx.gather_ids)
        if self.head_tp_exchanges(ctx):
            # The exchanged rows after narrowing are the group's live rows.
            output_rows = sum(self.head_tp_leg_row_counts(ctx, ctx.bs, collective=True))
        else:
            output_rows = ctx.bs
        attn_output = q.new_empty(output_rows, self.num_local_heads * self.v_head_dim)
        # One live row per request: decode spans every request, not the MIXED tail.
        with ctx.attn_backend.override_num_extends(0):
            self.forward_absorb_attn_v_proj(
                Q,
                decode_ctx,
                attn_output,
                # Real-mode record: decode_ctx would skip the PD cache-step here.
                record_kv_cache=not ctx.forward_mode.is_decode_or_idle(),
            )
        return attn_output

    def attention_output_rows(
        self, hidden_states: torch.Tensor, ctx: ForwardContext
    ) -> int:
        # A narrowing step keeps one live row per request (see _attn).
        if ctx.draft_narrowing is None:
            return hidden_states.shape[0]
        return ctx.bs


class DeepseekV3DecoderLayer(nn.Module):
    @property
    def attention_cls(self) -> type[nn.Module]:
        return DeepseekV3AttentionMLA

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.hidden_size = config.hidden_size
        rope_theta = get_rope_theta(config)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.self_attn = self.attention_cls(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=(
                config.q_lora_rank if hasattr(config, "q_lora_rank") else None
            ),
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=(
                None
                if "self_attn" in getattr(config, "disable_quant_module", [])
                else quant_config
            ),
            layer_id=layer_id,
            prefix=add_prefix("self_attn", prefix),
            reduce_attn_results=False,
            alt_stream=alt_stream,
            mapping=self.mapping,
        )

        self.layer_id = layer_id
        self.is_moe_layer = self._is_moe_layer(layer_id, is_nextn, config)
        # --tp-batch-invariant attn+dense: the dense tail transposes rows
        # instead of reduce-scattering head/channel partials.
        dense_batch_invariant = (
            global_server_args_dict["tp_batch_invariant"] == "attn+dense"
        )
        if self.is_moe_layer:
            self.mlp = DeepseekV3MoE(
                config=config,
                mapping=self.mapping,
                quant_config=quant_config,
                layer_index=layer_id,
                prefix=add_prefix("mlp", prefix),
                alt_stream=alt_stream,
            )
        else:
            self.mlp = DeepseekV3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=(
                    config.ffn_hidden_size
                    if hasattr(config, "ffn_hidden_size")
                    else config.intermediate_size
                ),
                hidden_act=config.hidden_act,
                mapping=self.mapping,
                quant_config=(
                    None
                    if "dense_mlp" in getattr(config, "disable_quant_module", [])
                    else quant_config
                ),
                prefix=add_prefix("mlp", prefix),
                is_shared_expert=False,
                batch_invariant=dense_batch_invariant,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.comm_manager = CommManager(
            mapping=self.mapping,
            layer_id=self.layer_id,
            is_moe=self.is_moe_layer,
            prev_is_moe=self._is_moe_layer(layer_id - 1, is_nextn, config),
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
            dense_batch_invariant=dense_batch_invariant and not self.is_moe_layer,
            query_sharded=False,
        )

    @staticmethod
    def _is_moe_layer(layer_id: int, is_nextn: bool, config):
        if is_nextn:
            return True
        if (
            config.n_routed_experts is not None
            and layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        ):
            return True
        return False

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:

        num_global_tokens, max_num_tokens_per_gpu = self.comm_manager.get_num_tokens(
            ctx
        )

        if ctx.forward_mode.is_idle():
            # No rows of its own: the attention joins its group's collectives
            # if the layout has any (the attention decides), then the MLP's.
            self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                ctx=ctx,
                comm_manager=self.comm_manager,
            )
            hidden_states = self.forward_mlp(
                hidden_states,
                residual,
                ctx,
                num_global_tokens,
                max_num_tokens_per_gpu,
            )
            return hidden_states, residual

        hidden_states, residual = self.comm_manager.input_reduce_norm(
            hidden_states, residual
        )
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            ctx=ctx,
            comm_manager=self.comm_manager,
        )
        residual = self.narrow_residual(residual, ctx)
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

    def narrow_residual(
        self, residual: torch.Tensor, ctx: ForwardContext
    ) -> torch.Tensor:
        """Align the residual with the attention output's rows; the draft
        layer narrows it to the live rows on a narrowing step."""
        return residual

    def input_layer_norm_fn(self, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        return hidden_states, residual

    def forward_mlp(
        self,
        hidden_states,
        residual,
        ctx: ForwardContext,
        num_global_tokens,
        max_num_tokens_per_gpu,
    ):
        hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
        if self.is_moe_layer:
            hidden_states = self.mlp(
                hidden_states, num_global_tokens, max_num_tokens_per_gpu
            )
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states, residual = self.comm_manager.post_mlp_fused(
            hidden_states, residual, ctx
        )
        return hidden_states


class DeepseekV3Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.alt_stream = torch.cuda.Stream()
        # config.num_hidden_layers = 5; self.start_layer,self.end_layer = 0, 5
        self.layers = nn.ModuleList(
            [
                DeepseekV3DecoderLayer(
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
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # For EAGLE3 support: set of layer indices whose *input* hidden states
        # are captured. Populated by set_eagle3_layers_to_capture().
        self.layers_to_capture: set = set()
        # DFLASH: each capture layer's positional tap index.
        self._dflash_capture_idx_map: dict[int, int] = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if input_embeds is not None:
            hidden_states = input_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        residual = None
        aux_hidden_states = [] if self.layers_to_capture else None
        for i in range(len(self.layers)):
            if aux_hidden_states is not None and i in self.layers_to_capture:
                # Under RSAG the inter-layer hidden/residual are reduce-
                # scattered across the attn TP group; aux consumers (e.g. the
                # EAGLE3 drafter) expect full rows, so gather before capturing.
                aux = (
                    hidden_states + residual if residual is not None else hidden_states
                )
                gathered = self.layers[i].comm_manager.gather_residual(aux, ctx)
                capture_idx = self._dflash_capture_idx_map.get(i)
                if ctx.target_capture_sink is not None and capture_idx is not None:
                    ctx.target_capture_sink.on_target_capture(capture_idx, gathered)
                aux_hidden_states.append(
                    gathered if gathered is aux else gathered.clone()
                )
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                ctx,
                residual,
            )
        if not ctx.forward_mode.is_idle():
            hidden_states, _ = layer.comm_manager.final_norm(
                hidden_states, residual, ctx, self.norm
            )
        return hidden_states, aux_hidden_states


class DeepseekV3ForCausalLM(BaseCausalLM):
    model_cls = DeepseekV3Model

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        model: DeepseekV3Model | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        self._model_override = model
        super().__init__(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
        )

    def resolve_model(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> DeepseekV3Model:
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
                if isinstance(layer.mlp, DeepseekV3MoE)
            }
        )

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None):
        # layer_ids are 0-indexed from the external API; +1 because the capture
        # check runs *before* the layer forward, so index i captures layer i-1's output.
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = {2, num_layers // 2, num_layers - 3}
        else:
            self.model.layers_to_capture = {val + 1 for val in layer_ids}

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        # DFlash checkpoints name 0-indexed target layer outputs. The capture
        # check runs before layer i, so capture at i + 1 for layer i's output.
        num_layers = len(self.model.layers)
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("DFLASH target_layer_ids must be unique.")

        invalid = [val for val in layer_ids if val < 0 or val + 1 >= num_layers]
        if invalid:
            raise ValueError(
                "DFLASH target_layer_ids must map to capturable target layer "
                f"outputs. Got invalid ids {invalid}; valid range is "
                f"[0, {num_layers - 2}] for {num_layers} target layers."
            )
        self.model.layers_to_capture = {val + 1 for val in layer_ids}
        self.model._dflash_capture_idx_map = {
            layer_idx: i
            for i, layer_idx in enumerate(sorted(self.model.layers_to_capture))
        }

    def get_param(self, params_dict, name):
        if name in params_dict:
            return params_dict[name]

        if "language_model." in name:
            name = name.replace("language_model.", "")
            if name in params_dict:
                return params_dict[name]

        if name.endswith(_OPTIONAL_MISSING_WEIGHT_SUFFIXES):
            return None

        logger.warning(f"The {name!s} is not in the model.")
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load a (possibly partial) checkpoint stream.

        Returns the ``named_parameters()`` names that received data (the
        ``BaseCausalLM`` weight-update contract).
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = getattr(self.config, "q_lora_rank", None) is not None

        params_dict = dict(self.named_parameters())
        # ``get_param`` remaps checkpoint names; report the parameter's own.
        param_names = {id(param): name for name, param in params_dict.items()}
        loaded: set[str] = set()
        moe_params_dict = dict(params_dict)
        for param_name, param in params_dict.items():
            if param_name.startswith("model."):
                moe_params_dict.setdefault(
                    param_name.replace("model.", "model.language_model.", 1),
                    param,
                )
                moe_params_dict.setdefault(
                    param_name.replace("model.", "language_model.model.", 1),
                    param,
                )
        # MoE expert weights, scales, and activation scales are handled
        # by the checkpoint loader.
        moe_loader = build_moe_checkpoint_loader(
            params_dict=moe_params_dict,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="gate_proj",
                down_proj_name="down_proj",
                up_proj_name="up_proj",
            ),
            num_experts=self.config.n_routed_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                if num_nextn_layers > 0 and name.startswith("model.layers"):
                    name_list = name.split(".")
                    if (
                        len(name_list) >= 3
                        and int(name_list[2]) >= self.config.num_hidden_layers
                    ):
                        continue
            if "rotary_emb.inv_freq" in name:
                continue
            if ".indexer." in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since moe_loader handles the experts below,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below by moe_loader
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = self.get_param(params_dict, name)
                if param is None:
                    continue
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded.add(param_names[id(param)])
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if moe_loader.matches(name):
                    loaded.add(moe_loader.load(name, loaded_weight))
                    continue

                if fuse_qkv_a_proj and (
                    "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                ):
                    quant_block_size = 1
                    # ``weight_block_size`` exists only on block-FP8 configs;
                    # elsewhere (e.g. compressed-tensors INT4) q/kv_a_proj is unquantized.
                    weight_block_size = getattr(
                        self.quant_config, "weight_block_size", None
                    )
                    if weight_block_size is not None:
                        quant_block_size = weight_block_size[0]
                    begin_size_mp = {
                        "q_a_proj": 0,
                        "kv_a_proj_with_mqa": self.config.q_lora_rank,
                    }
                    if "q_a_proj" in name:
                        param = self.get_param(
                            params_dict,
                            name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa"),
                        )
                        weight_loader = param.weight_loader
                        begin_size = begin_size_mp["q_a_proj"]
                    elif "kv_a_proj_with_mqa" in name:
                        param = self.get_param(
                            params_dict,
                            name.replace(
                                "kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa"
                            ),
                        )
                        weight_loader = param.weight_loader
                        begin_size = begin_size_mp["kv_a_proj_with_mqa"]
                    if "scale_inv" in name:
                        begin_size //= quant_block_size
                    weight_loader(param, loaded_weight, begin_size=begin_size)
                    loaded.add(param_names[id(param)])
                else:
                    # Owned-expert weights were already consumed by ``moe_loader.load(...)`` above (matches() == True branch).
                    # Anything reaching here that still looks like an expert weight is for an expert this rank does ot own under ep_size > 1.
                    if ".mlp.experts." in name:
                        continue
                    if "q_a_proj" in name and name not in params_dict:
                        name = name.replace("q_a_proj", "q_proj")
                    param = self.get_param(params_dict, name)
                    if param is None:
                        continue
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded.add(param_names[id(param)])

        self.post_load_weights()
        return loaded

    def post_load_weights(self):
        """Derive the absorbed MLA weights; re-runs write into the same storage."""
        for layer_id in range(self.config.num_hidden_layers):
            self_attn = self.model.layers[layer_id].self_attn
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
                    w = block_dequant(
                        self_attn.kv_b_proj.weight,
                        self_attn.kv_b_proj.weight_scale_inv,
                        weight_block_size,
                    ).to(dtype)
            else:
                w = self_attn.kv_b_proj.weight

            self_attn.w_kc, self_attn.w_vc = _prepare_mla_kv_b_proj_weights(
                w, self_attn
            )

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
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )


# ---------------------------------------------------------------------------
# Eagle3 MLA draft model
# ---------------------------------------------------------------------------


def _draft_rope_scaling(rope_scaling: dict | None) -> dict | None:
    """Keep only yarn-style rope_scaling for the EAGLE3 MLA draft layer.

    The layer implements plain rope and (deepseek-)yarn. transformers may
    normalize plain rope into a factor-less ``{"rope_type": "default"}``
    dict; anything else unsupported is dropped with a warning.
    """
    if not rope_scaling:
        return None
    if rope_scaling.get("rope_type", rope_scaling.get("type")) in (
        "yarn",
        "deepseek_yarn",
    ):
        return rope_scaling
    if "factor" in rope_scaling:
        logger.warning(
            f"EAGLE3 MLA draft ignores unsupported rope_scaling {rope_scaling!s}",
        )
    return None


class Eagle3MlaDecoderLayer(nn.Module):
    """Single decoder layer for Eagle3 MLA draft model.

    The fused_qkv_a_proj_with_mqa is overridden to accept 2x hidden_size
    input (concatenated [embeds, hidden_states]) while keeping o_proj at
    the standard hidden_size output.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        layer_id: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        rope_theta = get_rope_theta(config)
        rope_scaling = _draft_rope_scaling(getattr(config, "rope_scaling", None))
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.self_attn = DeepseekV3DraftAttentionMLA(
            config=config,
            mapping=self.mapping,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=getattr(config, "qk_nope_head_dim", 128),
            qk_rope_head_dim=getattr(config, "qk_rope_head_dim", 64),
            v_head_dim=getattr(config, "v_head_dim", 128),
            q_lora_rank=getattr(config, "q_lora_rank", None),
            kv_lora_rank=getattr(config, "kv_lora_rank", 512),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=add_prefix("self_attn", prefix),
            reduce_attn_results=False,
        )

        if hasattr(self.self_attn, "fused_qkv_a_proj_with_mqa"):
            q_lora_rank = getattr(config, "q_lora_rank", 0) or 0
            kv_lora_rank = getattr(config, "kv_lora_rank", 512)
            qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
            self.self_attn.fused_qkv_a_proj_with_mqa = DeepseekV3FusedQkvAProjWithMqa(
                2 * self.hidden_size,
                q_lora_rank + kv_lora_rank + qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix(
                    "fused_qkv_a_proj_with_mqa",
                    add_prefix("self_attn", prefix),
                ),
            )

        # --tp-batch-invariant attn+dense applies to this dense layer as well.
        dense_batch_invariant = (
            global_server_args_dict["tp_batch_invariant"] == "attn+dense"
        )
        self.mlp = DeepseekV3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=getattr(
                config, "intermediate_size", config.hidden_size * 4
            ),
            hidden_act=getattr(config, "hidden_act", "silu"),
            mapping=self.mapping,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            batch_invariant=dense_batch_invariant,
        )

        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fused_input_hidden_norm = FusedRMSNorm(
            self.input_layernorm,
            self.hidden_norm,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.comm_manager = CommManager(
            mapping=self.mapping,
            layer_id=self.layer_id,
            is_moe=False,
            prev_is_moe=False,
            dense_batch_invariant=dense_batch_invariant,
            post_attn_layernorm=self.post_attention_layernorm,
            query_sharded=False,
        )

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states

        # [embeds || hidden_states] after the two norms; empty on an idle
        # forward, which skips the norm kernels but still runs the attention
        # (its head group's collectives, if the layout has any).
        fused_norm_out = torch.empty(
            embeds.size(0),
            self.hidden_size * 2,
            dtype=embeds.dtype,
            device=embeds.device,
        )
        if not ctx.forward_mode.is_idle():
            # FusedRMSNorm's q_a/kv_a kwargs are MLA-specific names.
            # Here embeds and hidden_states corresponds to q_a and kv_a, separately.
            self.fused_input_hidden_norm(
                input_q_a=embeds,
                input_kv_a=hidden_states,
                output_q_a=fused_norm_out[..., : self.hidden_size],
                output_kv_a=fused_norm_out[..., self.hidden_size :],
            )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=fused_norm_out,
            ctx=ctx,
            comm_manager=self.comm_manager,
        )

        if not ctx.forward_mode.is_idle():
            # Active first draft step narrows attn output to [bs, H]; align the
            # residual to the same live rows before the post-attn reduce-norm.
            if ctx.draft_narrowing is not None:
                residual = residual.index_select(0, ctx.gather_ids)
            hidden_states, residual = self.comm_manager.post_attn_reduce_norm(
                hidden_states, residual, ctx
            )

        hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
        hidden_states = self.mlp(hidden_states)
        hidden_states, residual = self.comm_manager.post_mlp_fused(
            hidden_states, residual, ctx
        )

        return hidden_states, residual


class Eagle3MlaModel(nn.Module):
    @staticmethod
    def _get_eagle_layer_ids(config: PretrainedConfig):
        """Extract eagle aux hidden state layer IDs from config, or None if absent."""
        eagle_config = getattr(config, "eagle_config", None)
        if eagle_config is None:
            return getattr(config, "eagle_aux_hidden_state_layer_ids", None)
        if isinstance(eagle_config, dict):
            return eagle_config.get("eagle_aux_hidden_state_layer_ids", None)
        return getattr(eagle_config, "eagle_aux_hidden_state_layer_ids", None)

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        layer_ids = self._get_eagle_layer_ids(config)
        self.num_fc_input_dim = len(layer_ids) if layer_ids is not None else 3

        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        fc_input_size = target_hidden_size * self.num_fc_input_dim

        self.fc = ColumnParallelLinear(
            fc_input_size,
            config.hidden_size,
            bias=False,
            gather_output=True,
            quant_config=quant_config,
            prefix=add_prefix("fc", prefix),
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )

        self.midlayer = Eagle3MlaDecoderLayer(
            config,
            mapping=self.mapping,
            layer_id=0,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc_norm = (
            nn.ModuleList(
                [
                    RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                    for _ in range(self.num_fc_input_dim)
                ]
            )
            if getattr(config, "fc_norm", False)
            else None
        )
        self.fused_fc_norms = (
            nn.ModuleList(
                [
                    FusedRMSNorm(self.fc_norm[i], self.fc_norm[i + 1])
                    for i in range(0, self.num_fc_input_dim - 1, 2)
                ]
            )
            if self.fc_norm is not None
            else None
        )
        self.norm_output = getattr(config, "norm_output", False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        input_embeds: torch.Tensor | None = None,
        captured_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if captured_hidden_states is None:
            raise ValueError("Eagle3 MLA forward requires captured_hidden_states.")
        if input_embeds is None:
            embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        hidden_states = captured_hidden_states
        if hidden_states.size(-1) != embeds.size(-1):
            if self.fc_norm is not None and hidden_states.shape[0] > 0:
                chunks = hidden_states.chunk(self.num_fc_input_dim, dim=-1)
                normed = torch.empty_like(hidden_states)
                out_chunks = normed.chunk(self.num_fc_input_dim, dim=-1)
                i = 0
                for fused in self.fused_fc_norms:
                    fused(
                        input_q_a=chunks[i],
                        input_kv_a=chunks[i + 1],
                        output_q_a=out_chunks[i],
                        output_kv_a=out_chunks[i + 1],
                    )
                    i += 2
                if i < self.num_fc_input_dim:
                    # Odd count: single norm into the last slice.
                    self.fc_norm[i](chunks[i], out=out_chunks[i])
                hidden_states = normed
            hidden_states, _ = self.fc(hidden_states)

        residual = None
        hidden_states, residual = self.midlayer(
            positions,
            embeds,
            hidden_states,
            ctx,
            residual,
        )

        comm_manager = self.midlayer.comm_manager
        if comm_manager.should_fuse(hidden_states.size(0)):
            hidden_states_to_logits, hidden_states_to_aux, *_ = (
                self.norm.forward_with_allreduce_fusion(
                    self.mapping.dense.tp_rank,
                    self.mapping.dense.tp_group,
                    hidden_states,
                    residual,
                )
            )
        else:
            hidden_states_to_logits, hidden_states_to_aux = self.norm(
                hidden_states, residual
            )
            hidden_states_to_logits, _ = comm_manager.post_final_norm_comm(
                hidden_states_to_logits, None, ctx
            )
            hidden_states_to_aux, _ = comm_manager.post_final_norm_comm(
                hidden_states_to_aux, None, ctx
            )

        if self.norm_output:
            hidden_states_to_aux = hidden_states_to_logits

        return hidden_states_to_logits, [hidden_states_to_aux]


class Eagle3DeepseekV2ForCausalLM(DeepseekV3ForCausalLM):
    """Eagle3 MLA draft model for DeepSeek-V2/V3 / Kimi-K2 style architectures.

    Inherits weight-loading fusion logic from DeepseekV3ForCausalLM but uses
    Eagle3MlaModel internally with a single MLA decoder layer that accepts
    concatenated [embeds || hidden_states] as input.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config

        if self.config.num_hidden_layers != 1:
            raise ValueError("Eagle3 MLA drafter currently only supports 1 layer")

        self.model = Eagle3MlaModel(
            config,
            mapping=self.mapping,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        self.load_lm_head_from_target = False
        self._embed_loaded_from_checkpoint = False
        if self.config.tie_word_embeddings:
            if self.mapping.attn.has_dp and self.mapping.lm_head.has_tp:
                raise ValueError(
                    "--lm-head-tp-size > 1 vocab-shards the LM head, but this "
                    "draft ties it to its replicated embedding (tie_word_embeddings)"
                )
            self.lm_head = self.model.embed_tokens
        else:
            draft_vocab_size = (
                getattr(config, "draft_vocab_size", None) or config.vocab_size
            )
            if not hasattr(config, "draft_vocab_size"):
                self.load_lm_head_from_target = True
            # The draft's head follows the target's LM-head layout
            # (mapping.lm_head): the shared target head arrives in that
            # layout, and a draft-vocab head shards the same way.
            self.lm_head = ParallelLMHead(
                draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                tp_rank=self.mapping.lm_head.tp_rank,
                tp_size=self.mapping.lm_head.tp_size,
                tp_group=self.mapping.lm_head.tp_group,
                prefix=add_prefix("lm_head", prefix),
            )

        self.logits_processor = LogitsProcessor(
            config,
            skip_all_gather=self.mapping.attn.has_dp,
            do_argmax=True,
            tp_rank=self.mapping.lm_head.tp_rank,
            tp_size=self.mapping.lm_head.tp_size,
            tp_group=self.mapping.lm_head.tp_group,
            dp_lm_head_tp=self.mapping.attn.has_dp and self.mapping.lm_head.has_tp,
        )
        self.capture_aux_hidden_states = True
        self.hot_token_id = None

    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        with report_collective_sizing(ctx, ctx.bs, ctx.global_bs):
            return super().forward(ctx, input_ids, positions, **kwargs)

    def prepare_model_kwargs(
        self, ctx: ForwardContext, input_ids: torch.Tensor, kwargs: dict
    ) -> dict:
        model_kwargs = super().prepare_model_kwargs(ctx, input_ids, kwargs)
        captured_hidden_states = kwargs.get("captured_hidden_states")
        if captured_hidden_states is not None:
            model_kwargs["captured_hidden_states"] = captured_hidden_states
        else:
            # During CUDA graph capture warmup, provide dummy hidden states.
            target_hidden_size = getattr(
                self.config, "target_hidden_size", self.config.hidden_size
            )
            num_fc = self.model.num_fc_input_dim
            model_kwargs["captured_hidden_states"] = torch.zeros(
                input_ids.size(0),
                target_hidden_size * num_fc,
                dtype=torch.bfloat16,
                device=input_ids.device,
            )
        return model_kwargs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        remapped = []
        for name, loaded_weight in weights:
            if "d2t" in name:
                self.hot_token_id = loaded_weight + torch.arange(
                    loaded_weight.size(0), device=loaded_weight.device
                )
                continue
            if "t2d" in name:
                continue
            if "embed_tokens" in name:
                self._embed_loaded_from_checkpoint = True

            new_name = re.sub(r"^layers\.0\.", "midlayer.", name)

            if "lm_head" not in new_name:
                new_name = f"model.{new_name}"
            else:
                self.load_lm_head_from_target = False
            remapped.append((new_name, loaded_weight))

        return super().load_weights(remapped)

    def post_load_weights(self):
        self_attn = self.model.midlayer.self_attn
        if (
            self.quant_config is not None
            and hasattr(self.quant_config, "weight_block_size")
            and self_attn.kv_b_proj.weight.dtype
            in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        ):
            weight_block_size = self.quant_config.weight_block_size
            if weight_block_size is not None:
                if not hasattr(self_attn.kv_b_proj, "weight_scale_inv"):
                    raise RuntimeError(
                        "kv_b_proj.weight_scale_inv is required for block FP8 dequant."
                    )
                dtype = torch.get_default_dtype()
                w = block_dequant(
                    self_attn.kv_b_proj.weight,
                    self_attn.kv_b_proj.weight_scale_inv,
                    weight_block_size,
                ).to(dtype)
            else:
                w = self_attn.kv_b_proj.weight
        else:
            w = self_attn.kv_b_proj.weight

        self_attn.w_kc, self_attn.w_vc = _prepare_mla_kv_b_proj_weights(w, self_attn)

    def get_hot_token_id(self):
        return self.hot_token_id

    def mark_embedding_initialized(self) -> None:
        """Count the embedding as loaded, as when dummy weights fill it."""
        self._embed_loaded_from_checkpoint = True

    def set_embed_and_head(self, embed, head):
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        if self.model.embed_tokens.weight.shape == embed.shape:
            # A TP-sharded target embedding would read out of bounds here.
            del self.model.embed_tokens.weight
            self.model.embed_tokens.weight = embed
        elif not self._embed_loaded_from_checkpoint:
            raise ValueError(
                "EAGLE3 draft cannot share the target embedding "
                f"(target {tuple(embed.shape)} vs draft "
                f"{tuple(self.model.embed_tokens.weight.shape)}) and the draft "
                "checkpoint provided no embed_tokens weight"
            )
        else:
            logger.info(
                "EAGLE3 draft keeps its own embedding; target shape "
                f"{tuple(embed.shape)!s} differs",
            )
        if head is not None and self.load_lm_head_from_target:
            del self.lm_head.weight
            self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


EntryClass = [
    DeepseekV3ForCausalLM,
    Eagle3DeepseekV2ForCausalLM,
]
