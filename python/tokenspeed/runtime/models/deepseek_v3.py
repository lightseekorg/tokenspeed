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
from typing import Any

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.tokenspeed_mla import mla_kv_pack_quantize_fp8
from tokenspeed_kernel.ops.gemm.nvfp4_gemm_swiglu_nvfp4_quant import (
    nvfp4_gemm_swiglu_nvfp4_quant,
)
from tokenspeed_kernel.ops.gemm.trtllm import dsv3_fused_a_gemm
from tokenspeed_kernel.ops.moe.cuda import moe_finalize_fuse_shared
from tokenspeed_kernel.ops.quantization import fp8_quantize
from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize
from tokenspeed_kernel.ops.routing.cuda import dsv3_router_gemm
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cuda.merge_state import merge_state
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.configs.utils import get_rope_theta
from tokenspeed.runtime.layers.moe.checkpoint import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.utils import (
    CP_METADATA,
    ENABLE_CP,
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    get_layer_id,
)

_platform = current_platform()
_is_blackwell = _platform.is_blackwell
_is_hopper_plus = _platform.is_hopper_plus
_device_sm = _platform.arch_version.major * 10 + _platform.arch_version.minor

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.cuda_graph_wrapper import get_is_capture_mode
from tokenspeed.runtime.layers.activation import SiluAndMul
from tokenspeed.runtime.layers.attention.mla_fp8_utils import (
    mla_fused_rope_fp8_quantize,
)
from tokenspeed.runtime.layers.dense.nvfp4 import Nvfp4LinearMethod
from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
from tokenspeed.runtime.layers.moe.layer import MoELayer
from tokenspeed.runtime.layers.moe.topk import TopK
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType
from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.utils import (
    block_dequant,
    should_ignore_quant_layer,
)
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.model_loader.weight_utils import (
    default_weight_loader,
    kv_cache_scales_loader,
)
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.models.utils import (
    create_fused_set_kv_buffer_arg,
)
from tokenspeed.runtime.moe.distribution_recorder import (
    get_global_expert_distribution_recorder,
)
from tokenspeed.runtime.moe.expert_location import ModelConfigForExpertLocation
from tokenspeed.runtime.utils import LazyValue, add_prefix, get_colorful_logger
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.env import envs, global_server_args_dict
from tokenspeed.runtime.utils.pdl import pdl_enabled

logger = get_colorful_logger(__name__)

_OPTIONAL_MISSING_WEIGHT_SUFFIXES = (
    ".k_scale",
    ".v_scale",
)


class DeepseekV3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_shared_expert: bool = False,
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
        if x.size(0) == 0:
            return x

        if self._use_nvfp4_gemm_swiglu_nvfp4_quant:
            x_fc1_fp4, x_fc1_scale = fp4_quantize(
                x, self.gate_up_proj.input_scale_inv, enable_pdl=pdl_enabled()
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
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class MoEGate(nn.Module):
    _DSV3_ROUTER_GEMM_EXPERTS = (256, 384, 768)
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
            and config.n_routed_experts in self._DSV3_ROUTER_GEMM_EXPERTS
            and config.hidden_size in self._DSV3_ROUTER_GEMM_HIDDEN
        )

    def forward(self, hidden_states, comm_manager=None):
        if self.use_dsv3_router_gemm and hidden_states.size(0) > 0:
            logits = dsv3_router_gemm(
                hidden_states,
                self.weight,
                out_dtype=torch.float32,
                enable_pdl=pdl_enabled(),
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
            )

        self.experts = MoELayer(
            top_k=config.num_experts_per_tok,
            num_experts=config.n_routed_experts
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
                "n_group": getattr(config, "n_group", 0),
                "topk_group": getattr(config, "topk_group", 0),
                "routed_scaling_factor": getattr(config, "routed_scaling_factor", 1.0),
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
            apply_routed_scaling_factor_on_output=(
                self.experts.apply_routed_scaling_factor_on_output
            ),
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
        num_tokens = hidden_states.shape[0]

        with self.stream_fork.scope(enable=get_is_capture_mode()) as fork:
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states)
            if hidden_states.shape[0] > 0:
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
                enable_pdl=pdl_enabled(),
            )
        else:
            if not self.experts.apply_routed_scaling_factor_on_output:
                routed_expert_output *= self.routed_scaling_factor
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
            and block_scale is None
            and (output_dtype is None or output_dtype == torch.bfloat16)
        ):
            return dsv3_fused_a_gemm(x, self.weight.T)

        return super().forward(x, block_scale=block_scale, output_dtype=output_dtype)[0]


class DeepseekV3AttentionMLA(nn.Module):
    # Backends that use non-absorbed MLA kernels (ragged prefill, paged KV decode).
    _MLA_KERNEL_BACKENDS = ("trtllm_mla", "tokenspeed_mla")
    # Backends that support chunked ragged prefill with prefix replay.
    _RAGGED_PREFILL_BACKENDS = ("trtllm_mla", "tokenspeed_mla")

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
    ) -> None:
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
        self.num_heads = num_heads
        assert num_heads % self.mapping.attn.tp_size == 0
        self.num_local_heads = num_heads // self.mapping.attn.tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.config = config
        self.alt_stream = alt_stream
        self.attention_backend = global_server_args_dict["attention_backend"]
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
                tp_rank=self.mapping.attn.tp_rank,
                tp_size=self.mapping.attn.tp_size,
                tp_group=self.mapping.attn.tp_group,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
                tp_rank=self.mapping.attn.tp_rank,
                tp_size=self.mapping.attn.tp_size,
                tp_group=self.mapping.attn.tp_group,
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
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_attn_results,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
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
            elif hasattr(self.rotary_emb, "forward_cuda"):
                self.rotary_emb.forward = self.rotary_emb.forward_cuda
        else:
            self.rotary_emb = None

        # Fused RoPE+KV write kernel is incompatible with MLA: it assumes
        # K and V have the same head_dim, but MLA's KV cache is a single
        # [latent(512)|rope(64)] buffer where the two dimensions differ.
        # Passing this to the kernel causes thread overflow and silent
        # corruption of the latent cache.  All DeepSeek V2/V3 models use
        # MLA (kv_lora_rank > 0), so we unconditionally disable it here.
        self.use_fused_set_kv_buffer = False

        self.attn_mqa = PagedAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
        )

        self.attn_mha = PagedAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
        )

        self.w_kc = None
        self.w_vc = None

        self.dense_1_unqaunted = (
            self.check_unquanted(self.q_b_proj) if hasattr(self, "q_b_proj") else True
        )

    def check_unquanted(self, module) -> bool:
        return module.quant_config is None or should_ignore_quant_layer(
            module.prefix,
            ignored_layers=getattr(module.quant_config, "ignored_layers", []),
        )

    def no_absorb(self, ctx: ForwardContext) -> bool:
        if self.attention_backend in self._MLA_KERNEL_BACKENDS:
            # MLA kernel backends: Do not absorb when enabling ragged prefill
            # Target verify and draft extend have few tokens, go through absorb.
            return (
                not global_server_args_dict["mla_disable_ragged"]
                and ctx.forward_mode.is_extend()
                and not ctx.forward_mode.is_target_verify()
                and not ctx.forward_mode.is_draft_extend()
                and ctx.padded_static_len == -1
            )
        else:
            # Triton: Use normal computation for pure prefill and use weight absorption otherwise.
            # Note: extend_prefix_lens_cpu not available in ForwardContext, so we skip the
            # "no prefix cache" check. This is conservative — always absorb for Triton.
            return (
                ctx.forward_mode.is_extend()
                and not ctx.forward_mode.is_target_verify()
                and not ctx.forward_mode.is_draft_extend()
                and ctx.padded_static_len == -1
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None = None,
        can_run_flashinfer_fusion: bool = False,
    ) -> torch.Tensor:
        if self.no_absorb(ctx):
            return self.forward_normal_chunked(
                positions, hidden_states, ctx, out_cache_loc, comm_manager, block_scale
            )
        else:
            return self.forward_absorb(
                positions,
                hidden_states,
                ctx,
                out_cache_loc,
                comm_manager,
                block_scale,
                can_run_flashinfer_fusion,
            )

    def forward_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        comm_manager: CommManager,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            qkv = self.fused_qkv_a_proj_with_mqa(hidden_states)
            qkv = comm_manager.pre_attn_comm(qkv, ctx)
            q, latent_cache = qkv.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0]
        else:
            q = self.q_proj(hidden_states)[0]
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_a, k_pe = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]
        if self.rotary_emb is not None:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        latent_cache[:, : self.kv_lora_rank] = kv_a
        latent_cache[:, self.kv_lora_rank :] = k_pe
        latent_cache = latent_cache.unsqueeze(1)

        # Save latent cache
        ctx.token_to_kv_pool.set_kv_buffer(
            self.attn_mha, out_cache_loc, latent_cache, None
        )

        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe.unsqueeze(1)
        attn_output = self.attn_mha(q, k, v, ctx, out_cache_loc, save_kv_cache=False)

        attn_output = attn_output.view(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_absorb(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None = None,
        can_run_flashinfer_fusion: bool = False,
    ) -> torch.Tensor:
        Q, K = self.forward_absorb_qkv_proj(
            hidden_states,
            positions,
            ctx,
            out_cache_loc,
            comm_manager,
            block_scale,
            can_run_flashinfer_fusion,
        )
        output = self.forward_absorb_attn_o_proj(Q, K, ctx, out_cache_loc)
        return output

    def forward_absorb_qkv_proj(
        self,
        hidden_states,
        positions,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None = None,
        can_run_flashinfer_fusion: bool | None = None,
    ):
        if self.q_lora_rank is not None:
            qkv = self.fused_qkv_a_proj_with_mqa(
                hidden_states, block_scale, torch.bfloat16
            )

            if can_run_flashinfer_fusion and self.layer_id != 0:
                qkv, q_norm, k_nope, block_scale = (
                    self.fused_qk_layernorm.forward_with_allgather_fusion(
                        self.mapping.attn.tp_rank,
                        self.mapping.attn.tp_group,
                        qkv,
                        ctx.input_num_tokens,
                        fuse_block_quant_fp8=not self.dense_1_unqaunted,
                    )
                )
                latent_cache = qkv[..., self.q_lora_rank :]
                q = self.q_b_proj(q_norm, block_scale, torch.bfloat16)[0]
            else:
                qkv = comm_manager.pre_attn_comm(qkv, ctx)
                q, latent_cache = qkv.split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
                k_nope = latent_cache[..., : self.kv_lora_rank]

                # fused layernorm
                q_norm = torch.empty_like(q)
                if q.size(0) > 0:
                    self.fused_qk_layernorm(
                        input_q_a=q, input_kv_a=k_nope, output_q_a=q_norm
                    )

                q = self.q_b_proj(q_norm)[0]
        else:
            hidden_states = comm_manager.pre_attn_comm(hidden_states, ctx)
            q = self.q_proj(hidden_states)[0]
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            self.kv_a_layernorm(k_nope, inplace=True)

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        Q = torch.empty(
            q_nope.size(0),
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
        # k_nope do the RMSNorm inplace, so latent_cache contain k_nope after norm and k_pe *before* rotate
        K = latent_cache.unsqueeze(1)
        q_nope_out_view = Q[..., : self.kv_lora_rank]
        torch.bmm(
            q_nope.transpose(0, 1), self.w_kc, out=q_nope_out_view.transpose(0, 1)
        )
        # Model-owned fused FP8 decode: RoPE + quantize + KV cache write
        # all done here, so backend only needs to do attention.
        k_scale = getattr(self.attn_mqa, "k_scale_float", 1.0)
        use_fused_fp8_decode = (
            self.attention_backend in self._MLA_KERNEL_BACKENDS
            and (
                ctx.forward_mode.is_decode_or_idle()
                or ctx.forward_mode.is_target_verify()
            )
            and getattr(ctx.attn_backend, "data_type", None) == torch.float8_e4m3fn
            and self.rotary_emb is not None
            and k_scale == 1.0
        )

        if use_fused_fp8_decode:
            q_nope_absorbed = Q[..., : self.kv_lora_rank]
            k_nope_raw = K[..., : self.kv_lora_rank]
            k_pe_raw = K[..., self.kv_lora_rank :]

            query_fp8, key_fp8 = mla_fused_rope_fp8_quantize(
                q_nope=q_nope_absorbed,
                q_pe=q_pe,
                k_nope=k_nope_raw,
                k_pe=k_pe_raw,
                cos_sin_cache=self.rotary_emb.cos_sin_cache,
                positions=positions,
                is_neox=getattr(self.rotary_emb, "is_neox_style", True),
                k_scale_inv=1.0 / k_scale,
            )

            # Write FP8 KV cache (single write, no double-write)
            ctx.token_to_kv_pool.set_mla_kv_buffer(
                self.attn_mqa,
                out_cache_loc,
                cache_k_nope=key_fp8[..., : self.kv_lora_rank],
                cache_k_rope=key_fp8[..., self.kv_lora_rank :],
            )
            return query_fp8, key_fp8

        elif self.rotary_emb is not None and q_nope.size(0) > 0:
            # Apply RoPE directly on Q and K slices
            self.rotary_emb(
                positions,
                q_pe,
                K[..., self.kv_lora_rank :],
                fused_set_kv_buffer_arg=(
                    create_fused_set_kv_buffer_arg(
                        value=K[..., : self.kv_lora_rank],
                        layer=self.attn_mqa,
                        out_cache_loc=out_cache_loc,
                        token_to_kv_pool=ctx.token_to_kv_pool,
                    )
                    if self.use_fused_set_kv_buffer
                    else None
                ),
                output_q_rope=Q[..., self.kv_lora_rank :],
            )
        else:
            Q[..., self.kv_lora_rank :] = q_pe

        # For MLA kernel backends, write KV cache here (model-owned) so the
        # backend never has to. This unifies the FP8 fused path (written above)
        # and the BF16 path into a single ownership model.
        if (
            self.attention_backend in self._MLA_KERNEL_BACKENDS
            and not self.use_fused_set_kv_buffer
        ):
            ctx.token_to_kv_pool.set_mla_kv_buffer(
                self.attn_mqa,
                out_cache_loc,
                cache_k_nope=K[..., : self.kv_lora_rank],
                cache_k_rope=K[..., self.kv_lora_rank :],
            )

        return Q, K

    def forward_absorb_attn_o_proj(
        self,
        Q,
        K,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ):
        # MLA kernel backends: KV cache already written in forward_absorb_qkv_proj.
        # Other backends: write via fused_set_kv_buffer or let backend handle it.
        if self.attention_backend in self._MLA_KERNEL_BACKENDS:
            need_save_kv = False
        else:
            need_save_kv = not self.use_fused_set_kv_buffer

        attn_output = self.attn_mqa(
            Q,
            K,
            K[..., : self.kv_lora_rank],
            ctx,
            out_cache_loc,
            save_kv_cache=need_save_kv,
        )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
        attn_bmm_output = torch.empty(
            attn_output.size(0),
            self.num_local_heads,
            self.v_head_dim,
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        torch.bmm(
            attn_output.transpose(0, 1),
            self.w_vc,
            out=attn_bmm_output.transpose(0, 1),
        )
        output, _ = self.o_proj(attn_bmm_output.flatten(1, 2))
        return output

    def forward_normal_chunked(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None = None,
    ):
        q, k, v = self.forward_normal_chunked_kv_prepare(
            positions, hidden_states, ctx, out_cache_loc, comm_manager, block_scale
        )
        output = self.forward_normal_chunked_kv_core(q, k, v, ctx, out_cache_loc)
        return output

    def forward_normal_chunked_kv_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None = None,
    ):
        if self.q_lora_rank is not None:
            qkv = self.fused_qkv_a_proj_with_mqa(
                hidden_states, block_scale, torch.bfloat16
            )
            qkv = comm_manager.pre_attn_comm(qkv, ctx)
            q, kv_a, k_pe = qkv.split(
                [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            q_norm = torch.empty_like(q)
            kv_a_norm = torch.empty_like(kv_a)
            if q.size(0) > 0:
                self.fused_qk_layernorm(
                    input_q_a=q,
                    input_kv_a=kv_a,
                    output_q_a=q_norm,
                    output_kv_a=kv_a_norm,
                )
            q = self.q_b_proj(q_norm)[0]
            kv_a = kv_a_norm
            k_pe = k_pe.unsqueeze(1)
        else:
            hidden_states = comm_manager.pre_attn_comm(hidden_states, ctx)
            q = self.q_proj(hidden_states)[0]
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            kv_a, k_pe = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a = self.kv_a_layernorm(kv_a)
            k_pe = k_pe.unsqueeze(1)

        q = q.view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]

        # FP8 prefill: fused RoPE + FP8 quantize, direct FP8 KV cache write.
        # Disabled when k_scale != 1.0; mla_fp8_utils.py documents the current limitation.
        k_scale = getattr(self.attn_mha, "k_scale_float", 1.0)
        use_fp8_prefill = (
            self.attention_backend in self._MLA_KERNEL_BACKENDS
            and getattr(ctx.attn_backend, "data_type", None) == torch.float8_e4m3fn
            and self.rotary_emb is not None
            and k_scale == 1.0
        )

        if use_fp8_prefill:
            # Expand k_pe from [tokens,1,rope] to [tokens,heads,rope] for GQA
            k_pe_expanded = k_pe.expand(-1, self.num_local_heads, -1)

            q_fp8, k_fp8 = mla_fused_rope_fp8_quantize(
                q_nope=q_nope,
                q_pe=q_pe,
                k_nope=k_nope,
                k_pe=k_pe_expanded,
                cos_sin_cache=self.rotary_emb.cos_sin_cache,
                is_neox=getattr(self.rotary_emb, "is_neox_style", True),
                positions=positions,
                k_scale_inv=1.0 / k_scale,
            )

            v_fp8 = fp8_quantize(v, enable_pdl=pdl_enabled())

            # Write FP8 KV cache directly (skip BF16→FP8 conversion in pool)
            k_pe_for_cache = k_fp8[:, 0:1, self.qk_nope_head_dim :]
            kv_a_fp8 = fp8_quantize(kv_a, enable_pdl=pdl_enabled())
            ctx.token_to_kv_pool.set_mla_kv_buffer(
                self.attn_mha,
                out_cache_loc,
                cache_k_nope=kv_a_fp8.unsqueeze(1),
                cache_k_rope=k_pe_for_cache,
            )

            return q_fp8, k_fp8, v_fp8

        # BF16 path: apply RoPE, assemble Q/K, write cache
        if self.rotary_emb is not None:
            self.rotary_emb(
                positions,
                q_pe,
                k_pe,
                fused_set_kv_buffer_arg=(
                    create_fused_set_kv_buffer_arg(
                        value=kv_a.unsqueeze(1),
                        layer=self.attn_mha,
                        out_cache_loc=out_cache_loc,
                        token_to_kv_pool=ctx.token_to_kv_pool,
                    )
                    if self.use_fused_set_kv_buffer
                    else None
                ),
            )

        q[..., self.qk_nope_head_dim :] = q_pe
        k = torch.empty_like(q)
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe

        if not self.use_fused_set_kv_buffer:
            ctx.token_to_kv_pool.set_mla_kv_buffer(
                self.attn_mha,
                out_cache_loc,
                cache_k_nope=kv_a.unsqueeze(1),
                cache_k_rope=k_pe,
            )

        return q, k, v

    def forward_normal_chunked_kv_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ):
        attn_backend = ctx.attn_backend
        chunk_meta = attn_backend.chunked_prefill_metadata
        token_to_kv_pool = ctx.token_to_kv_pool

        # Scale compensation for FP8 prefill: bmm1_scale = k_scale * softmax_scale
        scaling = self.attn_mha.scaling
        k_scale = getattr(self.attn_mha, "k_scale_float", 1.0)
        if q.dtype == torch.float8_e4m3fn:
            scaling = k_scale * scaling

        # Causal self-attention over the new chunk tokens. q_lens == kv_lens ==
        # extend_seq_lens, so cum_seq_lens_q and cum_seq_lens_kv alias the same
        # cum_extend_seq_lens.
        num_extends = chunk_meta.extend_seq_lens.shape[0]
        accum_output, accum_lse = attn_backend.forward_extend_chunked(
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
        )

        # Always read KV cache as BF16 for kv_b_proj (weight is BF16), even if Q is FP8.
        read_dtype = (
            q.dtype
            if q.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
            else torch.bfloat16
        )

        for loop_idx in range(chunk_meta.chunked_loop_num):
            chunk_kv_indices = chunk_meta.chunk_kv_indices_list[loop_idx]

            kv_a_normed, k_pe = token_to_kv_pool.get_mla_kv_buffer(
                self.attn_mha, chunk_kv_indices, read_dtype
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
                k, v = mla_kv_pack_quantize_fp8(
                    k_nope, k_pe, v, k_scale_inv=1.0 / k_scale, enable_pdl=pdl_enabled()
                )
            else:
                # BF16 Attention
                k = torch.cat(
                    [k_nope, k_pe.expand(-1, self.num_local_heads, -1)], dim=-1
                )

            output, lse = attn_backend.forward_extend_chunked(
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

            accum_output, accum_lse = merge_state(
                accum_output, accum_lse, output, lse, enable_pdl=pdl_enabled()
            )

        attn_output = accum_output.view(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output


class DeepseekV3DecoderLayer(nn.Module):
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

        self.self_attn = DeepseekV3AttentionMLA(
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
        out_cache_loc: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:

        num_global_tokens, max_num_tokens_per_gpu = self.comm_manager.get_num_tokens(
            ctx
        )

        if not ctx.forward_mode.is_idle():
            hidden_states, residual = self.comm_manager.input_reduce_norm(
                hidden_states, residual
            )
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
                comm_manager=self.comm_manager,
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
        else:
            hidden_states = self.forward_mlp(
                hidden_states,
                residual,
                ctx,
                num_global_tokens,
                max_num_tokens_per_gpu,
            )
        return hidden_states, residual

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
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
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

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        if input_embeds is not None:
            hidden_states = input_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)
        if CP_METADATA:
            hidden_states = cp_split_and_rebuild_data(
                hidden_states,
                CP_METADATA.value.split_list,
                CP_METADATA.value.zigzag_index,
            )
            positions = cp_split_and_rebuild_data(
                positions, CP_METADATA.value.split_list, CP_METADATA.value.zigzag_index
            )
        residual = None
        aux_hidden_states = [] if self.layers_to_capture else None
        for i in range(len(self.layers)):
            if aux_hidden_states is not None and i in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    ctx,
                    out_cache_loc,
                    residual,
                )
        if not ctx.forward_mode.is_idle():
            if not ENABLE_CP:
                hidden_states = layer.comm_manager.final_norm(
                    hidden_states, residual, ctx, self.norm
                )
            else:
                hidden_states, _ = self.norm(hidden_states, residual)
        if CP_METADATA:
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                CP_METADATA.value,
                self.mapping.attn.tp_rank,
                self.mapping.attn.tp_group,
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

    def get_param(self, params_dict, name):
        if name in params_dict:
            return params_dict[name]

        if "language_model." in name:
            name = name.replace("language_model.", "")
            if name in params_dict:
                return params_dict[name]

        if name.endswith(_OPTIONAL_MISSING_WEIGHT_SUFFIXES):
            return None

        logger.warning("The %s is not in the model.", name)
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Fuse q_a_proj and kv_a_proj_with_mqa along output dimension when q_lora_rank is not None
        fuse_qkv_a_proj = getattr(self.config, "q_lora_rank", None) is not None

        params_dict = dict(self.named_parameters())
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
                break
            else:
                # Skip loading extra bias for GPTQ models.
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

        self.post_load_weights()

    def post_load_weights(self):
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
                    assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
                    dtype = torch.get_default_dtype()
                    w = block_dequant(
                        self_attn.kv_b_proj.weight,
                        self_attn.kv_b_proj.weight_scale_inv,
                        weight_block_size,
                    ).to(dtype)
            else:
                w = self_attn.kv_b_proj.weight

            w_kc, w_vc = w.unflatten(
                0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
            ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
            self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            self_attn.w_vc = w_vc.contiguous().transpose(1, 2)

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = self.mapping.attn.tp_size
        tp_rank = self.mapping.attn.tp_rank
        for layer_idx, scaling_factor in kv_cache_scales_loader(
            quantization_param_path,
            tp_rank,
            tp_size,
            self.config.num_hidden_layers,
            self.config.__class__.model_type,
        ):
            if not isinstance(self.model.layers[layer_idx], nn.Identity):
                self_attn = self.model.layers[layer_idx].self_attn
                # Set on both attn_mha (non-absorbed prefill) and attn_mqa (absorbed decode).
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
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=config.n_group,
        )


# ---------------------------------------------------------------------------
# Eagle3 MLA draft model
# ---------------------------------------------------------------------------


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
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.self_attn = DeepseekV3AttentionMLA(
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

        self.mlp = DeepseekV3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=getattr(
                config, "intermediate_size", config.hidden_size * 4
            ),
            hidden_act=getattr(config, "hidden_act", "silu"),
            mapping=self.mapping,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
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
        )

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states

        if not ctx.forward_mode.is_idle():
            fused_norm_out = torch.empty(
                embeds.size(0),
                self.hidden_size * 2,
                dtype=embeds.dtype,
                device=embeds.device,
            )
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
                out_cache_loc=out_cache_loc,
                comm_manager=self.comm_manager,
            )

            hidden_states, residual = self.comm_manager.post_attn_comm(
                hidden_states, residual, ctx
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

        hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
        hidden_states = self.mlp(hidden_states)
        hidden_states, residual = self.comm_manager.post_mlp_comm(
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
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
            prefix=add_prefix("embed_tokens", prefix),
        )

        layer_ids = self._get_eagle_layer_ids(config)
        self.num_fc_input_dim = len(layer_ids) if layer_ids is not None else 3

        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        fc_input_size = target_hidden_size * self.num_fc_input_dim

        self.fc = ReplicatedLinear(
            fc_input_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("fc", prefix),
        )

        self.midlayer = Eagle3MlaDecoderLayer(
            config,
            mapping=self.mapping,
            layer_id=0,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
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
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states, _ = self.fc(hidden_states)

        residual = None
        hidden_states, residual = self.midlayer(
            positions,
            embeds,
            hidden_states,
            ctx,
            out_cache_loc,
            residual,
        )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )
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
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            draft_vocab_size = (
                getattr(config, "draft_vocab_size", None) or config.vocab_size
            )
            if not hasattr(config, "draft_vocab_size"):
                self.load_lm_head_from_target = True
            self.lm_head = ParallelLMHead(
                draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                tp_rank=self.mapping.attn.tp_rank,
                tp_size=self.mapping.attn.tp_size,
                tp_group=self.mapping.attn.tp_group,
                prefix=add_prefix("lm_head", prefix),
            )

        self.logits_processor = LogitsProcessor(
            config,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
        )
        self.capture_aux_hidden_states = True
        self.hot_token_id = None

    def prepare_model_kwargs(
        self, ctx: ForwardContext, input_ids: torch.Tensor, kwargs: dict
    ) -> dict:
        model_kwargs = super().prepare_model_kwargs(ctx, input_ids, kwargs)
        captured_hidden_states = kwargs.get("captured_hidden_states")
        if captured_hidden_states is not None:
            model_kwargs["captured_hidden_states"] = captured_hidden_states
        else:
            # During CUDA graph capture warmup, provide dummy hidden states.
            num_tokens = input_ids.shape[0]
            target_hidden_size = getattr(
                self.config, "target_hidden_size", self.config.hidden_size
            )
            num_fc = self.model.num_fc_input_dim
            model_kwargs["captured_hidden_states"] = torch.zeros(
                num_tokens,
                target_hidden_size * num_fc,
                dtype=torch.bfloat16,
                device=input_ids.device,
            )
        return model_kwargs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        remapped = []
        for name, loaded_weight in weights:
            if "d2t" in name:
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue
            if "t2d" in name:
                continue

            new_name = re.sub(r"^layers\.0\.", "midlayer.", name)

            if "lm_head" not in new_name:
                new_name = f"model.{new_name}"
            else:
                self.load_lm_head_from_target = False
            remapped.append((new_name, loaded_weight))

        super().load_weights(remapped)

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
                assert hasattr(self_attn.kv_b_proj, "weight_scale_inv")
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

        w_kc, w_vc = w.unflatten(
            0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
        ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
        self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
        self_attn.w_vc = w_vc.contiguous().transpose(1, 2)

    def get_hot_token_id(self):
        return self.hot_token_id

    def set_embed_and_head(self, embed, head):
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        if head is not None and self.load_lm_head_from_target:
            del self.lm_head.weight
            self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


EntryClass = [
    DeepseekV3ForCausalLM,
    Eagle3DeepseekV2ForCausalLM,
]
