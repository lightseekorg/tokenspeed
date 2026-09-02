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

"""Kimi-K3 model.

Kimi-K3 = MoonViT3d vision tower (reused from Kimi-K2.5) + ``KimiLinear`` text
backbone (hybrid KDA linear-attention / NoPE-MLA full-attention decoder with a
DeepSeek-V3 style latent MoE and block-level attention residuals).

Implemented (full text path):

* ``KimiLinearMLAAttention`` — NoPE MLA + sigmoid output gate.
* ``KimiLinearKDA`` — per-head gated delta-rule linear attention; routes the
  conv + gated-delta scan + conv/recurrent state cache through the hybrid
  ``KdaAttnBackend``.
* ``KimiLinearMLP`` — dense / shared-expert MLP with the SiTU activation.
* ``KimiLinearMoE`` — sigmoid/noaux_tc router + Latent MoE + flashinfer's
  TRT-LLM fused SiTU + shared experts.
* ``KimiLinearDecoderLayer`` + ``KimiLinearModel`` — the AttnRes block-residual data
  flow.
* ``KimiLinearForCausalLM.load_weights`` — stacked / fused-qkv-a / expert
  mappings; post-load absorbed MLA ``w_kc``/``w_vc`` prep.
* ``KimiK3ForConditionalGeneration`` registration.

The multimodal path uses the shared MoonViT3d implementation with K3's
wide-QKV/RMSNorm configuration. At TP8 it runs as item-DP8 (vision TP1) via
``--mm-encoder-tp-mode data`` and gathers exact-size encoder outputs before
splicing them into the text embeddings.

Module hierarchy matches the checkpoint::

    KimiK3ForConditionalGeneration
      language_model (KimiLinearForCausalLM)
        model (KimiLinearModel): embed_tokens, layers[.self_attn/.mlp/...], norm
        lm_head
      vision (KimiK3Vision): vision_tower.*, mm_projector.*
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.activation.triton import (
    attnres_combine,
    attnres_partial,
    attnres_partial_dual,
    rmsnorm_gated_sigmoid,
    sigmoid_mul,
)
from tokenspeed_kernel.ops.attention import mla_normalize_project_query
from tokenspeed_kernel.ops.attn_res import attn_res_fwd, attn_res_fwd_available
from tokenspeed_kernel.ops.gemm import (
    kimi3_mla_qkv_gate_projection,
    kimi3_qkvfab_projection,
    kimi3_router_projection,
    kimi3_shared_down_projection,
    kimi3_shared_situ_projection,
    linear_attnres_partials,
    linear_attnres_partials_available,
)
from tokenspeed_kernel.ops.gemm.triton_gemv import (
    decode_gemv,
)
from tokenspeed_kernel.ops.moe import (
    latent_moe_decode_pipeline_available,
    latent_moe_input_projections,
)
from tokenspeed_kernel.ops.moe.flashinfer.trtllm_mxfp4 import (
    situ_moe_unavailable_reason,
)
from tokenspeed_kernel.ops.tuning import load_packaged_flashinfer_tuning_cache
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.configs.kimi_k3_config import KimiK3Config, KimiLinearConfig
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.comm_ops import (
    all_reduce,
    token_all_gather,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.pp_stage import PPStageState, pp_layer_window
from tokenspeed.runtime.execution.forward_step import (
    get_is_capture_mode,
    get_is_cuda_graph_phase,
)
from tokenspeed.runtime.layers.activation import SituAndMul
from tokenspeed.runtime.layers.layernorm import (
    RMSNorm,
)
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.latent import (
    Kimi3LatentProjection,
    Kimi3MoEExecutionPlan,
    LatentMoELayer,
    latent_moe_expert_shared_all_reduce,
)
from tokenspeed.runtime.layers.moe.loader import build_moe_checkpoint_loader
from tokenspeed.runtime.layers.moe.schema import ExpertCheckpointSchema
from tokenspeed.runtime.layers.moe.topk import TopK, TopKOutput, TopKOutputFormat
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType, get_moe_backend
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.modelopt_mixed import (
    preprocess_fp8_pb_wo_weights,
)
from tokenspeed.runtime.layers.quantization.utils import block_dequant
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM
from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3AttentionMLA,
    DeepseekV3FusedQkvAProjWithMqa,
    _prepare_mla_kv_b_proj_weights,
)
from tokenspeed.runtime.models.kimi_k3_comm import (
    K3AttnComm,
    K3AttnCommState,
    K3MoeTailComm,
)
from tokenspeed.runtime.models.moonvit import MoonViTVisionPath
from tokenspeed.runtime.multimodal.embedder import (
    EncoderSpec,
    VisionEmbedder,
    pad_input_tokens,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalInputs,
)
from tokenspeed.runtime.utils import add_prefix, ceil_div, make_layers
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.env import global_server_args_dict

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.multimodal.encoder_cudagraph import (
        EncoderForwardStepRunner,
    )

logger = logging.getLogger(__name__)

# ===----------------------------------------------------------------------=== #
# Multimodal vision path
# ===----------------------------------------------------------------------=== #


class KimiK3Vision(MoonViTVisionPath):
    """K3 MoonViT3d tower and patchmergerv2 projector.

    The encoder decomposition intentionally matches Kimi-K2.5: patch embedding
    and patch merging stay eager while only the shape-stable transformer block
    loop is captured. Keeping this object separate from the text wrapper also
    lets the top-level ``image_encoder`` callable be replaced by ModelExecutor's
    CUDA-graph wrapper without changing checkpoint parameter names.
    """

    def load_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
    ) -> None:
        name = name.replace("wqkv.", "attn.qkv_proj.")
        name = name.replace("wo.", "attn.proj.")
        name = name.replace("mm_projector.proj.0", "mm_projector.linear_1")
        name = name.replace("mm_projector.proj.2", "mm_projector.linear_2")
        if name not in params_dict:
            raise ValueError(f"Weight {name} not found in Kimi-K3 vision model")
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)


# ===----------------------------------------------------------------------=== #
# Text sublayers: dense MLP (SiTU), NoPE-MLA attention, AttnRes helper
# ===----------------------------------------------------------------------=== #


class KimiLinearMLP(nn.Module):
    """Dense / shared-expert MLP with the SiTU (SituGLU) activation.

    Mirrors ``DeepseekV3MLP`` (gate_up_proj + down_proj) but swaps SiLU for the
    Kimi SiTU activation. ``down_proj`` normally reduces its partial sum in
    place because Kimi-K3 runs the AttnRes residual path outside
    ``CommManager`` (see decision D4). EP Kimi can defer the shared-expert
    reduction so one Iris launch reduces it together with the routed latent.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        reduce_results: bool = True,
        is_shared_expert: bool = False,
        activation_situ_beta: float = 1.0,
        activation_situ_linear_beta: float | None = None,
    ) -> None:
        super().__init__()
        self.mapping = mapping
        if is_shared_expert:
            tp_rank = mapping.moe.tp_ep_rank
            tp_size = mapping.moe.tp_ep_size
            tp_group = mapping.moe.tp_ep_group
        else:
            tp_rank = mapping.dense.tp_rank
            tp_size = mapping.dense.tp_size
            tp_group = mapping.dense.tp_group

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
            reduce_results=reduce_results,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=tp_group,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SituAndMul(
            beta=activation_situ_beta, linear_beta=activation_situ_linear_beta
        )
        self.is_shared_expert = is_shared_expert

    def forward(
        self, x: torch.Tensor, down_out: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.size(0) == 0:
            return x
        if self.is_shared_expert:
            x = kimi3_shared_situ_projection(
                x,
                self.gate_up_proj.weight,
                beta=self.act_fn.beta,
                linear_beta=self.act_fn.linear_beta,
            )
            x = kimi3_shared_down_projection(
                x,
                self.down_proj.weight,
                out=down_out,
            )
            if self.down_proj.reduce_results and self.down_proj.tp_size > 1:
                x = all_reduce(x, self.down_proj.tp_group)
            return x
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        if down_out is not None:
            # Direct-write partial (unquantized bf16 shared experts only):
            # lands the TP partial straight into the fused-AR lane slice.
            torch.mm(x, self.down_proj.weight.t(), out=down_out)
            return down_out
        x, _ = self.down_proj(x)
        return x


class KimiLinearMLAAttention(DeepseekV3AttentionMLA):
    """Kimi-K3 full-attention layer: NoPE MLA + optional sigmoid output gate.

    Reuses ``DeepseekV3AttentionMLA`` wholesale (absorbed decode, chunked
    prefill, MLA kernels, latent KV pool) with two K3 deltas:

    * **NoPE** via the parent's ``skip_rope=True`` — no rotary embedding is
      built and every rope application in the parent is already guarded by
      ``self.rotary_emb is not None``.
    * **Output gate** (``mla_use_output_gate``): ``attn_out *= sigmoid(g_proj(x))``
      injected before the single ``o_proj`` call.

    ``reduce_attn_results=True`` (unlike DeepSeek's deferred RSAG reduce) so
    ``o_proj`` all-reduces here — the AttnRes path does not use
    ``CommManager`` to fold the attention comm into the residual.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        mapping: Mapping,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        layer_id=None,
        prefix: str = "",
        reduce_attn_results: bool = True,
        alt_stream: torch.cuda.Stream | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            mapping=mapping,
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=prefix,
            reduce_attn_results=reduce_attn_results,
            alt_stream=alt_stream,
            skip_rope=True,  # K3 MLA is NoPE (mla_use_nope=True)
        )
        self.use_output_gate = config.mla_use_output_gate
        if self.use_output_gate:
            assert q_lora_rank is not None, "gated MLA assumes the q-lora path"
            # The gate projection shares its input with the a-projections, so
            # its per-rank shard rides the same GEMV: one weight laid out as
            # [q_a | kv_a+rope | g_shard] x hidden. (The dsv3 min-latency
            # kernel is shape-locked to 2112 rows and measures below nvjet's
            # effective bandwidth here anyway.)
            self._qkv_a_width = q_lora_rank + kv_lora_rank + qk_rope_head_dim
            self._gate_width = num_heads * v_head_dim // mapping.attn.tp_size
            fused_prefix = add_prefix("fused_qkv_a_proj_with_mqa", prefix)
            fused_out = self._qkv_a_width + self._gate_width
            # FP8_PB_WO (w8a8) fused projection: pad the output rows to the
            # 128-block grid so Fp8LinearMethod keeps the flashinfer
            # blockscale GEMM (which requires N % 128 == 0) instead of the
            # Triton fallback. Rows use the PRIVATE [gate | q_a | kv_a | pad]
            # order (_FP8_FUSED_QKV_A_ORDER): all boundaries 128-aligned, so
            # checkpoint codes and scales load verbatim (zero
            # requantization); the zero pad rows complete kv_a's ragged
            # trailing block and are sliced off right after the GEMM.
            # bf16/mxfp4 checkpoints take pad 0, keep the canonical
            # [q_a | kv_a | gate] order, and construct exactly as before.
            self._fused_qkv_a_pad_rows = 0
            self._fused_qkv_a_fp8_layout = False
            if _fused_qkv_a_uses_fp8(quant_config, prefix):
                padded_out = ceil_div(fused_out, 128) * 128
                self._fused_qkv_a_pad_rows = padded_out - fused_out
                self._fused_qkv_a_fp8_layout = True
                fused_out = padded_out
            self.fused_qkv_a_proj_with_mqa = DeepseekV3FusedQkvAProjWithMqa(
                hidden_size,
                fused_out,
                bias=False,
                quant_config=quant_config,
                prefix=fused_prefix,
            )

    def _split_fused_qkv_a(
        self, qkv_gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the fused a-projection output by the active row layout.

        FP8 checkpoints load the PRIVATE [gate | q_a | kv_a] order
        (_FP8_FUSED_QKV_A_ORDER, 128-aligned boundaries, verbatim scales);
        bf16/mxfp4 keep the canonical [q_a | kv_a | gate].
        """
        if self._fused_qkv_a_fp8_layout:
            # Locked to the assembly order: a layout change edits the
            # constant, and this assertion turns any drift into a loud
            # failure instead of silently mis-splitting the projections.
            assert _FP8_FUSED_QKV_A_ORDER == (
                "g_proj",
                "q_a_proj",
                "kv_a_proj_with_mqa",
            )
            gate, q_a, latent_cache = qkv_gate.split(
                [
                    self._gate_width,
                    self.q_lora_rank,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                ],
                dim=-1,
            )
        else:
            q_a, latent_cache, gate = qkv_gate.split(
                [
                    self.q_lora_rank,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                    self._gate_width,
                ],
                dim=-1,
            )
        return q_a, latent_cache, gate

    def _project_q_latent_gated(
        self,
        hidden_states: torch.Tensor,
        ctx: "ForwardContext",
        comm_manager: CommManager,
        block_scale: torch.Tensor | None,
        attnres_partial_args: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Project MLA Q, latent KV, and the local output gate in one GEMM.

        Returns:
            query: Normalized and projected MLA query.
            latent_cache: Compressed latent KV and RoPE cache row.
            gate: Local output-gate shard.
            absorbed_query: Optional decode query projected into latent key space.
        """
        if block_scale is not None:
            qkv_gate = self.fused_qkv_a_proj_with_mqa(
                hidden_states, block_scale, torch.bfloat16
            )
            if attnres_partial_args is not None:
                attnres_partial_dual(*attnres_partial_args)
            if self._fused_qkv_a_pad_rows:
                # Drop the zero pad rows of the 128-aligned FP8 projection
                # before anything consumes the output.
                qkv_gate = qkv_gate[
                    ..., : self._qkv_a_width + self._gate_width
                ].contiguous()
            qkv_gate = comm_manager.pre_attn_comm(qkv_gate, ctx)
            q_a, latent_cache, gate = self._split_fused_qkv_a(qkv_gate)
        elif self.fused_qkv_a_proj_with_mqa.weight.dtype in _FP8_WEIGHT_DTYPES:
            # FP8-resident fused projection (FP8_PB_WO w8a8): the bf16 fast
            # kernels below cannot consume the quantized weight, so run the
            # quantized module GEMM and keep any hoisted dual-partials as a
            # standalone kernel — the same recipe as the block_scale branch.
            # (can_fuse_attnres_partials already returns False for FP8
            # weights, so args are normally None here.)
            if attnres_partial_args is not None:
                attnres_partial_dual(*attnres_partial_args)
            qkv_gate = self.fused_qkv_a_proj_with_mqa(hidden_states)
            if self._fused_qkv_a_pad_rows:
                # Drop the zero pad rows of the 128-aligned FP8 projection
                # before anything consumes the output.
                qkv_gate = qkv_gate[
                    ..., : self._qkv_a_width + self._gate_width
                ].contiguous()
            qkv_gate = comm_manager.pre_attn_comm(qkv_gate, ctx)
            q_a, latent_cache, gate = self._split_fused_qkv_a(qkv_gate)
        elif attnres_partial_args is not None:
            blocks, weight_a, weight_b, eps, scratch_a, scratch_b = attnres_partial_args
            qkv_gate = linear_attnres_partials(
                hidden_states,
                self.fused_qkv_a_proj_with_mqa.weight,
                blocks,
                weight_a,
                weight_b,
                scratch_a,
                scratch_b,
                eps=eps,
            )
            qkv_gate = comm_manager.pre_attn_comm(qkv_gate, ctx)
            q_a, latent_cache, gate = qkv_gate.split(
                [
                    self.q_lora_rank,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                    self._gate_width,
                ],
                dim=-1,
            )
        else:
            projection = kimi3_mla_qkv_gate_projection(
                hidden_states,
                self.fused_qkv_a_proj_with_mqa.weight,
                self._qkv_a_width,
            )
            if projection.packed is not None:
                qkv_gate = comm_manager.pre_attn_comm(projection.packed, ctx)
                q_a, latent_cache, gate = qkv_gate.split(
                    [
                        self.q_lora_rank,
                        self.kv_lora_rank + self.qk_rope_head_dim,
                        self._gate_width,
                    ],
                    dim=-1,
                )
            else:
                qkv = comm_manager.pre_attn_comm(projection.qkv, ctx)
                q_a, latent_cache = qkv.split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
                gate = projection.gate
        kv_a = latent_cache[..., : self.kv_lora_rank]
        if self.q_b_proj.weight.dtype in _FP8_WEIGHT_DTYPES:
            # FP8-resident q_b (FP8_PB_WO w8a8): mla_normalize_project_query
            # consumes a raw bf16 weight, so fall back to the unfused parent
            # recipe — fused q_a/kv_a norms, then the quantized q_b GEMM.
            # No absorbed-query preparation on this path; forward_absorb
            # projects the query itself when absorbed_query is None.
            q_norm = torch.empty_like(q_a)
            if q_a.size(0) > 0:
                self.fused_qk_layernorm(
                    input_q_a=q_a, input_kv_a=kv_a, output_q_a=q_norm
                )
            q = self.q_b_proj(q_norm)[0]
            return q, latent_cache, gate, None
        q, absorbed_query = mla_normalize_project_query(
            q_a,
            kv_a,
            self.fused_qk_layernorm.weight_q_a,
            self.fused_qk_layernorm.weight_kv_a,
            self.q_b_proj.weight,
            eps=self.q_a_layernorm.variance_epsilon,
            prepare_absorbed_query=ctx is not None and ctx.num_extends == 0,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
        )
        return q, latent_cache, gate, absorbed_query

    def can_fuse_attnres_partials(
        self,
        hidden_states: torch.Tensor,
        args: tuple,
    ) -> bool:
        projection = getattr(self, "fused_qkv_a_proj_with_mqa", None)
        if projection is None:
            return False
        blocks, weight_a, weight_b, eps, scratch_a, scratch_b = args
        return linear_attnres_partials_available(
            hidden_states,
            projection.weight,
            blocks,
            weight_a,
            weight_b,
            scratch_a,
            scratch_b,
            eps=eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: "ForwardContext",
        comm_manager,
        block_scale: torch.Tensor | None = None,
        attnres_partial_args: tuple | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        if self.use_output_gate:
            q, latent_cache, gate, absorbed_query = self._project_q_latent_gated(
                hidden_states,
                ctx,
                comm_manager,
                block_scale,
                attnres_partial_args,
            )
        else:
            if attnres_partial_args is not None:
                attnres_partial_dual(*attnres_partial_args)
            q, latent_cache = self._project_q_latent(
                hidden_states, ctx, comm_manager, block_scale
            )
            gate = None
            absorbed_query = None
        fuse_value_gate = (
            gate is not None
            and ctx.num_extends == 0
            and ctx.attn_backend.supports_mla_projected_value_decode
        )
        attn_output = self._attn(
            positions,
            q,
            latent_cache,
            ctx,
            output_gate=gate if fuse_value_gate else None,
            absorbed_query=absorbed_query,
        )
        if gate is not None and not fuse_value_gate:
            # Fused in-place fp32 sigmoid+mul; the gate shard matches the
            # head-sharded attn_output.
            attn_output = sigmoid_mul(attn_output, gate)
        output, _ = self.o_proj(attn_output)
        return output


def _sliced_scratch(like: torch.Tensor, slot: int, n_tokens: int):
    """The (m, s, acc) scratch views for the first ``n_tokens`` rows."""
    m, s_, acc = _attnres_scratch(like, slot=slot)
    return m[:n_tokens], s_[:n_tokens], acc[:n_tokens]


def _apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Module,
    norm: RMSNorm,
    num_valid_blocks: int,
    out_norm: RMSNorm | None = None,
    *,
    delta: torch.Tensor | None = None,
    block_write_idx: int = -1,
) -> torch.Tensor:
    """AttnRes mixing: a learned softmax attention over block-residual snapshots.

    Candidates are the ``num_valid_blocks`` historical snapshots
    ``block_residual[:num_valid_blocks]`` plus the current ``prefix_sum``, mixed
    by a learned per-block weight ``softmax(RMSNorm(v) @ (norm.weight *
    proj.weight))`` (mirrors the checkpoint's ``modeling_kimi.py``), replacing
    the plain residual add on the Kimi-K3 AttnRes path. The fused ``attn_res``
    kernel does the whole mix in fp32 (it sits on the global residual backbone,
    where bf16 rounding would drift the stream), with a torch fallback for
    unsupported shapes. Both paths are CUDA-graph capture-compatible.

    ``block_residual`` is block-major ``[num_blocks, T, hidden]``. When
    ``out_norm`` is given, the following RMSNorm is fused into the kernel
    epilogue and the normed mix is returned.
    """
    if num_valid_blocks <= 0 and delta is None and block_write_idx < 0:
        return prefix_sum if out_norm is None else out_norm(prefix_sum)

    # Calls that do not append a snapshot only need the valid rows. Preserve the
    # tightly sliced contract used by backends without in-kernel snapshot writes.
    kernel_blocks = (
        block_residual if block_write_idx >= 0 else block_residual[:num_valid_blocks]
    )
    return attn_res_fwd(
        prefix_sum,
        kernel_blocks,
        proj.weight.reshape(-1),
        norm.weight,
        norm.variance_epsilon,
        out_norm_weight=None if out_norm is None else out_norm.weight,
        out_norm_eps=None if out_norm is None else out_norm.variance_epsilon,
        delta=delta,
        num_valid_blocks=num_valid_blocks,
        block_write_idx=block_write_idx,
    )


def _situ_betas(config: KimiLinearConfig) -> tuple[float, float | None]:
    """(situ_beta, situ_linear_beta); every K3 MLP runs SiTU, so fail loud
    on any other ``hidden_act`` instead of silently running SiTU anyway."""
    if config.hidden_act != "situ":
        raise ValueError(
            f"KimiLinear MLPs only implement the 'situ' activation, got "
            f"{config.hidden_act!r}"
        )
    return (config.activation_situ_beta, config.activation_situ_linear_beta)


# FP8 storage dtypes accepted by the FP8-resident paths (matches the
# quantization layers' width: e4m3fn on NVIDIA, e4m3fnuz on older ROCm).
_FP8_WEIGHT_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


# FP8_PB_WO fused_qkv_a PRIVATE row layout (FP8 mode only). Segments are
# REORDERED to [gate | q_a | kv_a | zero tail pad] so that every segment
# boundary lands on the 128-row scale-block grid: the gate shard
# (num_heads*v_head_dim/attn_tp) and q_a (q_lora_rank) are 128-multiples,
# and kv_a (kv_lora_rank + rope = 576) sits last so its ragged half block is
# completed by the zero pad rows (which dequantize to exact zeros under any
# scale). Codes and scale rows are copied VERBATIM from the checkpoint — no
# requantization anywhere. bf16/mxfp4 checkpoints keep the canonical
# [q_a | kv_a | gate] order; consumers must split through
# ``KimiLinearMLAAttention._split_fused_qkv_a``, never with checkpoint
# offsets.
_FP8_FUSED_QKV_A_ORDER = ("g_proj", "q_a_proj", "kv_a_proj_with_mqa")


def _fused_qkv_a_uses_fp8(quant_config, layer_prefix: str) -> bool:
    """Whether the fused a-projection must be built FP8-resident.

    Real ModelOpt exports carry both the fused aliases and the per-segment
    entries, but either alone must suffice: the alias
    (``fused_qkv_a_proj_with_mqa``) or any source segment (q_a / kv_a / MLA
    g) resolving to FP8_PB_WO selects the FP8 layout. Segments that disagree
    (some FP8_PB_WO, some unquantized) cannot be fused verbatim — fail with
    the actual routes instead of a misleading 128-alignment error later.
    """
    fp8_pb_wo_route = getattr(quant_config, "fp8_pb_wo_route", None)
    if fp8_pb_wo_route is None:
        return False
    segment_routes = {
        leaf: fp8_pb_wo_route(add_prefix(leaf, layer_prefix))
        for leaf in ("q_a_proj", "kv_a_proj_with_mqa", "g_proj")
    }
    segment_values = set(segment_routes.values())
    if "w8a8" in segment_values and None in segment_values:
        raise ValueError(
            "fused_qkv_a segments are partially FP8_PB_WO-quantized; all of "
            f"q_a/kv_a/g must agree to fuse verbatim: {segment_routes}"
        )
    alias_route = fp8_pb_wo_route(add_prefix("fused_qkv_a_proj_with_mqa", layer_prefix))
    return alias_route == "w8a8" or "w8a8" in segment_values


def _assemble_fp8_fused_qkv_a(
    segments: list[tuple[torch.Tensor, torch.Tensor]],
    total_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack FP8 segments and their scale grids verbatim, zero tail pad.

    Args:
        segments: ``(codes_fp8 [n_i, K], scale_f32 [ceil(n_i/128), K/128])``
            in the fused row order; every segment except the last must be a
            128-row multiple so the scale grids concatenate 1:1.
        total_rows: Fused row count (the constructor's 128-padded output).

    Returns:
        ``(codes_fp8 [total_rows, K], scale_f32 [total_rows/128, K/128])``
        with bit-identical segment codes/scales and exact-zero pad rows.
    """
    k = segments[0][0].shape[1]
    n_real = sum(weight.shape[0] for weight, _ in segments)
    if total_rows % 128 or total_rows < n_real:
        raise ValueError(
            f"fused rows {total_rows} must be a 128-multiple covering "
            f"{n_real} segment rows"
        )
    device = segments[0][0].device
    fused_w = torch.zeros(total_rows, k, dtype=torch.float8_e4m3fn, device=device)
    fused_s = torch.ones(
        total_rows // 128, (k + 127) // 128, dtype=torch.float32, device=device
    )
    row = 0
    block = 0
    for index, (weight, scale) in enumerate(segments):
        if weight.dtype not in _FP8_WEIGHT_DTYPES:
            raise TypeError(f"fused segment {index} must be FP8, got {weight.dtype}")
        rows = weight.shape[0]
        if index < len(segments) - 1 and rows % 128:
            raise ValueError(
                f"interior fused segment {index} has {rows} rows; only the "
                "last segment may be ragged on the 128 grid"
            )
        nblocks = (rows + 127) // 128
        if scale.shape != (nblocks, fused_s.shape[1]):
            raise ValueError(
                f"segment {index} scale shape {tuple(scale.shape)} does not "
                f"match its {rows}x{k} codes"
            )
        fused_w[row : row + rows] = weight.to(device)
        fused_s[block : block + nblocks] = scale.to(device)
        row += rows
        block += nblocks
    # Any remaining scale rows (fully-pad blocks) keep the 1.0 guard: their
    # zero codes dequantize to exact zeros under any scale.
    return fused_w, fused_s


def _shard_k3_up_projection(mapping: Mapping, hidden_size: int) -> bool:
    """Whether to column-shard K3's routed up projection on NVIDIA."""
    return (
        torch.version.hip is None
        and mapping.moe.tp_ep_size > 1
        and hidden_size % mapping.moe.tp_ep_size == 0
    )


def _k3_trtllm_situ_internal_activation_dtype(
    quant_config: QuantizationConfig | None, prefix: str
) -> str:
    """Activation trait for FlashInfer TRT-LLM SiTU MoE by expert weight dtype.

    MXFP4 SiTU cubins are w4a8 (activations quantized to MXFP8 -> ``"fp8"``);
    NVFP4 SiTU runs w4a4 with the kernel wrapper quantizing the bf16 input to
    NVFP4 itself, which the kernel registry models as ``"input"``.
    """
    if quant_config is not None and quant_config.moe_weight_dtype(prefix) == "nvfp4":
        return "input"
    return "fp8"


# ===----------------------------------------------------------------------=== #
# Text decoder layers
# ===----------------------------------------------------------------------=== #


class KimiKDAMergedProj(nn.Module):
    """Merged KDA input projections: ``[q | k | v | g | f_a | b]``, one GEMM.

    All six consume the post-norm hidden states. q/k/v/g/b shard per-head
    over the attention TP group; ``f_a`` (low-rank decay-gate down
    projection) is replicated, so each rank carries a full copy. The ``q|k|v``
    slice reproduces the layout the hybrid backend's conv expects; ``g``,
    ``f_a`` and ``b`` (beta logits) ride along as strided slices, replacing two
    extra latency-bound GEMVs per layer. Loader ``shard_id`` in
    {"q","k","v","g","f_a","b"}.

    bf16 checkpoints (mxfp4 recipe): rows pad to a multiple of 16
    (off-multiple row counts fall off cublasLt's fast M=1 tactic, 30us vs
    13us at 6284 vs 6288 x 7168) and the GEMM is plain bf16 — unchanged.

    FP8_PB_WO checkpoints (``fp8_block_quant=True``): the buffer stays
    FP8-resident with a per-[128, 128]-block f32 dequant scale grid, and the
    used rows pad straight to the 128 grid (3206 -> 3328 @tp16,
    6284 -> 6400 @tp8; the bf16 16-row alignment does not apply here) so the
    w8a8 blockscale GEMM keeps the flashinfer kernel (N % 128 == 0). Every
    segment offset (0/p/2p/3p/4p/4p+head_dim) is 128-aligned, so the
    checkpoint's per-segment scale grids concatenate directly with no
    requantization; the zero pad rows share ``b``'s trailing block scale and
    dequantize to exact zeros (pad lemma).
    """

    _ROW_ALIGN = 16

    def __init__(
        self,
        hidden_size: int,
        proj: int,
        num_heads: int,
        head_dim: int,
        tp_rank: int,
        tp_size: int,
        fp8_block_quant: bool = False,
    ) -> None:
        super().__init__()
        self.proj_local = proj // tp_size
        self.local_num_heads = num_heads // tp_size
        self.head_dim = head_dim
        self.tp_rank = tp_rank
        self.fp8_block_quant = fp8_block_quant
        p = self.proj_local
        self._offsets = {
            "q": 0,
            "k": p,
            "v": 2 * p,
            "g": 3 * p,
            "f_a": 4 * p,
            "b": 4 * p + head_dim,
        }
        self._rows = {
            "q": p,
            "k": p,
            "v": p,
            "g": p,
            "f_a": head_dim,
            "b": self.local_num_heads,
        }
        used = 4 * p + head_dim + self.local_num_heads
        self.used_rows = used
        if fp8_block_quant:
            if p % 128 or head_dim % 128 or hidden_size % 128:
                raise ValueError(
                    "FP8 merged KDA projection requires 128-aligned segment "
                    f"offsets, got proj_local={p} head_dim={head_dim} "
                    f"hidden_size={hidden_size}"
                )
            total = ceil_div(used, 128) * 128
            # Zero codes: the pad rows (and any unwritten row) dequantize to
            # exact zeros under any scale.
            self.weight = nn.Parameter(
                torch.zeros(total, hidden_size, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )
            self.weight_scale_inv = nn.Parameter(
                torch.ones(total // 128, hidden_size // 128, dtype=torch.float32),
                requires_grad=False,
            )
            self.weight_scale_inv.weight_loader = self._load_scale
            # flashinfer MN-major prepacked scales, prepared post-load.
            self._flashinfer_scales_mn: torch.Tensor | None = None
            # Zero-initialized buffers make a missing shard silently read as
            # zeros; track loads explicitly and verify at post_load_weights.
            self._loaded_weight_shards: set[str] = set()
            self._loaded_scale_shards: set[str] = set()
        else:
            total = ceil_div(used, self._ROW_ALIGN) * self._ROW_ALIGN
            # Explicit bf16: default-dtype fp32 would cost +6 GiB/rank and starve the KV budget.
            self.weight = nn.Parameter(
                torch.empty(total, hidden_size, dtype=torch.bfloat16)
            )
            # Padding rows are never read back, but keep them finite.
            self.weight.data[used:].zero_()
        self.weight.weight_loader = self._load_weight

    def _load_weight(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str
    ) -> None:
        if self.fp8_block_quant and loaded_weight.dtype not in _FP8_WEIGHT_DTYPES:
            raise TypeError(
                "FP8-resident merged KDA projection cannot load a "
                f"{loaded_weight.dtype} shard (bf16 refit of FP8 KDA weights "
                "is unsupported)."
            )
        rows = self._rows[shard_id]
        # f_a is replicated (full copy per rank); the rest are row-sharded.
        src = (
            loaded_weight
            if shard_id == "f_a"
            else loaded_weight.narrow(0, self.tp_rank * rows, rows)
        )
        start = self._offsets[shard_id]
        param.data[start : start + rows].copy_(src)
        if self.fp8_block_quant:
            self._loaded_weight_shards.add(shard_id)

    def _load_scale(
        self, param: nn.Parameter, loaded_scale: torch.Tensor, shard_id: str
    ) -> None:
        """Concatenate the checkpoint's per-segment scale grids in place.

        Segment offsets are all 128-aligned, so each shard's scale rows map
        1:1 onto the merged grid. ``f_a`` is replicated; ``b``'s <=96-row
        shard lives inside the checkpoint's single scale block regardless of
        rank, and the zero pad rows in the trailing block dequantize to
        exact zeros under it.
        """
        block_start = self._offsets[shard_id] // 128
        if shard_id in ("f_a", "b"):
            src = loaded_scale[:1]
            nblocks = 1
        else:
            nblocks = self._rows[shard_id] // 128
            src = loaded_scale.narrow(0, self.tp_rank * nblocks, nblocks)
        param.data[block_start : block_start + nblocks].copy_(src)
        self._loaded_scale_shards.add(shard_id)

    def verify_fp8_load_complete(self) -> None:
        """Raise if any FP8 segment or scale shard never loaded.

        The FP8 buffers are zero-initialized, so a dropped shard would
        otherwise silently project to zeros.
        """
        if not self.fp8_block_quant:
            return
        expected = set(self._rows)
        missing_weights = expected - self._loaded_weight_shards
        missing_scales = expected - self._loaded_scale_shards
        if missing_weights or missing_scales:
            raise RuntimeError(
                "FP8 merged KDA projection incompletely loaded: missing "
                f"weight shards {sorted(missing_weights)}, scale shards "
                f"{sorted(missing_scales)}."
            )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.fp8_block_quant:
            # Fail fast instead of feeding FP8 codes to the bf16 GEMV:
            # FP8-resident KDA projections must run through the quantized
            # w8a8 branch of kimi3_qkvfab_projection (KimiLinearKDA
            # _project_qkvfab), which consumes the weight + scale directly.
            raise RuntimeError(
                "KimiKDAMergedProj.forward does not support the FP8-resident "
                "buffer; use kimi3_qkvfab_projection with weight_scale."
            )
        # Registry-dispatched: rowcta at M=1, cublasLt otherwise.
        out = decode_gemv(x, self.weight)
        p = self.proj_local
        mixed_qkv = out[:, : 3 * p]
        f_a_end = 4 * p + self.head_dim
        return (
            mixed_qkv,
            out[:, 3 * p : 4 * p],
            out[:, 4 * p : f_a_end],
            out[:, f_a_end : f_a_end + self.local_num_heads],
        )


class KimiLinearKDA(nn.Module):
    """KDA (Kimi Delta Attention) linear-attention sublayer.

    Gated delta-rule: ``q/k/v_proj`` + short causal conv (SiLU), a decay gate
    ``f_b(f_a(x))`` combined with a **per-head** ``A_log[num_heads]`` (stored in
    the checkpoint as a zero-padded ``[head_dim]`` buffer) / per-(head,
    channel) ``dt_bias``, a per-head ``beta``, then the gated-delta scan (``fla``
    ``chunk_kda`` for prefill), a gated RMSNorm with the ``g_proj`` sigmoid output
    gate, and ``o_proj``.

    The layer owns the projections + gates + output norm and routes the conv +
    gated-delta scan + conv/recurrent state cache through the hybrid attention
    backend (``ctx.attn_backend`` -> ``KdaAttnBackend``), mirroring
    ``Qwen3_5GatedDeltaNet``. The ``q/k/v_conv1d_weight`` parameters only hold the
    conv kernels; the convolution itself runs in the backend.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_id = layer_id

        la = config.linear_attn_config
        if not la.get("use_full_rank_gate", False):
            # The reference has a low-rank g_a_proj/g_b_proj output-gate variant;
            # only the full-rank g_proj (what the K3 checkpoint uses) is wired
            # here. Fail loudly instead of surfacing missing-weight errors.
            raise NotImplementedError(
                "KimiLinearKDA only implements the full-rank output gate "
                "(linear_attn_config.use_full_rank_gate=True)."
            )
        self.num_heads = la["num_heads"]
        self.head_dim = la["head_dim"]
        self.conv_size = la["short_conv_kernel_size"]
        self.gate_lower_bound = la.get("gate_lower_bound")
        proj = self.num_heads * self.head_dim
        hidden = config.hidden_size

        # KDA parallelism reads the linear-attention mapping: it defaults to
        # the attention TP width and diverges only under the
        # MLA-DP + linear-attn-TP hybrid.
        tp_rank = mapping.linear_attn.tp_rank
        tp_size = mapping.linear_attn.tp_size
        tp_group = mapping.linear_attn.tp_group
        self.local_num_heads = self.num_heads // tp_size
        proj_local = proj // tp_size

        def _col(in_f, out_f, name):
            return ColumnParallelLinear(
                in_f,
                out_f,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix(name, prefix),
                tp_rank=tp_rank,
                tp_size=tp_size,
                tp_group=tp_group,
            )

        # One merged GEMM replaces four per-head-sharded projections + the qkv concat.
        # FP8_PB_WO checkpoints keep the merged buffer FP8-resident (w8a8
        # blockscale GEMM); everything else stays bf16 exactly as before.
        fp8_pb_wo_route = getattr(quant_config, "fp8_pb_wo_route", None)
        merged_fp8 = (
            fp8_pb_wo_route is not None
            and fp8_pb_wo_route(add_prefix("q_proj", prefix)) == "w8a8"
        )
        self.qkvgb_proj = KimiKDAMergedProj(
            hidden_size=hidden,
            proj=proj,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            tp_rank=tp_rank,
            tp_size=tp_size,
            fp8_block_quant=merged_fp8,
        )
        # Decay-gate up projection (f_a and beta ride in the merged GEMM).
        self.f_b_proj = _col(self.head_dim, proj, "f_b_proj")

        # q/k/v short-conv kernels [proj_local, 1, W]. The conv itself runs in the
        # backend; these params only hold the weights. Named ``*_conv1d_weight``
        # (plain parameters, no wrapper module) -- the checkpoint's
        # ``<name>_conv1d.weight`` key is remapped in load_weights.
        self.q_conv1d_weight = nn.Parameter(torch.zeros(proj_local, 1, self.conv_size))
        self.k_conv1d_weight = nn.Parameter(torch.zeros(proj_local, 1, self.conv_size))
        self.v_conv1d_weight = nn.Parameter(torch.zeros(proj_local, 1, self.conv_size))

        # A_log is per-head [num_heads] (one log-decay per head). The
        # checkpoint stores it in a [head_dim]-sized buffer with only the first
        # num_heads entries populated (the rest zero-padded), so load this rank's
        # heads [local*rank : local*(rank+1)] and drop the padded tail. dt_bias
        # and the q/k/v conv weights are per-(head, channel), so they shard along
        # dim 0 by the attention TP rank.
        self.A_log = nn.Parameter(
            torch.zeros(self.local_num_heads, dtype=torch.float32)
        )
        _alog_start = self.local_num_heads * tp_rank
        _alog_n = self.local_num_heads

        def _a_log_head_loader(param, loaded_weight):
            param.data.copy_(loaded_weight.narrow(0, _alog_start, _alog_n))

        self.A_log.weight_loader = _a_log_head_loader
        self.dt_bias = nn.Parameter(torch.zeros(proj_local, dtype=torch.float32))
        self.dt_bias.weight_loader = sharded_weight_loader(0, tp_rank)
        for w in (self.q_conv1d_weight, self.k_conv1d_weight, self.v_conv1d_weight):
            w.weight_loader = sharded_weight_loader(0, tp_rank)
        # Fused (q, k, v) conv kernel bank; built once in post_load_weights.
        self.conv_weights: torch.Tensor | None = None

        self.o_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = RowParallelLinear(
            proj,
            hidden,
            bias=False,
            reduce_results=False,  # layer-level fused AR+residual owns the reduce
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

    def fuse_conv_weights(self) -> None:
        """Concatenate the loaded q/k/v conv kernels into ``self.conv_weights``."""
        self.conv_weights = torch.cat(
            (self.q_conv1d_weight, self.k_conv1d_weight, self.v_conv1d_weight), dim=0
        ).squeeze(1)

    def _project_qkvfab(
        self,
        hidden_states: torch.Tensor,
        attnres_partial_args: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project every KDA hidden-state consumer."""
        proj_local = self.local_num_heads * self.head_dim
        if attnres_partial_args is None:
            output = kimi3_qkvfab_projection(
                hidden_states,
                self.qkvgb_proj.weight,
                weight_scale=getattr(self.qkvgb_proj, "weight_scale_inv", None),
                prepacked_scales=getattr(
                    self.qkvgb_proj, "_flashinfer_scales_mn", None
                ),
            )
        else:
            blocks, weight_a, weight_b, eps, scratch_a, scratch_b = attnres_partial_args
            output = linear_attnres_partials(
                hidden_states,
                self.qkvgb_proj.weight,
                blocks,
                weight_a,
                weight_b,
                scratch_a,
                scratch_b,
                eps=eps,
            )
        f_a_end = 4 * proj_local + self.head_dim
        mixed_qkv = output[:, : 3 * proj_local]
        return (
            mixed_qkv,
            output[:, 3 * proj_local : 4 * proj_local],
            output[:, 4 * proj_local : f_a_end],
            output[:, f_a_end : self.qkvgb_proj.used_rows],
        )

    def can_fuse_attnres_partials(
        self,
        hidden_states: torch.Tensor,
        args: tuple,
    ) -> bool:
        blocks, weight_a, weight_b, eps, scratch_a, scratch_b = args
        return linear_attnres_partials_available(
            hidden_states,
            self.qkvgb_proj.weight,
            blocks,
            weight_a,
            weight_b,
            scratch_a,
            scratch_b,
            eps=eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: "ForwardContext",
        comm_manager,
        block_scale: torch.Tensor | None = None,
        attnres_partial_args: tuple | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states

        h = hidden_states
        num_tokens = h.shape[0]
        hn, hd = self.local_num_heads, self.head_dim
        # The hybrid backend re-splits key_dim/value_dim by attn_tp_size to
        # recover the per-rank head count, so pass the FULL (pre-TP) projection
        # width. The projected tensors (mixed_qkv, g_raw, beta) and the returned
        # core_out are already per-rank from their column-parallel layers, so the
        # output reshape below uses the per-rank head count ``hn``.
        proj = self.num_heads * hd

        # Raw (pre-conv) q/k/v projections concatenated; the hybrid backend runs
        # the short causal conv (+ SiLU) and manages the conv / recurrent state
        # cache (``KdaAttnBackend``). g_raw is the raw decay-gate
        # input, beta the per-head logits (sigmoid applied in-kernel).
        mixed_qkv, out_gate, f_a_out, beta = self._project_qkvfab(
            h, attnres_partial_args
        )
        # Prefill consumes compact QKV; perform this copy in the captured
        # projection segment rather than inside the eager attention break.
        if not ctx.forward_mode.is_decode():
            mixed_qkv = mixed_qkv.contiguous()
        # f_b runs inside the backend: fused into the decode scan kernel, a
        # plain GEMV on the prefill path.
        # Fused [3*proj, k] conv kernel bank, built once in post_load_weights.
        conv_weights = self.conv_weights
        fuse_decode_output_norm = ctx.forward_mode.is_decode() and num_tokens == ctx.bs

        core_out = ctx.attn_backend.forward(
            q=None,
            k=None,
            v=None,
            layer=None,
            token_to_kv_pool=ctx.token_to_kv_pool,
            forward_mode=ctx.forward_mode,
            bs=ctx.bs,
            mixed_qkv=mixed_qkv,
            conv_weights=conv_weights,
            bias=None,
            activation="silu",
            key_dim=proj,
            value_dim=proj,
            attention_tp_size=self.mapping.linear_attn.tp_size,
            head_k_dim=hd,
            head_v_dim=hd,
            f_a_out=f_a_out,
            f_b_weight=self.f_b_proj.weight,
            beta_raw=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            lower_bound=self.gate_lower_bound,
            output_gate=out_gate if fuse_decode_output_norm else None,
            norm_weight=self.o_norm.weight if fuse_decode_output_norm else None,
            norm_eps=self.o_norm.variance_epsilon if fuse_decode_output_norm else None,
            layer_id=self.layer_id,
            seq_len=num_tokens,
        )

        core_out = core_out.reshape(num_tokens, hn * hd)
        if not fuse_decode_output_norm:
            # Decode kernels may fuse this epilogue; prefill retains the shared
            # per-head norm implementation.
            core_out = rmsnorm_gated_sigmoid(
                core_out.contiguous(),
                out_gate,
                self.o_norm.weight,
                self.o_norm.variance_epsilon,
                hn,
                hd,
            )
        output, _ = self.o_proj(core_out)
        return output


class KimiLinearMoEGate(nn.Module):
    """Router for Kimi-K3 MoE: linear scorer + noaux_tc correction bias.

    Matches the checkpoint's ``block_sparse_moe.gate.{weight,e_score_correction_bias}``.
    Built inline (rather than reusing ``DeepseekV3.MoEGate``) because Kimi-K3's
    config uses ``num_experts`` where DeepSeek expects ``n_routed_experts``.
    """

    def __init__(self, hidden_size: int, num_experts: int) -> None:
        super().__init__()
        # Keep the checkpoint's BF16 weights at rest. Both the specialized AMD
        # decode GEMV and the portable fallback accumulate router logits in
        # FP32 before exact top-k selection.
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return kimi3_router_projection(
            hidden_states,
            self.weight,
        )


# Decode-shape ceiling for the split AttnRes fast paths (partial/combine and
# the scratch that links them). Decode steps carry at most --max-num-seqs
# tokens (1 token per sequence, no speculative decoding), so any batch above
# this is a prefill chunk and takes the unsplit path; the scratch buffers are
# allocated to exactly this many rows. Raise together with --max-num-seqs.
ATTNRES_FAST_PATH_MAX_TOKENS = 32
# Paired MI350 measurements show that stream scheduling costs more than the
# available attention/AttnRes overlap through M=16. Preserve NVIDIA's policy.
ATTNRES_STREAM_FORK_THRESHOLD = 16 if current_platform().is_amd else 0


def _attnres_mlp_slot(layer_id: int) -> int:
    """Slot (0 or 2) for this layer's mlp-side partial; alternates so a layer
    never reads the buffer its own aux branch is writing for the next one."""
    return 2 * (layer_id % 2)


_ATTNRES_SCRATCH: list | None = None


def _attnres_scratch(
    like: torch.Tensor, slot: int = 0, cap: int = ATTNRES_FAST_PATH_MAX_TOKENS
):
    """Shared (m, s, acc) scratch for the split attn_res mixing (bs <= cap).

    slot 1 = the next layer's attn-side mix; slots 0/2 = the mlp-side mix,
    ping-ponged on layer parity so the hoist needs no ordering edge.
    """
    global _ATTNRES_SCRATCH
    sc = _ATTNRES_SCRATCH
    if (
        sc is None
        or sc[0][2].shape[1] != like.shape[-1]
        or sc[0][2].device != like.device
    ):
        sc = [
            (
                torch.empty(cap, dtype=torch.float32, device=like.device),
                torch.empty(cap, dtype=torch.float32, device=like.device),
                torch.empty(
                    cap, like.shape[-1], dtype=torch.float32, device=like.device
                ),
            )
            for _ in range(3)
        ]
        _ATTNRES_SCRATCH = sc
    return sc[slot]


class KimiLinearMoE(nn.Module):
    """Kimi-K3 MoE block: sigmoid / noaux_tc router + Latent MoE + shared experts.

    Structure:

    * **Router** ``KimiLinearMoEGate`` + ``TopK`` (grouped, ``n_group=topk_group=1``,
      sigmoid scoring with ``e_score_correction_bias`` — DeepSeek-V3 noaux_tc).
    * **Latent MoE**: routed experts run at ``routed_expert_hidden_size`` (3584),
      so ``routed_expert_down_proj`` (7168->3584) feeds the experts and
      ``routed_expert_up_proj``/``routed_expert_norm`` project back (7168).
    * **Routed experts** (MXFP4): AMD uses the native ``MoELayer`` plan wrapped
      by ``LatentMoELayer`` so Triton/Gluon owns EP8 dispatch and SiTU. Non-AMD
      platforms use flashinfer's TRTLLM-Gen SiTU MoE. The selected MoE kernel
      advertises whether it consumes precomputed TopK or routes from logits.
    * **Shared experts**: a plain ``KimiLinearMLP`` (SiTU).
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        mapping: Mapping,
        layer_index: int,
        model_scope: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        # Router (gate+topk) and shared experts run on this stream during
        # graph capture, overlapped with the main-stream routed chain
        # (down_proj -> fused SiTU MoE -> up_proj). Collectives stay on the default
        # stream (aux-stream collectives can deadlock across ranks).
        self.stream_fork = StreamFork(alt_stream)
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.routed_scaling_factor = config.routed_scaling_factor
        # Latent MoE: routed experts run at routed_expert_hidden_size.
        self.routed_hidden = (
            config.routed_expert_hidden_size
            if config.routed_expert_hidden_size is not None
            else config.hidden_size
        )
        situ_beta, situ_linear_beta = _situ_betas(config)

        moe_backend = get_moe_backend()
        self.execution_plan = Kimi3MoEExecutionPlan.build(
            mapping,
            moe_backend,
            alt_stream,
            enforce_eager=bool(global_server_args_dict["enforce_eager"]),
        )
        # AUTO intentionally requests the flashinfer-backed SiTU plan when it was
        # registered at import time; AUTO cannot override MoELayer per model.
        plan = self.execution_plan
        if not plan.use_native and not plan.use_marlin:
            if not plan.use_trtllm:
                raise RuntimeError(
                    "Kimi-K3 MXFP4 SiTU MoE requires the native, FlashInfer "
                    "TRT-LLM, or Marlin (Hopper W4A16) backend; no portable SiTU "
                    f"Triton fallback exists (selected MoE backend: "
                    f"{moe_backend.value!r})."
                )
            # Fail here with the actual reason instead of letting MoELayer's
            # kernel selection miss the (never-registered) SiTU kernel.
            reason = situ_moe_unavailable_reason()
            if reason is not None:
                raise RuntimeError(
                    "Kimi-K3's fused SiTU MoE requires flashinfer > 0.6.15 "
                    f"with native SiTU, unavailable: {reason}. Upgrade "
                    "flashinfer; no portable SiTU Triton fallback exists."
                )
            # Out-of-box tactics: seed the autotuner from the in-tree
            # swept table for this GPU/flashinfer combo, if one ships
            # (flashinfer's own heuristic mispicks MoE tactics at prefill
            # batch sizes). A lookup miss leaves the startup autotune
            # window to tune these shapes.
            load_packaged_flashinfer_tuning_cache(
                "kimi-k3", mapping.moe.ep_size, mapping.moe.tp_size
            )
        # Attention DP partitions the batch when its TP group is smaller than
        # the MoE TP×EP group. Gather those DP shards so every rank enters the
        # K3 MoE with the same complete token batch.
        self._gather_dp_tokens_for_moe = mapping.attn.tp_size != mapping.moe.tp_ep_size
        if self._gather_dp_tokens_for_moe:
            if (
                mapping.attn.cp_size != 1
                or mapping.moe.dp_size != 1
                or mapping.moe.tp_ep_size != mapping.world_size
                or mapping.attn.tp_size * mapping.attn.dp_size != mapping.moe.tp_ep_size
            ):
                raise ValueError(
                    "Kimi-K3 cross-DP-EP MoE requires attn CP=1, MoE DP=1, "
                    "and attn TP*DP == MoE TP*EP == world size."
                )
        self.gate = KimiLinearMoEGate(config.hidden_size, config.num_experts)

        # Leave routing unconstrained: the registry prefers a kernel-routing
        # implementation when one exists and otherwise selects a precomputed-
        # TopK fallback (for example NVFP4, AMD-native, or Marlin SiTU).
        self.experts = MoELayer(
            top_k=self.top_k,
            num_experts=self.num_experts,
            hidden_size=self.routed_hidden,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            layer_index=layer_index,
            prefix=prefix,
            tp_rank=mapping.moe.tp_rank,
            tp_size=mapping.moe.tp_size,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
            activation="situ",
            activation_situ_beta=situ_beta,
            activation_situ_linear_beta=situ_linear_beta,
            routing_config={
                "n_group": config.num_expert_group,
                "topk_group": config.topk_group,
                "routed_scaling_factor": self.routed_scaling_factor,
                "normalize_topk_weights": config.moe_renormalize,
                "correction_bias": self.gate.e_score_correction_bias,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
                "activation_situ_beta": situ_beta,
                "activation_situ_linear_beta": situ_linear_beta,
            },
            routing_mode=None,
            # Native gfx950 and Hopper Marlin both run A16W4 (bf16 activations).
            # FlashInfer TRT-LLM SiTU depends on the expert weight dtype:
            # MXFP4 cubins are w4a8 (MXFP8 activations -> "fp8"), NVFP4 SiTU
            # runs w4a4 with the kernel wrapper quantizing the bf16 input
            # itself, which the registry models as "input".
            internal_activation_dtype_override=(
                "input"
                if self.execution_plan.use_native or self.execution_plan.use_marlin
                else (
                    _k3_trtllm_situ_internal_activation_dtype(quant_config, prefix)
                    if self.execution_plan.use_trtllm
                    else None
                )
            ),
        )

        # Derive the producer contract from the concrete registry selection;
        # backend family alone is too coarse (TRT-LLM has both kernel-routing
        # MXFP4 SiTU and precomputed-TopK NVFP4 SiTU implementations).
        self.topk = TopK(
            top_k=self.top_k,
            renormalize=config.moe_renormalize,
            use_grouped_topk=config.use_grouped_topk,
            num_expert_group=config.num_expert_group,
            num_fused_shared_experts=0,
            topk_group=config.topk_group,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            output_format=(
                TopKOutputFormat.STANDARD
                if self.experts.supports_precomputed_topk
                else self.experts.topk_output_format
            ),
            # bf16 weights out: makes precomputed TRT-LLM SiTU consume them
            # without a cast. This setting is unused by kernel-routing plans.
            topk_weights_dtype=(
                torch.bfloat16 if self.execution_plan.use_trtllm else torch.float32
            ),
        )

        self.routed_expert_down_proj = Kimi3LatentProjection(
            config.hidden_size,
            self.routed_hidden,
            prefix=add_prefix("routed_expert_down_proj", prefix),
        )
        # AMD keeps replicated weights because Iris cannot use the folded AR→GEMM→AR order.
        self._shard_up_projection = _shard_k3_up_projection(mapping, config.hidden_size)
        self.routed_expert_up_proj = Kimi3LatentProjection(
            self.routed_hidden,
            config.hidden_size,
            prefix=add_prefix("routed_expert_up_proj", prefix),
            shard_group=(
                mapping.moe.tp_ep_group if self._shard_up_projection else None
            ),
            shard_rank=mapping.moe.tp_ep_rank,
            shard_size=mapping.moe.tp_ep_size,
        )
        self.routed_expert_norm = (
            RMSNorm(self.routed_hidden, eps=config.rms_norm_eps)
            if config.latent_moe_use_norm
            else None
        )
        self.execution_plan = self.execution_plan.prepare_latent_fusion(
            mapping,
            lane_width=self.routed_hidden + config.hidden_size,
            has_latent_norm=self.routed_expert_norm is not None,
            max_token_num=max(
                int(global_server_args_dict["comm_fusion_max_num_tokens"]),
                1,
            ),
            shard_up_projection=self._shard_up_projection,
        )

        self._topk_ready = (
            torch.cuda.Event()
            if alt_stream is not None and self.experts.supports_precomputed_topk
            else None
        )

        # Shared experts (SiTU dense MLP over the full hidden size).
        self.shared_experts = KimiLinearMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("shared_experts", prefix),
            # Partial sums only: the reduce happens in forward on the default
            # stream after the aux-stream join, or in the joint Iris reduction.
            reduce_results=False,
            is_shared_expert=True,
            activation_situ_beta=situ_beta,
            activation_situ_linear_beta=situ_linear_beta,
        )
        self.packed_input_projection_weight: torch.Tensor | None = None
        # LatentMoELayer reduces routed partials across EP only. A TP-sharded
        # W2 also needs a TP reduction, which the graph-safe K3MoeTailComm path
        # below performs across the full TP x EP group.
        self.native_latent_moe = (
            LatentMoELayer(
                router=self.gate,
                topk=self.topk,
                routed_down_proj=self.routed_expert_down_proj,
                experts=self.experts,
                routed_norm=self.routed_expert_norm,
                routed_up_proj=self.routed_expert_up_proj,
                shared_experts=self.shared_experts,
                shared_reduce=(
                    None
                    if self.execution_plan.joint_moe_reduce
                    else self._reduce_shared
                ),
                joint_reduce=self.execution_plan.joint_moe_reduce,
                shared_expert_stream=(
                    alt_stream if self.execution_plan.overlap_shared_experts else None
                ),
                expert_parallel_group=mapping.moe.ep_group,
                input_projections=self._latent_input_projections,
            )
            if self.execution_plan.use_native and mapping.moe.tp_size == 1
            else None
        )
        # Dry-run exact registry selection for the fused decode pipeline.
        self._use_fused_decode_pipeline = (
            self.execution_plan.joint_moe_reduce
            and latent_moe_decode_pipeline_available(
                self.gate.weight,
                self.routed_expert_down_proj.weight,
                self.shared_experts.gate_up_proj.weight,
                self.shared_experts.down_proj.weight,
                self.experts.w13_weight,
                self.experts.w13_weight_scale,
                self.experts.w2_weight,
                self.experts.w2_weight_scale,
                self.experts.plan,
                topk=self.top_k,
            )
        )

        # Communication capability negotiation and tail routing live in
        # K3MoeTailComm (one MIN-vote per process; see kimi_k3_comm.py).
        self.comm = K3MoeTailComm(
            mapping=mapping,
            hidden_size=config.hidden_size,
            prefix=prefix,
            layer_index=layer_index,
            model_scope=model_scope,
            routed_hidden=self.routed_hidden,
            top_k=self.top_k,
            routed_norm=self.routed_expert_norm,
            up_proj=self.routed_expert_up_proj,
            execution_plan=self.execution_plan,
            # The experts kernel plan's capability bit (rank-uniform):
            # both SiTU variants register the deferred capability, while a
            # kernel without it (e.g. mxfp4 SwiGLU) arms the
            # materialized-input tail instead.
            # self.experts is constructed above, before this comm object.
            experts_supports_deferred_finalize=(
                self.experts.supports_deferred_finalize
            ),
        )

    def pack_input_projection_weights(self) -> None:
        """Back the router, routed-down, and shared gate/up weights with one tensor.

        The three projections consume the same activation and reduce over the
        same width, so one ``[experts + latent + 2 * shared, hidden]`` weight
        lets a single GEMM replace three. Each module keeps a contiguous row
        view of that tensor, so the rebinding adds no steady-state memory and
        leaves the separate composition working unchanged.
        """
        modules = (
            self.gate,
            self.routed_expert_down_proj,
            self.shared_experts.gate_up_proj,
        )
        # Held so the row views stay alive; deliberately not a registered
        # buffer, which would duplicate every projection in the state dict.
        # Any layout a packed GEMM cannot read is caught downstream by
        # ``packed_projection_weight_view``, leaving the composition in place.
        self.packed_input_projection_weight = torch.cat(
            [module.weight for module in modules], dim=0
        )
        offset = 0
        for module in modules:
            rows = module.weight.shape[0]
            module.weight.data = self.packed_input_projection_weight.narrow(
                0, offset, rows
            )
            offset += rows

    def _latent_input_projections(
        self,
        hidden_states: torch.Tensor,
        shared_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Project the router, routed latent, and shared partial in one pass.

        Returns ``None`` before the projection weights are concatenated, which
        leaves the caller on the separate per-module projections.
        """
        if self.packed_input_projection_weight is None:
            return None
        router_logits, routed_input, shared_input = latent_moe_input_projections(
            hidden_states,
            self.gate.weight,
            self.routed_expert_down_proj.weight,
            self.shared_experts.gate_up_proj.weight,
            gate_clamp=self.shared_experts.act_fn.beta,
            up_clamp=self.shared_experts.act_fn.linear_beta,
        )
        # The shared experts hold reduce_results=False, so this partial is
        # reduced by the layer's shared or joint reducer, not here.
        shared_output = kimi3_shared_down_projection(
            shared_input,
            self.shared_experts.down_proj.weight,
            out=shared_out,
        )
        return router_logits, routed_input, shared_output

    def _routed_experts(
        self,
        routed_in: torch.Tensor,
        topk_output: TopKOutput,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        do_finalize: bool = True,
    ) -> torch.Tensor:
        """Run the selected SiTU MoE (kernel-routing or precomputed-TopK)."""
        plan = self.execution_plan
        if not plan.use_native and not plan.use_trtllm and not plan.use_marlin:
            raise RuntimeError(
                "Kimi-K3 has no portable SiTU Triton fallback; use the native, "
                "FlashInfer TRT-LLM, or Marlin SiTU MoE path."
            )
        out = self.experts(
            hidden_states=routed_in,
            topk_output=topk_output,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            do_finalize=do_finalize,
        )
        # The kernel returns this rank's pre-reduce partial; the selected
        # tail tier owns the combining reduction.
        return out

    def _routing_output_format(self, ctx: ForwardContext | None) -> TopKOutputFormat:
        """Choose between the selected kernel's two routing entry points."""
        if not (
            self.execution_plan.use_trtllm
            and self.experts.support_routing
            and self.experts.supports_precomputed_topk
        ):
            return self.experts.topk_output_format
        # Keep the decode-optimized precomputed path. Extend/mixed forwards use
        # FlashInfer's public routing API, which wins for prefill-sized batches.
        # Cross-DP-EP gathers the same global batch onto every MoE rank, so use
        # the replicated phase decision rather than a rank-local IDLE mode.
        if ctx is None:
            return TopKOutputFormat.STANDARD
        is_decode = (
            ctx.all_decode_or_idle
            if self._gather_dp_tokens_for_moe
            else ctx.forward_mode.is_decode()
        )
        if is_decode:
            return TopKOutputFormat.STANDARD
        return TopKOutputFormat.BYPASSED

    def _forward_fused_decode_pipeline(
        self,
        hidden_states: torch.Tensor,
        prefix_sum: torch.Tensor,
    ) -> torch.Tensor:
        """Run the fused multi-launch K3 decode pipeline.

        The input-projection operation produces router logits, the routed
        latent, and the activated shared input from one packed weight. Routed
        W13 and SiTU run separately; the next launch computes routed W2 beside
        the shared down projection. One collective reduces both partials. The
        final routed up projection adds ``prefix_sum`` and the shared output in
        its epilogue when a specialized implementation is available. The
        optional routed norm folds into that final projection; top-k remains
        a separate launch.
        """

        router_logits, routed_input, shared_input = latent_moe_input_projections(
            hidden_states,
            self.gate.weight,
            self.routed_expert_down_proj.weight,
            self.shared_experts.gate_up_proj.weight,
            gate_clamp=self.shared_experts.act_fn.beta,
            up_clamp=self.shared_experts.act_fn.linear_beta,
        )
        topk_output = self.topk(hidden_states, router_logits)
        routed_latent, shared_output = latent_moe_expert_shared_all_reduce(
            routed_input,
            self.experts.w13_weight,
            self.experts.w13_weight_scale,
            self.experts.w2_weight,
            self.experts.w2_weight_scale,
            topk_output.topk_weights,
            topk_output.topk_ids,
            shared_input,
            self.shared_experts.down_proj.weight,
            activation_clamp=float(self.experts.activation_situ_beta),
            linear_clamp=self.experts.activation_situ_linear_beta,
            expert_start=self.experts.ep_rank * self.experts.num_local_experts,
            w13_interleaved=self.experts.w13_input_layout == "interleaved",
            group=self.mapping.moe.ep_group,
        )
        return self.native_latent_moe.finalize_output(
            routed_latent,
            prefix_sum,
            shared_output,
        )

    def _gather_dp_tokens(
        self,
        hidden_states: torch.Tensor,
        prefix_sum: torch.Tensor,
        ctx: ForwardContext,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Gather DP batches before the temporary cross-DP-EP MoE path.

        Attention-TP replicas hold identical rows. Gathering on the matching
        DP group collects each DP batch exactly once and leaves every MoE rank
        with the complete global batch for the existing expert all-reduce.
        """
        global_counts = (
            ctx.collective_global_num_tokens
            if ctx.collective_global_num_tokens is not None
            else ctx.global_num_tokens
        )
        if global_counts is None:
            raise RuntimeError(
                "Kimi-K3 cross-DP-EP MoE requires global_num_tokens for token-gather sizing."
            )
        dp_counts = [
            global_counts[
                dp_rank * self.mapping.attn.tp_size * self.mapping.attn.cp_size
            ]
            for dp_rank in range(self.mapping.attn.dp_size)
        ]
        local_count = dp_counts[self.mapping.attn.dp_rank]
        if hidden_states.shape[0] != local_count or prefix_sum.shape[0] != local_count:
            raise RuntimeError(
                "Kimi-K3 cross-DP-EP MoE rows do not match global_num_tokens."
            )
        return (
            token_all_gather(
                hidden_states,
                group=self.mapping.attn.dp_group,
                scattered_num_tokens=dp_counts,
            ),
            token_all_gather(
                prefix_sum,
                group=self.mapping.attn.dp_group,
                scattered_num_tokens=dp_counts,
            ),
            sum(dp_counts),
            sum(dp_counts[: self.mapping.attn.dp_rank]),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prefix_sum: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        ctx: ForwardContext | None = None,
    ) -> torch.Tensor:
        """Routed + shared experts, accumulated onto ``prefix_sum``.

        Returns the new prefix (``prefix_sum + routed + shared``); the tail
        tiers fuse the accumulate in-kernel.
        """
        local_num_tokens = hidden_states.shape[0]
        local_offset = 0
        if self._gather_dp_tokens_for_moe:
            if ctx is None:
                raise RuntimeError("Kimi-K3 cross-DP-EP MoE requires a ForwardContext.")
            hidden_states, prefix_sum, num_global_tokens, local_offset = (
                self._gather_dp_tokens(hidden_states, prefix_sum, ctx)
            )
            max_num_tokens_per_gpu = num_global_tokens

        if self.native_latent_moe is not None:
            if self._use_fused_decode_pipeline and 0 < hidden_states.shape[0] <= 4:
                output = self._forward_fused_decode_pipeline(hidden_states, prefix_sum)
            else:
                output = self.native_latent_moe(
                    hidden_states,
                    num_global_tokens=num_global_tokens,
                    max_num_tokens_per_gpu=max_num_tokens_per_gpu,
                    prefix_sum=prefix_sum,
                )
            return output[local_offset : local_offset + local_num_tokens]

        num_tokens, hidden_size = hidden_states.shape
        if num_tokens == 0:
            return prefix_sum

        routing_output_format = self._routing_output_format(ctx)
        precompute_topk = routing_output_format.is_standard()
        plan = self.comm.plan(
            num_tokens,
            hidden_states,
            # Rank-uniform by construction: DP-EP gather replicates the phase.
            is_decode=(
                ctx is not None
                and (
                    ctx.all_decode_or_idle
                    if self._gather_dp_tokens_for_moe
                    else ctx.forward_mode.is_decode()
                )
            ),
        )
        # Producer-direct destinations for the routed and shared partials. The
        # symmetric pair is preferred (the tail reduces it in place); the packed
        # lane is the fallback, and its two halves are slices of one buffer.
        if plan.symm_outputs is not None:
            routed_out_buf, shared_out_buf = plan.symm_outputs
        elif plan.lane is not None:
            routed_out_buf = plan.lane[:, : self.routed_hidden]
            shared_out_buf = plan.lane[:, self.routed_hidden :]
        else:
            routed_out_buf = shared_out_buf = None
        self.experts._situ_output_buffer = routed_out_buf

        # The router, routed latent and shared gate/up read the same activation
        # and reduce over the same width, so one GEMM replaces three. Returns
        # None when the packed weight is unavailable or the shapes are outside
        # the fused kernel, which leaves the separate projections below.
        fused_inputs = self._latent_input_projections(
            hidden_states, shared_out=shared_out_buf
        )
        if fused_inputs is not None:
            router_logits, routed_in, shared_partial = fused_inputs
        else:
            # Router runs uncontended on main (3us; on aux it starves to 14us
            # under concurrent GEMMs). When the selected experts need
            # precomputed TopK runs on the fork branch beside down_proj;
            # routing bypasses it.
            router_logits = self.gate(hidden_states)
            routed_in = shared_partial = None

        prepared_shared_shard = None
        # Enable the fork for the whole graph phase, but only overlap during
        # capture. The pre-capture warmup runs with capture mode off, so gating
        # ``enable`` on it left the auxiliary stream completely untouched until
        # capture itself -- the first hipBLASLt call on that stream then did its
        # lazy handle/workspace setup inside the capturing stream and raised
        # "operation not permitted when stream is capturing" (900) on every
        # rank, deadlocking startup. Enabling on the graph phase makes warmup
        # execute the same branches on the auxiliary stream, serially
        # (``overlap=False``), so every backend is initialized before capture.
        # This matches the contract _capture_one documents and the pattern the
        # other fork-using models already follow.
        with self.stream_fork.scope(
            enable=get_is_cuda_graph_phase(),
            overlap=get_is_capture_mode(),
        ) as fork:
            with fork.branch():
                topk_output = self.topk(
                    hidden_states,
                    router_logits,
                    output_format=routing_output_format,
                )
                if self._topk_ready is not None and precompute_topk and fork._active:
                    self._topk_ready.record(torch.cuda.current_stream())
                if shared_partial is None:
                    shared_partial = self.shared_experts(
                        hidden_states,
                        down_out=shared_out_buf,
                    )
                if plan.split_shared_rs and fork._active:
                    prepared_shared_shard = self.comm.reduce_scatter_shared(
                        shared_partial
                    )
            if routed_in is None:
                routed_in, _ = self.routed_expert_down_proj(hidden_states)
            if self._topk_ready is not None and precompute_topk and fork._active:
                self._topk_ready.wait(torch.cuda.current_stream())
            routed_partial = self._routed_experts(
                routed_in,
                topk_output,
                num_global_tokens,
                max_num_tokens_per_gpu,
                do_finalize=not plan.defer_finalize,
            )
            if plan.routed_in_fork:
                # No fused collective to hide behind: reduce and project here
                # so the work overlaps the shared branch inside the fork.
                routed_tail_input = self.comm.reduce_project_routed(routed_partial)
            else:
                routed_tail_input = routed_partial
        output = self.comm.run(
            plan,
            routed_tail_input,
            shared_partial,
            prefix_sum,
            num_tokens,
            hidden_size,
            prepared_shared_shard,
        )
        # Cross-DP-EP: the tail ran on the gathered token set; every DP rank
        # keeps only its slice (a no-op view when no gather happened).
        return output[local_offset : local_offset + local_num_tokens]

    def _reduce_shared(self, shared_partial: torch.Tensor) -> torch.Tensor:
        """Reduce the shared experts' TP partial (delegates to K3MoeTailComm)."""
        return self.comm.reduce_shared(shared_partial)


class KimiLinearDecoderLayer(nn.Module):
    """Kimi-K3 decoder layer: KDA/MLA dispatch + dense/MoE FFN + AttnRes.

    One class for both layer types — the AttnRes data flow is identical, only
    ``self_attn`` differs (dispatched by ``config.is_kda_layer(layer_id)``). The
    AttnRes path replaces the plain pre-norm residual with a
    learned block-residual mixing (``_apply_attn_res``) and runs *outside*
    ``CommManager`` fusion: attention/FFN output projections all-reduce in place
    (``reduce_results=True``) and the residual is threaded explicitly as the
    per-token ``block_residual`` buffer.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        mapping: Mapping,
        layer_id: int,
        model_scope: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_id = layer_id

        # --- attention: KDA (linear) or NoPE-MLA (full); both under "self_attn" ---
        attn_prefix = add_prefix("self_attn", prefix)
        if config.is_kda_layer(layer_id):
            self.self_attn = KimiLinearKDA(
                config, mapping, layer_id, quant_config, attn_prefix
            )
        else:
            self.self_attn = KimiLinearMLAAttention(
                config=config,
                mapping=mapping,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                max_position_embeddings=config.max_position_embeddings,
                quant_config=quant_config,
                layer_id=layer_id,
                prefix=attn_prefix,
                reduce_attn_results=False,  # layer fused AR+residual reduces
                alt_stream=alt_stream,
            )

        # --- FFN: dense MLP (first_k_dense_replace) or MoE block ---
        self.is_moe_layer = (
            config.num_experts is not None
            and layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        )
        situ_beta, situ_linear_beta = _situ_betas(config)
        if self.is_moe_layer:
            # Named for the checkpoint index; not aliased as self.mlp (double
            # registration would duplicate every MoE param in state_dict).
            self.block_sparse_moe = KimiLinearMoE(
                config=config,
                mapping=mapping,
                layer_index=layer_id,
                model_scope=model_scope,
                quant_config=quant_config,
                prefix=add_prefix("block_sparse_moe", prefix),
                alt_stream=alt_stream,
            )
        else:
            self.mlp = KimiLinearMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                mapping=mapping,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                reduce_results=True,
                activation_situ_beta=situ_beta,
                activation_situ_linear_beta=situ_linear_beta,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # --- AttnRes modules ---
        block = config.attn_res_block_size
        self.is_block_write_layer = layer_id % block == 0
        self.block_write_idx = layer_id // block
        self.prev_valid_blocks = ceil_div(layer_id, block)
        self.self_attention_res_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention_res_proj = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            prefix=add_prefix("self_attention_res_proj", prefix),
        )
        self.mlp_res_proj = ReplicatedLinear(
            config.hidden_size, 1, bias=False, prefix=add_prefix("mlp_res_proj", prefix)
        )

        # K3 AttnRes bypasses CommManager's fused residual, but the MLA attention
        # still uses it for the (no-op in AllReduce mode) pre_attn_comm.
        # AR+residual fusion arming and the dummy-norm ride live in
        # K3AttnCommState (armed once per process).
        self.k3_comm = K3AttnComm(
            K3AttnCommState.get(mapping=mapping, hidden_size=config.hidden_size)
        )

        self.attn_fork = StreamFork(alt_stream)
        # (proj_w_getter, norm, valid_blocks) for the NEXT layer's attn-side
        # mix; set by the backbone after all layers exist. The partial launches
        # from this (MoE) layer's aux branch, hidden under the routed experts.
        self._next_attn_mix = None
        # Whether the next layer's mlp-side partial rides our sweep too.
        self._hoist_next_mlp = False
        self._mlp_slot = _attnres_mlp_slot(layer_id)
        # Precomputed rms_w * res_w products (filled in post_load_weights).
        self._attn_wp = None
        self._mlp_wp = None
        # True when the PREVIOUS layer precomputes our attn-side block partial.
        self._attn_split = False
        # True when the PREVIOUS layer precomputes our mlp-side block partial.
        self._mlp_split = False
        self._dflash_attnres_capture_fallback = False
        # True when the NEXT layer folds our routed+shared residual accumulate
        # into its attn-side combine (we return the parts unsummed).
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=self.is_moe_layer,
            prev_is_moe=False,
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
        )

    def _reduce_attn_accumulate(
        self,
        attn_partial: torch.Tensor,
        prefix_sum: torch.Tensor | None,
        combine: tuple | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """All-reduce the attention partial and accumulate the residual.

        Delegates to :meth:`K3AttnComm.attn_reduce`; see kimi_k3_comm.py for
        the branch semantics (AttnRes-combine epilogue / fused AR+residual /
        B1 combine / plain reduce).
        """
        return self.k3_comm.attn_reduce(
            attn_partial, prefix_sum, combine, mlp_wp=self._mlp_wp
        )

    def _mix_into_attention(
        self, hidden_states: torch.Tensor, block_residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """AttnRes entry: mix the residual candidates into the attention input.

        Returns ``(h, prefix_sum)`` -- with ``prefix_sum`` None at block-write
        layers (the snapshot consumed it).
        """
        prefix_sum = hidden_states
        n_tok = prefix_sum.shape[0]
        fast_mix = (
            self._attn_split
            and 0 < n_tok <= ATTNRES_FAST_PATH_MAX_TOKENS
            and prefix_sum.is_cuda
            and self.prev_valid_blocks > 0
        )
        if fast_mix:
            # The block partial was precomputed on the previous layer's aux
            # stream; only the prefix candidate is folded here.
            h = attnres_combine(
                prefix_sum,
                self._attn_wp,
                self.input_layernorm.weight,
                self.self_attention_res_norm.variance_epsilon,
                _sliced_scratch(prefix_sum, 1, n_tok),
                torch.empty_like(prefix_sum),
            )
        else:
            h = _apply_attn_res(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                self.prev_valid_blocks,
                out_norm=self.input_layernorm,
            )
        if self.is_block_write_layer:
            block_residual[self.block_write_idx] = prefix_sum  # snapshot
            prefix_sum = None
        return h, prefix_sum

    def _fused_attnres_graph_available(
        self, hidden_states: torch.Tensor, block_residual: torch.Tensor
    ) -> bool:
        # Split beats fused at decode: 47 us/step faster at bs = 8 (aux-stream partial).
        if getattr(self, "_dflash_attnres_capture_fallback", False):
            return False

        # B1 already fuses the post-attention mix into the all-reduce.
        if hidden_states.shape[0] == 1:
            return False

        block_write_idx = self.block_write_idx if self.is_block_write_layer else -1
        pre_attn = attn_res_fwd_available(
            hidden_states,
            block_residual,
            self.self_attention_res_proj.weight.reshape(-1),
            self.self_attention_res_norm.weight,
            self.self_attention_res_norm.variance_epsilon,
            out_norm_weight=self.input_layernorm.weight,
            out_norm_eps=self.input_layernorm.variance_epsilon,
            num_valid_blocks=self.prev_valid_blocks,
            block_write_idx=block_write_idx,
        )
        if not pre_attn:
            return False

        mlp_valid_blocks = self.prev_valid_blocks + int(self.is_block_write_layer)
        return attn_res_fwd_available(
            hidden_states,
            block_residual,
            self.mlp_res_proj.weight.reshape(-1),
            self.mlp_res_norm.weight,
            self.mlp_res_norm.variance_epsilon,
            out_norm_weight=self.post_attention_layernorm.weight,
            out_norm_eps=self.post_attention_layernorm.variance_epsilon,
            delta=None if self.is_block_write_layer else hidden_states,
            num_valid_blocks=mlp_valid_blocks,
        )

    def _forward_fused_attnres_graph(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: "ForwardContext",
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one AttnRes launch on each side of the attention collective."""
        prefix_sum = hidden_states
        h = _apply_attn_res(
            prefix_sum,
            block_residual,
            self.self_attention_res_proj,
            self.self_attention_res_norm,
            self.prev_valid_blocks,
            out_norm=self.input_layernorm,
            block_write_idx=(self.block_write_idx if self.is_block_write_layer else -1),
        )

        attn_partial = self.self_attn(
            positions=positions,
            hidden_states=h,
            ctx=ctx,
            comm_manager=self.comm_manager,
            attnres_partial_args=None,
        )
        reduced = all_reduce(attn_partial, self.mapping.attn.tp_group)

        if self.is_block_write_layer:
            prefix_sum = reduced
            delta = None
        else:
            delta = reduced
        h = _apply_attn_res(
            prefix_sum,
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
            self.prev_valid_blocks + int(self.is_block_write_layer),
            out_norm=self.post_attention_layernorm,
            delta=delta,
        )

        # Before the MoE: the next layer's PDL combine prefetches this scratch early.
        self._prepare_next_fallback_attnres_partial(prefix_sum, block_residual)
        if self.is_moe_layer:
            num_global_tokens, max_num_tokens_per_gpu = (
                self.comm_manager.get_num_tokens(ctx)
            )
            prefix_sum = self.block_sparse_moe(
                h,
                prefix_sum,
                num_global_tokens=num_global_tokens,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu,
                # ctx is required by the cross-DP-EP token gather (was missing).
                ctx=ctx,
            )
        else:
            prefix_sum = prefix_sum + self.mlp(h)
        return prefix_sum, block_residual

    def _prepare_next_fallback_attnres_partial(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> None:
        """Bridge a fused layer to the existing split AttnRes fallback."""
        num_tokens = hidden_states.shape[0]
        if (
            self._next_attn_mix is None
            or not hidden_states.is_cuda
            or not 0 < num_tokens <= ATTNRES_FAST_PATH_MAX_TOKENS
        ):
            return

        next_layer, valid_blocks = self._next_attn_mix
        if next_layer._fused_attnres_graph_available(hidden_states, block_residual):
            return
        attn_scratch = _sliced_scratch(hidden_states, 1, num_tokens)
        if self._hoist_next_mlp:
            attnres_partial_dual(
                block_residual[:valid_blocks],
                next_layer._mlp_wp,
                next_layer._attn_wp,
                next_layer.mlp_res_norm.variance_epsilon,
                _sliced_scratch(hidden_states, next_layer._mlp_slot, num_tokens),
                attn_scratch,
            )
        else:
            attnres_partial(
                block_residual[:valid_blocks],
                next_layer._attn_wp,
                next_layer.self_attention_res_norm.variance_epsilon,
                attn_scratch,
            )

    @torch.no_grad()
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: "ForwardContext",
        block_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._fused_attnres_graph_available(hidden_states, block_residual):
            return self._forward_fused_attnres_graph(
                positions,
                hidden_states,
                ctx,
                block_residual,
            )

        h, prefix_sum = self._mix_into_attention(hidden_states, block_residual)
        # The mlp-side mixing's block partial hides under attention on the aux
        # stream (blocks are final for this layer once the snapshot above ran);
        # the combine after the attention AR only touches the prefix candidate.
        mlp_valid_blocks = self.prev_valid_blocks + (
            1 if self.is_block_write_layer else 0
        )
        num_tokens = h.shape[0]
        split_mix = (
            0 < num_tokens <= ATTNRES_FAST_PATH_MAX_TOKENS
            and h.is_cuda
            and mlp_valid_blocks > 0
        )
        scratch = _sliced_scratch(h, self._mlp_slot, num_tokens) if split_mix else None
        # Block-write layers only: theirs cannot ride the previous sweep.
        own_mlp = split_mix and not self._mlp_split
        next_mix = self._next_attn_mix if split_mix else None
        sc1 = _sliced_scratch(h, 1, num_tokens) if next_mix is not None else None
        # The mlp-side combine (blocks partial + post-AR prefix) rides the
        # attention AR epilogue on the fused path.
        ar_combine = (
            (
                scratch,
                self.mlp_res_proj.weight.reshape(-1),
                self.mlp_res_norm.weight,
                self.post_attention_layernorm.weight,
                self.mlp_res_norm.variance_epsilon,
            )
            if split_mix
            else None
        )
        attnres_partial_args = None
        if next_mix is not None and self._hoist_next_mlp:
            next_layer, _ = next_mix
            # This layer's attention projection writes both block partials
            # hoisted for the next layer, which consumes their scratch slots.
            # Kernel availability below is the token-count capability gate.
            candidate_args = (
                block_residual[:mlp_valid_blocks],
                next_layer._mlp_wp,
                next_layer._attn_wp,
                self.mlp_res_norm.variance_epsilon,
                _sliced_scratch(h, next_layer._mlp_slot, num_tokens),
                sc1,
            )
            if self.self_attn.can_fuse_attnres_partials(h, candidate_args):
                attnres_partial_args = candidate_args
        reduce_consumes_scratch = (
            own_mlp
            and ar_combine is not None
            and prefix_sum is not None
            and (
                num_tokens == 1
                or (
                    self.k3_comm.state.attn_ar_fusion_ok
                    and num_tokens
                    <= global_server_args_dict["comm_fusion_max_num_tokens"]
                )
                or (
                    # The gfx950 fused reducer consumes the AttnRes scratch
                    # through M=16, so its producer stream must join first.
                    current_platform().is_cdna4
                    and num_tokens <= ATTNRES_STREAM_FORK_THRESHOLD
                )
            )
        )
        with self.attn_fork.scope(
            enable=(
                get_is_capture_mode()
                and num_tokens > ATTNRES_STREAM_FORK_THRESHOLD
                and (attnres_partial_args is None or own_mlp)
            )
        ) as fork:
            with fork.branch():
                if next_mix is not None:
                    next_layer, _ = next_mix
                    if self._hoist_next_mlp:
                        if attnres_partial_args is None:
                            attnres_partial_dual(
                                block_residual[:mlp_valid_blocks],
                                next_layer._mlp_wp,
                                next_layer._attn_wp,
                                self.mlp_res_norm.variance_epsilon,
                                _sliced_scratch(h, next_layer._mlp_slot, num_tokens),
                                sc1,
                            )
                    else:
                        attnres_partial(
                            block_residual[:mlp_valid_blocks],
                            next_layer._attn_wp,
                            self.mlp_res_norm.variance_epsilon,
                            sc1,
                        )
                if own_mlp:
                    attnres_partial(
                        block_residual[:mlp_valid_blocks],
                        self._mlp_wp,
                        self.mlp_res_norm.variance_epsilon,
                        scratch,
                    )
            attn_out = self.self_attn(
                positions=positions,
                hidden_states=h,
                ctx=ctx,
                comm_manager=self.comm_manager,
                attnres_partial_args=attnres_partial_args,
            )
            if not reduce_consumes_scratch:
                prefix_sum, h_fused = self._reduce_attn_accumulate(
                    attn_out, prefix_sum, combine=ar_combine
                )
        if reduce_consumes_scratch:
            prefix_sum, h_fused = self._reduce_attn_accumulate(
                attn_out, prefix_sum, combine=ar_combine
            )
        # --- mlp: AttnRes mixing -> norm -> FFN -> accumulate ---
        if h_fused is not None:
            h = h_fused
        elif split_mix:
            h = attnres_combine(
                prefix_sum,
                self._mlp_wp,
                self.post_attention_layernorm.weight,
                self.mlp_res_norm.variance_epsilon,
                scratch,
                torch.empty_like(prefix_sum),
            )
        else:
            h = _apply_attn_res(
                prefix_sum,
                block_residual,
                self.mlp_res_proj,
                self.mlp_res_norm,
                mlp_valid_blocks,
                out_norm=self.post_attention_layernorm,
            )
        if self.is_moe_layer:
            num_global_tokens, max_num_tokens_per_gpu = (
                self.comm_manager.get_num_tokens(ctx)
            )
            prefix_sum = self.block_sparse_moe(
                h,
                prefix_sum,
                num_global_tokens=num_global_tokens,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu,
                ctx=ctx,
            )
        else:
            prefix_sum = prefix_sum + self.mlp(h)
        return prefix_sum, block_residual


# ===----------------------------------------------------------------------=== #
# Text backbone (KimiLinear)
# ===----------------------------------------------------------------------=== #


class KimiLinearModel(nn.Module):
    """Kimi-K3 text transformer: embedding + hybrid decoder layers + AttnRes.

    Runs the block-level attention-residual (AttnRes) data flow:
    a per-token ``block_residual`` buffer is threaded through the layers, and a
    final ``_apply_attn_res`` mixes the accumulated stream against the block
    snapshots before the output norm. The per-layer ``self_attn`` dispatch is
    ``KimiLinearDecoderLayer``'s job.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config

        alt_stream = (
            torch.cuda.Stream(priority=-1) if torch.cuda.is_available() else None
        )

        # Pipeline stage layer window: [pp_start_layer, pp_end_layer). Global
        # layer numbering everywhere; other stages' slots hold PPMissingLayer.
        self.pp_start_layer, self.pp_end_layer = pp_layer_window(
            config.num_hidden_layers, mapping
        )
        if mapping.is_first_pp_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                tp_rank=mapping.attn.tp_rank,
                tp_size=mapping.attn.tp_size,
                tp_group=mapping.attn.tp_group,
            )
        else:
            self.embed_tokens = None

        layers_scope = add_prefix("layers", prefix)

        def get_layer(idx: int, prefix: str):
            return KimiLinearDecoderLayer(
                config=config,
                mapping=mapping,
                layer_id=idx,
                model_scope=layers_scope,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            )

        self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=layers_scope,
            pp_start_layer=self.pp_start_layer,
            pp_end_layer=self.pp_end_layer,
        )
        # Cross-layer attn-side mix precompute: a layer's aux stream computes
        # the NEXT layer's block partial alongside its own mlp-side partial
        # (one dual sweep under attention; blocks are final by then). Under PP
        # the pair must live on the same stage — the stage boundary severs the
        # dual sweep there, and the downstream stage's first layer falls back
        # to its own (non-split) AttnRes mixing.
        for i in range(self.pp_start_layer, self.pp_end_layer - 1):
            cur, nxt = self.layers[i], self.layers[i + 1]
            if nxt.prev_valid_blocks > 0:
                assert (
                    cur.mlp_res_norm.variance_epsilon
                    == nxt.self_attention_res_norm.variance_epsilon
                    == nxt.mlp_res_norm.variance_epsilon
                ), "dual partial assumes one shared RMS epsilon"
                cur._next_attn_mix = (nxt, nxt.prev_valid_blocks)
                nxt._attn_split = True
                # A block-write layer's own block does not exist a layer earlier.
                cur._hoist_next_mlp = not nxt.is_block_write_layer
                nxt._mlp_split = cur._hoist_next_mlp

        if mapping.is_last_pp_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            # Model-level AttnRes output mixing.
            self.output_attn_res_norm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.output_attn_res_proj = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                prefix=add_prefix("output_attn_res_proj", prefix),
            )
        else:
            self.norm = None
            self.output_attn_res_norm = None
            self.output_attn_res_proj = None
        # One-based completed-layer ids; see set_eagle3_layers_to_capture.
        self.eagle3_layers_to_capture: tuple[int, ...] = ()

        # DFLASH/DSpark speculative decoding: layer indices whose *output*
        # stream is captured for the draft. Populated by
        # ``set_dflash_layers_to_capture``; empty means no capture.
        self.layers_to_capture: list[int] = []
        self.dflash_aux_stream: str = "prefix"
        self._dflash_incremental_callback = None
        self._dflash_slot_bufs = None
        self._dflash_capture_idx_map: dict[int, int] = {}

    def _refresh_dflash_capture_fallback(self) -> None:
        """Mark AttnRes consumers that need split execution."""
        captured = set(self.layers_to_capture)
        use_fallback = self.dflash_aux_stream == "attn_res"
        for layer_idx, layer in enumerate(self.layers):
            layer._dflash_attnres_capture_fallback = use_fallback and (
                layer_idx - 1 in captured
            )

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def pp_stage_state_spec(
        self, num_tokens: int, device: torch.device
    ) -> list[tuple[str, tuple[int, ...], torch.dtype]]:
        """Wire spec (name, shape, dtype) of the inter-stage boundary bundle.

        K3's boundary state is the accumulated ``prefix_sum`` stream plus the
        valid prefix of the AttnRes ``block_residual`` snapshot buffer. The
        upstream stage ships ``ceil_div(its pp_end_layer, block)`` snapshot
        rows and this stage expects ``ceil_div(pp_start_layer, block)`` — the
        two are the same number because the windows abut, so no shape
        metadata crosses the wire.
        """
        del device
        hidden = self.config.hidden_size
        valid_blocks = ceil_div(self.pp_start_layer, self.config.attn_res_block_size)
        # The activation dtype. get_default_dtype() would be wrong here (the
        # model-dtype context only wraps weight loading); take it from any
        # floating live parameter — mid-stage ranks have no embed/norm.
        dtype = None
        for param in self.parameters():
            if param.is_floating_point() and param.dtype in (
                torch.bfloat16,
                torch.float16,
            ):
                dtype = param.dtype
                break
        if dtype is None:
            dtype = torch.bfloat16
        return [
            ("hidden_states", (num_tokens, hidden), dtype),
            ("block_residual", (valid_blocks, num_tokens, hidden), dtype),
        ]

    def _dspark_capture_stream(
        self,
        layer_idx: int,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
    ) -> torch.Tensor:
        """Capture the configured target stream after layer_idx."""
        if self.dflash_aux_stream != "attn_res":
            return prefix_sum.clone()

        if layer_idx + 1 < len(self.layers):
            consumer = self.layers[layer_idx + 1]
            proj = consumer.self_attention_res_proj
            norm = consumer.self_attention_res_norm
            num_blocks = consumer.prev_valid_blocks
        else:
            proj = self.output_attn_res_proj
            norm = self.output_attn_res_norm
            num_blocks = ceil_div(
                self.config.num_hidden_layers, self.config.attn_res_block_size
            )

        mixed = _apply_attn_res(prefix_sum, block_residual, proj, norm, num_blocks)
        return prefix_sum.clone() if mixed is prefix_sum else mixed

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: "ForwardContext",
        input_embeds: torch.Tensor | None = None,
        pp_inbound: PPStageState | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor | PPStageState, list | None]:
        if pp_inbound is not None:
            # Mid-pipeline stage: resume the AttnRes stream received from the
            # upstream stage instead of embedding.
            hidden_states = pp_inbound.hidden_states
        elif input_embeds is not None:
            hidden_states = input_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        # Per-forward AttnRes scratch, block-major so block_residual[:m] is a
        # contiguous kernel slice (fresh alloc = CUDA-graph safe); new_empty is
        # safe: slot j is written at layer j*block_size before any read.
        num_blocks = ceil_div(
            self.config.num_hidden_layers, self.config.attn_res_block_size
        )
        block_residual = hidden_states.new_empty(
            num_blocks, hidden_states.size(0), hidden_states.size(1)
        )
        if pp_inbound is not None:
            # Seed the upstream stage's snapshot rows; this stage's own
            # block-write layers fill the rest.
            inbound_blocks = pp_inbound.block_residual
            block_residual[: inbound_blocks.size(0)].copy_(inbound_blocks)

        capture_layers = self.layers_to_capture
        capture_dflash = bool(capture_layers)
        capture_eagle3 = bool(self.eagle3_layers_to_capture)
        aux_hidden_states: list[torch.Tensor] | None = (
            [] if capture_dflash or capture_eagle3 else None
        )

        prefix_sum = hidden_states
        for layer_idx in range(self.pp_start_layer, self.pp_end_layer):
            layer = self.layers[layer_idx]
            prefix_sum, block_residual = layer(
                positions, prefix_sum, ctx, block_residual
            )
            if capture_dflash and layer_idx in capture_layers:
                captured = self._dspark_capture_stream(
                    layer_idx, prefix_sum, block_residual
                )
                capture_idx = self._dflash_capture_idx_map.get(layer_idx)
                if self._dflash_slot_bufs is not None and capture_idx is not None:
                    num_tokens = captured.shape[0]
                    self._dflash_slot_bufs[capture_idx][:num_tokens].copy_(captured)
                    if self._dflash_incremental_callback is not None:
                        self._dflash_incremental_callback(capture_idx, num_tokens)
                assert aux_hidden_states is not None
                aux_hidden_states.append(captured)
            # Clone: the copy must survive the next layer's in-place residual writes.
            elif capture_eagle3 and layer_idx + 1 in self.eagle3_layers_to_capture:
                assert aux_hidden_states is not None
                aux_hidden_states.append(prefix_sum.clone())

        if not self.mapping.is_last_pp_rank:
            # Hand the AttnRes thread state to the next stage: the prefix sum
            # plus the snapshot rows written so far. The wire count matches
            # the downstream spec because the windows abut on the same block
            # arithmetic (ceil_div of the shared boundary layer id).
            valid_blocks = ceil_div(self.pp_end_layer, self.config.attn_res_block_size)
            return (
                PPStageState(
                    hidden_states=prefix_sum,
                    block_residual=block_residual[:valid_blocks],
                ),
                None,
            )

        hidden_states = _apply_attn_res(
            prefix_sum,
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            num_blocks,
            out_norm=self.norm,
        )
        return hidden_states, aux_hidden_states


class KimiLinearForCausalLM(BaseCausalLM):
    """Kimi-K3 text backbone: ``KimiLinearModel`` + lm head + logits processor.

    Inherits ``BaseCausalLM`` so the ``model.*`` / ``lm_head.*`` weight hierarchy
    matches the checkpoint (``language_model.model.*`` / ``language_model.lm_head.*``
    after the wrapper strips the ``language_model.`` prefix).
    """

    model_cls = KimiLinearModel

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None) -> None:
        """Take the draft config's one-based completed-layer ids unchanged."""
        num_layers = len(self.model.layers)
        selected = (
            [2, num_layers // 2, num_layers - 3]
            if layer_ids is None
            else list(layer_ids)
        )
        if selected != sorted(selected) or len(set(selected)) != len(selected):
            raise ValueError(
                "K3 EAGLE3 layer ids must be unique and sorted ascending, "
                f"got {selected}."
            )
        invalid = [layer_id for layer_id in selected if not 1 <= layer_id <= num_layers]
        if invalid:
            raise ValueError(
                "K3 EAGLE3 layer ids must identify a completed K3 layer; "
                f"got invalid ids {invalid} for {num_layers} layers."
            )
        self.model.eagle3_layers_to_capture = tuple(selected)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_dflash_layers_to_capture(
        self,
        layer_ids: list[int],
        incremental_callback=None,
        slot_bufs: list | None = None,
    ) -> None:
        """Capture the K3 residual stream after each named target layer.

        DFLASH/DSpark checkpoints name 0-indexed completed-layer outputs. The
        per-layer return has already accumulated the layer output into K3's
        prefix stream, matching vLLM's target-side capture contract.
        """
        num_layers = len(self.model.layers)
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError("DFLASH target_layer_ids must be unique.")
        invalid = [val for val in layer_ids if val < 0 or val >= num_layers]
        if invalid:
            raise ValueError(
                "DFLASH target_layer_ids must map to capturable target layer "
                f"outputs. Got invalid ids {invalid}; valid range is "
                f"[0, {num_layers - 1}] for {num_layers} target layers."
            )
        self.capture_aux_hidden_states = True
        # Ascending: the draft concatenates the taps positionally, so the
        # capture order is part of the weight contract.
        self.model.layers_to_capture = sorted(layer_ids)
        self.model._dflash_capture_idx_map = {
            layer_idx: i for i, layer_idx in enumerate(self.model.layers_to_capture)
        }
        self.model._dflash_incremental_callback = incremental_callback
        self.model._dflash_slot_bufs = slot_bufs
        self.model._refresh_dflash_capture_fallback()

    def set_dflash_aux_hidden_stream(self, stream: str) -> None:
        """Select which K3 residual stream the DFLASH/DSpark taps read."""
        if stream not in ("prefix", "attn_res"):
            raise ValueError(
                f"Unknown DFLASH aux hidden stream {stream!r}; "
                "expected 'prefix' or 'attn_res'."
            )
        if stream == "attn_res" and not getattr(
            self.config, "attn_res_block_size", None
        ):
            raise ValueError(
                "The 'attn_res' aux hidden stream needs a target with AttnRes "
                "enabled (config.attn_res_block_size); this target has none."
            )
        self.model.dflash_aux_stream = stream
        self.model._refresh_dflash_capture_fallback()
        logger.info(
            "DFLASH/DSpark target capture: layers=%s stream=%s",
            tuple(self.model.layers_to_capture),
            stream,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Load the ``model.*`` / ``lm_head.*`` text weights.

        Reuses the DeepSeek machinery: ``gate_proj``/``up_proj`` stack into
        ``gate_up_proj``; ``q_a_proj``/``kv_a_proj_with_mqa`` fuse into
        ``fused_qkv_a_proj_with_mqa``; routed experts go through the MoE
        checkpoint loader (``w1``/``w3``/``w2`` -> ``w13``/``w2``, MXFP4).

        KDA layers' remaining ``self_attn.*`` weights (conv + A_log / dt_bias /
        f_b / o_norm / o_proj) load directly through the default path — their
        names match ``KimiLinearKDA``'s modules 1:1 (no fusion).

        ModelOpt FP8_PB_WO checkpoints (nvidia/Kimi-K3-NVFP4) are adapted by
        ``preprocess_fp8_pb_wo_weights`` before any of the above: only f_b
        (raw GEMV inside the KDA attention backend) arrives block-dequantized
        to bf16; every other attention projection stays FP8-resident with its
        scale renamed to ``weight_scale_inv`` — o_proj / kv_b / q_b flow to
        their w8a8 LinearBase params, the KDA merged q/k/v/g/f_a/b shards
        stack into the FP8 qkvgb buffer (segment-concatenated scale grid),
        and the MLA q_a/kv_a/g segments are reassembled verbatim into the
        fused_qkv_a private layout below.
        """
        weights = preprocess_fp8_pb_wo_weights(weights, self.quant_config)
        config = self.config
        stacked_params_mapping = [
            # KDA q/k/v/g/f_a/b stack into qkvgb_proj; MLA's g_proj falls
            # through (no such param on MLA layers).
            ("self_attn.qkvgb_proj", "self_attn.q_proj", "q"),
            ("self_attn.qkvgb_proj", "self_attn.k_proj", "k"),
            ("self_attn.qkvgb_proj", "self_attn.v_proj", "v"),
            ("self_attn.qkvgb_proj", "self_attn.g_proj", "g"),
            ("self_attn.qkvgb_proj", "self_attn.f_a_proj", "f_a"),
            ("self_attn.qkvgb_proj", "self_attn.b_proj", "b"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        fuse_qkv_a_proj = config.q_lora_rank is not None

        params_dict = dict(self.named_parameters())

        # ModelOpt FP8_PB_WO fused_qkv_a assembly: MLA q_a / kv_a / g_proj
        # arrive FP8-resident with per-segment block scales. The canonical
        # [q_a | kv_a | gate] order is not 128-block aligned (the gate would
        # start at row q_lora+kv_lora+rope = 2112), so collect the six
        # tensors per MLA layer and stack them VERBATIM in the private
        # [gate | q_a | kv_a | pad] order (_FP8_FUSED_QKV_A_ORDER) — zero
        # requantization. KDA g_proj (FP8 too, but stacked into the merged
        # qkvgb buffer) and bf16 checkpoints skip this path entirely.
        fp8_fused_pending: dict[str, dict[str, torch.Tensor]] = {}
        fp8_fused_segments = frozenset(_FP8_FUSED_QKV_A_ORDER)

        def _try_fp8_fused_assembly(name: str, loaded_weight: torch.Tensor) -> bool:
            is_scale = name.endswith(".weight_scale_inv")
            if not (is_scale or name.endswith(".weight")):
                return False
            module_name = name.rsplit(".", 1)[0]
            leaf = module_name.rsplit(".", 1)[-1]
            if leaf not in fp8_fused_segments:
                return False
            base = module_name.rsplit(".", 1)[0]
            fused_weight_name = f"{base}.fused_qkv_a_proj_with_mqa.weight"
            if fused_weight_name not in params_dict:
                return False  # KDA layers (g_proj) or unfused configs
            if not is_scale and loaded_weight.dtype not in _FP8_WEIGHT_DTYPES:
                return False  # bf16 checkpoints keep the existing fused path
            entry = fp8_fused_pending.setdefault(base, {})
            entry[f"{leaf}.{'scale' if is_scale else 'weight'}"] = loaded_weight
            if len(entry) < 2 * len(fp8_fused_segments):
                return True
            pieces = fp8_fused_pending.pop(base)
            # The checkpoint gate is globally head-sharded; take this
            # attention rank's rows (its scale grid slices with it as long
            # as the shard is block-aligned).
            gate_w = pieces["g_proj.weight"]
            gate_s = pieces["g_proj.scale"]
            gate_rows = gate_w.shape[0] // self.mapping.attn.tp_size
            if gate_rows % 128 != 0:
                raise ValueError(
                    f"FP8 fused_qkv_a gate shard ({gate_rows} rows) must "
                    "align to the 128-row scale-block grid; adjust the "
                    "attention TP size."
                )
            gate_start = self.mapping.attn.tp_rank * gate_rows
            blocks_per_shard = gate_rows // 128
            block_start = self.mapping.attn.tp_rank * blocks_per_shard
            weight_param = params_dict[fused_weight_name]
            # Verbatim stack in the private [gate | q_a | kv_a | pad] order:
            # every boundary is 128-aligned, so codes and scale rows copy
            # bit-identically — zero requantization; consumers split via
            # _split_fused_qkv_a. The segment order is derived from
            # _FP8_FUSED_QKV_A_ORDER so loader and consumers cannot drift.
            ordered = {
                "g_proj": (
                    gate_w[gate_start : gate_start + gate_rows],
                    gate_s[block_start : block_start + blocks_per_shard],
                ),
                "q_a_proj": (pieces["q_a_proj.weight"], pieces["q_a_proj.scale"]),
                "kv_a_proj_with_mqa": (
                    pieces["kv_a_proj_with_mqa.weight"],
                    pieces["kv_a_proj_with_mqa.scale"],
                ),
            }
            fused_w, fused_s = _assemble_fp8_fused_qkv_a(
                [ordered[leaf] for leaf in _FP8_FUSED_QKV_A_ORDER],
                total_rows=weight_param.shape[0],
            )
            weight_param.weight_loader(weight_param, fused_w)
            scale_param = params_dict[
                f"{base}.fused_qkv_a_proj_with_mqa.weight_scale_inv"
            ]
            scale_param.weight_loader(scale_param, fused_s)
            return True

        moe_loader = build_moe_checkpoint_loader(
            params_dict=params_dict,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="w1", up_proj_name="w3", down_proj_name="w2"
            ),
            num_experts=config.num_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )

        pp_start = self.model.pp_start_layer
        pp_end = self.model.pp_end_layer
        pp_windowed = not (pp_start == 0 and pp_end == config.num_hidden_layers)

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            # MTP checkpoints append NextN draft layer(s) past num_hidden_layers;
            # the draft worker loads those.
            if name.startswith("model.layers."):
                layer_str = name.split(".")[2]
                if layer_str.isdigit() and int(layer_str) >= config.num_hidden_layers:
                    continue
                # Skip layers another pipeline stage owns: the MoE loader
                # raises on a matched-but-unloadable expert weight, so
                # out-of-window layers must be dropped before it sees them.
                if (
                    pp_windowed
                    and layer_str.isdigit()
                    and not pp_start <= int(layer_str) < pp_end
                ):
                    continue
            # Compressed-tensors MXFP4 routed experts ship the packed weight as
            # ``...w{1,2,3}.weight_packed``; the mxfp4 MoE param is
            # ``w13_weight`` / ``w2_weight`` (packed uint8), so drop the
            # ``_packed`` suffix for the expert loader (scale keeps ``weight_scale``).
            if "experts." in name and name.endswith(".weight_packed"):
                name = name[: -len(".weight_packed")] + ".weight"
            # KDA conv weights are plain params named ``<qkv>_conv1d_weight``.
            if "_conv1d.weight" in name:
                name = name.replace("_conv1d.weight", "_conv1d_weight")

            # FP8-resident fused_qkv_a segments (verbatim reorder), see above.
            if _try_fp8_fused_assembly(name, loaded_weight):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ".experts." in name and name not in params_dict:
                    continue  # routed-expert weights handled by moe_loader below
                    # NB: ``.experts.`` (leading dot) so ``shared_experts`` is
                    # NOT matched -- its gate_proj/up_proj must stack here.
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if moe_loader.matches(name):
                    moe_loader.load(name, loaded_weight)
                    continue

                if fuse_qkv_a_proj and ".g_proj" in name:
                    # MLA output gate (KDA g_proj stacked into qkvgb above):
                    # the per-rank shard sits after [q_a | kv_a+rope] in the
                    # widened fused a-projection.
                    mapped = name.replace("g_proj", "fused_qkv_a_proj_with_mqa")
                    param = params_dict.get(mapped)
                    if param is not None:
                        gate_offset = (
                            config.q_lora_rank
                            + config.kv_lora_rank
                            + config.qk_rope_head_dim
                        )
                        # The checkpoint gate is globally head-sharded; load
                        # this attention rank's rows into the fused tail.
                        gate_rows = loaded_weight.shape[0] // self.mapping.attn.tp_size
                        gate_start = self.mapping.attn.tp_rank * gate_rows
                        gate_shard = loaded_weight[gate_start : gate_start + gate_rows]
                        param.weight_loader(param, gate_shard, begin_size=gate_offset)
                        continue

                if fuse_qkv_a_proj and (
                    "q_a_proj" in name or "kv_a_proj_with_mqa" in name
                ):
                    # Single targeted replace: chaining ``.replace`` corrupts the
                    # q_a case because ``fused_qkv_a_proj_with_mqa`` (the q_a
                    # result) itself contains ``kv_a_proj_with_mqa`` as a
                    # substring, so a second replace would mangle it.
                    if "q_a_proj" in name:
                        begin_size = 0
                        mapped = name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                    else:
                        begin_size = config.q_lora_rank
                        mapped = name.replace(
                            "kv_a_proj_with_mqa", "fused_qkv_a_proj_with_mqa"
                        )
                    param = params_dict.get(mapped)
                    if param is None:
                        continue
                    param.weight_loader(param, loaded_weight, begin_size=begin_size)
                    continue

                param = params_dict.get(name)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

        if fp8_fused_pending:
            raise RuntimeError(
                "FP8 fused_qkv_a layers missing segments at end of checkpoint "
                f"stream: { {k: sorted(v) for k, v in fp8_fused_pending.items()} }"
            )
        self.post_load_weights()
        # Return the loader's transient allocator blocks (block-dequant
        # intermediates for the dequant-routed weights, fused-assembly
        # segment buffers) to the device before the KV pool sizes itself
        # from free memory; otherwise the reserved-but-free slack shrinks
        # the pool by up to ~1 GiB per rank.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def post_load_weights(self) -> None:
        """Prepare the absorbed MLA weights (``w_kc``/``w_vc``) per MLA layer.

        With FP8_PB_WO checkpoints ``kv_b_proj.weight`` stays FP8 for the
        w8a8 prefill GEMM; the absorbed ``w_kc``/``w_vc`` copies follow the
        DeepSeek R1 precedent and block-dequantize to the activation dtype
        (``DeepseekV3ForCausalLM.post_load_weights``). bf16 checkpoints (e.g.
        moonshotai/Kimi-K3, whose MXFP4 config ignores ``self_attn.*``) skip
        the dequant. KDA layers have no ``kv_b_proj`` (not
        ``KimiLinearMLAAttention``) and are skipped.
        """
        for layer in self.model.layers:
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:
                continue  # PPMissingLayer: another pipeline stage owns it
            if isinstance(self_attn, KimiLinearMLAAttention):
                w = self_attn.kv_b_proj.weight
                if w.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
                    if not hasattr(self_attn.kv_b_proj, "weight_scale_inv"):
                        raise RuntimeError(
                            "kv_b_proj.weight_scale_inv is required for block "
                            "FP8 dequant of the absorbed MLA weights."
                        )
                    weight_block_size = (
                        self_attn.kv_b_proj.quant_method.quant_config.weight_block_size
                    )
                    w = block_dequant(
                        w,
                        self_attn.kv_b_proj.weight_scale_inv,
                        weight_block_size,
                    ).to(torch.get_default_dtype())
                self_attn.w_kc, self_attn.w_vc = _prepare_mla_kv_b_proj_weights(
                    w, self_attn
                )
            elif isinstance(self_attn, KimiLinearKDA):
                self_attn.fuse_conv_weights()
                merged = self_attn.qkvgb_proj
                if getattr(merged, "fp8_block_quant", False):
                    merged.verify_fp8_load_complete()
                    # Prepack the block scales for the flashinfer w8a8 GEMM
                    # (the same preparation Fp8LinearMethod does for
                    # LinearBase layers); rows are 128-padded at construction
                    # so the shape gate always holds on this path.
                    from tokenspeed_kernel.ops.gemm.flashinfer import (
                        has_flashinfer_fp8_blockscale,
                        prepare_flashinfer_fp8_blockscale_weight_scales,
                    )

                    n_rows, n_cols = merged.weight.shape
                    if (
                        merged._flashinfer_scales_mn is None
                        and has_flashinfer_fp8_blockscale is not None
                        and has_flashinfer_fp8_blockscale()
                        and n_rows % 128 == 0
                        and n_cols % 128 == 0
                    ):
                        # Build once: rebinding after a refit would leave any
                        # captured CUDA graph holding the stale buffer.
                        merged._flashinfer_scales_mn = (
                            prepare_flashinfer_fp8_blockscale_weight_scales(
                                merged.weight_scale_inv.data
                            )
                        )

        # Fold the AttnRes rms_w * res_w products once; the split kernels take a single wp pointer.
        for layer in self.model.layers:
            if not hasattr(layer, "self_attention_res_norm"):
                continue  # PPMissingLayer
            layer._attn_wp = (
                layer.self_attention_res_norm.weight.float()
                * layer.self_attention_res_proj.weight.reshape(-1).float()
            ).to(torch.bfloat16)
            layer._mlp_wp = (
                layer.mlp_res_norm.weight.float()
                * layer.mlp_res_proj.weight.reshape(-1).float()
            ).to(torch.bfloat16)

        for layer in self.model.layers:
            if getattr(layer, "is_moe_layer", False):
                layer.block_sparse_moe.pack_input_projection_weights()


# ===----------------------------------------------------------------------=== #
# Registered multimodal wrapper
# ===----------------------------------------------------------------------=== #


class KimiK3ForConditionalGeneration(nn.Module):
    """Kimi-K3 top-level model (registered architecture).

    Construction mirrors ``KimiK25ForConditionalGeneration``: a vision path plus
    a text ``language_model``. The text path is ``KimiLinearForCausalLM``.
    """

    def __init__(
        self,
        config: KimiK3Config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_multimodal_active: bool = True,
        mm_attention_backend: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.quant_config = quant_config
        self.is_multimodal_active = is_multimodal_active

        # EPD encode workers own only the vision tower.
        self.language_model = None
        if not getattr(config, "encoder_only", False):
            self.language_model = KimiLinearForCausalLM(
                config.text_config,
                mapping=mapping,
                quant_config=quant_config,
            )

        # Multimodal path. ``image_encoder`` may later be replaced by
        # ModelExecutor with the encoder CUDA-graph wrapper.
        if is_multimodal_active:
            self.vision = KimiK3Vision(
                config.vision_config,
                mapping=mapping,
                quant_config=quant_config,
                mm_attention_backend=mm_attention_backend,
            )
            # Normal serving follows the text embedding dtype. Encoder-only
            # construction uses ModelLoader's configured default dtype.
            # Non-first pipeline stages own no embedding (and never run the
            # vision encoder on live inputs); keep the loader dtype there.
            if (
                self.language_model is not None
                and self.language_model.model.embed_tokens is not None
            ):
                target_dtype = self.get_input_embeddings().weight.dtype
                self.vision = self.vision.to(dtype=target_dtype)
            self.vision_embedder = VisionEmbedder(encoder_mapping=mapping.vision)
            self.image_encoder = self.vision.embed_media
        else:
            self.vision = None
            self.vision_embedder = None
            self.image_encoder = None

    def get_input_embeddings(self) -> nn.Module:
        if self.language_model is None:
            raise AttributeError(
                "Kimi-K3 encoder-only mode does not expose text embeddings."
            )
        return self.language_model.model.get_input_embeddings()

    def get_embed_and_head(self):
        return self.language_model.get_embed_and_head()

    def set_dflash_layers_to_capture(
        self,
        layer_ids: list[int],
        incremental_callback=None,
        slot_bufs: list | None = None,
    ) -> None:
        if self.language_model is None:
            raise AttributeError(
                "Kimi-K3 encoder-only mode cannot capture target hidden states."
            )
        self.language_model.set_dflash_layers_to_capture(
            layer_ids,
            incremental_callback=incremental_callback,
            slot_bufs=slot_bufs,
        )

    def set_dflash_aux_hidden_stream(self, stream: str) -> None:
        if self.language_model is None:
            raise AttributeError(
                "Kimi-K3 encoder-only mode cannot capture target hidden states."
            )
        self.language_model.set_dflash_aux_hidden_stream(stream)

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None) -> None:
        if self.language_model is None:
            raise AttributeError(
                "Kimi-K3 encoder-only mode does not support EAGLE3 speculative decoding."
            )
        self.language_model.set_eagle3_layers_to_capture(layer_ids)

    @property
    def logits_processor(self):
        # The runtime reads ``model.logits_processor`` on the top-level model
        # (model_executor.py) to build its sampling topology; delegate to the
        # text backbone, which owns it (BaseCausalLM).
        if self.language_model is None:
            raise AttributeError(
                "Kimi-K3 encoder-only mode does not expose a logits processor."
            )
        return self.language_model.logits_processor

    @property
    def lm_head(self):
        if self.language_model is None:
            raise AttributeError(
                "Kimi-K3 encoder-only mode does not expose an LM head."
            )
        return self.language_model.lm_head

    @property
    def model(self):
        """Expose the text backbone under the ``model`` attribute other
        registered architectures use (the PP executor resolves the stage wire
        spec via ``model.pp_stage_state_spec``)."""
        if self.language_model is None:
            raise AttributeError(
                "Kimi-K3 encoder-only mode does not expose a text backbone."
            )
        return self.language_model.model

    @property
    def vision_tower(self):
        """Expose the shared MoonViT attribute expected by EPD prefill."""
        return self.vision.vision_tower if self.vision is not None else None

    def get_multimodal_encoder_specs(self) -> dict[Modality, EncoderSpec]:
        if self.vision is None or self.image_encoder is None:
            return {}
        return {
            Modality.IMAGE: EncoderSpec(
                self.image_encoder,
                make_warmup_items=self.vision.make_image_warmup_items,
            )
        }

    def make_encoder_cudagraph_wrapper(
        self, mapping: Mapping
    ) -> EncoderForwardStepRunner:
        return self.vision.make_encoder_cudagraph_wrapper(mapping)

    def make_encoder_cudagraph_wrappers(self, mapping: Mapping) -> dict:
        if self.vision is None:
            return {}
        return {"image_encoder": self.make_encoder_cudagraph_wrapper(mapping)}

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        return pad_input_tokens(input_ids, mm_inputs)

    @torch.no_grad()
    def multimodal_input_embeds(
        self,
        input_ids: torch.Tensor,
        ctx: "ForwardContext",
        multimodal_context,
    ) -> torch.Tensor | None:
        if (
            multimodal_context is None
            or self.vision_embedder is None
            or not multimodal_context.has_extend_inputs()
            or ctx.forward_mode.is_decode_or_idle()
            # Non-first pipeline stages receive the residual stream from
            # upstream; they own no embedding to splice vision embeds into.
            or not self.mapping.is_first_pp_rank
        ):
            return None
        input_embeds, model_kwargs = self.vision_embedder.apply(
            input_ids=input_ids,
            text_embedding=self.get_input_embeddings(),
            ctx=multimodal_context,
            encoders=self.get_multimodal_encoder_specs(),
            multimodal_model=self,
        )
        assert not model_kwargs, "Kimi-K3 multimodal path must stay embeds-only"
        return input_embeds

    @torch.no_grad()
    def forward(
        self,
        ctx: "ForwardContext",
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if self.language_model is None:
            raise RuntimeError(
                "Kimi-K3 encoder-only mode cannot execute language-model forward."
            )
        multimodal_context = kwargs.pop("multimodal_context", None)
        input_embeds = self.multimodal_input_embeds(input_ids, ctx, multimodal_context)
        if input_embeds is not None:
            kwargs["input_embeds"] = input_embeds
        return self.language_model.forward(
            ctx,
            input_ids,
            positions,
            **kwargs,
        )

    def post_load_weights(self) -> None:
        """Prepare text-model derived weights for loaders that skip checkpoints."""
        if self.language_model is not None:
            self.language_model.post_load_weights()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Route checkpoint weights by top-level prefix.

        The checkpoint stores ``language_model.*`` for the text model (whose
        params are named ``model.*`` / ``lm_head.*``) and
        ``vision_tower.*`` / ``mm_projector.*`` for the multimodal path.

        Routing stays streaming: K3's text checkpoint is too large to retain
        every yielded tensor in a temporary list. Vision tensors are loaded as
        the language loader advances the source iterator.
        """
        loaded_vision_weights = 0
        dropped_vision_weights = 0
        vision_params = (
            dict(self.vision.named_parameters(remove_duplicate=False))
            if self.vision is not None
            else None
        )

        def language_weights():
            nonlocal loaded_vision_weights, dropped_vision_weights
            for name, weight in weights:
                if name.startswith("vision_tower.") or name.startswith("mm_projector."):
                    if self.vision is None:
                        dropped_vision_weights += 1
                    else:
                        assert vision_params is not None
                        self.vision.load_weight(name, weight, vision_params)
                        loaded_vision_weights += 1
                    continue
                if name.startswith("language_model."):
                    name = name[len("language_model.") :]
                yield name, weight

        if self.language_model is not None:
            self.language_model.load_weights(language_weights())
        else:
            # Exhaust the stream so interleaved vision weights are still routed.
            for _ in language_weights():
                pass
        if dropped_vision_weights:
            logger.warning(
                "Dropping %d vision weights: multimodal path is inactive.",
                dropped_vision_weights,
            )
        logger.debug("Loaded %d Kimi-K3 vision tensors.", loaded_vision_weights)


EntryClass = [KimiK3ForConditionalGeneration]
