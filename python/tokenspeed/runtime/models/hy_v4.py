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

"""Inference-only HY V4 (``HYV4ForCausalLM``).

HY V4 shares the GLM-MoE-DSA backbone: MLA with the DeepSeek sparse indexer, a
sigmoid-routed ``noaux_tc`` MoE with one shared expert, and a dense first layer.
On top of that backbone this file adds what the HY V4 checkpoint declares:

* a gated MLA output projection (``linear_gate``, elementwise by default),
* a per-head learnable attention sink, applied inside the DSA kernels,
* independent Hyper-Connections (iHC) around both sub-blocks, and
* a clamped SwiGLU on the routed experts only (never the dense or shared ones).

The learnable sink is part of the architecture, not a serving option: a backend
that cannot apply it would change the model's output, so the DSA kernels declare
``sinks`` and selection drops any kernel that cannot consume it.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from dataclasses import replace
from functools import lru_cache
from typing import Any, NamedTuple

import torch
from tokenspeed_kernel.ops import residual as residual_ops
from tokenspeed_kernel.ops.attention import dsa as dsa_ops
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    current_valid_rows,
    slice_to_real_tokens,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.dcp.placement import resolve_cache_slots
from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, RMSNorm
from tokenspeed.runtime.layers.linear import ColumnParallelLinear
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    MoELayer,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.topk import TopK
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.modelopt_mxfp8 import (
    ModelOptMxfp8Config,
    dequantize_mxfp8_weight,
    preprocess_modelopt_mxfp8_weights,
)
from tokenspeed.runtime.layers.rotary_embedding import RotaryEmbedding
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3AttentionMLA,
    DeepseekV3MLP,
    DeepseekV3MoE,
    MoEGate,
    _prepare_mla_kv_b_proj_weights,
)
from tokenspeed.runtime.models.glm5 import (
    GlmDsaIndexer,
    GlmDsaIndexerOutput,
    GlmMoeDsaAttention,
    GlmMoeDsaForCausalLM,
    _glm_dsa_skip_indexer_topk,
)
from tokenspeed.runtime.utils import LazyValue, add_prefix, get_colorful_logger
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = get_colorful_logger(__name__)

# Checkpoint -> module renames for the iHC parameters. The checkpoint nests the
# sub-block boundary under ``hc_pre`` and prefixes the head's tensors with
# ``hc_head_``; the modules here keep one flat ``hc_fn/hc_scale/hc_base`` set
# per boundary so the pre and head boundaries share a single parameter layout.
_HC_CHECKPOINT_RENAMES = (
    (".hc_attn_layer.hc_pre.", ".hc_attn_layer."),
    (".hc_mlp_layer.hc_pre.", ".hc_mlp_layer."),
    (".hc_head.hc_head_fn", ".hc_head.hc_fn"),
    (".hc_head.hc_head_scale", ".hc_head.hc_scale"),
    (".hc_head.hc_head_base", ".hc_head.hc_base"),
)

_STACKED_DENSE_PROJECTIONS = (
    (".gate_up_proj.", ".gate_proj.", 0),
    (".gate_up_proj.", ".up_proj.", 1),
)


class HYV4MLP(DeepseekV3MLP):
    """Unclamped dense and shared-expert projections."""


def hyv4_rmsnorm(hidden_size: int, *, eps: float) -> RMSNorm:
    return RMSNorm(hidden_size, eps=eps)


def normalize_hyv4_checkpoint_name(name: str) -> str:
    """Map an HY V4 checkpoint tensor name onto this module tree."""
    for checkpoint_part, module_part in _HC_CHECKPOINT_RENAMES:
        if checkpoint_part in name:
            name = name.replace(checkpoint_part, module_part)
    return name


def check_hyv4_runtime(mapping: Mapping, *, enable_ihc: bool) -> None:
    """Reject unsupported topology or kernel packages before allocating layers.

    Import the residual module, not the new iHC symbols, so model discovery
    with an older kernel package can still serve unrelated architectures.
    HY V4 itself requires the iHC and sink-capable kernels from its release.
    """
    if global_server_args_dict["numerics"] != "auto":
        raise ValueError("HY V4 currently supports --numerics auto only")
    if mapping.has_pp:
        raise ValueError(
            "HY V4 does not support pipeline parallelism; "
            "--pipeline-parallel-size must be 1."
        )
    if mapping.attn.has_head_tp or mapping.attn.has_qcp:
        raise ValueError(
            "HY V4 currently requires ordinary attention TP or DP; "
            "attention head TP and query context parallelism are unsupported"
        )
    missing = []
    if enable_ihc:
        missing.extend(
            name
            for name in ("ihc_pre", "ihc_post", "ihc_head")
            if not callable(getattr(residual_ops, name, None))
        )
    for name in ("dsa_prefill", "dsa_decode"):
        operation = getattr(dsa_ops, name, None)
        if (
            operation is None
            or "attn_sink" not in inspect.signature(operation).parameters
        ):
            missing.append(f"{name}(attn_sink=...)")
    if missing:
        raise RuntimeError(
            f"HY V4 requires newer tokenspeed-kernel capabilities: {', '.join(missing)}. "
            "Install tokenspeed-kernel from the same source revision as the "
            "runtime, or a release including iHC, DSA sinks and MXFP8 MoE."
        )


class HYV4RotaryEmbedding(RotaryEmbedding):
    """Plain RoPE with coefficients rounded to the activation dtype."""

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        # The prologue stores coefficients in FP32, but HY V4 rotates with
        # coefficients rounded to the model dtype before FP32 arithmetic.
        return super()._compute_cos_sin_cache().to(self.dtype).float()


@lru_cache(maxsize=16)
def _cached_hyv4_rope(
    *,
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    is_neox_style: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> HYV4RotaryEmbedding:
    # Keep this cache separate from GLM/DeepSeek, including across devices.
    with torch.device(device):
        return HYV4RotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype
        )


def _make_hyv4_rope(
    *,
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict[str, Any] | None,
    is_neox_style: bool,
) -> HYV4RotaryEmbedding:
    if rope_scaling is not None:
        raise ValueError("HY V4 requires plain RoPE")
    return _cached_hyv4_rope(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=base,
        is_neox_style=is_neox_style,
        dtype=torch.get_default_dtype(),
        device=torch.empty(0).device,
    )


class HYV4DsaIndexer(GlmDsaIndexer):
    """DSA indexer using HY V4's checkpoint layout.

    The projections and the FP8 index-K cache are shared with the GLM-DSA
    indexer; only the RoPE slice differs. HY V4 stores the rope part in the
    *trailing* ``qk_rope_head_dim`` elements of each index head (GLM puts it
    first), and rotates it interleaved. Rotating the wrong slice would corrupt
    the sparse selection silently instead of failing loudly.
    """

    def _make_rotary_embedding(
        self,
        *,
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: float,
        rope_scaling: dict[str, Any] | None,
        is_neox_style: bool,
    ) -> RotaryEmbedding:
        return _make_hyv4_rope(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=base,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
    ) -> GlmDsaIndexerOutput:
        index_q = self.wq_b(q_lora)[0]
        index_q = index_q.view(-1, self.index_n_heads, self.index_head_dim)
        if self._wk_weights_proj_loaded:
            assert self.wk_weights_proj is not None
            key_weights = self.wk_weights_proj(hidden_states)[0]
            index_k, weights = key_weights.split(
                [self.index_head_dim, self.index_n_heads],
                dim=-1,
            )
        else:
            if self.wk is None or self.weights_proj is None:
                raise RuntimeError(
                    "Fused indexer weights must be loaded before forward"
                )
            index_k = self.wk(hidden_states)[0]
            weights = self.weights_proj(hidden_states)[0]
        index_k = self.k_norm(index_k)

        nope_dim = self.index_head_dim - self.rope_head_dim
        q_nope, q_pe = index_q.split([nope_dim, self.rope_head_dim], dim=-1)
        k_nope, k_pe = index_k.split([nope_dim, self.rope_head_dim], dim=-1)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe[:, None, :])
        index_q = torch.cat(
            (q_nope, q_pe.reshape(-1, self.index_n_heads, self.rope_head_dim)),
            dim=-1,
        )
        index_k = torch.cat(
            (k_nope, k_pe.reshape(-1, self.rope_head_dim)),
            dim=-1,
        )
        # Raw weights: the per-head normalization is folded into
        # ``weights_softmax_scale`` exactly as on the GLM-DSA path.
        return GlmDsaIndexerOutput(query=index_q, key=index_k, weights=weights)


class HYV4Attention(GlmMoeDsaAttention):
    """MLA + DSA attention with HY V4's output gate and learnable sink."""

    _MLA_KERNEL_BACKENDS = ("trtllm_mla", "tokenspeed_mla", "dsa")
    _RAGGED_PREFILL_BACKENDS = ("trtllm_mla", "tokenspeed_mla", "dsa")
    rope_is_neox_style = False

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
        *,
        is_nextn: bool,
    ) -> None:
        hidden_size = config.hidden_size
        # Bypass the GLM constructor: it builds its own indexer and normalizes
        # the latent with a fixed 1e-6, while HY V4 uses ``rms_norm_eps``.
        DeepseekV3AttentionMLA.__init__(
            self,
            config=config,
            mapping=mapping,
            hidden_size=hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=config.rope_theta,
            rope_scaling=None,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            prefix=prefix,
            reduce_attn_results=False,
            alt_stream=alt_stream,
        )
        self.q_a_layernorm = hyv4_rmsnorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.kv_a_layernorm = hyv4_rmsnorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.fused_qk_layernorm = FusedRMSNorm(
            self.q_a_layernorm,
            self.kv_a_layernorm,
        )
        self.index_topk = config.index_topk
        # Shared target layers own no indexer and attend over the top-k the
        # last full layer left in the shared slot. The MTP layer always selects
        # its own.
        self.skip_indexer_topk = not is_nextn and _glm_dsa_skip_indexer_topk(
            config, layer_id
        )
        self.indexer: HYV4DsaIndexer | None = (
            None
            if self.skip_indexer_topk
            else HYV4DsaIndexer(
                config=config,
                hidden_size=hidden_size,
                q_lora_rank=config.q_lora_rank,
                qk_rope_head_dim=config.qk_rope_head_dim,
                rope_theta=config.rope_theta,
                rope_scaling=None,
                max_position_embeddings=config.max_position_embeddings,
                quant_config=quant_config,
                prefix=add_prefix("indexer", prefix),
                # Match the reference: convert only indexer wk to BF16 at
                # load time, then fuse it with the unquantized weights_proj.
                fused_only=True,
            )
        )
        self._decode_topk_indices_buffer: torch.Tensor | None = None
        self._decode_topk_lens_buffer: torch.Tensor | None = None
        self._retired_decode_workspaces: list[torch.Tensor] = []

        self.gate_size_per_head = 0
        self.linear_gate: ColumnParallelLinear | None = None
        if config.gated_mla:
            self.gate_size_per_head = (
                1 if config.gating_type == "headwise" else config.v_head_dim
            )
            self.linear_gate = ColumnParallelLinear(
                hidden_size,
                config.num_attention_heads * self.gate_size_per_head,
                bias=False,
                quant_config=quant_config,
                tp_rank=mapping.attn.tp_rank,
                tp_size=mapping.attn.tp_size,
                tp_group=mapping.attn.tp_group,
                prefix=add_prefix("linear_gate", prefix),
            )

        # Each attention TP rank owns its contiguous range of sink heads.
        # The DSA backend reads them through attn_mqa after loading.
        self.learnable_sink_param: nn.Parameter | None = None
        if config.learnable_sink:
            num_sink_heads = self.num_local_heads
            self.learnable_sink_param = nn.Parameter(
                torch.empty(num_sink_heads, dtype=torch.float32),
                requires_grad=False,
            )
            head_start = mapping.attn.tp_rank * num_sink_heads

            def load_sink(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
                if loaded_weight.numel() != config.num_attention_heads:
                    raise ValueError(
                        f"HY V4 sink needs {config.num_attention_heads} heads, "
                        f"got {loaded_weight.numel()}"
                    )
                heads = loaded_weight.reshape(-1)[
                    head_start : head_start + num_sink_heads
                ]
                param.data.copy_(heads.to(torch.float32))

            self.learnable_sink_param.weight_loader = load_sink

    def bind_learnable_sink(self) -> None:
        """Publish the loaded sink to the attention layer the backend reads.

        Hands over a detached view of the loaded parameter, so the backend sees
        the live storage without registering the tensor twice.
        """
        if self.learnable_sink_param is None:
            return
        self.attn_mqa.attn_sink = self.learnable_sink_param.detach()

    def _make_rotary_embedding(
        self,
        *,
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: float,
        rope_scaling: dict[str, Any] | None,
        is_neox_style: bool,
    ) -> RotaryEmbedding:
        return _make_hyv4_rope(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=base,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )

    def _apply_output_gate(
        self,
        attn_output: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Scale the attention output by ``sigmoid(linear_gate(hidden_states))``.

        The gate reads exactly the rows the attention consumed, so callers pass
        hidden states already gathered to the attention's token set.
        """
        if self.linear_gate is None or attn_output.shape[0] == 0:
            return attn_output
        gate = self.linear_gate(hidden_states)[0]
        if self.gate_size_per_head == 1:
            shaped = attn_output.view(attn_output.shape[0], -1, self.v_head_dim)
            shaped = shaped * torch.sigmoid(gate).unsqueeze(-1)
            return shaped.reshape(attn_output.shape[0], -1)
        # Materialize the sigmoid in the gate dtype before the product.
        return attn_output * torch.sigmoid(gate)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """HY V4 sparse attention: the GLM-DSA path plus the output gate.

        The hidden states are gathered once and shared by the indexer and the
        gate. MTP calls ``attend_token_shaped`` and selects its live rows before
        ``o_proj``; the target keeps all rows.
        """
        if hidden_states.shape[0] == 0:
            return hidden_states
        attn_output = self.attend_token_shaped(
            positions, hidden_states, ctx, comm_manager, block_scale
        )
        output, _ = self.o_proj(attn_output)
        return output

    def attend_token_shaped(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Projections, sparse attention and the output gate; no draft narrowing.

        Token-shaped through the eager ``_sparse_attention`` break. The NextN
        draft selects its live rows after this gate, before ``o_proj``.
        """
        if block_scale is not None:
            raise ValueError("HY V4 attention requires unquantized BF16 activations")
        qkv = self.fused_qkv_a_proj_with_mqa(
            hidden_states,
            block_scale,
            torch.bfloat16,
        )
        qkv = comm_manager.pre_attn_comm(qkv, ctx)

        q_a, latent_cache = qkv.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        # The packed-cache prologue normalizes KV while writing it. Share
        # its normalization boundary so the learned weight is applied once.
        q_norm = self._norm_qkv(q_a, latent_cache, ctx)

        should_compute_indexer = not self.skip_indexer_topk
        if should_compute_indexer or self.linear_gate is not None:
            hidden_states = comm_manager.pre_attn_comm(hidden_states, ctx)
        index_q = index_k = index_weights = None
        if should_compute_indexer:
            indexer_output = self.indexer(hidden_states, q_norm, positions)
            index_q = indexer_output.query
            index_k = indexer_output.key
            index_weights = indexer_output.weights

        q = self.q_b_proj(q_norm)[0]
        attn_output = self._sparse_attention(
            positions, q, latent_cache, index_q, index_k, index_weights, ctx
        )
        return self._apply_output_gate(attn_output, hidden_states)

    @break_point
    def _sparse_attention(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        index_q: torch.Tensor | None,
        index_k: torch.Tensor | None,
        index_weights: torch.Tensor | None,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """Cache writes, top-k selection and the sparse FlashMLA kernels.

        Same break-point contract as the GLM-DSA attention: none of this is
        capturable. The index tensors are ``None`` when the layer reuses the
        shared top-k.
        """
        valid_rows = current_valid_rows()
        if valid_rows is not None:
            # A padded prefill-graph replay hands this break the whole token
            # bucket; the prefill/decode split and the DSA kernels below are
            # sized by the live metadata, so drop the filler rows up front.
            positions, q, latent_cache, index_q, index_k, index_weights = (
                slice_to_real_tokens(
                    valid_rows,
                    positions,
                    q,
                    latent_cache,
                    index_q,
                    index_k,
                    index_weights,
                )
            )
        decode_window = self._resolve_decode_window(
            ctx,
            ctx.attn_backend.forward_decode_metadata,
            total_tokens=int(q.shape[0]),
        )
        num_decode_tokens = decode_window.num_tokens
        num_prefill_tokens = decode_window.start

        backend = ctx.attn_backend
        shared_topk = backend.sparse_topk
        prefill_locs = (
            backend.write_locations(self.attn_mqa, ForwardMode.EXTEND)
            if num_prefill_tokens > 0
            else None
        )
        decode_locs = (
            backend.write_locations(self.attn_mqa, ForwardMode.DECODE)
            if num_decode_tokens > 0
            else None
        )
        if prefill_locs is not None and decode_locs is not None:
            out_cache_loc = torch.cat((prefill_locs, decode_locs))
        elif prefill_locs is not None:
            out_cache_loc = prefill_locs
        elif decode_locs is not None:
            out_cache_loc = decode_locs
        else:
            out_cache_loc = torch.empty(0, dtype=torch.int64, device=q.device)

        if index_k is not None:
            indexer_output = GlmDsaIndexerOutput(
                query=index_q, key=index_k, weights=index_weights
            )
            index_slots, index_mask = resolve_cache_slots(
                out_cache_loc, backend.cache_placement(self.attn_mqa)
            )
            ctx.token_to_kv_pool.set_index_k_buffer(
                self.attn_mqa.layer_id,
                index_slots,
                index_k,
                write_mask=index_mask,
            )
            if ctx.num_extends > 0:
                shared_topk.prefill = self._compute_prefill_topk_indices(
                    indexer_output,
                    ctx,
                    num_prefill_tokens,
                )
            if ctx.num_extends < ctx.bs:
                shared_topk.decode = self._compute_decode_topk_indices(
                    indexer_output,
                    ctx,
                )

        attn_output = torch.empty(
            q.size(0),
            self.num_local_heads * self.v_head_dim,
            dtype=q.dtype,
            device=q.device,
        )

        if ctx.num_extends > 0:
            if shared_topk.prefill is None:
                raise RuntimeError("HY V4 sparse prefill requires top-k indices.")
            prefill_ctx = replace(
                ctx,
                bs=ctx.num_extends,
                input_num_tokens=num_prefill_tokens,
                forward_mode=ForwardMode.EXTEND,
            )
            self.forward_dsa_sparse_prefill(
                positions[:num_prefill_tokens],
                q[:num_prefill_tokens],
                latent_cache[:num_prefill_tokens],
                prefill_ctx,
                prefill_locs,
                attn_output[:num_prefill_tokens],
                prefill_topk=shared_topk.prefill,
            )

        if num_decode_tokens > 0:
            if shared_topk.decode is None:
                raise RuntimeError("HY V4 sparse decode requires top-k indices.")
            decode_ctx = replace(
                ctx,
                bs=decode_window.num_reqs,
                num_extends=0,
                input_num_tokens=num_decode_tokens,
                forward_mode=ForwardMode.DECODE,
            )
            topk_indices, topk_lens = self._slice_decode_topk(
                shared_topk.decode,
                decode_window.start,
                decode_window.end,
            )
            self.forward_absorb(
                positions[decode_window.start : decode_window.end],
                q[decode_window.start : decode_window.end],
                latent_cache[decode_window.start : decode_window.end],
                decode_ctx,
                decode_locs,
                attn_output[decode_window.start : decode_window.end],
                topk_indices=topk_indices,
                topk_lens=topk_lens,
            )
        return attn_output


class HYV4MoEGate(MoEGate):
    """Project the router's FP32 operands without a BF16-input GEMM path."""

    def __init__(self, config: PretrainedConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
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
        return torch.nn.functional.linear(hidden_states.float(), self.weight, None)


class HYV4MoE(DeepseekV3MoE):
    """DeepSeek routing with HY V4's clamped SwiGLU routed experts.

    Only the routed experts clamp their SwiGLU (``swiglu_limit``); the shared
    expert is the stock unclamped SwiGLU, matching the reference.
    """

    def __init__(
        self,
        config: PretrainedConfig,
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
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. Only silu is supported."
            )

        # Router parameters and logits stay in FP32 even for a BF16 model.
        self.gate = HYV4MoEGate(
            config=config,
            prefix=add_prefix("gate", prefix),
        )
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
            activation="swiglu",
            activation_alpha=1.0,
            swiglu_limit=config.swiglu_limit if config.swiglu_limit > 0 else None,
            routing_config={
                "n_group": config.n_group,
                "topk_group": config.topk_group,
                "routed_scaling_factor": config.routed_scaling_factor,
                "normalize_topk_weights": config.norm_topk_prob,
                "correction_bias": self.gate.e_score_correction_bias,
                "routing_method_type": RoutingMethodType.DeepSeekV3,
            },
            # With three or more of a token's experts on one EP rank, an
            # atomic top-k reduction makes the BF16 output vary run to run, and
            # the drift then spreads through attention.
            deterministic_finalize=True,
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
        self.shared_experts = HYV4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
            hidden_act=config.hidden_act,
            mapping=mapping,
            quant_config=quant_config,
            prefix=add_prefix("shared_experts", prefix),
            is_shared_expert=True,
            batch_invariant=False,
        )


class HYV4HCBoundary(nn.Module):
    """FP32 parameters of one iHC boundary.

    A sub-block boundary projects to ``2 * hc_mult`` gates (pre and post) with
    two scales; the head boundary projects to ``hc_mult`` pre gates with one.
    There is no combination matrix, which is what separates iHC from mHC.
    """

    def __init__(self, config: PretrainedConfig, *, is_head: bool) -> None:
        super().__init__()
        gate_count = config.hc_mult if is_head else 2 * config.hc_mult
        self.hc_fn = nn.Parameter(
            torch.empty(
                gate_count, config.hc_mult * config.hidden_size, dtype=torch.float32
            ),
            requires_grad=False,
        )
        self.hc_base = nn.Parameter(
            torch.empty(gate_count, dtype=torch.float32), requires_grad=False
        )
        self.hc_scale = nn.Parameter(
            torch.empty(1 if is_head else 2, dtype=torch.float32),
            requires_grad=False,
        )


class IhcPost(NamedTuple):
    """A sub-block output whose iHC post boundary is still to be applied."""

    block_output: torch.Tensor
    gates: torch.Tensor


class HYV4DecoderLayer(nn.Module):
    """One HY V4 decoder layer: gated sparse MLA plus a dense or MoE MLP.

    With iHC enabled the residual lives on the channel axis of
    ``hidden_states`` (``[num_tokens, hc_mult, hidden_size]``). Each post
    boundary runs in the projection pass of the next pre boundary, so a layer
    returns its channels before its MLP's post boundary and that post
    (``IhcPost``) as the residual; the next layer, or the model's head, applies
    it. The pre boundary rounds the channel reduction to the residual
    dtype before computing the fused RMSNorm in FP32.

    iHC requires the attention and MLP to see full token rows: the boundary
    reduces and re-scatters every channel of a token at once. Attention data
    parallelism (``attn_tp_size == 1``) or a layout where both the dense and
    the MoE layers all-reduce provides them; a reduce-scatter layout is
    rejected at construction time.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
        *,
        is_nextn: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_id = layer_id
        # The MTP block carries no iHC weights, so it runs single-stream.
        self.enable_ihc = config.enable_ihc and not is_nextn
        self.is_moe_layer = True if is_nextn else config.is_moe_layer(layer_id)

        self.self_attn = HYV4Attention(
            config=config,
            mapping=mapping,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
            is_nextn=is_nextn,
        )
        if self.is_moe_layer:
            self.mlp: nn.Module = HYV4MoE(
                config=config,
                mapping=mapping,
                quant_config=quant_config,
                layer_index=layer_id,
                prefix=add_prefix("mlp", prefix),
                alt_stream=alt_stream,
            )
        else:
            self.mlp = HYV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                mapping=mapping,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                is_shared_expert=False,
                batch_invariant=False,
            )

        self.input_layernorm = hyv4_rmsnorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = hyv4_rmsnorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        prev_is_moe = (
            True
            if is_nextn
            else (config.is_moe_layer(layer_id - 1) if layer_id > 0 else False)
        )
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=self.is_moe_layer,
            prev_is_moe=prev_is_moe,
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
            dense_batch_invariant=False,
            query_sharded=False,
        )

        self.hc_attn_layer: HYV4HCBoundary | None = None
        self.hc_mlp_layer: HYV4HCBoundary | None = None
        if self.enable_ihc:
            # iHC keeps full token rows on every rank, so no layer may leave its
            # rows reduce-scattered for the next one to gather back.
            if mapping.has_attn_tp and not (
                self.comm_manager.use_all_reduce(True)
                and self.comm_manager.use_all_reduce(False)
            ):
                raise ValueError(
                    "HY V4 iHC needs full token rows around each sub-block; use "
                    "attention data parallelism (--attn-tp-size 1) or a layout "
                    "where both dense and MoE layers all-reduce."
                )
            self.hc_attn_layer = HYV4HCBoundary(config, is_head=False)
            self.hc_mlp_layer = HYV4HCBoundary(config, is_head=False)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        residual: torch.Tensor | IhcPost | None,
    ) -> tuple[torch.Tensor, torch.Tensor | IhcPost | None]:
        num_global_tokens, max_num_tokens_per_gpu = self.comm_manager.get_num_tokens(
            ctx
        )
        if self.enable_ihc:
            return self._forward_ihc(
                positions,
                hidden_states,
                ctx,
                residual,
                num_global_tokens,
                max_num_tokens_per_gpu,
            )

        if not ctx.forward_mode.is_idle():
            hidden_states, residual = self.comm_manager.input_reduce_norm(
                hidden_states, residual
            )
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                ctx=ctx,
                comm_manager=self.comm_manager,
            )
            hidden_states, residual = self.post_attention_norm(
                hidden_states, residual, ctx
            )
        hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
        hidden_states = self._run_mlp(
            hidden_states, num_global_tokens, max_num_tokens_per_gpu
        )
        hidden_states, residual = self.comm_manager.post_mlp_fused(
            hidden_states, residual, ctx
        )
        return hidden_states, residual

    def post_attention_norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        ctx: ForwardContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize the attention residual before its separate RMSNorm."""
        hidden_states, residual = self.comm_manager.post_attn_comm(
            hidden_states, residual, ctx
        )
        residual = hidden_states + residual
        return self.post_attention_layernorm(residual), residual

    def _run_mlp(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        if self.is_moe_layer:
            return self.mlp(hidden_states, num_global_tokens, max_num_tokens_per_gpu)
        return self.mlp(hidden_states)

    def _forward_ihc(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        pending: IhcPost | None,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> tuple[torch.Tensor, IhcPost | None]:
        """iHC forward: each sub-block reduces the channels, then re-scatters.

        ``pending`` is the previous layer's MLP post boundary, applied with the
        attention pre boundary; this layer's own is returned the same way.
        """
        if ctx.forward_mode.is_idle():
            # Idle ranks hold no tokens but must join the MoE collectives.
            # Mirror the active path's collectives exactly (pre gather, MoE,
            # post reduce-scatter); skipping the post comm deadlocks DP ranks.
            empty = hidden_states.new_empty(
                (0, self.config.hidden_size), dtype=torch.bfloat16
            )
            mlp_output = self._run_mlp(
                self.comm_manager.pre_mlp_comm(empty, ctx),
                num_global_tokens,
                max_num_tokens_per_gpu,
            )
            self.comm_manager.post_mlp_comm(mlp_output, mlp_output, ctx)
            return hidden_states, None

        hidden_states, attn_input, attn_post = self._ihc_pre(
            hidden_states, pending, self.hc_attn_layer, self.input_layernorm
        )
        attn_output = self.self_attn(
            positions=positions,
            hidden_states=attn_input,
            ctx=ctx,
            comm_manager=self.comm_manager,
        )
        attn_output, _ = self.comm_manager.post_attn_comm(attn_output, attn_output, ctx)

        hidden_states, mlp_input, mlp_post = self._ihc_pre(
            hidden_states,
            IhcPost(attn_output, attn_post),
            self.hc_mlp_layer,
            self.post_attention_layernorm,
        )
        mlp_output = self._run_mlp(
            self.comm_manager.pre_mlp_comm(mlp_input, ctx),
            num_global_tokens,
            max_num_tokens_per_gpu,
        )
        mlp_output, _ = self.comm_manager.post_mlp_comm(mlp_output, mlp_output, ctx)
        return hidden_states, IhcPost(mlp_output, mlp_post)

    def _ihc_pre(
        self,
        hidden_states: torch.Tensor,
        pending: IhcPost | None,
        boundary: HYV4HCBoundary,
        norm: RMSNorm,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply ``pending``, then reduce the channels for ``boundary``'s sub-block.

        Returns the channels after ``pending``, the sub-block input and its post
        gates.
        """
        args = (
            boundary.hc_fn,
            boundary.hc_scale,
            boundary.hc_base,
            self.config.hc_magnitude,
            self.config.rms_norm_eps,
            self.config.hc_eps,
        )
        if pending is not None:
            hidden_states = residual_ops.ihc_post(
                pending.block_output, hidden_states, pending.gates
            )
        block_input, gates = residual_ops.ihc_pre(
            hidden_states,
            *args,
            norm_weight=norm.weight,
            norm_eps=norm.variance_epsilon,
        )
        return hidden_states, block_input, gates


class HYV4Model(nn.Module):
    """HY V4 backbone: iHC channels over the whole decoder stack."""

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        check_hyv4_quant_config(quant_config)
        check_hyv4_speculative_algorithm()
        check_hyv4_runtime(mapping, enable_ihc=config.enable_ihc)
        self.config = config
        self.mapping = mapping
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size

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
                HYV4DecoderLayer(
                    config,
                    layer_id,
                    mapping=mapping,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                    alt_stream=self.alt_stream,
                    is_nextn=False,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = hyv4_rmsnorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_head: HYV4HCBoundary | None = (
            HYV4HCBoundary(config, is_head=True) if config.enable_ihc else None
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        residual = None
        if self.hc_head is not None:
            # Keep the embedding dtype across channels so every sub-block
            # materializes its residual before the next pre boundary.
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.config.hc_mult, 1)

        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, ctx, residual)

        if ctx.forward_mode.is_idle():
            if self.hc_head is not None:
                hidden_states = hidden_states.new_empty(
                    (0, self.config.hidden_size), dtype=torch.bfloat16
                )
            return hidden_states, None

        if self.hc_head is not None:
            if residual is not None:
                hidden_states = residual_ops.ihc_post(
                    residual.block_output,
                    hidden_states,
                    residual.gates,
                )
            hidden_states = residual_ops.ihc_head(
                hidden_states,
                self.hc_head.hc_fn,
                self.hc_head.hc_scale,
                self.hc_head.hc_base,
                self.config.rms_norm_eps,
                self.config.hc_eps,
            )
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.layers[-1].comm_manager.final_norm(
                hidden_states, residual, ctx, self.norm
            )
        aux_hidden_states = None
        if (
            ctx.capture_hidden_mode is not None
            and ctx.capture_hidden_mode.need_capture()
        ):
            aux_hidden_states = [hidden_states]
        return hidden_states, aux_hidden_states


class HYV4ForCausalLM(GlmMoeDsaForCausalLM):
    """HY V4 causal LM: GLM-DSA loading conventions plus HY V4's extras.

    ``enable_lm_head_fp32`` is honored by accumulating the head GEMM in FP32
    (``LogitsProcessor`` with ``head_dtype="float32"``) while the weight stays
    BF16, which reproduces the reference FP32 head without an FP32 weight copy.
    """

    model_cls = HYV4Model
    # The loader retains dense ModelOpt MXFP8 weights for Marlin W8A16.
    supports_modelopt_mxfp8 = True
    supports_expert_placement = False
    # Loading derives tensors a live update would leave stale: the absorbed
    # MLA weights and the MoE kernels' weight layouts.
    supports_live_weight_updates = False

    def resolve_lm_head(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> nn.Module:
        # The FP32 head reads a BF16 weight, and HY V4 checkpoints never
        # quantize ``lm_head``; a quantized head would bypass both.
        return super().resolve_lm_head(config, None, prefix)

    def post_init(self) -> None:
        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_routed_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, HYV4MoE)
            }
        )

    def set_eagle3_layers_to_capture(self, layer_ids: list[int] | None = None) -> None:
        raise ValueError(
            "HY V4 does not capture intermediate hidden states for an EAGLE3 "
            "draft; use --speculative-algorithm MTP."
        )

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        raise ValueError(
            "HY V4 does not capture intermediate hidden states for a DFLASH "
            "draft; use --speculative-algorithm MTP."
        )

    @property
    def checkpoint_load_group(self) -> tuple[int, ...]:
        """TP peers consume the same filtered target or MTP checkpoint stream."""
        return self.mapping.attn.tp_group

    def checkpoint_weight_name_filter(self, name: str) -> bool:
        """Skip the MTP block, which the NextN draft model loads."""
        return not name.startswith("model.mtp_layers.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        load_hyv4_weights(
            self,
            (
                (name, tensor)
                for name, tensor in weights
                if self.checkpoint_weight_name_filter(name)
            ),
            module_prefix="model",
            checkpoint_prefix="model",
        )
        self.post_load_weights()

    def post_load_weights(self) -> None:
        for layer in self.model.layers:
            finalize_hyv4_attention(layer.self_attn)


def check_hyv4_quant_config(quant_config: QuantizationConfig | None) -> None:
    """Require the checkpoint format supported by the HY4 MoE backend.

    The baseline covers the released ModelOpt MXFP8 checkpoint. Unquantized
    MoE kernels do not yet provide this model's EP, clamp and fixed-order
    reduction together, so a BF16 checkpoint must not silently pick one.
    """
    if not isinstance(quant_config, ModelOptMxfp8Config):
        format_name = (
            "unquantized BF16" if quant_config is None else quant_config.get_name()
        )
        raise ValueError(
            "HY V4 currently supports ModelOpt MXFP8 checkpoints only; "
            f"{format_name!r} is unsupported"
        )


def check_hyv4_speculative_algorithm() -> None:
    """Reject draft models that read hidden states HY V4 does not capture.

    HY V4 drafts with the checkpoint's own MTP layer. EAGLE3, DFlash and
    DSpark drafts would otherwise fail only once the target and the draft
    have both loaded.
    """
    algorithm = global_server_args_dict["speculative_algorithm"]
    if algorithm not in (None, "MTP"):
        raise ValueError(
            "HY V4 supports MTP speculative decoding only "
            f"(--speculative-algorithm MTP), got {algorithm}."
        )


def finalize_hyv4_attention(attn: HYV4Attention) -> None:
    """Derive the absorbed MLA weights and publish the sink after loading."""
    from tokenspeed.runtime.layers.dense.mxfp8 import ModelOptMxfp8LinearMethod

    method = attn.kv_b_proj.quant_method
    weight = (
        method.weight_for_absorption(attn.kv_b_proj)
        if isinstance(method, ModelOptMxfp8LinearMethod)
        else attn.kv_b_proj.weight
    )
    attn.w_kc, attn.w_vc = _prepare_mla_kv_b_proj_weights(weight, attn)
    attn.bind_learnable_sink()


def load_hyv4_weights(
    owner: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    module_prefix: str,
    checkpoint_prefix: str,
    shared_params: tuple[str, ...] = (),
) -> None:
    """Route HY V4 checkpoint tensors onto ``owner``'s module tree.

    Handles the four layout differences from the GLM-DSA loader: the iHC tensor
    names, the packed ``experts.gate_up_proj`` / ``experts.down_proj`` tensors
    (every expert in one tensor), the fused indexer ``wk`` + ``weights_proj``
    projection, and the fused MLA ``q_a_proj`` + ``kv_a_proj_with_mqa`` input.

    Args:
        owner: Model whose parameters receive the tensors.
        weights: ``(checkpoint_name, tensor)`` pairs.
        module_prefix: Module path replacing ``checkpoint_prefix`` in names.
        checkpoint_prefix: Checkpoint path prefix of the tensors to load.
        shared_params: Name prefixes of parameters bound from another model
            after loading (the MTP draft's embedding and LM head), which the
            checkpoint is not expected to fill.

    Raises:
        RuntimeError: A parameter outside ``shared_params`` is left unloaded,
            including a fused parameter that received only one of its shards.
    """
    config = owner.config
    params_dict = dict(owner.named_parameters())
    modules_dict = dict(owner.named_modules())
    loaded_params: set[str] = set()
    # Fused parameter -> shard ids loaded so far. Every fused parameter here
    # takes shards 0 and 1 and counts as loaded only once both have arrived.
    loaded_shards: dict[str, set[int]] = {}
    unmatched: list[str] = []
    pending_indexer_wk: dict[str, dict[str, torch.Tensor]] = {}

    def check_fused_shard(param_name: str, shard_id: int) -> None:
        if shard_id in loaded_shards.get(param_name, set()):
            raise RuntimeError(f"Duplicate fused shard {shard_id}: {param_name}")

    def record_fused_shard(param_name: str, shard_id: int) -> bool:
        """Record one shard of a fused parameter; True once both are loaded."""
        shards = loaded_shards.setdefault(param_name, set())
        if shard_id in shards:
            raise RuntimeError(f"Duplicate fused shard {shard_id}: {param_name}")
        shards.add(shard_id)
        complete = shards == {0, 1}
        if complete:
            loaded_params.add(param_name)
        return complete

    # Packed tensors only: each one fills its parameter for every local expert,
    # so the parameter name alone tracks completeness.
    moe_loader = build_moe_checkpoint_loader(
        params_dict=params_dict,
        fused_schema=ExpertCheckpointSchema(
            gate_up_fused_name="gate_up_proj",
            down_proj_name="down_proj",
        ),
        num_experts=config.n_routed_experts,
        ep_rank=owner.mapping.moe.ep_rank,
        ep_size=owner.mapping.moe.ep_size,
    )
    weights = preprocess_modelopt_mxfp8_weights(
        weights,
        owner.quant_config,
        keeps_mxfp8=lambda name: ".mlp.experts." in name,
    )

    for name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in name:
            continue
        if checkpoint_prefix != module_prefix and name.startswith(checkpoint_prefix):
            name = module_prefix + name[len(checkpoint_prefix) :]
        name = normalize_hyv4_checkpoint_name(name)
        if name in loaded_params:
            raise RuntimeError(f"Duplicate HY V4 checkpoint tensor: {name}")

        if ".indexer.wk." in name or ".indexer.weights_proj." in name:
            if name.endswith(".bias"):
                continue
            module_name, shard_id = (
                (name.rsplit(".wk.", 1)[0], 0)
                if ".indexer.wk." in name
                else (name.rsplit(".weights_proj.", 1)[0], 1)
            )
            direct = params_dict.get(name)
            if direct is not None:
                getattr(direct, "weight_loader", default_weight_loader)(
                    direct, loaded_weight
                )
                loaded_params.add(name)
            suffix = name.rsplit(".", 1)[-1]
            fused_name = f"{module_name}.wk_weights_proj.weight"
            fused = params_dict.get(fused_name)
            # Reference ModelOpt MXFP8 indexer: wk is FP8, weights_proj BF16,
            # but both are read by one BF16 GEMM. Keep the raw pair only until
            # both tensors arrive, dequantize once and load fused shard zero.
            is_wk = shard_id == 0
            if (
                fused is not None
                and is_wk
                and (
                    suffix == "weight_scale"
                    or (
                        suffix == "weight"
                        and loaded_weight.dtype == torch.float8_e4m3fn
                    )
                )
            ):
                pending = pending_indexer_wk.setdefault(module_name, {})
                part = "scale" if suffix == "weight_scale" else "weight"
                if part in pending:
                    raise RuntimeError(f"Duplicate indexer wk {part}: {module_name}")
                pending[part] = loaded_weight
                if len(pending) == 2:
                    weight = dequantize_mxfp8_weight(
                        pending["weight"], pending["scale"]
                    )
                    check_fused_shard(fused_name, 0)
                    fused.weight_loader(fused, weight, 0)
                    del pending_indexer_wk[module_name]
                    module = modules_dict.get(module_name)
                    if record_fused_shard(fused_name, 0) and isinstance(
                        module, GlmDsaIndexer
                    ):
                        module.set_wk_weights_proj_loaded(True)
                continue
            if fused is not None and suffix == "weight":
                check_fused_shard(fused_name, shard_id)
                fused.weight_loader(fused, loaded_weight, shard_id)
                module = modules_dict.get(module_name)
                if record_fused_shard(fused_name, shard_id) and isinstance(
                    module, GlmDsaIndexer
                ):
                    module.set_wk_weights_proj_loaded(True)
            elif direct is None:
                unmatched.append(name)
            continue

        if ".mlp.experts." in name:
            if name.endswith("_proj_scale"):
                # MXFP8 packed experts: the E8M0 scales feed the MoE
                # ``*_weight_scale_inv`` parameters.
                name += "_inv"
            if moe_loader.matches(name):
                loaded_params.add(moe_loader.load(name, loaded_weight))
            else:
                unmatched.append(name)
            continue

        if ".q_a_proj." in name or ".kv_a_proj_with_mqa." in name:
            is_q = ".q_a_proj." in name
            mapped = name.replace(
                ".q_a_proj." if is_q else ".kv_a_proj_with_mqa.",
                ".fused_qkv_a_proj_with_mqa.",
            )
            begin_size = 0 if is_q else config.q_lora_rank
            param = params_dict.get(mapped)
            if param is None:
                unmatched.append(name)
            else:
                check_fused_shard(mapped, 0 if is_q else 1)
                param.weight_loader(param, loaded_weight, begin_size=begin_size)
                record_fused_shard(mapped, 0 if is_q else 1)
            continue

        stacked = False
        for param_part, weight_part, shard_id in _STACKED_DENSE_PROJECTIONS:
            if weight_part not in name:
                continue
            stacked_name = name.replace(weight_part, param_part)
            param = params_dict.get(stacked_name)
            if param is None:
                unmatched.append(name)
            else:
                check_fused_shard(stacked_name, shard_id)
                param.weight_loader(param, loaded_weight, shard_id)
                record_fused_shard(stacked_name, shard_id)
            stacked = True
            break
        if stacked:
            continue

        param = params_dict.get(name)
        if param is None:
            unmatched.append(name)
            continue
        if (
            loaded_weight.dtype == torch.float8_e4m3fn
            and param.dtype != torch.float8_e4m3fn
        ):
            raise RuntimeError(
                f"Unexpected FP8 tensor for unquantized HY V4 parameter: {name}"
            )
        getattr(param, "weight_loader", default_weight_loader)(param, loaded_weight)
        loaded_params.add(name)

    if pending_indexer_wk:
        raise RuntimeError(
            f"HY V4 indexer FP8 wk weights/scales are incomplete: "
            f"{sorted(pending_indexer_wk)}"
        )
    unmatched_scales = [
        name
        for name in unmatched
        if name.endswith(
            (".weight_scale", ".weight_scale_inv", "_proj_scale", "_proj_scale_inv")
        )
    ]
    if unmatched_scales:
        raise RuntimeError(f"Unmatched HY V4 quantization scales: {unmatched_scales}")
    if unmatched:
        logger.warning(
            f"HY V4 loader skipped {len(unmatched)} checkpoint tensors with no "
            f"matching parameter, e.g. {unmatched[:8]}"
        )
    # Parameters are allocated uninitialized, so one the checkpoint never
    # reaches, or fills only in part, would serve garbage.
    missing = sorted(
        name
        for name in params_dict
        if name not in loaded_params and not name.startswith(shared_params)
    )
    if missing:
        raise RuntimeError(
            f"HY V4 checkpoint left {len(missing)} parameters unloaded, "
            f"e.g. {missing[:8]}"
        )


EntryClass = [HYV4ForCausalLM]
