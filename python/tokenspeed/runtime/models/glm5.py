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

"""Inference-only GLM 5 model."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

import torch
from tokenspeed_kernel.ops.attention import (
    dsa_decode_topk,
    dsa_prefill_topk,
)
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.configs.utils import get_rope_theta
from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    current_forward_ctx,
    slice_to_real_tokens,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, LayerNorm, RMSNorm
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.utils import block_dequant
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3AttentionMLA,
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3MLP,
    DeepseekV3Model,
    DeepseekV3MoE,
    get_layer_id,
)
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import global_server_args_dict

_INDEXER_PREFILL_MAX_LOGITS_MB_ARG = "deepseek_v4_indexer_prefill_max_logits_mb"


@dataclass
class GlmDsaIndexerOutput:
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor


@dataclass
class GlmDsaPrefillTopK:
    workspace_indices: torch.Tensor
    topk_lens: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
    kv_workspace_slots: torch.Tensor


@dataclass
class GlmDsaDecodeTopK:
    topk_indices: torch.Tensor
    topk_lens: torch.Tensor


@dataclass(frozen=True)
class GlmDsaDecodeWindow:
    start: int
    end: int
    num_tokens: int
    num_reqs: int
    q_len_per_req: int


def _glm_dsa_skip_indexer_topk(config, layer_id: int | None) -> bool:
    if layer_id is None:
        return False
    indexer_types = getattr(config, "indexer_types", None)
    if indexer_types is not None and layer_id < len(indexer_types):
        return indexer_types[layer_id] in ("S", "shared")
    pattern = getattr(config, "index_topk_pattern", None)
    if pattern is not None and layer_id < len(pattern):
        return pattern[layer_id] in ("S", "shared")
    freq = int(getattr(config, "index_topk_freq", 1) or 1)
    if freq <= 1:
        return False
    offset = getattr(config, "index_skip_topk_offset", None)
    if offset is None:
        return max(layer_id - 1, 0) % freq != 0
    if offset <= 0:
        raise ValueError(
            "index_skip_topk_offset must be positive; offset <= 0 marks "
            "layer 0 as shared with no prior top-k to reuse"
        )
    return max(layer_id - offset + 1, 0) % freq != 0


def _build_prefill_kv_workspace_slots(
    *,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    page_size: int,
    device: torch.device,
) -> torch.Tensor:
    local_offsets = torch.arange(
        int(max_seq_len),
        dtype=torch.int64,
        device=device,
    )
    page_offsets = torch.div(
        local_offsets,
        int(page_size),
        rounding_mode="floor",
    )
    block_offsets = local_offsets % int(page_size)
    pages = block_tables.to(device=device, dtype=torch.int64).index_select(
        1,
        page_offsets,
    )
    slots = pages * int(page_size) + block_offsets
    valid = local_offsets.unsqueeze(0) < seq_lens.to(
        device=device,
        dtype=torch.int64,
    ).unsqueeze(1)
    return slots[valid].contiguous()


def _glm_dsa_rope_scaling(
    rope_scaling: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not rope_scaling or "factor" not in rope_scaling:
        return None

    rope_scaling = dict(rope_scaling)
    rope_scaling["rope_type"] = "deepseek_yarn"
    return rope_scaling


class GlmDsaIndexer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        q_lora_rank: int,
        qk_rope_head_dim: int,
        rope_theta: float,
        rope_scaling: dict[str, Any] | None,
        max_position_embeddings: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.index_topk = config.index_topk
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = int(qk_rope_head_dim)
        self.softmax_scale = self.index_head_dim**-0.5

        if self.rope_head_dim <= 0 or self.rope_head_dim > self.index_head_dim:
            raise ValueError(
                "GLM DSA indexer requires 0 < qk_rope_head_dim <= index_head_dim; "
                f"got qk_rope_head_dim={self.rope_head_dim}, "
                f"index_head_dim={self.index_head_dim}"
            )

        self.wq_b = ReplicatedLinear(
            q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.wk = ReplicatedLinear(
            hidden_size,
            self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.index_n_heads,
            bias=False,
            quant_config=None,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.wk_weights_proj = MergedColumnParallelLinear(
            hidden_size,
            [self.index_head_dim, self.index_n_heads],
            bias=False,
            quant_config=None,
            prefix=add_prefix("wk_weights_proj", prefix),
        )
        self._wk_weights_proj_loaded = False
        self.k_norm = LayerNorm(self.index_head_dim, eps=1e-6)

        rope_scaling = _glm_dsa_rope_scaling(rope_scaling)
        self.rotary_emb = get_rope(
            self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=not getattr(config, "indexer_rope_interleave", False),
        )

    def set_wk_weights_proj_loaded(self, loaded: bool = True) -> None:
        self._wk_weights_proj_loaded = bool(loaded)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
    ) -> GlmDsaIndexerOutput:
        index_q = self.wq_b(q_lora)[0]
        index_q = index_q.view(-1, self.index_n_heads, self.index_head_dim)
        if self._wk_weights_proj_loaded:
            key_weights = self.wk_weights_proj(hidden_states)[0]
            index_k, weights = key_weights.split(
                [self.index_head_dim, self.index_n_heads],
                dim=-1,
            )
        else:
            index_k = self.wk(hidden_states)[0]
            weights = self.weights_proj(hidden_states)[0]
        index_k = self.k_norm(index_k)

        q_rope, k_rope = self.rotary_emb(
            positions,
            index_q[..., : self.rope_head_dim],
            index_k[:, None, : self.rope_head_dim],
        )
        # Noops if the RoPE is in-place applied
        index_q[..., : self.rope_head_dim] = q_rope
        index_k[:, : self.rope_head_dim] = k_rope.squeeze(1)

        return GlmDsaIndexerOutput(
            query=index_q,
            key=index_k,
            weights=weights.float() * (self.index_n_heads**-0.5),
        )


class GlmMoeDsaAttention(DeepseekV3AttentionMLA):
    _MLA_KERNEL_BACKENDS = ("trtllm_mla", "tokenspeed_mla", "dsa")
    _RAGGED_PREFILL_BACKENDS = ("trtllm_mla", "tokenspeed_mla", "dsa")
    rope_is_neox_style = False

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
        is_nextn: bool = False,
    ) -> None:
        rope_scaling = _glm_dsa_rope_scaling(rope_scaling)
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
            skip_rope=skip_rope,
        )
        if q_lora_rank is None:
            raise ValueError("GLM DSA requires q_lora_rank.")
        # Let process_weights choose DeepGEMM only after it has transformed
        # FP8 block scales into the layout that kernel expects.
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=1e-6)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=1e-6)
        self.fused_qk_layernorm = FusedRMSNorm(
            self.q_a_layernorm,
            self.kv_a_layernorm,
        )
        self.index_topk = config.index_topk
        self.is_nextn = is_nextn
        # NextN/MTP has its own indexer weights but may reuse the previous
        # draft iteration's top-k. Shared target layers do not have usable
        # indexer weights and must consume the context-carried top-k.
        self.skip_indexer_topk = (
            True if is_nextn else _glm_dsa_skip_indexer_topk(config, layer_id)
        )
        if self.skip_indexer_topk and not self.is_nextn:
            self.indexer = None
        else:
            self.indexer = GlmDsaIndexer(
                config=config,
                hidden_size=hidden_size,
                q_lora_rank=q_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                rope_theta=rope_theta,
                rope_scaling=rope_scaling,
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                prefix=add_prefix("indexer", prefix),
            )
        self._decode_topk_indices_buffer: torch.Tensor | None = None
        self._decode_topk_lens_buffer: torch.Tensor | None = None

    def _get_decode_topk_workspace(
        self,
        attr_name: str,
        rows: int,
        cols: int,
        device: torch.device,
        fill_value: int | None = -1,
    ) -> torch.Tensor:
        buffer = getattr(self, attr_name, None)
        if (
            buffer is None
            or buffer.device != device
            or buffer.shape[0] < rows
            or buffer.shape[1] != cols
        ):
            # A captured CUDA graph may still reference the old buffer; keep
            # it alive so a regrow never frees memory a graph replays into.
            if buffer is not None:
                self._retire_decode_workspace(buffer)
            buffer = torch.empty(
                (rows, cols),
                dtype=torch.int32,
                device=device,
            )
            setattr(self, attr_name, buffer)
        workspace = buffer[:rows]
        if fill_value is not None:
            workspace.fill_(fill_value)
        return workspace

    def _get_decode_topk_lens_workspace(
        self,
        rows: int,
        device: torch.device,
    ) -> torch.Tensor:
        buffer = getattr(self, "_decode_topk_lens_buffer", None)
        if buffer is None or buffer.device != device or buffer.numel() < rows:
            if buffer is not None:
                self._retire_decode_workspace(buffer)
            buffer = torch.empty(
                (rows,),
                dtype=torch.int32,
                device=device,
            )
            self._decode_topk_lens_buffer = buffer
        workspace = buffer[:rows]
        workspace.fill_(0)
        return workspace

    @staticmethod
    def _resolve_decode_q_len(
        ctx: ForwardContext,
        num_decode_tokens: int,
        num_decode_reqs: int,
    ) -> int:
        """Per-request query rows, derived from the actual batch shape.

        Spec-verify and the draft first step can both feed multiple query rows
        per request, while the draft model's later decode steps feed one row.
        The draft attention backend inherits the target verify width from the
        shared config, so trust the actual input row count instead of backend
        metadata.
        """
        if num_decode_reqs > 0 and num_decode_tokens > 0:
            q_len, rem = divmod(int(num_decode_tokens), int(num_decode_reqs))
            if rem == 0 and q_len > 0:
                return q_len
        return 1

    @staticmethod
    def _resolve_num_decode_tokens(
        ctx: ForwardContext,
        *,
        total_tokens: int,
        num_decode_reqs: int,
    ) -> int:
        if num_decode_reqs <= 0 or total_tokens <= 0:
            return 0
        spec_width = int(getattr(ctx.attn_backend, "spec_num_tokens", 1) or 1)
        expected_decode_tokens = num_decode_reqs * spec_width
        return min(int(total_tokens), int(expected_decode_tokens))

    @staticmethod
    def _resolve_decode_req_count(
        ctx: ForwardContext,
        metadata: Any,
    ) -> int:
        num_extends = int(getattr(metadata, "num_extends", 0) or 0)
        limits = [max(0, int(ctx.bs) - int(ctx.num_extends))]

        seq_lens = getattr(metadata, "seq_lens_k", None)
        if seq_lens is not None:
            limits.append(max(0, int(seq_lens.shape[0]) - num_extends))

        block_tables = getattr(metadata, "block_kv_indices", None)
        if block_tables is not None:
            limits.append(max(0, int(block_tables.shape[0]) - num_extends))

        return min(limits)

    @staticmethod
    def _resolve_decode_window(
        ctx: ForwardContext,
        metadata: Any,
        *,
        total_tokens: int,
    ) -> GlmDsaDecodeWindow:
        num_decode_reqs = GlmMoeDsaAttention._resolve_decode_req_count(ctx, metadata)
        num_decode_tokens = GlmMoeDsaAttention._resolve_num_decode_tokens(
            ctx,
            total_tokens=total_tokens,
            num_decode_reqs=num_decode_reqs,
        )
        if total_tokens < num_decode_tokens:
            raise RuntimeError(
                "GLM DSA decode token split is invalid: "
                f"tokens={total_tokens}, decode_tokens={num_decode_tokens}"
            )
        q_len_per_req = GlmMoeDsaAttention._resolve_decode_q_len(
            ctx, num_decode_tokens, num_decode_reqs
        )
        decode_start = int(total_tokens) - int(num_decode_tokens)
        return GlmDsaDecodeWindow(
            start=decode_start,
            end=decode_start + int(num_decode_tokens),
            num_tokens=int(num_decode_tokens),
            num_reqs=int(num_decode_reqs),
            q_len_per_req=int(q_len_per_req),
        )

    @staticmethod
    def _slice_decode_topk(
        decode_topk: GlmDsaDecodeTopK,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return decode_topk.topk_indices[start:end], decode_topk.topk_lens[start:end]

    def _retire_decode_workspace(self, buffer: torch.Tensor) -> None:
        retired = getattr(self, "_retired_decode_workspaces", None)
        if retired is None:
            retired = []
            self._retired_decode_workspaces = retired
        retired.append(buffer)

    @staticmethod
    def _check_decode_q_len_per_req(q_len_per_req: int) -> None:
        # Multi-step MTP verify runs num_draft_tokens query rows per request.
        # DeepGEMM paged MQA logits (our fork) and FlashMLA sparse decode are
        # both verified bit-exact against batch expansion up to next_n = 6,
        # which covers --speculative-num-steps 5 (5 draft + 1 bonus).
        if not 1 <= q_len_per_req <= 6:
            raise NotImplementedError(
                "GLM DSA sparse decode supports 1-6 query tokens per request "
                f"(verified next_n <= 6), got {q_len_per_req}."
            )

    def _compute_decode_topk_indices(
        self,
        indexer_output: GlmDsaIndexerOutput,
        ctx: ForwardContext,
    ) -> GlmDsaDecodeTopK | None:
        metadata = getattr(ctx.attn_backend, "forward_decode_metadata", None)
        if metadata is None or metadata.block_kv_indices is None:
            return None
        num_tokens = indexer_output.query.shape[0]
        decode_window = self._resolve_decode_window(
            ctx, metadata, total_tokens=num_tokens
        )
        if decode_window.num_reqs <= 0 or num_tokens == 0:
            return None
        self._check_decode_q_len_per_req(decode_window.q_len_per_req)

        num_extends = int(metadata.num_extends or 0)
        seq_lens = metadata.seq_lens_k[
            num_extends : num_extends + decode_window.num_reqs
        ]
        if seq_lens.numel() == 0:
            return None

        block_tables = metadata.block_kv_indices[
            num_extends : num_extends + decode_window.num_reqs
        ]
        topk = self.index_topk
        return self._compute_decode_topk_indices_portable(
            indexer_output=indexer_output,
            ctx=ctx,
            seq_lens=seq_lens,
            block_tables=block_tables,
            q_len_per_req=decode_window.q_len_per_req,
            decode_start=decode_window.start,
            num_tokens=num_tokens,
            num_decode_tokens=decode_window.num_tokens,
            topk=topk,
        )

    def _compute_decode_topk_indices_portable(
        self,
        *,
        indexer_output: GlmDsaIndexerOutput,
        ctx: ForwardContext,
        seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        q_len_per_req: int,
        decode_start: int,
        num_tokens: int,
        num_decode_tokens: int,
        topk: int,
    ) -> GlmDsaDecodeTopK:
        q = indexer_output.query[decode_start : decode_start + num_decode_tokens]
        weights = indexer_output.weights[
            decode_start : decode_start + num_decode_tokens
        ]
        index_k_cache = ctx.token_to_kv_pool.get_index_k_buffer(self.attn_mqa.layer_id)
        if index_k_cache is None:
            raise RuntimeError("GLM DSA top-k requires an index-K cache.")

        topk_indices = self._get_decode_topk_workspace(
            "_decode_topk_indices_buffer",
            num_tokens,
            topk,
            q.device,
            fill_value=-1,
        )
        topk_slice = topk_indices[decode_start : decode_start + num_decode_tokens]
        topk_lens = self._get_decode_topk_lens_workspace(num_tokens, q.device)
        topk_lens_slice = topk_lens[decode_start : decode_start + num_decode_tokens]

        metadata = ctx.attn_backend.forward_decode_metadata
        seq_lens_2d = (
            metadata._dsa_seq_lens_2d[ctx.num_extends :]
            if q_len_per_req > 1
            else seq_lens.unsqueeze(1)
        )
        dsa_decode_topk(
            q,
            weights,
            seq_lens,
            block_tables,
            page_size=ctx.token_to_kv_pool.page_size,
            topk=topk,
            softmax_scale=self.indexer.softmax_scale,
            q_len_per_req=q_len_per_req,
            index_k_cache=index_k_cache,
            seq_lens_2d=seq_lens_2d,
            plan=metadata._dsa_plan,
            out=topk_slice,
            lens_out=topk_lens_slice,
        )
        return GlmDsaDecodeTopK(
            topk_indices=topk_indices,
            topk_lens=topk_lens,
        )

    def _compute_prefill_topk_indices(
        self,
        indexer_output: GlmDsaIndexerOutput,
        ctx: ForwardContext,
        num_prefill_tokens: int,
    ) -> GlmDsaPrefillTopK | None:
        chunk_meta = ctx.attn_backend.chunked_prefill_metadata
        prefix_lens = chunk_meta.extend_prefix_lens[: ctx.num_extends].to(torch.int32)
        extend_lens = chunk_meta.extend_seq_lens[: ctx.num_extends].to(torch.int32)
        seq_lens = prefix_lens + extend_lens
        if seq_lens.numel() == 0:
            return None
        if int(extend_lens.sum().item()) != num_prefill_tokens:
            raise RuntimeError(
                "GLM DSA prefill token count mismatch: "
                f"metadata={int(extend_lens.sum().item())}, "
                f"tokens={num_prefill_tokens}"
            )
        if ctx.req_to_page is None:
            raise RuntimeError("GLM DSA sparse prefill requires req_to_page metadata")

        topk = self.index_topk
        page_size = ctx.token_to_kv_pool.page_size
        max_seq_len = int(seq_lens.max().item())
        max_pages = (max_seq_len + page_size - 1) // page_size
        block_tables = chunk_meta.block_tables[:, :max_pages].to(
            device=indexer_output.query.device,
            dtype=torch.int32,
        )
        kv_workspace_slots = _build_prefill_kv_workspace_slots(
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            page_size=page_size,
            device=indexer_output.query.device,
        )
        return self._compute_prefill_topk_indices_portable(
            indexer_output=indexer_output,
            ctx=ctx,
            prefix_lens=prefix_lens,
            extend_lens=extend_lens,
            seq_lens=seq_lens,
            block_tables=block_tables,
            kv_workspace_slots=kv_workspace_slots,
            max_seq_len=max_seq_len,
            num_prefill_tokens=num_prefill_tokens,
            topk=topk,
        )

    def _compute_prefill_topk_indices_portable(
        self,
        *,
        indexer_output: GlmDsaIndexerOutput,
        ctx: ForwardContext,
        prefix_lens: torch.Tensor,
        extend_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        kv_workspace_slots: torch.Tensor,
        max_seq_len: int,
        num_prefill_tokens: int,
        topk: int,
    ) -> GlmDsaPrefillTopK:
        q = indexer_output.query[:num_prefill_tokens].contiguous()
        weights = indexer_output.weights[:num_prefill_tokens].float().contiguous()

        req_ids = torch.arange(
            seq_lens.numel(),
            dtype=torch.int64,
            device=q.device,
        )
        token_req = torch.repeat_interleave(req_ids, extend_lens.to(torch.int64))
        extend_cu = torch.zeros(
            extend_lens.numel() + 1,
            dtype=torch.int64,
            device=q.device,
        )
        torch.cumsum(extend_lens.to(torch.int64), dim=0, out=extend_cu[1:])
        token_offsets = torch.arange(
            num_prefill_tokens, dtype=torch.int64, device=q.device
        ) - extend_cu.index_select(0, token_req)
        causal_lens = (
            prefix_lens.to(torch.int64).index_select(0, token_req) + token_offsets + 1
        )
        seq_cu = torch.zeros(
            seq_lens.numel() + 1,
            dtype=torch.int64,
            device=q.device,
        )
        torch.cumsum(seq_lens.to(torch.int64), dim=0, out=seq_cu[1:])
        row_starts = seq_cu.index_select(0, token_req)
        row_ends = row_starts + causal_lens

        index_k_cache = ctx.token_to_kv_pool.get_index_k_buffer(self.attn_mqa.layer_id)
        if index_k_cache is None:
            raise RuntimeError("GLM DSA top-k requires an index-K cache.")

        max_logits_mb = int(global_server_args_dict[_INDEXER_PREFILL_MAX_LOGITS_MB_ARG])
        workspace_indices, topk_lens = dsa_prefill_topk(
            q,
            weights,
            kv_workspace_slots,
            row_starts.to(torch.int32).contiguous(),
            row_ends.to(torch.int32).contiguous(),
            topk=topk,
            softmax_scale=self.indexer.softmax_scale,
            index_k_cache=index_k_cache,
            page_size=ctx.token_to_kv_pool.page_size,
            max_logits_bytes=max(1, max_logits_mb) * 1024 * 1024,
        )
        return GlmDsaPrefillTopK(
            workspace_indices=workspace_indices,
            topk_lens=topk_lens,
            block_tables=block_tables,
            seq_lens=seq_lens.to(device=q.device, dtype=torch.int32),
            max_seq_len=max_seq_len,
            kv_workspace_slots=kv_workspace_slots,
        )

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
        """GLM-5 DSA attention, one COARSE breakable-graph break point.

        Like DeepSeek-V4 it does paged-cache writes, a data-dependent indexer
        -> top-k stage and the FlashMLA sparse kernel (plus pre-attn
        collectives), none capturable. Under a prefill-graph capture the whole
        attention runs eager (reading the live ``ctx``) while the layer's
        norms + MoE stay graphed; direct call otherwise (see ``break_point``).
        Padded token-shaped inputs are sliced to the real count the live
        metadata describes -- DSA and the decode-window split (which derives
        ``decode_start`` from the total token count) must not see padded rows,
        or decode rows get sliced out of the padded tail. Mirrors the
        DeepSeek-V4 DSA break.
        """
        # Empty (idle / DP-idle) batch: explicit skip, like the sibling MLP/MoE forwards.
        if hidden_states.shape[0] == 0:
            return hidden_states
        qkv = self.fused_qkv_a_proj_with_mqa(
            hidden_states,
            block_scale,
            torch.bfloat16,
        )
        # The fused QKV-A weight may be zero-padded on its output dim to a
        # multiple of 128 so the FP8 block-scale GEMM stays numerically valid
        # (see GlmMoeDsaForCausalLM._pad_fused_qkv_a_proj_for_fp8_blockscale).
        # Drop the padding columns before the split / comm. No-op when the
        # projection output already matches the logical width.
        _qkv_width = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
        if qkv.shape[-1] != _qkv_width:
            qkv = qkv[..., :_qkv_width]
        qkv = comm_manager.pre_attn_comm(qkv, ctx)
        # Slice only under a breakable capture/replay (see the DeepSeek-V4 break):
        # eager forwards (incl. MTP draft steps) are never padded. Sliced AFTER
        # the pre-attn comm: at replay ``_padded_to`` pins ``global_num_tokens``
        # to the padded bucket, so the comm must see padded-length rows; only
        # the DSA stack below needs exactly the real rows.
        _metadata = getattr(ctx.attn_backend, "forward_metadata", None)
        _token_to_req = getattr(_metadata, "token_to_req_indices", None)
        if current_forward_ctx() is not None and _token_to_req is not None:
            positions, qkv, out_cache_loc = slice_to_real_tokens(
                _token_to_req.numel(), positions, qkv, out_cache_loc
            )
        q_a, latent_cache = qkv.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        kv_a = latent_cache[..., : self.kv_lora_rank]
        q_norm = torch.empty_like(q_a)
        if q_a.size(0) > 0:
            self.fused_qk_layernorm(input_q_a=q_a, input_kv_a=kv_a, output_q_a=q_norm)

        decode_metadata = getattr(ctx.attn_backend, "forward_decode_metadata", None)
        num_attn_tokens = int(q_norm.shape[0])
        decode_window = self._resolve_decode_window(
            ctx,
            decode_metadata,
            total_tokens=num_attn_tokens,
        )
        num_decode_tokens = decode_window.num_tokens
        num_prefill_tokens = decode_window.start
        decode_start = decode_window.start
        decode_end = decode_window.end

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
            ctx.token_to_kv_pool.set_index_k_buffer(
                self.attn_mqa.layer_id,
                out_cache_loc,
                indexer_output.key,
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
                input_num_tokens=num_prefill_tokens,
                forward_mode=ForwardMode.EXTEND,
            )
            if ctx.dsa_prefill_topk is None:
                raise RuntimeError(
                    "GLM DSA sparse prefill requires computed top-k indices."
                )
            self.forward_dsa_sparse_prefill(
                positions[:num_prefill_tokens],
                q[:num_prefill_tokens],
                latent_cache[:num_prefill_tokens],
                prefill_ctx,
                out_cache_loc[:num_prefill_tokens],
                attn_output[:num_prefill_tokens],
                prefill_topk=ctx.dsa_prefill_topk,
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
                    "GLM DSA sparse decode requires computed top-k indices."
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

        if ctx.draft_first_step_reduce:
            attn_output = attn_output.index_select(0, ctx.gather_ids)
        output, _ = self.o_proj(attn_output)
        return output

    def forward_dsa_sparse_prefill(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        output: torch.Tensor,
        *,
        prefill_topk: GlmDsaPrefillTopK,
    ) -> torch.Tensor:
        Q, _ = self.forward_absorb_qkv_proj(
            q,
            latent_cache,
            positions,
            ctx,
            out_cache_loc,
        )
        attn_output = ctx.attn_backend.forward_sparse_prefill(
            q=Q,
            layer=self.attn_mqa,
            token_to_kv_pool=ctx.token_to_kv_pool,
            block_tables=prefill_topk.block_tables,
            seq_lens=prefill_topk.seq_lens,
            workspace_indices=prefill_topk.workspace_indices,
            topk_lens=prefill_topk.topk_lens,
            kv_workspace_slots=prefill_topk.kv_workspace_slots,
            max_seq_len=prefill_topk.max_seq_len,
        )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
        output_view = output.view(-1, self.num_local_heads, self.v_head_dim)
        torch.bmm(
            attn_output.transpose(0, 1),
            self.w_vc,
            out=output_view.transpose(0, 1),
        )
        return output

    def forward_absorb(
        self,
        positions: torch.Tensor,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        output: torch.Tensor,
        topk_indices: torch.Tensor | None = None,
        topk_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        Q, K = self.forward_absorb_qkv_proj(
            q,
            latent_cache,
            positions,
            ctx,
            out_cache_loc,
        )
        return self.forward_absorb_attn_v_proj(
            Q,
            K,
            ctx,
            out_cache_loc,
            output,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
        )

    def forward_absorb_attn_v_proj(
        self,
        Q,
        K,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        output: torch.Tensor,
        topk_indices: torch.Tensor | None = None,
        topk_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        need_save_kv = False
        if self.attention_backend not in self._MLA_KERNEL_BACKENDS:
            need_save_kv = not self.use_fused_set_kv_buffer

        attn_output = self.attn_mqa(
            Q,
            K,
            K[..., : self.kv_lora_rank],
            ctx,
            out_cache_loc,
            save_kv_cache=need_save_kv,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
        )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
        output_view = output.view(-1, self.num_local_heads, self.v_head_dim)
        torch.bmm(
            attn_output.transpose(0, 1),
            self.w_vc,
            out=output_view.transpose(0, 1),
        )
        return output


class GlmMoeDsaDecoderLayer(DeepseekV3DecoderLayer):
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
        nn.Module.__init__(self)
        self.mapping = mapping
        self.hidden_size = config.hidden_size
        rope_theta = get_rope_theta(config)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.self_attn = GlmMoeDsaAttention(
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
            is_nextn=is_nextn,
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
            if ctx.draft_first_step_reduce:
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
        else:
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


class GlmMoeDsaModel(DeepseekV3Model):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.mapping = mapping
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.alt_stream = torch.cuda.Stream()
        self.layers = nn.ModuleList(
            [
                GlmMoeDsaDecoderLayer(
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
        self.layers_to_capture: set = set()


def pad_fused_qkv_a_proj_weight_for_fp8_blockscale(attn) -> None:
    """Pad one attention module's fused QKV-A projection output dim to 128.

    The FP8 block-scale dense GEMM (deep_gemm / default ``mm`` path) returns NaN
    when the output dim ``N`` is not a multiple of the 128 scale block. GLM-5.1's
    fused QKV-A projection has ``N = q_lora_rank + kv_lora_rank + qk_rope_head_dim``
    (e.g. 2624), which is not 128-aligned, so attention output goes NaN. We
    zero-pad the FP8 weight rows up to the next 128 multiple;
    ``weight_scale_inv`` already has ``ceil(N/128)`` row blocks (covering the
    padded rows) and the downstream ``qkv.split(...)`` drops the padding rows, so
    real outputs are unchanged. No-op for bf16 weights or already-aligned ``N``.

    Shared by the main model (per decoder layer) and the NextN draft model (its
    single DSA decoder), both of which carry the same fused QKV-A projection.

    Args:
        attn: A GLM DSA attention module exposing ``fused_qkv_a_proj_with_mqa``.
    """
    fp8_dtypes = (torch.float8_e4m3fn, getattr(torch, "float8_e4m3fnuz", None))
    fp8_dtypes = tuple(d for d in fp8_dtypes if d is not None)
    proj = getattr(attn, "fused_qkv_a_proj_with_mqa", None)
    weight = getattr(proj, "weight", None)
    if weight is None or weight.dtype not in fp8_dtypes:
        return
    n = weight.shape[0]
    if n % 128 == 0:
        return
    n_pad = ((n + 127) // 128) * 128
    pad = weight.new_zeros(n_pad - n, weight.shape[1])
    proj.weight = torch.nn.Parameter(
        torch.cat([weight.data, pad], dim=0), requires_grad=False
    )


class GlmMoeDsaForCausalLM(DeepseekV3ForCausalLM):
    model_cls = GlmMoeDsaModel

    def _record_fused_indexer_projection_shard(
        self,
        *,
        module_name: str,
        shard_id: int,
        loaded_shards: dict[str, set[int]],
        modules_dict: dict[str, nn.Module],
    ) -> None:
        shards = loaded_shards.setdefault(module_name, set())
        shards.add(int(shard_id))
        if shards != {0, 1}:
            return

        module = modules_dict.get(module_name)
        if isinstance(module, GlmDsaIndexer):
            module.set_wk_weights_proj_loaded()

    def _load_fused_indexer_projection_shard(
        self,
        *,
        module_name: str,
        shard_id: int,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, torch.Tensor],
        modules_dict: dict[str, nn.Module],
        loaded_shards: dict[str, set[int]],
    ) -> bool:
        param = params_dict.get(f"{module_name}.wk_weights_proj.weight")
        if param is None:
            return False

        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight, shard_id)
        self._record_fused_indexer_projection_shard(
            module_name=module_name,
            shard_id=shard_id,
            loaded_shards=loaded_shards,
            modules_dict=modules_dict,
        )
        return True

    def _flush_fused_indexer_fp8_wk(
        self,
        *,
        module_name: str,
        pending_fp8_wk: dict[str, dict[str, torch.Tensor]],
        params_dict: dict[str, torch.Tensor],
        modules_dict: dict[str, nn.Module],
        loaded_shards: dict[str, set[int]],
    ) -> None:
        entry = pending_fp8_wk.get(module_name)
        if not entry or "weight" not in entry or "scale" not in entry:
            return
        weight_block_size = getattr(self.quant_config, "weight_block_size", None)
        if weight_block_size is None:
            return

        weight_fp8 = entry["weight"]
        scale = entry["scale"]
        weight_bf16 = block_dequant(
            weight_fp8,
            scale,
            list(weight_block_size),
        ).to(torch.bfloat16)
        if self._load_fused_indexer_projection_shard(
            module_name=module_name,
            shard_id=0,
            loaded_weight=weight_bf16,
            params_dict=params_dict,
            modules_dict=modules_dict,
            loaded_shards=loaded_shards,
        ):
            del pending_fp8_wk[module_name]

    def _try_load_fused_indexer_projection(
        self,
        *,
        name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, torch.Tensor],
        modules_dict: dict[str, nn.Module],
        pending_fp8_wk: dict[str, dict[str, torch.Tensor]],
        loaded_shards: dict[str, set[int]],
    ) -> None:
        if ".indexer.wk_weights_proj." in name:
            return

        if ".indexer.weights_proj.weight" in name:
            module_name = name.rsplit(".weights_proj.weight", 1)[0]
            self._load_fused_indexer_projection_shard(
                module_name=module_name,
                shard_id=1,
                loaded_weight=loaded_weight,
                params_dict=params_dict,
                modules_dict=modules_dict,
                loaded_shards=loaded_shards,
            )
            return

        if ".indexer.wk." not in name:
            return

        module_name = name.rsplit(".wk.", 1)[0]
        if name.endswith(".weight") and loaded_weight.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ):
            pending_fp8_wk.setdefault(module_name, {})["weight"] = loaded_weight
            self._flush_fused_indexer_fp8_wk(
                module_name=module_name,
                pending_fp8_wk=pending_fp8_wk,
                params_dict=params_dict,
                modules_dict=modules_dict,
                loaded_shards=loaded_shards,
            )
            return

        if name.endswith(".weight"):
            self._load_fused_indexer_projection_shard(
                module_name=module_name,
                shard_id=0,
                loaded_weight=loaded_weight,
                params_dict=params_dict,
                modules_dict=modules_dict,
                loaded_shards=loaded_shards,
            )
            return

        if "weight_scale_inv" in name:
            pending_fp8_wk.setdefault(module_name, {})["scale"] = loaded_weight
            self._flush_fused_indexer_fp8_wk(
                module_name=module_name,
                pending_fp8_wk=pending_fp8_wk,
                params_dict=params_dict,
                modules_dict=modules_dict,
                loaded_shards=loaded_shards,
            )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        modules_dict = dict(self.named_modules())
        pending_fp8_wk: dict[str, dict[str, torch.Tensor]] = {}
        loaded_fused_indexer_shards: dict[str, set[int]] = {}

        def base_weights():
            for name, loaded_weight in weights:
                layer_id = get_layer_id(name)
                if layer_id is not None and layer_id >= self.config.num_hidden_layers:
                    continue
                if "rotary_emb.inv_freq" in name:
                    continue
                if ".indexer." not in name:
                    yield name, loaded_weight
                    continue

                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = self.get_param(params_dict, name)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                self._try_load_fused_indexer_projection(
                    name=name,
                    loaded_weight=loaded_weight,
                    params_dict=params_dict,
                    modules_dict=modules_dict,
                    pending_fp8_wk=pending_fp8_wk,
                    loaded_shards=loaded_fused_indexer_shards,
                )

        super().load_weights(base_weights())
        self._pad_fused_qkv_a_proj_for_fp8_blockscale()

    def _pad_fused_qkv_a_proj_for_fp8_blockscale(self) -> None:
        """Pad each decoder layer's fused QKV-A projection to a 128-multiple.

        See :func:`pad_fused_qkv_a_proj_weight_for_fp8_blockscale` for why this
        is needed (FP8 block-scale GEMM returns NaN for non-128-aligned ``N``).
        """
        for layer in getattr(self.model, "layers", []):
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                pad_fused_qkv_a_proj_weight_for_fp8_blockscale(attn)


EntryClass = [GlmMoeDsaForCausalLM]
