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

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

import torch
from tokenspeed_kernel.ops.activation.triton import sigmoid_mul
from tokenspeed_kernel.ops.attention.dsa import (
    dsa_decode_topk,
    dsa_prefill_topk,
)
from tokenspeed_kernel.ops.attention.dsa.triton import (
    local_topk_to_global_slots,
    workspace_topk_to_global_slots,
)
from tokenspeed_kernel.ops.quantization import quantize_fp8
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.configs.numerics import BITWISE_ENVELOPES
from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.comm_ops import all_gather_single
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    current_valid_rows,
    slice_to_real_tokens,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.dcp.indexer import (
    DCPIndexMetadata,
    build_dcp_index_metadata,
    select_dsa_topk,
)
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    resolve_cache_slots,
)
from tokenspeed.runtime.layers.attention.page_table import (
    build_prefill_kv_workspace_slots,
)
from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, LayerNorm, RMSNorm
from tokenspeed.runtime.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.utils import block_dequant
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.layers.utils import get_layer_id
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base.decoder_layer import capture_layer_input
from tokenspeed.runtime.models.deepseek_v3 import (
    DeepseekV3AttentionMLA,
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
    DeepseekV3Model,
)
from tokenspeed.runtime.utils import add_prefix, set_weight_attrs
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.device_stream import is_current_stream_capturing
from tokenspeed.runtime.utils.env import global_server_args_dict

_INDEXER_PREFILL_MAX_LOGITS_MB_ARG = "deepseek_v4_indexer_prefill_max_logits_mb"
# Fewer rows: the all-gather costs more than split scoring saves (TP8 50K, MI350X).
_PREFILL_TOPK_TP_SPLIT_MIN_ROWS = 384
_PREFILL_TOPK_ROW_ALIGN = 16
# Below this many rows every rank merges every KVP prefill row: the row-owner merge's two extra collectives cost more.
_KVP_PREFILL_ROW_OWNER_MIN_ROWS = 1024


def _int32_to_device(
    tensors: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    """Copy 1-D host integer tensors to ``device`` as int32 in one async copy.

    A pageable host-to-device copy blocks the host until the device has
    drained its queue, so the host loses its lead over the device for the
    rest of the forward. The tensors go through one pinned staging buffer
    instead; each device view starts 64-byte aligned.
    """
    align = _PREFILL_TOPK_ROW_ALIGN
    starts = [0]
    for tensor in tensors:
        starts.append(starts[-1] + -(-tensor.numel() // align) * align)
    staging = torch.empty(starts[-1], dtype=torch.int32, pin_memory=True)
    for tensor, start in zip(tensors, starts):
        staging[start : start + tensor.numel()] = tensor
    on_device = staging.to(device, non_blocking=True)
    return tuple(
        on_device[start : start + tensor.numel()]
        for tensor, start in zip(tensors, starts)
    )


def _prefill_query_requests(
    seq_lens: torch.Tensor, row_starts: torch.Tensor
) -> torch.Tensor:
    """The request of every prefill query row, as int32: the request whose
    span of the request-major history holds the row's start."""
    return torch.searchsorted(
        torch.cumsum(seq_lens.to(device=row_starts.device, dtype=torch.int64), 0),
        row_starts.long(),
        right=True,
    ).to(torch.int32)


@dataclass
class GlmDsaIndexerOutput:
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    # When set, key is pre-norm and the index-K write applies this LayerNorm.
    key_norm: LayerNorm | None = None
    # When set, query holds input rows and ranks project only their top-k share.
    query_proj: Callable[[torch.Tensor], torch.Tensor] | None = None
    # When set, the decode rows' E4M3 query and FP32 weights folding its scales and the score scale.
    quantized_query: tuple[torch.Tensor, torch.Tensor] | None = None

    def query_rows(self, start: int, end: int) -> torch.Tensor:
        rows = self.query[start:end]
        return rows if self.query_proj is None else self.query_proj(rows)


@dataclass
class GlmDsaPrefillTopK:
    # int32, or int16 holding unsigned 16-bit indices with -1 where empty.
    workspace_indices: torch.Tensor
    topk_lens: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    kv_seq_lens: torch.Tensor
    max_seq_len: int
    kv_workspace_slots: torch.Tensor
    row_starts: torch.Tensor | None = None
    row_ends: torch.Tensor | None = None
    candidate_lens_cpu: torch.Tensor | None = None
    # DCP index metadata for owned-width tiles, reused by later indexer layers.
    dcp_index: DCPIndexMetadata | None = None


@dataclass
class GlmDsaDecodeTopK:
    topk_indices: torch.Tensor
    topk_lens: torch.Tensor
    # DCP index metadata and its source, reused by the forward's later indexer layers.
    dcp_index: Any = None
    dcp_source: Any = None
    dcp_q_len: int = 0
    dcp_capturing: bool = False


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
        # Top-k scale with the per-head normalization folded in, so forward
        # needs no separate weights-scale kernel.
        self.weights_softmax_scale = self.softmax_scale * (self.index_n_heads**-0.5)

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
            # Raw BF16 weights: the head-normalization constant is folded into
            # weights_softmax_scale, while the top-k scorer upcasts each signed
            # head weight before the per-head ReLU reduction.
            weights=weights,
        )


class GlmDsaSparseAttentionMixin:
    """DSA sparse MLA shared by the GLM-DSA and Kimi-K3.1 attention layers.

    The index-K write, the decode/prefill top-k selection, and the sparse
    prefill and decode cores. A host class derives from this mixin and a
    ``DeepseekV3AttentionMLA`` subclass, and declares in its ``__init__``:
    ``indexer`` (its forward returns :class:`GlmDsaIndexerOutput`, and it has
    ``weights_softmax_scale``), ``index_topk``, ``skip_indexer_topk``,
    ``is_nextn``, the decode top-k workspaces
    ``_decode_topk_indices_buffer`` / ``_decode_topk_lens_buffer`` /
    ``_retired_decode_workspaces``, ``prefill_topk_tp_split``, and
    ``decode_topk_fork``: a :class:`StreamFork` whose open scope runs a
    decode round's top-k beside the query absorption (``StreamFork(None)``
    keeps it on the main stream).

    ``prefill_topk_tp_split`` lets a host whose indexer is replicated on every
    attention-TP rank divide the prefill top-k rows among those ranks: each
    rank selects for its contiguous slice of rows and an all-gather assembles
    the selection every rank's sparse attention reads. A host sets it only
    when every rank holds the same indexer output.

    ``index_query_fp8`` (default off) makes the paged top-k scorers of an
    ``fp8_scaled`` plane take FP8 E4M3 index queries with per-(token, head)
    scales (:meth:`_scorer_index_query`); a host sets it only where the
    platform registers such scorers.

    ``decode_topk_solution`` (default None: normal selection) pins the
    tensor-parallel decode top-k to one ``dsa_decode_topk`` solution; a host
    sets it only to a solution it validated and that selects on the platform.

    ``index_k_write_quantizes_query`` (default off) quantizes a decode round's
    index query in the launch of its index-K write (an ``fp8_scaled`` plane,
    keys with ``key_norm``) and hands the decode top-k E4M3 queries with unit
    scales; a host sets it only where its ``decode_topk_solution`` takes them.

    A host that hands ``forward_dsa_core`` the index keys and head weights it
    projected itself (``index_key_weights``) gives its indexer
    ``from_key_weights(q_lora, key_weights)``, which returns the indexer
    output over them without projecting the attention input.
    """

    index_query_fp8: bool = False
    decode_topk_solution: str | None = None
    index_k_write_quantizes_query: bool = False

    def _scorer_index_query(
        self, q: torch.Tensor, index_k_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The index query and its scales as the paged top-k scorer takes them.

        With ``index_query_fp8`` set and an ``fp8_scaled`` (uint8) plane, the
        query is quantized to FP8 E4M3 with one FP32 scale
        ``max(amax, 1e-10) / 448`` per (token, head) over ``head_dim``.
        Otherwise, a bf16 plane included, the BF16 query passes through, as it
        does under bitwise numerics: the quantizer's rounding path follows the
        batch's token count.

        Args:
            q: ``[tokens, index_heads, head_dim]`` BF16 index query, heads and
                head_dim contiguous.
            index_k_cache: The layer's index-K plane.

        Returns:
            ``(q, q_scales)``: FP8 ``[tokens, index_heads, head_dim]`` and
            contiguous FP32 ``[tokens, index_heads]``, or ``q`` itself and
            ``None``.
        """
        if (
            not self.index_query_fp8
            or index_k_cache.dtype != torch.uint8
            or global_server_args_dict["numerics"] in BITWISE_ENVELOPES
        ):
            return q, None
        tokens, heads, head_dim = q.shape
        q_fp8, q_scales = quantize_fp8(
            q.view(tokens, heads * head_dim),
            granularity="token_group",
            group_size=head_dim,
        )
        return q_fp8.view(tokens, heads, head_dim), q_scales

    def _get_decode_topk_workspace(
        self,
        rows: int,
        cols: int,
        device: torch.device,
        fill_value: int | None = -1,
    ) -> torch.Tensor:
        buffer = self._decode_topk_indices_buffer
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
            self._decode_topk_indices_buffer = buffer
        workspace = buffer[:rows]
        if fill_value is not None:
            workspace.fill_(fill_value)
        return workspace

    def _get_decode_topk_lens_workspace(
        self,
        rows: int,
        device: torch.device,
        fill: bool = True,
    ) -> torch.Tensor:
        buffer = self._decode_topk_lens_buffer
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
        if fill:
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

        page_table = getattr(metadata, "page_table", None)
        if page_table is not None:
            limits.append(max(0, int(page_table.shape[0]) - num_extends))

        return min(limits)

    @staticmethod
    def _resolve_decode_window(
        ctx: ForwardContext,
        metadata: Any,
        *,
        total_tokens: int,
    ) -> GlmDsaDecodeWindow:
        num_decode_reqs = GlmDsaSparseAttentionMixin._resolve_decode_req_count(
            ctx, metadata
        )
        num_decode_tokens = GlmDsaSparseAttentionMixin._resolve_num_decode_tokens(
            ctx,
            total_tokens=total_tokens,
            num_decode_reqs=num_decode_reqs,
        )
        if total_tokens < num_decode_tokens:
            raise RuntimeError(
                "GLM DSA decode token split is invalid: "
                f"tokens={total_tokens}, decode_tokens={num_decode_tokens}"
            )
        q_len_per_req = GlmDsaSparseAttentionMixin._resolve_decode_q_len(
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
        self._retired_decode_workspaces.append(buffer)

    @staticmethod
    def _check_decode_q_len_per_req(q_len_per_req: int) -> None:
        # DSpark verifies 7 drafts plus a bonus; leaves handle each row's causal bound.
        if not 1 <= q_len_per_req <= 8:
            raise NotImplementedError(
                "GLM DSA sparse decode supports 1-8 query tokens per request, "
                f"got {q_len_per_req}."
            )

    def _compute_decode_topk_indices(
        self,
        indexer_output: GlmDsaIndexerOutput,
        ctx: ForwardContext,
        *,
        logical_num_tokens: int | None = None,
    ) -> GlmDsaDecodeTopK | None:
        metadata = getattr(ctx.attn_backend, "forward_decode_metadata", None)
        if metadata is None or metadata.page_table is None:
            return None
        num_tokens = indexer_output.query.shape[0]
        if logical_num_tokens is None:
            logical_num_tokens = num_tokens
        if logical_num_tokens < 0 or logical_num_tokens > num_tokens:
            raise RuntimeError(
                "GLM DSA logical token count is outside the input capacity: "
                f"logical={logical_num_tokens}, capacity={num_tokens}"
            )
        decode_window = self._resolve_decode_window(
            ctx, metadata, total_tokens=logical_num_tokens
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

        page_table = metadata.page_table[
            num_extends : num_extends + decode_window.num_reqs
        ]
        topk = self.index_topk
        return self._compute_decode_topk_indices_portable(
            indexer_output=indexer_output,
            ctx=ctx,
            seq_lens=seq_lens,
            page_table=page_table,
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
        page_table: torch.Tensor,
        q_len_per_req: int,
        decode_start: int,
        num_tokens: int,
        num_decode_tokens: int,
        topk: int,
    ) -> GlmDsaDecodeTopK:
        quantized = indexer_output.quantized_query
        if quantized is None:
            q = indexer_output.query_rows(
                decode_start, decode_start + num_decode_tokens
            )
            weights = indexer_output.weights[
                decode_start : decode_start + num_decode_tokens
            ]
        elif decode_start != 0 or quantized[0].shape[0] != num_decode_tokens:
            raise RuntimeError(
                "the index-K write quantized rows [0, "
                f"{quantized[0].shape[0]}), not the decode window [{decode_start}, "
                f"{decode_start + num_decode_tokens})"
            )
        else:
            q, weights = quantized
        index_k_cache = ctx.token_to_kv_pool.get_index_k_buffer(self.attn_mqa.layer_id)
        if index_k_cache is None:
            raise RuntimeError("GLM DSA top-k requires an index-K cache.")

        # The top-k kernels overwrite the whole workspace when they cover all
        # rows; the sentinel pre-fills are only needed for mixed prefill,
        # where leading rows stay unwritten but readable.
        writes_full_workspace = decode_start == 0 and num_decode_tokens == num_tokens
        topk_indices = self._get_decode_topk_workspace(
            num_tokens,
            topk,
            q.device,
            fill_value=None if writes_full_workspace else -1,
        )
        topk_slice = topk_indices[decode_start : decode_start + num_decode_tokens]
        topk_lens = self._get_decode_topk_lens_workspace(
            num_tokens, q.device, fill=not writes_full_workspace
        )
        topk_lens_slice = topk_lens[decode_start : decode_start + num_decode_tokens]

        metadata = ctx.attn_backend.forward_decode_metadata
        # _dsa_seq_lens_2d is per-token ([bs * q_len_per_req, 1]); skip the extend
        # requests' per-token rows to align with this decode window.
        seq_lens_2d = (
            metadata._dsa_seq_lens_2d[ctx.num_extends * q_len_per_req :]
            if q_len_per_req > 1
            else seq_lens.unsqueeze(1)
        )
        placement = ctx.attn_backend.cache_placement(self.attn_mqa)
        dcp_index = None
        capturing = False
        if global_server_args_dict["simulate"] and placement is not None:
            raise NotImplementedError(
                "DSA simulation requires replicated index history"
            )
        if placement is not None:
            if quantized is not None:
                raise RuntimeError("the KV-parallel decode top-k scores BF16 queries")
            page_size = ctx.token_to_kv_pool.index_k_page_size
            capturing = is_current_stream_capturing(q.device)
            dcp_index = self._reusable_dcp_index(
                ctx,
                metadata,
                placement=placement,
                page_table=page_table,
                q_len_per_req=q_len_per_req,
                rows=q.shape[0],
                capturing=capturing,
            )
            if dcp_index is None:
                requests = (
                    torch.arange(q.shape[0], device=q.device, dtype=torch.int32)
                    // q_len_per_req
                )
                causal_lens = (
                    seq_lens[requests.long()]
                    - (q_len_per_req - 1)
                    + torch.arange(q.shape[0], device=q.device) % q_len_per_req
                )
                dcp_index = build_dcp_index_metadata(
                    q,
                    weights,
                    index_k_cache,
                    page_table,
                    requests,
                    causal_lens,
                    placement=placement,
                    page_size=page_size,
                )
            offsets, _ = select_dsa_topk(
                q,
                weights,
                index_k_cache,
                page_table,
                dcp_index.query_requests,
                dcp_index.causal_lens,
                placement=placement,
                page_size=page_size,
                topk=topk,
                softmax_scale=self.indexer.weights_softmax_scale,
                initial_tokens=0,
                local_tokens=0,
                max_logits_bytes=None,
                metadata=dcp_index,
            )
            local_topk_to_global_slots(
                local_topk_offsets=offsets,
                block_table=page_table,
                block_size=page_size,
                seq_lens=None,
                q_len_per_req=q_len_per_req,
                topk_layout="global_slots",
                out=topk_slice,
                lens_out=topk_lens_slice,
            )
        else:
            softmax_scale = self.indexer.weights_softmax_scale
            if quantized is None:
                q, q_scales = self._scorer_index_query(q, index_k_cache)
            else:
                # Unit query scales: the weights carry them and the score scale.
                q_scales, softmax_scale = None, 1.0
            dsa_decode_topk(
                q,
                weights,
                seq_lens,
                page_table,
                page_size=ctx.token_to_kv_pool.index_k_page_size,
                topk=topk,
                softmax_scale=softmax_scale,
                batch_invariant=False,
                simulation_seed=(
                    self.attn_mqa.layer_id
                    if global_server_args_dict["simulate"]
                    else None
                ),
                q_len_per_req=q_len_per_req,
                index_k_cache=index_k_cache,
                q_scales=q_scales,
                seq_lens_2d=seq_lens_2d,
                plan=metadata._dsa_plan,
                out=topk_slice,
                lens_out=topk_lens_slice,
                slot_order=global_server_args_dict["dsa_slot_order"],
                solution=self.decode_topk_solution,
            )
        return GlmDsaDecodeTopK(
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            dcp_index=dcp_index,
            dcp_source=None if dcp_index is None else metadata,
            dcp_q_len=q_len_per_req,
            dcp_capturing=capturing,
        )

    @staticmethod
    def _reusable_dcp_index(
        ctx: ForwardContext,
        metadata: Any,
        *,
        placement: Any,
        page_table: torch.Tensor,
        q_len_per_req: int,
        rows: int,
        capturing: bool,
    ) -> Any:
        """The DCP index metadata of this forward's previous indexer layer.

        The shared top-k is cleared whenever a forward's metadata is built,
        so its decode record is this forward's. Its metadata is reused when
        it was built from the same decode metadata, page table and rows, in
        the same capture state: an eager step followed by a capture on the
        same metadata (as a graph runner's warm-up does when the share is not
        cleared in between) must not bake the eager build into the graph,
        whose replays refresh lengths in place. A draft backend carries the
        share across draft steps whose lengths advance in place, so it always
        rebuilds.
        """
        if getattr(ctx.attn_backend, "is_draft", False):
            return None
        prev = ctx.attn_backend.sparse_topk.decode
        index = getattr(prev, "dcp_index", None)
        if (
            index is None
            or prev.dcp_source is not metadata
            or prev.dcp_capturing != capturing
            or prev.dcp_q_len != q_len_per_req
            or index.placement != placement
            or index.query_requests.shape[0] != rows
            or index.page_table.data_ptr() != page_table.data_ptr()
            or index.page_table.shape != page_table.shape
            or index.page_table.stride() != page_table.stride()
        ):
            return None
        return index

    def _compute_prefill_topk_indices(
        self,
        indexer_output: GlmDsaIndexerOutput,
        ctx: ForwardContext,
        num_prefill_tokens: int,
    ) -> GlmDsaPrefillTopK | None:
        cached = ctx.attn_backend.sparse_topk.prefill
        if (
            isinstance(cached, GlmDsaPrefillTopK)
            and cached.row_starts is not None
            and cached.row_ends is not None
            and cached.candidate_lens_cpu is not None
            and cached.row_starts.numel() == num_prefill_tokens
        ):
            return self._compute_prefill_topk_indices_portable(
                indexer_output=indexer_output,
                ctx=ctx,
                seq_lens=cached.seq_lens,
                page_table=cached.page_table,
                kv_workspace_slots=cached.kv_workspace_slots,
                row_starts=cached.row_starts,
                row_ends=cached.row_ends,
                candidate_lens_cpu=cached.candidate_lens_cpu,
                kv_seq_lens=cached.kv_seq_lens,
                max_seq_len=cached.max_seq_len,
                num_prefill_tokens=num_prefill_tokens,
                topk=self.index_topk,
                dcp_index=cached.dcp_index,
            )

        chunk_meta = ctx.attn_backend.chunked_prefill_metadata
        prefix_lens = getattr(chunk_meta, "extend_prefix_lens_cpu", None)
        if prefix_lens is None:
            prefix_lens = chunk_meta.extend_prefix_lens
        prefix_lens_cpu = torch.as_tensor(
            prefix_lens[: ctx.num_extends], dtype=torch.int64, device="cpu"
        )
        extend_lens_cpu = torch.as_tensor(
            chunk_meta.extend_seq_lens_cpu[: ctx.num_extends],
            dtype=torch.int64,
            device="cpu",
        )
        if extend_lens_cpu.numel() == 0:
            return None
        metadata_tokens = int(extend_lens_cpu.sum().item())
        if metadata_tokens != num_prefill_tokens:
            raise RuntimeError(
                "GLM DSA prefill token count mismatch: "
                f"metadata={metadata_tokens}, tokens={num_prefill_tokens}"
            )
        if chunk_meta.page_table is None:
            raise RuntimeError(
                "GLM DSA sparse prefill requires the chunked-prefill block tables"
            )

        device = indexer_output.query.device
        seq_lens_cpu = prefix_lens_cpu + extend_lens_cpu
        max_seq_len = int(seq_lens_cpu.max().item())
        req_ids_cpu = torch.arange(seq_lens_cpu.numel(), dtype=torch.int64)
        token_req_cpu = torch.repeat_interleave(
            req_ids_cpu,
            extend_lens_cpu,
            output_size=num_prefill_tokens,
        )
        extend_cu_cpu = torch.zeros(extend_lens_cpu.numel() + 1, dtype=torch.int64)
        torch.cumsum(extend_lens_cpu, dim=0, out=extend_cu_cpu[1:])
        token_offsets_cpu = torch.arange(num_prefill_tokens, dtype=torch.int64)
        token_offsets_cpu -= extend_cu_cpu.index_select(0, token_req_cpu)
        candidate_lens_cpu = (
            prefix_lens_cpu.index_select(0, token_req_cpu) + token_offsets_cpu + 1
        )
        seq_cu_cpu = torch.zeros(seq_lens_cpu.numel() + 1, dtype=torch.int64)
        torch.cumsum(seq_lens_cpu, dim=0, out=seq_cu_cpu[1:])
        row_starts_cpu = seq_cu_cpu.index_select(0, token_req_cpu)
        row_ends_cpu = row_starts_cpu + candidate_lens_cpu
        seq_lens, row_starts, row_ends, kv_seq_lens = _int32_to_device(
            (seq_lens_cpu, row_starts_cpu, row_ends_cpu, candidate_lens_cpu), device
        )
        page_size = ctx.token_to_kv_pool.index_k_page_size
        max_pages = (max_seq_len + page_size - 1) // page_size
        page_table = chunk_meta.page_table[:, :max_pages].to(
            device=device,
            dtype=torch.int32,
        )
        kv_workspace_slots = build_prefill_kv_workspace_slots(
            page_table=page_table,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            page_size=page_size,
            device=device,
            num_tokens=int(seq_lens_cpu.sum().item()),
        )

        return self._compute_prefill_topk_indices_portable(
            indexer_output=indexer_output,
            ctx=ctx,
            seq_lens=seq_lens,
            page_table=page_table,
            kv_workspace_slots=kv_workspace_slots,
            row_starts=row_starts,
            row_ends=row_ends,
            candidate_lens_cpu=candidate_lens_cpu,
            kv_seq_lens=kv_seq_lens,
            max_seq_len=max_seq_len,
            num_prefill_tokens=num_prefill_tokens,
            topk=self.index_topk,
            dcp_index=None,
        )

    def _select_prefill_topk_rows(
        self,
        *,
        q: torch.Tensor,
        weights: torch.Tensor,
        kv_workspace_slots: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        candidate_lens_cpu: torch.Tensor,
        index_k_cache: torch.Tensor,
        page_size: int,
        max_logits_bytes: int,
        topk: int,
        query_proj: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prefill top-k over the whole index-K plane, rows split over TP.

        Without the split (or with too few rows to pay for the gather) every
        rank selects for every row. With it, rank ``r`` selects for rows
        ``[r * per_rank, (r + 1) * per_rank)`` and the ranks all-gather the
        ``[rows, topk]`` selection and the per-row lengths. A workspace of
        fewer than 65535 rows gathers the selection as int16 (unsigned 16-bit
        indices, ``-1`` where empty), which halves the gathered bytes.
        With ``query_proj`` set, ``q`` holds its input rows and each rank
        projects (and, see :meth:`_scorer_index_query`, quantizes) only the
        rows it selects for.
        """

        def select(rows: slice, out=None, lens_out=None):
            query, q_scales = self._scorer_index_query(
                q[rows] if query_proj is None else query_proj(q[rows]),
                index_k_cache,
            )
            return dsa_prefill_topk(
                query,
                weights[rows],
                kv_workspace_slots,
                row_starts[rows],
                row_ends[rows],
                topk=topk,
                softmax_scale=self.indexer.weights_softmax_scale,
                batch_invariant=False,
                index_k_cache=index_k_cache,
                page_size=page_size,
                q_scales=q_scales,
                max_logits_bytes=max_logits_bytes,
                candidate_lens_cpu=candidate_lens_cpu[rows],
                slot_order=global_server_args_dict["dsa_slot_order"],
                out=out,
                lens_out=lens_out,
            )

        num_rows = q.shape[0]
        tp_size = self.mapping.attn.tp_size
        if (
            not self.prefill_topk_tp_split
            or tp_size == 1
            or num_rows < _PREFILL_TOPK_TP_SPLIT_MIN_ROWS
        ):
            return select(slice(0, num_rows))
        # 16-row slices stay 64-byte aligned, so Triton never recompiles during serving.
        per_rank = -(-num_rows // (tp_size * _PREFILL_TOPK_ROW_ALIGN)) * (
            _PREFILL_TOPK_ROW_ALIGN
        )
        begin = min(num_rows, self.mapping.attn.tp_rank * per_rank)
        end = min(num_rows, begin + per_rank)
        gather_dtype = (
            torch.int16 if kv_workspace_slots.numel() < 0xFFFF else torch.int32
        )
        local = torch.empty((per_rank, topk), dtype=torch.int32, device=q.device)
        local_lens = torch.zeros((per_rank,), dtype=torch.int32, device=q.device)
        if end > begin:
            select(
                slice(begin, end),
                out=local[: end - begin],
                lens_out=local_lens[: end - begin],
            )
        gathered, gathered_lens = self._gather_prefill_topk_rows(
            local, local_lens, gather_dtype
        )
        return gathered[:num_rows], gathered_lens[:num_rows]

    def _gather_prefill_topk_rows(
        self, local: torch.Tensor, local_lens: torch.Tensor, gather_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-gather every rank's int32 selection rows (as ``gather_dtype``) and lengths."""
        tp_size = self.mapping.attn.tp_size
        per_rank = local.shape[0]
        gathered = torch.empty(
            (tp_size * per_rank, local.shape[1]),
            dtype=gather_dtype,
            device=local.device,
        )
        gathered_lens = torch.empty(
            (tp_size * per_rank,), dtype=torch.int32, device=local.device
        )
        group = self.mapping.attn.tp_group
        # The int32 -> int16 cast keeps the low 16 bits, so -1 stays 0xFFFF.
        local = local.to(gather_dtype)
        if gather_dtype == torch.int16:
            # NCCL has no int16; the gather copies bytes, so ship bf16 bit patterns.
            all_gather_single(
                gathered.view(torch.bfloat16), local.view(torch.bfloat16), group
            )
        else:
            all_gather_single(gathered, local, group)
        all_gather_single(gathered_lens, local_lens, group)
        return gathered, gathered_lens

    def _compute_prefill_topk_indices_portable(
        self,
        *,
        indexer_output: GlmDsaIndexerOutput,
        ctx: ForwardContext,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        kv_workspace_slots: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        candidate_lens_cpu: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        max_seq_len: int,
        num_prefill_tokens: int,
        topk: int,
        dcp_index: DCPIndexMetadata | None,
    ) -> GlmDsaPrefillTopK:
        """Prefill top-k of the round's rows. ``dcp_index`` is this forward's
        previous indexer layer's DCP index metadata, or ``None``."""
        q = indexer_output.query[:num_prefill_tokens].contiguous()
        # Raw BF16 weights view (may be column-split and noncompact); the top-k
        # scorer consumes the row/head strides and upcasts weights in-kernel.
        weights = indexer_output.weights[:num_prefill_tokens]

        index_k_cache = ctx.token_to_kv_pool.get_index_k_buffer(self.attn_mqa.layer_id)
        if index_k_cache is None:
            raise RuntimeError("GLM DSA top-k requires an index-K cache.")

        max_logits_mb = int(global_server_args_dict[_INDEXER_PREFILL_MAX_LOGITS_MB_ARG])
        placement = ctx.attn_backend.cache_placement(self.attn_mqa)
        owned_index = None
        if placement is not None:
            if indexer_output.query_proj is not None:
                q = indexer_output.query_proj(q)
            # Both gates read host values equal on every KVP rank, so the ranks issue the same collectives.
            row_owner = q.shape[0] >= _KVP_PREFILL_ROW_OWNER_MIN_ROWS
            global_starts, global_ends = row_starts, row_ends
            owned_index = self._owned_width_dcp_index(
                ctx,
                dcp_index,
                q,
                weights,
                index_k_cache,
                page_table,
                seq_lens,
                row_starts,
                row_ends,
                placement=placement,
            )
            offsets, topk_lens = select_dsa_topk(
                q,
                weights,
                index_k_cache,
                page_table,
                (
                    _prefill_query_requests(seq_lens, row_starts)
                    if owned_index is None
                    else owned_index.query_requests
                ),
                (
                    global_ends - global_starts
                    if owned_index is None
                    else owned_index.causal_lens
                ),
                placement=placement,
                page_size=ctx.token_to_kv_pool.index_k_page_size,
                topk=topk,
                softmax_scale=self.indexer.weights_softmax_scale,
                initial_tokens=0,
                local_tokens=0,
                max_logits_bytes=max(1, max_logits_mb) * 1024 * 1024,
                metadata=owned_index,
                row_owner=row_owner,
            )
            workspace_indices = torch.where(
                offsets >= 0, offsets + global_starts.unsqueeze(1), -1
            ).to(torch.int32)
        else:
            workspace_indices, topk_lens = self._select_prefill_topk_rows(
                q=q,
                weights=weights,
                kv_workspace_slots=kv_workspace_slots,
                row_starts=row_starts.to(torch.int32).contiguous(),
                row_ends=row_ends.to(torch.int32).contiguous(),
                candidate_lens_cpu=candidate_lens_cpu,
                index_k_cache=index_k_cache,
                page_size=ctx.token_to_kv_pool.index_k_page_size,
                max_logits_bytes=max(1, max_logits_mb) * 1024 * 1024,
                topk=topk,
                query_proj=indexer_output.query_proj,
            )
        return GlmDsaPrefillTopK(
            workspace_indices=workspace_indices,
            topk_lens=topk_lens,
            page_table=page_table,
            seq_lens=seq_lens.to(device=q.device, dtype=torch.int32),
            kv_seq_lens=kv_seq_lens,
            max_seq_len=max_seq_len,
            kv_workspace_slots=kv_workspace_slots,
            row_starts=row_starts,
            row_ends=row_ends,
            candidate_lens_cpu=candidate_lens_cpu,
            dcp_index=owned_index,
        )

    def _owned_width_dcp_index(
        self,
        ctx: ForwardContext,
        previous: DCPIndexMetadata | None,
        q: torch.Tensor,
        weights: torch.Tensor,
        index_k_cache: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        *,
        placement: CachePlacement,
    ) -> DCPIndexMetadata | None:
        """This forward's DCP index metadata at the leaf's owned width, or
        ``None`` to keep the page table's width.

        The leaf plans the width per forward from the host page table, the
        same on every rank (``prefill_index_columns``). Every indexer layer
        of a forward scores the same rows against the same page table, so the
        first builds the local table, the row geometry and, within the memory
        cap, the owned-width pages; later layers reuse ``previous``, the
        shared prefill record that every forward's metadata build clears.
        """
        columns = ctx.attn_backend.prefill_index_columns()
        if columns is None:
            return None
        if (
            previous is not None
            and previous.placement == placement
            and previous.page_table is page_table
            and previous.query_requests.shape[0] == q.shape[0]
        ):
            return previous
        return build_dcp_index_metadata(
            q,
            weights,
            index_k_cache,
            page_table,
            _prefill_query_requests(seq_lens, row_starts),
            row_ends - row_starts,
            placement=placement,
            page_size=ctx.token_to_kv_pool.index_k_page_size,
            columns=columns,
        )

    def forward_dsa_core(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        q_norm: torch.Tensor,
        latent_cache: torch.Tensor,
        ctx: ForwardContext,
        comm_manager: CommManager,
        output_gate: torch.Tensor | None = None,
        projected_q: torch.Tensor | None = None,
        index_query: torch.Tensor | None = None,
        index_key_weights: torch.Tensor | None = None,
        written_index_query: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """DSA sparse MLA after the a-projections.

        It runs the indexer and its index-K write, the shared prefill/decode
        top-k, then the sparse prefill and decode cores over the round's rows.

        Args:
            positions: ``[T]`` token positions.
            hidden_states: ``[T, hidden]`` attention input (the indexer's
                ``wk`` / ``weights_proj`` input).
            q_norm: ``[T, q_lora_rank]`` normalized q latent (the indexer's
                ``wq_b`` input and the ``q_b_proj`` input).
            latent_cache: ``[T, kv_lora_rank + qk_rope_head_dim]`` latent rows,
                the latent part already normalized.
            ctx: The forward context; ``ctx.attn_backend`` is the DSA router.
            comm_manager: The layer's attention communication.
            output_gate: Optional raw ``[T, num_local_heads * v_head_dim]``
                sigmoid gate the value projection applies; decode rounds only.
            projected_q: Optional ``q_b_proj(q_norm)`` rows a caller projected
                ahead; None projects them here.
            index_query: Optional ``[T, index_heads, index_head_dim]`` index
                query a caller projected ahead; None leaves the indexer's.
            index_key_weights: Optional ``[T, index_head_dim + index_heads]``
                pre-norm index keys then head weights a caller projected
                ahead from the attention input, unit inner stride; the
                indexer then skips its own projection of ``hidden_states``.
                None projects them in the indexer.
            written_index_query: Optional decode-round E4M3 index queries
                ``[T, index_heads, index_head_dim]`` and FP32 head weights
                ``[T, index_heads]`` (``quantize_index_query_weights``' outputs)
                from a launch that also wrote the round's index keys
                (``index_k_layernorm_fp8_scatter``'s bytes); the core then skips
                its index-K write. Every row must decode, without KV
                placement, under ``index_k_write_quantizes_query``, and the
                keys must be the caller's ``index_key_weights``. None writes
                the keys here.

        Returns:
            ``[T, num_local_heads * v_head_dim]`` attention output before
            ``o_proj`` (gated when ``output_gate`` is given).
        """
        if output_gate is not None and ctx.num_extends > 0:
            raise ValueError("the DSA value projection gates decode rounds only")
        if written_index_query is not None and (
            ctx.num_extends > 0
            or not self.index_k_write_quantizes_query
            or index_key_weights is None
        ):
            raise ValueError(
                "a written index query serves decode rounds whose index-K write "
                "quantizes the query (index_k_write_quantizes_query) over the "
                "keys the caller projected (index_key_weights)"
            )
        decode_metadata = getattr(ctx.attn_backend, "forward_decode_metadata", None)
        num_attn_tokens = int(q_norm.shape[0])
        # Narrow padded-replay inputs to live rows; _land_in restores the bucket shape.
        valid_rows = current_valid_rows()
        if valid_rows is not None and valid_rows < num_attn_tokens:
            if written_index_query is not None:
                raise RuntimeError("a written index query cannot be narrowed")
            if positions.dim() != 1:
                raise RuntimeError("padded DSA replay expects 1-D positions")
            if comm_manager.needs_pre_attn_all_gather():
                raise RuntimeError(
                    "padded DSA replay needs an identity pre-attention comm"
                )
            (
                positions,
                q_norm,
                latent_cache,
                output_gate,
                projected_q,
                index_query,
                index_key_weights,
            ) = slice_to_real_tokens(
                valid_rows,
                positions,
                q_norm,
                latent_cache,
                output_gate,
                projected_q,
                index_query,
                index_key_weights,
            )
            num_attn_tokens = valid_rows
        decode_window = self._resolve_decode_window(
            ctx,
            decode_metadata,
            total_tokens=num_attn_tokens,
        )
        num_decode_tokens = decode_window.num_tokens
        num_prefill_tokens = decode_window.start
        decode_start = decode_window.start
        decode_end = decode_window.end

        # The round's KV write slots, by sub-mode: the extend span for the
        # prefill rows, the decode rows' verify window behind them. The
        # indexer's full-round index_k write consumes the concatenation
        # (an eager break; under a padded replay the rows were narrowed above).
        backend = ctx.attn_backend
        # The forward's shared top-k: an indexer layer publishes, "shared"
        # layers (and the MTP head reusing the target's) consume.
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
            out_cache_loc = torch.empty(0, dtype=torch.int64, device=q_norm.device)
        if valid_rows is not None and out_cache_loc.numel() != num_attn_tokens:
            # Host metadata only, checked before the first cache write.
            raise RuntimeError(
                f"padded DSA replay: write span {out_cache_loc.numel()} "
                f"!= live rows {num_attn_tokens}"
            )

        should_compute_indexer = not self.skip_indexer_topk or (
            self.is_nextn
            and (
                (num_prefill_tokens > 0 and shared_topk.prefill is None)
                or (num_decode_tokens > 0 and shared_topk.decode is None)
            )
        )
        if written_index_query is not None and not should_compute_indexer:
            raise RuntimeError("a written index query needs the layer's own top-k")
        if should_compute_indexer:
            if index_key_weights is None:
                hidden_states = comm_manager.pre_attn_comm(hidden_states, ctx)
                if valid_rows is not None:
                    (hidden_states,) = slice_to_real_tokens(
                        num_attn_tokens, hidden_states
                    )
                indexer_output = self.indexer(hidden_states, q_norm, positions)
            else:
                indexer_output = self.indexer.from_key_weights(
                    q_norm, index_key_weights
                )
            if index_query is not None:
                indexer_output = replace(
                    indexer_output, query=index_query, query_proj=None
                )
            placement = ctx.attn_backend.cache_placement(self.attn_mqa)
            index_slots, index_mask = resolve_cache_slots(out_cache_loc, placement)
            if written_index_query is not None:
                if num_prefill_tokens > 0 or placement is not None:
                    raise RuntimeError(
                        "a written index query needs a decode round without KV "
                        "placement"
                    )
                indexer_output = replace(
                    indexer_output, quantized_query=written_index_query
                )
            # Decode rounds only (every row decodes); the KV-parallel top-k scores BF16 queries.
            elif (
                self.index_k_write_quantizes_query
                and num_prefill_tokens == 0
                and placement is None
            ):
                quantized = ctx.token_to_kv_pool.set_index_k_buffer_and_quantize_query(
                    self.attn_mqa.layer_id,
                    index_slots,
                    indexer_output.key,
                    write_mask=index_mask,
                    key_norm=indexer_output.key_norm,
                    query=indexer_output.query_rows(0, num_attn_tokens),
                    query_weights=indexer_output.weights,
                    softmax_scale=self.indexer.weights_softmax_scale,
                )
                indexer_output = replace(indexer_output, quantized_query=quantized)
            else:
                ctx.token_to_kv_pool.set_index_k_buffer(
                    self.attn_mqa.layer_id,
                    index_slots,
                    indexer_output.key,
                    write_mask=index_mask,
                    key_norm=indexer_output.key_norm,
                )
            if ctx.num_extends > 0:
                shared_topk.prefill = self._compute_prefill_topk_indices(
                    indexer_output,
                    ctx,
                    num_prefill_tokens,
                )
            if ctx.num_extends < ctx.bs:
                # After the index-K write; decode joins before attention reads it.
                with self.decode_topk_fork.branch_after_main():
                    shared_topk.decode = self._compute_decode_topk_indices(
                        indexer_output,
                        ctx,
                    )

        q = self.q_b_proj(q_norm)[0] if projected_q is None else projected_q
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
            if shared_topk.prefill is None:
                raise RuntimeError(
                    "GLM DSA sparse prefill requires computed top-k indices."
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
            decode_ctx = replace(
                ctx,
                bs=decode_window.num_reqs,
                num_extends=0,
                input_num_tokens=num_decode_tokens,
                forward_mode=ForwardMode.DECODE,
            )
            if shared_topk.decode is None:
                raise RuntimeError(
                    "GLM DSA sparse decode requires computed top-k indices."
                )
            Q = self.forward_absorb_qkv_proj(
                q[decode_start:decode_end],
                latent_cache[decode_start:decode_end],
                positions[decode_start:decode_end],
                decode_ctx,
                decode_locs,
            )
            self.decode_topk_fork.join()
            topk_indices, topk_lens = self._slice_decode_topk(
                shared_topk.decode,
                decode_start,
                decode_end,
            )
            self.forward_absorb_attn_v_proj(
                Q,
                decode_ctx,
                attn_output[decode_start:decode_end],
                topk_indices=topk_indices,
                topk_lens=topk_lens,
                output_gate=(
                    None
                    if output_gate is None
                    else output_gate[decode_start:decode_end]
                ),
            )

        return attn_output

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
        cache_num_tokens: int | None = None,
    ) -> torch.Tensor:
        Q = self.forward_absorb_qkv_proj(
            q,
            latent_cache,
            positions,
            ctx,
            out_cache_loc,
            cache_num_tokens=cache_num_tokens,
        )
        # The host's sparse core + value projection (``mla_project_value``
        # is this bmm on NVIDIA; the gluon kernel elsewhere).
        return self.sparse_prefill_attn_v_proj(
            Q,
            ctx,
            output,
            kv_seq_lens=prefill_topk.kv_seq_lens,
            topk_slots=workspace_topk_to_global_slots(
                workspace_indices=prefill_topk.workspace_indices,
                kv_workspace_slots=prefill_topk.kv_workspace_slots,
            ),
            topk_lens=prefill_topk.topk_lens,
            max_seq_len=prefill_topk.max_seq_len,
        )

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
        output_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        Q = self.forward_absorb_qkv_proj(
            q,
            latent_cache,
            positions,
            ctx,
            out_cache_loc,
        )
        return self.forward_absorb_attn_v_proj(
            Q,
            ctx,
            output,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            output_gate=output_gate,
        )

    def forward_absorb_attn_v_proj(
        self,
        Q,
        ctx: ForwardContext,
        output: torch.Tensor,
        topk_indices: torch.Tensor | None = None,
        topk_lens: torch.Tensor | None = None,
        output_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_output = self.attn_mqa(
            Q,
            k=None,
            v=None,
            positions=None,
            ctx=ctx,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
        )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
        self.project_latent_values(attn_output, output, output_gate)
        return output

    def project_latent_values(
        self,
        attn_output: torch.Tensor,
        output: torch.Tensor,
        gate: torch.Tensor | None,
    ) -> None:
        """``output = (attn_output @ w_vc) * sigmoid(gate)`` per local head.

        ``attn_output`` is ``[T, heads, kv_lora_rank]``; ``output`` and the
        optional raw ``gate`` are ``[T, heads * v_head_dim]``.
        """
        torch.bmm(
            attn_output.transpose(0, 1),
            self.w_vc,
            out=output.view(-1, self.num_local_heads, self.v_head_dim).transpose(0, 1),
        )
        if gate is not None:
            sigmoid_mul(output, gate)


class GlmMoeDsaAttention(GlmDsaSparseAttentionMixin, DeepseekV3AttentionMLA):
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
        # process_weights picks DeepGEMM only after converting FP8 block scales for it.
        self.q_a_layernorm = RMSNorm(q_lora_rank, eps=1e-6)
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, eps=1e-6)
        self.fused_qk_layernorm = FusedRMSNorm(
            self.q_a_layernorm,
            self.kv_a_layernorm,
        )
        self.index_topk = config.index_topk
        self.is_nextn = is_nextn
        # Every rank selects every prefill row (validated configuration).
        self.prefill_topk_tp_split = False
        # NextN may reuse the prior draft's top-k; shared layers lack indexer weights.
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
        self._retired_decode_workspaces: list[torch.Tensor] = []
        self.decode_topk_fork = StreamFork(None)

    @break_point
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        comm_manager: CommManager,
        block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """GLM-5 DSA attention, one COARSE breakable-graph break point.

        Like DeepSeek V4.1 it does cache-group writes, a data-dependent indexer
        -> top-k stage and the FlashMLA sparse kernel (plus pre-attn
        collectives), none capturable. Under a prefill-graph capture the whole
        attention runs eager (reading the live ``ctx``) while the layer's
        norms + MoE stay graphed; direct call otherwise (see ``break_point``).
        """
        # Empty (idle / DP-idle) batch: explicit skip, like the sibling MLP/MoE forwards.
        if hidden_states.shape[0] == 0:
            return hidden_states
        qkv = self.fused_qkv_a_proj_with_mqa(
            hidden_states,
            block_scale,
            torch.bfloat16,
        )
        # Drop the FP8 block-scale padding columns of the fused QKV-A output, if any.
        _qkv_width = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
        if qkv.shape[-1] != _qkv_width:
            qkv = qkv[..., :_qkv_width]
        qkv = comm_manager.pre_attn_comm(qkv, ctx)

        q_a, latent_cache = qkv.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        kv_a = latent_cache[..., : self.kv_lora_rank]
        q_norm = torch.empty_like(q_a)
        if q_a.size(0) > 0:
            self.fused_qk_layernorm(input_q_a=q_a, input_kv_a=kv_a, output_q_a=q_norm)

        attn_output = self.forward_dsa_core(
            positions, hidden_states, q_norm, latent_cache, ctx, comm_manager
        )
        if ctx.draft_narrowing is not None:
            attn_output = attn_output.index_select(0, ctx.gather_ids)
        output, _ = self.o_proj(attn_output)
        return output


class GlmMoeDsaDecoderLayer(DeepseekV3DecoderLayer):
    @property
    def attention_cls(self) -> type[nn.Module]:
        return GlmMoeDsaAttention

    @property
    def dense_batch_invariant(self) -> bool:
        return False

    def attention_kwargs(self, prefix: str) -> dict:
        return {**super().attention_kwargs(prefix), "is_nextn": self.is_nextn}

    def narrow_residual(
        self, residual: torch.Tensor, ctx: ForwardContext
    ) -> torch.Tensor:
        if ctx.draft_narrowing is None:
            return residual
        return residual.index_select(0, ctx.gather_ids)

    def forward_idle_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        aux_hidden_states: list[torch.Tensor] | None,
    ) -> None:
        # No rows, but the tap and its gather stay aligned across ranks.
        capture_layer_input(self.comm_manager, hidden_states, ctx, aux_hidden_states)


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
        # DFLASH: each capture layer's positional tap index.
        self._dflash_capture_idx_map: dict[int, int] = {}


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
    padded = torch.nn.Parameter(
        torch.cat([weight.data, pad], dim=0), requires_grad=False
    )
    # Keep the attributes the quant method attached (``weight_loader``,
    # ``input_dim`` / ``output_dim``, ...): a live weight update streams the
    # q_a / kv_a shards into this parameter by row offset through
    # ``weight_loader``, and the padding rows stay zero.
    set_weight_attrs(padded, dict(vars(weight)))
    proj.weight = padded


def load_fused_indexer_projection(
    name: str,
    loaded_weight: torch.Tensor,
    *,
    params_dict: dict[str, torch.Tensor],
    modules_dict: dict[str, nn.Module],
    pending_fp8_wk: dict[str, dict[str, torch.Tensor]],
    loaded_shards: dict[str, set[int]],
    weight_block_size: list[int] | None,
) -> None:
    """Copy an indexer ``wk`` / ``weights_proj`` checkpoint tensor into shard
    0 / 1 of the bf16 fused ``wk_weights_proj``.

    An FP8 ``wk`` and its ``weight_scale_inv`` wait in ``pending_fp8_wk`` for
    each other, then load block-dequantized. Once both shards of a module have
    loaded, its indexer switches to the fused projection.
    """
    if ".indexer.weights_proj.weight" in name:
        module_name, shard_id = name.rsplit(".weights_proj.weight", 1)[0], 1
    elif ".indexer.wk." in name:
        module_name, shard_id = name.rsplit(".wk.", 1)[0], 0
        is_fp8_weight = name.endswith(".weight") and loaded_weight.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        )
        if is_fp8_weight or "weight_scale_inv" in name:
            entry = pending_fp8_wk.setdefault(module_name, {})
            entry["weight" if is_fp8_weight else "scale"] = loaded_weight
            if len(entry) < 2 or weight_block_size is None:
                return
            del pending_fp8_wk[module_name]
            loaded_weight = block_dequant(
                entry["weight"], entry["scale"], list(weight_block_size)
            ).to(torch.bfloat16)
        elif not name.endswith(".weight"):
            return
    else:
        return

    param = params_dict.get(f"{module_name}.wk_weights_proj.weight")
    if param is None:
        return
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded_weight, shard_id)
    shards = loaded_shards.setdefault(module_name, set())
    shards.add(shard_id)
    module = modules_dict.get(module_name)
    if shards == {0, 1} and isinstance(module, GlmDsaIndexer):
        module.set_wk_weights_proj_loaded()


class GlmMoeDsaForCausalLM(DeepseekV3ForCausalLM):
    model_cls = GlmMoeDsaModel

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config,
            mapping,
            quant_config=quant_config,
            prefix=prefix,
        )
        self._init_indexer_pairing_state()

    def _init_indexer_pairing_state(self) -> None:
        """Declare the indexer pairing caches (also for subclasses that skip
        ``__init__``, such as the NextN draft)."""
        # FP8 indexer ``wk`` weight and block scale waiting for each other
        # before the bf16 fused projection shard is written; a live update
        # streams the checkpoint in chunks, so the pair may straddle a
        # ``load_weights`` call.
        self._pending_fp8_wk: dict[str, dict[str, torch.Tensor]] = {}
        # Fused indexer projection shards seen so far, per module; both must
        # arrive before the module switches to the fused path.
        self._loaded_fused_indexer_shards: dict[str, set[int]] = {}

    def begin_weight_update(self) -> None:
        super().begin_weight_update()
        self._pending_fp8_wk.clear()
        self._loaded_fused_indexer_shards.clear()

    def abort_weight_update(self) -> None:
        super().abort_weight_update()
        self._pending_fp8_wk.clear()
        self._loaded_fused_indexer_shards.clear()

    def end_weight_update(self) -> None:
        """Close the session; an FP8 indexer weight without its scale fails it.

        Raises:
            RuntimeError: An FP8 ``indexer.wk`` weight or its block scale was
                streamed without the other, so the bf16 fused projection still
                holds the previous values.
        """
        if self._pending_fp8_wk:
            unpaired = sorted(self._pending_fp8_wk)
            self.abort_weight_update()
            raise RuntimeError(
                f"{type(self).__name__}: the update streamed an FP8 indexer wk "
                f"weight or scale without its partner for {unpaired}"
            )
        super().end_weight_update()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        modules_dict = dict(self.named_modules())
        pending_fp8_wk = self._pending_fp8_wk
        loaded_fused_indexer_shards = self._loaded_fused_indexer_shards
        # ``get_param`` remaps checkpoint names; report the parameter's own.
        param_names = {id(param): name for name, param in params_dict.items()}
        loaded: set[str] = set()

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
                loaded.add(param_names[id(param)])
                load_fused_indexer_projection(
                    name,
                    loaded_weight,
                    params_dict=params_dict,
                    modules_dict=modules_dict,
                    pending_fp8_wk=pending_fp8_wk,
                    loaded_shards=loaded_fused_indexer_shards,
                    weight_block_size=getattr(
                        self.quant_config, "weight_block_size", None
                    ),
                )

        loaded |= super().load_weights(base_weights())
        self._pad_fused_qkv_a_proj_for_fp8_blockscale()
        return loaded

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
