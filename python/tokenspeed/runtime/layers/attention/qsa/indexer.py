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

"""Weight-free QSA MQA indexer and sparse GQA runtime path."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.activation.triton import sigmoid_mul
from tokenspeed_kernel.ops.attention.triton.qwen4_exp_qsa import (
    qwen4_exp_qsa_block_topk,
    qwen4_exp_qsa_complete_blocks,
    qwen4_exp_qsa_compress_and_store,
    qwen4_exp_qsa_group_cache_locs,
    qwen4_exp_qsa_logical_layout,
    qwen4_exp_qsa_norm_rope,
    qwen4_exp_qsa_recent_write,
    qwen4_exp_qsa_selected_tokens,
    qwen4_exp_qsa_sparse_attention,
    qwen4_exp_qsa_sparse_slots,
    qwen4_exp_qsa_stage_draft,
    qwen4_exp_qsa_stage_verify,
)
from tokenspeed_kernel.platform import pdl_enabled
from torch import nn

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    current_valid_rows,
    slice_to_real_tokens,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_QSA_CACHE_GROUP,
    QWEN4_EXP_QSA_COMPRESSED_ROWS_PER_PAGE,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
    QWEN4_EXP_QSA_RECENT_ROWS_PER_PAGE,
    qsa_compressed_field,
    qsa_raw_key_field,
    qsa_rope_position_field,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.layernorm import GemmaRMSNorm
from tokenspeed.runtime.layers.linear import ReplicatedLinear
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import envs

_DRAFT_INVALID_POSITION = torch.iinfo(torch.int64).min


class QSAIndexer(nn.Module):
    """Config-driven QSA indexer backed by TokenSpeed cache groups."""

    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        rotary_emb,
    ) -> None:
        super().__init__()
        required = (
            "indexer_n_heads",
            "indexer_kv_heads",
            "indexer_head_dim",
            "indexer_budget",
            "indexer_compress_ratio",
        )
        missing = [name for name in required if getattr(config, name, None) is None]
        if missing:
            raise ValueError(f"Qwen4-Exp QSA config is missing {missing}")
        invalid = {
            name: getattr(config, name)
            for name in required
            if int(getattr(config, name)) <= 0
        }
        if invalid:
            raise ValueError(f"Qwen4-Exp QSA config values must be positive: {invalid}")
        self.layer_id = int(layer_id)
        self.index_n_heads = int(config.indexer_n_heads)
        self.index_kv_heads = int(config.indexer_kv_heads)
        self.index_head_dim = int(config.indexer_head_dim)
        self.token_topk = int(config.indexer_budget)
        self.compress_ratio = int(config.indexer_compress_ratio)
        self.compressed_page_size = QWEN4_EXP_QSA_COMPRESSED_ROWS_PER_PAGE
        self.compressed_token_page_size = (
            self.compressed_page_size * self.compress_ratio
        )
        self.recent_page_size = QWEN4_EXP_QSA_RECENT_ROWS_PER_PAGE
        self.block_topk = self.token_topk // self.compress_ratio
        if self.index_kv_heads != 1:
            raise ValueError("QSA requires indexer_kv_heads == 1")
        if self.token_topk % self.compress_ratio:
            raise ValueError("indexer_budget must be divisible by compress ratio")
        if self.block_topk not in (512, 2048):
            raise ValueError(
                "Qwen4-Exp QSA requires indexer_budget / "
                "indexer_compress_ratio to be 512 or 2048"
            )
        if rotary_emb.rotary_dim > self.index_head_dim:
            raise ValueError("QSA index head is narrower than the rotary dimension")
        self.rotary_emb = rotary_emb
        self.index_qk_proj = ReplicatedLinear(
            config.hidden_size,
            (self.index_n_heads + self.index_kv_heads) * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("index_qk_proj", prefix),
        )
        self.q_layernorm = GemmaRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = GemmaRMSNorm(self.index_head_dim, eps=config.rms_norm_eps)
        self.mapping = mapping
        # Draft QSA indexers can publish step-0 top-k through ForwardContext
        # and reuse the target-aligned rows on later MTP steps.
        self.share_topk_for_mtp_iteration = False
        self._verify_scratch: dict[
            tuple[int, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._active_verify_width: int | None = None
        self._last_pool = None
        self._draft_scratch: dict[
            tuple[int, torch.device],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}

    @staticmethod
    def _full_backend(ctx: ForwardContext):
        return getattr(ctx.attn_backend, "full_attn_backend", ctx.attn_backend)

    def _metadata(self, ctx: ForwardContext):
        backend = self._full_backend(ctx)
        candidates = []
        if ctx.forward_mode.is_extend_or_mixed():
            candidates.extend(("forward_extend_metadata", "forward_prefill_metadata"))
        elif getattr(backend, "spec_num_tokens", 1) > 1 and not getattr(
            backend, "is_draft", False
        ):
            candidates.append("forward_prefill_metadata")
        candidates.extend(("forward_decode_metadata", "forward_metadata"))
        for name in candidates:
            metadata = getattr(backend, name, None)
            if metadata is not None and getattr(metadata, "page_tables", None):
                return metadata
        raise RuntimeError("QSA requires group-aware MHA metadata")

    @staticmethod
    def _seq_lens(metadata) -> torch.Tensor:
        value = getattr(metadata, "seq_lens", None)
        if value is None:
            value = getattr(metadata, "cache_seqlens_int32", None)
        if value is None:
            raise RuntimeError("QSA metadata has no sequence lengths")
        return value

    @staticmethod
    def _query_lengths(metadata, total_tokens: int, bs: int):
        """Per-request query lengths as a view, diff, or uniform scalar."""

        values = getattr(metadata, "extend_seq_lens", None)
        if values is not None and values.numel() >= bs:
            return values[:bs]
        cu = getattr(metadata, "cu_seqlens_q", None)
        if cu is None:
            cu = getattr(metadata, "cu_extend_seq_lens", None)
        if cu is not None and cu.numel() >= bs + 1:
            return cu[1 : bs + 1] - cu[:bs]
        if bs and total_tokens % bs == 0:
            return total_tokens // bs
        raise RuntimeError("QSA could not infer query lengths")

    def _logical_layout(
        self,
        metadata,
        total_tokens: int,
        bs: int,
        query_lengths: torch.Tensor | int | None = None,
    ):
        seq_lens = self._seq_lens(metadata)[:bs]
        if query_lengths is None:
            lengths = self._query_lengths(metadata, total_tokens, bs)
        elif isinstance(query_lengths, int):
            lengths = query_lengths
        else:
            lengths = query_lengths[:bs]
        positions, requests = qwen4_exp_qsa_logical_layout(
            seq_lens, lengths, total_tokens
        )
        return positions, requests, lengths

    @staticmethod
    def _decode_query_lengths(
        ctx: ForwardContext,
        total_tokens: int,
        *,
        force_uniform: bool = False,
    ) -> int | None:
        """Derive uniform query lengths without stale draft-step metadata."""

        if not ctx.bs or (
            not force_uniform
            and (ctx.forward_mode is None or not ctx.forward_mode.is_decode())
        ):
            return None
        if total_tokens % ctx.bs:
            raise RuntimeError(
                "Qwen4-Exp QSA decode rows must be divisible by batch size"
            )
        return total_tokens // ctx.bs

    @staticmethod
    def _group_cache_locs(
        logical_positions: torch.Tensor,
        request_indices: torch.Tensor,
        qsa_page_table: torch.Tensor,
        qsa_expansion: int,
        qsa_page_size: int,
        recent_page_table: torch.Tensor,
        recent_expansion: int,
        recent_page_size: int,
        compress_ratio: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map flattened token positions into both QSA cache fields at once."""

        return qwen4_exp_qsa_group_cache_locs(
            logical_positions,
            request_indices,
            qsa_page_table,
            qsa_expansion,
            qsa_page_size,
            recent_page_table,
            recent_expansion,
            recent_page_size,
            compress_ratio,
        )

    def _field_layer_id(self, pool) -> int:
        mapper = getattr(pool, "_field_layer_id", None)
        return mapper(self.layer_id) if mapper is not None else self.layer_id

    @staticmethod
    def _backend_group_page_size(backend, group_id: str) -> int:
        """Return the page size used by one backend cache-group consumer."""
        return int(backend._consumer_page_size(group_id))

    def _fields(self, pool):
        layer_id = self._field_layer_id(pool)
        raw = pool.arena.field(qsa_raw_key_field(layer_id))
        compressed = pool.arena.field(qsa_compressed_field(layer_id))
        rope_positions = pool.arena.field(qsa_rope_position_field(layer_id))
        return raw, compressed, rope_positions

    @staticmethod
    def _page_table_expansion(consumer_page_size: int, logical_page_size: int) -> int:
        """Consumer entries per logical page, reversing the MHA expansion."""

        consumer_page_size = int(consumer_page_size)
        if logical_page_size == consumer_page_size:
            return 1
        if logical_page_size % consumer_page_size:
            raise ValueError(
                "Qwen4-Exp QSA logical page size must be divisible by the "
                "attention consumer page size "
                f"({logical_page_size} vs {consumer_page_size})"
            )
        return logical_page_size // consumer_page_size

    def _project_qk(self, hidden_states, positions):
        qk, _ = self.index_qk_proj(hidden_states)
        q, k = qk.split(
            [
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ],
            dim=-1,
        )
        k = k.reshape(-1, 1, self.index_head_dim)
        rotary = self.rotary_emb
        if not rotary.is_neox_style:
            raise ValueError("QSA indexer RoPE requires neox-style embeddings")
        sections = getattr(rotary, "mrope_section", None)
        # Fused per-head Gemma RMSNorm + neox RoPE straight off the GEMM
        # view; raw keys stay unnormalized until all members of a
        # compression group have been averaged, matching the checkpoint
        # reference.
        q = qwen4_exp_qsa_norm_rope(
            q,
            positions,
            self.q_layernorm.gemma_weight,
            self.q_layernorm.variance_epsilon,
            rotary.cos_sin_cache,
            num_heads=self.index_n_heads,
            sections=(tuple(sections) if positions.ndim == 2 and sections else None),
            interleaved=bool(getattr(rotary, "mrope_interleaved", False)),
        )
        return q, k

    @staticmethod
    def _position_values(rope_positions: torch.Tensor) -> torch.Tensor:
        """Reshape positions to ``[tokens, 3]`` as a strided, cast-free view."""

        values = (
            rope_positions
            if rope_positions.ndim == 2
            else rope_positions.unsqueeze(0).expand(3, -1)
        )
        return values.reshape(3, -1).T

    def _write_recent_cache(
        self,
        token_k: torch.Tensor,
        position_values: torch.Tensor,
        logical_positions: torch.Tensor,
        request_indices: torch.Tensor,
        recent_locs: torch.Tensor,
        write_mask: torch.Tensor,
        raw: torch.Tensor,
        position_cache: torch.Tensor,
    ) -> None:
        """Write one request's latest compression window into its cache page."""

        qwen4_exp_qsa_recent_write(
            token_k,
            logical_positions,
            request_indices,
            recent_locs,
            position_values,
            raw,
            position_cache,
            self.recent_page_size,
            self.compress_ratio,
            write_mask=write_mask,
        )

    def _stage_verified(
        self,
        token_k: torch.Tensor,
        position_values: torch.Tensor,
        logical_positions: torch.Tensor,
        recent_locs: torch.Tensor,
        bs: int,
        pool,
    ) -> None:
        """Stage target-verify candidates until their accepted width is known."""

        if bs <= 0 or token_k.shape[0] % bs:
            raise RuntimeError("QSA target-verify rows must be divisible by batch size")
        width = token_k.shape[0] // bs
        key = (bs, width)
        scratch = self._verify_scratch.get(key)
        if scratch is None:
            scratch = (
                token_k.new_empty((bs, width, 1, self.index_head_dim)),
                position_values.new_empty((bs, width, 3)),
                logical_positions.new_empty((bs, width)),
                recent_locs.new_empty((bs, width)),
            )
            self._verify_scratch[key] = scratch
        # One fused kernel snapshots all four tensors; separate ``copy_``
        # launches would add four small kernels to every layer's verify step.
        qwen4_exp_qsa_stage_verify(
            token_k,
            position_values,
            logical_positions,
            recent_locs,
            *scratch,
        )
        self._active_verify_width = width
        self._last_pool = pool

    def _draft_scratch_buffers(
        self,
        token_k: torch.Tensor,
        position_values: torch.Tensor,
        logical_positions: torch.Tensor,
        bs: int,
        *,
        reset: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the request-local raw-key ring for one draft MTP round."""

        key = (bs, token_k.device)
        scratch = self._draft_scratch.get(key)
        if scratch is None:
            scratch = (
                token_k.new_empty((bs, self.compress_ratio, 1, self.index_head_dim)),
                position_values.new_empty((bs, 3)),
                logical_positions.new_full(
                    (bs, self.compress_ratio), _DRAFT_INVALID_POSITION
                ),
            )
            self._draft_scratch[key] = scratch
        elif reset:
            scratch[2].fill_(_DRAFT_INVALID_POSITION)
        return scratch

    @staticmethod
    def _draft_accepted_write_mask(
        ctx: ForwardContext,
        seq_lens: torch.Tensor,
        query_lengths: torch.Tensor | int,
        logical_positions: torch.Tensor,
        request_indices: torch.Tensor,
        recent_locs: torch.Tensor,
    ) -> torch.Tensor:
        """Select extend rows and the accepted prefix of draft verify rows."""

        if ctx.accept_lengths is None:
            raise RuntimeError("QSA draft accepted writes require acceptance lengths")
        lengths = (
            query_lengths if isinstance(query_lengths, int) else query_lengths[: ctx.bs]
        )
        starts = seq_lens[: ctx.bs] - lengths
        request_rows = request_indices.to(torch.long)
        offsets = logical_positions - starts.index_select(0, request_rows)
        accepted = ctx.accept_lengths[: ctx.bs].to(logical_positions.dtype)
        limits = accepted.index_select(0, request_rows)
        return (
            (recent_locs > 0)
            & (offsets >= 0)
            & ((request_indices < ctx.num_extends) | (offsets < limits))
        )

    def commit_verified(self, accepted_lengths: torch.Tensor) -> None:
        """Commit only accepted target-verify raw keys and group-start positions."""

        bs = accepted_lengths.shape[0]
        width = self._active_verify_width
        candidates = [
            key
            for key in self._verify_scratch
            if key[0] >= bs and (width is None or key[1] == width)
        ]
        if not candidates or self._last_pool is None:
            return
        key = min(candidates, key=lambda value: value[0])
        _, width = key
        token_k, position_values, logical_positions, recent_locs = self._verify_scratch[
            key
        ]
        token_k = token_k[:bs]
        position_values = position_values[:bs]
        logical_positions = logical_positions[:bs]
        recent_locs = recent_locs[:bs]
        accepted = accepted_lengths.to(torch.long).clamp(min=0, max=width)
        steps = torch.arange(width, device=accepted.device).unsqueeze(0)
        write_mask = steps < accepted.unsqueeze(1)
        last_indices = (accepted - 1).clamp_min(0).unsqueeze(1)
        last_positions = logical_positions.gather(1, last_indices)
        write_mask &= logical_positions > last_positions - self.compress_ratio
        write_mask &= recent_locs > 0
        raw, _, position_cache = self._fields(self._last_pool)
        request_indices = torch.arange(bs, device=accepted.device).repeat_interleave(
            width
        )
        self._write_recent_cache(
            token_k.reshape(-1, 1, self.index_head_dim),
            position_values.reshape(-1, 3),
            logical_positions.reshape(-1),
            request_indices,
            recent_locs.reshape(-1),
            write_mask.reshape(-1),
            raw,
            position_cache,
        )

    def _write_and_compress(
        self,
        token_k: torch.Tensor,
        rope_positions: torch.Tensor,
        logical_positions: torch.Tensor,
        request_indices: torch.Tensor,
        qsa_locs: torch.Tensor,
        recent_locs: torch.Tensor,
        pool,
        *,
        recent_request_limit: int | None = None,
        write_mask: torch.Tensor | None = None,
        draft_scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        stage_draft: bool = False,
    ) -> None:
        raw, compressed, position_cache = self._fields(pool)
        if not logical_positions.shape[0]:
            return
        position_values = self._position_values(rope_positions)
        rotary = self.rotary_emb
        sections = getattr(rotary, "mrope_section", None)
        # PDL lets the raw-key write launch while the compression kernel's
        # tail drains; the writer still waits before touching raw pages.
        pdl = pdl_enabled()
        qwen4_exp_qsa_compress_and_store(
            token_k,
            logical_positions,
            request_indices,
            recent_locs,
            raw,
            position_values,
            position_cache,
            self.k_layernorm.gemma_weight,
            self.k_layernorm.variance_epsilon,
            rotary.cos_sin_cache,
            qsa_locs,
            compressed,
            self.recent_page_size,
            self.compress_ratio,
            self.compressed_token_page_size,
            sections=tuple(sections) if sections else None,
            interleaved=bool(getattr(rotary, "mrope_interleaved", False)),
            write_mask=write_mask,
            draft_raw_cache=None if draft_scratch is None else draft_scratch[0],
            draft_position_cache=None if draft_scratch is None else draft_scratch[1],
            draft_logical_positions=(
                None if draft_scratch is None else draft_scratch[2]
            ),
            enable_pdl=pdl,
        )
        if stage_draft:
            if draft_scratch is None:
                raise RuntimeError("QSA draft staging requires scratch buffers")
            qwen4_exp_qsa_stage_draft(
                token_k,
                position_values,
                logical_positions,
                request_indices,
                recent_locs,
                draft_scratch[0],
                draft_scratch[1],
                draft_scratch[2],
                self.compress_ratio,
                enable_pdl=pdl,
            )
            return
        qwen4_exp_qsa_recent_write(
            token_k,
            logical_positions,
            request_indices,
            recent_locs,
            position_values,
            raw,
            position_cache,
            self.recent_page_size,
            self.compress_ratio,
            write_mask=write_mask,
            request_limit=recent_request_limit,
            enable_pdl=pdl,
        )

    def _topk_solution(
        self,
        rows: int,
        page_table: torch.Tensor,
        page_expansion: int,
        page_size: int,
    ) -> str:
        """Route block top-k between the streaming and materialized paths.

        ``TOKENSPEED_QWEN4_EXP_QSA_TOPK_PATH`` pins the backend
        (``stream``/``logits``) and ``TOKENSPEED_QWEN4_EXP_QSA_MAX_LOGITS_MB``
        caps the materialized score matrix for ``auto`` routing; both are
        declared on the shared ``envs`` table in ``runtime.utils.env``.
        """

        path = envs.TOKENSPEED_QWEN4_EXP_QSA_TOPK_PATH.get()
        if path not in ("auto", "stream", "logits"):
            raise ValueError(
                "TOKENSPEED_QWEN4_EXP_QSA_TOPK_PATH must be 'auto', 'stream', "
                f"or 'logits', got {path!r}"
            )
        if path != "auto":
            return path
        num_blocks = (page_table.shape[1] + page_expansion - 1) // page_expansion
        num_blocks *= page_size
        budget_mb = envs.TOKENSPEED_QWEN4_EXP_QSA_MAX_LOGITS_MB.get()
        # The materialized DSA-selection path measured 2-24x faster than the
        # fused streaming path across decode/prefill shapes, so auto prefers
        # it whenever the score matrix fits the configured budget; beyond
        # the budget routing falls back to zero-materialization streaming.
        return "logits" if rows * num_blocks * 4 <= budget_mb << 20 else "stream"

    def _select_tokens(
        self,
        q: torch.Tensor,
        logical_positions: torch.Tensor,
        request_indices: torch.Tensor,
        qsa_page_table: torch.Tensor,
        compressed: torch.Tensor,
        *,
        qsa_page_expansion: int = 1,
        complete_blocks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output_width = self.token_topk + self.compress_ratio - 1
        if q.shape[0] == 0:
            return torch.empty((0, output_width), dtype=torch.int32, device=q.device)
        page_size = compressed.shape[1]
        if complete_blocks is None:
            complete_blocks = qwen4_exp_qsa_complete_blocks(
                logical_positions, self.compress_ratio
            )
        cache = compressed.view(-1, 1, self.index_head_dim)
        # Streaming block top-k: scores are computed tile by tile against
        # each request's page table, so neither the expanded slot matrix nor
        # the dense FP32 score matrix is ever materialized.
        selected_blocks = qwen4_exp_qsa_block_topk(
            q,
            cache,
            qsa_page_table,
            request_indices,
            complete_blocks,
            page_size=page_size,
            block_topk=self.block_topk,
            page_expansion=qsa_page_expansion,
            solution=self._topk_solution(
                q.shape[0], qsa_page_table, qsa_page_expansion, page_size
            ),
            enable_pdl=pdl_enabled(),
        )
        return qwen4_exp_qsa_selected_tokens(
            selected_blocks,
            complete_blocks,
            logical_positions,
            self.compress_ratio,
            self.token_topk,
        )

    @break_point
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        """Top-k token indices for this layer, one eager breakable-graph break.

        Everything here is per-request: the logical layout comes from the live
        query lengths, the compressed / recent writes address freshly built page
        tables, and the compress / top-k grids are sized by the batch. A prefill
        capture sees one dummy request, so graphing this would bake that request's
        layout and its dead per-forward page tables into every replay. Running it
        as a break keeps it all live; direct call off the capture path.
        """
        num_real = current_valid_rows()
        if num_real is not None:
            (hidden_states,) = slice_to_real_tokens(num_real, hidden_states)
            # mrope positions come in as [3, tokens]: the token axis is last.
            positions = (
                positions[..., :num_real]
                if positions.ndim == 2
                else positions[:num_real]
            )
        metadata = self._metadata(ctx)
        query_lengths = self._decode_query_lengths(ctx, hidden_states.shape[0])
        logical, requests, lengths = self._logical_layout(
            metadata,
            hidden_states.shape[0],
            ctx.bs,
            query_lengths=query_lengths,
        )
        q, token_k = self._project_qk(hidden_states, positions)
        pool = ctx.token_to_kv_pool
        load_tracker = getattr(pool, "layerwise_load_tracker", None)
        if load_tracker is not None:
            load_tracker.wait_for_layer(self.layer_id)
        _, compressed, _ = self._fields(pool)
        full_backend = self._full_backend(ctx)
        verify_bs = ctx.bs - ctx.num_extends
        is_target_verify = (
            (ctx.forward_mode.is_decode() or ctx.forward_mode.is_mixed())
            and verify_bs > 0
            and getattr(full_backend, "spec_num_tokens", 1) > 1
            and not getattr(full_backend, "is_draft", False)
        )
        is_draft = getattr(full_backend, "is_draft", False)
        is_draft_first_step = is_draft and ctx.accept_lengths is not None
        is_draft_decode_step = (
            is_draft and ctx.accept_lengths is None and ctx.forward_mode.is_decode()
        )
        consumer_page_size = self._backend_group_page_size(
            full_backend, QWEN4_EXP_QSA_CACHE_GROUP
        )
        qsa_expansion = self._page_table_expansion(
            consumer_page_size, self.compressed_token_page_size
        )
        recent_consumer_page_size = self._backend_group_page_size(
            full_backend, QWEN4_EXP_QSA_RECENT_CACHE_GROUP
        )
        recent_expansion = self._page_table_expansion(
            recent_consumer_page_size, self.recent_page_size
        )
        qsa_page_table = metadata.page_tables[QWEN4_EXP_QSA_CACHE_GROUP]
        recent_page_table = metadata.page_tables[QWEN4_EXP_QSA_RECENT_CACHE_GROUP]
        qsa_locs, recent_locs, complete_blocks = self._group_cache_locs(
            logical,
            requests,
            qsa_page_table,
            qsa_expansion,
            self.compressed_token_page_size,
            recent_page_table,
            recent_expansion,
            self.recent_page_size,
            self.compress_ratio,
        )
        draft_scratch = None
        write_mask = None
        if is_draft_first_step:
            draft_scratch = self._draft_scratch_buffers(
                token_k,
                self._position_values(positions),
                logical,
                ctx.bs,
                reset=True,
            )
            write_mask = self._draft_accepted_write_mask(
                ctx,
                self._seq_lens(metadata),
                lengths,
                logical,
                requests,
                recent_locs,
            )
        elif is_draft_decode_step:
            if token_k.shape[0] != ctx.bs:
                raise RuntimeError("QSA draft decode requires one row per request")
            draft_scratch = self._draft_scratch_buffers(
                token_k,
                self._position_values(positions),
                logical,
                ctx.bs,
            )
        self._write_and_compress(
            token_k,
            positions,
            logical,
            requests,
            qsa_locs,
            recent_locs,
            pool,
            recent_request_limit=ctx.num_extends if is_target_verify else None,
            write_mask=write_mask,
            draft_scratch=draft_scratch if is_draft_decode_step else None,
            stage_draft=is_draft_decode_step,
        )
        if is_target_verify:
            verify_tokens = verify_bs * int(full_backend.spec_num_tokens)
            if verify_tokens > token_k.shape[0]:
                raise RuntimeError("QSA verify rows exceed the current input")
            self._stage_verified(
                token_k[-verify_tokens:],
                self._position_values(positions)[-verify_tokens:],
                logical[-verify_tokens:],
                recent_locs[-verify_tokens:],
                verify_bs,
                pool,
            )
        if self.share_topk_for_mtp_iteration:
            shared_topk = getattr(ctx, "dsa_decode_topk", None)
            num_rows = hidden_states.shape[0]
            if shared_topk is not None and shared_topk.shape[0] < num_rows:
                raise RuntimeError(
                    "QSA MTP top-k reuse requires target-aligned step-0 indices"
                )
            if shared_topk is not None:
                return shared_topk[:num_rows]
        selected = self._select_tokens(
            q,
            logical,
            requests,
            qsa_page_table,
            compressed,
            qsa_page_expansion=qsa_expansion,
            complete_blocks=complete_blocks,
        )
        if self.share_topk_for_mtp_iteration:
            ctx.dsa_decode_topk = selected
        return selected

    @break_point
    def sparse_attention(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor | None,
        attention_layer,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run QSA while preserving the backend's layerwise PD step contract.

        The QSA layer's second eager breakable-graph break: it writes the paged KV
        cache at live cache locations and runs the sparse kernel over a per-request
        slot table, neither of which survives capture at the dummy batch shape. The
        padded rows are dropped up front because those kernels are indexed by the
        live metadata's token count.
        """
        num_real = current_valid_rows()
        if num_real is not None:
            q, k, v, gate, out_cache_loc, topk_indices = slice_to_real_tokens(
                num_real, q, k, v, gate, out_cache_loc, topk_indices
            )
        with ctx.attn_backend.record_pd_cache_step(
            ctx.forward_mode,
            save_kv_cache=True,
            record_kv_cache=None,
        ):
            return self._sparse_attention_impl(
                q=q,
                k=k,
                v=v,
                gate=gate,
                attention_layer=attention_layer,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
                topk_indices=topk_indices,
            )

    def _sparse_attention_impl(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor | None,
        attention_layer,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        metadata = self._metadata(ctx)
        full_backend = self._full_backend(ctx)
        full_locs = (
            out_cache_loc
            if getattr(full_backend, "is_draft", False)
            else metadata.out_cache_locs[FULL_ATTENTION]
        )[: k.shape[0]]
        q = q.view(-1, attention_layer.tp_q_head_num, attention_layer.head_dim)
        k = k.view(-1, attention_layer.tp_k_head_num, attention_layer.head_dim)
        v = v.view(-1, attention_layer.tp_v_head_num, attention_layer.v_head_dim)
        ctx.token_to_kv_pool.set_kv_buffer(
            attention_layer,
            full_locs,
            k,
            v,
            attention_layer.k_scale,
            attention_layer.v_scale,
        )
        query_lengths = self._decode_query_lengths(
            ctx,
            q.shape[0],
            force_uniform=ctx.accept_lengths is not None,
        )
        logical, requests, _ = self._logical_layout(
            metadata,
            q.shape[0],
            ctx.bs,
            query_lengths=query_lengths,
        )
        page_table = metadata.page_tables[FULL_ATTENTION]
        k_cache = ctx.token_to_kv_pool.get_key_buffer(attention_layer.layer_id)
        v_cache = ctx.token_to_kv_pool.get_value_buffer(attention_layer.layer_id)
        page_size = self._backend_group_page_size(full_backend, FULL_ATTENTION)
        slots = qwen4_exp_qsa_sparse_slots(
            topk_indices, logical, requests, page_table, page_size
        )
        fp8_dtypes = (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        )
        output = qwen4_exp_qsa_sparse_attention(
            q,
            k_cache,
            v_cache,
            slots,
            scale=attention_layer.scaling,
            k_scale=(
                (1.0 if attention_layer.k_scale is None else attention_layer.k_scale)
                if k_cache.dtype in fp8_dtypes
                else None
            ),
            v_scale=(
                (1.0 if attention_layer.v_scale is None else attention_layer.v_scale)
                if v_cache.dtype in fp8_dtypes
                else None
            ),
        )
        output = output.reshape(q.shape[0], -1)
        if gate is not None:
            sigmoid_mul(output, gate)
        return output


__all__ = [
    "QWEN4_EXP_QSA_CACHE_GROUP",
    "QWEN4_EXP_QSA_RECENT_CACHE_GROUP",
    "QSAIndexer",
    "qsa_compressed_field",
    "qsa_raw_key_field",
    "qsa_rope_position_field",
]
