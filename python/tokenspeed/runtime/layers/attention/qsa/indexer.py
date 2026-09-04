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

"""Model-owned QSA projection and top-k indexer."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.triton.qwen4_exp_qsa import (
    qwen4_exp_qsa_block_topk,
    qwen4_exp_qsa_compress_and_store,
    qwen4_exp_qsa_recent_write,
    qwen4_exp_qsa_selected_slots,
)
from tokenspeed_kernel.platform import pdl_enabled
from torch import nn

from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    current_valid_rows,
    slice_to_real_tokens,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.backends.specific.qsa import QSAAttnBackend
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
_PERSISTENT_TOPK_WORKSPACE_BYTES = 1024 * 1024


class QSAIndexer(nn.Module):
    """Config-driven QSA indexer backed by TokenSpeed cache groups."""

    def __init__(
        self,
        config,
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
        self.compressed_token_page_size = (
            QWEN4_EXP_QSA_COMPRESSED_ROWS_PER_PAGE * self.compress_ratio
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
        # Draft QSA indexers can publish step-0 top-k through backend scratch
        # and reuse the target-aligned rows on later MTP steps.
        self.share_topk_for_mtp_iteration = False
        self.qsa_coordinator: QSAAttnBackend | None = None
        self._draft_scratch: dict[
            tuple[int, torch.device],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        # Stable caller-owned scratch lets the materialized score path use the
        # persistent radix top-k without allocating workspace per layer call.
        self.register_buffer(
            "_persistent_topk_workspace",
            torch.empty((_PERSISTENT_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8),
            persistent=False,
        )

    @staticmethod
    def _full_backend(ctx: ForwardContext) -> QSAAttnBackend:
        """Return the registered QSA backend behind the optional hybrid node."""

        backend = getattr(ctx.attn_backend, "full_attn_backend", ctx.attn_backend)
        if not isinstance(backend, QSAAttnBackend):
            raise RuntimeError(
                "Qwen4-Exp QSA requires the qsa attention backend, got "
                f"{type(backend).__name__}"
            )
        return backend

    def _fields(self, pool):
        layer_id = pool._field_layer_id(self.layer_id)
        raw = pool.arena.field(qsa_raw_key_field(layer_id))
        compressed = pool.arena.field(qsa_compressed_field(layer_id))
        rope_positions = pool.arena.field(qsa_rope_position_field(layer_id))
        return raw, compressed, rope_positions

    def verify_commit_fields(self, pool) -> tuple[torch.Tensor, torch.Tensor]:
        """Return this layer's raw-key and group-start position fields."""

        raw, _, position_cache = self._fields(pool)
        return raw, position_cache

    def _project_qk_raw(self, hidden_states):
        """Project packed raw index queries and keys without materializing copies."""

        qk, _ = self.index_qk_proj(hidden_states)
        q, k = qk.split(
            [
                self.index_n_heads * self.index_head_dim,
                self.index_kv_heads * self.index_head_dim,
            ],
            dim=-1,
        )
        k = k.reshape(-1, 1, self.index_head_dim)
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

    def _verify_staging_buffers(
        self,
        token_k: torch.Tensor,
        position_values: torch.Tensor,
        logical_positions: torch.Tensor,
        recent_locs: torch.Tensor,
        bs: int,
        pool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return this layer's views into backend-owned verify staging."""

        if self.qsa_coordinator is None:
            raise RuntimeError("QSA verify staging ran before indexers were bound")
        return self.qsa_coordinator.verify_staging_buffers(
            self,
            token_k,
            position_values,
            logical_positions,
            recent_locs,
            bs,
            pool,
        )

    def _draft_scratch_buffers(
        self,
        token_k: torch.Tensor,
        position_values: torch.Tensor,
        logical_positions: torch.Tensor | None,
        bs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the request-local raw-key ring for one draft MTP round."""

        key = (bs, token_k.device)
        scratch = self._draft_scratch.get(key)
        if scratch is None:
            scratch = (
                token_k.new_empty((bs, self.compress_ratio, 1, self.index_head_dim)),
                position_values.new_empty((bs, 3)),
                torch.full(
                    (bs, self.compress_ratio),
                    _DRAFT_INVALID_POSITION,
                    dtype=torch.int64,
                    device=token_k.device,
                ),
            )
            self._draft_scratch[key] = scratch
        return scratch

    @staticmethod
    def _draft_accepted_write_mask(
        ctx: ForwardContext,
        accepted_seq_lens: torch.Tensor,
        logical_positions: torch.Tensor,
        request_indices: torch.Tensor,
        recent_locs: torch.Tensor,
    ) -> torch.Tensor:
        """Select extend rows and the accepted prefix of draft verify rows:
        the rows whose logical position lies under the published accepted
        frontier (``valid_cache_len + accept_len``)."""

        request_rows = request_indices.to(torch.long)
        frontier = (
            accepted_seq_lens[: ctx.bs]
            .to(logical_positions.dtype)
            .index_select(0, request_rows)
        )
        return (recent_locs > 0) & (
            (request_indices < ctx.num_extends) | (logical_positions < frontier)
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
        query: torch.Tensor | None = None,
        stage_verify_buffers: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None,
    ) -> torch.Tensor | None:
        raw, compressed, position_cache = self._fields(pool)
        position_values = self._position_values(rope_positions)
        rotary = self.rotary_emb
        if query is not None and not rotary.is_neox_style:
            raise ValueError("QSA indexer RoPE requires neox-style embeddings")
        sections = getattr(rotary, "mrope_section", None)
        # PDL lets the raw-key write launch while the compression kernel's
        # tail drains; the writer still waits before touching raw pages.
        pdl = pdl_enabled()
        prepared_query = qwen4_exp_qsa_compress_and_store(
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
            query=query,
            query_norm_weight=(
                self.q_layernorm.gemma_weight if query is not None else None
            ),
            query_norm_epsilon=(
                self.q_layernorm.variance_epsilon if query is not None else None
            ),
            num_query_heads=self.index_n_heads if query is not None else None,
            stage_verify_buffers=stage_verify_buffers,
            stage_draft=stage_draft,
        )
        if stage_draft:
            if draft_scratch is None:
                raise RuntimeError("QSA draft staging requires scratch buffers")
            return prepared_query
        if recent_request_limit == 0:
            # Pure target verification must not commit speculative raw keys;
            # the backend commits accepted rows after sampling.
            return prepared_query
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
        return prepared_query

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

    def _select_slots(
        self,
        q: torch.Tensor,
        logical_positions: torch.Tensor,
        request_indices: torch.Tensor,
        qsa_page_table: torch.Tensor,
        full_page_table: torch.Tensor,
        compressed: torch.Tensor,
        *,
        full_page_size: int,
        complete_blocks: torch.Tensor,
        qsa_page_expansion: int = 1,
    ) -> torch.Tensor:
        """Select logical QSA blocks and emit physical full-cache slots."""

        output_width = self.token_topk + self.compress_ratio - 1
        if q.shape[0] == 0:
            return torch.empty((0, output_width), dtype=torch.int32, device=q.device)
        page_size = compressed.shape[1]
        cache = compressed.view(-1, 1, self.index_head_dim)
        # Auto normally materializes scores for persistent radix selection;
        # oversized matrices retain the zero-materialization streaming path.
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
            persistent_topk_workspace=self._persistent_topk_workspace,
            enable_pdl=pdl_enabled(),
        )
        return qwen4_exp_qsa_selected_slots(
            selected_blocks,
            complete_blocks,
            logical_positions,
            request_indices,
            full_page_table,
            full_page_size,
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
        """Top-k physical cache slots for this layer, one eager graph break.

        Everything here is per-request: the logical layout comes from the live
        query lengths, the compressed / recent writes address this forward's
        rows of the router's page tables, and the compress / top-k grids are
        sized by the batch. A prefill capture sees one dummy request, so
        graphing this would bake that request's layout and one-row table views
        into every replay. Running it as a break keeps it all live; direct call
        off the capture path.
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
        raw_q, token_k = self._project_qk_raw(hidden_states)
        pool = ctx.token_to_kv_pool
        if pool.layerwise_load_tracker is not None:
            pool.layerwise_load_tracker.wait_for_layer(self.layer_id)
        _, compressed, _ = self._fields(pool)
        router = self._full_backend(ctx)
        verify_bs = ctx.bs - ctx.num_extends
        is_target_verify = (
            (ctx.forward_mode.is_decode() or ctx.forward_mode.is_mixed())
            and verify_bs > 0
            and router.spec_num_tokens > 1
            and not router.is_draft
        )
        is_draft = router.is_draft
        is_draft_first_step = is_draft and ctx.draft_narrowing is not None
        is_draft_decode_step = (
            is_draft and ctx.draft_narrowing is None and ctx.forward_mode.is_decode()
        )
        position_values = self._position_values(positions)
        draft_scratch = None
        if is_draft_decode_step and token_k.shape[0] != ctx.bs:
            raise RuntimeError("QSA draft decode requires one row per request")
        if is_draft_first_step or is_draft_decode_step:
            draft_scratch = self._draft_scratch_buffers(
                token_k,
                position_values,
                None,
                ctx.bs,
            )
        layout = router.qsa_forward_layout(
            ctx,
            hidden_states.shape[0],
            compressed_token_page_size=self.compressed_token_page_size,
            recent_page_size=self.recent_page_size,
            compress_ratio=self.compress_ratio,
            reset_draft_tags=(
                draft_scratch[2] if is_draft_first_step and draft_scratch else None
            ),
        )
        logical = layout.logical_positions
        requests = layout.request_indices
        qsa_locs = layout.qsa_locs
        recent_locs = layout.recent_locs
        complete_blocks = layout.complete_blocks
        write_mask = None
        if is_draft_first_step:
            # Layout uses the target's verify window. The draft then publishes
            # its accepted frontier for the write mask and the live attention.
            ctx.draft_narrowing.publish_accepted_prefix()
            write_mask = self._draft_accepted_write_mask(
                ctx,
                layout.seq_lens,
                logical,
                requests,
                recent_locs,
            )
        shared_topk = None
        if self.share_topk_for_mtp_iteration:
            shared_topk = router.sparse_topk.decode
            if (
                shared_topk is not None
                and shared_topk.shape[0] < hidden_states.shape[0]
            ):
                raise RuntimeError(
                    "QSA MTP top-k reuse requires target-aligned step-0 indices"
                )
        verify_scratch = None
        if is_target_verify:
            verify_tokens = verify_bs * router.spec_num_tokens
            if verify_tokens > token_k.shape[0]:
                raise RuntimeError("QSA verify rows exceed the current input")
            verify_scratch = self._verify_staging_buffers(
                token_k[-verify_tokens:],
                position_values[-verify_tokens:],
                logical[-verify_tokens:],
                recent_locs[-verify_tokens:],
                verify_bs,
                pool,
            )
        q = self._write_and_compress(
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
            # Later MTP iterations reuse step-0 top-k, so their projected Q
            # region is dead; K compression and draft staging still proceed.
            query=None if shared_topk is not None else raw_q,
            stage_verify_buffers=verify_scratch,
        )
        if shared_topk is not None:
            return shared_topk[: hidden_states.shape[0]]
        if q is None:
            raise RuntimeError("QSA fused query preparation did not return queries")
        selected_slots = self._select_slots(
            q,
            logical,
            requests,
            layout.qsa_page_table,
            router.stacks.table(FULL_ATTENTION, ctx.bs),
            compressed,
            full_page_size=router.stacks.group_kernel_page_size(FULL_ATTENTION),
            qsa_page_expansion=layout.qsa_page_expansion,
            complete_blocks=complete_blocks,
        )
        if self.share_topk_for_mtp_iteration:
            router.sparse_topk.decode = selected_slots
        return selected_slots


__all__ = [
    "QWEN4_EXP_QSA_CACHE_GROUP",
    "QWEN4_EXP_QSA_RECENT_CACHE_GROUP",
    "QSAIndexer",
    "qsa_compressed_field",
    "qsa_raw_key_field",
    "qsa_rope_position_field",
]
