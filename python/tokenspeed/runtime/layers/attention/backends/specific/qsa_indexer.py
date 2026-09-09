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

"""QSA cache-table metadata and target verification, independent of attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.paged.group_tables import (
    GroupTableSpec,
    GroupTableStacks,
)
from tokenspeed.runtime.layers.attention.backends.support import CudaGraphSupport
from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_QSA_CACHE_GROUP,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    cache_field_layer_id,
)
from tokenspeed.runtime.layers.attention.qsa.verify_state import QSAVerifyState

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tokenspeed.runtime.layers.attention.backends.paged.router import (
        CacheGroupRouter,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool


@dataclass(frozen=True)
class QSAIndexerMetadata:
    """One invocation's request lengths and stable raw cache-group tables."""

    seq_lens: torch.Tensor
    extend_seq_lens: torch.Tensor | None
    qsa_block_table: torch.Tensor
    recent_block_table: torch.Tensor


class QSAIndexerBackend(AttentionBackend):
    """Prepare the two indexer cache groups once for all local QSA layers."""

    # Token-shaped side writes do not support padded prefill-graph replay.
    cuda_graph_support = CudaGraphSupport(prefill_graph=False)
    cache_consumer_families = frozenset({"history"})

    def __init__(self, config: AttnConfig, full_attn_backend: CacheGroupRouter) -> None:
        super().__init__(config, config.component(SoftmaxAttnConfig))
        self._config = config
        self.full_attn_backend = full_attn_backend
        self.max_context_len = int(config.context_len)
        self._table_specs: tuple[GroupTableSpec, ...] = ()
        self._tables: GroupTableStacks | None = None
        self._seq_lens: torch.Tensor | None = None
        self._decode_views: dict[int, QSAIndexerMetadata] = {}
        self.forward_extend_metadata: QSAIndexerMetadata | None = None
        self.forward_decode_metadata: QSAIndexerMetadata | None = None
        self._active_metadata: QSAIndexerMetadata | None = None
        self._verify_state: QSAVerifyState | None = None

    def set_cache_pool(self, cache_pool: CachePool) -> None:
        if self.cache_pool is cache_pool:
            return
        if self.cache_pool is not None:
            raise RuntimeError("QSA indexer backend cannot be rebound to another pool")
        groups = (QWEN4_EXP_QSA_CACHE_GROUP, QWEN4_EXP_QSA_RECENT_CACHE_GROUP)
        local_groups = {
            field.group_id
            for field in cache_pool.arena.plan.fields
            if field.group_id in groups
            and cache_field_layer_id(field.field_id) in cache_pool.field_layer_range
        }
        if local_groups != set(groups):
            raise RuntimeError(
                "QSA indexer cache view requires compressed and recent fields"
            )
        specs = {spec.group_id: spec for spec in cache_pool.arena.cache_group_specs}
        self._table_specs = tuple(
            GroupTableSpec(
                group_id=gid,
                block_granularity=specs[gid].block_granularity,
                kernel_page_size=specs[gid].block_granularity,
                max_num_pages=(self.max_context_len + specs[gid].block_granularity - 1)
                // specs[gid].block_granularity,
            )
            for gid in groups
        )
        self.cache_pool = cache_pool
        if not self.is_draft and self.spec_num_tokens > 1:
            self._verify_state = QSAVerifyState(self._config, cache_pool)

    def preallocate_verify_workspace(self, max_bs: int, draft_token_num: int) -> int:
        """Allocate target staging for this capacity and return its byte count."""
        if self._verify_state is None:
            return 0
        return self._verify_state.preallocate_verify_workspace(max_bs, draft_token_num)

    def verify_staging_buffers(
        self, layer_id: int, bs: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return this local layer's key, position, logical-position and slot views."""
        if self._verify_state is None:
            raise RuntimeError(
                "QSA staging requires a speculative target indexer backend"
            )
        return self._verify_state.verify_staging_buffers(layer_id, bs)

    def commit_after_mtp_verify(
        self, accepted_lengths: torch.Tensor, *, num_extends: int
    ) -> None:
        """Commit accepted decode rows, excluding the leading extend requests."""
        if self._verify_state is not None:
            self._verify_state.commit_after_mtp_verify(
                accepted_lengths, num_extends=num_extends
            )

    def init_cuda_graph_state(self, max_bs: int, **kwargs) -> None:
        del kwargs
        if not self._table_specs:
            raise RuntimeError(
                "QSA cache pool must be bound before metadata allocation"
            )
        if self._tables is not None:
            if self._tables.max_bs != max_bs:
                raise RuntimeError(
                    "QSA metadata capacity cannot change after allocation"
                )
            return
        # Ratio one preserves scheduler block IDs. The shared fill only copies,
        # normalizes holes to zero, and clears padded requests and column tails.
        self._tables = GroupTableStacks(
            self._table_specs,
            max_bs=max_bs,
            max_tokens_per_req=self.spec_num_tokens,
            device=self.device,
        )
        self._seq_lens = torch.empty(max_bs, dtype=torch.int32, device=self.device)

    def _fill_tables(
        self, bs: int, actual_bs: int, block_tables: Mapping[str, torch.Tensor]
    ) -> None:
        if self._tables is None:
            raise RuntimeError("QSA metadata must be allocated before forward")
        if actual_bs > 0:
            missing = [gid for gid in self._tables.group_ids if gid not in block_tables]
            if missing:
                raise RuntimeError(f"QSA indexer is missing cache groups {missing}")
            if any(block_tables[gid].ndim != 2 for gid in self._tables.group_ids):
                raise RuntimeError("QSA indexer requires two-dimensional block tables")
            if any(block_tables[gid].shape[1] == 0 for gid in self._tables.group_ids):
                raise RuntimeError("QSA indexer requires nonempty block-table columns")
        self._tables.fill(bs, actual_bs, block_tables)

    def _metadata(
        self, bs: int, seq_lens: torch.Tensor, extend_seq_lens: torch.Tensor | None
    ) -> QSAIndexerMetadata:
        return QSAIndexerMetadata(
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            qsa_block_table=self._tables.table(QWEN4_EXP_QSA_CACHE_GROUP, bs),
            recent_block_table=self._tables.table(QWEN4_EXP_QSA_RECENT_CACHE_GROUP, bs),
        )

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        block_tables: Mapping[str, torch.Tensor],
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_with_prefix: bool,
        **kwargs,
    ) -> None:
        del req_pool_indices, extend_seq_lens_cpu, extend_prefix_lens
        del extend_prefix_lens_cpu, extend_with_prefix, kwargs
        if not (forward_mode.is_extend_or_mixed() or forward_mode.is_idle()):
            raise RuntimeError("QSA decode metadata uses refresh_decode_metadata")
        self._fill_tables(bs, 0 if forward_mode.is_idle() else bs, block_tables)
        self.full_attn_backend.sparse_topk.qsa_metadata = None
        # Keep this slot independent: draft extend init is followed by a decode
        # refresh before the first (still extend-shaped) model invocation.
        lengths = extend_seq_lens[:num_extends]
        if num_extends < bs:
            lengths = torch.cat(
                (
                    lengths,
                    torch.full_like(seq_lens[num_extends:bs], self.spec_num_tokens),
                )
            )
        self.forward_extend_metadata = self._metadata(
            bs, seq_lens[:bs].clone(), lengths
        )
        self._active_metadata = None

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        forward_mode: ForwardMode,
        block_tables: Mapping[str, torch.Tensor],
        **kwargs,
    ) -> None:
        del req_pool_indices, kwargs
        if not forward_mode.is_decode_or_idle():
            raise RuntimeError("QSA refresh_decode_metadata serves decode only")
        self._fill_tables(bs, actual_bs, block_tables)
        self.full_attn_backend.sparse_topk.qsa_metadata = None
        torch.clamp_min(
            seq_lens[:bs],
            1 if self.is_draft else self.spec_num_tokens,
            out=self._seq_lens[:bs],
        )
        metadata = self._decode_views.get(bs)
        if metadata is None:
            metadata = self._metadata(bs, self._seq_lens[:bs], None)
            self._decode_views[bs] = metadata
        self.forward_decode_metadata = metadata
        self._active_metadata = None

    def metadata_for(self, forward_mode: ForwardMode) -> QSAIndexerMetadata:
        metadata = (
            self.forward_extend_metadata
            if forward_mode.is_extend_or_mixed()
            else self.forward_decode_metadata
        )
        if metadata is None:
            raise RuntimeError(f"QSA indexer has no {forward_mode} metadata")
        self._active_metadata = metadata
        return metadata

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        if self._seq_lens is None:
            raise RuntimeError("QSA draft metadata must be allocated before advancing")
        bs = seq_lens.shape[0]
        self._seq_lens[:bs].copy_(seq_lens)
        # Step zero computes logical positions before publishing its accepted
        # frontier. Its layout retains this tensor so the subsequent write mask
        # sees the frontier without recomputing the original verify window.
        active = self._active_metadata
        if active is not None and active is not self.forward_decode_metadata:
            active.seq_lens[:bs].copy_(seq_lens)

    def update_draft_forward_metadata(self, frontier: torch.Tensor) -> None:
        self.advance_draft_forward_metadata(frontier)

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        self.advance_draft_forward_metadata(
            block_seq_lens[:bs].clamp(self.spec_num_tokens, self.max_context_len)
        )
