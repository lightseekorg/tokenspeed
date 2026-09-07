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

"""Backend-owned QSA layout and speculative verification lifecycle.

The ordinary cache-group router owns the page tables. This runtime consumes
its resolved views and owns shared verify staging; the paged QSA leaf only
executes attention, and the model indexer owns weights and top-k selection.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.triton.qwen4_exp_qsa import (
    qwen4_exp_qsa_prepare_metadata,
)

from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_QSA_CACHE_GROUP,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    cache_field_layer_id,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.backends.paged.router import (
        CacheGroupRouter,
    )
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.qsa.indexer import QSAIndexer


def decode_query_lengths(
    ctx: ForwardContext, total_tokens: int, *, force_uniform: bool
) -> int | None:
    """Return the uniform decode query width, or None for a ragged extend.

    Args:
        ctx: The live model forward context.
        total_tokens: Query rows, after any draft step-0 narrowing.
        force_uniform: Whether a narrowed draft requires uniform rows even
            when its original context still names an extend forward.

    Returns:
        Query rows per request for uniform forwards, otherwise None.
    """
    if not ctx.bs or (
        not force_uniform
        and (ctx.forward_mode is None or not ctx.forward_mode.is_decode())
    ):
        return None
    if total_tokens % ctx.bs:
        raise RuntimeError("Qwen4-Exp QSA decode rows must be divisible by batch size")
    return total_tokens // ctx.bs


@dataclass(frozen=True)
class QSALayout:
    """Layer-invariant cache geometry for one QSA model forward."""

    metadata: object
    seq_lens: torch.Tensor
    logical_positions: torch.Tensor
    request_indices: torch.Tensor
    qsa_locs: torch.Tensor
    recent_locs: torch.Tensor
    complete_blocks: torch.Tensor
    qsa_page_table: torch.Tensor
    qsa_page_expansion: int
    recent_page_table: torch.Tensor
    recent_page_expansion: int
    full_page_table: torch.Tensor
    full_kernel_page_size: int
    reset_draft_tags: torch.Tensor | None


class _QSAVerifyStaging:
    """One capacity-sized target-verify staging set shared by every layer."""

    __slots__ = (
        "token_k",
        "position_values",
        "logical_positions",
        "recent_locs",
        "pool",
        "capacity",
    )

    def __init__(
        self,
        token_k: torch.Tensor,
        position_values: torch.Tensor,
        logical_positions: torch.Tensor,
        recent_locs: torch.Tensor,
        pool,
        capacity: int,
    ) -> None:
        self.token_k = token_k
        self.position_values = position_values
        self.logical_positions = logical_positions
        self.recent_locs = recent_locs
        self.pool = pool
        self.capacity = capacity


class QSARuntime:
    """Shared QSA indexing layout and target-verify staging for one router."""

    def __init__(self, config: AttnConfig, router: CacheGroupRouter) -> None:
        self.router = router
        self.dtype = config.dtype
        self.device = config.device
        self._indexers: tuple[QSAIndexer, ...] = ()
        self._slots: dict[int, int] = {}
        self._staging: dict[int, _QSAVerifyStaging] = {}
        self._verify_max_bs = int(config.max_bs)
        self._active_verify_width: int | None = None
        self._commit_tables: dict | None = None
        self._commit_tables_key: tuple | None = None

    @property
    def is_draft(self) -> bool:
        return self.router.is_draft

    @property
    def spec_num_tokens(self) -> int:
        return self.router.spec_num_tokens

    @property
    def sparse_topk(self):
        return self.router.sparse_topk

    @property
    def cache_pool(self):
        return self.router.cache_pool

    def _metadata(self, ctx: ForwardContext):
        leaf = self.router.leaves[FULL_ATTENTION]
        metadata = (
            leaf.forward_extend_metadata
            if ctx.forward_mode.is_extend_or_mixed()
            else leaf.forward_decode_metadata
        )
        if metadata is None:
            raise RuntimeError(
                f"QSA found no {ctx.forward_mode} metadata on the full-attention leaf"
            )
        return metadata

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

    def qsa_forward_layout(
        self,
        ctx: ForwardContext,
        total_tokens: int,
        *,
        compressed_token_page_size: int,
        recent_page_size: int,
        compress_ratio: int,
        reset_draft_tags: torch.Tensor | None,
    ) -> QSALayout:
        """Build or reuse the QSA row layout shared by every local QSA layer."""

        cached = self.sparse_topk.qsa_metadata
        if cached is not None:
            if not isinstance(cached, QSALayout):
                raise RuntimeError("invalid QSA per-forward metadata memo")
            if (
                cached.logical_positions.shape[0] != total_tokens
                or cached.seq_lens.shape[0] < ctx.bs
            ):
                raise RuntimeError("stale QSA per-forward metadata memo")
            if (
                reset_draft_tags is not None
                and cached.reset_draft_tags is not reset_draft_tags
            ):
                reset_draft_tags.fill_(torch.iinfo(torch.int64).min)
            return cached

        metadata = self._metadata(ctx)
        query_lengths = decode_query_lengths(
            ctx,
            total_tokens,
            force_uniform=False,
        )
        if query_lengths is None:
            query_lengths = self._query_lengths(metadata, total_tokens, ctx.bs)
        qsa = self.router.group_view(QWEN4_EXP_QSA_CACHE_GROUP, ctx.bs)
        recent = self.router.group_view(QWEN4_EXP_QSA_RECENT_CACHE_GROUP, ctx.bs)
        full = self.router.group_view(FULL_ATTENTION, ctx.bs)
        qsa_page_table, qsa_expansion = qsa.page_table, qsa.pages_per_block
        recent_page_table, recent_expansion = recent.page_table, recent.pages_per_block
        seq_lens = self._seq_lens(metadata)[: ctx.bs]
        logical, requests, qsa_locs, recent_locs, complete_blocks = (
            qwen4_exp_qsa_prepare_metadata(
                seq_lens,
                query_lengths,
                total_tokens,
                qsa_page_table,
                qsa_expansion,
                compressed_token_page_size,
                recent_page_table,
                recent_expansion,
                recent_page_size,
                compress_ratio,
                draft_logical_positions=reset_draft_tags,
            )
        )
        layout = QSALayout(
            metadata=metadata,
            seq_lens=seq_lens,
            logical_positions=logical,
            request_indices=requests,
            qsa_locs=qsa_locs,
            recent_locs=recent_locs,
            complete_blocks=complete_blocks,
            qsa_page_table=qsa_page_table,
            qsa_page_expansion=qsa_expansion,
            recent_page_table=recent_page_table,
            recent_page_expansion=recent_expansion,
            full_page_table=full.page_table,
            full_kernel_page_size=full.kernel_page_size,
            reset_draft_tags=reset_draft_tags,
        )
        self.sparse_topk.qsa_metadata = layout
        return layout

    # ------------------------------------------------------------------
    # Target-verify staging and batched commit
    # ------------------------------------------------------------------

    def bind_indexers(self, indexers: Iterable[QSAIndexer]) -> None:
        """Bind the model-owned local QSA indexers in ascending layer order."""

        bound = tuple(indexers)
        if self._indexers is bound:
            return
        if self._indexers:
            same = len(self._indexers) == len(bound) and all(
                previous is current
                for previous, current in zip(self._indexers, bound, strict=True)
            )
            if not same:
                raise RuntimeError("QSA backend cannot be rebound to another model")
        self._check_uniform_geometry(bound)
        self._indexers = bound
        self._slots = {indexer.layer_id: slot for slot, indexer in enumerate(bound)}
        for indexer in bound:
            indexer.qsa_runtime = self

    @staticmethod
    def _check_uniform_geometry(indexers: tuple[QSAIndexer, ...]) -> None:
        if not indexers:
            return
        layer_ids = [indexer.layer_id for indexer in indexers]
        if len(set(layer_ids)) != len(layer_ids):
            raise RuntimeError(
                f"QSA indexers must have distinct layer ids: {layer_ids}"
            )
        first = indexers[0]
        for indexer in indexers[1:]:
            for name in ("index_head_dim", "compress_ratio", "recent_page_size"):
                if getattr(indexer, name) != getattr(first, name):
                    raise RuntimeError(
                        f"QSA layer {indexer.layer_id} disagrees with layer "
                        f"{first.layer_id} on {name}: {getattr(indexer, name)} vs "
                        f"{getattr(first, name)}"
                    )

    def verify_staging_buffers(
        self,
        indexer: QSAIndexer,
        token_k: torch.Tensor,
        position_values: torch.Tensor,
        logical_positions: torch.Tensor,
        recent_locs: torch.Tensor,
        bs: int,
        pool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return graph-stable destinations for one layer's verify rows."""

        if bs <= 0 or token_k.shape[0] % bs:
            raise RuntimeError("QSA target-verify rows must be divisible by batch size")
        try:
            slot = self._slots[indexer.layer_id]
        except KeyError:
            raise RuntimeError(
                f"QSA layer {indexer.layer_id} staged verify rows without being "
                "bound to this backend"
            ) from None
        width = token_k.shape[0] // bs
        if self._active_verify_width is not None and self._active_verify_width != width:
            raise RuntimeError(
                "QSA target-verify width changed from "
                f"{self._active_verify_width} to {width}"
            )
        staging = self._staging.get(width)
        if staging is None:
            if self._staging:
                raise RuntimeError(
                    "QSA verify staging saw a second verify width "
                    f"{width} (existing {sorted(self._staging)})"
                )
            capacity = max(self._verify_max_bs, bs)
            staging = _QSAVerifyStaging(
                token_k.new_empty(
                    (len(self._indexers), capacity, width, 1, indexer.index_head_dim)
                ),
                position_values.new_empty((capacity, width, 3)),
                logical_positions.new_empty((capacity, width)),
                recent_locs.new_empty((capacity, width)),
                pool,
                capacity,
            )
            self._staging[width] = staging
        elif staging.pool is not pool:
            raise RuntimeError("QSA verify staging saw two KV pools in one forward")
        if bs > staging.capacity:
            raise RuntimeError(
                f"QSA verify staging holds {staging.capacity} requests but this "
                f"forward has {bs}; the buffer must never be resized"
            )
        self._active_verify_width = width
        return (
            staging.token_k[slot, :bs],
            staging.position_values[:bs],
            staging.logical_positions[:bs],
            staging.recent_locs[:bs],
        )

    def preallocate_verify_workspace(self, max_bs: int, draft_token_num: int) -> int:
        """Allocate target-verify staging before graph capture and return bytes."""

        del draft_token_num
        width = int(self.spec_num_tokens)
        if self.is_draft or width <= 1:
            return 0
        arena = getattr(self.cache_pool, "arena", None)
        plan = getattr(arena, "plan", None)
        if plan is None:
            return 0
        fields = self._raw_key_fields(plan)
        if not fields:
            return 0
        capacity = max(int(max_bs), self._verify_max_bs)
        index_dim = int(fields[0].shape[-1])
        staging = self._staging.get(width)
        if staging is None:
            staging = _QSAVerifyStaging(
                torch.empty(
                    (len(fields), capacity, width, 1, index_dim),
                    dtype=self.dtype,
                    device=self.device,
                ),
                torch.empty(
                    (capacity, width, 3), dtype=torch.int64, device=self.device
                ),
                torch.empty((capacity, width), dtype=torch.int64, device=self.device),
                torch.empty((capacity, width), dtype=torch.int32, device=self.device),
                self.cache_pool,
                capacity,
            )
            self._staging[width] = staging
        return sum(
            tensor.nbytes
            for tensor in (
                staging.token_k,
                staging.position_values,
                staging.logical_positions,
                staging.recent_locs,
            )
        )

    def _raw_key_fields(self, plan) -> list:
        owned_layers = self.cache_pool.field_layer_range
        return [
            field
            for field in plan.fields
            if field.group_id == QWEN4_EXP_QSA_RECENT_CACHE_GROUP
            and field.field_id.endswith(".qsa.raw_key")
            and cache_field_layer_id(field.field_id) in owned_layers
        ]

    def _commit_tables_get(self, pool) -> dict:
        raw_fields = []
        position_fields = []
        for indexer in self._indexers:
            raw, position_cache = indexer.verify_commit_fields(pool)
            raw_fields.append(raw)
            position_fields.append(position_cache)
        key = (
            id(pool),
            tuple(indexer.layer_id for indexer in self._indexers),
            tuple(field.data_ptr() for field in raw_fields),
            tuple(field.data_ptr() for field in position_fields),
        )
        if self._commit_tables is not None and self._commit_tables_key == key:
            return self._commit_tables

        first_raw, first_position = raw_fields[0], position_fields[0]
        first_indexer = self._indexers[0]
        for indexer, raw, position_cache in zip(
            self._indexers, raw_fields, position_fields, strict=True
        ):
            if (
                raw.shape != first_raw.shape
                or raw.dtype != first_raw.dtype
                or raw.stride() != first_raw.stride()
            ):
                raise RuntimeError(
                    f"QSA layer {indexer.layer_id} raw-key field disagrees with "
                    f"layer {first_indexer.layer_id}"
                )
            if (
                position_cache.shape != first_position.shape
                or position_cache.dtype != first_position.dtype
                or position_cache.stride() != first_position.stride()
            ):
                raise RuntimeError(
                    f"QSA layer {indexer.layer_id} RoPE position field must match "
                    f"layer {first_indexer.layer_id}"
                )
            if (
                raw.shape[1] != indexer.compress_ratio
                or raw.shape[-1] != indexer.index_head_dim
            ):
                raise RuntimeError(
                    f"QSA layer {indexer.layer_id} raw-key field geometry "
                    "disagrees with its indexer"
                )
        tables = {
            "raw_addresses": torch.tensor(
                [field.data_ptr() for field in raw_fields],
                dtype=torch.uint64,
                device=first_raw.device,
            ),
            "position_addresses": torch.tensor(
                [field.data_ptr() for field in position_fields],
                dtype=torch.uint64,
                device=first_raw.device,
            ),
            "raw_cache": first_raw,
            "position_cache": first_position,
        }
        self._commit_tables = tables
        self._commit_tables_key = key
        return tables

    def commit_after_mtp_verify(
        self,
        accepted_lengths: torch.Tensor,
        *,
        num_extends: int,
    ) -> None:
        """Commit accepted target-verify candidates for all QSA layers once."""

        if num_extends < 0 or num_extends > accepted_lengths.shape[0]:
            raise ValueError(
                "QSA verify commit received an invalid extend prefix: "
                f"{num_extends} for {accepted_lengths.shape[0]} requests"
            )
        verify_lengths = accepted_lengths[num_extends:]
        bs = verify_lengths.shape[0]
        if bs == 0 or not self._indexers:
            return
        width = self._active_verify_width
        staging = None if width is None else self._staging.get(width)
        if staging is None:
            return
        tables = self._commit_tables_get(staging.pool)
        indexer = self._indexers[0]
        from tokenspeed_kernel.ops.attention.triton.qwen4_exp_qsa import (
            qwen4_exp_qsa_commit_verify_layers,
        )

        qwen4_exp_qsa_commit_verify_layers(
            tables["raw_addresses"],
            tables["position_addresses"],
            staging.token_k,
            staging.logical_positions[:bs].reshape(-1),
            staging.recent_locs[:bs].reshape(-1),
            staging.position_values[:bs].reshape(-1, 3),
            verify_lengths,
            tables["raw_cache"],
            tables["position_cache"],
            indexer.recent_page_size,
            indexer.compress_ratio,
            verify_width=width,
        )


def require_qsa_runtime(attn_backend: AttentionBackend) -> QSARuntime:
    """Return the registered router runtime used by a target or draft model.

    Args:
        attn_backend: The model's router or outer hybrid backend.

    Returns:
        The QSA runtime created by the backend registry at construction.
    """
    router = getattr(attn_backend, "full_attn_backend", attn_backend)
    runtime = getattr(router, "runtime", None)
    if not isinstance(runtime, QSARuntime):
        raise RuntimeError("Qwen4-Exp QSA indexers require a registered QSA runtime")
    return runtime


def bind_qsa_indexers(
    attn_backend: AttentionBackend,
    indexers: Iterable[QSAIndexer],
) -> QSARuntime | None:
    """Bind local target indexers to their pre-existing backend runtime.

    Args:
        attn_backend: The model's router or outer hybrid backend.
        indexers: Model-owned local indexers in ascending layer order.

    Returns:
        The owning runtime, or None for drafts, which do not commit target
        acceptance. Construction, budgeting and hook registration precede
        model binding and never depend on a model forward running.
    """
    runtime = require_qsa_runtime(attn_backend)
    if runtime.is_draft:
        return None
    runtime.bind_indexers(indexers)
    return runtime


__all__ = ["QSARuntime", "QSALayout", "bind_qsa_indexers", "require_qsa_runtime"]
