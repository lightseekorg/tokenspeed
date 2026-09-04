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

"""Registered Qwen4-Exp QSA attention backend.

The model-owned :class:`QSAIndexer` keeps the projection weights and produces
physical top-k slots. This backend owns the cache-group router, sparse dispatch,
graph-stable target-verify staging, and the batched post-verification commit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention import qsa_sparse_attention
from tokenspeed_kernel.ops.attention.triton.qwen4_exp_qsa import (
    qwen4_exp_qsa_prepare_metadata,
)
from tokenspeed_kernel.ops.kvcache.triton import fused_fp8_set_kv_buffer

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    current_forward_ctx,
    current_valid_rows,
    slice_to_real_tokens,
)
from tokenspeed.runtime.layers.attention.backends.paged.mha import MHAAttnBackend
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_QSA_CACHE_GROUP,
    QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    cache_field_layer_id,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import FULL_ATTENTION
from tokenspeed.runtime.layers.attention.registry import register_backend

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.configs.base import (
        AttnConfig,
        SoftmaxAttnConfig,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.attention.qsa.indexer import QSAIndexer
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


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


class QSAAttnBackend(CacheGroupRouter):
    """QSA sparse dispatch over MHA leaves and the shared group router."""

    def __init__(self, config: AttnConfig, spec: SoftmaxAttnConfig) -> None:
        def leaf_factory(group_id: str, block_granularity: int):
            del group_id
            kernel_page_size = MHAAttnBackend.resolve_kernel_page_size(
                config, block_granularity
            )
            leaf_spec = dataclasses.replace(spec, backend_name="mha")
            return MHAAttnBackend(
                config,
                leaf_spec,
                kernel_page_size=kernel_page_size,
            )

        super().__init__(
            leaf_factory,
            is_draft=bool(config.is_draft),
            spec_num_tokens=config.speculative_num_draft_tokens or 1,
            device=config.device,
        )
        self.dtype = config.dtype
        self._data_type = config.kv_cache_dtype
        self._indexers: tuple[QSAIndexer, ...] = ()
        self._slots: dict[int, int] = {}
        self._staging: dict[int, _QSAVerifyStaging] = {}
        self._verify_max_bs = int(config.max_bs)
        self._active_verify_width: int | None = None
        self._commit_tables: dict | None = None
        self._commit_tables_key: tuple | None = None

    @property
    def data_type(self):
        return self._data_type

    # ------------------------------------------------------------------
    # QSA forward geometry
    # ------------------------------------------------------------------

    def _metadata(self, ctx: ForwardContext):
        leaf = self.leaves[FULL_ATTENTION]
        candidates = []
        if ctx.forward_mode.is_extend_or_mixed():
            candidates.extend(("forward_extend_metadata", "forward_prefill_metadata"))
        elif self.spec_num_tokens > 1 and not self.is_draft:
            candidates.append("forward_prefill_metadata")
        candidates.append("forward_decode_metadata")
        for name in candidates:
            metadata = getattr(leaf, name, None)
            if metadata is not None:
                return metadata
        raise RuntimeError(
            f"QSA found no {ctx.forward_mode} metadata on the full-attention leaf"
        )

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

    @staticmethod
    def _decode_query_lengths(
        ctx: ForwardContext,
        total_tokens: int,
        *,
        force_uniform: bool,
    ) -> int | None:
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
    def _page_table_expansion(kernel_page_size: int, block_granularity: int) -> int:
        kernel_page_size = int(kernel_page_size)
        if block_granularity == kernel_page_size:
            return 1
        if block_granularity % kernel_page_size:
            raise ValueError(
                "Qwen4-Exp QSA block granularity must be divisible by the "
                f"group's kernel page size ({block_granularity} vs "
                f"{kernel_page_size})"
            )
        return block_granularity // kernel_page_size

    def _group_geometry(
        self,
        group_id: str,
        block_granularity: int,
        bs: int,
    ) -> tuple[torch.Tensor, int]:
        expansion = self._page_table_expansion(
            self.stacks.group_kernel_page_size(group_id), block_granularity
        )
        return self.stacks.table(group_id, bs), expansion

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
        query_lengths = self._decode_query_lengths(
            ctx,
            total_tokens,
            force_uniform=False,
        )
        if query_lengths is None:
            query_lengths = self._query_lengths(metadata, total_tokens, ctx.bs)
        qsa_page_table, qsa_expansion = self._group_geometry(
            QWEN4_EXP_QSA_CACHE_GROUP,
            compressed_token_page_size,
            ctx.bs,
        )
        recent_page_table, recent_expansion = self._group_geometry(
            QWEN4_EXP_QSA_RECENT_CACHE_GROUP,
            recent_page_size,
            ctx.bs,
        )
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
            indexer.qsa_coordinator = self

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

    # ------------------------------------------------------------------
    # Sparse dispatch
    # ------------------------------------------------------------------

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: CachePool,
        topk_indices: torch.Tensor,
        ctx: ForwardContext | None,
    ) -> torch.Tensor:
        num_real = current_valid_rows()
        if num_real is not None:
            q, k, v, out_cache_loc, topk_indices = slice_to_real_tokens(
                num_real, q, k, v, out_cache_loc, topk_indices
            )
        ctx = ctx or current_forward_ctx()
        if ctx is None:
            raise RuntimeError("QSA sparse attention requires a forward context")
        full_locs = out_cache_loc[: k.shape[0]]
        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
        if (
            k_cache.dtype == torch.float8_e4m3fn
            and v_cache.dtype == torch.float8_e4m3fn
            and k.dtype != k_cache.dtype
            and v.dtype != v_cache.dtype
        ):
            fused_fp8_set_kv_buffer(
                k=k,
                v=v,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_loc=full_locs,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
                page_size=self.stacks.group_kernel_page_size(FULL_ATTENTION),
            )
        else:
            token_to_kv_pool.set_kv_buffer(
                layer,
                full_locs,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )
        max_seqlen_q = self._decode_query_lengths(
            ctx,
            q.shape[0],
            force_uniform=ctx.draft_narrowing is not None,
        )
        output = qsa_sparse_attention(
            q,
            k_cache,
            v_cache,
            topk_indices,
            scale=layer.scaling,
            max_seqlen_q=max_seqlen_q if max_seqlen_q is not None else 1,
            metadata_capacity_rows=max(
                q.shape[0], self.stacks.max_bs * self.stacks.max_tokens_per_req
            ),
            k_scale=(
                (1.0 if layer.k_scale is None else layer.k_scale)
                if k_cache.dtype == torch.float8_e4m3fn
                else None
            ),
            v_scale=(
                (1.0 if layer.v_scale is None else layer.v_scale)
                if v_cache.dtype == torch.float8_e4m3fn
                else None
            ),
            override=None,
            solution=None,
        )
        return output.reshape(q.shape[0], -1)

    @break_point
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        token_to_kv_pool: CachePool,
        forward_mode: ForwardMode,
        bs: int,
        save_kv_cache: bool = True,
        record_kv_cache: bool | None = None,
        *,
        topk_indices: torch.Tensor | None,
        ctx: ForwardContext | None,
        **kwargs,
    ) -> torch.Tensor:
        if topk_indices is None:
            return super().forward(
                q,
                k,
                v,
                layer,
                token_to_kv_pool,
                forward_mode,
                bs,
                save_kv_cache,
                record_kv_cache,
                **kwargs,
            )
        out_cache_loc = self.write_locations(layer, forward_mode)
        with self.record_pd_cache_step(forward_mode, save_kv_cache, record_kv_cache):
            return self._sparse_attention(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                topk_indices,
                ctx,
            )

    def forward_decode(
        self,
        q,
        k,
        v,
        layer,
        token_to_kv_pool,
        bs,
        save_kv_cache=True,
        *,
        topk_indices,
        ctx,
        **kwargs,
    ):
        if topk_indices is None:
            return super().forward_decode(
                q,
                k,
                v,
                layer,
                token_to_kv_pool,
                bs,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

        return self._sparse_attention(
            q,
            k,
            v,
            layer,
            self.write_locations(layer, ForwardMode.DECODE),
            token_to_kv_pool,
            topk_indices,
            ctx,
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        token_to_kv_pool,
        bs,
        save_kv_cache=True,
        *,
        topk_indices,
        ctx,
        forward_mode,
        **kwargs,
    ):
        if topk_indices is None:
            return super().forward_extend(
                q,
                k,
                v,
                layer,
                token_to_kv_pool,
                bs,
                save_kv_cache=save_kv_cache,
                forward_mode=forward_mode,
                **kwargs,
            )
        return self._sparse_attention(
            q,
            k,
            v,
            layer,
            self.write_locations(layer, forward_mode),
            token_to_kv_pool,
            topk_indices,
            ctx,
        )


register_backend("qsa", {AttentionArch.MHA}, QSAAttnBackend)


def bind_qsa_indexers(
    attn_backend,
    indexers: Iterable[QSAIndexer],
) -> QSAAttnBackend | None:
    """Bind local QSA indexers to the registered full-attention backend."""

    full_backend = getattr(attn_backend, "full_attn_backend", attn_backend)
    if getattr(full_backend, "is_draft", False):
        return None
    if not isinstance(full_backend, QSAAttnBackend):
        raise RuntimeError(
            "Qwen4-Exp QSA indexers require the qsa attention backend, got "
            f"{type(full_backend).__name__}"
        )
    full_backend.bind_indexers(indexers)
    attn_backend.register_speculative_state_backend(full_backend)
    return full_backend


__all__ = ["QSAAttnBackend", "QSALayout", "bind_qsa_indexers"]
