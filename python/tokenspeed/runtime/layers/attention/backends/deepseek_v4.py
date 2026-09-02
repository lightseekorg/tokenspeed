# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

from collections.abc import Mapping

import torch
from tokenspeed_kernel import (
    dsv4_decode,
    dsv4_plan,
    dsv4_prefill,
    dsv4_reset_attention_state,
)
from tokenspeed_kernel.ops.attention.triton.dsv4 import (
    dsv4_build_dense_prefill_local_compressed_indices,
    dsv4_combine_dense_swa_indices,
    dsv4_combine_topk_swa_indices,
    dsv4_compute_global_topk_indices_and_lens,
    dsv4_decode_swa_indices_and_lens,
    dsv4_dequantize_and_gather_k_cache,
    dsv4_indexer_decode_metadata_compute,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.deepseek_v4.draft_rounds import (
    DeepseekV4DraftRounds,
)
from tokenspeed.runtime.layers.attention.deepseek_v4.graph_buffers import (
    DeepseekV4GraphBuffers,
)
from tokenspeed.runtime.layers.attention.deepseek_v4.metadata import (
    DeepseekV4ForwardMetadata,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_geometry import (
    DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT,
    V4_KERNEL_BLOCK_ROWS,
    first_v4_compressed_kv_group_id,
    v4_compressed_kv_group_id,
)
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    DEEPSEEK_V4_PAGE_SIZE,
)
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_deepseek_v4 import (
    DeepseekV4CacheMetadata,
    _split_block_tables_into_v4_metadata,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
)
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.nvtx import nvtx_range

DEEPSEEK_V4_DEFAULT_PREFILL_CHUNK_SIZE = 4


def _compressed_block_table_base_offsets(
    metadata: DeepseekV4ForwardMetadata,
    compress_ratio: int,
) -> torch.Tensor | None:
    return metadata.cache.block_table_base_offsets.get(
        v4_compressed_kv_group_id(compress_ratio)
    )


def _decode_positions_from_metadata(
    metadata: DeepseekV4ForwardMetadata,
    num_tokens: int,
) -> torch.Tensor:
    token_to_req = metadata.token_to_req_indices[:num_tokens].to(torch.int64)
    query_starts = metadata.query_start_loc[token_to_req].to(torch.int64)
    query_lens = metadata.query_lens[token_to_req].to(torch.int64)
    seq_lens = metadata.seq_lens[token_to_req].to(torch.int64)
    token_offsets = torch.arange(
        num_tokens,
        dtype=torch.int64,
        device=metadata.seq_lens.device,
    )
    return seq_lens - query_lens + token_offsets - query_starts


def _refresh_decode_indexer_plan_cache(
    metadata: DeepseekV4ForwardMetadata,
    *,
    max_context_len: int,
) -> None:
    """Pre-build decode-indexer plan tensors before per-layer parallel work.

    This keeps per-layer indexer calls read-only with respect to cached plan
    buffers while compressor work may run on an auxiliary stream.
    """
    indexer_metadata = metadata.indexer
    cache = indexer_metadata.decode_plan_cache
    if not cache:
        return
    refreshed_keys = indexer_metadata.decode_plan_refreshed_keys
    refreshed_keys.clear()
    for (
        compress_ratio,
        cache_block_size,
        num_tokens,
    ), plan in list(cache.items()):
        if num_tokens <= 0:
            plan.context_lens.zero_()
            plan.page_table.zero_()
            plan.max_context_len = 0
            refreshed_keys.add((compress_ratio, cache_block_size, num_tokens))
            continue
        positions = _decode_positions_from_metadata(metadata, num_tokens)
        token_to_req_indices = metadata.token_to_req_indices[:num_tokens]
        page_table = metadata.cache.compressed_page_table(
            compress_ratio,
            cache_block_size,
        )
        block_table_base_offsets = (
            _compressed_block_table_base_offsets(metadata, compress_ratio)
            if page_table is not metadata.cache.page_table
            else None
        )
        rows = int(page_table.shape[0]) if page_table.ndim >= 1 else 0
        cols = int(page_table.shape[1]) if page_table.ndim >= 2 else 0
        if rows <= 0 or cols <= 0:
            plan.context_lens.zero_()
            plan.page_table.zero_()
            plan.max_context_len = 0
            refreshed_keys.add((compress_ratio, cache_block_size, num_tokens))
            continue
        max_blocks = int(plan.page_table.shape[1])
        if max_context_len > 0:
            derived_max_len = max(
                1,
                (max_context_len + compress_ratio - 1) // compress_ratio,
            )
        else:
            derived_max_len = max(
                1,
                (page_table.shape[1] * cache_block_size + compress_ratio - 1)
                // compress_ratio,
            )
        if plan.max_context_len != derived_max_len:
            plan.max_context_len = derived_max_len
        dsv4_indexer_decode_metadata_compute(
            positions=positions,
            token_to_req_indices=token_to_req_indices,
            block_table=page_table,
            cache_block_size=cache_block_size,
            compress_ratio=compress_ratio,
            max_blocks=max_blocks,
            out_context_lens=plan.context_lens,
            out_block_tables=plan.page_table,
            block_table_base_offsets=block_table_base_offsets,
        )
        if metadata.is_valid_token is not None:
            valid = metadata.is_valid_token[:num_tokens].to(
                device=plan.context_lens.device,
                dtype=torch.bool,
            )
            with torch.inference_mode():
                plan.context_lens.masked_fill_(~valid.view(num_tokens, 1), 0)
                plan.page_table.masked_fill_(
                    ~valid.to(device=plan.page_table.device).view(num_tokens, 1),
                    0,
                )
        refreshed_keys.add((compress_ratio, cache_block_size, num_tokens))


def _refresh_decode_indexer_schedule_metadata(
    metadata: DeepseekV4ForwardMetadata,
) -> None:
    indexer_metadata = metadata.indexer
    if not indexer_metadata.decode_schedule_metadata_cache:
        return
    for (
        compress_ratio,
        cache_block_size,
        num_tokens,
    ), schedule_metadata in list(
        indexer_metadata.decode_schedule_metadata_cache.items()
    ):
        if num_tokens <= 0:
            continue
        key = (compress_ratio, cache_block_size, num_tokens)
        decode_plan = indexer_metadata.decode_plan_cache.get(key)
        context_lens = getattr(decode_plan, "context_lens", None)
        if (
            context_lens is not None
            and context_lens.shape == (num_tokens, 1)
            and context_lens.dtype == torch.int32
        ):
            context_lens = context_lens.contiguous()
        else:
            positions = _decode_positions_from_metadata(metadata, num_tokens)
            compressed_lens = torch.div(
                positions.to(torch.int32) + 1,
                compress_ratio,
                rounding_mode="floor",
            ).clamp_min(0)
            if metadata.is_valid_token is not None:
                valid = metadata.is_valid_token[:num_tokens].to(
                    device=compressed_lens.device,
                    dtype=torch.bool,
                )
                compressed_lens = torch.where(
                    valid,
                    compressed_lens,
                    torch.zeros_like(compressed_lens),
                )
            context_lens = compressed_lens.view(num_tokens, 1).contiguous()
        refreshed = dsv4_plan(
            seq_lens_2d=context_lens,
            page_size=cache_block_size,
            out=schedule_metadata,
        )
        if refreshed is not None and refreshed is not schedule_metadata:
            indexer_metadata.decode_schedule_metadata_cache[key] = refreshed


class DeepseekV4AttentionBackend(AttentionBackend):
    """Metadata owner for the model-local DeepSeek V4 attention path."""

    cache_consumer_families = frozenset({"history", "state"})
    # The refresh replaces metadata.cache wholesale each step (the bespoke
    # multi-group slot mapping travels on the live cache_metadata object);
    # the captured kernels read only the persistent buffers the refresh
    # fills FROM it, never this object through Python. Exempt it from the
    # capture-time pointer snapshot (graph_ptr_guard).
    graph_unstable_metadata_fields = frozenset({"cache"})

    def __init__(self, config: AttnConfig, spec: MLAConfig) -> None:
        super().__init__(config, spec)
        self.kernel_page_size = (
            config.kernel_page_size
            if config.kernel_page_size is not None
            else DEEPSEEK_V4_PAGE_SIZE
        )
        # V4 has no expansion path: its cache spec pre-folds group geometry
        # to the kernel page, so scheduler tables arrive kernel-ready for any
        # prefix granularity that is a positive multiple of the kernel page.
        if (
            config.prefix_granularity <= 0
            or config.prefix_granularity % self.kernel_page_size
        ):
            raise ValueError(
                "DeepSeek V4 kernels require prefix_granularity to be a "
                f"positive multiple of kernel_page_size "
                f"({self.kernel_page_size}), got {config.prefix_granularity}"
            )
        self.swa_storage_rows = int(
            spec.sliding_window_tokens
            if spec.sliding_window_tokens is not None
            else V4_KERNEL_BLOCK_ROWS * 2
        )
        self.context_len = config.context_len
        prefill_chunk_size = getattr(config, "deepseek_v4_prefill_chunk_size", None)
        if prefill_chunk_size is None:
            prefill_chunk_size = global_server_args_dict.get(
                "deepseek_v4_prefill_chunk_size",
                DEEPSEEK_V4_DEFAULT_PREFILL_CHUNK_SIZE,
            )
        self.prefill_chunk_size = max(1, int(prefill_chunk_size))
        self.max_num_pages = max(
            1,
            (self.context_len + self.kernel_page_size - 1) // self.kernel_page_size,
        )
        self.forward_metadata: DeepseekV4ForwardMetadata | None = None
        self.forward_prefill_metadata: DeepseekV4ForwardMetadata | None = None
        self.forward_decode_metadata: DeepseekV4ForwardMetadata | None = None
        # The persistent decode buffers + per-shape views; allocated by
        # init_cuda_graph_state (unconditionally at wrapper construction).
        self.graph: DeepseekV4GraphBuffers | None = None
        self._expected_cache_group_ids: tuple[str, ...] | None = None
        self._cache_group_raw_tokens_per_page: dict[str, int] = {}
        self._cache_group_max_page_ids: dict[str, int] = {}
        self._prefill_workspace_buffer: torch.Tensor | None = None
        self._prefill_workspace_rows = 0
        self._prefill_workspace_head_dim = 0
        self._prefill_dense_compressed_indices_buffer: torch.Tensor | None = None
        self._swa_window_size = 0
        self._swa_block_size = 0
        self.speculative_num_steps = getattr(config, "speculative_num_steps", 0) or 0
        self.speculative_num_draft_tokens = (
            getattr(config, "speculative_num_draft_tokens", 0) or 0
        )
        # Plain-step draft metadata views; built by init_cuda_graph_state
        # for draft instances only (see DeepseekV4DraftRounds).
        self.draft_rounds: DeepseekV4DraftRounds | None = None

    def record_layer_cache_ready(
        self,
        hidden_states: torch.Tensor,
        forward_mode: ForwardMode,
    ) -> torch.Tensor:
        """Publish readiness after V4's model-local cache writers complete."""
        if (
            self.step_counter is not None
            and not forward_mode.is_decode()
            and not forward_mode.is_idle()
        ):
            self.step_counter.record_cache()
        return hidden_states

    def _configure_cache_group_contract(
        self,
        cache_group_specs=(),
        cache_group_page_counts=None,
    ) -> tuple[tuple[object, ...], dict[str, int]]:
        specs = tuple(cache_group_specs or ())
        group_ids = tuple(getattr(spec, "group_id", None) for spec in specs)
        if any(not isinstance(group_id, str) or not group_id for group_id in group_ids):
            raise RuntimeError(
                "DeepSeek V4 cache group specs must use nonempty string IDs"
            )
        if len(group_ids) != len(set(group_ids)):
            raise RuntimeError("DeepSeek V4 cache group specs contain duplicate IDs")
        page_counts = dict(cache_group_page_counts or {})
        if not group_ids:
            if page_counts:
                raise RuntimeError(
                    "DeepSeek V4 cache page counts require matching group specs"
                )
            raw_tokens_per_page: dict[str, int] = {}
            max_page_ids: dict[str, int] = {}
        else:
            if set(page_counts) != set(group_ids):
                raise RuntimeError(
                    "DeepSeek V4 cache page counts disagree with group specs: "
                    f"missing={sorted(set(group_ids) - set(page_counts))} "
                    f"extra={sorted(set(page_counts) - set(group_ids))}"
                )
            invalid_counts = {
                group_id: page_counts[group_id]
                for group_id in group_ids
                if not isinstance(page_counts[group_id], int)
                or isinstance(page_counts[group_id], bool)
                or page_counts[group_id] <= 1
            }
            if invalid_counts:
                raise RuntimeError(
                    "DeepSeek V4 cache groups must reserve page 0 and at least one "
                    f"live page: {invalid_counts!r}"
                )
            raw_tokens_per_page = {}
            for spec, group_id in zip(specs, group_ids, strict=True):
                raw_tokens = int(spec.rows_per_page) * int(spec.entry_stride_tokens)
                if raw_tokens <= 0:
                    raise RuntimeError(
                        "DeepSeek V4 cache group has invalid page geometry: "
                        f"group={group_id!r} rows_per_page={spec.rows_per_page} "
                        f"entry_stride_tokens={spec.entry_stride_tokens}"
                    )
                raw_tokens_per_page[group_id] = raw_tokens
            max_page_ids = {
                group_id: int(page_counts[group_id]) - 1 for group_id in group_ids
            }

        if self._expected_cache_group_ids is not None:
            if (
                self._expected_cache_group_ids != group_ids
                or self._cache_group_raw_tokens_per_page != raw_tokens_per_page
                or self._cache_group_max_page_ids != max_page_ids
            ):
                raise RuntimeError(
                    "DeepSeek V4 cache group contract changed after initialization"
                )
        self._expected_cache_group_ids = group_ids
        self._cache_group_raw_tokens_per_page = raw_tokens_per_page
        self._cache_group_max_page_ids = max_page_ids
        return specs, page_counts

    def configure_runtime(self, **kwargs) -> None:
        self._configure_cache_group_contract(
            kwargs.pop("cache_group_specs", ()),
            kwargs.pop("cache_group_page_counts", None),
        )

    def _prepare_cache_group_tables(
        self,
        block_tables: Mapping[object, object],
        *,
        bs: int,
        actual_bs: int,
        seq_lens: torch.Tensor,
        device: torch.device,
        phase: str,
        output_buffers: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if actual_bs < 0 or actual_bs > bs:
            raise RuntimeError(
                f"DeepSeek V4 {phase} actual_bs={actual_bs} must be within 0..{bs}"
            )
        if seq_lens.ndim != 1 or int(seq_lens.shape[0]) < actual_bs:
            raise RuntimeError(
                f"DeepSeek V4 {phase} seq_lens has shape {tuple(seq_lens.shape)}, "
                f"expected at least {actual_bs} entries"
            )
        if not isinstance(block_tables, Mapping):
            raise RuntimeError(
                f"DeepSeek V4 {phase} cache tables must be a mapping, got "
                f"{type(block_tables).__name__}"
            )
        expected = self._expected_cache_group_ids
        if expected is None:
            raise RuntimeError(
                f"DeepSeek V4 cache group specs were not initialized before {phase}"
            )
        # Consume this backend's declared groups out of the delivered dict,
        # in contract order; extras ride to their own consumers (the runner
        # hands every node the same complete mapping). A missing declared
        # group is a delivery bug.
        missing = [gid for gid in expected if gid not in block_tables]
        if missing:
            raise RuntimeError(
                f"DeepSeek V4 {phase} cache group mismatch: missing={missing} "
                f"(delivered: {sorted(map(str, block_tables))})"
            )
        items = [(gid, block_tables[gid]) for gid in expected]

        expected_device = torch.device(device)
        if expected_device.type == "cuda" and expected_device.index is None:
            expected_device = torch.device("cuda", torch.cuda.current_device())
        validated: dict[str, torch.Tensor] = {}
        for group_id, value in items:
            if not isinstance(value, torch.Tensor):
                raise RuntimeError(
                    f"DeepSeek V4 {phase} table {group_id!r} must be torch.Tensor"
                )
            table = value
            if table.dtype != torch.int32:
                raise RuntimeError(
                    f"DeepSeek V4 {phase} table {group_id!r} must use torch.int32"
                )
            if table.device != expected_device:
                raise RuntimeError(
                    f"DeepSeek V4 {phase} table {group_id!r} is on "
                    f"{table.device}, expected {expected_device}"
                )
            if table.ndim != 2:
                raise RuntimeError(
                    f"DeepSeek V4 {phase} table {group_id!r} must be rank 2"
                )
            if int(table.shape[0]) < actual_bs:
                raise RuntimeError(
                    f"DeepSeek V4 {phase} table {group_id!r} has "
                    f"{int(table.shape[0])} rows, expected at least "
                    f"actual_bs={actual_bs}"
                )
            if int(table.shape[1]) <= 0:
                raise RuntimeError(
                    f"DeepSeek V4 {phase} table {group_id!r} has zero width"
                )
            validated[group_id] = table

        self._assert_active_cache_pages(
            validated,
            seq_lens=seq_lens,
            actual_bs=actual_bs,
            phase=phase,
        )
        materialized: dict[str, torch.Tensor] = {}
        for group_id, table in validated.items():
            if output_buffers is None:
                materialized[group_id] = table
                continue
            out = output_buffers[group_id]
            # The graph replays bs (padded) rows; the scheduler delivers rows
            # for the live requests only. Copy the live rows and null the
            # padding tail so padded replay rows resolve to no page.
            rows = min(int(table.shape[0]), bs)
            columns = int(table.shape[1])
            if out.ndim != 2 or out.shape[0] < bs or out.shape[1] < columns:
                raise ValueError(
                    f"output table for {group_id!r} cannot hold "
                    f"{bs}x{columns} rows"
                )
            if out.dtype != table.dtype or out.device != table.device:
                raise ValueError(
                    f"output table for {group_id!r} must match input dtype and device"
                )
            out[:bs].fill_(-1)
            out[:rows, :columns].copy_(table[:rows])
            materialized[group_id] = out[:bs]
        return materialized

    def _assert_active_cache_pages(
        self,
        tables: Mapping[str, torch.Tensor],
        *,
        seq_lens: torch.Tensor,
        actual_bs: int,
        phase: str,
    ) -> None:
        """Check only each live row's active page on GPU hot paths."""
        if actual_bs == 0:
            return
        live_seq_lens = seq_lens[:actual_bs].to(dtype=torch.int64)
        seq_lens_valid = live_seq_lens.ge(0).all()
        if live_seq_lens.device.type == "cpu":
            if not bool(seq_lens_valid.item()):
                raise RuntimeError(
                    f"DeepSeek V4 {phase} sequence lengths must be nonnegative"
                )
        else:
            torch._assert_async(
                seq_lens_valid,
                f"DeepSeek V4 {phase} sequence lengths must be nonnegative",
            )
        for group_id, table in tables.items():
            raw_tokens_per_page = self._cache_group_raw_tokens_per_page.get(group_id)
            max_page_id = self._cache_group_max_page_ids.get(group_id)
            if raw_tokens_per_page is None or max_page_id is None:
                raise RuntimeError(
                    f"DeepSeek V4 cache group contract is incomplete for {group_id!r}"
                )
            live = table[:actual_bs]
            required_page = torch.div(
                live_seq_lens.clamp_min(1) - 1,
                raw_tokens_per_page,
                rounding_mode="floor",
            )
            width = int(table.shape[1])
            in_bounds = required_page < width
            safe_page = required_page.clamp(min=0, max=width - 1)
            required_entries = live.gather(1, safe_page.unsqueeze(1)).squeeze(1)
            has_tokens = live_seq_lens > 0
            active_pages_valid = (
                ~has_tokens
                | (
                    in_bounds
                    & required_entries.gt(0)
                    & required_entries.le(max_page_id)
                )
            ).all()
            if table.device.type == "cpu":
                page_ids_valid = ((live >= -1) & (live <= max_page_id)).all()
                if not bool(page_ids_valid.item()):
                    raise RuntimeError(
                        f"DeepSeek V4 {phase} table {group_id!r} contains a page "
                        f"ID outside -1..{max_page_id}"
                    )
                if not bool(active_pages_valid.item()):
                    raise RuntimeError(
                        f"DeepSeek V4 {phase} table {group_id!r} is missing a "
                        "real page for an active sequence"
                    )
            else:
                torch._assert_async(
                    active_pages_valid,
                    f"DeepSeek V4 {phase} table is missing a real active page "
                    "or its page ID exceeds group capacity",
                )

    def _cuda_graph_group_table_width(
        self,
        spec: CacheGroupSpec,
        *,
        max_tokens_per_req: int,
        overlap_schedule_depth: int,
    ) -> int:
        page_size = spec.block_granularity
        if page_size <= 0:
            raise ValueError(
                f"DeepSeek V4 cache group {spec.group_id!r} has invalid page size"
            )
        # Scheduler tables keep their absolute cache-block index and punch
        # expired sliding pages into holes. They are sparse, not compact, so a
        # graph buffer must cover the complete request extent for every group.
        reservation_tokens = (overlap_schedule_depth + 1) * max_tokens_per_req
        extent = self.context_len + reservation_tokens
        return (extent + page_size - 1) // page_size

    def _get_prefill_workspace(
        self,
        *,
        num_reqs: int,
        workspace_width: int,
        head_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        workspace_reqs = max(1, num_reqs)
        rows = workspace_reqs * workspace_width
        needs_alloc = (
            self._prefill_workspace_buffer is None
            or self._prefill_workspace_buffer.device != device
            or self._prefill_workspace_head_dim != head_dim
            or self._prefill_workspace_rows < rows
        )
        if needs_alloc:
            self._prefill_workspace_buffer = torch.empty(
                (rows, head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            self._prefill_workspace_rows = rows
            self._prefill_workspace_head_dim = head_dim
        assert self._prefill_workspace_buffer is not None
        return self._prefill_workspace_buffer[:rows].view(
            workspace_reqs,
            workspace_width,
            head_dim,
        )

    # ------------------------------------------------------------------
    # Slot publication — the ONLY writers of the three metadata slots.
    #
    # | phase end                        | prefill slot   | decode slot | forward_metadata |
    # |----------------------------------|----------------|-------------|------------------|
    # | extend/mixed init                | new dynamic    | (unchanged) | = prefill        |
    # | idle init / decode refresh /     | (unchanged)    | views(bs,N) | = decode         |
    # |   target verify refresh / capture|                |             |                  |
    # | packed-draft refresh == capture  | packed views   | step views  | = decode         |
    # | advance (draft steps 1+)         | (unchanged)    | step views  | = decode         |
    #
    # Capture and replay refresh call the SAME publisher, so "replay must
    # reproduce capture's end state" is one line of code, not a cross-method
    # contract (graph_ptr_guard verifies the slots' object graph against the
    # capture-end snapshot). Forwards are read-only: resolution results
    # travel as parameters, never through a slot write.
    # ------------------------------------------------------------------

    def _publish_prefill(self, metadata: DeepseekV4ForwardMetadata) -> None:
        self.forward_prefill_metadata = metadata
        self.forward_metadata = metadata

    def _publish_decode(self, metadata: DeepseekV4ForwardMetadata) -> None:
        self.forward_decode_metadata = metadata
        self.forward_metadata = metadata

    def _publish_draft_round(
        self,
        packed: DeepseekV4ForwardMetadata,
        step: DeepseekV4ForwardMetadata,
    ) -> None:
        """A packed-draft round's end state: the bs*N verify views ride the
        prefill slot (the step-0 shape carrier — _select_decode_metadata's
        DECODE-gated fallback resolves them there), the bs-row step views own
        the decode slot."""
        self.forward_prefill_metadata = packed
        self.forward_decode_metadata = step
        self.forward_metadata = step

    def _prepare_draft_round(
        self,
        prefill_metadata: DeepseekV4ForwardMetadata,
        base_seq_lens: torch.Tensor,
    ) -> None:
        """Bind the draft's plain-step views for this round and run the
        indexer/slot-mapping refresh hooks over them (DraftRounds owns the
        copy-only binding; the kernel-facing refresh trio stays here; slot
        publication belongs to the publishers, never here)."""
        assert self.draft_rounds is not None
        metadata = self.draft_rounds.prepare(prefill_metadata, base_seq_lens)
        metadata.cache.refresh_decode_compressed_slot_mappings(
            token_to_req_indices=metadata.token_to_req_indices,
            query_start_loc=metadata.query_start_loc,
            seq_lens=metadata.seq_lens,
            is_valid_token=metadata.is_valid_token,
        )
        _refresh_decode_indexer_plan_cache(
            metadata,
            max_context_len=self.context_len,
        )
        _refresh_decode_indexer_schedule_metadata(metadata)

    def _select_decode_metadata(
        self,
        num_tokens: int,
    ) -> DeepseekV4ForwardMetadata | None:
        # The prefill slot is scanned LAST and only for DECODE-mode objects:
        # a packed draft round publishes its bs*N verify views there (the
        # step-0 shape carrier) while the decode slot holds the bs-row step
        # views — mirroring the model-side resolver's token-count fallback.
        # A stale extend-mode prefill object never passes the mode gate.
        for metadata in (
            self.forward_metadata,
            self.forward_decode_metadata,
            self.forward_prefill_metadata,
        ):
            if (
                metadata is not None
                and metadata.forward_mode is not None
                and metadata.forward_mode.is_decode()
                and metadata.token_to_req_indices.numel() == num_tokens
            ):
                return metadata
        return None

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
        if forward_mode.is_decode():
            raise RuntimeError(
                "DeepSeek V4 decode metadata goes through "
                "refresh_decode_metadata; init_forward_metadata only serves "
                "extend/mixed/idle"
            )
        dsv4_reset_attention_state()
        block_table_base_offsets = kwargs.pop("block_table_base_offsets", None) or {}
        num_tokens_arg = kwargs.pop("num_tokens", None)
        positions = kwargs.get("positions")
        if num_tokens_arg is not None:
            num_tokens = int(num_tokens_arg)
        elif isinstance(positions, torch.Tensor):
            num_tokens = int(positions.numel())
        else:
            num_tokens = bs
        del kwargs
        device = seq_lens.device
        # Batch-ordered base page table for ratio<=1 (uncompressed/SWA) indexer
        # layers, which have no dedicated group: the first (smallest-ratio)
        # compressed-KV full-history group's table (row i == batch position i).
        base_page_table = None
        if block_tables:
            base_group_id = first_v4_compressed_kv_group_id(block_tables)
            if base_group_id is not None:
                base_page_table = block_tables[base_group_id]
            if self._expected_cache_group_ids is not None:
                block_tables = self._prepare_cache_group_tables(
                    block_tables,
                    bs=bs,
                    actual_bs=bs,
                    seq_lens=seq_lens,
                    device=device,
                    phase="eager",
                )
            # Without a published group contract (warmup placeholders, unit
            # fixtures) the named tables are accepted as-is.
        elif bs > 0 and self._expected_cache_group_ids:
            raise RuntimeError(
                "DeepSeek V4 eager metadata is missing live cache group tables"
            )
        req_pool_indices = req_pool_indices[:bs]
        seq_lens = seq_lens[:bs].to(torch.int32)

        # Per-request query lengths, on device and as a host mirror: the
        # extend rows lead with their new-token counts, a MIXED round's decode
        # rows each carry the verify window, idle placeholders one token.
        verify_width = max(1, int(self.speculative_num_draft_tokens))
        num_prefill_reqs = 0 if forward_mode.is_idle() else num_extends
        if num_prefill_reqs and (
            extend_seq_lens_cpu.device.type != "cpu"
            or extend_prefix_lens_cpu.device.type != "cpu"
            or extend_seq_lens_cpu.numel() < num_prefill_reqs
            or extend_prefix_lens_cpu.numel() < num_prefill_reqs
        ):
            raise RuntimeError(
                "DeepSeek V4 prefill metadata requires complete CPU sequence "
                "and query length mirrors"
            )
        if forward_mode.is_idle():
            query_lens = torch.ones(bs, dtype=torch.int32, device=device)
            query_lens_cpu = torch.ones(bs, dtype=torch.int32)
        else:
            query_lens = torch.full(
                (bs,), verify_width, dtype=torch.int32, device=device
            )
            query_lens[:num_extends] = extend_seq_lens[:num_extends].to(
                device=device, dtype=torch.int32
            )
            query_lens_cpu = torch.full((bs,), verify_width, dtype=torch.int32)
            query_lens_cpu[:num_extends] = extend_seq_lens_cpu[:num_extends].to(
                dtype=torch.int32, device="cpu"
            )
        if num_prefill_reqs == bs:
            # Every row is an extend row: the host mirrors describe the whole
            # batch, so no device sync is needed.
            seq_lens_cpu = (
                extend_prefix_lens_cpu[:bs].to(dtype=torch.int32, device="cpu")
                + query_lens_cpu
            )
        else:
            # Mixed decode rows (and idle placeholders) have no host mirror of
            # their cache length: sync once for the batch, then overlay the
            # extend rows' exact mirrors.
            seq_lens_cpu = seq_lens.to(dtype=torch.int32, device="cpu")
            seq_lens_cpu[:num_prefill_reqs] = (
                extend_prefix_lens_cpu[:num_prefill_reqs].to(
                    dtype=torch.int32, device="cpu"
                )
                + query_lens_cpu[:num_prefill_reqs]
            )
        prefill_seq_lens = [int(v) for v in seq_lens_cpu[:num_prefill_reqs].tolist()]
        prefill_query_lens = [
            int(v) for v in query_lens_cpu[:num_prefill_reqs].tolist()
        ]
        if any(
            query_len < 0 or seq_len < query_len
            for seq_len, query_len in zip(
                prefill_seq_lens, prefill_query_lens, strict=True
            )
        ):
            raise RuntimeError(
                "DeepSeek V4 prefill CPU length mirrors contain an invalid "
                "sequence/query pair"
            )

        if num_prefill_reqs == bs:
            max_seq_len = max(prefill_seq_lens, default=0)
            if forward_mode.is_extend():
                max_seq_len += max(self.speculative_num_steps - 1, 0)
            max_pages = (
                max_seq_len + self.kernel_page_size - 1
            ) // self.kernel_page_size
        elif base_page_table is not None:
            # Mixed and packed decode have no complete host length mirror. The
            # kernels consume seq_lens, so retaining the live table width avoids
            # synchronizing on CUDA max reductions just to trim a view.
            max_pages = base_page_table.shape[1]
        else:
            max_pages = self.max_num_pages
        if base_page_table is not None:
            # The full-history group's batch-ordered table (row i == batch
            # position i); slice by batch rows directly.
            page_table = base_page_table[:bs, : max(max_pages, 1)].to(
                device=device, dtype=torch.int32
            )
        else:
            # Idle/warmup before any group table exists.
            page_table = torch.zeros(
                (bs, max(max_pages, 1)),
                dtype=torch.int32,
                device=device,
            )
        block_tables = {
            str(gid): table[:bs].to(device=device, dtype=torch.int32)
            for gid, table in block_tables.items()
        }
        base_offsets_on_device: dict[str, torch.Tensor] = {}
        for gid, off in block_table_base_offsets.items():
            if not isinstance(off, torch.Tensor):
                raise TypeError(
                    "DeepSeek V4 block_table_base_offsets values "
                    f"must be torch.Tensor, got {type(off).__name__} for "
                    f"group_id={gid!r}"
                )
            base_offsets_on_device[str(gid)] = off[:bs].to(
                device=device, dtype=torch.int32
            )
        (
            swa_page_table,
            compressor_state_block_tables,
            indexer_state_block_table,
            swa_base,
            compressor_state_base,
            indexer_state_base,
        ) = _split_block_tables_into_v4_metadata(
            block_tables,
            base_offsets_on_device,
        )
        req_ids = torch.arange(bs, device=device, dtype=torch.int32)
        num_prefill_tokens = sum(prefill_query_lens)
        if num_prefill_reqs == bs:
            token_to_req = torch.repeat_interleave(
                req_ids,
                query_lens.clamp_min(0),
                output_size=num_prefill_tokens,
            )
        else:
            metadata_tokens = sum(int(value) for value in query_lens_cpu.tolist())
            output_size = num_tokens
            if num_tokens_arg is None and not isinstance(positions, torch.Tensor):
                output_size = metadata_tokens
            if (
                forward_mode.is_mixed()
                and num_tokens_arg is not None
                and metadata_tokens != num_tokens
            ):
                raise RuntimeError(
                    "DeepSeek V4 mixed metadata token count mismatch: "
                    f"query_lens describe {metadata_tokens} tokens, packed input "
                    f"has {num_tokens}"
                )
            token_to_req = torch.repeat_interleave(
                req_ids,
                query_lens.clamp_min(0),
                output_size=output_size,
            )
        query_start_loc = torch.nn.functional.pad(
            torch.cumsum(query_lens.to(torch.int32), dim=0, dtype=torch.int32),
            (1, 0),
        )
        cache_metadata = DeepseekV4CacheMetadata(
            page_size=self.kernel_page_size,
            page_table=page_table,
            block_tables=block_tables,
            block_table_base_offsets=base_offsets_on_device,
            swa_page_table=swa_page_table,
            swa_base_logical_page=swa_base,
            compressor_state_block_tables=compressor_state_block_tables,
            compressor_state_base_logical_pages=compressor_state_base,
            indexer_state_block_table=indexer_state_block_table,
            indexer_state_base_logical_page=indexer_state_base,
        )
        metadata = DeepseekV4ForwardMetadata(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            query_lens=query_lens,
            query_start_loc=query_start_loc,
            token_to_req_indices=token_to_req,
            cache=cache_metadata,
            seq_lens_cpu=seq_lens_cpu,
            query_lens_cpu=query_lens_cpu,
            num_prefill_reqs=num_prefill_reqs,
            num_prefill_tokens=num_prefill_tokens,
            forward_mode=forward_mode,
        )
        if forward_mode.is_idle():
            # A pure DECODE init raises at the top, so idle is the only
            # decode-shaped mode left here.
            self._publish_decode(metadata)
        else:
            self._publish_prefill(metadata)

    def _update_decode_swa_metadata(
        self,
        metadata: DeepseekV4ForwardMetadata,
        *,
        window_size: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_metadata = metadata.attention
        num_tokens = metadata.token_to_req_indices.shape[0]
        needs_alloc = (
            attention_metadata.swa_indices is None
            or attention_metadata.swa_lens is None
            or attention_metadata.swa_indices.shape
            != (
                num_tokens,
                window_size,
            )
            or attention_metadata.swa_lens.shape != (num_tokens,)
            or attention_metadata.swa_indices.device != metadata.seq_lens.device
        )
        if needs_alloc:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "DeepSeek V4 decode SWA metadata must be allocated before "
                    "CUDA graph capture"
                )
            with torch.inference_mode(False):
                attention_metadata.swa_indices = torch.empty(
                    (num_tokens, window_size),
                    dtype=torch.int32,
                    device=metadata.seq_lens.device,
                )
                attention_metadata.swa_lens = torch.empty(
                    (num_tokens,),
                    dtype=torch.int32,
                    device=metadata.seq_lens.device,
                )

        cache_metadata = metadata.cache
        if cache_metadata.swa_page_table is None:
            raise RuntimeError("DeepSeek V4 missing cache-group block table for SWA KV")
        swa_page_table = cache_metadata.swa_page_table
        indices, lens = dsv4_decode_swa_indices_and_lens(
            query_start_loc=metadata.query_start_loc,
            seq_lens=metadata.seq_lens,
            token_to_req_indices=metadata.token_to_req_indices,
            block_table=swa_page_table,
            block_table_base_offsets=cache_metadata.swa_base_logical_page,
            window_size=window_size,
            block_size=block_size,
            is_valid_token=metadata.is_valid_token,
            out_indices=attention_metadata.swa_indices,
            out_lens=attention_metadata.swa_lens,
        )
        attention_metadata.swa_indices = indices
        attention_metadata.swa_lens = lens
        attention_metadata.swa_window_size = window_size
        attention_metadata.swa_block_size = block_size
        self._swa_window_size = window_size
        self._swa_block_size = block_size
        return indices, lens

    def _decode_compressed_attention_indices_and_lens(
        self,
        positions: torch.Tensor,
        *,
        compress_ratio: int,
        block_size: int,
        topk_indices: torch.Tensor | None,
        metadata: DeepseekV4ForwardMetadata | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if compress_ratio <= 1:
            return None, None
        if metadata is None:
            metadata = self.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 decode requires forward metadata")
        num_tokens = positions.numel()
        req_idx = metadata.token_to_req_indices[:num_tokens].to(torch.int64)
        page_table = metadata.cache.compressed_page_table(compress_ratio, block_size)
        block_table_base_offsets = (
            _compressed_block_table_base_offsets(metadata, compress_ratio)
            if page_table is not metadata.cache.page_table
            else None
        )
        is_valid_token = (
            metadata.is_valid_token[:num_tokens]
            if metadata.is_valid_token is not None
            else None
        )
        capturing = positions.is_cuda and torch.cuda.is_current_stream_capturing()
        if compress_ratio == 4:
            if topk_indices is None:
                raise RuntimeError("DeepSeek V4 CSA decode requires top-k indices")
            topk_local = topk_indices
            if block_table_base_offsets is not None:
                base_slots = block_table_base_offsets.to(
                    device=topk_indices.device,
                    dtype=torch.int64,
                )[req_idx] * int(block_size)
                topk_i64 = topk_indices.to(torch.int64)
                topk_local = torch.where(
                    topk_i64 >= 0,
                    topk_i64 - base_slots[:, None],
                    topk_i64,
                ).to(topk_indices.dtype)
            indices_2d, lens = dsv4_compute_global_topk_indices_and_lens(
                topk_indices=topk_local,
                token_to_req_indices=metadata.token_to_req_indices[:num_tokens],
                block_table=page_table,
                block_size=block_size,
                is_valid_token=is_valid_token,
            )
            return indices_2d.unsqueeze(1), lens

        cache_key = (
            int(compress_ratio),
            int(block_size),
            int(num_tokens),
            int(positions.data_ptr()) if positions.numel() else 0,
        )
        attention_metadata = metadata.attention
        dense_indices_cache = attention_metadata.decode_dense_compressed_indices_cache
        capture_safe_keys = (
            attention_metadata.decode_dense_compressed_indices_capture_safe_keys
        )
        cached = dense_indices_cache.get(cache_key)
        capture_cached = cache_key in capture_safe_keys
        if cached is not None and (not capturing or capture_cached):
            return cached

        width = self._dense_compressed_indices_width(compress_ratio)
        compressed_lens = torch.div(
            positions.to(torch.int64) + 1,
            compress_ratio,
            rounding_mode="floor",
        ).clamp(0, width)
        offsets = torch.arange(width, dtype=torch.int64, device=positions.device)
        local = offsets[None, :].expand(num_tokens, -1)
        valid = offsets[None, :] < compressed_lens[:, None]
        if is_valid_token is not None:
            valid = valid & is_valid_token.to(torch.bool)[:, None]
        lens = compressed_lens.to(torch.int32)
        if is_valid_token is not None:
            lens = torch.where(
                is_valid_token.to(torch.bool),
                lens,
                torch.zeros_like(lens),
            )

        safe_local = torch.where(valid, local, torch.zeros_like(local))
        pages = torch.div(safe_local, block_size, rounding_mode="floor")
        if block_table_base_offsets is not None:
            pages = (
                pages
                - block_table_base_offsets.to(
                    device=pages.device,
                    dtype=torch.int64,
                )[
                    req_idx
                ][:, None]
            )
        page_offsets = safe_local % block_size
        page_ids = metadata.cache.safe_page_ids(
            page_table,
            req_idx[:, None],
            pages.long(),
        )
        slots = page_ids * block_size + page_offsets
        indices_2d = torch.where(
            valid & (page_ids >= 0),
            slots,
            torch.full_like(slots, -1),
        )
        indices = indices_2d.to(torch.int32).unsqueeze(1)
        dense_indices_cache[cache_key] = (indices, lens)
        if capturing:
            capture_safe_keys.add(cache_key)
        return indices, lens

    def _dense_compressed_indices_width(self, compress_ratio: int) -> int:
        if compress_ratio <= 1:
            return 0
        width = max(1, (self.context_len + compress_ratio - 1) // compress_ratio)
        alignment = DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT
        return ((width + alignment - 1) // alignment) * alignment

    def _dense_prefill_local_compressed_indices(
        self,
        positions: torch.Tensor,
        *,
        compress_ratio: int,
        width: int,
        token_to_req_indices: torch.Tensor | None = None,
        block_table_base_offsets: torch.Tensor | None = None,
        compressed_block_size: int = 1,
        compressed_table_capacity: int | None = None,
    ) -> torch.Tensor:
        shape = (positions.numel(), width)
        if (
            self._prefill_dense_compressed_indices_buffer is None
            or self._prefill_dense_compressed_indices_buffer.device != positions.device
            or self._prefill_dense_compressed_indices_buffer.shape[0] < shape[0]
            or self._prefill_dense_compressed_indices_buffer.shape[1] < shape[1]
        ):
            self._prefill_dense_compressed_indices_buffer = torch.empty(
                shape,
                dtype=torch.int32,
                device=positions.device,
            )
        out = self._prefill_dense_compressed_indices_buffer[: shape[0], : shape[1]]
        return dsv4_build_dense_prefill_local_compressed_indices(
            positions=positions,
            compress_ratio=compress_ratio,
            width=width,
            out=out,
            token_to_req_indices=token_to_req_indices,
            block_table_base_offsets=block_table_base_offsets,
            compressed_block_size=compressed_block_size,
            compressed_table_capacity=compressed_table_capacity,
        )

    def forward_deepseek_v4_decode(
        self,
        *,
        q: torch.Tensor,
        positions: torch.Tensor,
        token_to_kv_pool,
        layer_id: int,
        kind: str,
        compress_ratio: int,
        num_local_heads: int,
        padded_heads: int,
        head_dim: int,
        window_size: int,
        softmax_scale: float,
        attn_sink: torch.Tensor,
        topk_indices: torch.Tensor | None,
        metadata: DeepseekV4ForwardMetadata | None = None,
    ) -> torch.Tensor:
        # Resolution is a READ: the mixed path passes its decode slice, the
        # model path resolves by token count; neither writes a slot.
        if metadata is None:
            metadata = self._select_decode_metadata(q.shape[0])
        if metadata is None:
            raise RuntimeError("DeepSeek V4 decode requires forward metadata")
        if metadata.forward_mode is None or not metadata.forward_mode.is_decode():
            raise RuntimeError(
                "forward_deepseek_v4_decode only supports ForwardMode.DECODE"
            )
        if metadata.token_to_req_indices.numel() != q.shape[0]:
            raise RuntimeError(
                "DeepSeek V4 decode metadata token count mismatch: "
                f"metadata_tokens={metadata.token_to_req_indices.numel()}, "
                f"q_tokens={q.shape[0]}"
            )
        if q.shape[1] == padded_heads:
            q_padded = q.contiguous()
        else:
            q_padded = torch.zeros(
                (q.shape[0], padded_heads, q.shape[2]),
                dtype=q.dtype,
                device=q.device,
            )
            q_padded[:, : q.shape[1]].copy_(q)
        swa_block_size = token_to_kv_pool.swa_block_size
        attention_metadata = metadata.attention
        if (
            attention_metadata.swa_indices is not None
            and attention_metadata.swa_lens is not None
            and attention_metadata.swa_window_size == window_size
            and attention_metadata.swa_block_size == swa_block_size
            and attention_metadata.swa_indices.shape[0] == positions.numel()
        ):
            swa_indices = attention_metadata.swa_indices
            swa_lens = attention_metadata.swa_lens
        else:
            swa_indices, swa_lens = self._update_decode_swa_metadata(
                metadata,
                window_size=window_size,
                block_size=swa_block_size,
            )
        compressed_block_size = token_to_kv_pool.get_compressed_block_size(layer_id)
        extra_indices, extra_lens = self._decode_compressed_attention_indices_and_lens(
            positions,
            compress_ratio=compress_ratio,
            block_size=compressed_block_size,
            topk_indices=topk_indices,
            metadata=metadata,
        )

        compressed_cache_2d = None
        if compress_ratio > 1:
            compressed_cache_2d = token_to_kv_pool.get_compressed_kv_buffer_2d(layer_id)

        out = dsv4_decode(
            q=q_padded,
            swa_kv_cache=token_to_kv_pool.get_swa_kv_buffer(layer_id),
            swa_slots=swa_indices,
            swa_lens=swa_lens,
            swa_page_size=swa_block_size,
            attn_sink=attn_sink,
            softmax_scale=softmax_scale,
            extra_kv_cache=compressed_cache_2d,
            extra_slots=extra_indices,
            extra_lens=extra_lens,
            extra_page_size=(
                compressed_block_size if compressed_cache_2d is not None else None
            ),
        )
        return out[:, :num_local_heads]

    def forward_deepseek_v4_mixed(
        self,
        *,
        q: torch.Tensor,
        positions: torch.Tensor,
        token_to_kv_pool,
        layer_id: int,
        kind: str,
        compress_ratio: int,
        num_local_heads: int,
        padded_heads: int,
        head_dim: int,
        window_size: int,
        softmax_scale: float,
        attn_sink: torch.Tensor,
        topk_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        metadata = self.forward_metadata
        if (
            metadata is None
            or metadata.forward_mode is None
            or not metadata.forward_mode.is_mixed()
        ):
            metadata = self.forward_prefill_metadata or metadata
        if (
            metadata is None
            or metadata.forward_mode is None
            or not metadata.forward_mode.is_mixed()
        ):
            raise RuntimeError("DeepSeek V4 mixed attention requires forward metadata")

        num_prefill_reqs = metadata.num_prefill_reqs
        num_prefill_tokens = metadata.num_prefill_tokens
        num_decode_reqs = metadata.decode_req_count()
        num_decode_tokens = metadata.decode_token_count()
        out = q.new_empty((q.shape[0], num_local_heads, head_dim))
        # Slices travel as parameters; the slots are never touched mid-call
        # (the save/restore dance was a mutation on a read path).
        if num_prefill_tokens > 0:
            prefill_metadata = self._metadata_slice(
                metadata,
                req_start=0,
                req_end=num_prefill_reqs,
                token_start=0,
                token_end=num_prefill_tokens,
                forward_mode=ForwardMode.EXTEND,
            )
            prefill_out = self.forward_deepseek_v4_prefill(
                q=q[:num_prefill_tokens],
                positions=positions[:num_prefill_tokens],
                token_to_kv_pool=token_to_kv_pool,
                layer_id=layer_id,
                kind=kind,
                compress_ratio=compress_ratio,
                num_local_heads=num_local_heads,
                padded_heads=padded_heads,
                head_dim=head_dim,
                window_size=window_size,
                softmax_scale=softmax_scale,
                attn_sink=attn_sink,
                topk_indices=(
                    topk_indices[:num_prefill_tokens]
                    if topk_indices is not None
                    else None
                ),
                metadata=prefill_metadata,
            )
            with nvtx_range(f"attn_{kind}_mixed_prefill_copy"):
                out[:num_prefill_tokens].copy_(prefill_out)
        if num_decode_tokens > 0:
            decode_end = num_prefill_tokens + num_decode_tokens
            decode_metadata = self._metadata_slice(
                metadata,
                req_start=num_prefill_reqs,
                req_end=num_prefill_reqs + num_decode_reqs,
                token_start=num_prefill_tokens,
                token_end=decode_end,
                forward_mode=ForwardMode.DECODE,
            )
            decode_out = self.forward_deepseek_v4_decode(
                q=q[num_prefill_tokens:decode_end],
                positions=positions[num_prefill_tokens:decode_end],
                token_to_kv_pool=token_to_kv_pool,
                layer_id=layer_id,
                kind=kind,
                compress_ratio=compress_ratio,
                num_local_heads=num_local_heads,
                padded_heads=padded_heads,
                head_dim=head_dim,
                window_size=window_size,
                softmax_scale=softmax_scale,
                attn_sink=attn_sink,
                topk_indices=(
                    topk_indices[num_prefill_tokens:decode_end]
                    if topk_indices is not None
                    else None
                ),
                metadata=decode_metadata,
            )
            with nvtx_range(f"attn_{kind}_mixed_decode_copy"):
                out[num_prefill_tokens:decode_end].copy_(decode_out)
        return out

    def _prefill_workspace(
        self,
        *,
        positions: torch.Tensor,
        token_to_kv_pool,
        layer_id: int,
        compress_ratio: int,
        window_size: int,
        head_dim: int,
        topk_indices: torch.Tensor | None,
        metadata: DeepseekV4ForwardMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if metadata is None:
            metadata = self.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 prefill requires forward metadata")
        cache_metadata = metadata.cache
        num_reqs = metadata.seq_lens.numel()
        prefix_lens = metadata.seq_lens - metadata.query_lens
        gather_lens = metadata.query_lens + torch.minimum(
            prefix_lens,
            torch.full_like(prefix_lens, max(window_size - 1, 0)),
        )
        if cache_metadata.swa_page_table is None:
            raise RuntimeError("DeepSeek V4 missing cache-group block table for SWA KV")
        swa_page_table = cache_metadata.swa_page_table
        compressed_lens = (
            torch.div(metadata.seq_lens, compress_ratio, rounding_mode="floor")
            if compress_ratio > 1
            else torch.zeros_like(metadata.seq_lens)
        )
        max_gather_len, compressed_base = self._prefill_workspace_bounds(
            metadata.seq_lens_cpu,
            metadata.query_lens_cpu,
            num_reqs=num_reqs,
            window_size=window_size,
            compress_ratio=compress_ratio,
        )
        workspace_width = max(1, compressed_base + max_gather_len)
        kv_workspace = self._get_prefill_workspace(
            num_reqs=num_reqs,
            workspace_width=workspace_width,
            head_dim=head_dim,
            device=positions.device,
        )

        if compress_ratio == 4 and topk_indices is not None:
            compressed_block_size = token_to_kv_pool.get_compressed_block_size(layer_id)
            compressed_cache = token_to_kv_pool.get_compressed_kv_buffer_2d(layer_id)
            compressed_page_table = cache_metadata.compressed_page_table(
                compress_ratio,
                compressed_block_size,
            )
            compressed_base_offsets = (
                _compressed_block_table_base_offsets(metadata, compress_ratio)
                if compressed_page_table is not cache_metadata.page_table
                else None
            )
            compressed_table_capacity = (
                compressed_page_table.shape[1] * compressed_block_size
            )
            dsv4_dequantize_and_gather_k_cache(
                out=kv_workspace,
                cache_2d=compressed_cache,
                seq_lens=compressed_lens,
                gather_lens=None,
                block_table=compressed_page_table,
                block_table_base_offsets=compressed_base_offsets,
                block_size=compressed_block_size,
                offset=0,
                max_gather_len=compressed_base,
            )
            dsv4_dequantize_and_gather_k_cache(
                out=kv_workspace,
                cache_2d=token_to_kv_pool.get_swa_kv_buffer(layer_id),
                seq_lens=metadata.seq_lens,
                gather_lens=gather_lens,
                block_table=swa_page_table,
                block_table_base_offsets=cache_metadata.swa_base_logical_page,
                block_size=token_to_kv_pool.swa_block_size,
                offset=compressed_base,
                max_gather_len=max_gather_len,
            )
            indices, lens = dsv4_combine_topk_swa_indices(
                topk_indices=topk_indices,
                query_start_loc=metadata.query_start_loc,
                seq_lens=metadata.seq_lens,
                gather_lens=gather_lens,
                window_size=window_size,
                compress_ratio=compress_ratio,
                topk=topk_indices.shape[-1],
                workspace_width=workspace_width,
                compressed_base=compressed_base,
                block_table_base_offsets=compressed_base_offsets,
                compressed_block_size=compressed_block_size,
                compressed_table_capacity=compressed_table_capacity,
            )
            return kv_workspace, indices, lens

        if compress_ratio == 4:
            raise RuntimeError("DeepSeek V4 CSA prefill requires top-k indices")

        swa_cache = token_to_kv_pool.get_swa_kv_buffer(layer_id)
        compressed_cache = (
            token_to_kv_pool.get_compressed_kv_buffer_2d(layer_id)
            if compress_ratio > 1
            else None
        )
        if compress_ratio > 1:
            assert compressed_cache is not None
            compressed_block_size = token_to_kv_pool.get_compressed_block_size(layer_id)
            compressed_page_table = cache_metadata.compressed_page_table(
                compress_ratio,
                compressed_block_size,
            )
            compressed_base_offsets = (
                _compressed_block_table_base_offsets(metadata, compress_ratio)
                if compressed_page_table is not cache_metadata.page_table
                else None
            )
            compressed_table_capacity = (
                compressed_page_table.shape[1] * compressed_block_size
            )
            dsv4_dequantize_and_gather_k_cache(
                out=kv_workspace,
                cache_2d=compressed_cache,
                seq_lens=compressed_lens,
                gather_lens=None,
                block_table=compressed_page_table,
                block_table_base_offsets=compressed_base_offsets,
                block_size=compressed_block_size,
                offset=0,
                max_gather_len=compressed_base,
            )
        dsv4_dequantize_and_gather_k_cache(
            out=kv_workspace,
            cache_2d=swa_cache,
            seq_lens=metadata.seq_lens,
            gather_lens=gather_lens,
            block_table=swa_page_table,
            block_table_base_offsets=cache_metadata.swa_base_logical_page,
            block_size=token_to_kv_pool.swa_block_size,
            offset=compressed_base,
            max_gather_len=max_gather_len,
        )
        if compress_ratio > 1:
            dense_compressed_indices = self._dense_prefill_local_compressed_indices(
                positions,
                compress_ratio=compress_ratio,
                width=self._dense_compressed_indices_width(compress_ratio),
                token_to_req_indices=metadata.token_to_req_indices[: positions.numel()],
                block_table_base_offsets=compressed_base_offsets,
                compressed_block_size=compressed_block_size,
                compressed_table_capacity=compressed_table_capacity,
            )
            indices, lens = dsv4_combine_topk_swa_indices(
                topk_indices=dense_compressed_indices,
                query_start_loc=metadata.query_start_loc,
                seq_lens=metadata.seq_lens,
                gather_lens=gather_lens,
                window_size=window_size,
                compress_ratio=compress_ratio,
                topk=dense_compressed_indices.shape[-1],
                workspace_width=workspace_width,
                compressed_base=compressed_base,
                block_table_base_offsets=compressed_base_offsets,
                compressed_block_size=compressed_block_size,
                compressed_table_capacity=compressed_table_capacity,
            )
            return kv_workspace, indices, lens

        indices, lens = dsv4_combine_dense_swa_indices(
            positions=positions,
            token_to_req_indices=metadata.token_to_req_indices[: positions.numel()],
            seq_lens=metadata.seq_lens,
            compressed_lens=compressed_lens,
            gather_lens=gather_lens,
            window_size=window_size,
            compress_ratio=compress_ratio,
            workspace_width=workspace_width,
            compressed_base=compressed_base,
        )
        return kv_workspace, indices, lens

    @staticmethod
    def _prefill_workspace_bounds(
        seq_lens_cpu: torch.Tensor | None,
        query_lens_cpu: torch.Tensor | None,
        *,
        num_reqs: int,
        window_size: int,
        compress_ratio: int,
    ) -> tuple[int, int]:
        """Compute prefill allocation bounds without reading the CUDA stream."""
        if num_reqs < 0:
            raise ValueError(f"num_reqs must be non-negative, got {num_reqs}")
        if num_reqs == 0:
            return 1, 0
        if (
            seq_lens_cpu is None
            or query_lens_cpu is None
            or seq_lens_cpu.device.type != "cpu"
            or query_lens_cpu.device.type != "cpu"
            or seq_lens_cpu.numel() != num_reqs
            or query_lens_cpu.numel() != num_reqs
        ):
            raise RuntimeError(
                "DeepSeek V4 prefill workspace sizing requires matching CPU "
                "sequence and query lengths"
            )
        seq_lens = [int(value) for value in seq_lens_cpu.tolist()]
        query_lens = [int(value) for value in query_lens_cpu.tolist()]
        if any(
            query_len < 0 or seq_len < query_len
            for seq_len, query_len in zip(seq_lens, query_lens, strict=True)
        ):
            raise RuntimeError(
                "DeepSeek V4 prefill workspace CPU length mirrors contain an "
                "invalid sequence/query pair"
            )
        gather_window = max(window_size - 1, 0)
        max_gather_len = max(
            query_len + min(seq_len - query_len, gather_window)
            for seq_len, query_len in zip(seq_lens, query_lens, strict=True)
        )
        compressed_base = (
            max(seq_len // compress_ratio for seq_len in seq_lens)
            if compress_ratio > 1
            else 0
        )
        return max_gather_len, compressed_base

    def _metadata_slice(
        self,
        metadata: DeepseekV4ForwardMetadata,
        *,
        req_start: int,
        req_end: int,
        token_start: int,
        token_end: int,
        forward_mode: ForwardMode,
    ) -> DeepseekV4ForwardMetadata:
        token_to_req = metadata.token_to_req_indices[token_start:token_end].to(
            torch.int32
        ) - int(req_start)
        cache_metadata = metadata.cache
        block_tables = {
            key: table[req_start:req_end]
            for key, table in cache_metadata.block_tables.items()
        }
        block_table_base_offsets = {
            key: offsets[req_start:req_end]
            for key, offsets in (cache_metadata.block_table_base_offsets.items())
        }
        compressor_state_block_tables = {
            key: table[req_start:req_end]
            for key, table in cache_metadata.compressor_state_block_tables.items()
        }
        compressor_state_base_logical_pages = {
            key: offsets[req_start:req_end]
            for key, offsets in (
                cache_metadata.compressor_state_base_logical_pages.items()
            )
        }
        query_lens = metadata.query_lens[req_start:req_end]
        req_count = max(0, req_end - req_start)
        token_count = max(0, token_end - token_start)
        num_prefill_reqs = req_count if forward_mode.is_extend_or_mixed() else 0
        num_prefill_tokens = token_count if forward_mode.is_extend_or_mixed() else 0
        query_start_loc = torch.nn.functional.pad(
            torch.cumsum(query_lens.to(torch.int32), dim=0, dtype=torch.int32),
            (1, 0),
        )
        sliced_cache = DeepseekV4CacheMetadata(
            page_size=cache_metadata.page_size,
            page_table=cache_metadata.page_table[req_start:req_end],
            block_tables=block_tables,
            block_table_base_offsets=block_table_base_offsets,
            swa_page_table=(
                cache_metadata.swa_page_table[req_start:req_end]
                if cache_metadata.swa_page_table is not None
                else None
            ),
            swa_base_logical_page=(
                cache_metadata.swa_base_logical_page[req_start:req_end]
                if cache_metadata.swa_base_logical_page is not None
                else None
            ),
            compressor_state_block_tables=compressor_state_block_tables,
            compressor_state_base_logical_pages=compressor_state_base_logical_pages,
            indexer_state_block_table=(
                cache_metadata.indexer_state_block_table[req_start:req_end]
                if cache_metadata.indexer_state_block_table is not None
                else None
            ),
            indexer_state_base_logical_page=(
                cache_metadata.indexer_state_base_logical_page[req_start:req_end]
                if cache_metadata.indexer_state_base_logical_page is not None
                else None
            ),
        )
        return DeepseekV4ForwardMetadata(
            req_pool_indices=metadata.req_pool_indices[req_start:req_end],
            seq_lens=metadata.seq_lens[req_start:req_end],
            query_lens=query_lens,
            query_start_loc=query_start_loc,
            token_to_req_indices=token_to_req,
            cache=sliced_cache,
            is_valid_token=(
                metadata.is_valid_token[token_start:token_end]
                if metadata.is_valid_token is not None
                else None
            ),
            seq_lens_cpu=(
                metadata.seq_lens_cpu[req_start:req_end]
                if metadata.seq_lens_cpu is not None
                else None
            ),
            query_lens_cpu=(
                metadata.query_lens_cpu[req_start:req_end]
                if metadata.query_lens_cpu is not None
                else None
            ),
            num_prefill_reqs=num_prefill_reqs,
            num_prefill_tokens=num_prefill_tokens,
            forward_mode=forward_mode,
        )

    def _forward_deepseek_v4_prefill_chunk(
        self,
        *,
        q: torch.Tensor,
        positions: torch.Tensor,
        token_to_kv_pool,
        layer_id: int,
        kind: str,
        compress_ratio: int,
        num_local_heads: int,
        padded_heads: int,
        head_dim: int,
        window_size: int,
        softmax_scale: float,
        attn_sink: torch.Tensor,
        topk_indices: torch.Tensor | None,
        metadata: DeepseekV4ForwardMetadata | None = None,
    ) -> torch.Tensor:
        if metadata is None:
            metadata = self.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 prefill requires forward metadata")
        with nvtx_range(f"attn_{kind}_prefill_pad_q"):
            if q.shape[1] == padded_heads:
                q_padded = q.contiguous()
            else:
                q_padded = torch.zeros(
                    (q.shape[0], padded_heads, q.shape[2]),
                    dtype=q.dtype,
                    device=q.device,
                )
                q_padded[:, : q.shape[1]].copy_(q)
        with nvtx_range(f"attn_{kind}_prefill_workspace"):
            kv_workspace, indices, lens = self._prefill_workspace(
                metadata=metadata,
                positions=positions,
                token_to_kv_pool=token_to_kv_pool,
                layer_id=layer_id,
                compress_ratio=compress_ratio,
                window_size=window_size,
                head_dim=head_dim,
                topk_indices=topk_indices,
            )
        with nvtx_range(f"attn_{kind}_prefill_selected_attention"):
            out = dsv4_prefill(
                q=q_padded,
                kv=kv_workspace,
                indices=indices,
                lens=lens,
                attn_sink=attn_sink,
                softmax_scale=softmax_scale,
            )
        return out[:, :num_local_heads]

    def forward_deepseek_v4_prefill(
        self,
        *,
        q: torch.Tensor,
        positions: torch.Tensor,
        token_to_kv_pool,
        layer_id: int,
        kind: str,
        compress_ratio: int,
        num_local_heads: int,
        padded_heads: int,
        head_dim: int,
        window_size: int,
        softmax_scale: float,
        attn_sink: torch.Tensor,
        topk_indices: torch.Tensor | None,
        metadata: DeepseekV4ForwardMetadata | None = None,
    ) -> torch.Tensor:
        if metadata is None:
            metadata = self.forward_metadata
        if (
            metadata is None
            or metadata.forward_mode is None
            or not metadata.forward_mode.is_extend_or_mixed()
        ):
            metadata = self.forward_prefill_metadata or metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 prefill requires forward metadata")
        if (
            metadata.forward_mode is None
            or not metadata.forward_mode.is_extend_or_mixed()
        ):
            raise RuntimeError(
                "forward_deepseek_v4_prefill only supports extend/prefill modes"
            )
        if metadata.token_to_req_indices.numel() != q.shape[0]:
            raise RuntimeError(
                "DeepSeek V4 prefill metadata token count mismatch: "
                f"metadata_tokens={metadata.token_to_req_indices.numel()}, "
                f"q_tokens={q.shape[0]}"
            )

        num_reqs = int(metadata.num_prefill_reqs or metadata.seq_lens.numel())
        if num_reqs <= self.prefill_chunk_size:
            return self._forward_deepseek_v4_prefill_chunk(
                q=q,
                positions=positions,
                token_to_kv_pool=token_to_kv_pool,
                layer_id=layer_id,
                kind=kind,
                compress_ratio=compress_ratio,
                num_local_heads=num_local_heads,
                padded_heads=padded_heads,
                head_dim=head_dim,
                window_size=window_size,
                softmax_scale=softmax_scale,
                attn_sink=attn_sink,
                topk_indices=topk_indices,
                metadata=metadata,
            )

        query_lens_cpu = metadata.query_lens_cpu
        if (
            query_lens_cpu is None
            or query_lens_cpu.device.type != "cpu"
            or query_lens_cpu.numel() < num_reqs
        ):
            raise RuntimeError(
                "DeepSeek V4 chunked prefill requires CPU query-length metadata"
            )
        token_offsets = [0]
        for query_len in query_lens_cpu[:num_reqs].tolist():
            token_offsets.append(token_offsets[-1] + int(query_len))
        if token_offsets[-1] != q.shape[0]:
            raise RuntimeError(
                "DeepSeek V4 chunked prefill query lengths do not match token "
                f"count: query_tokens={token_offsets[-1]}, q_tokens={q.shape[0]}"
            )
        out = q.new_empty((q.shape[0], num_local_heads, head_dim))
        for req_start in range(0, num_reqs, self.prefill_chunk_size):
            req_end = min(req_start + self.prefill_chunk_size, num_reqs)
            token_start = token_offsets[req_start]
            token_end = token_offsets[req_end]
            if token_end <= token_start:
                continue
            chunk_metadata = self._metadata_slice(
                metadata,
                req_start=req_start,
                req_end=req_end,
                token_start=token_start,
                token_end=token_end,
                forward_mode=ForwardMode.EXTEND,
            )
            chunk_out = self._forward_deepseek_v4_prefill_chunk(
                q=q[token_start:token_end],
                positions=positions[token_start:token_end],
                token_to_kv_pool=token_to_kv_pool,
                layer_id=layer_id,
                kind=kind,
                compress_ratio=compress_ratio,
                num_local_heads=num_local_heads,
                padded_heads=padded_heads,
                head_dim=head_dim,
                window_size=window_size,
                softmax_scale=softmax_scale,
                attn_sink=attn_sink,
                topk_indices=(
                    topk_indices[token_start:token_end]
                    if topk_indices is not None
                    else None
                ),
                metadata=chunk_metadata,
            )
            out[token_start:token_end].copy_(chunk_out)
        return out

    def init_cuda_graph_state(
        self,
        max_bs: int,
        cache_group_specs=(),
        cache_group_page_counts=None,
        max_tokens_per_req: int = 1,
        overlap_schedule_depth: int = 0,
        **kwargs,
    ):
        dsv4_reset_attention_state()
        # A fresh buffers object also drops every cached per-shape view:
        # views over reallocated storage must be rebuilt, never reused.
        self.graph = DeepseekV4GraphBuffers(
            max_bs=max_bs,
            max_tokens_per_req=max(
                1,
                int(max_tokens_per_req),
                int(self.speculative_num_draft_tokens or 0),
            ),
            max_num_pages=self.max_num_pages,
            device=self.device,
        )
        self.draft_rounds = DeepseekV4DraftRounds(self.graph) if self.is_draft else None
        specs, _ = self._configure_cache_group_contract(
            cache_group_specs,
            cache_group_page_counts,
        )
        group_ids = self._expected_cache_group_ids
        assert group_ids is not None
        if not group_ids:
            return
        # The backend owns the contract math (per-group table widths); the
        # buffers object owns the storage those widths size.
        group_widths = {
            gid: self._cuda_graph_group_table_width(
                spec,
                max_tokens_per_req=max_tokens_per_req,
                overlap_schedule_depth=overlap_schedule_depth,
            )
            for spec, gid in zip(specs, group_ids, strict=True)
        }
        sliding_gids = frozenset(
            gid
            for spec, gid in zip(specs, group_ids, strict=True)
            if str(getattr(spec, "retention", "")) == "sliding_window"
        )
        self.graph.allocate_group_tables(group_widths, sliding_gids)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        **kwargs,
    ):
        dsv4_reset_attention_state()
        block_tables = kwargs.pop("block_tables", None) or {}
        block_table_base_offsets = kwargs.pop("block_table_base_offsets", None) or {}
        num_tokens_arg = kwargs.pop("num_tokens", None)
        del kwargs
        if forward_mode is not None and not forward_mode.is_decode_or_idle():
            raise NotImplementedError(
                f"DeepSeek V4 CUDA graph capture not supported for {forward_mode}"
            )
        if num_tokens_arg is None:
            num_tokens = bs
        else:
            num_tokens = int(num_tokens_arg)
        tokens_per_req = self.graph.tokens_per_req(bs, num_tokens)
        is_packed_decode = (
            forward_mode is not None and forward_mode.is_decode() and num_tokens != bs
        )
        self.graph.fill_packed_rows(
            bs=bs,
            actual_bs=bs,
            tokens_per_req=tokens_per_req,
        )
        capture_seq_lens = seq_lens[:bs].to(torch.int32)
        if is_packed_decode:
            capture_seq_lens = torch.maximum(
                capture_seq_lens,
                torch.full_like(capture_seq_lens, tokens_per_req),
            )
        self.graph.req_pool_indices[:bs].copy_(req_pool_indices[:bs])
        self.graph.seq_lens[:bs].copy_(capture_seq_lens)
        metadata_forward_mode = forward_mode
        is_decode = (
            metadata_forward_mode is not None and metadata_forward_mode.is_decode()
        )
        offsets_on_device = {
            str(gid): off.to(device=self.device, dtype=torch.int32)
            for gid, off in block_table_base_offsets.items()
        }
        metadata_block_tables = self._prepare_cache_group_tables(
            block_tables,
            bs=bs,
            # Capture tables are placeholders and may name the null page. The
            # first replay must replace every live row with authoritative data.
            actual_bs=0,
            seq_lens=capture_seq_lens,
            device=self.device,
            phase="capture",
            output_buffers=self.graph.block_tables,
        )
        metadata_base_offsets = self.graph.refresh_base_offsets(
            bs,
            offsets_on_device,
        )
        (
            swa_page_table,
            compressor_state_block_tables,
            indexer_state_block_table,
            swa_base,
            compressor_state_base,
            indexer_state_base,
        ) = _split_block_tables_into_v4_metadata(
            metadata_block_tables,
            metadata_base_offsets,
        )
        # The single builder: every tensor field of the view is a fixed slice
        # assigned at construction; per-round state travels on the cache slot
        # (guard-exempt) swapped below.
        metadata = self.graph.views(
            bs,
            tokens_per_req,
            kernel_page_size=self.kernel_page_size,
            max_num_pages=self.max_num_pages,
            forward_mode=metadata_forward_mode,
        )
        metadata.cache = DeepseekV4CacheMetadata(
            page_size=self.kernel_page_size,
            page_table=self.graph.page_table[:bs, : self.max_num_pages],
            block_tables=metadata_block_tables,
            block_table_base_offsets=metadata_base_offsets,
            swa_page_table=swa_page_table,
            swa_base_logical_page=swa_base,
            compressor_state_block_tables=compressor_state_block_tables,
            compressor_state_base_logical_pages=compressor_state_base,
            indexer_state_block_table=indexer_state_block_table,
            indexer_state_base_logical_page=indexer_state_base,
            decode_compressed_slot_mappings=(
                metadata.cache.decode_compressed_slot_mappings
            ),
        )
        metadata.seq_lens_cpu = None
        metadata.query_lens_cpu = None
        if is_packed_decode and self.is_draft:
            self._prepare_draft_round(
                metadata,
                self.graph.seq_lens[:bs],
            )
            self._publish_draft_round(metadata, self.draft_rounds.current)
            return
        if is_decode:
            self._publish_decode(metadata)
        else:
            self.forward_metadata = metadata

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        forward_mode: ForwardMode,
        block_table: torch.Tensor | None = None,
        num_extends: int = 0,
        for_graph_replay: bool = False,
        **kwargs,
    ) -> None:
        dsv4_reset_attention_state()
        block_tables = kwargs.pop("block_tables", None) or {}
        block_table_base_offsets = kwargs.pop("block_table_base_offsets", None) or {}
        if actual_bs < 0 or actual_bs > bs:
            raise RuntimeError(
                f"DeepSeek V4 decode actual_bs={actual_bs} must be within 0..{bs}"
            )
        num_tokens_arg = kwargs.pop("num_tokens", None)
        del kwargs
        if forward_mode is not None and not forward_mode.is_decode_or_idle():
            raise NotImplementedError(
                f"DeepSeek V4 decode refresh not supported for {forward_mode}"
            )
        if num_tokens_arg is None:
            num_tokens = bs
        else:
            num_tokens = int(num_tokens_arg)
        tokens_per_req = self.graph.tokens_per_req(bs, num_tokens)
        is_packed_decode = (
            forward_mode is not None and forward_mode.is_decode() and num_tokens != bs
        )
        total_tokens = self.graph.fill_packed_rows(
            bs=bs,
            actual_bs=actual_bs,
            tokens_per_req=tokens_per_req,
        )
        # The same builder capture used: a bs/width never captured builds its
        # views lazily over the same persistent buffers (above-ladder eager,
        # enforce-eager), a captured one returns the recorded object.
        metadata = self.graph.views(
            bs,
            tokens_per_req,
            kernel_page_size=self.kernel_page_size,
            max_num_pages=self.max_num_pages,
            forward_mode=forward_mode,
        )
        self.graph.write_batch(bs, req_pool_indices, seq_lens)
        # Base page table for ratio<=1 indexer layers: the first (smallest-ratio)
        # compressed-KV full-history group's batch-ordered table.
        base_page_table = None
        if block_tables:
            base_group_id = first_v4_compressed_kv_group_id(block_tables)
            if base_group_id is not None:
                base_page_table = block_tables[base_group_id]
        if base_page_table is not None:
            width = min(base_page_table.shape[1], self.max_num_pages)
            if int(base_page_table.shape[0]) < actual_bs:
                raise RuntimeError(
                    "DeepSeek V4 CUDA graph base table row count "
                    f"{int(base_page_table.shape[0])} < actual_bs={actual_bs}"
                )
            self.graph.page_table[:bs].zero_()
            if actual_bs > 0 and width > 0:
                self.graph.page_table[:actual_bs, :width].copy_(
                    base_page_table[:actual_bs, :width].to(
                        device=self.device, dtype=torch.int32
                    )
                )
        offsets_on_device = {
            str(gid): off.to(device=self.device, dtype=torch.int32)
            for gid, off in block_table_base_offsets.items()
        }
        # actual_bs == 0 replays only ever carry capture/idle placeholder
        # tables; those must go through the refresh path below, never the
        # live-table prepare path.
        if actual_bs > 0 and block_tables and not self._expected_cache_group_ids:
            # No published group contract (warmup placeholders, unit
            # fixtures): accept the named tables as-is, like the former
            # eager path did.
            metadata_block_tables = {
                str(group_id): table.to(device=self.device, dtype=torch.int32)
                for group_id, table in block_tables.items()
            }
            metadata_base_offsets = offsets_on_device
        elif actual_bs > 0 and block_tables:
            metadata_block_tables = self._prepare_cache_group_tables(
                block_tables,
                bs=bs,
                actual_bs=actual_bs,
                seq_lens=seq_lens,
                device=seq_lens.device,
                phase="replay",
                output_buffers=self.graph.block_tables,
            )
            metadata_base_offsets = {}
        elif actual_bs > 0 and self._expected_cache_group_ids:
            raise RuntimeError(
                "DeepSeek V4 replay is missing live cache group tables; "
                "capture placeholders cannot be reused"
            )
        else:
            metadata_block_tables = self.graph.refresh_block_tables(
                bs,
                {
                    str(group_id): table.to(device=self.device, dtype=torch.int32)
                    for group_id, table in block_tables.items()
                },
                actual_bs=actual_bs,
                pad_value=-1,
            )
            metadata_base_offsets = self.graph.refresh_base_offsets(
                bs,
                offsets_on_device,
                actual_bs=actual_bs,
            )
        (
            swa_page_table,
            compressor_state_block_tables,
            indexer_state_block_table,
            swa_base,
            compressor_state_base,
            indexer_state_base,
        ) = _split_block_tables_into_v4_metadata(
            metadata_block_tables,
            metadata_base_offsets,
        )
        metadata_forward_mode = forward_mode
        is_decode = (
            metadata_forward_mode is not None and metadata_forward_mode.is_decode()
        )
        metadata.forward_mode = metadata_forward_mode
        metadata.token_to_req_indices = self.graph.token_to_req[:total_tokens]
        metadata.is_valid_token = self.graph.is_valid_token[:total_tokens]
        metadata.cache = DeepseekV4CacheMetadata(
            page_size=self.kernel_page_size,
            page_table=self.graph.page_table[:bs, : self.max_num_pages],
            block_tables=metadata_block_tables,
            block_table_base_offsets=metadata_base_offsets,
            swa_page_table=swa_page_table,
            swa_base_logical_page=swa_base,
            compressor_state_block_tables=compressor_state_block_tables,
            compressor_state_base_logical_pages=compressor_state_base,
            indexer_state_block_table=indexer_state_block_table,
            indexer_state_base_logical_page=indexer_state_base,
            decode_compressed_slot_mappings=(
                metadata.cache.decode_compressed_slot_mappings
            ),
        )
        metadata.num_prefill_reqs = 0
        metadata.num_prefill_tokens = 0
        if is_packed_decode and self.is_draft:
            self._prepare_draft_round(
                metadata,
                self.graph.seq_lens[:bs],
            )
        elif (
            not is_packed_decode
            and is_decode
            and self.is_draft
            and self.forward_prefill_metadata is not None
            and self.forward_prefill_metadata.req_pool_indices.numel() == bs
        ):
            # The extend round's plain-row draft refresh (unified draft
            # contract step two): rebuild the draft's step-1+ decode metadata
            # from the prefill state the init just published. Draft only — a
            # non-spec target's post-extend decode would otherwise build a
            # draft-decode object nothing ever reads.
            self._prepare_draft_round(
                self.forward_prefill_metadata,
                seq_lens[:bs].clone(),
            )
        if (
            metadata_forward_mode is not None
            and metadata_forward_mode.is_decode()
            and self._swa_window_size > 0
            and self._swa_block_size > 0
        ):
            self._update_decode_swa_metadata(
                metadata,
                window_size=self._swa_window_size,
                block_size=self._swa_block_size,
            )
            metadata.cache.refresh_decode_compressed_slot_mappings(
                token_to_req_indices=metadata.token_to_req_indices,
                query_start_loc=metadata.query_start_loc,
                seq_lens=metadata.seq_lens,
                is_valid_token=metadata.is_valid_token,
            )
            _refresh_decode_indexer_plan_cache(
                metadata,
                max_context_len=self.context_len,
            )
            _refresh_decode_indexer_schedule_metadata(metadata)
        if is_packed_decode and self.is_draft:
            self._publish_draft_round(metadata, self.draft_rounds.current)
            return
        if is_decode:
            self._publish_decode(metadata)
        else:
            self.forward_metadata = metadata

    def publish_draft_step_locations(self, cache_start, num_tokens):
        """No-op by design: V4's KV writes derive from its own per-group
        slot mappings, rebuilt each draft step by
        :meth:`advance_draft_forward_metadata` (the drafter calls both). No
        router window exists to publish; returning None keeps the drafter
        loop backend-agnostic."""
        del cache_start, num_tokens
        return None

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor | None = None):
        if self.draft_rounds is None or self.forward_prefill_metadata is None:
            raise RuntimeError("DeepSeek V4 draft metadata was not initialized")
        metadata = self.draft_rounds.advance(seq_lens)
        if self._swa_window_size > 0 and self._swa_block_size > 0:
            self._update_decode_swa_metadata(
                metadata,
                window_size=self._swa_window_size,
                block_size=self._swa_block_size,
            )
        # seq_lens just changed, so any previously-refreshed plan tensors are
        # stale. Re-run the same metadata-setup hooks the main path uses.
        metadata.cache.refresh_decode_compressed_slot_mappings(
            token_to_req_indices=metadata.token_to_req_indices,
            query_start_loc=metadata.query_start_loc,
            seq_lens=metadata.seq_lens,
            is_valid_token=metadata.is_valid_token,
        )
        _refresh_decode_indexer_plan_cache(
            metadata,
            max_context_len=self.context_len,
        )
        _refresh_decode_indexer_schedule_metadata(metadata)
        self._publish_decode(metadata)

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError("DeepSeek V4 uses the model-local attention forward")

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError("DeepSeek V4 uses the model-local attention forward")


register_backend("deepseek_v4", {AttentionArch.MLA}, DeepseekV4AttentionBackend)
