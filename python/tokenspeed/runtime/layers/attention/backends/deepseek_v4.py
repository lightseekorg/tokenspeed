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
from tokenspeed_kernel.ops.attention.flash_mla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_indexer_decode_metadata_compute,
)
from tokenspeed_kernel.ops.kvcache.triton import flat_tables_unpack
from tokenspeed_kernel.registry import error_fn

try:
    from tokenspeed_kernel.thirdparty import deep_gemm
except Exception:
    deep_gemm = None  # type: ignore[assignment]

from tokenspeed.runtime.configs.deepseek_v4_cache_spec import (
    DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT,
    deepseek_v4_swa_row_bytes,
    v4_compressed_kv_group_id,
)
from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.configs.paged_cache_spec import (
    compute_max_logical_pages_for_capture,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.flat_cache_tables import (
    CacheTableSource,
    FlatPackedTableOwnerSource,
    make_flat_packed_table_destinations,
    resolve_cache_table_source,
    validate_flat_table_base_offsets,
)
from tokenspeed.runtime.layers.attention.backends.base import (
    AttentionBackend,
    init_target_verify_draft_metadata,
)
from tokenspeed.runtime.layers.attention.deepseek_v4.metadata import (
    DeepseekV4ForwardMetadata,
)
from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    deepseek_v4_build_dense_prefill_local_compressed_indices,
    deepseek_v4_combine_dense_swa_indices,
    deepseek_v4_combine_topk_swa_indices,
    deepseek_v4_compute_global_topk_indices_and_lens,
    deepseek_v4_decode_swa_indices_and_lens,
    deepseek_v4_dequantize_and_gather_k_cache,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
    DeepseekV4CacheMetadata,
    _split_paged_cache_block_tables_into_v4_metadata,
)
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.nvtx import nvtx_range

DEEPSEEK_V4_DEFAULT_PREFILL_CHUNK_SIZE = 4


def _compressed_block_table_base_offsets(
    metadata: DeepseekV4ForwardMetadata,
    compress_ratio: int,
) -> torch.Tensor | None:
    return metadata.cache.paged_cache_block_table_base_offsets.get(
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
            plan.block_table.zero_()
            plan.max_context_len = 0
            refreshed_keys.add((compress_ratio, cache_block_size, num_tokens))
            continue
        positions = _decode_positions_from_metadata(metadata, num_tokens)
        token_to_req_indices = metadata.token_to_req_indices[:num_tokens]
        block_table = metadata.cache.compressed_block_table(
            compress_ratio,
            cache_block_size,
        )
        block_table_base_offsets = (
            _compressed_block_table_base_offsets(metadata, compress_ratio)
            if block_table is not metadata.cache.block_table
            else None
        )
        rows = int(block_table.shape[0]) if block_table.ndim >= 1 else 0
        cols = int(block_table.shape[1]) if block_table.ndim >= 2 else 0
        if rows <= 0 or cols <= 0:
            plan.context_lens.zero_()
            plan.block_table.zero_()
            plan.max_context_len = 0
            refreshed_keys.add((compress_ratio, cache_block_size, num_tokens))
            continue
        max_blocks = int(plan.block_table.shape[1])
        if max_context_len > 0:
            derived_max_len = max(
                1,
                (max_context_len + compress_ratio - 1) // compress_ratio,
            )
        else:
            derived_max_len = max(
                1,
                (block_table.shape[1] * cache_block_size + compress_ratio - 1)
                // compress_ratio,
            )
        if plan.max_context_len != derived_max_len:
            plan.max_context_len = derived_max_len
        deepseek_v4_indexer_decode_metadata_compute(
            positions=positions,
            token_to_req_indices=token_to_req_indices,
            block_table=block_table,
            cache_block_size=cache_block_size,
            compress_ratio=compress_ratio,
            max_blocks=max_blocks,
            out_context_lens=plan.context_lens,
            out_block_tables=plan.block_table,
            block_table_base_offsets=block_table_base_offsets,
        )
        if metadata.is_valid_token is not None:
            valid = metadata.is_valid_token[:num_tokens].to(
                device=plan.context_lens.device,
                dtype=torch.bool,
            )
            with torch.inference_mode():
                plan.context_lens.masked_fill_(~valid.view(num_tokens, 1), 0)
                plan.block_table.masked_fill_(
                    ~valid.to(device=plan.block_table.device).view(num_tokens, 1),
                    0,
                )
        refreshed_keys.add((compress_ratio, cache_block_size, num_tokens))


def _refresh_decode_indexer_schedule_metadata(
    metadata: DeepseekV4ForwardMetadata,
) -> None:
    indexer_metadata = metadata.indexer
    if not indexer_metadata.decode_schedule_metadata_cache:
        return
    if deep_gemm is None:
        return
    get_metadata = getattr(deep_gemm, "get_paged_mqa_logits_metadata", None)
    if get_metadata is None:
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
        refreshed = get_metadata(
            context_lens,
            cache_block_size,
            deep_gemm.get_num_sms(),
        )
        if (
            schedule_metadata.shape == refreshed.shape
            and schedule_metadata.device == refreshed.device
            and schedule_metadata.dtype == refreshed.dtype
        ):
            with torch.inference_mode():
                schedule_metadata.copy_(refreshed)
        else:
            indexer_metadata.decode_schedule_metadata_cache[key] = refreshed


class DeepseekV4AttentionBackend(AttentionBackend):
    """Metadata owner for the model-local DeepSeek V4 attention path."""

    uses_paged_cache_groups = True
    # Capability is source-neutral: the shared flat arena selects named flat
    # tables while the legacy per-pool allocation continues to select radix
    # paged tables. Both eager and CUDA-graph target/draft paths are wired.
    uses_flat_cache_groups = True
    flat_spec_capable = True
    requires_group_keyed_cache_locs = True
    uses_padded_decode_token_mask = True
    flat_tables_self_padding = True
    accepts_bound_cache_table_source = True
    # Explicit capability hook: the generic executor does not infer this
    # sequencing from model configuration.
    init_speculative_draft_metadata = init_target_verify_draft_metadata

    def __init__(self, config) -> None:
        super().__init__(config)
        self.page_size = config.page_size
        self.context_len = config.context_len
        rope_head_dim = getattr(config, "qk_rope_head_dim", None)
        self._fp8_ds_mla_row_bytes = (
            deepseek_v4_swa_row_bytes(config.head_dim, rope_head_dim)
            if rope_head_dim is not None
            else None
        )
        prefill_chunk_size = getattr(config, "deepseek_v4_prefill_chunk_size", None)
        if prefill_chunk_size is None:
            prefill_chunk_size = global_server_args_dict.get(
                "deepseek_v4_prefill_chunk_size",
                DEEPSEEK_V4_DEFAULT_PREFILL_CHUNK_SIZE,
            )
        self.prefill_chunk_size = max(1, int(prefill_chunk_size))
        self.max_num_pages = max(
            1,
            (self.context_len + self.page_size - 1) // self.page_size,
        )
        self.forward_metadata: DeepseekV4ForwardMetadata | None = None
        self.forward_prefill_metadata: DeepseekV4ForwardMetadata | None = None
        self.forward_decode_metadata: DeepseekV4ForwardMetadata | None = None
        self._decode_tile_metadata = {}
        self._cuda_graph_metadata = {}
        self._cuda_graph_paged_cache_block_tables: dict[str, torch.Tensor] = {}
        # Per-sliding-group [max_bs] int32 buffers mirroring the block-table
        # buffers; populated by init_cuda_graph_state.
        self._cuda_graph_paged_cache_base_offsets: dict[str, torch.Tensor] = {}
        self._cuda_graph_max_bs = 0
        self._prefill_workspace_buffer: torch.Tensor | None = None
        self._prefill_workspace_rows = 0
        self._prefill_workspace_head_dim = 0
        self._prefill_dense_compressed_indices_buffer: torch.Tensor | None = None
        self._decode_swa_window_size = 0
        self._decode_swa_block_size = 0
        self.speculative_num_steps = getattr(config, "speculative_num_steps", 0) or 0
        self.speculative_num_draft_tokens = (
            getattr(config, "speculative_num_draft_tokens", 0) or 0
        )
        self._draft_decode_step = 0
        self._draft_decode_base_seq_lens: torch.Tensor | None = None
        self._eager_draft_decode_metadata: DeepseekV4ForwardMetadata | None = None
        self._draft_decode_metadata: DeepseekV4ForwardMetadata | None = None
        self._cuda_graph_draft_decode_metadata = {}
        self._cuda_graph_query_start_by_tokens_per_req: dict[int, torch.Tensor] = {}
        self._cuda_graph_token_to_req_by_tokens_per_req: dict[int, torch.Tensor] = {}
        self._uses_flat_cache_tables = False
        self._place_cache_tables = self._copy_radix_cache_tables

    def _pop_cache_table_source(
        self,
        kwargs: dict,
        *,
        require_source: bool = False,
    ) -> CacheTableSource:
        """Normalize exactly one radix/flat ABI to existing named metadata."""
        source = kwargs.pop("cache_table_source", None)
        if source is not None:
            if not isinstance(source, CacheTableSource):
                raise TypeError(
                    "cache_table_source must be a CacheTableSource, got "
                    f"{type(source).__name__}"
                )
        else:
            source = resolve_cache_table_source(
                paged_tables=kwargs.pop("paged_cache_block_tables", None),
                paged_base_offsets=kwargs.pop(
                    "paged_cache_block_table_base_offsets", None
                ),
                flat_tables=kwargs.pop("flat_block_tables", None),
                flat_base_offsets=kwargs.pop("flat_block_table_base_offsets", None),
                require_source=require_source,
            )
        return source

    @staticmethod
    def _use_preplaced_flat_cache_tables(
        source: CacheTableSource,
        *,
        rows: int,
        device: torch.device | str,
    ) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
        del rows, device
        # Planned staging owns placement and stable mapping identity.
        return source.tables, source.base_offsets

    @staticmethod
    def _copy_radix_cache_tables(
        source: CacheTableSource,
        *,
        rows: int,
        device: torch.device | str,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        tables = {
            str(group_id): table[:rows].to(device=device, dtype=torch.int32)
            for group_id, table in source.tables.items()
        }
        base_offsets: dict[str, torch.Tensor] = {}
        for group_id, offsets in source.base_offsets.items():
            if not isinstance(offsets, torch.Tensor):
                raise TypeError(
                    "DeepSeek V4 paged cache base offsets must be torch.Tensor, "
                    f"got {type(offsets).__name__} for group_id={group_id!r}"
                )
            base_offsets[str(group_id)] = offsets[:rows].to(
                device=device,
                dtype=torch.int32,
            )
        return tables, base_offsets

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

    def _query_lens(
        self,
        bs: int,
        num_tokens: int,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode | None,
        num_extends: int,
        extend_seq_lens_cpu: torch.Tensor | None,
        extend_prefix_lens_cpu: torch.Tensor | None,
        extend_prefix_lens: torch.Tensor | None,
    ) -> torch.Tensor:
        if forward_mode is not None and forward_mode.is_decode_or_idle():
            if forward_mode.is_decode() and num_tokens != bs:
                if bs == 0:
                    return torch.zeros(0, dtype=torch.int32, device=seq_lens.device)
                if num_tokens % bs != 0:
                    raise RuntimeError(
                        "DeepSeek V4 packed decode metadata expects uniformly "
                        f"packed tokens per request, got num_tokens={num_tokens}, "
                        f"bs={bs}"
                    )
                tokens_per_req = num_tokens // bs
                return torch.full(
                    (bs,),
                    tokens_per_req,
                    dtype=torch.int32,
                    device=seq_lens.device,
                )
            return torch.ones(bs, dtype=torch.int32, device=seq_lens.device)
        if forward_mode is not None and forward_mode.is_mixed():
            verify_width = max(1, int(self.speculative_num_draft_tokens))
            lens = torch.full(
                (bs,),
                verify_width,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            num_prefill_reqs = max(0, min(int(num_extends), bs))
            if num_prefill_reqs == 0:
                return lens
            if extend_seq_lens_cpu is not None and extend_seq_lens_cpu.numel() > 0:
                lens[:num_prefill_reqs] = extend_seq_lens_cpu[:num_prefill_reqs].to(
                    seq_lens.device, dtype=torch.int32
                )
            elif extend_prefix_lens_cpu is not None:
                prefix = extend_prefix_lens_cpu[:num_prefill_reqs].to(
                    seq_lens.device, dtype=torch.int32
                )
                lens[:num_prefill_reqs] = (
                    seq_lens[:num_prefill_reqs].to(torch.int32) - prefix
                ).clamp_min(0)
            elif extend_prefix_lens is not None:
                prefix = extend_prefix_lens[:num_prefill_reqs].to(torch.int32)
                lens[:num_prefill_reqs] = (
                    seq_lens[:num_prefill_reqs].to(torch.int32) - prefix
                ).clamp_min(0)
            else:
                lens[:num_prefill_reqs] = seq_lens[:num_prefill_reqs].to(torch.int32)
            return lens
        if extend_seq_lens_cpu is not None:
            return extend_seq_lens_cpu[:bs].to(seq_lens.device, dtype=torch.int32)
        if extend_prefix_lens_cpu is not None:
            prefix = extend_prefix_lens_cpu[:bs].to(seq_lens.device, dtype=torch.int32)
            return (seq_lens[:bs].to(torch.int32) - prefix).clamp_min(0)
        if extend_prefix_lens is not None:
            prefix = extend_prefix_lens[:bs].to(torch.int32)
            return (seq_lens[:bs].to(torch.int32) - prefix).clamp_min(0)
        return seq_lens[:bs].to(torch.int32)

    def _query_lens_cpu(
        self,
        bs: int,
        forward_mode: ForwardMode | None,
        num_extends: int,
        extend_seq_lens_cpu: torch.Tensor | None,
        extend_prefix_lens_cpu: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if forward_mode is not None and forward_mode.is_mixed():
            verify_width = max(1, int(self.speculative_num_draft_tokens))
            lens = torch.full((bs,), verify_width, dtype=torch.int32)
            num_prefill_reqs = max(0, min(int(num_extends), bs))
            if num_prefill_reqs == 0:
                return lens
            if extend_seq_lens_cpu is None:
                return None
            lens[:num_prefill_reqs] = extend_seq_lens_cpu[:num_prefill_reqs].to(
                dtype=torch.int32, device="cpu"
            )
            return lens
        if extend_seq_lens_cpu is not None:
            return extend_seq_lens_cpu[:bs].to(dtype=torch.int32, device="cpu")
        if extend_prefix_lens_cpu is not None:
            return None
        return None

    @staticmethod
    def _query_start_offsets_from_cpu(
        query_lens_cpu: torch.Tensor | None,
        bs: int,
    ) -> tuple[int, ...] | None:
        """Build immutable host offsets without ever reading a CUDA tensor."""

        if query_lens_cpu is None:
            return None
        if query_lens_cpu.device.type != "cpu":
            raise RuntimeError("DeepSeek V4 query_lens_cpu must reside on CPU")
        if query_lens_cpu.numel() < bs:
            raise RuntimeError(
                "DeepSeek V4 query_lens_cpu is shorter than the request batch: "
                f"{query_lens_cpu.numel()} < {bs}"
            )
        offsets = [0]
        for length in query_lens_cpu[:bs].tolist():
            value = int(length)
            if value < 0:
                raise RuntimeError(
                    f"DeepSeek V4 query length must be non-negative, got {value}"
                )
            offsets.append(offsets[-1] + value)
        return tuple(offsets)

    def _draft_decode_is_valid_token(
        self,
        prefill_metadata: DeepseekV4ForwardMetadata,
    ) -> torch.Tensor | None:
        if prefill_metadata.is_valid_token is None:
            return None
        bs = prefill_metadata.req_pool_indices.numel()
        return prefill_metadata.is_valid_token[
            prefill_metadata.query_start_loc[:bs].to(torch.int64)
        ]

    def _is_cuda_graph_prefill_metadata(
        self,
        metadata: DeepseekV4ForwardMetadata,
    ) -> bool:
        bs = metadata.req_pool_indices.numel()
        return self._cuda_graph_metadata.get(bs) is metadata

    def _prepare_draft_decode_metadata(
        self,
        prefill_metadata: DeepseekV4ForwardMetadata,
        base_seq_lens: torch.Tensor,
    ) -> None:
        self.forward_prefill_metadata = prefill_metadata
        self._draft_decode_step = 0
        self._draft_decode_base_seq_lens = base_seq_lens

        bs = prefill_metadata.req_pool_indices.numel()
        device = prefill_metadata.req_pool_indices.device
        is_cuda_graph_metadata = self._is_cuda_graph_prefill_metadata(prefill_metadata)
        metadata = (
            self._cuda_graph_draft_decode_metadata.get(bs)
            if is_cuda_graph_metadata
            else self._eager_draft_decode_metadata
        )
        is_valid_token = self._draft_decode_is_valid_token(prefill_metadata)
        if (
            metadata is None
            or metadata.req_pool_indices.numel() != bs
            or metadata.seq_lens.numel() != bs
            or metadata.query_lens.numel() != bs
            or metadata.token_to_req_indices.numel() != bs
            or metadata.req_pool_indices.device != device
        ):
            query_lens = torch.ones(bs, dtype=torch.int32, device=device)
            token_to_req = torch.arange(bs, dtype=torch.int32, device=device)
            decode_seq_lens = torch.empty_like(base_seq_lens)
            decode_seq_lens.copy_(base_seq_lens)
            decode_is_valid_token = None
            if is_valid_token is not None:
                decode_is_valid_token = torch.empty_like(is_valid_token)
                decode_is_valid_token.copy_(is_valid_token)
            metadata = DeepseekV4ForwardMetadata(
                req_pool_indices=prefill_metadata.req_pool_indices,
                seq_lens=decode_seq_lens,
                query_lens=query_lens,
                query_start_loc=torch.nn.functional.pad(
                    torch.cumsum(
                        query_lens.to(torch.int32),
                        dim=0,
                        dtype=torch.int32,
                    ),
                    (1, 0),
                ),
                token_to_req_indices=token_to_req,
                cache=prefill_metadata.cache,
                query_start_offsets=tuple(range(bs + 1)),
                is_valid_token=decode_is_valid_token,
                forward_mode=ForwardMode.DECODE,
            )
            if is_cuda_graph_metadata:
                self._cuda_graph_draft_decode_metadata[bs] = metadata
            else:
                self._eager_draft_decode_metadata = metadata
            self._draft_decode_metadata = metadata
            return

        metadata.req_pool_indices = prefill_metadata.req_pool_indices
        metadata.cache = prefill_metadata.cache
        metadata.seq_lens.copy_(base_seq_lens)
        if is_valid_token is None:
            metadata.is_valid_token = None
        else:
            if (
                metadata.is_valid_token is None
                or metadata.is_valid_token.shape != is_valid_token.shape
                or metadata.is_valid_token.device != is_valid_token.device
            ):
                metadata.is_valid_token = torch.empty_like(is_valid_token)
            metadata.is_valid_token.copy_(is_valid_token)
        metadata.num_prefill_reqs = 0
        metadata.num_prefill_tokens = 0
        metadata.forward_mode = ForwardMode.DECODE
        # Reuse path: cached decode-indexer plans still describe the previous
        # prefill. Refresh after updating seq_lens so draft step 0 does not
        # reuse stale context_lens / block_table tensors.
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
        if not is_cuda_graph_metadata:
            self._eager_draft_decode_metadata = metadata
        self._draft_decode_metadata = metadata

    def _select_decode_metadata(
        self,
        num_tokens: int,
    ) -> DeepseekV4ForwardMetadata | None:
        for metadata in (self.forward_metadata, self.forward_decode_metadata):
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
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        table_source = self._pop_cache_table_source(
            kwargs,
            require_source=bool(self._uses_flat_cache_tables and bs > 0),
        )
        paged_cache_block_tables = table_source.tables
        num_tokens_arg = kwargs.pop("num_tokens", None)
        positions = kwargs.get("positions")
        num_extends_arg = kwargs.pop("num_extends", None)
        num_extends = bs if num_extends_arg is None else int(num_extends_arg)
        if num_tokens_arg is not None:
            num_tokens = int(num_tokens_arg)
        elif isinstance(positions, torch.Tensor):
            num_tokens = int(positions.numel())
        else:
            num_tokens = bs
        del kwargs
        device = seq_lens.device
        req_pool_indices = req_pool_indices[:bs]
        seq_lens = seq_lens[:bs].to(torch.int32)
        query_lens = self._query_lens(
            bs,
            num_tokens,
            seq_lens,
            forward_mode,
            num_extends,
            extend_seq_lens_cpu,
            extend_prefix_lens_cpu,
            extend_prefix_lens,
        )
        is_packed_decode = (
            forward_mode is not None and forward_mode.is_decode() and num_tokens != bs
        )
        metadata_forward_mode = forward_mode
        if forward_mode is not None and forward_mode.is_mixed():
            num_prefill_reqs = max(0, min(num_extends, bs))
        elif forward_mode is not None and forward_mode.is_extend_or_mixed():
            num_prefill_reqs = bs
        else:
            num_prefill_reqs = 0
        query_lens_cpu = None
        query_start_offsets = None
        if num_prefill_reqs:
            query_lens_cpu = self._query_lens_cpu(
                bs,
                forward_mode,
                num_extends,
                extend_seq_lens_cpu,
                extend_prefix_lens_cpu,
            )
            query_start_offsets = self._query_start_offsets_from_cpu(query_lens_cpu, bs)
        seq_lens_cpu = None
        if extend_prefix_lens_cpu is not None and query_lens_cpu is not None:
            # Build the CPU mirror from existing CPU inputs. A GPU->CPU copy
            # here would serialize every eager prefill before metadata setup.
            prefix_count = min(
                int(extend_prefix_lens_cpu.numel()),
                int(query_lens_cpu.numel()),
                num_prefill_reqs,
            )
            if prefix_count:
                # Mixed metadata intentionally retains only prefill rows: no
                # CPU total-length mirror exists for its decode suffix, and
                # zero-filling those rows would turn "unknown" into bad data.
                seq_lens_cpu = (
                    extend_prefix_lens_cpu[:prefix_count].to(
                        dtype=torch.int32,
                        device="cpu",
                    )
                    + query_lens_cpu[:prefix_count]
                )
        elif extend_seq_lens_cpu is not None and forward_mode is not None:
            if forward_mode.is_extend():
                seq_lens_cpu = extend_seq_lens_cpu[:bs].to(
                    dtype=torch.int32,
                    device="cpu",
                )
            # Mixed batches need prefix+query CPU mirrors to reconstruct total
            # lengths. Do not fall back to a synchronous GPU->CPU copy here.
        if self._uses_flat_cache_tables:
            # V4 flat consumers use only named group tables. Avoid both the
            # legacy req_to_page gather and seq_lens.max().item() GPU fence.
            # This zero-width view is a non-owning compatibility sentinel;
            # every flat single-table fallback rejects before dereferencing it.
            if paged_cache_block_tables:
                block_table = next(iter(paged_cache_block_tables.values()))[:bs, :0]
            else:
                block_table = seq_lens[:bs].reshape(bs, 1)[:, :0]
        else:
            max_seq_len = int(seq_lens.max().item()) if bs else 0
            if forward_mode is not None and forward_mode.is_extend():
                max_seq_len += max(self.speculative_num_steps - 1, 0)
            if is_packed_decode:
                max_seq_len += max(int(query_lens.max().item()) - 1, 0)
            max_pages = (max_seq_len + self.page_size - 1) // self.page_size
            if req_to_page is None:
                block_table = torch.zeros(
                    (bs, max(max_pages, 1)),
                    dtype=torch.int32,
                    device=device,
                )
            else:
                block_table = req_to_page[req_pool_indices, : max(max_pages, 1)]
        paged_cache_block_tables, base_offsets_on_device = self._place_cache_tables(
            table_source,
            rows=bs,
            device=device,
        )
        (
            swa_block_table,
            compressor_state_block_tables,
            indexer_state_block_table,
            swa_base,
            compressor_state_base,
            indexer_state_base,
        ) = _split_paged_cache_block_tables_into_v4_metadata(
            paged_cache_block_tables,
            base_offsets_on_device,
        )
        req_ids = torch.arange(bs, device=device, dtype=torch.int32)
        token_to_req = torch.repeat_interleave(req_ids, query_lens.clamp_min(0))
        if (
            forward_mode is not None
            and forward_mode.is_mixed()
            and num_tokens_arg is not None
        ):
            # numel() reads tensor shape metadata only. Reducing query_lens and
            # calling .item() here would synchronize its CUDA stream on every
            # eager mixed batch.
            metadata_tokens = token_to_req.numel()
            if metadata_tokens != num_tokens:
                raise RuntimeError(
                    "DeepSeek V4 mixed metadata token count mismatch: "
                    f"query_lens describe {metadata_tokens} tokens, packed input has "
                    f"{num_tokens}"
                )
        if (
            num_prefill_reqs
            and self._uses_flat_cache_tables
            and query_start_offsets is None
        ):
            raise RuntimeError(
                "DeepSeek V4 flat prefill metadata requires CPU query lengths; "
                "a CUDA query_start_loc fallback would synchronize every layer"
            )
        if not num_prefill_reqs:
            num_prefill_tokens = 0
        elif query_lens_cpu is not None:
            num_prefill_tokens = int(query_lens_cpu[:num_prefill_reqs].sum().item())
        else:
            # Compatibility fallback for direct callers that omit the CPU
            # mirrors; production prefill/mixed execution supplies them.
            num_prefill_tokens = int(query_lens[:num_prefill_reqs].sum().item())
        query_start_loc = torch.nn.functional.pad(
            torch.cumsum(query_lens.to(torch.int32), dim=0, dtype=torch.int32),
            (1, 0),
        )
        cache_metadata = DeepseekV4CacheMetadata(
            page_size=self.page_size,
            block_table=block_table,
            paged_cache_block_tables=paged_cache_block_tables,
            paged_cache_block_table_base_offsets=base_offsets_on_device,
            swa_block_table=swa_block_table,
            swa_base_logical_page=swa_base,
            compressor_state_block_tables=compressor_state_block_tables,
            compressor_state_base_logical_pages=compressor_state_base,
            indexer_state_block_table=indexer_state_block_table,
            indexer_state_base_logical_page=indexer_state_base,
        )
        self.forward_metadata = DeepseekV4ForwardMetadata(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            query_lens=query_lens,
            query_start_loc=query_start_loc,
            token_to_req_indices=token_to_req,
            cache=cache_metadata,
            seq_lens_cpu=seq_lens_cpu,
            query_lens_cpu=query_lens_cpu,
            query_start_offsets=query_start_offsets,
            num_prefill_reqs=num_prefill_reqs,
            num_prefill_tokens=num_prefill_tokens,
            forward_mode=metadata_forward_mode,
        )
        if is_packed_decode:
            self.forward_decode_metadata = self.forward_metadata
            if getattr(self, "is_draft", False):
                self._prepare_draft_decode_metadata(
                    self.forward_metadata,
                    seq_lens.clone(),
                )
        elif (
            metadata_forward_mode is not None
            and metadata_forward_mode.is_decode_or_idle()
        ):
            self.forward_decode_metadata = self.forward_metadata
            if (
                self.forward_prefill_metadata is not None
                and self.forward_prefill_metadata.req_pool_indices.numel()
                == seq_lens.numel()
            ):
                self._prepare_draft_decode_metadata(
                    self.forward_prefill_metadata,
                    seq_lens.clone(),
                )
        elif forward_mode is not None and forward_mode.is_extend_or_mixed():
            self.forward_prefill_metadata = self.forward_metadata
        self._decode_tile_metadata = {}

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
            attention_metadata.decode_swa_indices is None
            or attention_metadata.decode_swa_lens is None
            or attention_metadata.decode_swa_indices.shape
            != (
                num_tokens,
                window_size,
            )
            or attention_metadata.decode_swa_lens.shape != (num_tokens,)
            or attention_metadata.decode_swa_indices.device != metadata.seq_lens.device
        )
        if needs_alloc:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "DeepSeek V4 decode SWA metadata must be allocated before "
                    "CUDA graph capture"
                )
            with torch.inference_mode(False):
                attention_metadata.decode_swa_indices = torch.empty(
                    (num_tokens, window_size),
                    dtype=torch.int32,
                    device=metadata.seq_lens.device,
                )
                attention_metadata.decode_swa_lens = torch.empty(
                    (num_tokens,),
                    dtype=torch.int32,
                    device=metadata.seq_lens.device,
                )

        cache_metadata = metadata.cache
        if cache_metadata.swa_block_table is None:
            raise RuntimeError("DeepSeek V4 missing paged-cache block table for SWA KV")
        swa_block_table = cache_metadata.swa_block_table
        indices, lens = deepseek_v4_decode_swa_indices_and_lens(
            query_start_loc=metadata.query_start_loc,
            seq_lens=metadata.seq_lens,
            token_to_req_indices=metadata.token_to_req_indices,
            block_table=swa_block_table,
            block_table_base_offsets=cache_metadata.swa_base_logical_page,
            window_size=window_size,
            block_size=block_size,
            is_valid_token=metadata.is_valid_token,
            out_indices=attention_metadata.decode_swa_indices,
            out_lens=attention_metadata.decode_swa_lens,
        )
        attention_metadata.decode_swa_indices = indices
        attention_metadata.decode_swa_lens = lens
        attention_metadata.decode_swa_window_size = window_size
        attention_metadata.decode_swa_block_size = block_size
        self._decode_swa_window_size = window_size
        self._decode_swa_block_size = block_size
        return indices, lens

    def _decode_compressed_attention_indices_and_lens(
        self,
        positions: torch.Tensor,
        *,
        compress_ratio: int,
        block_size: int,
        topk_indices: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if compress_ratio <= 1:
            return None, None
        metadata = self.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 decode requires forward metadata")
        num_tokens = positions.numel()
        req_idx = metadata.token_to_req_indices[:num_tokens].to(torch.int64)
        block_table = metadata.cache.compressed_block_table(compress_ratio, block_size)
        block_table_base_offsets = (
            _compressed_block_table_base_offsets(metadata, compress_ratio)
            if block_table is not metadata.cache.block_table
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
            indices_2d, lens = deepseek_v4_compute_global_topk_indices_and_lens(
                topk_indices=topk_local,
                token_to_req_indices=metadata.token_to_req_indices[:num_tokens],
                block_table=block_table,
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
            block_table,
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
        return deepseek_v4_build_dense_prefill_local_compressed_indices(
            positions=positions,
            compress_ratio=compress_ratio,
            width=width,
            out=out,
        )

    def _get_decode_tile_metadata(self, kind: str, bs: int):
        phase = (
            "graph"
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
            else "eager"
        )
        tile_metadata = self._decode_tile_metadata.get((phase, kind, bs))
        if tile_metadata is not None:
            return tile_metadata
        if get_mla_metadata is error_fn:
            raise RuntimeError(
                "DeepSeek V4 decode requires FlashMLA latent attention. "
                "Build/install `tokenspeed-kernel/python` with FlashMLA."
            )
        tile_metadata = get_mla_metadata()[0]
        self._decode_tile_metadata[(phase, kind, bs)] = tile_metadata
        return tile_metadata

    def _fp8_ds_mla_cache_view(
        self,
        cache_2d: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        row_bytes = self._fp8_ds_mla_row_bytes
        if row_bytes is None:
            if cache_2d.shape[1] % block_size != 0:
                raise ValueError(
                    "DeepSeek V4 fp8_ds_mla cache width must be divisible by "
                    f"block_size={block_size}, got {cache_2d.shape[1]}"
                )
            row_bytes = cache_2d.shape[1] // block_size
        return torch.as_strided(
            cache_2d,
            (cache_2d.shape[0], block_size, 1, row_bytes),
            (
                cache_2d.stride(0),
                row_bytes,
                row_bytes,
                1,
            ),
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
    ) -> torch.Tensor:
        metadata = self._select_decode_metadata(q.shape[0])
        if metadata is None:
            raise RuntimeError("DeepSeek V4 decode requires forward metadata")
        self.forward_metadata = metadata
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
        if flash_mla_with_kvcache is error_fn:
            raise RuntimeError(
                "DeepSeek V4 decode requires FlashMLA latent attention. "
                "Build/install `tokenspeed-kernel/python` with FlashMLA."
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
            attention_metadata.decode_swa_indices is not None
            and attention_metadata.decode_swa_lens is not None
            and attention_metadata.decode_swa_window_size == window_size
            and attention_metadata.decode_swa_block_size == swa_block_size
            and attention_metadata.decode_swa_indices.shape[0] == positions.numel()
        ):
            swa_indices = attention_metadata.decode_swa_indices
            swa_lens = attention_metadata.decode_swa_lens
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
        )

        swa_cache = self._fp8_ds_mla_cache_view(
            token_to_kv_pool.get_swa_kv_buffer(layer_id),
            swa_block_size,
        )
        compressed_cache = None
        if compress_ratio > 1:
            compressed_cache = self._fp8_ds_mla_cache_view(
                token_to_kv_pool.get_compressed_kv_buffer_2d(layer_id),
                compressed_block_size,
            )

        out, _ = flash_mla_with_kvcache(
            q=q_padded.unsqueeze(1),
            k_cache=swa_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=head_dim,
            tile_scheduler_metadata=self._get_decode_tile_metadata(
                kind,
                q_padded.shape[0],
            ),
            softmax_scale=softmax_scale,
            is_fp8_kvcache=True,
            indices=swa_indices.unsqueeze(1),
            attn_sink=attn_sink,
            extra_k_cache=compressed_cache,
            extra_indices_in_kvcache=extra_indices,
            topk_length=swa_lens,
            extra_topk_length=extra_lens,
        )
        if out.dim() == 4:
            out = out.squeeze(1)
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
        saved_metadata = self.forward_metadata
        try:
            if num_prefill_tokens > 0:
                self.forward_metadata = self._metadata_slice(
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
                )
                with nvtx_range(f"attn_{kind}_mixed_prefill_copy"):
                    out[:num_prefill_tokens].copy_(prefill_out)
            if num_decode_tokens > 0:
                decode_end = num_prefill_tokens + num_decode_tokens
                self.forward_metadata = self._metadata_slice(
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
                )
                with nvtx_range(f"attn_{kind}_mixed_decode_copy"):
                    out[num_prefill_tokens:decode_end].copy_(decode_out)
        finally:
            self.forward_metadata = saved_metadata
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        if cache_metadata.swa_block_table is None:
            raise RuntimeError("DeepSeek V4 missing paged-cache block table for SWA KV")
        swa_block_table = cache_metadata.swa_block_table
        have_cpu_lengths = (
            metadata.seq_lens_cpu is not None
            and metadata.query_lens_cpu is not None
            and metadata.seq_lens_cpu.numel() >= num_reqs
            and metadata.query_lens_cpu.numel() >= num_reqs
        )
        if num_reqs and have_cpu_lengths:
            seq_lens_cpu = metadata.seq_lens_cpu[:num_reqs].to(torch.int64)
            query_lens_cpu = metadata.query_lens_cpu[:num_reqs].to(torch.int64)
            prefix_lens_cpu = seq_lens_cpu - query_lens_cpu
            gather_lens_cpu = query_lens_cpu + torch.minimum(
                prefix_lens_cpu,
                torch.full_like(prefix_lens_cpu, max(window_size - 1, 0)),
            )
            max_gather_len = int(gather_lens_cpu.max().item())
        else:
            if num_reqs and not cache_metadata.has_legacy_block_table:
                raise RuntimeError(
                    "DeepSeek V4 flat prefill workspace requires CPU sequence/query "
                    "length mirrors"
                )
            # Direct callers without CPU mirrors retain compatibility. The
            # production prefill path supplies mirrors and avoids this fence.
            max_gather_len = int(gather_lens.max().item()) if num_reqs else 1
        compressed_lens = (
            torch.div(metadata.seq_lens, compress_ratio, rounding_mode="floor")
            if compress_ratio > 1
            else torch.zeros_like(metadata.seq_lens)
        )
        if compress_ratio > 1 and num_reqs:
            if have_cpu_lengths:
                compressed_base = int(
                    torch.div(
                        seq_lens_cpu,
                        compress_ratio,
                        rounding_mode="floor",
                    )
                    .max()
                    .item()
                )
            else:
                compressed_base = int(compressed_lens.max().item())
        else:
            compressed_base = 0
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
            compressed_block_table = cache_metadata.compressed_block_table(
                compress_ratio,
                compressed_block_size,
            )
            deepseek_v4_dequantize_and_gather_k_cache(
                out=kv_workspace,
                cache_2d=compressed_cache,
                seq_lens=compressed_lens,
                gather_lens=None,
                block_table=compressed_block_table,
                block_size=compressed_block_size,
                offset=0,
            )
            deepseek_v4_dequantize_and_gather_k_cache(
                out=kv_workspace,
                cache_2d=token_to_kv_pool.get_swa_kv_buffer(layer_id),
                seq_lens=metadata.seq_lens,
                gather_lens=gather_lens,
                block_table=swa_block_table,
                block_table_base_offsets=cache_metadata.swa_base_logical_page,
                block_size=token_to_kv_pool.swa_block_size,
                offset=compressed_base,
            )
            indices, lens = deepseek_v4_combine_topk_swa_indices(
                topk_indices=topk_indices,
                query_start_loc=metadata.query_start_loc,
                seq_lens=metadata.seq_lens,
                gather_lens=gather_lens,
                window_size=window_size,
                compress_ratio=compress_ratio,
                topk=topk_indices.shape[-1],
                workspace_width=workspace_width,
                compressed_base=compressed_base,
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
            compressed_block_table = cache_metadata.compressed_block_table(
                compress_ratio,
                compressed_block_size,
            )
            deepseek_v4_dequantize_and_gather_k_cache(
                out=kv_workspace,
                cache_2d=compressed_cache,
                seq_lens=compressed_lens,
                gather_lens=None,
                block_table=compressed_block_table,
                block_size=compressed_block_size,
                offset=0,
            )
        deepseek_v4_dequantize_and_gather_k_cache(
            out=kv_workspace,
            cache_2d=swa_cache,
            seq_lens=metadata.seq_lens,
            gather_lens=gather_lens,
            block_table=swa_block_table,
            block_table_base_offsets=cache_metadata.swa_base_logical_page,
            block_size=token_to_kv_pool.swa_block_size,
            offset=compressed_base,
        )
        if compress_ratio > 1:
            dense_compressed_indices = self._dense_prefill_local_compressed_indices(
                positions,
                compress_ratio=compress_ratio,
                width=self._dense_compressed_indices_width(compress_ratio),
            )
            indices, lens = deepseek_v4_combine_topk_swa_indices(
                topk_indices=dense_compressed_indices,
                query_start_loc=metadata.query_start_loc,
                seq_lens=metadata.seq_lens,
                gather_lens=gather_lens,
                window_size=window_size,
                compress_ratio=compress_ratio,
                topk=dense_compressed_indices.shape[-1],
                workspace_width=workspace_width,
                compressed_base=compressed_base,
            )
            return kv_workspace, indices, lens

        indices, lens = deepseek_v4_combine_dense_swa_indices(
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
        paged_cache_block_tables = {
            key: table[req_start:req_end]
            for key, table in cache_metadata.paged_cache_block_tables.items()
        }
        paged_cache_block_table_base_offsets = {
            key: offsets[req_start:req_end]
            for key, offsets in (
                cache_metadata.paged_cache_block_table_base_offsets.items()
            )
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
        query_start_offsets = None
        if metadata.query_start_offsets is not None:
            base = metadata.query_start_offsets[req_start]
            query_start_offsets = tuple(
                offset - base
                for offset in metadata.query_start_offsets[req_start : req_end + 1]
            )
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
            block_table=cache_metadata.block_table[req_start:req_end],
            paged_cache_block_tables=paged_cache_block_tables,
            paged_cache_block_table_base_offsets=paged_cache_block_table_base_offsets,
            swa_block_table=(
                cache_metadata.swa_block_table[req_start:req_end]
                if cache_metadata.swa_block_table is not None
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
            query_start_offsets=query_start_offsets,
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
    ) -> torch.Tensor:
        metadata = self.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 prefill requires forward metadata")
        if flash_mla_sparse_fwd is error_fn:
            raise RuntimeError(
                "DeepSeek V4 prefill requires FlashMLA sparse attention. "
                "Build/install `tokenspeed-kernel/python` with FlashMLA."
            )

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
                positions=positions,
                token_to_kv_pool=token_to_kv_pool,
                layer_id=layer_id,
                compress_ratio=compress_ratio,
                window_size=window_size,
                head_dim=head_dim,
                topk_indices=topk_indices,
            )
        with nvtx_range(f"attn_{kind}_prefill_flashmla"):
            out, _, _ = flash_mla_sparse_fwd(
                q=q_padded,
                kv=kv_workspace.view(-1, 1, head_dim),
                indices=indices.unsqueeze(1),
                sm_scale=softmax_scale,
                attn_sink=attn_sink,
                topk_length=lens,
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
    ) -> torch.Tensor:
        metadata = self.forward_metadata
        if (
            metadata is None
            or metadata.forward_mode is None
            or not metadata.forward_mode.is_extend_or_mixed()
        ):
            metadata = self.forward_prefill_metadata or metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 prefill requires forward metadata")
        self.forward_metadata = metadata
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
            )

        token_offsets = metadata.query_start_offsets
        if token_offsets is None:
            raise RuntimeError(
                "DeepSeek V4 prefill chunking requires cached CPU query offsets; "
                "query_start_loc must not be synchronized once per layer"
            )
        out = q.new_empty((q.shape[0], num_local_heads, head_dim))
        saved_metadata = self.forward_metadata
        try:
            for req_start in range(0, num_reqs, self.prefill_chunk_size):
                req_end = min(req_start + self.prefill_chunk_size, num_reqs)
                token_start = token_offsets[req_start]
                token_end = token_offsets[req_end]
                if token_end <= token_start:
                    continue
                self.forward_metadata = self._metadata_slice(
                    saved_metadata,
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
                )
                out[token_start:token_end].copy_(chunk_out)
        finally:
            self.forward_metadata = saved_metadata
        return out

    def init_cuda_graph_state(
        self,
        max_bs: int,
        seq_lens_buf: torch.Tensor | None = None,
        paged_cache_group_specs=(),
        max_tokens_per_req: int = 1,
        overlap_schedule_depth: int = 0,
        flat_capture_cols_by_group: dict[str, int] | None = None,
        flat_graph_batch_rows: int | None = None,
        cache_table_source_kind: str | None = None,
    ):
        del seq_lens_buf
        paged_cache_group_specs = tuple(paged_cache_group_specs or ())
        spec_group_ids = tuple(str(spec.group_id) for spec in paged_cache_group_specs)
        planned_capture_cols: dict[str, int] | None = None
        if flat_capture_cols_by_group is not None:
            planned_capture_cols = {
                str(group_id): cols
                for group_id, cols in flat_capture_cols_by_group.items()
            }
            if set(planned_capture_cols) != set(spec_group_ids):
                raise RuntimeError(
                    "DeepSeek V4 CUDA graph flat plan groups disagree with the "
                    "owner-local pool specs: "
                    f"plan={sorted(planned_capture_cols)}, "
                    f"specs={sorted(spec_group_ids)}"
                )
            for group_id, cols in planned_capture_cols.items():
                if isinstance(cols, bool) or not isinstance(cols, int) or cols <= 0:
                    raise ValueError(
                        "DeepSeek V4 flat capture cols must be positive integers: "
                        f"group={group_id!r}, cols={cols!r}"
                    )
            if flat_graph_batch_rows is None:
                raise RuntimeError(
                    "DeepSeek V4 flat capture cols require planned graph batch rows"
                )
            if int(flat_graph_batch_rows) not in (0, int(max_bs)):
                raise RuntimeError(
                    "DeepSeek V4 CUDA graph batch rows disagree with flat plan: "
                    f"backend={max_bs}, plan={flat_graph_batch_rows}"
                )
        elif flat_graph_batch_rows is not None:
            raise RuntimeError(
                "DeepSeek V4 planned graph batch rows require flat capture cols"
            )
        inferred_source_kind = "flat" if planned_capture_cols is not None else "radix"
        if cache_table_source_kind is None:
            # Compatibility for focused backend tests and non-executor callers.
            cache_table_source_kind = inferred_source_kind
        if cache_table_source_kind not in ("radix", "flat"):
            raise ValueError(
                "DeepSeek V4 cache table source kind must be radix or flat, got "
                f"{cache_table_source_kind!r}"
            )
        if cache_table_source_kind != inferred_source_kind:
            raise RuntimeError(
                "DeepSeek V4 cache table source binding disagrees with its "
                f"graph plan: binding={cache_table_source_kind}, "
                f"plan={inferred_source_kind}"
            )
        self._flat_capture_cols_by_group = planned_capture_cols
        self._uses_flat_cache_tables = cache_table_source_kind == "flat"
        self._place_cache_tables = (
            self._use_preplaced_flat_cache_tables
            if self._uses_flat_cache_tables
            else self._copy_radix_cache_tables
        )
        self._flat_graph_batch_rows = (
            int(flat_graph_batch_rows) if flat_graph_batch_rows is not None else None
        )
        flat_graph_allocation_rows = (
            self._flat_graph_batch_rows if planned_capture_cols is not None else max_bs
        )
        self._decode_tile_metadata = {}
        self._cuda_graph_max_tokens_per_req = max(
            1,
            int(max_tokens_per_req),
            int(self.speculative_num_draft_tokens or 0),
        )
        max_tokens = max_bs * self._cuda_graph_max_tokens_per_req
        # The flat plan owns all group tables explicitly.  Keep the legacy
        # single radix table as a zero-byte sentinel so it cannot silently add
        # an unplanned O(max_bs * context) allocation or become a fallback.
        legacy_block_table_shape = (
            (0, 0) if planned_capture_cols is not None else (max_bs, self.max_num_pages)
        )
        self._cuda_graph_block_table = torch.zeros(
            legacy_block_table_shape,
            dtype=torch.int32,
            device=self.device,
        )
        self._cuda_graph_req_pool_indices = torch.zeros(
            (max_bs,),
            dtype=torch.int32,
            device=self.device,
        )
        self._cuda_graph_seq_lens = torch.ones(
            (max_bs,),
            dtype=torch.int32,
            device=self.device,
        )
        self._cuda_graph_query_lens = torch.ones(
            (max_bs,),
            dtype=torch.int32,
            device=self.device,
        )
        self._cuda_graph_query_start_loc = torch.arange(
            max_bs + 1,
            dtype=torch.int32,
            device=self.device,
        )
        self._cuda_graph_token_to_req = torch.arange(
            max_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        query_start_base = torch.arange(
            max_bs + 1,
            dtype=torch.int32,
            device=self.device,
        )
        token_to_req_base = torch.arange(
            max_bs,
            dtype=torch.int32,
            device=self.device,
        )
        self._cuda_graph_query_start_by_tokens_per_req = {}
        self._cuda_graph_token_to_req_by_tokens_per_req = {}
        for tokens_per_req in range(1, self._cuda_graph_max_tokens_per_req + 1):
            self._cuda_graph_query_start_by_tokens_per_req[tokens_per_req] = (
                query_start_base * tokens_per_req
            )
            self._cuda_graph_token_to_req_by_tokens_per_req[tokens_per_req] = (
                token_to_req_base.repeat_interleave(tokens_per_req)
            )
        self._cuda_graph_max_bs = max_bs
        self._cuda_graph_paged_cache_block_tables = {}
        self._cuda_graph_paged_cache_base_offsets = {}
        self._cuda_graph_table_views_by_bs: dict[int, dict[str, torch.Tensor]] = {}
        self._cuda_graph_base_views_by_bs: dict[int, dict[str, torch.Tensor]] = {}
        self._cuda_graph_flat_packed_storage: torch.Tensor | None = None
        self._cuda_graph_flat_max_cols = 0
        self._cuda_graph_flat_group_ids: tuple[str, ...] = ()
        self._cuda_graph_flat_capture_cols: tuple[int, ...] = ()
        self._cuda_graph_validated_packed_sources: dict[
            int,
            tuple[
                torch.Tensor,
                torch.Tensor,
                Mapping[str, torch.Tensor],
                Mapping[str, torch.Tensor],
                tuple[str, ...],
            ],
        ] = {}
        if planned_capture_cols is not None:
            destinations, packed_numel = make_flat_packed_table_destinations(
                planned_capture_cols,
                batch_rows=flat_graph_allocation_rows,
            )
            self._cuda_graph_flat_group_ids = tuple(
                destination.group_id for destination in destinations
            )
            self._cuda_graph_flat_capture_cols = tuple(
                destination.table_cols for destination in destinations
            )
            self._cuda_graph_flat_packed_storage = torch.zeros(
                (packed_numel,),
                dtype=torch.int32,
                device=self.device,
            )
            for destination in destinations:
                gid = destination.group_id
                table_numel = flat_graph_allocation_rows * destination.table_cols
                self._cuda_graph_paged_cache_block_tables[gid] = (
                    self._cuda_graph_flat_packed_storage[
                        destination.table_offset : destination.table_offset
                        + table_numel
                    ].view(flat_graph_allocation_rows, destination.table_cols)
                )
                self._cuda_graph_paged_cache_base_offsets[gid] = (
                    self._cuda_graph_flat_packed_storage[
                        destination.base_offset : destination.base_offset
                        + flat_graph_allocation_rows
                    ]
                )
            self._cuda_graph_flat_max_cols = max(
                destination.table_cols for destination in destinations
            )
        else:
            for spec in paged_cache_group_specs:
                gid = str(spec.group_id)
                max_pages = compute_max_logical_pages_for_capture(
                    spec,
                    max_context_len=self.context_len,
                    max_tokens_per_req=max_tokens_per_req,
                    overlap_schedule_depth=overlap_schedule_depth,
                )
                self._cuda_graph_paged_cache_block_tables[gid] = torch.zeros(
                    (flat_graph_allocation_rows, max_pages),
                    dtype=torch.int32,
                    device=self.device,
                )
                self._cuda_graph_paged_cache_base_offsets[gid] = torch.zeros(
                    (flat_graph_allocation_rows,),
                    dtype=torch.int32,
                    device=self.device,
                )
        if self._cuda_graph_flat_packed_storage is not None:
            self._flat_graph_metadata_nbytes = int(
                self._cuda_graph_flat_packed_storage.numel()
            ) * int(self._cuda_graph_flat_packed_storage.element_size())
        else:
            self._flat_graph_metadata_nbytes = sum(
                int(tensor.numel()) * int(tensor.element_size())
                for tensor in (
                    *self._cuda_graph_paged_cache_block_tables.values(),
                    *self._cuda_graph_paged_cache_base_offsets.values(),
                )
            )
        if planned_capture_cols is not None:
            expected_nbytes = (
                4
                * flat_graph_allocation_rows
                * sum(cols + 1 for cols in planned_capture_cols.values())
            )
            if self._flat_graph_metadata_nbytes != expected_nbytes:
                raise RuntimeError(
                    "DeepSeek V4 CUDA graph table/base allocation disagrees "
                    f"with owner plan: actual={self._flat_graph_metadata_nbytes}, "
                    f"expected={expected_nbytes}"
                )
        self._cuda_graph_is_valid_token = torch.ones(
            max_tokens,
            dtype=torch.bool,
            device=self.device,
        )

    def _refresh_cuda_graph_paged_cache_block_tables(
        self,
        bs: int,
        paged_cache_block_tables: dict[str, torch.Tensor],
        *,
        pad_value: int,
        require_all_groups: bool = False,
    ) -> dict[str, torch.Tensor]:
        if not self._cuda_graph_paged_cache_block_tables:
            return {}
        out = self._cuda_graph_table_views_by_bs.get(bs)
        if out is None:
            # Capture is the cold boundary where view dictionaries may be
            # allocated. Replay reuses the exact mapping and tensor views.
            out = {
                group_id: buf[:bs]
                for group_id, buf in self._cuda_graph_paged_cache_block_tables.items()
            }
            self._cuda_graph_table_views_by_bs[bs] = out
        for group_id, buf in self._cuda_graph_paged_cache_block_tables.items():
            table = paged_cache_block_tables.get(group_id)
            if table is None and require_all_groups:
                raise RuntimeError(
                    "DeepSeek V4 CUDA graph flat replay is missing captured "
                    f"cache group {group_id!r}"
                )
            if table is None:
                if bs > 0:
                    # Radix sources may omit a group; preserve their legacy
                    # invalid/default contents without touching flat replay.
                    buf[:bs].fill_(pad_value)
                continue
            rows = int(table.shape[0])
            if rows > bs:
                raise RuntimeError(
                    "DeepSeek V4 CUDA graph paged cache table row count "
                    f"mismatch for {group_id!r}: got {rows}, padded bs is {bs}"
                )
            cols = int(table.shape[1])
            if cols > int(buf.shape[1]):
                raise RuntimeError(
                    "DeepSeek V4 CUDA graph paged cache table width "
                    f"mismatch for {group_id!r}: got {cols}, capture "
                    f"buffer has {int(buf.shape[1])}"
                )
            if bs > 0:
                capture_cols = int(buf.shape[1])
                if cols < capture_cols and rows > 0:
                    buf[:rows, cols:].fill_(pad_value)
                if cols > 0:
                    source = table[:rows, :cols]
                    if not require_all_groups:
                        source = source.to(torch.int32)
                    buf[:rows, :cols].copy_(source)
                if rows < bs:
                    buf[rows:bs].zero_()
        return out

    def _refresh_cuda_graph_base_offsets(
        self,
        bs: int,
        base_offsets: dict[str, torch.Tensor],
        *,
        require_all_groups: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Refresh persistent base-offset buffers from per-step input.

        Radix groups whose key is missing retain the legacy zero default.
        Flat replay sets ``require_all_groups`` because its table/base export
        is atomic even for full-history groups. Returns the [:bs] views.
        """
        out = self._cuda_graph_base_views_by_bs.get(bs)
        if out is None:
            out = {
                group_id: buf[:bs]
                for group_id, buf in self._cuda_graph_paged_cache_base_offsets.items()
            }
            self._cuda_graph_base_views_by_bs[bs] = out
        for gid, buf in self._cuda_graph_paged_cache_base_offsets.items():
            src = base_offsets.get(gid)
            if src is None and require_all_groups:
                raise RuntimeError(
                    "DeepSeek V4 CUDA graph flat replay is missing captured "
                    f"base-offset group {gid!r}"
                )
            if src is None:
                if bs > 0:
                    # Missing radix bases retain the legacy logical-page zero.
                    buf[:bs].fill_(0)
                continue
            rows = int(src.shape[0])
            if rows > bs:
                raise RuntimeError(
                    "DeepSeek V4 CUDA-graph replay base-offsets row count "
                    f"{rows} exceeds bs={bs} for group {gid!r}"
                )
            if bs > 0:
                source = src[:rows]
                if not require_all_groups:
                    source = source.to(torch.int32)
                buf[:rows].copy_(source)
                if rows < bs:
                    buf[rows:bs].zero_()
        return out

    def _refresh_cuda_graph_flat_packed(
        self,
        source: CacheTableSource,
        *,
        bs: int,
        actual_bs: int,
    ) -> None:
        """Fused exact-width table/base refresh from one planned upload."""

        packed = source.packed
        if not isinstance(packed, FlatPackedTableOwnerSource):
            raise RuntimeError(
                "planned DeepSeek V4 flat replay requires owner-bound packed input"
            )
        destination = self._cuda_graph_flat_packed_storage
        if destination is None or self._cuda_graph_flat_max_cols <= 0:
            raise RuntimeError(
                "planned DeepSeek V4 flat replay has no packed graph destination"
            )
        cache_key = id(packed.unpack_meta)
        validated = self._cuda_graph_validated_packed_sources.get(cache_key)
        if (
            validated is None
            or validated[0] is not packed.unpack_meta
            or validated[1] is not packed.buffer
            or validated[2] is not source.tables
            or validated[3] is not source.base_offsets
            or validated[4] is not packed.group_ids
        ):
            # Descriptor, storage, and group schema are immutable per ring
            # slot. Dynamic rows and widths are checked below on every replay.
            self._validate_cuda_graph_flat_packed_source(source, packed)
            self._cuda_graph_validated_packed_sources[cache_key] = (
                packed.unpack_meta,
                packed.buffer,
                source.tables,
                source.base_offsets,
                packed.group_ids,
            )
        self._validate_cuda_graph_flat_packed_shape(
            source,
            packed,
            bs=bs,
            actual_bs=actual_bs,
        )
        flat_tables_unpack(
            packed.buffer,
            packed.unpack_meta,
            destination,
            bs,
            actual_bs=actual_bs,
            tail_pad=-1,
            max_cols=self._cuda_graph_flat_max_cols,
        )

    def _validate_cuda_graph_flat_packed_source(
        self,
        source: CacheTableSource,
        packed: FlatPackedTableOwnerSource,
    ) -> None:
        """Validate one immutable ring-slot layout on its first replay."""

        if packed.group_ids != self._cuda_graph_flat_group_ids:
            raise RuntimeError(
                "DeepSeek V4 packed replay groups disagree with graph capture: "
                f"source={list(packed.group_ids)}, "
                f"capture={list(self._cuda_graph_flat_group_ids)}"
            )
        if frozenset(source.tables) != frozenset(packed.group_ids):
            raise RuntimeError(
                "DeepSeek V4 packed replay source groups disagree with its header"
            )
        validate_flat_table_base_offsets(
            source.tables,
            source.base_offsets,
            where="DeepSeek V4 packed CUDA graph replay",
        )
        destination = self._cuda_graph_flat_packed_storage
        assert destination is not None
        if (
            not isinstance(packed.buffer, torch.Tensor)
            or packed.buffer.dtype != torch.int32
            or packed.buffer.dim() != 1
            or packed.buffer.device != destination.device
        ):
            raise RuntimeError(
                "DeepSeek V4 packed replay has an invalid packed source buffer"
            )
        meta = packed.unpack_meta
        if (
            not isinstance(meta, torch.Tensor)
            or meta.dtype != torch.int32
            or tuple(meta.shape) != (len(packed.group_ids), 6)
            or meta.device != destination.device
        ):
            raise RuntimeError(
                "DeepSeek V4 packed replay has an invalid owner unpack header"
            )
        source_storage = packed.buffer.untyped_storage().data_ptr()
        if meta.untyped_storage().data_ptr() != source_storage:
            raise RuntimeError(
                "DeepSeek V4 packed replay header escaped its planned upload storage"
            )
        for group_id in packed.group_ids:
            table = source.tables[group_id]
            base = source.base_offsets[group_id]
            if (
                table.untyped_storage().data_ptr() != source_storage
                or base.untyped_storage().data_ptr() != source_storage
            ):
                raise RuntimeError(
                    f"DeepSeek V4 packed replay group {group_id!r} escaped its "
                    "planned upload storage"
                )

    def _validate_cuda_graph_flat_packed_shape(
        self,
        source: CacheTableSource,
        packed: FlatPackedTableOwnerSource,
        *,
        bs: int,
        actual_bs: int,
    ) -> None:
        """Validate replay-varying rows and widths before the masked unpack."""

        source_rows = None
        for group_index, group_id in enumerate(packed.group_ids):
            table = source.tables[group_id]
            base = source.base_offsets[group_id]
            rows = int(table.shape[0])
            base_rows = int(base.shape[0])
            if rows != base_rows:
                raise RuntimeError(
                    "DeepSeek V4 packed replay table/base rows disagree for "
                    f"{group_id!r}: table={rows}, base={base_rows}"
                )
            if source_rows is None:
                source_rows = rows
            elif source_rows != rows:
                raise RuntimeError(
                    "DeepSeek V4 packed replay groups have different row counts"
                )
            source_cols = int(table.shape[1])
            capture_cols = self._cuda_graph_flat_capture_cols[group_index]
            if source_cols <= 0 or source_cols > capture_cols:
                raise RuntimeError(
                    "DeepSeek V4 packed replay table width exceeds capture for "
                    f"{group_id!r}: source={source_cols}, capture={capture_cols}"
                )
        if source_rows is None or source_rows < actual_bs or source_rows > bs:
            raise RuntimeError(
                "DeepSeek V4 packed replay source rows do not cover the live "
                f"batch: source={source_rows}, actual={actual_bs}, padded={bs}"
            )

    def _cuda_graph_tokens_per_req(
        self,
        bs: int,
        num_tokens: int,
        forward_mode: ForwardMode | None,
    ) -> int:
        if num_tokens != bs:
            if bs == 0:
                return self._cuda_graph_max_tokens_per_req
            if num_tokens % bs != 0:
                raise RuntimeError(
                    "DeepSeek V4 packed CUDA graph metadata expects uniformly "
                    f"packed tokens per request, got num_tokens={num_tokens}, "
                    f"bs={bs}"
                )
            tokens_per_req = num_tokens // bs
            if tokens_per_req > self._cuda_graph_max_tokens_per_req:
                raise RuntimeError(
                    "DeepSeek V4 packed CUDA graph metadata was initialized "
                    f"for at most {self._cuda_graph_max_tokens_per_req} tokens "
                    f"per request, got {tokens_per_req}"
                )
            return max(1, tokens_per_req)
        return 1

    def _refresh_cuda_graph_packed_metadata(
        self,
        *,
        bs: int,
        actual_bs: int,
        tokens_per_req: int,
    ) -> int:
        total_tokens = bs * tokens_per_req
        actual_tokens = actual_bs * tokens_per_req
        query_start = self._cuda_graph_query_start_by_tokens_per_req.get(tokens_per_req)
        token_to_req = self._cuda_graph_token_to_req_by_tokens_per_req.get(
            tokens_per_req
        )
        if query_start is None or token_to_req is None:
            raise RuntimeError(
                "DeepSeek V4 CUDA graph packed metadata was not precomputed "
                f"for tokens_per_req={tokens_per_req}"
            )
        self._cuda_graph_query_lens[:bs].fill_(tokens_per_req)
        self._cuda_graph_query_start_loc[: bs + 1].copy_(query_start[: bs + 1])
        self._cuda_graph_token_to_req[:total_tokens].copy_(token_to_req[:total_tokens])
        self._cuda_graph_is_valid_token[:actual_tokens].fill_(True)
        if actual_tokens < total_tokens:
            self._cuda_graph_is_valid_token[actual_tokens:total_tokens].fill_(False)
        return total_tokens

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        **kwargs,
    ):
        flat_cache_group_ids = tuple(kwargs.pop("flat_cache_group_ids", ()))
        table_source = self._pop_cache_table_source(kwargs)
        planned_capture_cols = getattr(self, "_flat_capture_cols_by_group", None)
        if flat_cache_group_ids and planned_capture_cols is not None:
            captured_group_ids = {str(group_id) for group_id in flat_cache_group_ids}
            if captured_group_ids != set(planned_capture_cols):
                raise RuntimeError(
                    "DeepSeek V4 CUDA graph capture groups disagree with flat "
                    f"plan: capture={sorted(captured_group_ids)}, "
                    f"plan={sorted(planned_capture_cols)}"
                )
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
        tokens_per_req = self._cuda_graph_tokens_per_req(bs, num_tokens, forward_mode)
        is_packed_decode = (
            forward_mode is not None and forward_mode.is_decode() and num_tokens != bs
        )
        total_tokens = self._refresh_cuda_graph_packed_metadata(
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
        self._cuda_graph_req_pool_indices[:bs].copy_(req_pool_indices[:bs])
        self._cuda_graph_seq_lens[:bs].copy_(capture_seq_lens)
        metadata_forward_mode = forward_mode
        is_decode = (
            metadata_forward_mode is not None and metadata_forward_mode.is_decode()
        )
        tables_on_device, offsets_on_device = self._place_cache_tables(
            table_source,
            rows=bs,
            device=self.device,
        )
        metadata_paged = self._refresh_cuda_graph_paged_cache_block_tables(
            bs,
            tables_on_device,
            pad_value=0,
            # Capture allocates persistent buffers from group ids only. Live
            # scheduler tables do not exist until replay.
            require_all_groups=False,
        )
        metadata_base_offsets = self._refresh_cuda_graph_base_offsets(
            bs,
            offsets_on_device,
            require_all_groups=False,
        )
        (
            swa_block_table,
            compressor_state_block_tables,
            indexer_state_block_table,
            swa_base,
            compressor_state_base,
            indexer_state_base,
        ) = _split_paged_cache_block_tables_into_v4_metadata(
            metadata_paged,
            metadata_base_offsets,
        )
        prior_metadata = self._cuda_graph_metadata.get(bs)
        prior_slot_mappings = (
            prior_metadata.cache.decode_compressed_slot_mappings
            if prior_metadata is not None
            else {}
        )
        prior_slot_capacity_pages = (
            prior_metadata.cache.decode_compressed_capacity_pages
            if prior_metadata is not None
            else {}
        )
        cache_metadata = DeepseekV4CacheMetadata(
            page_size=self.page_size,
            block_table=self._cuda_graph_block_table[:bs, : self.max_num_pages],
            paged_cache_block_tables=metadata_paged,
            paged_cache_block_table_base_offsets=metadata_base_offsets,
            swa_block_table=swa_block_table,
            swa_base_logical_page=swa_base,
            compressor_state_block_tables=compressor_state_block_tables,
            compressor_state_base_logical_pages=compressor_state_base,
            indexer_state_block_table=indexer_state_block_table,
            indexer_state_base_logical_page=indexer_state_base,
            decode_compressed_slot_mappings=prior_slot_mappings,
            decode_compressed_capacity_pages=prior_slot_capacity_pages,
        )
        metadata = prior_metadata
        if metadata is None:
            metadata = DeepseekV4ForwardMetadata(
                req_pool_indices=self._cuda_graph_req_pool_indices[:bs],
                seq_lens=self._cuda_graph_seq_lens[:bs],
                query_lens=self._cuda_graph_query_lens[:bs],
                query_start_loc=self._cuda_graph_query_start_loc[: bs + 1],
                token_to_req_indices=self._cuda_graph_token_to_req[:total_tokens],
                cache=cache_metadata,
                is_valid_token=self._cuda_graph_is_valid_token[:total_tokens],
                seq_lens_cpu=None,
                query_lens_cpu=None,
                forward_mode=metadata_forward_mode,
            )
        else:
            metadata.req_pool_indices = self._cuda_graph_req_pool_indices[:bs]
            metadata.seq_lens = self._cuda_graph_seq_lens[:bs]
            metadata.query_lens = self._cuda_graph_query_lens[:bs]
            metadata.query_start_loc = self._cuda_graph_query_start_loc[: bs + 1]
            metadata.token_to_req_indices = self._cuda_graph_token_to_req[:total_tokens]
            metadata.cache = cache_metadata
            metadata.is_valid_token = self._cuda_graph_is_valid_token[:total_tokens]
            metadata.seq_lens_cpu = None
            metadata.query_lens_cpu = None
            metadata.forward_mode = metadata_forward_mode
        self._cuda_graph_metadata[bs] = metadata
        if is_packed_decode and getattr(self, "is_draft", False):
            self._prepare_draft_decode_metadata(
                metadata,
                self._cuda_graph_seq_lens[:bs],
            )
        if is_decode:
            self.forward_decode_metadata = metadata
        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        **kwargs,
    ):
        table_source = self._pop_cache_table_source(
            kwargs,
            require_source=self._uses_flat_cache_tables,
        )
        actual_bs = max(0, min(int(kwargs.pop("actual_bs", bs)), bs))
        num_tokens_arg = kwargs.pop("num_tokens", None)
        del kwargs
        if forward_mode is not None and not forward_mode.is_decode_or_idle():
            raise NotImplementedError(
                f"DeepSeek V4 CUDA graph replay not supported for {forward_mode}"
            )
        if num_tokens_arg is None:
            num_tokens = bs
        else:
            num_tokens = int(num_tokens_arg)
        tokens_per_req = self._cuda_graph_tokens_per_req(bs, num_tokens, forward_mode)
        is_packed_decode = (
            forward_mode is not None and forward_mode.is_decode() and num_tokens != bs
        )
        total_tokens = self._refresh_cuda_graph_packed_metadata(
            bs=bs,
            actual_bs=actual_bs,
            tokens_per_req=tokens_per_req,
        )
        metadata = self._cuda_graph_metadata[bs]
        self._cuda_graph_req_pool_indices[:bs].copy_(req_pool_indices[:bs])
        self._cuda_graph_seq_lens[:bs].copy_(seq_lens[:bs].to(torch.int32))
        if req_to_page is not None and not self._uses_flat_cache_tables:
            self._cuda_graph_block_table[:bs, : self.max_num_pages].copy_(
                req_to_page[req_pool_indices[:bs], : self.max_num_pages]
            )
        tables_on_device, offsets_on_device = self._place_cache_tables(
            table_source,
            rows=bs,
            device=self.device,
        )
        if self._uses_flat_cache_tables and table_source.packed is not None:
            self._refresh_cuda_graph_flat_packed(
                table_source,
                bs=bs,
                actual_bs=actual_bs,
            )
        else:
            self._refresh_cuda_graph_paged_cache_block_tables(
                bs,
                tables_on_device,
                pad_value=-1,
                require_all_groups=self._uses_flat_cache_tables,
            )
            self._refresh_cuda_graph_base_offsets(
                bs,
                offsets_on_device,
                require_all_groups=self._uses_flat_cache_tables,
            )
        metadata_forward_mode = forward_mode
        is_decode = (
            metadata_forward_mode is not None and metadata_forward_mode.is_decode()
        )
        metadata.forward_mode = metadata_forward_mode
        metadata.token_to_req_indices = self._cuda_graph_token_to_req[:total_tokens]
        metadata.is_valid_token = self._cuda_graph_is_valid_token[:total_tokens]
        # Capture bound cache metadata to the persistent table/base views for
        # this batch size. Replay only refreshes their contents above; replacing
        # the mappings or cache dataclass here would add Python allocations to
        # every decode without changing any pointer consumed by the graph.
        metadata.num_prefill_reqs = 0
        metadata.num_prefill_tokens = 0
        if is_packed_decode and getattr(self, "is_draft", False):
            self._prepare_draft_decode_metadata(
                metadata,
                self._cuda_graph_seq_lens[:bs],
            )
        if (
            metadata_forward_mode is not None
            and metadata_forward_mode.is_decode()
            and self._decode_swa_window_size > 0
            and self._decode_swa_block_size > 0
        ):
            self._update_decode_swa_metadata(
                metadata,
                window_size=self._decode_swa_window_size,
                block_size=self._decode_swa_block_size,
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
        if is_decode:
            self.forward_decode_metadata = metadata
        self.forward_metadata = metadata

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor | None = None):
        if (
            self._draft_decode_base_seq_lens is None
            or self.forward_prefill_metadata is None
            or self._draft_decode_metadata is None
        ):
            raise RuntimeError("DeepSeek V4 draft metadata was not initialized")
        self._draft_decode_step += 1
        metadata = self._draft_decode_metadata
        if seq_lens is None:
            metadata.seq_lens.add_(1)
        else:
            metadata.seq_lens.copy_(seq_lens[: metadata.seq_lens.numel()])
        metadata.forward_mode = ForwardMode.DECODE
        if self._decode_swa_window_size > 0 and self._decode_swa_block_size > 0:
            self._update_decode_swa_metadata(
                metadata,
                window_size=self._decode_swa_window_size,
                block_size=self._decode_swa_block_size,
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
        self.forward_decode_metadata = metadata
        self.forward_metadata = metadata
        self._decode_tile_metadata = {}

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError("DeepSeek V4 uses the model-local attention forward")

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError("DeepSeek V4 uses the model-local attention forward")


register_backend("deepseek_v4", {AttentionArch.MLA}, DeepseekV4AttentionBackend)
