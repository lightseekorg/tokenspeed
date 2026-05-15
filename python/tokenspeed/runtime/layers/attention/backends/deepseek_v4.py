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

import torch
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    deepseek_v4_indexer_decode_metadata_compute,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    DEEPSEEK_V4_SPARSE_PREFILL_TOPK_ALIGNMENT,
    DEEPSEEK_V4_SWA_SCALE_DIM,
    DEEPSEEK_V4_SWA_TOKEN_STRIDE,
    DeepseekV4AttentionOpUnavailable,
    deepseek_v4_build_dense_prefill_local_compressed_indices,
    deepseek_v4_combine_dense_swa_indices,
    deepseek_v4_combine_topk_swa_indices,
    deepseek_v4_compute_global_topk_indices_and_lens,
    deepseek_v4_decode_swa_indices_and_lens,
    deepseek_v4_dequantize_and_gather_k_cache,
    deepseek_v4_profile_scope,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
    DeepseekV4ForwardMetadata,
    _split_paged_cache_block_tables_into_v4_metadata,
)
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.utils.env import global_server_args_dict

DEEPSEEK_V4_DEFAULT_PREFILL_CHUNK_SIZE = 4


def _swa_block_table(metadata: DeepseekV4ForwardMetadata) -> torch.Tensor:
    return (
        metadata.swa_block_table
        if metadata.swa_block_table is not None
        else metadata.block_table
    )


def _cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.pad(
        torch.cumsum(lengths.to(torch.int32), dim=0, dtype=torch.int32),
        (1, 0),
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
    cache = metadata.decode_indexer_plan_cache
    if not cache:
        return
    refreshed_keys = metadata.decode_indexer_plan_refreshed_keys
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
        block_table = metadata.compressed_block_table(
            compress_ratio,
            cache_block_size,
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
    if not metadata.decode_indexer_schedule_metadata:
        return
    try:
        from tokenspeed_kernel.thirdparty import deep_gemm
    except Exception:
        return
    get_metadata = getattr(deep_gemm, "get_paged_mqa_logits_metadata", None)
    if get_metadata is None:
        return
    for (
        compress_ratio,
        cache_block_size,
        num_tokens,
    ), schedule_metadata in list(metadata.decode_indexer_schedule_metadata.items()):
        if num_tokens <= 0:
            continue
        key = (compress_ratio, cache_block_size, num_tokens)
        decode_plan = metadata.decode_indexer_plan_cache.get(key)
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
            metadata.decode_indexer_schedule_metadata[key] = refreshed


class DeepseekV4AttentionBackend(AttentionBackend):
    """Metadata owner for the model-local DeepSeek V4 attention path."""

    uses_paged_cache_groups = True
    uses_padded_decode_token_mask = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.page_size = config.page_size
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
            (self.context_len + self.page_size - 1) // self.page_size,
        )
        self.forward_metadata: DeepseekV4ForwardMetadata | None = None
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
        self._prefill_dense_compressed_indices_buffer: Optional[torch.Tensor] = None
        self._decode_swa_window_size = 0
        self._decode_swa_block_size = 0

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
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode | None,
        num_extends: int,
        extend_seq_lens_cpu: torch.Tensor | None,
        extend_prefix_lens_cpu: torch.Tensor | None,
        extend_prefix_lens: torch.Tensor | None,
    ) -> torch.Tensor:
        if forward_mode is not None and forward_mode.is_decode_or_idle():
            return torch.ones(bs, dtype=torch.int32, device=seq_lens.device)
        if forward_mode is not None and forward_mode.is_mixed():
            lens = torch.ones(bs, dtype=torch.int32, device=seq_lens.device)
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
        forward_mode: Optional[ForwardMode],
        num_extends: int,
        extend_seq_lens_cpu: Optional[torch.Tensor],
        extend_prefix_lens_cpu: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if forward_mode is not None and forward_mode.is_decode_or_idle():
            return torch.ones(bs, dtype=torch.int32)
        if forward_mode is not None and forward_mode.is_mixed():
            lens = torch.ones(bs, dtype=torch.int32)
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

    def init_forward_metadata(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        paged_cache_block_tables = kwargs.pop("paged_cache_block_tables", None) or {}
        paged_cache_block_table_base_offsets = (
            kwargs.pop("paged_cache_block_table_base_offsets", None) or {}
        )
        num_extends_arg = kwargs.pop("num_extends", None)
        num_extends = bs if num_extends_arg is None else int(num_extends_arg)
        del num_tokens, kwargs
        device = seq_lens.device
        req_pool_indices = req_pool_indices[:bs]
        seq_lens = seq_lens[:bs].to(torch.int32)
        query_lens = self._query_lens(
            bs,
            seq_lens,
            forward_mode,
            num_extends,
            extend_seq_lens_cpu,
            extend_prefix_lens_cpu,
            extend_prefix_lens,
        )
        if forward_mode is not None and forward_mode.is_mixed():
            num_prefill_reqs = max(0, min(num_extends, bs))
        elif forward_mode is not None and forward_mode.is_extend():
            num_prefill_reqs = bs
        else:
            num_prefill_reqs = 0
        query_lens_cpu = self._query_lens_cpu(
            bs,
            forward_mode,
            num_extends,
            extend_seq_lens_cpu,
            extend_prefix_lens_cpu,
        )
        seq_lens_cpu = None
        if extend_prefix_lens_cpu is not None and query_lens_cpu is not None:
            seq_lens_cpu = seq_lens[:bs].to(dtype=torch.int32, device="cpu")
            prefix_count = min(
                int(extend_prefix_lens_cpu.numel()),
                (
                    num_prefill_reqs
                    if forward_mode is not None and forward_mode.is_mixed()
                    else bs
                ),
            )
            if prefix_count:
                seq_lens_cpu[:prefix_count] = (
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
            elif forward_mode.is_mixed():
                seq_lens_cpu = seq_lens[:bs].to(dtype=torch.int32, device="cpu")
        max_seq_len = int(seq_lens.max().item()) if bs else 0
        max_pages = (max_seq_len + self.page_size - 1) // self.page_size
        if req_to_page is None:
            block_table = torch.zeros(
                (bs, max(max_pages, 1)),
                dtype=torch.int32,
                device=device,
            )
        else:
            block_table = req_to_page[req_pool_indices, : max(max_pages, 1)]
        paged_cache_block_tables = {
            str(gid): table[:bs].to(device=device, dtype=torch.int32)
            for gid, table in paged_cache_block_tables.items()
        }
        base_offsets_on_device: dict[str, torch.Tensor] = {}
        for gid, off in paged_cache_block_table_base_offsets.items():
            if not isinstance(off, torch.Tensor):
                raise TypeError(
                    "DeepSeek V4 paged_cache_block_table_base_offsets values "
                    f"must be torch.Tensor, got {type(off).__name__} for "
                    f"group_id={gid!r}"
                )
            base_offsets_on_device[str(gid)] = off[:bs].to(
                device=device, dtype=torch.int32
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
        num_prefill_tokens = (
            int(query_lens[:num_prefill_reqs].sum().item()) if num_prefill_reqs else 0
        )
        self.forward_metadata = DeepseekV4ForwardMetadata(
            page_size=self.page_size,
            req_pool_indices=req_pool_indices,
            block_table=block_table,
            seq_lens=seq_lens,
            query_lens=query_lens,
            query_start_loc=_cu_seqlens(query_lens),
            token_to_req_indices=token_to_req,
            seq_lens_cpu=seq_lens_cpu,
            query_lens_cpu=query_lens_cpu,
            num_prefill_reqs=num_prefill_reqs,
            num_prefill_tokens=num_prefill_tokens,
            forward_mode=forward_mode,
            paged_cache_block_tables=paged_cache_block_tables,
            paged_cache_block_table_base_offsets=base_offsets_on_device,
            swa_block_table=swa_block_table,
            swa_base_logical_page=swa_base,
            compressor_state_block_tables=compressor_state_block_tables,
            compressor_state_base_logical_pages=compressor_state_base,
            indexer_state_block_table=indexer_state_block_table,
            indexer_state_base_logical_page=indexer_state_base,
        )
        self._decode_tile_metadata = {}

    def _update_decode_swa_metadata(
        self,
        metadata: DeepseekV4ForwardMetadata,
        *,
        window_size: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = metadata.token_to_req_indices.shape[0]
        needs_alloc = (
            metadata.decode_swa_indices is None
            or metadata.decode_swa_lens is None
            or metadata.decode_swa_indices.shape != (num_tokens, window_size)
            or metadata.decode_swa_lens.shape != (num_tokens,)
            or metadata.decode_swa_indices.device != metadata.seq_lens.device
        )
        if needs_alloc:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "DeepSeek V4 decode SWA metadata must be allocated before "
                    "CUDA graph capture"
                )
            with torch.inference_mode(False):
                metadata.decode_swa_indices = torch.empty(
                    (num_tokens, window_size),
                    dtype=torch.int32,
                    device=metadata.seq_lens.device,
                )
                metadata.decode_swa_lens = torch.empty(
                    (num_tokens,),
                    dtype=torch.int32,
                    device=metadata.seq_lens.device,
                )

        indices, lens = deepseek_v4_decode_swa_indices_and_lens(
            query_start_loc=metadata.query_start_loc,
            seq_lens=metadata.seq_lens,
            token_to_req_indices=metadata.token_to_req_indices,
            block_table=_swa_block_table(metadata),
            block_table_base_offsets=metadata.swa_base_logical_page,
            window_size=window_size,
            block_size=block_size,
            is_valid_token=metadata.is_valid_token,
            out_indices=metadata.decode_swa_indices,
            out_lens=metadata.decode_swa_lens,
        )
        metadata.decode_swa_indices = indices
        metadata.decode_swa_lens = lens
        metadata.decode_swa_window_size = window_size
        metadata.decode_swa_block_size = block_size
        self._decode_swa_window_size = window_size
        self._decode_swa_block_size = block_size
        return indices, lens

    def _get_decode_swa_metadata(
        self,
        metadata: DeepseekV4ForwardMetadata,
        *,
        positions: torch.Tensor,
        window_size: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            metadata.decode_swa_indices is not None
            and metadata.decode_swa_lens is not None
            and metadata.decode_swa_window_size == window_size
            and metadata.decode_swa_block_size == block_size
            and metadata.decode_swa_indices.shape[0] == positions.numel()
        ):
            return metadata.decode_swa_indices, metadata.decode_swa_lens
        return self._update_decode_swa_metadata(
            metadata,
            window_size=window_size,
            block_size=block_size,
        )

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
        block_table = metadata.compressed_block_table(compress_ratio, block_size)
        is_valid_token = (
            metadata.is_valid_token[:num_tokens]
            if metadata.is_valid_token is not None
            else None
        )
        capturing = positions.is_cuda and torch.cuda.is_current_stream_capturing()
        if compress_ratio == 4:
            if topk_indices is None:
                raise RuntimeError("DeepSeek V4 CSA decode requires top-k indices")
            indices_2d, lens = deepseek_v4_compute_global_topk_indices_and_lens(
                topk_indices=topk_indices,
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
        dense_indices_cache = metadata.decode_dense_compressed_indices_cache
        capture_safe_keys = metadata.decode_dense_compressed_indices_capture_safe_keys
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
        page_offsets = safe_local % block_size
        page_ids = metadata.safe_page_ids(block_table, req_idx[:, None], pages.long())
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
        try:
            from flash_mla import get_mla_metadata
        except Exception as exc:
            raise DeepseekV4AttentionOpUnavailable(
                "DeepSeek V4 decode requires FlashMLA latent attention. "
                "Build/install `tokenspeed-kernel/python` with FlashMLA."
            ) from exc
        tile_metadata = get_mla_metadata()[0]
        self._decode_tile_metadata[(phase, kind, bs)] = tile_metadata
        return tile_metadata

    def _pad_query(self, q: torch.Tensor, padded_heads: int) -> torch.Tensor:
        if q.shape[1] == padded_heads:
            return q
        q_padded = torch.zeros(
            (q.shape[0], padded_heads, q.shape[2]),
            dtype=q.dtype,
            device=q.device,
        )
        q_padded[:, : q.shape[1]].copy_(q)
        return q_padded

    def _fp8_ds_mla_cache_view(
        self,
        cache_2d: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        row_bytes = DEEPSEEK_V4_SWA_TOKEN_STRIDE + DEEPSEEK_V4_SWA_SCALE_DIM
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
        metadata = self.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 decode requires forward metadata")
        if metadata.forward_mode is None or not metadata.forward_mode.is_decode():
            raise RuntimeError(
                "forward_deepseek_v4_decode only supports ForwardMode.DECODE"
            )
        try:
            from flash_mla import flash_mla_with_kvcache
        except Exception as exc:
            raise DeepseekV4AttentionOpUnavailable(
                "DeepSeek V4 decode requires FlashMLA latent attention. "
                "Build/install `tokenspeed-kernel/python` with FlashMLA."
            ) from exc

        q_padded = self._pad_query(q, padded_heads).contiguous()
        swa_indices, swa_lens = self._get_decode_swa_metadata(
            metadata,
            positions=positions,
            window_size=window_size,
            block_size=token_to_kv_pool.swa_block_size,
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
            token_to_kv_pool.swa_block_size,
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
        topk_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        metadata = self.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 mixed attention requires forward metadata")
        if metadata.forward_mode is None or not metadata.forward_mode.is_mixed():
            raise RuntimeError(
                "forward_deepseek_v4_mixed only supports ForwardMode.MIXED"
            )

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
                with deepseek_v4_profile_scope(f"attn_{kind}_mixed_prefill_copy"):
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
                with deepseek_v4_profile_scope(f"attn_{kind}_mixed_decode_copy"):
                    out[num_prefill_tokens:decode_end].copy_(decode_out)
        finally:
            self.forward_metadata = saved_metadata
        return out

    def _prefill_gather_lens(
        self,
        *,
        window_size: int,
    ) -> torch.Tensor:
        metadata = self.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 prefill requires forward metadata")
        prefix_lens = metadata.seq_lens - metadata.query_lens
        return metadata.query_lens + torch.minimum(
            prefix_lens,
            torch.full_like(prefix_lens, max(window_size - 1, 0)),
        )

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
        num_reqs = metadata.seq_lens.numel()
        gather_lens = self._prefill_gather_lens(window_size=window_size)
        max_gather_len = int(gather_lens.max().item()) if num_reqs else 1
        compressed_lens = (
            torch.div(metadata.seq_lens, compress_ratio, rounding_mode="floor")
            if compress_ratio > 1
            else torch.zeros_like(metadata.seq_lens)
        )
        compressed_base = (
            int(compressed_lens.max().item()) if compress_ratio > 1 and num_reqs else 0
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
            compressed_block_table = metadata.compressed_block_table(
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
                block_table=_swa_block_table(metadata),
                block_table_base_offsets=metadata.swa_base_logical_page,
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
            compressed_block_table = metadata.compressed_block_table(
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
            block_table=_swa_block_table(metadata),
            block_table_base_offsets=metadata.swa_base_logical_page,
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
        paged_cache_block_tables = {
            key: table[req_start:req_end]
            for key, table in metadata.paged_cache_block_tables.items()
        }
        paged_cache_block_table_base_offsets = {
            key: offsets[req_start:req_end]
            for key, offsets in metadata.paged_cache_block_table_base_offsets.items()
        }
        compressor_state_block_tables = {
            key: table[req_start:req_end]
            for key, table in metadata.compressor_state_block_tables.items()
        }
        compressor_state_base_logical_pages = {
            key: offsets[req_start:req_end]
            for key, offsets in metadata.compressor_state_base_logical_pages.items()
        }
        req_count = max(0, req_end - req_start)
        token_count = max(0, token_end - token_start)
        num_prefill_reqs = req_count if forward_mode.is_extend() else 0
        num_prefill_tokens = token_count if forward_mode.is_extend() else 0
        return DeepseekV4ForwardMetadata(
            page_size=metadata.page_size,
            req_pool_indices=metadata.req_pool_indices[req_start:req_end],
            block_table=metadata.block_table[req_start:req_end],
            seq_lens=metadata.seq_lens[req_start:req_end],
            query_lens=metadata.query_lens[req_start:req_end],
            query_start_loc=_cu_seqlens(metadata.query_lens[req_start:req_end]),
            token_to_req_indices=token_to_req,
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
            paged_cache_block_tables=paged_cache_block_tables,
            paged_cache_block_table_base_offsets=paged_cache_block_table_base_offsets,
            swa_block_table=(
                metadata.swa_block_table[req_start:req_end]
                if metadata.swa_block_table is not None
                else None
            ),
            swa_base_logical_page=(
                metadata.swa_base_logical_page[req_start:req_end]
                if metadata.swa_base_logical_page is not None
                else None
            ),
            compressor_state_block_tables=compressor_state_block_tables,
            compressor_state_base_logical_pages=compressor_state_base_logical_pages,
            indexer_state_block_table=(
                metadata.indexer_state_block_table[req_start:req_end]
                if metadata.indexer_state_block_table is not None
                else None
            ),
            indexer_state_base_logical_page=(
                metadata.indexer_state_base_logical_page[req_start:req_end]
                if metadata.indexer_state_base_logical_page is not None
                else None
            ),
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
        if metadata.forward_mode is None or not metadata.forward_mode.is_extend():
            raise RuntimeError(
                "forward_deepseek_v4_prefill only supports extend/prefill modes"
            )
        try:
            from flash_mla import flash_mla_sparse_fwd
        except Exception as exc:
            raise DeepseekV4AttentionOpUnavailable(
                "DeepSeek V4 prefill requires FlashMLA sparse attention. "
                "Build/install `tokenspeed-kernel/python` with FlashMLA."
            ) from exc

        with deepseek_v4_profile_scope(f"attn_{kind}_prefill_pad_q"):
            q_padded = self._pad_query(q, padded_heads).contiguous()
        with deepseek_v4_profile_scope(f"attn_{kind}_prefill_workspace"):
            kv_workspace, indices, lens = self._prefill_workspace(
                positions=positions,
                token_to_kv_pool=token_to_kv_pool,
                layer_id=layer_id,
                compress_ratio=compress_ratio,
                window_size=window_size,
                head_dim=head_dim,
                topk_indices=topk_indices,
            )
        with deepseek_v4_profile_scope(f"attn_{kind}_prefill_flashmla"):
            out, _, _ = flash_mla_sparse_fwd(
                q=q_padded,
                kv=kv_workspace.view(-1, 1, head_dim),
                indices=indices.unsqueeze(1),
                sm_scale=softmax_scale,
                attn_sink=attn_sink,
                topk_length=lens,
            )
        return out[:, :num_local_heads]

    def _prefill_chunk_token_offsets(
        self,
        metadata: DeepseekV4ForwardMetadata,
        num_reqs: int,
    ) -> list[int]:
        return [
            int(x)
            for x in metadata.query_start_loc[: num_reqs + 1].detach().cpu().tolist()
        ]

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
        topk_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        metadata = self.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 prefill requires forward metadata")
        if metadata.forward_mode is None or not metadata.forward_mode.is_extend():
            raise RuntimeError(
                "forward_deepseek_v4_prefill only supports extend/prefill modes"
            )

        num_reqs = int(metadata.seq_lens.numel())
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

        token_offsets = self._prefill_chunk_token_offsets(metadata, num_reqs)
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
    ):
        del seq_lens_buf
        self._cuda_graph_block_table = torch.zeros(
            (max_bs, self.max_num_pages),
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
            max_bs,
            dtype=torch.int32,
            device=self.device,
        )
        self._cuda_graph_max_bs = max_bs
        self._cuda_graph_paged_cache_block_tables = {}
        self._cuda_graph_paged_cache_base_offsets = {}
        for spec in tuple(paged_cache_group_specs or ()):
            raw_per_page = max(
                1,
                int(spec.rows_per_page) * int(spec.entry_stride_tokens),
            )
            gid = str(spec.group_id)
            sliding = str(getattr(spec, "retention", "")) == "sliding_window"
            if sliding:
                window = int(getattr(spec, "sliding_window_tokens", 0) or 0)
                live_tokens = max(1, window - 1 + max(1, int(max_tokens_per_req)))
                if self.context_len > 0:
                    live_tokens = min(live_tokens, self.context_len)
                max_pages = max(1, (live_tokens + raw_per_page - 1) // raw_per_page + 1)
            else:
                max_pages = max(
                    1,
                    (self.context_len + raw_per_page - 1) // raw_per_page,
                )
            self._cuda_graph_paged_cache_block_tables[gid] = torch.zeros(
                (max_bs, max_pages),
                dtype=torch.int32,
                device=self.device,
            )
            if sliding:
                self._cuda_graph_paged_cache_base_offsets[gid] = torch.zeros(
                    (max_bs,),
                    dtype=torch.int32,
                    device=self.device,
                )
        self._cuda_graph_is_valid_token = torch.ones(
            max_bs,
            dtype=torch.bool,
            device=self.device,
        )

    def _refresh_cuda_graph_paged_cache_block_tables(
        self,
        bs: int,
        paged_cache_block_tables: dict[str, torch.Tensor],
        *,
        pad_value: int,
        paged_cache_block_table_base_offsets: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if not self._cuda_graph_paged_cache_block_tables:
            return out
        for group_id, buf in self._cuda_graph_paged_cache_block_tables.items():
            table = paged_cache_block_tables.get(group_id)
            buf[:bs].fill_(pad_value)
            if table is not None:
                if int(table.shape[0]) != bs:
                    raise RuntimeError(
                        "DeepSeek V4 CUDA graph paged cache table row count "
                        f"mismatch for {group_id!r}: got {int(table.shape[0])}, "
                        f"expected padded bs {bs}"
                    )
                cols = int(table.shape[1])
                if cols > int(buf.shape[1]):
                    raise RuntimeError(
                        "DeepSeek V4 CUDA graph paged cache table width "
                        f"mismatch for {group_id!r}: got {cols}, capture "
                        f"buffer has {int(buf.shape[1])}"
                    )
                if cols > 0:
                    buf[:bs, :cols].copy_(table[:bs, :cols].to(torch.int32))
            out[group_id] = buf[:bs]
        return out

    def _refresh_cuda_graph_base_offsets(
        self,
        bs: int,
        base_offsets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Refresh persistent base-offset buffers from per-step input.

        Sliding groups whose key is missing fall back to 0 (legacy
        absolute scheduler binding). Returns the [:bs] views keyed by gid.
        """
        out: dict[str, torch.Tensor] = {}
        for gid, buf in self._cuda_graph_paged_cache_base_offsets.items():
            buf[:bs].fill_(0)
            src = base_offsets.get(gid)
            if src is not None and bs > 0:
                rows = int(src.shape[0])
                if rows < bs:
                    raise RuntimeError(
                        "DeepSeek V4 CUDA-graph replay base-offsets row count "
                        f"{rows} < bs={bs} for group {gid!r}"
                    )
                buf[:bs].copy_(src[:bs].to(torch.int32))
            out[gid] = buf[:bs]
        return out

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        **kwargs,
    ):
        paged_cache_block_tables = kwargs.pop("paged_cache_block_tables", None) or {}
        paged_cache_block_table_base_offsets = (
            kwargs.pop("paged_cache_block_table_base_offsets", None) or {}
        )
        del num_tokens, kwargs
        if forward_mode is not None and not forward_mode.is_decode_or_idle():
            raise NotImplementedError(
                f"DeepSeek V4 CUDA graph capture not supported for {forward_mode}"
            )
        self._cuda_graph_req_pool_indices[:bs].copy_(req_pool_indices[:bs])
        self._cuda_graph_seq_lens[:bs].copy_(seq_lens[:bs].to(torch.int32))
        self._cuda_graph_query_lens[:bs].fill_(1)
        self._cuda_graph_query_start_loc[: bs + 1].copy_(
            torch.arange(bs + 1, dtype=torch.int32, device=self.device)
        )
        self._cuda_graph_token_to_req[:bs].copy_(
            torch.arange(bs, dtype=torch.int32, device=self.device)
        )
        offsets_on_device = {
            str(gid): off.to(device=self.device, dtype=torch.int32)
            for gid, off in paged_cache_block_table_base_offsets.items()
        }
        metadata_paged = self._refresh_cuda_graph_paged_cache_block_tables(
            bs,
            {
                str(group_id): table.to(device=self.device, dtype=torch.int32)
                for group_id, table in paged_cache_block_tables.items()
            },
            pad_value=0,
            paged_cache_block_table_base_offsets=offsets_on_device,
        )
        metadata_base_offsets = self._refresh_cuda_graph_base_offsets(
            bs,
            offsets_on_device,
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
        metadata = DeepseekV4ForwardMetadata(
            page_size=self.page_size,
            req_pool_indices=self._cuda_graph_req_pool_indices[:bs],
            block_table=self._cuda_graph_block_table[:bs, : self.max_num_pages],
            seq_lens=self._cuda_graph_seq_lens[:bs],
            query_lens=self._cuda_graph_query_lens[:bs],
            query_start_loc=self._cuda_graph_query_start_loc[: bs + 1],
            token_to_req_indices=self._cuda_graph_token_to_req[:bs],
            is_valid_token=self._cuda_graph_is_valid_token[:bs],
            seq_lens_cpu=None,
            query_lens_cpu=None,
            forward_mode=forward_mode,
            paged_cache_block_tables=metadata_paged,
            paged_cache_block_table_base_offsets=metadata_base_offsets,
            swa_block_table=swa_block_table,
            swa_base_logical_page=swa_base,
            compressor_state_block_tables=compressor_state_block_tables,
            compressor_state_base_logical_pages=compressor_state_base,
            indexer_state_block_table=indexer_state_block_table,
            indexer_state_base_logical_page=indexer_state_base,
        )
        self._cuda_graph_metadata[bs] = metadata
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
        paged_cache_block_tables = kwargs.pop("paged_cache_block_tables", None) or {}
        paged_cache_block_table_base_offsets = (
            kwargs.pop("paged_cache_block_table_base_offsets", None) or {}
        )
        actual_bs = max(0, min(int(kwargs.pop("actual_bs", bs)), bs))
        del kwargs
        if forward_mode is not None and not forward_mode.is_decode_or_idle():
            raise NotImplementedError(
                f"DeepSeek V4 CUDA graph replay not supported for {forward_mode}"
            )
        metadata = self._cuda_graph_metadata[bs]
        self._cuda_graph_req_pool_indices[:bs].copy_(req_pool_indices[:bs])
        self._cuda_graph_seq_lens[:bs].copy_(seq_lens[:bs].to(torch.int32))
        self._cuda_graph_query_lens[:bs].fill_(1)
        self._cuda_graph_query_start_loc[: bs + 1].copy_(
            torch.arange(bs + 1, dtype=torch.int32, device=self.device)
        )
        self._cuda_graph_token_to_req[:bs].copy_(
            torch.arange(bs, dtype=torch.int32, device=self.device)
        )
        self._cuda_graph_is_valid_token[:actual_bs].fill_(True)
        if actual_bs < bs:
            self._cuda_graph_is_valid_token[actual_bs:bs].fill_(False)
        if req_to_page is not None:
            self._cuda_graph_block_table[:bs, : self.max_num_pages].copy_(
                req_to_page[req_pool_indices[:bs], : self.max_num_pages]
            )
        offsets_on_device = {
            str(gid): off.to(device=self.device, dtype=torch.int32)
            for gid, off in paged_cache_block_table_base_offsets.items()
        }
        metadata_paged = self._refresh_cuda_graph_paged_cache_block_tables(
            bs,
            {
                str(group_id): table.to(device=self.device, dtype=torch.int32)
                for group_id, table in paged_cache_block_tables.items()
            },
            pad_value=-1,
            paged_cache_block_table_base_offsets=offsets_on_device,
        )
        metadata_base_offsets = self._refresh_cuda_graph_base_offsets(
            bs,
            offsets_on_device,
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
        metadata.forward_mode = forward_mode
        metadata.paged_cache_block_tables = metadata_paged
        metadata.paged_cache_block_table_base_offsets = metadata_base_offsets
        metadata.swa_block_table = swa_block_table
        metadata.swa_base_logical_page = swa_base
        metadata.compressor_state_block_tables = compressor_state_block_tables
        metadata.compressor_state_base_logical_pages = compressor_state_base
        metadata.indexer_state_block_table = indexer_state_block_table
        metadata.indexer_state_base_logical_page = indexer_state_base
        metadata.num_prefill_reqs = 0
        metadata.num_prefill_tokens = 0
        if (
            forward_mode is not None
            and forward_mode.is_decode()
            and self._decode_swa_window_size > 0
            and self._decode_swa_block_size > 0
        ):
            self._update_decode_swa_metadata(
                metadata,
                window_size=self._decode_swa_window_size,
                block_size=self._decode_swa_block_size,
            )
            metadata.refresh_decode_compressed_slot_mappings()
            _refresh_decode_indexer_plan_cache(
                metadata,
                max_context_len=self.context_len,
            )
            _refresh_decode_indexer_schedule_metadata(metadata)
        self.forward_metadata = metadata

    def advance_draft_forward_metadata(self):
        raise NotImplementedError(
            "DeepSeek V4 attention does not support draft graphs yet"
        )

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError("DeepSeek V4 uses the model-local attention forward")

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError("DeepSeek V4 uses the model-local attention forward")


register_backend("deepseek_v4", {AttentionArch.MLA}, DeepseekV4AttentionBackend)
