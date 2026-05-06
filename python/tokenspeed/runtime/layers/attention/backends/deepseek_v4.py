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

from typing import Any

import torch

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    DEEPSEEK_V4_SWA_SCALE_DIM,
    DEEPSEEK_V4_SWA_TOKEN_STRIDE,
    DeepseekV4AttentionOpUnavailable,
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


def _swa_block_table(metadata: DeepseekV4ForwardMetadata) -> torch.Tensor:
    return (
        metadata.swa_block_table
        if metadata.swa_block_table is not None
        else metadata.block_table
    )


def _expand_compact_block_table(
    compact: torch.Tensor,
    base_offsets: torch.Tensor,
    *,
    absolute_cols: int,
) -> torch.Tensor:
    if compact.ndim != 2:
        raise ValueError(
            f"compact block table must be 2D, got {tuple(compact.shape)}"
        )
    bs, k = int(compact.shape[0]), int(compact.shape[1])
    cols = max(int(absolute_cols), 0)
    device = compact.device
    if base_offsets.shape[0] != bs:
        raise ValueError(
            f"base_offsets shape {tuple(base_offsets.shape)} mismatches "
            f"compact rows {bs}"
        )
    out = torch.full((bs, max(cols, 1)), -1, dtype=torch.int32, device=device)
    if k == 0 or cols == 0:
        return out[:, :cols]
    col_idx = torch.arange(k, device=device, dtype=torch.int32)
    abs_cols = base_offsets.to(torch.int32).view(-1, 1) + col_idx.view(1, -1)
    in_range = (abs_cols >= 0) & (abs_cols < cols)
    safe = abs_cols.clamp(0, cols - 1).to(torch.int64)
    rows = (
        torch.arange(bs, device=device, dtype=torch.int64)
        .view(-1, 1)
        .expand(bs, k)
    )
    src = torch.where(
        in_range,
        compact.to(torch.int32),
        torch.full_like(compact, -1, dtype=torch.int32),
    )
    out.index_put_((rows, safe), src, accumulate=False)
    return out[:, :cols]


def _expand_paged_cache_block_tables(
    paged: dict[str, torch.Tensor],
    base_offsets: dict[str, torch.Tensor],
    *,
    absolute_cols: dict[str, int],
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for gid, table in paged.items():
        offsets = base_offsets.get(gid)
        if offsets is None:
            out[gid] = table
            continue
        target = int(absolute_cols.get(gid, int(table.shape[1])))
        out[gid] = _expand_compact_block_table(table, offsets, absolute_cols=target)
    return out


def _cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.pad(
        torch.cumsum(lengths.to(torch.int32), dim=0, dtype=torch.int32),
        (1, 0),
    )


class DeepseekV4AttentionBackend(AttentionBackend):
    """Metadata owner for the model-local DeepSeek V4 attention path."""

    uses_paged_cache_groups = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.page_size = config.page_size
        self.context_len = config.context_len
        self.max_num_pages = max(
            1,
            (self.context_len + self.page_size - 1) // self.page_size,
        )
        self.forward_metadata: DeepseekV4ForwardMetadata | None = None
        self._decode_tile_metadata = {}
        self._cuda_graph_metadata = {}
        self._cuda_graph_paged_cache_block_tables: dict[str, torch.Tensor] = {}
        self._cuda_graph_paged_cache_base_offsets: dict[str, torch.Tensor] = {}
        self._paged_cache_group_specs: dict[str, dict[str, Any]] = {}
        self._cuda_graph_max_bs = 0
        self._prefill_workspace_buffer: torch.Tensor | None = None
        self._prefill_workspace_rows = 0
        self._prefill_workspace_head_dim = 0
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
        rows = max(1, num_reqs * workspace_width)
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
            num_reqs,
            workspace_width,
            head_dim,
        )

    def _query_lens(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode | None,
        extend_seq_lens_cpu: torch.Tensor | None,
        extend_prefix_lens_cpu: torch.Tensor | None,
        extend_prefix_lens: torch.Tensor | None,
    ) -> torch.Tensor:
        if forward_mode is not None and forward_mode.is_decode_or_idle():
            return torch.ones(bs, dtype=torch.int32, device=seq_lens.device)
        if extend_seq_lens_cpu is not None:
            return extend_seq_lens_cpu[:bs].to(seq_lens.device, dtype=torch.int32)
        if extend_prefix_lens_cpu is not None:
            prefix = extend_prefix_lens_cpu[:bs].to(seq_lens.device, dtype=torch.int32)
            return (seq_lens[:bs].to(torch.int32) - prefix).clamp_min(0)
        if extend_prefix_lens is not None:
            prefix = extend_prefix_lens[:bs].to(torch.int32)
            return (seq_lens[:bs].to(torch.int32) - prefix).clamp_min(0)
        return seq_lens[:bs].to(torch.int32)

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
        del num_tokens, kwargs
        device = seq_lens.device
        req_pool_indices = req_pool_indices[:bs]
        seq_lens = seq_lens[:bs].to(torch.int32)
        query_lens = self._query_lens(
            bs,
            seq_lens,
            forward_mode,
            extend_seq_lens_cpu,
            extend_prefix_lens_cpu,
            extend_prefix_lens,
        )
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
        absolute_cols: dict[str, int] = {}
        for gid in base_offsets_on_device:
            spec = self._paged_cache_group_specs.get(gid)
            if spec is None or not spec["sliding"]:
                continue
            absolute_cols[gid] = max(
                1,
                (max_seq_len + spec["raw_per_page"] - 1) // spec["raw_per_page"],
            )
        paged_cache_block_tables = _expand_paged_cache_block_tables(
            paged_cache_block_tables,
            base_offsets_on_device,
            absolute_cols=absolute_cols,
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
        self.forward_metadata = DeepseekV4ForwardMetadata(
            page_size=self.page_size,
            req_pool_indices=req_pool_indices,
            block_table=block_table,
            seq_lens=seq_lens,
            query_lens=query_lens,
            query_start_loc=_cu_seqlens(query_lens),
            token_to_req_indices=token_to_req,
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
            window_size=window_size,
            block_size=block_size,
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

    def _decode_compressed_indices_and_lens(
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
        capturing = positions.is_cuda and torch.cuda.is_current_stream_capturing()
        if compress_ratio == 4:
            if topk_indices is None:
                raise RuntimeError("DeepSeek V4 CSA decode requires top-k indices")
            indices_2d, lens = deepseek_v4_compute_global_topk_indices_and_lens(
                topk_indices=topk_indices,
                token_to_req_indices=metadata.token_to_req_indices[:num_tokens],
                block_table=block_table,
                block_size=block_size,
            )
            return indices_2d.unsqueeze(1), lens
        else:
            compressed_lens = torch.div(
                positions.to(torch.int64) + 1,
                compress_ratio,
                rounding_mode="floor",
            ).clamp_min(0)
            if capturing:
                max_len = max(
                    1,
                    (self.context_len + compress_ratio - 1) // compress_ratio,
                )
            else:
                max_len = int(compressed_lens.max().item()) if num_tokens else 0
            width = max(64, ((max(max_len, 1) + 63) // 64) * 64)
            offsets = torch.arange(width, dtype=torch.int64, device=positions.device)
            local = offsets[None, :].expand(num_tokens, -1)
            valid = offsets[None, :] < compressed_lens[:, None]
            lens = compressed_lens.to(torch.int32)

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
        return indices, lens

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
        extra_indices, extra_lens = self._decode_compressed_indices_and_lens(
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
            block_size=token_to_kv_pool.swa_block_size,
            offset=compressed_base,
        )
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

    def init_cuda_graph_state(
        self,
        max_bs: int,
        seq_lens_buf: torch.Tensor | None = None,
        paged_cache_group_specs=(),
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
        self._paged_cache_group_specs = {}
        for spec in tuple(paged_cache_group_specs or ()):
            raw_per_page = max(
                1,
                int(spec.rows_per_page) * int(spec.entry_stride_tokens),
            )
            max_pages = max(
                1,
                (self.context_len + raw_per_page - 1) // raw_per_page,
            )
            gid = str(spec.group_id)
            sliding = str(getattr(spec, "retention", "")) == "sliding_window"
            self._paged_cache_group_specs[gid] = {
                "raw_per_page": raw_per_page,
                "sliding": sliding,
            }
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
        offsets_in = paged_cache_block_table_base_offsets or {}
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
                offsets = offsets_in.get(group_id)
                if offsets is None:
                    if cols > int(buf.shape[1]):
                        raise RuntimeError(
                            "DeepSeek V4 CUDA graph paged cache table width "
                            f"mismatch for {group_id!r}: got {cols}, capture "
                            f"buffer has {int(buf.shape[1])}"
                        )
                    if cols > 0:
                        buf[:bs, :cols].copy_(table[:bs, :cols].to(torch.int32))
                elif cols > 0:
                    rows_idx = (
                        torch.arange(bs, device=buf.device, dtype=torch.int64)
                        .view(-1, 1)
                        .expand(bs, cols)
                    )
                    col_idx = torch.arange(
                        cols, device=buf.device, dtype=torch.int32
                    ).view(1, -1)
                    abs_cols = offsets[:bs].to(torch.int32).view(-1, 1) + col_idx
                    in_range = (abs_cols >= 0) & (abs_cols < buf.shape[1])
                    safe_abs = abs_cols.clamp(0, buf.shape[1] - 1).to(torch.int64)
                    src = torch.where(
                        in_range,
                        table[:bs, :cols].to(torch.int32),
                        torch.full(
                            (bs, cols),
                            int(pad_value),
                            dtype=torch.int32,
                            device=buf.device,
                        ),
                    )
                    buf.index_put_((rows_idx, safe_abs), src, accumulate=False)
            out[group_id] = buf[:bs]
        return out

    def _refresh_cuda_graph_base_offsets(
        self,
        bs: int,
        base_offsets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
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
