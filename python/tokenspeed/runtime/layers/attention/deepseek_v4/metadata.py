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

from dataclasses import dataclass, field

import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
    DeepseekV4CacheMetadata,
)


@dataclass
class DeepseekV4IndexerPrefillChunkPlan:
    token_start: int
    token_end: int
    request_start: int
    request_end: int
    slot_start: int
    slot_end: int
    gather_row_start: int
    gather_row_end: int
    max_seq_len_k: int
    cu_seq_lens_start: int
    cu_seq_lens_end: int
    skip_kv_gather: bool = False


@dataclass
class DeepseekV4IndexerPrefillMetadata:
    chunks: tuple[DeepseekV4IndexerPrefillChunkPlan, ...]
    chunk_specs: torch.Tensor
    chunk_offsets: torch.Tensor
    slots: torch.Tensor
    cu_seq_lens: torch.Tensor
    cu_seqlen_k_start: torch.Tensor
    cu_seqlen_k_end: torch.Tensor
    seq_lens_k: torch.Tensor

    def max_gather_rows(self) -> int:
        if not self.chunks:
            return 0
        return max(max(0, chunk.slot_end - chunk.slot_start) for chunk in self.chunks)

    @classmethod
    def empty(cls, device: torch.device) -> "DeepseekV4IndexerPrefillMetadata":
        return cls(
            chunks=(),
            chunk_specs=torch.empty((0, 5), dtype=torch.int64, device="cpu"),
            chunk_offsets=torch.empty((0, 7), dtype=torch.int64, device="cpu"),
            slots=torch.empty(0, dtype=torch.int64, device=device),
            cu_seq_lens=torch.empty(0, dtype=torch.int32, device=device),
            cu_seqlen_k_start=torch.empty(0, dtype=torch.int32, device=device),
            cu_seqlen_k_end=torch.empty(0, dtype=torch.int32, device=device),
            seq_lens_k=torch.empty(0, dtype=torch.int32, device=device),
        )


@dataclass
class DeepseekV4IndexerDecodePlan:
    context_lens: torch.Tensor
    block_table: torch.Tensor
    max_context_len: int


@dataclass
class DeepseekV4IndexerBatchMetadata:
    positions: torch.Tensor
    token_to_req_indices: torch.Tensor
    seq_lens_cpu: torch.Tensor
    query_lens_cpu: torch.Tensor
    num_prefill_tokens: int
    num_decode_tokens: int


@dataclass
class DeepseekV4AttentionMetadata:
    decode_swa_indices: torch.Tensor | None = None
    decode_swa_lens: torch.Tensor | None = None
    decode_swa_window_size: int = 0
    decode_swa_block_size: int = 0
    # Cache for dense compressed decode attention indices/lens. CSA decode uses
    # dynamic top-k indices and does not populate this cache.
    decode_dense_compressed_indices_cache: dict[
        tuple[int, int, int, int], tuple[torch.Tensor, torch.Tensor]
    ] = field(default_factory=dict)
    decode_dense_compressed_indices_capture_safe_keys: set[
        tuple[int, int, int, int]
    ] = field(default_factory=set)


@dataclass
class DeepseekV4IndexerMetadata:
    decode_schedule_metadata_cache: dict[tuple[int, int, int], torch.Tensor] = field(
        default_factory=dict
    )
    decode_plan_cache: dict[tuple[int, int, int], DeepseekV4IndexerDecodePlan] = field(
        default_factory=dict
    )
    decode_plan_refreshed_keys: set[tuple[int, int, int]] = field(default_factory=set)
    prefill_plan_cache: dict[tuple[int, int, int], DeepseekV4IndexerPrefillMetadata] = (
        field(default_factory=dict)
    )


@dataclass
class DeepseekV4SparseIndexerMetadata:
    batch_metadata: DeepseekV4IndexerBatchMetadata | None = None
    prefill_metadata: DeepseekV4IndexerPrefillMetadata | None = None
    decode_plan: DeepseekV4IndexerDecodePlan | None = None
    decode_schedule_metadata: torch.Tensor | None = None


@dataclass
class DeepseekV4ForwardMetadata:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    query_lens: torch.Tensor
    query_start_loc: torch.Tensor
    token_to_req_indices: torch.Tensor
    cache: DeepseekV4CacheMetadata
    attention: DeepseekV4AttentionMetadata = field(
        default_factory=DeepseekV4AttentionMetadata
    )
    indexer: DeepseekV4IndexerMetadata = field(
        default_factory=DeepseekV4IndexerMetadata
    )
    forward_mode: ForwardMode | None = None
    # Padding mask for CUDA graph replay rows; this is not mixed-batch state.
    is_valid_token: torch.Tensor | None = None
    # CPU lens are retained for sparse prefill/indexer planning without
    # forcing another device-to-host sync in the model path.
    seq_lens_cpu: torch.Tensor | None = None
    query_lens_cpu: torch.Tensor | None = None
    # Cached split boundary derived from scheduler num_extends/query_lens.
    num_prefill_reqs: int = 0
    num_prefill_tokens: int = 0

    def decode_req_count(self) -> int:
        return max(0, int(self.req_pool_indices.shape[0]) - int(self.num_prefill_reqs))

    def decode_token_count(self) -> int:
        return max(
            0,
            int(self.token_to_req_indices.shape[0]) - int(self.num_prefill_tokens),
        )
