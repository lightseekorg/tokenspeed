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

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_deepseek_v4 import (
    DeepseekV4CacheMetadata,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalInputs,
)

DEFAULT_VISION_MAX_N_TOKEN = 384


def build_image_window(
    mm_inputs: Sequence[MultimodalInputs | None],
    prefix_lens: Sequence[int],
    query_lens: Sequence[int],
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    """Build per-token image visibility for prefill metadata."""
    image_start_type, image_end_type = 0, 4
    max_image_tokens = DEFAULT_VISION_MAX_N_TOKEN
    for mm_input in mm_inputs:
        if mm_input is not None:
            for item in mm_input.mm_items:
                data = item.model_specific_data
                if "vision_max_n_token" in data:
                    max_image_tokens = max(
                        max_image_tokens, int(data["vision_max_n_token"])
                    )

    lefts = [0] * sum(query_lens)
    rights = [0] * len(lefts)
    has_image = False
    token_start = 0
    for req_idx, (prefix, query_len) in enumerate(
        zip(prefix_lens, query_lens, strict=True)
    ):
        mm_input = mm_inputs[req_idx] if req_idx < len(mm_inputs) else None
        if mm_input is not None:
            for item in mm_input.mm_items:
                if (
                    item.modality != Modality.IMAGE
                    or not item.offsets
                    or "types" not in item.model_specific_data
                ):
                    continue
                block_length = sum(end - start + 1 for start, end in item.offsets)
                types = item.model_specific_data["types"][-block_length:].tolist()
                for offset_start, offset_end in item.offsets:
                    start, end = int(offset_start), int(offset_end) + 1
                    if end <= prefix or start >= prefix + query_len:
                        continue
                    if start < prefix or end > prefix + query_len:
                        raise ValueError(
                            f"DeepSeek V4 image block [{start}, {end}) crosses "
                            f"the prefill range [{prefix}, {prefix + query_len})."
                        )
                    try:
                        image_start = start + types.index(image_start_type)
                        image_end = (
                            start + len(types) - types[::-1].index(image_end_type)
                        )
                    except ValueError:
                        continue
                    has_image = True
                    for pos in range(image_start, image_end):
                        index = token_start + pos - prefix
                        lefts[index] = min(pos - image_start, max_image_tokens - 1)
                        rights[index] = min(image_end - 1 - pos, max_image_tokens)
        token_start += query_len
    if not has_image:
        return None, None, 0
    return (
        torch.tensor(lefts, dtype=torch.int32, device=device),
        torch.tensor(rights, dtype=torch.int32, device=device),
        max_image_tokens,
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
    page_table: torch.Tensor
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
    swa_indices: torch.Tensor | None = None
    swa_lens: torch.Tensor | None = None
    swa_window_size: int = 0
    swa_block_size: int = 0
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
    decode_schedule_metadata_refreshed_keys: set[tuple[int, int, int]] = field(
        default_factory=set
    )
    prefill_plan_cache: dict[tuple[int, ...], DeepseekV4IndexerPrefillMetadata] = field(
        default_factory=dict
    )


@dataclass
class DeepseekV4SparseIndexerMetadata:
    batch_metadata: DeepseekV4IndexerBatchMetadata | None = None
    prefill_metadata: DeepseekV4IndexerPrefillMetadata | None = None
    decode_plan: DeepseekV4IndexerDecodePlan | None = None
    decode_schedule_metadata: torch.Tensor | None = None


@dataclass
class DeepseekV4ForwardMetadata:
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
    # The CUDA-graph padding mask, one mission: True = live token, False =
    # a padded replay row (never mixed-batch state; prefill rows are always
    # live).
    is_valid_token: torch.Tensor | None = None
    # CPU lens are retained for sparse prefill/indexer planning without
    # forcing another device-to-host sync in the model path.
    seq_lens_cpu: torch.Tensor | None = None
    query_lens_cpu: torch.Tensor | None = None
    # Cached split boundary derived from scheduler num_extends/query_lens.
    num_prefill_reqs: int = 0
    num_prefill_tokens: int = 0
    swa_left: torch.Tensor | None = None
    swa_right: torch.Tensor | None = None
    swa_max_image_tokens: int = 0

    def decode_req_count(self) -> int:
        return max(0, int(self.seq_lens.shape[0]) - int(self.num_prefill_reqs))

    def decode_token_count(self) -> int:
        return max(
            0,
            int(self.token_to_req_indices.shape[0]) - int(self.num_prefill_tokens),
        )
