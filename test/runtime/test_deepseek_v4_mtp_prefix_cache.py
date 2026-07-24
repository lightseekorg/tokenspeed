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

import os
import sys
from types import SimpleNamespace

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.model_executor import (
    _draft_idle_global_num_tokens_for_step,
)
from tokenspeed.runtime.models.deepseek_v4 import _deepseek_v4_swa_slot_mapping


def test_deepseek_v4_swa_slot_mapping_expands_mtp_decode_requests():
    metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0, 1], dtype=torch.int32),
        cache=SimpleNamespace(
            swa_block_table=torch.tensor(
                [
                    [10, 11],
                    [20, 21],
                ],
                dtype=torch.int32,
            ),
            swa_base_logical_page=None,
        ),
    )
    ctx = SimpleNamespace(
        attn_backend=SimpleNamespace(forward_metadata=metadata),
        token_to_kv_pool=SimpleNamespace(
            swa_block_size=2,
            swa_capacity_pages=512,
            swa_capacity_slots=1024,
        ),
    )
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    out_cache_loc = torch.tensor([100, 101, 102, 103], dtype=torch.int32)

    slot_mapping = _deepseek_v4_swa_slot_mapping(ctx, positions, out_cache_loc)

    assert slot_mapping.tolist() == [20, 21, 42, 43]


def test_deepseek_v4_swa_slot_mapping_prefers_draft_prefill_metadata():
    cache = SimpleNamespace(
        swa_block_table=torch.tensor(
            [
                [10, 11],
                [20, 21],
            ],
            dtype=torch.int32,
        ),
        swa_base_logical_page=None,
    )
    decode_metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0, 1], dtype=torch.int32),
        cache=cache,
    )
    prefill_metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32),
        cache=cache,
    )
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        input_num_tokens=5,
        attn_backend=SimpleNamespace(
            forward_metadata=decode_metadata,
            forward_prefill_metadata=prefill_metadata,
        ),
        token_to_kv_pool=SimpleNamespace(
            swa_block_size=2,
            swa_capacity_pages=512,
            swa_capacity_slots=1024,
        ),
    )
    positions = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32)
    out_cache_loc = torch.tensor([100, 101, 102, 103, 104], dtype=torch.int32)

    slot_mapping = _deepseek_v4_swa_slot_mapping(ctx, positions, out_cache_loc)

    assert slot_mapping.tolist() == [20, 21, 22, 40, 41]


def test_deepseek_v4_swa_slot_mapping_masks_invalid_and_overflow_slots():
    # The mapping must arrive at per-layer SWA inserts already sanitized:
    # invalid CUDA-graph tokens and out-of-capacity slots masked to -1.
    metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0, 1], dtype=torch.int32),
        cache=SimpleNamespace(
            swa_block_table=torch.tensor(
                [
                    [10, 11],
                    [20, 21],
                ],
                dtype=torch.int32,
            ),
            swa_base_logical_page=None,
        ),
        is_valid_token=torch.tensor([True, True, False, True]),
    )
    ctx = SimpleNamespace(
        attn_backend=SimpleNamespace(forward_metadata=metadata),
        token_to_kv_pool=SimpleNamespace(
            swa_block_size=2,
            swa_capacity_pages=22,
            swa_capacity_slots=43,
        ),
    )
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    out_cache_loc = torch.tensor([100, 101, 102, 103], dtype=torch.int32)

    slot_mapping = _deepseek_v4_swa_slot_mapping(ctx, positions, out_cache_loc)

    # Raw mapping is [20, 21, 42, 43]: index 2 is masked by is_valid_token,
    # index 3 exceeds the 43-slot capacity.
    assert slot_mapping.tolist() == [20, 21, -1, -1]


def test_deepseek_v4_swa_slot_mapping_masks_out_of_range_page_ids():
    metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0, 0, 0], dtype=torch.int32),
        cache=SimpleNamespace(
            swa_block_table=torch.tensor([[0, 511, 512]], dtype=torch.int32),
            swa_base_logical_page=None,
        ),
    )
    ctx = SimpleNamespace(
        attn_backend=SimpleNamespace(forward_metadata=metadata),
        token_to_kv_pool=SimpleNamespace(
            swa_block_size=2,
            swa_capacity_pages=512,
            swa_capacity_slots=1024,
        ),
    )

    slot_mapping = _deepseek_v4_swa_slot_mapping(
        ctx,
        positions=torch.tensor([0, 2, 4], dtype=torch.int32),
        out_cache_loc=torch.zeros(3, dtype=torch.int32),
    )

    assert slot_mapping.tolist() == [-1, 1022, -1]


def test_deepseek_v4_swa_slot_mapping_fails_closed_without_capacity():
    # Zero capacity masks every slot; a pool without the property fails fast.
    # Both protect the fused cache-insert kernels now that per-layer
    # sanitization is gone.
    metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0, 1], dtype=torch.int32),
        cache=SimpleNamespace(
            swa_block_table=torch.tensor(
                [
                    [10, 11],
                    [20, 21],
                ],
                dtype=torch.int32,
            ),
            swa_base_logical_page=None,
        ),
    )
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    out_cache_loc = torch.tensor([100, 101, 102, 103], dtype=torch.int32)

    zero_capacity_ctx = SimpleNamespace(
        attn_backend=SimpleNamespace(forward_metadata=metadata),
        token_to_kv_pool=SimpleNamespace(
            swa_block_size=2,
            swa_capacity_pages=0,
            swa_capacity_slots=0,
        ),
    )
    slot_mapping = _deepseek_v4_swa_slot_mapping(
        zero_capacity_ctx, positions, out_cache_loc
    )
    assert slot_mapping.tolist() == [-1, -1, -1, -1]

    no_capacity_ctx = SimpleNamespace(
        attn_backend=SimpleNamespace(forward_metadata=metadata),
        token_to_kv_pool=SimpleNamespace(swa_block_size=2, swa_capacity_pages=512),
    )
    with pytest.raises(AttributeError, match="swa_capacity_slots"):
        _deepseek_v4_swa_slot_mapping(no_capacity_ctx, positions, out_cache_loc)


def test_deepseek_v4_swa_slot_mapping_falls_back_for_incompatible_draft_metadata():
    metadata = SimpleNamespace(
        token_to_req_indices=torch.tensor([0, 1], dtype=torch.int32),
        cache=SimpleNamespace(
            swa_block_table=torch.tensor(
                [
                    [10, 11],
                    [20, 21],
                ],
                dtype=torch.int32,
            ),
            swa_base_logical_page=None,
            has_legacy_block_table=True,
        ),
    )
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        input_num_tokens=5,
        attn_backend=SimpleNamespace(forward_metadata=metadata),
        token_to_kv_pool=SimpleNamespace(
            swa_block_size=2,
            swa_capacity_pages=512,
            swa_capacity_slots=1024,
        ),
    )
    positions = torch.arange(5, dtype=torch.int32)
    out_cache_loc = torch.tensor([100, 101, 102, 103, 104], dtype=torch.int32)

    slot_mapping = _deepseek_v4_swa_slot_mapping(ctx, positions, out_cache_loc)

    assert torch.equal(slot_mapping, out_cache_loc)


def test_draft_idle_global_num_tokens_match_multi_step_decode_shape():
    global_num_tokens = [6, 0, 3]
    global_bs = [2, 0, 1]

    assert (
        _draft_idle_global_num_tokens_for_step(0, global_num_tokens, global_bs)
        is global_num_tokens
    )
    assert (
        _draft_idle_global_num_tokens_for_step(1, global_num_tokens, global_bs)
        is global_bs
    )
    assert (
        _draft_idle_global_num_tokens_for_step(2, global_num_tokens, global_bs)
        is global_bs
    )
    assert (
        _draft_idle_global_num_tokens_for_step(1, global_num_tokens, None)
        is global_num_tokens
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
