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

from types import SimpleNamespace

import torch

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
        token_to_kv_pool=SimpleNamespace(swa_block_size=2),
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
        token_to_kv_pool=SimpleNamespace(swa_block_size=2),
    )
    positions = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int32)
    out_cache_loc = torch.tensor([100, 101, 102, 103, 104], dtype=torch.int32)

    slot_mapping = _deepseek_v4_swa_slot_mapping(ctx, positions, out_cache_loc)

    assert slot_mapping.tolist() == [20, 21, 22, 40, 41]


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
        ),
    )
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        input_num_tokens=5,
        attn_backend=SimpleNamespace(forward_metadata=metadata),
        token_to_kv_pool=SimpleNamespace(swa_block_size=2),
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
