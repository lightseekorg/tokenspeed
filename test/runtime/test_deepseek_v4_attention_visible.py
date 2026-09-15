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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v4 import (
    DeepseekV4AttentionBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import model_wide_kwargs
from tokenspeed.runtime.layers.attention.deepseek_v4.metadata import (
    DeepseekV4ForwardMetadata,
)
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_deepseek_v4 import (
    DeepseekV4CacheMetadata,
)


def _backend() -> DeepseekV4AttentionBackend:
    backend = object.__new__(DeepseekV4AttentionBackend)
    backend.vision_enabled = True
    backend.vision_max_n_token = 384
    return backend


def test_visible_static_capacity_comes_from_model_config_not_batch_payload():
    server_args = SimpleNamespace(
        device="cuda",
        kv_cache_quant_method="none",
        prefix_granularity=64,
        spec_context_pad=0,
        max_num_seqs=8,
        data_parallel_size=1,
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=1)),
        max_cudagraph_capture_size=8,
        chunked_prefill_size=8192,
        disaggregation_mode="null",
        speculative_algorithm=None,
    )
    vision_model = SimpleNamespace(
        dtype=torch.bfloat16,
        context_len=4096,
        hf_config=SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"],
            vision_n_layers=24,
            vision_max_n_token=384,
        ),
    )
    text_model = SimpleNamespace(
        dtype=torch.bfloat16,
        context_len=4096,
        hf_config=SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"],
            vision_n_layers=0,
            vision_max_n_token=384,
        ),
    )

    vision = model_wide_kwargs(
        server_args, vision_model, False, kv_cache_dtype=torch.bfloat16
    )
    text = model_wide_kwargs(
        server_args, text_model, False, kv_cache_dtype=torch.bfloat16
    )

    assert vision["vision_enabled"] is True
    assert vision["vision_max_n_token"] == 384
    assert text["vision_enabled"] is False
    assert text["vision_max_n_token"] == 0


def _metadata(*, partial: bool = False) -> DeepseekV4ForwardMetadata:
    atomic = [(8, 12)] if partial else [(1, 4), (12, 18)]
    return DeepseekV4ForwardMetadata(
        seq_lens=torch.tensor([20, 40], dtype=torch.int32),
        query_lens=torch.tensor([10, 20], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 10, 30], dtype=torch.int32),
        token_to_req_indices=torch.tensor([0] * 10 + [1] * 20, dtype=torch.int32),
        cache=DeepseekV4CacheMetadata(
            page_size=64,
            page_table=torch.zeros((2, 1), dtype=torch.int32),
            swa_page_table=torch.zeros((2, 1), dtype=torch.int32),
        ),
        seq_lens_cpu=torch.tensor([20, 40], dtype=torch.int32),
        query_lens_cpu=torch.tensor([10, 20], dtype=torch.int32),
        num_prefill_reqs=2,
        num_prefill_tokens=30,
        forward_mode=ForwardMode.EXTEND,
        vision_left=torch.zeros(30, dtype=torch.int32),
        vision_right=torch.zeros(30, dtype=torch.int32),
        vision_atomic_spans=[atomic, []],
        vision_visibility_spans=[[(11, 12)] if partial else [(1, 4), (15, 18)], []],
        vision_atomic_spans_in_chunk=[[(8, 12)] if partial else [(12, 18)], []],
        vision_visibility_spans_in_chunk=[[(11, 12)] if partial else [(15, 18)], []],
    )


def test_visible_metadata_filters_absolute_spans_and_keeps_leading_pad_causal():
    metadata = _metadata()
    left, right = _backend()._prepare_prefill_visibility(
        metadata, device=torch.device("cpu")
    )

    assert left is metadata.vision_left
    assert right is metadata.vision_right
    assert left.tolist()[2:5] == [0, 0, 0]
    assert right.tolist()[2:5] == [0, 0, 0]
    assert left.tolist()[5:9] == [0, 1, 2, 3]
    assert right.tolist()[5:9] == [3, 2, 1, 0]
    assert left.tolist()[10:] == [0] * 20
    assert metadata.vision_prefill_validated


def test_visible_metadata_rejects_partial_atomic_overlap_by_name():
    with pytest.raises(RuntimeError, match="atomic span .* partially overlaps"):
        _backend()._prepare_prefill_visibility(
            _metadata(partial=True), device=torch.device("cpu")
        )


def test_visible_metadata_slices_request_and_token_payloads_together():
    backend = _backend()
    metadata = _metadata()
    backend._prepare_prefill_visibility(metadata, device=torch.device("cpu"))

    image = backend._metadata_slice(
        metadata,
        req_start=0,
        req_end=1,
        token_start=0,
        token_end=10,
        forward_mode=ForwardMode.EXTEND,
    )
    text = backend._metadata_slice(
        metadata,
        req_start=1,
        req_end=2,
        token_start=10,
        token_end=30,
        forward_mode=ForwardMode.EXTEND,
    )

    assert image.vision_atomic_spans == [[(1, 4), (12, 18)]]
    assert image.vision_atomic_spans_in_chunk == [[(12, 18)]]
    assert image.vision_left.numel() == 10
    assert text.vision_atomic_spans == [[]]
    assert text.vision_left.numel() == 20
    assert not any(text.vision_atomic_spans_in_chunk)
