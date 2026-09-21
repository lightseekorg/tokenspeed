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

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import tokenspeed_kernel.benchmark.generators.dsa as dsa_generator
import torch
from tokenspeed_kernel.benchmark.generators.dsa import (
    prepare_dsa_decode,
    prepare_dsa_prefill,
    prepare_kpool_decode_append,
    prepare_kpool_decode_topk,
    prepare_kpool_prefill_topk,
    prepare_kpool_prefill_write,
)
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
)
from tokenspeed_kernel.platform import ArchVersion, PlatformInfo

_EXPLICIT_CONFIG = {
    "dtype": "bfloat16",
    "index_heads": 8,
    "index_head_dim": 128,
    "local_attention_heads": 4,
    "kv_lora_rank": 128,
    "qk_nope_head_dim": 256,
    "qk_rope_head_dim": 64,
    "pool_size": 8,
    "index_page_stride_bytes": 4096,
    "kv_page_size": 64,
    "topk_tokens": 512,
    "max_context": 4096,
    "max_logits_bytes": 1024 * 1024,
}


def _platform() -> PlatformInfo:
    return PlatformInfo(
        vendor="amd",
        arch_version=ArchVersion(9, 5),
        device_name="test device",
        device_count=1,
        total_memory=1,
        memory_bandwidth=1.0,
        sm_count=1,
        max_threads_per_sm=1,
        max_shared_memory_per_sm=1,
    )


def _request(
    mode: str = "test",
    parameters: dict[str, Any] | None = None,
) -> BenchmarkRequest:
    return BenchmarkRequest(
        family="attention",
        mode=mode,
        parameters=parameters or {},
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )


def _profile_config(**overrides: object) -> dsa_generator._DSAConfig:
    request = _request(parameters={"model_profile": "glm53_flash_tp4", **overrides})
    return dsa_generator._resolve_config(request, specific=set())


def test_dsa_generator_resolves_named_profile_and_derived_geometry() -> None:
    config = _profile_config()

    assert config.dtype is torch.bfloat16
    assert config.index_heads == 32
    assert config.local_attention_heads == 16
    assert config.index_rows_per_page == 16
    assert config.topk_pools == 512
    assert config.selected_width == 2051
    assert config.qk_head_dim == 512
    assert config.dsa_softmax_scale == pytest.approx(256**-0.5)
    assert dsa_generator._common_parameters(config)["attention_heads"] == 16


def test_dsa_generator_supports_profile_overrides_and_explicit_config() -> None:
    overridden = _profile_config(
        kv_page_size=128,
        topk_tokens=1024,
        max_context=8192,
    )
    explicit = dsa_generator._resolve_config(
        _request(parameters=_EXPLICIT_CONFIG),
        specific=set(),
    )

    assert overridden.index_rows_per_page == 32
    assert overridden.topk_pools == 256
    assert overridden.max_context == 8192
    assert explicit.model_profile is None
    assert explicit.index_rows_per_page == 8
    assert explicit.topk_pools == 64
    assert explicit.selected_width == 519
    assert explicit.qk_head_dim == 192
    assert explicit.dsa_softmax_scale == pytest.approx(320**-0.5)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"model_profile": "unknown"}, "Unknown DSA model_profile"),
        ({}, "without model_profile require"),
    ],
)
def test_dsa_generator_requires_a_known_profile_or_explicit_config(
    parameters: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(BenchmarkCaseError, match=message):
        dsa_generator._resolve_config(_request(parameters=parameters), specific=set())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"topk_tokens": 2050}, "topk_tokens must be divisible"),
        ({"kv_page_size": 65}, "kv_page_size must be divisible"),
        ({"index_page_stride_bytes": 2000}, "fit and align"),
    ],
)
def test_dsa_generator_rejects_inconsistent_derived_geometry(
    overrides: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(BenchmarkCaseError, match=message):
        _profile_config(**overrides)


@pytest.mark.parametrize(
    ("mode", "prepare", "shape"),
    [
        ("kpool_prefill_write", prepare_kpool_prefill_write, {"rows": 16}),
        (
            "kpool_prefill_topk",
            prepare_kpool_prefill_topk,
            {"batch": 1, "prefix_tokens": 0, "query_tokens_per_sequence": 64},
        ),
        (
            "dsa_prefill",
            prepare_dsa_prefill,
            {"batch": 1, "prefix_tokens": 0, "query_tokens_per_sequence": 64},
        ),
        (
            "kpool_decode_append",
            prepare_kpool_decode_append,
            {"batch": 1, "q_len_per_req": 1, "sequence_length": 8192},
        ),
        (
            "kpool_decode_topk",
            prepare_kpool_decode_topk,
            {"batch": 1, "q_len_per_req": 1, "sequence_length": 8192},
        ),
        (
            "dsa_decode",
            prepare_dsa_decode,
            {"batch": 1, "q_len_per_req": 1, "sequence_length": 8192},
        ),
    ],
)
def test_dsa_generators_reject_unsupported_validation_before_allocating_inputs(
    mode: str,
    prepare: Callable[[BenchmarkRequest, PlatformInfo], object],
    shape: dict[str, int],
) -> None:
    request = _request(
        mode,
        {
            "model_profile": "glm53_flash_tp4",
            **shape,
            "validation": {"runs": 1},
        },
    )

    with pytest.raises(BenchmarkCaseError, match="not implemented"):
        prepare(request, _platform())


def test_dsa_generator_builds_page_aligned_prefill_metadata() -> None:
    metadata = dsa_generator._prefill_metadata(
        2,
        0,
        68,
        config=_profile_config(),
        device="cpu",
    )

    assert metadata["query_start_loc"].tolist() == [0, 68, 136]
    assert metadata["req_ids"].tolist() == [0] * 68 + [1] * 68
    assert metadata["pool_workspace_slots"].tolist() == (
        list(range(17)) + list(range(32, 49))
    )
    assert metadata["row_starts"].tolist() == [0] * 68 + [17] * 68
    assert metadata["row_ends"][[0, 63, 64, 67, 68, 131, 132, 135]].tolist() == [
        0,
        16,
        16,
        17,
        17,
        33,
        33,
        34,
    ]
    assert metadata["max_num_pools"] == 17


def test_dsa_generator_uses_profile_index_page_stride(monkeypatch) -> None:
    original_empty = torch.empty
    config = _profile_config()

    def cpu_randn(shape, *, generator, dtype):
        _ = generator
        return torch.randn(shape, dtype=dtype)

    monkeypatch.setattr(dsa_generator, "_randn", cpu_randn)
    monkeypatch.setattr(
        dsa_generator.torch,
        "empty",
        lambda shape, *, dtype, device: original_empty(shape, dtype=dtype),
    )
    cache, values, scales = dsa_generator._packed_index_cache(
        3,
        config=config,
        generator=torch.Generator(device="cpu"),
    )

    assert cache.shape == (3, 2112)
    assert cache.stride() == (23296, 1)
    assert values.shape == (3, 16, 128)
    assert scales.shape == (3, 16, 1)
    assert values.stride(0) == 23296
    assert scales.stride(0) == 5824


def test_dsa_generator_reuses_runtime_history_table_for_index_and_kv() -> None:
    index_table, kv_table, pages_per_request = dsa_generator._page_tables(
        2,
        128,
        config=_profile_config(),
        device="cpu",
    )

    assert pages_per_request == 2
    assert index_table.tolist() == [[0, 1], [2, 3]]
    assert kv_table is index_table


def test_dsa_generator_uses_positive_decode_append_pages() -> None:
    table, index_pages = dsa_generator._decode_append_page_table(
        3,
        8192,
        config=_profile_config(),
        device="cpu",
    )

    assert table.shape == (3, 128)
    assert table[:, 0].tolist() == [1, 2, 3]
    assert torch.all(table > 0)
    assert int(table.max()) < index_pages
    assert index_pages == 4


def test_dsa_generator_matches_profile_decode_tail_widths() -> None:
    config = _profile_config()

    assert dsa_generator._decode_tail_width(1, config=config) == 4
    assert dsa_generator._decode_tail_width(4, config=config) == 8


def test_dsa_generator_preserves_fused_projection_weight_stride(monkeypatch) -> None:
    def cpu_randn(shape, *, generator, dtype):
        _ = generator
        return torch.randn(shape, dtype=dtype)

    monkeypatch.setattr(dsa_generator, "_randn", cpu_randn)
    weights = dsa_generator._kpool_scoring_weights(
        3,
        config=_profile_config(),
        generator=torch.Generator(device="cpu"),
    )

    assert weights.shape == (3, 32)
    assert weights.stride() == (160, 1)
    assert weights.storage_offset() == 128
    assert not weights.is_contiguous()


def test_dsa_generator_builds_request_local_selected_slots() -> None:
    causal_lens = torch.tensor([2048, 2049, 2050, 2051, 2048], dtype=torch.int32)
    req_ids = torch.tensor([0, 0, 0, 0, 1], dtype=torch.int32)

    slots, valid_lens = dsa_generator._selected_slots(
        causal_lens,
        req_ids,
        config=_profile_config(),
        region_slots=4096,
        device="cpu",
    )

    assert valid_lens.tolist() == [2048, 2049, 2050, 2051, 2048]
    assert slots.shape == (5, 2051)
    assert torch.all(slots[:4][slots[:4] >= 0] < 4096)
    assert torch.all(slots[4][slots[4] >= 0] >= 4096)
    assert slots[0, 2048:].tolist() == [-1, -1, -1]


def test_dsa_generator_state_reset_restores_mutated_buffers() -> None:
    tail = torch.arange(12, dtype=torch.float32).view(3, 4)
    index = torch.arange(8, dtype=torch.float32).view(2, 4)
    reset = dsa_generator._snapshot_reset(tail, index)

    tail.zero_()
    index.add_(100)
    reset()

    assert tail.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    assert index.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
