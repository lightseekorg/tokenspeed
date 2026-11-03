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

from typing import Any

import pytest
import tokenspeed_kernel.benchmark.generators.dsa as dsa_generator
import torch
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
)

_GLM53_CONFIG = {
    "model_profile": "glm53_flash_tp4",
    "dtype": "bfloat16",
    "kv_cache_dtype": "bfloat16",
    "index_heads": 32,
    "index_head_dim": 128,
    "local_attention_heads": 16,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 256,
    "qk_rope_head_dim": 0,
    "pool_size": 4,
    "index_page_stride_bytes": 23296,
    "kv_page_size": 64,
    "topk_tokens": 2048,
    "max_context": 131072,
    "max_logits_bytes": 512 * 1024 * 1024,
}


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


def _config(**overrides: object) -> dsa_generator._DSAConfig:
    request = _request(parameters={**_GLM53_CONFIG, **overrides})
    return dsa_generator._resolve_config(request)


def test_dsa_generator_resolves_explicit_config_and_derived_geometry() -> None:
    config = _config()

    assert config.dtype is torch.bfloat16
    assert config.kv_cache_dtype is torch.bfloat16
    assert config.index_heads == 32
    assert config.local_attention_heads == 16
    assert config.index_rows_per_page == 16
    assert config.topk_pools == 512
    assert config.selected_width == 2051
    assert config.qk_head_dim == 512
    assert config.dsa_softmax_scale == pytest.approx(256**-0.5)
    assert dsa_generator._common_parameters(config)["attention_heads"] == 16


def test_dsa_generator_supports_fp8_kv_cache_configuration() -> None:
    config = _config(kv_cache_dtype="float8_e4m3fn")

    assert config.dtype is torch.bfloat16
    assert config.kv_cache_dtype is torch.float8_e4m3fn
    assert dsa_generator._common_parameters(config)["kv_cache_dtype"] == "float8_e4m3fn"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("model_profile", "unknown"),
        ("dtype", "float32"),
        ("kv_cache_dtype", "float32"),
    ],
)
def test_dsa_generator_rejects_unimplemented_config_values(
    name: str,
    value: str,
) -> None:
    with pytest.raises(BenchmarkCaseError, match=f"Implemented DSA {name}"):
        _config(**{name: value})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"index_head_dim": 64}, "index_head_dim must be divisible"),
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
        _config(**overrides)


def test_dsa_generator_rejects_unsupported_validation() -> None:
    request = _request(parameters={"validation": {"runs": 1}})
    with pytest.raises(BenchmarkCaseError, match="not implemented"):
        dsa_generator._resolve_config(request)


@pytest.mark.parametrize(
    ("mode", "prepare", "shape", "trait_name", "trait_value"),
    [
        (
            "kpool_prefill_topk",
            dsa_generator.prepare_kpool_prefill_topk,
            {"batch": 1, "prefix_tokens": 0, "query_tokens_per_sequence": 64},
            "has_prefill_plan",
            True,
        ),
        (
            "kpool_decode_topk",
            dsa_generator.prepare_kpool_decode_topk,
            {"batch": 1, "q_len_per_req": 4, "sequence_length": 8192},
            "q_len",
            4,
        ),
    ],
)
def test_kpool_generator_selection_traits_match_operation_api(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    prepare: Any,
    shape: dict[str, object],
    trait_name: str,
    trait_value: object,
) -> None:
    captured: dict[str, object] = {}

    def capture_selection(*_args, traits, **_kwargs):
        captured.update(traits)
        raise RuntimeError("selection captured")

    monkeypatch.setattr(dsa_generator, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(dsa_generator, "_select_registration", capture_selection)
    request = _request(mode, {**_GLM53_CONFIG, **shape})

    with pytest.raises(RuntimeError, match="selection captured"):
        prepare(request, None)

    assert captured[trait_name] == trait_value


def test_dsa_generator_builds_page_aligned_prefill_metadata() -> None:
    metadata = dsa_generator._prefill_metadata(
        2,
        0,
        68,
        config=_config(),
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


def test_dsa_generator_uses_configured_index_page_stride(monkeypatch) -> None:
    original_empty = torch.empty
    config = _config()

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
        config=_config(),
        device="cpu",
    )

    assert pages_per_request == 2
    assert index_table.tolist() == [[0, 1], [2, 3]]
    assert kv_table is index_table


def test_dsa_generator_uses_positive_decode_append_pages() -> None:
    table, index_pages = dsa_generator._decode_append_page_table(
        3,
        8192,
        config=_config(),
        device="cpu",
    )

    assert table.shape == (3, 128)
    assert table[:, 0].tolist() == [1, 2, 3]
    assert torch.all(table > 0)
    assert int(table.max()) < index_pages
    assert index_pages == 4


def test_dsa_generator_matches_configured_decode_tail_widths() -> None:
    config = _config()

    assert dsa_generator._decode_tail_width(1, config=config) == 4
    assert dsa_generator._decode_tail_width(4, config=config) == 8


def test_dsa_generator_preserves_fused_projection_weight_stride(monkeypatch) -> None:
    def cpu_randn(shape, *, generator, dtype):
        _ = generator
        return torch.randn(shape, dtype=dtype)

    monkeypatch.setattr(dsa_generator, "_randn", cpu_randn)
    weights = dsa_generator._kpool_scoring_weights(
        3,
        config=_config(),
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
        config=_config(),
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
