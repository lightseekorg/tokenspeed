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


_K31_CONFIG = {
    "model_profile": "kimi_k31_tp8",
    "dtype": "bfloat16",
    "kv_cache_dtype": "float8_e4m3fn",
    "index_heads": 32,
    "index_head_dim": 128,
    "index_k_format": "mxfp4",
    "local_attention_heads": 12,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "kv_page_size": 64,
    "topk_tokens": 2048,
    "max_context": 1048560,
}


def _k31_config(**overrides: object) -> dsa_generator._IndexerConfig:
    request = _request(parameters={**_K31_CONFIG, **overrides})
    return dsa_generator._resolve_indexer_config(request)


def test_indexer_config_derives_kimi_k31_geometry() -> None:
    config = _k31_config()

    assert config.kv_cache_dtype is torch.float8_e4m3fn
    assert config.qk_head_dim == 576
    assert config.index_row_bytes == 68
    assert config.table_columns == 16384
    assert config.index_score_scale == pytest.approx(128**-0.5 * 32**-0.5)
    assert config.attention_softmax_scale == pytest.approx(192**-0.5)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_profile": "glm53_flash_tp4"}, "model_profile"),
        ({"index_k_format": "fp8_scaled"}, "index_k_format"),
        ({"kv_cache_dtype": "bfloat16"}, "kv_cache_dtype"),
        ({"validation": {"runs": 1}}, "not implemented"),
    ],
)
def test_indexer_config_rejects_unimplemented_values(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(BenchmarkCaseError, match=message):
        _k31_config(**overrides)


def test_indexer_config_rejects_exact_registration() -> None:
    request = BenchmarkRequest(
        family="attention",
        mode="dsa_decode_topk",
        parameters=dict(_K31_CONFIG),
        solution=None,
        registration="gluon_dsa_decode_topk_mxfp4_rows_gfx950",
        cold_cache=True,
        seed=42,
    )
    with pytest.raises(BenchmarkCaseError, match="not an exact registration"):
        dsa_generator._resolve_indexer_config(request)


@pytest.mark.parametrize(
    "shape",
    [
        {"batch": 1, "q_len_per_req": 8, "sequence_length": 2047},
        {"batch": 1, "q_len_per_req": 8, "sequence_length": 1048561},
        {"batch": 0, "q_len_per_req": 8, "sequence_length": 4096},
        {"batch": 1, "q_len_per_req": 8.0, "sequence_length": 4096},
    ],
)
def test_indexer_decode_rejects_invalid_verify_shapes(shape) -> None:
    request = _request("dsa_decode_topk", {**_K31_CONFIG, **shape})
    with pytest.raises(BenchmarkCaseError):
        dsa_generator._verify_shape(request, _k31_config())


def test_indexer_verify_rows_follow_each_window_causally() -> None:
    requests, causal_lens = dsa_generator._verify_rows(2, 3, 100, device="cpu")

    assert requests.tolist() == [0, 0, 0, 1, 1, 1]
    assert causal_lens.tolist() == [98, 99, 100, 98, 99, 100]


def test_indexer_owned_page_table_follows_runtime_cache_placement() -> None:
    config = _k31_config(max_context=64 * 20)
    scheduler = dsa_generator._scheduler_page_table(
        2, 64 * 9 + 1, config=config, device="cpu"
    )

    # Ten live pages per request at the decode width; the rest are null.
    assert scheduler.shape == (2, 20)
    assert scheduler[0, :10].tolist() == list(range(1, 11))
    assert scheduler[1, :10].tolist() == list(range(11, 21))
    assert torch.all(scheduler[:, 10:] == 0)

    rank0 = dsa_generator._owned_page_table(scheduler, kvp_size=8, kvp_rank=0)
    rank3 = dsa_generator._owned_page_table(scheduler, kvp_size=8, kvp_rank=3)

    # Rank r owns block b when (b - 1) % 8 == r, as local page (b - 1) // 8 + 1.
    assert rank0[0].tolist()[:10] == [1] + [-1] * 7 + [2, -1]
    assert rank0[1].tolist()[:10] == [-1] * 6 + [3] + [-1] * 3
    assert rank3[0].tolist()[:10] == [-1] * 3 + [1] + [-1] * 6
    assert torch.all(rank0[:, 10:] == -1)


def test_indexer_spread_slots_stay_in_each_rows_history() -> None:
    requests, causal_lens = dsa_generator._verify_rows(2, 4, 3000, device="cpu")
    slots, lens = dsa_generator._spread_slots(
        causal_lens, requests, topk=2048, region_slots=3000, device="cpu"
    )

    assert slots.shape == (8, 2048)
    assert lens.tolist() == [2048] * 8
    offsets = slots.to(torch.int64) - requests.to(torch.int64).unsqueeze(1) * 3000
    assert torch.all(offsets >= 0)
    assert torch.all(offsets < causal_lens.unsqueeze(1))
    # Spread over the history rather than its first or last top-k tokens.
    assert len(set(slots[0].tolist())) > 2000


def _capture_generator_selection(monkeypatch, mode, prepare, shape):
    captured = {}

    def capture(_request, _platform, selected_mode, signature, traits):
        captured.update(mode=selected_mode, signature=signature, traits=traits)
        raise RuntimeError("selection captured")

    monkeypatch.setattr(dsa_generator, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(dsa_generator, "_select_indexer_registration", capture)
    with pytest.raises(RuntimeError, match="selection captured"):
        prepare(_request(mode, {**_K31_CONFIG, **shape}), None)
    return captured


def _capture_operation_selection(monkeypatch, call):
    import tokenspeed_kernel.ops.attention.dsa as dsa_ops

    captured = {}

    def capture(_family, mode, signature, *, traits, **_kwargs):
        captured.update(mode=mode, signature=signature, traits=traits)
        raise RuntimeError("selection captured")

    monkeypatch.setattr(dsa_ops, "select_kernel", capture)
    with pytest.raises(RuntimeError, match="selection captured"):
        call(dsa_ops)
    return captured


def _index_operands(rows: int, slots: int = 64 * 4):
    q = torch.zeros((rows, 32, 64), dtype=torch.uint8)
    q_scales = torch.zeros((rows, 32), dtype=torch.int32)
    weights = torch.zeros((rows, 160), dtype=torch.bfloat16)[:, 128:]
    plane = torch.zeros((slots, 68), dtype=torch.uint8)
    return q, q_scales, weights, plane


def _decode_topk_call(dsa_ops):
    q, q_scales, weights, plane = _index_operands(16)
    return dsa_ops.dsa_decode_topk(
        q,
        weights,
        torch.full((2,), 4096, dtype=torch.int32),
        torch.zeros((2, 16384), dtype=torch.int32),
        page_size=64,
        topk=2048,
        softmax_scale=1.0,
        batch_invariant=False,
        q_len_per_req=8,
        index_k_cache=plane,
        q_scales=q_scales,
        slot_order="selection",
        index_k_format="mxfp4",
    )


def _index_candidates_call(dsa_ops):
    q, q_scales, weights, plane = _index_operands(48)
    return dsa_ops.dsa_index_candidates(
        q,
        weights,
        plane,
        torch.zeros((8, 16384), dtype=torch.int32),
        torch.zeros(48, dtype=torch.int32),
        torch.full((48,), 4096, dtype=torch.int32),
        page_size=64,
        topk=2048,
        softmax_scale=1.0,
        initial_tokens=0,
        local_tokens=0,
        solution=None,
        q_scales=q_scales,
        query_group_size=6,
        index_k_format="mxfp4",
    )


def _select_owned_call(dsa_ops):
    rows, query_words = 48, 48 * 12 * 576 // 4
    gathered = torch.zeros((8, rows * 2048 + query_words), dtype=torch.float32)
    return dsa_ops.dsa_select_owned_candidates(
        gathered[:, : rows * 2048].view(8, rows, 2048),
        torch.zeros((rows, 2048), dtype=torch.int32),
        torch.zeros((8, 16384), dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        rank=0,
        topk=2048,
        page_size=64,
        solution=None,
    )


def _projected_value_call(dsa_ops):
    rows = 16
    gate = torch.zeros((rows, 12 * 128), dtype=torch.bfloat16)
    return dsa_ops.dsa_decode_projected_value(
        torch.zeros((rows, 12, 576), dtype=torch.float8_e4m3fn),
        torch.zeros((64, 1, 576), dtype=torch.float8_e4m3fn),
        torch.zeros((rows, 2048), dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        1048560,
        128,
        512,
        64,
        1.0,
        64,
        torch.zeros((12, 128, 512), dtype=torch.bfloat16).transpose(1, 2),
        gate=gate,
        out=torch.empty_like(gate),
        q_len_per_req=8,
        slot_order="selection",
    )


def _dcp_records_call(dsa_ops):
    rows = 48
    return dsa_ops.dsa_decode_dcp_records(
        q=torch.zeros((8, rows, 12, 576), dtype=torch.float8_e4m3fn),
        kv_cache=torch.zeros((64, 1, 576), dtype=torch.float8_e4m3fn),
        topk_slots=torch.zeros((rows, 2048), dtype=torch.int32),
        topk_lens=torch.zeros(rows, dtype=torch.int32),
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=1.0,
        page_size=64,
        q_len_per_req=6,
        slot_order="selection",
    )


@pytest.mark.parametrize(
    ("mode", "prepare", "shape", "operation"),
    [
        (
            "dsa_decode_topk",
            dsa_generator.prepare_dsa_decode_topk,
            {"batch": 2, "q_len_per_req": 8, "sequence_length": 4096},
            _decode_topk_call,
        ),
        (
            "dsa_index_candidates",
            dsa_generator.prepare_dsa_index_candidates,
            {"batch": 8, "q_len_per_req": 6, "sequence_length": 4096, "kvp_size": 8},
            _index_candidates_call,
        ),
        (
            "dsa_select_owned_candidates",
            dsa_generator.prepare_dsa_select_owned_candidates,
            {"batch": 8, "q_len_per_req": 6, "sequence_length": 4096, "kvp_size": 8},
            _select_owned_call,
        ),
        (
            "dsa_decode_projected_value",
            dsa_generator.prepare_dsa_decode_projected_value,
            {"batch": 2, "q_len_per_req": 8, "sequence_length": 4096},
            _projected_value_call,
        ),
        (
            "dsa_decode_dcp_records",
            dsa_generator.prepare_dsa_decode_dcp_records,
            {"batch": 8, "q_len_per_req": 6, "sequence_length": 4096, "kvp_size": 8},
            _dcp_records_call,
        ),
    ],
)
def test_indexer_generator_selection_matches_operation_api(
    monkeypatch: pytest.MonkeyPatch, mode, prepare, shape, operation
) -> None:
    generator = _capture_generator_selection(monkeypatch, mode, prepare, shape)
    operation_selection = _capture_operation_selection(monkeypatch, operation)

    assert generator == operation_selection
