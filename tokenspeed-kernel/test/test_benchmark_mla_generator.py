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
import tokenspeed_kernel.benchmark.generators.mla as mla_generator
import torch
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
)
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_KIMI_K3_CONFIG = {
    "model_profile": "kimi_k3_tp8",
    "local_heads": 12,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "kv_page_size": 64,
}
_FP8 = "float8_e4m3fn"
_DECODE_SHAPE = {
    "requests": 2,
    "rows_per_request": 4,
    "cache_length": 50250,
    "max_context_len": 65536,
    "q_dtype": _FP8,
    "kv_cache_dtype": _FP8,
}
_PREFILL_SHAPE = {
    "batch": 1,
    "query_tokens_per_sequence": 256,
    "kv_tokens_per_sequence": 1024,
    "is_causal": False,
    "return_lse": True,
    "dtype": _FP8,
}
_NORMALIZE_SHAPE = {
    "tokens": 1,
    "dtype": "bfloat16",
    "prepare_absorbed_query": True,
    "eps": 1e-5,
}


def _request(mode: str, parameters: dict[str, Any]) -> BenchmarkRequest:
    return BenchmarkRequest(
        family="attention",
        mode=mode,
        parameters={**_KIMI_K3_CONFIG, **parameters},
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )


def _cpu_randn(shape, *, generator, dtype):
    _ = generator
    return torch.randn(shape, dtype=torch.float32).to(dtype)


def _config() -> mla_generator._MLAConfig:
    return mla_generator._resolve_config(_request("test", {}))


def test_mla_generator_rejects_unsupported_validation() -> None:
    request = _request(
        "mla_prefill",
        {**_PREFILL_SHAPE, "validation": {"runs": 1}},
    )

    with pytest.raises(BenchmarkCaseError, match="not implemented"):
        mla_generator.prepare_mla_prefill(request, None)


def test_mla_generator_rejects_unimplemented_model_profile() -> None:
    request = _request(
        "mla_prefill",
        {**_PREFILL_SHAPE, "model_profile": "unimplemented"},
    )

    with pytest.raises(BenchmarkCaseError, match="Implemented MLA model_profile"):
        mla_generator.prepare_mla_prefill(request, None)


@pytest.mark.parametrize(
    ("mode", "prepare", "parameters", "match"),
    [
        (
            "mla_prefill",
            mla_generator.prepare_mla_prefill,
            {**_PREFILL_SHAPE, "is_causal": True},
            "query and KV lengths must match",
        ),
        (
            "mla_decode_with_kvcache",
            mla_generator.prepare_mla_decode,
            {**_DECODE_SHAPE, "cache_length": 3},
            "cover every query row",
        ),
        (
            "mla_decode_with_kvcache",
            mla_generator.prepare_mla_decode,
            {**_DECODE_SHAPE, "max_context_len": 4096},
            "must not exceed max_context_len",
        ),
        (
            "mla_decode_with_kvcache",
            mla_generator.prepare_mla_decode,
            {**_DECODE_SHAPE, "kv_cache_dtype": "float16"},
            "Implemented MLA kv_cache_dtype",
        ),
        (
            "mla_decode_projected_value",
            mla_generator.prepare_mla_decode_projected_value,
            {**_DECODE_SHAPE, "output_gate": 1},
            "output_gate must be a boolean",
        ),
    ],
)
def test_mla_generator_rejects_invalid_shapes(
    mode: str,
    prepare: Any,
    parameters: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(BenchmarkCaseError, match=match) as error:
        prepare(_request(mode, parameters), None)

    assert error.value.status is BenchmarkStatus.INVALID_CASE


def test_mla_generator_builds_packed_qkv_gate_views(monkeypatch) -> None:
    monkeypatch.setattr(mla_generator, "_randn", _cpu_randn)
    query, kv, gate = mla_generator._packed_qkv_gate(
        4,
        config=_config(),
        generator=None,
    )

    # [q_a 1536 | kv_a 512 + rope 64 | gate 12 * 128] per token row.
    assert query.shape == (4, 1536)
    assert kv.shape == (4, 512)
    assert gate.shape == (4, 1536)
    assert query.stride() == (3648, 1)
    assert kv.stride() == query.stride()
    assert gate.stride() == query.stride()
    assert kv.storage_offset() == 1536
    assert gate.storage_offset() == 2112


def test_mla_generator_builds_verify_page_table() -> None:
    page_table, cache_seqlens, pages = mla_generator._decode_page_table(
        2,
        4,
        130,
        256,
        config=_config(),
        device="cpu",
    )

    assert pages == 6
    assert page_table.shape == (8, 4)
    assert page_table[:4].tolist() == [[0, 1, 2, 0]] * 4
    assert page_table[4:].tolist() == [[3, 4, 5, 0]] * 4
    assert cache_seqlens.tolist() == [127, 128, 129, 130] * 2


def _capture_generator_selection(
    monkeypatch: pytest.MonkeyPatch,
    prepare: Any,
    request: BenchmarkRequest,
) -> tuple[object, dict[str, object]]:
    captured: dict[str, object] = {}

    def capture(_request, _platform, *, signature_roles, traits):
        captured["signature"] = format_signature(
            **{
                role: dense_tensor_format(dtype)
                for role, dtype in signature_roles.items()
            }
        )
        captured["traits"] = traits
        raise RuntimeError("selection captured")

    monkeypatch.setattr(mla_generator, "_randn", _cpu_randn)
    monkeypatch.setattr(mla_generator, "_generator", lambda _seed: None)
    monkeypatch.setattr(mla_generator, "load_builtin_kernels", lambda: None)
    monkeypatch.setattr(mla_generator, "_select_registration", capture)
    with pytest.raises(RuntimeError, match="selection captured"):
        prepare(request, None)
    return captured["signature"], captured["traits"]


def _capture_operation_selection(
    monkeypatch: pytest.MonkeyPatch,
    call: Any,
) -> tuple[object, dict[str, object]]:
    from tokenspeed_kernel.ops.attention import mla as mla_ops

    captured: dict[str, object] = {}

    def capture(_family, _mode, signature, *, traits, **_kwargs):
        captured["signature"] = signature
        captured["traits"] = traits
        raise RuntimeError("selection captured")

    monkeypatch.setattr(mla_ops, "select_kernel", capture)
    with pytest.raises(RuntimeError, match="selection captured"):
        call(mla_ops)
    return captured["signature"], captured["traits"]


def _decode_operation_inputs(output_gate: bool) -> dict[str, object]:
    config = _config()
    page_table, cache_seqlens, pages = mla_generator._decode_page_table(
        _DECODE_SHAPE["requests"],
        _DECODE_SHAPE["rows_per_request"],
        _DECODE_SHAPE["cache_length"],
        _DECODE_SHAPE["max_context_len"],
        config=config,
        device="cpu",
    )
    rows = page_table.shape[0]
    inputs: dict[str, object] = {
        "q": torch.zeros((rows, 1, 12, 576), dtype=torch.float8_e4m3fn),
        "kv_cache": torch.zeros((pages, 64, 1, 576), dtype=torch.float8_e4m3fn),
        "page_table": page_table,
        "cache_seqlens": cache_seqlens,
        "max_seqlen_k": _DECODE_SHAPE["max_context_len"],
        "qk_nope_head_dim": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "softmax_scale": 192**-0.5,
    }
    if output_gate:
        inputs["value_weight"] = torch.zeros((12, 512, 128), dtype=torch.bfloat16)
        inputs["gate"] = torch.zeros((rows, 3648), dtype=torch.bfloat16)[:, 2112:]
        inputs["out"] = torch.empty((rows, 1536), dtype=torch.bfloat16)
    return inputs


def test_mla_decode_selection_matches_operation_api(monkeypatch) -> None:
    expected = _capture_operation_selection(
        monkeypatch,
        lambda ops: ops.mla_decode_with_kvcache(**_decode_operation_inputs(False)),
    )
    actual = _capture_generator_selection(
        monkeypatch,
        mla_generator.prepare_mla_decode,
        _request("mla_decode_with_kvcache", _DECODE_SHAPE),
    )

    assert actual == expected
    assert actual[1]["batch_size"] == 8


def test_mla_decode_projected_value_selection_matches_operation_api(
    monkeypatch,
) -> None:
    expected = _capture_operation_selection(
        monkeypatch,
        lambda ops: ops.mla_decode_with_kvcache(**_decode_operation_inputs(True)),
    )
    actual = _capture_generator_selection(
        monkeypatch,
        mla_generator.prepare_mla_decode_projected_value,
        _request(
            "mla_decode_projected_value",
            {**_DECODE_SHAPE, "output_gate": True},
        ),
    )

    assert actual == expected
    assert actual[1]["gate_kind"] == "sigmoid"


def test_mla_normalize_project_query_selection_matches_operation_api(
    monkeypatch,
) -> None:
    monkeypatch.setattr(mla_generator, "_randn", _cpu_randn)
    query, kv, _ = mla_generator._packed_qkv_gate(1, config=_config(), generator=None)
    expected = _capture_operation_selection(
        monkeypatch,
        lambda ops: ops.mla_normalize_project_query(
            query,
            kv,
            torch.ones(1536, dtype=torch.bfloat16),
            torch.ones(512, dtype=torch.bfloat16),
            torch.zeros((2304, 1536), dtype=torch.bfloat16),
            eps=1e-5,
            prepare_absorbed_query=True,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
        ),
    )
    actual = _capture_generator_selection(
        monkeypatch,
        mla_generator.prepare_mla_normalize_project_query,
        _request("mla_normalize_project_query", _NORMALIZE_SHAPE),
    )

    assert actual == expected
    assert actual[1]["split_output"] is True
    assert actual[1]["inputs_contiguous"] is True


def test_mla_prefill_selection_matches_operation_api(monkeypatch) -> None:
    fp8 = torch.float8_e4m3fn
    expected = _capture_operation_selection(
        monkeypatch,
        lambda ops: ops.mla_prefill(
            q=torch.zeros((256, 12, 192), dtype=fp8),
            k=torch.zeros((1024, 12, 192), dtype=fp8),
            v=torch.zeros((1024, 12, 128), dtype=fp8),
            cu_seqlens_q=torch.tensor([0, 256], dtype=torch.int32),
            cu_seqlens_kv=torch.tensor([0, 1024], dtype=torch.int32),
            max_seqlen_q=256,
            max_seqlen_kv=1024,
            softmax_scale=192**-0.5,
            is_causal=False,
            return_lse=True,
        ),
    )
    actual = _capture_generator_selection(
        monkeypatch,
        mla_generator.prepare_mla_prefill,
        _request("mla_prefill", _PREFILL_SHAPE),
    )

    assert actual == expected


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (_FP8, "gluon_mla_prefill_8wave_gfx950"),
        ("bfloat16", "gluon_mla_prefill_gfx950"),
    ],
)
def test_mla_prefill_generator_reports_executed_kernel_on_cdna4(
    monkeypatch, mi350_platform: PlatformInfo, dtype: str, expected: str
) -> None:
    """The reported kernel must be the one mla_prefill runs for the case."""
    load_builtin_kernels()
    if KernelRegistry.get().get_by_name(expected) is None:
        pytest.skip("gfx950 Gluon MLA prefill kernels are unavailable")

    select_registration = mla_generator._select_registration
    selected: dict[str, object] = {}

    def capture(request, platform, *, signature_roles, traits):
        selected["spec"] = select_registration(
            request, platform, signature_roles=signature_roles, traits=traits
        )
        raise RuntimeError("selection captured")

    monkeypatch.setattr(mla_generator, "_select_registration", capture)
    request = _request("mla_prefill", {**_PREFILL_SHAPE, "dtype": dtype})
    with pytest.raises(RuntimeError, match="selection captured"):
        mla_generator.prepare_mla_prefill(request, mi350_platform)

    assert selected["spec"].name == expected
