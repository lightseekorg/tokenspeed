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
import tokenspeed_kernel.benchmark.generators.dsv41 as dsv41_generator
import torch
from tokenspeed_kernel.benchmark.harness import BenchmarkCaseError, BenchmarkRequest

_DSV41_CONFIG = {
    "model_profile": "dsv41_flash_tp4",
    "local_heads": 16,
    "swa_window": 128,
    "index_topk": 512,
    "index_heads": 32,
    "candidate_block_size": 8,
    "candidate_topk_blocks": 2048,
}


def _request(mode: str, parameters: dict[str, Any]) -> BenchmarkRequest:
    return BenchmarkRequest(
        family="attention",
        mode=mode,
        parameters={**_DSV41_CONFIG, **parameters},
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )


def test_spread_rows_selects_distinct_sorted_visible_rows() -> None:
    rows, lens = dsv41_generator.spread_rows(torch.tensor([0, 3, 512, 5000]), 512)

    assert lens.tolist() == [0, 3, 512, 512]
    assert (rows[0] == -1).all()
    assert rows[1, :4].tolist() == [0, 1, 2, -1]
    assert rows[2].tolist() == list(range(512))
    long_history = rows[3]
    assert (long_history[1:] > long_history[:-1]).all()
    assert 0 <= long_history.min() and long_history.max() < 5000


def test_prefill_swa_indices_address_retained_prefix_and_chunk() -> None:
    # Prefix 200 retains positions 73..199; the chunk starts at row 127.
    indices = dsv41_generator.prefill_swa_indices(
        torch.tensor([200, 201]), prefix_begin=73, swa_window=128
    )

    assert indices[0].tolist() == list(range(0, 128))
    assert indices[1].tolist() == list(range(1, 129))


def test_prefill_swa_indices_cold_chunk_masks_missing_history() -> None:
    indices = dsv41_generator.prefill_swa_indices(
        torch.tensor([0, 2]), prefix_begin=0, swa_window=4
    )

    assert indices.tolist() == [[-1, -1, -1, 0], [-1, 0, 1, 2]]


@pytest.mark.parametrize(
    ("mode", "prepare", "parameters", "expected_traits"),
    [
        (
            "dsv41_index_topk",
            dsv41_generator.prepare_dsv41_index_topk,
            {
                "batch": 1,
                "q_len_per_req": 6,
                "context_length": 8192,
                "compress_ratio": 2,
                "selection": "full",
                "table_pages": 64,
                "query_chunk_size": 256,
            },
            {
                "index_heads": 32,
                "index_k_format": "mxfp4",
                "index_shards": 1,
                "native_indexer": False,
            },
        ),
        (
            "dsv4_prefill",
            dsv41_generator.prepare_dsv4_prefill,
            {"prefix_tokens": 0, "tokens": 64, "compress_ratio": 1},
            {"num_q_heads": 16, "selected_width": 640, "sinks": True},
        ),
        (
            "dsv4_prefill",
            dsv41_generator.prepare_dsv4_prefill,
            {"prefix_tokens": 0, "tokens": 64, "compress_ratio": 0},
            {"selected_width": 128},
        ),
    ],
)
def test_dsv41_generator_selection_traits_match_operation_api(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    prepare: Any,
    parameters: dict[str, object],
    expected_traits: dict[str, object],
) -> None:
    captured: dict[str, object] = {}

    def capture_selection(_request, _platform, selected_mode, _roles, traits):
        captured["mode"] = selected_mode
        captured.update(traits)
        raise RuntimeError("selection captured")

    monkeypatch.setattr(dsv41_generator, "_select", capture_selection)

    with pytest.raises(RuntimeError, match="selection captured"):
        prepare(_request(mode, parameters), None)

    assert captured["mode"] == mode
    assert expected_traits.items() <= captured.items()


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"compress_ratio": 0}, "compress their history"),
        ({"table_pages": 63}, "table_pages must cover"),
        ({"selection": "unknown"}, "Implemented DSV4.1 selection"),
    ],
)
def test_dsv41_index_topk_rejects_inconsistent_cases(
    parameters: dict[str, object], message: str
) -> None:
    request = _request(
        "dsv41_index_topk",
        {
            "batch": 1,
            "q_len_per_req": 6,
            "context_length": 8192,
            "compress_ratio": 2,
            "selection": "full",
            "table_pages": 64,
            "query_chunk_size": 256,
            **parameters,
        },
    )

    with pytest.raises(BenchmarkCaseError, match=message):
        dsv41_generator.prepare_dsv41_index_topk(request, None)


def test_dsv41_generator_rejects_unimplemented_model_profile() -> None:
    request = _request(
        "dsv4_prefill",
        {
            "model_profile": "unknown",
            "prefix_tokens": 0,
            "tokens": 64,
            "compress_ratio": 1,
        },
    )

    with pytest.raises(BenchmarkCaseError, match="model_profile"):
        dsv41_generator.prepare_dsv4_prefill(request, None)
