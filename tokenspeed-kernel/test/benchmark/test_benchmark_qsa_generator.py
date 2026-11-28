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

import pytest
import tokenspeed_kernel.benchmark.generators.qsa as qsa_generator
import torch
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
)


def _request(**parameters: object) -> BenchmarkRequest:
    return BenchmarkRequest(
        family="attention",
        mode="qsa_sparse_attention",
        parameters={
            "model_profile": "qwen38_flash_tp1",
            "dtype": "bfloat16",
            "kv_dtype": "bfloat16",
            **parameters,
        },
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )


def test_selected_slots_follow_each_rows_prefix() -> None:
    positions = torch.tensor([3, 5, 9, 9])
    request_index = torch.tensor([0, 0, 1, 1])
    slots = qsa_generator.qsa_selected_slots(
        positions,
        request_index,
        region_slots=16,
        width=6,
        generator=torch.Generator().manual_seed(0),
    )

    assert slots.shape == (4, 6) and slots.dtype == torch.int32
    for row, (position, request) in enumerate(zip(positions, request_index)):
        valid = slots[row][slots[row] >= 0]
        start = 2 + int(request) * 16
        assert valid.numel() == min(int(position), 6)
        assert torch.all(slots[row][valid.numel() :] == -1)
        assert torch.all(valid[1:] > valid[:-1])
        assert torch.all((valid >= start) & (valid < start + int(position)))
    # A row whose prefix fits the width selects every slot of it.
    assert slots[0].tolist() == [2, 3, 4, -1, -1, -1]


@pytest.mark.parametrize(
    "parameters",
    [
        {"phase": "decode", "batch": 2, "q_len": 9, "context_tokens": 8},
        {"phase": "prefill", "batch": 2, "extend_tokens": 9, "context_tokens": 8},
    ],
)
def test_query_rows_must_fit_the_context(parameters) -> None:
    with pytest.raises(BenchmarkCaseError, match="must not exceed") as error:
        qsa_generator.prepare_qsa_sparse_attention(_request(**parameters), None)
    assert error.value.status is BenchmarkStatus.INVALID_CASE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    "parameters, rows",
    [
        # Two prompts extending a 600-token cached prefix by 40 rows each.
        ({"phase": "prefill", "batch": 2, "extend_tokens": 40}, 80),
        # Three requests verifying 4 tokens each.
        ({"phase": "decode", "batch": 3, "q_len": 4}, 12),
    ],
)
@pytest.mark.parametrize(
    "model_profile, q_heads", [("qwen38_flash_tp1", 24), ("qwen38_flash_tp2", 12)]
)
def test_qsa_benchmark_runs_each_phase_at_each_profile(
    parameters, rows, model_profile, q_heads
) -> None:
    request = _request(model_profile=model_profile, context_tokens=640, **parameters)
    prepared = qsa_generator.prepare_qsa_sparse_attention(request, None)
    output = prepared.invocation.invoke()

    assert output.shape[0] == rows
    assert output.numel() == rows * q_heads * 256
    assert torch.isfinite(output).all()
