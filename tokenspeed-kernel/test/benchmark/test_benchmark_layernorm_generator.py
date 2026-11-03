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
import torch
from tokenspeed_kernel.benchmark.generators.layernorm import (
    prepare_weightless_rmsnorm_gated_sigmoid,
)
from tokenspeed_kernel.benchmark.harness import BenchmarkCaseError, BenchmarkRequest
from tokenspeed_kernel.platform import current_platform


def _request(**updates: object) -> BenchmarkRequest:
    return BenchmarkRequest(
        family="layernorm",
        mode="weightless_rmsnorm_gated_sigmoid",
        parameters={
            "tokens": 3,
            "num_heads": 4,
            "head_dim": 64,
            "gate_row_stride": 640,
            "eps": 1e-5,
            "dtype": "bfloat16",
            **updates,
        },
        solution=None,
        registration=None,
        cold_cache=True,
        seed=11,
    )


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"gate_row_stride": 255}, "gate_row_stride"),
        ({"eps": 0.0}, "eps"),
        ({"tokens": 0}, "tokens"),
        ({"dtype": "float16"}, "dtype"),
    ],
)
def test_gated_sigmoid_rejects_invalid_parameters(updates, match) -> None:
    with pytest.raises(BenchmarkCaseError, match=match):
        prepare_weightless_rmsnorm_gated_sigmoid(_request(**updates), None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_gated_sigmoid_gates_each_normalized_head_into_a_bounded_output() -> None:
    prepared = prepare_weightless_rmsnorm_gated_sigmoid(_request(), current_platform())
    out = prepared.invocation.invoke()

    assert prepared.registration.mode == "weightless_rmsnorm_gated_sigmoid"
    assert out.shape == (3, 256) and out.dtype == torch.bfloat16
    # Each head is unit-RMS before a sigmoid gate in (0, 1), so its RMS stays below one.
    head_rms = out.float().view(3, 4, 64).square().mean(-1).sqrt()
    assert torch.all(head_rms > 0) and torch.all(head_rms < 1.0 + 1e-2)
