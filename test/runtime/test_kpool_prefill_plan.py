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

"""KPool prefill plans: what a CUDA build produces, and that building one on
the forward thread does not wait for the step in flight."""

from __future__ import annotations

import os
import sys
from dataclasses import fields

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.kpool import (  # noqa: E402
    build_kpool_prefill_plan,
)

_KPOOL = 4
_ROWS_PER_PAGE = 8


def _inputs(device: str) -> dict:
    # Three requests: a prefix that ends mid-pool, a fresh prompt, and a
    # prefix on a pool boundary, so the plan has completed pools and tails.
    return dict(
        prefix_lens_cpu=torch.tensor([6, 0, 8]),
        extend_lens_cpu=torch.tensor([11, 5, 9]),
        index_block_table=torch.arange(3 * 4, dtype=torch.int32).view(3, 4).to(device),
        request_slots=torch.tensor([2, 0, 1], dtype=torch.int32).to(device),
        kpool=_KPOOL,
        index_rows_per_page=_ROWS_PER_PAGE,
        token_capacity=32,
    )


def _host(value):
    return value.cpu() if isinstance(value, torch.Tensor) else value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_a_cuda_build_matches_the_host_build_without_waiting():
    expected = build_kpool_prefill_plan(**_inputs("cpu"))
    inputs = _inputs("cuda")
    # The first pinned buffer of a size comes from cudaHostAlloc, which waits.
    build_kpool_prefill_plan(**inputs)
    torch.cuda.synchronize()
    # Hold the stream behind a long kernel: an upload that waits for it
    # returns only after the kernel finished.
    torch.cuda._sleep(1 << 30)
    in_flight = torch.cuda.Event()
    in_flight.record()
    previous_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        plan = build_kpool_prefill_plan(**inputs)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)
    assert not in_flight.query()
    torch.cuda.synchronize()
    for field in fields(plan.write):
        got = _host(getattr(plan.write, field.name))
        want = getattr(expected.write, field.name)
        assert torch.equal(got, want), field.name
    for field in fields(plan):
        if field.name == "write":
            continue
        got = _host(getattr(plan, field.name))
        want = getattr(expected, field.name)
        if isinstance(want, torch.Tensor):
            assert torch.equal(got, want), field.name
        else:
            assert got == want, field.name
