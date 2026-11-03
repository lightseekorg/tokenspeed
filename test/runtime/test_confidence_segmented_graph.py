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

"""One target-token graph must preserve request identity across changing B."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.execution.verify_layout import VerifyQueryLayout
from tokenspeed.runtime.layers.attention.backends.paged.group_tables import (
    GroupTableSpec,
    GroupTableStacks,
)


def _reference_lengths(rates, active, budget):
    lengths = [int(on) for on in active]
    scores = []
    for request, (row, on) in enumerate(zip(rates, active)):
        if not on:
            continue
        product = 1.0
        for position, value in enumerate(row):
            product *= value
            scores.append((product, request, position))
    scores.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, request, _ in scores[: max(0, budget - sum(lengths))]:
        lengths[request] += 1
    return lengths


@pytest.mark.parametrize("requests", [1, 2, 3, 4])
def test_expansion_keeps_the_live_request_extent(requests):
    layout = VerifyQueryLayout(4, 8, 4, "cpu")
    rates = torch.tensor(
        [[0.95, 0.9, 0.8], [0.8, 0.6, 0.5], [0.7, 0.5, 0.4], [0.6, 0.4, 0.3]]
    )
    active = torch.arange(4) < requests
    layout.refresh(rates, torch.ones(4, dtype=torch.bool), active)
    values = torch.arange(16 * 3).reshape(16, 3).float()
    compact = layout.pack(values, token_dim=0)
    actual = layout.expand(compact, token_dim=0, request_count=requests)
    expected = values[: requests * 4].clone()
    lengths = _reference_lengths(rates.tolist(), active.tolist(), 8)
    for request in range(requests):
        expected[request * 4 + lengths[request] : (request + 1) * 4].zero_()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        layout.expand(compact.T, token_dim=1, request_count=requests),
        expected.T,
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay")
def test_one_query_graph_replays_after_request_count_and_slots_change():
    device = "cuda"
    layout = VerifyQueryLayout(4, 8, 4, device)
    tables = GroupTableStacks(
        [
            GroupTableSpec(
                "state", block_granularity=4, kernel_page_size=4, max_num_pages=1
            )
        ],
        max_bs=4,
        max_tokens_per_req=4,
        max_extend_tokens=0,
        device=device,
    )
    inputs = torch.zeros(16, 8, device=device)
    weight = torch.arange(8 * 5, device=device).reshape(8, 5).float() / 128
    rates = torch.tensor(
        [[0.95, 0.9, 0.8], [0.8, 0.6, 0.5], [0.7, 0.5, 0.4], [0.6, 0.4, 0.3]],
        device=device,
    )
    ready = torch.ones(4, dtype=torch.bool, device=device)
    layout.refresh(rates, ready, ready)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            tags = tables.tables[0, :, 0].repeat_interleave(4)[:, None]
            torch.sin(layout.pack(inputs + tags, token_dim=0)) @ weight
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        tags = tables.tables[0, :, 0].repeat_interleave(4)[:, None]
        compact = torch.sin(layout.pack(inputs + tags, token_dim=0)) @ weight
    torch.cuda.current_stream().wait_stream(stream)
    for step, requests in enumerate([4, 2, 3, 1, 4]):
        live_inputs = (
            torch.arange(16 * 8, device=device).reshape(16, 8).float() / 37 + step
        )
        inputs.copy_(live_inputs)
        active = torch.arange(4, device=device) < requests
        shifted_rates = rates.roll(step, 0)
        page_ids = (
            torch.arange(1, requests + 1, dtype=torch.int32, device=device) + step * 7
        )
        tables.fill(4, requests, {"state": page_ids[:, None]})
        torch.testing.assert_close(
            tables.tables[0, requests:],
            torch.zeros_like(tables.tables[0, requests:]),
            atol=0,
            rtol=0,
        )
        layout.refresh(shifted_rates, ready, active)
        graph.replay()
        actual = layout.expand(compact, token_dim=0, request_count=requests)
        live_tags = page_ids.repeat_interleave(4)[:, None]
        expected = (torch.sin(live_inputs[: requests * 4] + live_tags) @ weight).clone()
        lengths = _reference_lengths(
            shifted_rates.cpu().tolist(), active.cpu().tolist(), 8
        )
        for request in range(requests):
            expected[request * 4 + lengths[request] : (request + 1) * 4].zero_()
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
