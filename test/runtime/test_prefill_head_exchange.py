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


import sys
from pathlib import Path

import pytest
import torch

from tokenspeed.runtime.execution.query_shard import scatter_count
from tokenspeed.runtime.layers.attention import prefill_head_exchange as exchange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")


@pytest.mark.parametrize("tokens", [1, 9, 1025])
@pytest.mark.parametrize("rank", [0, 3, 7])
def test_request_rows_and_padded_sink(monkeypatch, tokens, rank):
    group = tuple(range(8))
    counts = scatter_count(tokens, 8)
    start = sum(counts[:rank])
    rows = slice(start, start + counts[rank])
    full_q = torch.arange(tokens * 64 * 2).reshape(tokens, 64, 2).float()
    q = full_q[:, rank * 8 : (rank + 1) * 8].contiguous()
    sink = torch.arange(64).float()
    local_sink = torch.full((64,), -torch.inf)
    local_sink[:8] = sink[rank * 8 : (rank + 1) * 8]
    indices = torch.arange(tokens * 4, dtype=torch.int32).reshape(tokens, 4)
    workspace = torch.randn(16, 1, 2)
    expected = full_q + sink[None, :, None]

    def transpose(value, actual_group, splits, backend):
        torch.testing.assert_close(value, q.flatten(1))
        assert actual_group == group and splits == counts
        return full_q[rows].flatten(1)

    def gather(value, actual_group, dim, backend):
        assert actual_group == group and dim == 0
        torch.testing.assert_close(value, sink[rank * 8 : (rank + 1) * 8])
        return sink

    def attention(query, *args, lse):
        assert lse is None
        torch.testing.assert_close(query, full_q[rows])
        torch.testing.assert_close(args[6], sink)
        assert args[-2] is workspace
        torch.testing.assert_close(args[-1], indices[rows])
        return query + args[6][None, :, None]

    def scatter(value, actual_group, splits, backend):
        assert actual_group == group and splits == counts
        torch.testing.assert_close(value, expected[rows])
        return expected[:, rank * 8 : (rank + 1) * 8].contiguous()

    monkeypatch.setattr(exchange, "all_to_all_transpose", transpose)
    monkeypatch.setattr(exchange, "all_gather", gather)
    monkeypatch.setattr(exchange, "all_to_all_head_scatter", scatter)
    monkeypatch.setattr(exchange.dsv41, "selected_attention", attention)
    actual = exchange.selected_prefill_with_head_exchange(
        q,
        workspace,
        indices,
        local_sink,
        softmax_scale=0.1,
        query_chunk_size=256,
        group=group,
        rank=rank,
    )
    torch.testing.assert_close(actual, expected[:, rank * 8 : (rank + 1) * 8])


@pytest.mark.parametrize(
    "tokens, heads, size, threshold, expected",
    [
        (8192, 64, 1, 1024, False),
        (8192, 32, 2, 1024, True),
        (8192, 16, 4, 1024, True),
        (8192, 8, 8, 1024, True),
        (1023, 8, 8, 1024, False),
        (1024, 8, 8, 1024, True),
        (2047, 8, 8, 2048, False),
        (2048, 8, 8, 2048, True),
        (256, 8, 8, 256, True),
        (7, 8, 8, 1, False),
        (8, 8, 8, 1, True),
        (8192, 16, 8, 1024, False),
    ],
)
def test_explicit_request_threshold(tokens, heads, size, threshold, expected):
    assert (
        exchange.should_exchange_prefill_heads(
            num_queries=tokens,
            local_heads=heads,
            group_size=size,
            min_queries=threshold,
            swa_min_queries=4096,
            has_global=True,
        )
        is expected
    )


@pytest.mark.parametrize(
    "tokens, minimum, swa_minimum, has_global, expected",
    [
        (1024, 1024, 4096, False, False),
        (2048, 1024, 4096, False, False),
        (4096, 1024, 4096, False, True),
        (1024, 1024, 4096, True, True),
        (1024, 1024, 1024, False, True),
        (4096, 8192, 4096, False, False),
        (4096, 8192, 4096, True, False),
    ],
)
def test_swa_and_global_thresholds(tokens, minimum, swa_minimum, has_global, expected):
    assert (
        exchange.should_exchange_prefill_heads(
            num_queries=tokens,
            local_heads=8,
            group_size=8,
            min_queries=minimum,
            swa_min_queries=swa_minimum,
            has_global=has_global,
        )
        is expected
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
