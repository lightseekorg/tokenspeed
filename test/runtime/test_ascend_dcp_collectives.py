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

"""Verify head routing and empty-shard semantics independently of HCCL."""

import pytest
import torch

from tokenspeed.runtime.layers.attention.dcp.ascend import AscendDCPContext


@pytest.mark.parametrize("degree", [1, 2, 8])
def test_query_heads_follow_group_rank_order(monkeypatch, degree):
    ctx = AscendDCPContext(
        degree=degree, rank=0, ranks=tuple(range(degree)), auxiliary_namespace="test"
    )
    queries = torch.arange(degree * 3 * 2 * 4).reshape(degree, 3, 2, 4)

    def gather(query, *, auxiliary):
        assert auxiliary
        torch.testing.assert_close(query, queries[0])
        return queries.flatten(0, 1)

    monkeypatch.setattr(ctx, "all_gather", gather)
    result = ctx.gather_query(queries[0])
    for r in range(degree):
        torch.testing.assert_close(result[:, r * 2 : (r + 1) * 2], queries[r])


@pytest.mark.parametrize("degree", [1, 2, 8])
def test_combine_routes_heads_and_masks_poisoned_empty_shards(monkeypatch, degree):
    torch.manual_seed(10)
    tokens, local_heads, dim = 3, 2, 4
    outputs = torch.randn(degree, tokens, degree * local_heads, dim)
    lse = torch.randn(degree, tokens, degree * local_heads)
    lse[:, 0] = -torch.inf  # Whole padding query.
    lse[0, 1] = -torch.inf  # Valid query with an empty rank.
    outputs[torch.isneginf(lse)] = torch.nan
    weights = torch.softmax(lse, dim=0).nan_to_num()
    expected = (outputs.nan_to_num() * weights[..., None]).sum(0)
    for rank in range(degree):
        ctx = AscendDCPContext(
            degree=degree,
            rank=rank,
            ranks=tuple(range(degree)),
            auxiliary_namespace="test",
        )
        clean = outputs.nan_to_num()

        def exchange(recv, send, *, auxiliary):
            source = lse if auxiliary else clean
            expected_send = source[rank].transpose(0, 1).reshape(send.shape)
            torch.testing.assert_close(send, expected_send)
            owned = source[
                :, :, rank * local_heads : (rank + 1) * local_heads
            ].transpose(1, 2)
            recv.copy_(owned)

        monkeypatch.setattr(ctx, "all_to_all", exchange)

        def merge(o, logs):
            w = torch.softmax(logs, dim=0).nan_to_num()
            return (o * w[..., None]).sum(0)

        result = ctx.combine(outputs[rank], lse[rank], merge=merge)
        torch.testing.assert_close(
            result, expected[:, rank * local_heads : (rank + 1) * local_heads]
        )
