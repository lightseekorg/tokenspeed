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

"""Expert placement dispatch tables: the token-pure replica choice."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.moe import ExpertDispatch, dispatch_topk_ids


def _dispatch() -> ExpertDispatch:
    # Three logical experts: 0 on slots 0 and 3, 1 on slot 1, 2 on 2, 4, 5.
    return ExpertDispatch(
        replicas=torch.tensor([[0, 3, -1], [1, -1, -1], [2, 4, 5]], dtype=torch.int32),
        num_replicas=torch.tensor([2, 1, 3], dtype=torch.int32),
    )


def test_replica_is_a_pure_function_of_row_and_route_rank():
    dispatch = _dispatch()
    assert dispatch.max_replicas == 3
    ids = torch.tensor([[2, 0], [2, 0], [2, 1]], dtype=torch.int64)
    physical = dispatch_topk_ids(ids, dispatch)
    # Row r, route rank k picks replicas[logical, (r + k) % count].
    assert physical.tolist() == [[2, 3], [4, 0], [5, 1]]
    assert physical.dtype == torch.int64
    # Every rank running the same routing over the same rows agrees.
    assert torch.equal(physical, dispatch_topk_ids(ids, dispatch))


def test_dispatch_tables_are_validated():
    good = _dispatch()
    with pytest.raises(ValueError, match="int32"):
        ExpertDispatch(good.replicas.long(), good.num_replicas)
    with pytest.raises(ValueError, match="num_replicas"):
        ExpertDispatch(good.replicas, good.num_replicas[:2])
    with pytest.raises(ValueError, match="contiguous"):
        ExpertDispatch(good.replicas.t().contiguous().t(), good.num_replicas)
