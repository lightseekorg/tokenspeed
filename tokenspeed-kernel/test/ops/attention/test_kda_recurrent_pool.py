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

"""Retained T1 page ownership, strides and CUDA graph refresh."""

import pytest
import torch
from kda_retained_reference import check_prefix, commit, make_case, reference, verify

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)


# K3 decode shapes (TP8 rank): 12 heads, K=V=128; lower_bound from config.


@pytest.mark.parametrize("batch", [1, 4, 16])
@pytest.mark.parametrize("cross_page", [False, True])
def test_retained_t1_strided_pool_matches_reference(batch, cross_page):
    case = make_case(batch, 1, cross_page=cross_page, strided=True, ragged=True)
    expected = reference(case)
    state_before, conv_before = case["state"].clone(), case["conv"].clone()
    output = verify(case)
    torch.testing.assert_close(output[0].float(), expected[0], atol=2e-4, rtol=1e-2)
    accepted = torch.ones(batch, device="cuda", dtype=torch.int32)
    commit(case, accepted, ended_slots=())
    check_prefix(case, accepted, expected)
    live = set(case["records"].history.live_state_indices.tolist()) - {0}
    for page in range(case["state"].shape[0]):
        if page not in live:
            torch.testing.assert_close(
                case["state"][page], state_before[page], atol=0, rtol=0
            )
            torch.testing.assert_close(
                case["conv"][page], conv_before[page], atol=0, rtol=0
            )


def test_retained_verify_graph_refreshes_input_and_lengths():
    case = make_case(4, 1, strided=True)
    verify(case)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = verify(case)
    case["raw"].mul_(0.5)
    # Every request slot's accepted count changes between capture and replay.
    case["records"].history.length.fill_(1)
    expected = reference(case)
    graph.replay()
    torch.testing.assert_close(output[0].float(), expected[0], atol=2e-4, rtol=1e-2)
