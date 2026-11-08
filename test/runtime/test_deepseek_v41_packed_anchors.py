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

"""V4.1 packed anchor addressing and graph replay."""

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    return torch.device(request.param)


def test_packed_anchors_keep_fixed_output_stride(device):
    width = 4
    # One complete prefill, one incomplete prefill, then three decode requests.
    tokens = torch.arange(100, 113, dtype=torch.int32, device=device)
    accepted = torch.tensor([1, 0, 1, 3, 2], dtype=torch.int32, device=device)
    positions = torch.tensor([10, 20, 21, 22, 30, 31, -1, -1], device=device)
    indptr = torch.tensor([0, 1, 4, 6], device=device)
    proposals = torch.full((5, 6), -7, dtype=torch.int32, device=device)
    starts = torch.empty(3, dtype=torch.int64, device=device)
    dsv41.dspark_anchors(
        tokens,
        accepted,
        positions,
        2,
        width,
        proposals,
        starts,
        num_prefill_outputs=1,
        decode_query_indptr=indptr,
    )
    assert proposals.tolist() == [[100] * 6, [-7] * 6, [101] * 6, [107] * 6, [110] * 6]
    assert starts.tolist() == [10, 22, 31]


def test_packed_anchors_replay_with_new_lengths():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    device = torch.device("cuda")
    indptr = torch.tensor([0, 4, 5], device=device)
    positions = torch.tensor([10, 11, 12, 13, 20, -1, -1, -1], device=device)
    accepted = torch.ones(2, dtype=torch.int32, device=device)
    tokens = torch.arange(8, dtype=torch.int32, device=device)
    proposals = torch.empty((2, 4), dtype=torch.int32, device=device)
    starts = torch.empty(2, dtype=torch.int64, device=device)

    def run():
        dsv41.dspark_anchors(
            tokens,
            accepted,
            positions,
            0,
            4,
            proposals,
            starts,
            num_prefill_outputs=0,
            decode_query_indptr=indptr,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for offsets, packed, accepts, expected_starts in [
        ([0, 4, 5], [10, 11, 12, 13, 20], [4, 1], [13, 20]),
        ([0, 1, 5], [10, 20, 21, 22, 23], [1, 3], [10, 22]),
        ([0, 4, 5], [10, 11, 12, 13, 20], [2, 1], [11, 20]),
    ]:
        indptr.copy_(torch.tensor(offsets, device=device))
        positions[:5].copy_(torch.tensor(packed, device=device))
        accepted.copy_(torch.tensor(accepts, dtype=torch.int32, device=device))
        graph.replay()
        assert starts.tolist() == expected_starts
        expected_bonus = tokens[
            torch.tensor(accepts, device=device)
            - 1
            + torch.arange(2, device=device) * 4
        ]
        torch.testing.assert_close(
            proposals, expected_bonus[:, None].expand_as(proposals)
        )
    graph.reset()
