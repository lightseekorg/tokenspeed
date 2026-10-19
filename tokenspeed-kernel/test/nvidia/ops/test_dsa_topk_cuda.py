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

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa.cuda import (
    has_ragged_decode_topk,
    ragged_decode_topk,
)
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_hopper_plus,
    reason="CUDA persistent top-k requires NVIDIA SM90+",
)


@pytest.mark.parametrize("cols", [4096, 65536])
@pytest.mark.parametrize("topk", [512, 1024, 2048])
@pytest.mark.parametrize("q_len_per_req", [1, 4])
@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
def test_ragged_decode_topk_preserves_causal_prefixes_and_strides(
    cols, topk, q_len_per_req, length_dtype
):
    if not has_ragged_decode_topk():
        pytest.skip("CUDA persistent top-k is unavailable")
    torch.manual_seed(437)
    requests = 4
    rows = requests * q_len_per_req
    lengths = torch.empty((requests, 2), device="cuda", dtype=length_dtype)[:, :1]
    lengths[:, 0] = torch.tensor([0, 1, topk - 1, cols], device="cuda")
    effective_lengths = (
        lengths[:, 0].repeat_interleave(q_len_per_req)
        - (q_len_per_req - 1)
        + torch.arange(q_len_per_req, device="cuda").repeat(requests)
    ).clamp_min(0)
    columns = torch.arange(cols, device="cuda")
    valid = columns[None, :] < effective_lengths[:, None]
    logits = torch.empty((rows, cols * 2), device="cuda")[:, ::2]
    logits.copy_(
        torch.where(valid, torch.randn(rows, cols, device="cuda"), float("nan"))
    )
    output = torch.full((rows, topk), -77, dtype=torch.int32, device="cuda")

    ragged_decode_topk(
        logits,
        output,
        topk,
        lengths=lengths,
        max_seq_len=cols,
        q_len_per_req=q_len_per_req,
        enable_pdl=False,
    )

    live = output >= 0
    torch.testing.assert_close(
        live.sum(dim=1), effective_lengths.clamp_max(topk).long()
    )
    assert torch.all((output < effective_lengths[:, None]) | ~live)
    assert torch.all(live | (output == -1))
    ordered = output.sort(dim=1).values
    assert not torch.any((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0))
    selected_scores = logits.gather(1, output.clamp_min(0).long()).masked_fill(
        ~live, -float("inf")
    )
    reference = torch.topk(logits.masked_fill(~valid, -float("inf")), topk).values
    torch.testing.assert_close(
        selected_scores.sort(dim=1).values,
        reference.sort(dim=1).values,
        rtol=0,
        atol=0,
    )
