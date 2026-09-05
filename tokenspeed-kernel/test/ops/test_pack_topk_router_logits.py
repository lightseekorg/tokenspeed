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
from tokenspeed_kernel.ops.moe.pack_topk import pack_topk_router_logits


def _inputs(device: str):
    return (
        torch.tensor([[0.7, 0.3], [0.55, 0.45]], dtype=torch.float32, device=device),
        torch.tensor([[3, 1], [2, 0]], dtype=torch.int32, device=device),
    )


def test_cpu_reference_recovers_selected_weights():
    weights, ids = _inputs("cpu")
    packed = pack_topk_router_logits(weights, ids, 4)
    recovered = packed.softmax(dim=-1).gather(1, ids.long())
    torch.testing.assert_close(recovered, weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("tokens", [1, 2, 4, 8])
def test_cuda_matches_reference(tokens):
    torch.manual_seed(19)
    ids = torch.stack(
        [torch.randperm(256, device="cuda")[:8] for _ in range(tokens)]
    ).to(torch.int32)
    weights = torch.rand(tokens, 8, device="cuda", dtype=torch.float32)
    weights /= weights.sum(dim=1, keepdim=True)
    actual = pack_topk_router_logits(weights, ids, 256)
    expected = torch.full_like(actual, -1.0e20)
    expected.scatter_(1, ids.long(), weights.log())
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("weights", "ids", "message"),
    [
        (torch.ones(2), torch.ones(2, dtype=torch.int32), "same 2D shape"),
        (
            torch.ones(1, 2, dtype=torch.bfloat16),
            torch.ones(1, 2, dtype=torch.int32),
            "float32",
        ),
        (torch.ones(1, 2), torch.ones(1, 2, dtype=torch.int16), "int32 or int64"),
    ],
)
def test_rejects_invalid_inputs(weights, ids, message):
    with pytest.raises((TypeError, ValueError), match=message):
        pack_topk_router_logits(weights, ids, 4)
