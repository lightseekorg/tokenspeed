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

import torch
from tokenspeed_kernel.ops.attention.dsv4.deep_gemm import _trtllm_decode_topk


def test_trtllm_decode_topk_accepts_2d_lens(monkeypatch):
    captured = {}

    def fake_indexer_topk_decode(values, seq_lens, indices, next_n, topk):
        del values, indices
        captured["seq_lens"] = seq_lens
        captured["next_n"] = next_n
        captured["topk"] = topk

    monkeypatch.setattr(
        torch.ops.trtllm,
        "indexer_topk_decode",
        fake_indexer_topk_decode,
        raising=False,
    )
    values = torch.empty((2, 4), dtype=torch.float32)
    seq_lens = torch.tensor([[3], [4]], dtype=torch.int64)
    indices = torch.empty((2, 2), dtype=torch.int32)

    _trtllm_decode_topk(values, seq_lens, indices, topk=2)

    assert captured["next_n"] == 1
    assert captured["topk"] == 2
    assert captured["seq_lens"].dtype == torch.int32
    assert captured["seq_lens"].dim() == 1
    torch.testing.assert_close(
        captured["seq_lens"],
        torch.tensor([3, 4], dtype=torch.int32),
        atol=0,
        rtol=0,
    )
