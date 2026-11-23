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
from tokenspeed_kernel.ops.attention.dsv41.triton import _index_finish_parts


@pytest.fixture(autouse=True)
def enable_fused_index_pack(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_EXPERIMENT_FUSED_INDEX_PACK", "1")


@pytest.mark.parametrize("rows,width", [(1, 256), (6, 2048), (24, 16384)])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("k", [32, 128, 256])
@pytest.mark.parametrize("use_fused", ["0", "1"])
def test_index_finish_parts_exact(rows, width, strided, k, use_fused, monkeypatch):
    monkeypatch.setenv("TOKENSPEED_EXPERIMENT_FUSED_INDEX_PACK", use_fused)
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    torch.manual_seed(42)
    step = 2 if strided else 1
    scores = torch.randn(rows, width * step, device="cuda")[:, ::step]
    ids = torch.empty(rows, width * step, device="cuda", dtype=torch.int64)[:, ::step]
    ids.copy_(torch.arange(width, device="cuda"))
    scores[:, ::7] = 1.0
    scores[:, ::11] = -torch.inf
    scores[:, ::17] = torch.inf
    scores[:, ::19] = 0.0
    scores[:, ::23] = -0.0
    scores[0] = -torch.inf
    ids.masked_fill_(scores == -torch.inf, -1)
    bits = scores.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    ordered = bits ^ torch.where(bits & 0x80000000 != 0, 0xFFFFFFFF, 0x80000000)
    packed = ((ordered - 0x80000000) << 32) | (0xFFFFFFFF - ids)
    packed.masked_fill_(scores == -torch.inf, torch.iinfo(torch.int64).min)
    pos = packed.topk(k, dim=1, sorted=False).indices
    selected = ids.gather(1, pos)
    live = scores.gather(1, pos) > -torch.inf
    expected = (
        selected.masked_fill(~live, torch.iinfo(torch.int64).max).sort(dim=1).values
    )
    expected.masked_fill_(expected == torch.iinfo(torch.int64).max, -1)
    output = torch.empty(rows, k, device="cuda", dtype=torch.int32)
    lengths = torch.empty(rows, device="cuda", dtype=torch.int32)
    _index_finish_parts(scores, ids, k, output, lengths)
    torch.testing.assert_close(output.long(), expected, rtol=0, atol=0)
    torch.testing.assert_close(lengths.long(), live.sum(1), rtol=0, atol=0)


def test_index_pack_parts_does_not_recompile_for_aligned_widths():
    from tokenspeed_kernel.ops.attention.dsv41.triton import _index_pack_parts_kernel
    from utils import assert_no_triton_compile

    if not torch.cuda.is_available():
        pytest.skip("GPU required")

    def run(rows, width):
        scores = torch.zeros(rows, width, device="cuda")
        ids = torch.zeros(rows, width, device="cuda", dtype=torch.int64)
        out = torch.empty_like(ids)
        _index_pack_parts_kernel[(rows, width // 256)](
            scores,
            ids,
            out,
            width,
            width,
            1,
            width,
            1,
            256,
        )

    run(1, 256)
    with assert_no_triton_compile(_index_pack_parts_kernel):
        run(6, 512)
        run(24, 16384)
