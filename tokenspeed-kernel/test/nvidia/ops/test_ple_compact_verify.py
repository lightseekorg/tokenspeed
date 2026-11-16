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
from tokenspeed_kernel.ops.ple import ple_conv_sequences, ple_restore_verify_layers

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    ("width", "lengths"),
    [(4, [4]), (4, [4] * 32), (8, [8] * 64), (8, [0, 1, 7, 8]), (4, [0, 0])],
)
@pytest.mark.parametrize("channels", [129, 10240, 32768])
def test_fused_compact_verify(width, lengths, channels):
    device, dtype, state = "cuda", torch.bfloat16, 9
    batch = len(lengths)
    total = sum(lengths)
    lengths_tensor = torch.tensor(lengths, device=device)
    starts = lengths_tensor.cumsum(0) - lengths_tensor
    req = torch.arange(batch, device=device).repeat_interleave(lengths_tensor)
    col = torch.cat([torch.arange(n, device=device) for n in lengths])
    values = torch.randn(total, channels, device=device, dtype=dtype)
    initial = torch.randn(batch, channels, state, device=device, dtype=dtype)
    weight = torch.randn(channels, 4, device=device, dtype=dtype)
    add_terms = (torch.randn_like(values), torch.randn_like(values))
    scratch = torch.empty(
        batch * (width + 1), channels, state, device=device, dtype=dtype
    )
    flag = torch.zeros((), device=device, dtype=torch.int32)

    def conv(layout_flag):
        return ple_conv_sequences(
            values,
            initial,
            weight,
            req,
            col,
            lengths_tensor,
            starts,
            total_tokens=total,
            batch_size=batch,
            dilation=3,
            kernel_size=4,
            state_len=state,
            write_final=False,
            weights_independent=True,
            add_terms=add_terms,
            windows=scratch,
            windows_block_rows=width + 1,
            scatter_windows=True,
            compact_layout_flag=layout_flag,
        )[0]

    reference = conv(None)
    full_windows = scratch.clone()
    row_elems = channels * state
    pages = torch.full((batch + 2, row_elems + 32), -123, device=device, dtype=dtype)
    out = pages[:, :row_elems].view(batch + 2, channels, state)
    src = torch.arange(batch, device=device) * (width + 1)
    dst = torch.arange(1, batch + 1, device=device)
    scratch_addrs, page_addrs, flag_addrs = [
        torch.tensor([t.data_ptr()], device=device, dtype=torch.uint64)
        for t in (scratch, pages, flag)
    ]
    strides = torch.tensor([pages.stride(0)], device=device, dtype=torch.int64)

    def run():
        result = conv(flag)
        ple_restore_verify_layers(
            scratch,
            src,
            dst,
            scratch_addrs,
            page_addrs,
            flag_addrs,
            strides,
            verify_width=width,
            batch_size=batch,
        )
        return result

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    requests = torch.arange(batch, device=device)
    for accepted in range(width + 1):
        steps = (accepted + requests) % (lengths_tensor + 1)
        src.copy_(requests * (width + 1) + steps)
        pages.fill_(-123)
        graph.replay()
        assert torch.equal(result.view(torch.int16), reference.view(torch.int16))
        assert torch.equal(
            out[1:-1].view(torch.int16), full_windows[src].view(torch.int16)
        )
        assert torch.all(pages[[0, -1]] == -123) and torch.all(
            pages[:, row_elems:] == -123
        )

    dst.fill_(-1)
    pages.fill_(-123)
    graph.replay()
    assert torch.all(pages == -123)
