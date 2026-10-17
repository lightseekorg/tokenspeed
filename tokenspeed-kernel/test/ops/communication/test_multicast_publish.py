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

"""The mailbox publish writes whole words, bit for bit."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.communication.triton import multicast_publish
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform().is_nvidia or current_platform().is_amd),
    reason="multicast publish needs a GPU",
)


@pytest.mark.parametrize(
    "rows, width, slot_col", [(1, 1024, 1), (3, 1024, 1), (64, 8192, 3)]
)
def test_publish_copies_whole_words_into_a_strided_slot(
    rows: int, width: int, slot_col: int
) -> None:
    torch.manual_seed(1)
    block = torch.randn(rows, width, dtype=torch.bfloat16, device="cuda")
    block[0, :16] = -0.0
    block[-1, 100] = 0.0
    mailbox = torch.empty(64, 4 * width, dtype=torch.bfloat16, device="cuda")
    mailbox.view(torch.int32).fill_(-1)
    slot = mailbox[:rows, slot_col * width : (slot_col + 1) * width]
    multicast_publish(block, slot)
    torch.cuda.synchronize()
    # Bit for bit, negative zeros included: the sentinel is a NaN payload.
    assert torch.equal(slot.view(torch.int32), block.view(torch.int32))
    # Untouched columns keep the sentinel arming.
    assert mailbox[:rows, : slot_col * width].view(torch.int32).eq(-1).all()
    assert mailbox[:rows, (slot_col + 1) * width :].view(torch.int32).eq(-1).all()
    multicast_publish(block[:0], slot[:0])
    odd = torch.zeros(rows, 4 * width + 2, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="32-bit aligned"):
        multicast_publish(block, odd[:, 1 : width + 1])


@pytest.mark.skipif(
    not current_platform().is_nvidia, reason="half-polled mailboxes require NVIDIA"
)
@pytest.mark.parametrize("path", ["decode_gemv", "torch.mm"])
def test_no_gemm_output_spells_the_half_polled_sentinel(path: str) -> None:
    """NaN and Inf inputs, even a 0xFFFF-payload NaN, come out as 0x7FFF."""
    from tokenspeed_kernel.ops.gemm import mv

    torch.manual_seed(0)
    n, k = 1024, 8192
    weight = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    weight.view(torch.int16)[3] = -1
    x = torch.randn(1, k, device="cuda").to(torch.bfloat16)
    x[0, 5] = float("nan")
    x[0, 9] = float("inf")
    x[0, 10] = float("-inf")
    x.view(torch.int16)[0, 7] = -1
    out = mv(x, weight) if path == "decode_gemv" else torch.mm(x, weight.t())
    torch.cuda.synchronize()
    bits = out.view(torch.int16).to(torch.int32) & 0xFFFF
    assert torch.isnan(out).any()
    assert not bits.eq(0xFFFF).any()
    assert set(bits[torch.isnan(out)].tolist()) == {0x7FFF}
