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

"""The rank-major mailbox gather reads whole blocks, bit for bit, and re-arms them."""

from __future__ import annotations

import pytest
import torch
from fused_boundary_helpers import assert_child_keeps_waiting
from tokenspeed_kernel.ops.communication.triton import gather_mailbox_blocks
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform().is_nvidia or current_platform().is_amd),
    reason="the mailbox gather needs a GPU",
)

# The half-polled sentinel: a NaN payload no GEMM emits.
_SENTINEL = 0xFFFFFFFF


@pytest.mark.parametrize(
    "rows, blocks, block_width", [(1, 4, 1024), (5, 4, 2048), (64, 1, 8192)]
)
def test_gather_reads_each_blocks_rows_side_by_side_and_re_arms_them(
    rows: int, blocks: int, block_width: int
) -> None:
    torch.manual_seed(1)
    expected = torch.randn(
        rows, blocks * block_width, dtype=torch.bfloat16, device="cuda"
    )
    expected[0, :16] = -0.0
    expected[-1, 100] = 0.0
    capacity = 64
    mailbox = torch.empty(
        blocks, capacity, block_width, dtype=torch.bfloat16, device="cuda"
    )
    mailbox.view(torch.int32).fill_(-1)
    mailbox[:, :rows].copy_(expected.view(rows, blocks, block_width).transpose(0, 1))
    got = gather_mailbox_blocks(mailbox[:, :rows], _SENTINEL, False)
    # Bit for bit, negative zeros included; every word read is re-armed.
    assert torch.equal(got.view(torch.int16), expected.view(torch.int16))
    assert mailbox.view(torch.int32).eq(-1).all()
    assert gather_mailbox_blocks(mailbox[:, :0], _SENTINEL, False).shape == (
        0,
        blocks * block_width,
    )
    with pytest.raises(ValueError, match="contiguous blocks"):
        gather_mailbox_blocks(mailbox[..., : block_width // 2], _SENTINEL, False)


_GATHER_WAITER = """
from tokenspeed_kernel.ops.communication.triton import gather_mailbox_blocks
row = torch.full((1, 4096), 0.5, dtype=torch.bfloat16, device=device)
mailbox = torch.empty(4, 1, 1024, dtype=torch.bfloat16, device=device)
def consume():
    gather_mailbox_blocks(mailbox, HALF_POLLED_SENTINEL, True)
"""


@pytest.mark.skipif(
    not current_platform().is_nvidia, reason="half-polled mailboxes require NVIDIA"
)
@pytest.mark.parametrize("word, empty", [(777, 0xFFFF), (2047, -65536)])
def test_gather_waits_for_a_word_still_holding_a_sentinel_half(
    word: int, empty: int
) -> None:
    """A word with either BF16 half still empty stalls its block's row."""
    assert_child_keeps_waiting(_GATHER_WAITER, word, empty)


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
