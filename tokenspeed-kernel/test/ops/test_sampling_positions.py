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

"""Position-keyed sampling preserves draws without per-batch recompilation."""

from test.utils import assert_no_triton_compile

import pytest
import torch
from tokenspeed_kernel.ops.sampling import chain_verify_coins, prepare_sampling_offsets
from tokenspeed_kernel.ops.sampling.triton.positions import _prepare_sampling_offsets
from tokenspeed_kernel.ops.sampling.triton.verify_coins import _chain_verify_coins


def test_active_counts_reuse_the_preparation_kernel():
    assert torch.cuda.is_available(), "Sampling offset acceptance requires CUDA"
    frontier = torch.arange(257, dtype=torch.int32, device="cuda")
    slots = torch.arange(1, 129, dtype=torch.int64, device="cuda")
    lengths = torch.full((128,), 37, dtype=torch.int32, device="cuda")
    out = torch.zeros(257, dtype=torch.int64, device="cuda")
    prepare_sampling_offsets(frontier, slots, lengths, out=out)
    with assert_no_triton_compile(_prepare_sampling_offsets):
        for rows in (1, 3, 16, 17, 65, 128):
            for extends in (0, 1, rows):
                before = out.clone()
                prepare_sampling_offsets(
                    frontier,
                    slots[:rows],
                    lengths[:extends] if extends else None,
                    out=out,
                )
                expected = frontier[slots[:rows]].long() + 1
                expected[:extends] += 36
                assert torch.equal(out[slots[:rows]], expected)
                assert out[0].item() == 0
                assert torch.equal(out[rows + 1 :], before[rows + 1 :])


def _verify_coins(slots, seeds, positions, width):
    rows = slots.numel()
    accept = torch.empty((rows, width), dtype=torch.float32, device=slots.device)
    final = torch.empty((rows,), dtype=torch.float32, device=slots.device)
    chain_verify_coins(slots, seeds, positions, coins=accept, final_coins=final)
    return accept, final


def test_verify_coins_follow_seed_and_round_position():
    assert torch.cuda.is_available(), "Sampling coin acceptance requires CUDA"
    position = torch.iinfo(torch.int32).max
    seeds = torch.tensor([0, 17, 17, 17 + 2**40, 17], device="cuda")
    positions = torch.tensor(
        [0, position, position, position, position - 1], device="cuda"
    )
    slots = torch.tensor([1, 2, 3, 4, 0, 0], dtype=torch.int32, device="cuda")
    accept, final = _verify_coins(slots, seeds, positions, 8)

    assert ((accept > 0) & (accept < 1)).all()
    assert ((final > 0) & (final < 1)).all()
    assert torch.equal(accept[0], accept[1])
    assert torch.equal(final[0], final[1])
    assert torch.equal(accept[4], accept[5])
    assert torch.equal(final[4], final[5])
    # Change only the high seed bits, then only the round position.
    assert not torch.equal(accept[0], accept[2])
    assert not torch.equal(final[0], final[2])
    assert not torch.equal(accept[0], accept[3])
    assert not torch.equal(final[0], final[3])
    # A proposal at the same position in adjacent rounds gets a fresh draw.
    assert not torch.equal(accept[3, 1:], accept[0, :-1])
    assert (accept != final[:, None]).any(dim=0).all()

    with assert_no_triton_compile(_chain_verify_coins):
        # Vary row count, width within the same bucket, and input stride.
        for rows, width, stride in ((3, 5, 1), (7, 6, 2), (13, 7, 3)):
            row_order = torch.arange(rows - 1, -1, -1, device="cuda") % slots.numel()
            reordered = torch.empty_strided(
                (rows,), (stride,), dtype=slots.dtype, device="cuda"
            )
            reordered.copy_(slots[row_order])
            actual_accept, actual_final = _verify_coins(
                reordered, seeds, positions, width
            )
            assert torch.equal(actual_accept, accept[row_order, :width])
            assert torch.equal(actual_final, final[row_order])


def test_verify_coins_read_slot_past_int32_offset():
    assert torch.cuda.is_available(), "Sampling coin acceptance requires CUDA"
    stride = 2**30 + 1
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < (2 * stride + 1) * 4 + 2**30:
        pytest.skip("requires an 8 GiB strided slot buffer")
    # Only three slot IDs are touched; row two starts past 2**31 elements
    # even though the stride itself fits in int32.
    compact = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
    slots = torch.empty_strided((3,), (stride,), dtype=compact.dtype, device="cuda")
    slots.copy_(compact)
    seeds = torch.tensor([0, 100, 200, 300], dtype=torch.int64, device="cuda")
    positions = torch.tensor([0, 1024, 2048, 4096], dtype=torch.int64, device="cuda")
    expected_accept, expected_final = _verify_coins(compact, seeds, positions, 4)
    actual_accept, actual_final = _verify_coins(slots, seeds, positions, 4)
    assert torch.equal(actual_accept, expected_accept)
    assert torch.equal(actual_final, expected_final)
