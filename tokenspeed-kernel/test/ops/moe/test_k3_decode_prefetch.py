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

"""The Kimi-K3 decode chain's L2 prefetch ranges, tables and row counts (CPU)."""

import pytest
import torch
from tokenspeed_kernel.ops.moe._cute_dsl import k3_decode_prefetch as prefetch
from tokenspeed_kernel.ops.moe._cute_dsl import k3_decode_support

MIB = 1 << 20
# The experts kernel's table rows (``moe.experts.PREFETCH_ROWS``).
ROWS = 8


def _bytes(n):
    return torch.empty(n, dtype=torch.uint8)


def test_a_range_covers_the_weight_storage_span():
    whole = torch.empty(4, 32, dtype=torch.bfloat16)
    # Row-pitched: three full rows of 32 and 16 more elements, 2 bytes each.
    pitched = whole[:, :16]
    assert prefetch.weight_ranges((whole, pitched, whole.t()), 0) == [
        (whole.data_ptr(), 256),
        (whole.data_ptr(), (3 * 32 + 16) * 2),
        (whole.data_ptr(), 256),
    ]


def test_uncapped_ranges_keep_odd_sizes_and_skip_empty_weights():
    odd, empty = _bytes(1000), _bytes(0)
    assert prefetch.weight_ranges((empty, odd), 0) == [(odd.data_ptr(), 1000)]


def test_a_byte_cap_truncates_in_order_to_whole_16_byte_chunks():
    first, second, third = _bytes(1024), _bytes(2048), _bytes(64)
    ranges = prefetch.weight_ranges((first, second, third), 1536)
    assert ranges == [(first.data_ptr(), 1024), (second.data_ptr(), 512)]
    ranges = prefetch.weight_ranges((_bytes(1000), second), 1500)
    assert [size for _, size in ranges] == [992, 496]
    assert all(size % 16 == 0 for _, size in ranges)


def test_a_weight_off_16_byte_alignment_is_rejected():
    with pytest.raises(ValueError, match="16-byte aligned"):
        prefetch.weight_ranges((_bytes(64)[1:],), 0)


def test_the_ahead_table_takes_first_whole_then_the_capped_weights():
    up, attention, o_proj = _bytes(4096), _bytes(4096), _bytes(4096)
    table = prefetch.ahead_table((up,), (attention, o_proj), 1024, ROWS, "cpu")
    assert table.dtype == torch.int64 and table.shape == (ROWS, 2)
    assert table[:2].tolist() == [[up.data_ptr(), 4096], [attention.data_ptr(), 1024]]
    # Unused rows stay zero, which the kernel skips.
    assert not table[2:].any()


def test_the_ahead_table_drops_ranges_past_its_rows():
    first = (_bytes(32),)
    weights = tuple(_bytes(32) for _ in range(ROWS + 3))
    table = prefetch.ahead_table(first, weights, 0, ROWS, "cpu")
    kept = (*first, *weights)[:ROWS]
    assert table.tolist() == [[w.data_ptr(), 32] for w in kept]


def test_next_attention_launches_group_row_counts_by_cap_and_carveout(monkeypatch):
    monkeypatch.setattr(prefetch, "NEXT_ATTENTION", "1:96.100,2-3:96,4-7:64,8:0,16:8")
    monkeypatch.setattr(prefetch, "EXPERTS_AHEAD", "4-6:16,5:0")
    groups = prefetch.next_attention_groups((1, 2, 3, 5, 6, 7, 8, 32))
    # Row counts with an in-kernel table take no launch; a zero turns either off.
    assert groups == {
        (96 * MIB, 100): (1,),
        (96 * MIB, None): (2, 3),
        (64 * MIB, None): (5, 7),
    }
    assert prefetch.experts_ahead_caps((1, 5, 6, 8)) == {6: 16 * MIB}


def test_the_served_row_counts_take_one_next_attention_prefetch_each():
    """No row count gets both the in-kernel table and a side-stream launch."""
    groups = prefetch.next_attention_groups(k3_decode_support.BATCHES)
    launched = [b for batches in groups.values() for b in batches]
    ahead = prefetch.experts_ahead_caps(k3_decode_support.BATCHES)
    assert len(launched) == len(set(launched))
    assert not set(launched) & set(ahead)
    assert set(launched) | set(ahead) <= set(k3_decode_support.BATCHES)


def test_attention_prefetches_keep_the_served_row_counts_they_turn_on(monkeypatch):
    monkeypatch.setattr(prefetch, "OPROJ_DOWN", "3-8:1,6:0")
    monkeypatch.setitem(prefetch.ATTENTION_MOE, "kda", "5-8:1")
    monkeypatch.setitem(prefetch.ATTENTION_MOE, "mla", "8:1")
    served = (1, 3, 5, 6, 8, 16)
    assert prefetch.oproj_down_batches(served) == (3, 5, 8)
    assert prefetch.attention_moe_batches(False, served) == (5, 6, 8)
    assert prefetch.attention_moe_batches(True, served) == (8,)


def test_the_front_weight_is_prefetched_once_per_served_row_count():
    """The exit skips the front weight that both attention kinds already fetched.

    ``ATTENTION_MOE`` prefetches the MoE front's packed weight with the
    o-projection while the attention core runs; a second prefetch of it after
    the o-projection exit only adds traffic beside the front GEMM.
    """
    served = k3_decode_support.BATCHES
    kda = set(prefetch.attention_moe_batches(False, served))
    mla = set(prefetch.attention_moe_batches(True, served))
    assert kda & mla
    assert not kda & mla & set(prefetch.oproj_down_batches(served))
