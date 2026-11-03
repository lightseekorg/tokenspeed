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

"""Whole-plane MLA KV gather followed by virtual-slot selection."""

import weakref

import pytest
import torch

from tokenspeed.runtime.layers.attention.dcp.ascend import AscendDCPContext


@pytest.mark.parametrize("degree", [1, 2, 8])
@pytest.mark.parametrize("block_granularity", [64, 128, 256])
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_whole_cache_gather_preserves_bytes_and_slot_order(
    monkeypatch, degree, block_granularity, slot_dtype
):
    virtual_blocks = 20  # Uneven ownership and a partial final owner stripe.
    local_rows = ((virtual_blocks - 2) // degree + 2) * block_granularity
    generator = torch.Generator().manual_seed(31)
    # Include arbitrary BF16 bit patterns, not just finite random floats.
    raw = torch.randint(
        0,
        256,
        (virtual_blocks * block_granularity, 1, 16),
        dtype=torch.uint8,
        generator=generator,
    )
    raw[:block_granularity].zero_()
    global_cache = raw.view(torch.bfloat16)
    caches = []
    for rank in range(degree):
        cache = torch.full((local_rows, 1, 8), -99, dtype=torch.bfloat16)
        cache[:block_granularity].zero_()
        for block in range(1, virtual_blocks):
            if (block - 1) % degree == rank:
                local = ((block - 1) // degree + 1) * block_granularity
                cache[local : local + block_granularity].copy_(
                    global_cache[
                        block * block_granularity : (block + 1) * block_granularity
                    ]
                )
        caches.append(cache)
    slots = torch.tensor(
        [
            19 * block_granularity + 7,
            block_granularity,
            8 * block_granularity - 1,
            8 * block_granularity,
            3 * block_granularity + 11,
            block_granularity,
            0,
            -1,
            virtual_blocks * block_granularity,
        ],
        dtype=slot_dtype,
    )
    groups = [slots[:5], slots[5:], slots[:0]]
    valid = (slots >= block_granularity) & (slots < global_cache.shape[0])
    expected = global_cache.index_select(0, torch.where(valid, slots.long(), 0))
    gathered = torch.cat(caches).view(torch.uint8)
    for rank, cache in enumerate(caches):
        before = cache.clone()
        context = AscendDCPContext(
            degree=degree,
            rank=rank,
            ranks=tuple(range(8, 8 + degree)),
            auxiliary_namespace=None,
        )
        group = object()
        context.primary_process_group = group
        sent = []

        def gather(output, value, *, group):
            sent.append(value.clone())
            output.copy_(gathered)

        monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
        actual = list(
            context.gather_cache_rows(
                cache,
                groups,
                block_granularity=block_granularity,
                virtual_block_count=virtual_blocks,
            )
        )
        torch.testing.assert_close(
            torch.cat(actual).view(torch.uint8),
            expected.view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(cache.view(torch.uint8), before.view(torch.uint8))
        assert len(sent) == (1 if degree > 1 else 0)
        if sent:
            torch.testing.assert_close(
                sent[0], before.view(torch.uint8), rtol=0, atol=0
            )
        assert actual[-1].shape[0] == 0


def test_empty_prefix_never_gathers(monkeypatch):
    context = AscendDCPContext(
        degree=8,
        rank=0,
        ranks=tuple(range(8)),
        auxiliary_namespace=None,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("an empty prefix must not launch a collective")

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", forbidden)
    for groups in ([], [torch.empty(0, dtype=torch.int64)]):
        assert (
            list(
                context.gather_cache_rows(
                    torch.empty(256, 1, 576, dtype=torch.bfloat16),
                    groups,
                    block_granularity=128,
                    virtual_block_count=9,
                )
            )
            == []
        )


@pytest.mark.parametrize("local_rows", [128, 257])
def test_gather_refuses_incomplete_physical_cache(local_rows):
    context = AscendDCPContext(
        degree=8,
        rank=0,
        ranks=tuple(range(8)),
        auxiliary_namespace=None,
    )
    with pytest.raises(ValueError, match="local cache plane"):
        list(
            context.gather_cache_rows(
                torch.empty(local_rows, 1, 576, dtype=torch.bfloat16),
                [torch.tensor([128])],
                block_granularity=128,
                virtual_block_count=20,
            )
        )


def test_next_forward_reads_new_cache_and_releases_gathered_plane(monkeypatch):
    context = AscendDCPContext(
        degree=8, rank=0, ranks=tuple(range(8)), auxiliary_namespace=None
    )
    context.primary_process_group = object()
    cache = torch.zeros(256, 1, 576, dtype=torch.bfloat16)
    peer_caches = cache.repeat(8, 1, 1)
    planes = []

    def gather(output, value, *, group):
        output.copy_(peer_caches.view(torch.uint8))
        planes.append(weakref.ref(output))

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
    # Virtual block 8 belongs to owner 7, at its physical block 1.
    slots = [torch.tensor([8 * 128 + 17])]

    def read():
        return list(
            context.gather_cache_rows(
                cache, slots, block_granularity=128, virtual_block_count=9
            )
        )[0]

    before = read()
    peer_caches[7 * 256 + 128 + 17].fill_(3)
    after = read()
    assert not before.any()
    assert (after == 3).all()
    # Returned rows own their storage; neither they nor the context retain the
    # temporary full plane after the reader has exhausted the iterator.
    assert all(reference() is None for reference in planes)
