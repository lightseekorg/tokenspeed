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


"""Prepared TP4/TP8 broadcast of exact int32 words from group rank zero."""

import torch
from tokenspeed_kernel._triton import gl, gluon, triton
from tokenspeed_kernel.ops.communication._iris import IrisGroup, allocation_bytes
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    iris_arch_policy,
)


class IrisBroadcast:
    """Retain fixed mailboxes; bit-preserving broadcasts stream through tiles."""

    WORDS_PER_TILE = 64
    MAX_TILES = 128

    @staticmethod
    def heap_bytes(max_words: int) -> int:
        tiles = min(
            IrisBroadcast.MAX_TILES,
            triton.cdiv(max_words, IrisBroadcast.WORDS_PER_TILE),
        )
        return allocation_bytes((tiles, 2 * IrisBroadcast.WORDS_PER_TILE), torch.uint32)

    def __init__(self, peers: IrisGroup, max_words: int):
        self._arch = iris_arch_policy()
        assert peers.world_size in (4, 8) and max_words > 0
        self.peers = peers
        self.tiles = min(self.MAX_TILES, triton.cdiv(max_words, self.WORDS_PER_TILE))
        self.mailbox = peers.ctx.empty(
            (self.tiles, 2 * self.WORDS_PER_TILE), dtype=torch.uint32
        )
        self.mailbox.fill_(0x80000000)
        offset = self.mailbox.data_ptr() - peers.addresses[peers.rank]
        self.peer_addresses = tuple(base + offset for base in peers.addresses)

    def can_run(self, tensor: torch.Tensor) -> bool:
        return (
            tensor.device == self.peers.device
            and tensor.dtype == torch.int32
            and tensor.is_contiguous()
            and tensor.numel() > 0
        )

    def run(self, tensor: torch.Tensor) -> None:
        """Broadcast contiguous int32 words in place from group rank zero.

        Fixed tile mailboxes preserve every int32 bit pattern. All calls on
        this state must use the same stream and group order as its collectives.
        """
        assert self.mailbox is not None
        assert tensor.is_cuda and tensor.is_contiguous() and tensor.dtype == torch.int32
        subgroups = self.peers.world_size if self.peers.rank == 0 else 1
        programs = min(
            triton.cdiv(tensor.numel(), self.WORDS_PER_TILE),
            self.tiles,
        )
        iris_lamport_broadcast[(programs,)](
            tensor,
            self.mailbox,
            self.peer_addresses[: self.peers.world_size],
            tensor.numel(),
            RANK=self.peers.rank,
            SOURCE=0,
            BLOCK=self.WORDS_PER_TILE,
            SUBGROUPS=subgroups,
            num_warps=subgroups,
            WAIT_STORES=self._arch.wait_stores,
        )


@gluon.jit(do_not_specialize=["words"])
def iris_lamport_broadcast(
    tensor,
    mailbox,
    peer_addresses,
    words,
    RANK: gl.constexpr,
    SOURCE: gl.constexpr,
    BLOCK: gl.constexpr,
    SUBGROUPS: gl.constexpr,
    WAIT_STORES: gl.constexpr,
):
    """Broadcast int32 values in place through sentinel-marked TP4/TP8 mailboxes.

    The source writes each word as two 16-bit halves, so every int32 value is
    distinct from the empty marker. Receivers copy both halves into their local
    tensor and clear the mailbox before the source can reuse that tile.

    Args:
        tensor: Contiguous int32 input on SOURCE and output on every other rank.
        mailbox: Local uint32 storage, initialized to 0x80000000, with two
            BLOCK-word halves per workgroup.
        peer_addresses: Mailbox base addresses for the group ranks.
        words: Number of int32 elements to broadcast.
        RANK: This process's rank within the group.
        SOURCE: Group rank supplying the input; its tensor remains unchanged.
        BLOCK: Number of int32 elements handled by one mailbox tile.
        SUBGROUPS: Subgroups per workgroup; one per group rank at the source.

    Returns:
        Nothing. Writes the broadcast result into tensor on receiving ranks.
    """
    # Two 16-bit halves encode every int32 without colliding with the empty
    # DWORD sentinel. Polls consume the payload itself with system-scope reads.
    WORLD_SIZE: gl.constexpr = len(peer_addresses)
    sentinel: gl.constexpr = 0x80000000
    # Program i always owns mailbox i, including when later calls change size.
    mailbox_offset = gl.program_id(0) * 2 * BLOCK
    first_word = gl.program_id(0) * BLOCK
    word_stride = gl.num_programs(0) * BLOCK
    if RANK == SOURCE:
        gl.static_assert(SUBGROUPS == WORLD_SIZE)
        layout: gl.constexpr = gl.BlockedLayout(
            [1, 1], [1, 64], [WORLD_SIZE, 1], [1, 0]
        )
        peers = gl.arange(0, WORLD_SIZE, layout=gl.SliceLayout(1, layout))
        offsets = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))
        destinations = gl.full(
            (WORLD_SIZE,), peer_addresses[0], gl.uint64, gl.SliceLayout(1, layout)
        )
        for peer in gl.static_range(1, WORLD_SIZE):
            destinations = gl.where(peers == peer, peer_addresses[peer], destinations)
        destinations = destinations.to(gl.pointer_type(gl.uint32))
        pointers = destinations[:, None] + mailbox_offset + offsets[None, :]
        mask = (peers != SOURCE)[:, None]
        for start in range(first_word, words, word_stride):
            # Publish the whole tile, including the masked tail: every word
            # must complete empty -> payload -> empty before address reuse.
            bits = gl.load(
                tensor + start + offsets, start + offsets < words, other=0
            ).to(gl.uint32)
            lower = gl.load(
                pointers, mask, other=sentinel, cache_modifier=".cv", volatile=True
            )
            upper = gl.load(
                pointers + BLOCK,
                mask,
                other=sentinel,
                cache_modifier=".cv",
                volatile=True,
            )
            busy = gl.max(
                gl.max(((lower != sentinel) | (upper != sentinel)).to(gl.int32), 1), 0
            )
            while busy != 0:
                lower = gl.load(
                    pointers, mask, other=sentinel, cache_modifier=".cv", volatile=True
                )
                upper = gl.load(
                    pointers + BLOCK,
                    mask,
                    other=sentinel,
                    cache_modifier=".cv",
                    volatile=True,
                )
                busy = gl.max(
                    gl.max(((lower != sentinel) | (upper != sentinel)).to(gl.int32), 1),
                    0,
                )
            gl.store(pointers, (bits & 65535)[None, :], mask, cache_modifier=".wt")
            gl.store(
                pointers + BLOCK, (bits >> 16)[None, :], mask, cache_modifier=".wt"
            )
            # Every publishing subgroup must finish before the next tile/call.
            WAIT_STORES(args=[], constraints="=r,~{memory}")
            gl.barrier()
    else:
        layout: gl.constexpr = gl.BlockedLayout([1], [64], [SUBGROUPS], [0])
        offsets = gl.arange(0, BLOCK, layout=layout)
        mailbox += mailbox_offset
        for start in range(first_word, words, word_stride):
            lower = gl.load(mailbox + offsets, cache_modifier=".cv", volatile=True)
            upper = gl.load(
                mailbox + BLOCK + offsets, cache_modifier=".cv", volatile=True
            )
            while (
                gl.max(((lower == sentinel) | (upper == sentinel)).to(gl.int32), 0) != 0
            ):
                lower = gl.load(mailbox + offsets, cache_modifier=".cv", volatile=True)
                upper = gl.load(
                    mailbox + BLOCK + offsets, cache_modifier=".cv", volatile=True
                )
            gl.store(
                tensor + start + offsets,
                (lower | (upper << 16)).to(gl.int32),
                start + offsets < words,
            )
            gl.store(mailbox + offsets, sentinel, cache_modifier=".wt")
            gl.store(mailbox + BLOCK + offsets, sentinel, cache_modifier=".wt")
            # Finish clearing the full tile before a later receive can start.
            WAIT_STORES(args=[], constraints="=r,~{memory}")
            gl.barrier()
