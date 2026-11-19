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

"""Lamport all-reduce MAX of one int64 word per row, in one launch."""

import torch
from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris import IrisGroup, allocation_bytes
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_peer_pointers,
    _iris_subgroup_any,
    iris_arch_policy,
)
from tokenspeed_kernel.platform import current_platform


class LamportWordMax:
    """Own the mailboxes of one group's all-reduce MAX of int64 words.

    Every rank pushes each word into its own slot of every rank's mailbox,
    waits for every rank's slot, keeps the largest word and restores the
    slots' empty markers. A word travels as four 16-bit pieces, each in a
    32-bit mailbox word that never equals the empty marker, so every int64
    value is exchanged exactly and a partly arrived word reads as missing.

    Calls alternate between two mailbox stages by a per-rank call count.
    Finishing a call needs every peer's words of that call, so no rank gets
    more than one call ahead of a peer: the stage a call writes was read and
    restored by every peer one call earlier. Every rank makes the same calls
    in one stream order.
    """

    NUM_WARPS = 4
    MAX_ROWS = 4096

    @staticmethod
    def supported(world_size: int) -> bool:
        """Whether the kernel serves a group of ``world_size`` ranks."""
        return current_platform().is_cdna4 and world_size in (4, 8)

    @staticmethod
    def heap_bytes(max_rows: int, world_size: int) -> int:
        return allocation_bytes((2, world_size, max_rows, 4), torch.uint32)

    def __init__(self, peers: IrisGroup, max_rows: int):
        if not self.supported(peers.world_size) or not (1 <= max_rows <= self.MAX_ROWS):
            raise ValueError(
                f"unsupported Iris word max: world={peers.world_size}, "
                f"max_rows={max_rows}"
            )
        self.peers = peers
        self.max_rows = max_rows
        # mailbox[stage][source rank][row] holds the row's four pieces.
        self.mailbox = peers.ctx.empty(
            (2, peers.world_size, max_rows, 4), dtype=torch.uint32
        )
        self.mailbox.fill_(0x80000000)
        self.calls = torch.zeros((1,), dtype=torch.int32, device=peers.device)
        self._policy = iris_arch_policy()

    def run(self, words: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Return every rank's ``words`` reduced with MAX, into ``out``.

        ``out`` may be ``words`` itself; a new tensor when None.
        """
        rows = words.shape[0]
        assert words.dtype == torch.int64 and words.dim() == 1
        assert 0 < rows <= self.max_rows and words.is_contiguous()
        if out is None:
            out = torch.empty_like(words)
        assert out.shape == words.shape and out.dtype == torch.int64
        assert out.is_contiguous()
        world_size = self.peers.world_size
        iris_lamport_word_max[(1,)](
            words,
            out,
            self.mailbox,
            self.calls,
            self.peers.addresses[:world_size],
            rows,
            RANK=self.peers.rank,
            MAX_ROWS=self.max_rows,
            NUM_WARPS=self.NUM_WARPS,
            PEER_SIGNAL_STORE_CACHE=self._policy.peer_signal_store,
            num_warps=self.NUM_WARPS,
        )
        return out


@gluon.jit(do_not_specialize=["rows"])
def iris_lamport_word_max(
    words_ptr,
    out_ptr,
    mailbox,
    calls_ptr,
    heaps,
    rows,
    RANK: gl.constexpr,
    MAX_ROWS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """All-reduce MAX of int64 words across TP4/TP8.

    One workgroup; each lane owns a row at a time. All of a rank's pushes are
    issued before it waits, and its subgroups wait independently. ``out``
    may alias ``words``: a row is stored only after every row was pushed.
    Mailbox traffic uses buffer instructions: global loads and stores with
    these cache modifiers each wait for completion before the next issues.
    """
    WORLD_SIZE: gl.constexpr = len(heaps)
    BLOCK: gl.constexpr = 64 * NUM_WARPS
    SLOT: gl.constexpr = MAX_ROWS * 4
    empty: gl.constexpr = 0x80000000
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [NUM_WARPS, 1], [1, 0])
    lane = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, layout))
    piece = gl.arange(0, 4, layout=gl.SliceLayout(0, layout))
    shift = (16 * piece).to(gl.int64)[None, :]
    calls = gl.load(calls_ptr)
    # The int32 count wraps negative; & keeps the parity where % would not.
    stage = (calls & 1) * (WORLD_SIZE * SLOT)
    peers = _iris_peer_pointers(mailbox, heaps, RANK)
    for start in range(0, rows, BLOCK):
        row = start + lane
        in_rows = row < rows
        word = gl.load(words_ptr + row, mask=in_rows, other=0)
        pieces = ((word[:, None] >> shift) & 0xFFFF).to(gl.uint32)
        offsets = stage + RANK * SLOT + row[:, None] * 4 + piece[None, :]
        for peer in gl.static_range(WORLD_SIZE):
            gl.amd.cdna4.buffer_store(
                pieces,
                peers[peer],
                offsets,
                mask=in_rows[:, None],
                cache=PEER_SIGNAL_STORE_CACHE,
            )
    for start in range(0, rows, BLOCK):
        row = start + lane
        in_rows = row < rows
        offsets = stage + row[:, None] * 4 + piece[None, :]
        received = ()
        for _ in gl.static_range(WORLD_SIZE):
            received += (gl.zeros([BLOCK, 4], gl.uint32, layout),)
        waiting = True
        while waiting:
            # Issue every source's read before checking any; arrived words
            # stay until this rank restores them, so retries reread all.
            missing = gl.zeros_like(lane)
            loaded = ()
            for source in gl.static_range(WORLD_SIZE):
                arrived = gl.amd.cdna4.buffer_load(
                    mailbox,
                    source * SLOT + offsets,
                    mask=in_rows[:, None],
                    other=0,
                    cache=".cv",
                )
                loaded += (arrived,)
                missing |= gl.max((arrived == empty).to(gl.int32), 1)
            received = loaded
            waiting = _iris_subgroup_any(missing)
        # Disjoint pieces: their sum is the word, its top bit included.
        best = gl.sum(received[0].to(gl.int64) << shift, 1)
        for source in gl.static_range(1, WORLD_SIZE):
            best = gl.maximum(best, gl.sum(received[source].to(gl.int64) << shift, 1))
        for source in gl.static_range(WORLD_SIZE):
            gl.amd.cdna4.buffer_store(
                gl.full([BLOCK, 4], empty, gl.uint32, layout),
                mailbox,
                source * SLOT + offsets,
                mask=in_rows[:, None],
                cache=".wt",
            )
        gl.store(out_ptr + row, best, mask=in_rows)
    gl.store(calls_ptr, calls + 1)
