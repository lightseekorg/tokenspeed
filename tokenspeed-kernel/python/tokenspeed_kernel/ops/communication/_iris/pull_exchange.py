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

"""Gathers and all-to-alls that pull a symmetric buffer producers fill in place."""

import torch
from tokenspeed_kernel._triton import gl, gluon, triton
from tokenspeed_kernel.ops.communication._iris import IrisGroup, allocation_bytes
from tokenspeed_kernel.ops.communication._iris.all_reduce import _iris_subgroup_any
from tokenspeed_kernel.platform import current_platform

# Words one subgroup moves per tile: 64 lanes of 16 bytes.
_TILE_WORDS = 256


@gluon.jit
def _peer_rank_flags(flags, heaps, peers, RANK: gl.constexpr, WORLD: gl.constexpr):
    """Each lane's peer's copy of ``flags[RANK]``."""
    bases = peers.to(gl.uint64) * 0
    for peer in gl.static_range(WORLD):
        bases = gl.where(peers == peer, heaps[peer], bases)
    offset = flags.to(gl.uint64) - heaps[RANK]
    return (bases + offset).to(gl.pointer_type(gl.int32)) + RANK


@gluon.jit
def _wait_rank_flags(flags, peers, epoch, RANK: gl.constexpr, WORLD: gl.constexpr):
    """Spin until every peer's entry of local ``flags`` reaches ``epoch``.

    Accepts newer epochs across int32 wraparound. Subgroups poll on their
    own, so callers must not hold a workgroup barrier inside.
    """
    polled = (peers < WORLD) & (peers != RANK)
    waiting = _iris_subgroup_any(polled.to(gl.int32))
    while waiting:
        seen = gl.amd.cdna4.buffer_load(flags, peers, mask=polled, other=0, cache=".cv")
        waiting = _iris_subgroup_any((polled & ((seen - epoch) < 0)).to(gl.int32))


@gluon.jit(do_not_specialize=["words", "source_offset"])
def _iris_pull_exchange(
    output_ptr,
    buffer_ptr,
    ready_ptr,
    done_ptr,
    state_ptr,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    words,
    source_offset,
    RANK: gl.constexpr,
    WORLD: gl.constexpr,
    ITERS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """``output[p * words + i] = buffer_p[source_offset + i]`` for every rank
    ``p`` and ``i < words``, over int32 words.

    Program 0 releases this rank's buffer, which earlier kernels on the stream
    wrote, and publishes the call's epoch in each peer's ``ready``. Every
    subgroup waits for all peers' epochs, then reads each peer's buffer once,
    as the staged all-reduce reads its peers' inputs. The last program to
    finish publishes the epoch in each peer's ``done`` and waits for every
    peer's, so no rank's next producer overwrites a buffer a peer still reads.
    """
    heaps = (
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [NUM_WARPS, 1], [1, 0])
    LANES: gl.constexpr = 64 * NUM_WARPS
    lane = gl.arange(0, LANES, layout=gl.SliceLayout(1, layout))
    element = gl.arange(0, 4, layout=gl.SliceLayout(0, layout))
    # Every subgroup polls the peers on its first lanes.
    peers = lane % 64
    remote = (peers < WORLD) & (peers != RANK) & (lane < 64)
    epoch = gl.load(state_ptr) + 1
    pid = gl.program_id(0)
    if pid == 0:
        gl.atomic_xchg(
            _peer_rank_flags(ready_ptr, heaps, peers, RANK, WORLD),
            epoch,
            mask=remote,
            sem="release",
            scope="sys",
        )
    _wait_rank_flags(ready_ptr, peers, epoch, RANK, WORLD)
    offset = buffer_ptr.to(gl.uint64) - heaps[RANK]
    sources = ()
    for source in gl.static_range(WORLD):
        sources += (
            gl.multiple_of(
                (heaps[source] + offset).to(gl.pointer_type(gl.int32)) + source_offset,
                16,
            ),
        )
    # Programs start on different ranks to spread the reads over the links.
    rotation = pid % WORLD
    for it in gl.static_range(ITERS):
        rows = (pid * ITERS + it) * LANES + lane
        offsets = rows[:, None] * 4 + element[None, :]
        mask = offsets < words
        for first in gl.static_range(WORLD):
            if rotation == first:
                loaded = ()
                for step in gl.static_range(WORLD):
                    loaded += (
                        gl.amd.cdna4.buffer_load(
                            sources[(RANK + first + step) % WORLD],
                            offsets,
                            mask=mask,
                            other=0,
                            cache=".cg",
                        ),
                    )
                for step in gl.static_range(WORLD):
                    target = output_ptr + (RANK + first + step) % WORLD * words
                    gl.amd.cdna4.buffer_store(
                        loaded[step], gl.multiple_of(target, 16), offsets, mask=mask
                    )
    gl.barrier()
    # This program's loads returned before its stores of them could issue.
    finished = gl.atomic_add(state_ptr + 1, 1, sem="relaxed", scope="gpu")
    if finished == gl.num_programs(0) - 1:
        gl.atomic_xchg(
            _peer_rank_flags(done_ptr, heaps, peers, RANK, WORLD),
            epoch,
            mask=remote,
            sem="relaxed",
            scope="sys",
        )
        _wait_rank_flags(done_ptr, peers, epoch, RANK, WORLD)
        gl.store(state_ptr, epoch)
        gl.store(state_ptr + 1, 0)


def _words(nbytes: int) -> int:
    return (nbytes + 15) // 16 * 4


def _launch_shape(words: int) -> tuple[int, int]:
    """Subgroups and tiles per program for a ``words``-word share: one-subgroup
    programs spread a small exchange over the most compute units."""
    if words <= 16384:
        return 1, 1
    if words <= 65536:
        return 2, 1
    return 2, 2


class IrisPullExchange:
    """A symmetric buffer each rank's producers fill in place, which a gather or
    an all-to-all then pulls from every rank.

    :meth:`input` borrows the start of the buffer until the next exchange's
    producer overwrites it. A call waits for every peer's producer before
    pulling and for every peer's pulls before it returns, so calls ordered on
    one stream may reuse the buffer back to back. Every rank of the group makes
    the same calls.
    """

    @staticmethod
    def heap_bytes(max_bytes: int, world_size: int) -> int:
        return allocation_bytes(
            (_words(max_bytes),), torch.int32
        ) + 2 * allocation_bytes((world_size,), torch.int32)

    def __init__(self, peers: IrisGroup, max_bytes: int) -> None:
        if not current_platform().is_cdna4 or not 2 <= peers.world_size <= 8:
            raise ValueError(
                "Iris pull exchanges need two to eight CDNA4 ranks, got "
                f"{peers.world_size}"
            )
        self.peers = peers
        self.buffer = peers.ctx.zeros((_words(max_bytes),), dtype=torch.int32)
        self.ready = peers.ctx.zeros((peers.world_size,), dtype=torch.int32)
        self.done = peers.ctx.zeros((peers.world_size,), dtype=torch.int32)
        # The last call's epoch, then the current call's finished programs.
        self.state = torch.zeros((2,), dtype=torch.int32, device=peers.device)

    @property
    def max_bytes(self) -> int:
        return self.buffer.numel() * 4

    def input(self, nbytes: int) -> torch.Tensor | None:
        """The buffer's first ``nbytes`` (uint8), or None past its capacity."""
        if nbytes > self.max_bytes:
            return None
        return self.buffer.view(torch.uint8)[:nbytes]

    def _takes(self, output: torch.Tensor, input: torch.Tensor, share: int) -> bool:
        return (
            input.data_ptr() == self.buffer.data_ptr()
            and input.is_contiguous()
            and input.nbytes <= self.max_bytes
            and share > 0
            and share % 16 == 0
            and output.device == self.buffer.device
            and output.is_contiguous()
            and output.data_ptr() % 16 == 0
        )

    def can_all_gather(self, output: torch.Tensor, input: torch.Tensor) -> bool:
        """Whether :meth:`all_gather` serves ``input``, a view from :meth:`input`."""
        return (
            self._takes(output, input, input.nbytes)
            and output.nbytes == self.peers.world_size * input.nbytes
        )

    def can_all_to_all(self, output: torch.Tensor, input: torch.Tensor) -> bool:
        """Whether :meth:`all_to_all` serves ``input``, a view from :meth:`input`."""
        world = self.peers.world_size
        return (
            input.nbytes % world == 0
            and self._takes(output, input, input.nbytes // world)
            and output.nbytes == input.nbytes
        )

    def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None:
        """Every rank's ``input`` bytes into ``output``, in rank order."""
        self._pull(output, input.nbytes // 4, 0)

    def all_to_all(self, output: torch.Tensor, input: torch.Tensor) -> None:
        """Every rank's ``input`` chunk for this rank into ``output``, in rank
        order: ``input``'s bytes split evenly by destination rank."""
        words = input.nbytes // 4 // self.peers.world_size
        self._pull(output, words, self.peers.rank * words)

    def _pull(self, output: torch.Tensor, words: int, source_offset: int) -> None:
        warps, iters = _launch_shape(words)
        _iris_pull_exchange[(triton.cdiv(words, _TILE_WORDS * warps * iters),)](
            output.view(-1).view(torch.int32),
            self.buffer,
            self.ready,
            self.done,
            self.state,
            *self.peers.addresses,
            words,
            source_offset,
            RANK=self.peers.rank,
            WORLD=self.peers.world_size,
            ITERS=iters,
            NUM_WARPS=warps,
            num_warps=warps,
        )
