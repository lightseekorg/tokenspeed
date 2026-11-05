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


@gluon.jit
def _sum_of_eight(x):
    """``x[0] + ... + x[7]`` paired as Triton's sum over eight lanes pairs them."""
    return ((x[0] + x[4]) + (x[2] + x[6])) + ((x[1] + x[5]) + (x[3] + x[7]))


@gluon.jit(do_not_specialize=["records"])
def _iris_pull_combine_records(
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
    records,
    RANK: gl.constexpr,
    WORLD: gl.constexpr,
    WORDS: gl.constexpr,
    BLOCK: gl.constexpr,
    ITERS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    MIN_NORMAL: gl.constexpr,
):
    """``combine_dcp_attention_partials`` without a sink of the ``records``
    DCP records every rank's buffer holds for this rank (from word ``RANK *
    records * (WORDS + 1)``: ``WORDS`` words of BF16 pairs, then the FP32
    LSE), bit for bit, into ``output``'s ``records`` rows of BF16 pairs.

    Pulls as :func:`_iris_pull_exchange` does, with its handshake; a
    subgroup merges one record at a time, the ranks in order.
    """
    gl.static_assert(WORLD == 8)
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
    flag_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    layout: gl.constexpr = gl.BlockedLayout(
        [1, BLOCK // 64], [1, 64], [NUM_WARPS, 1], [1, 0]
    )
    lane = gl.arange(0, 64 * NUM_WARPS, layout=flag_layout)
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
            (heaps[source] + offset).to(gl.pointer_type(gl.int32))
            + RANK * records * (WORDS + 1),
        )
    row = gl.arange(0, NUM_WARPS, layout=gl.SliceLayout(1, layout))
    col = gl.arange(0, BLOCK, layout=gl.SliceLayout(0, layout))
    for it in gl.static_range(ITERS):
        record = (pid * ITERS + it) * NUM_WARPS + row
        live = record < records
        words = record[:, None] * (WORDS + 1) + col[None, :]
        mask = live[:, None] & (col[None, :] < WORDS)
        lses = ()
        packed = ()
        for source in gl.static_range(WORLD):
            bits = gl.amd.cdna4.buffer_load(
                sources[source],
                record * (WORDS + 1) + WORDS,
                mask=live,
                other=0,
                cache=".cg",
            )
            lse = bits.to(gl.float32, bitcast=True)
            lses += (gl.where(lse == float("inf"), -float("inf"), lse),)
            packed += (
                gl.amd.cdna4.buffer_load(
                    sources[source], words, mask=mask, other=0, cache=".cg"
                ),
            )
        maximum = lses[0]
        has_nan = lses[0] != lses[0]
        for source in gl.static_range(1, WORLD):
            maximum = gl.maximum(maximum, lses[source])
            has_nan = has_nan | (lses[source] != lses[source])
        safe_max = gl.where(maximum == -float("inf"), 0.0, maximum)
        scaled = ()
        for source in gl.static_range(WORLD):
            scaled += (gl.exp(lses[source] - safe_max),)
        denominator = gl.maximum(_sum_of_eight(scaled), MIN_NORMAL)
        low = gl.zeros([NUM_WARPS, BLOCK], gl.float32, layout)
        high = gl.zeros([NUM_WARPS, BLOCK], gl.float32, layout)
        for source in gl.static_range(WORLD):
            weight = gl.where(has_nan, float("nan"), scaled[source] / denominator)
            empty = (lses[source] == -float("inf"))[:, None]
            # BF16 bits are the upper half of the FP32 they widen to.
            word = packed[source]
            values = gl.where(empty, 0.0, (word << 16).to(gl.float32, bitcast=True))
            low = low + values * weight[:, None]
            values = gl.where(empty, 0.0, (word & -65536).to(gl.float32, bitcast=True))
            high = high + values * weight[:, None]
        low_bits = low.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.int32)
        high_bits = high.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.int32)
        gl.amd.cdna4.buffer_store(
            low_bits | (high_bits << 16),
            output_ptr,
            record[:, None] * WORDS + col[None, :],
            mask=mask,
        )
    gl.barrier()
    # Every merged word depends on its loads, which so returned before the
    # stores issued.
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


def _combine_shape(records: int) -> tuple[int, int]:
    """Subgroups and records per subgroup of a ``records``-record combine:
    one-subgroup programs spread a small combine over the most compute units."""
    if records <= 48:
        return 1, 1
    return 4, 1


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

    def can_combine_records(
        self,
        records: torch.Tensor,
        *,
        head_dim: int,
        partial_dtype: torch.dtype,
        output_dtype: torch.dtype,
    ) -> bool:
        """Whether :meth:`combine_records` serves ``records``, a nonempty view
        from :meth:`input`: eight ranks' BF16 records combined into BF16."""
        return (
            self.peers.world_size == 8
            and records.numel() > 0
            and records.data_ptr() == self.buffer.data_ptr()
            and records.dtype == torch.int32
            and records.dim() == 4
            and records.is_contiguous()
            and records.shape[0] == 8
            and records.nbytes <= self.max_bytes
            and partial_dtype == output_dtype == torch.bfloat16
            and head_dim % 2 == 0
            and 0 < head_dim // 2 <= 512
            and records.shape[3] == head_dim // 2 + 1
        )

    def combine_records(self, records: torch.Tensor, *, head_dim: int) -> torch.Tensor:
        """``combine_dcp_attention_partials`` (BF16, no sink) of the records
        each rank's ``records`` holds for this rank, as an all-to-all
        delivers them, bit for bit: ``[tokens, local_heads, head_dim]``."""
        _, tokens, heads, _ = records.shape
        output = torch.empty(
            (tokens, heads, head_dim), dtype=torch.bfloat16, device=records.device
        )
        count = tokens * heads
        words = head_dim // 2
        warps, iters = _combine_shape(count)
        _iris_pull_combine_records[(triton.cdiv(count, warps * iters),)](
            output.view(torch.int32),
            self.buffer,
            self.ready,
            self.done,
            self.state,
            *self.peers.addresses,
            count,
            RANK=self.peers.rank,
            WORLD=self.peers.world_size,
            WORDS=words,
            BLOCK=max(64, triton.next_power_of_2(words)),
            ITERS=iters,
            NUM_WARPS=warps,
            MIN_NORMAL=torch.finfo(torch.float32).tiny,
            num_warps=warps,
            enable_fp_fusion=False,
        )
        return output

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
