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

"""Row-owned Lamport all-reduce fused with the residual add and RMSNorm."""

import torch
from tokenspeed_kernel._triton import gl, gluon, triton
from tokenspeed_kernel.ops.communication._iris import IrisGroup, allocation_bytes
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_group_heap_base,
    _iris_sanitize_lamport_bf16x2,
    _iris_subgroup_any,
    iris_arch_policy,
)
from tokenspeed_kernel.platform import current_platform


class LamportResidualRMSNorm:
    """Own the scatter and gather mailboxes of one group's fused reduction.

    A launch gives each rank a block of ``ceil(rows / world)`` rows. Every rank
    pushes its BF16 partial of a row to the row's owner; the owner sums the
    partials in FP32 rank order, rounds once to BF16 and pushes the reduced
    row to every rank, which adds its residual and normalizes. Mailbox words
    hold BF16 pairs and an empty marker. A consumer restores the markers of
    what it read: an owner before it publishes the reduced row, which is what
    lets a peer reuse its scatter slot in the next launch; a rank's gather
    slots before its launch ends, which precedes the next launch's scatter.
    So the mailboxes need no stage counters, and launches may use any row
    count up to the prepared capacity, in any order, on every rank alike.
    """

    # Column chunks per row, each pushed by one workgroup; one more
    # workgroup per row waits for the reduced row and normalizes it.
    CHUNKS = 8
    NUM_WARPS = 16
    # Owners' chunk workgroups spin until every peer's partial lands, so a
    # launch relies on all of them being resident at once: at 128 rows that
    # is up to 32 owned rows x 8 chunks per rank (TP4), about one 16-wave
    # workgroup per CU on gfx950.
    MAX_ROWS = 128

    @staticmethod
    def supported(world_size: int, width: int) -> bool:
        """Whether the kernel's layouts cover a row of ``width`` BF16 columns.

        A row is split into one 32-bit word per lane per chunk at most, and
        the normalizing workgroup holds whole BF16 pairs in every lane.
        """
        threads = 64 * LamportResidualRMSNorm.NUM_WARPS
        return (
            current_platform().is_cdna4
            and world_size in (4, 8)
            and width & (width - 1) == 0
            and 2 * threads <= width <= 2 * threads * LamportResidualRMSNorm.CHUNKS
        )

    @staticmethod
    def mailbox_shapes(
        world_size: int, max_rows: int, width: int
    ) -> tuple[tuple[int, int, int], tuple[int, int]]:
        """The BF16 shapes of the two mailboxes on the Iris heap: ``scatter``
        (a slot per source rank for each owned row) and ``gather`` (every row)."""
        max_local_rows = triton.cdiv(max_rows, world_size)
        return (world_size, max_local_rows, width), (max_rows, width)

    @staticmethod
    def heap_bytes(world_size: int, max_rows: int, width: int) -> int:
        """The Iris heap bytes an instance allocates, at the allocator's
        alignment: for reserving them before the heap exists
        (``_iris.reserve_heap``)."""
        return sum(
            allocation_bytes(shape, torch.bfloat16)
            for shape in LamportResidualRMSNorm.mailbox_shapes(
                world_size, max_rows, width
            )
        )

    def __init__(self, peers: IrisGroup, max_rows: int, width: int):
        if not self.supported(peers.world_size, width) or not (
            1 <= max_rows <= self.MAX_ROWS
        ):
            raise ValueError(
                f"unsupported fused residual RMSNorm: world={peers.world_size}, "
                f"width={width}, max_rows={max_rows}"
            )
        self.peers = peers
        self.max_rows = max_rows
        self.width = width
        scatter, gather = self.mailbox_shapes(peers.world_size, max_rows, width)
        self.max_local_rows = scatter[1]
        # scatter[source rank][owned row slot] receives each peer's partial.
        self.scatter = peers.ctx.empty(scatter, dtype=torch.bfloat16)
        # gather[row] receives the row's reduced BF16 values from its owner.
        self.gather = peers.ctx.empty(gather, dtype=torch.bfloat16)
        for mailbox in (self.scatter, self.gather):
            # An empty word's high half is negative zero, which no payload sends.
            mailbox.view(torch.int32).fill_(-(1 << 31))

    def run(
        self,
        tensor: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(rmsnorm(sum + residual) * weight, sum + residual)``.

        ``sum`` is the BF16-rounded group sum of ``tensor``; the residual add
        stays in FP32 for the norm and is rounded to BF16 when stored, as
        ``rmsnorm(x, weight, eps, residual=residual)`` computes it. Both
        outputs are new tensors.
        """
        rows, width = tensor.shape
        assert 0 < rows <= self.max_rows and width == self.width
        assert residual.shape == tensor.shape and weight.shape == (width,)
        assert tensor.dtype == residual.dtype == torch.bfloat16
        # Local layout fix-ups only: every rank still runs the same launch.
        if not tensor.is_contiguous() or tensor.data_ptr() % 4:
            tensor = tensor.clone(memory_format=torch.contiguous_format)
        residual = residual.contiguous()
        weight = weight.contiguous()
        norm_out = torch.empty_like(tensor)
        residual_out = torch.empty_like(tensor)
        world_size = self.peers.world_size
        policy = iris_arch_policy()
        iris_lamport_residual_rmsnorm[(rows, self.CHUNKS + 1)](
            tensor,
            residual,
            weight,
            norm_out,
            residual_out,
            self.scatter,
            self.gather,
            self.peers.addresses[:world_size],
            rows,
            RANK=self.peers.rank,
            WIDTH=width,
            MAX_LOCAL_ROWS=self.max_local_rows,
            EPS=eps,
            CHUNKS=self.CHUNKS,
            ELEMENTS_PER_THREAD=width // (64 * self.NUM_WARPS),
            NUM_WARPS=self.NUM_WARPS,
            PEER_SIGNAL_STORE_CACHE=policy.peer_signal_store,
            WAIT_STORES=policy.wait_stores,
            num_warps=self.NUM_WARPS,
        )
        return norm_out, residual_out


@gluon.jit
def _iris_pack_bf16x2(even, odd):
    """Round FP32 pairs to BF16 and pack each pair into one 32-bit word."""
    low = even.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32)
    high = odd.to(gl.bfloat16).to(gl.uint16, bitcast=True).to(gl.uint32)
    return low | (high << 16)


@gluon.jit
def _iris_residual_rmsnorm_chunk(
    input_ptr,
    scatter_ptr,
    gather_ptr,
    heaps,
    row,
    chunk,
    rows,
    RANK: gl.constexpr,
    WIDTH: gl.constexpr,
    MAX_LOCAL_ROWS: gl.constexpr,
    CHUNKS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
    WAIT_STORES: gl.constexpr,
):
    """Send one column chunk of a row to its owner, or reduce it as the owner.

    The owner waits for every peer's chunk, sums in FP32 rank order, rounds
    once to BF16, restores its scatter slot's markers and only then pushes the
    reduced chunk to every rank's gather mailbox, itself included.
    """
    WORLD_SIZE: gl.constexpr = len(heaps)
    PAIRS: gl.constexpr = WIDTH // 2
    CHUNK: gl.constexpr = PAIRS // CHUNKS
    THREADS: gl.constexpr = 64 * NUM_WARPS
    PEER_STRIDE: gl.constexpr = MAX_LOCAL_ROWS * PAIRS
    # One word per lane; chunks smaller than the workgroup leave lanes idle.
    gl.static_assert(CHUNK <= THREADS)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    index = gl.arange(0, THREADS, layout=layout)
    valid = index < CHUNK
    pair = chunk * CHUNK + index
    local_rows = (rows + WORLD_SIZE - 1) // WORLD_SIZE
    owner = row // local_rows
    slot = row - owner * local_rows
    words = gl.pointer_type(gl.uint32)
    local = _iris_sanitize_lamport_bf16x2(
        gl.amd.cdna4.buffer_load(
            input_ptr.to(words, bitcast=True), row * PAIRS + pair, mask=valid
        )
    )
    local_base = _iris_group_heap_base(RANK, heaps)
    scatter_offset = scatter_ptr.to(gl.uint64) - local_base
    if owner != RANK:
        destination = gl.multiple_of(
            (_iris_group_heap_base(owner, heaps) + scatter_offset).to(words), 16
        )
        gl.amd.cdna4.buffer_store(
            local,
            destination,
            RANK * PEER_STRIDE + slot * PAIRS + pair,
            mask=valid,
            cache=PEER_SIGNAL_STORE_CACHE,
        )
    else:
        mailbox = scatter_ptr.to(words, bitcast=True)
        offsets = slot * PAIRS + pair
        received = ()
        for _ in gl.static_range(1, WORLD_SIZE):
            received += (gl.zeros_like(local),)
        waiting = True
        while waiting:
            # Issue every peer's read before checking any; arrived words stay
            # unchanged until this owner restores them, so retries reread all.
            missing = gl.zeros_like(index)
            loaded = ()
            for delta in gl.static_range(1, WORLD_SIZE):
                peer = (RANK + delta) % WORLD_SIZE
                arrived = gl.amd.cdna4.buffer_load(
                    mailbox, peer * PEER_STRIDE + offsets, mask=valid, cache=".cv"
                )
                loaded += (arrived,)
                # Payload words never carry negative zero in the high half.
                missing |= ((arrived >> 16) == 0x8000).to(gl.int32)
            received = loaded
            waiting = _iris_subgroup_any(missing)
        # Widening BF16 to FP32 places its bits in the high half of the word.
        for peer in gl.static_range(0, WORLD_SIZE):
            if peer == RANK:
                term = local
            else:
                term = received[(peer - RANK + WORLD_SIZE) % WORLD_SIZE - 1]
            even = (term << 16).to(gl.float32, bitcast=True)
            odd = (term & 0xFFFF0000).to(gl.float32, bitcast=True)
            if peer == 0:
                sum_even = even
                sum_odd = odd
            else:
                sum_even += even
                sum_odd += odd
        reduced = _iris_sanitize_lamport_bf16x2(_iris_pack_bf16x2(sum_even, sum_odd))
        empty = gl.full(index.shape, 0x80000000, gl.uint32, layout)
        for delta in gl.static_range(1, WORLD_SIZE):
            peer = (RANK + delta) % WORLD_SIZE
            gl.amd.cdna4.buffer_store(
                empty, mailbox, peer * PEER_STRIDE + offsets, mask=valid, cache=".wt"
            )
        # The restored markers must land before any peer can see the reduced
        # row: that is its permission to reuse this slot.
        WAIT_STORES(args=[], constraints="=r,~{memory}")
        gather_offset = gather_ptr.to(gl.uint64) - local_base
        for delta in gl.static_range(0, WORLD_SIZE):
            peer = (RANK + delta) % WORLD_SIZE
            destination = gl.multiple_of(
                (_iris_group_heap_base(peer, heaps) + gather_offset).to(words), 16
            )
            gl.amd.cdna4.buffer_store(
                reduced,
                destination,
                row * PAIRS + pair,
                mask=valid,
                cache=PEER_SIGNAL_STORE_CACHE,
            )


@gluon.jit
def _iris_residual_rmsnorm_finish(
    gather_ptr,
    residual_ptr,
    weight_ptr,
    norm_out_ptr,
    residual_out_ptr,
    row,
    WIDTH: gl.constexpr,
    EPS: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Wait for a reduced row, then add the residual and normalize it."""
    PAIRS: gl.constexpr = WIDTH // 2
    THREADS: gl.constexpr = 64 * NUM_WARPS
    WORDS: gl.constexpr = ELEMENTS_PER_THREAD // 2
    gl.static_assert(THREADS * ELEMENTS_PER_THREAD == WIDTH)
    layout: gl.constexpr = gl.BlockedLayout(
        [ELEMENTS_PER_THREAD], [64], [NUM_WARPS], [0]
    )
    word_layout: gl.constexpr = gl.BlockedLayout(
        [1, WORDS], [64, 1], [NUM_WARPS, 1], [1, 0]
    )
    # Overlap the residual and weight reads with the wait.
    columns = gl.arange(0, WIDTH, layout=layout)
    offsets = row * WIDTH + columns
    residual = gl.amd.cdna4.buffer_load(residual_ptr, offsets).to(gl.float32)
    weight = gl.amd.cdna4.buffer_load(weight_ptr, columns).to(gl.float32)
    lane = gl.arange(0, THREADS, layout=gl.SliceLayout(1, word_layout))
    word = gl.arange(0, WORDS, layout=gl.SliceLayout(0, word_layout))
    word_offsets = row * PAIRS + lane[:, None] * WORDS + word[None, :]
    mailbox = gather_ptr.to(gl.pointer_type(gl.uint32), bitcast=True)
    words = gl.amd.cdna4.buffer_load(mailbox, word_offsets, cache=".cv")
    missing = gl.max(((words >> 16) == 0x8000).to(gl.int32), 1)
    while _iris_subgroup_any(missing):
        words = gl.amd.cdna4.buffer_load(mailbox, word_offsets, cache=".cv")
        missing = gl.max(((words >> 16) == 0x8000).to(gl.int32), 1)
    gl.amd.cdna4.buffer_store(
        gl.full_like(words, 0x80000000), mailbox, word_offsets, cache=".wt"
    )
    even = (words << 16).to(gl.float32, bitcast=True)
    odd = (words & 0xFFFF0000).to(gl.float32, bitcast=True)
    reduced = gl.convert_layout(gl.reshape(gl.join(even, odd), [WIDTH]), layout)
    total = reduced + residual
    gl.amd.cdna4.buffer_store(total.to(gl.bfloat16), residual_out_ptr, offsets)
    total *= gl.rsqrt(gl.sum(total * total, 0) / WIDTH + EPS)
    gl.amd.cdna4.buffer_store((total * weight).to(gl.bfloat16), norm_out_ptr, offsets)


@gluon.jit(do_not_specialize=["rows"])
def iris_lamport_residual_rmsnorm(
    input_ptr,
    residual_ptr,
    weight_ptr,
    norm_out_ptr,
    residual_out_ptr,
    scatter_ptr,
    gather_ptr,
    heaps,
    rows,
    RANK: gl.constexpr,
    WIDTH: gl.constexpr,
    MAX_LOCAL_ROWS: gl.constexpr,
    EPS: gl.constexpr,
    CHUNKS: gl.constexpr,
    ELEMENTS_PER_THREAD: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
    WAIT_STORES: gl.constexpr,
):
    """All-reduce BF16 rows across TP4/TP8, then add the residual and RMSNorm.

    Grid (rows, CHUNKS + 1). Workgroups (row, c < CHUNKS) move column chunk c
    of the row: a non-owner pushes its partial to the row's owner, the owner
    reduces and broadcasts it. Workgroup (row, CHUNKS) waits for the reduced
    row, then writes ``residual_out = bf16(sum + residual)`` and
    ``norm_out = bf16(rmsnorm(sum + residual) * weight)`` from the FP32 sum.
    The normalizing workgroups are dispatched last, so the chunk workgroups
    they and the owners wait for are never stuck behind them. Negative zero
    is sent as zero.
    """
    gl.static_assert(len(heaps) == 4 or len(heaps) == 8)
    row = gl.program_id(0)
    role = gl.program_id(1)
    if role == CHUNKS:
        _iris_residual_rmsnorm_finish(
            gather_ptr,
            residual_ptr,
            weight_ptr,
            norm_out_ptr,
            residual_out_ptr,
            row,
            WIDTH,
            EPS,
            ELEMENTS_PER_THREAD,
            NUM_WARPS,
        )
    else:
        _iris_residual_rmsnorm_chunk(
            input_ptr,
            scatter_ptr,
            gather_ptr,
            heaps,
            row,
            role,
            rows,
            RANK,
            WIDTH,
            MAX_LOCAL_ROWS,
            CHUNKS,
            NUM_WARPS,
            PEER_SIGNAL_STORE_CACHE,
            WAIT_STORES,
        )
