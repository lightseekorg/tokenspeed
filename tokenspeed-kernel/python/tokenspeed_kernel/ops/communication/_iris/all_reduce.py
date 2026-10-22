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

"""Iris all-reduce protocols and the synchronization used by its fusions."""

import dataclasses
import math

import torch
from tokenspeed_kernel._triton import gl, gluon, tl, triton
from tokenspeed_kernel.ops.communication._iris import IrisGroup, allocation_bytes, iris
from tokenspeed_kernel.ops.communication.iris import (
    _PACKED_WORD_BYTES,
    _PRODUCER_DIRECT_GL_DTYPES,
    _TWO_SHOT_WORLD_SIZES,
    _use_two_stage_plain,
    _use_two_stage_producer_direct,
    producer_direct_all_reduce_can_run,
)
from tokenspeed_kernel.platform import ArchVersion, current_platform

hip_utils = triton.language.extra.hip.utils

_PULL_MAX_PROGRAMS = 84
_PULL_WORDS_PER_LANE = 2
_TWO_SHOT_NUM_SUBGROUPS = 8


@dataclasses.dataclass(frozen=True)
class IrisCachePolicy:
    """Cache modifiers for Iris accesses whose policy differs by architecture.

    Flag polls (".cv"), stores publishing into the local heap (".wt"), and
    system-scope atomics are the same on every architecture and stay literal.
    Kernels built only for gfx950 may keep literals for these roles too.

    Attributes:
        peer_payload_load: Streaming load of data a peer published, issued
            after its readiness is observed; empty selects the default.
        peer_signal_store: Store into a peer's memory that the peer polls,
            such as a barrier epoch or a Lamport mailbox; empty selects the
            default.
    """

    peer_payload_load: str
    peer_signal_store: str


_IRIS_CACHE_POLICIES = {
    ArchVersion(9, 5): IrisCachePolicy(
        peer_payload_load=".cg",
        peer_signal_store=".wt",
    ),
    # On gfx1250 a ".cg" load lowers to a non-temporal load that refetches every
    # partially read sector over the fabric, and ".wt" peer stores are an order
    # of magnitude slower than plain ones.
    ArchVersion(12, 5): IrisCachePolicy(
        peer_payload_load="",
        peer_signal_store="",
    ),
}


def iris_cache_policy(arch: ArchVersion | None = None) -> IrisCachePolicy:
    """Return the Iris cache policy for ``arch``, or for the current device.

    Raises:
        ValueError: ``arch`` has no Iris cache policy.
    """
    if arch is None:
        arch = current_platform().arch_version
    policy = _IRIS_CACHE_POLICIES.get(arch)
    if policy is None:
        raise ValueError(f"Iris has no cache policy for {arch}")
    return policy


def _pull_scratch_numel(max_numel: int, world_size: int) -> int:
    return (
        triton.cdiv(max_numel, world_size) if world_size in _TWO_SHOT_WORLD_SIZES else 0
    )


class ProducerBuffer:
    """Own consecutive symmetric outputs written by upstream producers."""

    @staticmethod
    def heap_bytes(max_numel: int, dtype: torch.dtype) -> int:
        return allocation_bytes((max_numel,), dtype)

    def __init__(self, peers: IrisGroup, max_numel: int, dtype: torch.dtype):
        self.peers = peers
        self.data = peers.ctx.zeros((max_numel,), dtype=dtype)

    @staticmethod
    def views(
        buffer: torch.Tensor, shapes: tuple[tuple[int, ...], ...]
    ) -> tuple[torch.Tensor, ...]:
        views = []
        offset = 0
        for shape in shapes:
            numel = math.prod(shape)
            views.append(buffer.narrow(0, offset, numel).view(shape))
            offset += numel
        return tuple(views)

    def acquire(self, shapes: tuple[tuple[int, ...], ...]) -> tuple[torch.Tensor, ...]:
        if not shapes or any(math.prod(shape) <= 0 for shape in shapes):
            raise ValueError("Iris requires non-empty symmetric output shapes")
        if sum(math.prod(shape) for shape in shapes) > self.data.numel():
            raise ValueError("Iris symmetric outputs exceed the input buffer")
        return self.views(self.data, shapes)

    def owns(self, tensors: tuple[torch.Tensor, ...]) -> bool:
        if not tensors or any(
            tensor.dtype != self.data.dtype
            or tensor.device != self.data.device
            or not tensor.is_contiguous()
            or tensor.numel() <= 0
            for tensor in tensors
        ):
            return False
        offset = 0
        for tensor in tensors:
            if (
                tensor.data_ptr()
                != self.data.data_ptr() + offset * self.data.element_size()
            ):
                return False
            offset += tensor.numel()
        elements_per_word = _PACKED_WORD_BYTES // self.data.element_size()
        return offset % elements_per_word == 0 and offset <= self.data.numel()


class StagedPullOneShot:
    """Own rotating slots and epochs for one staging geometry."""

    BLOCK_SIZE = 2048
    NUM_SUBGROUPS = 4
    INPUT_SLOTS = 2

    @staticmethod
    def heap_bytes(
        max_numel: int, world_size: int, dtype: torch.dtype, block_size: int
    ) -> int:
        return allocation_bytes(
            (StagedPullOneShot.INPUT_SLOTS, max_numel), dtype
        ) + allocation_bytes(
            (triton.cdiv(max_numel, block_size), world_size), torch.int32
        )

    def __init__(
        self,
        peers: IrisGroup,
        max_numel: int,
        dtype: torch.dtype,
        *,
        block_size: int,
        num_subgroups: int,
    ):
        self.peers = peers
        self.block_size = block_size
        self.num_subgroups = num_subgroups
        self.slots = peers.ctx.zeros((self.INPUT_SLOTS, max_numel), dtype=dtype)
        self.ready_flags = peers.ctx.zeros(
            (triton.cdiv(max_numel, self.block_size), peers.world_size),
            dtype=torch.int32,
        )

    def run(self, tensor: torch.Tensor, out: torch.Tensor) -> None:
        iris_pull_oneshot_staged[(triton.cdiv(tensor.numel(), self.block_size),)](
            tensor.view(-1),
            self.slots.view(-1),
            out.view(-1),
            self.ready_flags,
            self.peers.bases,
            tensor.numel(),
            RANK=self.peers.rank,
            WORLD_SIZE=self.peers.world_size,
            BLOCK_SIZE=self.block_size,
            SLOT_STRIDE=self.slots.shape[1],
            NUM_SLOTS=self.slots.shape[0],
            PEER_PAYLOAD_LOAD_CACHE=iris_cache_policy().peer_payload_load,
            num_warps=self.num_subgroups,
        )


def _launch_pull_twoshot(
    peers: IrisGroup,
    inputs: torch.Tensor,
    scratch: torch.Tensor,
    ready_flags: torch.Tensor,
    out: torch.Tensor,
    *,
    exit_barrier: bool,
    hc_post: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> None:
    policy = iris_cache_policy()
    elements_per_word = _PACKED_WORD_BYTES // inputs.element_size()
    numel = out.numel() if hc_post is None else out.numel() // 4
    partition_words = numel // peers.world_size // elements_per_word
    subgroup_size = triton.runtime.driver.active.get_current_target().warp_size
    block_words = (
        _TWO_SHOT_NUM_SUBGROUPS
        * subgroup_size
        * _PULL_WORDS_PER_LANE
        // peers.world_size
    )
    num_tiles = triton.cdiv(partition_words, block_words)
    num_programs = min(num_tiles, _PULL_MAX_PROGRAMS)
    iris_pull_twoshot[(num_programs,)](
        inputs,
        scratch,
        out,
        ready_flags,
        *peers.addresses,
        RANK=peers.rank,
        WORLD_SIZE=peers.world_size,
        PARTITION_WORDS=partition_words,
        BLOCK_WORDS=block_words,
        NUM_PROGRAMS=num_programs,
        NUM_TILES=num_tiles,
        NUM_WARPS=_TWO_SHOT_NUM_SUBGROUPS,
        WORDS_PER_LANE=_PULL_WORDS_PER_LANE,
        ELEMENT_DTYPE=_PRODUCER_DIRECT_GL_DTYPES[inputs.dtype],
        ELEMENTS_PER_WORD=elements_per_word,
        EXIT_BARRIER=exit_barrier,
        PEER_PAYLOAD_LOAD_CACHE=policy.peer_payload_load,
        PEER_SIGNAL_STORE_CACHE=policy.peer_signal_store,
        **_hc_post_args(hc_post, elements_per_word),
        num_warps=_TWO_SHOT_NUM_SUBGROUPS,
    )


def _hc_post_args(
    hc_post: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    elements_per_word: int,
) -> dict:
    if hc_post is None:
        return {}
    residual, post, comb = hc_post
    return {
        "residual": residual,
        "post": post,
        "comb": comb,
        "HIDDEN_WORDS": residual.shape[-1] // elements_per_word,
        "HC_POST": True,
    }


class StagedPullTwoShot:
    """Own stable staging, partition scratch, and completion epochs."""

    @staticmethod
    def supported(world_size: int, dtype: torch.dtype) -> bool:
        return (
            current_platform().is_cdna4
            and world_size in _TWO_SHOT_WORLD_SIZES
            and dtype in _PRODUCER_DIRECT_GL_DTYPES
        )

    @staticmethod
    def heap_bytes(max_numel: int, world_size: int, dtype: torch.dtype) -> int:
        return (
            allocation_bytes((max_numel,), dtype)
            + allocation_bytes((_pull_scratch_numel(max_numel, world_size),), dtype)
            + allocation_bytes((_PULL_MAX_PROGRAMS, world_size), torch.int32)
        )

    def __init__(self, peers: IrisGroup, max_numel: int, dtype: torch.dtype):
        self.peers = peers
        self.input = peers.ctx.zeros((max_numel,), dtype=dtype)
        self.scratch = peers.ctx.zeros(
            (_pull_scratch_numel(max_numel, peers.world_size),), dtype=dtype
        )
        self.ready_flags = peers.ctx.zeros(
            (_PULL_MAX_PROGRAMS, peers.world_size), dtype=torch.int32
        )

    def can_run(self, numel: int) -> bool:
        return 0 < numel <= self.input.numel() and _use_two_stage_plain(
            self.peers.world_size, numel, self.input.dtype
        )

    def run(self, tensor: torch.Tensor, out: torch.Tensor) -> None:
        self.input[: tensor.numel()].copy_(tensor.view(-1))
        # Packed stores require an aligned destination; ranks keep the same
        # collective when a caller supplies an element-aligned tensor view.
        aligned_output = (
            torch.empty_like(out) if out.data_ptr() % _PACKED_WORD_BYTES else out
        )
        # Graph replay reuses this staging address; peers must finish reading it.
        _launch_pull_twoshot(
            self.peers,
            self.input,
            self.scratch,
            self.ready_flags,
            aligned_output,
            exit_barrier=True,
        )
        if aligned_output is not out:
            out.copy_(aligned_output)


class ProducerPull:
    """Borrow producer input; own scratch and epochs shared by one-/two-shot pull."""

    BLOCK_SIZE = 512
    NUM_SUBGROUPS = 1

    @staticmethod
    def heap_bytes(max_numel: int, world_size: int, dtype: torch.dtype) -> int:
        return allocation_bytes(
            (_pull_scratch_numel(max_numel, world_size),), dtype
        ) + allocation_bytes((_PULL_MAX_PROGRAMS, world_size), torch.int32)

    def __init__(self, producer: ProducerBuffer):
        self.producer = producer
        peers = producer.peers
        scratch_numel = _pull_scratch_numel(producer.data.numel(), peers.world_size)
        self.scratch = (
            peers.ctx.zeros((scratch_numel,), dtype=producer.data.dtype)
            if scratch_numel
            else None
        )
        self.ready_flags = peers.ctx.zeros(
            (_PULL_MAX_PROGRAMS, peers.world_size), dtype=torch.int32
        )

    def can_run(self, shapes: tuple[tuple[int, ...], ...]) -> bool:
        data = self.producer.data
        return current_platform().is_cdna4 and producer_direct_all_reduce_can_run(
            self.producer.peers.world_size,
            sum(math.prod(shape) for shape in shapes),
            data.dtype,
            data.numel() * data.element_size(),
        )

    def run(
        self,
        out: torch.Tensor,
        hc_post: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Reduce the producer outputs into out.

        With hc_post = (residual [T,4,H], post [T,4], comb [T,4,4]), out is the
        new BF16 residual [T,4,H]: the mHC post of the reduced [T,H] rows.
        """
        peers = self.producer.peers
        inputs = self.producer.data
        numel = out.numel() if hc_post is None else out.numel() // 4
        if _use_two_stage_producer_direct(peers.world_size, numel, inputs.dtype):
            assert self.scratch is not None
            _launch_pull_twoshot(
                peers,
                inputs,
                self.scratch,
                self.ready_flags,
                out,
                exit_barrier=False,
                hc_post=hc_post,
            )
            return
        policy = iris_cache_policy()
        num_tiles = triton.cdiv(numel, self.BLOCK_SIZE)
        num_programs = min(num_tiles, _PULL_MAX_PROGRAMS)
        iris_pull_oneshot[(num_programs,)](
            inputs,
            out,
            self.ready_flags,
            *peers.addresses,
            RANK=peers.rank,
            WORLD_SIZE=peers.world_size,
            TOTAL_NUMEL=numel,
            BLOCK_SIZE=self.BLOCK_SIZE,
            NUM_PROGRAMS=num_programs,
            NUM_TILES=num_tiles,
            NUM_WARPS=self.NUM_SUBGROUPS,
            WORDS_PER_LANE=_PULL_WORDS_PER_LANE,
            PUBLISH_READY=False,
            ELEMENT_DTYPE=_PRODUCER_DIRECT_GL_DTYPES[inputs.dtype],
            ELEMENTS_PER_WORD=_PACKED_WORD_BYTES // inputs.element_size(),
            PEER_PAYLOAD_LOAD_CACHE=policy.peer_payload_load,
            PEER_SIGNAL_STORE_CACHE=policy.peer_signal_store,
            **_hc_post_args(hc_post, _PACKED_WORD_BYTES // inputs.element_size()),
            num_warps=self.NUM_SUBGROUPS,
        )


class ProducerLamport:
    """Own a TP8 BF16 Lamport mailbox for the shapes selected during preparation."""

    WORLD_SIZE = 8
    BLOCK_ELEMENTS = 512
    NUM_STAGES = 3
    NUM_SUBGROUPS = 1
    TRANSACTION_BYTES = 16

    @staticmethod
    def supported(world_size: int, dtype: torch.dtype) -> bool:
        return (
            current_platform().is_cdna4
            and world_size == ProducerLamport.WORLD_SIZE
            and dtype == torch.bfloat16
        )

    @staticmethod
    def heap_bytes(max_numel: int) -> int:
        return allocation_bytes(
            (ProducerLamport.NUM_STAGES, ProducerLamport.WORLD_SIZE, max_numel),
            torch.bfloat16,
        )

    def __init__(
        self,
        producer: ProducerBuffer,
        *,
        max_numel: int,
    ):
        self.producer = producer
        peers = producer.peers
        assert self.supported(peers.world_size, producer.data.dtype)
        assert 0 < max_numel <= producer.data.numel()
        assert max_numel % self.BLOCK_ELEMENTS == 0
        self.max_numel = max_numel
        self.mailbox = peers.ctx.zeros(
            (self.NUM_STAGES, peers.world_size, self.max_numel), dtype=torch.bfloat16
        )
        self.epochs = torch.zeros(
            (self.max_numel // self.BLOCK_ELEMENTS,),
            dtype=torch.int32,
            device=peers.device,
        )
        # Reserve negative zero as the mailbox sentinel.
        self.mailbox.view(torch.int32).fill_(-2147483648)
        offset = self.mailbox.data_ptr() - peers.addresses[peers.rank]
        self.peer_mailboxes = tuple(base + offset for base in peers.addresses)
        assert all(
            address % self.TRANSACTION_BYTES == 0 for address in self.peer_mailboxes
        )

    def run(self, out: torch.Tensor) -> None:
        iris_lamport_allreduce[(out.numel() // self.BLOCK_ELEMENTS,)](
            self.producer.data,
            self.mailbox,
            out,
            self.epochs,
            *self.peer_mailboxes,
            RANK=self.producer.peers.rank,
            WORLD_SIZE=self.producer.peers.world_size,
            TOTAL_ELEMENTS=out.numel(),
            MAX_ELEMENTS=self.max_numel,
            NUM_STAGES=self.NUM_STAGES,
            PEER_SIGNAL_STORE_CACHE=iris_cache_policy().peer_signal_store,
            num_warps=self.NUM_SUBGROUPS,
        )


def staged_tuning(
    max_numel: int, world_size: int, dtype: torch.dtype
) -> tuple[int, int, int] | None:
    """Return the measured TP4 geometry when this state can serve its payload."""
    # This exact CDNA4 TP4 payload is faster with the one-shot geometry.
    if (
        current_platform().is_cdna4
        and world_size == 4
        and dtype == torch.bfloat16
        and max_numel >= 16 * 4096
    ):
        return 16 * 4096, 512, 1
    return None


def all_reduce_heap_bytes(
    staged_max_numel: int,
    producer_max_numel: int,
    world_size: int,
    dtype: torch.dtype,
) -> int:
    size = 0
    if producer_max_numel:
        size += ProducerBuffer.heap_bytes(producer_max_numel, dtype)
        size += ProducerPull.heap_bytes(producer_max_numel, world_size, dtype)
    if staged_max_numel:
        size += StagedPullOneShot.heap_bytes(
            staged_max_numel, world_size, dtype, StagedPullOneShot.BLOCK_SIZE
        )
        tuning = staged_tuning(staged_max_numel, world_size, dtype)
        if tuning is not None:
            numel, block_size, _ = tuning
            size += StagedPullOneShot.heap_bytes(numel, world_size, dtype, block_size)
        if StagedPullTwoShot.supported(world_size, dtype):
            size += StagedPullTwoShot.heap_bytes(staged_max_numel, world_size, dtype)
    return size


@gluon.jit
def _iris_drain_subgroup_vmem():
    """Finish this subgroup's outstanding vector memory operations."""
    gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)",
        "=r,~{memory}",
        [],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _iris_subgroup_any(flags):
    """Return whether any lane of this subgroup holds a nonzero int32 flag.

    One v_cmp writes the subgroup's lane mask to scalar registers, so a loop
    can branch on it without a cross-lane reduction or, in multi-subgroup
    programs, a shared-memory exchange. The result is uniform per subgroup,
    not per workgroup: subgroups looping on it iterate independently, so such
    a loop must not contain a workgroup barrier. The memory clobber keeps the
    compiler from hoisting polling loads out of the loop. The 64-bit result is
    the whole lane mask only on wave64 (CDNA4) targets.

    Args:
        flags: Exactly one int32 per lane (the instruction takes one VGPR).
    """
    mask = gl.inline_asm(
        "v_cmp_ne_u32_e64 $0, 0, $1",
        "=s,v,~{memory}",
        [flags],
        gl.int64,
        is_pure=False,
    )
    return mask != 0


@triton.jit
def iris_pull_oneshot_staged(
    input_ptr,
    input_sym_ptr,
    output_ptr,
    ready_flags,
    heap_bases,
    NUMEL,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    PEER_PAYLOAD_LOAD_CACHE: tl.constexpr,
):
    """One-shot all-reduce over rotating staging slots.

    Copy input_ptr into input_sym_ptr's selected slot, sum peer slots in FP32,
    and write output_ptr at its supplied dtype.

    A single staging slot is not safe across calls. Each rank stages, publishes
    a ready epoch, waits for its peers, sums their slots and returns -- but the
    store happens *before* the wait, so a rank called straight back overwrites
    the slot a slower peer is still summing, and that peer silently sums the
    next collective's data. No error, no hang, just wrong numbers on whichever
    ranks lagged.

    Rotating over at least two slots closes it, and the existing entry barrier
    makes two sufficient. Writing epoch E+1 lands on a different slot. Before a
    rank can wrap back to E's slot, it must pass the entry wait at E+1; a peer
    publishes ready(E+1) only after it has finished reading E. By the time a
    slot is reused, every peer is provably done with it, and no consumption flag
    or exit barrier has to be paid for.

    Epochs are per-block, so a grid that shrinks between calls leaves the higher
    blocks' counters where they were. That is consistent across ranks -- every
    rank launches the same grid for the same collective -- and blocks never read
    each other's slots, so they may sit on different slots at once.

    Epochs are int32 and wrap after 2**31 calls per block. The slot index is
    taken from the unsigned epoch, so with a power-of-two slot count the
    rotation continues across the wrap, and the wait compares modulo 2**32,
    so an epoch just past the wrap still counts as newer.
    """
    # 2**32 must be a multiple of the slot count for the rotation to survive
    # the wrap.
    tl.static_assert(NUM_SLOTS & (NUM_SLOTS - 1) == 0)
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL

    flag_offset = block_id * WORLD_SIZE
    local_ready = ready_flags + flag_offset + RANK
    epoch = tl.load(local_ready).to(tl.int32) + 1
    slot = (epoch.to(tl.uint32, bitcast=True) % NUM_SLOTS).to(tl.int32) * SLOT_STRIDE

    local = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(input_sym_ptr + slot + offsets, local, mask=mask, cache_modifier=".wt")
    tl.debug_barrier()

    tl.atomic_xchg(local_ready, epoch, sem="release", scope="sys")
    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            # Start one behind so the first poll always happens, then compare
            # modulo 2**32: a peer may already have passed the wrap.
            seen = epoch - 1
            while (seen - epoch) < 0:
                seen = iris.load(
                    ready_flags + flag_offset + peer,
                    RANK,
                    peer,
                    heap_bases,
                    cache_modifier=".cv",
                    volatile=True,
                )

    acc = local.to(tl.float32)
    for peer in tl.static_range(0, WORLD_SIZE):
        if peer != RANK:
            acc += iris.load(
                input_sym_ptr + slot + offsets,
                RANK,
                peer,
                heap_bases,
                mask=mask,
                other=0.0,
                cache_modifier=PEER_PAYLOAD_LOAD_CACHE,
                hint=BLOCK_SIZE,
            ).to(tl.float32)
    tl.store(output_ptr + offsets, acc.to(output_ptr.type.element_ty), mask=mask)


@gluon.jit
def _iris_sanitize_lamport_bf16(values):
    """Replace negative zero with zero so BF16 data cannot match an empty marker."""
    bits = values.to(gl.uint16, bitcast=True)
    return gl.where(bits == 0x8000, 0, bits).to(gl.bfloat16, bitcast=True)


@gluon.jit
def _iris_sanitize_lamport_bf16x2(words):
    """Replace BF16 negative-zero halves so no payload word matches the empty marker."""
    words = gl.where((words & 0xFFFF) == 0x8000, words & 0xFFFF0000, words)
    return gl.where((words >> 16) == 0x8000, words & 0xFFFF, words)


@gluon.jit
def _iris_lamport_push(
    values,
    regions,
    offset,
    offsets,
    mask,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Store values[delta - 1] at offset in peer (RANK + delta)'s region.

    Each program starts at a different peer so that concurrent stores spread
    across all links instead of all targeting RANK + 1 first. One uniform
    branch selects the order, so peer bases stay scalar constants.
    """
    rotation = gl.program_id(0) % (WORLD_SIZE - 1)
    for first in gl.static_range(WORLD_SIZE - 1):
        if rotation == first:
            for step in gl.static_range(WORLD_SIZE - 1):
                # Preserve 16-byte stores after casting integer addresses.
                destination = gl.multiple_of(
                    regions[
                        (RANK + 1 + (first + step) % (WORLD_SIZE - 1)) % WORLD_SIZE
                    ].to(gl.pointer_type(gl.bfloat16)),
                    16,
                )
                gl.amd.cdna4.buffer_store(
                    values[(first + step) % (WORLD_SIZE - 1)],
                    destination + offset,
                    offsets,
                    mask=mask,
                    cache=PEER_SIGNAL_STORE_CACHE,
                )


@gluon.jit
def _iris_wait_lamport_slots(
    slots,
    offsets,
    valid,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    SLOT_ELEMENTS: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    """Wait for each peer's BF16 packs in its slot to replace the empty markers.

    Peer p's packs start at slots + p * SLOT_ELEMENTS. slots is 16-byte aligned,
    as Iris allocations are; each row of offsets indexes one pack of consecutive
    elements starting at a multiple of its width. Each subgroup polls the packs
    as 32-bit words until all of its valid lanes have every peer's pack, then
    returns them as BF16 in LAYOUT.
    """
    ROWS: gl.constexpr = offsets.shape[0]
    WORDS: gl.constexpr = offsets.shape[1] // 2
    gl.static_assert(offsets.shape[1] % 2 == 0 and SLOT_ELEMENTS % 8 == 0)
    # Peers store whole words. Polling words keeps each one a single load at any
    # pointer alignment; BF16 loads could split a word into a stale low half
    # and an arrived high half.
    words_ptr = gl.multiple_of(slots.to(gl.pointer_type(gl.uint32), bitcast=True), 16)
    first, _ = gl.split(gl.reshape(offsets, [ROWS, WORDS, 2]))
    # Each thread's words stay as consecutive as its BF16 elements.
    word_offsets = gl.max_contiguous(gl.multiple_of(first // 2, [1, WORDS]), [1, WORDS])
    word_valid = gl.convert_layout(valid, gl.SliceLayout(1, word_offsets.type.layout))
    loaded = ()
    for _ in gl.static_range(1, WORLD_SIZE):
        loaded += (gl.zeros_like(word_offsets).to(gl.uint32),)
    waiting = _iris_subgroup_any(valid.to(gl.int32))
    while waiting:
        # Issue every peer's read before checking any; .cv bypasses stale
        # cache lines. Arrived packs stay unchanged until this rank clears
        # them, so all valid lanes simply reread them. The compiler regression
        # tests check that these reads remain in the polling cycle.
        loaded = ()
        for delta in gl.static_range(1, WORLD_SIZE):
            peer = (RANK + delta) % WORLD_SIZE
            loaded += (
                gl.amd.cdna4.buffer_load(
                    words_ptr + peer * (SLOT_ELEMENTS // 2),
                    word_offsets,
                    mask=word_valid[:, None],
                    cache=".cv",
                ),
            )
        # Payloads never hold negative zero, so a word is still empty exactly
        # when its high half is 0x8000.
        missing = gl.full(
            word_valid.shape,
            False,
            gl.int1,
            gl.SliceLayout(1, word_offsets.type.layout),
        )
        for delta in gl.static_range(0, WORLD_SIZE - 1):
            empty = (loaded[delta] >> 16) == 0x8000
            missing |= word_valid & (gl.max(empty.to(gl.int32), 1) != 0)
        waiting = _iris_subgroup_any(missing.to(gl.int32))
    values = ()
    for delta in gl.static_range(0, WORLD_SIZE - 1):
        low = loaded[delta].to(gl.uint16).to(gl.bfloat16, bitcast=True)
        high = (loaded[delta] >> 16).to(gl.uint16).to(gl.bfloat16, bitcast=True)
        values += (
            gl.convert_layout(
                gl.reshape(gl.join(low, high), [ROWS, 2 * WORDS]), LAYOUT
            ),
        )
    return values


@gluon.jit
def _iris_wait_lamport_peers(
    region,
    generation,
    offsets,
    valid,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    """Wait for peer BF16 tiles to replace the empty mailbox markers."""
    return _iris_wait_lamport_slots(
        region + generation * WORLD_SIZE * MAX_ELEMENTS,
        offsets,
        valid,
        RANK,
        WORLD_SIZE,
        MAX_ELEMENTS,
        LAYOUT,
    )


@gluon.jit
def iris_lamport_allreduce(
    input_sym_ptr,
    region_sym_ptr,
    output_ptr,
    epochs,
    region_0,
    region_1,
    region_2,
    region_3,
    region_4,
    region_5,
    region_6,
    region_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    TOTAL_ELEMENTS: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """All-reduce BF16 tiles across eight ranks using data as the ready signal.

    Push input_sym_ptr into each peer's rotating mailbox and wait until the
    local mailbox contains every peer's contribution. Sum in FP32 rank order
    and write BF16 output_ptr. Clear consumed mailboxes and advance the tile's
    counter; negative zero is replaced with zero to preserve the empty marker.
    """
    gl.static_assert(WORLD_SIZE == 8)
    gl.static_assert(TOTAL_ELEMENTS % 512 == 0)
    gl.static_assert(TOTAL_ELEMENTS <= MAX_ELEMENTS)
    gl.static_assert(NUM_STAGES >= 3)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [1, 1], [0, 1])
    pack = gl.program_id(0) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    element = gl.arange(0, 8, layout=gl.SliceLayout(0, layout))
    offsets = pack[:, None] * 8 + element[None, :]
    mask = offsets < TOTAL_ELEMENTS
    generation = (
        gl.load(epochs + gl.program_id(0)).to(gl.uint32).to(gl.uint64) % NUM_STAGES
    )
    stride: gl.constexpr = WORLD_SIZE * MAX_ELEMENTS
    local = gl.amd.cdna4.buffer_load(input_sym_ptr, offsets, mask=mask, other=0.0)
    local = _iris_sanitize_lamport_bf16(local)
    regions = (
        region_0,
        region_1,
        region_2,
        region_3,
        region_4,
        region_5,
        region_6,
        region_7,
    )
    published = ()
    for _ in gl.static_range(1, WORLD_SIZE):
        published += (local,)
    _iris_lamport_push(
        published,
        regions,
        generation * stride + RANK * MAX_ELEMENTS,
        offsets,
        mask,
        RANK,
        WORLD_SIZE,
        PEER_SIGNAL_STORE_CACHE,
    )

    peers = _iris_wait_lamport_peers(
        region_sym_ptr,
        generation,
        offsets,
        pack * 8 < TOTAL_ELEMENTS,
        RANK,
        WORLD_SIZE,
        MAX_ELEMENTS,
        layout,
    )
    # All ranks use the same FP32 addition order, then round once to BF16.
    for peer in gl.static_range(0, WORLD_SIZE):
        if peer == RANK:
            term = local
        else:
            term = peers[(peer - RANK + WORLD_SIZE) % WORLD_SIZE - 1]
        if peer == 0:
            accumulator = term.to(gl.float32)
        else:
            accumulator += term.to(gl.float32)
    gl.amd.cdna4.buffer_store(
        accumulator.to(gl.bfloat16), output_ptr, offsets, mask=mask
    )

    # Clear only this tile's consumed generation. Skipped tiles keep their
    # own epochs, so mixed row counts need no global counter or tail clearing.
    sentinel = (
        gl.where(offsets % 2 == 0, 0, 0x8000)
        .to(gl.uint16)
        .to(gl.bfloat16, bitcast=True)
    )
    for delta in gl.static_range(1, WORLD_SIZE):
        peer = (RANK + delta) % WORLD_SIZE
        destination = region_sym_ptr + generation * stride + peer * MAX_ELEMENTS
        gl.amd.cdna4.buffer_store(
            sentinel, destination, offsets, mask=mask, cache=".wt"
        )
    gl.barrier()
    gl.store(epochs + gl.program_id(0), ((generation + 1) % NUM_STAGES).to(gl.int32))


@gluon.jit
def _iris_heap_base(
    rank: gl.constexpr,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
):
    """Return the supplied symmetric heap base for the requested group rank."""
    if rank == 0:
        return heap_base_0
    if rank == 1:
        return heap_base_1
    if rank == 2:
        return heap_base_2
    if rank == 3:
        return heap_base_3
    if rank == 4:
        return heap_base_4
    if rank == 5:
        return heap_base_5
    if rank == 6:
        return heap_base_6
    return heap_base_7


@gluon.jit
def _iris_sync_rank_token(
    flags,
    row,
    token,
    local_heap,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
):
    """Publish a counter to peers and wait for their matching counters."""
    layout: gl.constexpr = gl.BlockedLayout([1], [SUBGROUP_SIZE], [NUM_WARPS], [0])
    peers = gl.arange(0, WORLD_SIZE, layout=layout)
    peer_mask = peers != RANK
    peer_heaps = gl.where(peers == 0, heap_base_0, heap_base_7)
    peer_heaps = gl.where(peers == 1, heap_base_1, peer_heaps)
    peer_heaps = gl.where(peers == 2, heap_base_2, peer_heaps)
    peer_heaps = gl.where(peers == 3, heap_base_3, peer_heaps)
    peer_heaps = gl.where(peers == 4, heap_base_4, peer_heaps)
    peer_heaps = gl.where(peers == 5, heap_base_5, peer_heaps)
    peer_heaps = gl.where(peers == 6, heap_base_6, peer_heaps)
    flags_heap_offset = tl.cast(flags, gl.uint64) - local_heap
    peer_flags = tl.cast(
        peer_heaps + flags_heap_offset,
        gl.pointer_type(gl.int32),
    )
    gl.atomic_xchg(
        peer_flags + row * WORLD_SIZE + RANK,
        token,
        mask=peer_mask,
        sem="release",
        scope="sys",
    )
    local_flags = flags + row * WORLD_SIZE + peers
    seen = gl.load(
        local_flags,
        mask=peer_mask,
        other=token,
        cache_modifier=".cv",
        volatile=True,
    )
    while gl.max(gl.where(peer_mask & (seen != token), 1, 0), axis=0) != 0:
        seen = gl.load(
            local_flags,
            mask=peer_mask,
            other=token,
            cache_modifier=".cv",
            volatile=True,
        )
    gl.atomic_add(
        local_flags,
        0,
        mask=peer_mask,
        sem="acquire",
        scope="sys",
    )
    _iris_drain_subgroup_vmem()
    # The acquire is subgroup-local. Keep every subgroup at the protocol
    # boundary until all of them have observed the peer publications; the
    # caller consumes the peer inbox immediately after this helper returns.
    gl.barrier()


@gluon.jit
def _iris_sync_rank_epoch(
    ready_flags,
    block_id,
    epoch,
    local_heap,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    SUBGROUP_SIZE: gl.constexpr,
    PUBLISH: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Wait until each peer's workgroup counter reaches the requested epoch.

    PUBLISH sends this rank's epoch to peer flags and polls local flags;
    otherwise read each peer's counter directly. Counter comparisons allow
    wraparound and peers that have already advanced to the next epoch.
    """
    ready_layout: gl.constexpr = gl.BlockedLayout(
        [1], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    peer_ids = gl.arange(0, WORLD_SIZE, layout=ready_layout)
    peer_heaps = gl.where(peer_ids == 0, heap_base_0, heap_base_7)
    peer_heaps = gl.where(peer_ids == 1, heap_base_1, peer_heaps)
    peer_heaps = gl.where(peer_ids == 2, heap_base_2, peer_heaps)
    peer_heaps = gl.where(peer_ids == 3, heap_base_3, peer_heaps)
    peer_heaps = gl.where(peer_ids == 4, heap_base_4, peer_heaps)
    peer_heaps = gl.where(peer_ids == 5, heap_base_5, peer_heaps)
    peer_heaps = gl.where(peer_ids == 6, heap_base_6, peer_heaps)
    flags_heap_offset = tl.cast(ready_flags, gl.uint64) - local_heap
    peer_mask = peer_ids != RANK
    if PUBLISH:
        remote_flags = tl.cast(
            peer_heaps + flags_heap_offset,
            gl.pointer_type(gl.int32),
        )
        remote_flags += block_id * WORLD_SIZE + RANK
        gl.store(
            remote_flags,
            epoch,
            mask=peer_mask,
            cache_modifier=PEER_SIGNAL_STORE_CACHE,
        )
        wait_flags = ready_flags + block_id * WORLD_SIZE + peer_ids
    else:
        wait_flags = tl.cast(
            peer_heaps + flags_heap_offset,
            gl.pointer_type(gl.int32),
        )
        wait_flags += block_id * WORLD_SIZE + peer_ids
    seen = gl.load(
        wait_flags,
        mask=peer_mask,
        other=epoch,
        cache_modifier=".cv",
        volatile=True,
    )
    # Compare modulo 32 bits, including when a peer has passed the wrap before
    # this rank. A zero or negative epoch still requires observing every peer.
    while gl.min((seen - epoch).to(gl.int32), axis=0) < 0:
        seen = gl.load(
            wait_flags,
            mask=peer_mask,
            other=epoch,
            cache_modifier=".cv",
            volatile=True,
        )


@gluon.jit
def _unpack_16bitx4(packed, dtype: gl.constexpr):
    """Unpack four 16-bit values from one 64-bit word into FP32 values."""
    value_0 = (packed & 0xFFFF).to(gl.uint16).to(dtype, bitcast=True).to(gl.float32)
    value_1 = (
        ((packed >> 16) & 0xFFFF).to(gl.uint16).to(dtype, bitcast=True).to(gl.float32)
    )
    value_2 = (
        ((packed >> 32) & 0xFFFF).to(gl.uint16).to(dtype, bitcast=True).to(gl.float32)
    )
    value_3 = (
        ((packed >> 48) & 0xFFFF).to(gl.uint16).to(dtype, bitcast=True).to(gl.float32)
    )
    return value_0, value_1, value_2, value_3


@gluon.jit
def _pack_16bitx4(value_0, value_1, value_2, value_3, dtype: gl.constexpr):
    """Round four values to dtype and pack them into one 64-bit word."""
    bits_0 = value_0.to(dtype).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits_1 = value_1.to(dtype).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits_2 = value_2.to(dtype).to(gl.uint16, bitcast=True).to(gl.uint64)
    bits_3 = value_3.to(dtype).to(gl.uint16, bitcast=True).to(gl.uint64)
    return bits_0 | (bits_1 << 16) | (bits_2 << 32) | (bits_3 << 48)


@gluon.jit
def _unpack_word(packed, dtype: gl.constexpr, elements_per_word: gl.constexpr):
    """Unpack a 64-bit word into values for FP32 accumulation."""
    # Explicit branches keep Gluon from type-checking the inactive bitcast.
    if elements_per_word == 4:
        return _unpack_16bitx4(packed, dtype)
    else:
        value_0 = (packed & 0xFFFFFFFF).to(gl.uint32).to(dtype, bitcast=True)
        value_1 = ((packed >> 32) & 0xFFFFFFFF).to(gl.uint32).to(dtype, bitcast=True)
        return value_0, value_1, value_0, value_1


@gluon.jit
def _pack_word(
    value_0,
    value_1,
    value_2,
    value_3,
    dtype: gl.constexpr,
    elements_per_word: gl.constexpr,
):
    """Round accumulation values and pack them into the requested 64-bit layout."""
    if elements_per_word == 4:
        return _pack_16bitx4(value_0, value_1, value_2, value_3, dtype)
    else:
        bits_0 = value_0.to(dtype).to(gl.uint32, bitcast=True).to(gl.uint64)
        bits_1 = value_1.to(dtype).to(gl.uint32, bitcast=True).to(gl.uint64)
        return bits_0 | (bits_1 << 32)


@gluon.jit
def _mhc_post_words(
    x_0,
    x_1,
    x_2,
    x_3,
    word,
    mask,
    residual,
    post,
    comb,
    out,
    HIDDEN_WORDS: gl.constexpr,
):
    """Store the hc=4 mHC post of reduced BF16 words into the new residual.

    ``word`` indexes four consecutive hidden values of one token in the
    reduced [tokens, hidden] output. Residual and output are packed
    [tokens, 4, hidden] streams; ``post`` is FP32 [tokens, 4] and ``comb``
    FP32 [tokens, 4 input, 4 output]. The arithmetic follows
    gluon_mhc_post_gfx950: post * x first, then input streams 0..3.
    """
    token = word // HIDDEN_WORDS
    base = token * (4 * HIDDEN_WORDS) + word % HIDDEN_WORDS
    words = tl.cast(residual, gl.pointer_type(gl.uint64))
    r0 = _unpack_16bitx4(gl.load(words + base, mask=mask, other=0), gl.bfloat16)
    r1 = _unpack_16bitx4(
        gl.load(words + base + HIDDEN_WORDS, mask=mask, other=0), gl.bfloat16
    )
    r2 = _unpack_16bitx4(
        gl.load(words + base + 2 * HIDDEN_WORDS, mask=mask, other=0), gl.bfloat16
    )
    r3 = _unpack_16bitx4(
        gl.load(words + base + 3 * HIDDEN_WORDS, mask=mask, other=0), gl.bfloat16
    )
    x = (x_0, x_1, x_2, x_3)
    for j in gl.static_range(4):
        p = gl.load(post + token * 4 + j, mask=mask, other=0.0)
        c0 = gl.load(comb + token * 16 + j, mask=mask, other=0.0)
        c1 = gl.load(comb + token * 16 + 4 + j, mask=mask, other=0.0)
        c2 = gl.load(comb + token * 16 + 8 + j, mask=mask, other=0.0)
        c3 = gl.load(comb + token * 16 + 12 + j, mask=mask, other=0.0)
        values = ()
        for k in gl.static_range(4):
            value = p * x[k].to(gl.float32)
            value += c0 * r0[k].to(gl.float32)
            value += c1 * r1[k].to(gl.float32)
            value += c2 * r2[k].to(gl.float32)
            value += c3 * r3[k].to(gl.float32)
            values = values + (value,)
        gl.store(
            tl.cast(out, gl.pointer_type(gl.uint64)) + base + j * HIDDEN_WORDS,
            _pack_16bitx4(values[0], values[1], values[2], values[3], gl.bfloat16),
            mask=mask,
        )


@gluon.jit
def iris_pull_oneshot(
    input_sym_ptr,
    output_ptr,
    ready_flags,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    # The reduced size and the tile/program counts derived from it follow the
    # batch; runtime so every batch shape shares one binary.
    TOTAL_NUMEL,
    BLOCK_SIZE: gl.constexpr,
    NUM_PROGRAMS,
    NUM_TILES,
    NUM_WARPS: gl.constexpr,
    WORDS_PER_LANE: gl.constexpr,
    PUBLISH_READY: gl.constexpr,
    ELEMENT_DTYPE: gl.constexpr,
    ELEMENTS_PER_WORD: gl.constexpr,
    PEER_PAYLOAD_LOAD_CACHE: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
    residual=None,
    post=None,
    comb=None,
    HIDDEN_WORDS: gl.constexpr = 0,
    HC_POST: gl.constexpr = False,
):
    """Pull every rank's contributions and write the full reduced output locally.

    With HC_POST, output_ptr instead receives the BF16 hc=4 mHC post of the
    reduced rows (see _mhc_post_words).

    input_sym_ptr holds consecutive producer outputs; output_ptr receives their
    FP32 sum rounded to ELEMENT_DTYPE. ready_flags coordinate input publication
    and completion. All peer reads finish before the next producer can reuse
    the symmetric input; PUBLISH_READY selects how those counters are exchanged.
    """
    SUBGROUP_SIZE: gl.constexpr = hip_utils.num_threads() // NUM_WARPS
    block_id = gl.program_id(0)
    local_heap = _iris_heap_base(
        RANK,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    epoch_ptr = ready_flags + block_id * WORLD_SIZE + RANK
    epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
        epoch,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        SUBGROUP_SIZE,
        PUBLISH=PUBLISH_READY,
        PEER_SIGNAL_STORE_CACHE=PEER_SIGNAL_STORE_CACHE,
    )

    input_heap_offset = tl.cast(input_sym_ptr, gl.uint64) - local_heap
    layout: gl.constexpr = gl.BlockedLayout(
        [WORDS_PER_LANE], [SUBGROUP_SIZE], [NUM_WARPS], [0]
    )
    lane = gl.arange(0, BLOCK_SIZE // ELEMENTS_PER_WORD, layout=layout)
    total_packed = TOTAL_NUMEL // ELEMENTS_PER_WORD
    tile_id = block_id
    while tile_id < NUM_TILES:
        packed_offset = tile_id * (BLOCK_SIZE // ELEMENTS_PER_WORD) + lane
        mask = packed_offset < total_packed
        local_packed = gl.amd.cdna4.buffer_load(
            tl.cast(input_sym_ptr, gl.pointer_type(gl.uint64)),
            packed_offset.to(gl.int32),
            mask=mask,
            other=0,
        )
        acc_0, acc_1, acc_2, acc_3 = _unpack_word(
            local_packed, ELEMENT_DTYPE, ELEMENTS_PER_WORD
        )
        for peer in gl.static_range(0, WORLD_SIZE):
            if peer != RANK:
                peer_heap = _iris_heap_base(
                    peer,
                    heap_base_0,
                    heap_base_1,
                    heap_base_2,
                    heap_base_3,
                    heap_base_4,
                    heap_base_5,
                    heap_base_6,
                    heap_base_7,
                )
                peer_input = tl.cast(
                    peer_heap + input_heap_offset, gl.pointer_type(gl.uint64)
                )
                peer_packed = gl.amd.cdna4.buffer_load(
                    peer_input,
                    packed_offset.to(gl.int32),
                    mask=mask,
                    other=0,
                    cache=PEER_PAYLOAD_LOAD_CACHE,
                )
                peer_0, peer_1, peer_2, peer_3 = _unpack_word(
                    peer_packed, ELEMENT_DTYPE, ELEMENTS_PER_WORD
                )
                acc_0 += peer_0
                acc_1 += peer_1
                acc_2 += peer_2
                acc_3 += peer_3

        packed_output = _pack_word(
            acc_0,
            acc_1,
            acc_2,
            acc_3,
            ELEMENT_DTYPE,
            ELEMENTS_PER_WORD,
        )
        if HC_POST:
            x_0, x_1, x_2, x_3 = _unpack_16bitx4(packed_output, gl.bfloat16)
            _mhc_post_words(
                x_0,
                x_1,
                x_2,
                x_3,
                packed_offset,
                mask,
                residual,
                post,
                comb,
                output_ptr,
                HIDDEN_WORDS,
            )
        else:
            gl.amd.cdna4.buffer_store(
                packed_output,
                tl.cast(output_ptr, gl.pointer_type(gl.uint64)),
                packed_offset.to(gl.int32),
                mask=mask,
            )
        tile_id += NUM_PROGRAMS

    # Do not return while a peer program can still be reading this rank's
    # input. The next producer reuses the same symmetric buffer.
    completion_epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
        completion_epoch,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        SUBGROUP_SIZE,
        PUBLISH=PUBLISH_READY,
        PEER_SIGNAL_STORE_CACHE=PEER_SIGNAL_STORE_CACHE,
    )


@gluon.jit
def iris_pull_twoshot(
    input_sym_ptr,
    scratch_sym_ptr,
    output_ptr,
    ready_flags,
    heap_base_0,
    heap_base_1,
    heap_base_2,
    heap_base_3,
    heap_base_4,
    heap_base_5,
    heap_base_6,
    heap_base_7,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    # The partition size and the tile/program counts derived from it follow
    # the batch; runtime so every batch shape shares one binary.
    PARTITION_WORDS,
    BLOCK_WORDS: gl.constexpr,
    NUM_PROGRAMS,
    NUM_TILES,
    NUM_WARPS: gl.constexpr,
    WORDS_PER_LANE: gl.constexpr,
    ELEMENT_DTYPE: gl.constexpr,
    ELEMENTS_PER_WORD: gl.constexpr,
    EXIT_BARRIER: gl.constexpr,
    PEER_PAYLOAD_LOAD_CACHE: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
    residual=None,
    post=None,
    comb=None,
    HIDDEN_WORDS: gl.constexpr = 0,
    HC_POST: gl.constexpr = False,
):
    """Reduce-scatter producer outputs, then all-gather the rank partitions.

    With HC_POST, output_ptr instead receives the BF16 hc=4 mHC post of the
    gathered rows (see _mhc_post_words).

    Sum this rank's consecutive input partition in FP32 and store it in
    scratch_sym_ptr at the input dtype. Pull all peers' reduced partitions into
    output_ptr. ready_flags separate input readiness from partition readiness;
    EXIT_BARRIER adds a final peer wait for callers that stage inputs before
    launch. Both communication phases run in this one kernel.
    """
    SUBGROUP_SIZE: gl.constexpr = hip_utils.num_threads() // NUM_WARPS
    block_id = gl.program_id(0)
    local_heap = _iris_heap_base(
        RANK,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
    )
    epoch_ptr = ready_flags + block_id * WORLD_SIZE + RANK
    epoch = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
        epoch,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        SUBGROUP_SIZE,
        PUBLISH=True,
        PEER_SIGNAL_STORE_CACHE=PEER_SIGNAL_STORE_CACHE,
    )

    input_heap_offset = tl.cast(input_sym_ptr, gl.uint64) - local_heap
    scratch_heap_offset = tl.cast(scratch_sym_ptr, gl.uint64) - local_heap
    load_layout: gl.constexpr = gl.BlockedLayout(
        [1, WORDS_PER_LANE],
        [1, SUBGROUP_SIZE],
        [WORLD_SIZE, NUM_WARPS // WORLD_SIZE],
        [1, 0],
    )
    reduce_layout: gl.constexpr = gl.BlockedLayout(
        [WORLD_SIZE, 1], [1, SUBGROUP_SIZE], [1, NUM_WARPS], [0, 1]
    )
    peer_layout: gl.constexpr = gl.SliceLayout(1, load_layout)
    word_layout: gl.constexpr = gl.SliceLayout(0, load_layout)
    reduce_word_layout: gl.constexpr = gl.SliceLayout(0, reduce_layout)
    peer_ids = gl.arange(0, WORLD_SIZE, layout=peer_layout)
    words = gl.arange(0, BLOCK_WORDS, layout=word_layout)
    reduce_words = gl.arange(0, BLOCK_WORDS, layout=reduce_word_layout)
    peer_heaps = gl.where(peer_ids == 0, heap_base_0, heap_base_7)
    peer_heaps = gl.where(peer_ids == 1, heap_base_1, peer_heaps)
    peer_heaps = gl.where(peer_ids == 2, heap_base_2, peer_heaps)
    peer_heaps = gl.where(peer_ids == 3, heap_base_3, peer_heaps)
    peer_heaps = gl.where(peer_ids == 4, heap_base_4, peer_heaps)
    peer_heaps = gl.where(peer_ids == 5, heap_base_5, peer_heaps)
    peer_heaps = gl.where(peer_ids == 6, heap_base_6, peer_heaps)
    peer_inputs = tl.cast(
        gl.expand_dims(peer_heaps, 1) + input_heap_offset,
        gl.pointer_type(gl.uint64),
    )
    peer_scratch = tl.cast(
        gl.expand_dims(peer_heaps, 1) + scratch_heap_offset,
        gl.pointer_type(gl.uint64),
    )
    shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[32, 4]],
        [WORLD_SIZE, BLOCK_WORDS],
        [1, 0],
    )
    peer_values = gl.allocate_shared_memory(
        gl.uint64,
        [WORLD_SIZE, BLOCK_WORDS],
        shared_layout,
    )
    rank_start = RANK * PARTITION_WORDS

    # Reduce only this rank's partition of the full input into symmetric scratch.
    tile_id = block_id
    while tile_id < NUM_TILES:
        partition_offset = tile_id * BLOCK_WORDS + words
        input_offset = rank_start + partition_offset
        mask = partition_offset < PARTITION_WORDS
        values = gl.load(
            peer_inputs + gl.expand_dims(input_offset.to(gl.int32), 0),
            mask=gl.expand_dims(mask, 0),
            other=0,
            cache_modifier=PEER_PAYLOAD_LOAD_CACHE,
        )
        peer_values.store(values)

        packed = peer_values.load(reduce_layout)
        value_0, value_1, value_2, value_3 = _unpack_word(
            packed, ELEMENT_DTYPE, ELEMENTS_PER_WORD
        )
        reduced = _pack_word(
            gl.sum(value_0, axis=0),
            gl.sum(value_1, axis=0),
            gl.sum(value_2, axis=0),
            gl.sum(value_3, axis=0),
            ELEMENT_DTYPE,
            ELEMENTS_PER_WORD,
        )
        gl.amd.cdna4.buffer_store(
            reduced,
            tl.cast(scratch_sym_ptr, gl.pointer_type(gl.uint64)),
            (tile_id * BLOCK_WORDS + reduce_words).to(gl.int32),
            mask=tile_id * BLOCK_WORDS + reduce_words < PARTITION_WORDS,
            cache=".wt",
        )
        tile_id += NUM_PROGRAMS

    partitions_ready = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
    _iris_sync_rank_epoch(
        ready_flags,
        block_id,
        partitions_ready,
        local_heap,
        heap_base_0,
        heap_base_1,
        heap_base_2,
        heap_base_3,
        heap_base_4,
        heap_base_5,
        heap_base_6,
        heap_base_7,
        RANK,
        WORLD_SIZE,
        NUM_WARPS,
        SUBGROUP_SIZE,
        PUBLISH=True,
        PEER_SIGNAL_STORE_CACHE=PEER_SIGNAL_STORE_CACHE,
    )

    # Gather one reduced partition from every rank into the local output.
    tile_id = block_id
    while tile_id < NUM_TILES:
        partition_offset = tile_id * BLOCK_WORDS + words
        mask = partition_offset < PARTITION_WORDS
        values = gl.load(
            peer_scratch + gl.expand_dims(partition_offset.to(gl.int32), 0),
            mask=gl.expand_dims(mask, 0),
            other=0,
            cache_modifier=PEER_PAYLOAD_LOAD_CACHE,
        )
        output_offset = gl.expand_dims(peer_ids * PARTITION_WORDS, 1) + gl.expand_dims(
            partition_offset, 0
        )
        if HC_POST:
            x_0, x_1, x_2, x_3 = _unpack_16bitx4(values, gl.bfloat16)
            _mhc_post_words(
                x_0,
                x_1,
                x_2,
                x_3,
                output_offset,
                gl.expand_dims(mask, 0) & (output_offset >= 0),
                residual,
                post,
                comb,
                output_ptr,
                HIDDEN_WORDS,
            )
        else:
            gl.store(
                tl.cast(output_ptr, gl.pointer_type(gl.uint64)) + output_offset,
                values,
                mask=gl.expand_dims(mask, 0),
            )
        tile_id += NUM_PROGRAMS

    if EXIT_BARRIER:
        # Callers that stage into the symmetric input before launching cannot
        # rotate buffers safely: under graph capture the staging copy records a
        # fixed address and replays it, so a rank one invocation ahead would
        # overwrite an input its slower peers are still reducing. Holding the
        # kernel until every peer has finished reading makes a single staging
        # buffer correct by construction, at the price of one more rendezvous.
        # Producer-direct callers own their input and pass False.
        reads_done = gl.atomic_add(epoch_ptr, 1, sem="release", scope="sys") + 1
        _iris_sync_rank_epoch(
            ready_flags,
            block_id,
            reads_done,
            local_heap,
            heap_base_0,
            heap_base_1,
            heap_base_2,
            heap_base_3,
            heap_base_4,
            heap_base_5,
            heap_base_6,
            heap_base_7,
            RANK,
            WORLD_SIZE,
            NUM_WARPS,
            SUBGROUP_SIZE,
            PUBLISH=True,
            PEER_SIGNAL_STORE_CACHE=PEER_SIGNAL_STORE_CACHE,
        )


@triton.jit
def iris_pull_oneshot_rmsnorm(
    input_sym_ptr,  # base of symmetric (M, HIDDEN_SIZE) input buffer
    residual_ptr,  # local (M, HIDDEN_SIZE)
    weight_ptr,  # local (HIDDEN_SIZE,)
    norm_out_ptr,  # local (M, HIDDEN_SIZE)
    residual_out_ptr,  # local (M, HIDDEN_SIZE)
    M,
    heap_bases,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    """All-reduce each row, add its residual, and apply weighted RMSNorm.

    Read symmetric input_sym_ptr across the ranks selected by rank_start and
    rank_stride. Compute the reduction, residual addition and norm in FP32;
    write residual_out_ptr and norm_out_ptr at their supplied dtypes. The caller
    must synchronize input publication and completion of all peer reads.
    """
    row = tl.program_id(0)
    if row >= M:
        return

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HIDDEN_SIZE
    row_offsets = row * HIDDEN_SIZE + offsets
    in_row_ptr = input_sym_ptr + row_offsets

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for i in tl.static_range(0, world_size):
        remote_rank = rank_start + i * rank_stride
        acc += iris.load(
            in_row_ptr,
            iris_rank,
            remote_rank,
            heap_bases,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

    residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    residual_out = acc + residual

    res_out_dtype = residual_out_ptr.type.element_ty
    tl.store(
        residual_out_ptr + row_offsets,
        residual_out.to(res_out_dtype),
        mask=mask,
    )

    variance = tl.sum(residual_out * residual_out, axis=0) / HIDDEN_SIZE
    scale = tl.rsqrt(variance + EPS)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    norm = residual_out * scale * weight

    norm_dtype = norm_out_ptr.type.element_ty
    tl.store(
        norm_out_ptr + row_offsets,
        norm.to(norm_dtype),
        mask=mask,
    )


@triton.jit
def iris_pull_oneshot_rmsnorm_persistent(
    input_sym_ptr,
    residual_ptr,
    weight_ptr,
    norm_out_ptr,
    residual_out_ptr,
    M,
    heap_bases,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EPS: tl.constexpr,
):
    """Process several all-reduce/residual/RMSNorm rows per workgroup.

    Use the same inputs, FP32 arithmetic and output dtypes as
    iris_pull_oneshot_rmsnorm, but distribute M rows over a fixed grid and reuse
    each workgroup's norm weights. The caller synchronizes input publication
    and completion of all peer reads.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HIDDEN_SIZE
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    res_out_dtype = residual_out_ptr.type.element_ty
    norm_dtype = norm_out_ptr.type.element_ty

    for row in range(pid, M, num_programs):
        row_offsets = row * HIDDEN_SIZE + offsets
        in_row_ptr = input_sym_ptr + row_offsets

        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for i in tl.static_range(0, world_size):
            remote_rank = rank_start + i * rank_stride
            acc += iris.load(
                in_row_ptr,
                iris_rank,
                remote_rank,
                heap_bases,
                mask=mask,
                other=0.0,
            ).to(tl.float32)

        residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        residual_out = acc + residual

        tl.store(
            residual_out_ptr + row_offsets,
            residual_out.to(res_out_dtype),
            mask=mask,
        )

        variance = tl.sum(residual_out * residual_out, axis=0) / HIDDEN_SIZE
        scale = tl.rsqrt(variance + EPS)
        norm = residual_out * scale * weight

        tl.store(
            norm_out_ptr + row_offsets,
            norm.to(norm_dtype),
            mask=mask,
        )


class ProducerAllReduce:
    """Prepared BF16 TP4/TP8 Lamport, push and pull variants over producer storage.

    ``run`` selects the protocol from the payload and can return the push
    state's borrowed output instead of ``output``. Fusions may share the Lamport
    one-shot mailbox and its per-512-element stage counters; the two-shot owns
    its own. A use covers whole tiles from the start of each slot and every rank
    makes the same sequence of uses. Within a use, each push to a tile reads that
    tile's stage first, and the receiver restores the markers of everything it
    consumed and advances each tile's stage once; the next use of a tile, a
    later launch, observes both. Pull scratch is shared with the generic
    producer reduction; all uses and consumers must be stream ordered.
    """

    LAMPORT_MAX_ELEMENTS = 43 * 4096
    # The one-shot sends every peer the whole payload; from this size the
    # two-shot was faster in 8-rank MI355X graph replay. TP4 keeps push.
    LAMPORT_TWO_SHOT_MIN_ELEMENTS = 48 * 1024
    LAMPORT_TWO_SHOT_MAX_ELEMENTS = 128 * 8192
    # iris_lamport_twoshot_bf16_allreduce reduces one 128-element tile of every
    # shard per program.
    LAMPORT_TWO_SHOT_TILE = 128
    PUSH_MAX_ELEMENTS = 128 * 8192

    @staticmethod
    def _lamport_two_shot_capacity(
        max_numel: int, *, world_size: int, enable_lamport: bool
    ) -> int:
        """Return the TP8 Lamport two-shot capacity in elements, or 0."""
        alignment = world_size * ProducerAllReduce.LAMPORT_TWO_SHOT_TILE
        capacity = min(
            max_numel // alignment * alignment,
            ProducerAllReduce.LAMPORT_TWO_SHOT_MAX_ELEMENTS,
        )
        return (
            capacity
            if enable_lamport
            and world_size == 8
            and capacity >= ProducerAllReduce.LAMPORT_TWO_SHOT_MIN_ELEMENTS
            else 0
        )

    @staticmethod
    def heap_bytes(max_numel: int, *, world_size: int, enable_lamport: bool) -> int:
        lamport = (
            min(max_numel // 512 * 512, ProducerAllReduce.LAMPORT_MAX_ELEMENTS)
            if enable_lamport
            else 0
        )
        two_shot = ProducerAllReduce._lamport_two_shot_capacity(
            max_numel, world_size=world_size, enable_lamport=enable_lamport
        )
        alignment = world_size * 8
        push = min(
            max_numel // alignment * alignment, ProducerAllReduce.PUSH_MAX_ELEMENTS
        )
        return (
            allocation_bytes((_PULL_MAX_PROGRAMS, world_size), torch.uint32)
            + 2 * allocation_bytes((push,), torch.bfloat16)
            + (
                allocation_bytes((3, world_size, lamport), torch.bfloat16)
                if lamport
                else 0
            )
            + (
                allocation_bytes(
                    (3, 2, world_size, two_shot // world_size), torch.bfloat16
                )
                if two_shot
                else 0
            )
        )

    def __init__(self, pull: ProducerPull, *, enable_lamport: bool):
        producer = pull.producer
        peers = producer.peers
        assert current_platform().is_cdna4 and peers.world_size in (4, 8)
        assert (
            producer.data.dtype == torch.bfloat16
            and producer.data.numel() >= peers.world_size * 8
        )
        self.pull = pull
        self.world_size = peers.world_size
        max_numel = producer.data.numel()
        self.lamport_max_numel = (
            min(max_numel // 512 * 512, self.LAMPORT_MAX_ELEMENTS)
            if enable_lamport
            else 0
        )
        alignment = self.world_size * 8
        self.push_max_numel = min(
            max_numel // alignment * alignment, self.PUSH_MAX_ELEMENTS
        )
        self.ready_flags = peers.ctx.zeros(
            (_PULL_MAX_PROGRAMS, self.world_size), dtype=torch.uint32
        )
        self.epochs = torch.zeros(
            _PULL_MAX_PROGRAMS, dtype=torch.uint32, device=peers.device
        )
        self.push_inbox = peers.ctx.empty(
            (self.world_size, self.push_max_numel // self.world_size),
            dtype=torch.bfloat16,
        )
        self.push_output = peers.ctx.empty((self.push_max_numel,), dtype=torch.bfloat16)
        self.lamport_region = (
            peers.ctx.empty(
                (3, self.world_size, self.lamport_max_numel), dtype=torch.bfloat16
            )
            if self.lamport_max_numel
            else None
        )
        self.lamport_epochs = (
            torch.zeros(
                self.lamport_max_numel // 512, dtype=torch.int32, device=peers.device
            )
            if self.lamport_max_numel
            else None
        )
        self.lamport_peer_addresses = None
        if self.lamport_region is not None:
            self.lamport_region.view(torch.int32).fill_(-2147483648)
            offset = self.lamport_region.data_ptr() - peers.addresses[peers.rank]
            self.lamport_peer_addresses = tuple(
                base + offset for base in peers.addresses
            )
        self.lamport_two_shot_max_numel = self._lamport_two_shot_capacity(
            max_numel, world_size=self.world_size, enable_lamport=enable_lamport
        )
        two_shot_shard = self.lamport_two_shot_max_numel // self.world_size
        # [stage][reduce-scatter, all-gather][source rank][shard] mailboxes and
        # one stage counter per two-shot program.
        self.lamport_two_shot_region = (
            peers.ctx.empty(
                (3, 2, self.world_size, two_shot_shard), dtype=torch.bfloat16
            )
            if two_shot_shard
            else None
        )
        self.lamport_two_shot_epochs = (
            torch.zeros(
                two_shot_shard // self.LAMPORT_TWO_SHOT_TILE,
                dtype=torch.int32,
                device=peers.device,
            )
            if two_shot_shard
            else None
        )
        self.lamport_two_shot_peer_addresses = None
        if self.lamport_two_shot_region is not None:
            self.lamport_two_shot_region.view(torch.int32).fill_(-2147483648)
            offset = (
                self.lamport_two_shot_region.data_ptr() - peers.addresses[peers.rank]
            )
            self.lamport_two_shot_peer_addresses = tuple(
                base + offset for base in peers.addresses
            )

    def run(
        self, input: torch.Tensor, output: torch.Tensor, numel: int
    ) -> torch.Tensor:
        """Select Lamport one-shot or two-shot, bounded push, or large pull."""
        assert 0 < numel <= self.pull.producer.data.numel()
        assert numel % (self.world_size * 8) == 0
        two_shot_tile = self.world_size * self.LAMPORT_TWO_SHOT_TILE
        if (
            self.LAMPORT_TWO_SHOT_MIN_ELEMENTS
            <= numel
            <= self.lamport_two_shot_max_numel
            and numel % two_shot_tile == 0
        ):
            iris_lamport_twoshot_bf16_allreduce[(numel // two_shot_tile,)](
                input,
                self.lamport_two_shot_region,
                output,
                self.lamport_two_shot_epochs,
                self.lamport_two_shot_peer_addresses[: self.world_size],
                RANK=self.pull.producer.peers.rank,
                WORLD_SIZE=self.world_size,
                total_elements=numel,
                MAX_SHARD_ELEMENTS=self.lamport_two_shot_max_numel // self.world_size,
                NUM_STAGES=3,
                PEER_SIGNAL_STORE_CACHE=iris_cache_policy().peer_signal_store,
                num_warps=1,
            )
            return output
        if numel <= self.lamport_max_numel and numel % 512 == 0:
            iris_lamport_bf16_allreduce[(numel // 512,)](
                input,
                self.lamport_region,
                output,
                self.lamport_epochs,
                self.lamport_peer_addresses[: self.world_size],
                RANK=self.pull.producer.peers.rank,
                WORLD_SIZE=self.world_size,
                total_elements=numel,
                MAX_ELEMENTS=self.lamport_max_numel,
                NUM_STAGES=3,
                PEER_SIGNAL_STORE_CACHE=iris_cache_policy().peer_signal_store,
                num_warps=1,
            )
            return output
        if numel <= self.push_max_numel:
            iris_push_twoshot_allreduce[(_PULL_MAX_PROGRAMS,)](
                input,
                self.push_inbox,
                self.push_output,
                self.ready_flags,
                self.epochs,
                self.pull.producer.peers.addresses[: self.world_size],
                numel,
                MAX_PARTITION=self.push_max_numel // self.world_size,
                RANK=self.pull.producer.peers.rank,
                BLOCK=1024,
                NUM_SUBGROUPS=4,
                num_warps=4,
            )
            return self.push_output
        if input.data_ptr() != self.pull.producer.data.data_ptr():
            self.pull.producer.data[:numel].copy_(input.view(-1))
        iris_pull_twoshot_allreduce[(_PULL_MAX_PROGRAMS,)](
            self.pull.producer.data,
            self.pull.scratch,
            output,
            self.ready_flags,
            self.epochs,
            self.pull.producer.peers.addresses[: self.world_size],
            numel,
            RANK=self.pull.producer.peers.rank,
            BLOCK=2048,
            NUM_SUBGROUPS=4,
            num_warps=4,
        )
        return output


@gluon.jit
def _iris_group_heap_base(rank, heaps):
    """Select the supplied base address for a group rank."""
    base = heaps[0]
    for peer in gl.static_range(1, len(heaps)):
        base = gl.where(rank == peer, heaps[peer], base)
    return base


@gluon.jit
def _iris_peer_pointers(pointer, heaps, RANK: gl.constexpr):
    """Return peer pointers at pointer's offset within each symmetric heap."""
    offset = pointer.to(gl.uint64) - heaps[RANK]
    peers = ()
    for peer in gl.static_range(len(heaps)):
        peers += (gl.multiple_of((heaps[peer] + offset).to(pointer.dtype), 16),)
    return peers


@gluon.jit
def _iris_rendezvous(
    flags,
    heaps,
    block,
    epoch,
    RANK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PUBLISH_RELEASE: gl.constexpr,
):
    """Publish a workgroup counter and wait for all group ranks to reach it.

    PUBLISH_RELEASE publishes prior producer writes with a release atomic.
    Otherwise the caller must first finish its write-through stores. Acquire
    peer flags before returning so subsequent reads see the published data.
    """
    WORLD_SIZE: gl.constexpr = len(heaps)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    peers = gl.arange(0, WORLD_SIZE, layout=layout)
    remote = peers != RANK
    bases = gl.full((WORLD_SIZE,), 0, gl.uint64, layout)
    for peer in gl.static_range(WORLD_SIZE):
        bases = gl.where(peers == peer, heaps[peer], bases)
    peer_flags = (bases + flags.to(gl.uint64) - heaps[RANK]).to(flags.dtype)
    if PUBLISH_RELEASE:
        gl.atomic_xchg(
            peer_flags + block * WORLD_SIZE + RANK,
            epoch,
            mask=remote,
            sem="release",
            scope="sys",
        )
    else:
        # Callers have already drained every subgroup's write-through stores.
        gl.store(
            peer_flags + block * WORLD_SIZE + RANK,
            epoch,
            mask=remote,
            cache_modifier=".wt",
        )
    local = flags + block * WORLD_SIZE + peers
    seen = gl.load(local, mask=remote, other=epoch, cache_modifier=".cv", volatile=True)
    while gl.sum((remote & ((seen - epoch).to(gl.int32) < 0)).to(gl.int32), 0) != 0:
        seen = gl.load(
            local, mask=remote, other=epoch, cache_modifier=".cv", volatile=True
        )
    # Acquire invalidates peer payload cache lines and joins the workgroup.
    gl.atomic_add(local, 0, mask=remote, sem="acquire", scope="sys")


@gluon.jit(do_not_specialize=["total_elements"])
def iris_lamport_bf16_allreduce(
    input_sym_ptr,
    region_sym_ptr,
    output_ptr,
    epochs,
    regions,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    total_elements,
    MAX_ELEMENTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """All-reduce BF16 tiles across the group using rotating sentinel mailboxes.

    Push input_sym_ptr to the peer addresses in regions, wait for incoming data
    in region_sym_ptr, and write a rank-ordered FP32 sum rounded to BF16 at
    output_ptr. Clear consumed mailboxes and advance epochs for each tile.
    Replace negative zero with zero so data cannot match the empty marker.
    """
    gl.static_assert(WORLD_SIZE == 4 or WORLD_SIZE == 8)
    gl.static_assert(MAX_ELEMENTS % 512 == 0)
    gl.static_assert(NUM_STAGES >= 3)
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [64, 1], [1, 1], [0, 1])
    pack = gl.program_id(0) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    element = gl.arange(0, 8, layout=gl.SliceLayout(0, layout))
    offsets = pack[:, None] * 8 + element[None, :]
    total_elements = gl.multiple_of(total_elements, 512)
    mask = (pack * 8 < total_elements)[:, None]
    generation = (
        gl.load(epochs + gl.program_id(0)).to(gl.uint32).to(gl.uint64) % NUM_STAGES
    )
    stride: gl.constexpr = WORLD_SIZE * MAX_ELEMENTS
    local = gl.amd.cdna4.buffer_load(input_sym_ptr, offsets, mask=mask, other=0.0)
    local = _iris_sanitize_lamport_bf16(local)
    published = ()
    for _ in gl.static_range(1, WORLD_SIZE):
        published += (local,)
    _iris_lamport_push(
        published,
        regions,
        generation * stride + RANK * MAX_ELEMENTS,
        offsets,
        mask,
        RANK,
        WORLD_SIZE,
        PEER_SIGNAL_STORE_CACHE,
    )

    peers = _iris_wait_lamport_peers(
        region_sym_ptr,
        generation,
        offsets,
        pack * 8 < total_elements,
        RANK,
        WORLD_SIZE,
        MAX_ELEMENTS,
        layout,
    )
    # All ranks use the same FP32 addition order, then round once to BF16.
    for peer in gl.static_range(0, WORLD_SIZE):
        if peer == RANK:
            term = local
        else:
            term = peers[(peer - RANK + WORLD_SIZE) % WORLD_SIZE - 1]
        if peer == 0:
            accumulator = term.to(gl.float32)
        else:
            accumulator += term.to(gl.float32)
    gl.amd.cdna4.buffer_store(
        accumulator.to(gl.bfloat16), output_ptr, offsets, mask=mask
    )

    # Clear only this tile's consumed generation. Skipped tiles keep their
    # own epochs, so mixed row counts need no global counter or tail clearing.
    sentinel = (
        gl.where(offsets % 2 == 0, 0, 0x8000)
        .to(gl.uint16)
        .to(gl.bfloat16, bitcast=True)
    )
    for delta in gl.static_range(1, WORLD_SIZE):
        peer = (RANK + delta) % WORLD_SIZE
        destination = region_sym_ptr + generation * stride + peer * MAX_ELEMENTS
        gl.amd.cdna4.buffer_store(
            sentinel, destination, offsets, mask=mask, cache=".wt"
        )
    gl.barrier()
    gl.store(epochs + gl.program_id(0), ((generation + 1) % NUM_STAGES).to(gl.int32))


@gluon.jit(do_not_specialize=["total_elements"])
def iris_lamport_twoshot_bf16_allreduce(
    input_ptr,
    region_sym_ptr,
    output_ptr,
    epochs,
    regions,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    total_elements,
    MAX_SHARD_ELEMENTS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Reduce-scatter then all-gather BF16 through sentinel mailboxes.

    Each rank owns one contiguous 1/WORLD_SIZE shard of input_ptr. Program t
    pushes tile t of every peer's shard to that peer, sums its own shard tile in
    FP32 rank order, rounds once to BF16, and pushes the result to every peer.
    regions are the peers' region_sym_ptr addresses. A region holds
    [stage][reduce-scatter, all-gather][source rank][MAX_SHARD_ELEMENTS]
    mailboxes, indexed by program, so epochs[t] belongs to program t for every
    payload size. Negative zero becomes zero in inputs and in reduced values,
    which every rank receives from the shard owner.
    """
    gl.static_assert(WORLD_SIZE == 8)
    # One subgroup of 4-byte packs per program spreads pushes over more CUs.
    VECTOR: gl.constexpr = 2
    TILE: gl.constexpr = 64 * VECTOR
    gl.static_assert(MAX_SHARD_ELEMENTS % TILE == 0)
    gl.static_assert(NUM_STAGES >= 3)
    layout: gl.constexpr = gl.BlockedLayout([1, VECTOR], [64, 1], [1, 1], [0, 1])
    pack = gl.program_id(0) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    element = gl.arange(0, VECTOR, layout=gl.SliceLayout(0, layout))
    offsets = pack[:, None] * VECTOR + element[None, :]
    total_elements = gl.multiple_of(total_elements, WORLD_SIZE * TILE)
    shard = gl.multiple_of(total_elements // WORLD_SIZE, TILE)
    stage = gl.load(epochs + gl.program_id(0)).to(gl.uint32).to(gl.uint64) % NUM_STAGES
    PHASE: gl.constexpr = WORLD_SIZE * MAX_SHARD_ELEMENTS
    slots = stage * (2 * PHASE)
    # chunks[delta] is this rank's tile of shard (RANK + delta) % WORLD_SIZE.
    chunks = ()
    for delta in gl.static_range(WORLD_SIZE):
        owner = (RANK + delta) % WORLD_SIZE
        chunks += (
            _iris_sanitize_lamport_bf16(
                gl.amd.cdna4.buffer_load(input_ptr + owner * shard, offsets)
            ),
        )
    _iris_lamport_push(
        chunks[1:],
        regions,
        slots + RANK * MAX_SHARD_ELEMENTS,
        offsets,
        None,
        RANK,
        WORLD_SIZE,
        PEER_SIGNAL_STORE_CACHE,
    )
    valid = gl.full([64], True, gl.int1, gl.SliceLayout(1, layout))
    scattered = _iris_wait_lamport_slots(
        region_sym_ptr + slots,
        offsets,
        valid,
        RANK,
        WORLD_SIZE,
        MAX_SHARD_ELEMENTS,
        layout,
    )
    # Match the one-shot result: FP32 sum in rank order, one BF16 rounding.
    for source in gl.static_range(0, WORLD_SIZE):
        if source == RANK:
            term = chunks[0]
        else:
            term = scattered[(source - RANK + WORLD_SIZE) % WORLD_SIZE - 1]
        if source == 0:
            accumulator = term.to(gl.float32)
        else:
            accumulator += term.to(gl.float32)
    reduced = _iris_sanitize_lamport_bf16(accumulator.to(gl.bfloat16))
    published = ()
    for _ in gl.static_range(1, WORLD_SIZE):
        published += (reduced,)
    _iris_lamport_push(
        published,
        regions,
        slots + PHASE + RANK * MAX_SHARD_ELEMENTS,
        offsets,
        None,
        RANK,
        WORLD_SIZE,
        PEER_SIGNAL_STORE_CACHE,
    )
    gl.amd.cdna4.buffer_store(reduced, output_ptr + RANK * shard, offsets)
    gathered = _iris_wait_lamport_slots(
        region_sym_ptr + slots + PHASE,
        offsets,
        valid,
        RANK,
        WORLD_SIZE,
        MAX_SHARD_ELEMENTS,
        layout,
    )
    for delta in gl.static_range(1, WORLD_SIZE):
        owner = (RANK + delta) % WORLD_SIZE
        gl.amd.cdna4.buffer_store(
            gathered[delta - 1], output_ptr + owner * shard, offsets
        )

    # Clear this program's consumed mailboxes before advancing its stage.
    sentinel = (
        gl.where(offsets % 2 == 0, 0, 0x8000)
        .to(gl.uint16)
        .to(gl.bfloat16, bitcast=True)
    )
    for phase in gl.static_range(2):
        for delta in gl.static_range(1, WORLD_SIZE):
            source = (RANK + delta) % WORLD_SIZE
            gl.amd.cdna4.buffer_store(
                sentinel,
                region_sym_ptr + slots + phase * PHASE + source * MAX_SHARD_ELEMENTS,
                offsets,
                cache=".wt",
            )
    # Order the sentinel restores before this program's stage advance.
    gl.barrier()
    gl.store(epochs + gl.program_id(0), ((stage + 1) % NUM_STAGES).to(gl.int32))


@gluon.jit(do_not_specialize=["numel"])
def iris_push_twoshot_allreduce(
    input_ptr,
    inbox_ptr,
    output_ptr,
    flags,
    epochs,
    heaps,
    numel,
    MAX_PARTITION: gl.constexpr,
    RANK: gl.constexpr,
    BLOCK: gl.constexpr,
    NUM_SUBGROUPS: gl.constexpr,
):
    """Push BF16 partials to partition owners, then distribute their sums.

    Split input_ptr's numel elements into equal consecutive partitions. Each rank
    receives its partition in inbox_ptr, sums all inputs in FP32, rounds
    to BF16, and pushes the result into every output_ptr. flags and epochs
    separate the two phases and ensure all output writes finish before return.
    """
    WORLD_SIZE: gl.constexpr = len(heaps)
    inboxes = _iris_peer_pointers(inbox_ptr, heaps, RANK)
    outputs = _iris_peer_pointers(output_ptr, heaps, RANK)
    block = gl.program_id(0)
    epoch = gl.load(epochs + block).to(gl.uint32) + 2
    part = gl.multiple_of(numel // WORLD_SIZE, 8)
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_SUBGROUPS], [0])
    offsets = gl.arange(0, BLOCK, layout=layout)
    for tile in range(block, gl.cdiv(part, BLOCK), gl.num_programs(0)):
        index = tile * BLOCK + offsets
        mask = index < part
        for step in gl.static_range(WORLD_SIZE):
            owner = (RANK + step) % WORLD_SIZE
            values = gl.amd.cdna4.buffer_load(input_ptr, owner * part + index, mask, 0)
            gl.amd.cdna4.buffer_store(
                values,
                inboxes[(RANK + step) % WORLD_SIZE],
                RANK * MAX_PARTITION + index,
                mask,
                cache=".wt",
            )
    _iris_drain_subgroup_vmem()
    gl.barrier()
    _iris_rendezvous(flags, heaps, block, epoch - 1, RANK, NUM_SUBGROUPS, False)
    for tile in range(block, gl.cdiv(part, BLOCK), gl.num_programs(0)):
        index = tile * BLOCK + offsets
        mask = index < part
        values = ()
        for peer in gl.static_range(WORLD_SIZE):
            values += (
                gl.amd.cdna4.buffer_load(
                    inbox_ptr, peer * MAX_PARTITION + index, mask, 0, cache=".cg"
                ).to(gl.float32),
            )
        result = (values[0] + values[1]) + (values[2] + values[3])
        if WORLD_SIZE == 8:
            result += (values[4] + values[5]) + (values[6] + values[7])
        result = result.to(gl.bfloat16)
        for step in gl.static_range(WORLD_SIZE):
            gl.amd.cdna4.buffer_store(
                result,
                outputs[(RANK + step) % WORLD_SIZE],
                RANK * part + index,
                mask,
                cache=".wt",
            )
    _iris_drain_subgroup_vmem()
    gl.barrier()
    _iris_rendezvous(flags, heaps, block, epoch, RANK, NUM_SUBGROUPS, False)
    gl.store(epochs + block, epoch)


@gluon.jit(do_not_specialize=["numel"])
def iris_pull_twoshot_allreduce(
    input_ptr,
    scratch_ptr,
    output_ptr,
    flags,
    epochs,
    heaps,
    numel,
    RANK: gl.constexpr,
    BLOCK: gl.constexpr,
    NUM_SUBGROUPS: gl.constexpr,
):
    """Reduce BF16 input partitions, then gather the full result locally.

    Each rank pulls its consecutive partition of input_ptr from all peers, sums
    in FP32, and stores BF16 values in scratch_ptr. After exchanging ready
    counters, pull each peer's scratch into output_ptr. The next invocation's
    entry wait protects scratch while peers finish reading the previous result.
    """
    WORLD_SIZE: gl.constexpr = len(heaps)
    inputs = _iris_peer_pointers(input_ptr, heaps, RANK)
    scratch = _iris_peer_pointers(scratch_ptr, heaps, RANK)
    block = gl.program_id(0)
    epoch = gl.load(epochs + block).to(gl.uint32) + 2
    _iris_rendezvous(flags, heaps, block, epoch - 1, RANK, NUM_SUBGROUPS, True)
    part = gl.multiple_of(numel // WORLD_SIZE, 8)
    layout: gl.constexpr = gl.BlockedLayout([8], [64], [NUM_SUBGROUPS], [0])
    offsets = gl.arange(0, BLOCK, layout=layout)
    for tile in range(block, gl.cdiv(part, BLOCK), gl.num_programs(0)):
        index = tile * BLOCK + offsets
        mask = index < part
        values = ()
        for peer in gl.static_range(WORLD_SIZE):
            values += (
                gl.amd.cdna4.buffer_load(
                    inputs[peer], RANK * part + index, mask, 0, cache=".cg"
                ).to(gl.float32),
            )
        result = (values[0] + values[1]) + (values[2] + values[3])
        if WORLD_SIZE == 8:
            result += (values[4] + values[5]) + (values[6] + values[7])
        result = result.to(gl.bfloat16)
        gl.amd.cdna4.buffer_store(result, scratch_ptr, index, mask, cache=".wt")
    _iris_drain_subgroup_vmem()
    gl.barrier()
    _iris_rendezvous(flags, heaps, block, epoch, RANK, NUM_SUBGROUPS, False)
    for tile in range(block, gl.cdiv(part, BLOCK), gl.num_programs(0)):
        index = tile * BLOCK + offsets
        mask = index < part
        for step in gl.static_range(WORLD_SIZE):
            peer = (RANK + step) % WORLD_SIZE
            values = gl.amd.cdna4.buffer_load(
                scratch[(RANK + step) % WORLD_SIZE], index, mask, 0, cache=".cg"
            )
            gl.amd.cdna4.buffer_store(values, output_ptr, peer * part + index, mask)
    # Phase two proves every peer consumed the old input. The next scratch
    # writer must enter an all-peer rendezvous after these gathers complete.
    gl.store(epochs + block, epoch)


@gluon.jit
def _iris_sync_peers(
    ready_flags,
    slot,
    epoch,
    heaps,
    rank: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Publish one slot's epoch and wait for matching peer epochs across the group."""
    WORLD_SIZE: gl.constexpr = len(heaps)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [num_warps], [0])
    peers = gl.arange(0, WORLD_SIZE, layout=layout)
    peer_heaps = _iris_group_heap_base(peers, heaps)
    flags_offset = tl.cast(ready_flags, gl.uint64) - heaps[rank]
    remote_flags = tl.cast(
        peer_heaps + flags_offset,
        gl.pointer_type(gl.int32),
    )
    remote_flags += slot * WORLD_SIZE + rank
    peer_mask = peers != rank
    gl.store(remote_flags, epoch, mask=peer_mask, cache_modifier=".wt")

    wait_flags = ready_flags + slot * WORLD_SIZE + peers
    seen = gl.load(
        wait_flags,
        mask=peer_mask,
        other=epoch,
        cache_modifier=".cv",
        volatile=True,
    )
    while (
        gl.sum(
            (
                peer_mask
                & ((seen.to(gl.uint32) - epoch.to(gl.uint32)).to(gl.int32) < 0)
            ).to(gl.int32),
            axis=0,
        )
        != 0
    ):
        seen = gl.load(
            wait_flags,
            mask=peer_mask,
            other=epoch,
            cache_modifier=".cv",
            volatile=True,
        )
    # Peer payload addresses are reused; .cg loads can still hit stale L2 lines.
    gl.atomic_add(ready_flags + slot * WORLD_SIZE + rank, 0, sem="acquire", scope="sys")
