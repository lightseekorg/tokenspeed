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

"""All-gather of paged cache rows with the row pack fused into the transfer.

One kernel per window on the state's own stream, which runs it in the CUDA
remote memory-synchronization domain: every rank reads its own rows from a
paged plane in ``fetch_slots`` order (16-byte loads that do not allocate in
L1) and stores them straight into row ``rank * send_rows + i`` of a receive
slot that every rank of the group holds, through the NVLink multicast address
of a symmetric workspace (one 16-byte ``multimem.st`` reaches all ranks) or,
without multicast, through one unicast store per peer. The rows a consumer
reads through ``gathered_index`` are byte-identical to an all-gather of the
packed rows; padding rows ``[owned, send_rows)`` are never written. The
workspace layout and the caller's contract are in the ``communication``
README.

The workspace's flags block holds per-peer int64 words, each array on its own
128-byte line: ``credit[p]`` (peer ``p`` started kernel ``credit[p]``, so it
has finished reading the slot's previous epoch), ``arrive[p]`` (all of peer
``p``'s rows of epoch ``arrive[p]`` are written), ``hdr[p]`` (peer ``p``'s
window identity), and a local CTA counter. Flags only ever travel through
unicast addresses.

Protocol of kernel ``e`` on rank ``r`` (credits and arrivals are monotonic and
must equal ``e``; a larger value means the ranks launched different numbers of
windows and fails before any payload store):

1. Entry, every CTA: lane ``t`` of warp 0 release-stores ``credit[r] = e`` into
   peer ``t`` and acquires its own ``credit[t] == e``; a CTA barrier hands the
   acquire to the storing threads.
2. Payload stores.
3. Exit: CTA barrier, then thread 0 counts the CTA in with an acq_rel atomic;
   the last CTA (a CTA barrier hands thread 0's acquire to the publishing
   lanes) resets the counter, stores ``hdr[r]`` and release-stores
   ``arrive[r] = e`` into every peer, then acquires ``arrive[t] == e`` and
   checks ``hdr[t]`` for every peer. The consumer runs in a later kernel
   ordered by an event, which also orders the multicast payload aliases
   (memory order across the two domains: the ``communication`` README).

Every wait is bounded by ``%globaltimer``. A timeout, an early credit or
arrival, or a header mismatch claims the error record (pinned host memory,
readable after the trap), prints a diagnostic, and traps; the group is then
unusable, as after an NCCL timeout.

The kernel is compiled once per state; every per-window value (row counts,
offsets, epoch, header) is a runtime argument, and the values fixed for the
state's lifetime (addresses, rank, timeout; the timeout is lowered for the
state's first window) are read from a small device array, so a launch passes
few arguments.
"""

import logging
import time
from collections.abc import Callable

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cuda.bindings import runtime as cudart
from cutlass._mlir.dialects import llvm
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Boolean, Int32, Int64, dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator
from tokenspeed_kernel.ops.communication._cute_dsl.fused_tp_projection_kernel import (
    _queue_poll_backoff,
)

logger = logging.getLogger(__name__)

# Byte offsets inside the flags block; every array owns a 128-byte line.
FLAGS_BYTES = 4096
_CREDIT = 0
_ARRIVE = 128
_HDR = 256
_CTA_DONE = 384
# Launch parameters fixed for a state's lifetime: int64 words of a device array
# the kernel reads, so that a launch passes only the per-window arguments.
(
    _PARAM_MULTICAST,
    _PARAM_FLAGS,
    _PARAM_PEER_TABLE,
    _PARAM_FLAGS_OFFSET,
    _PARAM_TIMEOUT,
    _PARAM_RANK,
    _PARAM_ERROR,
) = range(7)
# Bound of the first window's waits: every rank launches it right after the
# compile vote, and a rank that fails before its launch cannot be voted on.
_FIRST_WINDOW_TIMEOUT_NS = 30_000_000_000
# Threads per CTA and 16-byte loads in flight per thread. The grid-stride
# indices stay int32: one round covers at most 24 CTAs x 512 threads x 8
# vectors, and a rank's rows at most cap_bytes / 16 < 2**27 vectors.
_THREADS = 512
_UNROLL = 8
# CTAs per window: a multicast store reaches every peer at once, unicast
# stores need more CTAs in flight to reach the same bandwidth.
_NUM_CTAS = {"multicast": 2, "peer": 24}

# Error record (pinned host int64 words): stage, epoch, peer mask, rank,
# block, expected header, a word set once those are written, then the value
# observed per peer.
ERR_WORDS = 16
(
    ERR_STAGE,
    ERR_EPOCH,
    ERR_MASK,
    ERR_RANK,
    ERR_BLOCK,
    ERR_HDR,
    ERR_COMPLETE,
    ERR_OBSERVED,
) = (0, 1, 2, 3, 4, 5, 6, 8)
# Bound of a failing warp's wait for another warp's claim of the record to be
# written; the claimant has no wait of its own, so it is only a safeguard.
_ERR_WAIT_NS = 1_000_000_000
STAGE_ENTRY_TIMEOUT = 1
STAGE_ENTRY_AHEAD = 2
STAGE_EXIT_TIMEOUT = 3
STAGE_EXIT_AHEAD = 4
STAGE_HEADER = 5
STAGE_NAMES = {
    STAGE_ENTRY_TIMEOUT: "entry credit timeout",
    STAGE_ENTRY_AHEAD: "entry credit ahead of epoch",
    STAGE_EXIT_TIMEOUT: "exit arrival timeout",
    STAGE_EXIT_AHEAD: "exit arrival ahead of epoch",
    STAGE_HEADER: "window header mismatch",
}


@dsl_user_op
def _load_v4_no_allocate(address: Int64, *, loc=None, ip=None):
    # Coherent 16-byte load that does not allocate in L1 (rows are read once).
    scalar = Int32.mlir_type
    result = llvm.inline_asm(
        llvm.StructType.get_literal([scalar] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.global.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Int32(llvm.extractvalue(scalar, result, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def _multimem_store_v4(
    address: Int64, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None
):
    # multimem.st takes at most 16 bytes and no .b32 vectors; .f32 moves the
    # register bits unchanged (no arithmetic happens on a store).
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            c.ir_value(loc=loc, ip=ip),
            d.ir_value(loc=loc, ip=ip),
        ],
        "multimem.st.relaxed.sys.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _store_v4(
    address: Int64, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None
):
    # latent_moe_tail's store_global_u32x4 issues the same store without a
    # memory clobber; the clobber keeps the payload stores where the exit's
    # barrier and release expect them, as in _multimem_store_v4.
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            c.ir_value(loc=loc, ip=ip),
            d.ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _trap(*, loc=None, ip=None):
    # offload_hash's fail without a memory clobber: the clobber keeps the
    # error record's stores ahead of the trap.
    llvm.inline_asm(
        None,
        [],
        "trap;",
        "~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _global_i64(address: Int64):
    return cute.make_ptr(Int64, address, AddressSpace.gmem, assumed_align=8)


class PagedRowAllGatherKernel:
    """The fused paged-row all-gather for one row width, group size and mode.

    Args:
        row_bytes: Bytes per cache row, a multiple of 16.
        degree: Ranks in the group (2, 4 or 8).
        mode: ``multicast`` or ``peer``.
    """

    def __init__(self, row_bytes: int, degree: int, mode: str):
        if row_bytes % 16 != 0 or row_bytes <= 0:
            raise ValueError(
                f"row_bytes must be a positive multiple of 16, got {row_bytes}"
            )
        if degree not in (2, 4, 8):
            raise ValueError(f"degree must be 2, 4 or 8, got {degree}")
        if mode not in _NUM_CTAS:
            raise ValueError(f"mode must be one of {tuple(_NUM_CTAS)}, got {mode}")
        self.row_bytes = row_bytes
        self.row_vecs = row_bytes // 16
        self.degree = degree
        self.mode = mode

    # ------------------------------------------------------------------
    # Cross-rank flags
    # ------------------------------------------------------------------
    @cute.jit
    def _fail(self, params, stage, epoch, mask, observed, hdr, lane, bidx):
        """Claim the error record, print, trap. Called by a converged warp.

        A trap ends every CTA of the kernel, so a warp that finds the record
        already claimed (several CTAs can fail at once) waits for the
        claimant's words before it traps.
        """
        err_addr = self._word(params, _PARAM_ERROR)
        rank = Int32(self._word(params, _PARAM_RANK))
        claimed = Int32(0)
        if lane == 0:
            old = cute.arch.atomic_cas(
                _global_i64(err_addr + ERR_STAGE * 8),
                cmp=Int64(0),
                val=Int64(stage),
                sem="relaxed",
                scope="sys",
            )
            if old == Int64(0):
                claimed = Int32(1)
        claimed = cute.arch.shuffle_sync(claimed, 0)
        if claimed != 0:
            if lane < self.degree:
                cute.arch.store(
                    _global_i64(err_addr + (ERR_OBSERVED + lane) * 8),
                    observed,
                    sem="relaxed",
                    scope="sys",
                )
            if lane == 0:
                cute.arch.store(
                    _global_i64(err_addr + ERR_EPOCH * 8),
                    epoch,
                    sem="relaxed",
                    scope="sys",
                )
                cute.arch.store(
                    _global_i64(err_addr + ERR_MASK * 8),
                    Int64(mask),
                    sem="relaxed",
                    scope="sys",
                )
                cute.arch.store(
                    _global_i64(err_addr + ERR_RANK * 8),
                    Int64(rank),
                    sem="relaxed",
                    scope="sys",
                )
                cute.arch.store(
                    _global_i64(err_addr + ERR_BLOCK * 8),
                    Int64(bidx),
                    sem="relaxed",
                    scope="sys",
                )
                cute.arch.store(
                    _global_i64(err_addr + ERR_HDR * 8), hdr, sem="relaxed", scope="sys"
                )
            # Orders every lane's words before lane 0's fence, which makes the
            # store below a release of the whole record.
            cute.arch.sync_warp()
            cute.arch.fence_acq_rel_sys()
            if lane == 0:
                cute.arch.store(
                    _global_i64(err_addr + ERR_COMPLETE * 8),
                    Int64(1),
                    sem="relaxed",
                    scope="sys",
                )
        else:
            # The claimant's warp barrier and fence order its other words
            # before this one.
            start = cute.arch.globaltimer()
            complete = Int64(0)
            while complete == Int64(0):
                complete = cute.arch.load(
                    _global_i64(err_addr + ERR_COMPLETE * 8),
                    Int64,
                    sem="relaxed",
                    scope="sys",
                )
                if cute.arch.globaltimer() - start > _ERR_WAIT_NS:
                    complete = Int64(1)
                _queue_poll_backoff()
        if lane == 0:
            cute.printf(
                "paged_row_all_gather: rank {} block {} error stage {} epoch {} mask {}",
                rank,
                bidx,
                Int32(stage),
                epoch,
                mask,
            )
        _trap()

    @cute.jit
    def _word(self, table: Int64, index):
        """Int64 word ``index`` of a device array (peer bases, launch parameters)."""
        return cute.arch.load(_global_i64(table + Int64(index) * 8), Int64)

    @cute.jit
    def _protocol(self, params):
        """(flags, peer table, flags offset, timeout, rank) of the entry / exit.

        Loaded by each phase that needs them (the error path loads its own)
        rather than once per kernel, so that they are not live across the
        payload loops and the calls.
        """
        return (
            self._word(params, _PARAM_FLAGS),
            self._word(params, _PARAM_PEER_TABLE),
            self._word(params, _PARAM_FLAGS_OFFSET),
            self._word(params, _PARAM_TIMEOUT),
            Int32(self._word(params, _PARAM_RANK)),
        )

    @cute.jit
    def _wait_peers(self, words, epoch, hdr, lane, timeout_ns, params, stage, bidx):
        """Warp 0: lane t acquires ``words[t] == epoch`` for every peer t."""
        done = lane >= self.degree
        observed = Int64(0)
        start = cute.arch.globaltimer()
        pending = cute.arch.vote_ballot_sync(lane < self.degree)
        while pending != 0:
            ahead = Boolean(False)
            if not done:
                observed = cute.arch.load(
                    _global_i64(words + Int64(lane) * 8),
                    Int64,
                    sem="acquire",
                    scope="sys",
                )
                if observed == epoch:
                    done = Boolean(True)
                ahead = observed > epoch
            ahead_mask = cute.arch.vote_ballot_sync(ahead)
            pending = cute.arch.vote_ballot_sync(not done)
            if ahead_mask != 0:
                self._fail(
                    params, stage + 1, epoch, ahead_mask, observed, hdr, lane, bidx
                )
            if pending != 0:
                # Voted so that the whole warp enters _fail (it shuffles).
                if cute.arch.vote_any_sync(
                    cute.arch.globaltimer() - start > timeout_ns
                ):
                    self._fail(params, stage, epoch, pending, observed, hdr, lane, bidx)
                _queue_poll_backoff()

    @cute.jit
    def _entry(self, params, epoch, hdr, lane, bidx):
        flags, peer_tab, flags_off, timeout_ns, rank = self._protocol(params)
        if lane < self.degree:
            peer_flags = self._word(peer_tab, lane) + flags_off
            cute.arch.store(
                _global_i64(peer_flags + _CREDIT + Int64(rank) * 8),
                epoch,
                sem="release",
                scope="sys",
            )
        self._wait_peers(
            flags + _CREDIT,
            epoch,
            hdr,
            lane,
            timeout_ns,
            params,
            STAGE_ENTRY_TIMEOUT,
            bidx,
        )

    @cute.jit
    def _exit(self, params, epoch, hdr, tidx, warp, lane, bidx, nblk, last):
        flags, peer_tab, flags_off, timeout_ns, rank = self._protocol(params)
        # Every payload store of this CTA precedes thread 0's release below.
        cute.arch.barrier()
        if tidx == 0:
            old = cute.arch.atomic_add(
                _global_i64(flags + _CTA_DONE), Int64(1), sem="acq_rel", scope="sys"
            )
            last[0] = Int32(0)
            if old == Int64(nblk - 1):
                last[0] = Int32(1)
        # Hands thread 0's acquire (every CTA's stores) to the publishers.
        cute.arch.barrier()
        is_last = last[0]
        if is_last != 0:
            if tidx == 0:
                cute.arch.store(
                    _global_i64(flags + _CTA_DONE), Int64(0), sem="relaxed", scope="gpu"
                )
            if warp == 0:
                if lane < self.degree:
                    peer_flags = self._word(peer_tab, lane) + flags_off
                    cute.arch.store(
                        _global_i64(peer_flags + _HDR + Int64(rank) * 8),
                        hdr,
                        sem="relaxed",
                        scope="sys",
                    )
                    cute.arch.store(
                        _global_i64(peer_flags + _ARRIVE + Int64(rank) * 8),
                        epoch,
                        sem="release",
                        scope="sys",
                    )
                self._wait_peers(
                    flags + _ARRIVE,
                    epoch,
                    hdr,
                    lane,
                    timeout_ns,
                    params,
                    STAGE_EXIT_TIMEOUT,
                    bidx,
                )
                seen = Int64(0)
                mismatch = Boolean(False)
                if lane < self.degree:
                    seen = cute.arch.load(
                        _global_i64(flags + _HDR + Int64(lane) * 8),
                        Int64,
                        sem="relaxed",
                        scope="sys",
                    )
                    mismatch = seen != hdr
                mismatch_mask = cute.arch.vote_ballot_sync(mismatch)
                if mismatch_mask != 0:
                    self._fail(
                        params,
                        STAGE_HEADER,
                        epoch,
                        mismatch_mask,
                        seen,
                        hdr,
                        lane,
                        bidx,
                    )

    # ------------------------------------------------------------------
    # Payload stores
    # ------------------------------------------------------------------
    @cute.jit
    def _copy(self, fetch, plane_base, owned, dst_base, targets, tidx, bidx, nblk):
        """Grid-stride copy of this rank's rows, 16 bytes per thread and step."""
        unroll = _UNROLL
        row_vecs = self.row_vecs
        nthr = nblk * _THREADS
        nvec = owned * row_vecs
        step = nthr * unroll
        slots = cute.make_rmem_tensor((unroll,), Int32)
        regs = cute.make_rmem_tensor((4 * unroll,), Int32)
        first = bidx * _THREADS + tidx
        for u in cutlass.range_constexpr(unroll):
            v_first = first + u * nthr
            slots[u] = Int32(0)
            if v_first < nvec:
                slots[u] = fetch[v_first // row_vecs]
        for base in range(first, nvec, step):
            for u in cutlass.range_constexpr(unroll):
                v_load = base + u * nthr
                if v_load < nvec:
                    row = v_load // row_vecs
                    src = plane_base + Int64(slots[u]) * self.row_bytes
                    a, b, c, d = _load_v4_no_allocate(
                        src + Int64((v_load - row * row_vecs) * 16)
                    )
                    regs[4 * u] = a
                    regs[4 * u + 1] = b
                    regs[4 * u + 2] = c
                    regs[4 * u + 3] = d
            # The next round's slots load while this round's rows are in flight.
            for u in cutlass.range_constexpr(unroll):
                v_next = base + step + u * nthr
                if v_next < nvec:
                    slots[u] = fetch[v_next // row_vecs]
            for u in cutlass.range_constexpr(unroll):
                v_store = base + u * nthr
                if v_store < nvec:
                    offset = v_store * 16
                    if cutlass.const_expr(self.mode == "multicast"):
                        _multimem_store_v4(
                            dst_base + Int64(offset),
                            regs[4 * u],
                            regs[4 * u + 1],
                            regs[4 * u + 2],
                            regs[4 * u + 3],
                        )
                    else:
                        for j in cutlass.range_constexpr(self.degree):
                            _store_v4(
                                targets[j] + Int64(offset),
                                regs[4 * u],
                                regs[4 * u + 1],
                                regs[4 * u + 2],
                                regs[4 * u + 3],
                            )

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        fetch: cute.Tensor,
        plane_base: Int64,
        owned: Int32,
        dst_off: Int64,
        epoch: Int64,
        hdr: Int64,
        params: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        nblk, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(tidx // 32)
        lane = tidx % 32
        dst_base = dst_off
        if cutlass.const_expr(self.mode == "multicast"):
            dst_base = self._word(params, _PARAM_MULTICAST) + dst_off
        # The exit's "last CTA" verdict, shared from thread 0.
        smem = SmemAllocator()
        last = smem.allocate_tensor(Int32, cute.make_layout((1,)), byte_alignment=16)
        if warp == 0:
            self._entry(params, epoch, hdr, lane, bidx)
        cute.arch.barrier()
        # Peer mode: per-thread peer bases, rotated so warps start on
        # different peers.
        targets = None
        if cutlass.const_expr(self.mode == "peer"):
            peer_tab = self._word(params, _PARAM_PEER_TABLE)
            rank = Int32(self._word(params, _PARAM_RANK))
            targets = cute.make_rmem_tensor((self.degree,), Int64)
            first = (rank + 1 + warp) % self.degree
            for j in cutlass.range_constexpr(self.degree):
                targets[j] = self._word(peer_tab, (first + j) % self.degree) + dst_off
        self._copy(fetch, plane_base, owned, dst_base, targets, tidx, bidx, nblk)
        self._exit(params, epoch, hdr, tidx, warp, lane, bidx, nblk, last)

    @cute.jit
    def __call__(
        self,
        fetch: cute.Tensor,
        plane_base: Int64,
        owned: Int32,
        dst_off: Int64,
        epoch: Int64,
        hdr: Int64,
        params: Int64,
        num_ctas: Int32,
        stream,
    ):
        self.kernel(fetch, plane_base, owned, dst_off, epoch, hdr, params).launch(
            grid=(num_ctas, 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )


def compile_paged_row_all_gather(
    row_bytes: int, degree: int, mode: str
) -> Callable[..., None]:
    """Compile the kernel for the current device (or ``CUTE_DSL_ARCH``).

    Args:
        row_bytes: Bytes per cache row, a positive multiple of 16.
        degree: Ranks in the group (2, 4 or 8).
        mode: ``multicast`` or ``peer``.

    Returns:
        A callable taking ``(fetch_slots, plane_base, owned, dst_off, epoch,
        header, params, num_ctas, stream)`` that launches on ``stream`` (a raw
        ``cudaStream_t`` handle). ``fetch_slots`` is a 1-D int32 CUDA tensor
        of any length (TVM-FFI checks dtype, rank and device type);
        ``params`` is the address of the state's launch-parameter array
        (``_PARAM_*`` words).
    """
    fetch = cute.runtime.make_fake_compact_tensor(
        Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=4
    )
    zero64 = Int64(0)
    return cute.compile(
        PagedRowAllGatherKernel(row_bytes, degree, mode),
        fetch,
        zero64,
        Int32(0),
        zero64,
        zero64,
        zero64,
        zero64,
        Int32(1),
        stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        options="--enable-tvm-ffi",
    )


# Workspaces that went through the rendezvous stay referenced once their state
# falls back or closes: peers may map them.
_RETIRED: list[torch.Tensor] = []


class PagedRowAllGatherState:
    """A group's symmetric workspace, flags, stream and compiled kernel.

    Constructed only by ``build_paged_row_all_gather_state`` (through
    ``create_paged_row_all_gather_state`` in ``communication.cute_dsl``),
    which also runs the state's first window, epoch 1. The state numbers the
    windows and orders each one after the caller's stream and before its
    readers with events; the caller's contract is in the ``communication``
    README. Callers read :attr:`stream` (where the kernels run), :attr:`mode`,
    :attr:`num_ctas`, :attr:`row_bytes`, :attr:`cap_bytes`, :attr:`epoch`,
    :attr:`workspace` and :attr:`device_bytes`.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        mode: str,
        row_bytes: int,
        cap_bytes: int,
        workspace: torch.Tensor,
        handle: object,
        error: torch.Tensor,
        stream: torch.cuda.Stream,
        params: torch.Tensor,
        compiled: Callable[..., None],
    ):
        self.group = group
        self.device = device
        self.rank = group.rank()
        self.degree = group.size()
        self.mode = mode
        self.num_ctas = _NUM_CTAS[mode]
        self.row_bytes = row_bytes
        self.cap_bytes = cap_bytes
        self.slot_bytes = self.degree * cap_bytes
        self.workspace = workspace
        # The peer mappings, and the peer base table the kernel reads through
        # params.
        self.handle: object | None = handle
        # Host-pinned so the record stays readable after the kernel traps.
        self.error = error
        # Owned by the state (destroyed by close()), in the remote
        # memory-synchronization domain; launches go here.
        self.stream = stream
        self.stream_handle = stream.cuda_stream
        # The _PARAM_* words of every launch.
        self.params = params
        self.params_addr = params.data_ptr()
        # Both slots as [degree * cap_bytes // row_bytes, row_bytes] bytes.
        self.slot_views = [
            workspace[k * self.slot_bytes : (k + 1) * self.slot_bytes].view(
                -1, row_bytes
            )
            for k in (0, 1)
        ]
        self.compiled: Callable[..., None] | None = compiled
        # The last epoch launched; epoch e writes slot e & 1.
        self.epoch = 0
        # Per slot: the caller's stream at the launch, and the kernel's end.
        self._ready = (torch.cuda.Event(), torch.cuda.Event())
        self._done = (torch.cuda.Event(), torch.cuda.Event())
        # Free device memory the state took, from before the workspace
        # allocation to after its first window.
        self.device_bytes = 0
        self.closed = False

    def receive_view(self, epoch: int) -> torch.Tensor:
        """Epoch ``epoch``'s slot as ``[degree * cap_bytes // row_bytes, row_bytes]``.

        Rank ``p``'s rows of the window start at row ``p * send_rows``; the
        rows a window does not write hold older data, and reading them
        raises no error (the view spans the whole slot), so the reader
        indexes only rows the window wrote. Valid after the
        launch's completion is ordered before the reader and until the
        launch of ``epoch + 2``. Do not keep the view.
        """
        return self.slot_views[epoch & 1]

    def launch(
        self,
        plane: torch.Tensor,
        fetch_slots: torch.Tensor,
        owned: int,
        send_rows: int,
        header: int,
    ) -> tuple[int, torch.cuda.Event]:
        """Gather ``fetch_slots[:owned]`` rows of ``plane`` from every rank
        into the next epoch's slot, after the current stream's work so far.

        Checked here: the state is open, neither stream is capturing, the
        bounds that keep the stores inside the slots, and the plane's and
        ``fetch_slots``' layout and device; TVM-FFI checks ``fetch_slots``'
        dtype and rank. The arguments are ``cute_dsl_paged_row_all_gather``'s.

        Returns:
            The window's epoch and the event its readers wait for.
        """
        if self.closed:
            raise RuntimeError("paged_row_all_gather: the state is closed")
        # A captured launch would replay with a stale epoch. The state's stream
        # would join a capture of the current stream through the ready event.
        current = torch.cuda.current_stream(self.device)
        for handle in (current.cuda_stream, self.stream_handle):
            err, capture = cudart.cudaStreamIsCapturing(handle)
            if (
                err != cudart.cudaError_t.cudaSuccess
                or capture != cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusNone
            ):
                raise RuntimeError(
                    "paged_row_all_gather launches only in eager mode "
                    f"({err}, {capture})"
                )
        # The kernel addresses rows as plane + slot * row_bytes (bytes) with
        # 16-byte loads. A tensor on another GPU would launch there, away from
        # the state's stream.
        if not (
            0 <= owned <= send_rows
            and send_rows * self.row_bytes <= self.cap_bytes
            and max(owned, 1) <= fetch_slots.numel()
            and fetch_slots.device == self.device
            and plane.dim() >= 2
            and plane.is_contiguous()
            and plane.shape[1:].numel() * plane.element_size() == self.row_bytes
            and plane.data_ptr() % 16 == 0
            and plane.device == self.device
            and 0 <= header < 2**63
        ):
            raise ValueError(
                f"launch out of range: owned {owned}, send_rows {send_rows} (slot "
                f"capacity), fetch_slots {fetch_slots.numel()} entries on "
                f"{fetch_slots.device}, plane {tuple(plane.shape)} of {plane.dtype} "
                f"on {plane.device} (contiguous {self.row_bytes}-byte rows from a "
                f"16-byte aligned base; state {self.device}), header {header} "
                "(< 2**63)"
            )
        epoch = self.epoch + 1
        slot = epoch & 1
        self._ready[slot].record(current)
        self.stream.wait_event(self._ready[slot])
        # Booked before the call: a call that fails once its kernel is
        # enqueued must not hand the epoch out again.
        self.epoch = epoch
        self.compiled(
            fetch_slots,
            plane.data_ptr(),
            owned,
            slot * self.slot_bytes + self.rank * send_rows * self.row_bytes,
            epoch,
            header,
            self.params_addr,
            self.num_ctas,
            self.stream_handle,
        )
        self._done[slot].record(self.stream)
        return epoch, self._done[slot]

    def read_error(self) -> dict | None:
        """The claimed error record, or None. Reads host memory only."""
        words = self.error.tolist()
        if words[ERR_STAGE] == 0:
            return None
        return {
            "stage": STAGE_NAMES.get(words[ERR_STAGE], str(words[ERR_STAGE])),
            "stage_code": words[ERR_STAGE],
            "epoch": words[ERR_EPOCH],
            "mask": words[ERR_MASK],
            "rank": words[ERR_RANK],
            "block": words[ERR_BLOCK],
            "header": words[ERR_HDR],
            # False when the kernel ended before the claimant wrote the rest.
            "complete": words[ERR_COMPLETE] != 0,
            "observed": words[ERR_OBSERVED : ERR_OBSERVED + self.degree],
        }

    def close(self) -> None:
        """Collective: drain, then drop the handle and destroy :attr:`stream`.

        The workspace stays mapped. Launches raise afterwards, and another
        call does nothing.
        """
        if self.closed:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self.closed = True
        _RETIRED.append(self.workspace)
        self.handle = None
        self.compiled = None
        (err,) = cudart.cudaStreamDestroy(self.stream_handle)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"paged_row_all_gather: cudaStreamDestroy: {err}")


def build_paged_row_all_gather_state(
    group: dist.ProcessGroup,
    device: torch.device,
    *,
    row_bytes: int,
    cap_bytes: int,
    timeout_ns: int,
    modes: tuple[str, ...],
    multicast: bool,
) -> PagedRowAllGatherState | None:
    """The collective steps after the group agreed it is eligible.

    Order (every vote is a MIN all-reduce of host integers over ``group``, so
    all ranks take the same branch): local allocation and zeroing; a vote on
    the allocation; the rendezvous; the error record, launch parameters,
    remote-domain stream and the first window's slot list, then one vote on
    them and on the mode; the compile, then a vote on it; the first window
    (epoch 1, no row), which loads the module and checks every peer's flag
    words through its unicast mapping. What is fatal rather than voted is in
    the ``communication`` README.

    Args:
        group: The CPU (gloo) group over the gather ranks.
        device: This rank's CUDA device.
        row_bytes: Bytes per cache row.
        cap_bytes: Receive bytes per rank and slot, whole rows.
        timeout_ns: Bound of every in-kernel wait after the first window.
        modes: Modes in preference order, from ``multicast`` and ``peer``.
        multicast: Every rank's device supports multicast objects.

    Returns:
        The state, or None on every rank when the group must keep its
        existing all-gather.
    """

    def agree(*values: int) -> list[int]:
        flags = torch.tensor(values, dtype=torch.int64)
        dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=group)
        return flags.tolist()

    degree = group.size()
    total = 2 * degree * cap_bytes + FLAGS_BYTES
    workspace = None
    free_before = torch.cuda.mem_get_info(device)[0]
    try:
        with torch.inference_mode(False), torch.no_grad():
            workspace = symm_mem.empty((total,), dtype=torch.uint8, device=device)
        workspace.zero_()
        torch.cuda.current_stream(device).synchronize()
        allocated = 1
    except Exception as exc:  # noqa: BLE001 - local failure is voted on
        logger.warning(f"paged_row_all_gather: workspace allocation failed: {exc!r}")
        allocated = 0
    (all_allocated,) = agree(allocated)
    if not all_allocated:
        # No peer mapped it: no rendezvous ran.
        return None

    handle = symm_mem.rendezvous(workspace, group=group)
    if handle.rank != group.rank() or handle.world_size != degree:
        raise RuntimeError(
            f"symmetric-memory handle rank {handle.rank}/{handle.world_size} "
            f"!= group rank {group.rank()}/{degree}"
        )
    stream_handle = 0
    multicast_ptr = 0
    try:
        # The kernel reads every peer's base from the handle's device table,
        # which holds each peer's allocation base: the workspace is all of it.
        if handle.buffer_ptrs[handle.rank] != workspace.data_ptr():
            raise RuntimeError("the symmetric buffer does not start at the workspace")
        error = torch.zeros(ERR_WORDS, dtype=torch.int64, pin_memory=True)
        # The multicast word is unused in peer mode (the vote below picks it).
        multicast_ptr = int(handle.multicast_ptr) if multicast else 0
        flags_off = 2 * degree * cap_bytes
        words = {
            _PARAM_MULTICAST: multicast_ptr,
            _PARAM_FLAGS: workspace.data_ptr() + flags_off,
            _PARAM_PEER_TABLE: int(handle.buffer_ptrs_dev),
            _PARAM_FLAGS_OFFSET: flags_off,
            # The first window's bound; timeout_ns once it has run.
            _PARAM_TIMEOUT: min(timeout_ns, _FIRST_WINDOW_TIMEOUT_NS),
            _PARAM_RANK: group.rank(),
            _PARAM_ERROR: error.data_ptr(),
        }
        params = torch.tensor(
            [words[k] for k in sorted(words)], dtype=torch.int64, device=device
        )
        # A stream of the state's own: torch.cuda.Stream() hands out pool
        # streams that wrap around and are shared with other components,
        # whose kernels would then run in this domain too, serialized behind
        # the gather's in-kernel waits.
        with torch.cuda.device(device):
            err, created = cudart.cudaStreamCreateWithFlags(
                cudart.cudaStreamNonBlocking
            )
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaStreamCreateWithFlags: {err}")
        stream_handle = int(created)
        stream = torch.cuda.ExternalStream(stream_handle, device=device)
        # The launch attribute applies to every kernel of the stream (CuteDSL
        # launches take no memory-synchronization domain); read it back.
        domain = cudart.cudaStreamAttrValue()
        domain.memSyncDomain = (
            cudart.cudaLaunchMemSyncDomain.cudaLaunchMemSyncDomainRemote
        )
        attr = cudart.cudaStreamAttrID.cudaLaunchAttributeMemSyncDomain
        (err,) = cudart.cudaStreamSetAttribute(stream_handle, attr, domain)
        err_get, got = cudart.cudaStreamGetAttribute(stream_handle, attr)
        if (
            err != cudart.cudaError_t.cudaSuccess
            or err_get != cudart.cudaError_t.cudaSuccess
            or got.memSyncDomain != domain.memSyncDomain
        ):
            raise RuntimeError(
                f"remote memory-synchronization domain not set: {err}, {err_get}"
            )
        # The first window's slot list (it reads no row).
        fetch = torch.zeros(1, dtype=torch.int32, device=device)
        torch.cuda.synchronize(device)
        prepared = 1
    except Exception as exc:  # noqa: BLE001 - local failure is voted on
        logger.warning(f"paged_row_all_gather: state setup failed: {exc!r}")
        prepared = 0
    has_mc, all_prepared = agree(int(multicast_ptr != 0), prepared)
    mode = next(
        (candidate for candidate in modes if candidate == "peer" or has_mc), None
    )
    compiled = None
    compile_s = 0.0
    if mode is not None and all_prepared:
        start = time.perf_counter()
        try:
            with torch.cuda.device(device):
                compiled = compile_paged_row_all_gather(row_bytes, degree, mode)
        except Exception as exc:  # noqa: BLE001 - local failure is voted on
            logger.warning(f"paged_row_all_gather: compile failed: {exc!r}")
        compile_s = time.perf_counter() - start
        (all_compiled,) = agree(int(compiled is not None))
        if not all_compiled:
            compiled = None
    if compiled is None:
        _RETIRED.append(workspace)
        if stream_handle:
            # Nothing was launched on it.
            cudart.cudaStreamDestroy(stream_handle)
        if group.rank() == 0:
            reason = (
                "no requested mode is available"
                if mode is None
                else "state setup or compile failed on some rank"
            )
            logger.info(f"paged_row_all_gather: {reason}")
        return None

    state = PagedRowAllGatherState(
        group,
        device,
        mode,
        row_bytes,
        cap_bytes,
        workspace,
        handle,
        error,
        stream,
        params,
        compiled,
    )
    # Epoch 1 on every rank: an empty window, no row read or written (any
    # valid plane will do).
    state.launch(state.slot_views[0], fetch, 0, 1, 0)
    try:
        torch.cuda.synchronize(device)
    except Exception as exc:  # noqa: BLE001 - a trap surfaces as a CUDA error
        raise RuntimeError(
            f"paged_row_all_gather: the first window failed: {state.read_error()}"
        ) from exc
    state.params[_PARAM_TIMEOUT] = timeout_ns
    torch.cuda.synchronize(device)
    state.device_bytes = free_before - torch.cuda.mem_get_info(device)[0]
    if group.rank() == 0:
        logger.info(
            f"paged_row_all_gather: {mode} mode on {degree} ranks, "
            f"{state.num_ctas} CTAs, {total / 2**20:.1f} MiB workspace, compiled "
            f"in {compile_s:.1f} s"
        )
    return state
