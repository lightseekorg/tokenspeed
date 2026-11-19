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

"""One-hop all-gather and all-to-all of small payloads over symmetric memory.

Every rank owns a receive workspace ``[2 parities][ranks][cap units][32 B]``
mapped by every peer. A 16-byte payload unit travels as four 64-bit LL
elements, each holding one payload word in its low half and the call's epoch
in its high half, in one 32-byte relaxed system-scope store into the
receiver's slot ``[parity][sender][unit]`` (all-gather: one store through the
NVLink multicast address reaches every rank). The receiver loads the four
elements until each carries the epoch, then stores the words. An aligned
64-bit element is single-copy atomic, so a matching epoch proves its word is
this call's: no flag, fence or sentinel, and any payload bits pass.

Calls alternate the two parity slots by a 64-bit call count. A rank writes
parity ``p`` of call ``n + 2`` only after receiving its peers' call ``n + 1``,
which each peer sends after finishing call ``n``, so no slot is overwritten
while it is read. Call ``n``'s epoch is ``(n - 1) mod (2**32 - 1) + 1``: never
0, the zeroed workspace's, so an element no call wrote never matches. A stale
element matches only where the slot's last write came a multiple of
``2 * (2**32 - 1)`` calls earlier and the read beats the call's own store. The
count lives in device memory: the last CTA of a call publishes it, so graph
replays need no host state. Every wait is bounded by ``%globaltimer``; a
timeout traps, as a hung NCCL kernel would end the group.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack, make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op

# Payload bytes per unit; a unit occupies twice that in a receive slot.
UNIT = 16
THREADS = 512
# Fewer CTAs than SMs, so a call's grid fits beside the kernels around it.
MAX_CTAS = 64
# Per-rank float-gather payloads from which a call takes every SM.
WIDE_FLOAT_BYTES = 256 << 10


def _asm(code, constraints, operands, loc, ip):
    llvm.inline_asm(
        None,
        [operand.ir_value(loc=loc, ip=ip) for operand in operands],
        code,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _push_unit(src, dst, epoch, *, loc=None, ip=None):
    """Store the 16 payload bytes at ``src`` as four LL elements at ``dst``."""
    code = (
        "{ .reg .b32 w0, w1, w2, w3; .reg .b64 a, b, c, d;\n"
        "ld.global.nc.v4.b32 {w0, w1, w2, w3}, [$0];\n"
        "mov.b64 a, {w0, $2}; mov.b64 b, {w1, $2};\n"
        "mov.b64 c, {w2, $2}; mov.b64 d, {w3, $2};\n"
        "st.relaxed.sys.global.v4.b64 [$1], {a, b, c, d}; }"
    )
    _asm(code, "l,l,r", [src, dst, epoch], loc, ip)


@dsl_user_op
def _poll_unit(slot, out, epoch, deadline, *, loc=None, ip=None):
    """Wait until the four LL elements at ``slot`` carry ``epoch``, store the words."""
    code = (
        "{ .reg .b64 a, b, c, d, now; .reg .b32 l0, h0, l1, h1, l2, h2, l3, h3;\n"
        ".reg .pred p;\n"
        "LL_WAIT${:uid}:\n"
        "ld.relaxed.sys.global.v4.b64 {a, b, c, d}, [$0];\n"
        "mov.b64 {l0, h0}, a; mov.b64 {l1, h1}, b;\n"
        "mov.b64 {l2, h2}, c; mov.b64 {l3, h3}, d;\n"
        "setp.ne.u32 p, h0, $2; @p bra LL_LATE${:uid};\n"
        "setp.ne.u32 p, h1, $2; @p bra LL_LATE${:uid};\n"
        "setp.ne.u32 p, h2, $2; @p bra LL_LATE${:uid};\n"
        "setp.ne.u32 p, h3, $2; @p bra LL_LATE${:uid};\n"
        "bra.uni LL_DONE${:uid};\n"
        "LL_LATE${:uid}:\n"
        "mov.u64 now, %globaltimer;\n"
        "setp.lt.u64 p, now, $3; @p bra LL_WAIT${:uid};\n"
        "trap;\n"
        "LL_DONE${:uid}:\n"
        "st.global.v4.b32 [$1], {l0, l1, l2, l3}; }"
    )
    _asm(code, "l,l,r,l", [slot, out, epoch, deadline], loc, ip)


# The float exchange's armed word: a NaN no sender stores (it sends 0x7FFFFFFF).
SENTINEL = 0xFFFFFFFF


@dsl_user_op
def _push_float_unit(src, dst, *, loc=None, ip=None):
    """Store the 16 bytes at ``src`` at ``dst``, a sentinel word as 0x7FFFFFFF."""
    code = "{ .reg .b32 w0, w1, w2, w3; .reg .pred p;\n"
    code += "ld.global.nc.v4.b32 {w0, w1, w2, w3}, [$0];\n"
    for w in range(4):
        code += f"setp.eq.u32 p, w{w}, {SENTINEL}; @p mov.b32 w{w}, 0x7FFFFFFF;\n"
    code += "st.relaxed.sys.global.v4.b32 [$1], {w0, w1, w2, w3}; }"
    _asm(code, "l,l", [src, dst], loc, ip)


@dsl_user_op
def _poll_float_unit(slot, out, deadline, *, loc=None, ip=None):
    """Wait until no word at ``slot`` is the sentinel, store the words, re-arm it."""
    code = "{ .reg .b32 w0, w1, w2, w3; .reg .b64 now; .reg .pred p;\n"
    code += "SENT_WAIT${:uid}:\n"
    code += "ld.relaxed.sys.global.v4.b32 {w0, w1, w2, w3}, [$0];\n"
    for w in range(4):
        code += f"setp.eq.u32 p, w{w}, {SENTINEL}; @p bra SENT_LATE${{:uid}};\n"
    code += "bra.uni SENT_DONE${:uid};\n"
    code += "SENT_LATE${:uid}:\n"
    code += "mov.u64 now, %globaltimer;\n"
    code += "setp.lt.u64 p, now, $2; @p bra SENT_WAIT${:uid};\n"
    code += "trap;\n"
    code += "SENT_DONE${:uid}:\n"
    code += "st.global.v4.b32 [$1], {w0, w1, w2, w3};\n"
    code += f"st.global.v4.b32 [$0], {{{SENTINEL}, {SENTINEL}, {SENTINEL}, {SENTINEL}}}; }}"
    _asm(code, "l,l,l", [slot, out, deadline], loc, ip)


@dsl_user_op
def _globaltimer(*, loc=None, ip=None):
    """The ``%globaltimer`` nanosecond clock that bounds every wait."""
    value = llvm.inline_asm(
        T.i64(),
        [],
        "mov.u64 $0, %globaltimer;",
        "=l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int64(value)


@dsl_user_op
def _load_count(ctrl, *, loc=None, ip=None):
    """The call count the last call's final CTA published at ``ctrl``."""
    value = llvm.inline_asm(
        T.i64(),
        [ctrl.ir_value(loc=loc, ip=ip)],
        "ld.relaxed.gpu.global.u64 $0, [$1];",
        "=l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint64(value)


@dsl_user_op
def _finish(ctrl, grid, count, *, loc=None, ip=None):
    """Count this CTA in; the last one resets the ticket and publishes ``count``."""
    code = (
        "{ .reg .u32 seen; .reg .pred last;\n"
        "atom.add.acq_rel.gpu.global.u32 seen, [$0+8], 1;\n"
        "add.u32 seen, seen, 1;\n"
        "setp.eq.u32 last, seen, $1;\n"
        "@last st.relaxed.gpu.global.u32 [$0+8], 0;\n"
        "@last st.relaxed.gpu.global.u64 [$0], $2; }"
    )
    _asm(code, "l,r,l", [ctrl, grid, count], loc, ip)


class LLExchange:
    """One call: push this rank's units, then receive every rank's."""

    def __init__(self, ranks: int, gather: bool, multicast: bool, floats: bool):
        self.ranks = ranks
        self.gather = gather
        self.multicast = multicast
        # Float words travel as themselves (see SENTINEL), else as LL elements.
        self.floats = floats
        self.slot = UNIT if floats else 2 * UNIT

    @cute.jit
    def __call__(self, src: Int64, out: Int64, peers: cute.Tensor, ctrl: Int64, units: Int32, cap: Int32, rank: Int32, grid: Int32, multicast: Int64, timeout: Int64, stream: cuda.CUstream):  # fmt: skip
        self.kernel(
            src, out, peers, ctrl, units, cap, rank, grid, multicast, timeout
        ).launch(  # fmt: skip
            grid=(grid, 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, src: Int64, out: Int64, peers: cute.Tensor, ctrl: Int64, units: Int32, cap: Int32, rank: Int32, grid: Int32, multicast: Int64, timeout: Int64):  # fmt: skip
        """Push this rank's units for the call's parity, then poll every rank's."""
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        deadline = _globaltimer() + timeout
        count = _load_count(ctrl) + Uint64(1)
        epoch = Uint32((count - Uint64(1)) % Uint64(0xFFFFFFFF) + Uint64(1))
        parity = Int64(count & Uint64(1))
        sender_row = (parity * self.ranks + Int64(rank)) * Int64(cap)
        total = Int64(self.ranks) * Int64(units)
        start = Int64(block) * THREADS + Int64(tid)
        stride = Int64(grid) * THREADS
        # Push: unit j for rank d into d's slot [parity][rank][j].
        i = start
        if cutlass.const_expr(self.multicast):
            while i < Int64(units):
                self.push(src + i * UNIT, multicast + (sender_row + i) * self.slot, epoch)  # fmt: skip
                i = i + stride
        else:
            while i < total:
                d = i // Int64(units)
                j = i - d * Int64(units)
                source = src + (j if self.gather else i) * UNIT
                self.push(source, peers[Int32(d)] + (sender_row + j) * self.slot, epoch)  # fmt: skip
                i = i + stride
        # Receive: unit j of rank s from this rank's slot [parity][s][j].
        mine = peers[rank]
        parity_row = parity * self.ranks
        i = start
        while i < total:
            s = i // Int64(units)
            j = i - s * Int64(units)
            slot = mine + ((parity_row + s) * Int64(cap) + j) * self.slot
            if cutlass.const_expr(self.floats):
                _poll_float_unit(slot, out + i * UNIT, deadline)
            else:
                _poll_unit(slot, out + i * UNIT, epoch, deadline)
            i = i + stride
        cute.arch.sync_threads()
        if tid == 0:
            _finish(ctrl, Uint32(grid), count)

    @cute.jit
    def push(self, src: Int64, dst: Int64, epoch: Uint32):
        if cutlass.const_expr(self.floats):
            _push_float_unit(src, dst)
        else:
            _push_unit(src, dst, epoch)


@functools.cache
def _compiled(ranks: int, gather: bool, multicast: bool, floats: bool, device_index: int):  # fmt: skip
    with torch.cuda.device(device_index):
        peers = make_fake_compact_tensor(cutlass.Int64, (ranks,), assumed_align=8)
        args = (Int64(0), Int64(0), peers, Int64(0), Int32(1), Int32(1), Int32(0))
        args += (Int32(1), Int64(0), Int64(1), make_fake_stream())
        return cute.compile(LLExchange(ranks, gather, multicast, floats), *args)


class LLExchangeState:
    """A group's receive workspace, epoch and compiled calls; see the module docstring."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        *,
        cap_bytes: int,
        float_cap_bytes: int,
        multicast: bool,
        timeout_ns: int,
    ) -> None:
        """Allocate, rendezvous, and compile the group's exchange (collective).

        Every rank calls this before graph capture; the workspace starts zeroed.
        """
        self.ranks = group.size()
        self.rank = group.rank()
        self.device = device
        self.cap_units = cap_bytes // UNIT
        self.cap_bytes = self.cap_units * UNIT
        self.float_cap_units = float_cap_bytes // UNIT
        self.float_cap_bytes = self.float_cap_units * UNIT
        self.timeout_ns = timeout_ns
        with torch.cuda.device(device):
            # Epoch 0 marks every LL element unwritten; no call's epoch is 0.
            self.workspace, self.handle, self.peers = self._workspace(
                group, self.cap_units * 2 * UNIT, 0, device
            )
            self.multicast_ptr = int(self.handle.multicast_ptr) if multicast else 0
            # The float exchange's slots start armed with the sentinel.
            self.float_workspace, self.float_handle, self.float_peers = (
                self._workspace(group, self.float_cap_units * UNIT, 0xFF, device)
            )
            self.float_multicast_ptr = (
                int(self.float_handle.multicast_ptr) if multicast else 0
            )
            # Each workspace's call count, then the CTA ticket in the low half
            # of word 1.
            self.ctrl = torch.zeros(2, dtype=torch.int64, device=device)
            self.float_ctrl = torch.zeros(2, dtype=torch.int64, device=device)
            self.peers_view = from_dlpack(self.peers, assumed_align=8)
            self.float_peers_view = from_dlpack(self.float_peers, assumed_align=8)
            self.sms = torch.cuda.get_device_properties(device).multi_processor_count
            for gather in (True, False):
                _compiled(self.ranks, gather, gather and multicast, False, device.index)
            _compiled(self.ranks, True, multicast, True, device.index)
            torch.cuda.synchronize(device)
        # Every rank's workspaces are set before any peer stores into them.
        dist.barrier(group=group)

    def _workspace(self, group, slot_bytes: int, fill: int, device):
        """Two parities of ``slot_bytes`` per rank, filled with ``fill`` and
        mapped by every rank: (workspace, handle, peer addresses)."""
        workspace = symm_mem.empty(
            (2 * self.ranks * max(slot_bytes, UNIT),), dtype=torch.uint8, device=device
        )
        workspace.fill_(fill)
        handle = symm_mem.rendezvous(workspace, group=group)
        peers = torch.tensor(list(handle.buffer_ptrs), dtype=torch.int64, device=device)
        return workspace, handle, peers

    def serves(self, payload_bytes: int, *, floats: bool) -> bool:
        """Whether a per-rank payload of this size fits one call; ``floats``:
        one of float words (an all-gather)."""
        cap = self.float_cap_bytes if floats else self.cap_bytes
        return 0 < payload_bytes <= cap and payload_bytes % UNIT == 0

    def launch(
        self, src: torch.Tensor, out: torch.Tensor, *, gather: bool, floats: bool
    ) -> None:
        """Run one call; the public entries validate the tensors first.
        ``floats`` (gathers only) sends float words as themselves."""
        payload = src.numel() * src.element_size() // (1 if gather else self.ranks)
        units = payload // UNIT
        # A large float gather is bandwidth-bound and takes every SM it can
        # use; smaller calls leave SMs to the kernels they overlap.
        cap = self.sms if floats and payload >= WIDE_FLOAT_BYTES else MAX_CTAS
        grid = max(1, min(cap, -(-self.ranks * units // THREADS)))
        stream = cuda.CUstream(torch.cuda.current_stream(self.device).cuda_stream)
        multicast_ptr = self.float_multicast_ptr if floats else self.multicast_ptr
        multicast = gather and multicast_ptr != 0
        _compiled(self.ranks, gather, multicast, floats, self.device.index)(
            Int64(src.data_ptr()),
            Int64(out.data_ptr()),
            self.float_peers_view if floats else self.peers_view,
            Int64((self.float_ctrl if floats else self.ctrl).data_ptr()),
            Int32(units),
            Int32(self.float_cap_units if floats else self.cap_units),
            Int32(self.rank),
            Int32(grid),
            Int64(multicast_ptr),
            Int64(self.timeout_ns),
            stream,
        )


def build_ll_exchange_state(
    group: dist.ProcessGroup,
    device: torch.device,
    *,
    cap_bytes: int,
    float_cap_bytes: int,
    multicast: bool,
    timeout_ns: int,
) -> LLExchangeState:
    """Allocate, rendezvous and compile a group's LL exchange (collective)."""
    return LLExchangeState(
        group,
        device,
        cap_bytes=cap_bytes,
        float_cap_bytes=float_cap_bytes,
        multicast=multicast,
        timeout_ns=timeout_ns,
    )
