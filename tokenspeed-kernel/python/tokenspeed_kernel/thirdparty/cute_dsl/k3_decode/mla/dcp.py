# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Vendored from vLLM's Kimi-K3.1 specialized decode (cutedsl/mla/dcp.py): DcpValue's
# BF16 exchange, without the query mailbox, the projected exchange and the rows a
# sparse decode pushes. Modified:
# - Row counts are runtime values: one compile per head count, degree, capacity and
#   gate kind.
# - The merge is combine_dcp_attention_partials' bit for bit (natural-log LSEs, +inf
#   as -inf, ex2.approx of (lse - max) log2(e), Triton's sum over eight lanes,
#   div.full, sources added in rank order with unfused products), every row is sent
#   and polled, and the projection and gate are MlaValue's (value.py): a call gives
#   the bits of an all-to-all, that combine and mla_value.
# - A use's parity comes from a call count in device memory, so any sequence of
#   calls alternates parities, graph replays included.
# - NaN LSEs are sent canonical, so no sent LSE is the sentinel.

"""The DCP tail of a sparse MLA decode: partials to their head owners, merged and projected.

Every rank of a KVP group holds attention partials of all ``D * H`` gathered heads
over its own keys: BF16 rows ``[B, D * H, 512]`` and natural-log LSEs ``[B, D * H]``.
Head ``gh`` belongs to rank ``gh // H``. One launch per rank:

  push   every CTA stores its share of the rows and LSEs into slot ``[parity][rank]``
         of their owners' Lamport mailboxes (peer stores; a row's negative BF16
         zeros go as +0, a NaN LSE as the canonical NaN, so neither is a sentinel)
  merge  CTA (h, tile) polls the D sources of 16 rows of local head h, re-arms what
         it read, merges them in FP32 as combine_dcp_attention_partials does, rounds
         to BF16, and projects them through W_UV with the optional sigmoid gate as
         MlaValue does

The mailbox's data marks its own arrival, so no rank waits on a barrier. A rank
publishes into parity ``p`` again only after the use between, which every peer
starts after its merge of this use: no slot is overwritten while it is read.
"""

import math
from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, Uint64
from cutlass.cute.runtime import make_fake_stream

from ..common.host import architecture, dynamic_view, fake, to_cute
from ..common.primitives import (
    add_rn,
    alloc,
    asm,
    at,
    bf16x2,
    bits_float,
    float_bits,
    launch,
    load_words,
    mma_into,
    no_neg_zero,
    pack_pair,
    published,
    store_relaxed,
    store_words,
    unpack_bf16,
    zeros,
)
from .value import LATENT, VALUE, triton_sigmoid

LSE_SENT = 0xFFFFFFFF  # a NaN no sender stores
PAIR_SENT = 0x80000000  # BF16 pair (+0, -0); sent pairs have no negative zero
CANONICAL_NAN = 0x7FC00000
THREADS = 512
ROWS = 16  # rows per merge tile
CHUNKS = LATENT // 8  # 16 B chunks per row
LOG2E = 0x3FB8AA3B
MIN_NORMAL = 0x00800000


@asm(Float32, pure=True)
def sub_rn(a, b):
    return "sub.rn.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def mul_rn(a, b):
    return "mul.rn.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def max_f32(a, b):
    return "max.f32 $0,$1,$2;", "=f,f,f", (Float32(a), Float32(b))


@asm(Float32, pure=True)
def triton_exp(value):
    """``tl.exp``: ex2.approx of the value times log2(e)."""
    code = f"{{ .reg .f32 t; mul.rn.f32 t,$1,0f{LOG2E:08X}; ex2.approx.f32 $0,t; }}"
    return code, "=f,f", (Float32(value),)


@asm(Float32, pure=True)
def triton_divide(a, b):
    """``a / max(b, FLT_MIN)`` as Triton's division of the weights."""
    code = f"{{ .reg .f32 t; max.f32 t,$2,0f{MIN_NORMAL:08X}; div.full.f32 $0,$1,t; }}"
    return code, "=f,f,f", (Float32(a), Float32(b))


@asm(Uint64)
def load_count(ctrl):
    return "ld.relaxed.gpu.global.u64 $0, [$1];", "=l,l", (ctrl,)


@asm()
def finish(ctrl, grid, count):
    """Count this CTA in; the last one resets the ticket and publishes ``count``."""
    code = (
        "{ .reg .u32 seen; .reg .pred last;\n"
        "atom.add.acq_rel.gpu.global.u32 seen, [$0+8], 1;\n"
        "add.u32 seen, seen, 1;\n"
        "setp.eq.u32 last, seen, $1;\n"
        "@last st.relaxed.gpu.global.u32 [$0+8], 0;\n"
        "@last st.relaxed.gpu.global.u64 [$0], $2; }"
    )
    return code, "l,r,l", (ctrl, Uint32(grid), Uint64(count))


class DcpValue:
    """One 512-thread CTA per (local head, merge tile, pusher): grid ``(H, tiles,
    pushers)``. Every CTA pushes its share first; CTAs of pusher 0 then merge the
    16-row tiles ``tile, tile + tiles, ..`` of their head. Warp ``r`` weights row
    ``t0 + r`` over the D sources; warp ``w`` then owns outputs ``32 (w // 4) + 8
    (w % 4) .. + 7`` of the head, as CTA ``w // 4``'s warp ``w % 4`` of MlaValue.
    The grid holds at most one CTA per SM, so every CTA's pushes land while the
    merging CTAs poll."""

    def __init__(self, heads: int, size: int, cap: int, gated: bool):
        self.heads, self.size, self.cap, self.gated = heads, size, cap, gated

    @cute.jit
    def __call__(self, rows, lse, box, lbox, peers, lpeers, w_uv, gate, output, ctrl: Int64, batch: Int32, rank: Int32, tiles: Int32, pushers: Int32, stream):  # fmt: skip
        args = (rows, lse, box, lbox, peers, lpeers, w_uv, gate, output, ctrl)
        args += (batch, rank, tiles, pushers, stream)
        launch(self.kernel, args, (self.heads, tiles, pushers), THREADS)

    @cute.kernel
    def kernel(self, args):
        w_uv, ctrl, tiles, pushers = args[6], args[9], args[12], args[13]
        smem = cutlass.utils.SmemAllocator()
        latent = alloc(smem, BFloat16, (ROWS, LATENT + 32), 128)
        weights = alloc(smem, Float32, (ROWS, 8))
        tid, _, _ = cute.arch.thread_idx()
        head, tile, pusher = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        v = 32 * (warp // 4) + 8 * (warp % 4) + group
        # B fragments: 8 latent values per 32-wide chunk; A uses the same permutation.
        # Only merging CTAs project.
        wb = []
        for c in cutlass.range_constexpr(16):
            words = (Uint32(0),) * 4
            if pusher == 0:
                words = load_words(at(w_uv, head, 32 * c + 8 * quarter, v), count=4)
            wb.append(words)
        cute.arch.griddepcontrol_wait()
        count = load_count(ctrl) + Uint64(1)
        parity = Int32(count & Uint64(1))
        self.push(args, parity, head, tile, pusher, tid)
        cute.arch.griddepcontrol_launch_dependents()
        if pusher == 0:
            batch = args[10]
            for t in range(tile, (batch + ROWS - 1) // ROWS, tiles):
                self.merge(args, latent, weights, wb, parity, t, tid)
        cute.arch.sync_threads()
        if tid == 0:
            finish(ctrl, self.heads * tiles * pushers, count)

    @cute.jit
    def push(self, args, parity, head, tile, pusher, tid):
        """This CTA's share of every LSE, then of every row (16 B chunks), into
        slot ``[parity][rank][row][gh % H]`` of head ``gh``'s owner's mailbox: the
        owners weight the sources while the rows are in flight."""
        rows, lse, peers, lpeers = args[0], args[1], args[4], args[5]
        batch, rank, tiles, pushers = args[10], args[11], args[12], args[13]
        H, D, cap = self.heads, self.size, self.cap
        threads = H * tiles * pushers * THREADS
        me = ((pusher * tiles + tile) * H + head) * THREADS + tid
        for i in range(me, batch * D * H, threads):
            b, gh = i // (D * H), i % (D * H)
            value = lse[b, gh]
            bits = Uint32(float_bits(value))
            if value != value:
                bits = Uint32(CANONICAL_NAN)
            slot = ((parity * D + rank) * cap + b) * H + gh % H
            store_relaxed(lpeers[gh // H] + Int64(slot) * 4, (bits,))
        for i in range(me, batch * D * H * CHUNKS, threads):
            r, k = i // CHUNKS, i % CHUNKS
            b, gh = r // (D * H), r % (D * H)
            words = load_words(at(rows, b, gh, 8 * k), count=4, mutable=True)
            slot = ((parity * D + rank) * cap + b) * H + gh % H
            address = peers[gh // H] + (Int64(slot) * LATENT + 8 * k) * 2
            store_relaxed(address, tuple(no_neg_zero(w) for w in words))

    @cute.jit
    def merge(self, args, latent, weights, wb, parity, t, tid):
        """Rows ``16 t ..`` of this head: weights, polled merge, W_UV and gate."""
        box, lbox, gate, output, batch = args[2], args[3], args[7], args[8], args[10]
        D = self.size
        head, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        group, quarter = lane // 4, lane % 4
        t0 = t * ROWS
        col = head * VALUE + 32 * (warp // 4) + 8 * (warp % 4) + 2 * quarter
        gates = []
        if cutlass.const_expr(self.gated):
            for r in cutlass.range_constexpr(2):
                bits = Uint32(0)
                if t0 + group + 8 * r < batch:
                    bits = load_words(
                        at(gate, t0 + group + 8 * r, col), count=1, mutable=True
                    )[0]
                gates.append(unpack_bf16(bits))
        if t0 + warp < batch:  # warp r: the source weights of row t0 + r
            value = Float32(-math.inf)
            source = at(lbox, parity, min(lane, D - 1), t0 + warp, head)
            word = Uint32(LSE_SENT)
            if lane < D:
                word = load_words(source, count=1, volatile=True, word_type="u32")[0]
                while word == Uint32(LSE_SENT):
                    word = load_words(source, count=1, volatile=True, word_type="u32")[0]  # fmt: skip
                store_words(source, (Uint32(LSE_SENT),))
                value = bits_float(word)
                if value == Float32(math.inf):
                    value = Float32(-math.inf)
            # Triton's eight-lane reductions: lane s holds source s.
            top = value
            nan = Int32(value != value)
            for shift in cutlass.range_constexpr(2, -1, -1):
                top = max_f32(top, cute.arch.shuffle_sync_bfly(top, 1 << shift))
                nan = nan | cute.arch.shuffle_sync_bfly(nan, 1 << shift)
            if top == Float32(-math.inf):
                top = Float32(0)
            scaled = triton_exp(sub_rn(value, top))
            total = scaled
            for shift in cutlass.range_constexpr(2, -1, -1):
                total = add_rn(total, cute.arch.shuffle_sync_bfly(total, 1 << shift))
            weight = triton_divide(scaled, total)
            if nan != 0:
                weight = bits_float(Uint32(CANONICAL_NAN))
            if value == Float32(-math.inf):
                weight = -weight - Float32(1)  # empty: its partial adds 0 * weight
            if lane < D:
                weights[warp, lane] = weight
        cute.arch.sync_threads()
        chunk = tid % CHUNKS
        for half in cutlass.range_constexpr(2):  # rows r and r + 8, 16 B chunk
            r = tid // CHUNKS + 8 * half
            merged = zeros(8, Float32)
            if t0 + r < batch:
                sources = [
                    at(box, parity, s, t0 + r, head, 8 * chunk) for s in range(D)
                ]
                loads = [
                    load_words(s, count=4, volatile=True, word_type="u32")
                    for s in sources
                ]
                for s in cutlass.range_constexpr(D):
                    self.add_source(sources[s], loads[s], weights[r, s], merged)
            pairs = tuple(
                pack_pair(merged[2 * e], merged[2 * e + 1], policy=None)
                for e in range(4)
            )
            store_words(latent.iterator + r * (LATENT + 32) + 8 * chunk, pairs, shared=True)  # fmt: skip
        cute.arch.sync_threads()
        parts = zeros((2, 4), Float32)  # two accumulation chains
        for c in cutlass.range_constexpr(16):
            x, y = (
                load_words(
                    latent.iterator + r * (LATENT + 32) + 32 * c + 8 * quarter,
                    count=4,
                    shared=True,
                )
                for r in (group, group + 8)
            )
            for s in cutlass.range_constexpr(2):
                a = (x[2 * s], y[2 * s], x[2 * s + 1], y[2 * s + 1])
                b = (wb[c][2 * s], wb[c][2 * s + 1])
                mma_into(parts[s, None], a, b, bf16=True)
        acc = [parts[0, k] + parts[1, k] for k in range(4)]
        for r in cutlass.range_constexpr(2):
            row = t0 + group + 8 * r
            if row < batch:
                lo, hi = (Float32(BFloat16(acc[2 * r + e])) for e in range(2))
                if cutlass.const_expr(self.gated):
                    lo = lo * triton_sigmoid(gates[r][0])
                    hi = hi * triton_sigmoid(gates[r][1])
                store_words(at(output, row, col), (bf16x2(lo, hi),))
        cute.arch.sync_threads()  # the next tile rewrites the latent and weights

    @cute.jit
    def add_source(self, source, words, weight, merged):
        """Poll a source's 16 B chunk until published, re-arm it and add its eight
        values times ``weight`` (an empty source's ``-1 - weight``: zeros) to
        ``merged``, unfused, as the combine's ``acc + values * weight``."""
        while not published(words, sentinel=PAIR_SENT):
            words = load_words(source, count=4, volatile=True, word_type="u32")
        store_words(source, (Uint32(PAIR_SENT),) * 4)
        empty = weight < Float32(0)
        if empty:
            weight = -weight - Float32(1)
        for e in cutlass.range_constexpr(4):
            pair = unpack_bf16(words[e])
            for k in cutlass.range_constexpr(2):
                value = pair[k]
                if empty:
                    value = Float32(0)
                merged[2 * e + k] = add_rn(merged[2 * e + k], mul_rn(value, weight))


class DcpValueMailbox:
    """A KVP group's DcpValue state: two parities of BF16 partial rows ``[2, D, cap,
    H, 512]`` and FP32 LSEs ``[2, D, cap, H]`` in symmetric memory, sentinel-armed,
    their peers' addresses, and the call count. Creation is collective over
    ``group`` (eagerly, outside graph capture)."""

    def __init__(self, group: dist.ProcessGroup, device: torch.device, *, heads: int, cap: int) -> None:  # fmt: skip
        self.size, self.rank = dist.get_world_size(group), dist.get_rank(group)
        self.heads, self.cap, self.device = heads, cap, device
        shape = (2, self.size, cap, heads)
        with torch.cuda.device(device):
            self.partial = symm_mem.empty((*shape, LATENT), dtype=torch.bfloat16, device=device)  # fmt: skip
            self.lse = symm_mem.empty(shape, dtype=torch.float32, device=device)
            handles = [symm_mem.rendezvous(t, group=group) for t in (self.partial, self.lse)]  # fmt: skip
            i64 = dict(dtype=torch.int64, device=device)
            self.peers = torch.tensor(list(handles[0].buffer_ptrs), **i64)
            self.lse_peers = torch.tensor(list(handles[1].buffer_ptrs), **i64)
            self.handles = handles
            self.partial.view(torch.int32).fill_(PAIR_SENT - (1 << 32))
            self.lse.view(torch.int32).fill_(-1)
            # The call count, then the CTA ticket in the low half of word 1.
            self.ctrl = torch.zeros(2, dtype=torch.int64, device=device)
            self.sms = torch.cuda.get_device_properties(device).multi_processor_count
            torch.cuda.synchronize(device)
        # Every rank's mailbox is armed before any peer stores into it.
        dist.barrier(group=group)

    def grid(self, batch: int) -> tuple[int, int]:
        """(merge tiles, pushers) of a ``batch``-row call: at most one CTA per SM."""
        tiles = min(-(-batch // ROWS), max(1, self.sms // self.heads))
        merge = self.heads * tiles
        chunks = batch * self.size * self.heads * CHUNKS
        pushers = min(max(1, self.sms // merge), -(-chunks // (merge * THREADS)))
        return tiles, max(1, pushers)


@cache
def compile_dcp_value(heads: int, size: int, cap: int, gated: bool, arch: str):
    """DcpValue over runtime row counts and row strides."""

    def rows(rank, align):
        shape = tuple(cute.sym_int() for _ in range(rank))
        return fake(
            shape, (*(cute.sym_int(64) for _ in range(rank - 1)), 1), align=align
        )

    def lses():
        shape = (cute.sym_int(), cute.sym_int())
        return fake(shape, (cute.sym_int(), 1), torch.float32, 4)

    w_uv = fake((heads, LATENT, VALUE), (LATENT * VALUE, 1, LATENT))
    box = fake((2, size, cap, heads, LATENT))
    lbox = fake((2, size, cap, heads), None, torch.float32, 4)
    peers = fake((size,), None, torch.int64, 8)
    args = (rows(3, 16), lses(), box, lbox, peers, peers, w_uv, rows(2, 4), rows(2, 4))
    args += (Int64(0), Int32(0), Int32(0), Int32(1), Int32(1), make_fake_stream())
    return cute.compile(
        DcpValue(heads, size, cap, gated), *args, options=f"--gpu-arch={arch}"
    )


def dcp_value(rows, lse, mailbox: DcpValueMailbox, w_uv, gate, output, stream):
    """Merge every rank's ``rows`` / ``lse`` partials of this rank's heads and
    project them: ``output = bf16(merge @ w_uv)``, times ``sigmoid(gate)`` when given.

    ``rows`` is ``[batch, >= D * H, 512]`` BF16 with unit inner stride and 16-byte
    aligned rows, ``lse`` ``[batch, >= D * H]`` FP32 natural-log with unit inner
    stride, ``w_uv`` ``[H, 512, 128]`` with the latent axis contiguous, ``gate``
    ``[batch, H * 128]`` (or None) and ``output`` ``[batch, H * 128]`` BF16 with unit
    inner stride and 4-byte aligned rows. ``batch`` is at most the mailbox capacity.
    Every rank of the group makes the same calls."""
    batch = rows.shape[0]
    tiles, pushers = mailbox.grid(batch)
    code = compile_dcp_value(
        mailbox.heads,
        mailbox.size,
        mailbox.cap,
        gate is not None,
        architecture(rows.device),
    )
    # An ungated launch passes the output for the gate, which it never reads.
    code(
        dynamic_view(rows),
        dynamic_view(lse, 4),
        to_cute(mailbox.partial),
        to_cute(mailbox.lse, 4),
        to_cute(mailbox.peers, 8),
        to_cute(mailbox.lse_peers, 8),
        to_cute(w_uv),
        dynamic_view(output if gate is None else gate, 4),
        dynamic_view(output, 4),
        Int64(mailbox.ctrl.data_ptr()),
        Int32(batch),
        Int32(mailbox.rank),
        Int32(tiles),
        Int32(pushers),
        stream,
    )
