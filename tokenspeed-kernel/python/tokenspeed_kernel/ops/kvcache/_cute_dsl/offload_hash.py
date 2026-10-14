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

"""Request-private selection hash and stable batch LRU (ordinary CTAs)."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import Int32, Int64, Uint32, cute, utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def fail(*, loc=None, ip=None):
    # An unconditional device trap survives release builds. No CPU polling.
    llvm.inline_asm(
        None,
        [],
        "trap;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def bucket(key: Int32, size: cutlass.Constexpr):
    x = Uint32(key)
    x = (x ^ (x >> 16)) * Uint32(0x7FEB352D)
    x = (x ^ (x >> 15)) * Uint32(0x846CA68B)
    return Int32((x ^ (x >> 16)) & Uint32(size - 1))


@cute.jit
def lookup(table: cute.Tensor, key: Int32, size: cutlass.Constexpr):
    pos = bucket(key, size)
    count = Int32(0)
    found = Int32(-1)
    active = key > 0
    while active and count < size:
        old = table[pos]
        if old == key:
            found = pos
        active = (old != key) and (old != -1)
        pos = (pos + 1) & (size - 1)
        count += 1
    if active:
        fail()
    return found


@cute.jit
def block_prefix(value: Int32, scratch: cute.Tensor):
    """Inclusive stable prefix for 256 threads; returns prefix and total."""
    tid = cute.arch.thread_idx()[0]
    lane = tid % 32
    warp = tid // 32
    x = value
    for shift in cutlass.range_constexpr(5):
        prev = cute.arch.shuffle_sync_up(x, 1 << shift)
        if lane >= (1 << shift):
            x += prev
    if lane == 31:
        scratch[warp] = x
    cute.arch.sync_threads()
    if warp == 0:
        y = Int32(0)
        if lane < 8:
            y = scratch[lane]
        for shift in cutlass.range_constexpr(3):
            prev = cute.arch.shuffle_sync_up(y, 1 << shift)
            if lane >= (1 << shift):
                y += prev
        if lane < 8:
            scratch[lane] = y
    cute.arch.sync_threads()
    prefix = x
    if warp > 0:
        prefix += scratch[warp - 1]
    total = scratch[7]
    # No caller may reuse the shared counters before every lane has read them.
    cute.arch.sync_threads()
    return prefix, total


class HashResolve:
    def __init__(self, hot, stride, queries, width, table_size, shared):
        self.h, self.s, self.q, self.n = hot, stride, queries, width
        self.t, self.shared = table_size, shared

    @cute.jit
    def __call__(self, ptrs: tuple, batch: Int32, stream: cuda.CUstream):
        arrays = tuple(cute.make_tensor(p, cute.make_layout(2147483647)) for p in ptrs)
        self.hash_resolve_lru(arrays).launch(
            grid=(batch, 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def hash_resolve_lru(self, arrays: tuple):
        (
            selected,
            rids,
            keys,
            current,
            miss_ids,
            miss_dst,
            output,
            lru,
            order,
            free_counts,
            miss_counts,
            dest,
            global_keys,
            global_owners,
        ) = arrays
        b = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        r = Int32(rids[b])
        smem = utils.SmemAllocator()
        scan = smem.allocate_tensor(Int32, cute.make_layout(8), byte_alignment=16)
        if cutlass.const_expr(self.shared):
            table = smem.allocate_tensor(
                Int32, cute.make_layout(self.t), byte_alignment=16
            )
            owners = smem.allocate_tensor(
                Int32, cute.make_layout(self.t), byte_alignment=16
            )
        else:
            table = cute.make_tensor(
                global_keys.iterator + b * self.t, cute.make_layout(self.t)
            )
            owners = cute.make_tensor(
                global_owners.iterator + b * self.t, cute.make_layout(self.t)
            )
        for i in range(tid, self.n, 256):
            miss_ids[b * self.n + i] = -1
            miss_dst[b * self.n + i] = -1
            output[b * self.n + i] = -1
            dest[b * self.n + i] = 2147483647
        if tid == 0:
            free_counts[b] = 0
            miss_counts[b] = 0
        if r > 0:
            for i in range(tid, self.t, 256):
                table[i] = -1
                owners[i] = 2147483647
            cute.arch.sync_threads()
            for i in range(tid, self.n, 256):
                g = selected[b * self.n + i]
                if g > 0:
                    pos = bucket(g, self.t)
                    probes = Int32(0)
                    active = True
                    while active and probes < self.t:
                        old = cute.arch.atomic_cas(
                            table.iterator + pos,
                            cmp=Int32(-1),
                            val=g,
                            sem="relaxed",
                            scope="cta",
                        )
                        if (old == -1) or (old == g):
                            cute.arch.atomic_min(
                                owners.iterator + pos,
                                Int32(i),
                                sem="relaxed",
                                scope="cta",
                            )
                            active = False
                        else:
                            pos = (pos + 1) & (self.t - 1)
                        probes += 1
                    if active:
                        fail()
            cute.arch.sync_threads()
            free = Int32(0)
            # Keep exactly the existing reverse-free / forward-protected layout.
            for start in range(0, self.h, 256):
                j = start + tid
                slot = Int32(0)
                protect = False
                if j < self.h:
                    slot = lru[r * self.h + j]
                    tag = keys[r * self.s + slot]
                    pos = lookup(table, tag, self.t)
                    protect = pos >= 0
                    if pos >= 0:
                        owner = owners[pos]
                        cute.arch.atomic_min(
                            dest.iterator + b * self.n + owner,
                            r * self.s + slot,
                            sem="relaxed",
                            scope="cta",
                        )
                    for qi in cutlass.range_constexpr(self.q):
                        protect = protect or (
                            current[b * self.q + qi] == r * self.s + slot
                        )
                is_free = Int32((j < self.h) and not protect)
                rank, total = block_prefix(is_free, scan)
                if j < self.h:
                    if protect:
                        order[b * self.h + j - free - rank] = slot
                    else:
                        order[b * self.h + self.h - 1 - free - (rank - 1)] = slot
                free += total
            for slot in range(self.h + tid, self.s, 256):
                tag = keys[r * self.s + slot]
                pos = lookup(table, tag, self.t)
                if pos >= 0:
                    cute.arch.atomic_min(
                        dest.iterator + b * self.n + owners[pos],
                        r * self.s + slot,
                        sem="relaxed",
                        scope="cta",
                    )
            cute.arch.sync_threads()
            misses = Int32(0)
            for start in range(0, self.n, 256):
                i = start + tid
                g = Int32(-1)
                missing = False
                if i < self.n:
                    g = selected[b * self.n + i]
                    if g > 0:
                        pos = lookup(table, g, self.t)
                        missing = (owners[pos] == i) and (
                            dest[b * self.n + i] == 2147483647
                        )
                rank, total = block_prefix(Int32(missing), scan)
                if missing:
                    miss_ids[b * self.n + i] = g
                    miss_dst[b * self.n + i] = misses + rank - 1
                misses += total
            # Validate the whole union before publishing any tag/LRU changes.
            if misses > free:
                fail()
            if tid == 0:
                free_counts[b] = free
                miss_counts[b] = misses
            cute.arch.sync_threads()
            for i in range(tid, self.n, 256):
                g = miss_ids[b * self.n + i]
                if g > 0:
                    ordinal = miss_dst[b * self.n + i]
                    slot = order[b * self.h + self.h - 1 - ordinal]
                    d = r * self.s + slot
                    keys[d] = g
                    miss_dst[b * self.n + i] = d
                    dest[b * self.n + i] = d
            cute.arch.sync_threads()
            for i in range(tid, self.n, 256):
                g = selected[b * self.n + i]
                if g > 0:
                    pos = lookup(table, g, self.t)
                    output[b * self.n + i] = dest[b * self.n + owners[pos]]
            for j in range(tid, self.h, 256):
                source = Int32(0)
                if j < free - misses:
                    source = self.h - 1 - (j + misses)
                elif j < free:
                    source = self.h - 1 - (j - (free - misses))
                else:
                    source = j - free
                lru[r * self.h + j] = order[b * self.h + source]


class CopyRows:
    def __init__(self, row_units, writeback):
        self.row, self.writeback = row_units, writeback

    @cute.jit
    def __call__(
        self,
        host: cute.Pointer,
        device: cute.Pointer,
        src: cute.Pointer,
        dst: cute.Pointer,
        n: Int32,
        stream: cuda.CUstream,
    ):
        args = tuple(
            cute.make_tensor(p, cute.make_layout(9223372036854775807))
            for p in (host, device, src, dst)
        )
        self.copy_rows(args).launch(grid=(n, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def copy_rows(self, args: tuple):
        host, device, src, dst = args
        row = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        g, d = src[row], dst[row]
        if (g > 0) and (d >= 0):
            for col in range(tid, self.row, 128):
                h = Int64(g) * self.row + col
                v = Int64(d) * self.row + col
                if cutlass.const_expr(self.writeback):
                    host[h] = device[v]
                else:
                    device[v] = host[h]


class RowMetadata:
    def __init__(self, mode, hot, stride, queries, cyclic):
        self.mode, self.h, self.s, self.q, self.c = mode, hot, stride, queries, cyclic

    @cute.jit
    def __call__(self, ptrs: tuple, n: Int32, extra: tuple, stream: cuda.CUstream):
        arrays = tuple(cute.make_tensor(p, cute.make_layout(2147483647)) for p in ptrs)
        self.row_metadata(arrays, n, extra).launch(
            grid=((n + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def row_metadata(self, arrays: tuple, n: Int32, extra: tuple):
        i = cute.arch.block_idx()[0] * 256 + cute.arch.thread_idx()[0]
        if i < n:
            if cutlass.const_expr(self.mode == "current"):
                rids, positions, full, keys, out = arrays
                r, p, g = Int32(rids[i // self.q]), Int32(positions[i]), Int32(full[i])
                d = Int32(0)
                if cutlass.const_expr(self.c > 0):
                    d = r * self.s + self.h + p % self.c
                else:
                    if p < self.h:
                        d = r * self.s + p
                    else:
                        d = r * self.s + self.h + i % self.q
                if g > 0:
                    keys[d] = g
                    out[i] = d
                else:
                    out[i] = 0
            elif cutlass.const_expr(self.mode == "accepted"):
                full, accepted, out = arrays
                g = Int32(-1)
                if i % self.q < accepted[i // self.q]:
                    g = full[i]
                out[i] = g
            elif cutlass.const_expr(self.mode == "reset_all"):
                arrays[0][i] = i % self.h
            elif cutlass.const_expr(self.mode == "reset"):
                rids, lru = arrays
                lru[rids[i // self.h] * self.h + i % self.h] = i % self.h
            elif cutlass.const_expr(self.mode == "mark_seeded"):
                rids, seeded = arrays
                r = rids[i]
                if r > 0:
                    seeded[r] = 1
            elif cutlass.const_expr(self.mode == "seed_locations"):
                table, positions, requests, history, hot_rows = arrays
                table_stride, table_width, page_size = extra
                b, index = i // (self.h + self.c), i % (self.h + self.c)
                length = max(Int32(positions[b * self.q]), 0)
                r = Int32(requests[b])
                token, slot = Int32(index), Int32(index)
                valid = False
                if cutlass.const_expr(self.c > 0):
                    tail = min(length, self.c - self.q)
                    ordinary = Int32(0)
                    if length < self.h:
                        ordinary = length - tail
                    if index >= self.h:
                        token = length - tail + index - self.h
                        slot = self.h + token % self.c
                        valid = index - self.h < tail
                    else:
                        valid = index < ordinary
                else:
                    valid = (length < self.h) and (token < length)
                page_index = token // page_size
                full = Int32(-1)
                if valid and (page_index >= 0) and (page_index < table_width):
                    page = table[b * table_stride + page_index]
                    if page > 0:
                        full = Int32(page * page_size + token % page_size)
                history[i] = full
                d = Int32(-1)
                if full > 0:
                    d = r * self.s + slot
                hot_rows[i] = d


class SeedRows:
    def __init__(self, row_units):
        self.row = row_units

    @cute.jit
    def __call__(self, ptrs: tuple, batch: Int32, width: Int32, stream: cuda.CUstream):
        arrays = tuple(
            cute.make_tensor(p, cute.make_layout(9223372036854775807)) for p in ptrs
        )
        self.seed_rows(arrays, width).launch(
            grid=(batch * width, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def seed_rows(self, arrays: tuple, width: Int32):
        host, device, keys, seeded, rids, full, hot = arrays
        i = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        r, g, d = rids[i // width], full[i], hot[i]
        if (r > 0) and (seeded[r] == 0) and (g > 0) and (d >= 0):
            for col in range(tid, self.row, 128):
                device[Int64(d) * self.row + col] = host[Int64(g) * self.row + col]
            if tid == 0:
                keys[d] = g
