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

"""DeepSelect-style FP32 row top-k index selection, written in CuTe DSL.

This is a from-scratch CuTe DSL rendition of the DeepSeek DeepSelect algorithm
(https://github.com/deepseek-ai/DeepSelect) for Hopper, where the upstream
package ships no cubin. One CTA (or one thread-block cluster) selects the
``topk`` largest entries of one FP32 row:

1. **Init window.** The first round of the row -- the row's last (at most)
   4096 entries plus the first pseudo-randomly ordered 512-entry segments --
   is loaded into registers and its exact top-k is found with a 256-bucket
   radix select over the order-preserving integer image of each value. The
   k-th largest value becomes the running threshold.
2. **Scan.** The remaining segments are visited in a fixed pseudo-random
   order. Every entry is read once; only entries above the threshold are
   appended, as ``(index, value)`` pairs, to a shared-memory candidate buffer.
3. **Compact.** Once ``RECONSTRUCT_THRESHOLD`` candidates have accumulated
   (and once more at the end) a radix select over the survivors plus the new
   candidates reduces them back to k pairs and raises the threshold.

With ``cluster_size > 1`` the segments of a row are split across the CTAs of
a cluster; every CTA runs the three steps on its share, the non-leader CTAs
ship their k survivors to the leader over distributed shared memory, and the
leader runs one more compaction over the gathered candidates. This mirrors
DeepSelect's cluster variant and is what makes small batches with long rows
fast on Hopper, whose GPCs accept portable 8-CTA clusters (H20 included).

Departures from the CUDA implementation: the main rounds stream through a
plain three-stage ``cp.async.bulk`` shared-memory ring instead of
TMA-swizzled buffers (only the init window is loaded straight into
registers), and the cluster gather uses plain distributed shared-memory
stores under a cluster barrier instead of ``st.async`` transactions. NaN
entries become the ``(-1, -inf)`` placeholder where the init window and
shortcut rows are formed, and the scan's ``value > threshold`` filter
discards them elsewhere, so a NaN entry is never selected.
"""

from __future__ import annotations

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.utils.smem_allocator import SmemAllocator

# Geometry shared with DeepSelect's FP32 configuration.
NUM_THREADS = 512
NUM_WARPS = NUM_THREADS // 32
ELEMS_PER_THREAD = 16
ROUND_ELEMS = NUM_THREADS * ELEMS_PER_THREAD  # 8192 entries per round
SEG_ELEMS = 512  # one warp visits one segment per round
SEGS_PER_ROUND = ROUND_ELEMS // SEG_ELEMS
TAIL_ELEMS = 4096  # the row suffix that is always visited in order
TAIL_SEGS = TAIL_ELEMS // SEG_ELEMS
# Candidates accumulated before a compaction. DeepSelect uses 4096; the 64
# fewer slots keep the max_topk=2048 configuration within Hopper's 227 KiB of
# dynamic shared memory (survivors + candidates + ring + histograms).
RECONSTRUCT_THRESHOLD = 4032
# The check runs after a round, so up to a full round of hits may follow.
INCOMING_SLOTS = RECONSTRUCT_THRESHOLD + ROUND_ELEMS
RADIX_BUCKETS = 256
# Keys outside the bucket under refinement (or padding slots) are counted in a
# sink slot past the real buckets, which keeps the histogram passes branch-free.
RADIX_SINK = RADIX_BUCKETS
RADIX_SLOTS = RADIX_BUCKETS + 4
# Three rotating histogram buffers let a pass clear the buffer two passes
# ahead while the other warps may still be reading the previous one.
NUM_HISTS = 3
# Shared-memory ring for the main rounds: NUM_STAGES - 1 rounds are in flight
# while one is being filtered.
NUM_STAGES = 3
SEG_BYTES = SEG_ELEMS * 4
SUPPORTED_MAX_TOPK = (512, 1024, 2048)
SUPPORTED_CLUSTER_SIZES = (1, 2, 4, 8)

# Pseudo-random segment permutation, ``perm(i) = (i * (P mod n) + A) mod n``.
PERM_ADD_BASE = 0x22262226
PERM_MUL_PRIME = 0xB559EB75

NEG_INF_BITS = 0xFF800000
# A pair packs ``value_bits << 32 | index``; the placeholder is -inf at index -1.
PLACEHOLDER_PAIR = (NEG_INF_BITS << 32) | 0xFFFFFFFF

assert SEGS_PER_ROUND == NUM_WARPS
assert TAIL_SEGS <= SEGS_PER_ROUND


@cute.jit
def _f32_bits(x):
    return cutlass.Uint32(llvm.bitcast(cutlass.Uint32.mlir_type, x.ir_value()))


@cute.jit
def _bits_f32(u):
    return cutlass.Float32(llvm.bitcast(cutlass.Float32.mlir_type, u.ir_value()))


@cute.jit
def _distort(bits):
    """Map FP32 bit patterns to unsigned integers that sort like the floats.

    A positive NaN would map above +inf, so NaN bits must be sanitized with
    :func:`_sanitized_bits` before they reach a key or a pair.
    """
    sign_fill = cutlass.Uint32(cutlass.Int32(bits) >> cutlass.Int32(31))
    return bits ^ (sign_fill | cutlass.Uint32(0x80000000))


@cute.jit
def _sanitized_bits(val):
    """Float bits of ``val`` with NaN mapped to -inf, the placeholder score.

    Applied wherever init-window values turn into radix keys or pairs, so a
    NaN entry competes exactly like a padding slot and is never selected."""
    bits = _f32_bits(val)
    if (bits & cutlass.Uint32(0x7FFFFFFF)) > cutlass.Uint32(0x7F800000):
        bits = cutlass.Uint32(NEG_INF_BITS)
    return bits


@cute.jit
def _undistort(key):
    sign_fill = cutlass.Uint32(~(cutlass.Int32(key) >> cutlass.Int32(31)))
    return key ^ (sign_fill | cutlass.Uint32(0x80000000))


@cute.jit
def _pack_pair(index, value_bits):
    return (cutlass.Uint64(value_bits) << cutlass.Uint64(32)) | cutlass.Uint64(
        cutlass.Uint32(index)
    )


@cute.jit
def _pair_index(pair):
    return cutlass.Int32(cutlass.Uint32(pair & cutlass.Uint64(0xFFFFFFFF)))


@cute.jit
def _pair_value(pair):
    return _bits_f32(cutlass.Uint32(pair >> cutlass.Uint64(32)))


@cute.jit
def _lowest_set_bit(mask):
    """Index of the lowest set bit of a non-zero Int32."""
    return cutlass.Int32(cute.arch.clz(cute.arch.brev(mask)))


@cute.jit
def _advance_perm(state, stride, perm_len):
    """``(state + stride) mod perm_len`` for ``state, stride < perm_len``."""
    nxt = state + stride
    if nxt >= perm_len:
        nxt = nxt - perm_len
    return nxt


@cute.jit
def _warp_inclusive_sum(value, lane):
    """Inclusive prefix sum of an Int32 across the 32 lanes of a warp."""
    total = value
    for shift in cutlass.range_constexpr(5):
        offset = 1 << shift
        up = cute.arch.shuffle_sync_up(total, offset)
        if lane >= offset:
            total = total + up
    return total


@cute.jit
def _permuted_segment(linear, perm_len, perm_mul):
    """Segment visited at position ``linear`` of the pseudo-random order."""
    return (
        (cutlass.Uint32(linear) % perm_len) * perm_mul + cutlass.Uint32(PERM_ADD_BASE)
    ) % perm_len


@cute.jit
def _map_shared_cluster(smem_ptr, rank):
    """Address of ``smem_ptr`` inside CTA ``rank`` of the cluster (DSMEM)."""
    return cute.arch.inline_ptx(
        "mapa.shared::cluster.u32 {$w0}, {$r0}, {$r1};",
        write_only_types=[cutlass.Int32],
        read_only_args=[cutlass.Int32(smem_ptr.toint()), rank],
    )


@cute.jit
def _store_shared_cluster_u64(remote_addr, value):
    cute.arch.inline_ptx(
        "st.shared::cluster.b64 [{$r0}], {$r1};",
        read_only_args=[remote_addr, value],
    )


@cute.jit
def _bulk_copy_g2s(dst_smem_addr, src_gmem_addr, num_bytes, mbar_addr):
    """One-dimensional TMA copy of ``num_bytes`` (a multiple of 16) into smem.

    Completion is signalled to the mbarrier at ``mbar_addr`` through its
    transaction count.
    """
    cute.arch.inline_ptx(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[{$r0}], [{$r1}], {$r2}, [{$r3}];",
        read_only_args=[dst_smem_addr, src_gmem_addr, num_bytes, mbar_addr],
    )


@cute.jit
def _find_pivot(hist, k, lane):
    """Locate the bucket holding the k-th largest key; run by one warp.

    Returns, uniformly across the warp, the bucket, how many keys of that
    bucket belong to the selection, and whether that is the whole bucket
    (so the refinement can stop). Lane ``l`` owns buckets ``8l .. 8l+7``;
    the search is branch-free and the three results are combined across
    lanes with single-instruction warp reductions. One warp does this while
    the others wait: the chain is latency-bound, and sixteen redundant
    copies would contend for issue slots instead.
    """
    counts = cute.make_rmem_tensor((8,), cutlass.Int32)
    local = cutlass.Int32(0)
    for j in cutlass.range_constexpr(8):
        counts[j] = hist[lane * 8 + j]
        local = local + counts[j]
    inclusive = _warp_inclusive_sum(local, lane)
    total = cute.arch.shuffle_sync(inclusive, 31)
    # suffix[j] counts the keys in buckets >= lane * 8 + j, i.e. values at
    # least as large as that bucket's floor; it decreases with j.
    suffix = cute.make_rmem_tensor((9,), cutlass.Int32)
    suffix[8] = total - inclusive
    for j in cutlass.range_constexpr(7, -1, -1):
        suffix[j] = suffix[j + 1] + counts[j]
    # The pivot bucket is the last one whose suffix still reaches k. Across
    # the whole warp: the keys above it are the largest suffix below k, the
    # keys from it on are the smallest suffix at or above k.
    above = cutlass.Int32(0)
    including = cutlass.Int32(0x7FFFFFFF)
    for j in cutlass.range_constexpr(9):
        below_k = suffix[j] < k
        above = cutlass.max(above, cutlass.Int32(below_k) * suffix[j])
        including = cutlass.min(
            including, suffix[j] + cutlass.Int32(below_k) * cutlass.Int32(0x3FFFFFFF)
        )
    bucket = cutlass.Int32(-1)
    if (suffix[8] < k) & (k <= suffix[0]):
        bucket = lane * 8
        for j in cutlass.range_constexpr(1, 8):
            bucket = bucket + cutlass.Int32(suffix[j] >= k)
    bucket = cute.arch.warp_redux_sync(bucket, "max")
    above = cute.arch.warp_redux_sync(above, "max")
    including = cute.arch.warp_redux_sync(including, "min")
    whole = cutlass.Int32(k == including)
    return bucket, k - above, whole


@cute.jit
def _radix_bucket(key, prefix, pass_idx):
    """Histogram slot of ``key`` in pass ``pass_idx``: its next byte when the
    leading bytes equal ``prefix``, otherwise the sink."""
    bucket_shift = cutlass.Uint32(24 - 8 * pass_idx)
    if cutlass.const_expr(pass_idx == 0):
        return cutlass.Int32(key >> bucket_shift)
    else:
        # The leading bytes minus the prefix leave the digit for matching
        # keys and an out-of-range number otherwise.
        relative = (key >> bucket_shift) - (prefix << cutlass.Uint32(8))
        return cutlass.Int32(cutlass.min(relative, cutlass.Uint32(RADIX_SINK)))


@cute.jit
def _key_of_pair(pair):
    return _distort(cutlass.Uint32(pair >> cutlass.Uint64(32)))


@cute.jit
def _selection_prefix(cnt_gt, cnt_eq, k, scratch, warp, lane):
    """CTA-wide exclusive prefix of per-thread (greater, equal) counts.

    Both counts are packed into one word (each stays below 2^16), scanned
    across the warp with shuffles, and carried across warps through
    ``scratch`` (NUM_WARPS ints) and one barrier. Returns the slot at which
    this thread starts writing and how many pivot-equal pairs it may still
    take.
    """
    packed = (cnt_gt << cutlass.Int32(16)) | cnt_eq
    inclusive = _warp_inclusive_sum(packed, lane)
    warp_total = cute.arch.shuffle_sync(inclusive, 31)
    if lane == 0:
        scratch[warp] = warp_total
    cute.arch.barrier()
    entry = scratch[lane & (NUM_WARPS - 1)]
    if lane >= NUM_WARPS:
        entry = cutlass.Int32(0)
    total = cute.arch.warp_redux_sync(entry, "add")
    if lane >= warp:
        entry = cutlass.Int32(0)
    before = cute.arch.warp_redux_sync(entry, "add") + inclusive - packed
    total_gt = total >> cutlass.Int32(16)
    quota_total = k - total_gt
    gt_before = before >> cutlass.Int32(16)
    eq_before = before & cutlass.Int32(0xFFFF)
    eq_quota = cutlass.Int32(0)
    if quota_total > eq_before:
        eq_quota = quota_total - eq_before
    start = gt_before + cutlass.min(eq_before, quota_total)
    return start, eq_quota


@cute.jit
def _emit_pair(pair, pivot_key, dst, eq_quota, out, out_base):
    """Append ``pair`` at ``out[out_base + dst]`` if its key beats the pivot,
    or equals it while this thread still has quota; advance ``dst``."""
    key = _key_of_pair(pair)
    take = key > pivot_key
    if (key == pivot_key) & (eq_quota > 0):
        eq_quota = eq_quota - 1
        take = cutlass.Boolean(True)
    if take:
        out[out_base + dst] = pair
        dst = dst + 1
    return dst, eq_quota


# The set a compaction works on is either the init window (this thread's 16
# loaded values, in registers) or the survivor buffer followed by n_incoming
# candidates (in shared memory). The three walks below are specialised on the
# compile-time flag ``from_registers``; the survivor part is a static number
# of slots per thread and the candidate part a dynamic loop, so the work
# scales with the candidate count. (A Python object standing for the source
# cannot be carried through the DSL's dynamic branches, hence flat arguments.)


@cute.jit
def _set_histogram(
    from_registers,
    vals,
    survivors,
    cur,
    incoming,
    n_incoming,
    max_topk,
    tidx,
    hist_buf,
    prefix,
    pass_idx,
):
    one = cutlass.Int32(1)
    if cutlass.const_expr(from_registers):
        for t in cutlass.range_constexpr(ELEMS_PER_THREAD):
            key = _distort(_sanitized_bits(vals[t]))
            cute.arch.atomic_add(hist_buf + _radix_bucket(key, prefix, pass_idx), one)
    else:
        for t in cutlass.range_constexpr(max_topk // NUM_THREADS):
            key = _key_of_pair(survivors[cur * max_topk + tidx + t * NUM_THREADS])
            cute.arch.atomic_add(hist_buf + _radix_bucket(key, prefix, pass_idx), one)
        for i in range(tidx, n_incoming, NUM_THREADS):
            key = _key_of_pair(incoming[i])
            cute.arch.atomic_add(hist_buf + _radix_bucket(key, prefix, pass_idx), one)


@cute.jit
def _set_census(
    from_registers,
    vals,
    survivors,
    cur,
    incoming,
    n_incoming,
    max_topk,
    tidx,
    pivot_key,
):
    """Count this thread's keys above and equal to the pivot."""
    cnt_gt = cutlass.Int32(0)
    cnt_eq = cutlass.Int32(0)
    if cutlass.const_expr(from_registers):
        for t in cutlass.range_constexpr(ELEMS_PER_THREAD):
            key = _distort(_sanitized_bits(vals[t]))
            cnt_gt = cnt_gt + cutlass.Int32(key > pivot_key)
            cnt_eq = cnt_eq + cutlass.Int32(key == pivot_key)
    else:
        for t in cutlass.range_constexpr(max_topk // NUM_THREADS):
            key = _key_of_pair(survivors[cur * max_topk + tidx + t * NUM_THREADS])
            cnt_gt = cnt_gt + cutlass.Int32(key > pivot_key)
            cnt_eq = cnt_eq + cutlass.Int32(key == pivot_key)
        for i in range(tidx, n_incoming, NUM_THREADS):
            key = _key_of_pair(incoming[i])
            cnt_gt = cnt_gt + cutlass.Int32(key > pivot_key)
            cnt_eq = cnt_eq + cutlass.Int32(key == pivot_key)
    return cnt_gt, cnt_eq


@cute.jit
def _window_pair(vals, t, seg_base, lane, end, active):
    """Init-window entry ``t`` of this lane as a pair: it sits at chunk
    ``t // 4``, element ``t % 4`` of the segment; entries past the row, of
    an idle warp, or holding NaN carry index -1."""
    index = seg_base + cutlass.Int32(128 * (t // 4)) + lane * 4 + cutlass.Int32(t % 4)
    if (index >= end) | (~active) | (vals[t] != vals[t]):
        index = cutlass.Int32(-1)
    return _pack_pair(index, _sanitized_bits(vals[t]))


@cute.jit
def _set_write_back(
    from_registers,
    vals,
    seg_base,
    lane,
    end,
    active,
    survivors,
    cur,
    incoming,
    n_incoming,
    max_topk,
    tidx,
    pivot_key,
    dst,
    eq_quota,
    out_base,
):
    if cutlass.const_expr(from_registers):
        for t in cutlass.range_constexpr(ELEMS_PER_THREAD):
            pair = _window_pair(vals, t, seg_base, lane, end, active)
            dst, eq_quota = _emit_pair(
                pair, pivot_key, dst, eq_quota, survivors, out_base
            )
    else:
        for t in cutlass.range_constexpr(max_topk // NUM_THREADS):
            pair = survivors[cur * max_topk + tidx + t * NUM_THREADS]
            dst, eq_quota = _emit_pair(
                pair, pivot_key, dst, eq_quota, survivors, out_base
            )
        for i in range(tidx, n_incoming, NUM_THREADS):
            pair = incoming[i]
            dst, eq_quota = _emit_pair(
                pair, pivot_key, dst, eq_quota, survivors, out_base
            )


class DeepSelectTopK:
    """One compiled configuration of the DeepSelect-style selector.

    Args:
        max_topk: Largest ``topk`` this instance serves; sizes the survivor
            buffers. One of ``SUPPORTED_MAX_TOPK``.
        cluster_size: CTAs cooperating on one row. One of
            ``SUPPORTED_CLUSTER_SIZES``; the gathered survivors of the
            non-leader CTAs must fit the candidate buffer.
    """

    def __init__(self, max_topk: int, cluster_size: int):
        if max_topk not in SUPPORTED_MAX_TOPK:
            raise ValueError(f"max_topk must be one of {SUPPORTED_MAX_TOPK}")
        if cluster_size not in SUPPORTED_CLUSTER_SIZES:
            raise ValueError(f"cluster_size must be one of {SUPPORTED_CLUSTER_SIZES}")
        if (cluster_size - 1) * max_topk > INCOMING_SLOTS:
            raise ValueError(
                f"cluster_size {cluster_size} cannot gather {max_topk} survivors per CTA"
            )
        # A CTA only drops candidates once it has run a main round, and every
        # such CTA had a full init window of at least TAIL_ELEMS + 1 real
        # entries, so its survivor buffer already held max_topk real pairs.
        if max_topk > TAIL_ELEMS:
            raise ValueError(f"max_topk must not exceed {TAIL_ELEMS}")
        self.max_topk = max_topk
        self.cluster_size = cluster_size
        assert max_topk % NUM_THREADS == 0

    # ------------------------------------------------------------------
    # Radix select over a CTA-wide set of (index, value) pairs
    # ------------------------------------------------------------------
    @cute.jit
    def _radix_select(
        self,
        from_registers,
        vals,
        seg_base,
        end,
        active,
        survivors,
        cur,
        incoming,
        n_incoming,
        k,
        out_base,
        hist,
        tidx,
        warp,
        lane,
    ):
        """Write the k best pairs of the set to ``survivors[out_base:]``.

        The set is the init window (``from_registers``: ``vals`` plus the
        positions derived from ``seg_base``) or ``survivors[cur]`` followed
        by ``incoming[:n_incoming]``. Returns the k-th value's float bits.
        Ends with a CTA barrier.
        """
        max_topk = cutlass.const_expr(self.max_topk)
        for i in range(tidx, 2 * RADIX_SLOTS, NUM_THREADS):
            hist[i] = cutlass.Int32(0)  # buffers 0 and 1; buffer 2 is cleared by pass 0
        cute.arch.barrier()

        # Pass 0 buckets the top byte; up to three refining passes bucket the
        # next byte of the keys sharing the prefix found so far. Non-matching
        # keys go to the sink, so no pass branches per key. Warp 0 locates the
        # pivot bucket and publishes it through ``scalars``; the histogram
        # written two passes later is cleared meanwhile.
        scalars = hist.iterator + NUM_HISTS * RADIX_SLOTS + NUM_WARPS
        prefix = cutlass.Uint32(0)
        remaining = k
        whole = cutlass.Int32(0)
        passes = cutlass.Int32(0)
        for pass_idx in cutlass.range_constexpr(4):
            if whole == 0:
                buf = pass_idx % NUM_HISTS
                hist_buf = hist.iterator + buf * RADIX_SLOTS
                _set_histogram(
                    from_registers,
                    vals,
                    survivors,
                    cur,
                    incoming,
                    n_incoming,
                    max_topk,
                    tidx,
                    hist_buf,
                    prefix,
                    pass_idx,
                )
                if cutlass.const_expr(pass_idx + 2 < 4):
                    for i in range(tidx, RADIX_SLOTS, NUM_THREADS):
                        hist[((pass_idx + 2) % NUM_HISTS) * RADIX_SLOTS + i] = (
                            cutlass.Int32(0)
                        )
                cute.arch.barrier()
                if warp == 0:
                    bucket, remaining, whole = _find_pivot(
                        cute.make_tensor(hist_buf, cute.make_layout((RADIX_BUCKETS,))),
                        remaining,
                        lane,
                    )
                    if lane == 0:
                        scalars[0] = bucket
                        scalars[1] = remaining
                        scalars[2] = whole
                cute.arch.barrier()
                prefix = (prefix << cutlass.Uint32(8)) | cutlass.Uint32(scalars[0])
                remaining = scalars[1]
                whole = scalars[2]
                passes = passes + 1
        # The pivot is the floor of the final bucket: the smallest key that
        # starts with ``prefix``. Keys above it are selected; keys equal to it
        # fill the rest of the k slots (all of them when the search stopped on
        # a whole bucket, since then above + bucket == k).
        pivot_key = prefix << cutlass.Uint32(cutlass.Int32(32) - passes * 8)
        cnt_gt, cnt_eq = _set_census(
            from_registers,
            vals,
            survivors,
            cur,
            incoming,
            n_incoming,
            max_topk,
            tidx,
            pivot_key,
        )
        dst, eq_quota = _selection_prefix(
            cnt_gt, cnt_eq, k, hist.iterator + NUM_HISTS * RADIX_SLOTS, warp, lane
        )
        _set_write_back(
            from_registers,
            vals,
            seg_base,
            lane,
            end,
            active,
            survivors,
            cur,
            incoming,
            n_incoming,
            max_topk,
            tidx,
            pivot_key,
            dst,
            eq_quota,
            out_base,
        )
        cute.arch.barrier()
        return _undistort(pivot_key)

    @cute.jit
    def _reconstruct(
        self, survivors, cur, incoming, n_incoming, k, hist, tidx, warp, lane
    ):
        """Select the k best of ``survivors[cur]`` plus ``incoming[:n_incoming]``
        into ``survivors[cur ^ 1][:k]``; the rest of that buffer keeps its
        placeholders. Returns the k-th value's float bits, the new threshold."""
        max_topk = cutlass.const_expr(self.max_topk)
        unused = cute.make_rmem_tensor((1,), cutlass.Float32)
        return self._radix_select(
            False,
            unused,
            cutlass.Int32(0),
            cutlass.Int32(0),
            cutlass.Boolean(False),
            survivors,
            cur,
            incoming,
            n_incoming,
            k,
            (cur ^ 1) * max_topk,
            hist,
            tidx,
            warp,
            lane,
        )

    # ------------------------------------------------------------------
    # Loading the init window into registers
    # ------------------------------------------------------------------
    @cute.jit
    def _load_segment(self, row_base, seg_elem_base, lane, vals):
        """Load this lane's 16 entries of the segment at ``seg_elem_base``.

        Lane ``l`` owns the four 16-byte chunks ``128 * j + 4 * l`` of the
        segment, so each warp-wide load touches 512 contiguous bytes.
        """
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128
        )
        for j in cutlass.range_constexpr(4):
            offset = seg_elem_base + cutlass.Int32(128 * j) + lane * 4
            src_ptr = cute.make_ptr(
                cutlass.Float32,
                row_base + cutlass.Int64(offset) * 4,
                assumed_align=16,
            )
            src = cute.make_tensor(src_ptr, cute.make_layout((4,)))
            frag = cute.make_rmem_tensor((4,), cutlass.Float32)
            cute.copy(copy_atom, src, frag)
            for c in cutlass.range_constexpr(4):
                vals[4 * j + c] = frag[c]

    @cute.jit
    def _load_segment_bounded(self, row_base, seg_elem_base, end, lane, vals):
        """As ``_load_segment`` but entries at or past ``end`` read as -inf."""
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128
        )
        neg_inf = _bits_f32(cutlass.Uint32(NEG_INF_BITS))
        for j in cutlass.range_constexpr(4):
            offset = seg_elem_base + cutlass.Int32(128 * j) + lane * 4
            frag = cute.make_rmem_tensor((4,), cutlass.Float32)
            if offset < end:
                src_ptr = cute.make_ptr(
                    cutlass.Float32,
                    row_base + cutlass.Int64(offset) * 4,
                    assumed_align=16,
                )
                src = cute.make_tensor(src_ptr, cute.make_layout((4,)))
                cute.copy(copy_atom, src, frag)
            else:
                for c in cutlass.range_constexpr(4):
                    frag[c] = neg_inf
            for c in cutlass.range_constexpr(4):
                if offset + c < end:
                    vals[4 * j + c] = frag[c]
                else:
                    vals[4 * j + c] = neg_inf

    # ------------------------------------------------------------------
    # Main-round pipeline: one TMA copy per warp per round into a smem ring
    # ------------------------------------------------------------------
    @cute.jit
    def _issue_round(
        self, stage, seg, is_active, row_base, ring, full_bars, warp, lane
    ):
        """Start the copy of segment ``seg`` into this warp's slot of ``stage``.

        Every warp arrives on the stage's mbarrier exactly once per round, so
        the phase completes once all sixteen segments (or their skipped
        counterparts) are accounted for. Called by the whole warp.
        """
        if lane == 0:
            bar = full_bars + stage
            if is_active:
                src = row_base + cutlass.Int64(cutlass.Int32(seg)) * SEG_BYTES
                dst = ring.iterator + (stage * ROUND_ELEMS + warp * SEG_ELEMS)
                # Order this warp's earlier generic reads of the slot before the
                # async-proxy write that refills it.
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive_and_expect_tx(bar, SEG_BYTES)
                _bulk_copy_g2s(
                    cutlass.Int32(dst.toint()),
                    src,
                    cutlass.Int32(SEG_BYTES),
                    cutlass.Int32(bar.toint()),
                )
            else:
                cute.arch.mbarrier_arrive(bar)

    # ------------------------------------------------------------------
    # The kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        scores: cute.Tensor,
        ends: cute.Tensor,
        out: cute.Tensor,
        values: cute.Tensor,
    ):
        max_topk = cutlass.const_expr(self.max_topk)

        tidx, _, _ = cute.arch.thread_idx()
        rank, row, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tidx // 32)
        lane = tidx % 32

        width = cutlass.Int32(scores.shape[1])
        topk = cutlass.Int32(out.shape[1])
        end = cutlass.Int32(ends[row])
        end = cutlass.max(cutlass.min(end, width), cutlass.Int32(0))

        smem = SmemAllocator()
        survivors = smem.allocate_tensor(
            element_type=cutlass.Uint64,
            layout=cute.make_layout((2 * max_topk,)),
            byte_alignment=16,
        )
        incoming = smem.allocate_tensor(
            element_type=cutlass.Uint64,
            layout=cute.make_layout((INCOMING_SLOTS,)),
            byte_alignment=16,
        )
        ring = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((NUM_STAGES * ROUND_ELEMS,)),
            byte_alignment=128,
        )
        hist = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_layout((NUM_HISTS * RADIX_SLOTS + NUM_WARPS + 4,)),
            byte_alignment=16,
        )
        warp_cnt = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_layout((2 * NUM_WARPS,)),
            byte_alignment=16,
        )
        full_bars = smem.allocate_array(cutlass.Int64, NUM_STAGES, byte_alignment=8)

        if end <= topk:
            # Shortcut: the whole row is selected, in order; slots past the
            # end and NaN entries yield the (-1, -inf) placeholder.
            if rank == 0:
                for i in range(tidx, topk, NUM_THREADS):
                    index = cutlass.Int32(-1)
                    value = _bits_f32(cutlass.Uint32(NEG_INF_BITS))
                    if i < end:
                        score = scores[row, i]
                        if score == score:
                            index = cutlass.Int32(i)
                            value = score
                    out[row, i] = index
                    values[row, i] = value
        else:
            self._select_row(
                scores,
                out,
                values,
                row,
                rank,
                end,
                topk,
                tidx,
                warp,
                lane,
                survivors,
                incoming,
                ring,
                hist,
                warp_cnt,
                full_bars,
            )

    @cute.jit
    def _select_row(
        self,
        scores,
        out,
        values,
        row,
        rank,
        end,
        topk,
        tidx,
        warp,
        lane,
        survivors,
        incoming,
        ring,
        hist,
        warp_cnt,
        full_bars,
    ):
        max_topk = cutlass.const_expr(self.max_topk)
        cluster_size = cutlass.const_expr(self.cluster_size)
        row_base = (scores.iterator + row * scores.stride[0]).toint()
        placeholder = cutlass.Uint64(PLACEHOLDER_PAIR)
        neg_inf = _bits_f32(cutlass.Uint32(NEG_INF_BITS))

        for i in range(tidx, 2 * max_topk, NUM_THREADS):
            survivors[i] = placeholder
        if tidx == 0:
            for stage in cutlass.range_constexpr(NUM_STAGES):
                cute.arch.mbarrier_init(full_bars + stage, NUM_WARPS)
            cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # ---- Partition the row: an in-order tail and permuted segments ----
        num_segs = (end + cutlass.Int32(SEG_ELEMS - 1)) // cutlass.Int32(SEG_ELEMS)
        num_perm_segs = cutlass.Int32(0)
        if num_segs > SEGS_PER_ROUND:
            num_perm_segs = (
                (num_segs - 1) // cutlass.Int32(TAIL_SEGS) * cutlass.Int32(TAIL_SEGS)
            )
        num_perm_elems = num_perm_segs * SEG_ELEMS
        tail_elems = end - num_perm_elems
        tail_padded = num_segs * SEG_ELEMS
        if num_perm_segs > 0:
            tail_padded = cutlass.Int32(TAIL_ELEMS)
        total_padded = tail_padded + num_perm_elems
        perm_len = cutlass.Uint32(cutlass.max(num_perm_segs, cutlass.Int32(1)))
        perm_mul = cutlass.Uint32(PERM_MUL_PRIME) % perm_len
        # Advancing one round moves every warp SEGS_PER_ROUND positions along
        # the pseudo-random order, i.e. by this fixed stride modulo perm_len.
        round_stride = (cutlass.Uint32(SEGS_PER_ROUND) * perm_mul) % perm_len

        # This CTA's share of the padded row is [lo, hi); the tail belongs to
        # rank 0 alone, so the boundaries are clamped into the permuted part.
        lo = rank * total_padded // cluster_size
        hi = (rank + 1) * total_padded // cluster_size
        local_start_seg = cutlass.max(lo - tail_padded, cutlass.Int32(0)) // SEG_ELEMS
        local_end_seg = cutlass.max(hi - tail_padded, cutlass.Int32(0)) // SEG_ELEMS
        local_perm_segs = local_end_seg - local_start_seg
        local_tail_padded = cutlass.Int32(0)
        local_tail_elems = cutlass.Int32(0)
        if rank == 0:
            local_tail_padded = tail_padded
            local_tail_elems = tail_elems
        local_tail_segs = local_tail_padded // SEG_ELEMS
        local_elems_padded = local_tail_padded + local_perm_segs * SEG_ELEMS
        local_rounds = (
            local_elems_padded + cutlass.Int32(ROUND_ELEMS - 1)
        ) // cutlass.Int32(ROUND_ELEMS)
        init_perm_segs = cutlass.min(
            local_perm_segs, cutlass.Int32(SEGS_PER_ROUND) - local_tail_segs
        )
        main_rounds = cutlass.max(local_rounds - 1, cutlass.Int32(0))

        threshold_bits = cutlass.Uint32(NEG_INF_BITS)
        cur = cutlass.Int32(0)
        num_incomers = cutlass.Int32(0)
        vals = cute.make_rmem_tensor((ELEMS_PER_THREAD,), cutlass.Float32)

        if local_elems_padded > 0:
            # ---- Init window: exact top-k of the first round ----
            # Warps below local_tail_segs hold the tail in order; the rest hold
            # the first permuted segments of this CTA's share; one compaction
            # straight from the registers selects the top-k.
            seg_base = cutlass.Int32(0)
            active = cutlass.Boolean(False)
            if warp < local_tail_segs:
                seg_base = num_perm_elems + warp * SEG_ELEMS
                active = cutlass.Boolean(True)
                self._load_segment_bounded(row_base, seg_base, end, lane, vals)
            else:
                perm_pos = warp - local_tail_segs
                if perm_pos < local_perm_segs:
                    active = cutlass.Boolean(True)
                    seg = _permuted_segment(
                        local_start_seg + perm_pos, perm_len, perm_mul
                    )
                    seg_base = cutlass.Int32(seg) * SEG_ELEMS
                    self._load_segment(row_base, seg_base, lane, vals)
                else:
                    for e in cutlass.range_constexpr(ELEMS_PER_THREAD):
                        vals[e] = neg_inf
            init_padded = local_tail_padded + init_perm_segs * SEG_ELEMS
            num_real = init_padded - (local_tail_padded - local_tail_elems)
            eff_k = cutlass.min(topk, num_real)
            threshold_bits = self._radix_select(
                True,
                vals,
                seg_base,
                end,
                active,
                survivors,
                cur,
                incoming,
                cutlass.Int32(0),
                eff_k,
                cur * max_topk,
                hist,
                tidx,
                warp,
                lane,
            )

        # ---- Main rounds over the remaining permuted segments ----
        # Warp w visits positions first_linear + m * 16 of the pseudo-random
        # order; the segment ids are tracked incrementally for the round being
        # copied (issue_seg) and the round being filtered (seg).
        first_linear = local_start_seg + init_perm_segs + warp
        seg = _permuted_segment(first_linear, perm_len, perm_mul)
        issue_seg = seg
        for r in cutlass.range_constexpr(NUM_STAGES - 1):
            if r < main_rounds:
                self._issue_round(
                    cutlass.Int32(r),
                    issue_seg,
                    first_linear + r * SEGS_PER_ROUND < local_end_seg,
                    row_base,
                    ring,
                    full_bars,
                    warp,
                    lane,
                )
                issue_seg = _advance_perm(issue_seg, round_stride, perm_len)
        threshold = _bits_f32(threshold_bits)
        for m in range(main_rounds):
            if m + (NUM_STAGES - 1) < main_rounds:
                # The stage being refilled was consumed by this warp in round
                # m - 1, and only this warp ever touches its slot of it.
                self._issue_round(
                    (m + (NUM_STAGES - 1)) % NUM_STAGES,
                    issue_seg,
                    first_linear + (m + (NUM_STAGES - 1)) * SEGS_PER_ROUND
                    < local_end_seg,
                    row_base,
                    ring,
                    full_bars,
                    warp,
                    lane,
                )
                issue_seg = _advance_perm(issue_seg, round_stride, perm_len)
            active = first_linear + m * SEGS_PER_ROUND < local_end_seg
            seg_base = cutlass.Int32(seg) * SEG_ELEMS
            stage = m % NUM_STAGES
            cute.arch.mbarrier_wait(full_bars + stage, (m // NUM_STAGES) & 1)
            slot = stage * ROUND_ELEMS + warp * SEG_ELEMS + lane * 4
            for j in cutlass.range_constexpr(4):
                for c in cutlass.range_constexpr(4):
                    vals[4 * j + c] = ring[slot + 128 * j + c]

            hit_mask = cutlass.Int32(0)
            for e in cutlass.range_constexpr(ELEMS_PER_THREAD):
                if vals[e] > threshold:
                    hit_mask = hit_mask | cutlass.Int32(1 << e)
            if ~active:
                hit_mask = cutlass.Int32(0)
            n_hits = cutlass.Int32(cute.arch.popc(hit_mask))
            lane_inclusive = _warp_inclusive_sum(n_hits, lane)
            warp_total = cute.arch.shuffle_sync(lane_inclusive, 31)
            # One barrier per round: the count exchange alternates between two
            # slots, so a warp racing ahead into the next round cannot clobber
            # a count a slower warp is still reading.
            parity_base = (m & 1) * NUM_WARPS
            if lane == 0:
                warp_cnt[parity_base + warp] = warp_total
            cute.arch.barrier()
            entry = warp_cnt[parity_base + (lane & (NUM_WARPS - 1))]
            if lane >= NUM_WARPS:
                entry = cutlass.Int32(0)
            round_total = cute.arch.warp_redux_sync(entry, "add")
            if lane >= warp:
                entry = cutlass.Int32(0)
            dst = (
                num_incomers
                + cute.arch.warp_redux_sync(entry, "add")
                + lane_inclusive
                - n_hits
            )
            # Hits are rare, so walking the set bits (re-reading the value from
            # the ring, since the bit index is dynamic) beats touching all 16
            # entries again.
            mask = hit_mask
            while mask != 0:
                e = _lowest_set_bit(mask)
                mask = mask & (mask - 1)
                # Entry e of this lane sits at chunk e // 4, element e % 4.
                off = ((e >> cutlass.Int32(2)) << cutlass.Int32(7)) + (
                    e & cutlass.Int32(3)
                )
                incoming[dst] = _pack_pair(
                    seg_base + lane * 4 + off, _f32_bits(ring[slot + off])
                )
                dst = dst + 1
            num_incomers = num_incomers + round_total
            seg = _advance_perm(seg, round_stride, perm_len)
            if (num_incomers >= RECONSTRUCT_THRESHOLD) & (m + 1 < main_rounds):
                threshold_bits = self._reconstruct(
                    survivors, cur, incoming, num_incomers, topk, hist, tidx, warp, lane
                )
                threshold = _bits_f32(threshold_bits)
                cur = cur ^ 1
                num_incomers = cutlass.Int32(0)

        if num_incomers > 0:
            self._reconstruct(
                survivors, cur, incoming, num_incomers, topk, hist, tidx, warp, lane
            )
            cur = cur ^ 1
            num_incomers = cutlass.Int32(0)

        # ---- Cluster gather: non-leaders ship survivors to rank 0 ----
        if cutlass.const_expr(cluster_size > 1):
            # Every CTA must have finished scanning before anyone writes into
            # the leader's candidate buffer.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            if rank != 0:
                gather_base = (rank - 1) * max_topk
                for i in range(tidx, max_topk, NUM_THREADS):
                    remote = _map_shared_cluster(
                        incoming.iterator + (gather_base + i), cutlass.Int32(0)
                    )
                    _store_shared_cluster_u64(remote, survivors[cur * max_topk + i])
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            if rank == 0:
                self._reconstruct(
                    survivors,
                    cur,
                    incoming,
                    cutlass.Int32((cluster_size - 1) * max_topk),
                    topk,
                    hist,
                    tidx,
                    warp,
                    lane,
                )
                cur = cur ^ 1

        if rank == 0:
            for i in range(tidx, topk, NUM_THREADS):
                pair = survivors[cur * max_topk + i]
                out[row, i] = _pair_index(pair)
                values[row, i] = _pair_value(pair)

    # ------------------------------------------------------------------
    # Host launcher
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        scores: cute.Tensor,
        ends: cute.Tensor,
        out: cute.Tensor,
        values: cute.Tensor,
        stream,
    ):
        cluster_size = cutlass.const_expr(self.cluster_size)
        num_rows = scores.shape[0]
        self.kernel(scores, ends, out, values).launch(
            grid=(cluster_size, num_rows, 1),
            block=(NUM_THREADS, 1, 1),
            cluster=(
                (cluster_size, 1, 1) if cutlass.const_expr(cluster_size > 1) else None
            ),
            stream=stream,
        )


# ----------------------------------------------------------------------
# Host side
# ----------------------------------------------------------------------
MAX_TOPK = SUPPORTED_MAX_TOPK[-1]
# Row stride granularity of the score matrix, in elements (16-byte loads).
SCORE_ALIGNMENT = 4


def capacity_for(topk: int) -> int:
    """The smallest compiled survivor capacity that serves ``topk``."""
    for capacity in SUPPORTED_MAX_TOPK:
        if topk <= capacity:
            return capacity
    raise ValueError(f"topk must be in [1, {MAX_TOPK}], got {topk}")


def cluster_sizes_for(capacity: int) -> tuple[int, ...]:
    """Cluster sizes whose gathered survivors fit the candidate buffer."""
    return tuple(
        c for c in SUPPORTED_CLUSTER_SIZES if (c - 1) * capacity <= INCOMING_SLOTS
    )


@functools.cache
def compiled_selector(capacity: int, cluster_size: int, device_index: int):
    """Compile (once per configuration and device) the selector.

    The compiled callable takes ``(scores, ends, indices, values)`` as torch
    tensors and runs on the caller's current CUDA stream, so it is safe to
    replay inside a CUDA graph once compiled.
    """
    rows = cute.sym_int()
    width = cute.sym_int()
    stride = cute.sym_int(divisibility=SCORE_ALIGNMENT)
    topk = cute.sym_int()
    scores = cute.runtime.make_fake_tensor(
        cutlass.Float32, (rows, width), (stride, 1), assumed_align=16
    )
    ends = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (rows,), stride_order=(0,)
    )
    indices = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (rows, topk), stride_order=(1, 0), assumed_align=4
    )
    values = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (rows, topk), stride_order=(1, 0), assumed_align=4
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    # cute.compile targets the current device, not the operands'.
    with torch.cuda.device(device_index):
        return cute.compile(
            DeepSelectTopK(capacity, cluster_size),
            scores,
            ends,
            indices,
            values,
            stream=stream,
            options="--enable-tvm-ffi",
        )


def warmup(capacities: tuple[int, ...], device: torch.device) -> None:
    """Compile every cluster variant of the given survivor capacities.

    Call before CUDA-graph capture; compiling inside a capture is not
    possible. ``device`` may be concrete or the index-less
    ``torch.device("cuda")``, which warms the current device.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Compile the DeepSelect selector before graph capture")
    if device.type != "cuda":
        raise ValueError(f"device must be a CUDA device, got {device}")
    # Selector calls cache under the concrete index of the scores device, so
    # an index-less device must warm the same key.
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    for capacity in capacities:
        if capacity not in SUPPORTED_MAX_TOPK:
            raise ValueError(
                f"capacity must be one of {SUPPORTED_MAX_TOPK}, got {capacity}"
            )
        for cluster_size in cluster_sizes_for(capacity):
            compiled_selector(capacity, cluster_size, device_index)


# Rows below this width finish in a couple of rounds; the cluster gather and
# extra compaction would cost more than the split saves.
CLUSTER_MIN_WIDTH = 4 * ROUND_ELEMS


def choose_cluster_size(rows: int, width: int, capacity: int, sm_count: int) -> int:
    """CTAs per row for a launch of ``rows`` rows of ``width`` entries.

    Splitting a row across a cluster pays a gather and one more compaction,
    so it only helps rows of at least ``CLUSTER_MIN_WIDTH`` entries, and
    only while every cluster is resident at once: a GPC holds few clusters,
    and once ``rows * cluster_size`` exceeds roughly 60% of the SMs (80% for
    pairs) a second wave queues up and doubles the time. Measured on an
    H20 (78 SMs): 8-CTA clusters serve up to 6 rows, 4-CTA up to 12, 2-CTA
    up to 32.
    """
    if width < CLUSTER_MIN_WIDTH:
        return 1
    for cluster_size in reversed(cluster_sizes_for(capacity)):
        if cluster_size == 1:
            continue
        budget = sm_count * 5 // 6 if cluster_size == 2 else sm_count * 5 // 8
        if rows * cluster_size <= budget:
            return cluster_size
    return 1


def deepselect_topk(
    scores: torch.Tensor,
    ends: torch.Tensor,
    topk: int,
    *,
    capacity: int,
    cluster_size: int,
    indices: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices and values of the ``topk`` largest entries of every row.

    Args:
        scores: CUDA FP32 ``[rows, width]`` with unit column stride, a row
            stride that is a multiple of ``SCORE_ALIGNMENT`` elements and a
            16-byte aligned base. Entries at or past a row's end are never
            used, but the row's storage must extend to the next multiple of
            ``SCORE_ALIGNMENT`` (any padded stride does).
        ends: CUDA int32 ``[rows]``; row ``r`` selects among its first
            ``ends[r]`` entries (clamped to ``[0, width]``).
        topk: Entries to select per row, in ``[1, capacity]``.
        capacity: Compiled survivor capacity, one of ``SUPPORTED_MAX_TOPK``;
            :func:`capacity_for` gives the smallest that serves ``topk``.
        cluster_size: CTAs cooperating on one row, one of
            :func:`cluster_sizes_for` the capacity; see
            :func:`choose_cluster_size`.
        indices: Optional CUDA int32 contiguous ``[rows, topk]`` destination.
        values: Optional CUDA FP32 contiguous ``[rows, topk]`` destination.

    Returns:
        ``(indices, values)``: unsorted column indices and their scores. A
        row with ``ends[r] <= topk`` yields ``0 .. ends[r] - 1`` in order,
        followed by ``-1`` slots scoring ``-inf``; otherwise all ``topk``
        slots hold distinct indices below ``ends[r]``. Ties at the k-th
        value are broken arbitrarily but deterministically, so ``-inf``
        masking pads may be selected when fewer than ``topk`` finite entries
        exist; callers filter those by value. NaN entries are never
        selected: each one encountered yields a ``-1`` slot scoring
        ``-inf`` instead (in a shortcut row, at its position).
    """
    if scores.ndim != 2 or scores.dtype != torch.float32 or not scores.is_cuda:
        raise ValueError("scores must be a CUDA FP32 [rows, width] matrix")
    if (
        scores.stride(1) != 1
        or scores.stride(0) % SCORE_ALIGNMENT
        or scores.stride(0) < scores.shape[1]
        or scores.data_ptr() % (SCORE_ALIGNMENT * 4)
    ):
        raise ValueError(
            f"scores rows must be unit-stride with a stride multiple of {SCORE_ALIGNMENT} "
            "elements from a 16-byte aligned base"
        )
    rows = scores.shape[0]
    if ends.shape != (rows,) or ends.dtype != torch.int32 or not ends.is_contiguous():
        raise ValueError("ends must be contiguous int32 [rows]")
    if ends.device != scores.device:
        raise ValueError("ends must live on the scores device")
    if capacity not in SUPPORTED_MAX_TOPK or not 1 <= topk <= capacity:
        raise ValueError(
            f"capacity must be one of {SUPPORTED_MAX_TOPK} and at least topk, "
            f"got capacity {capacity} for topk {topk}"
        )
    if cluster_size not in cluster_sizes_for(capacity):
        raise ValueError(
            f"cluster_size {cluster_size} is not available for capacity {capacity}: "
            f"choose from {cluster_sizes_for(capacity)}"
        )
    if indices is None:
        indices = torch.empty((rows, topk), dtype=torch.int32, device=scores.device)
    if values is None:
        values = torch.empty((rows, topk), dtype=torch.float32, device=scores.device)
    for name, tensor, dtype in (
        ("indices", indices, torch.int32),
        ("values", values, torch.float32),
    ):
        if (
            tensor.shape != (rows, topk)
            or tensor.dtype != dtype
            or not tensor.is_contiguous()
            or tensor.device != scores.device
        ):
            raise ValueError(
                f"{name} must be a contiguous {dtype} [rows, topk] tensor on the scores device"
            )
    if rows == 0:
        return indices, values
    compiled_selector(capacity, cluster_size, scores.device.index)(
        scores, ends, indices, values
    )
    return indices, values
