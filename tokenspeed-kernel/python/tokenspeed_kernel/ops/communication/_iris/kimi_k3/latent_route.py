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

"""Route K3 decode MoE rows and project the shared experts inside the gather.

The latent-input gather's programs mostly wait for peer words. When every rank
projects all router columns, the router program of each row routes it there,
exactly as ``kimi3_sigmoid_bias_topk_sorted`` does through 32 rows (the same
scratch protocol, so the two may alternate on one scratch), and programs
waiting on the shared SiTU input write the shared experts' down projection.
"""

from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris.synchronization import wait_vmem

K3_ROUTE_EXPERTS = gl.constexpr(896)
K3_ROUTE_TOPK = gl.constexpr(16)
# kimi3_sigmoid_bias_topk_sorted's scratch: [896 * 8] arrival masks and 16
# counters, then the self-sorting launches' two [896] mask buffers, their two
# arrival counters and the launch epoch.
K3_ROUTE_SELF_MASKS = gl.constexpr(896 * 8 + 16)
K3_ROUTE_SCRATCH_WORDS = 896 * 8 + 16 + 2 * 896 + 16


@gluon.jit
def _k3_add(a, b):
    return a + b


@gluon.jit
def _k3_max_pair(key_a, index_a, key_b, index_b):
    return gl.maximum(key_a, key_b), gl.maximum(index_a, index_b)


@gluon.jit
def _k3_order_pairs(keys, index, flip, AXIS: gl.constexpr, DIMS: gl.constexpr):
    """Order the pairs of hypercube ``keys`` along its ``AXIS``-th inner axis.

    ``triton.language.standard._compare_and_swap``: a pair's left key becomes
    the larger where ``flip`` holds and the smaller elsewhere. Bit
    ``AXIS + 2`` of ``index`` is a key's position along the axis.
    """
    bits = keys.to(gl.int64, bitcast=True)
    partner = bits ^ gl.xor_sum(bits, DIMS - 1 - AXIS, keep_dims=True)
    partner = partner.to(gl.uint64, bitcast=True)
    right = ((index >> (AXIS + 2)) & 1) != 0
    return gl.where((keys > partner) != (flip ^ right), partner, keys)


@gluon.jit
def _k3_top16(keys, index):
    """Return the 16 largest of 1024 distinct ``keys``, in descending order.

    ``triton.language.standard.sort_impl``'s bitonic top-k on a hypercube of
    ``keys``, which hold their position ``index`` four to a thread. The
    runs of 16 lie along lane bits 0-3 (``index`` bits 2-5), so pairs meet
    by lane swizzles, and the first two halvings fold a thread's registers,
    so it holds one key from then on. Flips and sides are bits of ``index``.
    """
    cube = gl.permute(gl.reshape(keys, [2] * 10), [0, 1, 2, 3, 8, 9, 4, 5, 6, 7])
    index = gl.permute(gl.reshape(index, [2] * 10), [0, 1, 2, 3, 8, 9, 4, 5, 6, 7])
    # Sort runs of 16, alternately ascending and descending; the flip of
    # stage 4 is the first halved axis, ``index`` bit 0.
    for stage in gl.static_range(1, 5):
        flip = ((index >> ((stage + 2) * (stage < 4))) & 1) != 0
        for i in gl.static_range(stage):
            cube = _k3_order_pairs(cube, index, flip, stage - 1 - i, 10)
    # The larger halves of two opposite runs form a run holding their top 16.
    # Halvings take ``index`` bits 0, 1, 6, 7, 8 and 9 in turn.
    for i in gl.static_range(5, 11):
        cube, index = gl.reduce((cube, index), 10 - i, _k3_max_pair)
        if i < 10:
            flip = ((index >> (i - 4 + 4 * (i >= 6))) & 1) != 0
        else:
            flip = True
        for j in gl.static_range(4):
            cube = _k3_order_pairs(cube, index, flip, 3 - j, 14 - i)
    return gl.reshape(cube, [16])


@gluon.jit
def _k3_sleep():
    gl.inline_asm_elementwise(
        asm="s_sleep 2",
        constraints="=r,~{memory}",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _k3_await_count(counter, target):
    """Wait until ``counter`` reaches ``target``, then acquire once.

    The polls are relaxed so they do not invalidate this CU's caches for the
    other programs; the final acquire does, and the barrier keeps every
    subgroup's loads after it.
    """
    arrived = gl.atomic_add(counter, 0, sem="relaxed", scope="gpu")
    while arrived < target:
        _k3_sleep()
        arrived = gl.atomic_add(counter, 0, sem="relaxed", scope="gpu")
    gl.atomic_add(counter, 0, sem="acquire", scope="gpu")
    gl.barrier()


@gluon.jit
def _k3_ordered_key(value):
    """Map FP32 ``value`` to a uint32 key that compares as unsigned in the
    same order as the floats, for ``_k3_top16``'s packed keys.

    FP32 bits are sign and magnitude. Setting the sign bit of a non-negative
    value puts it above every negative one, keeping magnitude order; inverting
    every bit of a negative value clears its sign bit and reverses its
    magnitude order, so a more negative value gets a smaller key.
    """
    bits = value.to(gl.uint32, bitcast=True)
    sign = gl.full_like(bits, 0x80000000)
    return bits ^ gl.where((bits & sign) != 0, gl.full_like(bits, 0xFFFFFFFF), sign)


@gluon.jit
def _k3_popcount(words):
    """Count the set bits in each 32-bit lane of the expert arrival ``words``.

    The SWAR population count: each step adds neighboring fields of the
    previous width in parallel, so every field holds the count of its bits.
    The int32 result is each lane's count.
    """
    bits = words.to(gl.uint32, bitcast=True)
    # Each 2-bit field: its count of set bits (0..2), as pair - (pair >> 1).
    bits = bits - ((bits >> 1) & 0x55555555)
    # Each 4-bit field: the sum of its two 2-bit counts (0..4).
    bits = (bits & 0x33333333) + ((bits >> 2) & 0x33333333)
    # Each byte: the sum of its two 4-bit counts (0..8), which cannot carry.
    bits = (bits + (bits >> 4)) & 0x0F0F0F0F
    # Multiplying by 0x01010101 sums the four byte counts into the top byte.
    return ((bits * 0x01010101) >> 24).to(gl.int32)


@gluon.jit
def _k3_route_row(
    partial_ptr,
    router_ptr,
    bias_ptr,
    topk_ids,
    topk_weights,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    scratch,
    row,
    rows,
    base,
    split_stride,
    SPLIT_K: gl.constexpr,
    ROUTED_SCALING_FACTOR: gl.constexpr,
    NORMALIZE_TOPK_WEIGHTS: gl.constexpr,
    BLOCK_SHIFT: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Route row ``row`` from its router partials and place its sorted routes.

    Sums the row's FP32 split-K router partials in split order, as the
    gather's router publishers do (a negative zero becomes zero), and writes
    them to ``router_ptr``. Routes like ``_kimi3_route_row``: the top 16 of
    sigmoid plus bias in descending order with the lower expert first on a
    tie, weighted by the normalized unbiased sigmoid, selected by the bitonic
    top-k of ``tl.topk``. Then the row arrives and places its routes as
    ``_kimi3_self_sorted_routes``.
    """
    EXPERTS: gl.constexpr = K3_ROUTE_EXPERTS
    TOPK: gl.constexpr = K3_ROUTE_TOPK
    PADDED: gl.constexpr = 1024
    BLOCK: gl.constexpr = 1 << BLOCK_SHIFT
    gl.static_assert(NUM_WARPS == 4)

    sort_layout: gl.constexpr = gl.BlockedLayout(
        [PADDED // (64 * NUM_WARPS)], [64], [NUM_WARPS], [0]
    )
    expert = gl.arange(0, PADDED, layout=sort_layout)
    valid = expert < EXPERTS
    # Every partial row starts on a multiple of four words.
    quad_stride = split_stride >> 2
    logit = gl.zeros([PADDED], gl.float32, sort_layout)
    for split in range(0, SPLIT_K):
        logit += gl.load(
            partial_ptr + split * quad_stride * 4 + base + expert,
            mask=valid,
            other=0.0,
        )
    logit = gl.where((logit.to(gl.uint32, bitcast=True) >> 16) == 0x8000, 0.0, logit)
    gl.store(router_ptr + row * EXPERTS + expert, logit, mask=valid)
    scores = 1.0 / (1.0 + gl.exp(-logit))
    bias = gl.load(bias_ptr + expert, mask=valid, other=0.0)
    choice = scores + bias
    choice = gl.where(valid & (choice == choice), choice, float("-inf"))
    packed = (_k3_ordered_key(choice).to(gl.uint64) << 32) | (PADDED - expert).to(
        gl.uint64
    )

    route_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    selected = gl.convert_layout(_k3_top16(packed, expert), route_layout)
    route_slot = gl.arange(0, TOPK, layout=route_layout)
    selected_ids = PADDED - (selected & 0xFFFFFFFF).to(gl.int32)
    weights = 1.0 / (1.0 + gl.exp(-gl.gather(logit, selected_ids, 0)))
    if NORMALIZE_TOPK_WEIGHTS:
        denominator = gl.sum(weights, 0)
        denominator = gl.where(denominator != 0.0, denominator, 1.0)
        weights = weights / denominator
    weights = weights * ROUTED_SCALING_FACTOR

    # Launches alternate between two mask buffers and counters, so none waits
    # to zero the buffer its rows read: each zeroes the one its predecessor
    # finished with, for its successor.
    self_masks = scratch + K3_ROUTE_SELF_MASKS
    epoch = self_masks + 2 * EXPERTS + 2
    parity = gl.load(epoch) & 1
    masks = self_masks + parity * EXPERTS
    counter = self_masks + 2 * EXPERTS + parity
    if row == 0:
        gl.store(
            self_masks + (1 - parity) * EXPERTS + expert,
            gl.zeros([PADDED], gl.int32, sort_layout),
            mask=valid,
        )
        gl.store(self_masks + 2 * EXPERTS + 1 - parity, 0)
    # Write through, so the arrival needs no L2 write-back.
    route_offsets = row * TOPK + route_slot
    gl.amd.cdna4.buffer_store(selected_ids, topk_ids, route_offsets, cache=".wt")
    gl.amd.cdna4.buffer_store(weights, topk_weights, route_offsets, cache=".wt")
    gl.atomic_or(
        masks + selected_ids,
        gl.full_like(selected_ids, 1) << row,
        sem="relaxed",
        scope="gpu",
    )
    # The barrier before the one-thread arrival does not wait for memory, so
    # every subgroup first waits until its stores and mask updates landed.
    wait_vmem()
    gl.barrier()
    gl.atomic_add(counter, 1, sem="relaxed", scope="gpu")
    _k3_await_count(counter, rows)

    expert_masks = gl.load(masks + expert, mask=valid, other=0)
    count = _k3_popcount(expert_masks)
    padded = ((count + BLOCK - 1) >> BLOCK_SHIFT) << BLOCK_SHIFT
    first = gl.associative_scan(padded, 0, _k3_add) - padded
    if row == 0:
        # Every row read the epoch before it arrived.
        gl.atomic_add(epoch, 1, sem="relaxed", scope="gpu")
        gl.store(num_valid_ids, gl.sum(padded, 0))
        gl.store(num_valid_ids + 1, rows)
    below = gl.gather(expert_masks, selected_ids, 0) & ((1 << row) - 1)
    expert_first = gl.gather(first, selected_ids, 0)
    routed = gl.gather(count, selected_ids, 0)
    target = expert_first + _k3_popcount(below)
    gl.store(sorted_ids + target, (route_slot << 24) | row)
    gl.store(sorted_weights + target, weights)
    gl.store(
        sorted_expert_ids + (target >> BLOCK_SHIFT),
        selected_ids,
        mask=(target & (BLOCK - 1)) == 0,
    )
    pad_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [TOPK, 64 // TOPK], [1, NUM_WARPS], [1, 0]
    )
    pad_slot: gl.constexpr = gl.SliceLayout(1, pad_layout)
    fill = gl.convert_layout(expert_first + routed, pad_slot)
    fill_end = gl.convert_layout(
        expert_first + (((routed + BLOCK - 1) >> BLOCK_SHIFT) << BLOCK_SHIFT), pad_slot
    )
    last = gl.convert_layout(target + 1, pad_slot) == fill
    padding = (
        fill[:, None]
        + gl.arange(0, BLOCK, layout=gl.SliceLayout(0, pad_layout))[None, :]
    )
    gl.store(
        sorted_ids + padding,
        gl.full([TOPK, BLOCK], TOPK << 24, gl.int32, pad_layout) | rows,
        mask=last[:, None] & (padding < fill_end[:, None]),
    )


@gluon.jit
def _k3_shared_down_tile(
    situ_ptr,
    weight_ptr,
    out_ptr,
    out_stride,
    down_sync,
    rows,
    tile,
    situ_programs,
    K: gl.constexpr,
    TILES: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Write columns ``tile * 16 * NUM_WARPS`` onward of ``situ @ weight.T``.

    The weight tile loads first; then the program waits until all
    ``situ_programs`` SiTU programs counted at ``down_sync[0]`` and reduces in
    FP32 with the MFMA chain of ``_kimi3_shared_down_tile``. The last tile
    to leave zeroes both counters for the next launch.
    """
    BLOCK_N: gl.constexpr = 16 * NUM_WARPS
    BLOCK_K: gl.constexpr = 256
    gl.static_assert(K % BLOCK_K == 0)
    STEPS: gl.constexpr = K // BLOCK_K
    mfma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, NUM_WARPS],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)
    b_k = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, dot_b))
    b_n = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, dot_b))
    weights = ()
    for step in gl.static_range(STEPS):
        weights += (
            gl.amd.cdna4.buffer_load(
                weight_ptr, b_n[None, :] * K + (step * BLOCK_K + b_k)[:, None]
            ),
        )

    _k3_await_count(down_sync, situ_programs)
    left = gl.atomic_add(down_sync + 1, 1, sem="relaxed", scope="gpu")
    if left == TILES - 1:
        gl.atomic_xchg(down_sync, 0, sem="relaxed", scope="gpu")
        gl.atomic_xchg(down_sync + 1, 0, sem="relaxed", scope="gpu")

    a_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, dot_a))
    a_k = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, dot_a))
    accumulator = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, mfma)
    for step in gl.static_range(STEPS):
        activation = gl.amd.cdna4.buffer_load(
            situ_ptr,
            a_m[:, None] * K + (step * BLOCK_K + a_k)[None, :],
            mask=(a_m < rows)[:, None],
            other=0.0,
        )
        accumulator = gl.amd.cdna4.mfma(activation, weights[step], accumulator)
    out_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mfma))
    out_n = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma))
    gl.store(
        out_ptr + out_m[:, None] * out_stride + out_n[None, :],
        accumulator.to(out_ptr.dtype.element_ty),
        mask=(out_m < rows)[:, None],
    )
