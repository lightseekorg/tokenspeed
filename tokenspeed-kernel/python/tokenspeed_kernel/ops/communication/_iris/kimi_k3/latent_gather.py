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

"""Finish K3 decode MoE input projections whose replicated columns are sharded.

Each rank projects only its eighth of the replicated router and routed-latent
columns (plus its own shared gate/up columns) with the small-batch split-K
GEMM. This epilogue sums the K splits exactly as the replicated epilogue does,
publishes the rank's FP32 router logits and BF16 latent columns through the
K3 Lamport mailbox, and one reducer program per row gathers every rank's
block. Every rank publishes the same 336 words of each row (112 router
columns and 448 latent columns), so each row's two Lamport tiles carry a word
from every rank in every round, which the three-stage mailbox relies on.
"""

from tokenspeed_kernel._triton import gl, gluon
from tokenspeed_kernel.ops.communication._iris.all_reduce import (
    _iris_sanitize_lamport_bf16,
    _iris_subgroup_any,
)

K3_LATENT_GATHER_ROW_WORDS = gl.constexpr(512)


@gluon.jit
def _k3_latent_publish(
    words,
    word,
    valid,
    epochs,
    regions,
    row,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
):
    """Store whole 32-bit words at ``word`` of slot RANK of every rank's row."""
    tile = row * 2 + word // 256
    stage = gl.load(epochs + tile, mask=valid, other=0).to(gl.uint32) % 3
    offsets = (
        (stage * (WORLD_SIZE * MAX_ELEMENTS // 2)).to(gl.int32)
        + RANK * (MAX_ELEMENTS // 2)
        + row * K3_LATENT_GATHER_ROW_WORDS
        + word
    )
    rotation = gl.program_id(0) % (WORLD_SIZE - 1)
    for first in gl.static_range(WORLD_SIZE - 1):
        if rotation == first:
            for step in gl.static_range(WORLD_SIZE - 1):
                destination = gl.multiple_of(
                    regions[
                        (RANK + 1 + (first + step) % (WORLD_SIZE - 1)) % WORLD_SIZE
                    ].to(gl.pointer_type(gl.uint32)),
                    16,
                )
                gl.amd.cdna4.buffer_store(
                    words,
                    destination,
                    offsets,
                    mask=valid,
                    cache=PEER_SIGNAL_STORE_CACHE,
                )
    # Write through: the reducer may poll the own slot from another XCD.
    own = gl.multiple_of(regions[RANK].to(gl.pointer_type(gl.uint32)), 16)
    gl.amd.cdna4.buffer_store(words, own, offsets, mask=valid, cache=".wt")


@gluon.jit
def _k3_latent_publish_row_max(
    payload,
    routed,
    sync_ptr,
    row,
    MAX_ROWS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Write this row's latent maximum |x| to its rendezvous slot.

    ``sync_ptr`` holds one slot per row, -1 while empty, so the non-negative
    FP32 bits of a maximum are their own flag, then a departure counter.
    """
    low = (payload << 16).to(gl.float32, bitcast=True)
    high = (payload & 0xFFFF0000).to(gl.float32, bitcast=True)
    magnitude = gl.where(routed, gl.maximum(gl.abs(low), gl.abs(high)), 0.0)
    row_max = gl.max(gl.max(magnitude, 1), 0).to(gl.int32, bitcast=True)
    slots_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    slot = gl.arange(0, MAX_ROWS, layout=slots_layout)
    gl.amd.cdna4.buffer_store(
        gl.full([MAX_ROWS], 0, gl.int32, slots_layout) + row_max,
        gl.multiple_of(sync_ptr, 16),
        slot,
        mask=slot == row,
        cache=".wt",
    )


@gluon.jit
def _k3_latent_quantize_row(
    payload,
    routed,
    routed_index,
    fp8_ptr,
    sync_ptr,
    scale_ptr,
    row,
    tokens,
    MIN_AMAX: gl.constexpr,
    FP8_MAX: gl.constexpr,
    MAX_ROWS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Quantize this row's BF16 latent words with the whole tensor's FP8 scale.

    The scale is ``max(amax |x|, MIN_AMAX) / FP8_MAX`` over every row, as the
    MoE's own dynamic quantizer computes it, from every row's published slot.
    The last program to leave empties the slots for the next launch.
    """
    slots_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    slot = gl.arange(0, MAX_ROWS, layout=slots_layout)
    slots = gl.multiple_of(sync_ptr, 16)
    active = slot < tokens
    maxima = gl.amd.cdna4.buffer_load(slots, slot, mask=active, other=0, cache=".cv")
    while _iris_subgroup_any((active & (maxima < 0)).to(gl.int32)):
        maxima = gl.amd.cdna4.buffer_load(
            slots, slot, mask=active, other=0, cache=".cv"
        )
    amax = gl.max(maxima, 0).to(gl.float32, bitcast=True)
    scale = gl.maximum(amax, MIN_AMAX) / FP8_MAX
    left = gl.atomic_add(sync_ptr + MAX_ROWS, 1, sem="relaxed", scope="gpu")
    if left == tokens - 1:
        gl.amd.cdna4.buffer_store(
            gl.full([MAX_ROWS], -1, gl.int32, slots_layout),
            slots,
            slot,
            cache=".wt",
        )
        gl.atomic_xchg(sync_ptr + MAX_ROWS, 0, sem="relaxed", scope="gpu")
    if row == 0:
        gl.store(scale_ptr, scale)
    low = (payload << 16).to(gl.float32, bitcast=True)
    high = (payload & 0xFFFF0000).to(gl.float32, bitcast=True)
    low_bits = (low / scale).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    high_bits = (high / scale).to(gl.float8e4nv).to(gl.uint8, bitcast=True)
    packed = low_bits.to(gl.uint16) | (high_bits.to(gl.uint16) << 8)
    halves = gl.multiple_of(fp8_ptr.to(gl.pointer_type(gl.uint16), bitcast=True), 16)
    gl.amd.cdna4.buffer_store(packed, halves, routed_index, mask=routed)


@gluon.jit
def _k3_latent_gather_row(
    region,
    epochs,
    router_ptr,
    routed_ptr,
    fp8_ptr,
    sync_ptr,
    scale_ptr,
    row,
    tokens,
    WORLD_SIZE: gl.constexpr,
    ROUTER_LOCAL: gl.constexpr,
    LATENT_LOCAL: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    QUANTIZE: gl.constexpr,
    MIN_AMAX: gl.constexpr,
    FP8_MAX: gl.constexpr,
    MAX_ROWS: gl.constexpr,
):
    """Wait for, copy out and clear every rank's words of ``row``."""
    WORDS: gl.constexpr = 16
    LANES: gl.constexpr = 64 * NUM_WARPS // WORLD_SIZE
    PUBLISHED: gl.constexpr = ROUTER_LOCAL + LATENT_LOCAL // 2
    gl.static_assert(LANES * WORDS >= PUBLISHED)
    gl.static_assert(ROUTER_LOCAL % WORDS == 0 and 256 % WORDS == 0)
    # Thread t polls WORDS words of slot t // LANES; a retry votes with one lane mask.
    layout: gl.constexpr = gl.BlockedLayout([1, WORDS], [64, 1], [NUM_WARPS, 1], [1, 0])
    thread = gl.arange(0, 64 * NUM_WARPS, layout=gl.SliceLayout(1, layout))
    lane_word = gl.arange(0, WORDS, layout=gl.SliceLayout(0, layout))
    slot = thread // LANES
    first = (thread % LANES) * WORDS
    valid = (first < PUBLISHED)[:, None] & (lane_word >= 0)[None, :]
    low_stage = gl.load(epochs + row * 2).to(gl.uint32) % 3
    high_stage = gl.load(epochs + row * 2 + 1).to(gl.uint32) % 3
    stage = gl.where(first < 256, low_stage, high_stage)
    base = (
        (stage * (WORLD_SIZE * MAX_ELEMENTS // 2)).to(gl.int32)
        + slot * (MAX_ELEMENTS // 2)
        + row * K3_LATENT_GATHER_ROW_WORDS
        + first
    )
    mailbox = gl.multiple_of(base, WORDS)[:, None] + lane_word[None, :]
    words_ptr = gl.multiple_of(region.to(gl.pointer_type(gl.uint32), bitcast=True), 16)
    # Published words never have 0x8000 as their high half; that marks an empty word.
    payload = gl.amd.cdna4.buffer_load(
        words_ptr, mailbox, mask=valid, other=0, cache=".cv"
    )
    while _iris_subgroup_any(
        gl.max((valid & ((payload >> 16) == 0x8000)).to(gl.int32), 1)
    ):
        payload = gl.amd.cdna4.buffer_load(
            words_ptr, mailbox, mask=valid, other=0, cache=".cv"
        )
    is_router = first < ROUTER_LOCAL
    routed = valid & (~is_router)[:, None]
    if QUANTIZE:
        _k3_latent_publish_row_max(payload, routed, sync_ptr, row, MAX_ROWS, NUM_WARPS)
    gl.amd.cdna4.buffer_store(
        gl.full_like(payload, 0x80000000), words_ptr, mailbox, mask=valid, cache=".wt"
    )
    router_words = gl.multiple_of(
        router_ptr.to(gl.pointer_type(gl.uint32), bitcast=True), 16
    )
    router_base = row * (WORLD_SIZE * ROUTER_LOCAL) + slot * ROUTER_LOCAL + first
    gl.amd.cdna4.buffer_store(
        payload,
        router_words,
        (gl.multiple_of(router_base, WORDS)[:, None] + lane_word[None, :]).to(gl.int32),
        mask=valid & is_router[:, None],
    )
    routed_words = gl.multiple_of(
        routed_ptr.to(gl.pointer_type(gl.uint32), bitcast=True), 16
    )
    routed_base = (
        row * (WORLD_SIZE * LATENT_LOCAL // 2)
        + slot * (LATENT_LOCAL // 2)
        + first
        - ROUTER_LOCAL
    )
    routed_index = (
        gl.multiple_of(routed_base, WORDS)[:, None] + lane_word[None, :]
    ).to(gl.int32)
    gl.amd.cdna4.buffer_store(payload, routed_words, routed_index, mask=routed)
    # All subgroups have left the poll, so local publishers have read this row's stages.
    gl.barrier()
    gl.store(epochs + row * 2, ((low_stage + 1) % 3).to(gl.int32))
    gl.store(epochs + row * 2 + 1, ((high_stage + 1) % 3).to(gl.int32))
    if QUANTIZE:
        _k3_latent_quantize_row(
            payload,
            routed,
            routed_index,
            fp8_ptr,
            sync_ptr,
            scale_ptr,
            row,
            tokens,
            MIN_AMAX,
            FP8_MAX,
            MAX_ROWS,
            NUM_WARPS,
        )


@gluon.jit
def _k3_latent_publish_router(
    partial_ptr,
    epochs,
    regions,
    row,
    base,
    split_stride,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    SPLIT_K: gl.constexpr,
    ROUTER_LOCAL: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    BLOCK_N: gl.constexpr = 128
    # The router logits: FP32 sums in split order, as the replicated epilogue.
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    local = gl.arange(0, BLOCK_N, layout=layout)
    valid = local < ROUTER_LOCAL
    column = RANK * ROUTER_LOCAL + local
    value = gl.zeros([BLOCK_N], gl.float32, layout)
    for split in range(0, SPLIT_K):
        value += gl.load(
            partial_ptr + split * split_stride + base + column,
            mask=valid,
            other=0.0,
        )
    words = value.to(gl.uint32, bitcast=True)
    words = gl.where((words >> 16) == 0x8000, 0, words)
    _k3_latent_publish(
        words,
        local,
        valid,
        epochs,
        regions,
        row,
        RANK,
        WORLD_SIZE,
        MAX_ELEMENTS,
        PEER_SIGNAL_STORE_CACHE,
    )


@gluon.jit
def _k3_latent_publish_latent(
    partial_ptr,
    epochs,
    regions,
    row,
    base,
    split_stride,
    tile,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    SPLIT_K: gl.constexpr,
    ROUTER_N: gl.constexpr,
    ROUTER_LOCAL: gl.constexpr,
    LATENT_LOCAL: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    BLOCK_N: gl.constexpr = 128
    # Latent columns in pairs, so each published word is one store.
    pair_layout: gl.constexpr = gl.BlockedLayout(
        [1, 2], [64, 1], [NUM_WARPS, 1], [1, 0]
    )
    pairs = tile * (BLOCK_N // 2) + gl.arange(
        0, BLOCK_N // 2, layout=gl.SliceLayout(1, pair_layout)
    )
    half = gl.arange(0, 2, layout=gl.SliceLayout(0, pair_layout))
    local = pairs[:, None] * 2 + half[None, :]
    valid = local < LATENT_LOCAL
    column = ROUTER_N + RANK * LATENT_LOCAL + local
    value = gl.zeros([BLOCK_N // 2, 2], gl.float32, pair_layout)
    for split in range(0, SPLIT_K):
        value += gl.load(
            partial_ptr + split * split_stride + base + column,
            mask=valid,
            other=0.0,
        )
    bits = _iris_sanitize_lamport_bf16(value.to(gl.bfloat16)).to(
        gl.uint16, bitcast=True
    )
    low, high = gl.split(bits)
    words = low.to(gl.uint32) | (high.to(gl.uint32) << 16)
    _k3_latent_publish(
        words,
        ROUTER_LOCAL + pairs,
        pairs * 2 < LATENT_LOCAL,
        epochs,
        regions,
        row,
        RANK,
        WORLD_SIZE,
        MAX_ELEMENTS,
        PEER_SIGNAL_STORE_CACHE,
    )


@gluon.jit
def _k3_latent_shared_input(
    partial_ptr,
    shared_ptr,
    beta,
    linear_beta,
    row,
    base,
    split_stride,
    tile,
    SPLIT_K: gl.constexpr,
    ROUTER_N: gl.constexpr,
    LATENT_N: gl.constexpr,
    SHARED_N: gl.constexpr,
    HAS_LINEAR_BETA: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    BLOCK_N: gl.constexpr = 128
    # The shared SiTU input, as gluon_latent_input_small_batch_epilogue_gfx950.
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    column = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=layout)
    mask = column < SHARED_N
    gate_base = base + ROUTER_N + LATENT_N + column
    gate_raw = gl.zeros([BLOCK_N], gl.float32, layout)
    up = gl.zeros([BLOCK_N], gl.float32, layout)
    for split in range(0, SPLIT_K):
        offset = split * split_stride + gate_base
        gate_raw += gl.load(partial_ptr + offset, mask=mask, other=0.0)
        up += gl.load(partial_ptr + offset + SHARED_N, mask=mask, other=0.0)
    gate_raw = gate_raw.to(gl.bfloat16).to(gl.float32)
    up = up.to(gl.bfloat16).to(gl.float32)
    gate = beta * gl.extra.libdevice.tanh(gate_raw / beta)
    gate *= 1.0 / (1.0 + gl.exp(-gate_raw))
    if HAS_LINEAR_BETA:
        up = linear_beta * gl.extra.libdevice.tanh(up / linear_beta)
    gl.store(
        shared_ptr + row * SHARED_N + column,
        (gate * up).to(shared_ptr.dtype.element_ty),
        mask=mask,
    )


@gluon.jit(do_not_specialize=("tokens", "split_stride"))
def iris_k3moe_latent_input_gather(
    partial_ptr,
    shared_ptr,
    router_ptr,
    routed_ptr,
    fp8_ptr,
    sync_ptr,
    scale_ptr,
    region,
    epochs,
    regions,
    beta,
    linear_beta,
    tokens,
    split_stride,
    RANK: gl.constexpr,
    WORLD_SIZE: gl.constexpr,
    SPLIT_K: gl.constexpr,
    TOTAL_N: gl.constexpr,
    ROUTER_N: gl.constexpr,
    LATENT_N: gl.constexpr,
    SHARED_N: gl.constexpr,
    HAS_LINEAR_BETA: gl.constexpr,
    MAX_ELEMENTS: gl.constexpr,
    PEER_SIGNAL_STORE_CACHE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    MAX_ROWS: gl.constexpr,
    QUANTIZE: gl.constexpr = False,
    MIN_AMAX: gl.constexpr = 1.0e-12,
    FP8_MAX: gl.constexpr = 448.0,
):
    """Sum the K splits of this rank's columns, publish and gather the rows.

    ``partial_ptr`` holds the ``[SPLIT_K, tokens, TOTAL_N]`` FP32 partials of
    router columns ``[RANK * R, (RANK + 1) * R)`` with R = ROUTER_N /
    WORLD_SIZE, latent columns ``ROUTER_N + [RANK * L, (RANK + 1) * L)`` with
    L = LATENT_N / WORLD_SIZE, and every shared gate/up column. Grid
    ``(tokens * (1 + L / 128 rounded up + SHARED_N / 128) + tokens,)``: per row
    one router program, the latent programs and the shared SiTU programs
    (as ``gluon_latent_input_small_batch_epilogue_gfx950``), then one
    gathering program per row writing the full ``router_ptr`` (FP32
    ``[tokens, ROUTER_N]``) and ``routed_ptr`` (BF16 ``[tokens, LATENT_N]``)
    rows. A router logit of negative zero arrives as zero.

    With QUANTIZE the gathering programs also write ``fp8_ptr`` (E4M3
    ``[tokens, LATENT_N]``) and its per-tensor ``scale_ptr`` exactly as the
    MoE's dynamic quantizer would from ``routed_ptr``; ``sync_ptr`` holds
    MAX_ROWS int32 slots of -1 and a zero counter, as the launch leaves them.
    """
    BLOCK_N: gl.constexpr = 128
    ROUTER_LOCAL: gl.constexpr = ROUTER_N // WORLD_SIZE
    LATENT_LOCAL: gl.constexpr = LATENT_N // WORLD_SIZE
    LATENT_TILES: gl.constexpr = (LATENT_LOCAL + BLOCK_N - 1) // BLOCK_N
    SHARED_TILES: gl.constexpr = SHARED_N // BLOCK_N
    PER_ROW: gl.constexpr = 1 + LATENT_TILES + SHARED_TILES
    gl.static_assert(WORLD_SIZE == 8 and ROUTER_LOCAL <= BLOCK_N)
    gl.static_assert(ROUTER_N % WORLD_SIZE == 0 and LATENT_LOCAL % 32 == 0)
    gl.static_assert(SHARED_N % BLOCK_N == 0)
    gl.static_assert(ROUTER_LOCAL + LATENT_LOCAL // 2 <= K3_LATENT_GATHER_ROW_WORDS)
    pid = gl.program_id(0)
    row = pid // PER_ROW
    job = pid % PER_ROW
    base = row * TOTAL_N
    # Gathering programs come last so every publisher dispatches before they spin.
    if pid >= tokens * PER_ROW:
        _k3_latent_gather_row(
            region,
            epochs,
            router_ptr,
            routed_ptr,
            fp8_ptr,
            sync_ptr,
            scale_ptr,
            pid - tokens * PER_ROW,
            tokens,
            WORLD_SIZE,
            ROUTER_LOCAL,
            LATENT_LOCAL,
            MAX_ELEMENTS,
            NUM_WARPS,
            QUANTIZE,
            MIN_AMAX,
            FP8_MAX,
            MAX_ROWS,
        )
    elif job == 0:
        _k3_latent_publish_router(
            partial_ptr,
            epochs,
            regions,
            row,
            base,
            split_stride,
            RANK,
            WORLD_SIZE,
            SPLIT_K,
            ROUTER_LOCAL,
            MAX_ELEMENTS,
            PEER_SIGNAL_STORE_CACHE,
            NUM_WARPS,
        )
    elif job <= LATENT_TILES:
        _k3_latent_publish_latent(
            partial_ptr,
            epochs,
            regions,
            row,
            base,
            split_stride,
            job - 1,
            RANK,
            WORLD_SIZE,
            SPLIT_K,
            ROUTER_N,
            ROUTER_LOCAL,
            LATENT_LOCAL,
            MAX_ELEMENTS,
            PEER_SIGNAL_STORE_CACHE,
            NUM_WARPS,
        )
    else:
        _k3_latent_shared_input(
            partial_ptr,
            shared_ptr,
            beta,
            linear_beta,
            row,
            base,
            split_stride,
            job - 1 - LATENT_TILES,
            SPLIT_K,
            ROUTER_N,
            LATENT_N,
            SHARED_N,
            HAS_LINEAR_BETA,
            NUM_WARPS,
        )
