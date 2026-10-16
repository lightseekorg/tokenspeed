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

"""Decode-batch latent-MoE input projections for gfx1250.

The weights dominate at these batch sizes, so both kernels here stream the
86 MB packed weight once and are bound by memory, and one launch covers the
whole projection including SiTU. What differs is how the weight is cut into
enough concurrent streams: up to 16 tokens every workgroup streams one
16-row weight slice, with the shared gate and up slices splitting K and
meeting through a per-block arrival counter; above that a workgroup owns
the whole reduction of a gate/up-paired 32-column tile.

One GEMM cannot serve the three consumers, which is why this is not a vendor
call: expert selection needs FP32 logits, the routed branch the activation
dtype, and the shared branch its gate and up halves combined.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx1250.moe.fp16._common import (
    _is_packed_projection_view,
)

_HIDDEN = 7168
_ROUTER_N = 896
_LATENT_N = 3584
_SHARED_N = 768
_TOTAL_N = _ROUTER_N + _LATENT_N + 2 * _SHARED_N

_counter_cache: dict[tuple[int, int, int], torch.Tensor] = {}


def _decode_launch_metadata(grid, kernel, args):
    """Report packed projection work and traffic to Proton."""
    m = args["actual_m"]
    # Shared tiles write and read back one FP32 partial per split.
    partial_bytes = (
        2 * args["SHARED_SPLIT"] * m * _SHARED_N * args["partial_ptr"].element_size()
    )
    return {
        "name": kernel.name,
        "flops16": 2 * m * _TOTAL_N * _HIDDEN,
        "bytes": m * _HIDDEN * args["a_ptr"].element_size()
        + _TOTAL_N * _HIDDEN * args["b_ptr"].element_size()
        + 2 * partial_bytes
        + m * _ROUTER_N * args["router_ptr"].element_size()
        + m * _LATENT_N * args["routed_ptr"].element_size()
        + m * _SHARED_N * args["shared_ptr"].element_size(),
    }


def _nosplit_launch_metadata(grid, kernel, args):
    """Report packed projection work and traffic to Proton."""
    m = args["actual_m"]
    return {
        "name": kernel.name,
        "flops16": 2 * m * _TOTAL_N * _HIDDEN,
        "bytes": m * _HIDDEN * args["a_ptr"].element_size()
        + _TOTAL_N * _HIDDEN * args["b_ptr"].element_size()
        + m * _ROUTER_N * args["router_ptr"].element_size()
        + m * _LATENT_N * args["routed_ptr"].element_size()
        + m * _SHARED_N * args["shared_ptr"].element_size(),
    }


@gluon.jit
def _situ(gate_raw, up, beta, linear_beta, HAS_LINEAR_BETA: gl.constexpr):
    """Combine FP32 gate and up projections into the SiTU activation."""
    # Round both projections through bf16 first so the result matches a
    # materialized gate/up projection.
    gate_raw = gate_raw.to(gl.bfloat16).to(gl.float32)
    up = up.to(gl.bfloat16).to(gl.float32)
    gate = beta * gl.extra.libdevice.tanh(gate_raw / beta)
    gate *= 1.0 / (1.0 + gl.exp(-gate_raw))
    if HAS_LINEAR_BETA:
        up = linear_beta * gl.extra.libdevice.tanh(up / linear_beta)
    return gate * up


@gluon.jit
def _store_direct(
    acc,
    router_ptr,
    routed_ptr,
    row,
    column,
    live,
    is_router,
    ROUTER_N: gl.constexpr,
    LATENT_N: gl.constexpr,
):
    """Store a router (FP32) or routed (activation dtype) tile."""
    if is_router:
        gl.amd.cdna5.buffer_store(
            acc,
            router_ptr,
            (row[:, None] * ROUTER_N + column[None, :]).to(gl.int32),
            mask=live,
        )
    else:
        gl.amd.cdna5.buffer_store(
            acc.to(routed_ptr.dtype.element_ty),
            routed_ptr,
            (row[:, None] * LATENT_N + column[None, :] - ROUTER_N).to(gl.int32),
            mask=live,
        )


@gluon.jit
def _wait_stores():
    """Wait until this wave's global stores are acknowledged."""
    gl.inline_asm_elementwise(
        asm="s_wait_storecnt 0x0",
        constraints="=v,0",
        args=[gl.arange(0, 32, gl.BlockedLayout([1], [32], [gl.num_warps()], [0]))],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit(launch_metadata=_decode_launch_metadata)
def gluon_latent_input_decode_gfx1250(
    a_ptr,
    b_ptr,
    partial_ptr,
    counter_ptr,
    router_ptr,
    routed_ptr,
    shared_ptr,
    beta,
    linear_beta,
    stride_am,
    stride_bn,
    actual_m,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    SHARED_SPLIT: gl.constexpr,
    ROUTER_N: gl.constexpr,
    LATENT_N: gl.constexpr,
    SHARED_N: gl.constexpr,
    HAS_LINEAR_BETA: gl.constexpr,
    K: gl.constexpr,
):
    """Project one ``BLOCK_N``-row weight slice on a single warp.

    Router and routed slices stream the whole reduction and store their
    columns. Shared gate and up slices instead split it ``SHARED_SPLIT``
    ways, so they finish early enough to hide the exchange: each stores an
    FP32 partial and bumps its column block's arrival counter, and the last
    arrival sums the partials in split order (deterministically), applies
    SiTU, and rearms the counter for the next launch or graph replay. The
    shared workgroups come first in launch order.
    """

    num_k_tiles: gl.constexpr = K // BLOCK_K
    shared_k_tiles: gl.constexpr = num_k_tiles // SHARED_SPLIT
    gl.static_assert(K % BLOCK_K == 0, "K must tile exactly")
    gl.static_assert(num_k_tiles % SHARED_SPLIT == 0, "K must split evenly")
    gl.static_assert(
        NUM_BUFFERS >= 2 and shared_k_tiles >= NUM_BUFFERS - 1,
        "each split must fill the TDM pipeline",
    )
    gl.static_assert(ROUTER_N % BLOCK_N == 0, "router region must tile exactly")
    gl.static_assert(LATENT_N % BLOCK_N == 0, "routed region must tile exactly")
    gl.static_assert(SHARED_N % BLOCK_N == 0, "shared region must tile exactly")

    shared_programs: gl.constexpr = (SHARED_N // BLOCK_N) * 2 * SHARED_SPLIT
    pid = gl.program_id(0)
    is_shared = pid < shared_programs
    # Shared programs are ordered (column block, split, gate/up).
    is_up = pid % 2
    split = (pid // 2) % SHARED_SPLIT
    block = pid // (2 * SHARED_SPLIT)
    b_row = gl.where(
        is_shared,
        ROUTER_N + LATENT_N + is_up * SHARED_N + block * BLOCK_N,
        (pid - shared_programs) * BLOCK_N,
    )
    k_tiles = gl.where(is_shared, shared_k_tiles, num_k_tiles)
    k_first = gl.where(is_shared, split * shared_k_tiles, 0)

    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[],
        reg_bases=[],
        instr_shape=[16, 16, 32],
    )
    dot_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma_layout, k_width=8
    )
    dot_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=wmma_layout, k_width=8
    )
    shared_layout_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [BLOCK_M, BLOCK_K], [1, 0]
    )
    shared_layout_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [BLOCK_N, BLOCK_K], [1, 0]
    )
    a_smem = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], shared_layout_a
    )
    b_smem = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_N, BLOCK_K], shared_layout_b
    )
    a_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=a_ptr,
        shape=(actual_m, K),
        strides=(stride_am, 1),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=shared_layout_a,
    )
    b_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=b_ptr + b_row * stride_bn,
        shape=(BLOCK_N, K),
        strides=(stride_bn, 1),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=shared_layout_b,
    )

    for tile in gl.static_range(NUM_BUFFERS - 1):
        offset = (k_first + tile) * BLOCK_K
        gl.amd.cdna5.tdm.async_load(a_desc, [0, offset], a_smem.index(tile))
        gl.amd.cdna5.tdm.async_load(b_desc, [0, offset], b_smem.index(tile))
    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, wmma_layout)
    main_k_tiles = k_tiles - (NUM_BUFFERS - 1)
    for tile in range(0, main_k_tiles):
        gl.amd.cdna5.tdm.async_wait(2 * (NUM_BUFFERS - 2))
        slot = tile % NUM_BUFFERS
        a = a_smem.index(slot).load(layout=dot_layout_a)
        b = b_smem.index(slot).permute([1, 0]).load(layout=dot_layout_b)
        refill = tile + NUM_BUFFERS - 1
        offset = (k_first + refill) * BLOCK_K
        gl.amd.cdna5.tdm.async_load(
            a_desc, [0, offset], a_smem.index(refill % NUM_BUFFERS)
        )
        gl.amd.cdna5.tdm.async_load(
            b_desc, [0, offset], b_smem.index(refill % NUM_BUFFERS)
        )
        acc = gl.amd.cdna5.wmma(a, b, acc)
    # Lower the wait before each tail tile instead of draining everything.
    for tail in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.cdna5.tdm.async_wait(2 * (NUM_BUFFERS - 2 - tail))
        slot = (main_k_tiles + tail) % NUM_BUFFERS
        a = a_smem.index(slot).load(layout=dot_layout_a)
        b = b_smem.index(slot).permute([1, 0]).load(layout=dot_layout_b)
        acc = gl.amd.cdna5.wmma(a, b, acc)

    row = gl.arange(0, BLOCK_M, gl.SliceLayout(1, wmma_layout))
    offs_n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, wmma_layout))
    live = row[:, None] < actual_m
    if is_shared:
        # Partials hold every row of the tile; rows past actual_m are zero.
        tile_elems: gl.constexpr = BLOCK_M * BLOCK_N
        offsets = (row[:, None] * BLOCK_N + offs_n[None, :]).to(gl.int32)
        block_ptr = partial_ptr + block * (2 * SHARED_SPLIT * tile_elems)
        # ".cg" stores and loads work at device scope, so partials another
        # workgroup published are not served stale from a non-coherent cache.
        gl.amd.cdna5.buffer_store(
            acc,
            block_ptr + (is_up * SHARED_SPLIT + split) * tile_elems,
            offsets,
            cache=".cg",
        )
        # Publish the partial before announcing it. One warp issues both, so
        # draining its stores orders them before the relaxed counter bump.
        _wait_stores()
        old = gl.atomic_add(counter_ptr + block, 1, sem="relaxed", scope="gpu")
        if old == 2 * SHARED_SPLIT - 1:
            # Every other arrival has incremented, so rearm the counter.
            gl.store(counter_ptr + block, 0)
            gate_raw = gl.amd.cdna5.buffer_load(block_ptr, offsets, cache=".cg")
            up = gl.amd.cdna5.buffer_load(
                block_ptr + SHARED_SPLIT * tile_elems, offsets, cache=".cg"
            )
            for s in gl.static_range(1, SHARED_SPLIT):
                gate_raw += gl.amd.cdna5.buffer_load(
                    block_ptr + s * tile_elems, offsets, cache=".cg"
                )
                up += gl.amd.cdna5.buffer_load(
                    block_ptr + (SHARED_SPLIT + s) * tile_elems, offsets, cache=".cg"
                )
            column = block * BLOCK_N + offs_n
            gl.amd.cdna5.buffer_store(
                _situ(gate_raw, up, beta, linear_beta, HAS_LINEAR_BETA).to(
                    shared_ptr.dtype.element_ty
                ),
                shared_ptr,
                (row[:, None] * SHARED_N + column[None, :]).to(gl.int32),
                mask=live,
            )
    else:
        _store_direct(
            acc,
            router_ptr,
            routed_ptr,
            row,
            b_row + offs_n,
            live,
            b_row < ROUTER_N,
            ROUTER_N,
            LATENT_N,
        )


@gluon.jit(launch_metadata=_nosplit_launch_metadata)
def gluon_latent_input_decode_nosplit_gfx1250(
    a_ptr,
    b_ptr,
    router_ptr,
    routed_ptr,
    shared_ptr,
    beta,
    linear_beta,
    stride_am,
    stride_bn,
    actual_m,
    BLOCK_M: gl.constexpr,
    HALF_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    WARP_BASES: gl.constexpr,
    ROUTER_N: gl.constexpr,
    LATENT_N: gl.constexpr,
    SHARED_N: gl.constexpr,
    HAS_LINEAR_BETA: gl.constexpr,
    K: gl.constexpr,
):
    """Project one column tile over the whole reduction.

    The tile is two ``HALF_N``-row slices of the packed weight: adjacent rows
    for the router and routed regions, and the matching gate and up rows for
    the shared region, so SiTU needs nothing from another workgroup.
    """

    BLOCK_N: gl.constexpr = 2 * HALF_N
    direct_tiles: gl.constexpr = (ROUTER_N + LATENT_N) // BLOCK_N
    # A tile goes to one consumer, so region boundaries must align with it.
    gl.static_assert(ROUTER_N % BLOCK_N == 0, "router region must tile exactly")
    gl.static_assert(LATENT_N % BLOCK_N == 0, "routed region must tile exactly")
    gl.static_assert(SHARED_N % HALF_N == 0, "shared region must tile exactly")
    gl.static_assert(K % BLOCK_K == 0, "K must tile exactly")
    num_k_tiles: gl.constexpr = K // BLOCK_K
    gl.static_assert(
        NUM_BUFFERS >= 2 and num_k_tiles >= NUM_BUFFERS - 1,
        "K must fill the TDM pipeline",
    )

    pid_n = gl.program_id(0)
    row_base = gl.program_id(1) * BLOCK_M
    if pid_n >= direct_tiles:
        lo_row = ROUTER_N + LATENT_N + (pid_n - direct_tiles) * HALF_N
        hi_row = lo_row + SHARED_N
    else:
        lo_row = pid_n * BLOCK_N
        hi_row = lo_row + HALF_N

    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=WARP_BASES,
        reg_bases=[],
        instr_shape=[16, 16, 32],
    )
    dot_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma_layout, k_width=8
    )
    dot_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=wmma_layout, k_width=8
    )
    shared_layout_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [BLOCK_M, BLOCK_K], [1, 0]
    )
    shared_layout_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[256, 8]], [HALF_N, BLOCK_K], [1, 0]
    )
    a_smem = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], shared_layout_a
    )
    lo_smem = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [NUM_BUFFERS, HALF_N, BLOCK_K], shared_layout_b
    )
    hi_smem = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [NUM_BUFFERS, HALF_N, BLOCK_K], shared_layout_b
    )
    a_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=a_ptr,
        shape=(actual_m, K),
        strides=(stride_am, 1),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=shared_layout_a,
    )
    lo_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=b_ptr + lo_row * stride_bn,
        shape=(HALF_N, K),
        strides=(stride_bn, 1),
        block_shape=(HALF_N, BLOCK_K),
        layout=shared_layout_b,
    )
    hi_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=b_ptr + hi_row * stride_bn,
        shape=(HALF_N, K),
        strides=(stride_bn, 1),
        block_shape=(HALF_N, BLOCK_K),
        layout=shared_layout_b,
    )

    for tile in gl.static_range(NUM_BUFFERS - 1):
        offset = tile * BLOCK_K
        gl.amd.cdna5.tdm.async_load(a_desc, [row_base, offset], a_smem.index(tile))
        gl.amd.cdna5.tdm.async_load(lo_desc, [0, offset], lo_smem.index(tile))
        gl.amd.cdna5.tdm.async_load(hi_desc, [0, offset], hi_smem.index(tile))

    acc_lo = gl.zeros((BLOCK_M, HALF_N), gl.float32, wmma_layout)
    acc_hi = gl.zeros((BLOCK_M, HALF_N), gl.float32, wmma_layout)
    # Three TDM loads fill each buffer.
    main_k_tiles: gl.constexpr = num_k_tiles - (NUM_BUFFERS - 1)
    for tile in range(0, main_k_tiles):
        gl.amd.cdna5.tdm.async_wait(3 * (NUM_BUFFERS - 2))
        slot = tile % NUM_BUFFERS
        a = a_smem.index(slot).load(layout=dot_layout_a)
        lo = lo_smem.index(slot).permute([1, 0]).load(layout=dot_layout_b)
        hi = hi_smem.index(slot).permute([1, 0]).load(layout=dot_layout_b)
        refill = (tile + NUM_BUFFERS - 1) % NUM_BUFFERS
        offset = (tile + NUM_BUFFERS - 1) * BLOCK_K
        gl.amd.cdna5.tdm.async_load(a_desc, [row_base, offset], a_smem.index(refill))
        gl.amd.cdna5.tdm.async_load(lo_desc, [0, offset], lo_smem.index(refill))
        gl.amd.cdna5.tdm.async_load(hi_desc, [0, offset], hi_smem.index(refill))
        acc_lo = gl.amd.cdna5.wmma(a, lo, acc_lo)
        acc_hi = gl.amd.cdna5.wmma(a, hi, acc_hi)
    # Lower the wait before each tail tile instead of draining everything.
    for tail in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.cdna5.tdm.async_wait(3 * (NUM_BUFFERS - 2 - tail))
        slot = (main_k_tiles + tail) % NUM_BUFFERS
        a = a_smem.index(slot).load(layout=dot_layout_a)
        lo = lo_smem.index(slot).permute([1, 0]).load(layout=dot_layout_b)
        hi = hi_smem.index(slot).permute([1, 0]).load(layout=dot_layout_b)
        acc_lo = gl.amd.cdna5.wmma(a, lo, acc_lo)
        acc_hi = gl.amd.cdna5.wmma(a, hi, acc_hi)

    row = row_base + gl.arange(0, BLOCK_M, gl.SliceLayout(1, wmma_layout))
    offs_n = gl.arange(0, HALF_N, gl.SliceLayout(0, wmma_layout))
    live = row[:, None] < actual_m
    if pid_n >= direct_tiles:
        column = (pid_n - direct_tiles) * HALF_N + offs_n
        gl.amd.cdna5.buffer_store(
            _situ(acc_lo, acc_hi, beta, linear_beta, HAS_LINEAR_BETA).to(
                shared_ptr.dtype.element_ty
            ),
            shared_ptr,
            (row[:, None] * SHARED_N + column[None, :]).to(gl.int32),
            mask=live,
        )
    else:
        _store_direct(
            acc_lo,
            router_ptr,
            routed_ptr,
            row,
            lo_row + offs_n,
            live,
            lo_row < ROUTER_N,
            ROUTER_N,
            LATENT_N,
        )
        _store_direct(
            acc_hi,
            router_ptr,
            routed_ptr,
            row,
            hi_row + offs_n,
            live,
            lo_row < ROUTER_N,
            ROUTER_N,
            LATENT_N,
        )


def _get_counters(device: torch.device, count: int) -> torch.Tensor:
    """Return ``count`` zeroed per-tile arrival counters.

    The last split of each tile rearms its counter, so the buffer is zero
    again whenever a launch completes. Launches ordered on one stream,
    including graph replays, can therefore share it; launches on different
    streams may overlap, so each stream gets its own buffer.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = triton.runtime.driver.active.get_current_stream(index)
    key = (index, stream, count)
    counters = _counter_cache.get(key)
    if counters is None:
        counters = torch.zeros((count,), dtype=torch.int32, device=device)
        # Never retain an allocation from a graph-private pool.
        if not torch.cuda.is_current_stream_capturing():
            _counter_cache[key] = counters
    return counters


@dataclass(frozen=True)
class _SmallBatchConfig:
    """One-warp ``block_m x 16`` tiles; shared tiles split K ``shared_split`` ways."""

    block_m: int
    block_k: int
    num_buffers: int
    shared_split: int


@dataclass(frozen=True)
class _NoSplitConfig:
    """Whole-reduction tiling: a ``block_m x 2 * half_n`` tile per workgroup."""

    block_m: int
    half_n: int
    block_k: int
    num_buffers: int
    warp_bases: tuple[tuple[int, int], ...]


def _decode_config(tokens: int) -> _SmallBatchConfig | _NoSplitConfig:
    """Return the tiling for a decode batch.

    A whole-reduction workgroup per 32 columns leaves 188 weight streams,
    which keeps memory busy once each also carries more than 16 rows. One
    16-row tile needs more streams, so small batches give every workgroup a
    single 16-row weight slice (472 workgroups).
    """
    if tokens <= 16:
        return _SmallBatchConfig(16, 256, 8, 2)
    if tokens <= 32:
        return _NoSplitConfig(32, 16, 256, 6, ((1, 0),))
    return _NoSplitConfig(64, 16, 256, 6, ((1, 0), (2, 0)))


def _launch_small_batch(
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    beta: float,
    linear_beta: float | None,
    config: _SmallBatchConfig,
) -> None:
    tokens = hidden_states.shape[0]
    if tokens > config.block_m:
        raise ValueError(
            f"the small-batch kernel takes at most {config.block_m} tokens"
        )
    block_n = 16
    blocks = _SHARED_N // block_n
    partials = torch.empty(
        (blocks, 2, config.shared_split, config.block_m, block_n),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    grid = blocks * 2 * config.shared_split + (_ROUTER_N + _LATENT_N) // block_n
    gluon_latent_input_decode_gfx1250[(grid,)](
        hidden_states,
        packed_weight,
        partials,
        _get_counters(hidden_states.device, blocks),
        *outputs,
        float(beta),
        1.0 if linear_beta is None else float(linear_beta),
        hidden_states.stride(0),
        packed_weight.stride(0),
        tokens,
        BLOCK_M=config.block_m,
        BLOCK_N=block_n,
        BLOCK_K=config.block_k,
        NUM_BUFFERS=config.num_buffers,
        SHARED_SPLIT=config.shared_split,
        ROUTER_N=_ROUTER_N,
        LATENT_N=_LATENT_N,
        SHARED_N=_SHARED_N,
        HAS_LINEAR_BETA=linear_beta is not None,
        K=_HIDDEN,
        num_warps=1,
        num_stages=1,
        waves_per_eu=0,
    )


def _launch_nosplit(
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    beta: float,
    linear_beta: float | None,
    config: _NoSplitConfig,
) -> None:
    tokens = hidden_states.shape[0]
    tiles = (_ROUTER_N + _LATENT_N) // (2 * config.half_n) + _SHARED_N // config.half_n
    gluon_latent_input_decode_nosplit_gfx1250[
        (tiles, triton.cdiv(tokens, config.block_m))
    ](
        hidden_states,
        packed_weight,
        *outputs,
        float(beta),
        1.0 if linear_beta is None else float(linear_beta),
        hidden_states.stride(0),
        packed_weight.stride(0),
        tokens,
        BLOCK_M=config.block_m,
        HALF_N=config.half_n,
        BLOCK_K=config.block_k,
        NUM_BUFFERS=config.num_buffers,
        WARP_BASES=config.warp_bases,
        ROUTER_N=_ROUTER_N,
        LATENT_N=_LATENT_N,
        SHARED_N=_SHARED_N,
        HAS_LINEAR_BETA=linear_beta is not None,
        K=_HIDDEN,
        num_warps=1 << len(config.warp_bases),
        num_stages=1,
        waves_per_eu=0,
    )


def _launch_decode(
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor,
    beta: float,
    linear_beta: float | None,
    config: _SmallBatchConfig | _NoSplitConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate the outputs and launch the kernel ``config`` selects."""
    tokens = hidden_states.shape[0]
    device = hidden_states.device
    outputs = (
        torch.empty((tokens, _ROUTER_N), dtype=torch.float32, device=device),
        torch.empty((tokens, _LATENT_N), dtype=torch.bfloat16, device=device),
        torch.empty((tokens, _SHARED_N), dtype=torch.bfloat16, device=device),
    )
    if isinstance(config, _SmallBatchConfig):
        _launch_small_batch(
            hidden_states, packed_weight, outputs, beta, linear_beta, config
        )
    else:
        _launch_nosplit(
            hidden_states, packed_weight, outputs, beta, linear_beta, config
        )
    return outputs


def launch_gluon_latent_input_decode_gfx1250(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    routed_down_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    packed_weight: torch.Tensor,
    *,
    beta: float,
    linear_beta: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a decode batch of Kimi-K3 latent-MoE inputs.

    Args:
        hidden_states: Contiguous BF16 activation shaped ``[tokens, 7168]``.
            Any positive ``tokens`` works; a partial final row tile is
            masked. Automatic dispatch uses this kernel to 64 tokens.
        router_weight: Packed weight view shaped ``[896, 7168]``.
        routed_down_weight: Packed weight view shaped ``[3584, 7168]``.
        shared_gate_up_weight: Packed weight view shaped ``[1536, 7168]``.
        packed_weight: Consecutive row view covering all three weights. The
            kernels read only this tensor, so it is rejected unless the
            three above live inside it.
        beta: Positive SiTU gate clipping scale.
        linear_beta: Optional positive SiTU linear-branch clipping scale.

    Returns:
        FP32 router logits, BF16 routed latent, and BF16 shared-expert input.
    """

    tokens = hidden_states.shape[0]
    expected = (
        (hidden_states, (tokens, _HIDDEN), "hidden states"),
        (router_weight, (_ROUTER_N, _HIDDEN), "router weight"),
        (routed_down_weight, (_LATENT_N, _HIDDEN), "routed-down weight"),
        (shared_gate_up_weight, (2 * _SHARED_N, _HIDDEN), "shared gate/up weight"),
        (packed_weight, (_TOTAL_N, _HIDDEN), "packed projection weight"),
    )
    for tensor, shape, name in expected:
        if tuple(tensor.shape) != shape:
            raise ValueError(f"Kimi K3 {name} must have shape {shape}")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"Kimi K3 {name} must be BF16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"Kimi K3 {name} must be contiguous on GPU")
        if tensor.device != hidden_states.device:
            raise ValueError("Kimi K3 MoE input tensors must be colocated")
    if not _is_packed_projection_view(
        packed_weight, router_weight, routed_down_weight, shared_gate_up_weight
    ):
        raise ValueError(
            "Kimi K3 packed weight must be the consecutive view of the "
            "router, routed, and shared weights"
        )
    if tokens <= 0:
        raise ValueError("projection needs at least one token")
    if beta <= 0.0 or (linear_beta is not None and linear_beta <= 0.0):
        raise ValueError("Kimi K3 SiTU beta values must be positive")

    return _launch_decode(
        hidden_states, packed_weight, beta, linear_beta, _decode_config(tokens)
    )


__all__ = ["launch_gluon_latent_input_decode_gfx1250"]
