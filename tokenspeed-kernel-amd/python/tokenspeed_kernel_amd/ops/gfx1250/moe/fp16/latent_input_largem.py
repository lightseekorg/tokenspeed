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

"""Large-M latent-MoE input projections for gfx1250.

Compute-bound at these lengths, the opposite of the decode kernels next door,
so this is one large-tile WMMA GEMM with no split reduction that fans the
accumulator out by region. The FP32 router logits therefore reach memory
without a round trip through BF16, and SiTU runs in the same launch.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx1250.gemm.fp16.mm import (
    _WARP_BASES_8,
    _largem_swizzle2d,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.fp16._common import (
    _is_packed_projection_view,
)

_HIDDEN = 7168
_ROUTER_N = 896
_LATENT_N = 3584
_SHARED_N = 768
_TOTAL_N = _ROUTER_N + _LATENT_N + 2 * _SHARED_N

_BLOCK_M = 256
_HALF_N = 128
_BLOCK_K = 128
_NUM_WARPS = 8
_GROUP_M = 8
# MI450 places consecutive workgroups on its eight XCDs in turn.
_NUM_XCDS = 8
# One TDM issuer per SIMD: with eight warps, warps 0-3 sit on distinct SIMDs.
_TDM_WARPS = 0b00001111


def _largem_launch_metadata(grid, kernel, args):
    """Report packed projection work and traffic to Proton."""
    m = args["M"]
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
def _consume_tile(
    a_smem,
    lo_smem,
    hi_smem,
    slot: gl.constexpr,
    a_desc,
    lo_desc,
    hi_desc,
    next_k,
    acc_lo,
    acc_hi,
    REFILL: gl.constexpr,
    DOT_A: gl.constexpr,
    DOT_B: gl.constexpr,
    BLOCK_K: gl.constexpr,
    TDM_WARPS: gl.constexpr,
):
    """Multiply the K tile in ``slot`` while refilling the other slot.

    Each K tile runs as two half-K steps of two stages: operand reads from
    LDS, then WMMA. The two warps sharing a SIMD run a stage apart, so one
    reads LDS while the other issues WMMA.
    """
    half_k: gl.constexpr = BLOCK_K // 2
    gl.amd.cdna5.tdm.async_wait(0)
    with gl.amd.warp_pipeline_stage("lds"):
        if REFILL:
            refill: gl.constexpr = 1 - slot
            gl.amd.cdna5.tdm.async_load(
                a_desc, [0, next_k], a_smem.index(refill), warp_used_hint=TDM_WARPS
            )
            gl.amd.cdna5.tdm.async_load(
                lo_desc, [0, next_k], lo_smem.index(refill), warp_used_hint=TDM_WARPS
            )
            gl.amd.cdna5.tdm.async_load(
                hi_desc, [0, next_k], hi_smem.index(refill), warp_used_hint=TDM_WARPS
            )
        a = a_smem.index(slot).slice(0, half_k, 1).load(layout=DOT_A)
        lo = lo_smem.index(slot).slice(0, half_k, 1).permute([1, 0]).load(layout=DOT_B)
        hi = hi_smem.index(slot).slice(0, half_k, 1).permute([1, 0]).load(layout=DOT_B)
    with gl.amd.warp_pipeline_stage("wmma"):
        acc_lo = gl.amd.cdna5.wmma(a, lo, acc_lo)
        acc_hi = gl.amd.cdna5.wmma(a, hi, acc_hi)
    with gl.amd.warp_pipeline_stage("lds"):
        a = a_smem.index(slot).slice(half_k, half_k, 1).load(layout=DOT_A)
        lo = (
            lo_smem.index(slot)
            .slice(half_k, half_k, 1)
            .permute([1, 0])
            .load(layout=DOT_B)
        )
        hi = (
            hi_smem.index(slot)
            .slice(half_k, half_k, 1)
            .permute([1, 0])
            .load(layout=DOT_B)
        )
    with gl.amd.warp_pipeline_stage("wmma"):
        acc_lo = gl.amd.cdna5.wmma(a, lo, acc_lo)
        acc_hi = gl.amd.cdna5.wmma(a, hi, acc_hi)
    return acc_lo, acc_hi


@gluon.jit
def _store_direct(
    acc,
    router_ptr,
    routed_ptr,
    rows,
    cols,
    live,
    ROUTER_N: gl.constexpr,
    LATENT_N: gl.constexpr,
):
    """Store a half tile of router (FP32) and routed (activation dtype) columns.

    A half tile can straddle the router/routed boundary, or reach past the
    routed region, so masks rather than the tile index place each column.
    """
    gl.amd.cdna5.buffer_store(
        acc,
        router_ptr,
        (rows[:, None] * ROUTER_N + cols[None, :]).to(gl.int32),
        mask=live & (cols[None, :] < ROUTER_N),
    )
    in_routed = (
        live & (cols[None, :] >= ROUTER_N) & (cols[None, :] < ROUTER_N + LATENT_N)
    )
    gl.amd.cdna5.buffer_store(
        acc.to(routed_ptr.dtype.element_ty),
        routed_ptr,
        (rows[:, None] * LATENT_N + cols[None, :] - ROUTER_N).to(gl.int32),
        mask=in_routed,
    )


@gluon.jit(launch_metadata=_largem_launch_metadata)
def gluon_latent_input_largem_gfx1250(
    a_ptr,
    b_ptr,
    router_ptr,
    routed_ptr,
    shared_ptr,
    beta,
    linear_beta,
    inv_beta,
    inv_linear_beta,
    stride_am,
    stride_bn,
    M,
    BLOCK_M: gl.constexpr,
    HALF_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    WARP_BASES: gl.constexpr,
    TDM_WARPS: gl.constexpr,
    ROUTER_N: gl.constexpr,
    LATENT_N: gl.constexpr,
    SHARED_N: gl.constexpr,
    HAS_LINEAR_BETA: gl.constexpr,
    K: gl.constexpr,
):
    """One output tile of the packed projection, fanned out by region.

    A tile is two ``HALF_N``-row slices of the packed weight: adjacent rows
    in the router and routed regions, and the gate rows with the matching up
    rows ``SHARED_N`` below them in the shared region, so a shared tile
    finishes SiTU itself.
    """

    BLOCK_N: gl.constexpr = 2 * HALF_N
    direct_tiles: gl.constexpr = (ROUTER_N + LATENT_N + BLOCK_N - 1) // BLOCK_N
    gl.static_assert(SHARED_N % HALF_N == 0, "shared region must tile exactly")
    num_k_tiles: gl.constexpr = K // BLOCK_K
    gl.static_assert(
        K % BLOCK_K == 0 and num_k_tiles % 2 == 0 and num_k_tiles >= 2,
        "the two-slot pipeline consumes K tiles in pairs",
    )

    pid = gl.program_id(0)
    grid_m = gl.cdiv(M, BLOCK_M)
    grid_n: gl.constexpr = direct_tiles + SHARED_N // HALF_N
    # Workgroups land on the XCDs in turn. Give each XCD a contiguous run of
    # the grouped order, so workgroups that share A or B rows share an L2.
    tiles = grid_m * grid_n
    xcd = pid % NUM_XCDS
    pid = (
        xcd * (tiles // NUM_XCDS) + gl.minimum(xcd, tiles % NUM_XCDS) + pid // NUM_XCDS
    )
    pid_m, pid_n = _largem_swizzle2d(pid, grid_m, grid_n, GROUP_M)
    off_m = pid_m * BLOCK_M
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
        [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0]
    )
    shared_layout_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 8]], [HALF_N, BLOCK_K], [1, 0]
    )
    a_smem = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [2, BLOCK_M, BLOCK_K], shared_layout_a
    )
    lo_smem = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [2, HALF_N, BLOCK_K], shared_layout_b
    )
    hi_smem = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [2, HALF_N, BLOCK_K], shared_layout_b
    )
    a_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=a_ptr + off_m * stride_am,
        shape=(M - off_m, K),
        strides=(stride_am, 1),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=shared_layout_a,
    )
    # Every weight slice lies inside the packed weight, so no bound clamps it.
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

    gl.amd.cdna5.tdm.async_load(
        a_desc, [0, 0], a_smem.index(0), warp_used_hint=TDM_WARPS
    )
    gl.amd.cdna5.tdm.async_load(
        lo_desc, [0, 0], lo_smem.index(0), warp_used_hint=TDM_WARPS
    )
    gl.amd.cdna5.tdm.async_load(
        hi_desc, [0, 0], hi_smem.index(0), warp_used_hint=TDM_WARPS
    )
    acc_lo = gl.zeros((BLOCK_M, HALF_N), gl.float32, wmma_layout)
    acc_hi = gl.zeros((BLOCK_M, HALF_N), gl.float32, wmma_layout)
    for i in range(0, num_k_tiles // 2 - 1):
        k = 2 * i * BLOCK_K
        acc_lo, acc_hi = _consume_tile(
            a_smem,
            lo_smem,
            hi_smem,
            0,
            a_desc,
            lo_desc,
            hi_desc,
            k + BLOCK_K,
            acc_lo,
            acc_hi,
            True,
            dot_layout_a,
            dot_layout_b,
            BLOCK_K,
            TDM_WARPS,
        )
        acc_lo, acc_hi = _consume_tile(
            a_smem,
            lo_smem,
            hi_smem,
            1,
            a_desc,
            lo_desc,
            hi_desc,
            k + 2 * BLOCK_K,
            acc_lo,
            acc_hi,
            True,
            dot_layout_a,
            dot_layout_b,
            BLOCK_K,
            TDM_WARPS,
        )
    acc_lo, acc_hi = _consume_tile(
        a_smem,
        lo_smem,
        hi_smem,
        0,
        a_desc,
        lo_desc,
        hi_desc,
        K - BLOCK_K,
        acc_lo,
        acc_hi,
        True,
        dot_layout_a,
        dot_layout_b,
        BLOCK_K,
        TDM_WARPS,
    )
    acc_lo, acc_hi = _consume_tile(
        a_smem,
        lo_smem,
        hi_smem,
        1,
        a_desc,
        lo_desc,
        hi_desc,
        0,
        acc_lo,
        acc_hi,
        False,
        dot_layout_a,
        dot_layout_b,
        BLOCK_K,
        TDM_WARPS,
    )

    # Write adjacent output columns together instead of retaining the WMMA
    # lane mapping. Eight BF16 values per lane give contiguous 16-byte stores.
    store_layout: gl.constexpr = gl.BlockedLayout(
        [1, 8], [2, 16], [gl.num_warps(), 1], [1, 0]
    )
    rows = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, store_layout))
    live = rows[:, None] < M
    offs_n = gl.arange(0, HALF_N, gl.SliceLayout(0, store_layout))
    if pid_n >= direct_tiles:
        # Round both projections through bf16 first so the result matches a
        # materialized gate/up projection.
        gate_raw = gl.convert_layout(acc_lo, store_layout).to(gl.bfloat16)
        up = gl.convert_layout(acc_hi, store_layout).to(gl.bfloat16)
        gate_raw = gate_raw.to(gl.float32)
        up = up.to(gl.float32)
        # sigmoid(x) = (1 + tanh(x / 2)) / 2 keeps the whole activation on
        # the hardware tanh, without exp or divide; a quarter of the tiles
        # run it, so its cost lands directly on the critical path.
        gate = beta * gl.extra.libdevice.tanh(gate_raw * inv_beta)
        gate *= 0.5 + 0.5 * gl.extra.libdevice.tanh(0.5 * gate_raw)
        if HAS_LINEAR_BETA:
            up = linear_beta * gl.extra.libdevice.tanh(up * inv_linear_beta)
        cols = (pid_n - direct_tiles) * HALF_N + offs_n
        gl.amd.cdna5.buffer_store(
            (gate * up).to(shared_ptr.dtype.element_ty),
            shared_ptr,
            (rows[:, None] * SHARED_N + cols[None, :]).to(gl.int32),
            mask=live,
        )
    else:
        _store_direct(
            gl.convert_layout(acc_lo, store_layout),
            router_ptr,
            routed_ptr,
            rows,
            lo_row + offs_n,
            live,
            ROUTER_N,
            LATENT_N,
        )
        _store_direct(
            gl.convert_layout(acc_hi, store_layout),
            router_ptr,
            routed_ptr,
            rows,
            hi_row + offs_n,
            live,
            ROUTER_N,
            LATENT_N,
        )


def launch_gluon_latent_input_largem_gfx1250(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    routed_down_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    packed_weight: torch.Tensor,
    *,
    beta: float,
    linear_beta: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a large-M block of Kimi-K3 latent-MoE inputs.

    Args:
        hidden_states: Contiguous BF16 activation shaped ``[tokens, 7168]``.
            Any positive ``tokens`` works; a partial final row tile is
            masked. Automatic dispatch uses this kernel from 1025 tokens.
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

    device = hidden_states.device
    router_out = torch.empty((tokens, _ROUTER_N), dtype=torch.float32, device=device)
    routed_out = torch.empty((tokens, _LATENT_N), dtype=torch.bfloat16, device=device)
    shared_out = torch.empty((tokens, _SHARED_N), dtype=torch.bfloat16, device=device)

    n_tiles = triton.cdiv(_ROUTER_N + _LATENT_N, 2 * _HALF_N) + _SHARED_N // _HALF_N
    grid = triton.cdiv(tokens, _BLOCK_M) * n_tiles
    linear = 1.0 if linear_beta is None else float(linear_beta)
    gluon_latent_input_largem_gfx1250[(grid,)](
        hidden_states,
        packed_weight,
        router_out,
        routed_out,
        shared_out,
        float(beta),
        linear,
        1.0 / float(beta),
        1.0 / linear,
        hidden_states.stride(0),
        packed_weight.stride(0),
        tokens,
        BLOCK_M=_BLOCK_M,
        HALF_N=_HALF_N,
        BLOCK_K=_BLOCK_K,
        GROUP_M=_GROUP_M,
        NUM_XCDS=_NUM_XCDS,
        WARP_BASES=_WARP_BASES_8,
        TDM_WARPS=_TDM_WARPS,
        ROUTER_N=_ROUTER_N,
        LATENT_N=_LATENT_N,
        SHARED_N=_SHARED_N,
        HAS_LINEAR_BETA=linear_beta is not None,
        K=_HIDDEN,
        num_warps=_NUM_WARPS,
        num_stages=1,
        waves_per_eu=2,
    )
    return router_out, routed_out, shared_out


__all__ = ["launch_gluon_latent_input_largem_gfx1250"]
