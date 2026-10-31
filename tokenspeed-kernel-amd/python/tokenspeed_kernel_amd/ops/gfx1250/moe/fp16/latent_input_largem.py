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
    _largem_warp_pipeline_xcd_tile,
)
from tokenspeed_kernel_amd.ops.gfx1250.moe.fp16._common import (
    _is_packed_projection_view,
)

_HIDDEN = 7168
_ROUTER_N = 896
_LATENT_N = 3584
_SHARED_N = 768
_TOTAL_N = _ROUTER_N + _LATENT_N + 2 * _SHARED_N

_TILE_M = 256
_HALF_N = 128
_BLOCK_K = 128
_NUM_WARPS = 8
# A 1 x 2 CTA cluster multicasts each activation panel to the two CTAs of a
# 512-column cluster. A 2 x 2 cluster is no faster at 8192 rows and leaves a
# nearly empty third wave at 5376.
_CLUSTER = (1, 2)
_GROUP_M = 4
# A row tile spans 24 CTAs of 128 output columns; while every row tile fits
# on the 256 CUs in one wave, 128-row tiles finish sooner.
_SMALL_TILE_M = 128
_SMALL_GROUP_M = 8
_N_SLICES = 24
_NUM_CUS = 256
# MI450 places consecutive workgroups on its eight XCDs in turn.
_NUM_XCDS = 8


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


@gluon.constexpr_function
def _latent_largem_layouts(cta_m, cta_n, tile_m, block_k, num_warps):
    """LDS, WMMA and output staging layouts for a ``cta_m x cta_n`` cluster.

    Each CTA owns a ``tile_m x 128`` slice of both half accumulators. A row
    panel is multicast to a cluster row and a weight panel to a cluster
    column; the layouts' CGA part places each CTA's slice.
    """
    tile_n = 128
    block_m, block_n = cta_m * tile_m, cta_n * tile_n
    cga = [[1 << i, 0] for i in range(cta_m.bit_length() - 1)] + [
        [0, 1 << i] for i in range(cta_n.bit_length() - 1)
    ]
    # Splitting the A rows and weight rows across the two LDS partitions lets
    # paired waves read different partitions in the same instruction.
    _, _, local = gl.amd.cdna5.make_partitioned_dot_layouts(
        block_m,
        block_n,
        gl.PaddedSharedLayout.with_identity_for(
            [[block_k, 8]], [block_m, block_k], [1, 0]
        ),
        gl.PaddedSharedLayout.with_identity_for(
            [[block_k, 8]], [block_n, block_k], [1, 0]
        ),
        num_warps,
        [16, 16, 32],
        a_transposed=False,
        b_transposed=True,
        slice_m=tile_m,
        slice_n=tile_n,
        transposed=True,
    )
    wmma_layout = gl.amd.AMDWMMALayout(
        3, local.transposed, local.warp_bases, local.reg_bases, local.instr_shape, cga
    )
    cga_a = gl.DotOperandLayout(0, wmma_layout, 8).cga_layout
    # The weight sits in LDS as [N, K], so its CGA bases swap the operand's.
    cga_b = [[x[1], x[0]] for x in gl.DotOperandLayout(1, wmma_layout, 8).cga_layout]
    shared_a, shared_b, _ = gl.amd.cdna5.make_partitioned_dot_layouts(
        tile_m,
        tile_n,
        gl.PaddedSharedLayout.with_identity_for(
            [[block_k, 8]], [block_m, block_k], [1, 0], cga_a
        ),
        gl.PaddedSharedLayout.with_identity_for(
            [[block_k, 8]], [block_n, block_k], [1, 0], cga_b
        ),
        num_warps,
        [16, 16, 32],
        a_transposed=False,
        b_transposed=True,
        slice_m=tile_m,
        slice_n=tile_n,
        transposed=True,
    )
    shared_out = gl.PaddedSharedLayout.with_identity_for(
        [[tile_n, 8]], [block_m, block_n], [1, 0], cga
    )
    return shared_a, shared_b, wmma_layout, shared_out


@gluon.jit
def _consume_tile(
    a_smem,
    lo_smem,
    hi_smem,
    slot,
    refill_slot,
    a_desc,
    lo_desc,
    hi_desc,
    acc_lo,
    acc_hi,
    DOT_A: gl.constexpr,
    DOT_B: gl.constexpr,
    BLOCK_K: gl.constexpr,
    REFILL: gl.constexpr,
    CLUSTER: gl.constexpr,
):
    """Multiply the K tile in ``slot`` while refilling ``refill_slot``.

    Each K tile runs as two half-K steps of two stages: operand reads from
    LDS, then WMMA. The two waves sharing a SIMD run a stage apart, so one
    reads LDS while the other issues WMMA.
    """
    half_k: gl.constexpr = BLOCK_K // 2
    gl.amd.cdna5.tdm.async_wait(0)
    with gl.amd.warp_pipeline_stage("lds"):
        if REFILL:
            # Waves 0-3 (one per SIMD) issue the refill of the other slot.
            gl.amd.cdna5.tdm.async_load(
                a_desc, [0, 0], a_smem.index(refill_slot), warp_used_hint=0b1111
            )
            gl.amd.cdna5.tdm.async_load(
                lo_desc, [0, 0], lo_smem.index(refill_slot), warp_used_hint=0b1111
            )
            gl.amd.cdna5.tdm.async_load(
                hi_desc, [0, 0], hi_smem.index(refill_slot), warp_used_hint=0b1111
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
        if REFILL:
            a_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
                a_desc, add_offsets=[0, BLOCK_K]
            )
            lo_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
                lo_desc, add_offsets=[0, BLOCK_K]
            )
            hi_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
                hi_desc, add_offsets=[0, BLOCK_K]
            )
        if CLUSTER:
            # Keep the cluster's CTAs within one K tile of each other so
            # each multicast refill finds every destination slot free.
            gl.amd.cdna5.cluster.arrive()
    with gl.amd.warp_pipeline_stage("wmma"):
        acc_lo = gl.amd.cdna5.wmma(a, lo, acc_lo)
        acc_hi = gl.amd.cdna5.wmma(a, hi, acc_hi)
        if CLUSTER:
            gl.amd.cdna5.cluster.wait()
    return a_desc, lo_desc, hi_desc, acc_lo, acc_hi


@gluon.jit
def _panel_row(panel, ROUTER_PANELS: gl.constexpr, PANEL_N: gl.constexpr, ROUTER_N):
    """First weight row of a router or routed panel."""
    if panel < ROUTER_PANELS:
        row = panel * PANEL_N
    else:
        row = ROUTER_N + (panel - ROUTER_PANELS) * PANEL_N
    return row


@gluon.jit
def _store_panel(
    acc,
    out_ptr,
    off_m,
    col0,
    M,
    N: gl.constexpr,
    BLOCK_M: gl.constexpr,
    PANEL_N: gl.constexpr,
    SHARED_OUT: gl.constexpr,
):
    """Store one ``BLOCK_M x PANEL_N`` panel at column ``col0`` of an ``N``-wide
    output through LDS; the TDM store clips it at the last row and column."""
    smem = gl.allocate_shared_memory(
        out_ptr.dtype.element_ty, [BLOCK_M, PANEL_N], SHARED_OUT
    )
    smem.store(acc.to(out_ptr.dtype.element_ty))
    desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=out_ptr + off_m * N + col0,
        shape=(M - off_m, N - col0),
        strides=(N, 1),
        block_shape=(BLOCK_M, PANEL_N),
        layout=SHARED_OUT,
    )
    gl.amd.cdna5.tdm.async_store(desc, [0, 0], smem)
    # The next panel may reuse this staging buffer.
    gl.amd.cdna5.tdm.async_wait(0)


@gluon.jit
def _store_direct(
    acc,
    panel,
    router_ptr,
    routed_ptr,
    off_m,
    M,
    ROUTER_PANELS: gl.constexpr,
    DIRECT_PANELS: gl.constexpr,
    ROUTER_N: gl.constexpr,
    LATENT_N: gl.constexpr,
    BLOCK_M: gl.constexpr,
    PANEL_N: gl.constexpr,
    SHARED_OUT: gl.constexpr,
):
    """Store a router (FP32) or routed (activation dtype) panel.

    Panels never straddle the two regions; the last router panel is clipped
    at the router width, and a panel past the routed region is dropped.
    """
    if panel < ROUTER_PANELS:
        _store_panel(
            acc,
            router_ptr,
            off_m,
            panel * PANEL_N,
            M,
            ROUTER_N,
            BLOCK_M,
            PANEL_N,
            SHARED_OUT,
        )
    elif panel < DIRECT_PANELS:
        _store_panel(
            acc,
            routed_ptr,
            off_m,
            (panel - ROUTER_PANELS) * PANEL_N,
            M,
            LATENT_N,
            BLOCK_M,
            PANEL_N,
            SHARED_OUT,
        )


@gluon.jit(launch_metadata=_largem_launch_metadata, do_not_specialize=["M"])
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
    CTA_M: gl.constexpr,
    CTA_N: gl.constexpr,
    TILE_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    ROUTER_N: gl.constexpr,
    LATENT_N: gl.constexpr,
    SHARED_N: gl.constexpr,
    HAS_LINEAR_BETA: gl.constexpr,
    K: gl.constexpr,
):
    """One ``CTA_M x CTA_N`` cluster of the packed projection, fanned out by region.

    A cluster column holds two ``CTA_N * 128``-row weight panels: consecutive
    panels of the router and routed regions, and the gate rows with the
    matching up rows ``SHARED_N`` below them in the shared region, so a shared
    cluster finishes SiTU itself. Each CTA takes one 128-row slice of each.
    CTAs past the last row still run the K loop, so every cluster barrier
    completes; the TDM stores clip their rows.
    """
    gl.static_assert(gl.num_warps() == 8, "the warp pipeline needs two waves per SIMD")
    gl.static_assert(gl.num_ctas() == CTA_M * CTA_N, "one CTA per cluster slice")
    PANEL_N: gl.constexpr = CTA_N * 128
    BLOCK_M: gl.constexpr = CTA_M * TILE_M
    CLUSTER: gl.constexpr = CTA_M * CTA_N > 1
    gl.static_assert(SHARED_N % PANEL_N == 0, "shared region must tile exactly")
    gl.static_assert(LATENT_N % PANEL_N == 0, "routed region must tile exactly")
    gl.static_assert(K % BLOCK_K == 0, "K must tile exactly into BLOCK_K")
    # The router region rounds up to whole panels, so no panel holds both
    # router and routed columns.
    ROUTER_PANELS: gl.constexpr = (ROUTER_N + PANEL_N - 1) // PANEL_N
    DIRECT_PANELS: gl.constexpr = ROUTER_PANELS + LATENT_N // PANEL_N
    direct_tiles: gl.constexpr = (DIRECT_PANELS + 1) // 2
    num_k_tiles: gl.constexpr = K // BLOCK_K
    layouts: gl.constexpr = _latent_largem_layouts(CTA_M, CTA_N, TILE_M, BLOCK_K, 8)
    SHARED_A: gl.constexpr = layouts[0]
    SHARED_B: gl.constexpr = layouts[1]
    WMMA_LAYOUT: gl.constexpr = layouts[2]
    SHARED_OUT: gl.constexpr = layouts[3]

    grid_n: gl.constexpr = direct_tiles + SHARED_N // PANEL_N
    pid_m, pid_n = _largem_warp_pipeline_xcd_tile(
        gl.program_id(0), gl.cdiv(M, BLOCK_M), grid_n, NUM_XCDS, GROUP_M
    )
    off_m = pid_m * BLOCK_M
    if pid_n >= direct_tiles:
        lo_row = ROUTER_N + LATENT_N + (pid_n - direct_tiles) * PANEL_N
        hi_row = lo_row + SHARED_N
    else:
        lo_row = _panel_row(2 * pid_n, ROUTER_PANELS, PANEL_N, ROUTER_N)
        # A panel past the routed region multiplies in-bounds rows that are
        # never stored.
        hi_row = _panel_row(
            gl.minimum(2 * pid_n + 1, DIRECT_PANELS - 1),
            ROUTER_PANELS,
            PANEL_N,
            ROUTER_N,
        )

    dot_a: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 8)
    a_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=a_ptr + off_m * stride_am,
        shape=(M - off_m, K),
        strides=(stride_am, 1),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_A,
    )
    # Every weight panel lies inside the packed weight, so no bound clamps it.
    lo_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=b_ptr + lo_row * stride_bn,
        shape=(PANEL_N, K),
        strides=(stride_bn, 1),
        block_shape=(PANEL_N, BLOCK_K),
        layout=SHARED_B,
    )
    hi_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=b_ptr + hi_row * stride_bn,
        shape=(PANEL_N, K),
        strides=(stride_bn, 1),
        block_shape=(PANEL_N, BLOCK_K),
        layout=SHARED_B,
    )
    a_smem = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [2, BLOCK_M, BLOCK_K], SHARED_A
    )
    lo_smem = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [2, PANEL_N, BLOCK_K], SHARED_B
    )
    hi_smem = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [2, PANEL_N, BLOCK_K], SHARED_B
    )

    gl.amd.cdna5.tdm.async_load(a_desc, [0, 0], a_smem.index(0), warp_used_hint=0b1111)
    gl.amd.cdna5.tdm.async_load(
        lo_desc, [0, 0], lo_smem.index(0), warp_used_hint=0b1111
    )
    gl.amd.cdna5.tdm.async_load(
        hi_desc, [0, 0], hi_smem.index(0), warp_used_hint=0b1111
    )
    a_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(a_desc, add_offsets=[0, BLOCK_K])
    lo_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
        lo_desc, add_offsets=[0, BLOCK_K]
    )
    hi_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
        hi_desc, add_offsets=[0, BLOCK_K]
    )
    acc_lo = gl.zeros((BLOCK_M, PANEL_N), gl.float32, WMMA_LAYOUT)
    acc_hi = gl.zeros((BLOCK_M, PANEL_N), gl.float32, WMMA_LAYOUT)
    # Double buffer: consuming K tile i refills the other slot with tile i + 1.
    for i in range(0, num_k_tiles - 1):
        a_desc, lo_desc, hi_desc, acc_lo, acc_hi = _consume_tile(
            a_smem,
            lo_smem,
            hi_smem,
            i % 2,
            (i + 1) % 2,
            a_desc,
            lo_desc,
            hi_desc,
            acc_lo,
            acc_hi,
            dot_a,
            dot_b,
            BLOCK_K,
            True,
            CLUSTER,
        )
    a_desc, lo_desc, hi_desc, acc_lo, acc_hi = _consume_tile(
        a_smem,
        lo_smem,
        hi_smem,
        (num_k_tiles - 1) % 2,
        0,
        a_desc,
        lo_desc,
        hi_desc,
        acc_lo,
        acc_hi,
        dot_a,
        dot_b,
        BLOCK_K,
        False,
        CLUSTER,
    )

    if pid_n >= direct_tiles:
        # Round both projections through bf16 first so the result matches a
        # materialized gate/up projection.
        gate_raw = acc_lo.to(gl.bfloat16).to(gl.float32)
        up = acc_hi.to(gl.bfloat16).to(gl.float32)
        # sigmoid(x) = (1 + tanh(x / 2)) / 2 keeps the whole activation on
        # the hardware tanh, without exp or divide; a quarter of the tiles
        # run it, so its cost lands directly on the critical path.
        gate = beta * gl.extra.libdevice.tanh(gate_raw * inv_beta)
        gate *= 0.5 + 0.5 * gl.extra.libdevice.tanh(0.5 * gate_raw)
        if HAS_LINEAR_BETA:
            up = linear_beta * gl.extra.libdevice.tanh(up * inv_linear_beta)
        _store_panel(
            gate * up,
            shared_ptr,
            off_m,
            (pid_n - direct_tiles) * PANEL_N,
            M,
            SHARED_N,
            BLOCK_M,
            PANEL_N,
            SHARED_OUT,
        )
    else:
        _store_direct(
            acc_lo,
            2 * pid_n,
            router_ptr,
            routed_ptr,
            off_m,
            M,
            ROUTER_PANELS,
            DIRECT_PANELS,
            ROUTER_N,
            LATENT_N,
            BLOCK_M,
            PANEL_N,
            SHARED_OUT,
        )
        _store_direct(
            acc_hi,
            2 * pid_n + 1,
            router_ptr,
            routed_ptr,
            off_m,
            M,
            ROUTER_PANELS,
            DIRECT_PANELS,
            ROUTER_N,
            LATENT_N,
            BLOCK_M,
            PANEL_N,
            SHARED_OUT,
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
            masked. Automatic dispatch uses this kernel from 129 tokens.
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

    if triton.cdiv(tokens, _SMALL_TILE_M) * _N_SLICES <= _NUM_CUS:
        # One short wave: a per-K-tile cluster barrier costs more than the
        # multicast saves.
        cluster, tile_m, group_m = (1, 1), _SMALL_TILE_M, _SMALL_GROUP_M
    else:
        cluster, tile_m, group_m = _CLUSTER, _TILE_M, _GROUP_M
    return _launch_largem(
        hidden_states,
        packed_weight,
        beta,
        linear_beta,
        cluster=cluster,
        tile_m=tile_m,
        group_m=group_m,
    )


def _launch_largem(
    hidden_states: torch.Tensor,
    packed_weight: torch.Tensor,
    beta: float,
    linear_beta: float | None,
    *,
    cluster: tuple[int, int],
    tile_m: int,
    group_m: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = hidden_states.shape[0]
    device = hidden_states.device
    router_out = torch.empty((tokens, _ROUTER_N), dtype=torch.float32, device=device)
    routed_out = torch.empty((tokens, _LATENT_N), dtype=torch.bfloat16, device=device)
    shared_out = torch.empty((tokens, _SHARED_N), dtype=torch.bfloat16, device=device)

    cta_m, cta_n = cluster
    panel_n = cta_n * _HALF_N
    direct_panels = triton.cdiv(_ROUTER_N, panel_n) + _LATENT_N // panel_n
    n_tiles = triton.cdiv(direct_panels, 2) + _SHARED_N // panel_n
    grid = triton.cdiv(tokens, cta_m * tile_m) * n_tiles
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
        CTA_M=cta_m,
        CTA_N=cta_n,
        TILE_M=tile_m,
        BLOCK_K=_BLOCK_K,
        GROUP_M=group_m,
        NUM_XCDS=_NUM_XCDS,
        ROUTER_N=_ROUTER_N,
        LATENT_N=_LATENT_N,
        SHARED_N=_SHARED_N,
        HAS_LINEAR_BETA=linear_beta is not None,
        K=_HIDDEN,
        num_warps=_NUM_WARPS,
        waves_per_eu=2,
        num_ctas=cta_m * cta_n,
    )
    return router_out, routed_out, shared_out


__all__ = ["launch_gluon_latent_input_largem_gfx1250"]
