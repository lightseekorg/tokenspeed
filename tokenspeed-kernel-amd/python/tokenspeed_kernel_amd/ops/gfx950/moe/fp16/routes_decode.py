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

"""Direct BF16 expert-route decode GEMMs for gfx950 (at most 16 tokens).

Both stages launch one program per (output tile, route) and read the expert
straight from ``topk_ids``: no sort and no routing metadata. A route is one
``topk_ids`` entry, ``token * TOPK + slot``. The first route that selects an
expert leads it: its program gathers every token routed to that expert (one
row per token, so at most 16 rows) and streams the expert's weight tile once
through a 16-row MFMA. Programs of later routes with the same expert exit, so
each selected weight tile is read once per launch. In stage 2 the program of
a route whose expert is not local (or invalid) writes that route's zero
output instead.

The waves split every K step (a batched MFMA whose batch dimension is the
wave) and reduce through LDS once, like the dense decode GEMV
(``gemm/fp16/gemv.py``). The K loop is unrolled with the next step's loads
issued before the current MFMAs; weights stream with ``.cg``.

  Stage 1: inter[route] = silu(x @ gate.T) * (x @ up.T), gate/up columns
           interleaved inside the tile so SwiGLU pairs live in one lane.
  Stage 2: out[route] = inter[route] @ w2.T, unweighted.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

__all__ = [
    "MAX_TOKENS",
    "gluon_bf16_moe_routes_decode_stage1_gfx950",
    "gluon_bf16_moe_routes_decode_stage2_gfx950",
    "launch_gluon_bf16_moe_routes_decode_gfx950",
]

MAX_TOKENS = 16
_NUM_WARPS = 4
_WAVE_SIZE = 64


def _decode_launch_metadata(grid, kernel, args):
    return {"name": f"{kernel.name} (tokens={args['num_tokens']})"}


@gluon.constexpr_function
def _mfma_layout(num_warps: int):
    return gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[num_warps, 1, 1],
    )


@gluon.constexpr_function
def _weight_layout(block_k: int, num_warps: int):
    # [waves, BLOCK_K, BLOCK_N]: 8 lanes x 16 B read 128 contiguous bytes of
    # one weight row per instruction.
    k_lanes = min(block_k // 8, _WAVE_SIZE)
    return gl.BlockedLayout(
        [1, 8, 1], [1, k_lanes, _WAVE_SIZE // k_lanes], [num_warps, 1, 1], [1, 2, 0]
    )


@gluon.constexpr_function
def _ids_layout(num_warps: int):
    return gl.BlockedLayout([1, 1], [16, 4], [1, num_warps], [1, 0])


@gluon.constexpr_function
def _swiglu_layout(block_n: int, num_warps: int):
    # Gate/up column pairs stay in one lane for the split.
    n_lanes = min(block_n // 2, _WAVE_SIZE)
    return gl.BlockedLayout(
        [1, 2], [_WAVE_SIZE // n_lanes, n_lanes], [num_warps, 1], [1, 0]
    )


@gluon.jit
def _expert_routes(
    ids_ptr, num_tokens, route, TOPK: gl.constexpr, BLOCK_M: gl.constexpr
):
    """Return ``(expert, leader, token_route, selected_routes)`` for ``route``.

    ``expert`` is the global expert of ``route`` and ``leader`` whether
    ``route`` is the first route selecting it. ``token_route`` holds, per
    token, the last route selecting the expert (-1 if none), and
    ``selected_routes`` the ``[BLOCK_M, TOPK]`` routes selecting it (-1
    elsewhere).
    """
    L: gl.constexpr = _ids_layout(gl.num_warps())
    token = gl.arange(0, BLOCK_M, gl.SliceLayout(1, L))
    slot = gl.arange(0, TOPK, gl.SliceLayout(0, L))
    routes = token[:, None] * TOPK + slot[None, :]
    ids = gl.load(ids_ptr + routes, mask=routes < num_tokens * TOPK, other=-1)
    expert = gl.load(ids_ptr + route)
    selected_routes = gl.where(ids == expert, routes, -1)
    first_route = gl.min(
        gl.min(gl.where(selected_routes >= 0, routes, BLOCK_M * TOPK), axis=1),
        axis=0,
    )
    return (
        expert,
        first_route == route,
        gl.max(selected_routes, axis=1),
        selected_routes,
    )


@gluon.jit
def _store_routes(value, out_ptr, selected_routes, col, WIDTH: gl.constexpr):
    """Store each token's ``value`` row to every route that selected the expert.

    A token may select one expert in several slots; each such route is written.
    """
    TOPK: gl.constexpr = selected_routes.shape[1]
    slots = gl.arange(0, TOPK, gl.SliceLayout(0, selected_routes.type.layout))
    for slot in gl.static_range(TOPK):
        route = gl.max(gl.where(slots[None, :] == slot, selected_routes, -1), axis=1)
        route = gl.convert_layout(route, gl.SliceLayout(1, value.type.layout))
        gl.amd.cdna4.buffer_store(
            value,
            out_ptr,
            route[:, None] * WIDTH + col[None, :],
            mask=(route >= 0)[:, None] & (col < WIDTH)[None, :],
        )


@gluon.jit
def _route_gemm(
    a_ptr,
    a_rows,
    w_ptr,
    w_rows,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    """``a[a_rows] @ w[w_rows].T`` over K; rows < 0 of ``a`` read zeros.

    ``a_rows`` is a ``[BLOCK_M]`` vector of any layout, ``w_rows`` a
    ``[BLOCK_N]`` vector in the weight-load column layout.
    """
    WAVES: gl.constexpr = gl.num_warps()
    K_STEP: gl.constexpr = WAVES * BLOCK_K
    gl.static_assert(K % K_STEP == 0, "K must be a multiple of waves * BLOCK_K")
    NUM_K_STEPS: gl.constexpr = K // K_STEP
    mfma: gl.constexpr = _mfma_layout(WAVES)
    a_l: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8)
    b_l: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8)
    WL: gl.constexpr = _weight_layout(BLOCK_K, WAVES)
    w_wave = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, WL)))
    w_k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gl.SliceLayout(2, WL)))
    w_offs = (w_wave[:, None, None] * BLOCK_K + w_k[None, :, None]) + w_rows[
        None, None, :
    ] * K
    w = gl.amd.cdna4.buffer_load(w_ptr, w_offs, cache=".cg")
    a_wave = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, a_l)))
    a_k = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gl.SliceLayout(1, a_l)))
    rows = gl.convert_layout(a_rows, gl.SliceLayout(0, gl.SliceLayout(2, a_l)))
    ak = a_wave[:, None, None] * BLOCK_K + a_k[None, None, :]
    a_mask = (rows >= 0)[None, :, None] & (ak < K)
    a_offs = rows[None, :, None] * K + ak
    a = gl.amd.cdna4.buffer_load(a_ptr, a_offs, mask=a_mask, other=0.0)
    acc = gl.zeros((WAVES, BLOCK_M, BLOCK_N), gl.float32, mfma)
    # Unrolled: a loop-carried copy of a pending load would wait for it.
    for step in gl.static_range(NUM_K_STEPS):
        if step + 1 < NUM_K_STEPS:
            next_w = gl.amd.cdna4.buffer_load(
                w_ptr, w_offs + (step + 1) * K_STEP, cache=".cg"
            )
            next_a = gl.amd.cdna4.buffer_load(
                a_ptr, a_offs + (step + 1) * K_STEP, mask=a_mask, other=0.0
            )
        acc = gl.amd.cdna4.mfma(
            gl.convert_layout(a, a_l), gl.convert_layout(w, b_l), acc
        )
        if step + 1 < NUM_K_STEPS:
            w = next_w
            a = next_a
    # The waves' K partials, summed through LDS.
    return gl.sum(acc, axis=0)


@gluon.jit(launch_metadata=_decode_launch_metadata, do_not_specialize=("num_tokens",))
def gluon_bf16_moe_routes_decode_stage1_gfx950(
    x_ptr,
    w13_ptr,
    ids_ptr,
    inter_ptr,
    num_tokens,
    expert_start,
    HIDDEN: gl.constexpr,
    INTERMEDIATE: gl.constexpr,
    TOPK: gl.constexpr,
    NUM_EXPERTS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    """Gate/up GEMM + SwiGLU of output columns ``[n_tile * BLOCK_N / 2, ...)``.

    ``x`` is ``[num_tokens, HIDDEN]``, ``w13`` ``[NUM_EXPERTS, 2 *
    INTERMEDIATE, HIDDEN]`` (gate rows first), ``ids`` ``[num_tokens, TOPK]``
    int32 global expert ids and ``inter`` ``[num_tokens * TOPK,
    INTERMEDIATE]``. Rows of routes whose expert is not local are left
    unwritten.
    """
    BLOCK_M: gl.constexpr = 16
    # Tiles vary fastest: consecutive workgroups go to different XCDs, so the
    # few local routes of an expert-parallel rank still spread over all of them.
    n_tile = gl.program_id(0)
    route = gl.program_id(1)
    expert, leader, token_route, selected_routes = _expert_routes(
        ids_ptr, num_tokens, route, TOPK, BLOCK_M
    )
    local_expert = expert - expert_start
    if leader & (local_expert >= 0) & (local_expert < NUM_EXPERTS):
        WL: gl.constexpr = _weight_layout(BLOCK_K, gl.num_warps())
        n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gl.SliceLayout(1, WL)))
        # Even columns are gate rows, odd columns the matching up rows.
        w_rows = (n % 2) * INTERMEDIATE + n_tile * (BLOCK_N // 2) + n // 2
        w_base = w13_ptr + local_expert.to(gl.int64) * (2 * INTERMEDIATE * HIDDEN)
        token = gl.arange(0, BLOCK_M, gl.SliceLayout(1, _ids_layout(gl.num_warps())))
        a_rows = gl.where(token_route >= 0, token, -1)
        acc = _route_gemm(
            x_ptr, a_rows, w_base, w_rows, HIDDEN, BLOCK_M, BLOCK_N, BLOCK_K
        )
        SL: gl.constexpr = _swiglu_layout(BLOCK_N, gl.num_warps())
        acc = gl.convert_layout(acc, SL)
        gate, up = gl.split(acc.reshape((BLOCK_M, BLOCK_N // 2, 2)))
        h = gate * (1.0 / (1.0 + gl.exp(-gate))) * up
        col = n_tile * (BLOCK_N // 2) + gl.arange(
            0, BLOCK_N // 2, gl.SliceLayout(0, h.type.layout)
        )
        _store_routes(
            h.to(inter_ptr.dtype.element_ty),
            inter_ptr,
            selected_routes,
            col,
            INTERMEDIATE,
        )


@gluon.jit(launch_metadata=_decode_launch_metadata, do_not_specialize=("num_tokens",))
def gluon_bf16_moe_routes_decode_stage2_gfx950(
    inter_ptr,
    w2_ptr,
    ids_ptr,
    out_ptr,
    num_tokens,
    expert_start,
    HIDDEN: gl.constexpr,
    INTERMEDIATE: gl.constexpr,
    TOPK: gl.constexpr,
    NUM_EXPERTS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    """Down GEMM of output columns ``[n_tile * BLOCK_N, ...)`` of every route.

    ``w2`` is ``[NUM_EXPERTS, HIDDEN, INTERMEDIATE]`` and ``out``
    ``[num_tokens * TOPK, HIDDEN]``; routes whose expert is not local get
    zeros.
    """
    BLOCK_M: gl.constexpr = 16
    # Tiles vary fastest: consecutive workgroups go to different XCDs, so the
    # few local routes of an expert-parallel rank still spread over all of them.
    n_tile = gl.program_id(0)
    route = gl.program_id(1)
    expert, leader, token_route, selected_routes = _expert_routes(
        ids_ptr, num_tokens, route, TOPK, BLOCK_M
    )
    local_expert = expert - expert_start
    owned = (local_expert >= 0) & (local_expert < NUM_EXPERTS)
    if leader & owned:
        WL: gl.constexpr = _weight_layout(BLOCK_K, gl.num_warps())
        n = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gl.SliceLayout(1, WL)))
        w_base = w2_ptr + local_expert.to(gl.int64) * (HIDDEN * INTERMEDIATE)
        acc = _route_gemm(
            inter_ptr,
            token_route,
            w_base,
            n_tile * BLOCK_N + n,
            INTERMEDIATE,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
        )
        col = n_tile * BLOCK_N + gl.arange(
            0, BLOCK_N, gl.SliceLayout(0, acc.type.layout)
        )
        _store_routes(
            acc.to(out_ptr.dtype.element_ty), out_ptr, selected_routes, col, HIDDEN
        )
    elif not owned:
        ZL: gl.constexpr = gl.BlockedLayout([8], [64], [gl.num_warps()], [0])
        col = n_tile * BLOCK_N + gl.arange(0, BLOCK_N, ZL)
        gl.amd.cdna4.buffer_store(
            gl.zeros((BLOCK_N,), out_ptr.dtype.element_ty, ZL),
            out_ptr,
            route * HIDDEN + col,
        )


def _decode_config(intermediate_size: int) -> tuple[int, int, int, int]:
    """Return ``(stage1 BLOCK_N, BLOCK_K, stage2 BLOCK_N, BLOCK_K)``.

    Stage 2 uses 16 output columns for long reductions (``I >= 2048``), else 32.
    """
    stage2_block_n = 16 if intermediate_size >= 2048 else 32
    stage2_block_k = 128 if intermediate_size % 512 == 0 else 64
    return 32, 64, stage2_block_n, stage2_block_k


def launch_gluon_bf16_moe_routes_decode_gfx950(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_start: int,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write unweighted BF16 expert routes of at most 16 tokens into ``output``.

    Args:
        x: Contiguous BF16 ``[tokens, hidden]`` activations,
            ``1 <= tokens <= 16``.
        w13: Contiguous ``[experts, 2I, hidden]`` gate/up weights (gate rows
            first).
        w2: Contiguous ``[experts, hidden, I]`` down weights.
        topk_ids: Contiguous int32 ``[tokens, top_k]`` global expert ids; a
            token may select one expert in several slots.
        expert_start: Global id of local expert 0.
        output: Contiguous ``[tokens * top_k, hidden]`` BF16 destination.
            Routes of non-local or invalid experts are written as zeros.

    Returns:
        ``output``.
    """
    num_tokens, hidden = x.shape
    num_experts, two_i, _ = w13.shape
    intermediate_size = two_i // 2
    top_k = topk_ids.shape[1]
    stage1_block_n, stage1_block_k, stage2_block_n, stage2_block_k = _decode_config(
        intermediate_size
    )
    inter = torch.empty(
        (num_tokens * top_k, intermediate_size), dtype=x.dtype, device=x.device
    )
    gluon_bf16_moe_routes_decode_stage1_gfx950[
        (2 * intermediate_size // stage1_block_n, num_tokens * top_k)
    ](
        x,
        w13,
        topk_ids,
        inter,
        num_tokens,
        expert_start,
        HIDDEN=hidden,
        INTERMEDIATE=intermediate_size,
        TOPK=top_k,
        NUM_EXPERTS=num_experts,
        BLOCK_N=stage1_block_n,
        BLOCK_K=stage1_block_k,
        num_warps=_NUM_WARPS,
    )
    gluon_bf16_moe_routes_decode_stage2_gfx950[
        (hidden // stage2_block_n, num_tokens * top_k)
    ](
        inter,
        w2,
        topk_ids,
        output,
        num_tokens,
        expert_start,
        HIDDEN=hidden,
        INTERMEDIATE=intermediate_size,
        TOPK=top_k,
        NUM_EXPERTS=num_experts,
        BLOCK_N=stage2_block_n,
        BLOCK_K=stage2_block_k,
        num_warps=_NUM_WARPS,
    )
    return output
