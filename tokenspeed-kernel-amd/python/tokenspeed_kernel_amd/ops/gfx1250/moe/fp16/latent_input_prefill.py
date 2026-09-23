# Copyright (c) 2026 LightSeek Foundation

"""Prefill latent-MoE input projections for gfx1250.

Compute-bound at these lengths, the opposite of the decode kernel next door,
so this runs the wide-N schedule from ``gfx1250/gemm/fp16/mm.py`` with no
split reduction and fans the accumulator out by region. The FP32 router
logits therefore reach memory without a round trip through BF16.

SiTU is a second launch, as on gfx950: a tile spans 256 contiguous columns
while the gate and up halves sit 768 apart, so no tile holds both.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

from ...gemm.fp16.mm import _WARP_BASES_8, _largem_swizzle2d

_HIDDEN = 7168
_ROUTER_N = 896
_LATENT_N = 3584
_SHARED_N = 768
_TOTAL_N = _ROUTER_N + _LATENT_N + 2 * _SHARED_N

_BLOCK_M = 256
_BLOCK_N = 256
_BLOCK_K = 128
_NUM_BUFFERS = 2
_NUM_WARPS = 8
_GROUP_M = 8
_SITU_BLOCK = 256


@gluon.jit
def _gluon_latent_input_prefill_gfx1250_kernel(
    a_ptr,
    b_ptr,
    router_ptr,
    routed_ptr,
    shared_raw_ptr,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    M,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    WARP_BASES: gl.constexpr,
    ROUTER_N: gl.constexpr,
    LATENT_N: gl.constexpr,
    SHARED_N: gl.constexpr,
    TOTAL_N: gl.constexpr,
    K: gl.constexpr,
):
    """One output tile of the packed projection, fanned out by region."""

    gl.static_assert(NUM_BUFFERS == 2, "candidate uses a double-buffer TDM pipeline")

    pid = gl.program_id(0)
    grid_m = gl.cdiv(M, BLOCK_M)
    grid_n: gl.constexpr = (TOTAL_N + BLOCK_N - 1) // BLOCK_N
    pid_m, pid_n = _largem_swizzle2d(pid, grid_m, grid_n, GROUP_M)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

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
        [[256, 8]], [BLOCK_N, BLOCK_K], [1, 0]
    )

    a_smem = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], shared_layout_a
    )
    b_smem = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_N, BLOCK_K], shared_layout_b
    )
    a_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=a_ptr + off_m * stride_am,
        shape=(M - off_m, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=shared_layout_a,
    )
    b_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
        base=b_ptr + off_n * stride_bn,
        shape=(TOTAL_N - off_n, K),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=shared_layout_b,
    )

    gl.amd.cdna5.tdm.async_load(a_desc, [0, 0], a_smem.index(0))
    gl.amd.cdna5.tdm.async_load(b_desc, [0, 0], b_smem.index(0))

    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, wmma_layout)
    num_k_tiles: gl.constexpr = K // BLOCK_K
    for i in range(0, num_k_tiles - 1):
        nxt = i + 1
        gl.amd.cdna5.tdm.async_load(
            a_desc, [0, nxt * BLOCK_K], a_smem.index(nxt % NUM_BUFFERS)
        )
        gl.amd.cdna5.tdm.async_load(
            b_desc, [0, nxt * BLOCK_K], b_smem.index(nxt % NUM_BUFFERS)
        )
        gl.amd.cdna5.tdm.async_wait(2)
        a = a_smem.index(i % NUM_BUFFERS).load(layout=dot_layout_a)
        b = b_smem.index(i % NUM_BUFFERS).permute([1, 0]).load(layout=dot_layout_b)
        acc = gl.amd.cdna5.wmma(a, b, acc)

    gl.amd.cdna5.tdm.async_wait(0)
    last = num_k_tiles - 1
    a = a_smem.index(last % NUM_BUFFERS).load(layout=dot_layout_a)
    b = b_smem.index(last % NUM_BUFFERS).permute([1, 0]).load(layout=dot_layout_b)
    acc = gl.amd.cdna5.wmma(a, b, acc)

    rows = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, wmma_layout))
    cols = off_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, wmma_layout))
    live = rows[:, None] < M

    # One masked store per region. A tile can straddle a boundary, because the
    # regions are 128-column aligned while the tile is 256 wide, so the masks
    # rather than the tile index decide where a column belongs.
    in_router = live & (cols[None, :] < ROUTER_N)
    gl.amd.cdna5.buffer_store(
        acc,
        router_ptr,
        (rows[:, None] * ROUTER_N + cols[None, :]).to(gl.int32),
        mask=in_router,
    )

    routed_col = cols - ROUTER_N
    in_routed = (
        live & (cols[None, :] >= ROUTER_N) & (cols[None, :] < ROUTER_N + LATENT_N)
    )
    gl.amd.cdna5.buffer_store(
        acc.to(routed_ptr.dtype.element_ty),
        routed_ptr,
        (rows[:, None] * LATENT_N + routed_col[None, :]).to(gl.int32),
        mask=in_routed,
    )

    shared_col = cols - ROUTER_N - LATENT_N
    in_shared = (
        live & (cols[None, :] >= ROUTER_N + LATENT_N) & (cols[None, :] < TOTAL_N)
    )
    gl.amd.cdna5.buffer_store(
        acc.to(shared_raw_ptr.dtype.element_ty),
        shared_raw_ptr,
        (rows[:, None] * (2 * SHARED_N) + shared_col[None, :]).to(gl.int32),
        mask=in_shared,
    )


@gluon.jit
def _gluon_latent_input_prefill_situ_gfx1250(
    shared_raw_ptr,
    shared_ptr,
    beta,
    linear_beta,
    M,
    SHARED_N: gl.constexpr,
    HAS_LINEAR_BETA: gl.constexpr,
    BLOCK: gl.constexpr,
):
    """Combine the gate and up halves of the shared projection."""

    row = gl.program_id(axis=0)
    tile = gl.program_id(axis=1)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    col = tile * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = col < SHARED_N
    base = row * (2 * SHARED_N) + col
    gate_raw = gl.load(shared_raw_ptr + base, mask=mask, other=0.0).to(gl.float32)
    up = gl.load(shared_raw_ptr + base + SHARED_N, mask=mask, other=0.0).to(gl.float32)
    gate = beta * gl.extra.libdevice.tanh(gate_raw / beta)
    gate *= 1.0 / (1.0 + gl.exp(-gate_raw))
    if HAS_LINEAR_BETA:
        up = linear_beta * gl.extra.libdevice.tanh(up / linear_beta)
    gl.store(
        shared_ptr + row * SHARED_N + col,
        (gate * up).to(shared_ptr.dtype.element_ty),
        mask=mask,
    )


def launch_gluon_latent_input_prefill_gfx1250(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    routed_down_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    packed_weight: torch.Tensor,
    *,
    beta: float,
    linear_beta: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a prefill chunk of Kimi-K3 latent-MoE inputs."""

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
    if beta <= 0.0 or (linear_beta is not None and linear_beta <= 0.0):
        raise ValueError("Kimi K3 SiTU beta values must be positive")

    device = hidden_states.device
    router_out = torch.empty((tokens, _ROUTER_N), dtype=torch.float32, device=device)
    routed_out = torch.empty((tokens, _LATENT_N), dtype=torch.bfloat16, device=device)
    shared_raw = torch.empty(
        (tokens, 2 * _SHARED_N), dtype=torch.bfloat16, device=device
    )
    shared_out = torch.empty((tokens, _SHARED_N), dtype=torch.bfloat16, device=device)

    grid = triton.cdiv(tokens, _BLOCK_M) * triton.cdiv(_TOTAL_N, _BLOCK_N)
    _gluon_latent_input_prefill_gfx1250_kernel[(grid,)](
        hidden_states,
        packed_weight,
        router_out,
        routed_out,
        shared_raw,
        hidden_states.stride(0),
        hidden_states.stride(1),
        packed_weight.stride(0),
        packed_weight.stride(1),
        tokens,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        GROUP_M=_GROUP_M,
        NUM_BUFFERS=_NUM_BUFFERS,
        WARP_BASES=_WARP_BASES_8,
        ROUTER_N=_ROUTER_N,
        LATENT_N=_LATENT_N,
        SHARED_N=_SHARED_N,
        TOTAL_N=_TOTAL_N,
        K=_HIDDEN,
        num_warps=_NUM_WARPS,
        num_stages=1,
    )
    _gluon_latent_input_prefill_situ_gfx1250[
        (tokens, triton.cdiv(_SHARED_N, _SITU_BLOCK))
    ](
        shared_raw,
        shared_out,
        float(beta),
        1.0 if linear_beta is None else float(linear_beta),
        tokens,
        SHARED_N=_SHARED_N,
        HAS_LINEAR_BETA=linear_beta is not None,
        BLOCK=_SITU_BLOCK,
        num_warps=4,
    )
    return router_out, routed_out, shared_out


__all__ = ["launch_gluon_latent_input_prefill_gfx1250"]
