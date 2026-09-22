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

"""Large-prefill Kimi K3 latent-MoE input projections for gfx950.

The GEMM main loop intentionally follows ``gluon_mm_a16w16_prefill_gfx950``:
an eight-wave 256x256x64, double-buffered MFMA/LDS pipeline.  K3's packed
projection is 128-column aligned rather than 256-column aligned, so the
epilogue independently routes each accumulator half to the FP32 router output
or the BF16 routed/shared outputs.  SiTU remains a second kernel because gate
and up rows are separated in the packed weight.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.largem import (
    largem_get_pids,
    largem_mfma_lds_tile,
)

cdna4 = gl.amd.cdna4

_BLOCK_M = 256
_BLOCK_N = 256
_BLOCK_K = 64
_NUM_WARPS = 8
_WARPS_M = 2
_WARPS_N = 4
_NUM_XCDS = 8
_GROUP_SIZE_M = 4

# SiTU epilogue tile. BLOCK_N divides the 768-wide shared output, so no lane is
# masked along the columns, and the layout gives each lane a dwordx4 of them.
_SITU_BLOCK_M = 8
_SITU_BLOCK_N = 256
_SITU_NUM_WARPS = 4

_K3_HIDDEN = 7168
_K3_ROUTER = 896
_K3_ROUTED = 3584
_K3_SHARED = 768
_K3_SHARED_RAW = 2 * _K3_SHARED
_K3_TOTAL = _K3_ROUTER + _K3_ROUTED + _K3_SHARED_RAW


def _prefill_launch_metadata(grid, kernel, args):
    """Report packed projection work and traffic to Proton."""
    m = args["M"]
    return {
        "name": kernel.name,
        "flops16": 2 * m * _K3_TOTAL * _K3_HIDDEN,
        "bytes": m * _K3_HIDDEN * args["a_ptr"].element_size()
        + _K3_TOTAL * _K3_HIDDEN * args["b_ptr"].element_size()
        + m * _K3_ROUTER * args["router_ptr"].element_size()
        + m * _K3_ROUTED * args["routed_ptr"].element_size()
        + m * _K3_SHARED_RAW * args["shared_raw_ptr"].element_size(),
    }


def _situ_launch_metadata(grid, kernel, args):
    """Report the materialized BF16 SiTU epilogue traffic to Proton."""
    m = args["M"]
    return {
        "name": kernel.name,
        "bytes": m
        * (
            _K3_SHARED_RAW * args["shared_raw_ptr"].element_size()
            + _K3_SHARED * args["shared_ptr"].element_size()
        ),
    }


@gluon.jit
def _store_latent_input_half(
    acc_top,
    acc_bottom,
    n_start,
    pid_m,
    M,
    router_ptr,
    routed_ptr,
    shared_raw_ptr,
    stride_router_m,
    stride_routed_m,
    stride_shared_m,
    offs_cm,
    offs_cn,
    STORE_LAYOUT: gl.constexpr,
    BLOCK_M: gl.constexpr,
    TOTAL_N: gl.constexpr,
    ROUTER_N: gl.constexpr,
    ROUTED_N: gl.constexpr,
):
    """Store one 128-column accumulator half in its consumer's dtype."""
    if n_start >= TOTAL_N:
        return

    row_base = pid_m * BLOCK_M
    # Rows the load clamp duplicated onto the last valid row carry a repeat of
    # that row's projection, so they must not reach memory.
    top_mask = (row_base + offs_cm < M)[:, None]
    bottom_mask = (row_base + BLOCK_M // 2 + offs_cm < M)[:, None]
    if n_start < ROUTER_N:
        top = gl.convert_layout(acc_top, layout=STORE_LAYOUT)
        bottom = gl.convert_layout(acc_bottom, layout=STORE_LAYOUT)
        offsets = offs_cm[:, None] * stride_router_m + offs_cn[None, :]
        top_base = router_ptr + row_base * stride_router_m + n_start
        bottom_base = top_base + (BLOCK_M // 2) * stride_router_m
        cdna4.buffer_store(
            ptr=top_base, offsets=offsets, stored_value=top, mask=top_mask
        )
        cdna4.buffer_store(
            ptr=bottom_base, offsets=offsets, stored_value=bottom, mask=bottom_mask
        )
        return

    if n_start < ROUTER_N + ROUTED_N:
        column = n_start - ROUTER_N
        top = gl.convert_layout(
            acc_top.to(routed_ptr.dtype.element_ty), layout=STORE_LAYOUT
        )
        bottom = gl.convert_layout(
            acc_bottom.to(routed_ptr.dtype.element_ty), layout=STORE_LAYOUT
        )
        offsets = offs_cm[:, None] * stride_routed_m + offs_cn[None, :]
        top_base = routed_ptr + row_base * stride_routed_m + column
        bottom_base = top_base + (BLOCK_M // 2) * stride_routed_m
        cdna4.buffer_store(
            ptr=top_base, offsets=offsets, stored_value=top, mask=top_mask
        )
        cdna4.buffer_store(
            ptr=bottom_base, offsets=offsets, stored_value=bottom, mask=bottom_mask
        )
        return

    column = n_start - ROUTER_N - ROUTED_N
    top = gl.convert_layout(
        acc_top.to(shared_raw_ptr.dtype.element_ty), layout=STORE_LAYOUT
    )
    bottom = gl.convert_layout(
        acc_bottom.to(shared_raw_ptr.dtype.element_ty), layout=STORE_LAYOUT
    )
    offsets = offs_cm[:, None] * stride_shared_m + offs_cn[None, :]
    top_base = shared_raw_ptr + row_base * stride_shared_m + column
    bottom_base = top_base + (BLOCK_M // 2) * stride_shared_m
    cdna4.buffer_store(ptr=top_base, offsets=offsets, stored_value=top, mask=top_mask)
    cdna4.buffer_store(
        ptr=bottom_base, offsets=offsets, stored_value=bottom, mask=bottom_mask
    )


@gluon.jit(launch_metadata=_prefill_launch_metadata)
def gluon_latent_input_prefill_gfx950(
    a_ptr,
    b_ptr,
    router_ptr,
    routed_ptr,
    shared_raw_ptr,
    M,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_router_m,
    stride_routed_m,
    stride_shared_m,
    TOTAL_N: gl.constexpr,
    ROUTER_N: gl.constexpr,
    ROUTED_N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    """Project K3's packed BF16 weight and route its three output regions."""
    pid_m, pid_n = largem_get_pids(
        M, TOTAL_N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M
    )
    row_base = pid_m * BLOCK_M
    col_base = pid_n * BLOCK_N

    right_delta = BLOCK_N // 2
    if col_base + right_delta >= TOTAL_N:
        # K3 has 47 half-tiles.  Park the unused right half on the last valid
        # one rather than issuing an out-of-bounds async copy.
        right_delta = 0

    acc_tl, acc_bl, acc_tr, acc_br = largem_mfma_lds_tile(
        a_ptr,
        b_ptr,
        row_base,
        col_base,
        # A partial row tile clamps onto the last valid row; the stores below
        # mask those duplicated rows back out.
        M - 1 - row_base,
        right_delta,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        WARPS_M=WARPS_M,
        WARPS_N=WARPS_N,
    )

    gStoreLayoutC: gl.constexpr = gl.BlockedLayout(
        [4, 8], [4, 16], [WARPS_M, WARPS_N], [1, 0]
    )
    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))

    _store_latent_input_half(
        acc_tl,
        acc_bl,
        pid_n * BLOCK_N,
        pid_m,
        M,
        router_ptr,
        routed_ptr,
        shared_raw_ptr,
        stride_router_m,
        stride_routed_m,
        stride_shared_m,
        offs_cm,
        offs_cn,
        STORE_LAYOUT=gStoreLayoutC,
        BLOCK_M=BLOCK_M,
        TOTAL_N=TOTAL_N,
        ROUTER_N=ROUTER_N,
        ROUTED_N=ROUTED_N,
    )

    _store_latent_input_half(
        acc_tr,
        acc_br,
        pid_n * BLOCK_N + BLOCK_N // 2,
        pid_m,
        M,
        router_ptr,
        routed_ptr,
        shared_raw_ptr,
        stride_router_m,
        stride_routed_m,
        stride_shared_m,
        offs_cm,
        offs_cn,
        STORE_LAYOUT=gStoreLayoutC,
        BLOCK_M=BLOCK_M,
        TOTAL_N=TOTAL_N,
        ROUTER_N=ROUTER_N,
        ROUTED_N=ROUTED_N,
    )


@gluon.jit(launch_metadata=_situ_launch_metadata)
def gluon_latent_input_prefill_situ_gfx950(
    shared_raw_ptr,
    shared_ptr,
    beta,
    inv_beta,
    linear_beta,
    inv_linear_beta,
    M,
    stride_raw_m,
    stride_shared_m,
    SHARED_N: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HAS_LINEAR_BETA: gl.constexpr,
):
    """Apply SiTU to one tile of the materialized BF16 gate/up projection.

    The tile is two dimensional so the column extent can divide the 768-wide
    shared width while each lane still holds 8 contiguous BF16 columns, which
    is one dwordx4.
    """
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 4], [1, 0])
    rows = gl.arange(0, BLOCK_M, gl.SliceLayout(1, layout))
    cols = gl.arange(0, BLOCK_N, gl.SliceLayout(0, layout))
    row_base = gl.program_id(0) * BLOCK_M
    col_base = gl.program_id(1) * BLOCK_N
    # Columns always land inside the tile; only the last row tile is partial.
    mask = (row_base + rows < M)[:, None]

    raw_base = shared_raw_ptr + row_base * stride_raw_m + col_base
    gate_offsets = rows[:, None] * stride_raw_m + cols[None, :]
    gate_raw = cdna4.buffer_load(
        ptr=raw_base, offsets=gate_offsets, mask=mask, other=0.0
    ).to(gl.float32)
    up = cdna4.buffer_load(
        ptr=raw_base + SHARED_N, offsets=gate_offsets, mask=mask, other=0.0
    ).to(gl.float32)
    # Both clamps scale by a reciprocal the launcher computed, so only the
    # sigmoid still expands to the div_scale/div_fmas/div_fixup sequence.
    gate = beta * gl.extra.libdevice.tanh(gate_raw * inv_beta)
    gate *= 1.0 / (1.0 + gl.exp(-gate_raw))
    if HAS_LINEAR_BETA:
        up = linear_beta * gl.extra.libdevice.tanh(up * inv_linear_beta)
    cdna4.buffer_store(
        ptr=shared_ptr + row_base * stride_shared_m + col_base,
        offsets=rows[:, None] * stride_shared_m + cols[None, :],
        stored_value=(gate * up).to(shared_ptr.dtype.element_ty),
        mask=mask,
    )


def launch_gluon_latent_input_prefill_gfx950(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    packed_weight: torch.Tensor,
    *,
    beta: float,
    linear_beta: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project the K3 prefill input from one packed weight pass.

    Args:
        hidden_states: Contiguous BF16 activation shaped ``[tokens, 7168]``.
            Any positive ``tokens`` works; a partial final row tile is masked.
            Automatic dispatch uses this kernel from 4096 tokens.
        router_weight: Packed weight view shaped ``[896, 7168]``.
        routed_weight: Packed weight view shaped ``[3584, 7168]``.
        shared_gate_up_weight: Packed weight view shaped ``[1536, 7168]``.
        packed_weight: Consecutive row view covering all three weights.
        beta: Positive SiTU gate clipping scale.
        linear_beta: Optional positive SiTU linear-branch clipping scale.

    Returns:
        FP32 router logits, BF16 routed latent, and BF16 shared-expert input.
    """
    expected = (
        (router_weight, (_K3_ROUTER, _K3_HIDDEN), "router weight"),
        (routed_weight, (_K3_ROUTED, _K3_HIDDEN), "routed weight"),
        (
            shared_gate_up_weight,
            (_K3_SHARED_RAW, _K3_HIDDEN),
            "shared gate/up weight",
        ),
        (packed_weight, (_K3_TOTAL, _K3_HIDDEN), "packed weight"),
    )
    if (
        hidden_states.ndim != 2
        or hidden_states.shape[1] != _K3_HIDDEN
        or hidden_states.shape[0] < 1
    ):
        raise ValueError(
            "Kimi K3 prefill hidden states must have shape [M, 7168] with M >= 1"
        )
    for tensor, shape, name in expected:
        if tuple(tensor.shape) != shape:
            raise ValueError(f"Kimi K3 {name} must have shape {shape}")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"Kimi K3 {name} must be BF16")
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(f"Kimi K3 {name} must be contiguous on GPU")
        if tensor.device != hidden_states.device:
            raise ValueError("Kimi K3 input projection tensors must be colocated")
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError("Kimi K3 prefill hidden states must be BF16")
    if not hidden_states.is_cuda or not hidden_states.is_contiguous():
        raise ValueError("Kimi K3 prefill hidden states must be contiguous on GPU")
    if beta <= 0.0 or (linear_beta is not None and linear_beta <= 0.0):
        raise ValueError("SiTU beta values must be positive")

    tokens = hidden_states.shape[0]
    device = hidden_states.device
    router_out = torch.empty((tokens, _K3_ROUTER), dtype=torch.float32, device=device)
    routed_out = torch.empty((tokens, _K3_ROUTED), dtype=torch.bfloat16, device=device)
    shared_raw = torch.empty(
        (tokens, _K3_SHARED_RAW), dtype=torch.bfloat16, device=device
    )
    shared_out = torch.empty((tokens, _K3_SHARED), dtype=torch.bfloat16, device=device)

    grid_mn = triton.cdiv(tokens, _BLOCK_M) * triton.cdiv(_K3_TOTAL, _BLOCK_N)
    gluon_latent_input_prefill_gfx950[(grid_mn,)](
        hidden_states,
        packed_weight,
        router_out,
        routed_out,
        shared_raw,
        tokens,
        hidden_states.stride(0),
        hidden_states.stride(1),
        packed_weight.stride(1),
        packed_weight.stride(0),
        router_out.stride(0),
        routed_out.stride(0),
        shared_raw.stride(0),
        TOTAL_N=_K3_TOTAL,
        ROUTER_N=_K3_ROUTER,
        ROUTED_N=_K3_ROUTED,
        K=_K3_HIDDEN,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        WARPS_M=_WARPS_M,
        WARPS_N=_WARPS_N,
        GRID_MN=grid_mn,
        NUM_XCDS=_NUM_XCDS,
        GROUP_SIZE_M=_GROUP_SIZE_M,
        num_warps=_NUM_WARPS,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
    )
    situ_grid = (triton.cdiv(tokens, _SITU_BLOCK_M), _K3_SHARED // _SITU_BLOCK_N)
    gluon_latent_input_prefill_situ_gfx950[situ_grid](
        shared_raw,
        shared_out,
        float(beta),
        1.0 / float(beta),
        1.0 if linear_beta is None else float(linear_beta),
        1.0 if linear_beta is None else 1.0 / float(linear_beta),
        tokens,
        shared_raw.stride(0),
        shared_out.stride(0),
        SHARED_N=_K3_SHARED,
        BLOCK_M=_SITU_BLOCK_M,
        BLOCK_N=_SITU_BLOCK_N,
        HAS_LINEAR_BETA=linear_beta is not None,
        num_warps=_SITU_NUM_WARPS,
    )
    return router_out, routed_out, shared_out


__all__ = ["launch_gluon_latent_input_prefill_gfx950"]
