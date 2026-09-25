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

"""Medium-M packed BF16 latent input projection on gfx950.

One eight-wave workgroup computes a 128x128 output tile. The shared layout
helpers from the dense16 GEMM give vectorized global-to-LDS copies and padded
LDS rows for native BF16 MFMA. Three K buffers keep two 64-wide copies in
flight, and a two-stage warp pipeline alternates the MFMA block with the next
tile's LDS reads and copy issue. The column tile divides all K3 output regions,
so one packed weight pass can store router logits as FP32 and the routed
projection as BF16. Each shared-expert tile reads 64 gate rows and the matching
64 up rows, so SiTU is applied in the epilogue and no gate/up intermediate or
second launch is needed.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.mm import (
    _mfma_lds_gload_layout_a,
    _mfma_lds_gload_layout_b,
    _mfma_lds_shared_layout_a,
    _mfma_lds_shared_layout_b,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16.latent_input_largem import (
    validate_k3_latent_input_gfx950,
)

cdna4 = gl.amd.cdna4
async_copy = cdna4.async_copy

_K3_HIDDEN = 7168
_K3_ROUTER = 896
_K3_ROUTED = 3584
_K3_SHARED = 768
_K3_TOTAL = _K3_ROUTER + _K3_ROUTED + 2 * _K3_SHARED
_BLOCK_N = 128
_BLOCK_K = 64
_BLOCK_M = 128
_WARPS_M = 2
_WARPS_N = 4
_NUM_BUFFERS = 3


def _mediumm_launch_metadata(grid, kernel, args):
    """Report packed projection work and traffic to Proton."""
    m = args["M"]
    return {
        "name": kernel.name,
        "flops16": 2 * m * _K3_TOTAL * _K3_HIDDEN,
        "bytes": m * _K3_HIDDEN * args["a_ptr"].element_size()
        + _K3_TOTAL * _K3_HIDDEN * args["b_ptr"].element_size()
        + m * _K3_ROUTER * args["router_ptr"].element_size()
        + m * _K3_ROUTED * args["routed_ptr"].element_size()
        + m * _K3_SHARED * args["shared_ptr"].element_size(),
    }


@gluon.jit(launch_metadata=_mediumm_launch_metadata)
def gluon_latent_input_mediumm_gfx950(
    a_ptr,
    b_ptr,
    router_ptr,
    routed_ptr,
    shared_ptr,
    beta,
    inv_beta,
    linear_beta,
    inv_linear_beta,
    M,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_router_m,
    stride_routed_m,
    stride_shared_m,
    K: gl.constexpr,
    ROUTER_N: gl.constexpr,
    ROUTED_N: gl.constexpr,
    SHARED_N: gl.constexpr,
    HAS_LINEAR_BETA: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
):
    """Compute one packed output tile with BF16 MFMA and mixed-dtype stores."""
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    num_warps: gl.constexpr = WARPS_M * WARPS_N
    gload_a: gl.constexpr = _mfma_lds_gload_layout_a(BLOCK_M, BLOCK_K, num_warps)
    gload_b: gl.constexpr = _mfma_lds_gload_layout_b(BLOCK_N, BLOCK_K, num_warps)
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WARPS_M, WARPS_N],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mfma_layout, k_width=8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mfma_layout, k_width=8)
    shared_a: gl.constexpr = _mfma_lds_shared_layout_a(
        dot_a, BLOCK_M, BLOCK_K, a_ptr.dtype.element_ty
    )
    shared_b: gl.constexpr = _mfma_lds_shared_layout_b(
        dot_b, BLOCK_N, BLOCK_K, b_ptr.dtype.element_ty
    )
    smem_a = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_M, BLOCK_K], shared_a
    )
    smem_b = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [NUM_BUFFERS, BLOCK_K, BLOCK_N], shared_b
    )

    offs_am = gl.arange(0, BLOCK_M, gl.SliceLayout(1, gload_a))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gload_a))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gload_b))
    offs_bn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, gload_b))
    # Rows past the end of the activation reread the last valid row, so the
    # copies need no mask; the epilogue masks those rows out of the stores.
    a_rows = gl.minimum(pid_m * BLOCK_M + offs_am, M - 1)
    a_offsets = (a_rows[:, None] * stride_am + offs_ak[None, :] * stride_ak).to(
        gl.int32
    )
    n_start = pid_n * BLOCK_N
    # A shared-expert tile pairs 64 gate rows with the matching 64 up rows:
    # tile columns [0, 64) read gate rows and [64, 128) the up rows SHARED_N
    # further on, so SiTU can combine them in the epilogue.
    shared_base: gl.constexpr = ROUTER_N + ROUTED_N
    half_n: gl.constexpr = BLOCK_N // 2
    b_rows = n_start + offs_bn
    if n_start >= shared_base:
        b_rows = (
            shared_base
            + (n_start - shared_base) // 2
            + (offs_bn // half_n) * SHARED_N
            + offs_bn % half_n
        )
    b_offsets = (b_rows[None, :] * stride_bn + offs_bk[:, None] * stride_bk).to(
        gl.int32
    )
    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma_layout)

    k_tiles: gl.constexpr = K // BLOCK_K
    gl.static_assert(
        k_tiles >= NUM_BUFFERS, "the K pipeline needs a tile for every buffer"
    )
    a_kstep = BLOCK_K * stride_ak
    b_kstep = BLOCK_K * stride_bk

    # Fill every buffer; each commit group holds one K tile of A and B.
    for slot in gl.static_range(NUM_BUFFERS):
        async_copy.buffer_load_to_shared(smem_a.index(slot), a_ptr, a_offsets)
        async_copy.buffer_load_to_shared(smem_b.index(slot), b_ptr, b_offsets)
        async_copy.commit_group()
        a_offsets += a_kstep
        b_offsets += b_kstep
    async_copy.wait_group(NUM_BUFFERS - 1)
    a = async_copy.load_shared_relaxed(smem_a.index(0), dot_a)
    b = async_copy.load_shared_relaxed(smem_b.index(0), dot_b)

    # Each step multiplies the operands already in registers while the other
    # stage reads the next tile from LDS and refills the slot just consumed,
    # so NUM_BUFFERS - 1 copies stay in flight behind the MFMA block.
    main_rounds: gl.constexpr = (k_tiles - NUM_BUFFERS) // NUM_BUFFERS
    for _ in range(0, main_rounds):
        for slot in gl.static_range(NUM_BUFFERS):
            async_copy.wait_group(NUM_BUFFERS - 2)
            with gl.amd.warp_pipeline_stage("mfma", priority=0):
                acc = cdna4.mfma(a, b, acc)
            with gl.amd.warp_pipeline_stage("mem", priority=1):
                a = async_copy.load_shared_relaxed(
                    smem_a.index((slot + 1) % NUM_BUFFERS), dot_a
                )
                b = async_copy.load_shared_relaxed(
                    smem_b.index((slot + 1) % NUM_BUFFERS), dot_b
                )
                async_copy.buffer_load_to_shared(smem_a.index(slot), a_ptr, a_offsets)
                async_copy.buffer_load_to_shared(smem_b.index(slot), b_ptr, b_offsets)
                async_copy.commit_group()
                a_offsets += a_kstep
                b_offsets += b_kstep

    # Drain the tiles already in LDS: slot 0 is in registers, the rest follow.
    async_copy.wait_group(0)
    for slot in gl.static_range(1, NUM_BUFFERS):
        acc = cdna4.mfma(a, b, acc)
        a = async_copy.load_shared_relaxed(smem_a.index(slot), dot_a)
        b = async_copy.load_shared_relaxed(smem_b.index(slot), dot_b)
    acc = cdna4.mfma(a, b, acc)

    # K tiles left over after whole rounds run unpipelined.
    tail_tiles: gl.constexpr = k_tiles - (main_rounds + 1) * NUM_BUFFERS
    for slot in gl.static_range(tail_tiles):
        async_copy.buffer_load_to_shared(smem_a.index(slot), a_ptr, a_offsets)
        async_copy.buffer_load_to_shared(smem_b.index(slot), b_ptr, b_offsets)
        async_copy.commit_group()
        a_offsets += a_kstep
        b_offsets += b_kstep
        async_copy.wait_group(0)
        a = async_copy.load_shared_relaxed(smem_a.index(slot), dot_a)
        b = async_copy.load_shared_relaxed(smem_b.index(slot), dot_b)
        acc = cdna4.mfma(a, b, acc)

    if n_start < ROUTER_N:
        # Each lane already holds four consecutive FP32 columns: one dwordx4.
        offs_cm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, mfma_layout))
        offs_cn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, mfma_layout))
        cdna4.buffer_store(
            ptr=router_ptr + pid_m * BLOCK_M * stride_router_m + n_start,
            offsets=offs_cm[:, None] * stride_router_m + offs_cn[None, :],
            stored_value=acc,
            mask=(pid_m * BLOCK_M + offs_cm < M)[:, None],
        )
    elif n_start < shared_base:
        # Four consecutive BF16 columns are only a dwordx2, so regroup the
        # tile to eight columns per lane for dwordx4 stores.
        store_layout: gl.constexpr = gl.BlockedLayout(
            [4, 8], [4, 16], [num_warps, 1], [1, 0]
        )
        gl.static_assert(
            BLOCK_M == 4 * 4 * num_warps and BLOCK_N == 8 * 16,
            "the BF16 store layout covers exactly one output tile",
        )
        value = gl.convert_layout(acc.to(gl.bfloat16), store_layout)
        offs_sm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, store_layout))
        offs_sn = gl.arange(0, BLOCK_N, gl.SliceLayout(0, store_layout))
        cdna4.buffer_store(
            ptr=routed_ptr + pid_m * BLOCK_M * stride_routed_m + n_start - ROUTER_N,
            offsets=offs_sm[:, None] * stride_routed_m + offs_sn[None, :],
            stored_value=value,
            mask=(pid_m * BLOCK_M + offs_sm < M)[:, None],
        )
    else:
        # The MFMA layout repeats every WARPS_N * 16 columns, so each lane
        # holds gate column c and up column c + 64 in its own registers and
        # the split below pairs them without moving data between lanes.
        gl.static_assert(WARPS_N * 16 == half_n, "gate and up halves must share lanes")
        gate, up = gl.split(
            gl.permute(gl.reshape(acc, [BLOCK_M, 2, half_n]), [0, 2, 1])
        )
        # Round to BF16 first, matching the unfused projection's output.
        gate = gate.to(gl.bfloat16).to(gl.float32)
        up = up.to(gl.bfloat16).to(gl.float32)
        situ_gate = beta * gl.extra.libdevice.tanh(gate * inv_beta)
        situ_gate *= 1.0 / (1.0 + gl.exp(-gate))
        if HAS_LINEAR_BETA:
            up = linear_beta * gl.extra.libdevice.tanh(up * inv_linear_beta)
        shared_layout: gl.constexpr = gl.BlockedLayout(
            [2, 8], [8, 8], [num_warps, 1], [1, 0]
        )
        gl.static_assert(
            BLOCK_M == 2 * 8 * num_warps and half_n == 8 * 8,
            "the SiTU store layout covers exactly one output tile",
        )
        shared = gl.convert_layout((situ_gate * up).to(gl.bfloat16), shared_layout)
        offs_hm = gl.arange(0, BLOCK_M, gl.SliceLayout(1, shared_layout))
        offs_hn = gl.arange(0, half_n, gl.SliceLayout(0, shared_layout))
        cdna4.buffer_store(
            ptr=shared_ptr
            + pid_m * BLOCK_M * stride_shared_m
            + (n_start - shared_base) // 2,
            offsets=offs_hm[:, None] * stride_shared_m + offs_hn[None, :],
            stored_value=shared,
            mask=(pid_m * BLOCK_M + offs_hm < M)[:, None],
        )


def launch_gluon_latent_input_mediumm_gfx950(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    packed_weight: torch.Tensor,
    *,
    beta: float,
    linear_beta: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project K3's packed weight and apply SiTU.

    Args:
        hidden_states: Contiguous BF16 input shaped ``[M, 7168]``.
        router_weight: Consecutive packed router rows.
        routed_weight: Consecutive packed latent rows.
        shared_gate_up_weight: Consecutive packed shared gate/up rows.
        packed_weight: View covering the three weight regions.
        beta: Positive gate clamp.
        linear_beta: Optional positive up clamp.

    Returns:
        FP32 router logits, BF16 routed latent, and BF16 shared input.
    """
    validate_k3_latent_input_gfx950(
        hidden_states,
        router_weight,
        routed_weight,
        shared_gate_up_weight,
        packed_weight,
        beta=beta,
        linear_beta=linear_beta,
    )
    m = hidden_states.shape[0]
    device = hidden_states.device
    router = torch.empty((m, _K3_ROUTER), dtype=torch.float32, device=device)
    routed = torch.empty((m, _K3_ROUTED), dtype=torch.bfloat16, device=device)
    shared = torch.empty((m, _K3_SHARED), dtype=torch.bfloat16, device=device)
    gluon_latent_input_mediumm_gfx950[
        (triton.cdiv(m, _BLOCK_M), _K3_TOTAL // _BLOCK_N)
    ](
        hidden_states,
        packed_weight,
        router,
        routed,
        shared,
        float(beta),
        1.0 / float(beta),
        1.0 if linear_beta is None else float(linear_beta),
        1.0 if linear_beta is None else 1.0 / float(linear_beta),
        m,
        hidden_states.stride(0),
        hidden_states.stride(1),
        packed_weight.stride(1),
        packed_weight.stride(0),
        router.stride(0),
        routed.stride(0),
        shared.stride(0),
        K=_K3_HIDDEN,
        ROUTER_N=_K3_ROUTER,
        ROUTED_N=_K3_ROUTED,
        SHARED_N=_K3_SHARED,
        HAS_LINEAR_BETA=linear_beta is not None,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        WARPS_M=_WARPS_M,
        WARPS_N=_WARPS_N,
        NUM_BUFFERS=_NUM_BUFFERS,
        num_warps=_WARPS_M * _WARPS_N,
        llvm_fn_attrs=(("amdgpu-agpr-alloc", "0,0"),),
    )
    return router, routed, shared


__all__ = ["launch_gluon_latent_input_mediumm_gfx950"]
