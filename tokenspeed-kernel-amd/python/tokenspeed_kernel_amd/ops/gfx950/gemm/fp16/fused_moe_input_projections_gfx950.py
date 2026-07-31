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

"""Latent-MoE input projections as one MFMA pass over a concatenated weight.

The router, routed-down, and shared gate/up weights are consecutive rows of one
tensor, so a single GEMM feeds all three projections. Their consumers disagree
on dtype: expert selection reads FP32 router logits while the other two take
the activation dtype. A vendor GEMM has one output dtype and would force the
router weight to be upcast, so this kernel keeps one weight stream and branches
the store on which region the column tile falls in.

The activation tile is staged in LDS and reused across the column tile, which
is what lets more than one token ride along on the same weight pass. The SiTU
activation stays in a separate pass so the gate tile and the up tile 768 rows
below it need not be resident together, which would not fit in LDS.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import cdna4_async_copy, gl, gluon, triton

# The MFMA instruction is 16 rows tall, so a decode tile any shorter pays for
# rows it does not have. Past that the weight would be re-read once per token
# tile, so wider batches move to the taller tile instead.
_BLOCK_M = 16
_LARGE_BLOCK_M = 64
_BLOCK_N = 128
_BLOCK_K = 64
_NUM_WARPS = 4
# Every region width is a multiple of 128, so a tile of this width never
# straddles a boundary and the branch stays uniform across the program.
_EPILOGUE_BLOCK_N = 128
_TARGET_PROGRAMS = 512


def _split_k(tokens: int, total_n: int, hidden: int, block_m: int) -> int:
    """Pick the K split that fills the machine without shrinking the K loop."""
    tiles = -(-tokens // block_m) * (total_n // _BLOCK_N)
    k_tiles = hidden // _BLOCK_K
    split = 1
    while split * 2 <= k_tiles and tiles * split < _TARGET_PROGRAMS:
        split *= 2
    return split


@gluon.jit
def _fused_projection_gemm_kernel(
    a_ptr,
    b_ptr,
    partial_ptr,
    tokens,
    split_stride,
    HIDDEN: gl.constexpr,
    TOTAL_N: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SPLIT_K: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    WARPS_M: gl.constexpr,
):
    """Project one token tile against one column tile of the fused weight.

    Only 47 column tiles span the fused weight, so at decode the token axis
    cannot fill the machine on its own. Splitting K spreads each tile over
    SPLIT_K programs, which is what keeps a bandwidth-bound projection off a
    handful of compute units. Each split owns its own partial rather than
    accumulating in place, because every split of a tile targets the same
    columns and would otherwise serialize on those cache lines.
    """
    pid = gl.program_id(axis=0)
    num_pid_n: gl.constexpr = TOTAL_N // BLOCK_N
    pid_n = pid % num_pid_n
    split_id = (pid // num_pid_n) % SPLIT_K
    pid_m = pid // (num_pid_n * SPLIT_K)

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WARPS_M, NUM_WARPS // WARPS_M],
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )
    gload_a: gl.constexpr = gl.BlockedLayout(
        [1, 8], [512 // BLOCK_K, BLOCK_K // 8], [NUM_WARPS, 1], [1, 0]
    )
    gload_b: gl.constexpr = gl.BlockedLayout(
        [8, 1], [BLOCK_K // 8, 512 // BLOCK_K], [1, NUM_WARPS], [0, 1]
    )
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[1, 0])
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(8, 2, 8, order=[0, 1])

    NBUF: gl.constexpr = 2
    smem_a = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [NBUF, BLOCK_M, BLOCK_K], shared_a
    )
    smem_b = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [NBUF, BLOCK_K, BLOCK_N], shared_b
    )

    am_layout: gl.constexpr = gl.SliceLayout(1, gload_a)
    ak_layout: gl.constexpr = gl.SliceLayout(0, gload_a)
    offs_am = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=am_layout)
    offs_ak = gl.arange(0, BLOCK_K, layout=ak_layout)
    token_mask = offs_am < tokens
    a_offsets = (offs_am[:, None] * HIDDEN + offs_ak[None, :]).to(gl.int32)

    # The weight is [rows, hidden] and contiguous, so a column of the GEMM's B
    # operand is a weight row: the k axis is the contiguous one.
    bk_layout: gl.constexpr = gl.SliceLayout(1, gload_b)
    bn_layout: gl.constexpr = gl.SliceLayout(0, gload_b)
    offs_bk = gl.arange(0, BLOCK_K, layout=bk_layout)
    offs_bn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=bn_layout)
    b_offsets = (offs_bn[None, :] * HIDDEN + offs_bk[:, None]).to(gl.int32)

    k_start = split_id * (HIDDEN // SPLIT_K)
    a_base = a_ptr + k_start
    b_base = b_ptr + k_start
    acc = gl.zeros((BLOCK_M, BLOCK_N), gl.float32, mfma_layout)
    num_k: gl.constexpr = HIDDEN // BLOCK_K // SPLIT_K

    cdna4_async_copy.buffer_load_to_shared(
        smem_a.index(0), a_base, a_offsets, mask=token_mask[:, None]
    )
    cdna4_async_copy.buffer_load_to_shared(smem_b.index(0), b_base, b_offsets)
    cdna4_async_copy.commit_group()
    a_base += BLOCK_K
    b_base += BLOCK_K
    for k in range(0, num_k - 1):
        l_idx = k % 2
        g_idx = 1 - l_idx
        cdna4_async_copy.buffer_load_to_shared(
            smem_a.index(g_idx), a_base, a_offsets, mask=token_mask[:, None]
        )
        cdna4_async_copy.buffer_load_to_shared(smem_b.index(g_idx), b_base, b_offsets)
        cdna4_async_copy.commit_group()
        cdna4_async_copy.wait_group(1)
        a = smem_a.index(l_idx).load(dot_a_layout)
        b = smem_b.index(l_idx).load(dot_b_layout)
        acc = gl.amd.cdna3.mfma(a, b, acc)
        a_base += BLOCK_K
        b_base += BLOCK_K
    cdna4_async_copy.wait_group(0)
    l_idx = (num_k - 1) % 2
    a = smem_a.index(l_idx).load(dot_a_layout)
    b = smem_b.index(l_idx).load(dot_b_layout)
    acc = gl.amd.cdna3.mfma(a, b, acc)

    cm_layout: gl.constexpr = gl.SliceLayout(1, mfma_layout)
    cn_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)
    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=cm_layout)
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=cn_layout)
    offsets = (
        split_id * split_stride + offs_cm[:, None] * TOTAL_N + offs_cn[None, :]
    ).to(gl.int32)
    gl.amd.cdna4.buffer_store(
        ptr=partial_ptr,
        offsets=offsets,
        stored_value=acc,
        mask=offs_cm[:, None] < tokens,
    )


@gluon.jit
def _split_epilogue_kernel(
    partial_ptr,
    router_ptr,
    routed_ptr,
    shared_ptr,
    beta,
    linear_beta,
    tokens,
    split_stride,
    SPLIT_K: gl.constexpr,
    TOTAL_N: gl.constexpr,
    ROUTER_N: gl.constexpr,
    LATENT_N: gl.constexpr,
    SHARED_N: gl.constexpr,
    HAS_LINEAR_BETA: gl.constexpr,
    BLOCK_N: gl.constexpr,
):
    """Fan the FP32 accumulator out to each consumer's own dtype.

    Expert selection needs FP32 logits while the routed and shared branches
    take the activation dtype, which is why the projection cannot simply be a
    vendor GEMM with one output dtype.
    """
    row = gl.program_id(axis=0)
    tile = gl.program_id(axis=1)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    offs = tile * BLOCK_N + gl.arange(0, BLOCK_N, layout=layout)
    base = row * TOTAL_N

    if tile * BLOCK_N < ROUTER_N:
        mask = offs < ROUTER_N
        value = gl.zeros([BLOCK_N], gl.float32, layout)
        for split in range(0, SPLIT_K):
            value += gl.load(
                partial_ptr + split * split_stride + base + offs, mask=mask, other=0.0
            )
        gl.store(router_ptr + row * ROUTER_N + offs, value, mask=mask)
        return
    if tile * BLOCK_N < ROUTER_N + LATENT_N:
        mask = offs < ROUTER_N + LATENT_N
        value = gl.zeros([BLOCK_N], gl.float32, layout)
        for split in range(0, SPLIT_K):
            value += gl.load(
                partial_ptr + split * split_stride + base + offs, mask=mask, other=0.0
            )
        column = offs - ROUTER_N
        gl.store(
            routed_ptr + row * LATENT_N + column,
            value.to(routed_ptr.dtype.element_ty),
            mask=mask,
        )
        return

    column = offs - ROUTER_N - LATENT_N
    mask = column < SHARED_N
    gate_base = base + ROUTER_N + LATENT_N + column
    gate_raw = gl.zeros([BLOCK_N], gl.float32, layout)
    up = gl.zeros([BLOCK_N], gl.float32, layout)
    for split in range(0, SPLIT_K):
        offset = split * split_stride + gate_base
        gate_raw += gl.load(partial_ptr + offset, mask=mask, other=0.0)
        up += gl.load(partial_ptr + offset + SHARED_N, mask=mask, other=0.0)
    # Round both projections through bf16 before the FP32 activation so the
    # result matches a materialized gate/up projection.
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


def gluon_fused_moe_input_projections_gfx950(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    concatenated_weight: torch.Tensor,
    *,
    beta: float,
    linear_beta: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project router, routed latent, and shared input from one weight pass.

    Args:
        hidden_states: Contiguous BF16 activation shaped ``[tokens, hidden]``.
        router_weight: Router rows of the concatenated weight.
        routed_weight: Latent-projection rows of the concatenated weight.
        shared_gate_up_weight: Stacked shared gate/up rows of the same weight.
        concatenated_weight: The single tensor the three weights above are
            consecutive row views of.
        beta: Positive SiTU gate clipping scale.
        linear_beta: Optional positive SiTU linear-branch clipping scale.

    Returns:
        FP32 router logits, the BF16 routed latent, and the BF16 activated
        shared-expert input.
    """
    tokens, hidden = hidden_states.shape
    router_n = router_weight.shape[0]
    latent_n = routed_weight.shape[0]
    shared_n = shared_gate_up_weight.shape[0] // 2
    if beta <= 0.0 or (linear_beta is not None and linear_beta <= 0.0):
        raise ValueError("SiTU beta values must be positive")

    device = hidden_states.device
    dtype = hidden_states.dtype
    total_n = router_n + latent_n + 2 * shared_n
    block_m, warps_m = (_BLOCK_M, 1) if tokens <= _BLOCK_M else (_LARGE_BLOCK_M, 2)
    split_k = _split_k(tokens, total_n, hidden, block_m)
    partials = torch.empty(
        (split_k, tokens, total_n), dtype=torch.float32, device=device
    )
    split_stride = tokens * total_n

    grid = (triton.cdiv(tokens, block_m) * split_k * (total_n // _BLOCK_N),)
    _fused_projection_gemm_kernel[grid](
        hidden_states,
        concatenated_weight,
        partials,
        tokens,
        split_stride,
        HIDDEN=hidden,
        TOTAL_N=total_n,
        BLOCK_M=block_m,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        SPLIT_K=split_k,
        NUM_WARPS=_NUM_WARPS,
        WARPS_M=warps_m,
        num_warps=_NUM_WARPS,
    )

    router_out = torch.empty((tokens, router_n), dtype=torch.float32, device=device)
    routed_out = torch.empty((tokens, latent_n), dtype=dtype, device=device)
    shared_out = torch.empty((tokens, shared_n), dtype=dtype, device=device)
    emitted_n = router_n + latent_n + shared_n
    _split_epilogue_kernel[(tokens, triton.cdiv(emitted_n, _EPILOGUE_BLOCK_N))](
        partials,
        router_out,
        routed_out,
        shared_out,
        float(beta),
        1.0 if linear_beta is None else float(linear_beta),
        tokens,
        split_stride,
        SPLIT_K=split_k,
        TOTAL_N=total_n,
        ROUTER_N=router_n,
        LATENT_N=latent_n,
        SHARED_N=shared_n,
        HAS_LINEAR_BETA=linear_beta is not None,
        BLOCK_N=_EPILOGUE_BLOCK_N,
        num_warps=4,
    )
    return router_out, routed_out, shared_out


__all__ = ["gluon_fused_moe_input_projections_gfx950"]
