# Copyright (c) 2026 LightSeek Foundation

"""Single-GEMM latent-MoE input projections over one concatenated weight.

The router, routed-down, and shared gate/up projections share an activation
and a reduction width, so a caller that stores them as consecutive rows of one
tensor turns three GEMMs into one. Their outputs differ, though: expert
selection needs FP32 router logits while the other two stay in the activation
dtype. A vendor GEMM has a single output dtype and would force the router
weight to be upcast, so this kernel keeps one weight stream and branches the
epilogue on the column tile instead.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.moe.latent_input import packed_projection_weight_view
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


@triton.jit
def _packed_input_projections_kernel(
    hidden_ptr,
    weight_ptr,
    router_ptr,
    routed_ptr,
    shared_raw_ptr,
    M,
    stride_hm,
    stride_wn,
    K: tl.constexpr,
    ROUTER_N: tl.constexpr,
    ROUTED_N: tl.constexpr,
    SHARED_RAW_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M

    hidden_ptrs = hidden_ptr + offs_m[:, None] * stride_hm + offs_k[None, :]
    weight_ptrs = weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_mask = k_start + offs_k < K
        activation = tl.load(
            hidden_ptrs,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        weight = tl.load(weight_ptrs, mask=k_mask[:, None], other=0.0)
        acc += tl.dot(activation, weight)
        hidden_ptrs += BLOCK_K
        weight_ptrs += BLOCK_K

    # The block width divides every region, so the tile lies wholly inside one
    # of them and the branch is uniform across the program.
    n_start = pid_n * BLOCK_N
    if n_start < ROUTER_N:
        tl.store(
            router_ptr + offs_m[:, None] * ROUTER_N + offs_n[None, :],
            acc,
            mask=m_mask[:, None],
        )
    elif n_start < ROUTER_N + ROUTED_N:
        columns = offs_n - ROUTER_N
        tl.store(
            routed_ptr + offs_m[:, None] * ROUTED_N + columns[None, :],
            acc.to(routed_ptr.dtype.element_ty),
            mask=m_mask[:, None],
        )
    else:
        columns = offs_n - ROUTER_N - ROUTED_N
        tl.store(
            shared_raw_ptr + offs_m[:, None] * SHARED_RAW_N + columns[None, :],
            acc.to(shared_raw_ptr.dtype.element_ty),
            mask=m_mask[:, None],
        )


@triton.jit
def _situ_kernel(
    raw_ptr,
    out_ptr,
    beta,
    linear_beta,
    M,
    SHARED_N: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < SHARED_N
    base = row * 2 * SHARED_N
    gate = tl.load(raw_ptr + base + offs_n, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(raw_ptr + base + SHARED_N + offs_n, mask=mask, other=0.0).to(
        tl.float32
    )
    clamped = beta * tl.extra.libdevice.tanh(gate / beta)
    clamped = clamped * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * tl.extra.libdevice.tanh(up / linear_beta)
    tl.store(
        out_ptr + row * SHARED_N + offs_n,
        (clamped * up).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


_NUM_STAGES = 3

# (max tokens, BLOCK_M, BLOCK_N, BLOCK_K, num_warps), measured on gfx950. Few
# tokens leave the GEMM bandwidth-bound on one weight pass, so the schedule
# narrows the column tile to spread the weight across more workgroups rather
# than widening the tile for reuse that does not exist yet.
_SCHEDULE = (
    (16, 16, 64, 256, 4),
    (32, 32, 32, 256, 4),
    (64, 64, 32, 256, 4),
    (128, 64, 64, 128, 4),
    (256, 128, 64, 64, 8),
    (512, 128, 128, 64, 8),
)
_LARGE_SCHEDULE = (256, 128, 64, 8)


def _schedule(tokens: int) -> tuple[int, int, int, int]:
    """Return the ``(BLOCK_M, BLOCK_N, BLOCK_K, num_warps)`` tiling for a size."""
    for limit, block_m, block_n, block_k, warps in _SCHEDULE:
        if tokens <= limit:
            return block_m, block_n, block_k, warps
    return _LARGE_SCHEDULE


@register_kernel(
    "moe",
    "latent_input",
    name="triton_latent_input_packed",
    solution="triton",
    signatures=frozenset(
        {
            format_signature(
                hidden_states=dense_tensor_format(torch.bfloat16),
                router_weight=dense_tensor_format(torch.bfloat16),
                routed_weight=dense_tensor_format(torch.bfloat16),
                shared_gate_up_weight=dense_tensor_format(torch.bfloat16),
            )
        }
    ),
    # Below the hand-written decode kernels, which stay ahead at one token.
    priority=Priority.PERFORMANT,
    traits={
        "weights_packed": frozenset({True}),
        "inputs_contiguous": frozenset({True}),
    },
)
def triton_latent_input_packed(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    *,
    gate_clamp: float,
    up_clamp: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project router, routed latent, and shared input in one weight pass.

    Args:
        hidden_states: Contiguous activation shaped ``[tokens, hidden_size]``.
        router_weight: Router rows of the concatenated weight.
        routed_weight: Latent-projection rows of the concatenated weight.
        shared_gate_up_weight: Stacked shared gate/up rows of the same weight.
        gate_clamp: Positive tanh clamp for the shared gate branch.
        up_clamp: Optional positive tanh clamp for the shared up branch.

    Returns:
        FP32 router logits, routed latent, and the activated shared input.
    """
    weight = packed_projection_weight_view(
        router_weight, routed_weight, shared_gate_up_weight
    )
    if weight is None:
        raise ValueError("projection weights are not one concatenated allocation")
    tokens, hidden_size = hidden_states.shape
    router_n = router_weight.shape[0]
    routed_n = routed_weight.shape[0]
    shared_raw_n = shared_gate_up_weight.shape[0]
    shared_n = shared_raw_n // 2

    device = hidden_states.device
    dtype = hidden_states.dtype
    router_out = torch.empty((tokens, router_n), dtype=torch.float32, device=device)
    routed_out = torch.empty((tokens, routed_n), dtype=dtype, device=device)
    shared_raw = torch.empty((tokens, shared_raw_n), dtype=dtype, device=device)

    block_m, block_n, block_k, warps = _schedule(tokens)
    _packed_input_projections_kernel[
        (triton.cdiv(tokens, block_m), (router_n + routed_n + shared_raw_n) // block_n)
    ](
        hidden_states,
        weight,
        router_out,
        routed_out,
        shared_raw,
        tokens,
        hidden_states.stride(0),
        weight.stride(0),
        K=hidden_size,
        ROUTER_N=router_n,
        ROUTED_N=routed_n,
        SHARED_RAW_N=shared_raw_n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=_NUM_STAGES,
    )

    shared_out = torch.empty((tokens, shared_n), dtype=dtype, device=device)
    situ_block = min(1024, triton.next_power_of_2(shared_n))
    _situ_kernel[(tokens, triton.cdiv(shared_n, situ_block))](
        shared_raw,
        shared_out,
        float(gate_clamp),
        1.0 if up_clamp is None else float(up_clamp),
        tokens,
        SHARED_N=shared_n,
        HAS_LINEAR_BETA=up_clamp is not None,
        BLOCK_N=situ_block,
        num_warps=4,
    )
    return router_out, routed_out, shared_out


__all__ = ["triton_latent_input_packed"]
