from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    out_ptr,
    residual_out_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n_cols
    row_offsets = row * n_cols + offsets

    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        x += residual
        tl.store(residual_out_ptr + row_offsets, x, mask=mask)

    variance = tl.sum(x * x, axis=0) / n_cols
    x *= tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row_offsets, x * weight, mask=mask)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if x.shape[0] == 0:
        if residual is None:
            return x if out is None else out
        return (x if out is None else out), residual
    if x.shape[-1] != weight.shape[0]:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match hidden size {x.shape[-1]}"
        )
    if residual is not None and residual.shape != x.shape:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} does not match input shape {tuple(x.shape)}"
        )

    if not x.is_contiguous():
        x = x.contiguous()
    if residual is not None and not residual.is_contiguous():
        residual = residual.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()

    hidden_size = x.shape[-1]
    x_2d = x.view(-1, hidden_size)
    out = torch.empty_like(x) if out is None else out
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    out_2d = out.view(-1, hidden_size)

    residual_out = torch.empty_like(x) if residual is not None else None
    block = triton.next_power_of_2(hidden_size)
    _rmsnorm_kernel[(x_2d.shape[0],)](
        x_2d,
        residual,
        weight,
        out_2d,
        residual_out,
        hidden_size,
        eps,
        BLOCK=block,
        HAS_RESIDUAL=residual is not None,
    )
    if residual is None:
        return out
    return out, residual_out


@triton.jit
def _fused_qk_rmsnorm_kernel(
    q_in_ptr,
    k_in_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    q_in_token_stride,
    k_in_token_stride,
    q_out_token_stride,
    k_out_token_stride,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # 2D grid: (token, head). Heads in [0, num_q_heads) handle q rows;
    # heads in [num_q_heads, num_q_heads + num_kv_heads) handle k rows.
    # Inputs may be non-contiguous along the leading axis (e.g. views from a
    # qkv split) — we use the explicit token strides to compute addresses.
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)

    offsets = tl.arange(0, BLOCK)
    mask = offsets < head_dim

    if is_k:
        in_addrs = (
            k_in_ptr + token * k_in_token_stride + local_head * head_dim + offsets
        )
        out_addrs = (
            k_out_ptr + token * k_out_token_stride + local_head * head_dim + offsets
        )
        w_addrs = k_weight_ptr + offsets
    else:
        in_addrs = (
            q_in_ptr + token * q_in_token_stride + local_head * head_dim + offsets
        )
        out_addrs = (
            q_out_ptr + token * q_out_token_stride + local_head * head_dim + offsets
        )
        w_addrs = q_weight_ptr + offsets

    x = tl.load(in_addrs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / head_dim
    x = x * tl.rsqrt(var + eps)
    w = tl.load(w_addrs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_addrs, x * w, mask=mask)


def qk_rmsnorm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head RMSNorm of q and k in a single kernel launch.

    Reads from possibly non-contiguous q/k (e.g. views into a qkv-split tensor)
    and writes to fresh contiguous output tensors. The kernel uses the input
    leading-axis stride directly, so no ``.contiguous()`` copy is required
    on the inputs.
    """
    if q.shape[0] == 0:
        return torch.empty_like(q), torch.empty_like(k)
    head_dim = q_weight.shape[0]
    assert k_weight.shape[0] == head_dim, "q/k_weight must share head_dim"
    assert q.shape[-1] % head_dim == 0 and k.shape[-1] % head_dim == 0
    assert (
        q.stride(-1) == 1 and k.stride(-1) == 1
    ), "qk_rmsnorm requires the last dim to be contiguous"

    num_q_heads = q.shape[-1] // head_dim
    num_kv_heads = k.shape[-1] // head_dim
    n_tokens = q.numel() // q.shape[-1]
    block = triton.next_power_of_2(head_dim)

    q_in_stride = q.stride(0) if q.dim() > 1 else q.shape[-1]
    k_in_stride = k.stride(0) if k.dim() > 1 else k.shape[-1]

    # Allocate fresh contiguous outputs so downstream RoPE/attention kernels
    # — which assume row-major layouts — work without further copies.
    q_out = torch.empty((n_tokens, q.shape[-1]), dtype=q.dtype, device=q.device)
    k_out = torch.empty((n_tokens, k.shape[-1]), dtype=k.dtype, device=k.device)

    _fused_qk_rmsnorm_kernel[(n_tokens, num_q_heads + num_kv_heads)](
        q,
        k,
        q_out,
        k_out,
        q_weight,
        k_weight,
        q_in_stride,
        k_in_stride,
        q_out.stride(0),
        k_out.stride(0),
        num_q_heads,
        num_kv_heads,
        head_dim,
        eps,
        BLOCK=block,
    )
    return q_out, k_out


__all__ = ["rmsnorm", "qk_rmsnorm"]
