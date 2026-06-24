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


@triton.jit
def _rmsnorm_fused_parallel_kernel(
    input1_ptr,
    weight1_ptr,
    output1_ptr,
    input2_ptr,
    weight2_ptr,
    output2_ptr,
    n_cols1: tl.constexpr,
    n_cols2: tl.constexpr,
    stride_input1: tl.constexpr,
    stride_output1: tl.constexpr,
    stride_input2: tl.constexpr,
    stride_output2: tl.constexpr,
    eps: tl.constexpr,
    BLOCK1: tl.constexpr,
    BLOCK2: tl.constexpr,
):
    row = tl.program_id(0)

    offsets1 = tl.arange(0, BLOCK1)
    mask1 = offsets1 < n_cols1
    input1_offsets = row * stride_input1 + offsets1
    output1_offsets = row * stride_output1 + offsets1
    input1 = tl.load(input1_ptr + input1_offsets, mask=mask1, other=0.0).to(tl.float32)
    variance1 = tl.sum(input1 * input1, axis=0) / n_cols1
    weight1 = tl.load(weight1_ptr + offsets1, mask=mask1, other=0.0).to(tl.float32)
    output1 = input1 * tl.rsqrt(variance1 + eps) * weight1
    tl.store(output1_ptr + output1_offsets, output1, mask=mask1)

    offsets2 = tl.arange(0, BLOCK2)
    mask2 = offsets2 < n_cols2
    input2_offsets = row * stride_input2 + offsets2
    output2_offsets = row * stride_output2 + offsets2
    input2 = tl.load(input2_ptr + input2_offsets, mask=mask2, other=0.0).to(tl.float32)
    variance2 = tl.sum(input2 * input2, axis=0) / n_cols2
    weight2 = tl.load(weight2_ptr + offsets2, mask=mask2, other=0.0).to(tl.float32)
    output2 = input2 * tl.rsqrt(variance2 + eps) * weight2
    tl.store(output2_ptr + output2_offsets, output2, mask=mask2)


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


@triton.jit
def _fused_qk_rmsnorm_rope_gate_kernel(
    q_gate_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    gate_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_cache_ptr,
    positions_ptr,
    q_gate_stride_t,
    k_stride_t,
    q_out_stride_t,
    k_out_stride_t,
    gate_out_stride_t,
    cache_stride_p,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    half_rotary: tl.constexpr,
    eps: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    ROT_HALF_BLOCK: tl.constexpr,
    HAS_PASS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    is_k = head >= num_q_heads
    local_head = tl.where(is_k, head - num_q_heads, head)

    if is_k:
        in_base = k_ptr + token * k_stride_t + local_head * head_dim
        w_ptr = k_weight_ptr
        out_base = k_out_ptr + token * k_out_stride_t + local_head * head_dim
    else:
        in_base = q_gate_ptr + token * q_gate_stride_t + local_head * 2 * head_dim
        w_ptr = q_weight_ptr
        out_base = q_out_ptr + token * q_out_stride_t + local_head * head_dim

    # --- RMSNorm: variance over the full head_dim ---
    head_offs = tl.arange(0, HEAD_BLOCK)
    head_mask = head_offs < head_dim
    x = tl.load(in_base + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / head_dim
    inv_rms = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + head_offs, mask=head_mask, other=0.0).to(tl.float32)
    # Round-trip through INPUT_DTYPE so the RoPE input matches the bf16-storage
    # behavior of the unfused (qk_rmsnorm → memory → apply_rope) reference path.
    x_norm = (x * inv_rms * w).to(INPUT_DTYPE).to(tl.float32)

    # --- Pass-through tail [rotary_dim, head_dim): RMSNorm-only, no rotation ---
    # The rotary head [0, rotary_dim) will be overwritten by the RoPE store below.
    if HAS_PASS:
        pass_mask = head_mask & (head_offs >= rotary_dim)
        tl.store(out_base + head_offs, x_norm, mask=pass_mask)

    # --- Partial RoPE on the first rotary_dim elements ---
    # Triton lacks easy sub-vector slicing of x_norm, so we recompute the
    # normalized rotary halves on a smaller block (next_pow2(half_rotary)).
    # The extra ~rotary_dim element reload hits L1, so the cost is negligible.
    rot_offs = tl.arange(0, ROT_HALF_BLOCK)
    rot_mask = rot_offs < half_rotary
    x_rot1 = tl.load(in_base + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    x_rot2 = tl.load(in_base + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    w_rot1 = tl.load(w_ptr + rot_offs, mask=rot_mask, other=0.0).to(tl.float32)
    w_rot2 = tl.load(w_ptr + half_rotary + rot_offs, mask=rot_mask, other=0.0).to(
        tl.float32
    )
    x_rot1 = (x_rot1 * inv_rms * w_rot1).to(INPUT_DTYPE).to(tl.float32)
    x_rot2 = (x_rot2 * inv_rms * w_rot2).to(INPUT_DTYPE).to(tl.float32)

    # Always use int64 for position to avoid overflow in address computation.
    pos = tl.load(positions_ptr + token).to(tl.int64)
    cache_offset = pos * cache_stride_p
    cos = tl.load(
        cos_sin_cache_ptr + cache_offset + rot_offs, mask=rot_mask, other=0.0
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache_ptr + cache_offset + half_rotary + rot_offs,
        mask=rot_mask,
        other=0.0,
    ).to(tl.float32)

    o1 = x_rot1 * cos - x_rot2 * sin
    o2 = x_rot2 * cos + x_rot1 * sin
    tl.store(out_base + rot_offs, o1, mask=rot_mask)
    tl.store(out_base + half_rotary + rot_offs, o2, mask=rot_mask)

    # --- Gate copy (q heads only, verbatim) ---
    if not is_k:
        gate_in_base = in_base + head_dim
        gate_out_base = gate_out_ptr + token * gate_out_stride_t + local_head * head_dim
        g = tl.load(gate_in_base + head_offs, mask=head_mask, other=0.0)
        tl.store(gate_out_base + head_offs, g, mask=head_mask)


def fused_qk_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused split + QK-RMSNorm + (partial) RoPE + gate copy for Qwen3.5 attn.

    Replaces 4 kernel launches (2 contiguous copies + qk_rmsnorm + RoPE)
    with a single triton kernel.  Supports partial RoPE: only the first
    ``rotary_dim`` elements of each head are rotated; the remaining
    ``head_dim - rotary_dim`` elements pass through after RMSNorm only.
    Qwen3.5 uses ``partial_rotary_factor=0.25``, so ``rotary_dim = head_dim // 4``.

    Args:
        q_gate: (n_tokens, num_q_heads * 2 * head_dim) — per head: [q|gate]
        k: (n_tokens, num_kv_heads * head_dim)
        q_weight: (head_dim,) GemmaRMSNorm weight (already +1)
        k_weight: (head_dim,) GemmaRMSNorm weight (already +1)
        cos_sin_cache: (max_pos, rotary_dim) packed [cos|sin], float32
        positions: (n_tokens,) int32 or int64
        eps: RMSNorm epsilon
        num_q_heads: number of Q heads (after TP split)
        num_kv_heads: number of KV heads (after TP split)
        head_dim: per-head dimension
        rotary_dim: rotary dimension; must be even and <= head_dim

    Returns:
        (q_out, k_out, gate_out) — all contiguous (n_tokens, heads * head_dim)
    """
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(
            f"rotary_dim must be a positive even integer <= head_dim, "
            f"got rotary_dim={rotary_dim}, head_dim={head_dim}"
        )

    n_tokens = q_gate.shape[0]
    if n_tokens == 0:
        q_out = torch.empty(
            (0, num_q_heads * head_dim), dtype=q_gate.dtype, device=q_gate.device
        )
        k_out = torch.empty(
            (0, num_kv_heads * head_dim), dtype=k.dtype, device=k.device
        )
        gate_out = torch.empty_like(q_out)
        return q_out, k_out, gate_out

    q_out = torch.empty(
        (n_tokens, num_q_heads * head_dim), dtype=q_gate.dtype, device=q_gate.device
    )
    k_out = torch.empty(
        (n_tokens, num_kv_heads * head_dim), dtype=k.dtype, device=k.device
    )
    gate_out = torch.empty_like(q_out)

    half_rotary = rotary_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    rot_half_block = triton.next_power_of_2(half_rotary)
    num_warps = max(1, head_block // 64)

    grid = (n_tokens, num_q_heads + num_kv_heads)
    _fused_qk_rmsnorm_rope_gate_kernel[grid](
        q_gate,
        k,
        q_out,
        k_out,
        gate_out,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        q_gate.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        gate_out.stride(0),
        cos_sin_cache.stride(0),
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        half_rotary,
        eps,
        INPUT_DTYPE=tl.bfloat16 if q_gate.dtype == torch.bfloat16 else tl.float16,
        HEAD_BLOCK=head_block,
        ROT_HALF_BLOCK=rot_half_block,
        HAS_PASS=rotary_dim < head_dim,
        num_warps=num_warps,
        num_stages=2,
    )
    return q_out, k_out, gate_out


def rmsnorm_fused_parallel(
    input1: torch.Tensor,
    weight1: torch.Tensor,
    output1: torch.Tensor,
    input2: torch.Tensor,
    weight2: torch.Tensor,
    output2: torch.Tensor,
    eps: float,
    enable_pdl: bool = False,
) -> None:
    del enable_pdl
    if input1.shape[0] == 0:
        return
    if input1.dim() != 2 or input2.dim() != 2:
        raise ValueError("rmsnorm_fused_parallel expects 2D inputs")
    if input1.shape[0] != input2.shape[0]:
        raise ValueError(f"input row mismatch: {input1.shape[0]} vs {input2.shape[0]}")
    if input1.shape != output1.shape:
        raise ValueError(
            f"output1 shape {tuple(output1.shape)} does not match input1 "
            f"shape {tuple(input1.shape)}"
        )
    if input2.shape != output2.shape:
        raise ValueError(
            f"output2 shape {tuple(output2.shape)} does not match input2 "
            f"shape {tuple(input2.shape)}"
        )
    if input1.shape[-1] != weight1.shape[0]:
        raise ValueError(
            f"weight1 shape {tuple(weight1.shape)} does not match hidden size "
            f"{input1.shape[-1]}"
        )
    if input2.shape[-1] != weight2.shape[0]:
        raise ValueError(
            f"weight2 shape {tuple(weight2.shape)} does not match hidden size "
            f"{input2.shape[-1]}"
        )
    tensors = (input1, weight1, output1, input2, weight2, output2)
    if any(t.stride(-1) != 1 for t in tensors):
        raise ValueError("rmsnorm_fused_parallel requires contiguous last dimension")

    n_cols1 = input1.shape[-1]
    n_cols2 = input2.shape[-1]
    block1 = triton.next_power_of_2(n_cols1)
    block2 = triton.next_power_of_2(n_cols2)
    _rmsnorm_fused_parallel_kernel[(input1.shape[0],)](
        input1,
        weight1,
        output1,
        input2,
        weight2,
        output2,
        n_cols1,
        n_cols2,
        input1.stride(0),
        output1.stride(0),
        input2.stride(0),
        output2.stride(0),
        eps,
        BLOCK1=block1,
        BLOCK2=block2,
    )


__all__ = [
    "rmsnorm",
    "qk_rmsnorm",
    "fused_qk_rmsnorm_rope_gate",
    "rmsnorm_fused_parallel",
]
