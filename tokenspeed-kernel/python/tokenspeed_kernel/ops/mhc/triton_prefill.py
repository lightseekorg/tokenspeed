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

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _mhc_prefill_project_hc4_kernel(
    residual,
    fn,
    out_mul,
    out_sqrsum,
    num_tokens,
    hidden_size: tl.constexpr,
    split_k: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    split_id = tl.program_id(0).to(tl.int64)
    token_block_id = tl.program_id(1).to(tl.int64)
    token_offsets = token_block_id * block_m + tl.arange(0, block_m).to(tl.int64)
    mix_offsets = tl.arange(0, 32).to(tl.int64)
    split_start = split_id * split_k
    hc_hidden_size: tl.constexpr = 4 * hidden_size

    projection = tl.zeros((block_m, 32), dtype=tl.float32)
    square_sum = tl.zeros((block_m,), dtype=tl.float32)
    for k_start in range(0, split_k, block_k):
        k_offsets = split_start + k_start + tl.arange(0, block_k).to(tl.int64)
        residual_values = tl.load(
            residual + token_offsets[:, None] * hc_hidden_size + k_offsets[None, :],
            mask=(token_offsets[:, None] < num_tokens)
            & (k_offsets[None, :] < hc_hidden_size),
            other=0.0,
        ).to(tl.float32)
        weight_values = tl.load(
            fn + k_offsets[:, None] + mix_offsets[None, :] * hc_hidden_size,
            mask=(k_offsets[:, None] < hc_hidden_size) & (mix_offsets[None, :] < 24),
            other=0.0,
        )
        projection = tl.dot(
            residual_values,
            weight_values,
            projection,
            input_precision="ieee",
        )
        square_sum += tl.sum(residual_values * residual_values, axis=1)

    output_base = split_id * num_tokens.to(tl.int64)
    tl.store(
        out_mul + output_base * 24 + token_offsets[:, None] * 24 + mix_offsets[None, :],
        projection,
        mask=(token_offsets[:, None] < num_tokens) & (mix_offsets[None, :] < 24),
    )
    tl.store(
        out_sqrsum + output_base + token_offsets,
        square_sum,
        mask=token_offsets < num_tokens,
    )


def mhc_prefill_project_hc4(
    residual: torch.Tensor,
    fn: torch.Tensor,
    out_mul: torch.Tensor,
    out_sqrsum: torch.Tensor,
    *,
    n_splits: int,
    block_m: int,
    block_k: int,
) -> None:
    """Project an hc=4 residual into split prefill mapping partials.

    Args:
        residual: Contiguous BF16 residual streams shaped ``[T, 4, H]``.
        fn: Contiguous FP32 projection weights shaped ``[24, 4 * H]``.
        out_mul: Contiguous FP32 projection output shaped
            ``[n_splits, T, 24]``.
        out_sqrsum: Contiguous FP32 squared-sum output shaped
            ``[n_splits, T]``.
        n_splits: Number of contiguous reduction-axis partitions.
        block_m: Number of token rows in each program tile.
        block_k: Reduction-axis tile width.

    Returns:
        None. ``out_mul`` and ``out_sqrsum`` are written in place.
    """
    if residual.ndim != 3 or residual.shape[1] != 4:
        raise ValueError("mhc_prefill_project_hc4 requires residual shaped [T, 4, H]")
    if residual.dtype != torch.bfloat16 or fn.dtype != torch.float32:
        raise ValueError(
            "mhc_prefill_project_hc4 requires BF16 residual and FP32 weights"
        )
    if out_mul.dtype != torch.float32 or out_sqrsum.dtype != torch.float32:
        raise ValueError("mhc_prefill_project_hc4 requires FP32 output buffers")
    tensors = (residual, fn, out_mul, out_sqrsum)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("mhc_prefill_project_hc4 requires CUDA tensors")
    if not all(tensor.device == residual.device for tensor in tensors):
        raise ValueError("mhc_prefill_project_hc4 tensors must share one device")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("mhc_prefill_project_hc4 tensors must be contiguous")
    if not isinstance(n_splits, int) or n_splits < 1:
        raise ValueError("n_splits must be a positive integer")
    for name, value in (("block_m", block_m), ("block_k", block_k)):
        if not isinstance(value, int) or value < 16 or value & (value - 1):
            raise ValueError(f"{name} must be a power of two of at least 16")

    num_tokens, _, hidden_size = residual.shape
    if hidden_size < 1:
        raise ValueError("hidden size must be positive")
    hc_hidden_size = 4 * hidden_size
    if fn.shape != (24, hc_hidden_size):
        raise ValueError(
            f"fn shape mismatch: expected {(24, hc_hidden_size)}, got {tuple(fn.shape)}"
        )
    if out_mul.shape != (n_splits, num_tokens, 24):
        raise ValueError(
            "out_mul shape mismatch: expected "
            f"{(n_splits, num_tokens, 24)}, got {tuple(out_mul.shape)}"
        )
    if out_sqrsum.shape != (n_splits, num_tokens):
        raise ValueError(
            "out_sqrsum shape mismatch: expected "
            f"{(n_splits, num_tokens)}, got {tuple(out_sqrsum.shape)}"
        )
    if num_tokens == 0:
        return

    split_k = triton.cdiv(triton.cdiv(hc_hidden_size, n_splits), block_k) * block_k
    _mhc_prefill_project_hc4_kernel[(n_splits, triton.cdiv(num_tokens, block_m))](
        residual,
        fn,
        out_mul,
        out_sqrsum,
        num_tokens,
        hidden_size=hidden_size,
        split_k=split_k,
        block_m=block_m,
        block_k=block_k,
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def _mhc_pre_mix_hc4_kernel(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    pre_mix,
    post_mix,
    comb_mix,
    hidden_size: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    n_splits: tl.constexpr,
    num_tokens,
):
    token_id = tl.program_id(0)
    pre_post_offsets = tl.arange(0, 8)
    comb_offsets = tl.arange(0, 16)
    pre_post_values = tl.zeros((8,), tl.float32)
    comb_values = tl.zeros((16,), tl.float32)
    rms_sum = tl.full((), 0.0, tl.float32)

    for split_id in tl.static_range(0, n_splits):
        split_base = split_id * num_tokens * 24 + token_id * 24
        pre_post_values += tl.load(gemm_out_mul + split_base + pre_post_offsets)
        comb_values += tl.load(gemm_out_mul + split_base + 8 + comb_offsets)
        rms_sum += tl.load(gemm_out_sqrsum + split_id * num_tokens + token_id)

    rms = tl.rsqrt(rms_sum / (4 * hidden_size) + rms_eps)
    pre_post_scale = tl.where(
        pre_post_offsets < 4,
        tl.load(hc_scale),
        tl.load(hc_scale + 1),
    )
    pre_post_values = tl.sigmoid(
        pre_post_values * rms * pre_post_scale + tl.load(hc_base + pre_post_offsets)
    )
    tl.store(
        pre_mix + token_id * 4 + pre_post_offsets,
        pre_post_values + hc_eps,
        mask=pre_post_offsets < 4,
    )
    tl.store(
        post_mix + token_id * 4 + pre_post_offsets - 4,
        pre_post_values * 2.0,
        mask=pre_post_offsets >= 4,
    )

    comb_values = comb_values * rms * tl.load(hc_scale + 2) + tl.load(
        hc_base + 8 + comb_offsets
    )
    comb_matrix = tl.reshape(comb_values, (4, 4))
    row_max = tl.max(comb_matrix, axis=1)
    comb_matrix = tl.exp(comb_matrix - tl.expand_dims(row_max, 1))
    row_sum = tl.sum(comb_matrix, axis=1)
    comb_matrix = comb_matrix / tl.expand_dims(row_sum, 1) + hc_eps
    col_sum = tl.sum(comb_matrix, axis=0)
    comb_matrix = comb_matrix / (tl.expand_dims(col_sum, 0) + hc_eps)

    for _ in tl.static_range(1, sinkhorn_iters):
        row_sum = tl.sum(comb_matrix, axis=1)
        comb_matrix = comb_matrix / (tl.expand_dims(row_sum, 1) + hc_eps)
        col_sum = tl.sum(comb_matrix, axis=0)
        comb_matrix = comb_matrix / (tl.expand_dims(col_sum, 0) + hc_eps)

    tl.store(
        comb_mix + token_id * 16 + comb_offsets,
        tl.reshape(comb_matrix, (16,)),
    )


def mhc_pre_mix_hc4(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    pre_mix: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    *,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    n_splits: int,
    num_tokens: int,
) -> None:
    """Reduce split-K mHC projections and form hc=4 mixing coefficients.

    Args:
        gemm_out_mul: FP32 split-K projections shaped ``[n_splits, T, 24]``.
        gemm_out_sqrsum: FP32 split-K squared sums shaped ``[n_splits, T]``.
        hc_scale: FP32 scales for pre, post, and combination mappings.
        hc_base: FP32 biases shaped ``[24]``.
        pre_mix: FP32 output buffer shaped ``[T, 4]``.
        post_mix: FP32 output buffer shaped ``[T, 4]``.
        comb_mix: FP32 output buffer shaped ``[T, 16]``.
        hidden_size: Hidden width of one residual stream.
        rms_eps: Epsilon used by the pre-projection RMS normalization.
        hc_eps: Epsilon used by the mHC mixing transforms.
        sinkhorn_iters: Number of row/column Sinkhorn normalization rounds.
        n_splits: Number of split-K projection partials.
        num_tokens: Number of token rows in each tensor.

    Returns:
        None. The three output buffers are written in place.
    """
    _mhc_pre_mix_hc4_kernel[(num_tokens,)](
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        pre_mix,
        post_mix,
        comb_mix,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        n_splits=n_splits,
        num_tokens=num_tokens,
        num_warps=1,
    )


@triton.jit
def _mhc_pre_only_hc4_kernel(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    pre_mix,
    hidden_size: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    n_splits: tl.constexpr,
    num_tokens,
):
    token_id = tl.program_id(0)
    pre_offsets = tl.arange(0, 4)
    pre_values = tl.zeros((4,), tl.float32)
    rms_sum = tl.full((), 0.0, tl.float32)

    for split_id in tl.static_range(0, n_splits):
        split_base = split_id * num_tokens * 24 + token_id * 24
        pre_values += tl.load(gemm_out_mul + split_base + pre_offsets)
        rms_sum += tl.load(gemm_out_sqrsum + split_id * num_tokens + token_id)

    rms = tl.rsqrt(rms_sum / (4 * hidden_size) + rms_eps)
    pre_values = (
        tl.sigmoid(
            pre_values * rms * tl.load(hc_scale) + tl.load(hc_base + pre_offsets)
        )
        + hc_eps
    )
    tl.store(pre_mix + token_id * 4 + pre_offsets, pre_values)


def mhc_pre_only_hc4(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    pre_mix: torch.Tensor,
    *,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    n_splits: int,
    num_tokens: int,
) -> None:
    """Form only the pre-mapping coefficients for an hc=4 mHC layer.

    Args:
        gemm_out_mul: FP32 split-K projections shaped ``[n_splits, T, 24]``.
        gemm_out_sqrsum: FP32 split-K squared sums shaped ``[n_splits, T]``.
        hc_scale: FP32 scales for pre, post, and combination mappings.
        hc_base: FP32 biases shaped ``[24]``.
        pre_mix: FP32 output buffer shaped ``[T, 4]``.
        hidden_size: Hidden width of one residual stream.
        rms_eps: Epsilon used by the pre-projection RMS normalization.
        hc_eps: Epsilon used by the mHC pre-mapping transform.
        n_splits: Number of split-K projection partials.
        num_tokens: Number of token rows in each tensor.

    Returns:
        None. ``pre_mix`` is written in place.
    """
    _mhc_pre_only_hc4_kernel[(num_tokens,)](
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        pre_mix,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        n_splits=n_splits,
        num_tokens=num_tokens,
        num_warps=1,
    )


@triton.jit
def _mhc_post_comb_hc4_kernel(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    post_mix,
    comb_mix,
    hidden_size: tl.constexpr,
    rms_eps: tl.constexpr,
    hc_eps: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    n_splits: tl.constexpr,
    num_tokens,
):
    token_id = tl.program_id(0)
    post_offsets = tl.arange(0, 4)
    comb_offsets = tl.arange(0, 16)
    post_values = tl.zeros((4,), tl.float32)
    comb_values = tl.zeros((16,), tl.float32)
    rms_sum = tl.full((), 0.0, tl.float32)

    for split_id in tl.static_range(0, n_splits):
        split_base = split_id * num_tokens * 24 + token_id * 24
        post_values += tl.load(gemm_out_mul + split_base + 4 + post_offsets)
        comb_values += tl.load(gemm_out_mul + split_base + 8 + comb_offsets)
        rms_sum += tl.load(gemm_out_sqrsum + split_id * num_tokens + token_id)

    rms = tl.rsqrt(rms_sum / (4 * hidden_size) + rms_eps)
    post_values = tl.sigmoid(
        post_values * rms * tl.load(hc_scale + 1) + tl.load(hc_base + 4 + post_offsets)
    )
    tl.store(post_mix + token_id * 4 + post_offsets, post_values * 2.0)

    comb_values = comb_values * rms * tl.load(hc_scale + 2) + tl.load(
        hc_base + 8 + comb_offsets
    )
    comb_matrix = tl.reshape(comb_values, (4, 4))
    row_max = tl.max(comb_matrix, axis=1)
    comb_matrix = tl.exp(comb_matrix - tl.expand_dims(row_max, 1))
    row_sum = tl.sum(comb_matrix, axis=1)
    comb_matrix = comb_matrix / tl.expand_dims(row_sum, 1) + hc_eps
    col_sum = tl.sum(comb_matrix, axis=0)
    comb_matrix = comb_matrix / (tl.expand_dims(col_sum, 0) + hc_eps)

    for _ in tl.static_range(1, sinkhorn_iters):
        row_sum = tl.sum(comb_matrix, axis=1)
        comb_matrix = comb_matrix / (tl.expand_dims(row_sum, 1) + hc_eps)
        col_sum = tl.sum(comb_matrix, axis=0)
        comb_matrix = comb_matrix / (tl.expand_dims(col_sum, 0) + hc_eps)

    tl.store(
        comb_mix + token_id * 16 + comb_offsets,
        tl.reshape(comb_matrix, (16,)),
    )


def mhc_post_comb_hc4(
    gemm_out_mul: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    *,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    n_splits: int,
    num_tokens: int,
) -> None:
    """Form post and combination coefficients for an hc=4 mHC layer.

    This is independent from the pre-mapping after the split-K projection, so
    callers may run it on a side stream while the layer consumes the pre path.

    Args:
        gemm_out_mul: FP32 split-K projections shaped ``[n_splits, T, 24]``.
        gemm_out_sqrsum: FP32 split-K squared sums shaped ``[n_splits, T]``.
        hc_scale: FP32 scales for pre, post, and combination mappings.
        hc_base: FP32 biases shaped ``[24]``.
        post_mix: FP32 output buffer shaped ``[T, 4]``.
        comb_mix: FP32 output buffer shaped ``[T, 16]``.
        hidden_size: Hidden width of one residual stream.
        rms_eps: Epsilon used by the pre-projection RMS normalization.
        hc_eps: Epsilon used by the mHC mixing transforms.
        sinkhorn_iters: Number of row/column Sinkhorn normalization rounds.
        n_splits: Number of split-K projection partials.
        num_tokens: Number of token rows in each tensor.

    Returns:
        None. ``post_mix`` and ``comb_mix`` are written in place.
    """
    _mhc_post_comb_hc4_kernel[(num_tokens,)](
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        post_mix,
        comb_mix,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        n_splits=n_splits,
        num_tokens=num_tokens,
        num_warps=1,
    )


@triton.jit
def _mhc_pre_layer_norm_hc4_kernel(
    pre_mix,
    residual,
    weight,
    out,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    block_h: tl.constexpr,
):
    token_id = tl.program_id(0)
    hidden_offsets = tl.arange(0, block_h)
    hidden_mask = hidden_offsets < hidden_size
    residual_base = token_id * 4 * hidden_size

    layer_input = tl.zeros((block_h,), tl.float32)
    for hc_id in tl.static_range(0, 4):
        pre = tl.load(pre_mix + token_id * 4 + hc_id).to(tl.float32)
        residual_values = tl.load(
            residual + residual_base + hc_id * hidden_size + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        layer_input += pre * residual_values

    # Preserve the rounding point of the unfused path, which materializes the
    # weighted residual sum as BF16 before RMSNorm reads it back.
    layer_input = layer_input.to(tl.bfloat16).to(tl.float32)
    variance = tl.sum(layer_input * layer_input, axis=0) / hidden_size
    norm_scale = tl.rsqrt(variance + eps)
    norm_weight = tl.load(weight + hidden_offsets, mask=hidden_mask, other=0.0).to(
        tl.float32
    )
    tl.store(
        out + token_id * hidden_size + hidden_offsets,
        layer_input * norm_scale * norm_weight,
        mask=hidden_mask,
    )


def mhc_pre_layer_norm_hc4(
    pre_mix: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    *,
    eps: float,
) -> None:
    """Form an hc=4 mHC layer input and apply RMSNorm in one kernel.

    Args:
        pre_mix: FP32 mixing coefficients shaped ``[..., 4]``.
        residual: BF16 residual streams shaped ``[..., 4, hidden_size]``.
        weight: BF16 or FP32 RMSNorm weight shaped ``[hidden_size]``.
        out: BF16 output buffer shaped ``[..., hidden_size]``.
        eps: RMSNorm epsilon.

    Returns:
        None. ``out`` is written in place.
    """
    if residual.shape[-2] != 4 or pre_mix.shape[-1] != 4:
        raise ValueError("mhc_pre_layer_norm_hc4 requires exactly four streams")
    hidden_size = residual.shape[-1]
    if weight.shape != (hidden_size,):
        raise ValueError(
            f"weight shape {tuple(weight.shape)} does not match hidden size "
            f"{hidden_size}"
        )
    if out.shape != (*residual.shape[:-2], hidden_size):
        raise ValueError(
            f"out shape {tuple(out.shape)} does not match residual prefix "
            f"{tuple(residual.shape[:-2])} and hidden size {hidden_size}"
        )
    if not (pre_mix.is_contiguous() and residual.is_contiguous()):
        raise ValueError("pre_mix and residual must be contiguous")
    if not (weight.is_contiguous() and out.is_contiguous()):
        raise ValueError("weight and out must be contiguous")

    num_tokens = residual.numel() // (4 * hidden_size)
    if num_tokens == 0:
        return
    block_h = triton.next_power_of_2(hidden_size)
    _mhc_pre_layer_norm_hc4_kernel[(num_tokens,)](
        pre_mix,
        residual,
        weight,
        out,
        hidden_size=hidden_size,
        eps=eps,
        block_h=block_h,
        num_warps=8,
    )


def _mhc_prefill_config_hc4(num_tokens: int) -> tuple[int, int, int]:
    if num_tokens <= 1024:
        return 8, 32, 256
    if num_tokens <= 2048:
        return 8, 64, 128
    if num_tokens <= 4096:
        return 4, 64, 256
    if num_tokens <= 8192:
        return 2, 64, 256
    return 1, 64, 256


def fused_mhc_prefill_hc4(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the tiled hc=4 mHC prefill path with fused output RMSNorm.

    Args:
        residual: Contiguous BF16 residual streams shaped ``[T, 4, H]``.
        fn: Contiguous FP32 projection weights shaped ``[24, 4 * H]``.
        hc_scale: Contiguous FP32 pre, post, and combination scales shaped
            ``[3]``.
        hc_base: Contiguous FP32 projection bias shaped ``[24]``.
        norm_weight: Contiguous BF16 or FP32 RMSNorm weight shaped ``[H]``.
        rms_eps: Epsilon used by the pre-projection RMS normalization.
        hc_eps: Epsilon used by the mHC mapping transforms.
        sinkhorn_iters: Number of row/column Sinkhorn normalization rounds.
        norm_eps: Epsilon used by the output RMSNorm.

    Returns:
        A tuple of BF16 normalized layer input ``[T, H]``, FP32 post
        coefficients ``[T, 4, 1]``, and FP32 combination coefficients
        ``[T, 4, 4]``.
    """
    if residual.ndim != 3 or residual.shape[1] != 4:
        raise ValueError("fused_mhc_prefill_hc4 requires residual shaped [T, 4, H]")
    if residual.dtype != torch.bfloat16 or fn.dtype != torch.float32:
        raise ValueError(
            "fused_mhc_prefill_hc4 requires BF16 residual and FP32 weights"
        )
    if hc_scale.dtype != torch.float32 or hc_base.dtype != torch.float32:
        raise ValueError("fused_mhc_prefill_hc4 requires FP32 scales and biases")
    if norm_weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("fused_mhc_prefill_hc4 requires BF16 or FP32 norm weight")
    tensors = (residual, fn, hc_scale, hc_base, norm_weight)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("fused_mhc_prefill_hc4 requires CUDA tensors")
    if not all(tensor.device == residual.device for tensor in tensors):
        raise ValueError("fused_mhc_prefill_hc4 tensors must share one device")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused_mhc_prefill_hc4 tensors must be contiguous")
    if sinkhorn_iters < 1:
        raise ValueError("sinkhorn_iters must be positive")

    num_tokens, _, hidden_size = residual.shape
    if hidden_size < 1:
        raise ValueError("hidden size must be positive")
    if fn.shape != (24, 4 * hidden_size):
        raise ValueError(
            f"fn shape mismatch: expected {(24, 4 * hidden_size)}, "
            f"got {tuple(fn.shape)}"
        )
    if hc_scale.shape != (3,) or hc_base.shape != (24,):
        raise ValueError("hc_scale and hc_base must have shapes [3] and [24]")
    if norm_weight.shape != (hidden_size,):
        raise ValueError(
            f"norm_weight shape mismatch: expected {(hidden_size,)}, "
            f"got {tuple(norm_weight.shape)}"
        )
    if num_tokens == 0:
        return (
            residual.new_empty(0, hidden_size),
            torch.empty(0, 4, 1, device=residual.device, dtype=torch.float32),
            torch.empty(0, 4, 4, device=residual.device, dtype=torch.float32),
        )

    n_splits, block_m, block_k = _mhc_prefill_config_hc4(num_tokens)
    n_splits = min(n_splits, triton.cdiv(4 * hidden_size, block_k))
    projection = torch.empty(
        n_splits, num_tokens, 24, device=residual.device, dtype=torch.float32
    )
    square_sum = torch.empty(
        n_splits, num_tokens, device=residual.device, dtype=torch.float32
    )
    pre_mix = torch.empty(num_tokens, 4, device=residual.device, dtype=torch.float32)
    post_mix = torch.empty_like(pre_mix)
    comb_mix = torch.empty(num_tokens, 16, device=residual.device, dtype=torch.float32)
    layer_input = torch.empty(
        num_tokens, hidden_size, device=residual.device, dtype=torch.bfloat16
    )

    mhc_prefill_project_hc4(
        residual,
        fn,
        projection,
        square_sum,
        n_splits=n_splits,
        block_m=block_m,
        block_k=block_k,
    )
    mhc_pre_mix_hc4(
        projection,
        square_sum,
        hc_scale,
        hc_base,
        pre_mix,
        post_mix,
        comb_mix,
        hidden_size=hidden_size,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        n_splits=n_splits,
        num_tokens=num_tokens,
    )
    mhc_pre_layer_norm_hc4(
        pre_mix,
        residual,
        norm_weight,
        layer_input,
        eps=norm_eps,
    )
    return layer_input, post_mix.unsqueeze(-1), comb_mix.view(num_tokens, 4, 4)
