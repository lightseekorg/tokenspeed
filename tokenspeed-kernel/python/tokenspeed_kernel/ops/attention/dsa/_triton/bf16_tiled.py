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
# out OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""BF16 sparse MLA using tiled online softmax."""

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _dsa_bf16_tiled_kernel(
    q,
    kv_cache,
    topk_slots,
    topk_lens,
    out,
    lse_out,
    num_heads: tl.constexpr,
    topk,
    q_stride_token,
    q_stride_head,
    q_stride_dim,
    kv_stride_page,
    kv_stride_token,
    kv_stride_dim,
    page_size,
    slots_stride_token,
    slots_stride_key,
    lens_stride,
    softmax_scale,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head_block = tl.program_id(1)
    split = tl.program_id(2)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    dims = tl.arange(0, HEAD_DIM)
    keys = tl.arange(0, BLOCK_K)
    q_ptr = q
    q = tl.load(
        q_ptr
        + token * q_stride_token
        + heads[:, None] * q_stride_head
        + dims[None, :] * q_stride_dim,
        heads[:, None] < num_heads,
        0,
    )
    if ROPE_DIM:
        rope = tl.arange(0, ROPE_DIM)
        q_rope = tl.load(
            q_ptr
            + token * q_stride_token
            + heads[:, None] * q_stride_head
            + (HEAD_DIM + rope[None, :]) * q_stride_dim,
            heads[:, None] < num_heads,
            0,
        )
    length = tl.load(topk_lens + token * lens_stride)
    tiles = tl.cdiv(topk, BLOCK_K * NUM_SPLITS)
    maximum = tl.full((BLOCK_H,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, HEAD_DIM), tl.float32)
    # Preserve split boundaries while skipping entirely masked trailing tiles.
    # The live length is device metadata and can change on graph replay.
    split_start = split * tiles * BLOCK_K
    live_tiles = tl.minimum(
        tiles, tl.cdiv(tl.maximum(length - split_start, 0), BLOCK_K)
    )
    for tile in range(live_tiles):
        pos = (split * tiles + tile) * BLOCK_K + keys
        slot = tl.load(
            topk_slots + token * slots_stride_token + pos * slots_stride_key,
            pos < topk,
            -1,
        ).to(tl.int64)
        valid = (pos < length) & (pos < topk) & (slot >= 0)
        kv = tl.load(
            kv_cache
            + (slot[:, None] // page_size) * kv_stride_page
            + (slot[:, None] % page_size) * kv_stride_token
            + dims[None, :] * kv_stride_dim,
            valid[:, None],
            0,
        )
        scores = tl.dot(q, tl.trans(kv))
        if ROPE_DIM:
            k_rope = tl.load(
                kv_cache
                + (slot[:, None] // page_size) * kv_stride_page
                + (slot[:, None] % page_size) * kv_stride_token
                + (HEAD_DIM + rope[None, :]) * kv_stride_dim,
                valid[:, None],
                0,
            )
            scores += tl.dot(q_rope, tl.trans(k_rope))
        scores *= softmax_scale
        scores = tl.where(valid[None, :], scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe_maximum = tl.where(next_maximum == -float("inf"), 0.0, next_maximum)
        alpha = tl.exp(maximum - safe_maximum)
        probability = tl.exp(scores - safe_maximum[:, None])
        denominator = denominator * alpha + tl.sum(probability, 1)
        acc *= alpha[:, None]
        acc = tl.dot(probability.to(tl.bfloat16), kv, acc)
        maximum = next_maximum
    safe_denominator = tl.where(denominator > 0, denominator, 1.0)
    values = acc / safe_denominator[:, None]
    offsets = (
        (token * num_heads + heads[:, None]) * NUM_SPLITS + split
    ) * HEAD_DIM + dims[None, :]
    tl.store(out + offsets, values, heads[:, None] < num_heads)
    if RETURN_LSE:
        lse = tl.where(
            denominator > 0, maximum + tl.log(safe_denominator), -float("inf")
        )
        tl.store(
            lse_out + (token * num_heads + heads) * NUM_SPLITS + split,
            lse,
            heads < num_heads,
        )


@triton.jit
def _dsa_bf16_tiled_reduce_kernel(
    partial_out,
    partial_lse,
    out,
    lse_out,
    NUM_SPLITS: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    splits = tl.arange(0, NUM_SPLITS)
    dims = tl.arange(0, HEAD_DIM)
    lse = tl.load(partial_lse + row * NUM_SPLITS + splits)
    maximum = tl.max(lse, 0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weight = tl.exp(lse - safe_maximum)
    denominator = tl.sum(weight, 0)
    part = tl.load(
        partial_out + (row * NUM_SPLITS + splits[:, None]) * HEAD_DIM + dims[None, :]
    )
    values = tl.sum(part * weight[:, None], 0) / tl.where(
        denominator > 0, denominator, 1.0
    )
    tl.store(out + row * HEAD_DIM + dims, values)
    if RETURN_LSE:
        tl.store(
            lse_out + row,
            tl.where(
                denominator > 0, safe_maximum + tl.log(denominator), -float("inf")
            ),
        )


def run_bf16_tiled(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor | None,
    *,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    scale: float,
    return_lse: bool,
    out: torch.Tensor | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Attend selected global KV slots with BF16 tensor-core products.

    Query shape is [tokens, heads, rank + rope] or its 4D equivalent.
    KV accepts flat rows, pages, or pages with a singleton KV-head dimension.
    Slots are [tokens, width]; optional lengths mask each row's visible prefix.
    The caller owns causal selection and slot bounds; negative slots are ignored.
    Returns BF16 [tokens, heads, rank] (or the supplied output buffer), optionally
    with FP32 natural-log LSE [tokens, heads]. Empty rows produce zero/-inf.
    Validation reads metadata only, so it also runs during CUDA Graph capture.
    """
    if q.ndim not in (3, 4) or kv.ndim not in (2, 3, 4):
        raise ValueError("Expected 3D/4D queries and 2D/3D/4D KV cache")
    if not q.is_cuda or kv.device != q.device:
        raise ValueError("Query and KV must share a CUDA device")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise ValueError("BF16 tiled DSA requires BF16 query and KV")
    query_shape = q.shape
    q = q.reshape(-1, q.shape[-2], q.shape[-1])
    if kv_lora_rank not in (128, 512) or qk_rope_head_dim not in (0, 64):
        raise ValueError("BF16 tiled DSA requires rank 128/512 and RoPE 0/64")
    if q.shape[-1] != kv_lora_rank + qk_rope_head_dim or kv.shape[-1] != q.shape[-1]:
        raise ValueError("Query and KV dimensions must match rank plus RoPE")
    if kv.dim() == 4:
        if kv.shape[1] == 1:
            kv = kv.squeeze(1)
        elif kv.shape[2] == 1:
            kv = kv.squeeze(2)
        else:
            raise ValueError("Expected one KV head")
    if kv.dim() == 2:
        ks0, ks1, ks2, page = kv.stride(0), 0, kv.stride(1), 1
    elif kv.dim() == 3:
        ks0, ks1, ks2, page = *kv.stride(), kv.shape[1]
    else:
        raise ValueError("Unsupported KV layout")
    n, heads, _ = q.shape
    head_dim = kv_lora_rank
    if heads == 0 or page == 0:
        raise ValueError("Query heads and KV page size must be positive")
    if indices.ndim != 2 or indices.shape[0] != n:
        raise ValueError("Slots must have shape [query rows, selection width]")
    if indices.device != q.device or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Slots must be integer tensors on the query device")
    if lengths is not None and (
        lengths.shape != (n,)
        or lengths.device != q.device
        or lengths.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "Lengths must be an integer vector with one entry per query row"
        )
    if out is not None and (
        out.device != q.device
        or out.dtype != q.dtype
        or out.shape
        not in (
            (*query_shape[:-1], head_dim),
            (n, heads, head_dim),
            (n, 1, heads, head_dim),
        )
    ):
        raise ValueError(
            "Output must match the query device, dtype and flattened shape"
        )
    width = indices.shape[-1]
    if lengths is None:
        lengths = torch.full((n,), width, device=q.device, dtype=torch.int32)
    # Write directly when the caller's layout matches the kernel's flat output.
    direct_out = out is not None and out.is_contiguous()
    result = (
        out.view(n, heads, head_dim)
        if direct_out
        else torch.empty((n, heads, head_dim), device=q.device, dtype=q.dtype)
    )
    lse = (
        torch.empty((n, heads), device=q.device, dtype=torch.float32)
        if return_lse
        else None
    )
    # Fixed buckets limit compiled variants; each graph retains its own allocations.
    splits = 16 if n <= 4 else 4 if n <= 32 else 1
    block_k = 32 if width < 64 or q.stride(-1) != 1 or ks2 != 1 else 64
    if n:
        if splits == 1:
            partial, partial_lse = result, lse
        else:
            partial = torch.empty(
                (n, heads, splits, head_dim), device=q.device, dtype=torch.float32
            )
            partial_lse = torch.empty(
                (n, heads, splits), device=q.device, dtype=torch.float32
            )
        _dsa_bf16_tiled_kernel[(n, triton.cdiv(heads, 16), splits)](
            q,
            kv,
            indices,
            lengths,
            partial,
            partial_lse,
            heads,
            width,
            *q.stride(),
            ks0,
            ks1,
            ks2,
            page,
            *indices.stride(),
            lengths.stride(0),
            float(scale),
            block_k,
            splits,
            return_lse or splits > 1,
            BLOCK_H=16,
            HEAD_DIM=head_dim,
            ROPE_DIM=qk_rope_head_dim,
            num_warps=8,
            num_stages=2,
        )
        if splits > 1:
            _dsa_bf16_tiled_reduce_kernel[(n * heads,)](
                partial,
                partial_lse,
                result,
                lse,
                splits,
                return_lse,
                head_dim,
                num_warps=4,
            )
    if out is not None:
        if not direct_out:
            out.copy_(result.reshape(out.shape))
        result = out
    return (result, lse) if return_lse else result
