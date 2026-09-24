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
from tokenspeed_kernel.platform import CapabilityRequirement, current_platform
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


# Selected-slot DSA attention in absorbed-MLA form: the values are the latent
# part of the selected KV rows, so each gathered tile feeds both S = Q K^T and
# O += P V on tensor cores. Decode splits the keys across programs and merges
# the splits with a log-sum-exp combine.
_BLOCK_H = 16
_BLOCK_N = 32


@triton.jit
def _dsa_key_range(
    topk_lens,
    token,
    split,
    topk: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # The first min(topk_lens[token], topk) selected slots are live. Each split
    # takes an equal share of them, rounded up to whole key tiles.
    valid_len = tl.load(topk_lens + token).to(tl.int32)
    valid_len = tl.minimum(tl.maximum(valid_len, 0), topk)
    split_len = tl.cdiv(tl.cdiv(valid_len, NUM_SPLITS), BLOCK_N) * BLOCK_N
    start = split * split_len
    return start, tl.minimum(start + split_len, valid_len)


@triton.jit
def _dsa_softmax_update(
    scores,
    valid,
    latent,
    m_i,
    l_i,
    acc,
    softmax_scale: tl.constexpr,
):
    # One online-softmax step in base 2; the latent tile that produced the
    # scores also holds the values.
    scores = tl.where(
        valid[None, :],
        scores * (softmax_scale * 1.4426950408889634),
        -float("inf"),
    )
    m_new = tl.maximum(m_i, tl.max(scores, axis=1))
    # Rows without a valid key so far use a zero offset to avoid -inf - -inf.
    m_offset = tl.where(m_new == -float("inf"), 0.0, m_new)
    alpha = tl.exp2(m_i - m_offset)
    p = tl.exp2(scores - m_offset[:, None])
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = tl.dot(p.to(latent.dtype), latent, acc * alpha[:, None])
    return m_new, l_i, acc


@triton.jit
def _dsa_store_output(
    out,
    partial_out,
    partial_lse,
    acc,
    m_i,
    l_i,
    rows,
    head_mask,
    split,
    kv_lora_rank: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    dims = tl.arange(0, kv_lora_rank)
    # Rows without any valid key are zero.
    result = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
    if NUM_SPLITS == 1:
        tl.store(
            out + rows[:, None] * kv_lora_rank + dims[None, :],
            result.to(out.dtype.element_ty),
            mask=head_mask[:, None],
        )
    else:
        split_rows = rows * NUM_SPLITS + split
        tl.store(
            partial_out + split_rows[:, None] * kv_lora_rank + dims[None, :],
            result,
            mask=head_mask[:, None],
        )
        tl.store(
            partial_lse + split_rows,
            tl.where(l_i > 0.0, m_i + tl.log2(l_i), -float("inf")),
            mask=head_mask,
        )


@triton.jit
def _dsa_packed_kv_kernel(
    q,
    kv_fp8,
    kv_scale,
    kv_rope,
    topk_indices,
    topk_lens,
    out,
    partial_out,
    partial_lse,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    row_bytes: tl.constexpr,
    topk: tl.constexpr,
    softmax_scale: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    split = tl.program_id(2)
    head_mask = heads < num_heads
    rows = token * num_heads + heads
    dims = tl.arange(0, kv_lora_rank)
    # A row holds kv_lora_rank FP8 latents, one FP32 scale per 128 of them,
    # then the BF16 RoPE key.
    num_groups: tl.constexpr = kv_lora_rank // 128
    groups = tl.arange(0, num_groups)

    q_nope = tl.load(
        q + rows[:, None] * head_dim + dims[None, :],
        mask=head_mask[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    if qk_rope_head_dim > 0:
        rope_start: tl.constexpr = kv_lora_rank + num_groups * 4
        rope_dims = tl.arange(0, BLOCK_ROPE)
        rope_mask = rope_dims < qk_rope_head_dim
        q_rope = tl.load(
            q + rows[:, None] * head_dim + kv_lora_rank + rope_dims[None, :],
            mask=head_mask[:, None] & rope_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)

    start, end = _dsa_key_range(topk_lens, token, split, topk, NUM_SPLITS, BLOCK_N)
    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, kv_lora_rank], tl.float32)
    for block_start in range(start, end, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        slots = tl.load(topk_indices + token * topk + cols, mask=cols < end, other=-1)
        valid = slots >= 0
        kv_rows = slots.to(tl.int64) * row_bytes
        # Dequantize the gathered rows once; the BF16 latent tile serves as
        # both keys and values.
        latent = tl.load(
            kv_fp8 + kv_rows[:, None] + dims[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scale = tl.load(
            kv_scale + (kv_rows[:, None] + kv_lora_rank + groups[None, :] * 4) // 4,
            mask=valid[:, None],
            other=0.0,
        )
        latent = tl.reshape(latent.to(tl.float32), [BLOCK_N, num_groups, 128])
        latent = tl.reshape(latent * scale[:, :, None], [BLOCK_N, kv_lora_rank])
        latent = latent.to(tl.bfloat16)
        scores = tl.dot(q_nope, tl.trans(latent))
        if qk_rope_head_dim > 0:
            k_rope = tl.load(
                kv_rope + (kv_rows[:, None] + rope_start) // 2 + rope_dims[None, :],
                mask=valid[:, None] & rope_mask[None, :],
                other=0.0,
            )
            scores = tl.dot(q_rope, tl.trans(k_rope), scores)
        m_i, l_i, acc = _dsa_softmax_update(
            scores, valid, latent, m_i, l_i, acc, softmax_scale
        )

    _dsa_store_output(
        out,
        partial_out,
        partial_lse,
        acc,
        m_i,
        l_i,
        rows,
        head_mask,
        split,
        kv_lora_rank,
        NUM_SPLITS,
    )


@triton.jit
def _dsa_dense_kv_kernel(
    q,
    kv,
    topk_indices,
    topk_lens,
    out,
    partial_out,
    partial_lse,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    kv_dim: tl.constexpr,
    topk: tl.constexpr,
    softmax_scale: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    split = tl.program_id(2)
    head_mask = heads < num_heads
    rows = token * num_heads + heads
    dims = tl.arange(0, kv_lora_rank)

    q_nope = tl.load(
        q + rows[:, None] * head_dim + dims[None, :],
        mask=head_mask[:, None],
        other=0.0,
    ).to(tl.bfloat16)
    if qk_rope_head_dim > 0:
        rope_dims = tl.arange(0, BLOCK_ROPE)
        rope_mask = rope_dims < qk_rope_head_dim
        q_rope = tl.load(
            q + rows[:, None] * head_dim + kv_lora_rank + rope_dims[None, :],
            mask=head_mask[:, None] & rope_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)

    start, end = _dsa_key_range(topk_lens, token, split, topk, NUM_SPLITS, BLOCK_N)
    m_i = tl.full([BLOCK_H], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, kv_lora_rank], tl.float32)
    for block_start in range(start, end, BLOCK_N):
        cols = block_start + tl.arange(0, BLOCK_N)
        slots = tl.load(topk_indices + token * topk + cols, mask=cols < end, other=-1)
        valid = slots >= 0
        kv_rows = slots.to(tl.int64) * kv_dim
        # Gather the rows once; the latent tile serves as both keys and values.
        latent = tl.load(
            kv + kv_rows[:, None] + dims[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        scores = tl.dot(q_nope, tl.trans(latent))
        if qk_rope_head_dim > 0:
            k_rope = tl.load(
                kv + kv_rows[:, None] + kv_lora_rank + rope_dims[None, :],
                mask=valid[:, None] & rope_mask[None, :],
                other=0.0,
            ).to(tl.bfloat16)
            scores = tl.dot(q_rope, tl.trans(k_rope), scores)
        m_i, l_i, acc = _dsa_softmax_update(
            scores, valid, latent, m_i, l_i, acc, softmax_scale
        )

    _dsa_store_output(
        out,
        partial_out,
        partial_lse,
        acc,
        m_i,
        l_i,
        rows,
        head_mask,
        split,
        kv_lora_rank,
        NUM_SPLITS,
    )


@triton.jit
def _dsa_merge_splits_kernel(
    partial_out,
    partial_lse,
    out,
    kv_lora_rank: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Log-sum-exp combine of the split partial outputs of one (token, head).
    row = tl.program_id(0).to(tl.int64)
    dims = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    split_rows = row * NUM_SPLITS + tl.arange(0, NUM_SPLITS)
    lse = tl.load(partial_lse + split_rows)
    max_lse = tl.max(lse, axis=0)
    # Splits without valid keys carry -inf and get zero weight.
    weights = tl.exp2(lse - tl.where(max_lse == -float("inf"), 0.0, max_lse))
    total = tl.sum(weights, axis=0)
    partial = tl.load(partial_out + split_rows[:, None] * kv_lora_rank + dims[None, :])
    merged = tl.sum(partial * weights[:, None], axis=0)
    tl.store(
        out + row * kv_lora_rank + dims,
        tl.where(total > 0.0, merged / total, 0.0).to(out.dtype.element_ty),
    )


def _num_kv_splits(num_programs: int, topk: int) -> int:
    # Split keys until the grid reaches about two programs per SM, at most one
    # split per two key tiles; prefill grids keep one split and skip the merge.
    # Powers of two bound the compiled variants.
    splits = min(
        2 * current_platform().sm_count // max(num_programs, 1),
        triton.cdiv(topk, 2 * _BLOCK_N),
    )
    return 1 << (max(splits, 1).bit_length() - 1)


def _launch_dsa_kernel(
    kernel: triton.JITFunction,
    q: torch.Tensor,
    kv_args: tuple[torch.Tensor, ...],
    kv_row_stride: int,
    topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    tokens, num_heads, head_dim = q.shape
    topk = topk_indices.shape[1]
    out = torch.empty(
        (tokens, num_heads, kv_lora_rank),
        dtype=torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype,
        device=q.device,
    )
    head_blocks = triton.cdiv(num_heads, _BLOCK_H)
    num_splits = _num_kv_splits(tokens * head_blocks, topk)
    # Split programs write FP32 partial outputs and their base-2 log-sum-exp
    # for the merge kernel; a single split writes the output directly.
    partial_out = partial_lse = out
    if num_splits > 1:
        partial_out = torch.empty(
            (tokens, num_heads, num_splits, kv_lora_rank),
            dtype=torch.float32,
            device=q.device,
        )
        partial_lse = torch.empty(
            (tokens, num_heads, num_splits), dtype=torch.float32, device=q.device
        )
    kernel[(tokens, head_blocks, num_splits)](
        q,
        *kv_args,
        topk_indices,
        topk_lens,
        out,
        partial_out,
        partial_lse,
        num_heads,
        head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        kv_row_stride,
        topk,
        float(softmax_scale),
        NUM_SPLITS=num_splits,
        BLOCK_H=_BLOCK_H,
        BLOCK_N=_BLOCK_N,
        BLOCK_ROPE=max(16, triton.next_power_of_2(qk_rope_head_dim)),
        num_warps=4,
        # AMD uses single-stage loops, like the grouped Triton decode kernel.
        num_stages=1 if current_platform().is_amd else 2,
    )
    if num_splits > 1:
        block_d = min(kv_lora_rank, 128)
        _dsa_merge_splits_kernel[(tokens * num_heads, kv_lora_rank // block_d)](
            partial_out,
            partial_lse,
            out,
            kv_lora_rank,
            num_splits,
            BLOCK_D=block_d,
            num_warps=4,
        )
    return out


def _run_packed_kv(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    return _launch_dsa_kernel(
        _dsa_packed_kv_kernel,
        q,
        (
            packed_kv.view(torch.float8_e4m3fn),
            packed_kv.view(torch.float32),
            packed_kv.view(torch.bfloat16),
        ),
        int(packed_kv.shape[1]),
        topk_indices,
        topk_lens,
        softmax_scale=softmax_scale,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
    )


def _run_dense_kv(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> torch.Tensor:
    return _launch_dsa_kernel(
        _dsa_dense_kv_kernel,
        q,
        (kv_cache,),
        int(kv_lora_rank) + int(qk_rope_head_dim),
        topk_indices,
        topk_lens,
        softmax_scale=softmax_scale,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
    )


def _flatten_packed_kv_cache(packed_kv_cache: torch.Tensor) -> torch.Tensor:
    if packed_kv_cache.dim() == 2:
        return packed_kv_cache
    return packed_kv_cache.reshape(-1, packed_kv_cache.shape[-1])


def _flatten_dense_kv_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    if kv_cache.dim() == 2:
        return kv_cache
    if kv_cache.dim() == 3:
        return kv_cache.squeeze(1)
    if kv_cache.shape[1] == 1:
        kv_cache = kv_cache.permute(0, 2, 1, 3)
    return kv_cache.reshape(-1, kv_cache.shape[-1])


def _flatten_query(q: torch.Tensor) -> torch.Tensor:
    if q.dim() == 3:
        return q
    return q.reshape(-1, q.shape[-2], q.shape[-1])


def _run_dsa(
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    packed_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    k_scale: float,
    out: torch.Tensor | None,
) -> torch.Tensor:
    q = _flatten_query(q).contiguous()
    topk_slots = topk_slots.contiguous()
    topk_lens = topk_lens.contiguous()
    softmax_scale = float(softmax_scale) * float(k_scale)

    if packed_kv_cache is not None:
        result = _run_packed_kv(
            q,
            _flatten_packed_kv_cache(packed_kv_cache).contiguous(),
            topk_slots,
            topk_lens,
            softmax_scale=softmax_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
    else:
        result = _run_dense_kv(
            q,
            _flatten_dense_kv_cache(kv_cache).contiguous(),
            topk_slots,
            topk_lens,
            softmax_scale=softmax_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )

    if out is None:
        return result
    out_view = out.reshape_as(result)
    out_view.copy_(result)
    return out


@register_kernel(
    "attention",
    "dsa_decode",
    name="triton_dsa_decode",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(q=dense_tensor_format(torch.bfloat16)),
            format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
        }
    ),
    traits={
        "q_len": frozenset({1, 2, 3, 4, 5, 6}),
        "qk_nope_head_dim": frozenset({128, 192, 256}),
        "kv_lora_rank": frozenset({128, 512}),
        "qk_rope_head_dim": frozenset({0, 64}),
        "page_size": frozenset({64}),
        "topk": frozenset({512, 1024, 2048, 2049, 2050, 2051}),
        "has_kv_cache": frozenset({False, True}),
        "has_sparse_kv_cache": frozenset({False, True}),
        "logit_cap": frozenset({False}),
        "return_lse": frozenset({False}),
        "topk_layout": frozenset({"global_slots"}),
    },
    priority=Priority.PORTABLE,
)
def triton_dsa_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    sparse_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    q_len_per_req: int = 1,
    kv_seq_lens: torch.Tensor | None = None,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    del kv_seq_lens
    return _run_dsa(
        q=q,
        kv_cache=kv_cache,
        packed_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        k_scale=k_scale,
        out=out,
    )


@register_kernel(
    "attention",
    "dsa_prefill",
    name="triton_dsa_prefill",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset(
        {
            format_signature(q=dense_tensor_format(torch.bfloat16)),
            format_signature(q=dense_tensor_format(torch.float8_e4m3fn)),
        }
    ),
    traits={
        "q_len": frozenset({1}),
        "qk_nope_head_dim": frozenset({128, 192, 256}),
        "kv_lora_rank": frozenset({128, 512}),
        "qk_rope_head_dim": frozenset({0, 64}),
        "page_size": frozenset({64}),
        "topk": frozenset({512, 1024, 2048, 2049, 2050, 2051}),
        "has_kv_cache": frozenset({False, True}),
        "has_sparse_kv_cache": frozenset({False, True}),
        "logit_cap": frozenset({False}),
        "return_lse": frozenset({False}),
        "topk_layout": frozenset({"global_slots"}),
    },
    priority=Priority.PORTABLE,
)
def triton_dsa_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    sparse_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    q_len_per_req: int = 1,
    kv_seq_lens: torch.Tensor | None = None,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    del kv_seq_lens
    return _run_dsa(
        q=q,
        kv_cache=kv_cache,
        packed_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        k_scale=k_scale,
        out=out,
    )


from tokenspeed_kernel.ops.attention.dsa._triton.topk import *  # noqa: E402,F403
from tokenspeed_kernel.ops.attention.dsa._triton.topk import (  # noqa: E402
    _topk_with_padding,
    _triton_dsa_decode_topk_fp8_impl,
    _triton_dsa_plan_impl,
    _triton_dsa_prefill_topk_fp8_impl,
)


@register_kernel(
    "attention",
    "dsa_plan",
    name="triton_dsa_plan",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=frozenset({format_signature()}),
    traits={"page_size": frozenset({64})},
    priority=Priority.PORTABLE,
)
def triton_dsa_plan(
    *,
    page_size: int,
    seq_lens_2d: torch.Tensor,
    out: object | None = None,
) -> torch.Tensor:
    return _triton_dsa_plan_impl(
        page_size=page_size,
        seq_lens_2d=seq_lens_2d,
        out=out,
    )


_TOPK_SIGNATURES = frozenset(
    {
        format_signature(
            q=dense_tensor_format(torch.bfloat16),
            weights=dense_tensor_format(torch.float32),
        ),
        format_signature(
            q=dense_tensor_format(torch.bfloat16),
            weights=dense_tensor_format(torch.bfloat16),
        ),
    }
)


@register_kernel(
    "attention",
    "dsa_decode_topk",
    name="triton_dsa_decode_topk_fp8",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=_TOPK_SIGNATURES,
    traits={
        "head_dim": frozenset({128}),
        "page_size": frozenset({64}),
        "topk": frozenset({512, 1024, 2048}),
        "index_k_format": frozenset({"fp8_scaled"}),
        "index_k_layout": frozenset({"packed", "page_planar"}),
    },
    features={"logical_offsets"},
    priority=Priority.PORTABLE,
)
def triton_dsa_decode_topk_fp8(
    q: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
    q_len_per_req: int = 1,
    topk_layout: str = "global_slots",
    block_table_base_offsets: torch.Tensor | None = None,
    index_k_cache: torch.Tensor | None = None,
    seq_lens_2d: torch.Tensor | None = None,
    plan: object | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _triton_dsa_decode_topk_fp8_impl(
        q=q,
        weights=weights,
        seq_lens=seq_lens,
        block_table=block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=softmax_scale,
        q_len_per_req=q_len_per_req,
        topk_layout=topk_layout,
        block_table_base_offsets=block_table_base_offsets,
        index_k_cache=index_k_cache,
        seq_lens_2d=seq_lens_2d,
        plan=plan,
        out=out,
        lens_out=lens_out,
    )


@register_kernel(
    "attention",
    "dsa_prefill_topk",
    name="triton_dsa_prefill_topk_fp8",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=_TOPK_SIGNATURES,
    traits={
        "head_dim": frozenset({128}),
        "topk": frozenset({512, 1024, 2048}),
        "index_k_format": frozenset({"fp8_scaled"}),
        "index_k_layout": frozenset({"packed", "page_planar"}),
    },
    priority=Priority.PORTABLE,
)
def triton_dsa_prefill_topk_fp8(
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    softmax_scale: float,
    index_k_cache: torch.Tensor | None = None,
    page_size: int | None = None,
    index_k_fp8: torch.Tensor | None = None,
    index_k_scale: torch.Tensor | None = None,
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _triton_dsa_prefill_topk_fp8_impl(
        q=q,
        weights=weights,
        kv_workspace_slots=kv_workspace_slots,
        row_starts=row_starts,
        row_ends=row_ends,
        topk=topk,
        softmax_scale=softmax_scale,
        index_k_cache=index_k_cache,
        page_size=page_size,
        index_k_fp8=index_k_fp8,
        index_k_scale=index_k_scale,
        max_logits_bytes=max_logits_bytes,
        out=out,
        lens_out=lens_out,
    )
