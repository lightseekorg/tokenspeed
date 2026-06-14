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

GLM_DSA_FP8_NOPE_DIM = 512
GLM_DSA_ROPE_DIM = 64
GLM_DSA_FP8_QUANT_BLOCK = 128
GLM_DSA_FP8_SCALE_BYTES = 4
GLM_DSA_FP8_MAX = 448.0
GLM_DSA_SPARSE_DECODE_ROW_BYTES = (
    GLM_DSA_FP8_NOPE_DIM
    + GLM_DSA_FP8_NOPE_DIM // GLM_DSA_FP8_QUANT_BLOCK * GLM_DSA_FP8_SCALE_BYTES
    + GLM_DSA_ROPE_DIM * 2
)

__all__ = [
    "GLM_DSA_SPARSE_DECODE_ROW_BYTES",
    "glm_dsa_full_context_topk_to_global_slots",
    "glm_dsa_local_topk_to_global_slots",
    "glm_dsa_pack_sparse_decode_kv",
]


@triton.jit
def _glm_dsa_pack_sparse_decode_kv_kernel(
    out_ptr,
    out_stride,
    loc_ptr,
    nope_ptr,
    nope_stride,
    rope_ptr,
    rope_stride,
    fp8_max: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    quant_block: tl.constexpr,
    scale_bytes: tl.constexpr,
    num_nope_blocks: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot = tl.load(loc_ptr + token_idx).to(tl.int64)
    row = out_ptr + slot * out_stride

    nope_offsets = tl.arange(0, nope_dim)
    x = tl.load(
        nope_ptr + token_idx * nope_stride + nope_offsets,
        mask=nope_offsets < nope_dim,
        other=0.0,
    ).to(tl.float32)
    x_2d = tl.reshape(x, (num_nope_blocks, quant_block))
    scale = tl.max(tl.abs(x_2d), axis=1) / fp8_max
    scale = tl.maximum(scale, 1.0e-26)
    x_scaled = x_2d / tl.reshape(scale, (num_nope_blocks, 1))
    x_fp8 = tl.clamp(x_scaled, -fp8_max, fp8_max).to(tl.float8e4nv)
    x_u8 = tl.reshape(x_fp8.to(tl.uint8, bitcast=True), (nope_dim,))
    tl.store(row + nope_offsets, x_u8, mask=nope_offsets < nope_dim)

    scale_offsets = tl.arange(0, num_nope_blocks)
    scale_ptr = (row + nope_dim).to(tl.pointer_type(tl.float32))
    tl.store(scale_ptr + scale_offsets, scale, mask=scale_offsets < num_nope_blocks)

    rope_offsets = tl.arange(0, rope_dim)
    rope_values = tl.load(
        rope_ptr + token_idx * rope_stride + rope_offsets,
        mask=rope_offsets < rope_dim,
        other=0.0,
    )
    rope_ptr_out = (row + nope_dim + num_nope_blocks * scale_bytes).to(
        tl.pointer_type(tl.bfloat16)
    )
    tl.store(rope_ptr_out + rope_offsets, rope_values, mask=rope_offsets < rope_dim)


def _glm_dsa_prepare_pack_inputs(
    *,
    out: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if out.dtype != torch.uint8:
        raise TypeError(f"out must be uint8, got {out.dtype}")
    if out.dim() != 2 or out.shape[1] != GLM_DSA_SPARSE_DECODE_ROW_BYTES:
        raise ValueError(
            "out must be [slots, row_bytes] for GLM DSA sparse decode, got "
            f"{tuple(out.shape)}"
        )
    if loc.dim() != 1:
        raise ValueError(f"loc must be 1-D, got {tuple(loc.shape)}")
    if loc.numel() == 0:
        return cache_k_nope, cache_k_rope

    if cache_k_nope.dim() == 3:
        if cache_k_nope.shape[1] != 1:
            raise ValueError(
                "cache_k_nope must have one KV head, got "
                f"{tuple(cache_k_nope.shape)}"
            )
        cache_k_nope = cache_k_nope.squeeze(1)
    if cache_k_rope.dim() == 3:
        if cache_k_rope.shape[1] != 1:
            raise ValueError(
                "cache_k_rope must have one KV head, got "
                f"{tuple(cache_k_rope.shape)}"
            )
        cache_k_rope = cache_k_rope.squeeze(1)
    if cache_k_nope.dim() != 2 or cache_k_nope.shape[1] != GLM_DSA_FP8_NOPE_DIM:
        raise ValueError(
            "cache_k_nope must be [tokens, 512], got " f"{tuple(cache_k_nope.shape)}"
        )
    if cache_k_rope.dim() != 2 or cache_k_rope.shape[1] != GLM_DSA_ROPE_DIM:
        raise ValueError(
            "cache_k_rope must be [tokens, 64], got " f"{tuple(cache_k_rope.shape)}"
        )
    if cache_k_nope.shape[0] != loc.numel() or cache_k_rope.shape[0] != loc.numel():
        raise ValueError(
            "GLM DSA sparse decode pack token mismatch: "
            f"loc={loc.numel()}, nope={cache_k_nope.shape[0]}, "
            f"rope={cache_k_rope.shape[0]}"
        )
    if cache_k_nope.dtype != torch.bfloat16 or cache_k_rope.dtype != torch.bfloat16:
        raise TypeError(
            "GLM DSA sparse decode cache pack requires BF16 source tensors, got "
            f"nope={cache_k_nope.dtype}, rope={cache_k_rope.dtype}"
        )
    return cache_k_nope.contiguous(), cache_k_rope.contiguous()


def _glm_dsa_pack_sparse_decode_kv_cpu(
    *,
    out: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
) -> None:
    nope = cache_k_nope.float().view(
        cache_k_nope.shape[0],
        GLM_DSA_FP8_NOPE_DIM // GLM_DSA_FP8_QUANT_BLOCK,
        GLM_DSA_FP8_QUANT_BLOCK,
    )
    scale = nope.abs().amax(dim=-1).clamp_min(1.0e-26) / GLM_DSA_FP8_MAX
    nope_u8 = (
        torch.clamp(
            nope / scale.unsqueeze(-1),
            min=-GLM_DSA_FP8_MAX,
            max=GLM_DSA_FP8_MAX,
        )
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .view(cache_k_nope.shape[0], GLM_DSA_FP8_NOPE_DIM)
    )
    scale_u8 = (
        scale.contiguous()
        .view(torch.uint8)
        .view(
            cache_k_nope.shape[0],
            GLM_DSA_FP8_NOPE_DIM // GLM_DSA_FP8_QUANT_BLOCK * GLM_DSA_FP8_SCALE_BYTES,
        )
    )
    rope_u8 = (
        cache_k_rope.contiguous()
        .view(torch.uint8)
        .view(
            cache_k_rope.shape[0],
            GLM_DSA_ROPE_DIM * 2,
        )
    )
    rows = out.view(-1, GLM_DSA_SPARSE_DECODE_ROW_BYTES)
    loc = loc.to(device=out.device, dtype=torch.long)
    rows[loc, :GLM_DSA_FP8_NOPE_DIM] = nope_u8
    scale_start = GLM_DSA_FP8_NOPE_DIM
    rope_start = scale_start + scale_u8.shape[1]
    rows[loc, scale_start:rope_start] = scale_u8
    rows[loc, rope_start:] = rope_u8


def glm_dsa_pack_sparse_decode_kv(
    *,
    out: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
) -> None:
    cache_k_nope, cache_k_rope = _glm_dsa_prepare_pack_inputs(
        out=out,
        loc=loc,
        cache_k_nope=cache_k_nope,
        cache_k_rope=cache_k_rope,
    )
    if loc.numel() == 0:
        return
    if not out.is_cuda:
        _glm_dsa_pack_sparse_decode_kv_cpu(
            out=out,
            loc=loc,
            cache_k_nope=cache_k_nope,
            cache_k_rope=cache_k_rope,
        )
        return

    _glm_dsa_pack_sparse_decode_kv_kernel[(loc.numel(),)](
        out,
        out.stride(0),
        loc.to(torch.int64),
        cache_k_nope,
        cache_k_nope.stride(0),
        cache_k_rope,
        cache_k_rope.stride(0),
        fp8_max=GLM_DSA_FP8_MAX,
        nope_dim=GLM_DSA_FP8_NOPE_DIM,
        rope_dim=GLM_DSA_ROPE_DIM,
        quant_block=GLM_DSA_FP8_QUANT_BLOCK,
        scale_bytes=GLM_DSA_FP8_SCALE_BYTES,
        num_nope_blocks=GLM_DSA_FP8_NOPE_DIM // GLM_DSA_FP8_QUANT_BLOCK,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def _glm_dsa_local_topk_to_global_slots_kernel(
    global_topk_slots_ptr,
    global_topk_slots_stride,
    topk_lens_ptr,
    local_topk_offsets_ptr,
    local_topk_offsets_stride,
    seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    block_table_cols: tl.constexpr,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    has_seq_lens: tl.constexpr,
    block: tl.constexpr,
):
    token_idx = tl.program_id(0)
    count = tl.zeros((), dtype=tl.int32)
    seq_len = tl.full((), block_table_cols * block_size, dtype=tl.int32)
    if has_seq_lens:
        seq_len = tl.load(seq_lens_ptr + token_idx).to(tl.int32)

    for start in range(0, topk, block):
        offsets = start + tl.arange(0, block)
        mask = offsets < topk
        local_idx = tl.load(
            local_topk_offsets_ptr + token_idx * local_topk_offsets_stride + offsets,
            mask=mask,
            other=-1,
        )
        valid = (local_idx >= 0) & (local_idx < seq_len)
        block_idx = local_idx // block_size
        block_offset = local_idx % block_size
        valid = valid & (block_idx >= 0) & (block_idx < block_table_cols)
        page = tl.load(
            block_table_ptr + token_idx * block_table_stride + block_idx,
            mask=mask & valid,
            other=0,
        )
        slot = page * block_size + block_offset
        tl.store(
            global_topk_slots_ptr + token_idx * global_topk_slots_stride + offsets,
            tl.where(valid, slot, -1),
            mask=mask,
        )
        count += tl.sum(valid.to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, count)


def glm_dsa_local_topk_to_global_slots(
    *,
    local_topk_offsets: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    seq_lens: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if local_topk_offsets.dtype != torch.int32:
        raise TypeError(
            f"local_topk_offsets must be int32, got {local_topk_offsets.dtype}"
        )
    if local_topk_offsets.dim() != 2:
        raise ValueError(
            "local_topk_offsets must be [tokens, topk], got "
            f"{tuple(local_topk_offsets.shape)}"
        )
    if block_table.dim() != 2:
        raise ValueError(
            f"block_table must be [tokens, pages], got {block_table.shape}"
        )
    if block_table.shape[1] == 0:
        raise ValueError("block_table must have at least one page column")
    num_tokens, topk = local_topk_offsets.shape
    if block_table.shape[0] < num_tokens:
        raise ValueError(
            "block_table must have at least one row per token: "
            f"rows={block_table.shape[0]}, tokens={num_tokens}"
        )
    if seq_lens is not None and seq_lens.dim() != 1:
        raise ValueError(f"seq_lens must be 1-D, got {tuple(seq_lens.shape)}")
    if seq_lens is not None and seq_lens.numel() < num_tokens:
        raise ValueError(
            "seq_lens must have at least one entry per token: "
            f"lens={seq_lens.numel()}, tokens={num_tokens}"
        )

    if out is None:
        global_slots = torch.empty_like(local_topk_offsets)
    else:
        if out.shape != local_topk_offsets.shape:
            raise ValueError(
                f"out must have shape {tuple(local_topk_offsets.shape)}, "
                f"got {tuple(out.shape)}"
            )
        if out.dtype != torch.int32 or out.device != local_topk_offsets.device:
            raise TypeError(
                "out must be int32 on the same device as local_topk_offsets, "
                f"got {out.dtype} on {out.device}"
            )
        global_slots = out
    if lens_out is None:
        lens = torch.empty(
            num_tokens,
            dtype=torch.int32,
            device=local_topk_offsets.device,
        )
    else:
        if lens_out.shape != (num_tokens,):
            raise ValueError(
                "lens_out must have shape "
                f"({num_tokens},), got {tuple(lens_out.shape)}"
            )
        if (
            lens_out.dtype != torch.int32
            or lens_out.device != local_topk_offsets.device
        ):
            raise TypeError(
                "lens_out must be int32 on the same device as local_topk_offsets, "
                f"got {lens_out.dtype} on {lens_out.device}"
            )
        lens = lens_out
    if num_tokens == 0:
        return global_slots, lens

    block_table = block_table.to(device=local_topk_offsets.device, dtype=torch.int32)
    if seq_lens is not None:
        seq_lens = seq_lens.to(device=local_topk_offsets.device, dtype=torch.int32)
    if not local_topk_offsets.is_cuda:
        valid = local_topk_offsets >= 0
        if seq_lens is not None:
            valid &= local_topk_offsets < seq_lens[:num_tokens].view(num_tokens, 1)
        safe_local = torch.where(
            valid,
            local_topk_offsets,
            torch.zeros_like(local_topk_offsets),
        )
        block_idx = torch.div(safe_local, block_size, rounding_mode="floor")
        valid &= (block_idx >= 0) & (block_idx < block_table.shape[1])
        block_offset = safe_local % block_size
        row_idx = torch.arange(num_tokens, device=local_topk_offsets.device)
        pages = block_table[
            row_idx[:, None],
            block_idx.long().clamp(0, block_table.shape[1] - 1),
        ]
        global_slots.copy_(
            torch.where(
                valid,
                pages.to(torch.int32) * int(block_size) + block_offset,
                torch.full_like(local_topk_offsets, -1),
            )
        )
        lens.copy_(valid.sum(dim=1, dtype=torch.int32))
        return global_slots, lens

    seq_lens_arg = block_table[:, 0] if seq_lens is None else seq_lens
    _glm_dsa_local_topk_to_global_slots_kernel[(num_tokens,)](
        global_slots,
        global_slots.stride(0),
        lens,
        local_topk_offsets,
        local_topk_offsets.stride(0),
        seq_lens_arg,
        block_table,
        block_table.stride(0),
        block_table.shape[1],
        block_size=int(block_size),
        topk=topk,
        has_seq_lens=seq_lens is not None,
        block=1024,
    )
    return global_slots, lens


@triton.jit
def _glm_dsa_full_context_topk_to_global_slots_kernel(
    global_topk_slots_ptr,
    global_topk_slots_stride,
    topk_lens_ptr,
    seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    block_table_cols: tl.constexpr,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    block: tl.constexpr,
):
    token_idx = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + token_idx).to(tl.int32)
    max_context_len = block_table_cols * block_size
    capped_seq_len = tl.minimum(seq_len, max_context_len)
    topk_len = tl.minimum(capped_seq_len, topk)

    for start in range(0, topk, block):
        offsets = start + tl.arange(0, block)
        mask = offsets < topk
        block_idx = offsets // block_size
        block_offset = offsets % block_size
        valid = (offsets < seq_len) & (block_idx >= 0) & (block_idx < block_table_cols)
        page = tl.load(
            block_table_ptr + token_idx * block_table_stride + block_idx,
            mask=mask & valid,
            other=0,
        )
        slot = page * block_size + block_offset
        tl.store(
            global_topk_slots_ptr + token_idx * global_topk_slots_stride + offsets,
            tl.where(valid, slot, -1),
            mask=mask,
        )

    tl.store(topk_lens_ptr + token_idx, topk_len)


def glm_dsa_full_context_topk_to_global_slots(
    *,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    topk: int,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if seq_lens.dim() != 1:
        raise ValueError(f"seq_lens must be 1-D, got {tuple(seq_lens.shape)}")
    if block_table.dim() != 2:
        raise ValueError(
            f"block_table must be [tokens, pages], got {tuple(block_table.shape)}"
        )
    if block_table.shape[1] == 0:
        raise ValueError("block_table must have at least one page column")
    num_tokens = seq_lens.numel()
    if block_table.shape[0] < num_tokens:
        raise ValueError(
            "block_table must have at least one row per token: "
            f"rows={block_table.shape[0]}, tokens={num_tokens}"
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")

    device = out.device if out is not None else seq_lens.device
    seq_lens = seq_lens.to(device=device, dtype=torch.int32)
    block_table = block_table.to(device=device, dtype=torch.int32)
    expected_shape = (num_tokens, int(topk))
    if out is None:
        global_slots = torch.empty(
            expected_shape,
            dtype=torch.int32,
            device=device,
        )
    else:
        if out.shape != expected_shape:
            raise ValueError(
                f"out must have shape {expected_shape}, got {tuple(out.shape)}"
            )
        if out.dtype != torch.int32:
            raise TypeError(f"out must be int32, got {out.dtype}")
        global_slots = out
    if lens_out is None:
        lens = torch.empty(num_tokens, dtype=torch.int32, device=device)
    else:
        if lens_out.shape != (num_tokens,):
            raise ValueError(
                "lens_out must have shape "
                f"({num_tokens},), got {tuple(lens_out.shape)}"
            )
        if lens_out.dtype != torch.int32 or lens_out.device != device:
            raise TypeError(
                f"lens_out must be int32 on {device}, got {lens_out.dtype} "
                f"on {lens_out.device}"
            )
        lens = lens_out
    if num_tokens == 0:
        return global_slots, lens

    if not seq_lens.is_cuda:
        offsets = torch.arange(int(topk), dtype=torch.int32, device=device)
        block_idx = torch.div(offsets, block_size, rounding_mode="floor")
        valid = (offsets.view(1, int(topk)) < seq_lens.view(num_tokens, 1)) & (
            block_idx.view(1, int(topk)) < block_table.shape[1]
        )
        block_offset = offsets % block_size
        pages = block_table[
            torch.arange(num_tokens, device=device).view(num_tokens, 1),
            block_idx.long().view(1, int(topk)).clamp(0, block_table.shape[1] - 1),
        ]
        global_slots.copy_(
            torch.where(
                valid,
                pages.to(torch.int32) * int(block_size) + block_offset.view(1, topk),
                torch.full_like(global_slots, -1),
            )
        )
        max_context_len = int(block_table.shape[1]) * int(block_size)
        lens.copy_(
            torch.minimum(
                torch.minimum(seq_lens, torch.full_like(seq_lens, max_context_len)),
                torch.full_like(seq_lens, int(topk)),
            )
        )
        return global_slots, lens

    _glm_dsa_full_context_topk_to_global_slots_kernel[(num_tokens,)](
        global_slots,
        global_slots.stride(0),
        lens,
        seq_lens,
        block_table,
        block_table.stride(0),
        block_table.shape[1],
        block_size=int(block_size),
        topk=int(topk),
        block=1024,
    )
    return global_slots, lens
