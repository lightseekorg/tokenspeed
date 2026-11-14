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
from tokenspeed_kernel.ops.kvcache.triton import (
    _index_k_layernorm_fp8_scatter_args,
    _index_k_layernorm_fp8_scatter_row,
)
from tokenspeed_kernel.platform import current_platform, pdl_enabled

_is_nvidia = current_platform().is_nvidia

_RADIX_TOPK_MIN_COLS = 32768
_RADIX_TOPK_BLOCK_N = 4096
# Query heads per program of the fused index-K write; the row's key rides on the first block.
_FUSED_QUERY_HEADS_PER_PROGRAM = 8


@triton.jit
def _mark_forced_initial_local_logits_kernel(
    logits_ptr,
    logits_row_stride,
    causal_lens_ptr,
    # Candidate width follows the batch; runtime so every batch shape shares
    # one binary.
    num_cols,
    initial_tokens: tl.constexpr,
    local_tokens: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK + tl.arange(0, BLOCK)
    causal_len = tl.load(causal_lens_ptr + row).to(tl.int32)
    initial_end = tl.minimum(causal_len, initial_tokens)
    local_start = tl.maximum(initial_end, causal_len - local_tokens)
    forced = (cols < initial_end) | ((cols >= local_start) & (cols < causal_len))
    tl.store(
        logits_ptr + row * logits_row_stride + cols,
        float("inf"),
        mask=(cols < num_cols) & forced,
    )


def mark_forced_initial_local_logits(
    logits: torch.Tensor,
    causal_lens: torch.Tensor,
    *,
    initial_tokens: int,
    local_tokens: int,
) -> None:
    """Force sequence-start and causal-tail candidates into the top-k.

    LongCat's DSA selection always keeps the first ``initial_tokens`` and the
    last ``local_tokens`` visible positions; marking their logits +inf makes
    any top-k over the marked logits include them.

    Args:
        logits: ``[rows, candidates]`` scores, unit column stride; mutated.
        causal_lens: ``[rows]`` visible candidate count per row.
        initial_tokens: Sequence-start candidates to force.
        local_tokens: Causal-tail candidates to force.
    """
    if logits.dim() != 2 or logits.stride(1) != 1:
        raise ValueError("logits must be a 2-D tensor with unit column stride")
    if causal_lens.shape != (logits.shape[0],):
        raise ValueError(
            f"causal_lens must have shape ({logits.shape[0]},), got "
            f"{tuple(causal_lens.shape)}"
        )
    if initial_tokens < 0 or local_tokens < 0:
        raise ValueError("initial_tokens and local_tokens must be non-negative")
    if logits.numel() == 0 or initial_tokens + local_tokens == 0:
        return
    causal_lens = causal_lens.to(device=logits.device, dtype=torch.int32).contiguous()
    block = 256
    _mark_forced_initial_local_logits_kernel[
        (logits.shape[0], triton.cdiv(logits.shape[1], block))
    ](
        logits,
        logits.stride(0),
        causal_lens,
        num_cols=logits.shape[1],
        initial_tokens=int(initial_tokens),
        local_tokens=int(local_tokens),
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def _scrub_nonfinite_logits_kernel(
    logits_ptr,
    logits_row_stride,
    seq_lens_ptr,
    # Width follows the block table; runtime so every width shares one binary.
    num_cols,
    NEXT_N: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    # Verify row j of a request sees its seq_len - (NEXT_N - 1 - j) causal candidates.
    seq_len = tl.load(seq_lens_ptr + row // NEXT_N).to(tl.int32)
    length = seq_len - (NEXT_N - 1) + (row % NEXT_N).to(tl.int32)
    length = tl.minimum(tl.maximum(length, 0), num_cols)
    for start in range(tl.program_id(1) * BLOCK, length, SPLITS * BLOCK):
        cols = start + tl.arange(0, BLOCK)
        ptrs = logits_ptr + row * logits_row_stride + cols
        score = tl.load(ptrs, mask=cols < length, other=0.0)
        bad = (score != score) | (score == float("inf"))
        tl.store(ptrs, float("-inf"), mask=(cols < length) & bad)


def scrub_nonfinite_logits(
    logits: torch.Tensor, seq_lens: torch.Tensor, *, next_n: int
) -> None:
    """Set NaN and +inf scores to -inf in each row's causal candidates.

    Row ``r`` holds verify row ``r % next_n`` of request ``r // next_n``,
    whose candidates are the first ``seq_lens[r // next_n] - next_n + 1 +
    r % next_n`` columns (clamped to the width). Columns past them are left as
    they are: the length-aware top-k kernels never read them, so the work
    follows the context, not the width.

    Args:
        logits: ``[requests * next_n, candidates]`` FP32 scores, unit column
            stride; mutated.
        seq_lens: ``[requests]`` context length per request.
        next_n: Query rows per request (1 for plain decode).
    """
    if logits.dim() != 2 or logits.stride(1) != 1:
        raise ValueError("logits must be a 2-D tensor with unit column stride")
    if next_n < 1 or seq_lens.shape != (logits.shape[0] // next_n,):
        raise ValueError(
            f"seq_lens must have shape ({logits.shape[0]} // next_n,) for "
            f"next_n={next_n}, got {tuple(seq_lens.shape)}"
        )
    if logits.shape[0] % next_n:
        raise ValueError(f"{logits.shape[0]} rows are not whole requests of {next_n}")
    if logits.numel() == 0:
        return
    seq_lens = seq_lens.to(device=logits.device, dtype=torch.int32).contiguous()
    splits = 16
    _scrub_nonfinite_logits_kernel[(logits.shape[0], splits)](
        logits,
        logits.stride(0),
        seq_lens,
        num_cols=logits.shape[1],
        NEXT_N=next_n,
        SPLITS=splits,
        BLOCK=1024,
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def _local_topk_to_global_slots_kernel(
    global_topk_slots_ptr,
    global_topk_slots_stride,
    topk_lens_ptr,
    local_topk_offsets_ptr,
    local_topk_offsets_stride,
    seq_lens_ptr,
    block_table_ptr,
    block_table_base_offsets_ptr,
    block_table_stride,
    # Block-table width follows the batch; runtime so every batch shape
    # shares one binary.
    block_table_cols,
    block_size: tl.constexpr,
    topk: tl.constexpr,
    has_seq_lens: tl.constexpr,
    has_block_table_base_offsets: tl.constexpr,
    return_logical_offsets: tl.constexpr,
    q_len_per_req: tl.constexpr,
    block: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_idx = token_idx // q_len_per_req
    count = tl.zeros((), dtype=tl.int32)
    seq_len = tl.full((), block_table_cols * block_size, dtype=tl.int32)
    if has_seq_lens:
        base = tl.load(seq_lens_ptr + req_idx).to(tl.int32)
        seq_len = base - (q_len_per_req - 1) + (token_idx % q_len_per_req)
        seq_len = tl.maximum(seq_len, 0)

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
        if return_logical_offsets:
            base_page = tl.zeros((), dtype=tl.int64)
            if has_block_table_base_offsets:
                base_page = tl.load(block_table_base_offsets_ptr + req_idx).to(tl.int64)
            slot = local_idx.to(tl.int64) + base_page * block_size
        else:
            page = tl.load(
                block_table_ptr + req_idx * block_table_stride + block_idx,
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


def local_topk_to_global_slots(
    *,
    local_topk_offsets: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    seq_lens: torch.Tensor | None = None,
    q_len_per_req: int = 1,
    topk_layout: str = "global_slots",
    block_table_base_offsets: torch.Tensor | None = None,
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
    if topk_layout not in ("global_slots", "logical_offsets"):
        raise ValueError(
            "topk_layout must be 'global_slots' or 'logical_offsets', got "
            f"{topk_layout!r}"
        )
    q_len_per_req = int(q_len_per_req)
    if q_len_per_req < 1 or num_tokens % q_len_per_req != 0:
        raise ValueError(
            f"q_len_per_req={q_len_per_req} must divide tokens={num_tokens}"
        )
    num_reqs = num_tokens // q_len_per_req
    if block_table.shape[0] < num_reqs:
        raise ValueError(
            "block_table must have at least one row per request: "
            f"rows={block_table.shape[0]}, reqs={num_reqs}"
        )
    if seq_lens is not None and seq_lens.dim() != 1:
        raise ValueError(f"seq_lens must be 1-D, got {tuple(seq_lens.shape)}")
    if seq_lens is not None and seq_lens.numel() < num_reqs:
        raise ValueError(
            "seq_lens must have at least one entry per request: "
            f"lens={seq_lens.numel()}, reqs={num_reqs}"
        )
    if block_table_base_offsets is not None and (
        block_table_base_offsets.dim() != 1
        or block_table_base_offsets.numel() < num_reqs
    ):
        raise ValueError(
            "block_table_base_offsets must have at least one entry per request"
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
                f"lens_out must have shape ({num_tokens},), got {tuple(lens_out.shape)}"
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

    if not local_topk_offsets.is_cuda:
        raise RuntimeError("DSA local top-k slot conversion requires CUDA tensors.")

    block_table = block_table.to(device=local_topk_offsets.device, dtype=torch.int32)
    if seq_lens is not None:
        seq_lens = seq_lens.to(device=local_topk_offsets.device, dtype=torch.int32)
    if block_table_base_offsets is not None:
        block_table_base_offsets = block_table_base_offsets.to(
            device=local_topk_offsets.device, dtype=torch.int32
        )
    seq_lens_arg = block_table[:, 0] if seq_lens is None else seq_lens
    base_offsets_arg = (
        block_table[:, 0]
        if block_table_base_offsets is None
        else block_table_base_offsets
    )
    _local_topk_to_global_slots_kernel[(num_tokens,)](
        global_slots,
        global_slots.stride(0),
        lens,
        local_topk_offsets,
        local_topk_offsets.stride(0),
        seq_lens_arg,
        block_table,
        base_offsets_arg,
        block_table.stride(0),
        block_table.shape[1],
        block_size=int(block_size),
        topk=topk,
        has_seq_lens=seq_lens is not None,
        has_block_table_base_offsets=block_table_base_offsets is not None,
        return_logical_offsets=topk_layout == "logical_offsets",
        q_len_per_req=q_len_per_req,
        block=1024,
    )
    return global_slots, lens


@triton.jit(
    do_not_specialize=["total"],
)
def _workspace_topk_to_global_slots_kernel(
    out_ptr,
    workspace_indices_ptr,
    kv_workspace_slots_ptr,
    total,
    BLOCK: tl.constexpr,
    U16_INDICES: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    workspace_idx = tl.load(workspace_indices_ptr + offsets, mask=mask, other=-1)
    if U16_INDICES:
        # int16 storage of unsigned indices; 0xFFFF (-1) marks an empty slot.
        workspace_idx = workspace_idx.to(tl.int32)
        valid = mask & (workspace_idx != -1)
        workspace_idx = workspace_idx & 0xFFFF
    else:
        valid = mask & (workspace_idx >= 0)
    slots = tl.load(kv_workspace_slots_ptr + workspace_idx, mask=valid, other=-1)
    tl.store(out_ptr + offsets, tl.where(valid, slots, -1), mask=mask)


def workspace_topk_to_global_slots(
    *,
    workspace_indices: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Map workspace top-k indices to int32 global KV slots, -1 where empty.

    int16 ``workspace_indices`` hold unsigned 16-bit indices with ``-1``
    (0xFFFF) as the empty marker, so the workspace must have fewer than 65535
    rows; int32 indices mark empty entries with any negative value.
    """
    if workspace_indices.dtype not in (torch.int32, torch.int16):
        raise TypeError(
            f"workspace_indices must be int32 or int16, got {workspace_indices.dtype}"
        )
    if workspace_indices.dim() != 2:
        raise ValueError(
            "workspace_indices must be [tokens, topk], got "
            f"{tuple(workspace_indices.shape)}"
        )
    if kv_workspace_slots.dim() != 1:
        raise ValueError(
            f"kv_workspace_slots must be 1-D, got {tuple(kv_workspace_slots.shape)}"
        )
    if workspace_indices.dtype == torch.int16 and kv_workspace_slots.numel() >= 0xFFFF:
        raise ValueError(
            "int16 workspace_indices need fewer than 65535 workspace rows, got "
            f"{kv_workspace_slots.numel()}"
        )
    if out is None:
        out = torch.empty_like(workspace_indices, dtype=torch.int32)
    elif (
        out.shape != workspace_indices.shape
        or out.dtype != torch.int32
        or out.device != workspace_indices.device
    ):
        raise ValueError(
            "out must be int32 with shape "
            f"{tuple(workspace_indices.shape)} on {workspace_indices.device}, "
            f"got {tuple(out.shape)} {out.dtype} on {out.device}"
        )
    if workspace_indices.numel() == 0:
        return out
    if not workspace_indices.is_cuda:
        raise RuntimeError("DSA workspace top-k slot conversion requires CUDA tensors.")

    workspace_indices = workspace_indices.contiguous()
    kv_workspace_slots = kv_workspace_slots.to(
        device=workspace_indices.device,
        dtype=torch.int64,
    ).contiguous()
    out_view = out.contiguous() if not out.is_contiguous() else out
    total = int(workspace_indices.numel())
    block = 256
    _workspace_topk_to_global_slots_kernel[(triton.cdiv(total, block),)](
        out_view,
        workspace_indices,
        kv_workspace_slots,
        total=total,
        BLOCK=block,
        U16_INDICES=workspace_indices.dtype == torch.int16,
        num_warps=4,
        num_stages=1,
    )
    if out_view.data_ptr() != out.data_ptr():
        out.copy_(out_view)
    return out


@triton.jit
def _quantize_index_query_heads(
    q_ptr,
    weights_ptr,
    q_out_ptr,
    weights_out_ptr,
    row,  # int64 token row
    head,  # [HEADS_PER_PROGRAM] head indices
    softmax_scale,
    fp8_max,
    amax_floor,
    q_row_stride,
    q_head_stride,
    weights_row_stride,
    heads,
    DIM: tl.constexpr,
):
    head_mask = head < heads
    cols = tl.arange(0, DIM)
    y = tl.load(
        q_ptr + row * q_row_stride + head[:, None] * q_head_stride + cols[None, :],
        mask=head_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    # TRT-LLM's 1x128 token-group quantizer: amax floored at 1e-10 in the input dtype, IEEE divisions.
    amax = tl.maximum(tl.max(tl.abs(y), axis=1), amax_floor)
    y_s_inv = tl.div_rn(fp8_max, amax)
    y_s = tl.div_rn(1.0, y_s_inv)
    y_q = tl.clamp(y * y_s_inv[:, None], -fp8_max, fp8_max)
    tl.store(
        q_out_ptr + (row * heads + head[:, None]) * DIM + cols[None, :],
        y_q.to(q_out_ptr.dtype.element_ty),
        mask=head_mask[:, None],
    )
    w = tl.load(weights_ptr + row * weights_row_stride + head, mask=head_mask)
    tl.store(
        weights_out_ptr + row * heads + head,
        w.to(tl.float32) * y_s * softmax_scale,
        mask=head_mask,
    )


@triton.jit
def _quantize_index_query_weights_kernel(
    q_ptr,
    weights_ptr,
    q_out_ptr,
    weights_out_ptr,
    softmax_scale,
    fp8_max,
    amax_floor,
    q_row_stride,
    q_head_stride,
    weights_row_stride,
    heads,
    DIM: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
):
    _quantize_index_query_heads(
        q_ptr,
        weights_ptr,
        q_out_ptr,
        weights_out_ptr,
        tl.program_id(0).to(tl.int64),
        tl.program_id(1) * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM),
        softmax_scale,
        fp8_max,
        amax_floor,
        q_row_stride,
        q_head_stride,
        weights_row_stride,
        heads,
        DIM,
    )


def _quantize_index_query_weights_args(
    q: torch.Tensor, weights: torch.Tensor, softmax_scale: float
) -> tuple[tuple, torch.Tensor, torch.Tensor]:
    """Validate :func:`quantize_index_query_weights`' inputs (same arguments).

    Returns ``(args, q_fp8, folded)``: the runtime arguments
    ``_quantize_index_query_weights_kernel`` takes and its two outputs.
    """
    tokens, heads, dim = q.shape
    if dim != 128 or q.stride(-1) != 1 or weights.stride(-1) != 1:
        raise ValueError("expects [tokens, heads, 128] queries and unit-stride rows")
    if weights.shape != (tokens, heads):
        raise ValueError(f"weights must have shape {(tokens, heads)}")
    q_fp8 = torch.empty(
        (tokens, heads, dim), dtype=torch.float8_e4m3fn, device=q.device
    )
    folded = torch.empty((tokens, heads), dtype=torch.float32, device=q.device)
    args = (
        q,
        weights,
        q_fp8,
        folded,
        float(softmax_scale),
        float(torch.finfo(torch.float8_e4m3fn).max),
        float(torch.tensor(1e-10, dtype=q.dtype)),
        q.stride(0),
        q.stride(1),
        weights.stride(0),
        heads,
    )
    return args, q_fp8, folded


def quantize_index_query_weights(
    q: torch.Tensor, weights: torch.Tensor, softmax_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize index queries to E4M3 with one scale per (token, head) and fold
    that scale and ``softmax_scale`` into the head weights, in one launch.

    Bit-exact with NVIDIA's 1x128 token-group ``quantize_fp8`` followed by
    ``combine_topk_weights``, which it replaces on the NVIDIA scoring paths.

    Args:
        q: ``[tokens, heads, 128]`` BF16 index queries, unit stride on the last dim.
        weights: ``[tokens, heads]`` head weights, unit stride on the last dim.
        softmax_scale: Constant score scale folded into the weights.

    Returns:
        ``[tokens, heads, 128]`` contiguous E4M3 queries and ``[tokens, heads]``
        contiguous FP32 combined weights.
    """
    args, q_fp8, folded = _quantize_index_query_weights_args(q, weights, softmax_scale)
    tokens, heads = q.shape[:2]
    if tokens:
        # One head per program while the grid is small, eight once rows fill the GPU.
        per_program = 1 if tokens * heads <= 4096 else 8
        _quantize_index_query_weights_kernel[(tokens, triton.cdiv(heads, per_program))](
            *args,
            DIM=q.shape[2],
            HEADS_PER_PROGRAM=per_program,
            num_warps=1 if per_program == 1 else 4,
            num_stages=1,
        )
    return q_fp8, folded


@triton.jit
def _index_k_write_quantize_query_kernel(
    fp8_buf_ptr,
    scale_buf_ptr,
    x_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    loc_ptr,
    write_mask_ptr,
    x_row_stride,
    page_bytes,
    scale_base_off,
    eps,
    q_ptr,
    weights_ptr,
    q_out_ptr,
    weights_out_ptr,
    softmax_scale,
    fp8_max,
    amax_floor,
    q_row_stride,
    q_head_stride,
    weights_row_stride,
    heads,
    HAS_WRITE_MASK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HD: tl.constexpr,
    GROUP: tl.constexpr,
    NG: tl.constexpr,
    BLOCK_NG: tl.constexpr,
    FP8_MAX: tl.constexpr,
    DIM: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    _quantize_index_query_heads(
        q_ptr,
        weights_ptr,
        q_out_ptr,
        weights_out_ptr,
        row,
        block * HEADS_PER_PROGRAM + tl.arange(0, HEADS_PER_PROGRAM),
        softmax_scale,
        fp8_max,
        amax_floor,
        q_row_stride,
        q_head_stride,
        weights_row_stride,
        heads,
        DIM,
    )
    # The first head block of a row also writes its key.
    if block == 0:
        _index_k_layernorm_fp8_scatter_row(
            fp8_buf_ptr,
            scale_buf_ptr,
            x_ptr,
            norm_weight_ptr,
            norm_bias_ptr,
            loc_ptr,
            write_mask_ptr,
            row,
            x_row_stride,
            page_bytes,
            scale_base_off,
            eps,
            HAS_WRITE_MASK,
            PAGE_SIZE,
            HD,
            GROUP,
            NG,
            BLOCK_NG,
            FP8_MAX,
        )


def index_k_layernorm_fp8_scatter_and_quantize_query(
    buf: torch.Tensor,
    index_k: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    loc: torch.Tensor,
    q: torch.Tensor,
    head_weights: torch.Tensor,
    *,
    eps: float,
    page_size: int,
    group_size: int,
    write_mask: torch.Tensor | None,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write index keys and quantize the same rows' index queries in one launch.

    The plane's bytes are exactly those of ``index_k_layernorm_fp8_scatter(buf,
    index_k, norm_weight, norm_bias, loc, ...)`` and the outputs exactly
    ``quantize_index_query_weights(q, head_weights, softmax_scale)``: each
    program quantizes one block of a row's query heads with that kernel's row
    function, and a row's first block also writes its key with the key
    kernel's, the LayerNorm in one warp as there. CUDA-graph safe; no
    per-batch constexpr.

    Args:
        buf: uint8 ``fp8_scaled`` index-K plane (a slot matrix or a page
            view), as for ``index_k_layernorm_fp8_scatter``.
        index_k: ``[tokens, head_dim]`` pre-norm keys with unit inner stride.
        norm_weight: ``[head_dim]`` LayerNorm weight.
        norm_bias: ``[head_dim]`` LayerNorm bias.
        loc: ``[tokens]`` non-negative integer slot indices.
        q: ``[tokens, heads, 128]`` BF16 index queries, unit last stride.
        head_weights: ``[tokens, heads]`` head weights, unit last stride.
        eps: Positive LayerNorm epsilon.
        page_size: Rows per page of the plane.
        group_size: Key elements per scale; divides ``head_dim``.
        write_mask: Optional bool ``[tokens]`` ownership mask; False rows'
            keys are not written. None writes every row.
        softmax_scale: Score scale folded into the head weights.

    Returns:
        ``[tokens, heads, 128]`` contiguous E4M3 queries and ``[tokens,
        heads]`` contiguous FP32 head weights carrying each query's scale and
        ``softmax_scale``; ``buf`` is written in place.
    """
    key_args, key_constexprs = _index_k_layernorm_fp8_scatter_args(
        buf,
        index_k,
        norm_weight,
        norm_bias,
        loc,
        eps=eps,
        page_size=page_size,
        group_size=group_size,
        write_mask=write_mask,
    )
    query_args, q_fp8, folded = _quantize_index_query_weights_args(
        q, head_weights, softmax_scale
    )
    tokens, heads = q.shape[:2]
    if index_k.shape[0] != tokens:
        raise ValueError(
            f"index_k has {index_k.shape[0]} rows and q has {tokens}; the "
            "fused write quantizes the queries of its own rows"
        )
    if tokens:
        enable_pdl = pdl_enabled()
        grid = (tokens, triton.cdiv(heads, _FUSED_QUERY_HEADS_PER_PROGRAM))
        _index_k_write_quantize_query_kernel[grid](
            *key_args,
            *query_args,
            **key_constexprs,
            DIM=q.shape[2],
            HEADS_PER_PROGRAM=_FUSED_QUERY_HEADS_PER_PROGRAM,
            ENABLE_PDL=enable_pdl,
            # One warp: the key's LayerNorm sums in the order of the scatter kernel's.
            num_warps=1,
            **({"launch_pdl": True} if enable_pdl else {}),
        )
    return q_fp8, folded


@triton.jit
def _combine_topk_weights_kernel(
    weights_ptr,
    q_scale_ptr,
    out_ptr,
    softmax_scale,
    weights_row_stride,
    heads,
    numel,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    t = offs // heads
    h = offs % heads
    w = tl.load(weights_ptr + t * weights_row_stride + h, mask=mask).to(tl.float32)
    q_scale = tl.load(q_scale_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, w * q_scale * softmax_scale, mask=mask)


def combine_topk_weights(
    weights: torch.Tensor,
    q_scale: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Build the combined per-(token, head) weights for DSA logits scoring.

    Single-launch, bit-exact equivalent of
    ``(weights.float().unsqueeze(-1) * q_scale * float(softmax_scale)).squeeze(-1)``
    (same fp32 operation order).

    Args:
        weights: ``[tokens, heads]`` bf16/fp16/fp32 indexer weights. The
            leading axis may have any stride (e.g. a column-``split`` view of
            the fused ``wk_weights_proj`` output); only ``stride(-1) == 1``
            is required.
        q_scale: fp32 per-``(token, head)`` dequant scales with
            ``tokens * heads`` elements in row-major order (any unit-trailing
            shape such as ``[tokens * heads, 1]`` or ``[tokens, heads, 1]``).
        softmax_scale: Constant score scale folded into the weights.

    Returns:
        ``[tokens, heads]`` contiguous fp32 combined weights.
    """
    tokens, heads = weights.shape
    if weights.stride(-1) != 1:
        raise ValueError("combine_topk_weights expects unit-stride weights rows")
    if q_scale.dtype != torch.float32:
        raise TypeError(f"q_scale must be fp32, got {q_scale.dtype}")
    # The quantizer may pad the per-row scale vector (an alignment multiple of
    # its launch width); real rows lead, padding trails. 16-head indexers hit
    # this on every odd token count.
    q_scale = q_scale.reshape(-1)
    if q_scale.numel() < tokens * heads:
        raise ValueError(
            f"q_scale has {q_scale.numel()} entries for {tokens * heads} rows"
        )
    q_scale = q_scale[: tokens * heads]
    out = torch.empty(tokens, heads, dtype=torch.float32, device=weights.device)
    numel = tokens * heads
    if numel == 0:
        return out
    block = 1024
    _combine_topk_weights_kernel[(triton.cdiv(numel, block),)](
        weights,
        q_scale,
        out,
        float(softmax_scale),
        weights.stride(0),
        heads,
        numel,
        BLOCK=block,
        num_warps=4,
        num_stages=1,
    )
    return out


@triton.jit(do_not_specialize=["block_table_stride", "logits_stride", "max_seq_len"])
def _dsa_decode_logits_fp8_kernel(
    q,
    index_k_fp8,
    index_k_scale,
    weights,
    seq_lens,
    block_table,
    logits,
    block_table_stride,
    logits_stride,
    query_requests,
    EXPLICIT_ROWS: tl.constexpr,
    INITIAL_TOKENS: tl.constexpr,
    LOCAL_TOKENS: tl.constexpr,
    page_size: tl.constexpr,
    row_bytes: tl.constexpr,
    page_stride_bytes: tl.constexpr,
    max_seq_len,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_groups: tl.constexpr,
    softmax_scale: tl.constexpr,
    q_len_per_req: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    if EXPLICIT_ROWS:
        # seq_lens is per token here; indexing it by req would read past the
        # buffer whenever a request ID exceeds the query-tile length.
        req = tl.load(query_requests + token)
        seq_len = tl.load(seq_lens + token)
    else:
        req = token // q_len_per_req
        base = tl.load(seq_lens + req).to(tl.int32)
        seq_len = base - (q_len_per_req - 1) + (token % q_len_per_req)
    seq_len = tl.maximum(seq_len, 0)
    valid = offsets < seq_len
    block_idx = offsets // page_size
    block_offset = offsets - block_idx * page_size
    valid = valid & (offsets < max_seq_len)
    if page_size % BLOCK_N == 0:
        # An aligned tile fits within one page, so its ownership is uniform.
        start = block_id * BLOCK_N
        page = tl.load(
            block_table + req * block_table_stride + start // page_size,
            mask=(start < seq_len) & (start < max_seq_len),
            other=-1,
        ).to(tl.int64)
        has_valid = page >= 0
    else:
        page = tl.load(
            block_table + req * block_table_stride + block_idx,
            mask=valid,
            other=-1,
        ).to(tl.int64)
        has_valid = tl.max((valid & (page >= 0)).to(tl.int32), axis=0) > 0
    valid = valid & (page >= 0)
    # Foreign pages and capacity beyond the causal length can leave a whole
    # tile empty. Skip Q/K scoring, but still overwrite its logits for Top-K.
    scores = tl.full((BLOCK_N,), -float("inf"), tl.float32)
    if has_valid:
        fp8_base = page * page_stride_bytes + block_offset * head_dim
        scale_base = (
            page * (page_stride_bytes // 4)
            + (page_size * head_dim) // 4
            + block_offset * num_groups
        )
        scores = tl.zeros((BLOCK_N,), tl.float32)

        dim_offsets = tl.arange(0, BLOCK_D)
        for head in tl.static_range(0, num_heads):
            head_weight = tl.load(weights + token * num_heads + head).to(tl.float32)
            head_score = tl.zeros((BLOCK_N,), tl.float32)
            for dim_start in tl.static_range(0, head_dim, BLOCK_D):
                dims = dim_start + dim_offsets
                q_vals = tl.load(
                    q + (token * num_heads + head) * head_dim + dims,
                    mask=dims < head_dim,
                    other=0.0,
                ).to(tl.float32)
                k_vals = tl.load(
                    index_k_fp8 + fp8_base[:, None] + dims[None, :],
                    mask=valid[:, None] & (dims[None, :] < head_dim),
                    other=0.0,
                ).to(tl.float32)
                k_scale = tl.load(
                    index_k_scale + scale_base + dim_start // 128,
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)
                head_score += tl.sum(
                    k_vals * k_scale[:, None] * q_vals[None, :], axis=1
                )
            scores += (
                tl.maximum(head_score, 0.0, propagate_nan=tl.PropagateNan.ALL)
                * head_weight
            )

        scores *= softmax_scale
        forced = (offsets < INITIAL_TOKENS) | (offsets >= seq_len - LOCAL_TOKENS)
        scores = tl.where(forced, float("inf"), scores)
        scores = tl.where(valid & (scores == scores), scores, -float("inf"))
    tl.store(
        logits + token * logits_stride + offsets,
        scores,
        mask=offsets < max_seq_len,
    )


@triton.jit
def _dsa_prefill_logits_fp8_kernel(
    q,
    index_k_fp8,
    index_k_scale,
    weights,
    kv_workspace_slots,
    row_starts,
    row_ends,
    logits,
    # Both derive from the batch's total KV length, which differs for every
    # request.  As ``tl.constexpr`` they would key the JIT cache, forcing a
    # fresh compile per prompt; they are only used for bounds masking and
    # address arithmetic, so a runtime value is equivalent.
    logits_stride,
    seq_len_sum,
    page_size: tl.constexpr,
    row_bytes: tl.constexpr,
    page_stride_bytes: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_groups: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
    row_start = tl.load(row_starts + token).to(tl.int32)
    row_end = tl.load(row_ends + token).to(tl.int32)
    valid = (offsets >= row_start) & (offsets < row_end) & (offsets < seq_len_sum)
    slots = tl.load(
        kv_workspace_slots + offsets,
        mask=offsets < seq_len_sum,
        other=0,
    ).to(tl.int64)
    page = slots // page_size
    block_offset = slots - page * page_size
    valid = valid & (page >= 0)
    fp8_base = page * page_stride_bytes + block_offset * head_dim
    scale_base = (
        page * (page_stride_bytes // 4)
        + (page_size * head_dim) // 4
        + block_offset * num_groups
    )
    scores = tl.zeros((BLOCK_N,), tl.float32)

    dim_offsets = tl.arange(0, BLOCK_D)
    for head in tl.static_range(0, num_heads):
        head_weight = tl.load(weights + token * num_heads + head).to(tl.float32)
        head_score = tl.zeros((BLOCK_N,), tl.float32)
        for dim_start in tl.static_range(0, head_dim, BLOCK_D):
            dims = dim_start + dim_offsets
            q_vals = tl.load(
                q + (token * num_heads + head) * head_dim + dims,
                mask=dims < head_dim,
                other=0.0,
            ).to(tl.float32)
            k_vals = tl.load(
                index_k_fp8 + fp8_base[:, None] + dims[None, :],
                mask=valid[:, None] & (dims[None, :] < head_dim),
                other=0.0,
            ).to(tl.float32)
            k_scale = tl.load(
                index_k_scale + scale_base + dim_start // 128,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            head_score += tl.sum(k_vals * k_scale[:, None] * q_vals[None, :], axis=1)
        scores += (
            tl.maximum(head_score, 0.0, propagate_nan=tl.PropagateNan.ALL) * head_weight
        )

    scores *= softmax_scale
    scores = tl.where(valid, scores, -float("inf"))
    tl.store(
        logits + token * logits_stride + offsets,
        scores,
        mask=offsets < seq_len_sum,
    )


def _check_q_weights(q: torch.Tensor, weights: torch.Tensor) -> None:
    if q.dtype != torch.bfloat16:
        raise TypeError(f"DSA Triton top-k expects BF16 q, got {q.dtype}")
    if weights.dtype != torch.float32:
        raise TypeError(f"DSA Triton top-k expects FP32 weights, got {weights.dtype}")
    if q.dim() != 3:
        raise ValueError(f"q must be [tokens, heads, dim], got {tuple(q.shape)}")
    if weights.shape != q.shape[:2]:
        raise ValueError(
            "weights must be [tokens, heads] matching q, got "
            f"weights={tuple(weights.shape)}, q={tuple(q.shape)}"
        )
    if q.shape[2] % 64 != 0:
        raise ValueError(
            f"DSA Triton top-k requires dim multiple of 64, got {q.shape[2]}"
        )


def _check_packed_fp8_inputs(
    q: torch.Tensor,
    index_k_cache: torch.Tensor,
    weights: torch.Tensor,
    page_size: int,
) -> tuple[int, int]:
    _check_q_weights(q, weights)
    if q.shape[2] % 128 != 0:
        raise ValueError(
            f"DSA Triton FP8 top-k requires dim multiple of 128, got {q.shape[2]}"
        )
    if index_k_cache.dtype != torch.uint8:
        raise TypeError(
            "DSA Triton FP8 top-k expects uint8 packed index_k_cache, got "
            f"{index_k_cache.dtype}"
        )
    row_bytes = q.shape[2] + q.shape[2] // 128 * 4
    if index_k_cache.dim() != 2:
        raise ValueError(
            "index_k_cache must be a packed slot matrix or page-planar matrix, "
            f"got shape {tuple(index_k_cache.shape)}"
        )
    page_bytes = int(page_size) * row_bytes
    if index_k_cache.shape[1] == row_bytes:
        if not index_k_cache.is_contiguous():
            raise ValueError("packed index_k_cache must be contiguous")
        if index_k_cache.shape[0] % int(page_size) != 0:
            raise ValueError(
                "index_k_cache slot count must be divisible by page_size, got "
                f"slots={index_k_cache.shape[0]}, page_size={page_size}"
            )
        page_stride_bytes = page_bytes
    elif (
        index_k_cache.shape[1] >= page_bytes
        and index_k_cache.stride(1) == 1
        and index_k_cache.stride(0) >= page_bytes
    ):
        page_stride_bytes = index_k_cache.stride(0)
    else:
        raise ValueError(
            "index_k_cache must be contiguous [slots, row_bytes] or page-planar "
            f"[pages, at least {page_bytes} bytes], got "
            f"shape={tuple(index_k_cache.shape)}, stride={index_k_cache.stride()}"
        )
    if index_k_cache.storage_offset() % 4 or page_stride_bytes % 4:
        raise ValueError("index_k_cache page storage must be float32 aligned")
    return row_bytes, page_stride_bytes


@triton.jit
def _fp32_to_ordered_key(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign = bits & 0x80000000
    return bits ^ tl.where(sign != 0, 0xFFFFFFFF, 0x80000000)


@triton.jit
def _ordered_key_to_fp32(x):
    sign = x & 0x80000000
    bits = x ^ tl.where(sign != 0, 0x80000000, 0xFFFFFFFF)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _topk_row_length(seq_lens, row, seq_lens_stride, n_cols):
    length = n_cols
    if seq_lens is not None:
        length = tl.minimum(
            tl.maximum(tl.load(seq_lens + row * seq_lens_stride), 0), n_cols
        ).to(tl.int32)
    return length


@triton.jit(
    do_not_specialize=[
        "logits_stride",
        "logits_col_stride",
        "out_stride",
        "seq_lens_stride",
        "n_cols",
    ]
)
def _dsa_logits_topk_kernel(
    logits,
    out,
    seq_lens,
    # Runtime values: these track the batch's KV length, so keying the JIT
    # cache on them recompiles per request.  ``n_cols_padded`` stays a
    # constexpr -- it is a power of two that drives the unroll below.
    logits_stride,
    logits_col_stride,
    out_stride,
    seq_lens_stride,
    n_cols,
    n_cols_padded: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0)
    row_cols = _topk_row_length(seq_lens, row, seq_lens_stride, n_cols)
    # Valid columns have a positive index key, so zero is an invalid sentinel.
    acc = tl.zeros((1, topk), dtype=tl.uint64)
    for tile in tl.static_range(n_cols_padded // BLOCK_N - 1, -1, -1):
        if tile * BLOCK_N < row_cols:
            offsets = tile * BLOCK_N + tl.arange(0, BLOCK_N)
            values = tl.load(
                logits + row * logits_stride + offsets * logits_col_stride,
                mask=offsets < row_cols,
                other=-float("inf"),
            )
            value_keys = _fp32_to_ordered_key(values).to(tl.uint64)
            index_keys = (n_cols_padded - offsets).to(tl.uint64)
            packed = (value_keys << 32) | index_keys
            acc = tl.bitonic_merge(acc)
            acc = tl.maximum(acc, tl.topk(packed[None, :], topk, dim=1))

    acc = tl.sort(acc, dim=1, descending=True)
    top_offsets = tl.arange(0, topk)
    packed_top = tl.reshape(acc, (topk,))
    indices = n_cols_padded - (packed_top & 0xFFFFFFFF).to(tl.int32)
    values = _ordered_key_to_fp32((packed_top >> 32).to(tl.uint32))
    valid_top = (top_offsets < row_cols) & (indices >= 0) & (indices < row_cols)
    valid_top = valid_top & (values != -float("inf"))
    tl.store(
        out + row * out_stride + top_offsets,
        tl.where(valid_top, indices, -1),
        mask=top_offsets < topk,
    )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(
    do_not_specialize=[
        "logits_stride",
        "logits_col_stride",
        "seq_lens_stride",
        "hist_tiles",
        "n_cols",
    ]
)
def _dsa_radix_hist_kernel(
    logits,
    prefixes,
    hist,
    seq_lens,
    logits_stride,
    logits_col_stride,
    seq_lens_stride,
    hist_tiles,
    n_cols,
    shift: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0)
    tile = tl.program_id(1)
    row_cols = _topk_row_length(seq_lens, row, seq_lens_stride, n_cols)
    if tile * BLOCK_N < row_cols:
        offsets = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < row_cols
        values = tl.load(
            logits + row * logits_stride + offsets * logits_col_stride,
            mask=mask,
            other=-float("inf"),
        )
        keys = _fp32_to_ordered_key(values)
        if shift == 28:
            prefix_match = mask
        else:
            prefix = tl.load(prefixes + row).to(tl.uint32)
            prefix_match = (keys >> (shift + 4)) == prefix
        bucket = (keys >> shift) & 0xF
        base = (row * hist_tiles + tile) * 16
        for b in tl.static_range(0, 16):
            count = tl.sum(tl.where(mask & prefix_match & (bucket == b), 1, 0))
            tl.store(hist + base + b, count)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(do_not_specialize=["seq_lens_stride", "hist_tiles", "n_cols"])
def _dsa_radix_update_kernel(
    prefixes,
    remaining,
    hist,
    seq_lens,
    seq_lens_stride,
    hist_tiles,
    n_cols,
    topk: tl.constexpr,
    INITIAL_PASS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0)
    row_cols = _topk_row_length(seq_lens, row, seq_lens_stride, n_cols)
    tile_offsets = tl.arange(0, BLOCK_TILES)
    # Histogram tiles outside this row's frontier were never written.
    tile_mask = tile_offsets < tl.cdiv(row_cols, BLOCK_N)
    row_hist = hist + row * hist_tiles * 16
    if INITIAL_PASS:
        kth = tl.minimum(topk, row_cols)
        prefix = tl.full((), 0, tl.uint32)
    else:
        kth = tl.load(remaining + row).to(tl.int32)
        prefix = tl.load(prefixes + row).to(tl.uint32)
    cumulative = tl.full((), 0, dtype=tl.int32)
    selected = tl.full((), 0, dtype=tl.uint32)
    selected_remaining = kth
    found = False
    for b_desc in tl.static_range(0, 16):
        b = 15 - b_desc
        counts = tl.load(row_hist + tile_offsets * 16 + b, mask=tile_mask, other=0)
        count = tl.sum(counts).to(tl.int32)
        take = (found == False) & (kth <= cumulative + count)
        selected = tl.where(take, b, selected)
        selected_remaining = tl.where(take, kth - cumulative, selected_remaining)
        cumulative += tl.where(found == False, count, 0)
        found = found | take
    tl.store(prefixes + row, (prefix << 4) | selected)
    tl.store(remaining + row, selected_remaining)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(
    do_not_specialize=[
        "logits_stride",
        "logits_col_stride",
        "out_stride",
        "seq_lens_stride",
        "n_cols",
    ]
)
def _dsa_radix_scatter_kernel(
    logits,
    prefixes,
    remaining,
    out_values,
    out_indices,
    seq_lens,
    logits_stride,
    logits_col_stride,
    out_stride,
    seq_lens_stride,
    n_cols,
    topk: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0)
    row_cols = _topk_row_length(seq_lens, row, seq_lens_stride, n_cols)
    top_offsets = tl.arange(0, topk)
    tl.store(out_values + row * out_stride + top_offsets, -float("inf"))
    tl.store(out_indices + row * out_stride + top_offsets, -1)

    threshold = tl.load(prefixes + row).to(tl.uint32)
    keep_equal = tl.load(remaining + row).to(tl.int32)
    count_greater = tl.minimum(topk, row_cols) - keep_equal
    num_greater = tl.full((), 0, dtype=tl.int32)
    num_equal = tl.full((), 0, dtype=tl.int32)
    for start in tl.range(0, row_cols, BLOCK_N):
        offsets = start + tl.arange(0, BLOCK_N)
        mask = offsets < row_cols
        values = tl.load(
            logits + row * logits_stride + offsets * logits_col_stride,
            mask=mask,
            other=-float("inf"),
        )
        finite = values != -float("inf")
        keys = _fp32_to_ordered_key(values)
        greater = mask & finite & (keys > threshold)
        equal = mask & finite & (keys == threshold)

        greater_pos = num_greater + tl.cumsum(greater.to(tl.int32), 0) - 1
        greater_mask = greater & (greater_pos < topk)
        tl.store(
            out_values + row * out_stride + greater_pos,
            values,
            mask=greater_mask,
        )
        tl.store(
            out_indices + row * out_stride + greater_pos,
            offsets,
            mask=greater_mask,
        )
        num_greater += tl.sum(greater.to(tl.int32))

        equal_pos = count_greater + num_equal + tl.cumsum(equal.to(tl.int32), 0) - 1
        equal_mask = (
            equal & (equal_pos < topk) & (equal_pos < count_greater + keep_equal)
        )
        tl.store(
            out_values + row * out_stride + equal_pos,
            values,
            mask=equal_mask,
        )
        tl.store(
            out_indices + row * out_stride + equal_pos,
            offsets,
            mask=equal_mask,
        )
        num_equal += tl.sum(equal.to(tl.int32))
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit(do_not_specialize=["stride"])
def _dsa_radix_sort_selected_kernel(
    values,
    indices,
    out,
    stride,
    n_cols_padded: tl.constexpr,
    topk: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0)
    offsets = tl.arange(0, topk)
    vals = tl.load(values + row * stride + offsets)
    idx = tl.load(indices + row * stride + offsets).to(tl.int32)
    valid = idx >= 0
    value_keys = _fp32_to_ordered_key(vals).to(tl.uint64)
    index_keys = (n_cols_padded - idx).to(tl.uint64)
    packed = (value_keys << 32) | index_keys
    packed = tl.where(valid, packed, tl.zeros_like(packed))
    packed = tl.sort(packed[None, :], dim=1, descending=True)
    packed = tl.reshape(packed, (topk,))
    sorted_idx = n_cols_padded - (packed & 0xFFFFFFFF).to(tl.int32)
    sorted_vals = _ordered_key_to_fp32((packed >> 32).to(tl.uint32))
    valid = (
        (sorted_idx >= 0)
        & (sorted_idx < n_cols_padded)
        & (sorted_vals != -float("inf"))
    )
    tl.store(out + row * stride + offsets, tl.where(valid, sorted_idx, -1))
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def _is_power_of_2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


def _radix_topk(
    logits: torch.Tensor,
    topk: int,
    *,
    seq_lens: torch.Tensor | None,
    enable_pdl: bool,
) -> torch.Tensor:
    rows, cols = logits.shape
    block_n = _RADIX_TOPK_BLOCK_N
    tiles = triton.cdiv(cols, block_n)
    hist = torch.empty((rows, tiles, 16), dtype=torch.int32, device=logits.device)
    # The first radix pass initializes both arrays from each row's valid length.
    prefixes = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    remaining = torch.empty((rows,), dtype=torch.int32, device=logits.device)
    out_values = torch.empty((rows, topk), dtype=torch.float32, device=logits.device)
    out_indices = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    out = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    block_tiles = _next_power_of_2(tiles)
    seq_lens_stride = 0 if seq_lens is None else seq_lens.stride(0)
    use_pdl = bool(enable_pdl and _is_nvidia)
    # Keep stage releases late to limit resource use by waiting successor CTAs.
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}

    for shift in range(28, -1, -4):
        _dsa_radix_hist_kernel[(rows, tiles)](
            logits,
            prefixes,
            hist,
            seq_lens,
            logits.stride(0),
            logits.stride(1),
            seq_lens_stride,
            tiles,
            n_cols=cols,
            shift=shift,
            BLOCK_N=block_n,
            ENABLE_PDL=use_pdl,
            num_warps=8,
            num_stages=1,
            **pdl_kwargs,
        )
        _dsa_radix_update_kernel[(rows,)](
            prefixes,
            remaining,
            hist,
            seq_lens,
            seq_lens_stride,
            hist_tiles=tiles,
            n_cols=cols,
            topk=int(topk),
            INITIAL_PASS=shift == 28,
            BLOCK_N=block_n,
            BLOCK_TILES=block_tiles,
            ENABLE_PDL=use_pdl,
            num_warps=8,
            num_stages=1,
            **pdl_kwargs,
        )

    _dsa_radix_scatter_kernel[(rows,)](
        logits,
        prefixes,
        remaining,
        out_values,
        out_indices,
        seq_lens,
        logits.stride(0),
        logits.stride(1),
        out_indices.stride(0),
        seq_lens_stride,
        n_cols=cols,
        topk=int(topk),
        BLOCK_N=block_n,
        ENABLE_PDL=use_pdl,
        num_warps=8,
        num_stages=1,
        **pdl_kwargs,
    )
    _dsa_radix_sort_selected_kernel[(rows,)](
        out_values,
        out_indices,
        out,
        out.stride(0),
        n_cols_padded=_next_power_of_2(max(cols, int(topk))),
        topk=int(topk),
        ENABLE_PDL=use_pdl,
        num_warps=8,
        num_stages=1,
        **pdl_kwargs,
    )
    return out


def _topk_with_padding(
    logits: torch.Tensor,
    topk: int,
    *,
    seq_lens: torch.Tensor | None,
    enable_pdl: bool,
) -> torch.Tensor:
    topk = int(topk)
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if not _is_power_of_2(topk):
        raise ValueError(f"DSA Triton top-k requires power-of-two topk, got {topk}")
    if logits.dim() != 2:
        raise ValueError(f"logits must be [rows, cols], got {tuple(logits.shape)}")
    if logits.dtype != torch.float32:
        raise TypeError(f"logits must be FP32, got {logits.dtype}")

    rows, cols = logits.shape
    if seq_lens is not None:
        if seq_lens.shape != (rows,):
            raise ValueError(f"seq_lens must be [{rows}], got {tuple(seq_lens.shape)}")
        if seq_lens.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"seq_lens must be int32 or int64, got {seq_lens.dtype}")
        if seq_lens.device != logits.device:
            raise ValueError("seq_lens and logits must be on the same device")
    if rows == 0 or cols == 0:
        return torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
    n_cols_padded = _next_power_of_2(max(cols, topk))
    # Unrolled sort cost follows the padded width, including speculative KV padding.
    if n_cols_padded >= _RADIX_TOPK_MIN_COLS:
        return _radix_topk(logits, topk, seq_lens=seq_lens, enable_pdl=enable_pdl)
    out = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    block_n = min(n_cols_padded, 2048)
    block_n = max(block_n, topk)
    if n_cols_padded % block_n != 0:
        raise ValueError(
            "DSA Triton top-k requires padded cols divisible by block size, got "
            f"cols={cols}, padded={n_cols_padded}, block={block_n}"
        )
    use_pdl = bool(enable_pdl and _is_nvidia)
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    _dsa_logits_topk_kernel[(rows,)](
        logits,
        out,
        seq_lens,
        logits.stride(0),
        logits.stride(1),
        out.stride(0),
        0 if seq_lens is None else seq_lens.stride(0),
        n_cols=cols,
        n_cols_padded=n_cols_padded,
        topk=topk,
        BLOCK_N=block_n,
        ENABLE_PDL=use_pdl,
        num_warps=8,
        num_stages=1,
        **pdl_kwargs,
    )
    return out


def triton_topk_from_logits(
    logits: torch.Tensor,
    topk: int,
    *,
    seq_lens: torch.Tensor | None,
    enable_pdl: bool,
) -> torch.Tensor:
    """Select per-row top-k column indices from a materialized logits matrix.

    This is the selection backend behind the DSA prefill/decode top-k entry
    points, exposed for other indexers that materialize their own scores
    (e.g. the Qwen4-Exp QSA block top-k ``logits`` path). Column counts
    at or above the radix threshold switch to the radix selection pipeline
    automatically.

    Args:
        logits: FP32 scores shaped ``[rows, cols]``. Invalid entries within
            each row's valid prefix must be ``-inf``; entries after that
            prefix are never read and may be uninitialized.
        topk: Entries selected per row; must be a power of two.
        seq_lens: Int32 or int64 valid prefix lengths shaped ``[rows]`` on
            the logits device, clamped to ``[0, cols]`` in-kernel. Pass None
            to select over the full width. Strided scores and lengths are
            read directly without materializing contiguous copies.
        enable_pdl: Overlap the selection launch with the producer kernel
            on NVIDIA GPUs via programmatic dependent launch.

    Returns:
        Int32 column indices shaped ``[rows, topk]``; invalid entries are
        ``-1``.
    """

    return _topk_with_padding(logits, topk, seq_lens=seq_lens, enable_pdl=enable_pdl)


def dsa_decode_topk_fp8(
    q: torch.Tensor,
    index_k_cache: torch.Tensor,
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
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_bytes, page_stride_bytes = _check_packed_fp8_inputs(
        q, index_k_cache, weights, page_size
    )
    q_len_per_req = int(q_len_per_req)
    if q_len_per_req < 1 or q.shape[0] % q_len_per_req != 0:
        raise ValueError(
            f"q_len_per_req={q_len_per_req} must divide tokens={q.shape[0]}"
        )
    num_reqs = q.shape[0] // q_len_per_req
    if seq_lens.dim() != 1 or seq_lens.numel() != num_reqs:
        raise ValueError(
            "seq_lens must be [num_reqs], got "
            f"{tuple(seq_lens.shape)} for q={tuple(q.shape)}, "
            f"q_len_per_req={q_len_per_req}"
        )
    if block_table.dim() != 2 or block_table.shape[0] < num_reqs:
        raise ValueError(
            "block_table must have at least one row per request, got "
            f"block_table={tuple(block_table.shape)}, num_reqs={num_reqs}"
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if q.shape[0] == 0:
        return (
            (
                torch.empty((0, int(topk)), dtype=torch.int32, device=q.device)
                if out is None
                else out
            ),
            (
                torch.empty((0,), dtype=torch.int32, device=q.device)
                if lens_out is None
                else lens_out
            ),
        )
    if not q.is_cuda:
        raise RuntimeError("DSA Triton FP8 decode top-k requires CUDA tensors")

    q = q.contiguous()
    weights = weights.contiguous()
    seq_lens = seq_lens.to(device=q.device, dtype=torch.int32).contiguous()
    block_table = block_table.to(device=q.device, dtype=torch.int32).contiguous()
    max_seq_len = int(block_table.shape[1]) * int(page_size)
    logits = torch.empty(
        (q.shape[0], max_seq_len), dtype=torch.float32, device=q.device
    )
    block_n = 64
    grid = (q.shape[0], triton.cdiv(max_seq_len, block_n))
    _dsa_decode_logits_fp8_kernel[grid](
        q,
        index_k_cache.view(torch.float8_e4m3fn),
        index_k_cache.view(torch.float32),
        weights,
        seq_lens,
        block_table,
        logits,
        block_table.stride(0),
        logits.stride(0),
        None,
        EXPLICIT_ROWS=False,
        INITIAL_TOKENS=0,
        LOCAL_TOKENS=0,
        page_size=int(page_size),
        row_bytes=row_bytes,
        page_stride_bytes=page_stride_bytes,
        max_seq_len=max_seq_len,
        num_heads=q.shape[1],
        head_dim=q.shape[2],
        num_groups=q.shape[2] // 128,
        softmax_scale=float(softmax_scale),
        q_len_per_req=q_len_per_req,
        BLOCK_N=block_n,
        BLOCK_D=64,
        num_warps=4,
        num_stages=1,
    )
    local_topk_offsets = _topk_with_padding(
        logits, int(topk), seq_lens=None, enable_pdl=False
    )
    return local_topk_to_global_slots(
        local_topk_offsets=local_topk_offsets,
        block_table=block_table,
        block_size=int(page_size),
        seq_lens=seq_lens,
        q_len_per_req=q_len_per_req,
        topk_layout=topk_layout,
        block_table_base_offsets=block_table_base_offsets,
        out=out,
        lens_out=lens_out,
    )


def dsa_prefill_topk_fp8(
    q: torch.Tensor,
    index_k_cache: torch.Tensor,
    weights: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_bytes, page_stride_bytes = _check_packed_fp8_inputs(
        q, index_k_cache, weights, page_size
    )
    if kv_workspace_slots.dim() != 1:
        raise ValueError(
            f"kv_workspace_slots must be 1-D, got {tuple(kv_workspace_slots.shape)}"
        )
    if row_starts.shape != (q.shape[0],) or row_ends.shape != (q.shape[0],):
        raise ValueError(
            "row_starts/row_ends must be [tokens], got "
            f"row_starts={tuple(row_starts.shape)}, row_ends={tuple(row_ends.shape)}, "
            f"q={tuple(q.shape)}"
        )
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if out is None:
        out = torch.empty((q.shape[0], int(topk)), dtype=torch.int32, device=q.device)
    if lens_out is None:
        lens_out = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    out.fill_(-1)
    lens_out.zero_()
    if q.shape[0] == 0:
        return out, lens_out
    if not q.is_cuda:
        raise RuntimeError("DSA Triton FP8 prefill top-k requires CUDA tensors")

    q = q.contiguous()
    weights = weights.contiguous()
    kv_workspace_slots = kv_workspace_slots.to(
        device=q.device, dtype=torch.int64
    ).contiguous()
    row_starts = row_starts.to(device=q.device, dtype=torch.int32).contiguous()
    row_ends = row_ends.to(device=q.device, dtype=torch.int32).contiguous()
    seq_len_sum = int(kv_workspace_slots.numel())
    candidate_lens = (row_ends - row_starts).clamp_min(0)
    lens_out.copy_(
        torch.minimum(candidate_lens, torch.full_like(candidate_lens, int(topk)))
    )
    if seq_len_sum == 0:
        return out, lens_out

    if max_logits_bytes is None:
        max_query_rows = q.shape[0]
    else:
        max_query_rows = max(1, int(max_logits_bytes) // (max(seq_len_sum, 1) * 4))
    block_n = 64
    for start in range(0, q.shape[0], max_query_rows):
        end = min(start + max_query_rows, q.shape[0])
        logits = torch.empty(
            (end - start, seq_len_sum), dtype=torch.float32, device=q.device
        )
        grid = (end - start, triton.cdiv(seq_len_sum, block_n))
        _dsa_prefill_logits_fp8_kernel[grid](
            q[start:end],
            index_k_cache.view(torch.float8_e4m3fn),
            index_k_cache.view(torch.float32),
            weights[start:end],
            kv_workspace_slots,
            row_starts[start:end],
            row_ends[start:end],
            logits,
            logits.stride(0),
            seq_len_sum=seq_len_sum,
            page_size=int(page_size),
            row_bytes=row_bytes,
            page_stride_bytes=page_stride_bytes,
            num_heads=q.shape[1],
            head_dim=q.shape[2],
            num_groups=q.shape[2] // 128,
            softmax_scale=float(softmax_scale),
            BLOCK_N=block_n,
            BLOCK_D=64,
            num_warps=4,
            num_stages=1,
        )
        workspace_indices = _topk_with_padding(
            logits, int(topk), seq_lens=None, enable_pdl=False
        )
        valid = (workspace_indices >= row_starts[start:end, None]) & (
            workspace_indices < row_ends[start:end, None]
        )
        out[start:end].copy_(torch.where(valid, workspace_indices, -1))
    return out, lens_out


def _triton_dsa_plan_impl(
    *,
    page_size: int,
    seq_lens_2d: torch.Tensor,
    out: object | None = None,
) -> torch.Tensor:
    # plan is unused
    return object() if out is None else out


def _triton_dsa_decode_topk_fp8_impl(
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
    if index_k_cache is None:
        raise RuntimeError("Triton DSA paged top-k requires packed FP8 index_k_cache")
    if weights.dtype != torch.float32:
        weights = weights.float()
    return dsa_decode_topk_fp8(
        q,
        index_k_cache,
        weights,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=softmax_scale,
        q_len_per_req=q_len_per_req,
        topk_layout=topk_layout,
        block_table_base_offsets=block_table_base_offsets,
        out=out,
        lens_out=lens_out,
    )


def _triton_dsa_prefill_topk_fp8_impl(
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
    max_logits_bytes: int | None = None,
    out: torch.Tensor | None = None,
    lens_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if index_k_cache is None or page_size is None:
        raise RuntimeError(
            "Triton DSA top-k requires packed FP8 index_k_cache and page_size"
        )
    if weights.dtype != torch.float32:
        weights = weights.float()
    return dsa_prefill_topk_fp8(
        q,
        index_k_cache,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        page_size=page_size,
        topk=topk,
        softmax_scale=softmax_scale,
        max_logits_bytes=max_logits_bytes,
        out=out,
        lens_out=lens_out,
    )


__all__ = [
    "combine_topk_weights",
    "dsa_decode_topk_fp8",
    "dsa_prefill_topk_fp8",
    "index_k_layernorm_fp8_scatter_and_quantize_query",
    "triton_topk_from_logits",
    "local_topk_to_global_slots",
    "mark_forced_initial_local_logits",
    "quantize_index_query_weights",
    "scrub_nonfinite_logits",
    "workspace_topk_to_global_slots",
]
