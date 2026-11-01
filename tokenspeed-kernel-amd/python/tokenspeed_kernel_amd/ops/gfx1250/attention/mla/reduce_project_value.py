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

"""Split-MLA reduction with value projection and optional gate for gfx1250."""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon

_LANES = gl.constexpr(32)
_BLOCK_N = 32


@gluon.jit
def _mla_reduce_project_value_kernel(
    split_output_ptr,
    split_max_ptr,
    split_expsum_ptr,
    seq_len_ptr,
    weight_ptr,
    gate_ptr,
    output_ptr,
    LATENT: gl.constexpr,
    VALUE: gl.constexpr,
    HEADS: gl.constexpr,
    GATE_STRIDE_B: gl.constexpr,
    GATE_STRIDE_N: gl.constexpr,
    HAS_GATE: gl.constexpr,
    BATCHED: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    TILE_SIZE: gl.constexpr,
):
    """Reduce log2-space split partials into the BF16 value/gate epilogue."""

    pid = gl.program_id(0)
    blocks_per_head: gl.constexpr = VALUE // BLOCK_N
    batch_head = pid // blocks_per_head
    if BATCHED:
        batch = batch_head // HEADS
        head = batch_head % HEADS
    else:
        batch = 0
        head = batch_head
    pid_n = pid % blocks_per_head
    layout: gl.constexpr = gl.BlockedLayout(
        [(BLOCK_N + NUM_WARPS - 1) // NUM_WARPS, LATENT // _LANES],
        [1, _LANES],
        [NUM_WARPS, 1],
        [1, 0],
    )
    n_layout: gl.constexpr = gl.SliceLayout(1, layout)
    k_layout: gl.constexpr = gl.SliceLayout(0, layout)
    offs_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=n_layout)
    offs_k = gl.arange(0, LATENT, layout=k_layout)

    # Match the standalone decode reducer's split reduction layout and order.
    # This preserves its materialized BF16 latent boundary before projection.
    tpw_k: gl.constexpr = gl.constexpr(min(_LANES, LATENT))
    wpc_k: gl.constexpr = gl.constexpr(min(NUM_WARPS, LATENT // min(_LANES, LATENT)))
    spt_k: gl.constexpr = gl.constexpr(
        LATENT // (min(_LANES, LATENT) * min(NUM_WARPS, LATENT // min(_LANES, LATENT)))
    )
    reduce_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[NUM_KV_SPLITS, spt_k],
        threads_per_warp=[1, tpw_k],
        warps_per_cta=[1, wpc_k],
        order=[1, 0],
    )
    split_layout: gl.constexpr = gl.SliceLayout(1, reduce_layout)
    reduce_k_layout: gl.constexpr = gl.SliceLayout(0, reduce_layout)
    offs_split = gl.arange(0, NUM_KV_SPLITS, layout=split_layout)
    reduce_offs_k = gl.arange(0, LATENT, layout=reduce_k_layout)

    seq_len = gl.load(seq_len_ptr + batch)
    tiles_per_split = gl.cdiv(seq_len, NUM_KV_SPLITS * TILE_SIZE)
    active_splits = gl.cdiv(seq_len, tiles_per_split * TILE_SIZE)
    split_mask = offs_split < gl.full(
        [NUM_KV_SPLITS],
        active_splits,
        dtype=gl.int32,
        layout=split_layout,
    )
    split_offset = batch_head * NUM_KV_SPLITS + offs_split
    split_max = gl.load(
        split_max_ptr + split_offset,
        mask=split_mask,
        other=-float("inf"),
    ).to(gl.float32)
    overall_max = gl.max(split_max)
    split_expsum = gl.load(
        split_expsum_ptr + split_offset,
        mask=split_mask,
        other=0.0,
    ).to(gl.float32)
    split_scale = gl.exp2(split_max - overall_max)
    overall_expsum = gl.sum(split_expsum * split_scale)
    partial = gl.load(
        split_output_ptr + split_offset[:, None] * LATENT + reduce_offs_k[None, :],
        mask=split_mask[:, None],
        other=0.0,
    ).to(gl.float32)
    acc_sum = gl.sum(partial * split_scale[:, None], axis=0)

    # Preserve both materialized BF16 boundaries: the latent reducer output
    # and the following value projection.
    attention = gl.where(
        overall_expsum == 0.0,
        0.0,
        acc_sum / overall_expsum,
    ).to(gl.bfloat16)
    weight = gl.amd.cdna5.buffer_load(
        weight_ptr,
        (
            head * LATENT * VALUE
            + offs_k[None, :].to(gl.int64) * VALUE
            + offs_n[:, None].to(gl.int64)
        ).to(gl.int32),
    ).to(gl.float32)
    attention = gl.convert_layout(attention[None, :], layout)
    projected = gl.sum(weight * attention.to(gl.float32), axis=1)
    projected = projected.to(gl.bfloat16).to(gl.float32)
    output_offset = batch_head * VALUE + offs_n
    if HAS_GATE:
        gate = gl.load(
            gate_ptr + batch * GATE_STRIDE_B + (head * VALUE + offs_n) * GATE_STRIDE_N
        ).to(gl.float32)
        projected *= 1.0 / (1.0 + gl.exp(-gate))
    gl.store(
        output_ptr + output_offset,
        projected.to(output_ptr.dtype.element_ty),
    )


def gluon_mla_reduce_project_value_gfx1250(
    split_output: torch.Tensor,
    split_max: torch.Tensor,
    split_expsum: torch.Tensor,
    cache_seqlens: torch.Tensor,
    weight: torch.Tensor,
    *,
    gate: torch.Tensor | None = None,
    page_size: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Reduce split attention and emit projected, optionally gated BF16 values."""

    batch = split_output.shape[0]
    num_splits = split_output.shape[2]
    heads, latent, value = weight.shape
    expected = (
        (
            split_output,
            (batch, heads, num_splits, latent),
            torch.float32,
            "split output",
        ),
        (split_max, (batch, heads, num_splits), torch.float32, "split maxima"),
        (
            split_expsum,
            (batch, heads, num_splits),
            torch.float32,
            "split exponent sums",
        ),
        (cache_seqlens, (batch,), torch.int32, "sequence lengths"),
        (weight, (heads, latent, value), torch.bfloat16, "value weight"),
    )
    if page_size != 64:
        raise ValueError("MLA projected-value reducer requires page size 64")
    for tensor, shape, dtype, name in expected:
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or not tensor.is_cuda
            or not tensor.is_contiguous()
            or tensor.device != split_output.device
        ):
            raise ValueError(
                f"MLA projected-value reducer requires contiguous colocated {name} "
                f"{shape} {dtype}"
            )
    gate_shape = (batch, heads * value)
    if gate is not None and (
        tuple(gate.shape) != gate_shape
        or gate.dtype != torch.bfloat16
        or not gate.is_cuda
        or gate.device != split_output.device
        or gate.stride(1) != 1
    ):
        raise ValueError(
            "MLA projected-value reducer requires a colocated BF16 gate "
            f"{gate_shape} with contiguous inner dimension"
        )
    if (
        tuple(out.shape) != (batch, heads * value)
        or out.dtype != torch.bfloat16
        or not out.is_cuda
        or not out.is_contiguous()
        or out.device != split_output.device
    ):
        raise ValueError(
            "MLA projected-value reducer requires contiguous colocated out "
            f"{(batch, heads * value)} {torch.bfloat16}"
        )

    gate_tensor = split_output if gate is None else gate
    num_warps = 8 if batch == 1 else 4
    _mla_reduce_project_value_kernel[(batch * heads * value // _BLOCK_N,)](
        split_output,
        split_max,
        split_expsum,
        cache_seqlens,
        weight,
        gate_tensor,
        out,
        LATENT=latent,
        VALUE=value,
        HEADS=heads,
        GATE_STRIDE_B=0 if gate is None else gate.stride(0),
        GATE_STRIDE_N=0 if gate is None else gate.stride(1),
        HAS_GATE=gate is not None,
        BATCHED=batch > 1,
        BLOCK_N=_BLOCK_N,
        NUM_WARPS=num_warps,
        NUM_KV_SPLITS=num_splits,
        TILE_SIZE=page_size,
        num_warps=num_warps,
        num_stages=1,
        waves_per_eu=0,
    )
    return out


# Latent columns each query-block reduce program merges and projects.
_QUERY_BLOCK_D = 128
_QUERY_BLOCK_NUM_WARPS = 4


@gluon.jit
def _wait_stores():
    """Wait until this wave's global stores are acknowledged."""
    gl.inline_asm_elementwise(
        asm="s_wait_storecnt 0x0",
        constraints="=v,0",
        args=[gl.arange(0, 32, gl.BlockedLayout([1], [32], [gl.num_warps()], [0]))],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _mla_query_block_reduce_project_value_kernel(
    split_output_ptr,
    split_max_ptr,
    split_expsum_ptr,
    seq_len_ptr,
    weight_ptr,
    gate_ptr,
    output_ptr,
    partial_ptr,
    counter_ptr,
    LATENT: gl.constexpr,
    VALUE: gl.constexpr,
    HEADS: gl.constexpr,
    QUERIES: gl.constexpr,
    GATE_STRIDE_B: gl.constexpr,
    GATE_STRIDE_N: gl.constexpr,
    HAS_GATE: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    SPLIT_CHUNK: gl.constexpr,
    BLOCK_D: gl.constexpr,
    TILE_SIZE: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Merge BLOCK_D latent columns of one (token, head) row and project them.

    The merge matches ``_mla_decode_fwd_reduce_kernel``. Each program then
    multiplies its BF16 latent slice by the matching rows of the value
    weight and publishes the FP32 partial; the last of the row's
    ``LATENT // BLOCK_D`` programs adds the partials in a fixed order, so
    the result does not depend on arrival order, applies the gate and resets
    the row's counter for the next launch.
    """
    token = gl.program_id(0)
    head = gl.program_id(1)
    d_block = gl.program_id(2)
    # The weight slice depends on nothing else; issue it first.
    W_LAYOUT: gl.constexpr = gl.BlockedLayout(
        [BLOCK_D // (2 * num_warps), VALUE // 16], [2, 16], [num_warps, 1], [1, 0]
    )
    offs_wd = d_block * BLOCK_D + gl.arange(0, BLOCK_D, gl.SliceLayout(1, W_LAYOUT))
    offs_v = gl.arange(0, VALUE, gl.SliceLayout(0, W_LAYOUT))
    weight = gl.amd.cdna5.buffer_load(
        weight_ptr,
        (head * LATENT * VALUE + offs_wd[:, None] * VALUE + offs_v[None, :]).to(
            gl.int32
        ),
    )
    NUM_D: gl.constexpr = LATENT // BLOCK_D
    row = token * HEADS + head
    VEC: gl.constexpr = BLOCK_D // 32
    TILE_LAYOUT: gl.constexpr = gl.BlockedLayout(
        [1, VEC], [1, 32], [num_warps, 1], [1, 0]
    )
    CHUNK_LAYOUT: gl.constexpr = gl.SliceLayout(1, TILE_LAYOUT)
    D_LAYOUT: gl.constexpr = gl.SliceLayout(0, TILE_LAYOUT)
    STAT_LAYOUT: gl.constexpr = gl.BlockedLayout([1], [32], [num_warps], [0])

    seq_len = gl.load(seq_len_ptr + token // QUERIES)
    tiles_per_split = gl.cdiv(seq_len, NUM_KV_SPLITS * TILE_SIZE)
    active_splits = gl.cdiv(seq_len, tiles_per_split * TILE_SIZE)
    stat_base = row.to(gl.int64) * NUM_KV_SPLITS

    offs_all = gl.arange(0, NUM_KV_SPLITS, layout=STAT_LAYOUT)
    split_max = gl.load(
        split_max_ptr + stat_base + offs_all,
        mask=offs_all < active_splits,
        other=float("-inf"),
    )
    overall_max = gl.max(split_max, axis=0)
    # A causal query with no visible key has no finite split maximum.
    overall_max = gl.where(overall_max > float("-inf"), overall_max, 0.0)
    split_expsum = gl.load(
        split_expsum_ptr + stat_base + offs_all,
        mask=offs_all < active_splits,
        other=0.0,
    )
    overall_expsum = gl.sum(split_expsum * gl.exp2(split_max - overall_max), axis=0)

    offs_d = d_block * BLOCK_D + gl.arange(0, BLOCK_D, layout=D_LAYOUT)
    acc = gl.zeros([SPLIT_CHUNK, BLOCK_D], gl.float32, layout=TILE_LAYOUT)
    for chunk in range(0, gl.cdiv(active_splits, SPLIT_CHUNK)):
        offs_split = chunk * SPLIT_CHUNK + gl.arange(
            0, SPLIT_CHUNK, layout=CHUNK_LAYOUT
        )
        live = offs_split < active_splits
        scale = gl.exp2(
            gl.load(
                split_max_ptr + stat_base + offs_split, mask=live, other=float("-inf")
            )
            - overall_max
        )
        partial = gl.load(
            split_output_ptr
            + (stat_base + offs_split)[:, None] * LATENT
            + offs_d[None, :],
            mask=live[:, None],
            other=0.0,
        ).to(gl.float32)
        acc += partial * scale[:, None]
    merged = gl.sum(acc, axis=0)
    merged = gl.where(overall_expsum == 0.0, 0.0, merged / overall_expsum)
    # Keep the BF16 latent boundary of the unfused reduce and projection.
    attention = gl.convert_layout(
        merged.to(gl.bfloat16).to(gl.float32), gl.SliceLayout(1, W_LAYOUT)
    )
    projected = gl.sum(weight.to(gl.float32) * attention[:, None], axis=0)

    partial_row = partial_ptr + row.to(gl.int64) * (NUM_D * VALUE)
    gl.store(partial_row + d_block * VALUE + offs_v, projected, cache_modifier=".cg")
    # Every wave publishes its stores before one atomic announces the slice.
    _wait_stores()
    gl.barrier()
    old = gl.atomic_add(counter_ptr + row, 1, sem="acq_rel", scope="gpu")
    if old == NUM_D - 1:
        gl.store(counter_ptr + row, 0)
        total = gl.load(partial_row + offs_v, cache_modifier=".cg", volatile=True)
        for d in gl.static_range(1, NUM_D):
            total += gl.load(
                partial_row + d * VALUE + offs_v, cache_modifier=".cg", volatile=True
            )
        total = total.to(gl.bfloat16).to(gl.float32)
        if HAS_GATE:
            gate = gl.load(
                gate_ptr
                + token * GATE_STRIDE_B
                + (head * VALUE + offs_v) * GATE_STRIDE_N
            ).to(gl.float32)
            total *= 1.0 / (1.0 + gl.exp(-gate))
        gl.store(
            output_ptr + row * VALUE + offs_v,
            total.to(output_ptr.dtype.element_ty),
        )


def gluon_mla_query_block_reduce_project_value_gfx1250(
    split_output: torch.Tensor,
    split_max: torch.Tensor,
    split_expsum: torch.Tensor,
    cache_seqlens: torch.Tensor,
    weight: torch.Tensor,
    *,
    queries_per_seq: int,
    counters: torch.Tensor,
    gate: torch.Tensor | None = None,
    page_size: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Merge query-block split partials into projected, optionally gated values.

    Args:
        split_output: BF16 or FP32 partials ``[tokens, heads, splits, latent]``
            where each request owns ``queries_per_seq`` consecutive tokens.
        split_max: FP32 log2-space split maxima ``[tokens, heads, splits]``.
        split_expsum: FP32 split exponent sums ``[tokens, heads, splits]``.
        cache_seqlens: Int32 KV length of each request.
        weight: BF16 value projection ``[heads, latent, value]``.
        queries_per_seq: Query rows per request.
        counters: Int32 ``[tokens * heads]`` zeros; left zeroed again.
        gate: Optional BF16 raw sigmoid gate ``[tokens, heads * value]``.
        page_size: KV tile the split boundaries were computed with; must be 64.
        out: Contiguous BF16 output ``[tokens, heads * value]``.
    """
    tokens, heads, num_splits, latent = split_output.shape
    value = weight.shape[2]
    if page_size != 64:
        raise ValueError("MLA query-block reducer requires page size 64")
    if num_splits & (num_splits - 1) or latent % _QUERY_BLOCK_D or value % 32:
        raise ValueError(
            f"MLA query-block reducer needs power-of-two splits, a latent multiple "
            f"of {_QUERY_BLOCK_D} and a value multiple of 32, got {num_splits}, "
            f"{latent} and {value}"
        )
    if queries_per_seq < 1 or tokens != cache_seqlens.shape[0] * queries_per_seq:
        raise ValueError("tokens must be requests times queries_per_seq")
    expected = (
        (split_output, (tokens, heads, num_splits, latent), "split output"),
        (split_max, (tokens, heads, num_splits), "split maxima"),
        (split_expsum, (tokens, heads, num_splits), "split exponent sums"),
        (weight, (heads, latent, value), "value weight"),
        (out, (tokens, heads * value), "out"),
        (counters, (tokens * heads,), "counters"),
    )
    for tensor, shape, name in expected:
        if (
            tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
            or tensor.device != split_output.device
        ):
            raise ValueError(
                f"MLA query-block reducer requires contiguous colocated {name} {shape}"
            )
    if split_output.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("split output must be BF16 or FP32")
    if split_max.dtype != torch.float32 or split_expsum.dtype != torch.float32:
        raise TypeError("split statistics must be FP32")
    if weight.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise TypeError("value weight and out must be BF16")
    if counters.dtype != torch.int32:
        raise TypeError("counters must be int32")
    if cache_seqlens.dtype != torch.int32 or cache_seqlens.ndim != 1:
        raise TypeError("cache_seqlens must be a rank-1 int32 tensor")
    if gate is not None and (
        tuple(gate.shape) != (tokens, heads * value)
        or gate.dtype != torch.bfloat16
        or gate.device != split_output.device
        or gate.stride(1) != 1
    ):
        raise ValueError(
            "MLA query-block reducer requires a colocated BF16 gate "
            f"{(tokens, heads * value)} with contiguous inner dimension"
        )

    num_d = latent // _QUERY_BLOCK_D
    partials = torch.empty(
        (tokens * heads, num_d, value), dtype=torch.float32, device=out.device
    )
    split_chunk = min(num_splits, 64)
    _mla_query_block_reduce_project_value_kernel[(tokens, heads, num_d)](
        split_output,
        split_max,
        split_expsum,
        cache_seqlens,
        weight,
        split_output if gate is None else gate,
        out,
        partials,
        counters,
        LATENT=latent,
        VALUE=value,
        HEADS=heads,
        QUERIES=queries_per_seq,
        GATE_STRIDE_B=0 if gate is None else gate.stride(0),
        GATE_STRIDE_N=0 if gate is None else gate.stride(1),
        HAS_GATE=gate is not None,
        NUM_KV_SPLITS=num_splits,
        SPLIT_CHUNK=split_chunk,
        BLOCK_D=_QUERY_BLOCK_D,
        TILE_SIZE=page_size,
        num_warps=min(_QUERY_BLOCK_NUM_WARPS, split_chunk),
    )
    return out


__all__ = [
    "gluon_mla_reduce_project_value_gfx1250",
    "gluon_mla_query_block_reduce_project_value_gfx1250",
]
