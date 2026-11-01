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

"""Device-side MoE routing + block alignment (pure Gluon).

Drop-in replacement for the pure-torch ``moe_align_block_size`` that produces
the identical contract (``sorted_token_ids`` / ``sorted_expert_ids`` /
``sorted_weights`` / ``num_valid_ids``) but runs entirely on the GPU. Used by
the prefill path (large M), where the fused single-CTA align does not scale
(O(G^2) rank tile). Written in **Gluon** so the whole MoE compiles through one
Gluon backend.

Large-route kernels (more than 1024 routed slots), split into chunks of
``_ROUTE_CHUNK`` slots:
  1. ``_chunk_count_kernel``  -- each chunk's per-expert counts
                                 (``gl.histogram``, one row per chunk, no
                                 atomics), the padding sentinel / zero-weight
                                 fill, and the optional output clear.
  2. ``_chunk_scatter_kernel``-- every chunk sums the small count table to get
                                 each expert's padded start row plus the routes
                                 of earlier chunks, then its routes claim rows
                                 with shared-memory atomics. Workgroups also
                                 write slices of ``sorted_expert_ids`` and the
                                 valid row count.

For at most 1024 routed slots, ``_prepare_small_kernel`` combines output
initialization, histogram, and prefix construction in one workgroup. The
existing parallel scatter remains separate, avoiding both three launches and
the quadratic stable-rank tile used by the decode-only fused aligner.

Callers that accumulate into an output with atomics can pass it as
``output_to_zero``: extra workgroups of the first launch clear it, replacing a
separate fill launch.

Order within an expert is arbitrary (atomics), unlike the stable reference, but
MoE is order-independent. Outputs stay padded to their compile-time upper bound
so this path is safe under CUDA/HIP graph capture; downstream stages read the
device-side valid-count tensor and early-out on padding. CDNA3/4
``buffer_atomic_add`` is float-only, so the small-route rank counters are fp32
(values are small ints, exact in fp32).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel_amd._triton import gl, gluon, triton

# Routed slots per workgroup in the large-route count and scatter kernels.
_ROUTE_CHUNK = 2048
# 16-byte vectors cleared per workgroup when a caller passes ``output_to_zero``.
_ZERO_BLOCK = 2048


@gluon.jit
def _zero_block(ptr, numel, block, BLOCK: gl.constexpr, NW: gl.constexpr):
    """Clear 16-byte vectors ``[block * BLOCK, (block + 1) * BLOCK)`` of a
    buffer holding ``numel`` such vectors, addressed as int32 words."""
    L: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [NW, 1], [1, 0])
    vec = block.to(gl.int64) * BLOCK + gl.arange(0, BLOCK, layout=gl.SliceLayout(1, L))
    words = vec[:, None] * 4 + gl.arange(0, 4, layout=gl.SliceLayout(0, L))[None, :]
    # One mask per vector keeps whole 16-byte stores.
    gl.store(
        ptr + words, gl.zeros([BLOCK, 4], gl.int32, L), mask=(vec < numel)[:, None]
    )


def _zero_target(
    output: torch.Tensor | None, unused: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Return ``output`` as int32 words and its count of 16-byte vectors.

    Without an output, return ``unused`` with no vectors; kernels then write
    nothing through it.
    """
    if output is None:
        return unused, 0
    if not output.is_contiguous():
        raise ValueError("output_to_zero must be contiguous")
    nbytes = output.numel() * output.element_size()
    if nbytes % 16 or output.data_ptr() % 16:
        raise ValueError("output_to_zero must be whole, aligned 16-byte vectors")
    return output.view(-1).view(torch.int32), nbytes // 16


def _align_launch_metadata(grid, kernel, args):
    return {
        "name": f"{kernel.name} (routes={args['N']}, experts={args['num_experts']})"
    }


@gluon.jit(
    launch_metadata=_align_launch_metadata,
    do_not_specialize=("N", "em_max", "sentinel", "zero_numel"),
)
def _chunk_count_kernel(
    exp_ptr,  # [N] int32 flat topk_ids
    N,
    num_experts,
    counts_ptr,  # [num_chunks, BLOCK_E] int32 out: expert counts per chunk
    sti_ptr,  # [em_max] int32 out: padding sentinel
    sw_ptr,  # [em_max] float32 out: zero weights
    em_max,
    sentinel,
    zero_ptr,  # buffer cleared for the caller (unused when zero_numel is 0)
    zero_numel,  # 16-byte vectors to clear
    CHUNK: gl.constexpr,
    BLOCK_E: gl.constexpr,
    FILL_BLOCK: gl.constexpr,
    ZERO_BLOCK: gl.constexpr,
    EXPERT_START: gl.constexpr,
    NW: gl.constexpr,
):
    L: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    pid = gl.program_id(0)
    if pid * CHUNK < N:
        offs = pid * CHUNK + gl.arange(0, CHUNK, layout=L)
        in_range = offs < N
        e = (
            gl.load(exp_ptr + offs, mask=in_range, other=EXPERT_START + num_experts)
            - EXPERT_START
        )
        # EP may pass global expert IDs directly. Exclude routes outside this
        # rank's contiguous local expert range from the aligned prefix.
        valid = in_range & (e >= 0) & (e < num_experts)
        counts = gl.histogram(
            gl.where(valid, e, num_experts), BLOCK_E, mask=valid, layout=L
        )
        gl.store(counts_ptr + pid * BLOCK_E + gl.arange(0, BLOCK_E, layout=L), counts)

    rows = pid * FILL_BLOCK + gl.arange(0, FILL_BLOCK, layout=L)
    fill = rows < em_max
    gl.store(sti_ptr + rows, gl.full([FILL_BLOCK], sentinel, gl.int32, L), mask=fill)
    gl.store(sw_ptr + rows, gl.zeros([FILL_BLOCK], gl.float32, L), mask=fill)
    _zero_block(zero_ptr, zero_numel, pid, ZERO_BLOCK, NW)


@gluon.jit(
    launch_metadata=_align_launch_metadata,
    do_not_specialize=("N", "num_chunks", "nb_max"),
)
def _chunk_scatter_kernel(
    exp_ptr,  # [N] int32 flat topk_ids
    w_ptr,  # [N] float32 flat topk_weights
    N,
    num_experts,
    block_m,
    counts_ptr,  # [num_chunks, BLOCK_E] int32 from _chunk_count_kernel
    num_chunks,
    sti_ptr,  # [em_max] int32 out
    sw_ptr,  # [em_max] float32 out
    sei_ptr,  # [nb_max] int32 out: sorted_expert_ids
    nb_max,
    num_valid_ptr,  # [1] int32 out: EM
    TOPK: gl.constexpr,
    CHUNK: gl.constexpr,
    BLOCK_E: gl.constexpr,
    LOG2_BLOCK_E: gl.constexpr,
    COUNT_ROWS: gl.constexpr,
    EXPERT_START: gl.constexpr,
    NW: gl.constexpr,
):
    # Each thread owns whole expert columns of the count table, so the sums
    # over chunks need no cross-thread reduction.
    LC: gl.constexpr = gl.BlockedLayout(
        [COUNT_ROWS, BLOCK_E // (64 * NW)], [1, 64], [1, NW], [0, 1]
    )
    LE: gl.constexpr = gl.SliceLayout(0, LC)
    L: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    pid = gl.program_id(0)

    offs = pid * CHUNK + gl.arange(0, CHUNK, layout=L)
    in_range = offs < N
    e = (
        gl.load(exp_ptr + offs, mask=in_range, other=EXPERT_START + num_experts)
        - EXPERT_START
    )
    valid = in_range & (e >= 0) & (e < num_experts)
    w = gl.load(w_ptr + offs, mask=valid, other=0.0)

    # ---- per-expert routes in all chunks and in chunks before this one ----
    chunk_rows = gl.arange(0, COUNT_ROWS, layout=gl.SliceLayout(1, LC))
    count_cols = gl.arange(0, BLOCK_E, layout=LE)
    total = gl.zeros([BLOCK_E], gl.int32, LE)
    earlier = gl.zeros([BLOCK_E], gl.int32, LE)
    for r0 in range(0, num_chunks, COUNT_ROWS):
        r = r0 + chunk_rows
        counts = gl.load(
            counts_ptr + r[:, None] * BLOCK_E + count_cols[None, :],
            mask=(r < num_chunks)[:, None],
            other=0,
        )
        total += gl.sum(counts, axis=0)
        earlier += gl.sum(gl.where((r < pid)[:, None], counts, 0), axis=0)

    # Same ownership as the column sums; the scan needs a blocked layout.
    LB: gl.constexpr = gl.BlockedLayout([BLOCK_E // (64 * NW)], [64], [NW], [0])
    e_ids = gl.arange(0, BLOCK_E, layout=LB)
    total = gl.convert_layout(total, LB)
    earlier = gl.convert_layout(earlier, LB)
    # Bins at or above num_experts count no routes, so they add no rows.
    padded = (total + block_m - 1) // block_m * block_m
    row_end = gl.cumsum(padded, 0)  # inclusive
    # The last bin's inclusive end is the valid row count.
    gl.store(
        num_valid_ptr + e_ids * 0,
        row_end,
        mask=(e_ids == BLOCK_E - 1) & (pid == 0),
    )
    shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    next_row = gl.allocate_shared_memory(gl.int32, [BLOCK_E], shared)
    next_row.store(row_end - padded + earlier)
    block_end = gl.allocate_shared_memory(gl.int32, [BLOCK_E], shared)
    block_end.store(row_end // block_m)

    # ---- each routed slot claims the next free row of its expert ----
    # The shared-memory atomic returns the pre-add value: the claimed row.
    dest = next_row.atomic_scatter_add(
        gl.full([CHUNK], 1, gl.int32, L), gl.where(valid, e, 0), 0, mask=valid
    )
    packed = ((offs % TOPK) << 24) | (offs // TOPK)
    gl.store(sti_ptr + dest, packed, mask=valid)
    gl.store(sw_ptr + dest, w, mask=valid)

    # ---- sorted_expert_ids: expert(b) = #{e : block_end[e] <= b} ----
    # Binary search of the non-decreasing block ends. Blocks past the last
    # expert's end count every bin, which marks them as padding (-1).
    BLOCK_B: gl.constexpr = 64 * NW
    for b0 in range(pid * BLOCK_B, nb_max, num_chunks * BLOCK_B):
        b = b0 + gl.arange(0, BLOCK_B, layout=L)
        expert = gl.zeros([BLOCK_B], gl.int32, L)
        for step in gl.static_range(1, LOG2_BLOCK_E + 1):
            probe = block_end.gather(expert + ((BLOCK_E >> step) - 1), 0)
            expert = gl.where(probe <= b, expert + (BLOCK_E >> step), expert)
        gl.store(
            sei_ptr + b,
            gl.where(expert < num_experts, expert, gl.full([BLOCK_B], -1, gl.int32, L)),
            mask=b < nb_max,
        )


@gluon.jit(do_not_specialize=("N", "sentinel", "zero_numel"))
def _prepare_small_kernel(
    exp_ptr,
    N,
    num_experts,
    block_m,
    nb_max,
    sti_ptr,
    sw_ptr,
    sei_ptr,
    row_off_ptr,
    fill_ctr_ptr,
    num_valid_ptr,
    num_blocks_ptr,
    sentinel,
    zero_ptr,  # buffer cleared for the caller (unused when zero_numel is 0)
    zero_numel,  # 16-byte vectors to clear
    BLOCK_G: gl.constexpr,
    BLOCK_E: gl.constexpr,
    BLOCK_NB: gl.constexpr,
    MAX_BLOCKS_PER_EXPERT: gl.constexpr,
    EM_MAX: gl.constexpr,
    INIT_TILE: gl.constexpr,
    ZERO_BLOCK: gl.constexpr,
    EXPERT_START: gl.constexpr,
    NW: gl.constexpr,
):
    """Initialize outputs and construct small-route offsets in one workgroup.

    Workgroups after the first only clear the caller's output.
    """
    LG: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    LE: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    LI: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])

    pid = gl.program_id(0)
    if pid > 0:
        _zero_block(zero_ptr, zero_numel, pid - 1, ZERO_BLOCK, NW)
        return

    for r0 in gl.static_range(0, EM_MAX, INIT_TILE):
        r = r0 + gl.arange(0, INIT_TILE, layout=LI)
        gl.store(
            sti_ptr + r,
            gl.full([INIT_TILE], sentinel, gl.int32, layout=LI),
            mask=r < EM_MAX,
        )
        gl.store(
            sw_ptr + r,
            gl.full([INIT_TILE], 0.0, gl.float32, layout=LI),
            mask=r < EM_MAX,
        )

    g = gl.arange(0, BLOCK_G, layout=LG)
    gmask = g < N
    global_e = gl.load(exp_ptr + g, mask=gmask, other=EXPERT_START + num_experts)
    local_e = global_e - EXPERT_START
    route_mask = gmask & (local_e >= 0) & (local_e < num_experts)
    safe_e = gl.where(route_mask, local_e, num_experts)
    counts = gl.histogram(safe_e, BLOCK_E, mask=route_mask, layout=LE)

    e_ids = gl.arange(0, BLOCK_E, layout=LE)
    valid_e = e_ids < num_experts
    blocks_pe = (counts + block_m - 1) // block_m
    padded = blocks_pe * block_m
    row_off = gl.cumsum(padded, 0) - padded
    blocks_incl = gl.cumsum(blocks_pe, 0)
    block_start = blocks_incl - blocks_pe
    num_valid = gl.sum(padded, 0)
    num_blocks = gl.sum(blocks_pe, 0)

    gl.store(row_off_ptr + e_ids, row_off, mask=valid_e)
    gl.store(
        fill_ctr_ptr + e_ids,
        gl.full([BLOCK_E], 0.0, gl.float32, layout=LE),
        mask=valid_e,
    )
    gl.store(num_valid_ptr, num_valid)
    gl.store(num_blocks_ptr, num_blocks)

    for b0 in range(0, nb_max, BLOCK_NB):
        block_ids = b0 + gl.arange(0, BLOCK_NB, layout=LE)
        gl.store(
            sei_ptr + block_ids,
            gl.full([BLOCK_NB], -1, gl.int32, layout=LE),
            mask=(block_ids >= num_blocks) & (block_ids < nb_max),
        )

    for block_offset in gl.static_range(0, MAX_BLOCKS_PER_EXPERT):
        gl.store(
            sei_ptr + block_start + block_offset,
            e_ids,
            mask=valid_e & (block_offset < blocks_pe),
        )


@gluon.jit(do_not_specialize=("N",))
def _scatter_kernel(
    exp_ptr,  # [N] int32 flat topk_ids
    w_ptr,  # [N] float32 flat topk_weights
    N,
    topk,
    num_experts,
    row_off_ptr,  # [BLOCK_E]
    fill_ctr_ptr,  # [BLOCK_E] float32
    sti_ptr,  # [EM_max]
    sw_ptr,  # [EM_max]
    BLOCK: gl.constexpr,
    EXPERT_START: gl.constexpr,
    NW: gl.constexpr,
):
    L: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    pid = gl.program_id(0)
    offs = pid * BLOCK + gl.arange(0, BLOCK, layout=L)
    mask = offs < N
    global_e = gl.load(
        exp_ptr + offs,
        mask=mask,
        other=EXPERT_START + num_experts,
    )
    e = global_e - EXPERT_START
    valid = mask & (e >= 0) & (e < num_experts)
    safe_e = gl.where(valid, e, 0)
    # Each slot atomically claims the next free rank within its expert; the
    # buffer_atomic_add returns the pre-add value (the rank). CDNA3/4 only
    # supports float atomic add, so the counter is fp32 (ranks exact in fp32).
    ones = gl.full([BLOCK], 1.0, gl.float32, layout=L)
    rank_f = gl.amd.cdna4.buffer_atomic_add(
        fill_ctr_ptr,
        safe_e,
        ones,
        mask=valid,
    )
    rank = rank_f.to(gl.int32)
    ro = gl.load(row_off_ptr + safe_e, mask=valid, other=0)
    dest = ro + rank
    tok = offs // topk
    slot = offs % topk
    packed = ((slot << 24) | tok).to(gl.int32)
    w = gl.load(w_ptr + offs, mask=mask, other=0.0)
    gl.store(sti_ptr + dest, packed, mask=valid)
    gl.store(sw_ptr + dest, w, mask=valid)


def moe_align_block_size_device(
    topk_ids: torch.Tensor,  # [M, topk] int32
    topk_weights: torch.Tensor,  # [M, topk] float32
    num_experts: int,
    block_m: int,
    *,
    expert_start: int = 0,
    output_to_zero: torch.Tensor | None = None,
):
    """Fully-device MoE block alignment (pure Gluon). Same return contract as
    the torch ``moe_align_block_size``.

    ``topk_ids`` may use global expert IDs. ``expert_start`` identifies the
    first expert owned by this rank; routes outside the contiguous local range
    are discarded in the count and scatter kernels.

    ``output_to_zero`` is an optional contiguous tensor that the first launch
    clears, for callers that accumulate expert outputs into it with atomics.
    """
    assert topk_ids.shape == topk_weights.shape
    if expert_start < 0:
        raise ValueError("expert_start must be non-negative")
    device = topk_ids.device
    M, topk = topk_ids.shape
    N = M * topk
    sentinel = M

    exp_flat = topk_ids.reshape(-1).to(torch.int32).contiguous()
    w_flat = topk_weights.reshape(-1).to(torch.float32).contiguous()

    # Bucket static preparation bounds so changing live route counts does not
    # compile a new sorter for each token count. Consumers read num_valid_ids.
    route_bucket = triton.next_power_of_2(max(1, N))
    EM_max = route_bucket + num_experts * block_m
    nb_max = triton.cdiv(EM_max, block_m) + 1
    BLOCK_E = triton.next_power_of_2(num_experts + 1)  # +1 so the dump bin is in-range

    sti = torch.empty(EM_max, dtype=torch.int32, device=device)
    sw = torch.empty(EM_max, dtype=torch.float32, device=device)
    sei = torch.empty(nb_max, dtype=torch.int32, device=device)
    meta = torch.empty(2, dtype=torch.int32, device=device)  # [EM, num_blocks]
    zero_target, zero_numel = _zero_target(output_to_zero, sti)
    zero_blocks = triton.cdiv(zero_numel, _ZERO_BLOCK)

    if N <= 1024:
        row_off = torch.empty(BLOCK_E, dtype=torch.int32, device=device)
        fill_ctr = torch.empty(BLOCK_E, dtype=torch.float32, device=device)
        _prepare_small_kernel[(1 + zero_blocks,)](
            exp_flat,
            N,
            num_experts,
            block_m,
            nb_max,
            sti,
            sw,
            sei,
            row_off,
            fill_ctr,
            meta[0:1],
            meta[1:2],
            sentinel,
            zero_target,
            zero_numel,
            BLOCK_G=triton.next_power_of_2(N),
            BLOCK_E=BLOCK_E,
            BLOCK_NB=256,
            # N is the flattened route count (token rows * top-k), not a GEMM dimension.
            # block_m is routed rows per expert block; this matches the worst
            # case, where all routes hit one expert.
            MAX_BLOCKS_PER_EXPERT=triton.cdiv(route_bucket, block_m),
            EM_MAX=EM_max,
            INIT_TILE=triton.next_power_of_2(min(1024, EM_max)),
            ZERO_BLOCK=_ZERO_BLOCK,
            EXPERT_START=expert_start,
            NW=4,
            num_warps=4,
        )
        _scatter_kernel[(triton.cdiv(N, 256),)](
            exp_flat,
            w_flat,
            N,
            topk,
            num_experts,
            row_off,
            fill_ctr,
            sti,
            sw,
            BLOCK=256,
            EXPERT_START=expert_start,
            NW=4,
            num_warps=4,
        )
    else:
        num_warps = 8
        num_chunks = triton.cdiv(N, _ROUTE_CHUNK)
        # The count table needs no initialization: every chunk writes its row.
        # At least one bin per scatter thread: each owns whole expert columns.
        count_width = max(BLOCK_E, 64 * num_warps)
        chunk_counts = torch.empty(
            (num_chunks, count_width), dtype=torch.int32, device=device
        )
        fill_block = 2048
        _chunk_count_kernel[
            (max(num_chunks, triton.cdiv(EM_max, fill_block), zero_blocks),)
        ](
            exp_flat,
            N,
            num_experts,
            chunk_counts,
            sti,
            sw,
            EM_max,
            sentinel,
            zero_target,
            zero_numel,
            CHUNK=_ROUTE_CHUNK,
            BLOCK_E=count_width,
            FILL_BLOCK=fill_block,
            ZERO_BLOCK=_ZERO_BLOCK,
            EXPERT_START=expert_start,
            NW=num_warps,
            num_warps=num_warps,
        )
        _chunk_scatter_kernel[(num_chunks,)](
            exp_flat,
            w_flat,
            N,
            num_experts,
            block_m,
            chunk_counts,
            num_chunks,
            sti,
            sw,
            sei,
            nb_max,
            meta[0:1],
            TOPK=topk,
            CHUNK=_ROUTE_CHUNK,
            BLOCK_E=count_width,
            LOG2_BLOCK_E=count_width.bit_length() - 1,
            COUNT_ROWS=32,
            EXPERT_START=expert_start,
            NW=num_warps,
            num_warps=num_warps,
        )

    num_valid_ids = meta[0:1]
    return sti, sei, sw, num_valid_ids
