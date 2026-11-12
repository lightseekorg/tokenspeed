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

"""Packed-key sigmoid top-k routing for Kimi K3 decode."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import Platform, current_platform


@triton.jit
def _float32_to_ordered_key(value):
    bits = value.to(tl.uint32, bitcast=True)
    sign = tl.full(bits.shape, 0x80000000, tl.uint32)
    full = tl.full(bits.shape, 0xFFFFFFFF, tl.uint32)
    return bits ^ tl.where((bits & sign) != 0, full, sign)


@triton.jit
def _kimi3_route_row(
    logits,
    correction_bias,
    logical_to_physical,
    ROUTED_SCALING_FACTOR: tl.constexpr,
    NORMALIZE_TOPK_WEIGHTS: tl.constexpr,
    HAS_DISPATCH_MAP: tl.constexpr,
):
    """Return the top-16 expert ids and route weights of the row at ``logits``."""
    num_experts: tl.constexpr = 896
    padded_experts: tl.constexpr = 1024
    topk: tl.constexpr = 16
    expert = tl.arange(0, padded_experts)
    valid = expert < num_experts

    all_logits = tl.load(
        logits + expert,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)
    scores = tl.sigmoid(all_logits)
    bias = tl.load(
        correction_bias + expert,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    choice = scores + bias
    # Squash NaN like the grouped kernel: the packed key sorts NaN above finite.
    choice = tl.where(valid & (choice == choice), choice, -float("inf"))

    # Pack the ordered FP32 selection score and inverse expert id. A single
    # bitonic top-k then implements the reference's descending score order and
    # lower-id tie break without sixteen full-row reductions.
    packed = (_float32_to_ordered_key(choice).to(tl.uint64) << 32) | (
        padded_experts - expert
    ).to(tl.uint64)
    selected = tl.topk(packed, topk, dim=0)
    selected_ids = (padded_experts - (selected & 0xFFFFFFFF).to(tl.int32)).to(tl.int32)

    # Selection uses score+bias, while route weights use the unbiased sigmoid.
    selected_logits = tl.load(logits + selected_ids).to(tl.float32)
    selected_weights = tl.sigmoid(selected_logits)
    if NORMALIZE_TOPK_WEIGHTS:
        denominator = tl.sum(selected_weights, axis=0)
        denominator = tl.where(denominator != 0.0, denominator, 1.0)
        selected_weights /= denominator
    selected_weights *= ROUTED_SCALING_FACTOR

    if HAS_DISPATCH_MAP:
        # Static expert-location dispatch: route each selected logical expert
        # to this rank's physical replica inside the same launch.
        selected_ids = tl.load(logical_to_physical + selected_ids).to(tl.int32)
    return selected_ids, selected_weights


@triton.jit
def _kimi3_sigmoid_bias_topk_kernel(
    logits,
    correction_bias,
    topk_ids,
    topk_weights,
    logical_to_physical,
    ROUTED_SCALING_FACTOR: tl.constexpr,
    NORMALIZE_TOPK_WEIGHTS: tl.constexpr,
    HAS_DISPATCH_MAP: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    num_experts: tl.constexpr = 896
    topk: tl.constexpr = 16
    # One program per row; every reduction below is row-local.
    row = tl.program_id(0)
    logits += row * num_experts
    topk_ids += row * topk
    topk_weights += row * topk

    if ENABLE_PDL:
        # ``logits`` is the router GEMV output written by the predecessor
        # kernel; correction_bias / logical_to_physical are static, but the
        # very first load below consumes logits, so fence up front.
        tl.extra.cuda.gdc_wait()

    selected_ids, selected_weights = _kimi3_route_row(
        logits,
        correction_bias,
        logical_to_physical,
        ROUTED_SCALING_FACTOR,
        NORMALIZE_TOPK_WEIGHTS,
        HAS_DISPATCH_MAP,
    )
    offset = tl.arange(0, topk)
    tl.store(topk_ids + offset, selected_ids)
    tl.store(topk_weights + offset, selected_weights)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def kimi3_sigmoid_bias_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    logical_to_physical_map: torch.Tensor | None = None,
    weights_dtype: torch.dtype = torch.float32,
    enable_pdl: bool = False,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route Kimi K3 tokens to their top 16 of 896 experts, one row per CTA.

    Args:
        router_logits: Contiguous FP32 logits shaped ``[tokens, 896]``.
        correction_bias: Contiguous FP32 selection bias shaped ``[896]``.
        routed_scaling_factor: Scale applied to selected route weights.
        normalize_topk_weights: Normalize selected sigmoid scores when true.
        logical_to_physical_map: Optional contiguous integer map shaped
            ``[896]``; when given, selected logical expert ids are translated
            to physical ids inside the kernel (static EP dispatch).
        weights_dtype: Output dtype for the route weights (selection and
            normalization always run in FP32; the store casts).
        enable_pdl: Launch with programmatic dependent launch and fence the
            router-logits read with ``gdc_wait`` / ``gdc_launch_dependents``
            (NVIDIA only; ignored elsewhere). Safe only when the router GEMV
            that writes ``router_logits`` is chained on the same stream/graph.
        out: Optional contiguous weight and expert-ID output buffers. Weights
            use the supplied buffer's precision. Omission allocates weights in
            ``weights_dtype`` and IDs in INT32.

    Returns:
        ``(topk_weights, topk_ids)`` shaped ``[tokens, 16]``. Supplied output
        buffers are returned in place. A dispatch map produces physical IDs.
    """
    if (
        router_logits.dim() != 2
        or router_logits.shape[1] != 896
        # The kernel offsets its row base with an INT32 program id.
        or not 0 < router_logits.shape[0] <= (2**31 - 1) // 896
        or router_logits.dtype != torch.float32
        or not router_logits.is_cuda
        or not router_logits.is_contiguous()
    ):
        raise ValueError(
            "Kimi K3 top-k requires contiguous GPU FP32 logits [tokens, 896]"
        )
    tokens = router_logits.shape[0]
    if (
        correction_bias.shape != (896,)
        or correction_bias.dtype != torch.float32
        or correction_bias.device != router_logits.device
        or not correction_bias.is_contiguous()
    ):
        raise ValueError("Kimi K3 top-k requires contiguous colocated FP32 bias [896]")
    if logical_to_physical_map is not None and (
        logical_to_physical_map.shape != (896,)
        or logical_to_physical_map.dtype not in (torch.int32, torch.int64)
        or logical_to_physical_map.device != router_logits.device
        or not logical_to_physical_map.is_contiguous()
    ):
        raise ValueError(
            "Kimi K3 top-k dispatch map must be a contiguous colocated "
            "int32/int64 [896] tensor"
        )

    if out is None:
        topk_weights = torch.empty(
            (tokens, 16), dtype=weights_dtype, device=router_logits.device
        )
        topk_ids = torch.empty(
            (tokens, 16), dtype=torch.int32, device=router_logits.device
        )
    else:
        topk_weights, topk_ids = out
    # waves_per_eu is a CDNA-only occupancy hint; NVIDIA Triton rejects it.
    amd_kwargs = {"waves_per_eu": 1} if Platform.get().is_amd else {}
    pdl_kwargs = (
        {"launch_pdl": True} if enable_pdl and current_platform().is_nvidia else {}
    )
    _kimi3_sigmoid_bias_topk_kernel[(tokens,)](
        router_logits,
        correction_bias,
        topk_ids,
        topk_weights,
        logical_to_physical_map,
        ROUTED_SCALING_FACTOR=float(routed_scaling_factor),
        NORMALIZE_TOPK_WEIGHTS=normalize_topk_weights,
        HAS_DISPATCH_MAP=logical_to_physical_map is not None,
        ENABLE_PDL=enable_pdl,
        num_warps=8,
        num_stages=1,
        **amd_kwargs,
        **pdl_kwargs,
    )
    return topk_weights, topk_ids


# Rows of one sorted launch: the arrival masks keep one bit per row, eight
# 32-bit words per expert.
_SORTED_ROW_WORDS = 8
_SORTED_MAX_ROWS = 32 * _SORTED_ROW_WORDS
# Routes each sorting program places, one per thread.
_SORTED_HELPER_ROUTES = 512
# The shared-expert down projection a sorted launch can carry. Each program
# writes a block of output columns of every row, reducing K in 256-wide steps:
# 32 columns spread the weight over more CUs through 32 rows, and 64 halve the
# activation reads above that.
_SHARED_DOWN_BLOCK_K = 256
_SHARED_DOWN_NARROW_ROWS = 32
_SHARED_DOWN_MAX_ROWS = 128


def _shared_down_block_n(rows: int) -> int:
    return 32 if rows <= _SHARED_DOWN_NARROW_ROWS else 64


def _sorted_scratch_words() -> int:
    # [896 * ROW_WORDS] arrival masks, the arrival and sorted counters.
    return 896 * _SORTED_ROW_WORDS + 16


@triton.jit
def _popcount(words):
    bits = words.to(tl.uint32, bitcast=True)
    bits = bits - ((bits >> 1) & 0x55555555)
    bits = (bits & 0x33333333) + ((bits >> 2) & 0x33333333)
    bits = (bits + (bits >> 4)) & 0x0F0F0F0F
    return ((bits * 0x01010101) >> 24).to(tl.int32)


@triton.jit
def _kimi3_sorted_routes(
    topk_ids,
    topk_weights,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    masks,
    sorted_count,
    counter,
    rows,
    helper,
    helpers,
    BLOCK_SHIFT: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    WORDS: tl.constexpr,
    HELPER_ROUTES: tl.constexpr,
):
    """Place sorting program ``helper``'s share of the block-aligned expert sort.

    ``masks[e * ROW_WORDS + r // 32]`` has bit ``r % 32`` set when row ``r``
    routes to expert ``e``; rows end below ``32 * WORDS``. A row picks
    distinct experts, so the routes before ``(r, e)`` in expert ``e``'s run
    are its rows below ``r``: the sorted order is the flat top-k order,
    without atomics. Every sorting program derives the expert offsets from
    the masks and places ``HELPER_ROUTES`` routes; the route that opens a
    block writes the block's expert, and an expert's last route its padding.

    The rows wrote through to memory before arriving, and the caller's
    acquire invalidated this CU's caches, so plain loads read their writes.
    """
    num_experts: tl.constexpr = 896
    padded_experts: tl.constexpr = 1024
    topk: tl.constexpr = 16
    block: tl.constexpr = 1 << BLOCK_SHIFT
    route = helper * HELPER_ROUTES + tl.arange(0, HELPER_ROUTES)
    valid = route < rows * topk
    expert_id = tl.load(topk_ids + route, mask=valid, other=0)
    weight = tl.load(topk_weights + route, mask=valid, other=0.0)
    expert = tl.arange(0, padded_experts)
    expert_valid = expert < num_experts
    word = tl.arange(0, WORDS)
    word_offsets = expert[:, None] * ROW_WORDS + word[None, :]
    expert_masks = tl.load(masks + word_offsets, mask=expert_valid[:, None], other=0)
    word_counts = _popcount(expert_masks)
    count = tl.sum(word_counts, axis=1)
    padded = ((count + block - 1) >> BLOCK_SHIFT) << BLOCK_SHIFT
    first = tl.cumsum(padded, 0) - padded
    # Each mask word's first slot: the expert's first slot plus the rows of
    # its lower words.
    word_bases = first[:, None] + tl.cumsum(word_counts, 1) - word_counts
    if helper == 0:
        tl.store(num_valid_ids, tl.sum(padded, 0))
        tl.store(num_valid_ids + 1, rows)

    row = route // topk
    word_offset = expert_id * WORDS + row // 32
    below = tl.gather(
        tl.reshape(expert_masks, [padded_experts * WORDS]), word_offset, 0
    ) & ((1 << (row % 32)) - 1)
    target = tl.gather(
        tl.reshape(word_bases, [padded_experts * WORDS]), word_offset, 0
    ) + _popcount(below)
    tl.store(sorted_ids + target, ((route % topk) << 24) | row, mask=valid)
    tl.store(sorted_weights + target, weight, mask=valid)
    tl.store(
        sorted_expert_ids + (target >> BLOCK_SHIFT),
        expert_id,
        mask=valid & ((target & (block - 1)) == 0),
    )
    fill = tl.gather(first + count, expert_id, 0)
    fill_end = tl.gather(first + padded, expert_id, 0)
    padding = fill[:, None] + tl.arange(0, block)[None, :]
    tl.store(
        sorted_ids + padding,
        tl.full([HELPER_ROUTES, block], topk << 24, tl.int32) | rows,
        mask=(valid & (target + 1 == fill))[:, None] & (padding < fill_end[:, None]),
    )
    # The last program done with the masks leaves them and both counters
    # zeroed for the next launch; every wave here consumed its mask loads
    # before the barrier.
    tl.debug_barrier()
    last = helpers == 1
    if not last:
        last = tl.atomic_add(sorted_count, 1, sem="relaxed", scope="gpu") == helpers - 1
    if last:
        tl.store(
            masks + word_offsets,
            tl.zeros([padded_experts, WORDS], tl.int32),
            mask=expert_valid[:, None],
        )
        tl.atomic_xchg(counter, 0, sem="relaxed", scope="gpu")
        tl.atomic_xchg(sorted_count, 0, sem="relaxed", scope="gpu")


@triton.jit
def _kimi3_shared_down_tile(
    hidden,
    weight,
    out,
    out_stride,
    rows,
    tile,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Write columns ``tile * BLOCK_N`` onward of ``hidden @ weight.T`` for every row.

    Reduces in FP32 and stores the output dtype; rows past ``rows`` pad the
    MFMA tile and are neither read nor written.
    """
    row = tl.arange(0, BLOCK_M)
    column = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    row_valid = (row < rows)[:, None]
    accumulator = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    for k_start in tl.static_range(0, K, BLOCK_K):
        activation = tl.load(
            hidden + row[:, None] * K + (k_start + k)[None, :],
            mask=row_valid,
            other=0.0,
        )
        weight_tile = tl.load(
            weight + column[:, None] * K + (k_start + k)[None, :],
            cache_modifier=".cg",
        )
        accumulator = tl.dot(activation, tl.trans(weight_tile), accumulator)
    tl.store(
        out + row[:, None] * out_stride + column[None, :],
        accumulator.to(out.dtype.element_ty),
        mask=row_valid,
    )


@triton.jit(do_not_specialize=["rows", "helpers", "shared_out_stride"])
def _kimi3_sigmoid_bias_topk_sorted_kernel(
    logits,
    correction_bias,
    topk_ids,
    topk_weights,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    scratch,
    shared_hidden,
    shared_weight,
    shared_out,
    shared_out_stride,
    rows,
    helpers,
    ROUTED_SCALING_FACTOR: tl.constexpr,
    NORMALIZE_TOPK_WEIGHTS: tl.constexpr,
    BLOCK_SHIFT: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    WORDS: tl.constexpr,
    HELPER_ROUTES: tl.constexpr,
    SHARED_TILES: tl.constexpr,
    SHARED_K: tl.constexpr,
    SHARED_BLOCK_M: tl.constexpr,
    SHARED_BLOCK_N: tl.constexpr,
    SHARED_BLOCK_K: tl.constexpr,
):
    num_experts: tl.constexpr = 896
    topk: tl.constexpr = 16
    masks = scratch
    counter = scratch + num_experts * ROW_WORDS
    sorted_count = counter + 1
    pid = tl.program_id(0)
    if pid < rows:
        # One program per row routes it exactly as
        # _kimi3_sigmoid_bias_topk_kernel and marks its experts' masks.
        selected_ids, selected_weights = _kimi3_route_row(
            logits + pid * num_experts,
            correction_bias,
            correction_bias,
            ROUTED_SCALING_FACTOR,
            NORMALIZE_TOPK_WEIGHTS,
            False,
        )
        offset = tl.arange(0, topk)
        # Write through, so the arrival needs no L2 write-back (a release's
        # write-back serializes the programs sharing an L2).
        tl.store(topk_ids + pid * topk + offset, selected_ids, cache_modifier=".wt")
        tl.store(
            topk_weights + pid * topk + offset, selected_weights, cache_modifier=".wt"
        )
        tl.atomic_or(
            masks + selected_ids * ROW_WORDS + pid // 32,
            tl.full([topk], 1, tl.int32) << (pid % 32),
            sem="relaxed",
            scope="gpu",
        )
        # The compiler's barrier before the one-thread arrival does not wait
        # for memory, so every wave first waits until its stores and mask
        # updates have landed.
        tl.inline_asm_elementwise(
            asm="s_waitcnt vmcnt(0)",
            constraints="=r,~{memory}",
            args=[],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        tl.debug_barrier()
        tl.atomic_add(counter, 1, sem="relaxed", scope="gpu")
    else:
        helper = pid - rows - SHARED_TILES
        if helper >= 0:
            # The sorting programs come last in the grid, so every row program
            # has dispatched before they wait for its arrival. One thread
            # polls; its acquire invalidates this CU's caches, and the barrier
            # keeps every wave's loads after the last poll.
            arrived = tl.atomic_add(counter, 0, sem="acquire", scope="gpu")
            while arrived < rows:
                tl.inline_asm_elementwise(
                    asm="s_sleep 2",
                    constraints="=r",
                    args=[],
                    dtype=tl.int32,
                    is_pure=False,
                    pack=1,
                )
                arrived = tl.atomic_add(counter, 0, sem="acquire", scope="gpu")
            tl.debug_barrier()
            _kimi3_sorted_routes(
                topk_ids,
                topk_weights,
                sorted_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                masks,
                sorted_count,
                counter,
                rows,
                helper,
                helpers,
                BLOCK_SHIFT,
                ROW_WORDS,
                WORDS,
                HELPER_ROUTES,
            )
        else:
            if SHARED_TILES > 0:
                # The shared-expert down projection shares no data with the
                # routing; its programs run while the rows route and sort.
                _kimi3_shared_down_tile(
                    shared_hidden,
                    shared_weight,
                    shared_out,
                    shared_out_stride,
                    rows,
                    pid - rows,
                    SHARED_K,
                    SHARED_BLOCK_M,
                    SHARED_BLOCK_N,
                    SHARED_BLOCK_K,
                )


def kimi3_sorted_routing_scratch(device: torch.device) -> torch.Tensor:
    """Allocate the zeroed scratch ``kimi3_sigmoid_bias_topk_sorted`` reuses.

    Every launch leaves it zeroed again, so one buffer serves any number of
    sequential (including graph-replayed) launches on one stream.
    """
    return torch.zeros(_sorted_scratch_words(), dtype=torch.int32, device=device)


def kimi3_sorted_routing_fuses_shared_down(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None,
) -> bool:
    """Whether ``kimi3_sigmoid_bias_topk_sorted`` can carry this down projection.

    ``hidden_states`` and ``weight`` must be contiguous BF16 ``[rows, K]`` and
    ``[N, K]`` on one device, with at most 128 rows, ``K`` a multiple of 256
    and ``N`` of 64 (of 32 through 32 rows). ``out``, when given, must be
    BF16 ``[rows, N]`` with unit inner stride and non-overlapping rows.
    """
    if (
        hidden_states.dim() != 2
        or weight.dim() != 2
        or hidden_states.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not hidden_states.is_cuda
        or weight.device != hidden_states.device
        or not hidden_states.is_contiguous()
        or not weight.is_contiguous()
    ):
        return False
    rows, width = hidden_states.shape
    output_width = weight.shape[0]
    if (
        not 0 < rows <= _SHARED_DOWN_MAX_ROWS
        or weight.shape[1] != width
        or width % _SHARED_DOWN_BLOCK_K
        or output_width % _shared_down_block_n(rows)
    ):
        return False
    return out is None or (
        out.shape == (rows, output_width)
        and out.dtype == torch.bfloat16
        and out.device == hidden_states.device
        and out.stride(1) == 1
        and out.stride(0) >= output_width
    )


def kimi3_sigmoid_bias_topk_sorted(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    routed_scaling_factor: float,
    normalize_topk_weights: bool,
    block_size: int,
    scratch: torch.Tensor,
    shared_down: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Route like ``kimi3_sigmoid_bias_topk`` and sort the routes by expert.

    One launch: each of the first ``rows`` programs routes one row exactly as
    ``kimi3_sigmoid_bias_topk`` (FP32 weights, no dispatch map) and marks its
    experts' arrival masks. ``ceil(rows * 16 / 512)`` sorting programs after
    them wait for every row, then write the block-aligned expert sort of all
    routes in the format of the gfx950 MoE ``gluon_moe_sorting`` (896 local
    experts from zero), with each expert's routes in flat top-k order.
    Given ``shared_down``, programs between the rows and the sort also write
    that independent projection, so it costs no launch of its own.

    Args:
        router_logits: Contiguous FP32 logits shaped ``[rows, 896]``, with
            ``1 <= rows <= 256``.
        correction_bias: Contiguous FP32 selection bias shaped ``[896]``.
        routed_scaling_factor: Scale applied to selected route weights.
        normalize_topk_weights: Normalize selected sigmoid scores when true.
        block_size: Power-of-two expert padding granularity of the sort.
        scratch: ``kimi3_sorted_routing_scratch`` buffer on the logits'
            device; launches sharing it must be stream-ordered.
        shared_down: Optional ``(hidden_states, weight, out)`` of a BF16
            projection ``hidden_states @ weight.T`` over the same rows,
            accepted by ``kimi3_sorted_routing_fuses_shared_down``. It reduces
            in FP32 into ``out``, or a new contiguous tensor when ``out`` is
            None.

    Returns:
        ``(topk_weights, topk_ids, sorted_ids, sorted_weights,
        sorted_expert_ids, num_valid_ids)``: FP32 and INT32 ``[rows, 16]``
        routes, then the sort. ``sorted_ids`` holds ``(slot << 24) | row``
        per route and ``(16 << 24) | rows`` per padding slot;
        ``num_valid_ids`` is ``[padded routes, rows]``. Entries past the
        padded routes (and their blocks) are unwritten. Given
        ``shared_down``, its output follows as a seventh tensor.
    """
    if (
        router_logits.dim() != 2
        or router_logits.shape[1] != 896
        or not 0 < router_logits.shape[0] <= _SORTED_MAX_ROWS
        or router_logits.dtype != torch.float32
        or not router_logits.is_cuda
        or not router_logits.is_contiguous()
    ):
        raise ValueError(
            f"sorted Kimi K3 top-k requires contiguous GPU FP32 logits "
            f"[rows <= {_SORTED_MAX_ROWS}, 896]"
        )
    if (
        correction_bias.shape != (896,)
        or correction_bias.dtype != torch.float32
        or correction_bias.device != router_logits.device
        or not correction_bias.is_contiguous()
    ):
        raise ValueError("Kimi K3 top-k requires contiguous colocated FP32 bias [896]")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a power of two")
    if not current_platform().is_cdna4:
        raise RuntimeError("sorted Kimi K3 top-k waits for stores as gfx950 does")
    if (
        scratch.shape != (_sorted_scratch_words(),)
        or scratch.dtype != torch.int32
        or scratch.device != router_logits.device
        or not scratch.is_contiguous()
    ):
        raise ValueError("scratch must come from kimi3_sorted_routing_scratch")
    rows = router_logits.shape[0]
    device = router_logits.device
    routes = rows * 16
    # gluon_moe_sorting's capacity: every nonempty expert may add a partial block.
    capacity = (routes + min(routes, 896) * (block_size - 1)) // block_size * block_size
    topk_weights = torch.empty((rows, 16), dtype=torch.float32, device=device)
    topk_ids = torch.empty((rows, 16), dtype=torch.int32, device=device)
    sorted_ids = torch.empty((capacity,), dtype=torch.int32, device=device)
    sorted_weights = torch.empty((capacity,), dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty(
        (triton.cdiv(capacity, block_size),), dtype=torch.int32, device=device
    )
    num_valid_ids = torch.empty((2,), dtype=torch.int32, device=device)
    helpers = triton.cdiv(routes, _SORTED_HELPER_ROUTES)
    shared_block_m = 16
    shared_block_n = 32
    if shared_down is None:
        shared_hidden = shared_weight = shared_out = router_logits
        shared_tiles = shared_width = shared_out_stride = 0
    else:
        shared_hidden, shared_weight, shared_out = shared_down
        if (
            shared_hidden.shape[0] != rows
            or shared_hidden.device != device
            or not kimi3_sorted_routing_fuses_shared_down(
                shared_hidden, shared_weight, shared_out
            )
        ):
            raise ValueError(
                "shared_down must be a BF16 projection of the routed rows that "
                "kimi3_sorted_routing_fuses_shared_down accepts"
            )
        if shared_out is None:
            shared_out = shared_hidden.new_empty((rows, shared_weight.shape[0]))
        shared_width = shared_hidden.shape[1]
        shared_block_n = _shared_down_block_n(rows)
        shared_tiles = shared_weight.shape[0] // shared_block_n
        shared_out_stride = shared_out.stride(0)
        # MFMA row tiles: one binary each through 16, 32, 64 and 128 rows.
        shared_block_m = max(16, triton.next_power_of_2(rows))
    _kimi3_sigmoid_bias_topk_sorted_kernel[(rows + shared_tiles + helpers,)](
        router_logits,
        correction_bias,
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        scratch,
        shared_hidden,
        shared_weight,
        shared_out,
        shared_out_stride,
        rows,
        helpers,
        ROUTED_SCALING_FACTOR=float(routed_scaling_factor),
        NORMALIZE_TOPK_WEIGHTS=normalize_topk_weights,
        BLOCK_SHIFT=block_size.bit_length() - 1,
        ROW_WORDS=_SORTED_ROW_WORDS,
        # Mask words the rows use, in power-of-two buckets: one binary each
        # through 32, 64, 128 and 256 rows.
        WORDS=triton.next_power_of_2(triton.cdiv(rows, 32)),
        HELPER_ROUTES=_SORTED_HELPER_ROUTES,
        SHARED_TILES=shared_tiles,
        SHARED_K=shared_width,
        SHARED_BLOCK_M=shared_block_m,
        SHARED_BLOCK_N=shared_block_n,
        SHARED_BLOCK_K=_SHARED_DOWN_BLOCK_K,
        num_warps=8,
        num_stages=1,
        waves_per_eu=1,
    )
    routes_and_sort = (
        topk_weights,
        topk_ids,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
    )
    if shared_down is None:
        return routes_and_sort
    return (*routes_and_sort, shared_out)


__all__ = [
    "kimi3_sigmoid_bias_topk",
    "kimi3_sigmoid_bias_topk_sorted",
    "kimi3_sorted_routing_fuses_shared_down",
    "kimi3_sorted_routing_scratch",
]
