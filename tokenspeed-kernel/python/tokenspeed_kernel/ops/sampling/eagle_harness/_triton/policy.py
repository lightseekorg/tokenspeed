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

"""Triton kernels of the EAGLE-Harness device policy: one program each, a
scalar state machine over the int32 state, the ``[4, depth + 1]`` slots and
the go flags, with the CUDA build's layouts (``thirdparty/cuda/csrc/
eagle_harness_policy.cu``) and arithmetic: fp64 cost comparisons with one
IEEE rounding per operation (launched without FMA contraction; fp64 ``/``
is correctly rounded). The
launchers live in ``ops/sampling/eagle_harness/triton.py``.

The go flags live in mapped pinned host memory, which the host polls: they
are stored write-through.

``select_verify``'s cumulative ranked scores replicate torch's ROCm
reductions bit for bit: ``mean(dim=0)`` as its reduce kernel sums at the
shapes the policy ranks (four strided accumulators per column; a single
column as a pairwise tree), and ``cumsum`` as rocPRIM's block scan of 16
items per thread (a sequential thread reduce, a DPP warp scan of 64 lanes,
the warp prefixes, then a sequential thread scan).
"""

from __future__ import annotations

from tokenspeed_kernel._triton import tl, triton

# Scalars of the int32 state tensor.
_TINY = tl.constexpr(0)
_TIER = tl.constexpr(1)
_SAMPLED_NEXT = tl.constexpr(2)
_SAMPLED_PLAN = tl.constexpr(3)
_SAMPLED_SEEN = tl.constexpr(4)
_BENCH_BATCH_SIZE = tl.constexpr(5)
_RANKED_COUNT = tl.constexpr(6)
_BATCH_SIZE = tl.constexpr(7)
_SAMPLED = tl.constexpr(8)
_COST_ID = tl.constexpr(9)
_INTERVAL = tl.constexpr(10)
_VARIANT = tl.constexpr(11)
_HIST_MAX = tl.constexpr(12)
_PAST_VERIFY_NUM = tl.constexpr(13)
_LAST_DEPTH = tl.constexpr(14)
_STOP_WIDTH = tl.constexpr(15)
_CUMU_COUNT = tl.constexpr(16)
_VERIFY_COUNT = tl.constexpr(17)
_FLAT = tl.constexpr(18)
_VERIFY_NUM = tl.constexpr(19)
_TABLE_WIDTH = tl.constexpr(20)
NUM_STATE = 21

# Tiers, then the cost variant (and verify floor) of a tiny round; tier_table holds
# the verify floor per variant, then the width per tier.
_TINY_VARIANT = tl.constexpr(4)
_VARIANTS = tl.constexpr(5)
TIER_TABLE_SIZE = 9
# Columns of batch_table.
_COST_ID_COL = tl.constexpr(0)
_CHAIN_TIER_ROW_COL = tl.constexpr(1)
_INTERVALS_COL = tl.constexpr(2)
_CACHE_STEP_COL = tl.constexpr(3)
_TABLE_WIDTH_COL = tl.constexpr(4)
BATCH_COLS = 5
# Entries of a chain cost row per depth, and of a tree round's cost row copy.
_GAIN_DENOM = tl.constexpr(0)
_STEP_COST = tl.constexpr(1)
_CHAIN_THRESHOLD = tl.constexpr(2)
_CLIFF_GAIN_DENOM = tl.constexpr(3)
_CLIFF_COST = tl.constexpr(4)
CHAIN_F64 = 5
_ROUND_DRAFT_COST = tl.constexpr(0)
_ROUND_MODEL_COST_RATIO = tl.constexpr(1)
_ROUND_THRESHOLDS = tl.constexpr(2)
# Columns of interval_i32: the measured width count, then the widths.
_WIDTHS_COL = tl.constexpr(1)
# Capacities: verify history, measured widths, tree depth, ranked shapes.
MAX_HISTORY = 32
MAX_WIDTHS = 32
MAX_DEPTH = 32
RANK_MAX_ROWS = 32
RANK_MAX_COLS = 64
# rocPRIM's block scan of the ranked scores: 128 threads hold 16 items each.
_SCAN_THREADS = tl.constexpr(128)
_SCAN_ITEMS = tl.constexpr(16)

_INT32_MIN = tl.constexpr(-(2**31))
_INT32_MAX = tl.constexpr(2**31 - 1)


@triton.jit
def _f64_arg(bits):
    """A double passed as its int64 bit pattern (Triton passes Python floats
    as fp32)."""
    return bits.to(tl.int64).to(tl.float64, bitcast=True)


@triton.jit
def _publish_go(go_host_ptr, depth, seq, draft, enabled):
    """Depth ``depth``'s go flag, ``(seq << 1) | draft``, into the pinned
    mirror the host polls."""
    value = (seq << 1) | draft.to(tl.int64)
    # Write-through: the store reaches host memory, not only the device's caches.
    tl.store(go_host_ptr + depth, value, mask=enabled, cache_modifier=".wt")


@triton.jit
def _deltas_depth(d, tiny, TREE_DEPTH: tl.constexpr, CHAIN: tl.constexpr):
    """Whether depth d's slot asks for work deltas, which the next decision
    compares."""
    if CHAIN:
        result = (d % 2 == 1) & (d < TREE_DEPTH - 1)
    else:
        result = ~((tiny != 0) & (d < 2)) & (d != TREE_DEPTH - 1) & (d % 2 == 0)
    return result


@triton.jit
def _early_decision(
    d, narrow, tiny, NODE_TOP_K, TREE_DEPTH: tl.constexpr, CHAIN: tl.constexpr
):
    """Depth d's decision when it compares no work deltas: (decided, width,
    whether the cost criterion stops)."""
    last = d == TREE_DEPTH - 1
    if CHAIN:
        decided = last | (d == 0) | (d % 2 == 1)
        width = tl.where(last, -1, 1)
        stop = last
    else:
        tiny_early = (tiny != 0) & (d < 2)
        decided = tiny_early | last | (d % 2 == 0)
        width = tl.where(tiny_early, NODE_TOP_K, tl.where(last, -1, narrow))
        stop = ~tiny_early & last
    return decided, width, stop


@triton.jit
def _batch_cut(bench_batch_size, exec_mode, bs):
    """Exec mode 1's batch cut of bs."""
    bench = bench_batch_size.to(tl.int64)
    return (exec_mode == 1) & (bs != 1) & (bench > 4 * bs) & (bench > bs + 4)


@triton.jit
def _commit(
    d,
    width,
    cost_stop,
    cur_width,
    cur_start,
    tiny,
    tier,
    sampled,
    sampled_plan,
    sampled_seen,
    sampled_next,
    past_verify_num,
    hist_max,
    last_depth,
    stop_width,
    cumu_count,
    verify_count,
    verify_num_state,
    flat,
    tier_table_ptr,
    verify_num,
    min_verify_count,
    NODE_TOP_K,
    TREE_DEPTH: tl.constexpr,
    CHAIN: tl.constexpr,
):
    """Takes depth d's decision (a sampled round stops only at its planned
    depth): stops the round (its counts, the next sampled round's plan) or
    returns depth d + 1's slots."""
    is_sampled = sampled != 0
    sampled_seen = tl.where(
        is_sampled & cost_stop & (sampled_seen < 0), d, sampled_seen
    )
    stop = tl.where(is_sampled, d >= sampled_plan, cost_stop)
    start = cur_start + NODE_TOP_K * cur_width
    if CHAIN:
        stop_verify_count = tl.minimum(start, verify_num)
        stop_verify_num = stop_verify_count
        stop_flat = flat
    else:
        # past * 2 in int64: a cut round records a measured width up to 2**31 - 1.
        floor = tl.load(tier_table_ptr + tl.where(tiny != 0, _TINY_VARIANT, tier))
        past = tl.maximum(hist_max, floor).to(tl.int64)
        stop_verify_count = tl.minimum(
            tl.minimum(verify_num, start).to(tl.int64),
            tl.maximum(past * 2, min_verify_count),
        ).to(tl.int32)
        stop_verify_num = verify_num_state
        stop_flat = tiny
    # The next sampled round's plan: the first depth whose rule said stop, else two deeper.
    plan = tl.where(
        sampled_seen >= 0, sampled_seen, tl.minimum(sampled_plan + 2, TREE_DEPTH - 1)
    )
    last_depth = tl.where(stop, d, last_depth)
    stop_width = tl.where(stop, width, stop_width)
    cumu_count = tl.where(stop, start, cumu_count)
    verify_count = tl.where(stop, stop_verify_count, verify_count)
    verify_num_state = tl.where(stop, stop_verify_num, verify_num_state)
    flat = tl.where(stop, stop_flat, flat)
    sampled_next = tl.where(stop & is_sampled, plan, sampled_next)
    go = ~stop
    next_width = tl.where(go, width, 0)
    next_start = tl.where(go, start, 0)
    deltas = go & _deltas_depth(d + 1, tiny, TREE_DEPTH, CHAIN)
    next_work = tl.where(deltas, tl.minimum(past_verify_num, start), 0)
    return (
        go,
        next_width,
        next_start,
        next_work,
        sampled_seen,
        sampled_next,
        last_depth,
        stop_width,
        cumu_count,
        verify_count,
        verify_num_state,
        flat,
    )


@triton.jit(
    do_not_specialize=[
        "frontier_stride",
        "hist_len",
        "num_intervals",
        "cost_cols",
        "chain_past_verify_num",
        "verify_num",
        "min_verify_count",
    ]
)
def _policy_begin_round_kernel(
    state_ptr,
    seq_ptr,
    hist_ptr,
    slots_ptr,
    go_host_ptr,
    batch_table_ptr,
    tier_table_ptr,
    cost_f64_ptr,
    round_f64_ptr,
    live_rows_ptr,
    frontier_ptr,
    sampled_ptr,
    frontier_stride,
    hist_len,
    num_intervals,
    cost_cols,
    chain_past_verify_num,
    verify_num,
    min_verify_count,
    NODE_TOP_K: tl.constexpr,
    TREE_DEPTH: tl.constexpr,
    CHAIN: tl.constexpr,
    DEPTH_PAD: tl.constexpr,
    COPY_PAD: tl.constexpr,
):
    """Round start: past verify width, sampled plan, cost interval, depth 0's
    decision and depth 1's slots (a width table round: every depth's)."""
    tiny = tl.load(state_ptr + _TINY)
    tier = tl.load(state_ptr + _TIER)
    sampled_next = tl.load(state_ptr + _SAMPLED_NEXT)
    sampled_plan = tl.load(state_ptr + _SAMPLED_PLAN)
    sampled_seen = tl.load(state_ptr + _SAMPLED_SEEN)
    seq = tl.load(seq_ptr) + 1
    bs = tl.load(live_rows_ptr)
    live = bs > 0
    sampled = tl.where(live, tl.load(sampled_ptr).to(tl.int32), 0)
    # The batch size's table row, verify history and cached lengths.
    row_index = tl.maximum(bs - 1, 0).to(tl.int64)
    batch_row = batch_table_ptr + row_index * 5
    cost_id = tl.load(batch_row + _COST_ID_COL, mask=live, other=0)
    chain_tier_row = tl.load(batch_row + _CHAIN_TIER_ROW_COL, mask=live, other=0)
    intervals = tl.load(batch_row + _INTERVALS_COL, mask=live, other=1)
    cache_step = tl.load(batch_row + _CACHE_STEP_COL, mask=live, other=1)
    table_width = tl.load(batch_row + _TABLE_WIDTH_COL, mask=live, other=0).to(tl.int32)
    h = tl.arange(0, 32)
    hist = tl.load(
        hist_ptr + row_index * hist_len + h,
        mask=live & (h < hist_len),
        other=_INT32_MIN,
    )
    hist_max = tl.max(hist)
    lengths = tl.zeros([1024], dtype=tl.int64)
    for start in range(0, bs, 1024):
        i = start + tl.arange(0, 1024)
        lengths += tl.load(
            frontier_ptr + i.to(tl.int64) * frontier_stride, mask=i < bs, other=0
        )
    frontier_sum = tl.sum(lengths)

    # The round's scalars start cleared.
    zero = bs * 0
    if CHAIN:
        past_verify_num = tl.where(live, chain_past_verify_num, 0)
        state_hist_max = zero
    else:
        floor = tl.load(tier_table_ptr + tl.where(tiny != 0, _TINY_VARIANT, tier))
        past_verify_num = tl.where(live, tl.maximum(hist_max, floor), 0)
        state_hist_max = tl.where(live, hist_max, 0)
        # Only chains have width table rounds.
        table_width = zero
    table_round = live & (table_width > 0)
    normal = live & ~table_round
    is_sampled = normal & (sampled != 0)
    sampled_plan = tl.where(is_sampled, sampled_next, sampled_plan)
    sampled_seen = tl.where(is_sampled, -1, sampled_seen)
    # The cost context's interval of the mean cached length (non-negative lengths:
    # // is Python's floor division).
    average_cache_len = frontier_sum // tl.maximum(bs, 1)
    interval = tl.minimum(
        tl.maximum(average_cache_len // cache_step - 1, 0), intervals - 1
    )
    interval = tl.where(normal, interval, 0).to(tl.int32)
    cost_id = tl.where(normal, cost_id, 0).to(tl.int32)
    if CHAIN:
        variant = tl.where(normal, chain_tier_row, 0).to(tl.int32)
        narrow = zero + 1
    else:
        variant = tl.where(normal, tl.where(tiny != 0, _TINY_VARIANT, tier), 0)
        narrow = tl.load(tier_table_ptr + _VARIANTS + tier)
    # Depth 0: the root (one node) and its K children from the first step.
    cur_width = tl.where(normal, 1, 0)
    cur_start = tl.where(normal, 1, 0)
    cur_work = tl.where(normal & _deltas_depth(zero, tiny, TREE_DEPTH, CHAIN), 1, 0)
    cur_go = tl.where(normal, 1, 0)
    _, width, cost_stop = _early_decision(
        zero, narrow, tiny, NODE_TOP_K, TREE_DEPTH, CHAIN
    )
    (
        go,
        next_width,
        next_start,
        next_work,
        sampled_seen_c,
        sampled_next_c,
        last_depth,
        stop_width,
        cumu_count,
        verify_count,
        verify_num_state,
        flat,
    ) = _commit(
        zero, width, cost_stop, cur_width, cur_start, tiny, tier, sampled, sampled_plan,
        sampled_seen, sampled_next, past_verify_num, state_hist_max, zero, zero, zero,
        zero, zero, zero, tier_table_ptr, verify_num, min_verify_count, NODE_TOP_K,
        TREE_DEPTH, CHAIN,
    )  # fmt: skip
    sampled_seen = tl.where(normal, sampled_seen_c, sampled_seen)
    sampled_next = tl.where(normal, sampled_next_c, sampled_next)
    go = normal & go
    next_width = tl.where(normal, next_width, 0)
    next_start = tl.where(normal, next_start, 0)
    next_work = tl.where(normal, next_work, 0)
    # A width-W table round: depths 1..W-2 draft, all W nodes verify.
    table_last = table_width - 2
    last_depth = tl.where(normal, last_depth, tl.where(table_round, table_last, 0))
    stop_width = tl.where(
        normal,
        stop_width,
        tl.where(table_round, tl.where(table_last == TREE_DEPTH - 1, -1, 1), 0),
    )
    cumu_count = tl.where(normal, cumu_count, tl.where(table_round, table_width, 0))
    table_verify = tl.minimum(table_width, verify_num)
    verify_count = tl.where(
        normal, verify_count, tl.where(table_round, table_verify, 0)
    )
    verify_num_state = tl.where(
        normal, verify_num_state, tl.where(table_round, table_verify, 0)
    )
    flat = tl.where(normal, flat, 0)
    table_width = tl.where(table_round, table_width, 0)

    # Go flags: depth 1's from the decision (none for a table round), a table
    # round's through its last drafted depth + 1, at least depth 2.
    _publish_go(go_host_ptr, 1, seq, go, ~table_round)
    d = tl.arange(0, DEPTH_PAD)
    table_go = (d >= 1) & (d <= tl.maximum(table_last + 1, 2)) & (d <= TREE_DEPTH)
    _publish_go(go_host_ptr, d, seq, d <= table_last, table_round & table_go)

    # The round's cost row, for its decisions (a table round takes none).
    row = cost_id.to(tl.int64) * num_intervals + interval
    c = tl.arange(0, COPY_PAD)
    if CHAIN:
        count = TREE_DEPTH * 5
        src = cost_f64_ptr + (row * 2 + variant) * count + c
    else:
        count = _ROUND_THRESHOLDS + TREE_DEPTH
        f64_row = cost_f64_ptr + row * cost_cols
        src = f64_row + tl.where(
            c < _ROUND_THRESHOLDS,
            c,
            _ROUND_THRESHOLDS + variant * TREE_DEPTH + c - _ROUND_THRESHOLDS,
        )
    copy = normal & (c < count)
    tl.store(round_f64_ptr + c, tl.load(src, mask=copy), mask=copy)

    tl.store(seq_ptr, seq)
    tl.store(state_ptr + _SAMPLED_NEXT, sampled_next)
    tl.store(state_ptr + _SAMPLED_PLAN, sampled_plan)
    tl.store(state_ptr + _SAMPLED_SEEN, sampled_seen)
    tl.store(state_ptr + _BATCH_SIZE, bs)
    tl.store(state_ptr + _SAMPLED, sampled)
    tl.store(state_ptr + _COST_ID, cost_id)
    tl.store(state_ptr + _INTERVAL, interval)
    tl.store(state_ptr + _VARIANT, variant.to(tl.int32))
    tl.store(state_ptr + _HIST_MAX, state_hist_max.to(tl.int32))
    tl.store(state_ptr + _PAST_VERIFY_NUM, past_verify_num.to(tl.int32))
    tl.store(state_ptr + _LAST_DEPTH, last_depth.to(tl.int32))
    tl.store(state_ptr + _STOP_WIDTH, stop_width.to(tl.int32))
    tl.store(state_ptr + _CUMU_COUNT, cumu_count.to(tl.int32))
    tl.store(state_ptr + _VERIFY_COUNT, verify_count.to(tl.int32))
    tl.store(state_ptr + _FLAT, flat.to(tl.int32))
    tl.store(state_ptr + _VERIFY_NUM, verify_num_state.to(tl.int32))
    tl.store(state_ptr + _TABLE_WIDTH, table_width.to(tl.int32))

    # Depths 0 and 1 from the round, the others cleared; a table round's every depth
    # (depth 1 always runs, so a stop at depth 0 keeps an unranked depth 1).
    slots = d
    runs = d <= tl.maximum(table_last, 1)
    depth_slots: tl.constexpr = TREE_DEPTH + 1
    in_depth = d < depth_slots
    first = d == 0
    second = d == 1
    width_row = tl.where(
        table_round,
        runs.to(tl.int32),
        tl.where(first, cur_width, tl.where(second, next_width, 0)),
    )
    start_row = tl.where(
        table_round,
        tl.where(runs, d + 1, 0),
        tl.where(first, cur_start, tl.where(second, next_start, 0)),
    )
    work_row = tl.where(
        table_round, 0, tl.where(first, cur_work, tl.where(second, next_work, 0))
    )
    go_row = tl.where(
        table_round,
        (first | (d <= table_last)).to(tl.int32),
        tl.where(first, cur_go, tl.where(second, go.to(tl.int32), 0)),
    )
    tl.store(slots_ptr + slots, width_row.to(tl.int32), mask=in_depth)
    tl.store(slots_ptr + depth_slots + slots, start_row.to(tl.int32), mask=in_depth)
    tl.store(slots_ptr + 2 * depth_slots + slots, work_row.to(tl.int32), mask=in_depth)
    tl.store(slots_ptr + 3 * depth_slots + slots, go_row.to(tl.int32), mask=in_depth)


@triton.jit(
    do_not_specialize=[
        "depth",
        "exec_mode",
        "async_layer_ratio_bits",
        "verify_num",
        "min_verify_count",
    ]
)
def _policy_decide_kernel(
    state_ptr,
    seq_ptr,
    slots_ptr,
    go_host_ptr,
    work_deltas_ptr,
    round_f64_ptr,
    tier_table_ptr,
    depth,
    exec_mode,
    async_layer_ratio_bits,
    verify_num,
    min_verify_count,
    NODE_TOP_K: tl.constexpr,
    TREE_DEPTH: tl.constexpr,
    CHAIN: tl.constexpr,
):
    """Depth ``depth``'s decision of a tree or a chain round, before depth
    ``depth`` drafts: depth d + 1's slots, or the round's stop."""
    d = depth
    depth_slots: tl.constexpr = TREE_DEPTH + 1
    tiny = tl.load(state_ptr + _TINY)
    tier = tl.load(state_ptr + _TIER)
    sampled_next = tl.load(state_ptr + _SAMPLED_NEXT)
    sampled_plan = tl.load(state_ptr + _SAMPLED_PLAN)
    sampled_seen = tl.load(state_ptr + _SAMPLED_SEEN)
    bench_batch_size = tl.load(state_ptr + _BENCH_BATCH_SIZE)
    bs = tl.load(state_ptr + _BATCH_SIZE)
    sampled = tl.load(state_ptr + _SAMPLED)
    hist_max = tl.load(state_ptr + _HIST_MAX)
    past_verify_num = tl.load(state_ptr + _PAST_VERIFY_NUM)
    last_depth = tl.load(state_ptr + _LAST_DEPTH)
    stop_width = tl.load(state_ptr + _STOP_WIDTH)
    cumu_count = tl.load(state_ptr + _CUMU_COUNT)
    verify_count = tl.load(state_ptr + _VERIFY_COUNT)
    flat = tl.load(state_ptr + _FLAT)
    verify_num_state = tl.load(state_ptr + _VERIFY_NUM)
    table_width = tl.load(state_ptr + _TABLE_WIDTH)
    cur_width = tl.load(slots_ptr + d)
    cur_start = tl.load(slots_ptr + depth_slots + d)
    cur_go = tl.load(slots_ptr + 3 * depth_slots + d)
    seq = tl.load(seq_ptr)
    live = bs > 0
    # A (chain) width table round: begin_round wrote the slots; past its last
    # depth, only clear the next go flag.
    table_round = table_width > 0
    if CHAIN:
        top_k = 1
        narrow = 1
    else:
        top_k = NODE_TOP_K
        narrow = tl.load(tier_table_ptr + _VARIANTS + tier)
    running = live & ~table_round
    drafting = cur_go == 1

    decided, width, cost_stop = _early_decision(
        d, narrow, tiny, top_k, TREE_DEPTH, CHAIN
    )
    delta_0 = tl.load(work_deltas_ptr).to(tl.float64)
    if CHAIN:
        # raw_gain * 2.0 / (1.0 + depth * depth_loss), against the step's cost.
        terms = round_f64_ptr + d * 5
        raw_gain = tl.load(work_deltas_ptr + 1).to(tl.float64) - delta_0
        doubled = raw_gain * 2.0
        curr_gain = doubled / tl.load(terms + _GAIN_DENOM)
        threshold = tl.load(terms + _CHAIN_THRESHOLD)
        stop = curr_gain / tl.load(terms + _STEP_COST) < threshold
        cliff_cost = tl.load(terms + _CLIFF_COST)
        # A verify cost cliff: look two depths ahead.
        two_steps = curr_gain + doubled / tl.load(terms + _CLIFF_GAIN_DENOM)
        cliff_stop = two_steps / cliff_cost < threshold
        stop = tl.where(stop & (cliff_cost != 0.0), cliff_stop, stop)
        cost_stop = tl.where(decided, cost_stop, stop)
        width = tl.where(decided, width, 1)
    else:
        cut = _batch_cut(bench_batch_size, exec_mode, bs)
        compares = ~decided & ~cut
        tiny = tl.where(running & drafting & compares, (bs == 1).to(tl.int32), tiny)
        # (work_deltas[narrow] - work_deltas[0]) / draft_cost * ratios, one rounding per op.
        gain = tl.load(work_deltas_ptr + narrow).to(tl.float64) - delta_0
        gain = gain / tl.load(round_f64_ptr + _ROUND_DRAFT_COST)
        gain = gain * tl.load(round_f64_ptr + _ROUND_MODEL_COST_RATIO)
        gain = gain * _f64_arg(async_layer_ratio_bits)
        stop = gain < tl.load(round_f64_ptr + _ROUND_THRESHOLDS + d)
        cost_stop = tl.where(decided, cost_stop, tl.where(cut, True, stop))
        width = tl.where(decided, width, tl.where(cut, 1, narrow))
    (
        go,
        next_width,
        next_start,
        next_work,
        sampled_seen,
        sampled_next,
        last_depth,
        stop_width,
        cumu_count,
        verify_count,
        verify_num_state,
        flat,
    ) = _commit(
        d, width, cost_stop, cur_width, cur_start, tiny, tier, sampled, sampled_plan,
        sampled_seen, sampled_next, past_verify_num, hist_max, last_depth, stop_width,
        cumu_count, verify_count, verify_num_state, flat, tier_table_ptr, verify_num,
        min_verify_count, top_k, TREE_DEPTH, CHAIN,
    )  # fmt: skip
    decides = running & drafting
    go = decides & go
    # A round past its last drafting depth only clears the next go flag.
    _publish_go(go_host_ptr, d + 1, seq, go, live & (running | ~drafting))
    write = decides
    tl.store(state_ptr + _TINY, tiny, mask=write)
    tl.store(state_ptr + _SAMPLED_NEXT, sampled_next, mask=write)
    tl.store(state_ptr + _SAMPLED_SEEN, sampled_seen, mask=write)
    tl.store(state_ptr + _LAST_DEPTH, last_depth.to(tl.int32), mask=write)
    tl.store(state_ptr + _STOP_WIDTH, stop_width.to(tl.int32), mask=write)
    tl.store(state_ptr + _CUMU_COUNT, cumu_count.to(tl.int32), mask=write)
    tl.store(state_ptr + _VERIFY_COUNT, verify_count.to(tl.int32), mask=write)
    tl.store(state_ptr + _FLAT, flat.to(tl.int32), mask=write)
    tl.store(state_ptr + _VERIFY_NUM, verify_num_state.to(tl.int32), mask=write)
    next_slot = slots_ptr + d + 1
    tl.store(next_slot, tl.where(write, next_width, 0).to(tl.int32), mask=running)
    tl.store(
        next_slot + depth_slots,
        tl.where(write, next_start, 0).to(tl.int32),
        mask=running,
    )
    tl.store(
        next_slot + 2 * depth_slots,
        tl.where(write, next_work, 0).to(tl.int32),
        mask=running,
    )
    tl.store(next_slot + 3 * depth_slots, go.to(tl.int32), mask=running)


@triton.jit
def _take_i32(values, offs, index):
    """values[index] of an int vector, exactly."""
    return tl.sum(tl.where(offs == index, values, 0))


@triton.jit
def _take_f32(values, offs, index):
    """values[index] of an fp32 vector, bit for bit (through its integer bits)."""
    bits = values.to(tl.int32, bitcast=True)
    return tl.sum(tl.where(offs == index, bits, 0)).to(tl.float32, bitcast=True)


@triton.jit
def _shift_in(values, offs, n, value):
    """values[:n] shifted left by one, ``value`` last."""
    shifted = tl.gather(values, tl.minimum(offs + 1, values.shape[0] - 1), 0)
    return tl.where(offs < n - 1, shifted, tl.where(offs == n - 1, value, values))


@triton.jit
def _scan_item(top_ptr, top_stride, means, t, i, rows, cols, flat):
    """rocPRIM's blocked item i of thread t: entry t * 16 + i of the scanned
    scores (zero past them)."""
    e = t * _SCAN_ITEMS + i
    n = tl.where(flat, rows * cols, cols)
    ok = e < n
    row = e // cols
    col = e % cols
    flat_value = tl.load(
        top_ptr + row.to(tl.int64) * top_stride + col, mask=ok & flat, other=0.0
    )
    mean_value = tl.gather(means, tl.where(e < means.shape[0], e, 0), 0)
    return tl.where(ok, tl.where(flat, flat_value, mean_value), 0.0)


@triton.jit(
    do_not_specialize=[
        "recent_len",
        "hist_len",
        "published_capacity",
        "top_stride",
        "num_intervals",
        "i32_cols",
        "f64_cols",
        "cost_delta_col",
        "max_widths",
        "exec_mode",
    ]
)
def _policy_select_verify_kernel(
    state_ptr,
    recent_ptr,
    hist_ptr,
    published_ptr,
    top_ptr,
    interval_i32_ptr,
    interval_f64_ptr,
    verify_i32_ptr,
    verify_f64_ptr,
    recent_len,
    hist_len,
    published_capacity,
    top_stride,
    num_intervals,
    i32_cols,
    f64_cols,
    cost_delta_col,
    max_widths,
    exec_mode,
):
    """Publishes this round's cumulative ranked scores, then sizes the verify
    on the previous round's (the first round: its own) and books the width
    into the verify history."""
    bs = tl.load(state_ptr + _BATCH_SIZE)
    if bs > 0:
        bench_batch_size = tl.load(state_ptr + _BENCH_BATCH_SIZE)
        previous_count = tl.load(state_ptr + _RANKED_COUNT)
        sampled = tl.load(state_ptr + _SAMPLED)
        cost_id = tl.load(state_ptr + _COST_ID)
        interval = tl.load(state_ptr + _INTERVAL)
        verify_count = tl.load(state_ptr + _VERIFY_COUNT)
        flat = tl.load(state_ptr + _FLAT) != 0
        rows = bs
        cols = verify_count
        own_count = tl.where(flat, rows * cols, cols)

        r = tl.arange(0, 32)
        recent = tl.load(recent_ptr + r, mask=r < recent_len, other=0)
        c = tl.arange(0, 64)
        previous = tl.load(
            published_ptr + c, mask=c < tl.minimum(published_capacity, 64), other=0.0
        )
        history_row = hist_ptr + (bs - 1).to(tl.int64) * hist_len
        history = tl.load(history_row + r, mask=r < hist_len, other=0)
        verify_row = (bs - 1).to(tl.int64) * num_intervals + interval
        interval_row = cost_id.to(tl.int64) * num_intervals + interval
        widths_row = interval_i32_ptr + interval_row * i32_cols
        deltas_row = interval_f64_ptr + interval_row * f64_cols + cost_delta_col
        index = tl.load(widths_row + _WIDTHS_COL + r, mask=r < max_widths, other=0)
        tier = tl.load(verify_i32_ptr + verify_row)
        num_widths = tl.load(widths_row)
        threshold_stable = tl.load(verify_f64_ptr + verify_row * 2)
        threshold_unstable = tl.load(verify_f64_ptr + verify_row * 2 + 1)

        # Column means: torch's mean(dim=0) order at rows <= 32, cols <= 64.
        factor = tl.math.div_rn(cols.to(tl.float32), (rows * cols).to(tl.float32))
        col_ok = c < cols
        acc0 = tl.zeros([64], dtype=tl.float32)
        acc1 = tl.zeros([64], dtype=tl.float32)
        acc2 = tl.zeros([64], dtype=tl.float32)
        acc3 = tl.zeros([64], dtype=tl.float32)
        if ~flat:
            for k in range(0, rows):
                x = tl.load(top_ptr + k * top_stride + c, mask=col_ok, other=0.0)
                acc0 = tl.where(k % 4 == 0, acc0 + x, acc0)
                acc1 = tl.where(k % 4 == 1, acc1 + x, acc1)
                acc2 = tl.where(k % 4 == 2, acc2 + x, acc2)
                acc3 = tl.where(k % 4 == 3, acc3 + x, acc3)
        column_total = ((acc0 + acc1) + acc2) + acc3
        # One column: a pairwise tree over the last power of two <= rows lanes.
        width = rows * 0 + 1
        while width * 2 <= rows:
            width *= 2
        lane = tl.arange(0, 32)
        first = tl.load(
            top_ptr + lane * top_stride, mask=(lane < width) & ~flat, other=0.0
        )
        second = tl.load(
            top_ptr + (lane + width) * top_stride,
            mask=(lane < width) & (lane + width < rows) & ~flat,
            other=0.0,
        )
        lanes = tl.where(lane < width, first + second, 0.0)
        for _ in tl.static_range(5):
            lanes = tl.sum(tl.reshape(lanes, (lanes.shape[0] // 2, 2)), axis=1)
        single_total = tl.reshape(lanes, ())
        means = tl.where(cols == 1, tl.where(c == 0, single_total, 0.0), column_total)
        means = tl.where(col_ok, means * factor, 0.0)

        # rocPRIM's cumsum: thread reduce, 64-lane warp scan, warp prefix, thread scan.
        t = tl.arange(0, _SCAN_THREADS)
        partial = _scan_item(top_ptr, top_stride, means, t, 0, rows, cols, flat)
        for i in tl.static_range(1, _SCAN_ITEMS):
            partial = partial + _scan_item(
                top_ptr, top_stride, means, t, i, rows, cols, flat
            )
        row_lane = t % 16
        for shift_log in tl.static_range(4):
            shift = 1 << shift_log
            shifted = tl.gather(partial, tl.maximum(t - shift, 0), 0)
            partial = tl.where(row_lane >= shift, shifted + partial, partial)
        row15 = tl.gather(partial, tl.maximum(t - row_lane - 1, 0), 0)
        partial = tl.where(t % 32 >= 16, row15 + partial, partial)
        lane31 = tl.gather(partial, (t // 64) * 64 + 31, 0)
        partial = tl.where(t % 64 >= 32, lane31 + partial, partial)
        warp0 = _take_f32(partial, t, 63)
        before = tl.gather(partial, tl.maximum(t - 1, 0), 0)
        exclusive = tl.where(t < 64, before, tl.where(t == 64, warp0, warp0 + before))
        out = _scan_item(top_ptr, top_stride, means, t, 0, rows, cols, flat)
        out = tl.where(t > 0, exclusive + out, out)
        # Scores below 64 (the verify compares at most verify_count of them) stay in
        # registers for a first round.
        own = tl.gather(out, c // _SCAN_ITEMS, 0)
        tl.debug_barrier()  # the previous scores are read before they are overwritten
        tl.store(published_ptr + t * _SCAN_ITEMS, out, mask=t * _SCAN_ITEMS < own_count)
        for i in tl.static_range(1, _SCAN_ITEMS):
            out = out + _scan_item(top_ptr, top_stride, means, t, i, rows, cols, flat)
            e = t * _SCAN_ITEMS + i
            tl.store(published_ptr + e, out, mask=e < own_count)
            own = tl.where(
                c % _SCAN_ITEMS == i, tl.gather(out, c // _SCAN_ITEMS, 0), own
            )
        own = tl.where(c < own_count, own, 0.0)

        # The verify width.
        unranked = (previous_count <= 0) & (sampled != 0)
        ranked = tl.where(previous_count > 0, previous, own)
        ranked_len = tl.where(previous_count > 0, previous_count, own_count)
        # A sampled first round's own scores depend on its draws: the largest measured
        # width below the ranked count.
        best = tl.max(tl.where((r < num_widths) & (index < verify_count), index, -1))
        unranked_width = tl.where(best >= 0, best + 1, 1)
        # The majority of the recent widths (-1: none).
        same = (recent[None, :] == recent[:, None]) & (r[None, :] < recent_len)
        count = tl.sum(same.to(tl.int32), axis=1)
        majority = (r < recent_len) & (count > recent_len // 2)
        last_verify_num = tl.where(
            tl.max(majority.to(tl.int32)) > 0,
            tl.max(tl.where(majority, recent, _INT32_MIN)),
            -1,
        )
        cut = _batch_cut(bench_batch_size, exec_mode, bs)
        limit = tl.minimum(ranked_len, verify_count)
        keep = (r < num_widths) & (index < limit)
        largest = tl.max(tl.where(keep, index, 0))
        n = tl.sum(keep.to(tl.int32))
        position = tl.cumsum(keep.to(tl.int32), 0) - keep.to(tl.int32)
        widths = tl.sum(
            tl.where(
                keep[:, None] & (position[:, None] == r[None, :]), index[:, None], 0
            ),
            axis=0,
        )
        scores = tl.gather(ranked, tl.minimum(tl.maximum(widths, 0), 63), 0).to(
            tl.float64
        )
        # Width j's mask: reset to widths[1] whenever its gain per cost over an earlier
        # width i is below the stability threshold its current mask selects.
        reset = _take_i32(widths, r, 1)
        mask = widths
        for i in range(0, n):
            score_i = tl.sum(tl.where(r == i, scores, 0.0))
            delta = tl.load(deltas_row + i * max_widths + r, mask=r < n, other=0.0)
            threshold = tl.where(
                last_verify_num <= mask + 1, threshold_stable, threshold_unstable
            )
            below = (
                (r > i)
                & (r < n)
                & (delta > 0.0)
                & ((scores - score_i) / delta < threshold)
            )
            mask = tl.where(below, reset, mask)
        chosen = tl.max(tl.where(r < n, mask, _INT32_MIN)) + 1
        smallest = tl.min(tl.where((r >= 1) & (r < n), widths, _INT32_MAX))
        compared = tl.where(
            (chosen == 1) | (verify_count < chosen), smallest + 1, chosen
        )

        planned = _shift_in(recent, r, recent_len, verify_count)
        last = recent_len - 1
        planned_width = tl.where(
            cut, _take_i32(index, r, 1) + 1, tl.where(largest == 0, 1, compared)
        )
        planned_recent = tl.where(
            cut | (r != last), planned, tl.where(largest == 0, 1, chosen)
        )
        verify_num = tl.where(unranked, unranked_width, planned_width)
        recent_out = tl.where(
            unranked, _shift_in(recent, r, recent_len, unranked_width), planned_recent
        )

        tl.store(state_ptr + _TIER, tier)
        tl.store(state_ptr + _VERIFY_NUM, verify_num)
        tl.store(state_ptr + _RANKED_COUNT, own_count)
        tl.store(recent_ptr + r, recent_out, mask=r < recent_len)
        # The history shifted by one, the width last.
        tl.store(
            history_row + r,
            _shift_in(history, r, hist_len, verify_num),
            mask=r < hist_len,
        )
