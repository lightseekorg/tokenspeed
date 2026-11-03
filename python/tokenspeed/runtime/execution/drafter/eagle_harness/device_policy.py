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

"""EAGLE-Harness tree policy: how wide and how deep a round drafts, and how
many draft nodes it verifies, from the measured cost data. The decisions are
EAGLE-4's, bit for bit, taken by kernels that read and write
device memory only, so a round needs no host synchronization. A chain round
of a batch size the width table covers decides nothing: its width is fixed
(``begin_round`` writes the whole round).

Every value the cost data determines is computed here, on the host, with
EAGLE-4's expressions and constants in Python double precision, and
uploaded as tables; the kernels compare against them and evaluate only the
arithmetic on device inputs (work deltas, ranked scores), one IEEE rounding
per Python operation, in EAGLE-4's order.

Batch sizes fall in tiers by the target's verify cost at eight nodes relative
to one (``_cost_tier``): small (or batch size one), medium, large, huge; a
batch of one also counts as tiny once its first compared decision has run.
Chains (``chain_intensity``, one child per node) stop every other depth when
one more pair of draft steps gains too little accepted length per cost, and
verify every drafted node.

A round, on one stream (every launch graph-capturable, sizes on the device):

0. On the host, eagerly: ``next_round`` (the round count, and exec mode 1's
   benchmark batch size).
1. ``begin_round``: depth 0's decision, depth 1's slots (a width table
   round: every depth's slots and go flags and the round's counts, which its
   decisions leave as they are; its slots queue no work deltas).
2. Depth 0's work deltas when ``work_sizes[0]`` (node count 1).
3. For ``d = 1, 2, ...`` while ``go[d]``: ``decide(d, work_deltas)`` (its
   decision reads only depth ``d - 1``'s work deltas, so it runs before depth
   ``d`` drafts and leaves ``go[d + 1]`` one depth ahead), then draft depth
   ``d`` (``widths[d]`` lanes, children from ``cumu_starts[d]``) and queue
   its work deltas when ``work_sizes[d]`` (that node count). The work deltas
   of every depth take ``past_verify_num`` as their verify width. The
   decision that stops the round also writes ``verify_count`` and ``flat``
   and, in a sampled round, the next sampled round's planned stop depth.
4. Rank the pool's best ``verify_count`` of ``cumu_count`` nodes;
   ``select_verify`` on their scores (which size the next round's verify,
   averaged over the batch unless ``flat``); verify ``verify_num`` nodes.

``go_host`` mirrors ``go`` in mapped pinned memory, tagged with the round's
``begin_round`` count (``next_round``'s on the host), for polling without a
CUDA synchronization (``go_ready``). Cost data EAGLE-4's decisions would
reject in some round is refused when the tables are built instead.
MeasurePolicy (exec mode 2) decides on the host.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from tokenspeed_kernel.thirdparty.cuda.eagle_harness_policy import (
    eagle_harness_policy_begin_round,
    eagle_harness_policy_decide_chain,
    eagle_harness_policy_decide_tree,
    eagle_harness_policy_load,
    eagle_harness_policy_select_verify,
)

from tokenspeed.runtime.execution.drafter.eagle_harness.config import (
    ChainWidthTable,
    chain_table_width,
    first_rank_value,
    load_run_config,
)
from tokenspeed.runtime.execution.drafter.eagle_harness.cost import (
    BatchCost,
    IntervalCost,
    max_batch_size,
)

__all__ = ["DevicePolicy", "write_round_slots"]


# Scalars of the int32 state tensor (``State`` in eagle_harness_policy.cu).
(
    _TINY,
    _TIER,
    _SAMPLED_NEXT,
    _SAMPLED_PLAN,
    _SAMPLED_SEEN,
    _BENCH_BATCH_SIZE,
    _RANKED_COUNT,
    _BATCH_SIZE,
    _SAMPLED,
    _COST_ID,
    _INTERVAL,
    _VARIANT,
    _HIST_MAX,
    _PAST_VERIFY_NUM,
    _LAST_DEPTH,
    _STOP_WIDTH,
    _CUMU_COUNT,
    _VERIFY_COUNT,
    _FLAT,
    _VERIFY_NUM,
    _TABLE_WIDTH,
    _NUM_STATE,
) = range(22)
# Rows of the slot tensor (``SlotRow``).
_WIDTH_ROW, _CUMU_START_ROW, _WORK_SIZE_ROW, _GO_ROW, _NUM_SLOT_ROWS = range(5)
# Columns of the batch table (``BatchCol``).
(
    _COST_ID_COL,
    _CHAIN_TIER_ROW_COL,
    _INTERVALS_COL,
    _CACHE_STEP_COL,
    _TABLE_WIDTH_COL,
    _BATCH_COLS,
) = range(6)
# Columns of the interval tables (``IntervalI32Col`` / ``IntervalF64Col``).
_NUM_WIDTHS_COL, _WIDTHS_COL = range(2)
_DRAFT_COST_COL, _MODEL_COST_RATIO_COL, _STOP_THRESHOLD_COL = range(3)
# Entries of the chain table (``ChainF64``).
(
    _GAIN_DENOM,
    _STEP_COST,
    _CHAIN_THRESHOLD,
    _CLIFF_GAIN_DENOM,
    _CLIFF_COST,
    _CHAIN_F64,
) = range(6)
# Entries of a tree round's cost row copy (``RoundF64``), its thresholds last.
_ROUND_DRAFT_COST, _ROUND_MODEL_COST_RATIO, _ROUND_THRESHOLDS = range(3)

# Pass threshold of a draft step's gain per cost.
_THRESHOLD = 2.5
# Per-depth growth of the pass threshold for an EAGLE-4 draft (doubled for an MoE target).
_DEPTH_LOSS_RATIO = 0.04
_ASYNC_LAYER_RATIO = 0.9
# Rounds of verify widths a verify decision's hysteresis looks back on.
_RECENT_VERIFY_ROUNDS = 5
# Rounds of verify widths a batch size's draft budget looks back on.
_PAST_VERIFY_ROUNDS = 16
# Tree batch tiers (_cost_tier), then the cost variant of a tiny round.
_SMALL, _MEDIUM, _LARGE, _HUGE, _TINY_VARIANT = range(5)
_VARIANTS = 5
# Per cost variant: a draft budget's least starting verify width; the stop threshold's batch ratio.
_VERIFY_FLOOR = (16, 8, 4, 2, 32)
_BATCH_SIZE_RATIO = (1.25, 1.50, 1.75, 2.00, 1.00)
# Least ranked node budget of a tree round.
_MIN_VERIFY_COUNT = 16
# Verify width a chain round starts its work lists from; chains verify every drafted node.
_CHAIN_VERIFY_NUM = 16
# Chain batch tiers by scale index: how fast the target's verify cost grows with its width.
_CHAIN_SCALE_TINY = 1.25
_CHAIN_TINY_MAX_BS = 2
_CHAIN_SCALE_SMALL = 1.45
_CHAIN_SCALE_MEDIUM = 2.00
_CHAIN_COEFS = (
    (2.60, 0.24, 0.00),
    (2.85, 0.24, 0.00),
    (2.54, 0.24, 0.70),
    (1.55, 0.24, 0.00),
)
# The mean round cost the thresholds were tuned at, in one-token verify units.
_CHAIN_REF_COST = 2.43
# Cliff: a cost step above the floor and ratio x the next; the stop test then looks 2 depths ahead.
_CHAIN_CLIFF_MIN_COST = 0.15
_CHAIN_CLIFF_RATIO = 1.5
# A chain's measured verify widths 1, 2, 4, ..., 16 as width - 1, which the stop test indexes by step.
_CHAIN_WIDTH_INDEX = [0, *range(1, 16, 2)]
# The kernels' limit of measured target widths per interval.
_MAX_WIDTHS = 32
_INT32_MAX = 2**31 - 1


@dataclass(frozen=True)
class _Tables:
    """The cost data's device tables; the other policy kind's are empty."""

    # int64 [max_bs, 5]: cost id, tier row, intervals, step, chain width table width.
    batch_table: torch.Tensor
    interval_i32: torch.Tensor  # trees: int32 [costs, intervals, cols]: widths
    # Trees: float64 [costs, intervals, 2 + 5 * depth + widths**2].
    interval_f64: torch.Tensor
    # Trees: int32 [9], the verify floor per variant, then the width per tier.
    tier_table: torch.Tensor
    verify_i32: torch.Tensor  # trees: int32 [max_bs, intervals], tier
    verify_f64: torch.Tensor  # trees: float64 [max_bs, intervals, 2], thresholds
    chain_f64: torch.Tensor  # chains: float64 [costs, intervals, 2, depth, 5]


def _cost_state(target: IntervalCost) -> float:
    """The target's verify cost at the first measured width of 8 or more,
    relative to width 1."""
    for i, width in enumerate(target.index):
        if width >= 7:
            return target.ratio[i]
    return target.ratio[-1]


def _cost_tier(cost_state: float, batch_size: int) -> int:
    if cost_state <= 1.25 or batch_size == 1:
        return _SMALL
    if cost_state < 1.60:
        return _MEDIUM
    if cost_state < 2.20:
        return _LARGE
    return _HUGE


def _adjust_cost_ratio(model_cost_ratio: float) -> float:
    if model_cost_ratio < 8:
        return 1.30
    if model_cost_ratio < 12:
        return 1.10
    if model_cost_ratio < 17:
        return 1.30
    if model_cost_ratio < 23:
        return 1.50
    if model_cost_ratio < 32:
        return 1.80
    if model_cost_ratio < 64:
        return 2.20
    return 2.70


def _chain_tier(small_batch: bool, verify_ratio: list[float]) -> int:
    # Mean verify cost of widths 2, 6 and 14 over width one.
    scale = (verify_ratio[1] + verify_ratio[3] + verify_ratio[7]) / (
        3.0 * verify_ratio[0]
    )
    if scale <= _CHAIN_SCALE_TINY or small_batch:
        return 0
    if scale < _CHAIN_SCALE_SMALL:
        return 1
    if scale < _CHAIN_SCALE_MEDIUM:
        return 2
    return 3


def _tree_stop_threshold(
    depth: int,
    depth_loss_ratio: float,
    adjust_cost_ratio: float,
    batch_size_ratio: float,
) -> float:
    """The right side of a tree decision's stop test."""
    ratio = batch_size_ratio if depth > 1 else (batch_size_ratio + 1.00) / 2
    ratio = batch_size_ratio * 1.1 if depth > 5 and ratio > 1.3 else ratio
    return _THRESHOLD * (depth * depth_loss_ratio + 1.0) * adjust_cost_ratio * ratio


def _tree_interval_rows(
    cost: BatchCost,
    interval: int,
    max_widths: int,
    depth: int,
    depth_loss_ratio: float,
    where: str,
) -> tuple[list[int], list[float]]:
    """One (cost data, interval)'s int32 row (measured widths) and float64 row
    (draft_cost[-1], model cost ratio, stop thresholds per cost variant and
    depth, cost deltas); refuses a model cost ratio or draft cost
    EAGLE-4's decisions reject."""
    target, draft = cost.intervals[interval]
    model_cost_ratio = cost.model_ratio[interval]
    if model_cost_ratio < 5 or model_cost_ratio > 100:
        raise ValueError(
            f"{where}: model cost ratio {model_cost_ratio:.2f} outside [5, 100] "
            "(a draft too costly for speculation, or a broken measurement)"
        )
    if not draft.ratio or draft.ratio[-1] == 0:
        raise ValueError(f"{where}: no nonzero draft cost at the widest draft width")
    adjust_cost_ratio = _adjust_cost_ratio(model_cost_ratio)
    thresholds = [0.0] * _VARIANTS * depth
    for variant in range(_VARIANTS):
        batch_size_ratio = max(
            1.0, _BATCH_SIZE_RATIO[variant] / ((20 / model_cost_ratio + 1) / 2)
        )
        for d in range(depth):
            thresholds[variant * depth + d] = _tree_stop_threshold(
                d, depth_loss_ratio, adjust_cost_ratio, batch_size_ratio
            )
    # The verify width's cost deltas by position; 0.0 where cost_delta > 0.04 fails.
    cost_deltas = [0.0] * max_widths * max_widths
    for i in range(len(target.index)):
        for j in range(i + 1, len(target.index)):
            cost_delta = target.ratio[j] - target.ratio[i]
            if cost_delta > 0.04:
                cost_deltas[i * max_widths + j] = cost_delta
    i32_row = [0] * (_WIDTHS_COL + max_widths)
    i32_row[_NUM_WIDTHS_COL] = len(target.index)
    i32_row[_WIDTHS_COL : _WIDTHS_COL + len(target.index)] = target.index
    f64_row = [0.0] * _STOP_THRESHOLD_COL + thresholds + cost_deltas
    f64_row[_DRAFT_COST_COL] = draft.ratio[-1]
    f64_row[_MODEL_COST_RATIO_COL] = model_cost_ratio
    return i32_row, f64_row


def _verify_row(
    target: IntervalCost, batch_size: int, high: bool, moe: bool, where: str
) -> tuple[int, list[float]]:
    """The verify cost context at one batch size and interval: the tier and
    the gain-per-cost thresholds times 0.95 and 1.05, of a high (or low)
    intensity, MoE (or dense) target; refuses a cost state EAGLE-4
    rejects."""
    cost_state = _cost_state(target)
    tier = _cost_tier(cost_state, batch_size)
    if tier == _SMALL:
        threshold = 2.40 if high else 2.00
        if cost_state < 0.85 and (cost_state < 0.70 or not moe):
            raise ValueError(f"{where}: verify cost state {cost_state} too low")
    elif tier == _MEDIUM:
        threshold = 2.10 if high else 1.80
    elif tier == _LARGE:
        threshold = 1.80 if high else 1.60
    else:
        threshold = 1.50 if high else 1.40
        if cost_state > 4.0:
            raise ValueError(f"{where}: verify cost state {cost_state} too high")
    return tier, [threshold * 0.95, threshold * 1.05]


def _chain_terms(
    cost: BatchCost, interval: int, small_batch: bool, depth: int, where: str
) -> list[float]:
    """A chain's stop test at an even depth without the work deltas: gain
    denominator, step cost, threshold, and when a verify cost cliff makes it
    look two depths ahead, that gain denominator and cost (0.0 else); refuses
    terms EAGLE-4's stop test divides by zero on or turns into a complex
    threshold."""
    verify_ratio = cost.intervals[interval][0].ratio
    model_cost_ratio = cost.model_ratio[interval]
    cost_llm = verify_ratio[1:]
    terms = [0.0] * _CHAIN_F64
    zero_division = ValueError(
        f"{where}: the chain stop test at depth {depth} divides by zero"
    )
    if verify_ratio[0] == 0 or model_cost_ratio == 0:
        raise zero_division
    threshold, depth_loss, cost_power = _CHAIN_COEFS[
        _chain_tier(small_batch, verify_ratio)
    ]
    curr_step = depth // 2
    next_step = min(curr_step + 1, len(cost_llm) - 1)
    draft_step_cost = 1.0 / model_cost_ratio
    curr_cost = draft_step_cost * 2 + max(0, cost_llm[next_step] - cost_llm[curr_step])
    ref_iter_cost = 0.0
    for i, verify_cost in enumerate(cost_llm):
        ref_iter_cost += verify_cost + draft_step_cost * (2 * i + 1)
    ref_iter_cost /= len(cost_llm)
    if ref_iter_cost == 0 or curr_cost == 0:
        raise zero_division
    curr_threshold = threshold * (_CHAIN_REF_COST / ref_iter_cost) ** cost_power
    if isinstance(curr_threshold, complex):
        raise ValueError(f"{where}: complex chain stop threshold at depth {depth}")
    terms[_GAIN_DENOM] = 1.0 + depth * depth_loss
    terms[_STEP_COST] = curr_cost
    terms[_CHAIN_THRESHOLD] = curr_threshold
    if next_step + 1 < len(cost_llm):
        step_cost_next = cost_llm[next_step] - cost_llm[curr_step]
        step_cost_after = cost_llm[next_step + 1] - cost_llm[next_step]
        if step_cost_next > _CHAIN_CLIFF_MIN_COST and step_cost_next > (
            _CHAIN_CLIFF_RATIO * max(step_cost_after, 0.0)
        ):
            cost_two_steps = draft_step_cost * 4 + max(
                0, cost_llm[next_step + 1] - cost_llm[curr_step]
            )
            if cost_two_steps == 0:
                raise zero_division
            terms[_CLIFF_GAIN_DENOM] = 1.0 + (depth + 2) * depth_loss
            terms[_CLIFF_COST] = cost_two_steps
    return terms


def _check_cost_data(
    cost: BatchCost, where: str, chain: bool, exec_mode: int, verify_num: int
) -> None:
    """Refuse cost data the tables cannot hold or a round cannot use: no
    context interval, a cache_length_step that is not a nonzero integer, fewer
    ratios than measured widths, more than 32 widths, widths that are not
    strictly increasing integers in [0, 2**31 - 1); a chain's widths other
    than 1, 2, 4, ..., 16; a tree's widths without width 1, or (exec mode 1)
    without a second width within the verify budget ``verify_num``, the width
    its batch cut verifies."""
    if not cost.intervals:
        raise ValueError(f"{where}: no context interval")
    if not isinstance(cost.cache_length_step, int) or cost.cache_length_step == 0:
        raise ValueError(
            f"{where}: cache_length_step {cost.cache_length_step!r} is not a "
            "nonzero integer"
        )
    for interval, (target, _) in enumerate(cost.intervals):
        index = target.index
        if (
            len(target.ratio) < len(index)
            or len(index) > _MAX_WIDTHS
            or not all(isinstance(w, int) and 0 <= w < _INT32_MAX for w in index)
            or any(b <= a for a, b in zip(index, index[1:]))
        ):
            raise ValueError(
                f"{where} interval {interval}: the device policy needs a ratio per "
                f"measured width and at most {_MAX_WIDTHS} strictly increasing "
                f"widths in [0, 2**31 - 1), got {index}"
            )
        if chain and list(index) != _CHAIN_WIDTH_INDEX:
            raise ValueError(
                f"{where} interval {interval}: chain cost data needs widths 1, 2, "
                f"4, ..., 16 (index {_CHAIN_WIDTH_INDEX}), got {index}"
            )
        if not chain and (
            0 not in index
            or (exec_mode == 1 and (len(index) < 2 or index[1] >= verify_num))
        ):
            raise ValueError(
                f"{where} interval {interval}: the verify cost needs width 1 "
                f"(index 0), and exec mode 1 a second width within the verify "
                f"budget {verify_num}, got {index}"
            )


def _tree_tables(
    policy: DevicePolicy,
    costs: list[BatchCost],
    names: list[str],
    cost_ids: list[int],
) -> dict[str, torch.Tensor]:
    """A tree's interval, verify and tier tables (``cost_ids``: per batch size)."""
    depth, max_bs = policy.tree_depth, len(cost_ids)
    max_widths = max(len(t.index) for c in costs for t, _ in c.intervals)
    num_intervals = max(len(cost.intervals) for cost in costs)
    interval_i32 = torch.zeros(
        (len(costs), num_intervals, _WIDTHS_COL + max_widths), dtype=torch.int32
    )
    f64_cols = _STOP_THRESHOLD_COL + _VARIANTS * depth + max_widths**2
    interval_f64 = torch.zeros(
        (len(costs), num_intervals, f64_cols), dtype=torch.float64
    )
    for cost_id, cost in enumerate(costs):
        for interval in range(len(cost.intervals)):
            i32_row, f64_row = _tree_interval_rows(
                cost,
                interval,
                max_widths,
                depth,
                policy.depth_loss_ratio,
                f"{names[cost_id]} interval {interval}",
            )
            interval_i32[cost_id, interval] = torch.tensor(i32_row)
            interval_f64[cost_id, interval] = torch.tensor(f64_row, dtype=torch.float64)
    verify_i32 = torch.zeros((max_bs, num_intervals), dtype=torch.int32)
    verify_f64 = torch.zeros((max_bs, num_intervals, 2), dtype=torch.float64)
    for bs in range(1, max_bs + 1):
        cost_id = cost_ids[bs - 1]
        for interval, (target, _) in enumerate(costs[cost_id].intervals):
            tier, thresholds = _verify_row(
                target,
                bs,
                policy.high_intensity,
                policy.is_moe_target,
                f"{names[cost_id]} interval {interval} at batch size {bs}",
            )
            verify_i32[bs - 1, interval] = tier
            verify_f64[bs - 1, interval] = torch.tensor(thresholds, dtype=torch.float64)
    # A decision's width per tier.
    widths = [min(policy.node_top_k, 8 if tier < _LARGE else 4) for tier in range(4)]
    return {
        "interval_i32": interval_i32,
        "interval_f64": interval_f64,
        "tier_table": torch.tensor(list(_VERIFY_FLOOR) + widths, dtype=torch.int32),
        "verify_i32": verify_i32,
        "verify_f64": verify_f64,
    }


def _chain_tables(
    costs: list[BatchCost], names: list[str], depth: int
) -> dict[str, torch.Tensor]:
    """A chain's stop terms per (cost data, interval, tier row, depth)."""
    num_intervals = max(len(cost.intervals) for cost in costs)
    chain_f64 = torch.zeros(
        (len(costs), num_intervals, 2, depth, _CHAIN_F64), dtype=torch.float64
    )
    for cost_id, cost in enumerate(costs):
        for interval in range(len(cost.intervals)):
            for small_batch in (0, 1):
                for d in range(2, depth - 1, 2):
                    terms = _chain_terms(
                        cost,
                        interval,
                        bool(small_batch),
                        d,
                        f"{names[cost_id]} interval {interval}",
                    )
                    chain_f64[cost_id, interval, small_batch, d] = torch.tensor(
                        terms, dtype=torch.float64
                    )
    return {"chain_f64": chain_f64}


def _check_chain_width_table(policy: DevicePolicy) -> None:
    """Refuse a width table a round cannot run: on a tree, or a width outside
    2 (one draft token) to the chain's full width."""
    table = policy.chain_width_table
    if table and not policy.chain:
        raise ValueError("EAGLE-Harness trees run no chain width table")
    full_width = min(policy.verify_num_limit, policy.tree_depth + 1)
    for entry in table:
        if not 2 <= entry.width <= full_width:
            raise ValueError(
                f"EAGLE-Harness chain width table width {entry.width} (from "
                f"{entry.min_bs} requests) outside 2..{full_width}"
            )


def _build_tables(
    policy: DevicePolicy, cost_data: dict[int, BatchCost], max_bs: int
) -> _Tables:
    """The cost data's tables on the policy's device. Each batch size reads
    its nearest measured batch's cost data (the larger on a tie; ``cost
    id``); tables per interval pad to the most intervals. A chain's batch
    sizes also hold the width table's width (0 where the confidence rule
    decides).

    Raises:
        ValueError: The cost data is missing, or one a batch size reads is
            one the tables cannot hold or a round cannot use; or the width
            table is one a round cannot run.
    """
    _check_chain_width_table(policy)
    if not cost_data:
        raise ValueError("EAGLE-Harness needs cost data")
    nearest = [
        min(cost_data, key=lambda m: (abs(m - bs), -m)) for bs in range(1, max_bs + 1)
    ]
    # Each measured batch size's cost id, in order of first use.
    ids = {bs: i for i, bs in enumerate(dict.fromkeys(nearest))}
    costs = [cost_data[bs] for bs in ids]
    names = [f"EAGLE-Harness cost data bs_{bs}" for bs in ids]
    for cost, name in zip(costs, names):
        _check_cost_data(
            cost, name, policy.chain, policy.exec_mode, policy.verify_num_limit
        )
    cost_ids = [ids[bs] for bs in nearest]
    batch_table = torch.zeros((max_bs, _BATCH_COLS), dtype=torch.int64)
    for bs in range(1, max_bs + 1):
        cost_id = cost_ids[bs - 1]
        cost = costs[cost_id]
        row = batch_table[bs - 1]
        row[_COST_ID_COL] = cost_id
        row[_CHAIN_TIER_ROW_COL] = int(bs <= _CHAIN_TINY_MAX_BS)
        row[_INTERVALS_COL] = len(cost.intervals)
        row[_CACHE_STEP_COL] = cost.cache_length_step
        row[_TABLE_WIDTH_COL] = chain_table_width(policy.chain_width_table, bs) or 0
    tables = {
        "batch_table": batch_table,
        "interval_i32": torch.zeros(0, dtype=torch.int32),
        "interval_f64": torch.zeros(0, dtype=torch.float64),
        "tier_table": torch.zeros(0, dtype=torch.int32),
        "verify_i32": torch.zeros(0, dtype=torch.int32),
        "verify_f64": torch.zeros(0, dtype=torch.float64),
        "chain_f64": torch.zeros(0, dtype=torch.float64),
    }
    if policy.chain:
        tables.update(_chain_tables(costs, names, policy.tree_depth))
    else:
        tables.update(_tree_tables(policy, costs, names, cost_ids))
    return _Tables(**{name: t.to(policy.device) for name, t in tables.items()})


def write_round_slots(
    slots: torch.Tensor,
    node_top_k: int,
    widths: list[int],
    work_size: Callable[[int, int], int],
) -> int:
    """Write the ``[4, tree_depth + 1]`` device ``slots`` of a round drafting
    depth ``d >= 1`` at ``widths[d - 1]`` with ``work_size(d, cumu_start)``
    work deltas; returns its pool size."""
    host = torch.zeros(slots.shape, dtype=torch.int32)
    start = 1 + node_top_k
    for depth, width in enumerate(widths, start=1):
        host[:, depth] = torch.tensor([width, start, work_size(depth, start), 1])
        start += node_top_k * width
    slots.copy_(host)
    return start


class DevicePolicy:
    """The round decisions of one EAGLE-Harness drafter (exec mode 0 or 1).

    Args:
        cost_data: Measured cost per batch size (``load_cost_data``).
        verify_num: Most draft nodes a round verifies
            (``--speculative-num-draft-tokens``).
        node_top_k: Children per expanded node (``--speculative-eagle-topk``).
        tree_depth: Most draft depths a round runs
            (``--speculative-num-steps``).
        compute_intensity: ``high_intensity`` or ``low_intensity`` (trees),
            or ``chain_intensity`` (chains, one child per node).
        is_moe_target: The target has routed experts (doubles the per-depth
            growth of a tree's pass threshold and lowers the smallest
            accepted verify cost).
        exec_mode: 0 (serving) or 1 (benchmarking: a batch far below the
            benchmark's batch size, read from ``benchmark_config``, drafts
            only the root's children and verifies narrowly).
        benchmark_config: ``<data path>/config/benchmark_config.json``,
            ``{"batch_size": N}`` (read once, when exec mode 1 first needs it).
        file_group: The TP ranks' CPU group of their own, over which the
            first rank's read of ``benchmark_config`` reaches every rank
            (None: this rank reads it).
        chain_width_table: A chain's verify width per live batch size
            (``--speculative-chain-width-table``; empty: the confidence rule
            decides every round). A round it covers drafts depths 1 to
            ``width - 2`` and verifies ``width`` nodes, deciding nothing.
        device: The CUDA device the rounds run on.

    Raises:
        ValueError: No cost data, or cost data the tables cannot hold or that
            EAGLE-4's decisions reject in some round, or a width
            table a round cannot run (``_build_tables``).

    Per-round outputs (device int32, valid once the step writing them ran):
    ``past_verify_num``; per depth ``widths``, ``cumu_starts``,
    ``work_sizes``, ``go`` (``[tree_depth + 1]``); ``cumu_count`` (pool size
    after the round), ``verify_count``, ``verify_num``.
    """

    def __init__(
        self,
        cost_data: dict[int, BatchCost],
        *,
        verify_num: int,
        node_top_k: int,
        tree_depth: int,
        compute_intensity: str,
        is_moe_target: bool,
        exec_mode: int,
        benchmark_config: str,
        file_group: dist.ProcessGroup | None,
        chain_width_table: ChainWidthTable,
        device: torch.device,
    ) -> None:
        self.chain = compute_intensity == "chain_intensity"
        self.node_top_k = node_top_k
        self.tree_depth = tree_depth
        self.verify_num_limit = verify_num
        self.high_intensity = compute_intensity == "high_intensity"
        self.is_moe_target = is_moe_target
        self.depth_loss_ratio = _DEPTH_LOSS_RATIO * (2.0 if is_moe_target else 1.0)
        self.exec_mode = exec_mode
        self.benchmark_config = benchmark_config
        self.file_group = file_group
        self.chain_width_table = chain_width_table
        self.device = device
        # Exec mode 1: next_round reads the benchmark's batch size until it loads.
        self._benchmark_pending = exec_mode == 1
        # Rounds begun (next_round), which tags the go mirror's flags.
        self._rounds = 0
        max_bs = max_batch_size(compute_intensity)
        self._tables = _build_tables(self, cost_data, max_bs)
        # Loaded here: the first launch may be inside a graph capture.
        eagle_harness_policy_load()

        state = [0] * _NUM_STATE
        state[_SAMPLED_SEEN] = -1
        # The first sampled round plans the deepest stop.
        state[_SAMPLED_NEXT] = self.tree_depth - 1
        if self.chain:
            # Chains keep no verify history.
            hist = torch.zeros((max_bs, 0), dtype=torch.int32)
            recent = torch.zeros((0,), dtype=torch.int32)
        else:
            # Trees start at tier small (zero) with a full-width verify history.
            hist = torch.full((max_bs, _PAST_VERIFY_ROUNDS), verify_num)
            recent = torch.arange(-_RECENT_VERIFY_ROUNDS, 0)
        self._state = torch.tensor(state, dtype=torch.int32, device=device)
        self._hist = hist.to(device=device, dtype=torch.int32)
        self._recent = recent.to(device=device, dtype=torch.int32)
        # The ranked scores the next round sizes its verify on (none before the first).
        self._ranked = torch.zeros(
            (max_bs * verify_num,), dtype=torch.float32, device=device
        )
        self._slots = torch.zeros(
            (_NUM_SLOT_ROWS, self.tree_depth + 1), dtype=torch.int32, device=device
        )
        # The round's cost row, copied by begin_round for its decisions.
        depth = self.tree_depth
        self._round_f64 = torch.zeros(
            (_CHAIN_F64 * depth if self.chain else _ROUND_THRESHOLDS + depth,),
            dtype=torch.float64,
            device=device,
        )
        # begin_round's count on the device (_rounds on the host).
        self._device_rounds = torch.zeros((1,), dtype=torch.int64, device=device)
        self.go_host = torch.zeros(
            (self.tree_depth + 1,), dtype=torch.int64, pin_memory=True
        )
        self._go_host_view = self.go_host.numpy()

        self.past_verify_num = self._state[_PAST_VERIFY_NUM : _PAST_VERIFY_NUM + 1]
        self.widths = self._slots[_WIDTH_ROW]
        self.cumu_starts = self._slots[_CUMU_START_ROW]
        self.work_sizes = self._slots[_WORK_SIZE_ROW]
        self.go = self._slots[_GO_ROW]
        self.cumu_count = self._state[_CUMU_COUNT : _CUMU_COUNT + 1]
        self.verify_count = self._state[_VERIFY_COUNT : _VERIFY_COUNT + 1]
        self.verify_num = self._state[_VERIFY_NUM : _VERIFY_NUM + 1]

    def next_round(self, batch_size: int) -> None:
        """On the host before each round (its ``begin_round``) of
        ``batch_size`` requests (0: none): counts it, and in exec mode 1 reads
        the benchmark's batch size at its first round past one request, which
        a batch cut may read. A load is enqueued on the current stream, the
        rounds' stream.

        Raises:
            FileNotFoundError: Exec mode 1's benchmark config is missing then.
        """
        if self._benchmark_pending and batch_size > 1:
            benchmark_batch_size = first_rank_value(
                self._benchmark_batch_size, self.file_group
            )
            self._state[_BENCH_BATCH_SIZE].fill_(benchmark_batch_size)
            self._benchmark_pending = False
        self._rounds += 1

    def _benchmark_batch_size(self) -> int:
        if not os.path.exists(self.benchmark_config):
            raise FileNotFoundError(
                "EAGLE-Harness exec mode 1 reads the benchmark's batch size from "
                f'{self.benchmark_config} ({{"batch_size": N}}), which the '
                "benchmark client writes before its first request"
            )
        return int(load_run_config(self.benchmark_config)["batch_size"])

    def begin_round(
        self, live_rows: torch.Tensor, frontier: torch.Tensor, sampled: torch.Tensor
    ) -> None:
        """Start a round of ``live_rows`` (int32 ``[1]``) requests (0: no
        round) whose cached lengths are ``frontier``'s first entries (int64
        ``[rows]``) and which draw their children when ``sampled`` (bool
        ``[1]``); device work only, so a captured graph can hold it."""
        t = self._tables
        eagle_harness_policy_begin_round(
            self._state,
            self._device_rounds,
            self._hist,
            self._slots,
            self.go_host,
            t.batch_table,
            t.tier_table,
            t.chain_f64 if self.chain else t.interval_f64,
            self._round_f64,
            live_rows,
            frontier,
            sampled,
            self.chain,
            self.node_top_k,
            _CHAIN_VERIFY_NUM,
            self.verify_num_limit,
            _MIN_VERIFY_COUNT,
        )

    def deltas_depth(self, depth: int) -> bool:
        """Whether depth ``depth``'s slot can ask for work deltas: a tree's
        even depths and a chain's odd ones, never the last."""
        return depth < self.tree_depth - 1 and depth % 2 == int(self.chain)

    def write_full_round(self) -> None:
        """Slots of a round drafting every depth at full width, which the
        drafter warms its depths up on outside a round (rounds rewrite them).
        No round is live afterwards: decisions launched on these slots (a
        warm-up) write nothing."""
        past = _CHAIN_VERIFY_NUM if self.chain else self.verify_num_limit
        write_round_slots(
            self._slots,
            self.node_top_k,
            [self.node_top_k] * (self.tree_depth - 1),
            lambda depth, start: min(past, start) if self.deltas_depth(depth) else 0,
        )
        self.past_verify_num.fill_(past)
        self._state[_BATCH_SIZE].fill_(0)

    def decide(self, depth: int, work_deltas: torch.Tensor) -> None:
        """The decision after ``depth`` (``1 <= depth < tree_depth``), launched
        before depth ``depth`` drafts with depth ``depth - 1``'s work deltas
        (fp32 ``[node_top_k + 1]``) in ``work_deltas``: writes depth
        ``depth + 1``'s slots, or stops the round (``verify_count`` and
        ``flat``, a chain's ``verify_num``, and in a sampled round the next
        sampled round's planned stop depth)."""
        if self.chain:
            eagle_harness_policy_decide_chain(
                self._state,
                self._device_rounds,
                self._slots,
                self.go_host,
                work_deltas,
                self._round_f64,
                depth,
                self.verify_num_limit,
            )
        else:
            eagle_harness_policy_decide_tree(
                self._state,
                self._device_rounds,
                self._slots,
                self.go_host,
                work_deltas,
                self._round_f64,
                self._tables.tier_table,
                depth,
                self.node_top_k,
                self.exec_mode,
                _ASYNC_LAYER_RATIO,
                self.verify_num_limit,
                _MIN_VERIFY_COUNT,
            )

    def select_verify(self, top_scores: torch.Tensor) -> None:
        """A tree round's cumulative ranked scores from ``top_scores`` (fp32
        ``[rows >= live rows, cols >= verify_count]``, each row's best pool
        scores in rank order), bit for bit as
        ``torch.cumsum(top_scores.mean(dim=0), dim=0)`` (of the flattened
        scores when ``flat``) with torch's default (non-deterministic-mode)
        cumsum computes them, published for the next round; then its
        ``verify_num`` from the previous round's (its own in the first round,
        none in a sampled first round), booked into the verify history.
        Chains: nothing (they verify every node; their stopping decision sets
        it).

        Raises:
            RuntimeError: torch's deterministic algorithms are enabled (its
                cumsum then sums in another order).
        """
        if self.chain:
            return
        if torch.are_deterministic_algorithms_enabled():
            raise RuntimeError(
                "the device ranked scores replicate torch's default cumsum, not its "
                "deterministic one"
            )
        t = self._tables
        eagle_harness_policy_select_verify(
            self._state,
            self._recent,
            self._hist,
            self._ranked,
            top_scores,
            t.interval_i32,
            t.interval_f64,
            t.verify_i32,
            t.verify_f64,
            self.tree_depth,
            self.exec_mode,
        )

    def go_ready(self, depth: int) -> bool | None:
        """Whether depth ``depth`` drafts in the current round, from the
        pinned mirror without a synchronization: None until the decision
        wrote it."""
        value = int(self._go_host_view[depth])
        if value >> 1 != self._rounds:
            return None
        return bool(value & 1)
