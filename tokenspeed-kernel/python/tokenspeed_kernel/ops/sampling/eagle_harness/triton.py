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

"""Triton build of the EAGLE-Harness draft-tree and device-policy ops
(greedy trees and chains), registered in family ``sampling`` with the op
name as mode; the kernels live in ``_triton/``.

Every op launches a fixed number of programs and reads its per-round counts
(live rows, widths, pool sizes) from device int32 ``[1]`` tensors, so a
launch captured in a graph once is replayed for every round. Kernels run
without FMA contraction in a fixed reduction order: results are
deterministic bit for bit.

Not built here: the sampled-tree ops (``sample_children``,
``tree_reject_sample``, ``residual_sample``, ``publish_draft_q``) and the
device-chain ops, which need device-side graph launch.
``policy_select_verify`` registers for AMD only; its fixed summation order
(column means, then a blocked prefix sum) is spelled out on
``triton_eagle_harness_policy_select_verify``.
"""

from __future__ import annotations

import struct

import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.sampling.eagle_harness._triton import policy as _policy
from tokenspeed_kernel.ops.sampling.eagle_harness._triton import tree as _tree
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signature

__all__ = [
    "triton_eagle_harness_update_node_data",
    "triton_eagle_harness_work_deltas",
    "triton_eagle_harness_draft_depth_update",
    "triton_eagle_harness_stable_topk_desc",
    "triton_eagle_harness_write_verify_tree",
    "triton_eagle_harness_policy_load",
    "triton_eagle_harness_policy_begin_round",
    "triton_eagle_harness_policy_decide_tree",
    "triton_eagle_harness_policy_decide_chain",
    "triton_eagle_harness_policy_select_verify",
]

_PORTABLE = CapabilityRequirement(vendors=frozenset({"nvidia", "amd"}))
_AMD = CapabilityRequirement(vendors=frozenset({"amd"}))
# One IEEE rounding per operation (no FMA contraction): reproducible bits.
_EXACT = dict(enable_fp_fusion=False)
# Rows the work deltas sum over (a graph's padded batch).
_MAX_BATCH_SIZE = 512


@register_kernel(
    "sampling",
    "eagle_harness_update_node_data",
    name="triton_eagle_harness_update_node_data",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)
def triton_eagle_harness_update_node_data(
    pool_scores: torch.Tensor,
    pool_tokens: torch.Tensor,
    work_scores: torch.Tensor,
    sort_scores: torch.Tensor,
    child_scores: torch.Tensor,
    child_tokens: torch.Tensor,
) -> None:
    """Writes the root's children into pool nodes ``1..K``, in place.

    ``child_scores`` / ``child_tokens`` (contiguous ``[bs, K]`` fp32 /
    int64) go to columns ``1..K`` of ``pool_scores`` / ``pool_tokens``
    (``[rows >= bs, nodes]``, equal row strides), ``work_scores`` and
    ``sort_scores`` (row-strided fp32). Grid: ``cdiv(bs * K, 256)``
    programs of 256 elements; ``K`` is constexpr.
    """
    if pool_tokens.stride(0) != pool_scores.stride(0):
        raise ValueError("eagle_harness: pool row strides")
    if not (child_scores.is_contiguous() and child_tokens.is_contiguous()):
        raise ValueError("eagle_harness: contiguous [bs, K] children")
    rows, k = child_scores.shape
    if rows * k == 0:
        return
    block = 256
    _tree._update_node_data_kernel[(triton.cdiv(rows * k, block),)](
        pool_scores,
        pool_tokens,
        work_scores,
        sort_scores,
        child_scores,
        child_tokens,
        rows,
        pool_scores.stride(0),
        work_scores.stride(0),
        sort_scores.stride(0),
        K=k,
        BLOCK=block,
    )


@register_kernel(
    "sampling",
    "eagle_harness_work_deltas",
    name="triton_eagle_harness_work_deltas",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)
def triton_eagle_harness_work_deltas(
    work_scores: torch.Tensor,
    work_deltas: torch.Tensor,
    live_rows: torch.Tensor,
    node_count: torch.Tensor,
    verify_num: torch.Tensor,
) -> None:
    """Writes ``work_deltas[j]`` (contiguous fp32 ``[K + 1]``), in place: the
    mean over the first ``live_rows`` rows of the sum of each row's best
    ``min(node_count + j, verify_num)`` scores, taken from two descending
    segments of the row (``work_scores``, fp32, unit column stride): the
    ``node_count`` best nodes so far, then the next frontier's ``j`` best
    scores (ties to the first segment).

    ``live_rows``, ``node_count`` and ``verify_num`` are device int32
    ``[1]``; a zero ``node_count`` writes nothing. ``work_scores`` holds at
    most 512 rows. Grid: one program per ``j``; the batch sum is a halving
    tree over 256 lanes (512 when ``work_scores`` has more than 256 rows),
    so the sum's order depends only on that row capacity.
    """
    if work_scores.size(0) > _MAX_BATCH_SIZE:
        raise ValueError("eagle_harness: work deltas past 512 rows")
    if work_scores.stride(1) != 1:
        raise ValueError("eagle_harness: work score rows")
    # The rows the live ones are among fix the sum tree's width.
    threads = _MAX_BATCH_SIZE // 2
    if work_scores.size(0) > threads:
        threads = _MAX_BATCH_SIZE
    _tree._work_deltas_kernel[(work_deltas.size(0),)](
        work_scores,
        work_deltas,
        live_rows,
        node_count,
        verify_num,
        work_scores.stride(0),
        THREADS=threads,
        LOG_THREADS=threads.bit_length() - 1,
        **_EXACT,
    )


@register_kernel(
    "sampling",
    "eagle_harness_draft_depth_update",
    name="triton_eagle_harness_draft_depth_update",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)
def triton_eagle_harness_draft_depth_update(
    child_probs: torch.Tensor,
    child_tokens: torch.Tensor,
    expanded: torch.Tensor,
    child_hidden: torch.Tensor,
    frontier_tokens: torch.Tensor,
    frontier_scores: torch.Tensor,
    frontier_hidden: torch.Tensor,
    frontier_index: torch.Tensor,
    pool_tokens: torch.Tensor,
    pool_scores: torch.Tensor,
    pool_prev: torch.Tensor,
    work_scores: torch.Tensor,
    sort_scores: torch.Tensor,
    positions: torch.Tensor,
    slot_starts: torch.Tensor,
    lane_mask: torch.Tensor,
    width: torch.Tensor,
    depth: int,
    cumu_count: torch.Tensor,
    verify_num: torch.Tensor,
    num_slots: int,
) -> None:
    """One draft depth's bookkeeping after its lane forward, in place.

    All tensors are contiguous with ``rows`` requests first. The forward
    expanded the first ``width`` of each request's ``K`` lanes:
    ``child_probs`` / ``child_tokens`` (``[rows, K, K]`` fp32 / int64,
    lanes past ``width`` ignored) are their children, ``expanded``
    (``[rows, K]`` int64) their pool indices and ``child_hidden``
    (``[rows, K, H]``, must not overlap ``frontier_hidden``) their output
    hidden rows. A child's path score is its probability times its lane's
    ``frontier_scores`` entry.

    The children, sorted descending by a bitonic network (equal scores
    keep their order), give the next frontier, the best ``K``:
    ``frontier_tokens``, ``frontier_scores``, ``frontier_index`` (pool
    indices) and ``frontier_hidden`` (``[rows, K, H]``, their parent lanes'
    ``child_hidden`` rows). All children are appended to the pool
    (``pool_tokens`` / ``pool_scores`` / ``pool_prev``, ``[rows, nodes]``)
    at node ``cumu_count``: tokens, ``path score * 0.9999 ** (depth + 1)``
    and parent pool indices. ``sort_scores`` (``[rows, <= 256]``) keeps the
    best ``verify_num`` of its first ``min(verify_num, cumu_count)`` entries
    and the children (stable, ties to the earlier); ``work_scores`` gets
    that head and then the frontier's scores, for the next work deltas.

    The lanes move one slot on: ``positions`` (``[rows, K]``) + 1,
    ``slot_starts`` (``[rows]``) + K. When depth ``depth + 1`` fits in the
    ``num_slots``-slot lane window, ``lane_mask`` (``[rows, K, words <=
    4]`` int64, window slot ``j`` at bit ``j % 64`` of word ``j // 64``)
    becomes the next depth's: lane ``r``'s parent lane's mask plus slot
    ``depth * K + r``.

    ``width``, ``cumu_count`` and ``verify_num`` are device int32 ``[1]``;
    ``depth >= 1`` and ``num_slots`` are host ints passed unspecialized.
    Constraints: ``K * K <= 64``. Grid: one program per request row.
    """
    rows, k = child_probs.shape[:2]
    for t in (
        child_probs,
        child_tokens,
        expanded,
        child_hidden,
        frontier_tokens,
        frontier_scores,
        frontier_hidden,
        frontier_index,
        pool_tokens,
        pool_scores,
        pool_prev,
        work_scores,
        sort_scores,
        positions,
        slot_starts,
        lane_mask,
    ):
        if not (t.is_contiguous() and t.size(0) == rows):
            raise ValueError("eagle_harness: contiguous [rows, ...]")
    for t in (child_probs, child_tokens, child_hidden, frontier_hidden, lane_mask):
        if not (t.dim() == 3 and t.size(1) == k):
            raise ValueError("eagle_harness: [rows, lanes, *] tensors")
    if not (
        child_probs.size(2) == k
        and child_tokens.size(2) == k
        and k * k <= _tree.MAX_CHILDREN
    ):
        raise ValueError(f"eagle_harness: {k} lanes of {child_probs.size(2)} children")
    for t in (expanded, frontier_tokens, frontier_scores, frontier_index, positions):
        if not (t.dim() == 2 and t.size(1) == k):
            raise ValueError("eagle_harness: frontier")
    num_nodes = pool_tokens.size(1)
    if not (pool_scores.size(1) == num_nodes and pool_prev.size(1) == num_nodes):
        raise ValueError("eagle_harness: node pool")
    if not (
        depth >= 1
        and sort_scores.size(1) <= _tree.MAX_VERIFY
        and sort_scores.size(1) + k <= work_scores.size(1)
    ):
        raise ValueError("eagle_harness: verify lists")
    mask_words = lane_mask.size(2)
    if not (mask_words <= _tree.MAX_MASK_WORDS and 0 <= num_slots <= 64 * mask_words):
        raise ValueError("eagle_harness: lane masks")
    hidden_size = child_hidden.size(2)
    if not (
        child_hidden.dtype == frontier_hidden.dtype
        and frontier_hidden.size(2) == hidden_size
    ):
        raise ValueError("eagle_harness: hidden rows")
    # The copy gathers rows across lanes: it cannot run in place.
    span = rows * k * hidden_size * child_hidden.element_size()
    a, b = child_hidden.data_ptr(), frontier_hidden.data_ptr()
    if not (a + span <= b or b + span <= a):
        raise ValueError("eagle_harness: child_hidden overlaps frontier_hidden")
    if rows == 0:
        return
    _tree._draft_depth_update_kernel[(rows,)](
        child_probs,
        child_tokens,
        expanded,
        child_hidden,
        frontier_tokens,
        frontier_scores,
        frontier_hidden,
        frontier_index,
        pool_tokens,
        pool_scores,
        pool_prev,
        work_scores,
        sort_scores,
        positions,
        slot_starts,
        lane_mask,
        width,
        cumu_count,
        verify_num,
        num_nodes,
        work_scores.size(1),
        sort_scores.size(1),
        depth,
        num_slots,
        hidden_size,
        K=k,
        K_PAD=triton.next_power_of_2(k),
        MASK_WORDS=mask_words,
        MASK_PAD=triton.next_power_of_2(mask_words),
        BLOCK_H=min(triton.next_power_of_2(hidden_size), 1024),
        **_EXACT,
    )


@register_kernel(
    "sampling",
    "eagle_harness_stable_topk_desc",
    name="triton_eagle_harness_stable_topk_desc",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)
def triton_eagle_harness_stable_topk_desc(
    scores: torch.Tensor,
    live_rows: torch.Tensor,
    n: torch.Tensor,
    k: torch.Tensor,
    out_values: torch.Tensor,
    out_indices: torch.Tensor,
) -> None:
    """The first ``k`` entries of a stable descending sort of the first
    ``n`` columns of each of the first ``live_rows`` rows of ``scores``
    (``[rows, <= 1024]`` fp32, unit column stride), written to the first
    ``k`` columns of ``out_values`` / ``out_indices`` (``[rows, max_k]``
    fp32 / int64, equal row strides).

    Order: NaN first, then the larger value, then the lower index, so the
    output is unique. ``live_rows``, ``n`` and ``k`` (``1 <= k <= n``) are
    device int32 ``[1]``. Grid: one program per row of ``scores``.
    """
    if not (
        scores.dim() == 2
        and out_values.dim() == 2
        and out_indices.dim() == 2
        and scores.stride(1) == 1
        and out_values.stride(1) == 1
        and out_indices.stride(1) == 1
        and out_values.stride(0) == out_indices.stride(0)
    ):
        raise ValueError("eagle_harness: stable_topk_desc layouts")
    rows = scores.size(0)
    if not (
        scores.size(1) <= 1024
        and out_values.size(0) == rows
        and out_indices.size(0) == rows
        and out_values.size(1) == out_indices.size(1)
    ):
        raise ValueError(
            "eagle_harness: stable_topk_desc needs <= 1024 score columns and [rows, max_k] outputs"
        )
    if rows == 0:
        return
    _tree._stable_topk_desc_kernel[(rows,)](
        scores,
        live_rows,
        n,
        k,
        out_values,
        out_indices,
        scores.stride(0),
        out_values.stride(0),
        BLOCK_N=max(triton.next_power_of_2(scores.size(1)), 16),
        CHUNK=16,
    )


@register_kernel(
    "sampling",
    "eagle_harness_write_verify_tree",
    name="triton_eagle_harness_write_verify_tree",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)
def triton_eagle_harness_write_verify_tree(
    ranked: torch.Tensor,
    pool_tokens: torch.Tensor,
    pool_prev: torch.Tensor,
    rows: torch.Tensor,
    input_map: torch.Tensor,
    parent_map: torch.Tensor,
    live_rows: torch.Tensor,
    verify_num: torch.Tensor,
) -> None:
    """Writes each of the first ``live_rows`` requests' verify tree into row
    ``rows[b]`` (int64 ``[bs]``) of the next round's ``input_map`` /
    ``parent_map`` (``[*, N]`` int32, ``N <= 64``).

    Slot ``i < verify_num`` is pool node ``ranked[b, i]`` (int64, ancestor
    first): its ``pool_tokens`` token and, as parent, the latest earlier
    slot holding its ``pool_prev`` parent. The remaining slots repeat the
    root (pool node 0) with parent -1. ``live_rows`` and ``verify_num`` are
    device int32 ``[1]``. Grid: one program per request.
    """
    if not (
        ranked.stride(1) == 1
        and pool_tokens.stride(1) == 1
        and pool_prev.stride(1) == 1
    ):
        raise ValueError("eagle_harness: unit column strides")
    if pool_tokens.stride(0) != pool_prev.stride(0):
        raise ValueError("eagle_harness: pool row strides")
    batch_size = ranked.size(0)
    if batch_size == 0:
        return
    num_slots = input_map.size(1)
    _tree._write_verify_tree_kernel[(batch_size,)](
        ranked,
        pool_tokens,
        pool_prev,
        rows,
        input_map,
        parent_map,
        live_rows,
        verify_num,
        ranked.stride(0),
        pool_tokens.stride(0),
        input_map.stride(0),
        parent_map.stride(0),
        NUM_SLOTS=num_slots,
        BLOCK_S=max(triton.next_power_of_2(num_slots), 16),
    )


@register_kernel(
    "sampling",
    "eagle_harness_policy_load",
    name="triton_eagle_harness_policy_load",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)
def triton_eagle_harness_policy_load() -> None:
    """Nothing to load: the kernels compile on first launch."""


def _round_depth(slots: torch.Tensor) -> int:
    """tree_depth from the [4, tree_depth + 1] slots."""
    if not (
        slots.dim() == 2
        and slots.size(0) == 4
        and slots.is_contiguous()
        and 2 <= slots.size(1) <= _policy.MAX_DEPTH + 1
    ):
        raise ValueError(
            f"eagle_harness: slots must be [4, depth + 1], depth <= {_policy.MAX_DEPTH}"
        )
    return slots.size(1) - 1


def _check_decision(depth: int, tree_depth: int, work_deltas: torch.Tensor, n: int):
    if not 1 <= depth < tree_depth:
        raise ValueError("eagle_harness: decision depth")
    if not (
        work_deltas.dim() == 1
        and work_deltas.size(0) == n
        and work_deltas.stride(0) == 1
    ):
        raise ValueError(f"eagle_harness: work deltas must be contiguous [{n}]")


def _f64_bits(value: float) -> int:
    """A double as the int64 of its bits (Triton would pass a float as fp32)."""
    return struct.unpack("<q", struct.pack("<d", value))[0]


@register_kernel(
    "sampling",
    "eagle_harness_policy_begin_round",
    name="triton_eagle_harness_policy_begin_round",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)
def triton_eagle_harness_policy_begin_round(
    state: torch.Tensor,
    seq: torch.Tensor,
    hist: torch.Tensor,
    slots: torch.Tensor,
    go_host: torch.Tensor,
    batch_table: torch.Tensor,
    tier_table: torch.Tensor,
    cost_f64: torch.Tensor,
    round_f64: torch.Tensor,
    live_rows: torch.Tensor,
    frontier: torch.Tensor,
    sampled: torch.Tensor,
    chain: bool,
    node_top_k: int,
    chain_past_verify_num: int,
    verify_num: int,
    min_verify_count: int,
) -> None:
    """Starts a policy round of ``live_rows[0]`` requests (int32; 0 runs no
    round) whose cached lengths are the first ``live_rows[0]`` entries of
    ``frontier`` (int64, strided) and which draw their children when
    ``sampled`` (bool ``[1]``).

    Increments ``seq`` (int64 ``[1]``), clears the round's scalars of
    ``state`` (int32) and the ``[4, depth + 1]`` int32 ``slots`` (width,
    cumu_start, work_size, go per depth), writes depth 0's slots, takes
    depth 0's decision and writes depth 1's, publishing ``go_host[1]``
    (pinned int64 ``[depth + 1]``, ``(seq << 1) | go``). A chain round whose
    ``batch_table`` row holds a width table width ``W`` instead writes
    every depth's slots (depths 1..W-2 draft) and go flags up to
    ``go_host[max(W - 1, 2)]``, and verifies ``min(W, verify_num)`` nodes.

    ``hist`` (``[max_bs, rounds <= 32]`` int32) holds the past verify
    widths; ``tier_table`` (int32 ``[9]``) the verify floor per cost
    variant (four tiers, then tiny), then the width per tier (unused for a
    chain); ``batch_table`` (int64 ``[max_bs, 5]``) per batch size the cost
    id, chain tier row, interval count, cache step and chain table width
    (0: the cost rule decides). The round's row of ``cost_f64`` (a tree's
    interval table, a chain's per-depth stop terms) is copied to
    ``round_f64`` for the decisions. A chain's past verify width is
    ``chain_past_verify_num``; ``verify_num`` and ``min_verify_count`` cap
    the ranked node count when a decision stops the round. ``node_top_k``,
    the tree depth (``<= 32``) and ``chain`` are compiled in. Grid:
    ``(1,)``.
    """
    tree_depth = _round_depth(slots)
    if hist.size(1) > _policy.MAX_HISTORY:
        raise ValueError("eagle_harness: verify history")
    if not (
        batch_table.size(1) == _policy.BATCH_COLS
        and (chain or tier_table.size(0) == _policy.TIER_TABLE_SIZE)
    ):
        raise ValueError("eagle_harness: policy tables")
    cost_cols = 0 if chain else cost_f64.size(2)
    copy = tree_depth * _policy.CHAIN_F64 if chain else 2 + tree_depth
    _policy._policy_begin_round_kernel[(1,)](
        state,
        seq,
        hist,
        slots,
        go_host,
        batch_table,
        tier_table,
        cost_f64,
        round_f64,
        live_rows,
        frontier,
        sampled,
        frontier.stride(0),
        hist.size(1),
        cost_f64.size(1),
        cost_cols,
        chain_past_verify_num,
        verify_num,
        min_verify_count,
        NODE_TOP_K=node_top_k,
        TREE_DEPTH=tree_depth,
        CHAIN=chain,
        DEPTH_PAD=triton.next_power_of_2(tree_depth + 1),
        COPY_PAD=triton.next_power_of_2(copy),
        **_EXACT,
    )


@register_kernel(
    "sampling",
    "eagle_harness_policy_decide_tree",
    name="triton_eagle_harness_policy_decide_tree",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)
def triton_eagle_harness_policy_decide_tree(
    state: torch.Tensor,
    seq: torch.Tensor,
    slots: torch.Tensor,
    go_host: torch.Tensor,
    work_deltas: torch.Tensor,
    round_f64: torch.Tensor,
    tier_table: torch.Tensor,
    depth: int,
    node_top_k: int,
    exec_mode: int,
    async_layer_ratio: float,
    verify_num: int,
    min_verify_count: int,
) -> None:
    """A tree round's decision after ``depth`` (``1 <= depth < tree_depth``),
    on the work deltas (contiguous fp32 ``[node_top_k + 1]``) queued after
    ``depth - 1``.

    Writes depth ``depth + 1``'s slots and publishes its go flag; once the
    round stopped, only clears that flag. The gain of going on,
    ``(deltas[narrow] - deltas[0]) / draft_cost * model_cost_ratio *
    async_layer_ratio`` (fp64, ``narrow`` the tier's width), is compared
    with the depth's stop threshold from ``round_f64``; exec mode 1 cuts a
    batch much smaller than the benchmark batch to width 1. A stopping
    decision records the round's ranked node count, ``min(verify_num, pool
    size, max(2 * past verify width, min_verify_count))``, and in a
    sampled round the next sampled round's planned stop depth (a sampled
    round stops only at its planned depth). Grid: ``(1,)``.
    """
    tree_depth = _round_depth(slots)
    _check_decision(depth, tree_depth, work_deltas, node_top_k + 1)
    _policy._policy_decide_kernel[(1,)](
        state,
        seq,
        slots,
        go_host,
        work_deltas,
        round_f64,
        tier_table,
        depth,
        exec_mode,
        _f64_bits(async_layer_ratio),
        verify_num,
        min_verify_count,
        NODE_TOP_K=node_top_k,
        TREE_DEPTH=tree_depth,
        CHAIN=False,
        **_EXACT,
    )


@register_kernel(
    "sampling",
    "eagle_harness_policy_decide_chain",
    name="triton_eagle_harness_policy_decide_chain",
    solution="triton",
    capability=_PORTABLE,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)
def triton_eagle_harness_policy_decide_chain(
    state: torch.Tensor,
    seq: torch.Tensor,
    slots: torch.Tensor,
    go_host: torch.Tensor,
    work_deltas: torch.Tensor,
    round_f64: torch.Tensor,
    depth: int,
    verify_num: int,
) -> None:
    """A chain round's decision after ``depth`` (``1 <= depth <
    tree_depth``), on contiguous fp32 ``[2]`` work deltas.

    The gain ``2 * (deltas[1] - deltas[0]) / gain_denom`` per step cost
    (fp64, the depth's terms in ``round_f64``) below the depth's threshold
    stops the round; a depth with a verify cost cliff instead looks two
    depths ahead. Writes depth ``depth + 1``'s slots and go flag, or on a
    stop the round's counts (every ranked node, ``verify_num`` at most, is
    verified). A width table round's decision changes no state or slot:
    past its last drafted depth it only clears ``go_host[depth + 1]``.
    Grid: ``(1,)``.
    """
    tree_depth = _round_depth(slots)
    _check_decision(depth, tree_depth, work_deltas, 2)
    # A chain reads no tier table; any int32 tensor stands in for the pointer.
    _policy._policy_decide_kernel[(1,)](
        state,
        seq,
        slots,
        go_host,
        work_deltas,
        round_f64,
        state,
        depth,
        0,
        0,
        verify_num,
        0,
        NODE_TOP_K=1,
        TREE_DEPTH=tree_depth,
        CHAIN=True,
        **_EXACT,
    )


@register_kernel(
    "sampling",
    "eagle_harness_policy_select_verify",
    name="triton_eagle_harness_policy_select_verify",
    solution="triton",
    capability=_AMD,
    signatures=frozenset({format_signature()}),
    priority=Priority.PORTABLE,
)
def triton_eagle_harness_policy_select_verify(
    state: torch.Tensor,
    recent: torch.Tensor,
    hist: torch.Tensor,
    published: torch.Tensor,
    top_scores: torch.Tensor,
    interval_i32: torch.Tensor,
    interval_f64: torch.Tensor,
    verify_i32: torch.Tensor,
    verify_f64: torch.Tensor,
    tree_depth: int,
    exec_mode: int,
) -> None:
    """A tree round's cumulative ranked scores and verify width.

    The scores of ``t = top_scores[:live rows, :verify count]`` (fp32, unit
    column stride, at most 32 rows and 64 columns) are the inclusive prefix
    sums of its column means, or of ``t`` flattened row-major for a flat
    round, in fp32 with one rounding per operation in this order:

    * column means, ``cols > 1``: four accumulators per column, row ``k``
      added in increasing ``k`` into accumulator ``k % 4``, combined as
      ``((a0 + a1) + a2) + a3`` and multiplied by the fp32 quotient
      ``cols / (rows * cols)``. One column: with ``w`` the largest power of
      two ``<= rows``, lane ``i < w`` holds ``t[i] + t[i + w]`` (``+ 0`` when
      ``i + w >= rows``), the 32 zero-padded lanes are summed as adjacent
      pairs ``(2j, 2j + 1)`` five times, and the total is multiplied by the
      same quotient (``1 / rows``);
    * prefix sum: the values ``v`` (zero past the count) form one tile of
      128 lanes x 16 consecutive items (lane ``l`` holds ``v[16 l]`` to
      ``v[16 l + 15]``). Each lane sums its items left to right; an
      inclusive scan runs within each 16-lane row with steps 1, 2, 4, 8
      (``p[l] = p[l - s] + p[l]``); lanes 16..31 of each 32-lane half add
      lane 15's value, lanes 32..63 of each 64-lane group add lane 31's;
      lane ``l``'s exclusive prefix ``x`` is ``p[l - 1]`` for ``l < 64``,
      ``p[63]`` for ``l = 64`` and ``p[63] + p[l - 1]`` for ``l > 64``. The
      outputs are ``x + v[16 l]`` (just ``v[0]`` for lane 0), then each
      next item added to the previous output.

    ``published`` (fp32) holds
    the previous round's scores, which size the verify (this round's in the
    first round; a sampled first round takes the largest measured width
    below its ranked count), and then receives this round's. The width is
    written to ``state`` and shifted into ``hist`` (``[max_bs, <= 32]``
    int32) and ``recent`` (``[<= 32]`` int32). ``verify_i32`` /
    ``verify_f64`` hold per batch size and interval the tier and the
    stable / unstable thresholds; ``interval_i32`` the measured widths and
    ``interval_f64`` the cost deltas after the ``2 + 5 * tree_depth`` stop
    columns. Grid: ``(1,)``.
    """
    if not (
        hist.size(1) <= _policy.MAX_HISTORY and recent.size(0) <= _policy.MAX_HISTORY
    ):
        raise ValueError("eagle_harness: verify history")
    if not (
        hist.size(0) <= _policy.RANK_MAX_ROWS
        and top_scores.size(1) <= _policy.RANK_MAX_COLS
        and top_scores.stride(1) == 1
    ):
        raise ValueError(
            "eagle_harness: ranked scores: at most 32 rows and 64 unit-stride columns"
        )
    max_widths = interval_i32.size(2) - 1
    if max_widths > _policy.MAX_WIDTHS:
        raise ValueError("eagle_harness: measured widths")
    _policy._policy_select_verify_kernel[(1,)](
        state,
        recent,
        hist,
        published,
        top_scores,
        interval_i32,
        interval_f64,
        verify_i32,
        verify_f64,
        recent.size(0),
        hist.size(1),
        published.size(0),
        top_scores.stride(0),
        interval_i32.size(1),
        interval_i32.size(2),
        interval_f64.size(2),
        2 + 5 * tree_depth,
        max_widths,
        exec_mode,
        **_EXACT,
    )
