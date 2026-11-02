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

"""Portable Triton build of the EAGLE-Harness draft-tree and device-policy
ops (greedy trees and chains). Each op keeps the contract of the CUDA wrapper
of the same name and its floating-point order, so both builds give the same
bits; the kernels live in ``_triton/``.

Not built here: the sampled-tree ops (``sample_children``,
``tree_reject_sample``, ``residual_sample``, ``publish_draft_q``) and the
device-chain ops, which need device-side graph launch. ``select_verify``
replicates ROCm torch's reduction and scan order, so it registers for AMD
only; NVIDIA keeps the CUDA build, whose order is CUB's.
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
# One IEEE rounding per operation, as the CUDA build computes.
_EXACT = dict(enable_fp_fusion=False)
# Rows the work deltas sum over (a graph's padded batch).
_MAX_BATCH_SIZE = 512


def _register(op: str, capability: CapabilityRequirement = _PORTABLE):
    return register_kernel(
        "sampling",
        op,
        name=f"triton_{op}",
        solution="triton",
        capability=capability,
        signatures=frozenset({format_signature()}),
        priority=Priority.PORTABLE,
    )


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"eagle_harness: {message}")


@_register("eagle_harness_update_node_data")
def triton_eagle_harness_update_node_data(
    pool_scores: torch.Tensor,
    pool_tokens: torch.Tensor,
    work_scores: torch.Tensor,
    sort_scores: torch.Tensor,
    child_scores: torch.Tensor,
    child_tokens: torch.Tensor,
) -> None:
    """See ``eagle_harness_update_node_data`` of the CUDA build."""
    _check(pool_tokens.stride(0) == pool_scores.stride(0), "pool row strides")
    _check(
        child_scores.is_contiguous() and child_tokens.is_contiguous(),
        "contiguous [bs, K] children",
    )
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


@_register("eagle_harness_work_deltas")
def triton_eagle_harness_work_deltas(
    work_scores: torch.Tensor,
    work_deltas: torch.Tensor,
    live_rows: torch.Tensor,
    node_count: torch.Tensor,
    verify_num: torch.Tensor,
) -> None:
    """See ``eagle_harness_work_deltas`` of the CUDA build."""
    _check(work_scores.size(0) <= _MAX_BATCH_SIZE, "work deltas past 512 rows")
    _check(work_scores.stride(1) == 1, "work score rows")
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


@_register("eagle_harness_draft_depth_update")
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
    """See ``eagle_harness_draft_depth_update`` of the CUDA build."""
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
        _check(t.is_contiguous() and t.size(0) == rows, "contiguous [rows, ...]")
    for t in (child_probs, child_tokens, child_hidden, frontier_hidden, lane_mask):
        _check(t.dim() == 3 and t.size(1) == k, "[rows, lanes, *] tensors")
    _check(
        child_probs.size(2) == k
        and child_tokens.size(2) == k
        and k * k <= _tree.MAX_CHILDREN,
        f"{k} lanes of {child_probs.size(2)} children",
    )
    for t in (expanded, frontier_tokens, frontier_scores, frontier_index, positions):
        _check(t.dim() == 2 and t.size(1) == k, "frontier")
    num_nodes = pool_tokens.size(1)
    _check(
        pool_scores.size(1) == num_nodes and pool_prev.size(1) == num_nodes,
        "node pool",
    )
    _check(
        depth >= 1
        and sort_scores.size(1) <= _tree.MAX_VERIFY
        and sort_scores.size(1) + k <= work_scores.size(1),
        "verify lists",
    )
    mask_words = lane_mask.size(2)
    _check(
        mask_words <= _tree.MAX_MASK_WORDS and 0 <= num_slots <= 64 * mask_words,
        "lane masks",
    )
    hidden_size = child_hidden.size(2)
    _check(
        child_hidden.dtype == frontier_hidden.dtype
        and frontier_hidden.size(2) == hidden_size,
        "hidden rows",
    )
    # The copy gathers rows across lanes: it cannot run in place.
    span = rows * k * hidden_size * child_hidden.element_size()
    a, b = child_hidden.data_ptr(), frontier_hidden.data_ptr()
    _check(a + span <= b or b + span <= a, "child_hidden overlaps frontier_hidden")
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


@_register("eagle_harness_stable_topk_desc")
def triton_eagle_harness_stable_topk_desc(
    scores: torch.Tensor,
    live_rows: torch.Tensor,
    n: torch.Tensor,
    k: torch.Tensor,
    out_values: torch.Tensor,
    out_indices: torch.Tensor,
) -> None:
    """See ``eagle_harness_stable_topk_desc`` of the CUDA build."""
    _check(
        scores.dim() == 2
        and out_values.dim() == 2
        and out_indices.dim() == 2
        and scores.stride(1) == 1
        and out_values.stride(1) == 1
        and out_indices.stride(1) == 1
        and out_values.stride(0) == out_indices.stride(0),
        "stable_topk_desc layouts",
    )
    rows = scores.size(0)
    _check(
        scores.size(1) <= 1024
        and out_values.size(0) == rows
        and out_indices.size(0) == rows
        and out_values.size(1) == out_indices.size(1),
        "stable_topk_desc needs <= 1024 score columns and [rows, max_k] outputs",
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


@_register("eagle_harness_write_verify_tree")
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
    """See ``eagle_harness_write_verify_tree`` of the CUDA build."""
    _check(
        ranked.stride(1) == 1
        and pool_tokens.stride(1) == 1
        and pool_prev.stride(1) == 1,
        "unit column strides",
    )
    _check(pool_tokens.stride(0) == pool_prev.stride(0), "pool row strides")
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


@_register("eagle_harness_policy_load")
def triton_eagle_harness_policy_load() -> None:
    """Nothing to load: the kernels compile on first launch."""


def _round_depth(slots: torch.Tensor) -> int:
    """tree_depth from the [4, tree_depth + 1] slots."""
    _check(
        slots.dim() == 2
        and slots.size(0) == 4
        and slots.is_contiguous()
        and 2 <= slots.size(1) <= _policy.MAX_DEPTH + 1,
        f"slots must be [4, depth + 1], depth <= {_policy.MAX_DEPTH}",
    )
    return slots.size(1) - 1


def _check_decision(depth: int, tree_depth: int, work_deltas: torch.Tensor, n: int):
    _check(1 <= depth < tree_depth, "decision depth")
    _check(
        work_deltas.dim() == 1
        and work_deltas.size(0) == n
        and work_deltas.stride(0) == 1,
        f"work deltas must be contiguous [{n}]",
    )


def _f64_bits(value: float) -> int:
    """A double as the int64 of its bits (Triton would pass a float as fp32)."""
    return struct.unpack("<q", struct.pack("<d", value))[0]


@_register("eagle_harness_policy_begin_round")
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
    """See ``eagle_harness_policy_begin_round`` of the CUDA build."""
    tree_depth = _round_depth(slots)
    _check(hist.size(1) <= _policy.MAX_HISTORY, "verify history")
    _check(
        batch_table.size(1) == _policy.BATCH_COLS
        and (chain or tier_table.size(0) == _policy.TIER_TABLE_SIZE),
        "policy tables",
    )
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


@_register("eagle_harness_policy_decide_tree")
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
    """See ``eagle_harness_policy_decide_tree`` of the CUDA build."""
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


@_register("eagle_harness_policy_decide_chain")
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
    """See ``eagle_harness_policy_decide_chain`` of the CUDA build."""
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


@_register("eagle_harness_policy_select_verify", capability=_AMD)
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
    """See ``eagle_harness_policy_select_verify`` of the CUDA build; the
    cumulative ranked scores are ROCm torch's bits."""
    _check(
        hist.size(1) <= _policy.MAX_HISTORY and recent.size(0) <= _policy.MAX_HISTORY,
        "verify history",
    )
    _check(
        hist.size(0) <= _policy.RANK_MAX_ROWS
        and top_scores.size(1) <= _policy.RANK_MAX_COLS
        and top_scores.stride(1) == 1,
        "ranked scores past the replicated torch shapes",
    )
    max_widths = interval_i32.size(2) - 1
    _check(max_widths <= _policy.MAX_WIDTHS, "measured widths")
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
