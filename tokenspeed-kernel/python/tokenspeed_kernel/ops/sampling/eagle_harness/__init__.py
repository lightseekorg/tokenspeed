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

"""EAGLE-Harness (EAGLE-4) draft-tree, device-policy and device-chain ops.

Each op dispatches through the registry (family ``sampling``, mode = the op
name) to the kernel registered for the current platform: the CUDA build
(``cuda.py``) on NVIDIA, the HIP build (``hip.py``) on AMD gfx950. Both builds
keep one contract per op, documented on the CUDA wrapper of the same name in
``tokenspeed_kernel.thirdparty.cuda.eagle_harness``,
``...eagle_harness_policy`` and ``...eagle_harness_chain``.

The device-chain ops (a depth's graph choosing the round's next graph on the
device) exist where the platform can launch that choice without the Python
host: by device-side graph launch (CUDA), or by a native launcher thread the
GPU posts its choice to (HIP). Check ``eagle_harness_device_chain_available()``
before using them.
"""

from __future__ import annotations

import functools

import tokenspeed_kernel.ops.sampling.eagle_harness.cuda  # noqa: F401
import tokenspeed_kernel.ops.sampling.eagle_harness.hip  # noqa: F401
import torch
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    SelectedKernel,
    select_kernel,
)
from tokenspeed_kernel.signature import format_signature

__all__ = [
    "eagle_harness_device_chain_available",
    "eagle_harness_update_node_data",
    "eagle_harness_work_deltas",
    "eagle_harness_draft_depth_update",
    "eagle_harness_stable_topk_desc",
    "eagle_harness_write_verify_tree",
    "eagle_harness_sample_children",
    "eagle_harness_tree_reject_sample",
    "eagle_harness_residual_sample",
    "eagle_harness_publish_draft_q",
    "eagle_harness_policy_load",
    "eagle_harness_policy_begin_round",
    "eagle_harness_policy_decide_tree",
    "eagle_harness_policy_decide_chain",
    "eagle_harness_policy_select_verify",
    "eagle_harness_chain_load",
    "eagle_harness_chain_instantiate",
    "eagle_harness_chain_destroy",
    "eagle_harness_chain_next",
    "eagle_harness_chain_begin",
    "eagle_harness_chain_join",
]


@functools.cache
def _kernel(op: str) -> SelectedKernel:
    # The ops take no format-dispatched tensors: the platform alone selects.
    return select_kernel("sampling", op, format_signature())


def eagle_harness_device_chain_available() -> bool:
    """Whether the platform has the device-chain ops: ``eagle_harness_chain_*``
    are usable only when this is True."""
    try:
        _kernel("eagle_harness_chain_next")
    except NoKernelFoundError:
        return False
    return True


def eagle_harness_update_node_data(
    pool_scores: torch.Tensor,
    pool_tokens: torch.Tensor,
    work_scores: torch.Tensor,
    sort_scores: torch.Tensor,
    child_scores: torch.Tensor,
    child_tokens: torch.Tensor,
) -> None:
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_update_node_data")(
        pool_scores, pool_tokens, work_scores, sort_scores, child_scores, child_tokens
    )


def eagle_harness_work_deltas(
    work_scores: torch.Tensor,
    work_deltas: torch.Tensor,
    live_rows: torch.Tensor,
    node_count: torch.Tensor,
    verify_num: torch.Tensor,
) -> None:
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_work_deltas")(
        work_scores, work_deltas, live_rows, node_count, verify_num
    )


def eagle_harness_draft_depth_update(
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
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_draft_depth_update")(
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
        depth,
        cumu_count,
        verify_num,
        num_slots,
    )


def eagle_harness_stable_topk_desc(
    scores: torch.Tensor,
    live_rows: torch.Tensor,
    n: torch.Tensor,
    k: torch.Tensor,
    out_values: torch.Tensor,
    out_indices: torch.Tensor,
) -> None:
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_stable_topk_desc")(
        scores, live_rows, n, k, out_values, out_indices
    )


def eagle_harness_write_verify_tree(
    ranked: torch.Tensor,
    pool_tokens: torch.Tensor,
    pool_prev: torch.Tensor,
    rows: torch.Tensor,
    input_map: torch.Tensor,
    parent_map: torch.Tensor,
    live_rows: torch.Tensor,
    verify_num: torch.Tensor,
) -> None:
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_write_verify_tree")(
        ranked,
        pool_tokens,
        pool_prev,
        rows,
        input_map,
        parent_map,
        live_rows,
        verify_num,
    )


def eagle_harness_sample_children(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    tokens: torch.Tensor,
    scores: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    q_store: torch.Tensor,
    qid_store: torch.Tensor,
    node_slot: torch.Tensor,
    node_pos: torch.Tensor,
    width: torch.Tensor | None,
    slot_base: int,
    cumu_start: torch.Tensor,
    vocab_limit: int,
    seed: int,
    round_counter: torch.Tensor,
    launch: int,
) -> None:
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_sample_children")(
        logits,
        candidates,
        tokens,
        scores,
        temperatures,
        top_ks,
        top_ps,
        q_store,
        qid_store,
        node_slot,
        node_pos,
        width,
        slot_base,
        cumu_start,
        vocab_limit,
        seed,
        round_counter,
        launch,
    )


def eagle_harness_tree_reject_sample(
    probas: torch.Tensor,
    tokens: torch.Tensor,
    parents: torch.Tensor,
    q_rows: torch.Tensor | None,
    q_slot: torch.Tensor,
    q_pos: torch.Tensor,
    q_store: torch.Tensor,
    qid_store: torch.Tensor,
    coins: torch.Tensor,
    top_ks: torch.Tensor,
    argmax: torch.Tensor,
    path: torch.Tensor,
    predicts: torch.Tensor,
    accept_length: torch.Tensor,
    last_row: torch.Tensor,
) -> None:
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_tree_reject_sample")(
        probas,
        tokens,
        parents,
        q_rows,
        q_slot,
        q_pos,
        q_store,
        qid_store,
        coins,
        top_ks,
        argmax,
        path,
        predicts,
        accept_length,
        last_row,
    )


def eagle_harness_residual_sample(
    probas: torch.Tensor,
    last_row: torch.Tensor,
    coins: torch.Tensor,
    top_ks: torch.Tensor,
    argmax: torch.Tensor,
    accept_length: torch.Tensor,
    predicts: torch.Tensor,
    num_nodes: int,
) -> None:
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_residual_sample")(
        probas, last_row, coins, top_ks, argmax, accept_length, predicts, num_nodes
    )


def eagle_harness_publish_draft_q(
    ranked: torch.Tensor,
    node_slot: torch.Tensor,
    node_pos: torch.Tensor,
    q_round: torch.Tensor,
    qid_round: torch.Tensor,
    rows: torch.Tensor,
    q_tab: torch.Tensor,
    qid_tab: torch.Tensor,
    slot_tab: torch.Tensor,
    pos_tab: torch.Tensor,
    live_rows: torch.Tensor,
    verify_num: torch.Tensor,
) -> None:
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_publish_draft_q")(
        ranked,
        node_slot,
        node_pos,
        q_round,
        qid_round,
        rows,
        q_tab,
        qid_tab,
        slot_tab,
        pos_tab,
        live_rows,
        verify_num,
    )


def eagle_harness_policy_load() -> None:
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_policy_load")()


def eagle_harness_policy_begin_round(
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
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_policy_begin_round")(
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
        chain,
        node_top_k,
        chain_past_verify_num,
        verify_num,
        min_verify_count,
    )


def eagle_harness_policy_decide_tree(
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
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_policy_decide_tree")(
        state,
        seq,
        slots,
        go_host,
        work_deltas,
        round_f64,
        tier_table,
        depth,
        node_top_k,
        exec_mode,
        async_layer_ratio,
        verify_num,
        min_verify_count,
    )


def eagle_harness_policy_decide_chain(
    state: torch.Tensor,
    seq: torch.Tensor,
    slots: torch.Tensor,
    go_host: torch.Tensor,
    work_deltas: torch.Tensor,
    round_f64: torch.Tensor,
    depth: int,
    verify_num: int,
) -> None:
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_policy_decide_chain")(
        state, seq, slots, go_host, work_deltas, round_f64, depth, verify_num
    )


def eagle_harness_policy_select_verify(
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
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_policy_select_verify")(
        state,
        recent,
        hist,
        published,
        top_scores,
        interval_i32,
        interval_f64,
        verify_i32,
        verify_f64,
        tree_depth,
        exec_mode,
    )


def eagle_harness_chain_load() -> None:
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_chain_load")()


def eagle_harness_chain_instantiate(graph: int, stream: int) -> int:
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_chain_instantiate")(graph, stream)


def eagle_harness_chain_destroy(exec_handle: int) -> None:
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_chain_destroy")(exec_handle)


def eagle_harness_chain_next(
    execs: torch.Tensor,
    go: torch.Tensor,
    sampled: torch.Tensor,
    depth: int,
    last_depth: int,
) -> None:
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_chain_next")(execs, go, sampled, depth, last_depth)


def eagle_harness_chain_begin(stream: int) -> None:
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_chain_begin")(stream)


def eagle_harness_chain_join(stream: int) -> None:
    """Registry-dispatched; the contract is on the CUDA wrapper of this name."""
    return _kernel("eagle_harness_chain_join")(stream)
