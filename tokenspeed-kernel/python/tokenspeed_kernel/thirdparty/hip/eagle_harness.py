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

"""EAGLE-Harness draft-tree bookkeeping kernels (EAGLE-4), HIP build; see
``csrc/eagle_harness_tree.hip``. Each op has the contract of the CUDA build's
``tokenspeed_kernel.thirdparty.cuda.eagle_harness`` op of the same name."""

import functools
from pathlib import Path

import torch

__all__ = [
    "eagle_harness_draft_depth_update",
    "eagle_harness_publish_draft_q",
    "eagle_harness_residual_sample",
    "eagle_harness_sample_children",
    "eagle_harness_tree_reject_sample",
    "eagle_harness_stable_topk_desc",
    "eagle_harness_update_node_data",
    "eagle_harness_work_deltas",
    "eagle_harness_write_verify_tree",
]


@functools.cache
def _module():
    import tvm_ffi

    so_path = (
        Path(__file__).parent / "objs" / "eagle_harness_tree" / "eagle_harness_tree.so"
    )
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel eagle_harness_tree library not found at {so_path}. "
            "Run: pip install -e tokenspeed-kernel/python/ (ROCm backend)"
        )
    return tvm_ffi.load_module(str(so_path))


def eagle_harness_update_node_data(
    pool_scores: torch.Tensor,
    pool_tokens: torch.Tensor,
    work_scores: torch.Tensor,
    sort_scores: torch.Tensor,
    child_scores: torch.Tensor,
    child_tokens: torch.Tensor,
) -> None:
    """HIP build of the CUDA op of the same name."""
    _module().eagle_harness_update_node_data(
        pool_scores, pool_tokens, work_scores, sort_scores, child_scores, child_tokens
    )


def eagle_harness_work_deltas(
    work_scores: torch.Tensor,
    work_deltas: torch.Tensor,
    live_rows: torch.Tensor,
    node_count: torch.Tensor,
    verify_num: torch.Tensor,
) -> None:
    """HIP build of the CUDA op of the same name."""
    _module().eagle_harness_work_deltas(
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
    """HIP build of the CUDA op of the same name."""
    _module().eagle_harness_draft_depth_update(
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
    """HIP build of the CUDA op of the same name."""
    _module().eagle_harness_stable_topk_desc(
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
    """HIP build of the CUDA op of the same name."""
    _module().eagle_harness_write_verify_tree(
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
    """HIP build of the CUDA op of the same name."""
    _module().eagle_harness_sample_children(
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
    """HIP build of the CUDA op of the same name."""
    _module().eagle_harness_tree_reject_sample(
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
    """HIP build of the CUDA op of the same name."""
    _module().eagle_harness_residual_sample(
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
    """HIP build of the CUDA op of the same name."""
    _module().eagle_harness_publish_draft_q(
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
