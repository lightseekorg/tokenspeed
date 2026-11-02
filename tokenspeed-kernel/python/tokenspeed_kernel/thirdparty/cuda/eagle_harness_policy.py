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

"""EAGLE-Harness device policy kernels (EAGLE-4); see
``csrc/eagle_harness_policy.cu`` for the state and slot layouts, and the
runtime's ``eagle_harness/device_policy.py`` for the tables and call order.

Every kernel is one CTA reading its sizes from device memory, so each launch
can be captured in a CUDA graph once and replayed for any round. Tables and
state are contiguous CUDA tensors; ``frontier`` and the rows of
``top_scores`` may be strided."""

import functools
from pathlib import Path

import torch

__all__ = [
    "eagle_harness_policy_load",
    "eagle_harness_policy_begin_round",
    "eagle_harness_policy_decide_chain",
    "eagle_harness_policy_decide_tree",
    "eagle_harness_policy_select_verify",
]


@functools.cache
def _module():
    import tvm_ffi

    so_path = (
        Path(__file__).parent
        / "objs"
        / "eagle_harness_policy"
        / "eagle_harness_policy.so"
    )
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel eagle_harness_policy library not found at {so_path}. "
            "Run: pip install -e tokenspeed_kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def eagle_harness_policy_load() -> None:
    """Load the kernel library (no launch), before a graph capture uses it."""
    _module()


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
    """Start a round of ``live_rows[0]`` requests (int32; 0 runs no round)
    whose cached lengths are the first ``live_rows[0]`` entries of
    ``frontier`` (int64 ``[rows >= live_rows[0]]``, summed in the kernel) and
    which draw their children when ``sampled`` (bool):
    increments ``seq`` (int64 ``[1]``), clears the round's scalars of
    ``state`` and the ``[4, depth + 1]`` int32 ``slots`` (width, cumu_start,
    work_size, go per depth), writes depth 0's slots, takes depth 0's
    decision and writes depth 1's, mirroring ``go[1]`` into the pinned
    ``go_host`` (int64 ``[depth + 1]``, ``(seq << 1) | go``); a chain
    round whose ``batch_table`` row holds a width table width ``W`` instead
    writes every depth's slots (depths 1..W-2 draft) and go flags, up to
    ``go_host[max(W - 1, 2)]``, and its counts (verify ``W``). ``hist``
    (``[max_bs, rounds]`` int32, at most 32 rounds) holds the past verify
    widths, ``tier_table`` (int32 ``[9]``) the verify floor per cost variant
    (four tiers, then tiny) then the width per tier (empty for a chain),
    ``batch_table`` (int64 ``[max_bs, 5]``, ``BatchCol``) the cost data,
    interval count, step and chain width table width (0: the confidence
    rule decides) per batch size. The round's row of ``cost_f64``
    (a tree's ``interval_f64``, a chain's ``chain_f64``) is copied to
    ``round_f64`` for the decisions (``RoundF64`` for a tree, the chain
    entries per depth for a chain; depth at most 32). A chain's
    past verify width is ``chain_past_verify_num``. ``verify_num`` and
    ``min_verify_count`` size the ranked node count when a decision stops
    the round (``eagle_harness_policy_decide_tree``).
    """
    _module().eagle_harness_policy_begin_round(
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
    """A tree round's decision after ``depth`` (``1 <= depth < tree_depth``),
    on the work deltas (contiguous fp32 ``[node_top_k + 1]``) queued after
    ``depth - 1``:
    writes depth ``depth + 1``'s slots and mirrors its go flag; once the round
    stopped, only clears that go flag. ``round_f64`` holds the round's cost
    row (``eagle_harness_policy_begin_round``). A stopping decision also
    writes the round's ranked node count, ``min(verify_num, pool size,
    max(2 * past verify width, min_verify_count))``, its flatness (a
    chain's: every ranked node, ``verify_num`` at most, verified) and, in a
    sampled round, the next sampled round's planned stop depth."""
    _module().eagle_harness_policy_decide_tree(
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
    """A chain round's decision after ``depth``, as
    ``eagle_harness_policy_decide_tree`` on contiguous fp32 ``[2]`` work
    deltas, with the round's stop terms per depth in ``round_f64``. A width
    table round's decision changes no state or slot: past its last drafted
    depth it only clears ``go_host[depth + 1]``."""
    _module().eagle_harness_policy_decide_chain(
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
    """A tree round's cumulative ranked scores and verify width. The scores
    of ``t = top_scores[:live rows, :verify count]`` (fp32, unit column
    stride, at most 32 rows and 64 columns) are, bit for bit,
    ``torch.cumsum(t.mean(dim=0), dim=0)``, or ``torch.cumsum(t.view(-1),
    dim=0)`` for a flat round, on the contiguous ``t``. ``published`` (fp32)
    holds the previous round's, which size the verify (this round's in the
    first round, none for a sampled first round), and gets this round's for
    the next before the verify is sized; the width is booked into ``hist``
    and the ``recent`` widths. ``verify_i32`` / ``verify_f64`` hold per batch
    size and interval the tier and the stability thresholds; ``interval_f64``
    the cost deltas after the stop thresholds."""
    _module().eagle_harness_policy_select_verify(
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
