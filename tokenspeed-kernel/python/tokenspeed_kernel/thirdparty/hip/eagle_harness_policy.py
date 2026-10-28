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

"""EAGLE-Harness device policy kernels (EAGLE-4), HIP build; see
``csrc/eagle_harness_policy.hip``. Each op has the contract of the CUDA build's
``tokenspeed_kernel.thirdparty.cuda.eagle_harness_policy`` op of the same name.

Every kernel is one block reading its sizes from device memory, so each launch
can be captured in a HIP graph once and replayed for any round."""

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
            "Run: pip install -e tokenspeed-kernel/python/ (ROCm backend)"
        )
    return tvm_ffi.load_module(str(so_path))


def eagle_harness_policy_load() -> None:
    """HIP build of the CUDA op of the same name."""
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
    """HIP build of the CUDA op of the same name."""
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
    """HIP build of the CUDA op of the same name."""
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
    """HIP build of the CUDA op of the same name."""
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
    """HIP build of the CUDA op of the same name."""
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
