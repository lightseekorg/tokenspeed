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

"""Regression test for the MTP draft NaN under CUDA graph (DP + MoE).

Root cause: the draft first-step MoE all-gather (``draft_first_step_reduce``)
sizes its TritonRSAG collective from ``ctx.global_bs``. The CUDA-graph capture
path (``CudaGraphWrapper._capture_one``) set ``ctx.global_num_tokens`` to a
uniform dummy but left ``ctx.global_bs`` as ``None``. With ``global_bs is None``
the draft MoE scattered-token-count helper falls back to a *single-rank* layout
(only the local rank has tokens), whereas at replay ``ctx.global_bs`` is the live
per-rank batch list and yields a *multi-rank* layout. Because the RSAG kernel's
offsets/launch are frozen at capture, the captured single-rank layout no longer
matches the replayed multi-rank layout and the gather reads uninitialized
symmetric memory -> NaN draft logits -> ``accept_rate`` collapses to 0.

The fix sets ``ctx.global_bs`` at capture the same way ``global_num_tokens`` is
set, so the captured layout matches the replayed layout.

These tests are pure-Python (no GPU / NCCL): they assert the layout invariant
that the fix restores.
"""

import pytest

from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext, ForwardMode


def _make_mapping(rank: int = 0) -> Mapping:
    # Mirrors the repro config: DP=4, dense-tp=4, moe-tp=1, ep=4 on a 4-GPU node.
    return Mapping(
        rank=rank,
        world_size=4,
        attn_dp_size=4,
        dense_tp_size=4,
        moe_tp_size=1,
        moe_ep_size=4,
    )


def _draft_first_step_ctx(bs: int, global_bs, global_num_tokens) -> ForwardContext:
    """A draft first-step decode ctx (draft_first_step_reduce=True)."""
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=bs,
        num_extends=0,
        input_num_tokens=bs,
        forward_mode=ForwardMode.DECODE,
        draft_first_step_reduce=True,
        global_num_tokens=global_num_tokens,
        global_bs=global_bs,
    )


def test_capture_global_bs_none_diverges_from_replay():
    """Documents the bug: global_bs=None at capture -> single-rank layout that
    does NOT match the multi-rank layout produced from a live global_bs."""
    cm = CommManager(mapping=_make_mapping(), layer_id=0, is_moe=True, prev_is_moe=True)
    bs = 1

    # Pre-fix capture: global_num_tokens set (uniform dummy), global_bs left None.
    capture_buggy = _draft_first_step_ctx(
        bs, global_bs=None, global_num_tokens=[bs] * 4
    )
    # Replay: live per-rank batch sizes (uniform across the padded DP bucket).
    replay = _draft_first_step_ctx(bs, global_bs=[bs] * 4, global_num_tokens=[bs] * 4)

    buggy = cm.moe_tp_ep_group_scattered_num_tokens(capture_buggy)
    live = cm.moe_tp_ep_group_scattered_num_tokens(replay)

    # global_bs=None takes the single-rank fallback (only local rank populated)...
    assert buggy == [1, 0, 0, 0]
    # ...which differs from the replay multi-rank layout -> the NaN-causing mismatch.
    assert buggy != live


def test_capture_global_bs_set_matches_replay():
    """The fix: setting global_bs at capture (same as global_num_tokens) makes the
    captured draft MoE all-gather layout identical to the replayed one."""
    cm = CommManager(mapping=_make_mapping(), layer_id=0, is_moe=True, prev_is_moe=True)
    bs = 1

    capture_fixed = _draft_first_step_ctx(
        bs, global_bs=[bs] * 4, global_num_tokens=[bs] * 4
    )
    replay = _draft_first_step_ctx(bs, global_bs=[bs] * 4, global_num_tokens=[bs] * 4)

    assert cm.moe_tp_ep_group_scattered_num_tokens(
        capture_fixed
    ) == cm.moe_tp_ep_group_scattered_num_tokens(replay)


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_capture_matches_replay_all_ranks(rank: int):
    """For every DP rank, the fixed capture layout matches replay (the scattered
    counts for the rank's MoE tp_ep group are identical)."""
    cm = CommManager(
        mapping=_make_mapping(rank), layer_id=0, is_moe=True, prev_is_moe=True
    )
    bs = 1
    ctx = _draft_first_step_ctx(bs, global_bs=[bs] * 4, global_num_tokens=[bs] * 4)
    scattered = cm.moe_tp_ep_group_scattered_num_tokens(ctx)
    # moe tp_ep group spans all 4 ranks (moe_tp=1 * ep=4); each contributes bs.
    assert scattered == [bs] * 4


def test_capture_one_sets_global_bs(monkeypatch):
    """Guards the fix at its source: CudaGraphWrapper._capture_one must set
    ctx.global_bs (not leave it None) for DP, matching how global_num_tokens is
    set. Verified by inspecting the source so the test needs no GPU/capture."""
    import inspect

    from tokenspeed.runtime.execution import cuda_graph_wrapper

    src = inspect.getsource(cuda_graph_wrapper.CudaGraphWrapper._capture_one)
    # Both DP token-metadata fields must be assigned in the capture path.
    assert "ctx.global_num_tokens" in src
    assert "ctx.global_bs" in src
