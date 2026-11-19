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

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.forward_step import (
    DecodeLadderArgs,
    ForwardStepRunner,
    get_batch_sizes_to_capture,
)
from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout


def test_drafter_refuses_an_unsupported_local_graph_scope():
    drafter = BaseDrafter(spec_num_tokens=6)
    drafter.configure_decode_graph_scope(local=False)
    with pytest.raises(ValueError, match="enable-dp-local-cuda-graph"):
        drafter.configure_decode_graph_scope(local=True)
    assert not drafter.dp_local_graph


def _runner(
    *,
    local: bool,
    capture_bs: list[int],
    disable_padding: bool = False,
    verify_width: int = 1,
):
    # Avoid constructing model/cache/GPU resources; exercise the real routing
    # methods with the runner state those methods read.
    runner = ForwardStepRunner.__new__(ForwardStepRunner)
    runner.disable = False
    runner.dp_size = 4
    runner.disable_padding = disable_padding
    runner.dp_local_graph = local
    runner.capture_bs = capture_bs
    runner.max_capture_bs = max(capture_bs)
    runner.graphs = {("default", bs) for bs in capture_bs}
    runner.max_tokens_per_req = verify_width
    return runner


def _context(
    local_bs: int,
    global_bs: list[int],
    *,
    all_decode_or_idle: bool = True,
    verify_width: int = 1,
) -> ForwardContext:
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=local_bs,
        num_extends=0,
        output_layout=ForwardOutputLayout(0, 0, local_bs, verify_width),
        input_num_tokens=local_bs * verify_width,
        forward_mode=ForwardMode.DECODE,
        global_num_tokens=[bs * verify_width for bs in global_bs],
        all_decode_or_idle=all_decode_or_idle,
    )


def test_global_graph_padding_uses_max_dp_batch_for_every_rank():
    runner = _runner(local=False, capture_bs=[1, 2, 4, 8])
    global_bs = [8, 5, 2, 0]

    for local_bs in global_bs:
        ctx = _context(local_bs, global_bs)
        assert runner.can_run(local_bs, ctx)
        assert runner.padded_bs(local_bs, ctx) == 8


@pytest.mark.parametrize("local_bs,expected", [(8, 8), (5, 8), (2, 2), (0, 1)])
@pytest.mark.parametrize("verify_width", [1, 6])
def test_local_graph_padding_rounds_up_each_rank_and_idle_uses_smallest(
    local_bs, expected, verify_width
):
    runner = _runner(local=True, capture_bs=[1, 2, 4, 8], verify_width=verify_width)
    ctx = _context(local_bs, [8, 5, 2, 0], verify_width=verify_width)

    assert runner.can_run(local_bs, ctx)
    assert runner.padded_bs(local_bs, ctx) == expected


def test_mixed_step_and_global_batch_over_ladder_force_the_whole_group_eager():
    runner = _runner(local=True, capture_bs=[1, 2, 4, 8])

    mixed_ctx = _context(2, [8, 5, 2, 0], all_decode_or_idle=False)
    assert not runner.can_run(2, mixed_ctx)

    oversized_ctx = _context(9, [8, 5, 9, 0])
    assert not runner.can_run(9, oversized_ctx)


@pytest.mark.parametrize("disable_padding", [False, True])
def test_idle_respects_the_smallest_explicit_capture_bucket(disable_padding):
    sizes = get_batch_sizes_to_capture(
        DecodeLadderArgs(
            cudagraph_capture_sizes=[4, 8],
            max_num_seqs=32,
            data_parallel_size=4,
            disable_cuda_graph_padding=disable_padding,
            max_cudagraph_capture_size=8,
        )
    )
    runner = _runner(
        local=True, capture_bs=sizes, disable_padding=disable_padding, verify_width=6
    )
    ctx = _context(0, [8, 0, 0, 0], verify_width=6)

    assert sizes == [4, 8]
    assert runner.can_run(0, ctx)
    assert runner.padded_bs(0, ctx) == 4


def test_disable_padding_requires_every_nonzero_peer_graph_to_be_captured():
    runner = _runner(local=True, capture_bs=[1, 2], disable_padding=True)
    global_bs = [4, 2, 0, 0]

    for local_bs in global_bs:
        assert not runner.can_run(local_bs, _context(local_bs, global_bs))
