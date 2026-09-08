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

"""Verification participants run after execution, outside the backend tree."""

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.forward_step import ForwardStepRunner


def _runner(*, use_graph: bool, has_drafter: bool, fail_forward: bool):
    events = []
    commits = []
    wrapper = object.__new__(ForwardStepRunner)
    wrapper.config = SimpleNamespace(spec_algo="MTP", max_req_pool_size=8)
    wrapper.device = "cpu"
    wrapper.drafter = object() if has_drafter else None
    wrapper.max_tokens_per_req = 4
    wrapper.input_buffers = SimpleNamespace(
        req_pool_indices_buf=torch.tensor([2, 3, 0, 0], dtype=torch.int32),
        seq_lens_buf=torch.tensor([10, 10, 1, 1], dtype=torch.int32),
        state_write_req_pool_indices_buf=torch.zeros(4, dtype=torch.int32),
    )
    wrapper.token_to_kv_pool = SimpleNamespace(
        arena=SimpleNamespace(cache_group_specs=())
    )
    # The backend deliberately has no indexer verification hook.
    wrapper.attn_backend = SimpleNamespace(
        update_mamba_state_after_mtp_verify=lambda lengths: events.append("recurrent")
    )
    wrapper._can_use_graph = lambda bs, ctx: use_graph
    wrapper._padded_bs = lambda bs, ctx: 4
    wrapper._prepare_decode_metadata = lambda *args, **kwargs: events.append("metadata")
    wrapper._init_forward_metadata = lambda *args, **kwargs: events.append("metadata")
    wrapper._cuda_graph_key = lambda bs: bs
    wrapper._graph_debug = False
    wrapper.deepep_adapter = SimpleNamespace(replay=lambda: None)
    result = (
        torch.arange(16, dtype=torch.int32),
        torch.tensor([3, 1, 99, 99], dtype=torch.int32),
        None,
    )

    def execute():
        events.append("execute")
        if fail_forward:
            raise RuntimeError("forward failed")
        return result[0][:8], result[1][:2], None

    def commit(accepted_lengths, *, num_extends):
        events.append("commit")
        commits.append((accepted_lengths.tolist(), num_extends))

    wrapper._forward_func = lambda **kwargs: execute()
    wrapper.graphs = {4: SimpleNamespace(replay=execute)}
    wrapper.output_buffers = {4: result}
    wrapper.speculative_states = (SimpleNamespace(commit_after_verify=commit),)
    return wrapper, events, commits


def _run(wrapper, mode):
    ctx = SimpleNamespace(
        bs=2,
        num_extends=1 if mode.is_mixed() else (2 if mode.is_extend() else 0),
        forward_mode=mode,
        global_num_tokens=None,
        all_decode_or_idle=mode.is_decode(),
        capture_hidden_mode=None,
        input_num_tokens=8,
    )
    empty = torch.empty(0, dtype=torch.int32)
    result = wrapper(
        2,
        ctx,
        None,
        extend_with_prefix=False,
        extend_prefix_lens=empty,
        extend_prefix_lens_cpu=empty,
        extend_seq_lens=empty,
        extend_seq_lens_cpu=empty,
        positions=None,
        block_tables={},
    )
    assert ctx.bs == 2
    return result


@pytest.mark.parametrize(
    "mode,use_graph,has_drafter,expected",
    [
        (ForwardMode.DECODE, False, True, [([3, 1], 0)]),
        (ForwardMode.DECODE, True, True, [([3, 1], 0)]),
        (ForwardMode.MIXED, False, True, [([3, 1], 1)]),
        (ForwardMode.EXTEND, False, True, []),
        (ForwardMode.DECODE, False, False, []),
    ],
)
def test_runner_commits_live_acceptance_once_after_execution(
    mode, use_graph, has_drafter, expected
):
    wrapper, events, commits = _runner(
        use_graph=use_graph, has_drafter=has_drafter, fail_forward=False
    )
    _run(wrapper, mode)
    assert commits == expected
    assert events[:2] == ["metadata", "execute"]
    if expected:
        assert events[-1] == "commit"
        assert events.count("commit") == 1


@pytest.mark.parametrize("use_graph", [False, True])
def test_failed_execution_does_not_commit_stale_staging(use_graph):
    wrapper, events, commits = _runner(
        use_graph=use_graph, has_drafter=True, fail_forward=True
    )
    with pytest.raises(RuntimeError, match="forward failed"):
        _run(wrapper, ForwardMode.DECODE)
    assert events == ["metadata", "execute"]
    assert commits == []
