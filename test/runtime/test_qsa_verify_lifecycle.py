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

"""Check commit order, live acceptance and failure handling after eager or replay.

Exercise execution modes with all consumers, then each optional consumer alone.
"""

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.forward_step import ForwardStepRunner
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
    HybridLinearAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp import (
    Qwen4ExpBackend,
)


def _runner(
    *,
    use_graph: bool,
    has_drafter: bool,
    consumers: tuple[str, ...],
    fail_forward: bool,
):
    events = []
    commits = {"recurrent": [], "ple": [], "qsa": []}
    wrapper = object.__new__(ForwardStepRunner)
    wrapper.config = SimpleNamespace(spec_algo="MTP", max_req_pool_size=8)
    wrapper.device = "cpu"
    full_backend = object.__new__(AttentionBackend)
    full_backend.device = "cpu"

    def commit_recurrent(accepted_lengths):
        events.append("recurrent")
        commits["recurrent"].append(accepted_lengths.tolist())

    def commit_ple(accepted_lengths):
        events.append("ple")
        commits["ple"].append(accepted_lengths.tolist())

    def commit_qsa(accepted_lengths, *, num_extends):
        events.append("qsa")
        commits["qsa"].append((accepted_lengths.tolist(), num_extends))

    def config(is_draft):
        return SimpleNamespace(
            device="cpu",
            dtype=torch.bfloat16,
            is_draft=is_draft,
            speculative_num_draft_tokens=4,
            component=lambda component_type: SimpleNamespace(
                num_attention_heads=1, num_kv_heads=1, attn_tp_size=1, head_dim=8
            ),
        )

    wrapper.attn_backend = Qwen4ExpBackend(
        config=config(False),
        attention_backend=(
            HybridLinearAttnBackend(
                full_backend,
                SimpleNamespace(commit_verified_state=commit_recurrent),
                [1, 3],
            )
            if "recurrent" in consumers
            else full_backend
        ),
        ple_backend=(
            SimpleNamespace(commit_verified_state=commit_ple)
            if "ple" in consumers
            else None
        ),
        indexer_backend=(
            SimpleNamespace(commit_after_mtp_verify=commit_qsa)
            if "qsa" in consumers
            else None
        ),
    )
    draft_backend = Qwen4ExpBackend(
        config=config(True),
        attention_backend=full_backend,
        ple_backend=None,
        indexer_backend=SimpleNamespace(
            commit_after_mtp_verify=lambda *args, **kwargs: events.append("draft_qsa")
        ),
    )
    wrapper.drafter = (
        SimpleNamespace(attn_backend=draft_backend) if has_drafter else None
    )
    wrapper.max_tokens_per_req = 4
    wrapper.input_buffers = SimpleNamespace(
        req_pool_indices_buf=torch.tensor([2, 3, 0, 0], dtype=torch.int32),
        seq_lens_buf=torch.tensor([10, 10, 1, 1], dtype=torch.int32),
        state_write_req_pool_indices_buf=torch.zeros(4, dtype=torch.int32),
    )
    wrapper.token_to_kv_pool = SimpleNamespace(
        arena=SimpleNamespace(cache_group_specs=())
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

    wrapper._forward_func = lambda **kwargs: execute()
    wrapper.graphs = {4: SimpleNamespace(replay=execute)}
    wrapper.output_buffers = {4: result}
    return wrapper, events, commits


def _run(wrapper, mode):
    ctx = SimpleNamespace(
        attn_backend=wrapper.attn_backend,
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
    "mode,use_graph,has_drafter,consumers",
    [
        (ForwardMode.DECODE, False, True, ("recurrent", "ple", "qsa")),
        (ForwardMode.DECODE, True, True, ("recurrent", "ple", "qsa")),
        (ForwardMode.MIXED, False, True, ("recurrent", "ple", "qsa")),
        (ForwardMode.EXTEND, False, True, ("recurrent", "ple", "qsa")),
        (ForwardMode.DECODE, False, False, ("recurrent", "ple", "qsa")),
        (ForwardMode.DECODE, False, True, ("recurrent",)),
        (ForwardMode.DECODE, True, True, ("ple",)),
        (ForwardMode.DECODE, True, True, ("qsa",)),
        (ForwardMode.DECODE, False, True, ()),
        (ForwardMode.MIXED, False, True, ("ple",)),
    ],
)
def test_runner_commits_live_acceptance_once_after_execution(
    mode, use_graph, has_drafter, consumers
):
    wrapper, events, commits = _runner(
        use_graph=use_graph,
        has_drafter=has_drafter,
        consumers=consumers,
        fail_forward=False,
    )
    _run(wrapper, mode)
    expected_qsa = (
        [([3, 1], int(mode.is_mixed()))]
        if has_drafter
        and "qsa" in consumers
        and mode in (ForwardMode.DECODE, ForwardMode.MIXED)
        else []
    )
    assert commits["qsa"] == expected_qsa
    assert events[:2] == ["metadata", "execute"]
    assert "draft_qsa" not in events
    verifies_decode = has_drafter and mode.is_decode()
    expected_commits = []
    for consumer in ("recurrent", "ple"):
        active = verifies_decode and consumer in consumers
        assert commits[consumer] == ([[3, 1]] if active else [])
        if active:
            expected_commits.append(consumer)
    if expected_qsa:
        expected_commits.append("qsa")
    assert events[2:] == expected_commits


@pytest.mark.parametrize("use_graph", [False, True])
def test_failed_execution_does_not_commit_stale_staging(use_graph):
    wrapper, events, commits = _runner(
        use_graph=use_graph,
        has_drafter=True,
        consumers=("recurrent", "ple", "qsa"),
        fail_forward=True,
    )
    with pytest.raises(RuntimeError, match="forward failed"):
        _run(wrapper, ForwardMode.DECODE)
    assert events == ["metadata", "execute"]
    assert commits == {"recurrent": [], "ple": [], "qsa": []}
