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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tokenspeed.runtime.execution.forward_step import ForwardStepRunner
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
    DeepseekV41AttentionBackend,
)


def v41():
    return object.__new__(DeepseekV41AttentionBackend)


def composite(*children):
    node = SimpleNamespace(child_backends=lambda: children)
    node.npu_graph_requires_host_lengths = lambda: (
        AttentionBackend.npu_graph_requires_host_lengths(node)
    )
    return node


@pytest.mark.parametrize("draft", [None, v41(), composite(v41(), v41())])
def test_device_lengths_do_not_sync_or_update(draft):
    runner = SimpleNamespace(attn_backend=v41(), draft_attn_backend=draft)
    lengths = Mock()
    lengths.to.side_effect = AssertionError("device lengths must stay on device")
    graph = Mock()
    ForwardStepRunner._update_npu_graph_host_lengths(runner, graph, lengths)
    graph.update.assert_not_called()


@pytest.mark.parametrize(
    "target,draft",
    [
        (composite(), None),
        (v41(), composite()),
        (composite(v41(), composite()), v41()),
    ],
)
def test_unknown_attention_keeps_host_update(target, draft):
    runner = SimpleNamespace(attn_backend=target, draft_attn_backend=draft)
    graph = Mock()
    for values in ([1, 65537, 51200], [9, 3, 51000]):
        ForwardStepRunner._update_npu_graph_host_lengths(
            runner, graph, torch.tensor(values, dtype=torch.int32)
        )
        graph.update.assert_called_with(
            cpu_update_input=[{"actual_seq_lengths_kv": values}]
        )
