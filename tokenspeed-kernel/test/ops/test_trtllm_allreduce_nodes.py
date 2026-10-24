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

import os

import pytest
import torch.distributed as dist
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="the TRT-LLM all-reduce is NVIDIA-only"
)


def _local_key() -> tuple[int, int]:
    from tokenspeed_kernel.ops.communication.fabric import host_identity

    return host_identity(), os.stat("/proc/self/ns/ipc").st_ino


def _spans(monkeypatch, peers: list[tuple[int, int]]) -> tuple[bool, object]:
    """``group_spans_nodes`` when the group's other ranks report ``peers``."""
    from tokenspeed_kernel.ops.communication import trtllm

    sent = []

    def all_gather_object(out, obj, group=None):
        sent.append(obj)
        out[:] = [obj, *peers]

    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 1 + len(peers))
    monkeypatch.setattr(dist, "all_gather_object", all_gather_object)
    return trtllm.group_spans_nodes(object()), sent[0]


def test_ranks_gather_their_host_and_ipc_namespace(monkeypatch) -> None:
    spans, sent = _spans(monkeypatch, [_local_key()])

    assert sent == _local_key()
    assert not spans


def test_a_peer_on_another_machine_spans_nodes(monkeypatch) -> None:
    # IPC namespace inodes are numbered per machine, so they can coincide.
    host, ipc = _local_key()

    assert _spans(monkeypatch, [(host + 1, ipc)])[0]


def test_a_peer_in_another_ipc_namespace_spans_nodes(monkeypatch) -> None:
    host, ipc = _local_key()

    assert _spans(monkeypatch, [(host, ipc + 1)])[0]


def test_no_process_group_is_one_node(monkeypatch) -> None:
    from tokenspeed_kernel.ops.communication import trtllm

    def uninitialized(group=None):
        raise ValueError("Default process group has not been initialized")

    monkeypatch.setattr(dist, "get_world_size", uninitialized)

    assert not trtllm.group_spans_nodes(None)
