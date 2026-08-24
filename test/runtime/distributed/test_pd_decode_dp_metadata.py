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

"""CPU-only tests for EventLoop._dp_sync_and_check's per-rank metadata.

The all_gather is faked (each fake rank's local row is written into the
global tensor directly), so the tests pin down what each rank REPORTS and
how the gathered rows are interpreted — in particular the decode-side PD
rule: an EXTEND op that only starts remote KV receive is reported as IDLE,
so idle DP ranks don't enter dummy collectives the active rank won't match.
"""

from __future__ import annotations

import torch
import torch.distributed

from tokenspeed.runtime.engine.event_loop import EventLoop
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.pd.decode_executor import DisaggDecodeExecutor


class FakeForwardOp:
    def __init__(
        self, *, input_lengths, request_ids=None, num_extends=0, local_prefill=False
    ):
        self.input_lengths = input_lengths
        self.request_ids = request_ids or [
            f"req-{i}" for i in range(len(input_lengths))
        ]
        self._num_extends = num_extends
        self._local_prefill = local_prefill

    def num_extends(self):
        return self._num_extends

    def is_local_prefill(self):
        return self._local_prefill


class _FakeLoop:
    """Only the state read by ``EventLoop._dp_sync_and_check``."""

    def __init__(self, *, world_size=1, disagg_decode=False):
        self.kv_transfer = (
            object.__new__(DisaggDecodeExecutor) if disagg_decode else None
        )
        self._dp_local_info = torch.zeros(1, 3, dtype=torch.int32)
        self._dp_global_info = torch.zeros(world_size, 3, dtype=torch.int32)
        self.world_cpu_group = None


def _sync(loop, forward_op, monkeypatch, other_rank_rows=()):
    """Run _dp_sync_and_check with the collective replaced by direct writes:
    this rank's row lands in slot 0, ``other_rank_rows`` fill the rest."""

    def fake_gather(global_info, local_info, group=None):
        global_info[0] = local_info[0]
        for i, row in enumerate(other_rank_rows, start=1):
            global_info[i] = torch.tensor(row, dtype=torch.int32)

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", fake_gather)
    return EventLoop._dp_sync_and_check(loop, forward_op)


def test_pd_decode_extend_only_is_reported_idle(monkeypatch):
    # Decode-side PD EXTEND starts KV receive only; no model collectives run on
    # the active DP rank yet, so it must report IDLE (0 tokens) to its peers.
    loop = _FakeLoop(disagg_decode=True)
    op = FakeForwardOp(input_lengths=[17], num_extends=1)

    meta = _sync(loop, op, monkeypatch)

    assert meta.global_num_tokens == [0]
    assert meta.global_forward_mode == [int(ForwardMode.IDLE)]
    assert not meta.need_idle_forward


def test_pd_decode_decode_step_is_reported_as_decode(monkeypatch):
    loop = _FakeLoop(disagg_decode=True)
    op = FakeForwardOp(input_lengths=[1], num_extends=0)

    meta = _sync(loop, op, monkeypatch)

    assert meta.global_num_tokens == [1]
    assert meta.global_forward_mode == [int(ForwardMode.DECODE)]
    assert meta.all_decode_or_idle


def test_pd_decode_local_recovery_prefill_is_model_work(monkeypatch):
    loop = _FakeLoop(disagg_decode=True)
    op = FakeForwardOp(input_lengths=[17], num_extends=1, local_prefill=True)

    meta = _sync(loop, op, monkeypatch)

    assert meta.global_num_tokens == [17]
    assert meta.global_forward_mode == [int(ForwardMode.EXTEND)]
    assert meta.all_extend


def test_non_pd_extend_is_model_work(monkeypatch):
    loop = _FakeLoop()
    op = FakeForwardOp(input_lengths=[17], num_extends=1)

    meta = _sync(loop, op, monkeypatch)

    assert meta.global_num_tokens == [17]
    assert meta.global_forward_mode == [int(ForwardMode.EXTEND)]


def test_zero_token_forward_op_is_not_model_work(monkeypatch):
    loop = _FakeLoop()
    op = FakeForwardOp(input_lengths=[0], num_extends=1)

    meta = _sync(loop, op, monkeypatch)

    assert meta.global_num_tokens == [0]
    assert meta.global_forward_mode == [int(ForwardMode.IDLE)]


def test_idle_rank_joins_dummy_forward_only_when_a_peer_has_work(monkeypatch):
    # Two ranks: this one idle, the peer running a 4-token decode batch.
    loop = _FakeLoop(world_size=2)
    busy_peer = (4, 4, int(ForwardMode.DECODE))

    meta = _sync(loop, None, monkeypatch, other_rank_rows=[busy_peer])

    assert meta.need_idle_forward
    assert meta.all_decode_or_idle
    assert meta.global_num_tokens == [0, 4]

    # Fully idle world: nothing to keep in lockstep with.
    idle_peer = (0, 0, int(ForwardMode.IDLE))
    meta = _sync(loop, None, monkeypatch, other_rank_rows=[idle_peer])
    assert not meta.need_idle_forward
