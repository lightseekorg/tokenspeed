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

import logging
import threading

from tokenspeed.runtime.disaggregation.base.poll import TransferPoll
from tokenspeed.runtime.disaggregation.kv.mooncake.prefill import (
    MooncakeKVManagerPrefill,
)
from tokenspeed.runtime.disaggregation.kv.mooncake.sender import MooncakeKVSender

# The sender's timeout path calls ``logger.warning_once`` -- a method that only
# exists because ``transformers`` monkeypatches ``logging.Logger`` at import
# time. This focused test does not import transformers, so define the same no-op
# fallback to mirror that production guarantee; without it poll()'s timeout
# branch raises AttributeError before it can conclude Failed.
if not hasattr(logging.Logger, "warning_once"):
    logging.Logger.warning_once = lambda self, *args, **kwargs: None

# --- (a) The KV sender's Bootstrapping  was dead code: init_time was
# left None and never assigned, so poll()'s `if self.init_time is not None`
# branch never ran and a decode peer that never registered kept the room
# Bootstrapping forever. The fix sets init_time at registration (__init__). ---


class _FakeKVMgr:
    def __init__(self, bootstrap_time_out=120.0, waiting_timeout=300.0):
        self.request_status = {}
        self.failure_records = {}
        self.failure_lock = threading.Lock()
        self.bootstrap_time_out = bootstrap_time_out
        self.waiting_timeout = waiting_timeout
        self.failures = []

    def update_status(self, room, status):
        if room not in self.request_status:
            self.request_status[room] = status
        elif status == TransferPoll.Failed:
            self.request_status[room] = TransferPoll.Failed
        else:
            self.request_status[room] = max(self.request_status[room], status)

    def check_status(self, room):
        return self.request_status[room]

    def record_failure(self, room, reason):
        self.failures.append((room, reason))
        with self.failure_lock:
            self.failure_records[room] = reason


def test_sender_sets_init_time_at_registration():
    # Regression guard for the dead-code fix: init_time must be a real timestamp
    # right after construction (it used to stay None forever).
    mgr = _FakeKVMgr()
    s = MooncakeKVSender(mgr, "bs:1", 7)
    assert s.init_time is not None


def test_sender_bootstrap_timeout_fires_when_peer_never_registers():
    mgr = _FakeKVMgr(bootstrap_time_out=120.0)
    s = MooncakeKVSender(mgr, "bs:1", 7)
    # Fresh: still within the window -> Bootstrapping, no failure recorded.
    assert s.poll() == TransferPoll.Bootstrapping
    assert mgr.failures == []
    # Simulate the bootstrap window elapsing with the peer still absent.
    s.init_time -= mgr.bootstrap_time_out + 1
    assert s.poll() == TransferPoll.Failed
    assert mgr.failures and mgr.failures[0][0] == 7
    # conclude_state caches the terminal verdict.
    assert s.poll() == TransferPoll.Failed


def test_sender_concludes_success_without_timeout():
    mgr = _FakeKVMgr()
    s = MooncakeKVSender(mgr, "bs:1", 9)
    mgr.request_status[9] = TransferPoll.Success
    assert s.poll() == TransferPoll.Success
    assert s.poll() == TransferPoll.Success  # cached
    assert mgr.failures == []


# --- Post-bootstrap deadline: once the decode pre-allocated (room -> Bootstrapped),
# the Bootstrapping timeout no longer applies. A decode that stalls after pre-alloc
# but before Success must still conclude Failed, on a SEPARATE timer (not init_time). ---


def test_sender_post_bootstrap_timeout_fires_when_transfer_stalls():
    mgr = _FakeKVMgr(waiting_timeout=300.0)
    s = MooncakeKVSender(mgr, "bs:1", 11)
    mgr.request_status[11] = TransferPoll.Bootstrapped  # decode pre-allocated
    # First Bootstrapped poll lazily stamps the deadline; still within window.
    assert s.poll() == TransferPoll.Bootstrapped
    assert s.bootstrapped_time is not None
    assert mgr.failures == []
    # Elapse the post-bootstrap window with the transfer still not Success.
    s.bootstrapped_time -= mgr.waiting_timeout + 1
    assert s.poll() == TransferPoll.Failed
    assert mgr.failures and mgr.failures[0][0] == 11
    assert s.poll() == TransferPoll.Failed  # conclude_state cached


def test_sender_bootstrapped_does_not_reuse_init_time():
    # The deadline must measure from the FIRST Bootstrapped poll, not from
    # construction. A request that spent the whole Bootstrapping window then
    # advanced to Bootstrapped must NOT immediately fail (the premature-abort trap
    # of reusing init_time).
    mgr = _FakeKVMgr(bootstrap_time_out=120.0, waiting_timeout=300.0)
    s = MooncakeKVSender(mgr, "bs:1", 12)
    s.init_time -= mgr.bootstrap_time_out + 1  # bootstrapping epoch long elapsed
    mgr.request_status[12] = TransferPoll.Bootstrapped
    assert s.poll() == TransferPoll.Bootstrapped  # fresh post-bootstrap clock, no abort
    assert mgr.failures == []


def test_sender_bootstrapped_then_success_no_spurious_timeout():
    mgr = _FakeKVMgr(waiting_timeout=300.0)
    s = MooncakeKVSender(mgr, "bs:1", 13)
    mgr.request_status[13] = TransferPoll.Bootstrapped
    assert s.poll() == TransferPoll.Bootstrapped  # stamps the deadline
    mgr.request_status[13] = TransferPoll.Success  # transfer completes within window
    assert s.poll() == TransferPoll.Success
    assert mgr.failures == []


# --- (c) A fault in the prefill transfer_worker loop used to `raise`, killing
# the daemon thread and leaving its FastQueue shard with no consumer so every
# room on that shard wedged forever. The fix routes the fault through
# _fail_transfer_chunk: conclude the room Failed, notify the decode peers, and
# keep the loop alive. ---


class _FakeInfo:
    def __init__(self, endpoint, dst_port, room):
        self.endpoint = endpoint
        self.dst_port = dst_port
        self.room = room


class _Chunk:
    def __init__(self, room):
        self.room = room


def _bare_prefill_mgr():
    # Skip the heavy __init__ (zmq sockets, mooncake engine, threads); wire up
    # only the attributes _fail_transfer_chunk + the real base record_failure /
    # update_status touch, and record the decode-sync calls.
    m = object.__new__(MooncakeKVManagerPrefill)
    m.request_status = {}
    m.failure_records = {}
    m.failure_lock = threading.Lock()
    m.transfer_infos = {}
    m.attn_tp_rank = 0
    m.bootstrap_port = 12345
    m.synced = []
    m.sync_status_to_decode_endpoint = (
        lambda remote, dst_port, room, status, prefill_rank, *a, **k: m.synced.append(
            (remote, dst_port, room, status, prefill_rank)
        )
    )
    return m


def test_fail_transfer_chunk_concludes_room_and_notifies_decode():
    m = _bare_prefill_mgr()
    m.transfer_infos[7] = {"s1": _FakeInfo("10.0.0.7", 4000, 7)}
    m._fail_transfer_chunk(_Chunk(7), RuntimeError("poison chunk"))
    assert m.request_status[7] == TransferPoll.Failed
    assert 7 in m.failure_records
    assert m.synced == [("10.0.0.7", 4000, 7, TransferPoll.Failed, 0)]
    assert 7 not in m.transfer_infos  # bookkeeping released


def test_fail_transfer_chunk_handles_fault_before_dequeue():
    # queue.get() itself faulted -> kv_chunk is None; must be a quiet no-op so
    # the loop keeps draining (no room to conclude, no raise).
    m = _bare_prefill_mgr()
    m._fail_transfer_chunk(None, RuntimeError("queue failed"))
    assert m.request_status == {}
    assert m.synced == []


def test_fail_transfer_chunk_survives_a_failing_decode_sync():
    # A secondary failure while notifying the decode peer must not re-raise into
    # the worker loop; the room is still concluded Failed locally.
    m = _bare_prefill_mgr()

    def boom(*a, **k):
        raise OSError("decode unreachable")

    m.sync_status_to_decode_endpoint = boom
    m.transfer_infos[5] = {"s": _FakeInfo("h", 1, 5)}
    m._fail_transfer_chunk(_Chunk(5), RuntimeError("x"))
    assert m.request_status[5] == TransferPoll.Failed
    assert 5 not in m.transfer_infos
