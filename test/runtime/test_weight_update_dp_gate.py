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

"""Weight ops under attention-DP > 1: the scheduler's same-round gate and the
frontend's fan-in.

The scheduler completes a weight op (NCCL group init/teardown, distributed or
Mooncake load) only in a round where every DP rank holds the same kind of op
at the head of its queue, decided on the per-round DP MAX all-reduce that
already carries flush intent. These tests drive ``RequestHandler`` with a fake
``torch.distributed.all_reduce`` that plays the peer rank.
"""

from __future__ import annotations

import unittest
from collections import deque
from unittest import mock

import torch

from tokenspeed.runtime.engine.io_struct import (
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    FlushCacheReqInput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    RebalanceExpertsReqInput,
    RebalanceExpertsReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromMooncakeReqInput,
    UpdateWeightsFromMooncakeReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)
from tokenspeed.runtime.engine.request_handler import (
    _INTERNAL_OP_CODES,
    _WEIGHT_OP_CODES,
    RequestHandler,
)
from tokenspeed.runtime.engine.scheduler_control_client import (
    SchedulerControlClient,
    combined_weight_update_output,
)
from tokenspeed.runtime.moe.expert_rebalance import (
    EplbApplyChunk,
    EplbCommit,
    EplbSnapshot,
)

MAX = torch.distributed.ReduceOp.MAX
MIN = torch.distributed.ReduceOp.MIN


def _distributed(flush_cache: bool = True, weight_version: str | None = "v2"):
    return UpdateWeightsFromDistributedReqInput(
        names=["w"],
        dtype_names=["float16"],
        shapes=[[1]],
        flush_cache=flush_cache,
        weight_version=weight_version,
    )


def _mooncake(version: int = 7, flush_cache: bool = True, weight_version=None):
    return UpdateWeightsFromMooncakeReqInput(
        version=version, flush_cache=flush_cache, weight_version=weight_version
    )


class _DpPeer:
    """The other attention-DP rank, as seen through ``all_reduce``.

    MAX reduces merge this rank's seven-lane gate vector with the peer's
    (``head``: the peer's frontend op, ``internal``: its internal op); MIN
    reduces (replica decisions) pass unless ``min_ok`` is False.
    """

    def __init__(
        self,
        *,
        head: type | None,
        flush: bool = False,
        min_ok=True,
        internal: type | None = None,
    ):
        self.head = head
        self.internal = internal
        self.flush = flush
        self.min_ok = min_ok
        self.calls: list[tuple[object, object]] = []
        self.gate_vectors: list[list[int]] = []

    def all_reduce(self, buf, *, op, group):
        self.calls.append((group, op))
        if op == MAX:
            self.gate_vectors.append(buf.tolist())
            code = 0 if self.head is None else _WEIGHT_OP_CODES[self.head]
            internal = 0 if self.internal is None else _INTERNAL_OP_CODES[self.internal]
            peer = torch.tensor(
                [
                    1 if self.flush else 0,
                    -1 if self.head else 0,
                    code,
                    -code,
                    -1 if self.internal else 0,
                    internal,
                    -internal,
                ],
                dtype=buf.dtype,
            )
            buf.copy_(torch.maximum(buf, peer))
        elif op == MIN and not self.min_ok:
            buf.fill_(0)


def _handler(*, dp_size: int = 1):
    handler = RequestHandler.__new__(RequestHandler)
    handler.send_func = mock.Mock()
    handler.server_args = mock.Mock(weight_version="v1", kvstore_storage_backend=None)
    handler.can_clear_cache_fn = mock.Mock(return_value=True)
    handler.clear_cache_fn = mock.Mock(return_value=True)
    handler._replica_tp_size = 1
    handler._replica_tp_cpu_group = None
    handler.pp_size = 1
    handler.pp_cpu_group = None
    handler.attn_dp_size = dp_size
    handler.attn_dp_cpu_group = "dp" if dp_size > 1 else None
    handler._replica_decision_buf = torch.zeros(1, dtype=torch.int32)
    handler._replica_flush_want_buf = torch.zeros(7, dtype=torch.int32)
    handler._pending_weight_ops = deque()
    handler._pending_internal_ops = deque()
    handler._internal_op_completer = None
    handler._device = mock.Mock()
    handler._device.delete_l3_namespace.return_value = True
    handler._device.update_weights.return_value = (True, "ok")
    return handler


def _replies(handler) -> list:
    return [call.args[0] for call in handler.send_func.send_pyobj.call_args_list]


class TestSameRoundGate(unittest.TestCase):
    def test_op_waits_until_every_dp_rank_holds_one(self):
        handler = _handler(dp_size=2)
        peer = _DpPeer(head=None)
        req = _distributed()

        with mock.patch.object(torch.distributed, "all_reduce", peer.all_reduce):
            handler.process_requests([req])

        handler._device.update_weights.assert_not_called()
        handler.can_clear_cache_fn.assert_not_called()
        handler.send_func.send_pyobj.assert_not_called()
        self.assertEqual(list(handler._pending_weight_ops), [req])
        # Only the gate itself was reduced; no flush collectives.
        self.assertEqual(peer.calls, [("dp", MAX)])
        self.assertEqual(peer.gate_vectors, [[0, -1, 2, -2, 0, 0, 0]])

    def test_ready_op_flushes_updates_and_replies_once(self):
        handler = _handler(dp_size=2)
        peer = _DpPeer(head=UpdateWeightsFromDistributedReqInput)
        req = _distributed()

        with mock.patch.object(torch.distributed, "all_reduce", peer.all_reduce):
            handler.process_requests([req])

        # The head's flush_cache enters the flush collectives without any
        # standalone intent on the gate vector.
        self.assertEqual(peer.gate_vectors[0][0], 0)
        self.assertEqual(
            peer.calls,
            [("dp", MAX), ("dp", MIN), ("dp", MIN), ("dp", MIN)],
        )
        handler.can_clear_cache_fn.assert_called_once_with()
        handler.clear_cache_fn.assert_called_once_with()
        handler._device.update_weights.assert_called_once_with(req)
        handler._device.set_l3_weight_version.assert_called_once_with("v2")
        replies = _replies(handler)
        self.assertEqual(len(replies), 1)
        self.assertIsInstance(replies[0], UpdateWeightsFromDistributedReqOutput)
        self.assertTrue(replies[0].success)
        self.assertEqual(handler._pending_weight_ops, deque())

    def test_queued_op_completes_in_a_later_round(self):
        handler = _handler(dp_size=2)
        req = _distributed(flush_cache=False, weight_version=None)
        waiting = _DpPeer(head=None)
        with mock.patch.object(torch.distributed, "all_reduce", waiting.all_reduce):
            handler.process_requests([req])
        handler._device.update_weights.assert_not_called()

        arrived = _DpPeer(head=UpdateWeightsFromDistributedReqInput)
        with mock.patch.object(torch.distributed, "all_reduce", arrived.all_reduce):
            handler.process_requests([])

        handler._device.update_weights.assert_called_once_with(req)
        self.assertEqual(len(_replies(handler)), 1)

    def test_one_op_per_round(self):
        handler = _handler(dp_size=2)
        first, second = _distributed(flush_cache=False, weight_version=None), _mooncake(
            flush_cache=False
        )
        peer = _DpPeer(head=UpdateWeightsFromDistributedReqInput)

        with mock.patch.object(torch.distributed, "all_reduce", peer.all_reduce):
            handler.process_requests([first, second])

        handler._device.update_weights.assert_called_once_with(first)
        self.assertEqual(list(handler._pending_weight_ops), [second])

    def test_mismatched_head_types_raise(self):
        handler = _handler(dp_size=2)
        peer = _DpPeer(head=InitWeightsUpdateGroupReqInput)

        with mock.patch.object(torch.distributed, "all_reduce", peer.all_reduce):
            with self.assertRaisesRegex(RuntimeError, "different weight-update"):
                handler.process_requests([_distributed()])

        handler._device.update_weights.assert_not_called()

    def test_peer_failure_fails_every_rank_without_l3_commit(self):
        handler = _handler(dp_size=2)
        peer = _DpPeer(head=UpdateWeightsFromDistributedReqInput)
        req = _distributed()
        seen_min = []

        def all_reduce(buf, *, op, group):
            # Preflight and L3-delete MINs agree; the update-result MIN fails.
            if op == MIN:
                seen_min.append(group)
                if len(seen_min) == 3:
                    buf.fill_(0)
                    return
            peer.all_reduce(buf, op=op, group=group)

        with mock.patch.object(torch.distributed, "all_reduce", all_reduce):
            handler.process_requests([req])

        handler._device.update_weights.assert_called_once_with(req)
        handler._device.set_l3_weight_version.assert_not_called()
        self.assertEqual(handler.server_args.weight_version, "v1")
        reply = _replies(handler)[0]
        self.assertFalse(reply.success)
        self.assertIn("another rank", reply.message)

    def test_init_and_destroy_are_gated(self):
        for req_type, out_type in (
            (InitWeightsUpdateGroupReqInput, InitWeightsUpdateGroupReqOutput),
            (DestroyWeightsUpdateGroupReqInput, DestroyWeightsUpdateGroupReqOutput),
        ):
            with self.subTest(op=req_type.__name__):
                req = (
                    req_type(
                        master_address="h", master_port=1, rank_offset=0, world_size=2
                    )
                    if req_type is InitWeightsUpdateGroupReqInput
                    else req_type()
                )
                handler = _handler(dp_size=2)
                waiting = _DpPeer(head=None)
                with mock.patch.object(
                    torch.distributed, "all_reduce", waiting.all_reduce
                ):
                    handler.process_requests([req])
                handler._device.update_weights.assert_not_called()
                handler.send_func.send_pyobj.assert_not_called()

                ready = _DpPeer(head=req_type)
                with mock.patch.object(
                    torch.distributed, "all_reduce", ready.all_reduce
                ):
                    handler.process_requests([])
                handler._device.update_weights.assert_called_once_with(req)
                # Gate MAX, then the result MIN; no flush collectives.
                self.assertEqual(ready.calls, [("dp", MAX), ("dp", MIN)])
                handler.can_clear_cache_fn.assert_not_called()
                reply = _replies(handler)[0]
                self.assertIsInstance(reply, out_type)
                self.assertTrue(reply.success)

    def test_standalone_flush_and_waiting_op_share_the_gate(self):
        handler = _handler(dp_size=2)
        peer = _DpPeer(head=None, flush=True)

        with mock.patch.object(torch.distributed, "all_reduce", peer.all_reduce):
            handler.process_requests([FlushCacheReqInput(), _distributed()])

        # The peer's standalone flush runs here; the load stays queued.
        handler.clear_cache_fn.assert_called_once_with()
        handler._device.update_weights.assert_not_called()
        self.assertEqual(len(handler._pending_weight_ops), 1)
        replies = _replies(handler)
        self.assertEqual(len(replies), 1)
        self.assertTrue(replies[0].success)


class _Completer:
    """The rebalance hooks as the handler sees them: completes ops, triggers."""

    def __init__(self, *, manual_ok: bool = True):
        self.completed: list = []
        self.manual_ok = manual_ok
        self.manual_calls = 0

    def complete(self, op) -> None:
        self.completed.append(op)

    def begin_manual_rebalance(self) -> tuple[bool, str]:
        self.manual_calls += 1
        return (True, "snapshot taken") if self.manual_ok else (False, "busy")


def _with_internal_ops(handler) -> _Completer:
    completer = _Completer()
    handler.set_internal_op_completer(completer)
    return completer


class TestInternalOpGate(unittest.TestCase):
    """Internal ops (the expert rebalance) ride the same gate from a second FIFO."""

    def test_internal_op_waits_until_every_dp_rank_holds_one(self):
        handler = _handler(dp_size=2)
        completer = _with_internal_ops(handler)
        handler.enqueue_internal_op(EplbSnapshot())
        peer = _DpPeer(head=None)

        with mock.patch.object(torch.distributed, "all_reduce", peer.all_reduce):
            handler.process_requests([])

        self.assertEqual(completer.completed, [])
        self.assertEqual(list(handler._pending_internal_ops), [EplbSnapshot()])
        self.assertEqual(peer.gate_vectors, [[0, 0, 0, 0, -1, 1, -1]])

        arrived = _DpPeer(head=None, internal=EplbSnapshot)
        with mock.patch.object(torch.distributed, "all_reduce", arrived.all_reduce):
            handler.process_requests([])
        self.assertEqual(completer.completed, [EplbSnapshot()])
        self.assertEqual(handler._pending_internal_ops, deque())
        # No reply and no flush collectives for an internal op.
        handler.send_func.send_pyobj.assert_not_called()
        self.assertEqual(arrived.calls, [("dp", MAX)])

    def test_frontend_op_arriving_later_on_a_peer_does_not_trip_the_gate(self):
        # Rank A holds an internal op and receives a frontend op; the peer
        # holds only the internal op this round. Two FIFOs: the heads compare
        # kind-for-kind, the internal op completes, the frontend op waits.
        handler = _handler(dp_size=2)
        completer = _with_internal_ops(handler)
        handler.enqueue_internal_op(EplbCommit())
        peer = _DpPeer(head=None, internal=EplbCommit)
        req = _distributed(flush_cache=False, weight_version=None)

        with mock.patch.object(torch.distributed, "all_reduce", peer.all_reduce):
            handler.process_requests([req])

        self.assertEqual(completer.completed, [EplbCommit()])
        self.assertEqual(list(handler._pending_weight_ops), [req])
        handler._device.update_weights.assert_not_called()

    def test_frontend_op_takes_precedence_and_internal_waits_a_round(self):
        handler = _handler(dp_size=2)
        completer = _with_internal_ops(handler)
        chunk = EplbApplyChunk((0, 1))
        handler.enqueue_internal_op(chunk)
        req = _distributed(flush_cache=False, weight_version=None)
        peer = _DpPeer(
            head=UpdateWeightsFromDistributedReqInput, internal=EplbApplyChunk
        )

        with mock.patch.object(torch.distributed, "all_reduce", peer.all_reduce):
            handler.process_requests([req])

        # One blocking device call per round: the frontend op this round.
        handler._device.update_weights.assert_called_once_with(req)
        self.assertEqual(completer.completed, [])
        self.assertEqual(list(handler._pending_internal_ops), [chunk])

        with mock.patch.object(torch.distributed, "all_reduce", peer.all_reduce):
            handler.process_requests([])
        self.assertEqual(completer.completed, [chunk])
        self.assertEqual(handler._pending_internal_ops, deque())

    def test_one_internal_op_per_round_in_enqueue_order(self):
        handler = _handler(dp_size=2)
        completer = _with_internal_ops(handler)
        ops = [EplbCommit(), EplbApplyChunk((0,)), EplbApplyChunk((1,))]
        for op in ops:
            handler.enqueue_internal_op(op)
        for expected, internal in zip(
            ops, (EplbCommit, EplbApplyChunk, EplbApplyChunk)
        ):
            peer = _DpPeer(head=None, internal=internal)
            with mock.patch.object(torch.distributed, "all_reduce", peer.all_reduce):
                handler.process_requests([])
            self.assertEqual(completer.completed[-1], expected)
        self.assertEqual(completer.completed, ops)

    def test_mismatched_internal_heads_raise(self):
        handler = _handler(dp_size=2)
        _with_internal_ops(handler)
        handler.enqueue_internal_op(EplbSnapshot())
        peer = _DpPeer(head=None, internal=EplbCommit)
        with mock.patch.object(torch.distributed, "all_reduce", peer.all_reduce):
            with self.assertRaisesRegex(RuntimeError, "internal control"):
                handler.process_requests([])

    def test_enqueue_requires_a_completer_and_an_internal_kind(self):
        handler = _handler()
        with self.assertRaisesRegex(RuntimeError, "completer"):
            handler.enqueue_internal_op(EplbSnapshot())
        _with_internal_ops(handler)
        with self.assertRaisesRegex(TypeError, "not an internal"):
            handler.enqueue_internal_op(_distributed())
        with self.assertRaisesRegex(RuntimeError, "already installed"):
            handler.set_internal_op_completer(_Completer())

    def test_manual_trigger_rides_the_frontend_fifo_and_replies(self):
        handler = _handler(dp_size=2)
        completer = _with_internal_ops(handler)
        peer = _DpPeer(head=RebalanceExpertsReqInput)
        with mock.patch.object(torch.distributed, "all_reduce", peer.all_reduce):
            handler.process_requests([RebalanceExpertsReqInput()])
        self.assertEqual(completer.manual_calls, 1)
        handler._device.update_weights.assert_not_called()
        # Gate MAX, then the replica MIN on the decision; no flush.
        self.assertEqual(peer.calls, [("dp", MAX), ("dp", MIN)])
        reply = _replies(handler)[0]
        self.assertIsInstance(reply, RebalanceExpertsReqOutput)
        self.assertTrue(reply.success)

        # A peer that refused fails the trigger everywhere.
        handler = _handler(dp_size=2)
        _with_internal_ops(handler)
        peer = _DpPeer(head=RebalanceExpertsReqInput, min_ok=False)
        with mock.patch.object(torch.distributed, "all_reduce", peer.all_reduce):
            handler.process_requests([RebalanceExpertsReqInput()])
        reply = _replies(handler)[0]
        self.assertFalse(reply.success)
        self.assertIn("another rank", reply.message)

    def test_manual_trigger_without_eplb_is_refused(self):
        handler = _handler()
        handler.process_requests([RebalanceExpertsReqInput()])
        reply = _replies(handler)[0]
        self.assertIsInstance(reply, RebalanceExpertsReqOutput)
        self.assertFalse(reply.success)
        self.assertIn("--enable-eplb", reply.message)


class TestSingleReplicaUnchanged(unittest.TestCase):
    def test_init_completes_in_the_same_round(self):
        handler = _handler()
        req = InitWeightsUpdateGroupReqInput(
            master_address="h", master_port=1, rank_offset=0, world_size=2
        )

        handler.process_requests([req])

        handler._device.update_weights.assert_called_once_with(req)
        reply = _replies(handler)[0]
        self.assertIsInstance(reply, InitWeightsUpdateGroupReqOutput)
        self.assertTrue(reply.success)
        self.assertEqual(handler._pending_weight_ops, deque())

    def test_no_collectives_without_dp(self):
        handler = _handler()
        with mock.patch.object(
            torch.distributed,
            "all_reduce",
            side_effect=AssertionError("collective on a single replica"),
        ):
            handler.process_requests([_distributed()])
        handler._device.update_weights.assert_called_once()

    def test_tensor_and_disk_updates_reply_unsupported(self):
        handler = _handler()

        handler.process_requests(
            [
                UpdateWeightsFromTensorReqInput(
                    serialized_named_tensors=[b""], load_format=None, flush_cache=True
                ),
                UpdateWeightFromDiskReqInput(model_path="/m"),
            ]
        )

        handler._device.update_weights.assert_not_called()
        replies = _replies(handler)
        self.assertIsInstance(replies[0], UpdateWeightsFromTensorReqOutput)
        self.assertIsInstance(replies[1], UpdateWeightFromDiskReqOutput)
        for reply in replies:
            self.assertFalse(reply.success)
            self.assertIn("not supported on this engine", reply.message)


class TestMooncakeOp(unittest.TestCase):
    def test_flushed_load_publishes_the_version_as_namespace(self):
        handler = _handler()
        handler._device.update_weights.return_value = (True, "applied")
        req = _mooncake(version=12, flush_cache=True)

        handler.process_requests([req])

        handler.clear_cache_fn.assert_called_once_with()
        handler._device.update_weights.assert_called_once_with(req)
        handler._device.set_l3_weight_version.assert_called_once_with("12")
        self.assertEqual(handler.server_args.weight_version, "12")
        # The default is resolved at the point of use; the request is not
        # rewritten on its way to the device.
        self.assertIsNone(req.weight_version)
        reply = _replies(handler)[0]
        self.assertIsInstance(reply, UpdateWeightsFromMooncakeReqOutput)
        self.assertTrue(reply.success)

    def test_flushed_load_default_satisfies_the_l3_identity_rule(self):
        handler = _handler()
        handler.server_args.kvstore_storage_backend = "memory"
        req = _mooncake(version=12, flush_cache=True)

        handler.process_requests([req])

        handler._device.update_weights.assert_called_once_with(req)
        handler._device.set_l3_weight_version.assert_called_once_with("12")
        self.assertIsNone(req.weight_version)

    def test_explicit_weight_version_wins(self):
        handler = _handler()
        req = _mooncake(version=12, flush_cache=True, weight_version="ckpt-12")

        handler.process_requests([req])

        handler._device.set_l3_weight_version.assert_called_once_with("ckpt-12")

    def test_unflushed_load_keeps_the_namespace(self):
        handler = _handler()
        req = _mooncake(version=12, flush_cache=False)

        handler.process_requests([req])

        handler.clear_cache_fn.assert_not_called()
        handler._device.update_weights.assert_called_once_with(req)
        handler._device.set_l3_weight_version.assert_not_called()
        self.assertEqual(handler.server_args.weight_version, "v1")

    def test_l3_rejects_unflushed_version_switch(self):
        handler = _handler()
        handler.server_args.kvstore_storage_backend = "memory"
        req = _mooncake(version=12, flush_cache=False, weight_version="12")

        handler.process_requests([req])

        handler._device.update_weights.assert_not_called()
        reply = _replies(handler)[0]
        self.assertIsInstance(reply, UpdateWeightsFromMooncakeReqOutput)
        self.assertFalse(reply.success)
        self.assertIn("cannot change without flush_cache", reply.message)

    def test_failed_flush_skips_the_load(self):
        handler = _handler()
        handler.can_clear_cache_fn = mock.Mock(return_value=False)

        handler.process_requests([_mooncake(version=12)])

        handler._device.update_weights.assert_not_called()
        reply = _replies(handler)[0]
        self.assertFalse(reply.success)
        self.assertIn("cache flush failed", reply.message)


class TestCombinedWeightUpdateOutput(unittest.TestCase):
    def test_empty_is_failure(self):
        self.assertEqual(combined_weight_update_output([]), (False, ""))

    def test_ands_success_and_keeps_distinct_messages_once(self):
        ok, message = combined_weight_update_output(
            [
                UpdateWeightsFromDistributedReqOutput(success=True, message="ok"),
                UpdateWeightsFromDistributedReqOutput(success=False, message="nccl"),
                UpdateWeightsFromDistributedReqOutput(success=True, message="ok"),
            ]
        )
        self.assertFalse(ok)
        self.assertEqual(message, "ok | nccl")

    def test_single_replica_message_is_unchanged(self):
        self.assertEqual(
            combined_weight_update_output(
                [InitWeightsUpdateGroupReqOutput(success=True, message="joined")]
            ),
            (True, "joined"),
        )


class _Lock:
    def __init__(self) -> None:
        self.entered = 0

    async def __aenter__(self):
        self.entered += 1

    async def __aexit__(self, *exc):
        return False


class TestSchedulerControlClientFanIn(unittest.IsolatedAsyncioTestCase):
    def _client(self, replies):
        client = SchedulerControlClient.__new__(SchedulerControlClient)
        client.server_args = mock.Mock()
        client.server_args.mapping.attn.has_dp = True
        client.server_args.mapping.attn.dp_size = len(replies)
        client.auto_create_handle_loop = lambda: None
        client.model_update_lock = mock.Mock(writer_lock=_Lock())
        sent = []

        async def comm(req):
            sent.append(req)
            return list(replies)

        client.init_weights_update_group_communicator = comm
        client.destroy_weights_update_group_communicator = comm
        client.update_weights_from_distributed_communicator = comm
        client.update_weights_from_mooncake_communicator = comm
        return client, sent

    async def test_dp_replies_are_anded_without_rejecting_dp(self):
        client, sent = self._client(
            [
                UpdateWeightsFromDistributedReqOutput(success=True, message="ok"),
                UpdateWeightsFromDistributedReqOutput(success=False, message="bad"),
            ]
        )
        req = _distributed()

        ok, message = await SchedulerControlClient.update_weights_from_distributed(
            client, req
        )

        self.assertFalse(ok)
        self.assertEqual(message, "ok | bad")
        self.assertEqual(sent, [req])
        self.assertEqual(client.model_update_lock.writer_lock.entered, 1)

    async def test_init_destroy_and_mooncake_fan_in(self):
        # Only the loads rewrite weights; group init/teardown must not wait
        # for in-flight generation behind the writer lock.
        for method, req, out, takes_writer_lock in (
            (
                SchedulerControlClient.init_weights_update_group,
                InitWeightsUpdateGroupReqInput(
                    master_address="h", master_port=1, rank_offset=0, world_size=3
                ),
                InitWeightsUpdateGroupReqOutput,
                False,
            ),
            (
                SchedulerControlClient.destroy_weights_update_group,
                DestroyWeightsUpdateGroupReqInput(),
                DestroyWeightsUpdateGroupReqOutput,
                False,
            ),
            (
                SchedulerControlClient.update_weights_from_mooncake,
                _mooncake(),
                UpdateWeightsFromMooncakeReqOutput,
                True,
            ),
        ):
            with self.subTest(op=type(req).__name__):
                client, _ = self._client(
                    [
                        out(success=True, message="done"),
                        out(success=True, message="done"),
                    ]
                )
                self.assertEqual(await method(client, req), (True, "done"))
                self.assertEqual(
                    client.model_update_lock.writer_lock.entered,
                    1 if takes_writer_lock else 0,
                )


if __name__ == "__main__":
    unittest.main()
