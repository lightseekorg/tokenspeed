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

"""The frontend's half of the control operations: every attention-DP
engine's reply ANDed into one result, the model-update lock taken only by
the operations that rewrite weights, and the L3 weight version the ``Engine``
API stamps on a flushed update."""

from __future__ import annotations

import os
import sys
from unittest import mock

import pytest

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.engine.io_struct import (  # noqa: E402
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    FlushCacheReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromMooncakeReqInput,
    UpdateWeightsFromMooncakeReqOutput,
)
from tokenspeed.runtime.engine.scheduler_control_client import (  # noqa: E402
    SchedulerControlClient,
    combined_flush_cache_output,
    combined_weight_update_output,
)
from tokenspeed.runtime.entrypoints.engine import Engine  # noqa: E402


class TestEngineStampsL3Version:
    def _engine(self, *, storage_backend):
        engine = Engine.__new__(Engine)
        engine.server_args = mock.Mock(
            weight_version="v1", kvstore_storage_backend=storage_backend
        )
        engine.tokenizer_manager = mock.Mock()
        engine.llm = mock.Mock()
        return engine

    def test_l3_flush_without_version_does_not_send(self):
        engine = self._engine(storage_backend="memory")
        engine.llm.run.return_value = (True, "ok")

        success, message = engine.update_weights_from_distributed(
            names=["w"],
            dtypes=["float16"],
            shapes=[[1]],
            group_name="weight_update_group",
            flush_cache=True,
            weight_version=None,
        )

        assert not (success)
        assert ("require weight_version") in (message)
        assert (engine.server_args.weight_version) == ("v1")
        engine.llm.run.assert_not_called()

    def test_successful_update_persists_explicit_namespace(self):
        engine = self._engine(storage_backend="memory")
        engine.llm.run.return_value = (True, "ok")

        engine.update_weights_from_distributed(
            names=["w"],
            dtypes=["float16"],
            shapes=[[1]],
            group_name="weight_update_group",
            flush_cache=True,
            weight_version="v2",
        )

        assert (engine.server_args.weight_version) == ("v2")
        req = engine.tokenizer_manager.update_weights_from_distributed.call_args.args[0]
        assert (req.weight_version) == ("v2")

    def test_failed_update_does_not_persist_explicit_namespace(self):
        engine = self._engine(storage_backend="memory")
        engine.llm.run.return_value = (False, "nccl failed")

        engine.update_weights_from_distributed(
            names=["w"],
            dtypes=["float16"],
            shapes=[[1]],
            group_name="weight_update_group",
            flush_cache=True,
            weight_version="v2",
        )

        assert (engine.server_args.weight_version) == ("v1")

    def test_without_l3_omitted_version_does_not_stamp(self):
        engine = self._engine(storage_backend=None)
        engine.llm.run.return_value = (True, "ok")

        engine.update_weights_from_distributed(
            names=["w"],
            dtypes=["float16"],
            shapes=[[1]],
            group_name="weight_update_group",
            flush_cache=True,
            weight_version=None,
        )

        assert (engine.server_args.weight_version) == ("v1")

    def test_weight_version_has_no_default(self):
        engine = self._engine(storage_backend="memory")
        with pytest.raises((TypeError)):
            engine.update_weights_from_distributed(
                names=["w"],
                dtypes=["float16"],
                shapes=[[1]],
                group_name="weight_update_group",
                flush_cache=True,
            )
        with pytest.raises((TypeError)):
            UpdateWeightsFromDistributedReqInput(
                names=["w"],
                dtype_names=["float16"],
                shapes=[[1]],
            )


class TestCombinedFlushCacheOutput:
    def test_empty_results_are_failure(self):
        output = combined_flush_cache_output([])
        assert not (output.success)

    def test_ands_every_dp_replica(self):
        output = combined_flush_cache_output(
            [
                FlushCacheReqOutput(success=True),
                FlushCacheReqOutput(success=False),
            ]
        )
        assert not (output.success)

    def test_all_success_is_success(self):
        output = combined_flush_cache_output(
            [
                FlushCacheReqOutput(success=True),
                FlushCacheReqOutput(success=True),
            ]
        )
        assert output.success


@pytest.mark.asyncio(loop_scope="function")
class TestSchedulerControlFlushCache:
    async def test_ands_dp_replica_replies(self):
        client = SchedulerControlClient.__new__(SchedulerControlClient)

        async def fake_comm(req):
            del req
            return [
                FlushCacheReqOutput(success=True),
                FlushCacheReqOutput(success=False),
            ]

        client._control = fake_comm
        output = await SchedulerControlClient.flush_cache(client)
        assert not (output.success)

    async def test_all_replicas_success(self):
        client = SchedulerControlClient.__new__(SchedulerControlClient)

        async def fake_comm(req):
            del req
            return [
                FlushCacheReqOutput(success=True),
                FlushCacheReqOutput(success=True),
            ]

        client._control = fake_comm
        output = await SchedulerControlClient.flush_cache(client)
        assert output.success


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


class TestCombinedWeightUpdateOutput:
    def test_empty_is_failure(self):
        assert (combined_weight_update_output([])) == ((False, ""))

    def test_ands_success_and_keeps_distinct_messages_once(self):
        ok, message = combined_weight_update_output(
            [
                UpdateWeightsFromDistributedReqOutput(success=True, message="ok"),
                UpdateWeightsFromDistributedReqOutput(success=False, message="nccl"),
                UpdateWeightsFromDistributedReqOutput(success=True, message="ok"),
            ]
        )
        assert not (ok)
        assert (message) == ("ok | nccl")

    def test_single_replica_message_is_unchanged(self):
        assert (
            combined_weight_update_output(
                [InitWeightsUpdateGroupReqOutput(success=True, message="joined")]
            )
        ) == ((True, "joined"))


class _Lock:
    def __init__(self) -> None:
        self.entered = 0

    async def __aenter__(self):
        self.entered += 1

    async def __aexit__(self, *exc):
        return False


@pytest.mark.asyncio
class TestSchedulerControlClientFanIn:
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

        client._control = comm
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

        assert not (ok)
        assert (message) == ("ok | bad")
        assert (sent) == ([req])
        assert (client.model_update_lock.writer_lock.entered) == (1)

    async def test_init_destroy_and_mooncake_fan_in(self, subtests):
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
            with subtests.test(op=type(req).__name__):
                client, _ = self._client(
                    [
                        out(success=True, message="done"),
                        out(success=True, message="done"),
                    ]
                )
                assert (await method(client, req)) == ((True, "done"))
                assert (client.model_update_lock.writer_lock.entered) == (
                    1 if takes_writer_lock else 0
                )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
