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

"""The Python halves of the Rust engine host, without a GPU.

The launcher's half runs against real child processes speaking the host's
control-pipe protocol; the data plane's half drives a real ``DeviceHandle``
and scheduler, with a CPU stand-in for the model executor.
"""

from __future__ import annotations

import os
import signal
import stat
import sys
import textwrap
import threading
from contextlib import nullcontext

import msgspec
import pytest
import torch

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="runtime-1gpu")

from test.l3_test_utils import storage_scheduler  # noqa: E402

from tokenspeed_scheduler import (  # noqa: E402
    AttnKind,
    Cache,
    CacheGroupConfig,
    RequestSpec,
    Scheduler,
    SchedulerConfig,
)

from tokenspeed.runtime.engine.profiler import Profiler  # noqa: E402
from tokenspeed.runtime.engine.rank_setup import request_length_limits  # noqa: E402
from tokenspeed.runtime.engine.rust_host import (  # noqa: E402
    HostDataPlane,
    rust_host_refusals,
    start_rust_host,
)
from tokenspeed.runtime.execution.device import DeviceHandle  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.types import (  # noqa: E402
    InputLogprobPlan,
    ModelExecutionResult,
    RequestHistorySeeds,
    StopCriteria,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402
from tokenspeed.runtime.utils.process import receive_ready_messages  # noqa: E402
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs  # noqa: E402

# ----------------------------------------------------------------------
# The launcher's side
# ----------------------------------------------------------------------


def test_the_default_engine_is_served():
    # The Host L2 tier and the retraction snapshot pool are on by default.
    server_args = ServerArgs(model="x")
    assert server_args.enable_kvstore
    assert not server_args.retraction_snapshot_pool_disabled
    assert rust_host_refusals(server_args) == []


def test_unsupported_subsystems_are_named():
    custom = ServerArgs(
        model="x",
        kv_events_config='{"enable_kv_cache_events": true, "publisher": "custom"}',
    )
    assert rust_host_refusals(custom) == ["the KV event publisher 'custom'"]


def test_the_subsystems_ported_to_the_host_are_served():
    served = [
        ServerArgs(model="x", enable_metrics=True, enable_log_request_stats=True),
        ServerArgs(model="x", kvstore_storage_backend="memory"),
        ServerArgs(model="x", kv_events_config='{"enable_kv_cache_events": true}'),
        ServerArgs(
            model="x",
            attn_tp_size=2,
            kv_parallel_size=2,
            decode_context_parallel_size=2,
        ),
    ]
    for server_args in served:
        assert rust_host_refusals(server_args) == []


def test_tensor_and_data_parallel_layouts_are_served():
    assert rust_host_refusals(ServerArgs(model="x", attn_tp_size=4)) == []
    data_parallel = ServerArgs(
        model="x",
        attn_tp_size=2,
        data_parallel_size=2,
        dist_init_addr="127.0.0.1:4000",
    )
    assert data_parallel.mapping.has_attn_dp
    assert rust_host_refusals(data_parallel) == []


def test_both_pd_roles_are_served_with_layerwise_streaming():
    def pd(mode: str, **fields) -> ServerArgs:
        return ServerArgs(
            model="x",
            disaggregation_mode=mode,
            disaggregation_transfer_backend="mooncake",
            **fields,
        )

    assert rust_host_refusals(pd("decode")) == []
    assert rust_host_refusals(pd("prefill")) == []
    assert rust_host_refusals(pd("prefill", disaggregation_layerwise_interval=0)) == []


def _fake_host(tmp_path, body: str) -> str:
    """An executable taking the host's command line, running ``body`` with
    ``connection`` and ``python`` bound."""
    path = tmp_path / "ts-host"
    path.write_text(f"#!{sys.executable}\n" + textwrap.dedent("""
            import sys, time
            from multiprocessing.connection import Connection
            args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
            python = args["--python"]
            connection = Connection(int(args["--control-fd"]))
            """) + textwrap.dedent(body))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _rank_args(rank: int) -> tuple[ServerArgs, PortArgs]:
    server_args = ServerArgs(model="x")
    server_args.mapping.rank = rank
    return server_args, PortArgs.init_new(server_args)


def test_the_launcher_hands_a_host_its_rank_and_hears_it_ready(tmp_path):
    binary = _fake_host(
        tmp_path,
        """
        server_args, port_args, relay = connection.recv()
        connection.send({"status": "ready", "rank": server_args.mapping.rank,
                         "python": python, "relay": relay})
        time.sleep(600)
        """,
    )
    proc, reader = start_rust_host(
        binary, *_rank_args(3), nullcontext(), "ipc:///tmp/relay", False
    )
    try:
        (ready,) = receive_ready_messages([reader], [proc])
        assert ready == {
            "status": "ready",
            "rank": 3,
            "python": sys.executable,
            "relay": "ipc:///tmp/relay",
        }
        assert proc.is_alive() and proc.exitcode is None
    finally:
        proc.terminate()
        proc.join(timeout=10)
    assert not proc.is_alive()
    assert proc.exitcode == -signal.SIGTERM


def test_a_host_that_dies_before_ready_fails_the_launch(tmp_path):
    binary = _fake_host(tmp_path, "connection.recv()\nsys.exit(3)\n")
    proc, reader = start_rust_host(binary, *_rank_args(0), nullcontext(), None, False)
    with pytest.raises(RuntimeError, match="Scheduler process 0"):
        receive_ready_messages([reader], [proc])
    proc.join(timeout=10)
    assert proc.exitcode == 3


def test_request_length_limits_reserve_the_first_decode_window():
    assert request_length_limits(
        context_len=4096, max_single_request_tokens=1000, decode_input_tokens=4
    ) == (1000, 996)
    with pytest.raises(RuntimeError, match="cannot admit one request"):
        request_length_limits(
            context_len=4096, max_single_request_tokens=4, decode_input_tokens=4
        )


# ----------------------------------------------------------------------
# The data plane's side
# ----------------------------------------------------------------------


class _LandedCopy:
    """The CPU stand-in for a result's copy event: already landed; counts
    the waits on it."""

    def __init__(self) -> None:
        self.waits = 0

    def synchronize(self) -> None:
        self.waits += 1


class _CpuExecutor:
    """Stands in for the model executor: samples ``100 + row`` per row and
    records what each forward was given."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.forwards: list[tuple[list[str], list, tuple]] = []
        self.dp_metadata: list = []
        self.idle_forwards: list = []
        self.logprob_plans: list = []
        self.history_seeds: list = []

    def execute_forward_op(self, forward_op, sampling_params_list, **kwargs):
        rows = len(forward_op.request_ids)
        self.dp_metadata.append(kwargs["dp_metadata"])
        self.forwards.append(
            (
                list(forward_op.request_ids),
                list(sampling_params_list),
                kwargs["stop_criteria"],
            )
        )
        self.history_seeds.append(kwargs["request_history_seeds"])
        plan = kwargs["input_logprob_plan"]
        sampled_top_k = kwargs["sampled_top_k"]
        self.logprob_plans.append((plan, sampled_top_k))
        # Each gathered prompt row's logprob is -position/10, its top-k
        # alternatives the row's position and k - 1 more.
        positions = (
            [
                start + j
                for start, count in zip(plan.position_starts, plan.counts)
                for j in range(count)
            ]
            if plan is not None
            else []
        )

        def alternatives(firsts: list[int], k: int):
            values = [[-first / 10 - r for r in range(k)] for first in firsts]
            ids = [[first + r for r in range(k)] for first in firsts]
            return torch.tensor(values), torch.tensor(ids)

        input_top = alternatives(positions, plan.top_k) if plan and plan.top_k else None
        output_top = (
            alternatives(list(range(100, 100 + rows)), sampled_top_k)
            if sampled_top_k
            else None
        )
        return ModelExecutionResult(
            output_tokens=torch.arange(100, 100 + rows, dtype=torch.int32),
            output_lengths=torch.ones(rows, dtype=torch.int32),
            output_logprobs=torch.full((rows,), -0.5),
            copy_event=_LandedCopy(),
            input_token_logprobs=(
                torch.tensor([-p / 10 for p in positions]) if plan is not None else None
            ),
            input_logprob_plan=plan,
            input_top_logprobs_val=input_top[0] if input_top else None,
            input_top_logprobs_idx=input_top[1] if input_top else None,
            output_top_logprobs_val=output_top[0] if output_top else None,
            output_top_logprobs_idx=output_top[1] if output_top else None,
        )

    def execute_idle_forward(self, dp_metadata) -> None:
        self.idle_forwards.append(dp_metadata)

    def complete_pending_drafting(self) -> None:
        # No drafter: no round is ever pending.
        pass

    def zero_cache_pages(self, pages) -> None:
        pass

    def copy_cache_pages(self, copies) -> None:
        pass


def _scheduler() -> Scheduler:
    config = SchedulerConfig()
    config.prefix_granularity = 16
    config.max_scheduled_tokens = 64
    config.max_batch_size = 4
    config.num_device_pages = 64
    config.num_snapshot_pages = 1
    config.cache_groups = [
        CacheGroupConfig(
            group_id="full_attention",
            block_granularity=16,
            total_pages=64,
            attn_kind=AttnKind.Full,
        )
    ]
    return Scheduler(config)


def _resolved(rid: str, **fields) -> SamplingParams:
    """Parameters as the control plane admits them."""
    params = SamplingParams(**fields)
    params.resolve_seed(rid)
    params.normalize(None)
    return params


def _data_plane(handle: DeviceHandle) -> HostDataPlane:
    """A text-only, aggregated rank's data plane over ``handle``."""
    server_args = ServerArgs(model="x")
    server_args.mapping.rank = 0
    return HostDataPlane(
        handle,
        Profiler(
            server_args=server_args,
            device=handle,
            attn_mapping=server_args.mapping.attn,
            sync_size=1,
            sync_group=None,
        ),
        {},
        report_compiles=False,
        max_encoder_tokens=None,
        attn_tp_cpu_group=None,
        epd_publisher=None,
    )


@pytest.fixture
def executor():
    return _CpuExecutor()


def test_the_data_plane_runs_a_plan_with_the_admitted_requests(executor):
    plane = _data_plane(DeviceHandle(executor))
    params = _resolved("r1", temperature=0.0, max_new_tokens=8, stop_token_ids={9})
    plane.admit([("r1", msgspec.msgpack.encode(params), [2, 9], 11, None)])
    scheduler = _scheduler()
    spec = RequestSpec()
    spec.request_id = "r1"
    spec.tokens = [1, 2, 3]
    spec.max_new_tokens = 8
    scheduler.submit_requests([spec])

    pending = plane.execute(
        scheduler.next_execution_plan(),
        None,
        None,
        None,
        (None, 0),
        ([1], [2], [[7, 8]]),
    )

    assert pending is not None
    assert plane.complete(pending) == ([100], [1], [-0.5], None, None, None, None, None)
    ((rids, sampling_params, stop_criteria),) = executor.forwards
    assert rids == ["r1"]
    assert sampling_params == [params]
    assert stop_criteria == (StopCriteria(stop_token_ids=(2, 9), token_limit=11),)
    assert executor.history_seeds == [
        RequestHistorySeeds(slots=(1,), prefix_lengths=(2,), tokens=((7, 8),))
    ]


def test_the_data_plane_hands_the_logprob_plan_to_the_forward(executor):
    plane = _data_plane(DeviceHandle(executor))
    plane.admit([("r1", msgspec.msgpack.encode(_resolved("r1")), [], 11, None)])

    pending = plane.execute(
        _plan_with_one_request("r1"), None, None, None, (([0], [2], [0], 2), 1), None
    )

    *_, logprobs = plane.complete(pending)
    ((plan, sampled_top_k),) = executor.logprob_plans
    assert plan == InputLogprobPlan(
        row_starts=(0,), counts=(2,), position_starts=(0,), top_k=2
    )
    assert sampled_top_k == 1
    input_logprobs, input_top, output_top = logprobs
    assert input_logprobs == pytest.approx([0.0, -0.1])
    assert input_top[1] == [[0, 1], [1, 2]]
    assert output_top == ([[-10.0]], [[100]])


def _plan_with_one_request(rid: str):
    scheduler = _scheduler()
    spec = RequestSpec()
    spec.request_id = rid
    spec.tokens = [1, 2, 3]
    spec.max_new_tokens = 8
    scheduler.submit_requests([spec])
    return scheduler.next_execution_plan()


def test_a_data_parallel_forward_carries_every_ranks_batch(executor):
    plane = _data_plane(DeviceHandle(executor))
    plane.admit([("r1", msgspec.msgpack.encode(_resolved("r1")), [], 11, None)])

    pending = plane.execute(
        _plan_with_one_request("r1"),
        ([3, 0], [1, 0], [1, 0], False),
        None,
        None,
        (None, 0),
        None,
    )

    assert plane.complete(pending)[0] == [100]
    (metadata,) = executor.dp_metadata
    assert metadata.global_num_tokens == [3, 0]
    assert metadata.global_forward_mode == [
        int(ForwardMode.EXTEND),
        int(ForwardMode.IDLE),
    ]
    assert not metadata.need_idle_forward
    assert executor.idle_forwards == []


def test_an_idle_data_parallel_rank_runs_an_idle_forward(executor):
    plane = _data_plane(DeviceHandle(executor))
    plane.admit([("r1", msgspec.msgpack.encode(_resolved("r1")), [], 11, None)])

    # This rank's plan has a batch, but the control plane decided it idles.
    assert (
        plane.execute(
            _plan_with_one_request("r1"),
            ([0, 6], [0, 2], [0, 0], True),
            None,
            None,
            None,
            None,
        )
        is None
    )

    assert executor.forwards == []
    (metadata,) = executor.idle_forwards
    assert metadata.need_idle_forward
    assert metadata.global_num_decodes == [0, 2]
    assert metadata.all_decode_or_idle


def test_a_plan_without_a_batch_runs_no_forward(executor):
    plane = _data_plane(DeviceHandle(executor))
    assert (
        plane.execute(_scheduler().next_execution_plan(), None, None, None, None, None)
        is None
    )
    assert executor.forwards == []


class _HostCache:
    """Stands in for the Host cache executor: its acknowledgments, landed."""

    def __init__(self, landed) -> None:
        self.landed = landed

    def poll_results(self) -> list:
        landed, self.landed = self.landed, []
        return landed


def test_the_data_plane_reports_cache_acknowledgments_in_landing_order(executor):
    snapshot = Cache.SnapshotDoneEvent()
    snapshot.op_id = 7
    write_back = Cache.WriteBackDoneEvent()
    write_back.op_id = 3
    landed = [snapshot, write_back, Cache.LoadBackDoneEvent(9)]
    host_cache = _HostCache(landed)
    plane = _data_plane(DeviceHandle(executor, host_cache_executor=host_cache))
    assert plane.poll_cache_events() == [
        ("SnapshotDoneEvent", 7),
        ("WriteBackDoneEvent", 3),
        ("LoadBackDoneEvent", 9),
    ]
    assert plane.poll_cache_events() == []


def test_the_data_plane_leaves_a_plans_storage_operations_to_the_host(executor):
    plane = _data_plane(DeviceHandle(executor))
    # The waiting request's probe: a storage operation, and no forward. The
    # host's L3 executor runs it; the data plane, without a Host cache
    # executor here, launches nothing.
    plan = storage_scheduler().next_execution_plan()
    assert [type(op).__name__ for op in plan.cache] == ["StorageQueryOp"]
    assert plane.execute(plan, None, None, None, None, None) is None


def test_a_released_request_is_forgotten(executor):
    plane = _data_plane(DeviceHandle(executor))
    params = _resolved("r1")
    plane.admit([("r1", msgspec.msgpack.encode(params), [], 10, None)])
    plane.release(["r1"])
    with pytest.raises(KeyError):
        plane.release(["r1"])


def test_a_forward_lands_once_and_counts_toward_a_profile_at_its_note(executor):
    """The commit waits on the copy event exactly once; the profiler counts
    the forward only when the forward thread runs its committed note."""
    profiled = _data_plane(DeviceHandle(executor))
    profiled.admit([("r1", msgspec.msgpack.encode(_resolved("r1")), [], 11, None)])
    launched = profiled.execute(
        _plan_with_one_request("r1"), None, None, None, (None, 0), None
    )

    assert profiled.complete(launched)[0] == [100]
    assert launched.results.copy_event.waits == 1
    with pytest.raises(RuntimeError, match="exactly once"):
        profiled.complete(launched)
    assert profiled._profiler.forward_ct == 0
    profiled.note_committed(launched)
    assert profiled._profiler.forward_ct == 1


def test_a_result_without_a_copy_event_cannot_land():
    results = ModelExecutionResult(output_tokens=torch.tensor([7], dtype=torch.int32))
    with pytest.raises(RuntimeError, match="copy_event is required"):
        results.sync()


def test_the_forward_thread_is_named_for_its_role(executor):
    data_plane = _data_plane(DeviceHandle(executor))
    names = []

    def forward_thread():
        data_plane.enter_forward_thread()
        names.append(threading.current_thread().name)

    thread = threading.Thread(target=forward_thread)
    thread.start()
    thread.join()
    assert names == ["tokenspeed::forward"]
