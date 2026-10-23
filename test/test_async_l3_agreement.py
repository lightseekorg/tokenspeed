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

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.cache.l3.config import StorageRuntimeConfig
from tokenspeed.runtime.cache.l3.io_types import StorageCompletion, StoragePoll
from tokenspeed.runtime.engine.l3_cache_hooks import L3CacheHooks


class Device:
    def __init__(self, rank):
        self.rank = rank
        self.round = 0

    def poll(self):
        return StoragePoll(tuple(self._completions()), 0, 0, 0)

    def _completions(self):
        self.round += 1
        if self.round == 1:
            return (
                [StorageCompletion(1, (True, True), True, (False, False))]
                if self.rank == 0
                else [
                    StorageCompletion(
                        2,
                        (True, False),
                        True,
                        (False, True),
                    )
                ]
            )
        if self.round == 2:
            return (
                [StorageCompletion(2, (True, True), True, (False, False))]
                if self.rank == 0
                else []
            )
        if self.round == 3:
            return (
                []
                if self.rank == 0
                else [StorageCompletion(1, (False, False), False, (False, False))]
            )
        if self.rank == 0:
            raise RuntimeError("injected local poll fault")
        return []


def worker(rank, path):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        config = StorageRuntimeConfig.from_extra_config(
            None, host_blocks=16, prefix_granularity=2
        )
        hooks = L3CacheHooks(
            SimpleNamespace(),
            Device(rank),
            config=config.scheduler,
            attn_tp_size=2,
            attn_tp_cpu_group=dist.group.WORLD,
            pp_size=1,
            pp_cpu_group=None,
            metrics=None,
        )
        first = hooks.poll_ready_events(allow_forward=True, allow_storage=True)
        assert len(first) == 1  # only the common tick; both operations incomplete
        second = hooks.poll_ready_events(allow_forward=True, allow_storage=True)
        assert len(second) == 2
        assert second[1].op_id == 2 and list(second[1].success) == [1, 0]
        assert second[1].retired
        third = hooks.poll_ready_events(allow_forward=True, allow_storage=True)
        assert third[1].op_id == 1 and not third[1].retired
        assert not hooks._reports
        try:
            hooks.poll_ready_events(allow_forward=False, allow_storage=False)
        except RuntimeError as error:
            assert "injected local poll fault" in str(error)
        else:
            raise AssertionError("every rank must observe the peer's polling failure")
    finally:
        dist.destroy_process_group()


def test_real_gloo_owners_complete_out_of_order_and_converge_errors(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def test_disabled_hook_has_no_storage_or_collective_work():
    hooks = L3CacheHooks(
        SimpleNamespace(),
        None,
        config=None,
        attn_tp_size=1,
        attn_tp_cpu_group=None,
        pp_size=1,
        pp_cpu_group=None,
        metrics=None,
    )
    assert hooks.poll_ready_events(allow_forward=True, allow_storage=True) == []


class IdleDevice:
    def poll(self):
        return StoragePoll((), 0, 0, 0)


def idle_worker(rank, path, world_size):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    original = dist.all_gather_object

    def unexpected_object_exchange(*args, **kwargs):
        raise AssertionError("An idle L3 round serialized an object payload")

    try:
        config = StorageRuntimeConfig.from_extra_config(
            None, host_blocks=16, prefix_granularity=2
        )
        hooks = L3CacheHooks(
            SimpleNamespace(),
            IdleDevice(),
            config=config.scheduler,
            attn_tp_size=world_size,
            attn_tp_cpu_group=dist.group.WORLD,
            pp_size=1,
            pp_cpu_group=None,
            metrics=None,
        )
        dist.all_gather_object = unexpected_object_exchange

        def tick_record(now_us, allow_forward, allow_storage, pressure):
            return SimpleNamespace(
                now_us=now_us,
                allow_forward=allow_forward,
                allow_storage=allow_storage,
                pressure=pressure,
            )

        with patch(
            "tokenspeed.runtime.engine.l3_cache_hooks.Cache",
            SimpleNamespace(StorageTick=tick_record),
        ):
            previous = 0
            for _ in range(3):
                events = hooks.poll_ready_events(
                    allow_forward=rank == 0, allow_storage=rank == 1
                )
                assert len(events) == 1
                tick = events[0]
                assert tick.now_us >= previous
                assert not tick.allow_forward and not tick.allow_storage
                previous = tick.now_us
        reports = hooks._exchange_reports(
            (
                1000 + rank,
                rank == 0,
                rank == 1,
                [],
                None,
                (rank + 1, 2**33 + rank, rank * 100),
            )
        )
        assert reports == [
            (
                1000 + peer,
                peer == 0,
                peer == 1,
                [],
                None,
                (peer + 1, 2**33 + peer, peer * 100),
            )
            for peer in range(world_size)
        ]
    finally:
        dist.all_gather_object = original
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 3])
def test_idle_gloo_round_agrees_clock_and_permissions_without_objects(
    tmp_path, monkeypatch, world_size
):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(
        idle_worker,
        args=(str(tmp_path / "idle"), world_size),
        nprocs=world_size,
        join=True,
    )


class PressureDevice:
    def __init__(self, rank):
        self.rank = rank
        self.round = 0

    def poll(self):
        self.round += 1
        if self.round == 1:
            return StoragePoll(
                (),
                9 if self.rank == 0 else 0,
                4096 if self.rank == 1 else 0,
                700000 * self.rank,
            )
        if self.round == 2:
            return StoragePoll((), 0, 0, 0)
        return StoragePoll((), -1 if self.rank == 0 else 0, 0, 0)


def pressure_worker(rank, path):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        hooks = L3CacheHooks(
            SimpleNamespace(),
            PressureDevice(rank),
            config=StorageRuntimeConfig.from_extra_config(
                None, host_blocks=16, prefix_granularity=2
            ).scheduler,
            attn_tp_size=2,
            attn_tp_cpu_group=dist.group.WORLD,
            pp_size=1,
            pp_cpu_group=None,
            metrics=None,
        )
        with patch.object(
            dist,
            "all_gather_object",
            side_effect=AssertionError(
                "pressure-only rounds must use the fixed header"
            ),
        ):
            tick = hooks.poll_ready_events(allow_forward=True, allow_storage=True)[0]
            assert (
                tick.pressure.queued_operations,
                tick.pressure.queued_bytes,
                tick.pressure.oldest_operation_us,
            ) == (9, 4096, 700000)
            tick = hooks.poll_ready_events(allow_forward=True, allow_storage=True)[0]
            assert (
                tick.pressure.queued_operations,
                tick.pressure.queued_bytes,
                tick.pressure.oldest_operation_us,
            ) == (0, 0, 0)
        try:
            hooks.poll_ready_events(allow_forward=True, allow_storage=True)
        except RuntimeError as error:
            assert "Invalid L3 pressure observation" in str(error)
        else:
            raise AssertionError(
                "invalid local pressure must converge as a protocol error"
            )
    finally:
        dist.destroy_process_group()


def test_pressure_gloo_uses_worst_owner_without_object_serialization(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(pressure_worker, args=(str(tmp_path / "pressure"),), nprocs=2, join=True)


class MalformedDevice:
    def __init__(self, rank, invalid_field):
        self.rank = rank
        self.invalid_field = invalid_field

    def poll(self):
        if self.rank != 0:
            return StoragePoll((), 0, 0, 0)
        item = {
            "missing": object(),
            "retirement": StorageCompletion(7, (True,), "pending", (False,)),
            "mask_length": StorageCompletion(7, (True,), True, ()),
            "mask_type": StorageCompletion(7, (False,), True, (1,)),
            "contradiction": StorageCompletion(7, (True,), True, (True,)),
        }[self.invalid_field]
        return StoragePoll((item,), 0, 0, 0)


def malformed_worker(rank, path, invalid_field):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:
        hooks = L3CacheHooks(
            SimpleNamespace(),
            MalformedDevice(rank, invalid_field),
            config=StorageRuntimeConfig.from_extra_config(
                None, host_blocks=16, prefix_granularity=2
            ).scheduler,
            attn_tp_size=2,
            attn_tp_cpu_group=dist.group.WORLD,
            pp_size=1,
            pp_cpu_group=None,
            metrics=None,
        )
        try:
            hooks.poll_ready_events(allow_forward=True, allow_storage=True)
        except RuntimeError as error:
            assert "L3 replica polling failed" in str(error)
        else:
            raise AssertionError("Malformed completion reached scheduler publication")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "invalid_field",
    ["missing", "retirement", "mask_length", "mask_type", "contradiction"],
)
def test_malformed_owner_completions_converge_before_any_rank_raises(
    tmp_path, monkeypatch, invalid_field
):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(
        malformed_worker,
        args=(str(tmp_path / f"malformed-{invalid_field}"), invalid_field),
        nprocs=2,
        join=True,
    )


class ObjectFailureDevice:
    def __init__(self, rank):
        self.rank = rank

    def poll(self):
        completion = (
            StorageCompletion(1, (False, False, True), True, (False, False, False))
            if self.rank == 0
            else StorageCompletion(1, (True, False, True), True, (False, True, False))
        )
        return StoragePoll((completion,), 0, 0, 0)


def object_failure_worker(rank, path):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=10),
    )
    try:
        hooks = L3CacheHooks(
            SimpleNamespace(),
            ObjectFailureDevice(rank),
            config=StorageRuntimeConfig.from_extra_config(
                None, host_blocks=16, prefix_granularity=2
            ).scheduler,
            attn_tp_size=2,
            attn_tp_cpu_group=dist.group.WORLD,
            pp_size=1,
            pp_cpu_group=None,
            metrics=None,
        )
        result = hooks.poll_ready_events(allow_forward=True, allow_storage=True)[1]
        assert list(result.success) == [0, 0, 1]
        assert list(result.object_failures) == [0, 1, 0]
        assert result.retired
    finally:
        dist.destroy_process_group()


def test_real_gloo_rejection_is_distinct_from_confirmed_peer_miss(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    mp.spawn(
        object_failure_worker,
        args=(str(tmp_path / "object-failure"),),
        nprocs=2,
        join=True,
    )
