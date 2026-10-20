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

"""GPU e2e: compact Host L2 round-trips through the Mooncake-compatible L3 store.

CI has no Mooncake master. ``MemoryKvStore`` implements the same
``batch_put_from`` / ``batch_get_into`` contract the Mooncake client uses on
packed Host CacheBlocks.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")


class TestL3FlatKvRoundTripTest:
    def setup_method(self):
        try:
            import torch

            import tokenspeed.runtime.cache.l2.executor as executor_module
            from tokenspeed.runtime.cache.l2.sizing import RetractionPoolRequest
            from tokenspeed.runtime.cache.l3.backend import MemoryKvStore
            from tokenspeed.runtime.cache.transfer.layout import (
                CacheField,
                CacheGroupLayout,
                CacheTransferLayout,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"needs runtime dependencies: {exc}")
        if not torch.cuda.is_available():
            pytest.skip("needs a CUDA device")
        self.torch = torch
        self.executor_module = executor_module
        self.no_pool = RetractionPoolRequest(
            host_gb=0.0,
            ratio=0.0,
            max_retracted_requests=0,
            tail_lcm_blocks_per_request=0,
        )
        self.MemoryKvStore = MemoryKvStore
        self.CacheField = CacheField
        self.CacheGroupLayout = CacheGroupLayout
        self.CacheTransferLayout = CacheTransferLayout

    def test_cp_restores_sharded_bytes_on_a_different_owner(self, request):
        from tokenspeed.runtime.cache.transfer.ops import PrefetchOp, PrefetchRow
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
            CacheRuntimeContract,
        )
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
            CacheGroupSpec,
        )

        torch = self.torch
        specs = (
            CacheGroupSpec(
                group_id="history",
                attn_kind="full_history",
                rows_per_page=2,
                entry_stride_tokens=1,
                sliding_window_tokens=None,
                replayable=False,
                transfer_policy="full_suffix",
                shard_count=2,
            ),
            CacheGroupSpec(
                group_id="state",
                attn_kind="state",
                rows_per_page=None,
                entry_stride_tokens=None,
                sliding_window_tokens=None,
                replayable=False,
                transfer_policy="latest_snapshot",
                shard_count=1,
            ),
        )
        contract = CacheRuntimeContract(
            prefix_granularity=2,
            num_lcm_blocks=8,
            token_capacity=16,
            group_specs=specs,
            group_page_counts={"history": 9, "state": 9},
            group_packing={"history": 1, "state": 1},
        )

        class Pool:
            def __init__(self, layout):
                self.layout = layout
                self.arena = self
                self.runtime_contract = contract
                self.cache_group_specs = specs
                self.load_tracker = None

            def cache_transfer_layout(self):
                return self.layout

            def register_layerwise_load_tracker(self, tracker):
                self.load_tracker = tracker

        backend = self.MemoryKvStore()
        ranks = []
        for rank in range(2):
            history = torch.zeros(72, dtype=torch.uint8, device="cuda")
            state = torch.zeros_like(history)
            layout = self.CacheTransferLayout(
                num_lcm_blocks=8,
                groups=(
                    self.CacheGroupLayout(
                        "history", 1, (self.CacheField("layer.0.kv", 0, 0, 8, 8),)
                    ),
                    self.CacheGroupLayout(
                        "state", 1, (self.CacheField("layer.1.state", 1, 0, 8, 8),)
                    ),
                ),
                buffers=(history, state),
                consumers=(("layer.0.kv",), ("layer.1.state",)),
            )
            pool = Pool(layout)
            with patch.object(self.executor_module, "_HOST_MEM_HEADROOM_BYTES", 0):
                executor = self.executor_module.HostCacheExecutor(
                    pool,
                    draft_pool=None,
                    l2_tier=True,
                    host_ratio=1.0,
                    host_size_gb=0,
                    snapshot_pool=self.no_pool,
                    slot_state_exporters=None,
                    io_backend="direct",
                    attn_tp_rank=rank,
                    kvp_rank=rank,
                    nprocs_per_node=1,
                    world_cpu_group=None,
                )
            executor.attach_l3_storage(
                backend,
                key_prefix="cp-e2e",
                rank=rank,
                prefix_for_weight_version=lambda version: f"cp-e2e-{version}",
                prefetch_timeout_base_s=10.0,
                prefetch_timeout_per_page_s=0.0,
                prefetch_batch_pages=2,
            )
            request.addfinalizer(executor.shutdown)
            history[8:16].fill_(0x10 + rank)
            state[32:40].fill_(0x20 + rank)
            executor._start_writing(
                [7],
                [(0, 1, 1), (1, 4, 4)],
                [(0, 1, f"h{rank}", 0), (1, 4, "state", 0)],
                lane=executor._pinned_write_lane,
                prerequisite_stream=torch.cuda.current_stream(),
            )
            executor.write_stream.synchronize()
            deadline = time.monotonic() + 10
            results = []
            while not results and time.monotonic() < deadline:
                results = executor.poll_results()
                if not results:
                    time.sleep(0.001)
            assert [event.op_id for event in results] == [7]
            ranks.append((executor, pool, history, state))

        # Cold local buffers, and each sharded page now lands on the other rank.
        op = PrefetchOp(
            op_id=8,
            request_id="reader",
            first_page=0,
            num_pages=2,
            rows=(
                PrefetchRow(0, 2, "h0", 0, 0),
                PrefetchRow(0, 1, "h1", 0, 1),
                PrefetchRow(1, 4, "state", 0, 1),
            ),
        )
        for executor, _, history, state in ranks:
            executor.host_storage.host_buffer.zero_()
            history.zero_()
            state.zero_()
            executor.submit_prefetches([op])
        for rank, (executor, pool, history, state) in enumerate(ranks):
            deadline = time.monotonic() + 10
            while (
                not executor.l3_prefetch_progress()[8][0]
                and time.monotonic() < deadline
            ):
                time.sleep(0.001)
            assert executor.l3_prefetch_progress()[8] == (True, 2)
            executor.complete_l3_prefetch(8, 2)
            (ack,) = executor.poll_results()
            assert ack.op_id == 8 and ack.landed_pages == 2
            # One owned history object plus this TP rank's replicated state.
            assert executor.l3_store.stats().get.ok_keys == 2
            load = executor._start_loading(
                [9],
                [(0, 2, 1), (1, 5, 4)],
                prerequisite_stream=torch.cuda.current_stream(),
            )
            pool.load_tracker.set_consumers(load)
            pool.load_tracker.wait_for_layer(0)
            pool.load_tracker.wait_for_layer(1)
            torch.cuda.synchronize()
            assert [event.op_id for event in executor.poll_results()] == [9]
            assert torch.equal(
                history[16:24].cpu(),
                torch.full((8,), 0x10 + 1 - rank, dtype=torch.uint8),
            )
            assert torch.equal(
                state[40:48].cpu(), torch.full((8,), 0x20 + rank, dtype=torch.uint8)
            )

    @pytest.mark.parametrize("capacity_pressure", [False, True])
    def test_d2h_backup_host_wipe_prefetch_h2d_restores_bytes(
        self, request, capacity_pressure
    ):
        torch = self.torch
        first = torch.full((128,), 0xCC, dtype=torch.uint8, device="cuda")
        second = torch.full((128,), 0xCC, dtype=torch.uint8, device="cuda")
        layout = self.CacheTransferLayout(
            num_lcm_blocks=4,
            groups=(
                self.CacheGroupLayout(
                    group_id="full",
                    cache_blocks_per_lcm_block=2,
                    fields=(
                        self.CacheField("layer.0.k", 0, 8, 8, 4),
                        self.CacheField("layer.0.v", 1, 16, 12, 6),
                    ),
                ),
                self.CacheGroupLayout(
                    group_id="state",
                    cache_blocks_per_lcm_block=1,
                    fields=(self.CacheField("layer.1.state", 0, 64, 10, 5),),
                ),
            ),
            buffers=(first, second),
            consumers=(("layer.0.k", "layer.0.v"), ("layer.1.state",)),
        )

        class SyntheticPool:
            def __init__(self):
                self.load_tracker = None
                self.arena = None

            def cache_transfer_layout(self):
                return layout

            def register_layerwise_load_tracker(self, tracker):
                self.load_tracker = tracker

        pool = SyntheticPool()
        specs = tuple(
            SimpleNamespace(group_id=group.group_id, shard_count=1)
            for group in layout.groups
        )
        pool.arena = SimpleNamespace(
            cache_group_specs=specs,
            runtime_contract=SimpleNamespace(
                group_specs=specs,
                virtual_block_counts={
                    group.group_id: 1
                    + layout.num_lcm_blocks * group.cache_blocks_per_lcm_block
                    for group in layout.groups
                },
            ),
        )
        with patch.object(self.executor_module, "_HOST_MEM_HEADROOM_BYTES", 0):
            executor = self.executor_module.HostCacheExecutor(
                pool,
                l2_tier=True,
                host_ratio=1.0,
                host_size_gb=0,
                snapshot_pool=self.no_pool,
                slot_state_exporters=None,
                io_backend="direct",
                attn_tp_rank=0,
                kvp_rank=0,
                nprocs_per_node=1,
                world_cpu_group=None,
            )
        store = self.MemoryKvStore()
        retry_started = threading.Event()
        release_retry = threading.Event()
        if capacity_pressure:
            from tokenspeed.runtime.cache.l3.mooncake import MooncakeKvStore

            class RpcStore:
                def __init__(self):
                    self.objects = {}
                    self.failed = False

                def batch_is_exist(self, keys):
                    return [int(key in self.objects) for key in keys]

                def batch_put_from(self, keys, ptrs, sizes):
                    if self.failed:
                        retry_started.set()
                        if not release_retry.wait(timeout=10):
                            raise TimeoutError("test did not release capacity retry")
                    results = []
                    for key, ptr, size in zip(keys, ptrs, sizes):
                        if not self.failed:
                            self.failed = True
                            results.append(-200)
                        else:
                            self.objects[key] = ctypes.string_at(ptr, size)
                            results.append(0)
                    return results

                def batch_get_into(self, keys, ptrs, sizes):
                    for key, ptr, size in zip(keys, ptrs, sizes):
                        payload = self.objects[key]
                        assert len(payload) == size
                        ctypes.memmove(ptr, payload, size)
                    return list(sizes)

            store = object.__new__(MooncakeKvStore)
            store.store = RpcStore()
        executor.attach_l3_storage(
            store,
            key_prefix="e2e",
            rank=0,
            prefix_for_weight_version=lambda version: f"e2e-{version}",
            prefetch_timeout_base_s=10.0,
            prefetch_timeout_per_page_s=0.0,
            prefetch_batch_pages=128,
        )
        request.addfinalizer(executor.shutdown)
        request.addfinalizer(release_retry.set)

        # A CUDA fence cannot complete a CPU storage future. Hold PUT until
        # the test observes the pending ACK, then let normal polling finish it.
        put_started = threading.Event()
        release_put = threading.Event()
        request.addfinalizer(release_put.set)
        backend = executor.l3_store.backend
        original_put = backend.batch_put_from

        def gated_put(keys, host_buffer, offsets, sizes):
            put_started.set()
            if not release_put.wait(timeout=10):
                raise TimeoutError("test did not release the L3 backup")
            return original_put(keys, host_buffer, offsets, sizes)

        put_patch = patch.object(backend, "batch_put_from", gated_put)
        put_patch.start()
        request.addfinalizer(put_patch.stop)

        first[16:20].fill_(0x11)
        second[28:34].fill_(0x12)
        first[40:44].fill_(0x41)
        second[64:70].fill_(0x42)
        first[94:99].fill_(0x73)
        torch.cuda.synchronize()

        backup_pages = [
            (0, 1, "h0", 0),
            (0, 4, "h1", 0),
            (1, 3, "h2", 0),
        ]
        executor._start_writing(  # pylint: disable=protected-access
            [7],
            [(0, 1, 1), (0, 4, 4), (1, 3, 3)],
            backup_pages,
            lane=executor._pinned_write_lane,
            prerequisite_stream=torch.cuda.current_stream(),
        )
        executor.write_stream.synchronize()
        write_results = executor.poll_results()
        assert (write_results) == ([])
        assert put_started.wait(timeout=10), "L3 backup did not start"
        release_put.set()
        if capacity_pressure:
            assert retry_started.wait(timeout=2), "capacity failure was not retried"
            assert executor.poll_results() == []
            assert not executor.consume_backup_poll_failure()
            assert executor.l3_backup_backlog().pending_pages == 3
            release_retry.set()
        deadline = time.monotonic() + 10
        while not write_results and time.monotonic() < deadline:
            write_results = executor.poll_results()
            if not write_results:
                time.sleep(0.001)
        assert ([int(event.op_id) for event in write_results]) == ([7])
        assert not executor.consume_backup_poll_failure()
        assert (executor.l3_store.exists(backup_pages)) == ([True, True, True])

        # Prove restore comes from L3, not leftover Host bytes.
        executor.host_storage.host_buffer.fill_(0)
        first.fill_(0xEE)
        second.fill_(0xEE)
        torch.cuda.synchronize()
        assert not (bool(executor.host_storage.host_buffer.any().item()))

        assert (executor.l3_store.prefetch(backup_pages)) == ([True, True, True])
        assert bool(executor.host_storage.host_buffer.any().item())

        load_index = executor._start_loading(  # pylint: disable=protected-access
            [9],
            [(0, 2, 1), (0, 5, 4), (1, 4, 3)],
            prerequisite_stream=torch.cuda.current_stream(),
        )
        assert (load_index) is not None
        pool.load_tracker.set_consumers(load_index)
        pool.load_tracker.wait_for_layer(0)
        pool.load_tracker.wait_for_layer(1)
        torch.cuda.synchronize()
        load_results = executor.poll_results()
        assert ([int(event.op_id) for event in load_results]) == ([9])

        assert torch.equal(
            first[24:28].cpu(), torch.full((4,), 0x11, dtype=torch.uint8)
        )
        assert torch.equal(
            second[40:46].cpu(), torch.full((6,), 0x12, dtype=torch.uint8)
        )
        assert torch.equal(
            first[48:52].cpu(), torch.full((4,), 0x41, dtype=torch.uint8)
        )
        assert torch.equal(
            second[76:82].cpu(), torch.full((6,), 0x42, dtype=torch.uint8)
        )
        assert torch.equal(
            first[104:109].cpu(), torch.full((5,), 0x73, dtype=torch.uint8)
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
