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

"""GPU D2H/L3/H2D byte tests, including independent native Mooncake clients.

The default test uses MemoryKvStore. Set MOONCAKE_GPU_TEST_CONFIG to a JSON
Mooncake configuration with global_segment_size=0 to also run native tests.
Start a master and an independent storage owner first. Each native writer
exits before its reader starts, so neither GPU nor Host allocations survive.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import ExitStack
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from test.runtime.test_host_executor import _synthetic_arena

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")


class TestL3FlatKvRoundTripTest:
    def setup_method(self):
        try:
            import torch

            import tokenspeed.runtime.cache.l2.executor as executor_module
            from tokenspeed.runtime.cache.l3.backend import MemoryKvStore
            from tokenspeed.runtime.cache.transfer.layout import (
                CacheField,
                CacheGroupLayout,
                CacheTransferLayout,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            if os.environ.get("MOONCAKE_GPU_TEST_CONFIG"):
                raise
            pytest.skip(f"needs runtime dependencies: {exc}")
        if not torch.cuda.is_available():
            if os.environ.get("MOONCAKE_GPU_TEST_CONFIG"):
                pytest.fail("native Mooncake GPU test requires a CUDA device")
            pytest.skip("needs a CUDA device")
        self.torch = torch
        self.executor_module = executor_module
        self.MemoryKvStore = MemoryKvStore
        self.CacheField = CacheField
        self.CacheGroupLayout = CacheGroupLayout
        self.CacheTransferLayout = CacheTransferLayout

    def test_d2h_backup_host_wipe_prefetch_h2d_restores_bytes(self):
        self._exercise(
            backend="memory",
            phase="roundtrip",
            io_backend="direct",
            namespace=f"e2e-{uuid.uuid4().hex}",
            native_config=None,
        )

    def test_native_mooncake_survives_writer_process_exit(self):
        raw = os.environ.get("MOONCAKE_GPU_TEST_CONFIG")
        if not raw:
            pytest.skip("set MOONCAKE_GPU_TEST_CONFIG for a live storage owner")
        config = json.loads(raw)
        assert config["global_segment_size"] == 0
        for io_backend in ("direct", "kernel"):
            self._native_roundtrip(layout_kind="flat", io_backend=io_backend)

    @pytest.mark.parametrize("backend", ["memory", "mooncake"])
    def test_qwen_packed_ssm_conv_restores_every_field(self, backend):
        if backend == "mooncake":
            raw = os.environ.get("MOONCAKE_GPU_TEST_CONFIG")
            if not raw:
                pytest.skip("set MOONCAKE_GPU_TEST_CONFIG for a live storage owner")
            assert json.loads(raw)["global_segment_size"] == 0
            self._native_roundtrip(layout_kind="qwen_gdn", io_backend="kernel")
            return

        store = self.MemoryKvStore()
        try:
            namespace = f"qwen-packed-{uuid.uuid4().hex}"
            for phase in ("put", "get"):
                self._exercise_qwen(
                    phase=phase, namespace=namespace, store=store, io_backend="kernel"
                )
        finally:
            store.close()

    def _native_roundtrip(self, *, layout_kind, io_backend):
        namespace = f"gpu-native-{uuid.uuid4().hex}"
        for phase in ("put", "get"):
            result = subprocess.run(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--native-worker",
                    phase,
                    "--io-backend",
                    io_backend,
                    "--layout-kind",
                    layout_kind,
                    "--namespace",
                    namespace,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
            print(
                f"native Mooncake {layout_kind} {io_backend} {phase}: passed",
                flush=True,
            )

    def _complete(self, executor, *, op_id, expected):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            completed = executor.poll().completions
            if completed:
                assert len(completed) == 1
                assert completed[0].op_id == op_id
                assert completed[0].success == expected
                assert completed[0].retired
                return
            time.sleep(0.001)
        pytest.fail(f"storage operation {op_id} did not complete")

    def _host_executor(self, pool, *, io_backend, host_ratio):
        from tokenspeed.runtime.cache.l2.sizing import RetractionPoolRequest

        return self.executor_module.HostCacheExecutor(
            pool,
            l2_tier=True,
            snapshot_pool=RetractionPoolRequest(
                host_gb=0.0,
                ratio=0.0,
                max_retracted_requests=0,
                tail_lcm_blocks_per_request=0,
            ),
            slot_state_exporters=None,
            kvp_rank=0,
            host_ratio=host_ratio,
            host_size_gb=0,
            io_backend=io_backend,
            attn_tp_rank=0,
            nprocs_per_node=1,
            world_cpu_group=None,
        )

    def _l3_executor(self, executor, *, store, namespace, prefix_granularity):
        from tokenspeed.runtime.cache.l3.config import StorageRuntimeConfig
        from tokenspeed.runtime.cache.l3.executor import L3Executor

        return L3Executor(
            store,
            executor.host_storage,
            key_prefix=namespace,
            rank=0,
            ownership=None,
            prefix_for_weight_version=lambda version: f"{namespace}-{version}",
            config=StorageRuntimeConfig.from_extra_config(
                None,
                host_blocks=executor.host_storage.num_host_lcm_blocks,
                prefix_granularity=prefix_granularity,
            ),
        )

    def _exercise_qwen(self, *, phase, namespace, store, io_backend):
        from test.runtime.test_cache_setup import _qwen_gdn_setup

        from tokenspeed.runtime.cache.l3.executor import StorageJob
        from tokenspeed.runtime.layers.attention.kv_cache.factory import (
            create_cache_arena,
        )
        from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
            HybridMHATokenToKVPool,
        )

        torch = self.torch
        spec = _qwen_gdn_setup(2).spec
        plan = spec.memory_plan
        arena = create_cache_arena(spec, device="cuda", enable_memory_saver=False)
        pool = HybridMHATokenToKVPool(
            arena=arena,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=128,
            layer_num=len(spec.layer_types),
            rank=0,
            layer_types=spec.layer_types,
        )
        layout = pool.cache_transfer_layout()
        groups = {group.group_id: index for index, group in enumerate(layout.groups)}
        # Distinct parents keep different logical groups from aliasing each other.
        source_blocks = {group_id: index + 1 for group_id, index in groups.items()}
        destination_blocks = {group_id: index + 5 for group_id, index in groups.items()}
        keys = [(index, f"qwen-{index}", 0) for index in groups.values()]
        payloads = {
            field.field_id: (
                torch.arange(field.payload_bytes, dtype=torch.int32) * 17 + index * 23
            )
            .remainder(251)
            .to(torch.uint8)
            for index, field in enumerate(plan.fields)
        }
        with ExitStack() as cleanup:
            executor = self._host_executor(
                pool,
                io_backend=io_backend,
                host_ratio=8 / plan.num_lcm_blocks,
            )
            cleanup.callback(executor.shutdown)
            shared_backend = store is not None
            if store is None:
                from tokenspeed.runtime.cache.l3.mooncake import MooncakeKvStore

                store = MooncakeKvStore(
                    json.loads(os.environ["MOONCAKE_GPU_TEST_CONFIG"]),
                    host_buffer=executor.host_storage.host_buffer,
                    tp_size=1,
                    pp_size=1,
                )
            l3 = self._l3_executor(
                executor,
                store=store,
                namespace=namespace,
                prefix_granularity=plan.prefix_granularity,
            )
            # Memory's writer and reader share only the backend, never an arena
            # or Host allocation. Native workers additionally exit between phases.
            cleanup.callback(
                l3._shutdown_workers if shared_backend else l3.close, timeout_s=5.0
            )
            executor.host_storage.host_buffer.fill_(0)
            arena.buffer.fill_(0xEE)
            if phase == "put":
                for field in plan.fields:
                    offset = plan.field_page_byte_offset(
                        field.field_id, source_blocks[field.group_id]
                    )
                    arena.buffer[offset : offset + field.payload_bytes].copy_(
                        payloads[field.field_id]
                    )
                executor._start_writing(  # pylint: disable=protected-access
                    [7],
                    [
                        (index, source_blocks[group_id], source_blocks[group_id])
                        for group_id, index in groups.items()
                    ],
                    lane=executor._pinned_write_lane,
                    prerequisite_stream=torch.cuda.current_stream(),
                )
                executor.write_stream.synchronize()
                assert [int(event.op_id) for event in executor.poll_results()] == [7]
                objects, offsets, sizes = l3._page_ranges(
                    [
                        (g, block, h, o)
                        for (g, h, o), block in zip(
                            keys, source_blocks.values(), strict=True
                        )
                    ]
                )
                l3.submit(
                    StorageJob(1, "put", tuple(objects), tuple(offsets), tuple(sizes))
                )
                self._complete(l3, op_id=1, expected=(True,) * len(groups))
                return

            objects, offsets, sizes = l3._page_ranges(
                [
                    (g, block, h, o)
                    for (g, h, o), block in zip(
                        keys, destination_blocks.values(), strict=True
                    )
                ]
            )
            l3.submit(
                StorageJob(1, "get", tuple(objects), tuple(offsets), tuple(sizes))
            )
            self._complete(l3, op_id=1, expected=(True,) * len(groups))
            load_index = executor._start_loading(  # pylint: disable=protected-access
                [9],
                [
                    (index, destination_blocks[group_id], destination_blocks[group_id])
                    for group_id, index in groups.items()
                ],
                prerequisite_stream=torch.cuda.current_stream(),
            )
            assert load_index is not None
            pool.layerwise_load_tracker.set_consumers(load_index)
            for layer_id in range(len(spec.layer_types)):
                pool.layerwise_load_tracker.wait_for_layer(layer_id)
            torch.cuda.synchronize()
            assert [int(event.op_id) for event in executor.poll_results()] == [9]
            expected = torch.full((plan.arena_bytes,), 0xEE, dtype=torch.uint8)
            # The oracle uses recipe fields, not the transfer descriptor under test.
            # Compare the entire allocation to catch missing conv/SSM bytes and
            # writes into padding, the null parent, or an unrelated block.
            for field in plan.fields:
                offset = plan.field_page_byte_offset(
                    field.field_id, destination_blocks[field.group_id]
                )
                expected[offset : offset + field.payload_bytes] = payloads[
                    field.field_id
                ]
            assert torch.equal(arena.buffer.cpu(), expected)
            assert store.remove_by_prefix(namespace)

    def _exercise(self, *, backend, phase, io_backend, namespace, native_config):
        with ExitStack() as cleanup:
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
                    self.arena = _synthetic_arena(layout)

                def cache_transfer_layout(self):
                    return layout

                def register_layerwise_load_tracker(self, tracker):
                    self.load_tracker = tracker

            pool = SyntheticPool()
            executor = self._host_executor(pool, io_backend=io_backend, host_ratio=1.0)
            if backend == "memory":
                store = self.MemoryKvStore()
            else:
                assert backend == "mooncake"
                from tokenspeed.runtime.cache.l3.mooncake import MooncakeKvStore

                store = MooncakeKvStore(
                    native_config,
                    host_buffer=executor.host_storage.host_buffer,
                    tp_size=1,
                    pp_size=1,
                )
            from tokenspeed_scheduler import StorageKind

            l3 = self._l3_executor(
                executor, store=store, namespace=namespace, prefix_granularity=2
            )
            cleanup.callback(l3.close, timeout_s=5.0)
            cleanup.callback(executor.shutdown)

            keys = [(0, "h0", 0), (0, "h1", 0), (1, "h2", 0)]
            if phase in ("put", "roundtrip"):
                l3.submit_operations(
                    [SimpleNamespace(op_id=1, kind=StorageKind.Query, keys=keys)]
                )
                self._complete(l3, op_id=1, expected=(False,) * 3)
                first[16:20].fill_(0x11)
                second[28:34].fill_(0x12)
                first[40:44].fill_(0x41)
                second[64:70].fill_(0x42)
                first[94:99].fill_(0x73)
                torch.cuda.synchronize()

                executor._start_writing(  # pylint: disable=protected-access
                    [7],
                    [(0, 1, 1), (0, 4, 4), (1, 3, 3)],
                    lane=executor._pinned_write_lane,
                    prerequisite_stream=torch.cuda.current_stream(),
                )
                executor.write_stream.synchronize()
                write_results = executor.poll_results()
                assert [int(event.op_id) for event in write_results] == [7]
                l3.submit_operations(
                    [
                        SimpleNamespace(
                            op_id=2,
                            kind=StorageKind.Put,
                            keys=keys,
                            host_blocks=[1, 4, 3],
                        )
                    ]
                )
                self._complete(l3, op_id=2, expected=(True,) * 3)
                if phase == "put":
                    return

            # Prove restore comes from L3, not leftover Host bytes.
            executor.host_storage.host_buffer.fill_(0)
            first.fill_(0xEE)
            second.fill_(0xEE)
            torch.cuda.synchronize()
            assert not bool(executor.host_storage.host_buffer.any().item())

            l3.submit_operations(
                [
                    SimpleNamespace(
                        op_id=3, kind=StorageKind.Get, keys=keys, host_blocks=[2, 5, 4]
                    )
                ]
            )
            self._complete(l3, op_id=3, expected=(True,) * 3)
            assert bool(executor.host_storage.host_buffer.any().item())

            load_index = executor._start_loading(  # pylint: disable=protected-access
                [9],
                [(0, 2, 2), (0, 5, 5), (1, 4, 4)],
                prerequisite_stream=torch.cuda.current_stream(),
            )
            assert load_index is not None
            pool.load_tracker.set_consumers(load_index)
            pool.load_tracker.wait_for_layer(0)
            pool.load_tracker.wait_for_layer(1)
            torch.cuda.synchronize()
            load_results = executor.poll_results()
            assert [int(event.op_id) for event in load_results] == [9]

            # Check entire allocations, including padding and untouched blocks.
            expected_first = torch.full((128,), 0xEE, dtype=torch.uint8)
            expected_second = torch.full((128,), 0xEE, dtype=torch.uint8)
            expected_first[24:28].fill_(0x11)
            expected_second[40:46].fill_(0x12)
            expected_first[48:52].fill_(0x41)
            expected_second[76:82].fill_(0x42)
            expected_first[104:109].fill_(0x73)
            assert torch.equal(first.cpu(), expected_first)
            assert torch.equal(second.cpu(), expected_second)

            assert l3.backend.remove_by_prefix(namespace)
            executor.host_storage.host_buffer.fill_(0)
            l3.submit_operations(
                [
                    SimpleNamespace(
                        op_id=4, kind=StorageKind.Get, keys=keys, host_blocks=[2, 5, 4]
                    )
                ]
            )
            self._complete(l3, op_id=4, expected=(False,) * 3)
            assert not bool(executor.host_storage.host_buffer.any().item())


if __name__ == "__main__":
    if "--native-worker" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--native-worker", choices=("put", "get"), required=True)
        parser.add_argument("--io-backend", choices=("direct", "kernel"), required=True)
        parser.add_argument("--namespace", required=True)
        parser.add_argument(
            "--layout-kind", choices=("flat", "qwen_gdn"), required=True
        )
        args = parser.parse_args()
        case = TestL3FlatKvRoundTripTest()
        case.setup_method()
        if args.layout_kind == "qwen_gdn":
            case._exercise_qwen(
                phase=args.native_worker,
                namespace=args.namespace,
                store=None,
                io_backend=args.io_backend,
            )
        else:
            case._exercise(
                backend="mooncake",
                phase=args.native_worker,
                io_backend=args.io_backend,
                namespace=args.namespace,
                native_config=json.loads(os.environ["MOONCAKE_GPU_TEST_CONFIG"]),
            )
    else:
        raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
