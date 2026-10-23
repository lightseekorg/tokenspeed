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

"""Opt-in RDMA owner relocation test using MOONCAKE_GPU_TEST_CONFIG.

Requires two GPUs, a master and an independent memory storage owner. Four
sequential processes exercise both owners of a PUT and GET. No writer Host or
GPU allocation survives the reader, and the new Host and Device pages have a
different owner. This validates transfer ownership, not model CP execution.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import unittest
import uuid
from dataclasses import dataclass
from pathlib import Path

# The owner processes import the shared test helpers from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def exercise_owner(*, phase: str, rank: int, namespace: str) -> None:
    from test.l3_test_utils import (
        StorageOp,
        host_storage_executor,
        run_storage,
        store_library,
    )

    import torch
    from tokenspeed_scheduler import StorageKind

    from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor
    from tokenspeed.runtime.cache.l2.sizing import RetractionPoolRequest
    from tokenspeed.runtime.cache.l3.config import MooncakeStoreSettings
    from tokenspeed.runtime.cache.transfer.layout import (
        CacheField,
        CacheGroupLayout,
        CacheTransferLayout,
    )
    from tokenspeed.runtime.cache.transfer.ownership import BlockOwnerTranslation
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
        CacheRuntimeContract,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec

    torch.cuda.set_device(rank)
    size = 4096
    buffer = torch.full((5 * size,), 0xA5, dtype=torch.uint8, device="cuda")
    layout = CacheTransferLayout(
        num_lcm_blocks=4,
        groups=(
            CacheGroupLayout("full", 1, (CacheField("layer.0.kv", 0, 0, size, size),)),
        ),
        buffers=(buffer,),
        consumers=(("layer.0.kv",),),
    )
    contract = CacheRuntimeContract(
        prefix_granularity=1,
        num_lcm_blocks=4,
        token_capacity=8,
        group_specs=(
            CacheGroupSpec(
                "full",
                "full_history",
                rows_per_page=1,
                entry_stride_tokens=1,
                replayable=False,
                shard_count=2,
            ),
        ),
        group_page_counts={"full": 5},
        group_packing={"full": 1},
    )

    @dataclass(frozen=True)
    class Arena:
        runtime_contract: CacheRuntimeContract

    class Pool:
        def __init__(self):
            self.arena = Arena(contract)
            self.load_tracker = None

        def cache_transfer_layout(self):
            return layout

        def register_layerwise_load_tracker(self, tracker):
            self.load_tracker = tracker

    executor = HostCacheExecutor(
        Pool(),
        l2_tier=True,
        host_ratio=1,
        host_size_gb=0,
        snapshot_pool=RetractionPoolRequest(
            host_gb=0, ratio=0, max_retracted_requests=0, tail_lcm_blocks_per_request=0
        ),
        slot_state_exporters=None,
        io_backend="kernel",
        attn_tp_rank=rank,
        kvp_rank=rank,
        # These owner processes run sequentially without a process group.
        nprocs_per_node=1,
        world_cpu_group=None,
    )
    config = json.loads(os.environ["MOONCAKE_GPU_TEST_CONFIG"])
    assert config["global_segment_size"] == 0 and config["protocol"] == "rdma"
    ownership = BlockOwnerTranslation.for_host_pool(
        layout, contract, num_host_lcm_blocks=4, rank=rank
    )
    l3 = host_storage_executor(
        executor.host_storage,
        shard_counts=ownership.shard_counts,
        rank=rank,
        deployment_id=namespace,
        mooncake=MooncakeStoreSettings.resolve(
            config, library=store_library(), tp_size=2, pp_size=1
        ),
    )
    expected = torch.arange(size, dtype=torch.int32).remainder(251).to(torch.uint8)
    try:
        if phase == "put":
            buffer[size : 2 * size].copy_(expected)
            torch.cuda.synchronize()
            rows = [row for _, row in ownership.owned_positions([(0, 2, 2)])]
            if rows:
                executor._start_writing(
                    [1],
                    rows,
                    lane=executor._pinned_write_lane,
                    prerequisite_stream=torch.cuda.current_stream(),
                )
                executor.write_stream.synchronize()
                executor.poll_results()
            block = 2
        else:
            executor.host_storage.host_buffer.fill_(0)
            block = 3
        kind = StorageKind.Put if phase == "put" else StorageKind.Get
        done = run_storage(
            l3, StorageOp(1, kind, [(0, "h", 0)], [block]), timeout_s=30.0
        )
        assert done[1:] == ([True], True, [False])
        if phase == "get":
            rows = [row for _, row in ownership.owned_positions([(0, 5, 3)])]
            if rows:
                executor._start_loading(
                    [2], rows, prerequisite_stream=torch.cuda.current_stream()
                )
                executor.load_stream.synchronize()
                executor.poll_results()
                assert torch.equal(buffer[3 * size : 4 * size].cpu(), expected)
            else:
                assert bool((buffer == 0xA5).all().item())
        owned = rank == (1 if phase == "put" else 0)
        operations = l3.snapshot()[0]
        # Each row's successful bytes (``L3Executor.snapshot``).
        assert sum(row[4] for row in operations) == (size if owned else 0)
    finally:
        l3.close(5.0)
        executor.shutdown()


class L3KvpGpuTest(unittest.TestCase):
    def test_rdma_object_survives_writer_exit_and_new_cyclic_owner(self):
        if not os.environ.get("MOONCAKE_GPU_TEST_CONFIG"):
            self.skipTest("set MOONCAKE_GPU_TEST_CONFIG for a live RDMA memory store")
        import torch

        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        namespace = f"kvp-native-{uuid.uuid4().hex}"
        for phase in ("put", "get"):
            for rank in (0, 1):
                with self.subTest(phase=phase, rank=rank):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--phase",
                            phase,
                            "--rank",
                            str(rank),
                            "--namespace",
                            namespace,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        check=False,
                    )
                    self.assertEqual(
                        result.returncode, 0, f"{result.stdout}\n{result.stderr}"
                    )


if __name__ == "__main__":
    if "--phase" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--phase", choices=("put", "get"), required=True)
        parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
        parser.add_argument("--namespace", required=True)
        args = parser.parse_args()
        exercise_owner(phase=args.phase, rank=args.rank, namespace=args.namespace)
    else:
        unittest.main()
