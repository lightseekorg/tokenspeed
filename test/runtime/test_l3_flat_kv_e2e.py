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

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")


class L3FlatKvRoundTripTest(unittest.TestCase):
    def setUp(self):
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
            self.skipTest(f"needs runtime dependencies: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device")
        self.torch = torch
        self.executor_module = executor_module
        self.MemoryKvStore = MemoryKvStore
        self.CacheField = CacheField
        self.CacheGroupLayout = CacheGroupLayout
        self.CacheTransferLayout = CacheTransferLayout

    def test_d2h_backup_host_wipe_prefetch_h2d_restores_bytes(self):
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
            def cache_transfer_layout(self):
                return layout

            def register_layerwise_load_tracker(self, tracker):
                self.load_tracker = tracker

        pool = SyntheticPool()
        pool.arena = SimpleNamespace(
            cache_group_specs=tuple(
                SimpleNamespace(group_id=group.group_id) for group in layout.groups
            ),
        )
        with patch.object(self.executor_module, "_HOST_MEM_HEADROOM_BYTES", 0):
            executor = self.executor_module.L2CacheExecutor(
                pool,
                host_ratio=1.0,
                host_size_gb=0,
                io_backend="direct",
            )
        store = self.MemoryKvStore()
        executor.attach_l3_storage(store, key_prefix="e2e", rank=0)

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
        )
        torch.cuda.current_stream().synchronize()
        write_results = executor.poll_results()
        self.assertEqual([int(event.op_id) for event in write_results], [7])
        self.assertEqual(executor.l3_store.exists(backup_pages), [True, True, True])

        # Prove restore comes from L3, not leftover Host bytes.
        executor.host_storage.host_buffer.fill_(0)
        first.fill_(0xEE)
        second.fill_(0xEE)
        torch.cuda.synchronize()
        self.assertFalse(bool(executor.host_storage.host_buffer.any().item()))

        executor._prefetch_from_storage(
            backup_pages
        )  # pylint: disable=protected-access
        self.assertTrue(bool(executor.host_storage.host_buffer.any().item()))

        load_index = executor._start_loading(  # pylint: disable=protected-access
            [9],
            [(0, 2, 1), (0, 5, 4), (1, 4, 3)],
        )
        self.assertIsNotNone(load_index)
        pool.load_tracker.set_consumers(load_index)
        pool.load_tracker.wait_for_layer(0)
        pool.load_tracker.wait_for_layer(1)
        torch.cuda.synchronize()
        load_results = executor.poll_results()
        self.assertEqual([int(event.op_id) for event in load_results], [9])

        self.assertTrue(
            torch.equal(first[24:28].cpu(), torch.full((4,), 0x11, dtype=torch.uint8))
        )
        self.assertTrue(
            torch.equal(second[40:46].cpu(), torch.full((6,), 0x12, dtype=torch.uint8))
        )
        self.assertTrue(
            torch.equal(first[48:52].cpu(), torch.full((4,), 0x41, dtype=torch.uint8))
        )
        self.assertTrue(
            torch.equal(second[76:82].cpu(), torch.full((6,), 0x42, dtype=torch.uint8))
        )
        self.assertTrue(
            torch.equal(first[104:109].cpu(), torch.full((5,), 0x73, dtype=torch.uint8))
        )
        executor.shutdown()


if __name__ == "__main__":
    unittest.main()
