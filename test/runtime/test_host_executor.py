"""Compact Host cache executor tests."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")


class GroupAwareWireTest(unittest.TestCase):
    def test_submit_preserves_group_identity(self):
        try:
            from tokenspeed.runtime.cache.l2.executor import L2CacheExecutor
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

        op_ids = []
        transfers = []
        L2CacheExecutor._submit(
            [7],
            [[0, 1]],
            [[5, 5]],
            [[9, 9]],
            pending_op_ids=op_ids,
            pending_transfers=transfers,
            source_is_device=True,
        )
        self.assertEqual(op_ids, [7])
        self.assertEqual(transfers, [(0, 5, 9), (1, 5, 9)])

    def test_writeback_calls_transfer_with_compact_layout(self):
        try:
            import tokenspeed.runtime.cache.l2.executor as executor_module

            L2CacheExecutor = executor_module.L2CacheExecutor
        except (ImportError, ModuleNotFoundError) as exc:
            self.skipTest(f"needs runtime dependencies: {exc}")

        executor = L2CacheExecutor.__new__(L2CacheExecutor)
        executor._pending_write_op_ids = [7]
        executor._pending_write_transfers = [(0, 5, 9)]
        executor._immediate_write_op_ids = []
        executor.layout = SimpleNamespace(buffers=("device",))
        executor.host_storage = SimpleNamespace(host_buffer="host")
        executor.write_stream = object()
        executor.ack_write_queue = []
        ranges = [(0, 64, 128, 32)]
        executor._transfer_ranges = Mock(return_value=ranges)
        start = Mock()
        finish = Mock()

        with (
            patch.object(
                executor_module.torch.cuda, "Event", side_effect=(start, finish)
            ),
            patch.object(executor_module, "transfer_cache_ranges") as transfer,
        ):
            executor._start_writing()

        transfer.assert_called_once_with(
            "d2h",
            executor.layout.buffers,
            executor.host_storage.host_buffer,
            ranges,
            executor.write_stream,
        )
        start.record.assert_called_once_with()
        start.wait.assert_called_once_with(executor.write_stream)
        finish.record.assert_called_once_with(executor.write_stream)


if __name__ == "__main__":
    unittest.main()
