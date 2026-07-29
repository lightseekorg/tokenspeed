import argparse
import os
import sys
import tempfile
import time
import unittest
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.model_loader import weight_utils
from tokenspeed.runtime.model_loader.weight_utils import CheckpointPrefetcher
from tokenspeed.runtime.utils.server_args import ServerArgs

_POLL_TIMEOUT_S = 5.0


def _wait_until(predicate, timeout=_POLL_TIMEOUT_S):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _fake_available_memory(available):
    """Patch psutil so the prefetch window computes to available / 4."""
    return mock.patch.object(
        weight_utils.psutil,
        "virtual_memory",
        return_value=mock.Mock(available=available),
    )


class TestWeightLoaderPrefetch(unittest.TestCase):
    def test_cli_flag_maps_to_server_args(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(
            [
                "--model",
                "test/model",
                "--weight-loader-prefetch-checkpoints",
                "--weight-loader-prefetch-num-threads",
                "2",
            ]
        )
        with mock.patch.object(ServerArgs, "__post_init__"):
            server_args = ServerArgs.from_cli_args(args)

        self.assertTrue(server_args.weight_loader_prefetch_checkpoints)
        self.assertEqual(server_args.weight_loader_prefetch_num_threads, 2)

    def test_load_config_defaults_keep_prefetch_disabled(self):
        load_config = LoadConfig()

        self.assertFalse(load_config.weight_loader_prefetch_checkpoints)
        self.assertEqual(load_config.weight_loader_prefetch_num_threads, 4)

    def _make_files(self, tmpdir, count, size):
        files = []
        for idx in range(count):
            path = os.path.join(tmpdir, f"model-{idx}.safetensors")
            with open(path, "wb") as f:
                f.write(b"x" * size)
            files.append(path)
        return files

    def test_window_clamps_to_available_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._make_files(tmpdir, count=1, size=100)
            with _fake_available_memory(100 * 1024**3):
                prefetcher = CheckpointPrefetcher(files)
            self.assertEqual(prefetcher._window_bytes, 25 * 1024**3)
            with _fake_available_memory(1000 * 1024**3):
                prefetcher = CheckpointPrefetcher(files)
            self.assertEqual(
                prefetcher._window_bytes, CheckpointPrefetcher._WINDOW_MAX_BYTES
            )

    def test_window_bounds_read_ahead_and_advances_on_consumption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._make_files(tmpdir, count=4, size=100)
            read_order = []

            def record_read(path):
                read_order.append(path)
                return 100

            # available=1000 -> window of 250 bytes: fits two 100-byte shards;
            # the third must wait for the consumer to advance.
            with (
                _fake_available_memory(1000),
                mock.patch.object(
                    CheckpointPrefetcher, "_read_file", side_effect=record_read
                ),
            ):
                prefetcher = CheckpointPrefetcher(files, num_threads=1)
                prefetcher.start()

                self.assertTrue(_wait_until(lambda: len(read_order) == 2))
                prefetcher.wait_file(0)
                prefetcher.wait_file(1)
                # Third shard would exceed the window until we consume one.
                time.sleep(0.05)
                self.assertEqual(read_order, files[:2])

                prefetcher.advance(0)
                self.assertTrue(_wait_until(lambda: len(read_order) == 3))
                self.assertEqual(read_order, files[:3])

                prefetcher.advance(1)
                prefetcher.advance(2)
                self.assertTrue(_wait_until(lambda: read_order == files))
                prefetcher.wait_file(3)

    def test_oversized_shard_still_prefetched_alone(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._make_files(tmpdir, count=2, size=100)
            read_order = []

            # available=40 -> window of 10 bytes, smaller than one shard.
            with (
                _fake_available_memory(40),
                mock.patch.object(
                    CheckpointPrefetcher,
                    "_read_file",
                    side_effect=lambda p: read_order.append(p) or 100,
                ),
            ):
                prefetcher = CheckpointPrefetcher(files, num_threads=1)
                prefetcher.start()
                prefetcher.wait_file(0)
                prefetcher.advance(0)
                prefetcher.wait_file(1)
                self.assertEqual(read_order, files)

    def test_read_failure_unblocks_consumer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._make_files(tmpdir, count=1, size=100)

            def broken_read(path):
                raise OSError("boom")

            with mock.patch.object(
                CheckpointPrefetcher, "_read_file", side_effect=broken_read
            ):
                prefetcher = CheckpointPrefetcher(files, num_threads=1)
                prefetcher.start()
                # Must not hang; the consumer falls back to demand paging.
                prefetcher.wait_file(0)


if __name__ == "__main__":
    unittest.main()
