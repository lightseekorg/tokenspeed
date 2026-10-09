import argparse
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.model_loader import weight_utils
from tokenspeed.runtime.model_loader.loader import DefaultModelLoader
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
    def _parse_server_args(self, cli):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(["--model", "test/model", *cli])
        with mock.patch.object(ServerArgs, "__post_init__"):
            return ServerArgs.from_cli_args(args)

    def test_prefetch_enabled_by_default(self):
        server_args = self._parse_server_args(
            ["--weight-loader-prefetch-num-threads", "2"]
        )
        self.assertTrue(server_args.weight_loader_prefetch_checkpoints)
        self.assertEqual(server_args.weight_loader_prefetch_num_threads, 2)

    def test_disable_flag_turns_prefetch_off(self):
        server_args = self._parse_server_args(
            ["--disable-weight-loader-prefetch-checkpoints"]
        )
        self.assertFalse(server_args.weight_loader_prefetch_checkpoints)

    def test_load_config_defaults_enable_prefetch(self):
        load_config = LoadConfig()

        self.assertTrue(load_config.weight_loader_prefetch_checkpoints)
        self.assertEqual(load_config.weight_loader_prefetch_num_threads, 8)

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
                prefetcher = CheckpointPrefetcher(
                    files, local_rank=0, local_world_size=1
                )
            self.assertEqual(prefetcher._window_bytes, 25 * 1024**3)
            with _fake_available_memory(1000 * 1024**3):
                prefetcher = CheckpointPrefetcher(
                    files, local_rank=0, local_world_size=1
                )
            self.assertEqual(
                prefetcher._window_bytes, CheckpointPrefetcher._WINDOW_MAX_BYTES
            )

    def test_window_bounds_read_ahead_and_advances_on_consumption(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._make_files(tmpdir, count=4, size=100)
            read_order = []

            def record_read(path, start, end):
                read_order.append(path)
                return 100

            # available=1000 -> window of 250 bytes: fits two 100-byte shards;
            # the third must wait for the consumer to advance.
            with (
                _fake_available_memory(1000),
                mock.patch.object(
                    CheckpointPrefetcher, "_read_range", side_effect=record_read
                ),
            ):
                prefetcher = CheckpointPrefetcher(
                    files, num_threads=1, local_rank=0, local_world_size=1
                )
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
                    "_read_range",
                    side_effect=lambda p, start, end: read_order.append(p) or 100,
                ),
            ):
                prefetcher = CheckpointPrefetcher(
                    files, num_threads=1, local_rank=0, local_world_size=1
                )
                prefetcher.start()
                prefetcher.wait_file(0)
                prefetcher.advance(0)
                prefetcher.wait_file(1)
                self.assertEqual(read_order, files)

    def test_parallel_ranges_cover_shard_and_wait_for_every_range(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._make_files(tmpdir, count=1, size=261)[0]
            release = threading.Event()
            last_finished = threading.Event()
            ranges = []
            original = CheckpointPrefetcher._read_range

            def read_range(file_path, start, end):
                if start == 0:
                    self.assertTrue(release.wait(_POLL_TIMEOUT_S))
                count = original(file_path, start, end)
                ranges.append((start, end))
                if end == 261:
                    last_finished.set()
                return count

            with (
                mock.patch.object(CheckpointPrefetcher, "_BLOCK_SIZE", 32),
                mock.patch.object(CheckpointPrefetcher, "_MIN_RANGE_SIZE", 64),
                mock.patch.object(
                    CheckpointPrefetcher, "_read_range", side_effect=read_range
                ),
            ):
                prefetcher = CheckpointPrefetcher(
                    [path], num_threads=4, local_rank=0, local_world_size=1
                )
                prefetcher.start()
                try:
                    self.assertTrue(last_finished.wait(_POLL_TIMEOUT_S))
                    self.assertFalse(prefetcher._ready[0].is_set())
                finally:
                    release.set()
                    prefetcher.close()
                self.assertTrue(prefetcher._ready[0].is_set())
            ranges.sort()
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(ranges[-1][1], 261)
            for previous, following in zip(ranges, ranges[1:]):
                self.assertEqual(previous[1], following[0])

    def test_local_readers_share_reads_without_expanding_the_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._make_files(tmpdir, count=6, size=100)
            reads = []
            with (
                _fake_available_memory(1000),
                mock.patch.object(
                    CheckpointPrefetcher,
                    "_read_range",
                    side_effect=lambda path, start, end: reads.append(path) or 100,
                ),
            ):
                readers = [
                    CheckpointPrefetcher(
                        files, num_threads=1, local_rank=rank, local_world_size=2
                    )
                    for rank in range(2)
                ]
                try:
                    for reader in readers:
                        reader.start()
                    for reader in readers:
                        reader.wait_file(0)
                        reader.wait_file(1)
                    self.assertCountEqual(reads, files[:2])
                    self.assertTrue(
                        all(reader._inflight_bytes == 200 for reader in readers)
                    )
                    for idx in range(len(files)):
                        for reader in readers:
                            reader.wait_file(idx)
                            reader.advance(idx)
                    self.assertCountEqual(reads, files)
                finally:
                    for reader in readers:
                        reader.close()

    def test_loader_uses_node_local_participants_and_keeps_filtered_reads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._make_files(tmpdir, count=2, size=100)
            loader = DefaultModelLoader(LoadConfig())
            source = DefaultModelLoader.Source(tmpdir, revision=None)
            with (
                mock.patch.object(
                    loader, "_prepare_weights", return_value=(tmpdir, files[::-1], True)
                ),
                mock.patch.object(
                    weight_utils.safetensors.torch,
                    "load_file",
                    return_value={"weight": weight_utils.torch.ones(1)},
                ),
                mock.patch.object(
                    CheckpointPrefetcher, "_read_range", return_value=100
                ) as read,
            ):
                # Global ranks 4 and 5 are the only participants on node 1.
                for rank in (4, 5):
                    list(
                        loader._get_weights_iterator(
                            source,
                            None,
                            (4, 5),
                            Mapping(rank=rank, world_size=8, nprocs_per_node=4),
                        )
                    )
                self.assertCountEqual(
                    [call.args[0] for call in read.call_args_list], files
                )

            # Different per-rank filters must not leave shards assigned to
            # a peer that never opens them. Each filtered iterator prefetches
            # its complete selected list independently.
            with (
                mock.patch.object(
                    loader, "_prepare_weights", return_value=(tmpdir, files, True)
                ),
                mock.patch(
                    "tokenspeed.runtime.model_loader.loader.filter_safetensors_files_by_weight_names",
                    side_effect=lambda paths, folder, index, accept: [
                        path for path in paths if accept(path)
                    ],
                ),
                mock.patch.object(weight_utils, "safe_open") as open_shard,
                mock.patch.object(
                    CheckpointPrefetcher, "_read_range", return_value=100
                ) as read,
            ):
                open_shard.return_value.__enter__.return_value.keys.return_value = []
                for rank, path in enumerate(files):
                    list(
                        loader._get_weights_iterator(
                            source,
                            lambda name, selected=path: name == selected,
                            None,
                            Mapping(rank=rank, world_size=2),
                        )
                    )
                self.assertCountEqual(
                    [call.args[0] for call in read.call_args_list], files
                )

    def test_iterator_close_stops_workers_waiting_for_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = [
                os.path.join(tmpdir, f"weights-{idx}.safetensors") for idx in range(2)
            ]
            for path in files:
                weight_utils.safetensors.torch.save_file(
                    {"weight": weight_utils.torch.ones(1)}, path
                )
            for filtered in [False, True]:
                with (
                    self.subTest(filtered=filtered),
                    _fake_available_memory(os.path.getsize(files[0]) * 4),
                ):
                    prefetcher = CheckpointPrefetcher(
                        files, num_threads=2, local_rank=0, local_world_size=1
                    )
                    with mock.patch.object(
                        weight_utils, "CheckpointPrefetcher", return_value=prefetcher
                    ):
                        if filtered:
                            iterator = (
                                weight_utils.safetensors_filtered_weights_iterator(
                                    files,
                                    lambda name: True,
                                    prefetch=True,
                                    local_rank=0,
                                    local_world_size=1,
                                )
                            )
                        else:
                            iterator = weight_utils.safetensors_weights_iterator(
                                files, prefetch=True, local_rank=0, local_world_size=1
                            )
                        try:
                            next(iterator)
                        finally:
                            iterator.close()
                    self.assertEqual(prefetcher._files_read, 1)
                    self.assertTrue(
                        all(not thread.is_alive() for thread in prefetcher._threads)
                    )

    def test_read_failure_unblocks_consumer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self._make_files(tmpdir, count=1, size=100)

            def broken_read(path, start, end):
                raise OSError("boom")

            with mock.patch.object(
                CheckpointPrefetcher, "_read_range", side_effect=broken_read
            ):
                prefetcher = CheckpointPrefetcher(
                    files, num_threads=1, local_rank=0, local_world_size=1
                )
                prefetcher.start()
                # Must not hang; the consumer falls back to demand paging.
                prefetcher.wait_file(0)


if __name__ == "__main__":
    unittest.main()
