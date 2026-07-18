import argparse
import os
import sys
import unittest
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.model_loader import weight_utils
from tokenspeed.runtime.utils.server_args import ServerArgs


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

    def test_prefetch_splits_files_by_local_rank(self):
        files = [f"/tmp/model-{idx}.safetensors" for idx in range(6)]
        seen = []

        def record_prefetch(path):
            seen.append(path)
            return 1

        env = {"LOCAL_RANK": "1", "LOCAL_WORLD_SIZE": "3"}
        with (
            mock.patch.dict(os.environ, env, clear=False),
            mock.patch.object(
                weight_utils, "_prefetch_checkpoint_file", side_effect=record_prefetch
            ),
        ):
            thread = weight_utils.prefetch_checkpoint_files(files, num_threads=2)
            thread.join(timeout=5)

        self.assertFalse(thread.is_alive())
        self.assertEqual(sorted(seen), [files[1], files[4]])


if __name__ == "__main__":
    unittest.main()
