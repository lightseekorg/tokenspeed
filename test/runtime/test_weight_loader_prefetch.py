import argparse
import contextlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.execution.weight_loader import WeightLoader
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
        self.assertFalse(load_config.use_modelscope)

    def test_modelscope_setting_reaches_weight_loader_config(self):
        server_args = SimpleNamespace(
            load_format="auto",
            download_dir=None,
            ext_yaml=None,
            weight_loader_prefetch_checkpoints=False,
            weight_loader_prefetch_num_threads=4,
            use_modelscope=True,
            kv_cache_dtype="bfloat16",
            quantization_param_path=None,
        )
        memory_saver_adapter = mock.Mock()
        memory_saver_adapter.region.return_value = contextlib.nullcontext()
        model = object()

        with (
            mock.patch(
                "tokenspeed.runtime.execution.weight_loader.get_available_gpu_memory",
                return_value=1,
            ),
            mock.patch("tokenspeed.runtime.execution.weight_loader.set_cuda_arch"),
            mock.patch("tokenspeed.runtime.execution.weight_loader.DeviceConfig"),
            mock.patch(
                "tokenspeed.runtime.execution.weight_loader.get_model",
                return_value=model,
            ) as get_model,
        ):
            loaded = WeightLoader.load_model(
                model_config=SimpleNamespace(dtype="bfloat16"),
                server_args=server_args,
                device="cpu",
                gpu_id=0,
                memory_saver_adapter=memory_saver_adapter,
            )

        self.assertIs(loaded, model)
        load_config = get_model.call_args.kwargs["load_config"]
        self.assertTrue(load_config.use_modelscope)

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
