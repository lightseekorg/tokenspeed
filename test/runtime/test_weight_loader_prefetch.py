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


class _NullMemorySaverAdapter:
    @contextlib.contextmanager
    def region(self):
        yield


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

    def test_tokenspeed_mla_fp8_kv_requires_scaling_factors(self):
        server_args = SimpleNamespace(
            load_format="auto",
            download_dir=None,
            ext_yaml=None,
            weight_loader_prefetch_checkpoints=False,
            weight_loader_prefetch_num_threads=4,
            kv_cache_dtype="fp8_e4m3",
            quantization_param_path=None,
            attention_backend="tokenspeed_mla",
        )
        model_config = SimpleNamespace(
            dtype="bfloat16",
            hf_config=SimpleNamespace(architectures=["DeepseekV3ForCausalLM"]),
        )

        with (
            mock.patch(
                "tokenspeed.runtime.execution.weight_loader.set_cuda_arch",
            ),
            mock.patch(
                "tokenspeed.runtime.execution.weight_loader.get_available_gpu_memory",
                return_value=0,
            ),
            mock.patch(
                "tokenspeed.runtime.execution.weight_loader.get_model",
            ) as get_model,
        ):
            with self.assertRaisesRegex(RuntimeError, "DeepSeek V3"):
                WeightLoader.load_model(
                    model_config=model_config,
                    server_args=server_args,
                    device="cpu",
                    gpu_id=0,
                    memory_saver_adapter=_NullMemorySaverAdapter(),
                )

        get_model.assert_not_called()

    def test_fp8_kv_scale_guard_is_deepseek_v3_specific(self):
        server_args = SimpleNamespace(
            load_format="auto",
            download_dir=None,
            ext_yaml=None,
            weight_loader_prefetch_checkpoints=False,
            weight_loader_prefetch_num_threads=4,
            kv_cache_dtype="fp8_e4m3",
            quantization_param_path=None,
            attention_backend="tokenspeed_mla",
        )
        model_config = SimpleNamespace(
            dtype="bfloat16",
            hf_config=SimpleNamespace(architectures=["KimiK2ForCausalLM"]),
        )
        model = object()

        with (
            mock.patch(
                "tokenspeed.runtime.execution.weight_loader.set_cuda_arch",
            ),
            mock.patch(
                "tokenspeed.runtime.execution.weight_loader.get_available_gpu_memory",
                return_value=0,
            ),
            mock.patch(
                "tokenspeed.runtime.execution.weight_loader.get_model",
                return_value=model,
            ) as get_model,
        ):
            loaded = WeightLoader.load_model(
                model_config=model_config,
                server_args=server_args,
                device="cuda",
                gpu_id=0,
                memory_saver_adapter=_NullMemorySaverAdapter(),
            )

        self.assertIs(loaded, model)
        get_model.assert_called_once()


if __name__ == "__main__":
    unittest.main()
