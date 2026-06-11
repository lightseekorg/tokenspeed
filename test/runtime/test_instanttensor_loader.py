import argparse
import glob
import os
import sys
import tempfile
import unittest
from importlib.util import find_spec

import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.load_config import LoadConfig, LoadFormat
from tokenspeed.runtime.model_loader.loader import DefaultModelLoader
from tokenspeed.runtime.model_loader.weight_utils import (
    download_weights_from_hf,
    instanttensor_weights_iterator,
    safetensors_weights_iterator,
)
from tokenspeed.runtime.utils.server_args import ServerArgs

INSTANTTENSOR_AVAILABLE = find_spec("instanttensor") is not None
# InstantTensor is NVIDIA-only. torch.cuda.is_available() is also True on ROCm,
# so guard on the platform vendor to keep the parity test off AMD runners.
IS_NVIDIA = current_platform().is_nvidia


class TestInstantTensorConfig(unittest.TestCase):
    """Config/CLI wiring that needs neither a GPU nor instanttensor."""

    def test_cli_flag_maps_to_load_format(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(
            ["--model", "test/model", "--load-format", "instanttensor"]
        )
        self.assertEqual(args.load_format, "instanttensor")

    def test_load_config_normalizes_to_enum(self):
        load_config = LoadConfig(load_format="instanttensor")
        self.assertEqual(load_config.load_format, LoadFormat.INSTANTTENSOR)

    def test_prepare_weights_treats_instanttensor_as_safetensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # _prepare_weights only globs/filters file paths; it never reads the
            # tensor data, so an empty placeholder shard is sufficient here.
            open(os.path.join(tmpdir, "model.safetensors"), "wb").close()

            loader = DefaultModelLoader(LoadConfig(load_format="instanttensor"))
            _, hf_weights_files, use_safetensors = loader._prepare_weights(
                tmpdir, revision=None, fall_back_to_pt=False
            )

            self.assertTrue(use_safetensors)
            self.assertEqual(len(hf_weights_files), 1)
            self.assertTrue(hf_weights_files[0].endswith("model.safetensors"))


@unittest.skipIf(not IS_NVIDIA, "InstantTensor requires NVIDIA GPUs")
@unittest.skipIf(not INSTANTTENSOR_AVAILABLE, "instanttensor is not installed")
class TestInstantTensorWeights(unittest.TestCase):
    """Iterator parity test (requires an NVIDIA GPU and instanttensor)."""

    def test_instanttensor_matches_safetensors(self):
        model = "openai-community/gpt2"
        with tempfile.TemporaryDirectory() as tmpdir:
            download_weights_from_hf(
                model, cache_dir=tmpdir, allow_patterns=["*.safetensors"]
            )
            safetensors_files = glob.glob(f"{tmpdir}/**/*.safetensors", recursive=True)
            self.assertGreater(len(safetensors_files), 0)

            instanttensor_tensors = {}
            for name, tensor in instanttensor_weights_iterator(safetensors_files):
                # Copy immediately in case InstantTensor exposes internal buffers.
                instanttensor_tensors[name] = tensor.to("cpu")

            reference_tensors = dict(safetensors_weights_iterator(safetensors_files))

            self.assertEqual(len(instanttensor_tensors), len(reference_tensors))
            for name, got in instanttensor_tensors.items():
                ref = reference_tensors[name]
                self.assertEqual(got.dtype, ref.dtype)
                self.assertEqual(got.shape, ref.shape)
                self.assertTrue(torch.equal(got, ref))


if __name__ == "__main__":
    unittest.main()
