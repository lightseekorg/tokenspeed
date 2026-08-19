import argparse
import glob
import json
import os
import struct
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
    _find_sub_byte_dtype,
    download_weights_from_hf,
    instanttensor_weights_iterator,
    safetensors_weights_iterator,
)
from tokenspeed.runtime.utils.server_args import ServerArgs

INSTANTTENSOR_AVAILABLE = find_spec("instanttensor") is not None
# torch.cuda.is_available() is True on ROCm too, so guard on the vendor.
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
            # _prepare_weights only globs paths, never tensor data.
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


class TestSubByteDtypeGuard(unittest.TestCase):
    """Sub-byte dtypes must be refused, since InstantTensor misloads them."""

    def _write(self, path: str, dtype: str, shape: list[int], nbytes: int) -> str:
        header = {"t": {"dtype": dtype, "shape": shape, "data_offsets": [0, nbytes]}}
        blob = json.dumps(header).encode()
        blob += b" " * ((8 - len(blob) % 8) % 8)
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(blob)))
            f.write(blob)
            f.write(b"\x11" * nbytes)
        return path

    def test_packed_fp4_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            shard = self._write(
                os.path.join(d, "model.safetensors"), "F4", [2048], 1024
            )
            self.assertEqual(_find_sub_byte_dtype([shard]), "F4")

    def test_byte_aligned_shard_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            shards = [
                self._write(os.path.join(d, "a.safetensors"), "BF16", [512], 1024),
                self._write(os.path.join(d, "b.safetensors"), "F8_E4M3", [1024], 1024),
                self._write(os.path.join(d, "c.safetensors"), "U8", [1024], 1024),
            ]
            self.assertIsNone(_find_sub_byte_dtype(shards))

    def test_unparsable_header_raises_rather_than_waving_the_shard_through(
        self,
    ) -> None:
        # Failing open would wave through the shard the guard cannot inspect.
        with tempfile.TemporaryDirectory() as d:
            cases = {
                "truncated.safetensors": b"\x00",
                "not_an_object.safetensors": struct.pack("<Q", 8) + b"[1,2,3] ",
                "absurd_header.safetensors": struct.pack("<Q", 2**62),
            }
            for filename, blob in cases.items():
                path = os.path.join(d, filename)
                with open(path, "wb") as f:
                    f.write(blob)
                with self.subTest(filename):
                    with self.assertRaises((ValueError, struct.error)):
                        _find_sub_byte_dtype([path])


@unittest.skipIf(not IS_NVIDIA, "InstantTensor requires NVIDIA GPUs")
@unittest.skipIf(not INSTANTTENSOR_AVAILABLE, "instanttensor is not installed")
class TestInstantTensorRejectsSubByteCheckpoints(unittest.TestCase):
    def test_iterator_raises_at_call_time_not_at_first_pull(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            shard = os.path.join(d, "model.safetensors")
            header = {"t": {"dtype": "F4", "shape": [2048], "data_offsets": [0, 1024]}}
            blob = json.dumps(header).encode()
            blob += b" " * ((8 - len(blob) % 8) % 8)
            with open(shard, "wb") as f:
                f.write(struct.pack("<Q", len(blob)))
                f.write(blob)
                f.write(b"\x11" * 1024)

            # Validating on the first ``next()`` would defer this past alloc.
            with self.assertRaises(ValueError) as caught:
                instanttensor_weights_iterator([shard])
            self.assertIn("F4", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
