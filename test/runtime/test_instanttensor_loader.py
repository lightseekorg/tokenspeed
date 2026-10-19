import argparse
import glob
import json
import os
import struct
import sys
import tempfile
from importlib.util import find_spec

import pytest
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


class TestInstantTensorConfig:
    """Config/CLI wiring that needs neither a GPU nor instanttensor."""

    def test_cli_flag_maps_to_load_format(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(
            ["--model", "test/model", "--load-format", "instanttensor"]
        )
        assert (args.load_format) == ("instanttensor")

    def test_load_config_normalizes_to_enum(self):
        load_config = LoadConfig(load_format="instanttensor")
        assert (load_config.load_format) == (LoadFormat.INSTANTTENSOR)

    def test_prepare_weights_treats_instanttensor_as_safetensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # _prepare_weights only globs paths, never tensor data.
            open(os.path.join(tmpdir, "model.safetensors"), "wb").close()

            loader = DefaultModelLoader(LoadConfig(load_format="instanttensor"))
            _, hf_weights_files, use_safetensors = loader._prepare_weights(
                tmpdir, revision=None, fall_back_to_pt=False
            )

            assert use_safetensors
            assert (len(hf_weights_files)) == (1)
            assert hf_weights_files[0].endswith("model.safetensors")


@pytest.mark.skipif((not IS_NVIDIA), reason=("InstantTensor requires NVIDIA GPUs"))
@pytest.mark.skipif(
    (not INSTANTTENSOR_AVAILABLE), reason=("instanttensor is not installed")
)
class TestInstantTensorWeights:
    """Iterator parity test (requires an NVIDIA GPU and instanttensor)."""

    def test_instanttensor_matches_safetensors(self):
        model = "openai-community/gpt2"
        with tempfile.TemporaryDirectory() as tmpdir:
            download_weights_from_hf(
                model, cache_dir=tmpdir, allow_patterns=["*.safetensors"]
            )
            safetensors_files = glob.glob(f"{tmpdir}/**/*.safetensors", recursive=True)
            assert (len(safetensors_files)) > (0)

            instanttensor_tensors = {}
            for name, tensor in instanttensor_weights_iterator(
                safetensors_files, process_group=None
            ):
                # Copy immediately in case InstantTensor exposes internal buffers.
                instanttensor_tensors[name] = tensor.to("cpu")

            reference_tensors = dict(
                safetensors_weights_iterator(
                    safetensors_files, local_rank=0, local_world_size=1
                )
            )

            assert (len(instanttensor_tensors)) == (len(reference_tensors))
            for name, got in instanttensor_tensors.items():
                ref = reference_tensors[name]
                assert (got.dtype) == (ref.dtype)
                assert (got.shape) == (ref.shape)
                assert torch.equal(got, ref)


class TestSubByteDtypeGuard:
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
            assert (_find_sub_byte_dtype([shard])) == ("F4")

    def test_byte_aligned_shard_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            shards = [
                self._write(os.path.join(d, "a.safetensors"), "BF16", [512], 1024),
                self._write(os.path.join(d, "b.safetensors"), "F8_E4M3", [1024], 1024),
                self._write(os.path.join(d, "c.safetensors"), "U8", [1024], 1024),
            ]
            assert (_find_sub_byte_dtype(shards)) is None

    def test_unparsable_header_raises_rather_than_waving_the_shard_through(
        self,
        subtests,
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
                with subtests.test(filename):
                    with pytest.raises(((ValueError, struct.error))):
                        _find_sub_byte_dtype([path])


@pytest.mark.skipif((not IS_NVIDIA), reason=("InstantTensor requires NVIDIA GPUs"))
@pytest.mark.skipif(
    (not INSTANTTENSOR_AVAILABLE), reason=("instanttensor is not installed")
)
class TestInstantTensorRejectsSubByteCheckpoints:
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
            with pytest.raises((ValueError)) as caught:
                instanttensor_weights_iterator([shard], process_group=None)
            assert ("F4") in (str(caught.value))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
