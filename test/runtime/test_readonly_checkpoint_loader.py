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

"""Checkpoint loading must not require or use write access to the source."""

import json
import os
import struct
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.model_loader.loader import DefaultModelLoader


class _HostWeights(torch.nn.Module):
    # Older model implementations may still expose the retired shared-mapping
    # hook. Loading them must retain normal checkpoint read semantics.
    def checkpoint_shared_weight_name_filter(self, name: str) -> bool:
        return name.startswith("host.")


class _FilteredHostWeights(_HostWeights):
    def checkpoint_weight_name_filter(self, name: str) -> bool:
        return name != "skip"


@pytest.mark.parametrize("model_type", [_HostWeights, _FilteredHostWeights])
@pytest.mark.parametrize("load_format", ["auto", "safetensors"])
@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("read_only", [False, True])
def test_checkpoint_loading_preserves_source_bytes(
    tmp_path, model_type, load_format, prefetch, read_only
):
    expected = {
        "host.table": torch.arange(128, dtype=torch.bfloat16).reshape(8, 16),
        "host.scalar": torch.tensor(3, dtype=torch.bfloat16),
        "host.empty": torch.empty(0, 16, dtype=torch.bfloat16),
        "projection": torch.arange(15, dtype=torch.float32).reshape(3, 5),
        "skip": torch.tensor([1], dtype=torch.uint8),
    }
    path = tmp_path / "model.safetensors"
    save_file(expected, str(path))
    original = path.read_bytes()
    if read_only:
        path.chmod(0o444)
    try:
        if read_only:
            try:
                with path.open("r+b"):
                    pass
            except PermissionError:
                pass
            else:
                pytest.skip("This process can bypass read-only file permissions")
        loader = DefaultModelLoader(
            LoadConfig(
                load_format=load_format,
                weight_loader_prefetch_checkpoints=prefetch,
            )
        )
        model = model_type()
        loaded = dict(
            loader.get_checkpoint_weights(
                str(tmp_path), None, model, Mapping(rank=0, world_size=1)
            )
        )
        assert loader.checkpoint_shapes == {
            name: tuple(tensor.shape) for name, tensor in expected.items()
        }
        expected_names = set(expected)
        if model_type is _FilteredHostWeights:
            expected_names.remove("skip")
        assert loaded.keys() == expected_names
        for name, tensor in loaded.items():
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)

        # Updating a loaded tensor must never write through to the checkpoint,
        # even when its file permissions would allow writes.
        loaded["host.table"].add_(1)
        torch.testing.assert_close(
            loaded["host.table"], expected["host.table"] + 1, rtol=0, atol=0
        )
        assert path.read_bytes() == original
        for name, tensor in load_file(str(path)).items():
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
    finally:
        path.chmod(0o600)


@pytest.mark.skipif(
    not hasattr(torch, "float4_e2m1fn_x2"), reason="requires packed FP4 tensor support"
)
def test_checkpoint_metadata_keeps_logical_subbyte_shapes(tmp_path):
    header = {"packed": {"dtype": "F4", "shape": [2, 2048], "data_offsets": [0, 2048]}}
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(blob)) + blob + b"\x11" * 2048
    )
    loader = DefaultModelLoader(LoadConfig(weight_loader_prefetch_checkpoints=False))
    # Packed tensors have a different materialized shape; reloads compare headers.
    list(
        loader.get_checkpoint_weights(
            str(tmp_path), None, _HostWeights(), Mapping(rank=0)
        )
    )
    assert loader.checkpoint_shapes == {"packed": (2, 2048)}


def test_checkpoint_metadata_ignores_unselected_shards(tmp_path):
    expected = {"host.table": torch.arange(6).reshape(2, 3)}
    save_file(expected, str(tmp_path / "kept.safetensors"))
    (tmp_path / "ignored.safetensors").write_bytes(b"not a readable shard")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "host.table": "kept.safetensors",
                    "skip": "ignored.safetensors",
                }
            }
        )
    )
    loader = DefaultModelLoader(LoadConfig(weight_loader_prefetch_checkpoints=False))
    loaded = dict(
        loader.get_checkpoint_weights(
            str(tmp_path), None, _FilteredHostWeights(), Mapping(rank=0)
        )
    )
    assert loaded.keys() == expected.keys()
    torch.testing.assert_close(loaded["host.table"], expected["host.table"])
    assert loader.checkpoint_shapes == {"host.table": (2, 3)}
