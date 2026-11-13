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

"""Selected host weights retain writable file mappings without changing bytes."""

import gc
import os
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.model_loader.loader import DefaultModelLoader


class _HostWeights(torch.nn.Module):
    def checkpoint_shared_weight_name_filter(self, name: str) -> bool:
        return name.startswith("host.")


class _FilteredHostWeights(_HostWeights):
    def checkpoint_weight_name_filter(self, name: str) -> bool:
        return name != "skip"


class _SelectedFileHostWeights(_HostWeights):
    checkpoint_weight_files = ("selected.safetensors",)


def _mapping_permissions(pointer: int) -> str | None:
    for line in Path("/proc/self/maps").read_text().splitlines():
        address, permissions, *_ = line.split()
        begin, end = (int(part, 16) for part in address.split("-"))
        if begin <= pointer < end:
            return permissions
    return None


def _load(directory, model, *, load_format, prefetch):
    loader = DefaultModelLoader(
        LoadConfig(
            load_format=load_format,
            weight_loader_prefetch_checkpoints=prefetch,
        )
    )
    return loader.get_checkpoint_weights(
        str(directory), None, model, Mapping(rank=0, world_size=1)
    )


@pytest.mark.parametrize("model_type", [_HostWeights, _FilteredHostWeights])
@pytest.mark.parametrize("load_format", ["auto", "safetensors"])
@pytest.mark.parametrize("prefetch", [False, True])
def test_selected_weights_keep_shared_mapping_and_exact_bytes(
    tmp_path, model_type, load_format, prefetch
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
    loaded = dict(
        _load(tmp_path, model_type(), load_format=load_format, prefetch=prefetch)
    )
    if model_type is _FilteredHostWeights:
        expected.pop("skip")
    assert loaded.keys() == expected.keys()
    for name, tensor in loaded.items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
        if tensor.numel():
            permissions = _mapping_permissions(tensor.data_ptr())
            assert permissions is not None
            assert (permissions == "rw-s") == name.startswith("host.")

    # A strided view must keep the file alive after the iterator and other
    # tensors are gone, just as a registered feature-column fragment does.
    columns = loaded["host.table"][:, 8:]
    pointer = columns.data_ptr()
    del tensor, loaded
    gc.collect()
    assert _mapping_permissions(pointer) == "rw-s"
    torch.testing.assert_close(columns, expected["host.table"][:, 8:], rtol=0, atol=0)
    del columns
    gc.collect()
    assert _mapping_permissions(pointer) is None
    assert path.read_bytes() == original


def test_early_iterator_close_preserves_retained_view(tmp_path):
    expected = torch.arange(128, dtype=torch.bfloat16).reshape(8, 16)
    path = tmp_path / "model.safetensors"
    save_file({"host.table": expected, "projection": torch.ones(8)}, str(path))
    iterator = _load(
        tmp_path, _FilteredHostWeights(), load_format="auto", prefetch=False
    )
    name, table = next(iterator)
    assert name == "host.table"
    iterator.close()
    gc.collect()
    assert _mapping_permissions(table.data_ptr()) == "rw-s"
    torch.testing.assert_close(table, expected, rtol=0, atol=0)


def test_shared_mapping_requires_write_permission_but_default_loading_does_not(
    tmp_path,
):
    expected = torch.ones(8, 16, dtype=torch.bfloat16)
    path = tmp_path / "model.safetensors"
    save_file({"host.table": expected}, str(path))
    path.chmod(0o444)
    try:
        with pytest.raises(PermissionError):
            dict(_load(tmp_path, _HostWeights(), load_format="auto", prefetch=False))
        loaded = dict(
            _load(tmp_path, torch.nn.Module(), load_format="auto", prefetch=False)
        )
        torch.testing.assert_close(loaded["host.table"], expected, rtol=0, atol=0)
    finally:
        path.chmod(0o644)


def test_shared_weight_filter_sees_source_prefix(tmp_path):
    expected = torch.ones(8, 16, dtype=torch.bfloat16)
    save_file(
        {"table": expected, "projection": torch.ones(8)},
        str(tmp_path / "model.safetensors"),
    )
    loader = DefaultModelLoader(LoadConfig(load_format="auto"))
    source = DefaultModelLoader.Source(
        str(tmp_path), None, weight_files=None, prefix="draft."
    )
    loaded = dict(
        loader._get_weights_iterator(
            source,
            None,
            None,
            Mapping(rank=0, world_size=1),
            shared_weight_name_filter=lambda name: name == "draft.table",
        )
    )
    assert _mapping_permissions(loaded["draft.table"].data_ptr()) == "rw-s"
    assert _mapping_permissions(loaded["draft.projection"].data_ptr()) != "rw-s"
    torch.testing.assert_close(loaded["draft.table"], expected, rtol=0, atol=0)


def test_shared_mapping_respects_explicit_checkpoint_files(tmp_path):
    expected = torch.ones(8, 16, dtype=torch.bfloat16)
    save_file({"host.table": expected}, str(tmp_path / "selected.safetensors"))
    save_file({"wrong_file": torch.zeros(8)}, str(tmp_path / "model.safetensors"))
    loaded = dict(
        _load(
            tmp_path,
            _SelectedFileHostWeights(),
            load_format="auto",
            prefetch=False,
        )
    )
    assert loaded.keys() == {"host.table"}
    assert _mapping_permissions(loaded["host.table"].data_ptr()) == "rw-s"
    torch.testing.assert_close(loaded["host.table"], expected, rtol=0, atol=0)
