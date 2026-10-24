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

"""CPU validation of preconverted W4A4 shard selection and lossless loading."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel_npu.ops.dsv41_w4a4_weights import (
    load_projection,
    read_projection,
)


@pytest.fixture
def artifact(tmp_path):
    data = dict(
        version=1,
        key="weights",
        input_size=8,
        output_size=3,
        weight=torch.full((2, 3, 1), 0x12345678, dtype=torch.int32),
        scales=torch.tensor([[0.125, 0.25, 0.5], [1.0, 2.0, 4.0]]),
        left=torch.eye(2, dtype=torch.bfloat16),
        right=torch.eye(4, dtype=torch.bfloat16),
        clip_ratio=0.9,
    )
    torch.save(data, tmp_path / "weights.pt")
    config = {
        "format": "tokenspeed-v41-w4a4",
        "version": 1,
        "ep_size": 2,
        "layers": {"7": {"1": {"w13": "weights.pt"}}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(config))
    args = dict(layer_index=7, ep_rank=1, ep_size=2, projection="w13", shape=(2, 3, 8))
    return path, data, config, args


def test_load_preserves_codes_scale_bits_and_transforms(artifact, monkeypatch):
    path, data, _, args = artifact
    casts = []

    def format_cast(weight, fmt, *, customize_dtype):
        casts.append((fmt, customize_dtype))
        return weight

    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu_format_cast=format_cast)
    )
    monkeypatch.setitem(
        sys.modules,
        "tokenspeed_kernel_npu.ops.dsv41_moe",
        SimpleNamespace(
            AscendGroupedLinear=lambda *values: values,
            FlatQuantParameters=lambda *values: values,
        ),
    )
    result = load_projection(path, device="cpu", **args)
    assert result[0] == "w4a4" and result[4:] == (8, 3)
    assert casts == [(29, 3)]
    assert torch.equal(result[1], data["weight"])
    restored = (
        result[2].view(torch.int64).to(torch.int32).view(torch.float32).squeeze(1)
    )
    assert torch.equal(restored, data["scales"])
    assert torch.equal(result[3][0], data["left"])
    assert torch.equal(result[3][1], data["right"])
    assert result[3][2] == data["clip_ratio"]


@pytest.mark.parametrize("change", ["ep_size", "ep_rank", "layer_index", "shape"])
def test_rejects_wrong_local_shard(artifact, change):
    path, _, _, args = artifact
    args[change] = {"ep_size": 4, "ep_rank": 0, "layer_index": 8, "shape": (1, 3, 8)}[
        change
    ]
    with pytest.raises(ValueError):
        read_projection(path, **args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("key", "wrong"),
        ("scales", torch.zeros(2, 3)),
        ("scales", torch.full((2, 3), float("nan"))),
        ("left", torch.eye(3, dtype=torch.bfloat16)),
        ("clip_ratio", float("nan")),
        ("weight", torch.zeros(2, 3, 1, dtype=torch.int64)),
    ],
)
def test_rejects_invalid_payload(artifact, field, value):
    path, data, _, args = artifact
    data[field] = value
    torch.save(data, path.parent / "weights.pt")
    with pytest.raises(ValueError):
        read_projection(path, **args)


def test_missing_artifact_does_not_convert_or_write(artifact):
    path, _, _, args = artifact
    (path.parent / "weights.pt").unlink()
    with pytest.raises(FileNotFoundError):
        read_projection(path, **args)
    assert list(path.parent.iterdir()) == [path]


def test_rejects_legacy_random_initialization_config(artifact):
    path, _, _, args = artifact
    path.write_text(
        json.dumps({"initialization": "random-orthogonal-uncalibrated", "seed": 1})
    )
    with pytest.raises(ValueError, match="online conversion is unsupported"):
        read_projection(path, **args)
