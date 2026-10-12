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

"""Dummy loading gives integer parameters valid values before post-processing
reads them, and tells an EAGLE3 draft its embedding is initialized."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
from torch import nn

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs.load_config import LoadConfig  # noqa: E402
from tokenspeed.runtime.model_loader import loader  # noqa: E402
from tokenspeed.runtime.model_loader.weight_utils import (  # noqa: E402
    initialize_dummy_integer_weights,
)


class _QuantizedLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 0xFF stands in for uninitialized memory; as an E8M0 scale it is NaN.
        self.weight = nn.Parameter(
            torch.full((4, 4), 0xFF, dtype=torch.uint8), requires_grad=False
        )
        self.bias = nn.Parameter(torch.ones(4))
        self.register_buffer("order", torch.arange(4))


def test_integer_parameters_are_zeroed_and_everything_else_kept():
    layer = _QuantizedLinear()
    initialize_dummy_integer_weights(layer)
    assert torch.all(layer.weight == 0)
    assert torch.all(layer.bias == 1)
    assert torch.equal(layer.order, torch.arange(4))


def test_the_dummy_loader_fills_integers_before_post_processing(monkeypatch):
    seen = []

    class _Model(_QuantizedLinear):
        def post_load_weights(self) -> None:
            seen.append(self.weight.clone())

    monkeypatch.setattr(loader, "_initialize_model", lambda *args: _Model())
    loader.DummyModelLoader(LoadConfig()).load_model(
        model_config=SimpleNamespace(dtype=torch.float32),
        device_config=SimpleNamespace(device="cpu"),
    )
    assert len(seen) == 1 and torch.all(seen[0] == 0)


def test_the_dummy_loader_marks_the_draft_embedding_initialized(monkeypatch):
    marked = []

    class _Draft(_QuantizedLinear):
        def mark_embedding_initialized(self) -> None:
            marked.append(True)

    monkeypatch.setattr(loader, "_initialize_model", lambda *args: _Draft())
    loader.DummyModelLoader(LoadConfig()).load_model(
        model_config=SimpleNamespace(dtype=torch.float32),
        device_config=SimpleNamespace(device="cpu"),
    )
    assert marked == [True]
