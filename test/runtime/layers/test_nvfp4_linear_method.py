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

"""NVFP4 linear-method weight processing for the fused GEMM+SwiGLU path."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def _has_sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


@pytest.mark.skipif(not _has_sm100(), reason="Blackwell SM100 CUDA GPU required")
def test_nvfp4_process_weights_releases_normal_fc1_tensors() -> None:
    from torch import nn
    from torch.nn.parameter import Parameter

    from tokenspeed.runtime.layers.dense.nvfp4 import Nvfp4LinearMethod

    class QuantConfig:
        group_size = 16

    layer = nn.Module()
    layer.prefix = "test.gate_up_proj"
    layer.interleave_linear_and_gate = True
    layer.weight = Parameter(
        torch.randint(0, 256, (256, 8), device="cuda", dtype=torch.uint8),
        requires_grad=False,
    )
    layer.weight_scale = Parameter(
        torch.empty((256, 4), device="cuda", dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.input_scale = Parameter(
        torch.tensor([2.0], device="cuda", dtype=torch.float32),
        requires_grad=False,
    )
    layer.weight_scale_2 = Parameter(
        torch.tensor([3.0], device="cuda", dtype=torch.float32),
        requires_grad=False,
    )

    Nvfp4LinearMethod(QuantConfig()).process_weights_after_loading(layer)

    assert not hasattr(layer, "weight")
    assert not hasattr(layer, "weight_scale")
    assert not hasattr(layer, "weight_scale_interleaved")
    assert hasattr(layer, "weight_swiglu_interleaved")
    assert hasattr(layer, "weight_scale_swiglu_interleaved")


@pytest.mark.skipif(not _has_sm100(), reason="Blackwell SM100 CUDA GPU required")
def test_nvfp4_process_weights_releases_normal_weight_scale() -> None:
    from torch import nn
    from torch.nn.parameter import Parameter

    from tokenspeed.runtime.layers.dense.nvfp4 import Nvfp4LinearMethod

    class QuantConfig:
        group_size = 16

    layer = nn.Module()
    layer.prefix = "test.down_proj"
    layer.interleave_linear_and_gate = False
    layer.weight = Parameter(
        torch.randint(0, 256, (128, 8), device="cuda", dtype=torch.uint8),
        requires_grad=False,
    )
    layer.weight_scale = Parameter(
        torch.empty((128, 4), device="cuda", dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.input_scale = Parameter(
        torch.tensor([2.0], device="cuda", dtype=torch.float32),
        requires_grad=False,
    )
    layer.weight_scale_2 = Parameter(
        torch.tensor([3.0], device="cuda", dtype=torch.float32),
        requires_grad=False,
    )

    Nvfp4LinearMethod(QuantConfig()).process_weights_after_loading(layer)

    assert hasattr(layer, "weight")
    assert not hasattr(layer, "weight_scale")
    assert hasattr(layer, "weight_scale_interleaved")
    assert not hasattr(layer, "weight_swiglu_interleaved")
    assert not hasattr(layer, "weight_scale_swiglu_interleaved")
