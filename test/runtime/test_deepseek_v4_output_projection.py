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

from __future__ import annotations

from types import SimpleNamespace

import tokenspeed_kernel
import torch

import tokenspeed.runtime.models.deepseek_v4 as deepseek_v4
from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config


def test_runtime_output_projection_delegates_to_public_api(monkeypatch) -> None:
    plan = object()
    weight = torch.empty((4, 2))
    weight_scale = torch.empty((1, 1))
    grouped_output = torch.randn((3, 2, 5))
    calls = []

    def project(*args):
        calls.append(args)
        return grouped_output

    projected = grouped_output.flatten(1) + 1
    attention_module = SimpleNamespace(
        _wo_a_output_projection_plan=plan,
        wo_a=SimpleNamespace(weight=weight, weight_scale_inv=weight_scale),
        wo_b=lambda value: (value + 1, None),
    )
    monkeypatch.setattr(deepseek_v4, "deepseek_v4_grouped_output_projection", project)

    actual = deepseek_v4.DeepseekV4Attention._project_attention_output(
        attention_module,
        torch.empty((3, 2, 2)),
        torch.arange(3),
        torch.empty((3, 2)),
    )

    assert actual is not grouped_output
    torch.testing.assert_close(actual, projected)
    assert calls[0][0] is plan
    assert calls[0][-2] is weight
    assert calls[0][-1] is weight_scale


def test_fp8_loader_delegates_grouped_scale_preprocessing(monkeypatch) -> None:
    method = Fp8LinearMethod(
        Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[128, 128],
            scale_fmt="ue8m0",
        )
    )
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.empty((256, 128), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_scale_inv",
        torch.nn.Parameter(torch.ones((2, 1)), requires_grad=False),
    )
    plan = object()
    layer._deepseek_v4_grouped_output_projection_plan = plan
    prepared = torch.empty((7,), dtype=torch.int32)
    calls = []

    def process(plan_arg, weight_arg, scale_arg):
        calls.append((plan_arg, weight_arg, scale_arg))
        return prepared

    monkeypatch.setattr(
        tokenspeed_kernel,
        "deepseek_v4_grouped_output_projection_process_weights",
        process,
    )

    method.process_weights_after_loading(layer)

    assert layer.weight_scale_inv.dtype == torch.int32
    assert layer.weight_scale_inv.data_ptr() == prepared.data_ptr()
    assert calls[0][0] is plan
    assert calls[0][1].data_ptr() == layer.weight.data_ptr()
