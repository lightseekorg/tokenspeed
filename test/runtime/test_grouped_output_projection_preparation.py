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

"""The FP8 layer owns grouped weight transformation and canonical enrollment."""

import pytest
import torch
from tokenspeed_kernel.weights import UNKNOWN_LAYOUT, get_weight_broker

from tokenspeed.runtime.layers.dense import fp8
from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config


@pytest.mark.parametrize("transformed", [False, True])
@pytest.mark.parametrize("serialized", [False, True])
def test_layer_installs_selected_preprocessing(monkeypatch, transformed, serialized):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros(128, 128).to(torch.float8_e4m3fn), requires_grad=False
    )
    layer.weight_scale_inv = torch.nn.Parameter(torch.ones(1, 1), requires_grad=False)
    layer._dsv4_grouped_output_projection_plan = plan = object()
    broker = get_weight_broker()
    observed = {}

    def quantize(weight, *, granularity, block_size):
        assert granularity == "block"
        assert block_size == [128, 128]
        return weight.clone(), torch.ones(1, 1)

    monkeypatch.setattr(fp8, "quantize_fp8", quantize)

    def transform(*, weight, weight_scale_inv):
        return {"weight": weight, "weight_scale_inv": weight_scale_inv + 1}

    def select_preprocessor(requested_plan):
        assert requested_plan is plan
        assert broker.tracked_layout(layer.weight) is UNKNOWN_LAYOUT
        observed["parameter"] = layer.weight_scale_inv
        observed["storage"] = layer.weight_scale_inv.untyped_storage()
        return (("weight", "weight_scale_inv"), transform) if transformed else None

    monkeypatch.setattr(
        fp8,
        "kernel_dsv4_grouped_output_projection_preprocessor",
        select_preprocessor,
    )
    method = Fp8LinearMethod(
        Fp8Config(
            is_checkpoint_fp8_serialized=serialized,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
    )
    method.process_weights_after_loading(layer)

    assert layer.weight_scale_inv is observed["parameter"]
    assert (
        layer.weight_scale_inv.untyped_storage() is not observed["storage"]
    ) is transformed
    assert layer.weight_scale_inv.item() == (2 if transformed else 1)
    expected_layout = transform if transformed else None
    assert broker.layout(layer.weight) is expected_layout
    assert broker.layout(layer.weight_scale_inv.view(-1)) is expected_layout
