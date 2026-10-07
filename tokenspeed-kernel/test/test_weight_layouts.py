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

import gc
import weakref
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.weights import UNKNOWN_LAYOUT, WeightBroker


def _transpose(*, weight):
    return {"weight": weight.T.contiguous()}


def test_views_share_layout_but_copies_need_enrollment():
    broker = WeightBroker()
    weight = torch.arange(12.0).reshape(3, 4)
    broker.enroll(weight, _transpose)
    views = [weight.T, weight[1:, 1:], weight.view(torch.uint8), weight.detach()]
    copy = weight.clone()
    storage = weakref.ref(weight.untyped_storage())
    del weight
    gc.collect()

    assert all(broker.layout(view) is _transpose for view in views)
    assert broker.tracked_layout(copy) is UNKNOWN_LAYOUT
    with pytest.raises(ValueError, match="untracked weight storage"):
        broker.layout(copy)
    assert broker.layout(copy, allow_unknown=True) is None
    broker.enroll(copy)
    assert broker.layout(copy) is None
    del views
    gc.collect()
    assert storage() is None


def test_preprocess_preserves_parameter_metadata_and_releases_old_storage():
    broker = WeightBroker()
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(torch.arange(12.0).reshape(3, 4))
    layer.weight.weight_loader = "original loader"
    layer.weight.quant_method = "preserved quantization metadata"
    parameter = layer.weight
    old_storage = weakref.ref(parameter.untyped_storage())

    def transpose(*, weight):
        result = weight.T.contiguous()
        result.weight_loader = f"wrapped {weight.weight_loader}"
        return {"weight": result}

    bindings = broker.preprocess((("weight",), transpose), layer)
    gc.collect()

    assert bindings == {"weight": (layer, "weight")}
    assert layer.weight is parameter
    assert layer.weight.weight_loader == "wrapped original loader"
    assert layer.weight.quant_method == "preserved quantization metadata"
    assert old_storage() is None
    torch.testing.assert_close(layer.weight, torch.arange(12.0).reshape(3, 4).T)
    assert broker.layout(layer.weight) is transpose


def test_preprocess_can_create_destinations_and_clear_superseded_sources():
    broker = WeightBroker()
    layer = SimpleNamespace(weight=torch.arange(8.0), scale=torch.ones(1), bias=None)
    original = weakref.ref(layer.weight)
    old_storage = weakref.ref(layer.weight.untyped_storage())

    def split(*, weight, scale, bias):
        backend = torch.nn.Module()
        transformed = weight.flip(0)
        return (
            {
                "left": transformed[:4],
                "right": transformed[4:],
                "scale": scale,
                "bias": bias,
            },
            {"left": (backend, "left"), "right": (backend, "right"), "scale": None},
        )

    bindings = broker.preprocess((("weight", "scale", "bias"), split), layer)
    backend = bindings["left"][0]
    gc.collect()

    assert layer.weight is None
    assert original() is None
    assert old_storage() is None
    assert dict(backend.named_parameters()).keys() == {"left", "right"}
    assert backend.left.untyped_storage() is backend.right.untyped_storage()
    torch.testing.assert_close(backend.left, torch.tensor([7.0, 6.0, 5.0, 4.0]))
    assert broker.layout(backend.left) is split
    assert broker.layout(backend.right) is split
    assert broker.layout(layer.scale) is split
    assert layer.bias is None
    assert "bias" not in bindings


@pytest.mark.parametrize("reverse", [False, True])
def test_preprocess_records_explicit_variant_and_in_place_aliases(reverse):
    broker = WeightBroker()
    layer = SimpleNamespace(weight=torch.arange(4.0))
    alias = layer.weight.view(2, 2)
    broker.enroll(alias)

    def reverse_layout(weight):
        weight.copy_(weight.flip(0))

    def prepare(*, weight, config):
        if config:
            reverse_layout(weight)
        return {"weight": weight}, None, reverse_layout if config else None

    broker.preprocess((("weight",), prepare), layer, config=reverse)

    assert broker.layout(alias) is (reverse_layout if reverse else None)
    torch.testing.assert_close(
        alias.flatten(), torch.arange(4.0).flip(0) if reverse else torch.arange(4.0)
    )


def test_partial_configuration_does_not_change_layout_identity():
    broker = WeightBroker()
    layer = SimpleNamespace(weight=torch.ones(4))

    def prepare(*, weight, factor):
        return {"weight": weight * factor}

    broker.preprocess((("weight",), partial(prepare, factor=2)), layer)

    assert broker.layout(layer.weight) is prepare
    torch.testing.assert_close(layer.weight, torch.full((4,), 2.0))
