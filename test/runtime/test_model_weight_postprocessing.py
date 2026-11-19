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


"""Root-model operands must see processed and populated child weights."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.model_loader import loader


class PackedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 3), requires_grad=False)

    def process_weights_after_loading(self, module):
        self.weight.data = self.weight.data.T.contiguous()


class PreparedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = PackedLinear()
        self.operand = None
        self.warmup_output = None

    def load_weights(self, weights):
        for _, value in weights:
            self.linear.weight.data.copy_(value)

    def process_weights_after_loading(self, module):
        # Derive a separate operand, so a stale layout/value cannot pass.
        self.operand = self.linear.weight.detach().clone() * 2

    def post_quant_warmup(self):
        self.warmup_output = self(torch.ones(1, 3))

    def forward(self, value):
        return value @ self.operand


@pytest.mark.parametrize("kind", ["auto", "dummy", "sharded_state"])
def test_root_derives_operands_after_child_layout_and_copy(tmp_path, monkeypatch, kind):
    raw = torch.arange(6).reshape(2, 3).float()
    if kind == "sharded_state":
        save_file(
            {"linear.weight": raw.T.contiguous()},
            str(tmp_path / "model-rank-0-part-0.safetensors"),
        )
    monkeypatch.setattr(loader, "_initialize_model", lambda *args: PreparedModel())
    monkeypatch.setattr(
        loader.DefaultModelLoader,
        "_get_all_weights",
        lambda *args: iter([("linear.weight", raw)]),
    )
    if kind == "dummy":
        # Dummy randomization happens in resident layouts. Use known values,
        # retaining the loader's real order and the real postprocessing hooks.
        monkeypatch.setattr(
            loader,
            "initialize_dummy_weights",
            lambda model: model.linear.weight.data.copy_(raw.T),
        )
    model_loader = loader.get_model_loader(LoadConfig(load_format=kind))
    model = model_loader.load_model(
        model_config=SimpleNamespace(
            model_path=str(tmp_path),
            revision=None,
            dtype=torch.float32,
            mapping=SimpleNamespace(rank=0),
        ),
        device_config=SimpleNamespace(device="cpu"),
    )
    expected = torch.ones(1, 3) @ (raw.T * 2)
    torch.testing.assert_close(model(torch.ones(1, 3)), expected, rtol=0, atol=0)
    torch.testing.assert_close(model.warmup_output, expected, rtol=0, atol=0)
    # In-place updates are refused for transformed modules, including root
    # derived operands; they must not be mistaken for graph-safe live copies.
    assert any(
        "PreparedModel" in name for name in loader.post_load_transformed_modules(model)
    )


def test_plain_model_postprocessing_keeps_parameter_values():
    model = nn.Linear(3, 2)
    before = model.weight.detach().clone()
    loader.process_weights_after_loading(model, torch.device("cpu"))
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
