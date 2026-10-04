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

"""``routed_experts_weights_of_layer`` has one shape across the MoE models.

The base class declares it as a setter-less property; a model that assigned
the attribute instead (as GPT-OSS once did at the end of ``load_weights``)
raised ``AttributeError`` on every launch. The static scan below guards the
whole models tree, and the GPT-OSS case exercises the accessor body.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch

from tokenspeed.runtime.models import gpt_oss
from tokenspeed.runtime.models.base.causal_lm import BaseCausalLM

MODELS_DIR = Path(gpt_oss.__file__).resolve().parent
ACCESSOR = "routed_experts_weights_of_layer"


def _stores_to_self_accessor(tree: ast.AST) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Store)
        and node.attr == ACCESSOR
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ]


def _property_classes(tree: ast.AST) -> dict[str, bool]:
    """Class name -> whether it defines the accessor as a ``@property``."""
    found: dict[str, bool] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == ACCESSOR:
                found[node.name] = any(
                    isinstance(d, ast.Name) and d.id == "property"
                    for d in item.decorator_list
                )
    return found


def test_no_model_assigns_the_accessor_and_every_definition_is_a_property():
    offenders = {}
    non_properties = {}
    for path in sorted(MODELS_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text())
        stores = _stores_to_self_accessor(tree)
        if stores:
            offenders[path.name] = stores
        for cls, is_property in _property_classes(tree).items():
            if not is_property:
                non_properties[f"{path.name}:{cls}"] = True
    assert offenders == {}, f"assignments to self.{ACCESSOR}: {offenders}"
    assert non_properties == {}
    assert isinstance(inspect.getattr_static(BaseCausalLM, ACCESSOR), property)


def test_gpt_oss_collects_every_layers_routed_weights_through_the_property():
    assert isinstance(
        inspect.getattr_static(gpt_oss.GptOssForCausalLM, ACCESSOR), property
    )
    slot_tensors = {
        layer_id: [torch.zeros(2, 3) + layer_id, torch.ones(2) * layer_id]
        for layer_id in range(3)
    }
    model = gpt_oss.GptOssForCausalLM.__new__(gpt_oss.GptOssForCausalLM)
    model.model = SimpleNamespace(
        layers=[
            SimpleNamespace(
                mlp=SimpleNamespace(
                    get_moe_routed_weights=lambda tensors=tensors: tensors
                )
            )
            for tensors in slot_tensors.values()
        ]
    )
    weights = model.routed_experts_weights_of_layer
    assert weights.keys() == slot_tensors.keys()
    for layer_id, tensors in weights.items():
        assert all(t is s for t, s in zip(tensors, slot_tensors[layer_id]))
    # The MoE block exposes the same accessor name as the other MoE models.
    block = gpt_oss.GptOssSparseMoeBlock.__new__(gpt_oss.GptOssSparseMoeBlock)
    torch.nn.Module.__init__(block)
    block.experts = torch.nn.Linear(2, 2)
    assert [t.shape for t in block.get_moe_routed_weights()] == [
        torch.Size([2, 2]),
        torch.Size([2]),
    ]
