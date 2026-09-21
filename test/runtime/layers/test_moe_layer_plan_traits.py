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

"""MoELayer states its layer geometry and routing facts to ``moe_plan``.

A kernel that cannot serve a layer must drop out at plan time, not fail in
weight preprocessing or return wrong sums at runtime, so the layer declares
the MoE input width, the SwiGLU form and whether expert ids may repeat.
"""

from __future__ import annotations

import pytest

from tokenspeed.runtime.layers.moe import expert as expert_module
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config
from tokenspeed.runtime.utils.env import global_server_args_dict


def _plan_kwargs(monkeypatch, **layer_kwargs) -> dict:
    plans = []

    def fake_plan(weight_dtype, **kwargs):
        plans.append(kwargs)
        return {"solution": "fake", "apply_kernel_name": "fake"}

    monkeypatch.setattr(expert_module.tokenspeed_kernel, "moe_plan", fake_plan)
    monkeypatch.setattr(expert_module, "create_layer_weights", lambda *a, **k: None)
    monkeypatch.setitem(global_server_args_dict, "moe_mxfp4_fp8_activation", False)
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)
    kwargs = dict(
        top_k=2,
        num_experts=8,
        hidden_size=2880,
        intermediate_size=128,
        quant_config=Mxfp4Config(
            ignored_layers=[], is_checkpoint_mxfp4_serialized=True
        ),
        layer_index=0,
        prefix="model.layers.0.mlp",
    )
    kwargs.update(layer_kwargs)
    MoELayer(**kwargs)
    assert len(plans) == 1
    return plans[0]


def test_hidden_width_is_a_plan_trait(monkeypatch):
    assert _plan_kwargs(monkeypatch, activation="swiglu")["hidden"] == 2880


@pytest.mark.parametrize(
    "layer, form",
    [
        (dict(activation="swiglu", swiglu_limit=10.0), "standard"),
        (dict(activation="swiglu", activation_alpha=1.0, swiglu_beta=0.0), "standard"),
        # gpt-oss / MiniMax-M3: silu(alpha * gate) * (up + 1).
        (
            dict(activation="swiglu", activation_alpha=1.702, swiglu_beta=1.0),
            "generalized",
        ),
        (dict(activation="swiglu", swiglu_beta=1.0), "generalized"),
        (dict(activation="silu"), None),
    ],
)
def test_swiglu_form_follows_alpha_and_beta(monkeypatch, layer, form):
    assert _plan_kwargs(monkeypatch, **layer)["swiglu_form"] == form


def test_zero_experts_declare_repeated_expert_ids(monkeypatch):
    plain = _plan_kwargs(monkeypatch, activation="swiglu")
    assert plain["expert_id_repeats"] is False
    longcat = _plan_kwargs(
        monkeypatch, activation="swiglu", zero_expert_type="copy", zero_expert_num=2
    )
    assert longcat["expert_id_repeats"] is True


@pytest.mark.parametrize(
    "layer, clamped",
    [
        (dict(activation="swiglu", swiglu_limit=10.0), True),
        (dict(activation="swiglu"), False),
        (dict(activation="silu"), False),
    ],
)
def test_activation_clamped_follows_the_swiglu_limit(monkeypatch, layer, clamped):
    # The W4A8 kernel's fixed FC2 activation scale assumes a bounded SwiGLU
    # output; the plan states whether the checkpoint provides that bound.
    assert _plan_kwargs(monkeypatch, **layer)["activation_clamped"] is clamped
