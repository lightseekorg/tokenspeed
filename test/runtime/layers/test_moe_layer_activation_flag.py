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

"""``--moe-mxfp4-fp8-activation`` is honoured or rejected by every MoELayer.

The flag lives in the common MoE construction, not in one model, so an
explicitly requested activation precision can never be dropped on the floor:
MXFP4 experts plan with FP8 activations, anything else refuses to start.
"""

from __future__ import annotations

import pytest

from tokenspeed.runtime.layers.moe import expert as expert_module
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.layers.quantization.mxfp4 import Mxfp4Config
from tokenspeed.runtime.utils.env import global_server_args_dict


def _layer(monkeypatch, quant_config, *, flag: bool, override=None):
    plans = []

    def fake_plan(weight_dtype, **kwargs):
        plans.append({"weight_dtype": weight_dtype, **kwargs})
        return {"solution": "fake", "apply_kernel_name": "fake"}

    monkeypatch.setattr(expert_module.tokenspeed_kernel, "moe_plan", fake_plan)
    monkeypatch.setattr(expert_module, "create_layer_weights", lambda *a, **k: None)
    monkeypatch.setitem(global_server_args_dict, "moe_mxfp4_fp8_activation", flag)
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)
    MoELayer(
        top_k=2,
        num_experts=8,
        hidden_size=256,
        intermediate_size=128,
        quant_config=quant_config,
        layer_index=0,
        prefix="model.layers.0.mlp",
        activation="swiglu",
        swiglu_limit=10.0,
        internal_activation_dtype_override=override,
    )
    assert len(plans) == 1
    return plans[0]


def _mxfp4():
    return Mxfp4Config(ignored_layers=[], is_checkpoint_mxfp4_serialized=True)


def test_flag_plans_fp8_activations_for_mxfp4_experts(monkeypatch):
    plan = _layer(monkeypatch, _mxfp4(), flag=True)
    assert plan["weight_dtype"] == "mxfp4"
    assert plan["internal_activation_dtype"] == "fp8"


def test_without_the_flag_mxfp4_experts_keep_the_input_dtype(monkeypatch):
    plan = _layer(monkeypatch, _mxfp4(), flag=False)
    assert plan["internal_activation_dtype"] == "input"


def test_flag_is_rejected_for_experts_that_are_not_mxfp4(monkeypatch):
    block_fp8 = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        ignored_layers=[],
        weight_block_size=[128, 128],
    )
    with pytest.raises(ValueError, match="applies to MXFP4 routed experts"):
        _layer(monkeypatch, block_fp8, flag=True)


def test_flag_is_rejected_when_the_model_pins_another_activation(monkeypatch):
    # Kimi-K3 pins "input" for its Marlin and gfx950 paths; the flag cannot be
    # honoured there and must not be overwritten silently.
    with pytest.raises(ValueError, match="pins its MXFP4 experts to 'input'"):
        _layer(monkeypatch, _mxfp4(), flag=True, override="input")


def test_flag_agrees_with_a_matching_model_override(monkeypatch):
    # Kimi-K3's TRT-LLM MXFP4 SiTU path already runs w4a8 and pins "fp8".
    plan = _layer(monkeypatch, _mxfp4(), flag=True, override="fp8")
    assert plan["internal_activation_dtype"] == "fp8"


def test_without_the_flag_the_model_override_is_authoritative(monkeypatch):
    plan = _layer(monkeypatch, _mxfp4(), flag=False, override="input")
    assert plan["internal_activation_dtype"] == "input"


def test_flag_is_accepted_for_any_backend_and_fails_closed_in_the_plan():
    # Which backends can honour FP8 activations is a registry fact (Hopper
    # cutlass, SM100 trtllm SiTU, AMD Gluon...), so ServerArgs does not keep
    # an allowlist; an unsupported backend fails at plan time instead.
    from tokenspeed.runtime.utils.server_args import prepare_server_args

    for argv in (
        ["--moe-backend", "flashinfer_trtllm"],
        ["--draft-moe-backend", "triton"],
    ):
        args = prepare_server_args(
            ["--model", "x", "--moe-mxfp4-fp8-activation", *argv]
        )
        assert args.moe_mxfp4_fp8_activation
