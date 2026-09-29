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

import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.layers.moe import expert as expert_mod
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.utils.env import global_server_args_dict

register_cuda_ci(est_time=5, suite="runtime-1gpu")

# The alignment the stock FlashInfer TRT-LLM BF16 MoE launcher requires.
_UNQUANT_ALIGNMENT = 128


@dataclass
class _SpecStub:
    intermediate_size: int
    gated: bool


def _padding_stub(intermediate_size: int, tp_size: int = 2, gated: bool = True):
    stub = SimpleNamespace(
        intermediate_size=intermediate_size,
        tp_size=tp_size,
        prefix="model.layers.0.mlp.experts",
        _spec=_SpecStub(intermediate_size=intermediate_size, gated=gated),
    )
    apply = MoELayer._apply_trtllm_ispp_padding.__get__(stub)
    return stub, apply


def test_trtllm_ispp_padding_rounds_up_unaligned(monkeypatch):
    monkeypatch.setattr(
        expert_mod,
        "get_moe_backend",
        lambda: SimpleNamespace(value="flashinfer_trtllm"),
    )
    # ispp = 1000 -> padded to 1024 with the unquant kernel's 128 alignment.
    stub, apply = _padding_stub(intermediate_size=2000, tp_size=2)

    apply(_UNQUANT_ALIGNMENT, "test")

    assert stub.intermediate_size == 1024 * 2
    assert stub._spec.intermediate_size == 1024 * 2


def test_trtllm_ispp_padding_noop_when_aligned(monkeypatch):
    monkeypatch.setattr(
        expert_mod,
        "get_moe_backend",
        lambda: SimpleNamespace(value="flashinfer_trtllm"),
    )
    stub, apply = _padding_stub(intermediate_size=1024 * 2, tp_size=2)

    apply(_UNQUANT_ALIGNMENT, "test")

    assert stub.intermediate_size == 1024 * 2
    assert stub._spec.intermediate_size == 1024 * 2


def test_trtllm_ispp_padding_noop_for_other_backends(monkeypatch):
    monkeypatch.setattr(
        expert_mod,
        "get_moe_backend",
        lambda: SimpleNamespace(value="auto"),
    )
    stub, apply = _padding_stub(intermediate_size=2000, tp_size=2)

    apply(_UNQUANT_ALIGNMENT, "test")

    assert stub.intermediate_size == 2000
    assert stub._spec.intermediate_size == 2000


def test_auto_backend_pads_non_gated_experts_for_trtllm(monkeypatch):
    monkeypatch.setattr(
        expert_mod, "get_moe_backend", lambda: SimpleNamespace(value="auto")
    )
    # Nemotron-3 Super at TP2: ispp = 2688 / 2 = 1344 -> 1408, the relu2 kernels' alignment.
    relu2, apply_relu2 = _padding_stub(intermediate_size=2688, tp_size=2, gated=False)
    swiglu, apply_swiglu = _padding_stub(intermediate_size=2688, tp_size=2)

    apply_relu2(_UNQUANT_ALIGNMENT, "test")
    apply_swiglu(_UNQUANT_ALIGNMENT, "test")

    assert relu2.intermediate_size == 1408 * 2
    assert relu2._spec.intermediate_size == 1408 * 2
    assert swiglu.intermediate_size == 2688


@pytest.mark.parametrize(
    "activation, kernel_alignment, ispp, planned",
    [
        # Gated sizes the 64-aligned launcher accepts are not padded.
        ("silu", 64, 64, 64),
        ("swiglu", 64, 192, 192),
        # Other gated sizes pad to the next multiple of 64, not 128.
        ("silu", 64, 96, 128),
        ("silu", 64, 160, 192),
        # Without the 64-aligned launcher, padding stays at 128.
        ("silu", 128, 192, 256),
        # Activations the flashinfer_trtllm unquant kernels do not serve keep 128.
        ("situ", 64, 192, 256),
    ],
)
def test_unquant_padding_follows_the_trtllm_kernel_alignment(
    monkeypatch, activation, kernel_alignment, ispp, planned
):
    plans = []
    monkeypatch.setattr(
        expert_mod,
        "get_moe_backend",
        lambda: SimpleNamespace(value="flashinfer_trtllm"),
    )
    monkeypatch.setattr(expert_mod, "TRTLLM_UNQUANT_ISPP_ALIGNMENT", kernel_alignment)
    monkeypatch.setattr(
        expert_mod.tokenspeed_kernel,
        "moe_plan",
        lambda weight_dtype, **kwargs: plans.append(kwargs) or {"solution": "fake"},
    )
    monkeypatch.setattr(expert_mod, "create_layer_weights", lambda *a, **k: None)
    monkeypatch.setitem(global_server_args_dict, "moe_mxfp4_fp8_activation", False)
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)
    tp_size = 4
    MoELayer(
        top_k=2,
        num_experts=8,
        hidden_size=2048,
        intermediate_size=ispp * tp_size,
        quant_config=None,
        layer_index=0,
        prefix="model.layers.0.mlp.experts",
        tp_rank=0,
        tp_size=tp_size,
        activation=activation,
        activation_situ_beta=1.0 if activation == "situ" else None,
    )

    assert [plan["ispp"] for plan in plans] == [planned]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
