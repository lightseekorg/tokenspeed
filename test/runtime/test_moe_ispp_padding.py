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

from dataclasses import dataclass
from types import SimpleNamespace

from tokenspeed.runtime.layers.moe import expert as expert_mod
from tokenspeed.runtime.layers.moe.expert import MoELayer

# The alignment declared by the flashinfer_trtllm unquant MoE kernels
# (tokenspeed_kernel/ops/moe/flashinfer/trtllm_unquant.py, ispp_alignment).
_UNQUANT_ALIGNMENT = 128


@dataclass
class _SpecStub:
    intermediate_size: int


def _padding_stub(intermediate_size: int, tp_size: int = 2):
    stub = SimpleNamespace(
        intermediate_size=intermediate_size,
        tp_size=tp_size,
        prefix="model.layers.0.mlp.experts",
        _spec=_SpecStub(intermediate_size=intermediate_size),
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
