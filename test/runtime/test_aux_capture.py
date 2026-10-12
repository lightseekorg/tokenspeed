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

"""Aux hidden-state capture under a deferred (fused) all-reduce."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

# CPU-only test scheduled in runtime-1gpu because it imports the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.models.base.decoder_layer import (  # noqa: E402
    capture_layer_input,
)
from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3Model  # noqa: E402
from tokenspeed.runtime.models.longcat_flash import _RuntimeLongcatModel  # noqa: E402
from tokenspeed.runtime.models.qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402

TP = 2


class _DeferredReduceLayer:
    """A TP rank whose MLP output leaves the layer un-reduced: the next layer's
    opening add reduces it, as the fused all-reduce + norm does."""

    def __init__(self):
        comm = SimpleNamespace(
            gather_residual=lambda rows, ctx: rows,
            final_norm=lambda hidden, residual, ctx, norm: (hidden, None),
        )
        self.comm_manager = comm
        self.branch_comm = [comm, comm]
        self.final_norm_comm = comm
        self.input_stream: torch.Tensor | None = None

    def __call__(self, *args, aux_hidden_states=None, **kwargs):
        # Qwen3.5 passes keywords; DeepSeek-V3 and LongCat pass (positions, hidden, ctx, residual).
        if args:
            _, hidden_states, _, residual = args
        else:
            hidden_states, residual = kwargs["hidden_states"], kwargs["residual"]
        stream = hidden_states * TP + (0 if residual is None else residual)
        self.input_stream = stream.clone()
        capture_layer_input(self.comm_manager, stream, None, aux_hidden_states)
        partial = stream * 0.5
        # A later norm may update the residual in place; the capture must not see it.
        stream.mul_(0.25)
        return partial, stream


@pytest.mark.parametrize(
    "model_cls", [Qwen3_5ForCausalLM, DeepseekV3Model, _RuntimeLongcatModel]
)
def test_aux_capture_is_the_reduced_layer_input(model_cls):
    layers = [_DeferredReduceLayer() for _ in range(4)]
    model = SimpleNamespace(
        layers=layers,
        layers_to_capture={1, 3},
        _dflash_capture_idx_map={},
        norm=None,
    )
    x = torch.randn(3, 8)
    ctx = SimpleNamespace(
        target_capture_sink=None,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
    )

    _, aux = model_cls.forward(
        model,
        torch.zeros(3, dtype=torch.long),
        None,
        ctx,
        input_embeds=x,
    )

    assert len(aux) == 2
    for got, layer in zip(aux, (layers[1], layers[3])):
        torch.testing.assert_close(got, layer.input_stream, rtol=0, atol=0)


def test_capture_layer_input_gathers_and_detaches():
    gathered_from = []

    def gather_residual(rows, ctx):
        gathered_from.append(rows)
        return rows

    comm = SimpleNamespace(gather_residual=gather_residual)
    residual = torch.randn(3, 8)
    expected = residual.clone()
    aux: list[torch.Tensor] = []

    capture_layer_input(comm, residual, None, None)
    assert gathered_from == []

    capture_layer_input(comm, residual, None, aux)
    residual.mul_(0.25)

    assert gathered_from == [residual]
    torch.testing.assert_close(aux[0], expected, rtol=0, atol=0)
