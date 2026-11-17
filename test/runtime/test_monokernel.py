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
"""The part of a decode monokernel seam that every model shares: when a layer
launches, how a forward moves the step counter, and the startup error. CPU
only."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.models.monokernel import (  # noqa: E402
    MonokernelLayer,
    MonokernelState,
    unserved_error,
    unserved_report,
)


def test_a_forward_ends_with_one_new_tag_when_it_launched() -> None:
    """The step counter changes once per forward that launches: in
    finish_step, unless the last launch gave the LM head's input (and moved
    the counter itself). A forward without launches adds no kernel."""
    state = MonokernelState(torch.zeros(1, dtype=torch.int64))

    def forward(launched: bool, folded: bool):
        state.start_step()
        state.launched = launched
        if folded:
            state.lm_head_input = torch.ones(1, 2)
        return state.finish_step()

    assert forward(launched=False, folded=False) is None
    assert state.step.tolist() == [0, 0]
    assert forward(launched=True, folded=False) is None
    assert state.step.tolist() == [1, 0]
    lm_head_input = forward(launched=True, folded=True)
    assert lm_head_input is not None and lm_head_input.tolist() == [[1.0, 1.0]]
    assert state.step.tolist() == [1, 0]


def test_the_report_groups_layers_by_reason() -> None:
    unserved = {3: ["a"], 0: ["a", "b"], 7: ["c"]}
    assert unserved_report(unserved) == "a: layers 0, 3; b: layer 0; c: layer 7"


def test_the_startup_error_names_every_layer_without_a_monokernel() -> None:
    """The switch never serves a layer without its kernel: startup fails and
    names each such layer with its reasons."""
    assert unserved_error("Target", {0: ["a"], 1: ["a", "b"]}) == (
        "--decode-monokernel: 2 Target layers have no monokernel "
        "(a: layers 0, 1; b: layer 1). Serve without --decode-monokernel."
    )


class _RegularLayer(torch.nn.Module):
    """A regular decoder layer that records the forwards it runs."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_id = 0
        self.forwards = 0

    def forward(self, hidden_states, residual, ctx):
        self.forwards += 1
        return hidden_states, residual


class _Layer(MonokernelLayer, _RegularLayer):
    """A monokernel layer whose kernels serve forwards of at most 2 rows."""

    def launch_chain(self, rows: int, ctx) -> int | None:
        return 1 if rows <= 2 else None

    def _decode(self, chain: int, *, hidden_states, residual, ctx):
        return hidden_states + chain, None


def test_a_monokernel_layer_runs_other_forwards_with_its_regular_code() -> None:
    layer = _Layer()
    rows = torch.zeros(3, 2)
    out = layer(hidden_states=rows, residual=rows, ctx=None)
    assert out[0] is rows and out[1] is rows and layer.forwards == 1


def test_a_monokernel_layer_refuses_a_decode_step_before_packing() -> None:
    """A step that the kernels serve needs the packed layer: without it the
    layer fails instead of running its regular code."""
    layer = _Layer()
    with pytest.raises(RuntimeError, match="before post_quant_warmup"):
        layer(hidden_states=torch.zeros(1, 2), residual=None, ctx=None)
    assert layer.forwards == 0


def test_a_packed_monokernel_layer_launches() -> None:
    """A packed layer returns its launch's outputs and records the launch for
    finish_step."""
    layer = _Layer()
    layer.monokernel_state = MonokernelState(torch.zeros(1, dtype=torch.int64))
    layer.monokernel_state.start_step()
    out = layer(hidden_states=torch.zeros(1, 2), residual=None, ctx=None)
    assert out[0].tolist() == [[1.0, 1.0]] and out[1] is None
    assert layer.monokernel_state.launched and layer.forwards == 0
