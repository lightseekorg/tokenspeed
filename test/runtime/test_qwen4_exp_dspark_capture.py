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

"""Completed Qwen4-Exp HC outputs used by ordinary and HyperDSpark drafts."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.models.qwen4_exp import (
    Qwen4ExpForCausalLM,
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpModel,
)


class _HC:
    """CPU injection with distinct, nonuniform branch gates."""

    @staticmethod
    def combine(output, residual):
        branches = residual[0].unflatten(-1, (2, 2))
        gates = output.new_tensor([0.25, 0.75]).view(1, 2, 1)
        return (branches + output.unsqueeze(-2) * gates).flatten(-2)


class _Comm:
    def __init__(self, layer_id, gather, events):
        self.layer_id = layer_id
        self.gather = gather
        self.events = events

    def needs_final_all_gather(self):
        return self.gather

    def post_final_norm_comm(self, hidden, residual, ctx):
        if self.gather:
            self.events.append(self.layer_id)
            hidden = torch.cat((hidden, hidden + 100), dim=0)
        return hidden, residual


class _Layer(torch.nn.Module):
    def __init__(self, layer_id, gather, events):
        super().__init__()
        self.layer_id = layer_id
        self.ple = None
        self.mlp_hyper_connection = _HC()
        self.comm_manager = _Comm(layer_id, gather, events)

    def forward(self, positions, hidden_states, residual, ctx, input_ids):
        if residual is not None:
            hidden_states = self.mlp_hyper_connection.combine(hidden_states, residual)
        elif hidden_states.shape[-1] == 2:
            hidden_states = hidden_states.repeat(1, 2)
        output = hidden_states.new_full((hidden_states.shape[0], 2), self.layer_id + 1)
        return output, (hidden_states,)


class _Mixer(torch.nn.Module):
    def combine_norm(self, output, residual):
        return _HC.combine(output, residual), None

    def mix(self, hidden_states, normalized):
        return hidden_states.unflatten(-1, (2, 2)).mean(dim=-2), (hidden_states,)


def _make_model(*, gather: bool):
    events = []
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hc_count=2)
    model.hidden_size = 2
    model.layers = torch.nn.ModuleList(_Layer(i, gather, events) for i in range(3))
    model.hyper_connection_mixer = _Mixer()
    model.dspark_layers_to_capture = ()
    model.dspark_capture_hc = None
    model._dspark_capture_idx_map = {}
    return model, events


def _forward(model, values, *, sink, deepstack):
    return model(
        input_ids=torch.arange(values.shape[0]),
        positions=torch.arange(values.shape[0]),
        ctx=SimpleNamespace(target_capture_sink=sink),
        input_embeds=values,
        input_deepstack_embeds=deepstack,
    )


@pytest.mark.parametrize("capture_hc", [False, True])
@pytest.mark.parametrize("gather", [False, True])
def test_capture_includes_final_mlp_and_preserves_checkpoint_order(
    capture_hc: bool, gather: bool
) -> None:
    model, gathers = _make_model(gather=gather)
    baseline, _ = _make_model(gather=gather)
    model.set_dspark_layers_to_capture([0, 2], capture_hc=capture_hc)
    values = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    callbacks = []
    sink = SimpleNamespace(
        on_target_capture=lambda index, rows: callbacks.append((index, rows))
    )
    output, captures = _forward(model, values, sink=sink, deepstack=None)
    baseline_output, mtp_capture = _forward(baseline, values, sink=None, deepstack=None)

    torch.testing.assert_close(output, baseline_output)
    assert [index for index, _ in callbacks] == [0, 1]
    for capture, total, (_, callback) in zip(captures, [1, 6], callbacks, strict=True):
        expected = torch.cat((values + total * 0.25, values + total * 0.75), dim=-1)
        if not capture_hc:
            expected = expected.unflatten(-1, (2, 2)).mean(dim=-2)
        if gather:
            expected = torch.cat((expected, expected + 100), dim=0)
        torch.testing.assert_close(capture, expected)
        torch.testing.assert_close(callback, capture)
    assert gathers == ([0, 2, 2] if gather else [])
    assert mtp_capture[0].shape[-1] == 4
    if capture_hc:
        torch.testing.assert_close(captures[1], mtp_capture[0])


def test_completed_capture_respects_deepstack_without_double_injection() -> None:
    model, _ = _make_model(gather=False)
    model.set_dspark_layers_to_capture([0, 2], capture_hc=False)
    values = torch.tensor([[1.0, 3.0]])
    deepstack = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]])

    output, captures = _forward(model, values, sink=None, deepstack=deepstack)

    torch.testing.assert_close(captures[0], values + deepstack[:, :2] + 0.5)
    expected_final = values + deepstack.unflatten(-1, (3, 2)).sum(dim=-2) + 3
    torch.testing.assert_close(captures[1], expected_final)
    torch.testing.assert_close(output, expected_final)


def test_raw_capture_does_not_alias_a_resolved_residual() -> None:
    model, _ = _make_model(gather=False)
    model.set_dspark_layers_to_capture([0], capture_hc=True)
    residual = torch.arange(8, dtype=torch.float32).view(2, 4)
    expected = residual.clone()

    captured = model._capture_dspark_hidden(0, residual, None, object())
    residual.add_(100)

    torch.testing.assert_close(captured, expected)


@pytest.mark.parametrize(
    "entry_class", [Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration]
)
def test_text_and_multimodal_entries_forward_the_hc_capture_choice(entry_class) -> None:
    model, _ = _make_model(gather=False)
    holder = SimpleNamespace(model=model, capture_aux_hidden_states=False)

    entry_class.set_dspark_layers_to_capture(holder, [0, 2], capture_hc=True)

    assert holder.capture_aux_hidden_states
    assert model.dspark_layers_to_capture == (0, 2)
    assert model.dspark_capture_hc is True


@pytest.mark.parametrize("layer_ids", [[], [0, 0], [2, 0], [-1, 2], [0, 3]])
def test_invalid_checkpoint_taps_are_rejected(layer_ids) -> None:
    model, _ = _make_model(gather=False)
    with pytest.raises(ValueError):
        model.set_dspark_layers_to_capture(layer_ids, capture_hc=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
