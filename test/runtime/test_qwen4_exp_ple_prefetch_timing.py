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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import tokenspeed.runtime.models.qwen4_exp as qwen4_exp_module
from tokenspeed.runtime.models.qwen4_exp import Qwen4ExpModel


class _RecordingPLE:
    def __init__(self, layer_id: int, events: list[tuple]) -> None:
        self.layer_id = layer_id
        self.events = events

    def start_prefetch(self, input_ids: torch.Tensor, ctx: object) -> None:
        self.events.append(("prefetch", self.layer_id, ctx))


class _RecordingLayer(torch.nn.Module):
    def __init__(self, layer_id: int, ple: _RecordingPLE | None, events: list[tuple]):
        super().__init__()
        self.layer_id = layer_id
        self.ple = ple
        self.events = events
        self.comm_manager = SimpleNamespace(needs_final_all_gather=lambda: False)

    def forward(self, positions, hidden_states, residual, ctx, input_ids):
        self.events.append(("layer", self.layer_id, ctx))
        return hidden_states, residual


class _RecordingMixer(torch.nn.Module):
    def mix(self, hidden_states, normalized):
        return hidden_states, (hidden_states,)


class _RecordingCapture:
    _capturing = True

    def __init__(self) -> None:
        self.callbacks = []

    def add_eager(self, callback):
        self.callbacks.append(callback)
        return callback()


@pytest.mark.parametrize("ple_layer_ids", [(), (0,), (1,), (1, 2), (0, 3)])
@pytest.mark.parametrize("capturing", [False, True])
def test_qwen4_exp_prefetch_starts_at_preceding_layer(
    monkeypatch, ple_layer_ids: tuple[int, ...], capturing: bool
) -> None:
    events: list[tuple] = []
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    torch.nn.Module.__init__(model)
    model.layers = torch.nn.ModuleList(
        _RecordingLayer(
            layer_id,
            _RecordingPLE(layer_id, events) if layer_id in ple_layer_ids else None,
            events,
        )
        for layer_id in range(4)
    )
    model.hyper_connection_mixer = _RecordingMixer()
    monkeypatch.setattr(
        qwen4_exp_module,
        "get_global_expert_distribution_recorder",
        lambda: SimpleNamespace(with_current_layer=lambda _: nullcontext()),
    )
    capture = _RecordingCapture() if capturing else None
    monkeypatch.setattr(
        qwen4_exp_module.BreakableCapture,
        "current",
        classmethod(lambda cls: capture),
    )
    ctx = object()
    live_ctx = [ctx]
    monkeypatch.setattr(qwen4_exp_module, "current_forward_ctx", lambda: live_ctx[0])
    input_ids = torch.arange(2)
    hidden_states = torch.zeros(2, 4)

    model(
        input_ids=input_ids,
        positions=input_ids,
        ctx=ctx,
        input_embeds=hidden_states,
    )

    expected = []
    if 0 in ple_layer_ids:
        expected.append(("prefetch", 0, ctx))
    for layer_id in range(4):
        if layer_id + 1 in ple_layer_ids:
            expected.append(("prefetch", layer_id + 1, ctx))
        expected.append(("layer", layer_id, ctx))
    assert events == expected

    if capture is not None:
        assert len(capture.callbacks) == len(ple_layer_ids)
        replay_ctx = object()
        live_ctx[0] = replay_ctx
        events.clear()
        for callback in capture.callbacks:
            callback()
        assert events == [
            ("prefetch", layer_id, replay_ctx) for layer_id in ple_layer_ids
        ]
