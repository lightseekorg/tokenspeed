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

"""Visual deepstack injection under a deferred (fused) all-reduce."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

# CPU-only test scheduled in runtime-1gpu because it imports the full runtime.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.models.qwen3_5 import Qwen3_5ForCausalLM  # noqa: E402
from tokenspeed.runtime.models.qwen3_omni import Qwen3OmniMoeTextModel  # noqa: E402

TP = 2
HIDDEN = 8
NUM_DEEPSTACK = 3


class _DeferredReduceLayer:
    """A TP rank whose MLP output leaves the layer un-reduced: the next layer's
    opening add reduces it, as the fused all-reduce + norm does."""

    def __init__(self):
        self.comm_manager = SimpleNamespace(
            final_norm=lambda hidden, residual, ctx, norm: (hidden, None),
        )
        self.input_stream: torch.Tensor | None = None
        self.partial: torch.Tensor | None = None
        self.residual: torch.Tensor | None = None

    def __call__(self, *args, **kwargs):
        # Qwen3.5 passes keywords; Qwen3-Omni passes (positions, hidden, ctx, residual).
        if args:
            _, hidden_states, _, residual = args
        else:
            hidden_states, residual = kwargs["hidden_states"], kwargs["residual"]
        stream = hidden_states * TP + (0 if residual is None else residual)
        self.input_stream = stream.clone()
        self.partial = stream * 0.5
        self.residual = stream.clone()
        return self.partial.clone(), stream


@pytest.mark.parametrize("model_cls", [Qwen3_5ForCausalLM, Qwen3OmniMoeTextModel])
def test_deepstack_enters_the_reduced_stream_once(model_cls):
    layers = [_DeferredReduceLayer() for _ in range(NUM_DEEPSTACK + 1)]
    model = SimpleNamespace(
        layers=layers,
        layers_to_capture=set(),
        _dflash_capture_idx_map={},
        norm=None,
        hidden_size=HIDDEN,
        config=SimpleNamespace(hidden_size=HIDDEN),
    )
    x = torch.randn(3, HIDDEN)
    deepstack = torch.randn(3, HIDDEN * NUM_DEEPSTACK)
    ctx = SimpleNamespace(
        target_capture_sink=None,
        forward_mode=SimpleNamespace(is_idle=lambda: False),
    )

    model_cls.forward(
        model,
        torch.zeros(3, dtype=torch.long),
        None,
        ctx,
        input_embeds=x,
        input_deepstack_embeds=deepstack,
    )

    for i in range(NUM_DEEPSTACK):
        prev = layers[i]
        expected = (
            prev.partial * TP
            + prev.residual
            + deepstack[:, i * HIDDEN : (i + 1) * HIDDEN]
        )
        torch.testing.assert_close(layers[i + 1].input_stream, expected)
