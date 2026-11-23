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

"""PagedAttention.attend's rmsnorm_output_gate: the decode of a backend that
gates its own output receives the gate; every other forward returns the
separate weightless_rmsnorm_gated_sigmoid of the attention output."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.layernorm import weightless_rmsnorm_gated_sigmoid

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.paged_attention import PagedAttention

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

HEADS, HEAD_DIM, ROWS = 4, 64, 3


class _Backend:
    """A backend that returns fixed attention rows, gating them itself when
    handed the gate, as a gating decode kernel does."""

    def __init__(self, rows: torch.Tensor, gates: bool) -> None:
        self.rows = rows
        self.gates = gates
        self.received_gate = False

    def gates_decode_output(self, layer: PagedAttention) -> bool:
        return self.gates

    def forward(self, q, k, v, layer, pool, mode, bs, *, save_kv_cache, ctx, **kwargs):
        gate = kwargs.pop("rmsnorm_output_gate", None)
        assert not kwargs
        if gate is None:
            return self.rows
        self.received_gate = True
        return weightless_rmsnorm_gated_sigmoid(
            self.rows, gate[0], HEADS, HEAD_DIM, gate[1], False
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("mode", [ForwardMode.DECODE, ForwardMode.EXTEND])
@pytest.mark.parametrize("gates", [True, False], ids=["gating", "plain"])
def test_attend_gates_once_where_the_backend_cannot(mode, gates):
    generator = torch.Generator(device="cuda").manual_seed(0)
    rows = torch.randn(
        ROWS, HEADS * HEAD_DIM, device="cuda", generator=generator
    ).bfloat16()
    gate = torch.randn(
        ROWS, 2 * HEADS * HEAD_DIM, device="cuda", generator=generator
    ).bfloat16()[:, HEADS * HEAD_DIM :]
    backend = _Backend(rows, gates)
    layer = PagedAttention(HEADS, HEAD_DIM, 0.125, 1, 0, rotary_emb=None, qk_norm=None)
    ctx = SimpleNamespace(
        forward_mode=mode, attn_backend=backend, token_to_kv_pool=None, bs=1
    )
    q = torch.zeros(ROWS, HEADS, HEAD_DIM, device="cuda").bfloat16()
    output = layer.attend(q, None, None, ctx, rmsnorm_output_gate=(gate, 1e-6))
    expected = weightless_rmsnorm_gated_sigmoid(
        rows, gate, HEADS, HEAD_DIM, 1e-6, False
    )
    assert torch.equal(output, expected)
    assert backend.received_gate == (gates and mode.is_decode())
