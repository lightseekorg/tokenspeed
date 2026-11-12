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

"""The relative-attention gate/sink epilogue against the reference's PTX sigmoid.

The reference kernels compute ``sigmoid(x)`` as ``rcp.approx(ex2.approx(-x *
log2(e)) + 1)`` without flushing subnormals, gate first, then the sink, each
product rounded alone. The oracle restates that kernel; the epilogue must
equal it for every finite BF16 gate.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.attention.rmha import rel_mha_gate_sink

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NVIDIA PTX approximations"
)


@triton.jit
def _reference_sigmoid(x):
    exp = tl.inline_asm_elementwise(
        "ex2.approx.f32 $0, $1;",
        "=f,f",
        [-x * 1.4426950408889634],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return tl.inline_asm_elementwise(
        "rcp.approx.f32 $0, $1;",
        "=f,f",
        [exp + 1.0],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _reference_gate_sink(
    X, Gate, LSE, Sink, Y, SX, SG, SL, SS, H: tl.constexpr, D: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.program_id(1) * 256 + tl.arange(0, 256)
    mask = col < H * D
    head = col // D
    x = tl.load(X + row * SX + col, mask, other=0).to(tl.float32)
    gate = tl.load(Gate + row * SG + col, mask, other=0).to(tl.float32)
    lse = tl.load(LSE + row * SL + head, mask, other=0).to(tl.float32)
    sink = tl.load(Sink + row * SS + head, mask, other=0).to(tl.float32)
    result = x * _reference_sigmoid(gate)
    result = result * _reference_sigmoid(lse - sink)
    tl.store(Y + row * H * D + col, result, mask)


def _reference(attention, gate, lse, sink, heads, head_dim):
    out = torch.empty(attention.shape, device="cuda", dtype=torch.bfloat16)
    _reference_gate_sink[(attention.shape[0], triton.cdiv(heads * head_dim, 256))](
        attention,
        gate,
        lse,
        sink,
        out,
        attention.stride(0),
        gate.stride(0),
        lse.stride(0),
        sink.stride(0),
        heads,
        head_dim,
        enable_fp_fusion=False,
    )
    return out


def test_gate_sink_equals_the_reference_ptx_for_every_bf16_gate():
    tokens, heads, head_dim = 128, 8, 128
    generator = torch.Generator(device="cuda").manual_seed(5)
    patterns = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    gates = patterns.view(torch.bfloat16)[patterns.view(torch.bfloat16).isfinite()]
    gate = gates.repeat(3)[: tokens * heads * head_dim].view(tokens, -1).contiguous()
    attention = torch.randn(
        tokens, heads * head_dim, generator=generator, device="cuda"
    )
    lse = torch.randn(tokens, heads, generator=generator, device="cuda") * 8
    sink = torch.randn(tokens, heads, generator=generator, device="cuda") * 8
    actual = rel_mha_gate_sink(
        attention, gate, lse, sink, head_dim=head_dim, out_dtype=torch.bfloat16
    )
    expected = _reference(attention, gate, lse, sink, heads, head_dim)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


# Past 256 tokens a program walks several tokens of its head.
@pytest.mark.parametrize("tokens", [37, 600])
def test_strided_gate_and_sink_and_infinite_lse_follow_the_reference(tokens):
    generator = torch.Generator(device="cuda").manual_seed(11)
    heads, head_dim = 12, 128
    attention = torch.randn(
        tokens, heads * head_dim, generator=generator, device="cuda"
    )
    # Gate and sink are strided column slices of one projection, as in the model.
    projected = (
        torch.randn(
            (tokens, heads * head_dim + heads + 64), generator=generator, device="cuda"
        )
        * 8
    ).bfloat16()
    gate = projected[:, : heads * head_dim]
    sink = projected[:, heads * head_dim : heads * head_dim + heads]
    lse = torch.randn(tokens, heads, generator=generator, device="cuda") * 20
    lse[0, 0], lse[1, 1] = float("inf"), -float("inf")
    actual = rel_mha_gate_sink(
        attention, gate, lse, sink, head_dim=head_dim, out_dtype=torch.bfloat16
    )
    expected = _reference(attention, gate, lse, sink, heads, head_dim)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
