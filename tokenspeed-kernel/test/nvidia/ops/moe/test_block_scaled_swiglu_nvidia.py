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

"""The block-scaled experts' capped SwiGLU against the checkpoint reference's PTX.

The reference kernels compute the sigmoid of the uncapped gate as
``rcp.approx.ftz(1 + ex2.approx.ftz(-log2(e) * gate))`` and each cap as
``cap * tanh.approx(div_rn(x, cap))``, multiply gamma last without FMA
contraction and round once to BF16. The oracle below restates that sequence;
the route quantizer's BF16 SwiGLU rows must equal it for every finite BF16
gate, in its decode-sized and prefill-sized forms.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.ops.moe._triton import block_scaled as triton_routes

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NVIDIA PTX approximations"
)


@triton.jit
def _reference_swiglu(X, G, Y, N, SIZE, GATE_CAP: tl.constexpr, UP_CAP: tl.constexpr):
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    gate = tl.load(X + 2 * i, i < SIZE, other=0).to(tl.float32)
    up = tl.load(X + 2 * i + 1, i < SIZE, other=0).to(tl.float32)
    sigmoid = tl.inline_asm_elementwise(
        "{ .reg .f32 t; mul.rn.f32 t, $1, 0fBFB8AA3B; "
        "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
        "rcp.approx.ftz.f32 $0, t; }",
        "=f,f",
        [gate],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    gate = GATE_CAP * tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        "=f,f",
        [tl.div_rn(gate, GATE_CAP)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    up = UP_CAP * tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        "=f,f",
        [tl.div_rn(up, UP_CAP)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    gamma = tl.load(G + i // N, i < SIZE, other=0)
    tl.store(Y + i, ((gate * sigmoid) * up) * gamma, i < SIZE)


@pytest.mark.parametrize("routes", [256, 2048])  # chunk and per-route programs
def test_swiglu_rows_equal_the_reference_ptx_for_every_bf16_gate(routes):
    width = 256
    generator = torch.Generator(device="cuda").manual_seed(3)
    # Every finite BF16 bit pattern as a gate, cycled over the routes' columns.
    patterns = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    finite = patterns.view(torch.bfloat16).isfinite()
    gates = patterns.view(torch.bfloat16)[finite]
    gates = gates.repeat(routes * width // gates.numel() + 1)[: routes * width]
    ups = (
        torch.randn(routes * width, generator=generator, device="cuda") * 30
    ).bfloat16()
    gate_up = torch.stack((gates, ups), dim=-1).view(routes, 2 * width).contiguous()
    gamma = torch.rand(routes, generator=generator, device="cuda") * 2
    ids = torch.zeros(routes, device="cuda", dtype=torch.int32)
    output = torch.empty(routes, 2 * width, device="cuda", dtype=torch.bfloat16)
    second, second_scale, second_amax = triton_routes._quantized_buffers(
        routes, width, True, gate_up.device
    )
    triton_routes._swiglu_quantize_routes(
        gate_up,
        gamma,
        ids,
        output,
        second,
        second_scale,
        second_amax,
        None,
        second_format="nvfp4",
        expert_start=0,
        num_experts=1,
        gate_cap=4.0,
        up_cap=25.0,
        zero_remote_routes=False,
        route_rows=None,
        w_amax=None,
        w_denominator=None,
        layout=0,
    )
    expected = torch.empty(routes, width, device="cuda", dtype=torch.bfloat16)
    _reference_swiglu[(triton.cdiv(routes * width, 256),)](
        gate_up,
        gamma,
        expected,
        width,
        routes * width,
        4.0,
        25.0,
        enable_fp_fusion=False,
    )
    # The quantizer keeps each route's BF16 SwiGLU row in its output row.
    assert torch.equal(output[:, :width].view(torch.int16), expected.view(torch.int16))
