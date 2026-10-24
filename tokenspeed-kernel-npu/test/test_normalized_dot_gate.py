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

"""Exact parity with the prior Engram gate, including changing graph inputs."""

import pytest
import torch
import torch_npu  # noqa: F401
from tokenspeed_kernel.ops.residual import normalized_dot_gate


def reference(h, kv, qw, kw, mask, eps):
    hc, dim = h.shape[-2:]
    key, value = kv.split([hc * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc, dim))
    x = h.float()
    weight = qw.float() * kw.float()
    rstd = torch.rsqrt(x.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (x * weight * key).sum(-1) * rstd * dim**-0.5
    magnitude = dot.abs().clamp_min(1e-6).sqrt()
    negative = dot.contiguous().view(torch.int32) < 0
    gate = torch.sigmoid(torch.where(negative, -magnitude, magnitude))
    gate = gate.masked_fill(~mask.unsqueeze(-1), 0)
    return (x + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(h.dtype)


def operands(shape):
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    hc, dim = shape[-2:]
    return (
        torch.randn(shape, device="npu", dtype=torch.bfloat16),
        torch.randn((*shape[:-2], (hc + 1) * dim), device="npu", dtype=torch.bfloat16),
        torch.randn((hc, dim), device="npu", dtype=torch.bfloat16),
        torch.randn((hc, dim), device="npu", dtype=torch.bfloat16),
        torch.randint(2, shape[:-2], device="npu", dtype=torch.int32).bool(),
    )


def assert_exact(actual, expected):
    a, b = actual.cpu(), expected.cpu()
    torch.testing.assert_close(a, b, atol=0, rtol=0, equal_nan=True)
    finite = torch.isfinite(b)
    assert torch.equal(torch.signbit(a)[finite], torch.signbit(b)[finite])


@pytest.mark.parametrize(
    "shape",
    [
        (0, 4, 5120),
        (1, 4, 32),
        (7, 4, 128),
        (8, 4, 512),
        (48, 4, 5120),
        (49, 4, 5120),
        (2, 3, 4, 5120),
        (2, 4, 8192),
        (3, 4, 1),
        (3, 4, 33),
        (2, 4, 8193),
    ],
)
def test_exact_previous_expression(shape):
    for seed in range(4):
        torch.manual_seed(20261020 + seed)
        values = operands(shape)
        assert_exact(normalized_dot_gate(*values, 1e-20), reference(*values, 1e-20))


@pytest.mark.parametrize("shape", [(8, 4, 32), (48, 4, 5120)])
def test_native_graph_reads_live_inputs_weights_and_mask(shape):
    values = operands(shape)
    for _ in range(3):
        normalized_dot_gate(*values, 1e-20)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = normalized_dot_gate(*values, 1e-20)
    for seed in range(8):
        torch.manual_seed(20261020 + seed)
        for value in values[:-1]:
            value.normal_()
        values[-1].copy_(
            torch.randint(2, shape[:-2], device="npu", dtype=torch.int32).bool()
        )
        if seed == 2:
            values[0].zero_()
        if seed == 3:
            values[2].zero_()
            values[3].fill_(-0.0)
        if seed == 4:
            values[-1].zero_()
        graph.replay()
        torch.npu.synchronize()
        assert_exact(output, reference(*values, 1e-20))


def test_nonfinite_and_mask_preserve_original_arithmetic():
    values = operands((8, 4, 5120))
    h, kv, _qw, _kw, mask = values
    h[0].fill_(float("inf"))
    h[1].fill_(float("nan"))
    kv[2].fill_(float("inf"))
    h[3].zero_()
    kv[3].zero_()
    h[4].fill_(torch.finfo(torch.bfloat16).max)
    mask[::2] = False
    assert_exact(normalized_dot_gate(*values, 1e-20), reference(*values, 1e-20))
