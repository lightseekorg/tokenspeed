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

"""HC4 application: BF16 rounding, input/output orientation and live graph inputs."""

import pytest
import torch
import torch_npu
from tokenspeed_kernel.ops.residual import mhc_apply_pre_norm, mhc_post


@pytest.fixture(scope="module", autouse=True)
def device():
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(20261017)


def pre_reference(x, pre, weight, eps):
    mixed = (pre.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype).float()
    normalized = mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + eps)
    return (weight.float() * normalized).to(x.dtype)


def post_reference(x, residual, post, comb):
    mixed = (comb.unsqueeze(-1) * residual.float().unsqueeze(-2)).sum(-3)
    return (post * x.float().unsqueeze(-2) + mixed).to(x.dtype)


def check(actual, expected):
    a = actual.cpu()
    e = expected.cpu()
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a, e, atol=1e-6, rtol=1 / 128)
    if a.numel():
        relative_l2 = (a.float() - e.float()).norm() / e.float().norm().clamp_min(1e-20)
        assert relative_l2 < 1e-4, (
            float(relative_l2),
            float((a.float() - e.float()).abs().max()),
        )


def inputs(n, h, strided):
    step = 2 if strided else 1
    residual = torch.randn(n, 4, h * step, device="npu", dtype=torch.bfloat16)[
        :, :, ::step
    ]
    x = torch.randn(n, h * step, device="npu", dtype=torch.bfloat16)[:, ::step]
    pre = torch.rand(n, 16, device="npu")[:, :4]
    post = torch.rand(n, 16, device="npu")[:, :4, None] * 2
    comb = torch.rand(n, 4, 4, device="npu")
    comb = comb / comb.sum(-2, keepdim=True)
    if strided:
        comb = comb.transpose(1, 2)
    weight = torch.randn(h * step, device="npu", dtype=torch.bfloat16)[::step]
    return x, residual, pre, post, comb, weight


@pytest.mark.parametrize(
    "n,h,strided",
    [
        (0, 5120, False),
        (1, 5120, False),
        (8, 5120, False),
        (16, 5120, False),
        (32, 5120, False),
        (129, 5120, False),
        (2048, 5120, False),
        (3, 37, True),
        (32, 5120, True),
        (8, 4096, False),
        (8, 512, False),
    ],
)
def test_eager(n, h, strided):
    x, r, pre, post, comb, w = inputs(n, h, strided)
    for eps in (1e-6, 1e-20):
        check(mhc_apply_pre_norm(r, pre, w, eps), pre_reference(r, pre, w, eps))
    check(
        mhc_post(x, r, post, comb, override=None, solution=None),
        post_reference(x, r, post, comb),
    )


@pytest.mark.parametrize("n", [1, 8, 16, 32])
def test_graph_runtime(n):
    from tokenspeed.runtime.layers.layernorm import RMSNorm
    from tokenspeed.runtime.models.deepseek_v41 import _v41_hc_input, v41_hc_post

    x, r, pre, post, comb, w = inputs(n, 5120, False)
    norm = RMSNorm(5120, eps=1e-6).to(device="npu", dtype=torch.bfloat16)
    norm.weight.data.copy_(w)

    def forward():
        z = _v41_hc_input(r, pre, norm)
        return z, v41_hc_post(x, r, post.squeeze(-1), comb)

    forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, auto_dispatch_capture=True):
        a, b = forward()
    for scale in (0.0, 1.0, 1e-3, 100.0, 0.25):
        r.copy_(torch.randn_like(r) * scale)
        x.copy_(torch.randn_like(x) * scale)
        pre.copy_(torch.rand_like(pre))
        post.copy_(torch.rand_like(post) * 2)
        comb.copy_(torch.rand_like(comb))
        norm.weight.data.copy_(torch.randn_like(w))
        graph.replay()
        torch.npu.synchronize()
        check(a, pre_reference(r, pre, norm.weight, 1e-6))
        check(b, post_reference(x, r, post, comb))


@pytest.mark.parametrize("h", [37, 5120])
def test_orientation_rounding_and_cancellation(h):
    x, r, pre, post, comb, w = inputs(2, h, True)
    # Distinct powers of two and a cyclic permutation expose transposed comb.
    for i in range(4):
        r[:, i, :] = 2.0**i
    x.fill_(0.5)
    post.fill_(0.25)
    comb.zero_()
    pre.zero_()
    w.fill_(1)
    for i in range(4):
        comb[:, i, (i + 1) % 4] = 1
    pre[:, 2] = 1
    assert torch.equal(
        mhc_post(x, r, post, comb, override=None, solution=None),
        post_reference(x, r, post, comb),
    )
    check(mhc_apply_pre_norm(r, pre, w, 1e-6), pre_reference(r, pre, w, 1e-6))
    r[:, 0, :] = 1
    r[:, 1, :] = -1
    r[:, 2, :] = 1 / 256
    r[:, 3, :] = -1 / 512
    pre.fill_(1)
    comb.fill_(0.25)
    check(mhc_apply_pre_norm(r, pre, w, 1e-6), pre_reference(r, pre, w, 1e-6))
    assert torch.equal(
        mhc_post(x, r, post, comb, override=None, solution=None),
        post_reference(x, r, post, comb),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fp32_norm_weight_and_bf16_rounding(dtype):
    x, r, pre, post, comb, w = inputs(32, 5120, False)
    w = w.to(dtype)
    pre.copy_(torch.rand_like(pre) * 0.5)
    expected = pre_reference(r, pre, w, 1e-6)
    mixed = (pre.unsqueeze(-1) * r.float()).sum(-2)
    wrong = (
        mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-6) * w.float()
    ).bfloat16()
    assert not torch.equal(expected, wrong)
    check(mhc_apply_pre_norm(r, pre, w, 1e-6), expected)
