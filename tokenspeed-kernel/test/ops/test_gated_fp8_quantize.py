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

import pytest
import torch
from tokenspeed_kernel.ops.activation.triton import rmsnorm_gated_sigmoid, sigmoid_mul
from tokenspeed_kernel.ops.quantization import quantize_fp8_with_gate
from tokenspeed_kernel.platform import current_platform, pdl_enabled

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize(
    "rows,heads,head_dim,strided",
    [(1, 12, 64, False), (64, 96, 128, True), (257, 3, 256, True), (5, 20, 96, True)],
)
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    "enable_pdl", [False, True] if current_platform().is_hopper_plus else [False]
)
def test_gated_fp8_matches_separate_ops(
    rows, heads, head_dim, normalize, enable_pdl, strided
):
    torch.manual_seed(rows + heads)
    width = heads * head_dim
    x = torch.randn(
        rows, width + (128 if strided else 0), device="cuda", dtype=torch.bfloat16
    )[:, :width]
    gate = torch.randn(rows, width + 256, device="cuda", dtype=torch.bfloat16)[
        :, :width
    ]
    weight = (
        torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
        if normalize
        else None
    )
    if head_dim & (head_dim - 1):
        # Integer inputs and zero logits avoid sigmoid approximation differences.
        x.copy_(torch.arange(rows * width, device="cuda").reshape_as(x) % 13 - 6)
        gate.zero_()
        if weight is not None:
            weight.copy_(
                2.0 ** ((torch.arange(head_dim, device="cuda") % 5) - 2).float()
            )
    # Include all-zero groups to exercise the scale floor.
    x[:, :128] = 0
    if normalize and head_dim & (head_dim - 1):
        values = x.float().reshape(rows, heads, head_dim)
        activated = (
            (
                values
                * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-6)
                * weight.float()
                * gate.float().reshape_as(values).sigmoid()
            )
            .to(x.dtype)
            .reshape_as(x)
        )
    elif normalize:
        activated = rmsnorm_gated_sigmoid(
            x.contiguous(), gate, weight, 1e-6, heads, head_dim, enable_pdl=enable_pdl
        )
    else:
        activated = sigmoid_mul(
            x.clone(memory_format=torch.contiguous_format), gate.contiguous()
        )
    groups = activated.float().reshape(rows, width // 128, 128)
    floor = torch.tensor(1e-10, device=x.device, dtype=x.dtype).float()
    amax = groups.abs().amax(dim=-1).clamp_min(floor)
    quant_scale = torch.full_like(amax, 448.0) / amax
    expected_scales = torch.ones_like(quant_scale) / quant_scale
    expected = (groups * quant_scale[..., None]).to(torch.float8_e4m3fn).reshape_as(x)
    actual, scales = quantize_fp8_with_gate(
        x,
        gate,
        norm_weight=weight,
        eps=1e-6,
        enable_pdl=enable_pdl,
    )
    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)


@pytest.mark.parametrize("normalize", [False, True])
def test_gated_fp8_cuda_graph_uses_updated_inputs(normalize):
    x = torch.randn(64, 20 * 96, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(x)
    weight = torch.randn(96, device="cuda", dtype=torch.bfloat16) if normalize else None
    kwargs = dict(norm_weight=weight, eps=1e-6, enable_pdl=pdl_enabled())
    quantize_fp8_with_gate(x, gate, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, scales = quantize_fp8_with_gate(x, gate, **kwargs)
    for _ in range(3):
        x.normal_()
        gate.normal_()
        graph.replay()
        expected, expected_scales = quantize_fp8_with_gate(x, gate, **kwargs)
        torch.testing.assert_close(
            values.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0
        )
        torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
