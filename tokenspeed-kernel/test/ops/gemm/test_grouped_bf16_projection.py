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

"""The torch grouped projection keeps the einsum rounding and a flat row layout."""

import pytest
import torch
from tokenspeed_kernel.numerics.torch.gemm import torch_grouped_bf16_projection
from tokenspeed_kernel.ops.gemm import grouped_bf16_projection

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("groups", [1, 2, 8])
@pytest.mark.parametrize("tokens", [1, 3, 256])
def test_torch_projection_matches_einsum_bitwise_in_token_major_rows(groups, tokens):
    torch.manual_seed(groups * 1000 + tokens)
    # A slice of padded heads, as the attention output reaches the projection.
    heads = torch.randn(tokens, 16 * groups, 256, device="cuda", dtype=torch.bfloat16)
    x = heads[:, : 8 * groups].reshape(tokens, groups, -1)
    # Module weights are parameters, which still require grad outside serving.
    weight = torch.nn.Parameter(
        torch.randn(groups, 384, 2048, device="cuda", dtype=torch.bfloat16)
    )
    expected = torch.einsum("tgd,grd->tgr", x, weight)

    result = grouped_bf16_projection(x, weight, None, "torch")
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    # Token-major rows let the caller flatten the groups without a copy.
    assert result.is_contiguous()

    out = torch.full(expected.shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    assert grouped_bf16_projection(x, weight, out, "torch") is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [5, 8, 128, 200])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("dispatch", [False, True])
def test_grouped_projection_matches_exact_dyadic_product(tokens, padded, dispatch):
    torch.manual_seed(2031)
    storage = torch.randint(-8, 9, (tokens, 16 if padded else 8, 4096), device="cuda")
    x = storage.bfloat16().mul_(0.125)[:, :8]
    weight = torch.nn.Parameter(
        torch.randint(-8, 9, (8, 128, 4096), device="cuda").bfloat16().mul_(0.125)
    )
    # Products and all partial sums are exactly representable in FP32. The
    # FP64 reference therefore needs no tolerance for GEMM reduction order.
    expected = (
        torch.bmm(
            x.transpose(0, 1).double().contiguous(),
            weight.transpose(1, 2).double().contiguous(),
        )
        .transpose(0, 1)
        .bfloat16()
    )
    for supplied in (False, True):
        out = (
            torch.full(expected.shape, float("nan"), dtype=x.dtype, device=x.device)
            if supplied
            else None
        )
        result = (
            grouped_bf16_projection(x, weight, out, "torch")
            if dispatch
            else torch_grouped_bf16_projection(x, weight, out)
        )
        assert result.is_contiguous()
        if supplied:
            assert result is out
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dispatch", [False, True])
def test_grouped_projection_graph_replay_refreshes_rows(dispatch):
    x = torch.zeros(5, 8, 4096, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(8, 128, 4096, dtype=x.dtype, device=x.device)
    out = torch.empty(5, 8, 128, dtype=x.dtype, device=x.device)

    def project():
        if dispatch:
            return grouped_bf16_projection(x, weight, out, "torch")
        return torch_grouped_bf16_projection(x, weight, out)

    project()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        project()
    for value in (0.125, -0.25, 0.5):
        x.fill_(value)
        graph.replay()
        torch.testing.assert_close(
            out, torch.full_like(out, 4096 * value), rtol=0, atol=0
        )
