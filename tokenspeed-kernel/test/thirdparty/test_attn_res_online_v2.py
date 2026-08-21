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
from tokenspeed_kernel.thirdparty.cuda.attn_res import attn_res_fwd_packed


def _blackwell_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


pytestmark = pytest.mark.skipif(
    not _blackwell_available(), reason="SM100-family CUDA GPU is required"
)


def _reference(
    layer: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    out_norm_weight: torch.Tensor | None,
    num_blocks: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    updated_layer = layer.clone()
    if delta is not None:
        updated_layer = (updated_layer + delta).to(torch.bfloat16)
    values = torch.cat((blocks[:num_blocks], updated_layer.unsqueeze(0)), dim=0)
    keys = torch.nn.functional.rms_norm(
        values.float(), (values.shape[-1],), rms_weight.float(), eps
    )
    probs = (keys @ res_weight.float()).softmax(dim=0)
    output = (probs.unsqueeze(-1) * values.float()).sum(dim=0)
    if out_norm_weight is not None:
        output = torch.nn.functional.rms_norm(
            output, (output.shape[-1],), out_norm_weight.float(), eps
        )
    return output.to(torch.bfloat16), updated_layer


@pytest.mark.parametrize(
    ("block_capacity", "num_blocks"),
    [(1, 1), (3, 3), (7, 7), (11, 11), (8, 3)],
)
@pytest.mark.parametrize("has_delta", [False, True])
@pytest.mark.parametrize("with_out_norm", [False, True])
def test_online_v2_matches_reference(
    block_capacity: int,
    num_blocks: int,
    has_delta: bool,
    with_out_norm: bool,
) -> None:
    torch.manual_seed(17)
    layer = torch.randn(1, 1, 7168, device="cuda", dtype=torch.bfloat16)
    original_layer = layer.clone()
    delta = torch.randn_like(layer) if has_delta else None
    blocks = torch.randn(
        block_capacity, 1, 1, 7168, device="cuda", dtype=torch.bfloat16
    )
    res_weight = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
    rms_weight = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
    out_norm_weight = (
        torch.randn(7168, device="cuda", dtype=torch.bfloat16)
        if with_out_norm
        else None
    )

    expected, expected_layer = _reference(
        layer,
        delta,
        blocks,
        res_weight,
        rms_weight,
        out_norm_weight,
        num_blocks,
        1e-5,
    )
    actual = attn_res_fwd_packed(
        layer,
        blocks,
        res_weight,
        rms_weight,
        1e-5,
        out_norm_weight,
        delta=delta,
        num_blocks=num_blocks,
    )

    torch.testing.assert_close(actual, expected, atol=0.5, rtol=2e-2)
    torch.testing.assert_close(layer, expected_layer, atol=0, rtol=0)
    if delta is None:
        torch.testing.assert_close(layer, original_layer, atol=0, rtol=0)


@pytest.mark.parametrize("with_out_norm", [False, True])
def test_online_v2_cuda_graph_capture(with_out_norm: bool) -> None:
    torch.manual_seed(23)
    layer = torch.randn(1, 1, 7168, device="cuda", dtype=torch.bfloat16)
    blocks = torch.randn(3, 1, 1, 7168, device="cuda", dtype=torch.bfloat16)
    res_weight = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
    rms_weight = torch.randn(7168, device="cuda", dtype=torch.bfloat16)
    out_norm_weight = (
        torch.randn(7168, device="cuda", dtype=torch.bfloat16)
        if with_out_norm
        else None
    )

    expected = attn_res_fwd_packed(
        layer, blocks, res_weight, rms_weight, 1e-5, out_norm_weight
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = attn_res_fwd_packed(
            layer, blocks, res_weight, rms_weight, 1e-5, out_norm_weight
        )
    graph.replay()

    torch.testing.assert_close(captured, expected, atol=0.5, rtol=2e-2)
