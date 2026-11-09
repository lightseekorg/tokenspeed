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

"""Compare fused output quantization against materialized BF16 intermediates."""

import pytest
import torch
from tokenspeed_kernel.ops.activation.triton import sigmoid_mul
from tokenspeed_kernel.ops.layernorm import gated_rmsnorm
from tokenspeed_kernel.ops.quantization import output_fp8_quant_pack, quantize_fp8
from tokenspeed_kernel.platform import pdl_enabled

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires SM100/103",
)


def reference(x, z, w, *, head_dim, sigmoid_gate, pdl):
    if w is None:
        y = x.clone()
        sigmoid_mul(y, z)
    else:
        y = gated_rmsnorm(
            x,
            w,
            z,
            eps=1e-6,
            group_size=head_dim,
            norm_before_gate=True,
            sigmoid_gate=sigmoid_gate,
            fp8_scale=None,
            weights_independent=True,
        )
    return quantize_fp8(
        y,
        granularity="token_group",
        group_size=128,
        scale_encoding="packed_ue8m0",
        enable_pdl=pdl,
        solution="cute_dsl",
    )


@pytest.mark.parametrize(
    "rows,width,head",
    [
        (1, 128, 128),
        (5, 640, 128),
        (32, 6144, 128),
        (129, 6144, 128),
        (4096, 6144, 128),
        (4, 1024, 256),
    ],
)
@pytest.mark.parametrize("kind", ["sigmoid", "norm_sigmoid", "norm_silu"])
@pytest.mark.parametrize("pdl", [False, True])
def test_output_quant_graph(rows, width, head, kind, pdl):
    torch.manual_seed(43)
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    # Model QKV gates are interleaved; GDN gates are row-strided views.
    backing = torch.randn(
        rows, width // head, 2, head, device="cuda", dtype=torch.bfloat16
    )
    z = (
        backing[:, :, 1, :]
        if kind == "sigmoid"
        else torch.randn(rows, width * 2, device="cuda", dtype=torch.bfloat16)[
            :, width:
        ]
    )
    w = (
        None
        if kind == "sigmoid"
        else torch.randn(head, device="cuda", dtype=torch.bfloat16)
    )
    kwargs = dict(
        head_dim=head,
        eps=1e-6,
        sigmoid_gate=kind != "norm_silu",
        enable_pdl=pdl,
    )
    previous = pdl_enabled()
    pdl_enabled(pdl)
    try:
        output_fp8_quant_pack(x, z, w, **kwargs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            # Include the producer, so PDL waits must protect its stores.
            produced = x * 1.0
            actual, scales = output_fp8_quant_pack(produced, z, w, **kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        for magnitude in (0.0, 0.01, 1.0, 16.0):
            x.normal_().mul_(magnitude)
            z.normal_()
            graph.replay()
            expected, expected_scales = reference(
                x,
                z,
                w,
                head_dim=head,
                sigmoid_gate=kind != "norm_silu",
                pdl=pdl,
            )
            torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
            torch.testing.assert_close(
                actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
            )
            padded = (rows + 3) // 4 * 4
            complete = scales.as_strided((padded, scales.shape[1]), scales.stride())
            assert torch.count_nonzero(complete[rows:]) == 0
    finally:
        pdl_enabled(previous)


def test_output_quant_reuses_compilation():
    from tokenspeed_kernel.ops.quantization._triton.output_fp8 import (
        _output_fp8_quant_pack_kernel,
    )
    from utils import assert_no_triton_compile

    def run(rows):
        x = torch.randn(rows, 640, device="cuda", dtype=torch.bfloat16)
        return output_fp8_quant_pack(
            x,
            x,
            None,
            head_dim=128,
            eps=0.0,
            sigmoid_gate=True,
            enable_pdl=True,
        )

    for rows in (1, 4, 16, 17):
        run(rows)
    with assert_no_triton_compile(_output_fp8_quant_pack_kernel):
        for rows in (2, 3, 5, 8, 15, 31, 32, 129, 256):
            run(rows)


@pytest.mark.parametrize("normalize", [False, True])
def test_output_quant_large_row_stride(normalize):
    # Only three small rows are touched; the last starts beyond 2**31 elements.
    x = torch.empty_strided(
        (3, 128), ((1 << 30) + 128, 1), device="cuda", dtype=torch.bfloat16
    )
    x.copy_(torch.randn(3, 128, device="cuda", dtype=torch.bfloat16))
    w = torch.ones(128, device="cuda", dtype=torch.bfloat16) if normalize else None
    actual, scales = output_fp8_quant_pack(
        x,
        x,
        w,
        head_dim=128,
        eps=1e-6,
        sigmoid_gate=True,
        enable_pdl=True,
    )
    dense = x.contiguous()
    expected, expected_scales = reference(
        dense,
        dense,
        w,
        head_dim=128,
        sigmoid_gate=True,
        pdl=True,
    )
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
    )
