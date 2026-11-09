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

"""Dynamic FP8 input fusion preserves BF16, scales, graph and PDL semantics."""

import pytest
import torch
from tokenspeed_kernel.ops.layernorm.triton import add_rmsnorm
from tokenspeed_kernel.ops.quantization import (
    add_rmsnorm_quantize_fp8_block,
    quantize_fp8,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires SM100/103",
)


def reference(x, residual, weight, *, scale_encoding):
    x, residual = x.clone(), residual.clone()
    add_rmsnorm(
        x,
        residual,
        weight,
        1e-6,
        x2=None,
        out=x,
        out_fp8=None,
        fp8_scale=None,
        out_fp4=None,
        fp4_scale=None,
        gemma=True,
    )
    codes, scales = quantize_fp8(
        x,
        granularity="token_group",
        group_size=128,
        scale_encoding=scale_encoding,
    )
    return x, codes, scales, residual


@pytest.mark.parametrize("encoding", ["float32", "packed_ue8m0"])
@pytest.mark.parametrize(
    "rows,width", [(0, 640), (1, 640), (4, 2560), (32, 5120), (129, 6144), (4096, 2560)]
)
@pytest.mark.parametrize("pdl", [False, True])
def test_input_quant_graph(encoding, rows, width, pdl):
    torch.manual_seed(25)
    source = torch.randn(rows, width * 2, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(source)
    # Match strided residual/captured input views used by projection callers.
    x, r = source[:, :width], residual[:, :width]
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16) * 0.3
    kwargs = dict(eps=1e-6, scale_encoding=encoding, enable_pdl=pdl)
    add_rmsnorm_quantize_fp8_block(x.clone(), r.clone(), weight, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Include producers so PDL waits must protect their stores.
        actual = add_rmsnorm_quantize_fp8_block(x * 1.0, r * 1.0, weight, **kwargs)
    for magnitude in (0.0, 1e-15, 0.01, 1.0, 32.0):
        x.normal_().mul_(magnitude)
        r.normal_().mul_(magnitude * 3)
        graph.replay()
        expected = reference(x, r, weight, scale_encoding=encoding)
        for index in (0, 3):
            torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)
        torch.testing.assert_close(
            actual[1].view(torch.uint8), expected[1].view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
        if encoding == "packed_ue8m0" and rows:
            padded = (rows + 3) // 4 * 4
            storage = actual[2].as_strided(
                (padded, actual[2].shape[1]), actual[2].stride()
            )
            assert torch.count_nonzero(storage[rows:]) == 0


def test_input_quant_large_stride():
    width = 128
    stride = 2**30 + 128
    x = torch.empty_strided(
        (3, width), (stride, 1), device="cuda", dtype=torch.bfloat16
    )
    r = torch.empty_strided(
        (3, width), (stride, 1), device="cuda", dtype=torch.bfloat16
    )
    x.normal_()
    r.normal_()
    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)
    expected = reference(x, r, weight, scale_encoding="float32")
    actual = add_rmsnorm_quantize_fp8_block(
        x,
        r,
        weight,
        eps=1e-6,
        scale_encoding="float32",
        enable_pdl=True,
    )
    for i in (0, 2, 3):
        torch.testing.assert_close(actual[i], expected[i], rtol=0, atol=0)
    torch.testing.assert_close(
        actual[1].view(torch.uint8), expected[1].view(torch.uint8), rtol=0, atol=0
    )


def test_input_quant_reuses_compilation():
    from tokenspeed_kernel.ops.quantization._triton.input_fp8 import (
        _add_rmsnorm_fp8_block_kernel,
    )
    from utils import assert_no_triton_compile

    def run(rows):
        x = torch.randn(rows, 640, device="cuda", dtype=torch.bfloat16)
        return add_rmsnorm_quantize_fp8_block(
            x,
            x.clone(),
            torch.ones(640, device="cuda", dtype=torch.bfloat16),
            eps=1e-6,
            scale_encoding="packed_ue8m0",
            enable_pdl=True,
        )

    for rows in (1, 4, 16, 17):
        run(rows)
    with assert_no_triton_compile(_add_rmsnorm_fp8_block_kernel):
        for rows in (2, 3, 5, 8, 15, 31, 32, 129, 256):
            run(rows)
