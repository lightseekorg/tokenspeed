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
from tokenspeed_kernel.ops.quantization import (
    plan_rmsnorm_quantize_fp8_group32,
    rmsnorm_quantize_fp8_group32,
)
from tokenspeed_kernel.ops.quantization.triton import (
    _rmsnorm_fp8_group32_ue8m0,
    triton_rmsnorm_fp8_group32_ue8m0,
)
from tokenspeed_kernel.ops.tuning import kernel_warmup
from utils import assert_no_triton_compile


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_packed_scale_word_offsets_beyond_int32():
    # Exercise one row with widely spaced scale words, without computing the
    # millions of rows that normally give the scale plane this stride.
    stride = (1 << 30) + 4
    prefix = 1 << 31
    size = prefix + 2 * stride + 1
    if torch.cuda.mem_get_info()[0] < size * 4 + (1 << 30):
        pytest.skip("requires 17 GiB of free GPU memory")
    storage = torch.empty(size, dtype=torch.int32, device="cuda")
    # Keep a wrapped negative offset inside the allocation, so a regression
    # fails the value check without poisoning the test process's GPU context.
    scales = storage[prefix:].as_strided((1, 3), (1, stride))
    scales.fill_(-1)
    x = torch.ones(1, 384, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(384, dtype=torch.bfloat16, device="cuda")
    expected = triton_rmsnorm_fp8_group32_ue8m0(
        x, weight, 1e-6, scale_layout="packed", return_normalized=True
    )
    normalized, codes = torch.empty_like(x), torch.empty_like(expected[1])
    _rmsnorm_fp8_group32_ue8m0[(1,)](
        x,
        weight,
        normalized,
        codes,
        scales,
        1,
        384,
        stride,
        384,
        1e-6,
        512,
        "packed",
        16,
        True,
        False,
        num_warps=4,
    )
    torch.testing.assert_close(scales, expected[2], rtol=0, atol=0)
    torch.testing.assert_close(normalized, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(
        codes.view(torch.uint8), expected[1].view(torch.uint8), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("layout", ["canonical", "swizzled", "packed"])
@pytest.mark.parametrize("rows", [0, 1, 17, 128, 129])
def test_rmsnorm_fp8_group32_skip_normalized_matches_and_omits_tensor(layout, rows):
    torch.manual_seed(7)
    x = torch.randn(rows, 256, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(256, dtype=torch.bfloat16, device="cuda")
    plan = plan_rmsnorm_quantize_fp8_group32(x.dtype, solution="triton")
    ref_norm, ref_q, ref_s = rmsnorm_quantize_fp8_group32(
        x,
        weight,
        1e-6,
        scale_layout=layout,
        return_normalized=True,
        override=None,
        solution="triton",
        plan=plan,
    )
    assert ref_norm is not None
    got_norm, got_q, got_s = rmsnorm_quantize_fp8_group32(
        x,
        weight,
        1e-6,
        scale_layout=layout,
        return_normalized=False,
        override=None,
        solution="triton",
        plan=plan,
    )
    assert got_norm is None
    assert torch.equal(got_q.view(torch.uint8), ref_q.view(torch.uint8))
    if rows == 0 and layout == "swizzled":
        assert got_s.shape == ref_s.shape
    else:
        assert torch.equal(got_s, ref_s)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_rmsnorm_fp8_group32_warmup_compiles_both_normalized_variants():
    torch.cuda.synchronize()
    _rmsnorm_fp8_group32_ue8m0.device_caches.clear()
    x = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(256, dtype=torch.bfloat16, device="cuda")
    with kernel_warmup(enabled=True):
        norm, _, _ = triton_rmsnorm_fp8_group32_ue8m0(
            x[:16], weight, 1e-6, scale_layout="packed", return_normalized=False
        )
        assert norm is None
    with assert_no_triton_compile(_rmsnorm_fp8_group32_ue8m0):
        for rows in (1, 17, 64):
            for return_normalized in (False, True):
                out_norm, _, _ = triton_rmsnorm_fp8_group32_ue8m0(
                    x[:rows],
                    weight,
                    1e-6,
                    scale_layout="packed",
                    return_normalized=return_normalized,
                )
                assert (out_norm is None) != return_normalized
