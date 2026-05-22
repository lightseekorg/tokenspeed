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

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.quantization.triton import fp8_quantize

FP8_E4M3_MAX = 448.0


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_pure_cast_contig_bf16(device: str) -> None:
    torch.manual_seed(0)
    x = torch.randn(2048, 512, device=device, dtype=torch.bfloat16) * 50
    ref = x.to(torch.float8_e4m3fn)
    out = fp8_quantize(x)
    torch.cuda.synchronize()
    assert out.shape == ref.shape and out.dtype == ref.dtype
    assert _bitwise_equal(out, ref)


def test_pure_cast_strided_v_slice(device: str) -> None:
    """v = kv[..., qk_nope:] — the deepseek_v3.py call site shape."""
    torch.manual_seed(0)
    s, h, qk_nope, v_head = 4096, 16, 128, 128
    kv = torch.randn(s, h, qk_nope + v_head, device=device, dtype=torch.bfloat16) * 50
    v = kv[..., qk_nope:]
    assert not v.is_contiguous(), "v must be a strided slice for this test"
    ref = v.to(torch.float8_e4m3fn)
    out = fp8_quantize(v)
    torch.cuda.synchronize()
    assert _bitwise_equal(out, ref)


def test_pure_cast_1d(device: str) -> None:
    torch.manual_seed(0)
    x = torch.randn(1024, device=device, dtype=torch.bfloat16) * 50
    ref = x.to(torch.float8_e4m3fn)
    out = fp8_quantize(x)
    torch.cuda.synchronize()
    assert _bitwise_equal(out, ref)


def test_pure_cast_fp16(device: str) -> None:
    torch.manual_seed(0)
    x = torch.randn(1024, 512, device=device, dtype=torch.float16) * 50
    ref = x.to(torch.float8_e4m3fn)
    out = fp8_quantize(x)
    torch.cuda.synchronize()
    assert _bitwise_equal(out, ref)


def test_pure_cast_e5m2(device: str) -> None:
    torch.manual_seed(0)
    x = torch.randn(1024, 512, device=device, dtype=torch.bfloat16) * 50
    ref = x.to(torch.float8_e5m2)
    out = fp8_quantize(x, fp8_dtype=torch.float8_e5m2)
    torch.cuda.synchronize()
    assert out.dtype == torch.float8_e5m2
    assert _bitwise_equal(out, ref)


@pytest.mark.parametrize("scale_inv", [0.5, 2.0, 1.0 / 7.5])
def test_scaled_cast_matches_reference(device: str, scale_inv: float) -> None:
    """Scaled path: out = saturate((x * scale_inv) -> fp8). Compare bitwise to
    the same recipe applied in eager torch."""
    torch.manual_seed(0)
    x = torch.randn(2048, 512, device=device, dtype=torch.bfloat16) * 100
    ref = (
        (x.to(torch.float32) * scale_inv)
        .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
    )
    out = fp8_quantize(x, scale_inv=scale_inv)
    torch.cuda.synchronize()
    assert _bitwise_equal(out, ref)


def test_preallocated_output(device: str) -> None:
    torch.manual_seed(0)
    x = torch.randn(1024, 512, device=device, dtype=torch.bfloat16) * 50
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    ret = fp8_quantize(x, out=out)
    torch.cuda.synchronize()
    assert ret.data_ptr() == out.data_ptr()
    assert _bitwise_equal(out, x.to(torch.float8_e4m3fn))


def test_rejects_unsupported_dtype(device: str) -> None:
    x = torch.randn(64, 128, device=device, dtype=torch.float32)
    with pytest.raises(AssertionError):
        fp8_quantize(x)


def test_rejects_non_stride1_inner(device: str) -> None:
    """fp8_quantize requires stride(-1) == 1; reject otherwise."""
    x = torch.randn(64, 128, device=device, dtype=torch.bfloat16)
    transposed = x.t()  # stride(-1) = 128, stride(-2) = 1
    assert transposed.stride(-1) != 1
    with pytest.raises(AssertionError):
        fp8_quantize(transposed)
