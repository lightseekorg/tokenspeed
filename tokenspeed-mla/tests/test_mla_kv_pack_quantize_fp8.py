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

"""K/V pack-quantize tests; the far-offset case only touches rows past 2**31."""

import pytest
import torch
from tokenspeed_mla.mla_kv_pack_quantize_fp8 import mla_kv_pack_quantize_fp8


def _reference(k_nope, k_pe, v, fp8_dtype):
    s, h, _ = k_nope.shape
    pe = k_pe.reshape(s, -1).float()[:, None, :].expand(s, h, k_pe.shape[-1])
    k = torch.cat([k_nope.float(), pe], dim=-1).to(fp8_dtype)
    return k, v.float().to(fp8_dtype)


def _far_row_stride(rows, row_elems):
    """Row stride putting the last of ``rows`` rows past 2**31 elements."""
    stride = (2**31 // (rows - 1) // row_elems + 1) * row_elems
    assert (rows - 1) * stride > 2**31
    return stride


class TestGPU:
    @pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_matches_torch(self, fp8_dtype):
        torch.manual_seed(0)
        s, h = 37, 5
        k_nope = torch.randn(s, h, 512, dtype=torch.bfloat16, device="cuda")
        k_pe = torch.randn(s, 1, 64, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(s, h, 512, dtype=torch.bfloat16, device="cuda")
        k_out, v_out = mla_kv_pack_quantize_fp8(k_nope, k_pe, v, fp8_dtype=fp8_dtype)
        k_ref, v_ref = _reference(k_nope, k_pe, v, fp8_dtype)
        torch.cuda.synchronize()
        assert torch.equal(k_out.float(), k_ref.float())
        assert torch.equal(v_out.float(), v_ref.float())

    def test_row_offsets_past_int32(self):
        """Rows past 2**31 elements quantize correctly on both loads and stores."""
        torch.manual_seed(1)
        s, h = 8, 128
        fp8_dtype = torch.float8_e4m3fn
        specs = [
            ((s, h, 512), torch.bfloat16),  # k_nope
            ((s, h, 512), torch.bfloat16),  # v
            ((s, h, 576), fp8_dtype),  # k_out
            ((s, h, 512), fp8_dtype),  # v_out
        ]
        strides = [_far_row_stride(s, s_ * d) for (_, s_, d), _ in specs]
        needed = sum(
            ((s - 1) * stride + s_ * d) * dtype.itemsize
            for stride, ((_, s_, d), dtype) in zip(strides, specs)
        )
        free, _ = torch.cuda.mem_get_info()
        if free < needed * 1.1:
            pytest.skip(f"needs {needed >> 30} GiB free GPU memory")

        views = []
        for (shape, dtype), stride in zip(specs, strides):
            s_, h_, d = shape
            storage = torch.empty((s - 1) * stride + h_ * d, dtype=dtype, device="cuda")
            views.append(storage.as_strided(shape, (stride, d, 1)))
        k_nope, v, k_out, v_out = views
        k_nope.normal_()
        v.normal_()
        k_pe = torch.randn(s, 64, dtype=torch.bfloat16, device="cuda")

        k_res, v_res = mla_kv_pack_quantize_fp8(
            k_nope, k_pe, v, k_out=k_out, v_out=v_out
        )
        k_ref, v_ref = _reference(k_nope, k_pe, v, fp8_dtype)
        torch.cuda.synchronize()
        assert torch.equal(k_res.float(), k_ref.float())
        assert torch.equal(v_res.float(), v_ref.float())
