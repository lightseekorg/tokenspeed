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

"""Ascend MLA prefill accepts split Q/K/V views like the model produces."""

import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.mla import mla_prefill

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="Ascend MLA tests require an NPU"
)


def _inputs(dtype):
    torch.npu.set_device(0)
    torch.manual_seed(211)
    tokens, heads, kv_heads, qk_dim, v_dim = 8, 4, 2, 192, 128
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32, device="npu")
    q_base = torch.randn(tokens, heads, 2 * qk_dim, dtype=dtype, device="npu")
    k_base = torch.randn(tokens, kv_heads, 2 * qk_dim, dtype=dtype, device="npu")
    v_base = torch.randn(tokens, kv_heads, 2 * v_dim, dtype=dtype, device="npu")
    q, k, v = q_base[..., :qk_dim], k_base[..., :qk_dim], v_base[..., :v_dim]
    assert not q.is_contiguous() and not k.is_contiguous() and not v.is_contiguous()
    return q, k, v, cu_seqlens, 1.0 / math.sqrt(qk_dim)


def _run(kernel, q, k, v, cu_seqlens, scale):
    return kernel(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        5,
        5,
        scale,
        None,
        True,
        0.0,
        True,
        None,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("public", [False, True])
@torch.inference_mode()
def test_mla_prefill_accepts_non_contiguous_qkv(dtype, public):
    q, k, v, cu_seqlens, scale = _inputs(dtype)
    if public:
        from tokenspeed_kernel.ops.attention.mla import mla_prefill as kernel
    else:
        kernel = mla_prefill

    expected, expected_lse = _run(
        kernel, q.contiguous(), k.contiguous(), v.contiguous(), cu_seqlens, scale
    )
    actual, actual_lse = _run(kernel, q, k, v, cu_seqlens, scale)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual_lse, expected_lse, atol=0, rtol=0)
