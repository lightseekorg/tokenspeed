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
from tokenspeed_kernel.ops.attention.mla._triton.kv_pack import (
    _mla_kv_pack_rows,
    triton_mla_kv_pack_quantize_fp8,
)
from tokenspeed_kernel.ops.embedding.triton import (
    _mla_nope_quantize_fp8_kernel,
    mla_nope_quantize_fp8_triton,
)
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_blackwell, reason="requires Blackwell"
)


def inputs(tokens, heads, width, broadcast, dtype):
    packed = torch.randn(tokens, heads, width + 64, device="cuda", dtype=dtype)
    key_heads = 1 if broadcast else heads
    kn = torch.randn(tokens, key_heads, width, device="cuda", dtype=dtype)
    kr = torch.randn(tokens, 1, 64, device="cuda", dtype=dtype)
    qo = torch.empty_like(packed, dtype=torch.float8_e4m3fn)
    ko = torch.empty(
        tokens, key_heads, width + 64, device="cuda", dtype=torch.float8_e4m3fn
    )
    args = dict(
        q_nope=packed[..., :width],
        q_rope=packed[..., width:],
        k_nope=kn,
        k_rope=kr,
        q_nope_out=qo[..., :width],
        q_rope_out=qo[..., width:],
        k_nope_out=ko[..., :width],
        k_rope_out=ko[..., width:],
    )
    return packed, kn, kr, qo, ko, args


@pytest.mark.parametrize("tokens", [0, 1, 3, 65, 257])
@pytest.mark.parametrize("heads", [1, 12, 96])
@pytest.mark.parametrize("width", [128, 512])
@pytest.mark.parametrize("broadcast", [False, True])
def test_grouped_nope_matches_reference(tokens, heads, width, broadcast):
    q, kn, kr, qo, ko, args = inputs(tokens, heads, width, broadcast, torch.bfloat16)
    mla_nope_quantize_fp8_triton(
        **args, quant_scale_q=0.7, quant_scale_kv=1.3, enable_pdl=False
    )
    expected_q = (q.float() * 0.7).to(qo.dtype)
    expected_k = (
        torch.cat((kn, kr.expand(tokens, kn.shape[1], 64)), -1)
        .float()
        .mul_(1.3)
        .to(ko.dtype)
    )
    torch.testing.assert_close(
        qo.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        ko.view(torch.uint8), expected_k.view(torch.uint8), rtol=0, atol=0
    )


@pytest.mark.parametrize("width", [128, 512])
@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("tokens", [37, 1025])
def test_grouped_nope_graph_tensor_scales(width, pdl, tokens):
    q, kn, kr, qo, ko, args = inputs(tokens, 12, width, True, torch.float16)
    sq = torch.tensor(0.5, device="cuda")
    sk = torch.tensor(1.7, device="cuda")

    def run():
        mla_nope_quantize_fp8_triton(
            **args, quant_scale_q=sq, quant_scale_kv=sk, enable_pdl=pdl
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    q.normal_()
    kn.normal_()
    kr.normal_()
    sq.fill_(0.75)
    sk.fill_(0.25)
    graph.replay()
    eq = (q.float() * sq).to(qo.dtype)
    ek = torch.cat((kn, kr), -1).float().mul_(sk).to(ko.dtype)
    torch.testing.assert_close(
        qo.view(torch.uint8), eq.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        ko.view(torch.uint8), ek.view(torch.uint8), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("pdl", [False, True])
def test_grouped_kv_pack_strided_output_graph(dtype, pdl):
    kv = torch.randn(65, 12, 256, device="cuda", dtype=torch.bfloat16)
    kr = torch.randn(65, 64, device="cuda", dtype=torch.bfloat16)
    key_storage = torch.full((65, 12, 208), 7, device="cuda", dtype=torch.float32).to(
        dtype
    )
    value_storage = torch.full((65, 12, 144), 7, device="cuda", dtype=torch.float32).to(
        dtype
    )
    ko, vo = key_storage[..., :192], value_storage[..., :128]

    def run():
        return triton_mla_kv_pack_quantize_fp8(
            kv[..., :128], kr, kv[..., 128:], 0.7, 1.3, ko, vo, dtype, pdl
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_k, actual_v = run()
    kv.normal_()
    kr.normal_()
    graph.replay()
    ek = (
        torch.cat((kv[..., :128], kr[:, None, :].expand(65, 12, 64)), -1)
        .float()
        .mul_(0.7)
        .to(dtype)
    )
    ev = (kv[..., 128:].float() * 1.3).to(dtype)
    torch.testing.assert_close(
        actual_k.view(torch.uint8), ek.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_v.view(torch.uint8), ev.view(torch.uint8), rtol=0, atol=0
    )
    assert bool((key_storage[..., 192:].float() == 7).all())
    assert bool((value_storage[..., 128:].float() == 7).all())


def test_grouped_mla_conversion_reuses_compilation():
    from utils import assert_no_triton_compile

    def run(tokens):
        q, kn, kr, qo, ko, args = inputs(tokens, 12, 128, False, torch.bfloat16)
        mla_nope_quantize_fp8_triton(**args, enable_pdl=False)
        triton_mla_kv_pack_quantize_fp8(kn, kr, kn, enable_pdl=False)

    run(16)
    run(1024)
    with assert_no_triton_compile(
        _mla_nope_quantize_fp8_kernel,
        _mla_kv_pack_rows,
    ):
        for tokens in (1, 2, 3, 17, 33, 65, 129, 257, 1025, 2049):
            run(tokens)
