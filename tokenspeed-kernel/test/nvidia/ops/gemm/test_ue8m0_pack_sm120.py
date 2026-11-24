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
from tokenspeed_kernel.ops.gemm._triton.ue8m0_pack import _pack_ue8m0, pack_ue8m0
from tokenspeed_kernel.platform import ArchVersion, Platform
from tokenspeed_kernel.thirdparty.deepgemm import mxfp8_mm_sm120

pytestmark = pytest.mark.skipif(
    Platform.get().arch_version != ArchVersion(12, 0), reason="SM120 only"
)


def reference(s):
    dg = pytest.importorskip("deep_gemm_sm120")
    return dg.transform_sf_into_required_layout(
        torch.exp2(s.float() - 127), s.shape[0], s.shape[1] * 32, (1, 32)
    )


@pytest.mark.parametrize("m", [1, 4, 6, 24, 32, 65, 128])
@pytest.mark.parametrize("c", [1, 4, 7, 40, 128, 160, 256])
def test_bytes_and_layout(m, c):
    s = (torch.arange(m * c, device="cuda").reshape(m, c) % 256).to(torch.uint8)
    a, b = pack_ue8m0(s), reference(s)
    assert a.shape == b.shape and a.stride() == b.stride()
    assert torch.equal(a, b)


@pytest.mark.parametrize("stride", [False, True])
def test_graph_replay(stride):
    s = torch.randint(0, 256, (24, 256), device="cuda", dtype=torch.uint8)
    if stride:
        s = s[::2, ::2]
    pack_ue8m0(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        a = pack_ue8m0(s)
    for _ in range(3):
        s.random_(0, 256)
        graph.replay()
        assert torch.equal(a, reference(s))


@pytest.mark.parametrize(
    "m,n,k", [(6, 1792, 5120), (24, 5120, 8192), (24, 32768, 1280)]
)
def test_gemm_exact(monkeypatch, m, n, k):
    a = (torch.randn(m, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    sa = torch.randint(120, 130, (m, k // 32), device="cuda", dtype=torch.uint8)
    sb = reference(torch.full((n, k // 32), 127, device="cuda", dtype=torch.uint8))
    x = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)
    monkeypatch.setenv("TOKENSPEED_EXPERIMENT_SM120_FUSED_SCALE_PACK", "0")
    mxfp8_mm_sm120(a, b, sa, sb, x)
    monkeypatch.setenv("TOKENSPEED_EXPERIMENT_SM120_FUSED_SCALE_PACK", "1")
    mxfp8_mm_sm120(a, b, sa, sb, y)
    assert torch.equal(x, y)


def test_empty():
    assert pack_ue8m0(
        torch.empty((0, 160), device="cuda", dtype=torch.uint8)
    ).shape == (0, 40)


def test_large_offset():
    stride = 2**31 + 16
    s = torch.empty_strided((2, 4), (stride, 1), device="cuda", dtype=torch.uint8)
    s.fill_(127)
    assert torch.equal(pack_ue8m0(s), reference(s))


def test_dynamic_sizes_no_compile():
    from utils import assert_no_triton_compile

    for m, c in [(4, 4), (6, 7), (24, 160)]:
        pack_ue8m0(torch.full((m, c), 127, device="cuda", dtype=torch.uint8))
    with assert_no_triton_compile(_pack_ue8m0):
        for m, c in [(8, 12), (10, 11), (32, 128)]:
            s = torch.full((m, c), 127, device="cuda", dtype=torch.uint8)
            assert torch.equal(pack_ue8m0(s), reference(s))


def test_all_bytes_in_each_lane():
    for lane in range(4):
        s = torch.full((256, 4), 127, device="cuda", dtype=torch.uint8)
        s[:, lane] = torch.arange(256, device="cuda").to(torch.uint8)
        assert torch.equal(pack_ue8m0(s), reference(s))


def test_transposed_scales_gemm_fallback(monkeypatch):
    m, n, k = 6, 1792, 5120
    a = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.full((k // 32, m), 127, device="cuda", dtype=torch.uint8).t()
    sb = reference(torch.full((n, k // 32), 127, device="cuda", dtype=torch.uint8))
    x = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)
    monkeypatch.setenv("TOKENSPEED_EXPERIMENT_SM120_FUSED_SCALE_PACK", "0")
    mxfp8_mm_sm120(a, b, sa, sb, x)
    monkeypatch.setenv("TOKENSPEED_EXPERIMENT_SM120_FUSED_SCALE_PACK", "1")
    mxfp8_mm_sm120(a, b, sa, sb, y)
    assert torch.equal(x, y)
