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


"""K3 fused skinny-add3 correctness and capture-safety contracts."""

import pytest
import torch
from tokenspeed_kernel.ops.gemm import kimi3
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cute_dsl import skinny_gemm


def _is_add3_arch() -> bool:
    # _SKINNY_ADD3_CONFIGS stores sm103-tuned TILE CONFIGS, not just a backend choice, so
    # it stays gated where it was swept -- mirrors _skinny_add3_arch_supported. Widening
    # it would run another architecture's tuning parameters unmeasured.
    return (
        current_platform().vendor == "nvidia"
        and torch.cuda.get_device_capability() >= (10, 3)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_add3_arch(),
    reason="add3 route is measured for sm103 only",
)
@pytest.mark.parametrize("m", [1, 2])
def test_skinny_add3_matches_reference(m):
    torch.manual_seed(0)
    n, k = 7168, 3584
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 8
    a = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    # Column slice of a wider tensor, as the serving call site passes it.
    c_wide = torch.randn(m, 2 * n, device="cuda", dtype=torch.bfloat16)
    c = c_wide[:, n:]
    got = kimi3._skinny_gemv_add3(x, w, a, c, None).float()
    ref = a.float() + x.float() @ w.float().t() + c.float()
    assert torch.allclose(got, ref, atol=0.5, rtol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_add3_arch(),
    reason="_SKINNY_ADD3_CONFIGS holds sm103-tuned tile configs; unswept below that",
)
def test_kimi3_add3_auto_selects_the_skinny_epilogue():
    torch.manual_seed(1)
    m, n, k = 1, 7168, 3584
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 8
    a = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    auto = kimi3.kimi3_latent_projection_add3(x, w, a, c).float()
    forced = kimi3.kimi3_latent_projection_add3(
        x, w, a, c, solution="skinny_add3"
    ).float()
    composed = kimi3.kimi3_latent_projection_add3(
        x, w, a, c, solution="composed"
    ).float()
    assert torch.allclose(auto, forced, atol=0.0, rtol=0.0)
    assert torch.allclose(auto, composed, atol=0.5, rtol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_add3_arch(),
    reason="add3 route is measured for sm103 only",
)
def test_skinny_add3_unwarmed_capture_falls_back(monkeypatch):
    torch.manual_seed(2)
    m, n, k = 1, 7168, 3584
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 8
    a = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    dev = x.device.index or 0
    with kimi3._skinny_add3_warmed_lock:
        kimi3._skinny_add3_warmed.discard((dev, m, n, k))

    def _no_jit(*args, **kwargs):
        raise AssertionError("JIT compile attempted inside capture")

    monkeypatch.setattr(skinny_gemm.shape_dynamic_skinny_gemm, "_compile", _no_jit)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        ref = a.float() + x.float() @ w.float().t() + c.float()
        with torch.cuda.graph(g):
            out = kimi3._skinny_gemv_add3(x, w, a, c, None)
        g.replay()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    assert torch.allclose(out.float(), ref, atol=0.5, rtol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_add3_arch(),
    reason="fused add3 requires its measured architecture",
)
@pytest.mark.parametrize("m", [1, 2])
def test_skinny_add3_changed_input_replay_and_out(m):
    torch.manual_seed(31 + m)
    n, k = 7168, 3584
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / 8
    prefix = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    shared = torch.randn(m, n * 2, dtype=torch.bfloat16, device="cuda")[:, n:]
    storage = torch.full((m * n + 32,), 42, dtype=torch.bfloat16, device="cuda")
    out = storage[: m * n].view(m, n)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        assert kimi3._skinny_gemv_add3(x, weight, prefix, shared, out) is out
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        kimi3._skinny_gemv_add3(x, weight, prefix, shared, out)
    x.mul_(0.75)
    prefix.add_(0.25)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    expected = prefix.float() + x.float() @ weight.float().T + shared.float()
    torch.testing.assert_close(out.float(), expected, atol=0.5, rtol=2e-2)
    assert torch.all(storage[m * n :] == 42)
