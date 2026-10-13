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

"""The persistent cuBLASLt FP8 runner: same results and tactics, cached algorithms."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

platform = current_platform()

pytestmark = pytest.mark.skipif(
    not platform.is_blackwell, reason="FlashInfer cuBLASLt FP8 GEMM requires Blackwell"
)


def _operands(m: int, n: int, k: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(m, k, generator=g, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    w = (torch.randn(n, k, generator=g, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    scale_a = torch.full((), 0.75, device="cuda")
    scale_b = torch.full((), 1.25, device="cuda")
    return a, w.t(), scale_a, scale_b


@pytest.mark.parametrize("tuned", [False, True])
@pytest.mark.parametrize(
    "m,n,k",
    [(1, 4096, 4096), (37, 10240, 4096), (2048, 4096, 2048), (8000, 10240, 4096)],
)
def test_persistent_runner_matches_upstream_bmm_fp8(m, n, k, tuned):
    from flashinfer import bmm_fp8
    from flashinfer.autotuner import autotune
    from tokenspeed_kernel.ops.gemm.flashinfer import flashinfer_mm_fp8_tensor_scaled

    a, b, scale_a, scale_b = _operands(m, n, k, seed=m + n)
    if tuned:
        with autotune(True):
            flashinfer_mm_fp8_tensor_scaled(a, b, scale_a, scale_b, torch.bfloat16)
    ours = flashinfer_mm_fp8_tensor_scaled(a, b, scale_a, scale_b, torch.bfloat16)
    upstream = bmm_fp8(
        a.unsqueeze(0),
        b.unsqueeze(0),
        scale_a,
        scale_b,
        torch.bfloat16,
        backend="cublas",
    ).squeeze(0)
    assert torch.equal(ours, upstream)
    strided = torch.empty(m, n + 8, dtype=torch.bfloat16, device="cuda")[:, :n]
    flashinfer_mm_fp8_tensor_scaled(a, b, scale_a, scale_b, torch.bfloat16, out=strided)
    assert torch.equal(strided, upstream)


def test_wrapper_calls_reuse_one_runner_per_device(monkeypatch):
    from flashinfer.gemm import gemm_base
    from tokenspeed_kernel.ops.gemm.flashinfer import flashinfer_mm_fp8_tensor_scaled
    from tokenspeed_kernel.thirdparty.flashinfer import fp8_gemm

    used = []
    real = fp8_gemm._cublas_fp8_runner

    def spy(device_index):
        used.append(real(device_index))
        return used[-1]

    monkeypatch.setattr(fp8_gemm, "_cublas_fp8_runner", spy)
    a, b, scale_a, scale_b = _operands(300, 4096, 4096, seed=7)
    for _ in range(2):
        flashinfer_mm_fp8_tensor_scaled(a, b, scale_a, scale_b, torch.bfloat16)
    assert len(used) == 2 and used[0] is used[1]
    runner = used[0]
    assert real(0) is not real(1)
    upstream = gemm_base.get_gemm_module().cublas_fp8_gemm_runner()
    assert type(runner).__name__ == type(upstream).__name__
    out = torch.empty(1, 300, 4096, dtype=torch.bfloat16, device="cuda")
    workspace = gemm_base._get_cache_buf(
        "bmm_fp8_workspace", gemm_base.DEFAULT_WORKSPACE_SIZE, a.device
    )
    inputs = [a.unsqueeze(0), b.unsqueeze(0), scale_a, scale_b, out, workspace]
    assert runner._get_algos(inputs) is runner._get_algos(inputs)


def test_concurrent_streams_do_not_share_the_cublaslt_workspace(monkeypatch):
    from flashinfer.gemm import gemm_base
    from tokenspeed_kernel.thirdparty.flashinfer import fp8_gemm

    used = []
    real = gemm_base._get_cache_buf

    def spy(name, num_bytes, device):
        used.append(real(name, num_bytes, device))
        return used[-1]

    monkeypatch.setattr(gemm_base, "_get_cache_buf", spy)
    a, b, scale_a, scale_b = _operands(64, 4096, 4096, seed=11)
    out = torch.empty(1, 64, 4096, dtype=torch.bfloat16, device="cuda")
    operands = (a.unsqueeze(0), b.unsqueeze(0), scale_a, scale_b, out)
    side = torch.cuda.Stream()
    fp8_gemm.cublas_fp8_gemm(*operands)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fp8_gemm.cublas_fp8_gemm(*operands)
    torch.cuda.current_stream().wait_stream(side)
    fp8_gemm.cublas_fp8_gemm(*operands)
    main, other, again = (buffer.data_ptr() for buffer in used)
    assert main == again != other


def test_the_algorithm_cache_keeps_only_recent_shapes(monkeypatch):
    from tokenspeed_kernel.thirdparty.flashinfer import fp8_gemm

    monkeypatch.setattr(fp8_gemm, "_ALGO_CACHE_SHAPES", 2)
    cache = fp8_gemm._RecentShapes()
    cache["a"], cache["b"] = 1, 2
    assert cache.get("a") == 1
    cache["c"] = 3
    assert list(cache) == ["a", "c"]
    assert cache.get("b") is None
