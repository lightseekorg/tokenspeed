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

"""Startup tuning and numerical contracts for SM107 attention output projection."""

import pytest
import torch
from tokenspeed_kernel import compile_monitor
from tokenspeed_kernel.ops.gemm import flashinfer as fi
from tokenspeed_kernel.ops.gemm import mm
from tokenspeed_kernel.ops.tuning import (
    autotune,
    load_autotune_cache,
    save_autotune_cache,
)
from tokenspeed_kernel.platform import pdl_enabled

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or fi.platform.arch_version != fi.ArchVersion(10, 7)
    or not fi.has_flashinfer_cute_dsl_bf16(),
    reason="SM107 with FlashInfer CuTe DSL required",
)

N, K = 2560, 6144
AutoTuner = pytest.importorskip("flashinfer.autotuner").AutoTuner


def inputs(m, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, K, device="cuda", generator=generator).bfloat16()
    w = (torch.randn(N, K, device="cuda", generator=generator) / K**0.5).bfloat16()
    return x, w


def check_result(got, x, w):
    reference = x.double() @ w.double().T
    error = (got.double() - reference).norm() / reference.norm().clamp_min(1e-12)
    assert error < 0.002
    torch.testing.assert_close(got.double(), reference, rtol=0.008, atol=3e-5)


@pytest.fixture(scope="module", autouse=True)
def tuned(tmp_path_factory):
    if not torch.cuda.is_available() or fi.platform.arch_version != fi.ArchVersion(
        10, 7
    ):
        return
    x, w = inputs(128, 19)
    tuner = AutoTuner.get()
    tuner.clear_cache()
    with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
        # Discovery on prefill must expose small decode profiles too.
        result = mm(x, w)
        assert torch.equal(result, x @ w.T)
    path = str(tmp_path_factory.mktemp("out-projection") / "tactics.json")
    assert save_autotune_cache(path, None, 0)
    tuner.clear_cache()
    assert load_autotune_cache(path, None, 0)


@pytest.mark.parametrize("m", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", [7, 23, 41])
def test_tuned_projection_and_changed_input_graph(m, seed):
    x, w = inputs(m, seed)
    storage = torch.full((m * N + 32,), 42, device="cuda", dtype=torch.bfloat16)
    out = storage[: m * N].view(m, N)
    assert fi.flashinfer_joint_bf16_supported(x, w, out)
    assert mm(x, w, out=out) is out
    check_result(out, x, w)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mm(x, w, out=out)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        mm(x, w, out=out)
    for scale in [0.125, -8.0, 0.75]:
        x.mul_(scale)
        expected = mm(x, w).clone()
        for _ in range(5):
            out.fill_(float("nan"))
            graph.replay()
            assert torch.equal(out, expected)
        check_result(out, x, w)
        assert torch.all(storage[m * N :] == 42)


@pytest.mark.parametrize("m", [1, 3, 4])
def test_pdl_and_cancellation(m):
    x, w = inputs(m, 91)
    x[:, K // 2 :] = -x[:, : K // 2]
    w[:, K // 2 :] = w[:, : K // 2]
    previous = pdl_enabled()
    try:
        for enabled in [False, True]:
            pdl_enabled(enabled)
            out = mm(x, w)
            torch.testing.assert_close(
                out.float(), torch.zeros_like(out.float()), rtol=0, atol=3e-5
            )
    finally:
        pdl_enabled(previous)


@pytest.mark.parametrize("m", [5, 17, 32, 48, 128, 192])
def test_larger_rows_keep_original_gemm(m):
    x, w = inputs(m, 99)
    out = torch.empty(m, N, dtype=x.dtype, device=x.device)
    with pytest.raises(ValueError, match="Unsupported input"):
        fi.flashinfer_bf16_gemm(x, w, out)
    assert torch.equal(mm(x, w, out=out), x @ w.T)


def test_other_projections_and_invalid_layouts_keep_original_gemm():
    x, w = inputs(4, 10)
    for a, b in [
        (x[:, :2560].contiguous(), w[:, :2560].contiguous()),
        (x, w[:512]),
        (x[:, ::2], w[:, ::2]),
    ]:
        assert not fi.flashinfer_joint_bf16_supported(a, b, None)
        assert torch.equal(mm(a, b), a @ b.T)
    bias = torch.randn(N, device=x.device, dtype=x.dtype)
    assert torch.equal(mm(x, w, bias=bias), torch.nn.functional.linear(x, w, bias))


def test_serving_new_rows_uses_original_gemm(monkeypatch):
    # Eager shapes that were not captured must not trigger per-M CuTe compilation.
    x, w = inputs(32, 103)
    from flashinfer.gemm.kernels import dense_bf16_gemm_direct as direct
    from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as cluster
    from flashinfer.gemm.kernels import dense_bf16_gemm_warp_splitk as warp

    compilers = [
        direct._get_compiled_direct_kernel,
        cluster._get_compiled_splitk_kernel,
        warp._compile,
    ]
    misses = [compiler.cache_info().misses for compiler in compilers]
    monkeypatch.setattr(compile_monitor, "_serving", True)
    for m in [3, 5, 19, 31]:
        assert torch.equal(mm(x[:m], w), x[:m] @ w.T)
    assert [compiler.cache_info().misses for compiler in compilers] == misses


def test_singleton_strides_and_alignment():
    x, w = inputs(1, 211)
    padded_x = torch.zeros((1, K + 64), dtype=x.dtype, device=x.device)
    padded_x[:, 32:-32].copy_(x)
    storage = torch.full((1, N + 64), 42, dtype=x.dtype, device=x.device)
    out = storage[:, 32:-32]
    assert fi.flashinfer_joint_bf16_supported(padded_x[:, 32:-32], w, out)
    assert mm(padded_x[:, 32:-32], w, out=out) is out
    check_result(out, x, w)
    assert torch.all(storage[:, :32] == 42)
    assert torch.all(storage[:, -32:] == 42)
    unaligned = torch.empty(K + 1, device=x.device, dtype=x.dtype)[1:].view_as(x)
    unaligned.copy_(x)
    assert not fi.flashinfer_joint_bf16_supported(unaligned, w, None)
    # The baseline can choose a different reduction for an unaligned pointer.
    # Compare the fallback using the same storage, not its aligned copy.
    actual = mm(unaligned, w)
    assert torch.equal(actual, unaligned @ w.T)
    check_result(actual, unaligned, w)
