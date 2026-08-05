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

"""Tests for the CuTe DSL argmax kernel wrapper.

These exercise:
  * Parity with ``torch.argmax`` across vocab sizes used by real LLMs and
    dtypes (fp32 / bf16 / fp16). The kernel upcasts at load and reduces in
    Float32, so low-precision inputs match torch bit-for-bit.
  * Correct routing through the torch fallback for unsupported shapes
    (small N, unaligned N, 1D, CPU).
  * CUDA-graph capture/replay — the kernel must stay in-place under graph
    capture, since the sampling backends run under captured graphs.
  * Pure-torch fallback path on non-NVIDIA hosts (AMD ROCm / CPU-only /
    sm_80 / sm_120+ / missing nvidia-cutlass-dsl).
"""

from __future__ import annotations

import pytest
import torch

cute_dsl = pytest.importorskip("tokenspeed_kernel.ops.sampling.cute_dsl")
cute_argmax = cute_dsl.argmax
cute_argmax_pair = cute_dsl.argmax_pair

requires_nvidia = pytest.mark.skipif(
    not cute_dsl.current_platform().is_nvidia,
    reason="CuTe DSL argmax kernel is NVIDIA-only",
)


# Vocab sizes for the models tokenspeed actively serves — same list as
# ``tmp/integrate_cutedsl_argmax/bench_argmax.py::DEFAULT_N``.
MODEL_VOCABS = {
    "deepseek_v4": 129280,
    "qwen3_5": 151936,
    "kimi_k2_5": 163840,
    "minimax_m2": 200064,
    "gpt_oss": 201088,
}


def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the CuTe argmax kernel")


# ---------------------------------------------------------------------------
# Correctness — kernel path (N >= 256, N % 32 == 0, fp32, supported SM)
# ---------------------------------------------------------------------------


# Speculative-decoding-style small batches × all 5 real model vocabs. Mirrors
# ``bench_argmax.py::DEFAULT_M`` so the kernel coverage tested here matches the
# one we benchmark.
_KERNEL_M_VALUES = [1, 4, 16, 64, 128]
_KERNEL_SHAPES = [(m, n) for n in MODEL_VOCABS.values() for m in _KERNEL_M_VALUES] + [
    # Asymmetric M values to catch row-tail edge cases the powers-of-two miss.
    (37, MODEL_VOCABS["deepseek_v4"]),
    (199, MODEL_VOCABS["qwen3_5"]),
]


@pytest.mark.parametrize("M,N", _KERNEL_SHAPES)
def test_argmax_matches_torch_for_kernel_shapes(M, N):
    """Parity with ``torch.argmax`` across all served model vocabs."""
    _need_cuda()
    torch.manual_seed(M * 13 + N)
    x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch.float32)
    out = cute_argmax(x)
    ref = torch.argmax(x, dim=-1)
    assert out.dtype == torch.int64
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.mark.parametrize(
    "M,N",
    [
        (1, MODEL_VOCABS["deepseek_v4"]),
        (16, MODEL_VOCABS["qwen3_5"]),
        (64, MODEL_VOCABS["kimi_k2_5"]),
        (4, MODEL_VOCABS["minimax_m2"]),
        (32, MODEL_VOCABS["gpt_oss"]),
    ],
)
def test_argmax_pair_matches_torch_max(M, N):
    _need_cuda()
    torch.manual_seed(M ^ N)
    x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch.float32)
    pair = cute_argmax_pair(x)
    ref_max, ref_idx = torch.max(x, dim=-1, keepdim=True)
    assert pair.shape == (M, 2)
    assert pair.dtype == torch.float32
    torch.testing.assert_close(pair[:, 0:1], ref_max, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(pair[:, 1:2].long(), ref_idx, atol=0, rtol=0)


def test_argmax_pair_writes_into_caller_buffer():
    """argmax_pair must populate the caller-provided ``out`` (CUDA graph hot path)."""
    _need_cuda()
    M, N = 4, MODEL_VOCABS["qwen3_5"]
    x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch.float32)
    out = torch.empty((M, 2), dtype=torch.float32, device="cuda")
    returned = cute_argmax_pair(x, out=out)
    assert returned.data_ptr() == out.data_ptr()
    ref_max, ref_idx = torch.max(x, dim=-1, keepdim=True)
    torch.testing.assert_close(out[:, 0:1], ref_max, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(out[:, 1:2].long(), ref_idx, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Fallback paths — these must still return correct indices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("N", [8, 32, 64, 128, 256, 1024, 2048, 3072])
def test_argmax_falls_back_for_small_N(N):
    """N ≤ 3072 (incl. all multiples) hangs the kernel on B200 (see
    ``_MIN_VOCAB_SIZE = 4096`` in cute_dsl.py). The wrapper must route every
    such N through ``torch.argmax``."""
    _need_cuda()
    M = 4
    torch.manual_seed(N)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(cute_argmax(x), torch.argmax(x, dim=-1), atol=0, rtol=0)


@pytest.mark.parametrize("N", [257, 1023, 1025, 32001])
def test_argmax_falls_back_for_unaligned_N(N):
    _need_cuda()
    M = 4
    torch.manual_seed(N)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    torch.testing.assert_close(cute_argmax(x), torch.argmax(x, dim=-1), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_argmax_matches_torch_for_low_precision_dtypes(dtype):
    _need_cuda()
    x = torch.randn(8, 4096, device="cuda", dtype=dtype)
    torch.testing.assert_close(cute_argmax(x), torch.argmax(x, dim=-1), atol=0, rtol=0)


def test_argmax_empty_batch_falls_back_to_torch():
    """M=0 (empty batch) must fall back to torch.argmax instead of launching
    the CuTe kernel with zero grid blocks, which raises
    ``CUDA_ERROR_INVALID_VALUE``.

    This path is hit during ``execute_idle_forward()`` in DP/EP configs
    where an idle rank runs the drafter model with ``bs=0``.
    """
    _need_cuda()
    N = MODEL_VOCABS["qwen3_5"]
    x = torch.empty((0, N), device="cuda", dtype=torch.float32)
    out = cute_argmax(x)
    assert out.shape == (0,)
    assert out.dtype == torch.int64
    ref = torch.argmax(x, dim=-1)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)

    # With caller-provided out buffer.
    out_buf = torch.empty(0, dtype=torch.int32, device="cuda")
    ret = cute_argmax(x, out=out_buf)
    assert ret.data_ptr() == out_buf.data_ptr()

    # argmax_pair must also handle M=0.
    pair = cute_argmax_pair(x)
    assert pair.shape == (0, 2)
    assert pair.dtype == torch.float32


def test_argmax_falls_back_for_1d_input():
    _need_cuda()
    x = torch.randn(4096, device="cuda", dtype=torch.float32)
    out = cute_argmax(x)
    assert out.shape == ()
    assert out.item() == torch.argmax(x, dim=-1).item()


def test_argmax_falls_back_on_cpu():
    x = torch.randn(4, 4096, dtype=torch.float32)
    out = cute_argmax(x)
    torch.testing.assert_close(out, torch.argmax(x, dim=-1), atol=0, rtol=0)


def test_argmax_falls_back_when_cute_unavailable(monkeypatch):
    """Simulate AMD ROCm / CPU-only / missing-cutlass platforms.

    Forces the kernel availability flag off and verifies that both ``argmax``
    and ``argmax_pair`` still return correct results via ``torch`` ops.
    """
    monkeypatch.setattr(cute_dsl, "_CUTE_AVAILABLE", False)
    assert not cute_dsl.is_available()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # argmax: should match torch.argmax across kernel-eligible and ineligible shapes.
    for shape in [
        (4, MODEL_VOCABS["deepseek_v4"]),
        (1, MODEL_VOCABS["gpt_oss"]),
        (8, 257),
    ]:
        x = torch.randn(*shape, device=device, dtype=torch.float32)
        torch.testing.assert_close(
            cute_argmax(x), torch.argmax(x, dim=-1), atol=0, rtol=0
        )

    # argmax_pair: should still pack (max, idx) via torch fallback when CUDA.
    if torch.cuda.is_available():
        x = torch.randn(8, MODEL_VOCABS["qwen3_5"], device="cuda", dtype=torch.float32)
        pair = cute_argmax_pair(x)
        ref_max, ref_idx = torch.max(x, dim=-1, keepdim=True)
        torch.testing.assert_close(pair[:, 0:1], ref_max, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(pair[:, 1:2].long(), ref_idx, atol=0, rtol=0)


def _fake_platform(vendor: str, major: int, minor: int):
    from tokenspeed_kernel.platform import ArchVersion, PlatformInfo

    return PlatformInfo(
        vendor=vendor,
        arch_version=ArchVersion(major, minor),
        device_name=f"fake-{vendor}-{major}{minor}",
        device_count=1,
        total_memory=0,
        memory_bandwidth=0.0,
        sm_count=0,
        max_threads_per_sm=0,
        max_shared_memory_per_sm=0,
    )


def test_ts_supported_arch_gates_non_nvidia():
    """_ts_supported_arch must return False for any non-NVIDIA platform.

    Verifies the vendor gate by faking a PlatformInfo with vendor='amd'.
    """
    from tokenspeed_kernel.platform import Platform

    real_platform = Platform.get()
    try:
        Platform.override(_fake_platform("amd", 9, 4))
        assert cute_dsl._ts_supported_arch() is False
    finally:
        Platform.override(real_platform)


def test_ts_supported_arch_gates_unsupported_sm():
    """SM versions outside [9.0, 12.0) must be rejected, even on NVIDIA."""
    from tokenspeed_kernel.platform import Platform

    real_platform = Platform.get()
    try:
        # sm_80 (A100) — too old.
        Platform.override(_fake_platform("nvidia", 8, 0))
        assert cute_dsl._ts_supported_arch() is False

        # sm_120 — upstream-flagged as unstable.
        Platform.override(_fake_platform("nvidia", 12, 0))
        assert cute_dsl._ts_supported_arch() is False

        # sm_90 (Hopper) — supported.
        Platform.override(_fake_platform("nvidia", 9, 0))
        assert cute_dsl._ts_supported_arch() is True

        # sm_100 (Blackwell) — supported.
        Platform.override(_fake_platform("nvidia", 10, 0))
        assert cute_dsl._ts_supported_arch() is True
    finally:
        Platform.override(real_platform)


# ---------------------------------------------------------------------------
# Special vocab patterns — ties / structured maxima
# ---------------------------------------------------------------------------


def test_argmax_returns_first_index_on_ties_like_torch():
    """Tied values: torch.argmax returns the lowest index. The CuTe kernel uses
    a redux.sync.min on candidates whose value equals the warp max, so it must
    do the same."""
    _need_cuda()
    M, N = 4, MODEL_VOCABS["qwen3_5"]
    x = torch.full((M, N), -100.0, device="cuda", dtype=torch.float32)
    # Plant several rows where many positions hold the maximum value 0.0 at
    # known indices — the kernel must return the first.
    plant_positions = [
        [0, 7, 9],
        [3, 4],
        [128, 1024, 65536],
        [N - 1, 17],
    ]
    for row, positions in enumerate(plant_positions):
        for pos in positions:
            x[row, pos] = 0.0
    out = cute_argmax(x)
    torch.testing.assert_close(out, torch.argmax(x, dim=-1), atol=0, rtol=0)


@requires_nvidia
@pytest.mark.parametrize(
    "N", [4096, MODEL_VOCABS["kimi_k2_5"], MODEL_VOCABS["qwen3_5"]]
)
def test_argmax_in_range_for_nan_and_neg_inf_rows(N):
    """A row whose elements never beat ``-inf`` (all-NaN or all ``-inf``) must
    still yield an *in-range* index, not the ``0xFFFFFFFF`` (-1) sentinel.

    The kernel suppresses NaN (IEEE ``NaN > x`` is false) and the warp/block/
    cluster max reductions are NaN-suppressing, so NaN never wins; the argmax
    sentinels are seeded to 0 so such rows resolve to index 0. ``N`` covers both
    the single-block (``cluster_n == 1``) and the cluster reduction path (fp32
    ``N > 32K``), since the cluster path has its own sentinel seed.
    """
    _need_cuda()
    x = torch.full((6, N), -100.0, device="cuda", dtype=torch.float32)
    x[0].fill_(float("nan"))  # all NaN -> sentinel row, must map to a valid index (0)
    x[1].fill_(float("inf"))  # all inf -> first element wins, so output index 0
    x[2].fill_(float("-inf"))  # all -inf -> never beats the -inf init, also sentinel
    x[3, 2000] = 5.0  # mixed: real max at 2000 ...
    x[3, 100] = float("nan")  # ... plus a NaN the kernel must ignore
    x[4, 2000] = float("nan")
    x[4, 100] = 5.0
    x[5, 1234] = 0.0
    out = cute_argmax(x)

    assert ((out >= 0) & (out < N)).all(), f"out-of-range index: {out.tolist()}"
    # Degenerate rows resolve to index 0 (lowest-index tie among equal maxima).
    assert out[0].item() == 0
    assert out[1].item() == 0
    assert out[2].item() == 0
    # Mixed row: NaN is suppressed, so the kernel returns the finite argmax.
    assert out[3].item() == 2000
    assert out[4].item() == 100
    assert out[5].item() == 1234


def test_argmax_mtp_pattern():
    """Matches the test_argmax_mtp_case in TRT-LLM: one hot row at vocab[1].

    Uses DeepSeek V4 vocab (already JIT-cached by earlier kernel-shape tests
    within the same process) to keep this test fast — the row pattern is what's
    being verified, not the vocab size."""
    _need_cuda()
    N = MODEL_VOCABS["deepseek_v4"]
    x = torch.full((1, N), -100.0, device="cuda", dtype=torch.float32)
    x[0, 1] = 0.0
    out = cute_argmax(x)
    assert out[0].item() == 1


# ---------------------------------------------------------------------------
# ``out=`` parameter — the kernel must honor caller-provided int32 / int64
# buffers (used by greedy / flashinfer / eagle to skip a downstream cast).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("out_dtype", [torch.int32, torch.int64])
def test_argmax_writes_into_caller_buffer(out_dtype):
    """Caller-provided int32 / int64 buffer: kernel writes indices directly,
    no post-kernel cast on the hot path."""
    _need_cuda()
    M, N = 8, MODEL_VOCABS["deepseek_v4"]
    torch.manual_seed(M ^ N ^ 1)
    x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch.float32)
    out = torch.empty(M, dtype=out_dtype, device="cuda")
    returned = cute_argmax(x, out=out)
    assert returned.data_ptr() == out.data_ptr()
    assert out.dtype == out_dtype
    ref = torch.argmax(x, dim=-1)
    torch.testing.assert_close(out.long(), ref, atol=0, rtol=0)


@pytest.mark.parametrize("out_dtype", [torch.int32, torch.int64])
def test_argmax_caller_buffer_via_fallback(out_dtype):
    """``out=`` must still receive correct indices when the wrapper takes the
    torch.argmax fallback (small N here). Catches dtype-cast issues in the
    fallback's ``out.copy_(result)`` path."""
    _need_cuda()
    M, N = 4, 257  # unaligned N forces fallback.
    torch.manual_seed(N)
    x = torch.randn(M, N, device="cuda", dtype=torch.float32)
    out = torch.empty(M, dtype=out_dtype, device="cuda")
    cute_argmax(x, out=out)
    torch.testing.assert_close(out.long(), torch.argmax(x, dim=-1), atol=0, rtol=0)


def test_argmax_rejects_invalid_out():
    _need_cuda()
    x = torch.randn(4, MODEL_VOCABS["deepseek_v4"], device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="shape"):
        cute_argmax(x, out=torch.empty(5, dtype=torch.int32, device="cuda"))
    with pytest.raises(ValueError, match="int32 or int64"):
        cute_argmax(x, out=torch.empty(4, dtype=torch.float32, device="cuda"))
    with pytest.raises(ValueError, match="device"):
        cute_argmax(x, out=torch.empty(4, dtype=torch.int32, device="cpu"))


# ---------------------------------------------------------------------------
# CUDA graph compatibility — the kernel must run from inside a captured graph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M,N",
    [
        (4, MODEL_VOCABS["deepseek_v4"]),
        (16, MODEL_VOCABS["kimi_k2_5"]),
        (64, MODEL_VOCABS["gpt_oss"]),
    ],
)
def test_argmax_under_cuda_graph(M, N):
    _need_cuda()
    torch.manual_seed(M ^ N ^ 0xC0DE)
    x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch.float32)
    out = torch.empty(M, dtype=torch.int64, device="cuda")

    # Warmup.
    out.copy_(cute_argmax(x))
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(cute_argmax(x))

    # Mutate input, replay — output must reflect the new values.
    new_x = 0.1 * torch.randn_like(x)
    x.copy_(new_x)
    graph.replay()
    torch.cuda.synchronize()
    ref = torch.argmax(x, dim=-1)
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.mark.parametrize("out_dtype", [torch.int32, torch.int64])
def test_argmax_out_buffer_under_cuda_graph(out_dtype):
    """``cute_argmax(x, out=buf)`` must be CUDA-graph-safe — this is the
    pattern GreedySamplingBackend uses inside the captured sampling graph.

    Caller provides a pre-allocated buffer (no graph-internal allocation of
    the output tensor); the kernel writes int32 / int64 indices straight
    into it. Replay must observe input mutations.
    """
    _need_cuda()
    M, N = 16, MODEL_VOCABS["deepseek_v4"]
    torch.manual_seed(M ^ N ^ 0xBEEF)
    x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch.float32)
    buf = torch.empty(M, dtype=out_dtype, device="cuda")

    # Warmup so cute DSL JIT compiles outside graph capture.
    cute_argmax(x, out=buf)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cute_argmax(x, out=buf)

    new_x = 0.1 * torch.randn_like(x)
    x.copy_(new_x)
    graph.replay()
    torch.cuda.synchronize()
    ref = torch.argmax(x, dim=-1)
    torch.testing.assert_close(buf.long(), ref, atol=0, rtol=0)


def test_argmax_pair_under_cuda_graph():
    """argmax_pair with caller-provided (M, 2) f32 buffer must work inside
    a CUDA graph. The kernel splits its output into two scratch tensors
    internally; we verify the assembly still survives capture/replay."""
    _need_cuda()
    M, N = 8, MODEL_VOCABS["qwen3_5"]
    torch.manual_seed(0xABCD)
    x = 0.1 * torch.randn(M, N, device="cuda", dtype=torch.float32)
    pair = torch.empty((M, 2), dtype=torch.float32, device="cuda")

    cute_argmax_pair(x, out=pair)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cute_argmax_pair(x, out=pair)

    new_x = 0.1 * torch.randn_like(x)
    x.copy_(new_x)
    graph.replay()
    torch.cuda.synchronize()
    ref_max, ref_idx = torch.max(x, dim=-1, keepdim=True)
    torch.testing.assert_close(pair[:, 0:1], ref_max, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(pair[:, 1:2].long(), ref_idx, atol=0, rtol=0)


def test_greedy_sample_pattern_under_cuda_graph():
    """Verbatim of GreedySamplingBackend.sample's call pattern: caller holds a
    pre-allocated int32 buffer sized for max_bs, slices it per request, and
    captures the slice into the graph. Replay with new logits must produce
    new tokens."""
    _need_cuda()
    max_bs, bs, N = 32, 16, MODEL_VOCABS["kimi_k2_5"]
    torch.manual_seed(0xCAFE)
    logits = 0.1 * torch.randn(bs, N, device="cuda", dtype=torch.float32)
    sample_token_buf = torch.empty((max_bs,), dtype=torch.int32, device="cuda")

    tokens = cute_argmax(logits, out=sample_token_buf[:bs])
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tokens = cute_argmax(logits, out=sample_token_buf[:bs])

    new_logits = 0.1 * torch.randn_like(logits)
    logits.copy_(new_logits)
    graph.replay()
    torch.cuda.synchronize()
    ref = torch.argmax(logits, dim=-1).to(torch.int32)
    torch.testing.assert_close(tokens, ref, atol=0, rtol=0)
    # Slice view must alias the underlying buffer.
    torch.testing.assert_close(sample_token_buf[:bs], ref, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Pure-torch fallback on hosts without the CuTe DSL kernel — exercises the
# code path CPU-only / sm_80 / sm_120+ / missing-nvidia-cutlass-dsl
# would take. Selected at import time via the module-level dispatch in
# cute_dsl.py; we reach it here by calling the underscore-prefixed
# implementation directly so the test runs regardless of the host hardware.
# ---------------------------------------------------------------------------


def test_argmax_torch_fallback_on_cpu_tensor():
    """Pure-torch fallback must handle CPU input — non-CUDA hosts route here."""
    x = torch.randn(8, 4096, dtype=torch.float32)
    out = cute_dsl._argmax_torch_fallback(x)
    torch.testing.assert_close(out, torch.argmax(x, dim=-1), atol=0, rtol=0)
    assert out.dtype == torch.int64


def test_argmax_torch_fallback_with_int32_out():
    x = torch.randn(8, 4096, dtype=torch.float32)
    out = torch.empty(8, dtype=torch.int32)
    cute_dsl._argmax_torch_fallback(x, out=out)
    torch.testing.assert_close(out.long(), torch.argmax(x, dim=-1), atol=0, rtol=0)


def test_argmax_pair_torch_fallback_on_cpu_tensor():
    """argmax_pair fallback must handle CPU input — AMD/CPU hosts route here.

    This case was a regression in an earlier version of the wrapper that
    short-circuited on ``not logits.is_cuda``; AMD/CPU could not even reach
    the fallback. Keep this test to guard against that regression.
    """
    x = torch.randn(4, 4096, dtype=torch.float32)
    pair = cute_dsl._argmax_pair_torch_fallback(x)
    assert pair.shape == (4, 2)
    assert pair.dtype == torch.float32
    ref_max, ref_idx = torch.max(x, dim=-1, keepdim=True)
    torch.testing.assert_close(pair[:, 0:1], ref_max, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(pair[:, 1:2].long(), ref_idx, atol=0, rtol=0)


def test_public_binding_dispatch_matches_arch():
    """The CuTe module owns only the NVIDIA CuTe direct API."""
    if cute_dsl.is_available():
        assert cute_dsl.argmax is cute_dsl._argmax_cute
        assert cute_dsl.argmax_pair is cute_dsl._argmax_pair_cute
    else:
        assert cute_dsl.argmax is cute_dsl._argmax_torch_fallback
        assert cute_dsl.argmax_pair is cute_dsl._argmax_pair_torch_fallback
