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

"""GPU contracts for joint FI GEMV dispatch and registry fallbacks."""

from __future__ import annotations

import pytest
import tokenspeed_kernel.ops.gemm  # noqa: F401  (registration side effects)
import torch
from tokenspeed_kernel.ops.gemm.triton_gemv import _select, decode_gemv, use_decode_gemv
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _is_joint_fi_arch() -> bool:
    # Joint FI tuning covers supported Blackwell devices, not just sm103.
    return (
        current_platform().vendor == "nvidia"
        and torch.cuda.get_device_capability() >= (10, 0)
    )


def test_registry_fallback_selection():
    _select.cache_clear()
    impl = _select(1, 999, 4096, True)
    assert "rowcta" in getattr(impl, "__name__", "")
    impl = _select(4, 3217, 7168, True)
    assert "torch" in getattr(impl, "__name__", "")
    # A width no call site produces.
    impl = _select(3, 6289, 7168, True)
    assert "torch" in getattr(impl, "__name__", "")
    impl = _select(1, 2305, 1536, True)
    assert "rowcta" in getattr(impl, "__name__", "")


@pytest.mark.parametrize("projection,n", [("kda", 3216), ("mla", 3648)])
def test_kimi_projections_honor_small_m_joint_route(monkeypatch, projection, n):
    from tokenspeed_kernel.ops.gemm import kimi3, triton_gemv

    x = torch.empty(32, 7168, device="meta", dtype=torch.bfloat16)
    weight = torch.empty(n, 7168, device="meta", dtype=torch.bfloat16)
    output = torch.empty(32, n, device="meta", dtype=torch.bfloat16)
    monkeypatch.setattr(kimi3, "use_decode_gemv", lambda *_: True)
    monkeypatch.setattr(triton_gemv, "decode_gemv", lambda *_: output)
    if projection == "kda":
        assert kimi3.kimi3_qkvfab_projection(x, weight) is output
    else:
        result = kimi3.kimi3_mla_qkv_gate_projection(x, weight, 2112)
        assert result.packed is output
        assert result.qkv.shape == (32, 2112)
        assert result.gate.shape == (32, 1536)


@pytest.mark.parametrize("m", [1, 2, 4])
def test_shared_projections_route_and_match_torch(m):
    """The K3 shared gate_up/down call sites use the joint path and
    agree with the Torch composition within BF16 tolerances."""
    from tokenspeed_kernel.ops.gemm.kimi3 import (
        kimi3_shared_down_projection,
        kimi3_shared_situ_projection,
    )

    torch.manual_seed(m)
    x = torch.randn(m, 7168, device="cuda", dtype=torch.bfloat16)
    gate_up_w = torch.randn(1536, 7168, device="cuda", dtype=torch.bfloat16)
    act = kimi3_shared_situ_projection(x, gate_up_w, beta=4.0, linear_beta=25.0)
    ref = kimi3_shared_situ_projection(
        x, gate_up_w, beta=4.0, linear_beta=25.0, solution="torch"
    )
    torch.testing.assert_close(act, ref, atol=5e-2, rtol=2e-2)

    y = torch.randn(m, 768, device="cuda", dtype=torch.bfloat16)
    down_w = torch.randn(7168, 768, device="cuda", dtype=torch.bfloat16)
    got = kimi3_shared_down_projection(y, down_w)
    want = kimi3_shared_down_projection(y, down_w, solution="torch")
    torch.testing.assert_close(got, want, atol=5e-2, rtol=2e-2)


def test_forced_torch_solution_is_not_routed():
    """solution="torch" must stay the vendor-BLAS baseline even for shapes the
    joint FI path supports, or A/B comparisons silently measure the route."""
    from unittest.mock import patch

    from tokenspeed_kernel.ops.gemm import kimi3

    x = torch.randn(1, 7168, device="cuda", dtype=torch.bfloat16)
    latent_w = torch.randn(3584, 7168, device="cuda", dtype=torch.bfloat16)
    down_w = torch.randn(7168, 768, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(1, 768, device="cuda", dtype=torch.bfloat16)

    with patch(
        "tokenspeed_kernel.ops.gemm.triton_gemv.decode_gemv",
        side_effect=AssertionError("forced torch path must not route"),
    ):
        kimi3.kimi3_latent_projection(x, latent_w, solution="torch")
        kimi3.kimi3_shared_down_projection(y, down_w, solution="torch")
    # Drain the queued vendor-BLAS work here: under emulation it otherwise keeps
    # executing into the next test and charges its runtime against that test.
    torch.cuda.synchronize()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_cdna5,
    reason="CDNA5 required",
)
@pytest.mark.parametrize("m", [1, 2, 16, 17, 32])
@pytest.mark.parametrize("n,k", [(7168, 1536), (7168, 3584)])
def test_cdna5_route_admits_decode_shapes(m, n, k):
    """K3 decode shapes must route on CDNA5 through its registered kernels.

    The unquantized Linear path relies on this: o_proj is 93 calls a forward
    and reached the vendor GEMM. M == 1 lands on row-CTA and the rest on the
    dense16 WMMA kernel; 17 and 32 cross into a second 16-row chunk, where
    the launcher masks the tail.
    """
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

    assert use_decode_gemv(x, w)

    got = decode_gemv(x, w).float()
    ref = (x @ w.t()).float()
    assert torch.allclose(
        got, ref, atol=0.5, rtol=2e-2
    ), f"M={m} N={n} K={k}: max abs err {(got - ref).abs().max().item():.4f}"


@pytest.mark.skipif(
    not torch.cuda.is_available() or not current_platform().is_cdna5,
    reason="CDNA5 required",
)
def test_cdna5_route_declines_unregistered_calls():
    """Anything the registry has no specialized kernel for keeps its caller's path."""
    w_bf16 = torch.randn(7168, 1536, device="cuda", dtype=torch.bfloat16)

    x_fp16 = torch.randn(1, 1536, device="cuda", dtype=torch.float16)
    assert not use_decode_gemv(x_fp16, w_bf16.half())

    strided = torch.randn(1, 3072, device="cuda", dtype=torch.bfloat16)[:, ::2]
    assert not use_decode_gemv(strided, w_bf16)

    # K=128 is registered but measured slower than the vendor GEMM at wide N.
    thin = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
    thin_w = torch.randn(8448, 128, device="cuda", dtype=torch.bfloat16)
    assert not use_decode_gemv(thin, thin_w)

    # Past the WMMA band nothing is registered: each 16-row chunk re-reads the
    # whole weight, and by this M rocBLAS is measured faster.
    wide = torch.randn(64, 1536, device="cuda", dtype=torch.bfloat16)
    assert not use_decode_gemv(wide, w_bf16)

    # In-band M, but the WMMA kernel tiles K by 128 and N by its output tile.
    unaligned_k = torch.randn(16, 1600, device="cuda", dtype=torch.bfloat16)
    assert not use_decode_gemv(
        unaligned_k, torch.randn(7168, 1600, device="cuda", dtype=torch.bfloat16)
    )
    unaligned_n = torch.randn(16, 1536, device="cuda", dtype=torch.bfloat16)
    assert not use_decode_gemv(
        unaligned_n, torch.randn(7000, 1536, device="cuda", dtype=torch.bfloat16)
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _is_joint_fi_arch(),
    reason="Blackwell required",
)
@pytest.mark.parametrize(
    "n,k,expected_runners",
    [
        (
            512,
            2560,
            {
                "TGVRunner",
                "CuteDSLDirectBf16Runner",
                "CuteDSLSplitKBf16Runner",
                "CuteDSLWarpSplitKBf16Runner",
            },
        ),
        (2560, 160, {"TGVRunner"}),
        (2560, 320, {"TGVRunner"}),
        (7168, 2112, {"TGVRunner"}),
        (
            4120,
            2560,
            {"TGVRunner", "CuteDSLDirectBf16Runner", "CuteDSLSplitKBf16Runner"},
        ),
    ],
    ids=["aligned", "tgv-k160", "tgv-k320", "tgv-k2112", "ragged-n4120"],
)
def test_joint_fi_tuning_roundtrip_and_changed_input_replay(
    monkeypatch, tmp_path, n, k, expected_runners
):
    """Native candidates form a union, including non-128 K and non-16 N."""
    from flashinfer.autotuner import AutoTuner
    from tokenspeed_kernel.ops.gemm import flashinfer as fi_adapter
    from tokenspeed_kernel.ops.gemm import mm
    from tokenspeed_kernel.ops.tuning import (
        autotune,
        load_autotune_cache,
        save_autotune_cache,
    )

    x = torch.randn(128, k, device="cuda", dtype=torch.bfloat16)
    weight = (
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    ).contiguous()
    if not fi_adapter.flashinfer_joint_bf16_supported(x, weight, None):
        pytest.skip("FI joint adapter unavailable")
    if not hasattr(fi_adapter._fi_gemm, "_cute_dsl_bf16_runners"):
        expected_runners = expected_runners - {"CuteDSLWarpSplitKBf16Runner"}
    assert fi_adapter.BF16_GEMM_MAX_M == 32
    tuner = AutoTuner.get()
    tuner.clear_cache()
    native_key = AutoTuner.__dict__["_get_cache_key"]
    profiles = set()
    original = tuner._profile_single_kernel

    def record(runner, tensors, *args, **kwargs):
        profiles.add((type(runner).__name__, tensors[0].shape[0], tensors[4].dtype))
        return original(runner, tensors, *args, **kwargs)

    monkeypatch.setattr(tuner, "_profile_single_kernel", record)
    # Empty-cache defaults must be legal too, without profiling or fallback to Torch.
    with torch.no_grad(), autotune(tune_mode=False, tuning_buckets=None, round_up=None):
        for m in (1, 3, 31, 32):
            got = fi_adapter.flashinfer_bf16_gemm(x[:m], weight, None)
            ref = x[:m].float() @ weight.float().T
            assert torch.isfinite(got).all()
            assert (
                torch.linalg.vector_norm(got.float() - ref)
                / torch.linalg.vector_norm(ref)
                < 0.01
            )
    assert not profiles

    with torch.no_grad(), autotune(tune_mode=True, tuning_buckets=None, round_up=None):
        fi_adapter.autotune_bf16_gemm(x, weight)
    assert {name for name, _, _ in profiles} == expected_runners
    assert {m for _, m, _ in profiles} == {1, 2, 4, 8, 16, 32}
    assert {dtype for _, _, dtype in profiles} == {torch.bfloat16}
    assert AutoTuner.__dict__["_get_cache_key"] is native_key
    print(
        "PROFILE UNION", n, k, sorted((name, m) for name, m, _ in profiles), flush=True
    )

    path = str(tmp_path / "configs.json")
    assert save_autotune_cache(path, None, 0)
    tuner.clear_cache()
    assert load_autotune_cache(path, None, 0)
    attempts = []

    def unexpected_profile(*args, **kwargs):
        # FI can catch profiling failures, so also verify the attempt counter.
        attempts.append(1)
        raise AssertionError("cache hit profiled")

    monkeypatch.setattr(tuner, "_profile_single_kernel", unexpected_profile)
    with torch.no_grad(), autotune(tune_mode=True, tuning_buckets=None, round_up=None):
        fi_adapter.autotune_bf16_gemm(x, weight)
        for m in (1, 3, 31, 32):
            fi_adapter.flashinfer_bf16_gemm(x[:m], weight, None)
    assert not attempts

    with torch.no_grad(), autotune(tune_mode=False, tuning_buckets=None, round_up=None):
        for m in range(1, 33):
            assert use_decode_gemv(x[:m], weight)
            buffer = torch.full((m * n + 32,), 42, dtype=torch.bfloat16, device="cuda")
            out = buffer[: m * n].view(m, n)
            out.fill_(float("nan"))
            assert mm(x[:m], weight, out=out).data_ptr() == out.data_ptr()
            ref = x[:m].float() @ weight.float().T
            assert torch.isfinite(out).all()
            assert (
                torch.linalg.vector_norm(out.float() - ref)
                / torch.linalg.vector_norm(ref)
                < 0.01
            )
            assert torch.all(buffer[m * n :] == 42)
            if m in (1, 3, 8, 17, 31, 32):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    fi_adapter.flashinfer_bf16_gemm(x[:m], weight, out)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    fi_adapter.flashinfer_bf16_gemm(x[:m], weight, out)
                x[:m].mul_(0.875)
                out.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                ref = x[:m].float() @ weight.float().T
                assert torch.isfinite(out).all()
                assert (
                    torch.linalg.vector_norm(out.float() - ref)
                    / torch.linalg.vector_norm(ref)
                    < 0.01
                )
                assert torch.all(buffer[m * n :] == 42)

        def reject_fi(**kwargs):
            raise AssertionError("M > 32 reached FI joint dispatch")

        monkeypatch.setattr(fi_adapter._fi_gemm, "bf16_gemm_sm100", reject_fi)
        for m in (33, 48, 64, 128):
            assert not use_decode_gemv(x[:m], weight)
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            assert mm(x[:m], weight, out=out).data_ptr() == out.data_ptr()
            torch.testing.assert_close(out, x[:m] @ weight.T, rtol=0, atol=0)
            torch.testing.assert_close(
                decode_gemv(x[:m], weight), x[:m] @ weight.T, rtol=0, atol=0
            )
    assert not attempts
    assert AutoTuner.__dict__["_get_cache_key"] is native_key
    print(
        "PASSED UNION",
        n,
        k,
        "32 sizes; 6 changed-input graphs; 4 large-M fallbacks; 0 reprofile",
        flush=True,
    )
