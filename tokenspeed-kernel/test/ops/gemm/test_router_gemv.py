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

"""Deterministic split-K FP32 GEMV for decode-sized batches."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.gemm import (
    fp32_router_gemv,
    fp32_router_gemv_counters,
    fp32_router_gemv_execute,
    fp32_router_gemv_plan,
    fp32_router_gemv_process_weight,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import NoKernelFoundError

pytestmark = pytest.mark.skipif(
    not (current_platform().is_amd or current_platform().is_nvidia),
    reason="split-K GEMV tests require an AMD or NVIDIA GPU.",
)


@pytest.mark.parametrize(
    "rows, hidden, columns",
    [(1, 8192, 256), (2, 8192, 256), (3, 8192, 256), (8, 8192, 256)]
    + [(2, 256, 4), (2, 1000, 100), (2, 4608, 80), (9, 1024, 64)],
)
def test_fp32_router_gemv_matches_matmul_and_is_deterministic(
    device: str, rows: int, hidden: int, columns: int
) -> None:
    """Two rows of ragged shapes test the K and N masks; without TF32 nine rows fall back."""
    torch.manual_seed(23)
    x = torch.randn(rows, hidden, dtype=torch.float32, device=device)
    weight = torch.randn(hidden, columns, dtype=torch.float32, device=device)
    counters = fp32_router_gemv_counters(columns, torch.device(device))
    out = fp32_router_gemv(x, weight, 0.1767766953, counters, False)
    expected = torch.matmul(x, weight) * 0.1767766953
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    # The fixed split order makes repeat and PDL calls bitwise identical.
    for enable_pdl in (False, True):
        assert torch.equal(
            fp32_router_gemv(x, weight, 0.1767766953, counters, enable_pdl), out
        )
    assert counters.eq(0).all()
    assert fp32_router_gemv(x[:0], weight, 0.5, counters, False).shape == (0, columns)
    with pytest.raises(ValueError, match="counters"):
        fp32_router_gemv(x, weight, 1.0, counters[1:], False)


def _tf32(t: torch.Tensor) -> torch.Tensor:
    """Round finite FP32 values to the nearest TF32, ties to even."""
    bits = t.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rounded = (bits + 0xFFF + ((bits >> 13) & 1)) & 0xFFFFE000
    signed = torch.where(rounded >= 2**31, rounded - 2**32, rounded)
    return signed.to(torch.int32).view(torch.float32)


@pytest.mark.parametrize(
    "rows, hidden, columns",
    [(3, 8192, 256), (8, 8192, 256), (17, 8192, 256), (32, 8192, 256)]
    + [(5, 1000, 100), (20, 4608, 80)],
)
def test_tf32_router_rounds_like_cublas_where_cublas_uses_tf32(
    device: str, monkeypatch: pytest.MonkeyPatch, rows: int, hidden: int, columns: int
) -> None:
    """Three to 32 rows run on tensor cores when cuBLAS would use TF32 anyway."""
    if not current_platform().is_nvidia:
        pytest.skip("the TF32 tensor-core path is NVIDIA-only")
    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "tf32")
    monkeypatch.delenv("NVIDIA_TF32_OVERRIDE", raising=False)
    torch.manual_seed(29)
    x = torch.randn(rows, hidden, dtype=torch.float32, device=device)
    weight = torch.randn(hidden, columns, dtype=torch.float32, device=device)
    # Row 0 sits exactly between two TF32 values; row 1 carries a NaN that must survive.
    x[0] = 1 + 2**-11
    weight[:, 0] = 1.0
    x.view(torch.int32)[1, 0] = 0x7FFFFFFF
    counters = fp32_router_gemv_counters(columns, torch.device(device))

    def no_cublas(*args: object) -> torch.Tensor:
        raise AssertionError("3-32 rows must not fall back to torch.matmul")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "matmul", no_cublas)
        out = fp32_router_gemv(x, weight, 0.125, counters, False)
        for enable_pdl in (False, True):
            again = fp32_router_gemv(x, weight, 0.125, counters, enable_pdl)
            assert torch.equal(again.view(torch.int32), out.view(torch.int32))
    assert out[1].isnan().all()
    keep = torch.arange(rows, device=device) != 1
    expected = (_tf32(x[keep]).double() @ _tf32(weight).double()) * 0.125
    torch.testing.assert_close(
        out[keep].double(), expected, rtol=1e-5, atol=1e-5 * expected.abs().max().item()
    )
    assert counters.eq(0).all()
    # Without TF32 the rows go to cuBLAS's FP32 matmul.
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    fallback = fp32_router_gemv(x, weight, 0.125, counters, False)
    cublas = torch.matmul(x, weight) * 0.125
    assert torch.equal(fallback.view(torch.int32), cublas.view(torch.int32))


def test_fp32_router_gemv_replays_under_graph_capture(device: str) -> None:
    torch.manual_seed(25)
    hidden, columns = 4096, 96
    x = torch.randn(2, hidden, dtype=torch.float32, device=device)
    weight = torch.randn(hidden, columns, dtype=torch.float32, device=device)
    counters = fp32_router_gemv_counters(columns, torch.device(device))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fp32_router_gemv(x, weight, 2.0, counters, False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = fp32_router_gemv(x, weight, 2.0, counters, False)
    for _ in range(3):
        x.normal_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out, torch.matmul(x, weight) * 2.0, rtol=1e-5, atol=1e-3
        )
    assert counters.eq(0).all()


def test_planned_fp32_router_gemv_prepares_weight_and_replays(device: str) -> None:
    """The selected implementation owns its persistent layout and capture path."""
    torch.manual_seed(27)
    hidden, columns = 10240, 1024
    x = torch.randn(1, hidden, dtype=torch.float32, device=device)
    weight = torch.randn(hidden, columns, dtype=torch.float32, device=device)
    plan = fp32_router_gemv_plan(
        hidden_size=hidden,
        output_size=columns,
        dtype=torch.float32,
        solution=None,
        batch_invariant=False,
    )
    prepared_weight = fp32_router_gemv_process_weight(plan, weight)
    counters = fp32_router_gemv_counters(columns, torch.device(device))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fp32_router_gemv_execute(
            plan,
            x,
            prepared_weight,
            0.125,
            counters,
            False,
        )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = fp32_router_gemv_execute(
            plan,
            x,
            prepared_weight,
            0.125,
            counters,
            False,
        )
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        out,
        torch.matmul(x, weight) * 0.125,
        rtol=2e-5,
        atol=2e-5,
    )


def test_bitwise_plan_needs_a_batch_invariant_provider() -> None:
    """No router GEMV here declares batch invariance, so bitwise planning rejects them."""
    with pytest.raises(NoKernelFoundError):
        fp32_router_gemv_plan(
            hidden_size=8192,
            output_size=256,
            dtype=torch.float32,
            solution=None,
            batch_invariant=True,
        )
