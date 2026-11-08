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

"""The Kimi-K3 decode chain's attention input projection (``DecodeProjection``).

Up to 16 rows the projection runs the warp split-K and above 16 rows the cluster
split-K, which also takes a weight, activations or outputs the warp split-K cannot.
Both multiply BF16, accumulate in FP32 and round once to BF16; only the order in
which the K partials are summed differs. One TP8 rank's KDA and MLA shards run on
one GPU.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.moe._cute_dsl.k3_decode_support import (
    HIDDEN,
    KDA_ROWS,
    MLA_ROWS,
)
from utils import assert_no_cute_compile


def _is_blackwell() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability(0)[0] == 10
    )


pytestmark = pytest.mark.skipif(not _is_blackwell(), reason="requires NVIDIA sm100+")

WARP_ROWS = tuple(range(1, 17))
# 17 rows take the cluster split-K's 32-row tactic; 32 and 64 are chain batches.
BATCHES = (*WARP_ROWS, 17, 32, 64)


def _randn(*shape, std, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, device="cuda", generator=g) * std).bfloat16()


def _integers(*shape, seed):
    """BF16 integers in [-8, 8]."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.randint(-8, 9, shape, device="cuda", generator=g)
    return values.bfloat16()


def _projection(weight, batches):
    """The projection the chain binds for ``weight`` at ``batches`` rows."""
    from tokenspeed_kernel.ops.moe._cute_dsl.k3_decode import _projection_tactics
    from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.gemm.decode import (
        DecodeProjection,
    )

    tactics = _projection_tactics(weight.shape[0], batches)
    return DecodeProjection(weight, batches, tactics)


def _run(projection, x):
    return projection.run(x, x.new_empty(x.shape[0], projection.weight.shape[0]))


def _strided(t, pad):
    """``t``'s values in a view whose rows lie ``pad`` elements past their width."""
    rows, cols = t.shape
    view = t.new_empty(rows, cols + pad)[:, :cols]
    view.copy_(t)
    return view


def _misaligned(t):
    """``t``'s values, packed, 16 bytes past a 32-byte boundary."""
    view = t.new_empty(t.numel() + 8)[8:].view(t.shape)
    view.copy_(t)
    return view


def _assert_fp32_sum_rounded_once(out, x, weight):
    """Per element ``|out - ref| <= 2**-8 |ref| + (1 + 2**-8) K 2**-24 S``.

    ``ref`` is the exact product (float64 holds BF16 products and these sums
    exactly) and ``S = sum_k |x_k w_k|``. Each of the four compute warps sums its
    K/4 products in FP32 through K/64 mma.sync steps of 16 products into the
    accumulator, then the four partials are added. Even if a step truncated each
    of its 17 addends to the largest one's exponent and its result as well, it
    would err by at most 18 FP32 ulps of a value below S, 18 * 2**-23 S, so the
    FP32 sum errs by at most ``(K/64 * 18 + 3) * 2**-23 S < K 2**-24 S``. Rounding
    that sum once to BF16 adds at most 2**-8 of it.
    """
    ref = x.double() @ weight.double().T
    terms = x.double().abs() @ weight.double().abs().T
    k = x.shape[1]
    bound = 2**-8 * ref.abs() + (1 + 2**-8) * k * 2**-24 * terms
    error = (out.double() - ref).abs()
    worst = (error / bound).max().item()
    assert (error <= bound).all(), f"error reaches {worst:.3f} of the bound"


@pytest.fixture(scope="module", params=(KDA_ROWS, MLA_ROWS), ids=("kda", "mla"))
def projection(request):
    """One rank's input projection, bound at all warp and three cluster row counts."""
    weight = _randn(request.param, HIDDEN, std=0.02, seed=request.param)
    return _projection(weight, BATCHES)


@pytest.mark.parametrize("rows", WARP_ROWS)
def test_warp_split_k_sums_in_fp32_and_rounds_once(projection, rows):
    x = _randn(rows, HIDDEN, std=1.0, seed=100 + rows)
    out = x.new_empty(rows, projection.weight.shape[0])
    assert projection.engine(x, out) == "warp"
    _assert_fp32_sum_rounded_once(projection.run(x, out), x, projection.weight)


@pytest.fixture(scope="module")
def integer_projection(projection):
    """The same shard's projection over a weight of integers in [-8, 8]."""
    n = projection.weight.shape[0]
    return _projection(_integers(n, HIDDEN, seed=n), WARP_ROWS)


@pytest.mark.parametrize("rows", WARP_ROWS)
def test_warp_split_k_is_exact_on_small_integers(integer_projection, rows):
    """Small-integer operands make the warp split-K output exact in any summation order.

    Integers in [-8, 8] keep every FP32 partial sum an integer below
    K * 64 < 2**24, exact in any order, so each output must be the exact product
    rounded once to BF16: a K slice read twice or not at all, a dropped warp
    partial or a second rounding changes it.
    """
    weight = integer_projection.weight
    x = _integers(rows, HIDDEN, seed=200 + rows)
    out = x.new_empty(rows, weight.shape[0])
    assert integer_projection.engine(x, out) == "warp"
    expected = (x.double() @ weight.double().T).float().bfloat16()
    assert torch.equal(integer_projection.run(x, out), expected)


def test_a_rows_output_does_not_depend_on_the_other_rows(projection):
    x = _randn(16, HIDDEN, std=1.0, seed=300)
    alone = [_run(projection, x[r : r + 1]) for r in range(16)]
    for rows in WARP_ROWS:
        out = _run(projection, x[:rows])
        for r in range(rows):
            assert torch.equal(out[r], alone[r][0]), f"row {r} of {rows}"


def test_rows_above_16_run_the_cluster_split_k(projection):
    """The two engines sum in different orders, so their outputs tell them apart."""
    differing = 0
    for rows in BATCHES:
        x = _randn(rows, HIDDEN, std=1.0, seed=400 + rows)
        out = _run(projection, x)
        warp = rows <= 16
        assert projection.engine(x, out) == ("warp" if warp else "split_k")
        engine = projection.warp if warp else projection.split
        assert torch.equal(out, engine.run(x, torch.empty_like(out))), rows
        if warp:
            split = projection.split.run(x, torch.empty_like(out))
            differing += (out != split).sum().item()
    assert differing > 0


def test_operands_the_warp_split_k_cannot_take_run_the_cluster_split_k(projection):
    rows, n = 16, projection.weight.shape[0]
    x = _randn(rows, HIDDEN, std=1.0, seed=500)
    out = x.new_empty(rows, n)
    split = projection.split.run(x, torch.empty_like(out))
    assert not torch.equal(split, projection.warp.run(x, torch.empty_like(out)))
    cases = {
        "strided x": (_strided(x, 64), out),
        "misaligned x": (_misaligned(x), out),
        "strided out": (x, _strided(out, 32)),
        "misaligned out": (x, _misaligned(out)),
    }
    for case, (a, b) in cases.items():
        assert projection.engine(a, b) == "split_k", case
        b.fill_(float("nan"))
        assert projection.run(a, b) is b
        assert torch.equal(b, split), case


def test_a_weight_the_warp_split_k_cannot_take_runs_the_cluster_split_k(projection):
    weight = projection.weight
    for case, w in (
        ("strided", _strided(weight, 64)),
        ("misaligned", _misaligned(weight)),
    ):
        fallback = _projection(w, BATCHES)
        for rows in (1, 9, 16, 17):
            x = _randn(rows, HIDDEN, std=1.0, seed=600 + rows)
            out = _run(fallback, x)
            assert fallback.engine(x, out) == "split_k", (case, rows)
            expected = projection.split.run(x, torch.empty_like(out))
            assert torch.equal(out, expected), (case, rows)


def test_unprepared_rows_and_other_operands_are_rejected(projection):
    n = projection.weight.shape[0]
    x = _randn(18, HIDDEN, std=1.0, seed=700)
    with pytest.raises(ValueError, match="prepared"):
        _run(projection, x)
    with pytest.raises(ValueError, match="prepared"):
        projection.run(x[:4], x.new_empty(4, n - 16))
    with pytest.raises(ValueError, match="prepared"):
        projection.run(x[:4].float(), x.new_empty(4, n))


@pytest.fixture(scope="module")
def producer():
    """A projection of the attention input whose output feeds the tested one."""
    return _projection(_randn(HIDDEN, HIDDEN, std=0.02, seed=7), BATCHES)


def test_captured_chains_replay_eager_bit_for_bit(projection, producer):
    """A PDL projection chain replays in CUDA graphs bit for bit like the eager chain.

    A projection feeds the tested one, both launched with PDL, in one CUDA graph
    per row count. The middle and output buffers hold NaN before each replay, so
    a read of the middle before its producer finished shows; every replay must
    equal the eager chain on the same input.
    """
    n, graphs = projection.weight.shape[0], {}
    for rows in (1, 5, 9, 16, 17, 32):
        x = _randn(rows, HIDDEN, std=1.0, seed=rows)
        mid, out = x.new_empty(rows, HIDDEN), x.new_empty(rows, n)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            projection.run(producer.run(x, mid), out)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            projection.run(producer.run(x, mid), out)
        graphs[rows] = graph, x, mid, out
    for step in range(3):
        for rows, (graph, x, mid, out) in graphs.items():
            x.copy_(_randn(rows, HIDDEN, std=1.0, seed=1000 * (step + 1) + rows))
            mid.fill_(float("nan"))
            out.fill_(float("nan"))
            graph.replay()
            expected_mid = _run(producer, x)
            expected = _run(projection, expected_mid)
            torch.cuda.synchronize()
            assert torch.equal(mid, expected_mid), (step, rows)
            assert torch.equal(out, expected), (step, rows)


def test_binding_compiles_every_engine_before_the_forward(projection):
    """Binding compiles every engine, so the forward afterwards traces nothing.

    With the compile caches emptied, binding compiles every engine; running every
    row count, a fallback and a graph capture afterwards traces nothing.
    """
    from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.gemm import (
        projection as cluster,
    )
    from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.gemm import (
        warp_splitk,
    )

    warp_splitk.compile_warp_splitk.cache_clear()
    cluster.compile_projection.cache_clear()
    bound = _projection(projection.weight, BATCHES)
    inputs = {rows: _randn(rows, HIDDEN, std=1.0, seed=800 + rows) for rows in BATCHES}
    with assert_no_cute_compile():
        for x in inputs.values():
            _run(bound, x)
        _run(bound, _strided(inputs[16], 64))
        for rows in (1, 16, 32):
            x = inputs[rows]
            out = x.new_empty(rows, bound.weight.shape[0])
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                bound.run(x, out)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out, _run(bound, x)), rows
