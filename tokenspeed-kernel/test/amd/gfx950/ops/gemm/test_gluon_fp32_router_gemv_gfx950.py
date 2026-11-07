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

"""gfx950 FP32 router GEMV and split-K GEMM."""

from __future__ import annotations

import re

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4, vmcnt_zero_before_barrier

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.gemm import (  # noqa: E402
    fp32_router_gemv_counters,
    fp32_router_gemv_execute,
    fp32_router_gemv_execute_with_mm,
    fp32_router_gemv_plan,
    fp32_router_gemv_process_weight,
    fp32_router_gemv_with_mm_supported,
    mm,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp32.router_gemv import (  # noqa: E402
    MAX_K,
    MAX_ROWS,
    gluon_fp32_router_gemv_gfx950,
    gluon_fp32_router_gemv_mm_a16w16_gfx950,
    gluon_fp32_router_gemv_num_counters_gfx950,
    gluon_fp32_router_gemv_splitk_gfx950,
    launch_gluon_fp32_router_gemv_gfx950,
    transpose_fp32_router_weight_gfx950,
)

MULTIPLIER = 0.17677669529663687
# Row counts in the measured split-K buckets of the 8192 x 256 router, with
# ragged last row tiles.
SPLITK_ROWS = (9, 16, 17, 31, 64, 65, 96)


def _counters(x: torch.Tensor, prepared: torch.Tensor) -> torch.Tensor:
    numel = gluon_fp32_router_gemv_num_counters_gfx950(
        x.shape[0], prepared.shape[0], x.shape[1]
    )
    return torch.zeros(numel, dtype=torch.int32, device=x.device)


def _launch(x: torch.Tensor, prepared: torch.Tensor, multiplier: float):
    counters = _counters(x, prepared)
    return launch_gluon_fp32_router_gemv_gfx950(
        x, prepared, multiplier, counters, False
    )


def _problem(rows: int, hidden: int, columns: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, hidden, device="cuda", generator=generator)
    weight = torch.randn(hidden, columns, device="cuda", generator=generator)
    return x, weight


@pytest.mark.parametrize(
    ("rows", "hidden", "columns"),
    # Decode rows on an 8192 x 256 router, plus ragged K and N tails; K = 1002
    # takes the scalar-load variant.
    [(r, 8192, 256) for r in (1, 2, 3, 4, 5, 8, 12, 16)]
    + [(3, 1000, 100), (7, 4608, 80), (16, 300, 33), (5, 1002, 64)],
)
def test_router_gemv_matches_fp64_reference(
    rows: int, hidden: int, columns: int
) -> None:
    x, weight = _problem(rows, hidden, columns, 20261020 + rows + hidden + columns)
    prepared = transpose_fp32_router_weight_gfx950(weight)
    out = _launch(x, prepared, MULTIPLIER)
    expected = (x.double() @ weight.double()) * MULTIPLIER
    # FP32 accumulation over K terms of unit variance: allow a few ulps of the
    # sum's scale, far below the error of any lower-precision path.
    torch.testing.assert_close(out.double(), expected, rtol=0, atol=2e-6 * hidden**0.5)


def test_logits_do_not_depend_on_the_row_count() -> None:
    """A row's logits are bitwise the same alone or batched with up to 16 rows."""
    # 128 outputs have no split-K config, so the GEMV serves every row count.
    x, weight = _problem(MAX_ROWS, 8192, 128, 7)
    prepared = transpose_fp32_router_weight_gfx950(weight)
    full = _launch(x, prepared, MULTIPLIER)
    for rows in range(1, MAX_ROWS):
        part = _launch(x[:rows].contiguous(), prepared, MULTIPLIER)
        assert torch.equal(part, full[:rows]), rows


# No split-K config for K = 512; K past MAX_K exceeds the GEMV's LDS row.
@pytest.mark.parametrize(("rows", "hidden"), [(MAX_ROWS + 1, 512), (3, MAX_K + 4)])
def test_batches_past_the_kernel_limits_fall_back_to_matmul(
    rows: int, hidden: int
) -> None:
    x, weight = _problem(rows, hidden, 64, 11)
    prepared = transpose_fp32_router_weight_gfx950(weight)
    out = _launch(x, prepared, 2.0)
    assert torch.equal(out, torch.matmul(x, prepared.T) * 2.0)


def test_planned_router_replays_under_graph_capture_for_every_decode_size() -> None:
    hidden, columns = 8192, 256
    plan = fp32_router_gemv_plan(
        hidden_size=hidden,
        output_size=columns,
        dtype=torch.float32,
        solution=None,
        batch_invariant=False,
    )
    x, weight = _problem(max(SPLITK_ROWS), hidden, columns, 13)
    prepared = fp32_router_gemv_process_weight(plan, weight)
    counters = fp32_router_gemv_counters(columns, x.device)
    sizes = (1, 2, 3, 4, 8) + SPLITK_ROWS
    inputs = {rows: x[:rows].clone() for rows in sizes}
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for rows in sizes:
            fp32_router_gemv_execute(
                plan, inputs[rows], prepared, MULTIPLIER, counters, False
            )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = {
            rows: fp32_router_gemv_execute(
                plan, inputs[rows], prepared, MULTIPLIER, counters, False
            )
            for rows in sizes
        }
    for _ in range(2):
        for rows in sizes:
            inputs[rows].normal_()
        graph.replay()
        torch.cuda.synchronize()
        for rows in sizes:
            expected = (inputs[rows].double() @ weight.double()) * MULTIPLIER
            torch.testing.assert_close(
                outs[rows].double(), expected, rtol=0, atol=2e-6 * hidden**0.5
            )


def test_router_kernel_compiles_once_per_row_bucket() -> None:
    hidden, columns = 8192, 128
    x, weight = _problem(MAX_ROWS, hidden, columns, 17)
    prepared = transpose_fp32_router_weight_gfx950(weight)
    for rows in (1, 2, 4, 8, 16):
        _launch(x[:rows].contiguous(), prepared, MULTIPLIER)
    with assert_no_triton_compile(gluon_fp32_router_gemv_gfx950):
        for rows in range(1, MAX_ROWS + 1):
            _launch(x[:rows].contiguous(), prepared, MULTIPLIER)


# Both counts exceed the measured split-K buckets and take BLAS with alpha.
@pytest.mark.parametrize("rows", [97, 1031])
def test_planned_router_matches_scaled_matmul_bitwise(rows: int) -> None:
    """Rows above the split-K buckets go to BLAS with the multiplier as alpha."""
    torch.manual_seed(31)
    hidden, columns = 8192, 256
    x = torch.randn(rows, hidden, dtype=torch.float32, device="cuda")
    weight = torch.randn(hidden, columns, dtype=torch.float32, device="cuda") * 0.02
    plan = fp32_router_gemv_plan(
        hidden_size=hidden,
        output_size=columns,
        dtype=torch.float32,
        solution=None,
        batch_invariant=False,
    )
    prepared_weight = fp32_router_gemv_process_weight(plan, weight)
    counters = fp32_router_gemv_counters(columns, x.device)
    out = fp32_router_gemv_execute(
        plan, x, prepared_weight, MULTIPLIER, counters, False
    )
    expected = torch.matmul(x, prepared_weight.T) * MULTIPLIER
    assert torch.equal(out.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("rows", SPLITK_ROWS)
def test_splitk_router_matches_fp64_reference_and_top_k(rows: int) -> None:
    x, weight = _problem(rows, 8192, 256, 20261029 + rows)
    weight *= 0.02
    prepared = transpose_fp32_router_weight_gfx950(weight)
    counters = _counters(x, prepared)
    out = launch_gluon_fp32_router_gemv_gfx950(x, prepared, MULTIPLIER, counters, False)
    expected = (x.double() @ weight.double()) * MULTIPLIER
    torch.testing.assert_close(
        out.double(), expected, rtol=0, atol=2e-6 * 0.02 * 8192**0.5
    )
    assert torch.equal(out.topk(8).indices, expected.topk(8).indices)
    # The last program of every tile returns its counter to zero, so the
    # next call reuses the buffer and sums the same partials bitwise.
    assert not counters.any()
    again = launch_gluon_fp32_router_gemv_gfx950(
        x, prepared, MULTIPLIER, counters, False
    )
    assert torch.equal(again, out)


def test_splitk_router_compiles_once_per_bucket() -> None:
    hidden, columns = 8192, 256
    x, weight = _problem(max(SPLITK_ROWS), hidden, columns, 19)
    prepared = transpose_fp32_router_weight_gfx950(weight)
    # One row count per bucket; M is never specialized on alignment.
    for rows in (9, 65):
        _launch(x[:rows].contiguous(), prepared, MULTIPLIER)
    with assert_no_triton_compile(gluon_fp32_router_gemv_splitk_gfx950):
        for rows in range(min(SPLITK_ROWS), max(SPLITK_ROWS) + 1):
            _launch(x[:rows].contiguous(), prepared, MULTIPLIER)


def _compile_splitk(**config):
    """Compile the split-K kernel for gfx950 without launching it.

    ``config`` overrides the 8192 x 256 router's 64-row bucket.
    """
    from tokenspeed_kernel._triton import gluon, triton
    from tokenspeed_kernel_amd._scheduling import sched_compile_options

    fn = gluon_fp32_router_gemv_splitk_gfx950
    pointers = ("x_ptr", "weight_ptr", "out_ptr", "partial_ptr", "counter_ptr")
    constexprs = {
        "K": 8192,
        "N": 256,
        "BLOCK_M": 16,
        "BLOCK_N": 16,
        "SPLIT_K": 8,
        "CHUNKS": 1,
        "MULTIPLIER": 1.0,
        **config,
    }
    signature = {name: "*fp32" for name in pointers}
    signature.update(counter_ptr="*i32", M="i32")
    signature.update({name: "constexpr" for name in constexprs})
    attrs = {
        (index,): [["tt.divisibility", 16]]
        for index, name in enumerate(fn.arg_names)
        if name in pointers
    }
    return triton.compile(
        gluon._runtime.GluonASTSource(
            fn,
            {name: signature[name] for name in fn.arg_names},
            constexprs=constexprs,
            attrs=attrs,
        ),
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": 4, **sched_compile_options()},
    )


@pytest.mark.parametrize(
    "config",
    [
        dict(BLOCK_M=16, BLOCK_N=16, SPLIT_K=8, CHUNKS=1),
        dict(BLOCK_M=32, BLOCK_N=16, SPLIT_K=16, CHUNKS=2),
    ],
)
def test_splitk_partials_bypass_l2_and_weights_stream(config) -> None:
    # The programs of a tile can run on different XCDs, so every partial store
    # and load must be system scope (SC0 SC1). Each weight element is read
    # once, so its loads are non-temporal (NT); x loads keep the defaults.
    lines = _compile_splitk(**config).asm["amdgcn"].splitlines()

    def cache_bits(op):
        return [
            frozenset(b for b in ("sc0", "sc1", "nt") if re.search(rf"\b{b}\b", line))
            for line in lines
            if re.match(rf"\s+{op}_dword", line)
        ]

    system, stream, default = frozenset({"sc0", "sc1"}), frozenset({"nt"}), frozenset()
    loads, stores = cache_bits("buffer_load"), cache_bits("buffer_store")
    assert set(loads) == {system, stream, default}, loads
    assert loads.count(system) % config["SPLIT_K"] == 0
    assert system in stores, stores


def test_splitk_partials_reach_l2_before_arrival() -> None:
    """Every wave's partial reaches L2 before the split arrives.

    All waves write the split's partial, and one thread bumps the tile's
    relaxed counter. Compile without launching and check that, on every
    path, each wave runs s_waitcnt vmcnt(0) before the barrier preceding
    the counter atomic.
    """
    arrivals = vmcnt_zero_before_barrier(
        _compile_splitk().asm["amdgcn"],
        "gluon_fp32_router_gemv_splitk_gfx950",
        r"\bglobal_atomic_add\w*\b",
    )
    assert arrivals, "each split must arrive with an atomic add"
    assert all(waited for _, waited, _ in arrivals), arrivals


def _router_plan():
    return fp32_router_gemv_plan(
        hidden_size=8192,
        output_size=256,
        dtype=torch.float32,
        solution=None,
        batch_invariant=False,
    )


# 8 rows are below the split-K buckets, 16 rows inside the first one.
@pytest.mark.parametrize("rows", [8, 16])
def test_router_with_mm_matches_the_split_k_router_and_mm(rows: int) -> None:
    """One launch writes mm's bits and the split-K router's logits."""
    plan = _router_plan()
    x, weight = _problem(16, 8192, 256, 20261103)
    weight *= 0.02
    prepared = fp32_router_gemv_process_weight(plan, weight)
    generator = torch.Generator(device="cuda").manual_seed(rows)
    mm_a = torch.randn(rows, 8192, device="cuda", generator=generator).bfloat16()
    mm_b = (
        torch.randn(512, 8192, device="cuda", generator=generator) * 0.02
    ).bfloat16()
    router_rows = x[:rows].contiguous()
    assert fp32_router_gemv_with_mm_supported(plan, router_rows, mm_a, mm_b)
    mm_out = torch.full((rows, 512), float("nan"), dtype=torch.bfloat16, device="cuda")
    logits = fp32_router_gemv_execute_with_mm(
        plan, router_rows, prepared, MULTIPLIER, mm_a, mm_b, mm_out
    )
    assert torch.equal(mm_out, mm(mm_a, mm_b))
    # A row's split-K logits do not depend on the other rows of its tile, so
    # the standalone 16-row split-K launch gives the same bits.
    assert torch.equal(logits, _launch(x, prepared, MULTIPLIER)[:rows])


# One row has no decode GEMM config, mm sends 2-4 rows of this shape to the
# rowcta GEMV, and 17 rows exceed the shared router tile.
@pytest.mark.parametrize("rows", [1, 2, 17])
def test_router_with_mm_declines_outside_its_contract(rows: int) -> None:
    plan = _router_plan()
    x = torch.randn(rows, 8192, device="cuda")
    mm_a = torch.randn(rows, 8192, device="cuda").bfloat16()
    mm_b = torch.randn(512, 8192, device="cuda").bfloat16()
    assert not fp32_router_gemv_with_mm_supported(plan, x, mm_a, mm_b)


def test_router_with_mm_compiles_once_per_row_bucket() -> None:
    """Row counts inside a warmed bucket reuse its binary: the row count is a
    runtime argument, and only the GEMM's row bucket is compiled in."""
    plan = _router_plan()
    x, weight = _problem(16, 8192, 256, 20261104)
    prepared = fp32_router_gemv_process_weight(plan, weight)
    mm_b = torch.randn(512, 8192, device="cuda").bfloat16()

    def run(rows: int) -> None:
        mm_a = torch.randn(rows, 8192, device="cuda").bfloat16()
        mm_out = torch.empty((rows, 512), dtype=torch.bfloat16, device="cuda")
        router_rows = x[:rows].contiguous()
        assert fp32_router_gemv_with_mm_supported(plan, router_rows, mm_a, mm_b)
        fp32_router_gemv_execute_with_mm(
            plan, router_rows, prepared, MULTIPLIER, mm_a, mm_b, mm_out
        )

    for rows in (8, 16):
        run(rows)
    with assert_no_triton_compile(gluon_fp32_router_gemv_mm_a16w16_gfx950):
        for rows in (5, 7, 9, 12, 15, 16, 8):
            run(rows)
