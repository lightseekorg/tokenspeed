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

"""gfx950 FP32 router GEMV for decode-sized batches."""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.gemm import (  # noqa: E402
    fp32_router_gemv_counters,
    fp32_router_gemv_execute,
    fp32_router_gemv_plan,
    fp32_router_gemv_process_weight,
)
from tokenspeed_kernel_amd.ops.gfx950.gemm.fp32.router_gemv import (  # noqa: E402
    MAX_K,
    MAX_ROWS,
    gluon_fp32_router_gemv_gfx950,
    launch_gluon_fp32_router_gemv_gfx950,
    transpose_fp32_router_weight_gfx950,
)

MULTIPLIER = 0.17677669529663687


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
    out = launch_gluon_fp32_router_gemv_gfx950(x, prepared, MULTIPLIER, False)
    expected = (x.double() @ weight.double()) * MULTIPLIER
    # FP32 accumulation over K terms of unit variance: allow a few ulps of the
    # sum's scale, far below the error of any lower-precision path.
    torch.testing.assert_close(out.double(), expected, rtol=0, atol=2e-6 * hidden**0.5)


def test_logits_do_not_depend_on_the_row_count() -> None:
    """A row's logits are bitwise the same alone or batched with up to 16 rows."""
    x, weight = _problem(MAX_ROWS, 8192, 256, 7)
    prepared = transpose_fp32_router_weight_gfx950(weight)
    full = launch_gluon_fp32_router_gemv_gfx950(x, prepared, MULTIPLIER, False)
    for rows in range(1, MAX_ROWS):
        part = launch_gluon_fp32_router_gemv_gfx950(
            x[:rows].contiguous(), prepared, MULTIPLIER, False
        )
        assert torch.equal(part, full[:rows]), rows


@pytest.mark.parametrize(("rows", "hidden"), [(MAX_ROWS + 1, 512), (3, MAX_K + 4)])
def test_batches_past_the_kernel_limits_fall_back_to_matmul(
    rows: int, hidden: int
) -> None:
    x, weight = _problem(rows, hidden, 64, 11)
    prepared = transpose_fp32_router_weight_gfx950(weight)
    out = launch_gluon_fp32_router_gemv_gfx950(x, prepared, 2.0, False)
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
    x, weight = _problem(MAX_ROWS, hidden, columns, 13)
    prepared = fp32_router_gemv_process_weight(plan, weight)
    counters = fp32_router_gemv_counters(columns, x.device)
    sizes = (1, 2, 3, 4, 8, 16)
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
    hidden, columns = 8192, 256
    x, weight = _problem(MAX_ROWS, hidden, columns, 17)
    prepared = transpose_fp32_router_weight_gfx950(weight)
    for rows in (1, 2, 4, 8, 16):
        launch_gluon_fp32_router_gemv_gfx950(
            x[:rows].contiguous(), prepared, MULTIPLIER, False
        )
    with assert_no_triton_compile(gluon_fp32_router_gemv_gfx950):
        for rows in range(1, MAX_ROWS + 1):
            launch_gluon_fp32_router_gemv_gfx950(
                x[:rows].contiguous(), prepared, MULTIPLIER, False
            )


# Both counts exceed the GEMV row limit and take BLAS with alpha.
@pytest.mark.parametrize("rows", [17, 1031])
def test_planned_router_matches_scaled_matmul_bitwise(rows: int) -> None:
    """Rows above the GEMV go to BLAS with the multiplier folded into alpha."""
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
