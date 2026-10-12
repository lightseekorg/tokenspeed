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

"""GFX950 Gluon Qwen4 hyper-connection kernels."""

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.residual import gated_residual_mix  # noqa: E402
from tokenspeed_kernel.profiling import ShapeCapture  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950 import hyperconnection  # noqa: E402

HC_COUNT = 4
HIDDEN_SIZE = 2560
LOWRANK = 320
WIDE = HC_COUNT * HIDDEN_SIZE


def _inputs(rows, projection_rows, dtype, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=generator) * scale).to(
            dtype
        )

    return (
        randn(rows, WIDE),
        randn(projection_rows, WIDE, scale=0.01),
        randn(WIDE, LOWRANK, scale=0.01),
    )


def _mix(inputs, override, projection_scale=1.0):
    return gated_residual_mix(
        *inputs,
        HC_COUNT,
        HIDDEN_SIZE,
        LOWRANK,
        weights_independent=True,
        projection_scale=projection_scale,
        override=override,
    )


def _fp64_reference(normalized, projection, up, projection_scale):
    x = normalized.double()
    projected = (x @ projection.double().T) * projection_scale
    gate = torch.nn.functional.silu(projected[:, :LOWRANK]) @ up.double().T
    mixed = (
        torch.sigmoid(gate).unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
        * x.unflatten(-1, (HC_COUNT, HIDDEN_SIZE))
    ).mean(dim=-2)
    return mixed, projected[:, LOWRANK:]


@pytest.mark.parametrize("rows", [1, 7, 16, 17, 32, 33, 64, 100, 256])
@pytest.mark.parametrize("projection_rows", [LOWRANK + HC_COUNT, LOWRANK])
@pytest.mark.parametrize("projection_scale", [1.0, 0.25])
def test_mix_matches_gemm_epilogue_path(rows, projection_rows, projection_scale):
    inputs = _inputs(rows, projection_rows, torch.bfloat16, seed=rows)
    mixed, inject = _mix(inputs, "gluon_hyperconnection_mix_gfx950", projection_scale)
    ref_mixed, ref_inject = _mix(inputs, "triton_hyperconnection_mix", projection_scale)
    # Both round the projections to BF16 like standalone GEMMs; only the FP32
    # accumulation order differs, which moves an output by at most one BF16
    # ulp.
    torch.testing.assert_close(mixed, ref_mixed, rtol=8e-3, atol=4e-3)
    if projection_rows == LOWRANK:
        assert inject is None and ref_inject is None
    else:
        torch.testing.assert_close(inject, ref_inject, rtol=8e-3, atol=1e-3)
    exact_mixed, _ = _fp64_reference(*inputs, projection_scale)
    torch.testing.assert_close(mixed.double(), exact_mixed, rtol=3e-2, atol=3e-2)


def test_mix_dispatch_and_float16():
    capture = ShapeCapture.get()
    previous = capture.enabled
    capture.enabled = True
    capture.clear()
    try:
        for rows in (1, 64, 256, 257):
            _mix(_inputs(rows, LOWRANK + HC_COUNT, torch.float16, seed=5), None)
        names = [record.kernel_name for record in capture._records]
    finally:
        capture.enabled = previous
        capture.clear()
    assert names == ["gluon_hyperconnection_mix_gfx950"] * 3 + [
        "triton_hyperconnection_mix"
    ]
    inputs = _inputs(48, LOWRANK + HC_COUNT, torch.float16, seed=6)
    mixed, _ = _mix(inputs, "gluon_hyperconnection_mix_gfx950")
    exact_mixed, _ = _fp64_reference(*inputs, 1.0)
    torch.testing.assert_close(mixed.double(), exact_mixed, rtol=8e-3, atol=8e-3)


def test_mix_graph_replay_reuses_workspace():
    inputs = _inputs(64, LOWRANK + HC_COUNT, torch.bfloat16, seed=9)
    expected = _mix(inputs, "gluon_hyperconnection_mix_gfx950")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = [_mix(inputs, "gluon_hyperconnection_mix_gfx950") for _ in range(3)]
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    for actual in outputs:
        # Split partials are summed in split order, so results are bitwise
        # reproducible.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mix_row_count_does_not_recompile():
    inputs = _inputs(256, LOWRANK + HC_COUNT, torch.bfloat16, seed=11)

    def run(rows):
        normalized, projection, up = inputs
        return hyperconnection.launch_gluon_hyperconnection_mix_gfx950(
            normalized[:rows], projection, up, HC_COUNT, HIDDEN_SIZE, LOWRANK, 1.0
        )

    # Row buckets (<= 16, <= 32, larger) select the tile shape.
    for rows in (16, 32, 64):
        run(rows)
    with assert_no_triton_compile(
        hyperconnection.gluon_hyperconnection_mix_gfx950,
        hyperconnection.gluon_hyperconnection_mix_up_gfx950,
    ):
        for rows in (1, 3, 15, 17, 31, 33, 63, 65, 200, 256):
            run(rows)
