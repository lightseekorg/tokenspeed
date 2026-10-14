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

from __future__ import annotations

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for the gfx950 MXFP4 linear tests",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.activation.triton import silu_and_mul  # noqa: E402
from tokenspeed_kernel.ops.gemm import (  # noqa: E402
    mxfp4_linear,
    mxfp4_swiglu_linear,
    prepare_mxfp4_linear,
)
from tokenspeed_kernel.ops.gemm.triton import triton_mm_mxfp4  # noqa: E402
from tokenspeed_kernel.ops.quantization.triton import mxfp4_quantize  # noqa: E402
from tokenspeed_kernel.workspace import capture_session, zeroed_state  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.gemm.mxfp4.mm import (  # noqa: E402
    gluon_mxfp4_linear_gfx950,
    gluon_mxfp4_linear_num_counters_gfx950,
    gluon_mxfp4_swiglu_linear_gfx950,
    launch_gluon_mxfp4_linear_gfx950,
    launch_gluon_mxfp4_swiglu_linear_gfx950,
    preshuffle_mxfp4_weight,
)

# The kernel quantizes activations with the mxfp4_quantize recipe bit for bit
# and multiplies exact E2M1 x E2M1 products under power-of-two scales, so it
# matches the unfused path up to FP32 summation order: at most one BF16 ULP.
_ATOL = 1e-3
_RTOL = 2**-8

# Qwen3.8-Flash-Next TP2 shared expert: gate_up [2 * 320, 2560], down
# [2560, 320].
_HIDDEN = 2560
_INTERMEDIATE = 320


def _counters(x, weight, swiglu):
    return zeroed_state(
        gluon_mxfp4_linear_num_counters_gfx950(
            x.shape[0], weight.shape[0], x.shape[1], swiglu
        ),
        dtype=torch.int32,
        device=x.device,
    )


def _linear(x, weight, scale, out=None):
    counters = _counters(x, weight, False)
    return launch_gluon_mxfp4_linear_gfx950(
        x, weight, scale, counters=counters, out=out
    )


def _swiglu(x, weight, scale, out=None):
    counters = _counters(x, weight, True)
    return launch_gluon_mxfp4_swiglu_linear_gfx950(
        x, weight, scale, counters=counters, out=out
    )


def _weight(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
    scale = torch.randint(118, 124, (n, k // 32), device="cuda", dtype=torch.uint8)
    return weight, scale


def _unfused_linear(x, weight, scale) -> torch.Tensor:
    x_fp4, x_scale = mxfp4_quantize(x)
    return triton_mm_mxfp4(x_fp4, weight, x_scale, scale, torch.bfloat16)


def _unfused_swiglu(x, weight, scale) -> torch.Tensor:
    gate_up = _unfused_linear(x, weight, scale)
    out = torch.empty(
        (x.shape[0], weight.shape[0] // 2), device=x.device, dtype=torch.bfloat16
    )
    return silu_and_mul(gate_up, out)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.float(), expected.float(), atol=_ATOL, rtol=_RTOL)


@pytest.mark.parametrize(
    "m",
    # Decode (MTP draft and verify), mid-size and prefill M buckets.
    [1, 7, 16, 33, 64, 200, 600, 2500],
)
def test_shared_expert_matches_unfused_path(m: int) -> None:
    torch.manual_seed(0)
    x = torch.randn((m, _HIDDEN), device="cuda", dtype=torch.bfloat16) * 3
    gate_up, gate_up_scale = _weight(2 * _INTERMEDIATE, _HIDDEN)
    down, down_scale = _weight(_HIDDEN, _INTERMEDIATE)

    act = _swiglu(x, preshuffle_mxfp4_weight(gate_up), gate_up_scale)
    expected_act = _unfused_swiglu(x, gate_up, gate_up_scale)
    _assert_close(act, expected_act)

    out = _linear(expected_act, preshuffle_mxfp4_weight(down), down_scale)
    _assert_close(out, _unfused_linear(expected_act, down, down_scale))


@pytest.mark.parametrize(
    "m,n,k",
    [
        # Untuned shapes take the heuristic tiles: ragged N, a K tail, split-K.
        (5, 1000, 1024),
        (48, 384, 4096),
        (300, 2048, 704),
    ],
)
def test_heuristic_shapes_match_unfused_path(m: int, n: int, k: int) -> None:
    torch.manual_seed(0)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight, scale = _weight(n - n % 16, k)
    actual = _linear(x, preshuffle_mxfp4_weight(weight), scale)
    _assert_close(actual, _unfused_linear(x, weight, scale))


def test_writes_strided_out_deterministically() -> None:
    torch.manual_seed(0)
    m = 12  # split-K route
    x = torch.randn((m, _HIDDEN), device="cuda", dtype=torch.bfloat16)
    weight, scale = _weight(2 * _INTERMEDIATE, _HIDDEN)
    shuffled = preshuffle_mxfp4_weight(weight)
    backing = torch.empty((m, _INTERMEDIATE + 24), device="cuda", dtype=torch.bfloat16)
    out = backing[:, :_INTERMEDIATE]
    actual = _swiglu(x, shuffled, scale, out=out)
    assert actual is out
    _assert_close(out, _unfused_swiglu(x, weight, scale))
    # The split-K reduction sums partials in split order.
    for _ in range(3):
        assert torch.equal(
            _swiglu(x, shuffled, scale),
            out,
        )


def test_row_count_does_not_recompile() -> None:
    torch.manual_seed(0)
    x = torch.randn((4096, _HIDDEN), device="cuda", dtype=torch.bfloat16)
    gate_up, gate_up_scale = _weight(2 * _INTERMEDIATE, _HIDDEN)
    down, down_scale = _weight(_HIDDEN, _INTERMEDIATE)
    gate_up = preshuffle_mxfp4_weight(gate_up)
    down = preshuffle_mxfp4_weight(down)
    act = torch.randn((4096, _INTERMEDIATE), device="cuda", dtype=torch.bfloat16)

    def run(rows: int) -> None:
        _swiglu(x[:rows], gate_up, gate_up_scale)
        _linear(act[:rows], down, down_scale)

    # One row count per M bucket compiles each tile configuration.
    for rows in (1, 65, 257, 513, 2049):
        run(rows)
    with assert_no_triton_compile(
        gluon_mxfp4_linear_gfx950, gluon_mxfp4_swiglu_linear_gfx950
    ):
        for rows in (2, 16, 17, 48, 64, 100, 256, 300, 512, 1000, 2048, 3000, 4096):
            run(rows)


def test_hip_graph_replay_keeps_counters_zeroed() -> None:
    # A captured split-K launch gets counters zeroed outside the graph; every
    # replay leaves them zeroed for the next.
    torch.manual_seed(0)
    m = 16  # split-K route
    x = torch.randn((m, _HIDDEN), device="cuda", dtype=torch.bfloat16)
    weight, scale = _weight(2 * _INTERMEDIATE, _HIDDEN)
    shuffled = preshuffle_mxfp4_weight(weight)
    out = torch.empty((m, _INTERMEDIATE), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        _swiglu(x, shuffled, scale, out=out)
        with capture_session(graph), torch.cuda.graph(graph, stream=stream):
            _swiglu(x, shuffled, scale, out=out)
    stream.synchronize()

    for _ in range(3):
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        _assert_close(out, _unfused_swiglu(x, weight, scale))


def test_registry_selects_gluon_and_matches_portable_path() -> None:
    torch.manual_seed(0)
    x = torch.randn((24, _HIDDEN), device="cuda", dtype=torch.bfloat16)
    weight, scale = _weight(2 * _INTERMEDIATE, _HIDDEN)

    gluon = prepare_mxfp4_linear(weight, scale, swiglu=True)
    portable = prepare_mxfp4_linear(weight, scale, swiglu=True, solution="triton")
    assert gluon.layout == "mfma_16x16"
    assert portable.layout == "linear" and portable.weight is weight
    _assert_close(mxfp4_swiglu_linear(x, gluon), mxfp4_swiglu_linear(x, portable))
    _assert_close(mxfp4_linear(x, gluon), mxfp4_linear(x, portable))
