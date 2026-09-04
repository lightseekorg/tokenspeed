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

"""Tensor-core vs reduction KDA gate precompute: agreement and tolerance."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
    _gate_tiling,
    _gate_tiling_dot,
    batched_kda_gate_precompute_dot_kernel,
    batched_kda_gate_precompute_kernel,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

LAYERS, HEADS, HEAD_DIM, D_FA = 4, 12, 128, 128
LOWER_BOUND = 0.5


def _run(kernel, block_t, block_k, rows, tensors, num_warps):
    """Launch one gate kernel over a descriptor table and return the gates."""
    f_a, f_b, a_log, dt_bias = tensors
    gates = [
        torch.zeros(rows, HEADS * HEAD_DIM, dtype=torch.float32, device="cuda")
        for _ in range(LAYERS)
    ]
    addresses = torch.zeros(LAYERS, 10, dtype=torch.int64, device="cuda")
    for layer in range(LAYERS):
        addresses[layer, 3] = f_a[layer].data_ptr()
        addresses[layer, 4] = f_b[layer].data_ptr()
        addresses[layer, 6] = a_log[layer].data_ptr()
        addresses[layer, 7] = dt_bias[layer].data_ptr()
        addresses[layer, 9] = gates[layer].data_ptr()
    kernel[
        (
            LAYERS * HEADS,
            triton.cdiv(rows, block_t),
            triton.cdiv(HEAD_DIM, block_k),
        )
    ](
        addresses,
        rows,
        stride_fa=D_FA,
        stride_gate=HEADS * HEAD_DIM,
        lower_bound=LOWER_BOUND,
        HV=HEADS,
        K=HEAD_DIM,
        D_FA=D_FA,
        BK=block_k,
        BT=block_t,
        num_warps=num_warps,
    )
    torch.cuda.synchronize()
    return torch.stack(gates)


def _tensors(rows):
    gen = torch.Generator(device="cuda").manual_seed(0)
    rand = lambda *shape, dtype: torch.randn(  # noqa: E731
        *shape, generator=gen, dtype=dtype, device="cuda"
    )
    return (
        [rand(rows, D_FA, dtype=torch.bfloat16) for _ in range(LAYERS)],
        [rand(HEADS * HEAD_DIM, D_FA, dtype=torch.bfloat16) for _ in range(LAYERS)],
        [rand(HEADS, dtype=torch.float32) * 0.1 for _ in range(LAYERS)],
        [rand(HEADS * HEAD_DIM, dtype=torch.float32) * 0.1 for _ in range(LAYERS)],
    )


@pytest.mark.parametrize("rows", [16, 128, 512])
def test_dot_path_matches_reduction_within_tiling_spread(rows) -> None:
    """The two forms agree as closely as two reduction tilings agree."""
    # The rocJITsu gfx1250 simulator cannot finish rows=512 inside CI limits.
    if rows >= 512 and "gfx1250" in getattr(
        torch.cuda.get_device_properties(0), "gcnArchName", ""
    ):
        pytest.skip("simulated gfx1250 is too slow for the production-shape case")
    tensors = _tensors(rows)
    device = torch.device("cuda")
    block_t, block_k = _gate_tiling(rows, HEADS, HEAD_DIM, device, layers=LAYERS)
    baseline = _run(
        batched_kda_gate_precompute_kernel,
        block_t,
        block_k,
        rows,
        tensors,
        1 if block_t >= 4 else 2,
    )
    # A second legal reduction tiling: the spread between these is the tolerance
    # the shipped code already accepts.
    other = _run(batched_kda_gate_precompute_kernel, 1, 16, rows, tensors, 2)
    tiling_spread = (baseline - other).abs().max()

    dot_t, dot_k = _gate_tiling_dot(rows, HEAD_DIM)
    dot = _run(batched_kda_gate_precompute_dot_kernel, dot_t, dot_k, rows, tensors, 1)
    assert dot.shape == baseline.shape
    assert torch.isfinite(dot).all()
    # lower_bound * sigmoid(...) is bounded by construction; a wrong reduction
    # axis would leave this range immediately.
    assert dot.min() >= 0.0 and dot.max() <= LOWER_BOUND

    gap = (baseline - dot).abs().max()
    assert gap <= max(
        tiling_spread * 4, 1e-5
    ), f"dot path differs by {gap:.3e}, tiling spread is {tiling_spread:.3e}"


def test_dot_tiling_never_returns_a_tile_tl_dot_cannot_serve() -> None:
    """tl.dot needs 16 rows; the dot tiling must never propose fewer."""
    device = torch.device("cuda")
    for rows in (16, 32, 64, 128, 512, 1024):
        block_t, block_k = _gate_tiling_dot(rows, HEAD_DIM)
        assert block_t >= 16, f"rows={rows} produced BT={block_t}"
        assert block_k >= 16, f"rows={rows} produced BK={block_k}"
