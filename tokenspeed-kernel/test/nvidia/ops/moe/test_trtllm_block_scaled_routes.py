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

"""Block-scaled route rows on TRT-LLM-gen batched GEMMs vs the Triton scaled dot.

Both sum each element's K blocks in one order and scale it alike, so the route
rows must match bit for bit at every token tile, through the weight
preprocessor, the expert tiling and the gather back to route rows.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch
from block_quantize_helpers import triton_block_quantize
from tokenspeed_kernel.ops.moe.block_scaled import (
    block_scaled_moe_routes,
    block_scaled_moe_routes_plan,
    block_scaled_moe_routes_process_tensor,
)
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not (10, 0) <= torch.cuda.get_device_capability() <= (10, 3),
    reason="TRT-LLM-gen batched GEMMs target the sm_100 family",
)


class Layer(NamedTuple):
    hidden: int
    intermediate: int
    local: int
    experts: int
    start: int


# Four of 16 experts on the rank; and EP2's 192 of 384 at narrow widths.
SMALL = Layer(hidden=1536, intermediate=768, local=4, experts=16, start=4)
WIDE = Layer(hidden=512, intermediate=256, local=192, experts=384, start=192)
TOP_K = 8


def _experts(layer, formats, generator):
    """Quantized local experts of each projection: (values, scales, amax, den)."""
    out = []
    for fmt, rows, width in (
        (formats[0], 2 * layer.intermediate, layer.hidden),
        (formats[1], layer.hidden, layer.intermediate),
    ):
        dense = torch.randn(
            layer.local * rows, width, generator=generator, device="cuda"
        )
        values, scales, _ = triton_block_quantize((dense * 0.05).bfloat16(), fmt)
        values = values.view(layer.local, rows, -1)
        scales = scales.view(layer.local, rows, -1)
        if fmt == "mxfp8":
            out.append((values, scales, None, None))
        else:
            amax = torch.rand(layer.local, generator=generator, device="cuda") + 0.5
            out.append((values, scales, amax, torch.tensor([2688.0], device="cuda")))
    return out


def _routes(layer, solution, formats, w13, w2):
    """Planned routes on copies of the experts, prepared for ``solution``."""
    plan = block_scaled_moe_routes_plan(
        formats=formats,
        hidden=layer.hidden,
        intermediate=layer.intermediate,
        device=torch.device("cuda"),
        solution=solution,
    )
    w13 = tuple(None if t is None else t.clone() for t in w13)
    w2 = tuple(None if t is None else t.clone() for t in w2)
    for tensor, role in (
        (w13[0], "w13_weight"),
        (w13[1], "w13_scale"),
        (w2[0], "w2_weight"),
        (w2[1], "w2_scale"),
    ):
        block_scaled_moe_routes_process_tensor(plan, tensor, role=role)

    def run(x, weights, ids, zero_remote_routes, fold_slots=False):
        return block_scaled_moe_routes(
            plan,
            x,
            *w13,
            *w2,
            weights,
            ids,
            experts=layer.experts,
            expert_start=layer.start,
            gate_cap=4.0,
            up_cap=25.0,
            zero_remote_routes=zero_remote_routes,
            fold_slots=fold_slots,
            fold_into=None,
        )

    return run


def _inputs(layer, tokens, generator):
    x = (
        torch.randn(tokens, layer.hidden, generator=generator, device="cuda") * 0.5
    ).bfloat16()
    ids = torch.rand((tokens, layer.experts), generator=generator, device="cuda")
    ids = ids.argsort(dim=1)[:, :TOP_K].to(torch.int32).contiguous()
    weights = torch.rand((tokens, TOP_K), generator=generator, device="cuda")
    return x, weights, ids


def _assert_equal_local_rows(layer, actual, expected, ids, zero_remote_routes):
    local = (ids >= layer.start) & (ids < layer.start + layer.local)
    assert torch.equal(actual[local], expected[local])
    if zero_remote_routes:
        assert actual[~local].count_nonzero() == 0


@pytest.mark.parametrize(
    "formats", [("nvfp4", "nvfp4"), ("mxfp8", "mxfp8"), ("nvfp4", "mxfp8")]
)
@pytest.mark.parametrize("zero_remote_routes", [True, False])
def test_routes_match_the_triton_scaled_dot_at_every_tile(formats, zero_remote_routes):
    generator = torch.Generator(device="cuda").manual_seed(5)
    w13, w2 = _experts(SMALL, formats, generator)
    trtllm = _routes(SMALL, "flashinfer_trtllm", formats, w13, w2)
    triton = _routes(SMALL, "triton", formats, w13, w2)
    # 1..600 tokens put 0.5..300 routes on each expert: token tiles 8 to 256,
    # the one-launch tiling up to 1024 routes and the two-launch one past it.
    for tokens in (1, 7, 33, 128, 129, 160, 600):
        x, weights, ids = _inputs(SMALL, tokens, generator)
        expected = triton(x, weights, ids, zero_remote_routes)
        local = (ids >= SMALL.start) & (ids < SMALL.start + SMALL.local)
        assert expected[local].abs().amax() > 0
        actual = trtllm(x, weights, ids, zero_remote_routes)
        _assert_equal_local_rows(SMALL, actual, expected, ids, zero_remote_routes)


@pytest.mark.parametrize("formats", [("nvfp4", "nvfp4"), ("mxfp8", "mxfp8")])
@pytest.mark.parametrize("tokens", [40, 600])
def test_skewed_and_remote_only_routes(formats, tokens):
    generator = torch.Generator(device="cuda").manual_seed(9)
    w13, w2 = _experts(SMALL, formats, generator)
    trtllm = _routes(SMALL, "flashinfer_trtllm", formats, w13, w2)
    triton = _routes(SMALL, "triton", formats, w13, w2)
    x, weights, _ = _inputs(SMALL, tokens, generator)
    # Every token on the same eight experts, the four local ones among them:
    # each local expert takes every token.
    crowded = torch.arange(TOP_K, device="cuda", dtype=torch.int32).repeat(tokens, 1)
    remote = (crowded + 8).contiguous()  # experts 8..15, none local
    for ids in (crowded, remote):
        for zero_remote_routes in (True, False):
            expected = triton(x, weights, ids, zero_remote_routes)
            actual = trtllm(x, weights, ids, zero_remote_routes)
            _assert_equal_local_rows(SMALL, actual, expected, ids, zero_remote_routes)


@pytest.mark.parametrize("formats", [("nvfp4", "nvfp4"), ("mxfp8", "mxfp8")])
def test_folded_routes_match_the_triton_fold_and_the_route_rows(formats):
    generator = torch.Generator(device="cuda").manual_seed(17)
    w13, w2 = _experts(SMALL, formats, generator)
    trtllm = _routes(SMALL, "flashinfer_trtllm", formats, w13, w2)
    triton = _routes(SMALL, "triton", formats, w13, w2)
    for tokens in (1, 7, 33, 160, 600):
        x, weights, ids = _inputs(SMALL, tokens, generator)
        folded = trtllm(x, weights, ids, False, fold_slots=True)
        assert folded.shape == (tokens, 1, SMALL.hidden)
        assert torch.equal(folded, triton(x, weights, ids, False, fold_slots=True))
        # The local route rows summed in FP32 slot order, rounded once.
        rows = triton(x, weights, ids, True)
        local = (ids >= SMALL.start) & (ids < SMALL.start + SMALL.local)
        expected = torch.zeros((tokens, SMALL.hidden), device="cuda")
        for slot in range(TOP_K):
            expected += torch.where(local[:, slot, None], rows[:, slot], 0.0).float()
        assert torch.equal(folded[:, 0], expected.bfloat16())


@pytest.mark.parametrize("formats", [("nvfp4", "nvfp4"), ("mxfp8", "mxfp8")])
def test_wide_expert_shard_matches_the_triton_scaled_dot(formats):
    generator = torch.Generator(device="cuda").manual_seed(21)
    w13, w2 = _experts(WIDE, formats, generator)
    trtllm = _routes(WIDE, "flashinfer_trtllm", formats, w13, w2)
    triton = _routes(WIDE, "triton", formats, w13, w2)
    for tokens in (7, 128, 600):
        x, weights, ids = _inputs(WIDE, tokens, generator)
        expected = triton(x, weights, ids, True)
        actual = trtllm(x, weights, ids, True)
        _assert_equal_local_rows(WIDE, actual, expected, ids, True)


@pytest.mark.parametrize("formats", [("nvfp4", "nvfp4"), ("mxfp8", "mxfp8")])
def test_graph_replay_and_new_tiles_reuse_the_compiled_kernels(formats):
    from tokenspeed_kernel.ops.moe._flashinfer import route_tiles
    from tokenspeed_kernel.ops.moe._triton import block_scaled as triton_routes

    generator = torch.Generator(device="cuda").manual_seed(13)
    w13, w2 = _experts(SMALL, formats, generator)
    routes = _routes(SMALL, "flashinfer_trtllm", formats, w13, w2)
    for tokens in (32, 200):  # one-launch and two-launch tiling
        x, weights, ids = _inputs(SMALL, tokens, generator)
        expected = routes(x, weights, ids, True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = routes(x, weights, ids, True)
        x.mul_(0.5)
        weights.mul_(1.7)
        ids.copy_(ids.flip(1))
        graph.replay()
        assert not torch.equal(captured, expected)
        assert torch.equal(captured, routes(x, weights, ids, True))

    # Route counts divisible by 16 or not, and the per-route SwiGLU tier past
    # 1024 routes, compile once each; tiles and scale layouts are runtime.
    for tokens in (1, 200):
        routes(*_inputs(SMALL, tokens, generator), True)
    with assert_no_triton_compile(
        route_tiles._route_tile_quantize_kernel,
        triton_routes._route_and_block_quantize,
        route_tiles._tile_routes_kernel,
        triton_routes._swiglu_quantize,
        triton_routes._swiglu_chunk_max,
        triton_routes._quantize_swiglu_chunks,
        route_tiles._gather_route_rows_kernel,
    ):
        for tokens in (3, 40, 300):
            routes(*_inputs(SMALL, tokens, generator), True)
