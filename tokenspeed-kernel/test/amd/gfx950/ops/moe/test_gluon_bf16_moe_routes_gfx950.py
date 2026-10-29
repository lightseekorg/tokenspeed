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
        "AMD CDNA4 is required for the Gluon BF16 expert-route tests",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.moe import (  # noqa: E402
    _bf16_route_offsets_fit_int32,
    _select_bf16_routes_kernel,
    bf16_moe_routes,
    bf16_moe_routes_warmup,
)
from tokenspeed_kernel.ops.moe._triton import bf16 as triton_bf16  # noqa: E402
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel  # noqa: E402
from tokenspeed_kernel.signature import dense_format_signature  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.fp16 import (  # noqa: E402
    routes_decode,
    routes_prefill,
)

HIDDEN = 1024
INTERMEDIATE = 512
TOPK = 8


def _weights(experts: int, hidden: int, intermediate: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w13 = torch.randn((experts, 2 * intermediate, hidden), generator=gen, device="cuda")
    w2 = torch.randn((experts, hidden, intermediate), generator=gen, device="cuda")
    return (w13 * 0.03).bfloat16(), (w2 * 0.03).bfloat16()


def _ids(tokens: int, global_experts: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand((tokens, global_experts), generator=gen, device="cuda")
    return scores.topk(TOPK, dim=-1).indices.to(torch.int32)


def _reference(x, w13, w2, ids, expert_start):
    """FP32 expert outputs per route, with the kernels' BF16 intermediate."""
    experts, two_i, _ = w13.shape
    intermediate = two_i // 2
    out = torch.zeros((ids.numel(), x.shape[1]), dtype=torch.float32, device="cuda")
    for route, expert in enumerate(ids.view(-1).tolist()):
        local = expert - expert_start
        if 0 <= local < experts:
            gate_up = x[route // ids.shape[1]].float() @ w13[local].float().T
            h = (
                torch.nn.functional.silu(gate_up[:intermediate])
                * gate_up[intermediate:]
            )
            out[route] = h.bfloat16().float() @ w2[local].float().T
    return out.view(*ids.shape, x.shape[1])


def _assert_routes(actual, expected):
    scale = expected.abs().max().item()
    torch.testing.assert_close(actual.float(), expected, atol=4e-3 * scale, rtol=1.6e-2)


@pytest.mark.parametrize(
    ("hidden", "intermediate", "topk", "fits_int32", "expected"),
    [
        (HIDDEN, INTERMEDIATE, TOPK, True, "gluon_bf16_moe_routes_gfx950"),
        # Hidden sizes off the zero kernel's 512-column block, intermediate
        # sizes off the 256-wide GEMM tiles, other top-k widths and route
        # buffers past 32-bit offsets keep Triton.
        (640, INTERMEDIATE, TOPK, True, "triton_bf16_moe_routes"),
        (HIDDEN, 96, TOPK, True, "triton_bf16_moe_routes"),
        (HIDDEN, INTERMEDIATE, 6, True, "triton_bf16_moe_routes"),
        (HIDDEN, INTERMEDIATE, TOPK, False, "triton_bf16_moe_routes"),
    ],
)
def test_public_routes_selection(
    hidden, intermediate, topk, fits_int32, expected
) -> None:
    # The traits bf16_moe_routes derives from its tensors.
    kernel = select_kernel(
        "moe",
        "routes",
        dense_format_signature(("x",), (torch.bfloat16,)),
        traits={
            "buffer_offsets_fit_int32": fits_int32,
            "hidden": hidden,
            "ispp": intermediate,
            "topk": topk,
        },
    )
    assert kernel.name == expected


def test_route_offsets_bound_bytes() -> None:
    # BF16 routes reach 2 GiB at 2**30 elements: 2**17 tokens * 8 * 1024 * 2 B.
    assert _bf16_route_offsets_fit_int32(2**17 - 1, TOPK, HIDDEN, INTERMEDIATE, 2)
    assert not _bf16_route_offsets_fit_int32(2**17, TOPK, HIDDEN, INTERMEDIATE, 2)
    # One expert's gate/up weights bound the weight offsets.
    assert not _bf16_route_offsets_fit_int32(1, TOPK, 16384, 32768, 2)


def test_bitwise_routes_need_a_batch_invariant_provider() -> None:
    """Neither route kernel declares batch invariance, so bitwise selection fails."""
    w13, w2 = _weights(4, HIDDEN, INTERMEDIATE, seed=7)
    x = torch.zeros((2, HIDDEN), device="cuda", dtype=torch.bfloat16)
    ids = torch.zeros((2, TOPK), device="cuda", dtype=torch.int32)
    with pytest.raises(NoKernelFoundError):
        bf16_moe_routes(x, w13, w2, ids, 0, batch_invariant=True)


# Decode (<= 16 tokens) and grouped prefill with 128- and 256-row expert tiles.
@pytest.mark.parametrize(
    ("tokens", "experts", "global_experts", "expert_start"),
    [
        (1, 16, 16, 0),
        (5, 16, 64, 16),
        (16, 16, 16, 0),
        (17, 16, 32, 0),
        (300, 16, 64, 32),
        (160, 4, 8, 0),
        (400, 4, 8, 4),
    ],
)
def test_routes_match_reference(tokens, experts, global_experts, expert_start) -> None:
    w13, w2 = _weights(experts, HIDDEN, INTERMEDIATE, seed=tokens)
    x = (torch.randn((tokens, HIDDEN), device="cuda") * 0.5).bfloat16()
    ids = _ids(tokens, global_experts, seed=tokens + 1)
    out = torch.full((tokens * TOPK, HIDDEN), float("nan"), device="cuda").bfloat16()

    routes = bf16_moe_routes(x, w13, w2, ids, expert_start, out, batch_invariant=False)

    assert routes.data_ptr() == out.data_ptr()
    expected = _reference(x, w13, w2, ids, expert_start)
    _assert_routes(routes, expected)
    absent = (ids < expert_start) | (ids >= expert_start + experts)
    assert not routes[absent].any()


@pytest.mark.parametrize("tokens", [3, 40])
def test_token_selecting_one_expert_twice_fills_both_routes(tokens) -> None:
    w13, w2 = _weights(4, HIDDEN, INTERMEDIATE, seed=7)
    x = (torch.randn((tokens, HIDDEN), device="cuda") * 0.5).bfloat16()
    ids = torch.randint(0, 6, (tokens, TOPK), device="cuda", dtype=torch.int32)
    ids[0, :2] = 1

    routes = bf16_moe_routes(x, w13, w2, ids, 0, batch_invariant=False)

    _assert_routes(routes, _reference(x, w13, w2, ids, 0))


# 2 tokens take the decode kernels, 17 the prefill stages.
@pytest.mark.parametrize("tokens", [2, 17])
def test_large_expert_offset_addresses_beyond_int32(tokens: int) -> None:
    experts, hidden, intermediate = 80, 4096, 4096
    w13, w2 = _weights(experts, hidden, intermediate, seed=3)
    x = (torch.randn((tokens, hidden), device="cuda") * 0.5).bfloat16()
    ids = torch.full((tokens, TOPK), experts + 5, device="cuda", dtype=torch.int32)
    ids[0, 0] = experts
    ids[1, 3] = 1

    routes = bf16_moe_routes(x, w13, w2, ids, 1, batch_invariant=False)

    _assert_routes(routes, _reference(x, w13, w2, ids, 1))


def test_graph_replay_follows_fresh_ids() -> None:
    experts = 16
    w13, w2 = _weights(experts, HIDDEN, INTERMEDIATE, seed=11)
    for tokens in (4, 48):
        x = (torch.randn((tokens, HIDDEN), device="cuda") * 0.5).bfloat16()
        ids = _ids(tokens, 2 * experts, seed=tokens)
        out = torch.empty((tokens * TOPK, HIDDEN), device="cuda").bfloat16()
        bf16_moe_routes(x, w13, w2, ids, 0, out, batch_invariant=False)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bf16_moe_routes(x, w13, w2, ids, 0, out, batch_invariant=False)
        for seed in (1, 2):
            ids.copy_(_ids(tokens, 2 * experts, seed=100 * tokens + seed))
            x.copy_((torch.randn((tokens, HIDDEN), device="cuda") * 0.5).bfloat16())
            graph.replay()
            eager = bf16_moe_routes(x, w13, w2, ids, 0, batch_invariant=False)
            assert torch.equal(out.view_as(eager), eager)


def test_warmup_covers_token_counts() -> None:
    experts = 16
    w13, w2 = _weights(experts, HIDDEN, INTERMEDIATE, seed=5)
    bf16_moe_routes_warmup(w13, w2, TOPK, 0, max_num_tokens=4096, batch_invariant=False)
    kernels = (
        routes_decode.gluon_bf16_moe_routes_decode_stage1_gfx950,
        routes_decode.gluon_bf16_moe_routes_decode_stage2_gfx950,
        routes_prefill.gluon_bf16_moe_routes_prefill_stage1_gfx950,
        routes_prefill.gluon_bf16_moe_routes_prefill_stage2_gfx950,
        routes_prefill.gluon_bf16_moe_routes_zero_gfx950,
        routes_prefill.gluon_bf16_moe_routes_count_gfx950,
        routes_prefill.gluon_bf16_moe_routes_offsets_gfx950,
        routes_prefill.gluon_bf16_moe_routes_scatter_gfx950,
    )
    with assert_no_triton_compile(*kernels):
        for tokens in (1, 3, 7, 12, 16, 19, 33, 64, 97, 256, 999, 4096):
            x = torch.randn((tokens, HIDDEN), device="cuda").bfloat16()
            bf16_moe_routes(
                x, w13, w2, _ids(tokens, 2 * experts, tokens), 0, batch_invariant=False
            )


def test_warmup_compiles_the_large_batch_kernel() -> None:
    experts = 16
    w13, w2 = _weights(experts, HIDDEN, INTERMEDIATE, seed=6)
    # 2**18 tokens (twice the 2**17 boundary) put the routes past 2 GiB.
    bf16_moe_routes_warmup(
        w13, w2, TOPK, 0, max_num_tokens=2**18, batch_invariant=False
    )
    kernel = _select_bf16_routes_kernel(
        torch.bfloat16,
        HIDDEN,
        INTERMEDIATE,
        TOPK,
        buffer_offsets_fit_int32=False,
        batch_invariant=False,
    )
    assert kernel.name == "triton_bf16_moe_routes"
    with assert_no_triton_compile(
        triton_bf16._stage1_kernel, triton_bf16._stage2_kernel
    ):
        for tokens in (20, 33, 48, 100):
            x = torch.randn((tokens, HIDDEN), device="cuda").bfloat16()
            kernel(x, w13, w2, _ids(tokens, 2 * experts, tokens), 0, output=None)
