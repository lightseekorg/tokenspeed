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
from tokenspeed_kernel.numerics.torch.moe import torch_latent_input
from utils import assert_no_triton_compile, is_cdna5

pytest.importorskip(
    "tokenspeed_kernel_amd.ops.gfx1250.moe.fp16",
    reason="tokenspeed-kernel-amd is required for Gluon latent input tests",
)

from tokenspeed_kernel.ops.moe import latent_moe_input_projections  # noqa: E402
from tokenspeed_kernel_amd._triton import triton  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx1250.moe.fp16.latent_input_decode import (  # noqa: E402
    gluon_latent_input_decode_gfx1250,
    gluon_latent_input_decode_nosplit_gfx1250,
    launch_gluon_latent_input_decode_gfx1250,
)

HIDDEN = 7168
WIDTHS = (896, 3584, 1536)
GATE_CLAMP = 4.0
UP_CLAMP = 25.0

pytestmark = pytest.mark.skipif(
    not is_cdna5(), reason="requires the gfx1250 latent-input kernels"
)


def _weights():
    packed = torch.randn(sum(WIDTHS), HIDDEN, dtype=torch.bfloat16, device="cuda")
    return packed, list(packed.split(WIDTHS))


def _reference(hidden, views):
    return torch_latent_input(
        hidden_states=hidden,
        router_weight=views[0],
        routed_weight=views[1],
        shared_gate_up_weight=views[2],
        gate_clamp=GATE_CLAMP,
        up_clamp=UP_CLAMP,
    )


def _project(hidden, views, override):
    return latent_moe_input_projections(
        hidden,
        *views,
        gate_clamp=GATE_CLAMP,
        up_clamp=UP_CLAMP,
        override=override,
    )


# 7 leaves a partial row tile in the small-batch kernel; 16 and 17 straddle
# the change to the whole-reduction kernel; 33 leaves a partial row tile past
# its row-tile change; 64 is the top of the registered range.
@pytest.mark.parametrize("tokens", [1, 7, 16, 17, 32, 33, 64])
def test_decode_matches_reference(tokens: int) -> None:
    _packed, views = _weights()
    hidden = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    got = _project(hidden, views, "gluon_latent_input_decode_gfx1250")
    assert got[0].dtype == torch.float32
    for output, reference in zip(got, _reference(hidden, views), strict=True):
        torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)


def test_decode_batch_sizes_share_binaries() -> None:
    """Only the row-tile bucket specializes the decode kernels. Warm each
    bucket and integer class once; every other batch size reuses them."""
    _packed, views = _weights()
    hidden = torch.randn(64, HIDDEN, dtype=torch.bfloat16, device="cuda")

    def run(tokens):
        return _project(hidden[:tokens], views, "gluon_latent_input_decode_gfx1250")

    for tokens in (1, 2, 16, 17, 32, 33, 48):
        run(tokens)
    with assert_no_triton_compile(
        gluon_latent_input_decode_gfx1250, gluon_latent_input_decode_nosplit_gfx1250
    ):
        for tokens in (3, 4, 8, 15, 20, 31, 40, 63, 64):
            run(tokens)


def test_decode_replays_in_a_graph() -> None:
    """Replays reuse the arrival counters, so each launch must leave them
    rearmed; repeated replays agreeing bit for bit with eager runs shows that
    every tile was reduced exactly once."""
    _packed, views = _weights()
    hidden = torch.randn(4, HIDDEN, dtype=torch.bfloat16, device="cuda")

    def run():
        return _project(hidden, views, "gluon_latent_input_decode_gfx1250")

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for _ in range(3):
        hidden.copy_(torch.randn_like(hidden))
        graph.replay()
        for output, reference in zip(captured, run(), strict=True):
            torch.testing.assert_close(output, reference, atol=0.0, rtol=0.0)


# 2000 is not a multiple of the 256-row tile, so it exercises the clamped
# descriptor and the masked store that a whole number of tiles never reaches.
@pytest.mark.parametrize("tokens", [1025, 2000, 8192])
def test_prefill_matches_reference(tokens: int) -> None:
    _packed, views = _weights()
    hidden = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    got = _project(hidden, views, "gluon_latent_input_largem_gfx1250")
    assert got[0].dtype == torch.float32
    for output, reference in zip(got, _reference(hidden, views), strict=True):
        torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (1, "gluon_latent_input_decode_gfx1250"),
        (64, "gluon_latent_input_decode_gfx1250"),
        (1025, "gluon_latent_input_largem_gfx1250"),
        (8192, "gluon_latent_input_largem_gfx1250"),
    ],
)
def test_dispatch_selects_by_token_count(tokens: int, expected: str) -> None:
    """Two kernels over one shape do not agree bit for bit, so an exact match
    with the override is what shows dispatch chose this one."""
    _packed, views = _weights()
    hidden = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    automatic = latent_moe_input_projections(
        hidden, *views, gate_clamp=GATE_CLAMP, up_clamp=UP_CLAMP
    )
    for chosen, forced in zip(
        automatic, _project(hidden, views, expected), strict=True
    ):
        torch.testing.assert_close(chosen, forced, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("foreign", ["all", "tail"])
def test_launcher_rejects_a_packed_weight_that_is_not_the_views(
    foreign: str,
) -> None:
    """A partly aliasing set is the dangerous case: the router can match while
    the routed and shared weights belong to another allocation entirely."""
    packed, views = _weights()
    _other, other_views = _weights()
    if foreign == "all":
        views = other_views
    else:
        views = [packed[: WIDTHS[0]], other_views[1], other_views[2]]
    hidden = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="consecutive view"):
        launch_gluon_latent_input_decode_gfx1250(
            hidden, *views, packed, beta=GATE_CLAMP, linear_beta=UP_CLAMP
        )


def _expected(hidden, packed, router_n, latent_n, shared_n, beta, linear_beta):
    router, routed, shared = packed.split((router_n, latent_n, 2 * shared_n))
    return torch_latent_input(
        hidden_states=hidden,
        router_weight=router,
        routed_weight=routed,
        shared_gate_up_weight=shared,
        gate_clamp=beta,
        up_clamp=linear_beta,
    )


# The launcher pins the K3 shape, so these bypass it and set the dimensions
# themselves. That is what lets the sim cases exist: the same pipelines,
# region routing and SiTU over about a five-hundredth of the data, which is
# what the MI450 simulator affords.
_PIPELINE_SHAPES = [
    pytest.param(WIDTHS[0], WIDTHS[1], WIDTHS[2] // 2, HIDDEN, id="k3"),
    pytest.param(64, 128, 64, 512, id="sim"),
]


@pytest.mark.parametrize(("router_n", "latent_n", "shared_n", "k"), _PIPELINE_SHAPES)
def test_decode_small_batch_pipeline(
    router_n: int, latent_n: int, shared_n: int, k: int
) -> None:
    tokens, block_m, block_n, shared_split = 4, 16, 16, 2
    beta, linear_beta = 4.0, 25.0
    total_n = router_n + latent_n + 2 * shared_n
    blocks = shared_n // block_n

    hidden = torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
    packed = torch.randn(total_n, k, dtype=torch.bfloat16, device="cuda")
    partials = torch.empty(
        blocks * 2 * shared_split * block_m * block_n,
        dtype=torch.float32,
        device="cuda",
    )
    counters = torch.zeros(blocks, dtype=torch.int32, device="cuda")
    router = torch.empty(tokens, router_n, dtype=torch.float32, device="cuda")
    routed = torch.empty(tokens, latent_n, dtype=torch.bfloat16, device="cuda")
    shared = torch.empty(tokens, shared_n, dtype=torch.bfloat16, device="cuda")

    grid = blocks * 2 * shared_split + (router_n + latent_n) // block_n
    gluon_latent_input_decode_gfx1250[(grid,)](
        hidden,
        packed,
        partials,
        counters,
        router,
        routed,
        shared,
        beta,
        linear_beta,
        hidden.stride(0),
        packed.stride(0),
        tokens,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=128,
        NUM_BUFFERS=2,
        SHARED_SPLIT=shared_split,
        ROUTER_N=router_n,
        LATENT_N=latent_n,
        SHARED_N=shared_n,
        HAS_LINEAR_BETA=True,
        K=k,
        num_warps=1,
        num_stages=1,
        waves_per_eu=0,
    )

    expected = _expected(
        hidden, packed, router_n, latent_n, shared_n, beta, linear_beta
    )
    for output, reference in zip((router, routed, shared), expected, strict=True):
        torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)
    # The last arrival of every shared block rearms its counter.
    assert not counters.any()


@pytest.mark.parametrize(("router_n", "latent_n", "shared_n", "k"), _PIPELINE_SHAPES)
def test_decode_nosplit_pipeline(
    router_n: int, latent_n: int, shared_n: int, k: int
) -> None:
    tokens, block_m, half_n = 20, 32, 16
    beta, linear_beta = 4.0, 25.0
    total_n = router_n + latent_n + 2 * shared_n

    hidden = torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
    packed = torch.randn(total_n, k, dtype=torch.bfloat16, device="cuda")
    router = torch.empty(tokens, router_n, dtype=torch.float32, device="cuda")
    routed = torch.empty(tokens, latent_n, dtype=torch.bfloat16, device="cuda")
    shared = torch.empty(tokens, shared_n, dtype=torch.bfloat16, device="cuda")

    tiles = (router_n + latent_n) // (2 * half_n) + shared_n // half_n
    gluon_latent_input_decode_nosplit_gfx1250[(tiles, triton.cdiv(tokens, block_m))](
        hidden,
        packed,
        router,
        routed,
        shared,
        beta,
        linear_beta,
        hidden.stride(0),
        packed.stride(0),
        tokens,
        BLOCK_M=block_m,
        HALF_N=half_n,
        BLOCK_K=128,
        NUM_BUFFERS=2,
        WARP_BASES=((1, 0),),
        ROUTER_N=router_n,
        LATENT_N=latent_n,
        SHARED_N=shared_n,
        HAS_LINEAR_BETA=True,
        K=k,
        num_warps=2,
        num_stages=1,
        waves_per_eu=0,
    )

    expected = _expected(
        hidden, packed, router_n, latent_n, shared_n, beta, linear_beta
    )
    for output, reference in zip((router, routed, shared), expected, strict=True):
        torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)
