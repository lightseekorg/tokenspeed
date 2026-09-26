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
from utils import is_cdna5

pytest.importorskip(
    "tokenspeed_kernel_amd.ops.gfx1250.moe.fp16",
    reason="tokenspeed-kernel-amd is required for Gluon latent input tests",
)

from tokenspeed_kernel.ops.moe import latent_moe_input_projections  # noqa: E402
from tokenspeed_kernel_amd._triton import triton  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx1250.moe.fp16.latent_input_decode import (  # noqa: E402
    gluon_latent_input_decode_epilogue_gfx1250,
    gluon_latent_input_decode_gfx1250,
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
    router = torch.nn.functional.linear(hidden.float(), views[0].float())
    routed = torch.nn.functional.linear(hidden, views[1])
    gate, up = torch.nn.functional.linear(hidden, views[2]).chunk(2, dim=-1)
    shared = (
        GATE_CLAMP
        * torch.tanh(gate.float() / GATE_CLAMP)
        * torch.sigmoid(gate.float())
        * (UP_CLAMP * torch.tanh(up.float() / UP_CLAMP))
    ).to(hidden.dtype)
    return router, routed, shared


def _project(hidden, views, override):
    return latent_moe_input_projections(
        hidden,
        *views,
        gate_clamp=GATE_CLAMP,
        up_clamp=UP_CLAMP,
        override=override,
    )


# 7 and 17 straddle the split-k change at sixteen tokens and leave a partial
# row tile; 32 is the top of the registered range.
@pytest.mark.parametrize("tokens", [1, 7, 16, 17, 32])
def test_decode_matches_reference(tokens: int) -> None:
    _packed, views = _weights()
    hidden = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    got = _project(hidden, views, "gluon_latent_input_decode_gfx1250")
    assert got[0].dtype == torch.float32
    for output, reference in zip(got, _reference(hidden, views), strict=True):
        torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)


def test_decode_replays_in_a_graph() -> None:
    _packed, views = _weights()
    hidden = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device="cuda")

    def run():
        return _project(hidden, views, "gluon_latent_input_decode_gfx1250")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    hidden.copy_(torch.randn_like(hidden))
    graph.replay()
    for output, reference in zip(captured, run(), strict=True):
        torch.testing.assert_close(output, reference, atol=2e-2, rtol=2e-2)


# 2000 is not a multiple of the 256-row tile, so it exercises the clamped
# descriptor and the masked store that a whole number of tiles never reaches.
@pytest.mark.parametrize("tokens", [1536, 2000, 8192])
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
        (32, "gluon_latent_input_decode_gfx1250"),
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


@pytest.mark.parametrize(
    ("router_n", "latent_n", "shared_n", "k", "split_k"),
    [
        pytest.param(WIDTHS[0], WIDTHS[1], WIDTHS[2] // 2, HIDDEN, 8, id="k3"),
        pytest.param(64, 128, 64, 256, 2, id="sim"),
    ],
)
def test_decode_pipeline(
    router_n: int, latent_n: int, shared_n: int, k: int, split_k: int
) -> None:
    """The launcher pins the K3 shape, so these bypass it and set the
    dimensions themselves. That is what lets the sim case exist: the same
    split reduction, WMMA pipeline, region routing and SiTU over about a
    five-hundredth of the data, which is what the MI450 simulator affords."""
    total_n = router_n + latent_n + 2 * shared_n
    tokens, block_m, block_n = 4, 16, 64
    beta, linear_beta = 4.0, 25.0

    hidden = torch.randn(tokens, k, dtype=torch.bfloat16, device="cuda")
    packed = torch.randn(total_n, k, dtype=torch.bfloat16, device="cuda")
    partials = torch.empty(split_k, tokens, total_n, dtype=torch.float32, device="cuda")
    router = torch.empty(tokens, router_n, dtype=torch.float32, device="cuda")
    routed = torch.empty(tokens, latent_n, dtype=torch.bfloat16, device="cuda")
    shared = torch.empty(tokens, shared_n, dtype=torch.bfloat16, device="cuda")

    grid = (split_k * (total_n // block_n), triton.cdiv(tokens, block_m))
    gluon_latent_input_decode_gfx1250[grid](
        hidden,
        packed,
        partials,
        hidden.stride(0),
        hidden.stride(1),
        packed.stride(0),
        packed.stride(1),
        tokens * total_n,
        tokens,
        BLOCK_N=block_n,
        BLOCK_K=128,
        NUM_BUFFERS=3,
        SPLIT_K=split_k,
        TOTAL_N=total_n,
        K=k,
        num_warps=4,
        num_stages=1,
        waves_per_eu=0,
    )
    gluon_latent_input_decode_epilogue_gfx1250[
        (tokens, triton.cdiv(router_n + latent_n + shared_n, block_n))
    ](
        partials,
        router,
        routed,
        shared,
        beta,
        linear_beta,
        tokens * total_n,
        SPLIT_K=split_k,
        TOTAL_N=total_n,
        ROUTER_N=router_n,
        LATENT_N=latent_n,
        SHARED_N=shared_n,
        HAS_LINEAR_BETA=True,
        BLOCK_N=block_n,
        num_warps=4,
    )

    acc = hidden.float() @ packed.float().T
    gate = acc[:, router_n + latent_n : router_n + latent_n + shared_n]
    up = acc[:, router_n + latent_n + shared_n :]
    gate = gate.to(torch.bfloat16).float()
    up = up.to(torch.bfloat16).float()
    expected_shared = (
        beta
        * torch.tanh(gate / beta)
        * torch.sigmoid(gate)
        * (linear_beta * torch.tanh(up / linear_beta))
    ).to(torch.bfloat16)
    torch.testing.assert_close(router, acc[:, :router_n], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        routed,
        acc[:, router_n : router_n + latent_n].to(torch.bfloat16),
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(shared, expected_shared, atol=2e-2, rtol=2e-2)
