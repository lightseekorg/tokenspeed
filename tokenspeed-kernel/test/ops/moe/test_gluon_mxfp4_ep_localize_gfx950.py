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

"""Expert-parallel id localization for the Gluon MXFP4 swiglu MoE apply.

The Gluon precomputed router drops expert ids outside ``[0, num_experts)``, so
expert parallelism needs only a global-to-local remap ahead of it rather than a
separate EP kernel. These tests pin both halves of that claim: the remap itself,
and the router's behaviour that makes it sound.
"""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip(
        "AMD CDNA4 is required for Gluon MXFP4 EP localization tests",
        allow_module_level=True,
    )

from tokenspeed_kernel.ops.moe.gluon.mxfp4 import (  # noqa: E402
    _localize_topk_ids_for_ep,
)
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4.fused.routing import (  # noqa: E402
    gluon_precomputed_topk_fused_route,
)

GLOBAL_EXPERTS = 256
EP_SIZE = 8
LOCAL_EXPERTS = GLOBAL_EXPERTS // EP_SIZE
TOP_K = 6


class _Weights(torch.nn.Module):
    def __init__(self, ep_size: int, ep_rank: int, num_local_experts: int) -> None:
        super().__init__()
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.num_local_experts = num_local_experts


def _ids(num_tokens: int, device: torch.device) -> torch.Tensor:
    return torch.randint(
        0, GLOBAL_EXPERTS, (num_tokens, TOP_K), device=device, dtype=torch.int32
    )


@pytest.mark.parametrize("ep_rank", [0, 3, EP_SIZE - 1])
def test_localize_maps_only_this_rank(ep_rank: int) -> None:
    """Ids in this rank's window become local; every other id becomes -1."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    ids = _ids(64, device)
    w = _Weights(EP_SIZE, ep_rank, LOCAL_EXPERTS)

    local = _localize_topk_ids_for_ep(w, ids)

    start = ep_rank * LOCAL_EXPERTS
    owned = (ids >= start) & (ids < start + LOCAL_EXPERTS)
    torch.testing.assert_close(local[owned], (ids - start)[owned], rtol=0, atol=0)
    assert bool((local[~owned] == -1).all())
    assert local.dtype == torch.int32


def test_every_pair_is_owned_by_exactly_one_rank() -> None:
    """Summed over ranks, EP must dispatch each pair once and only once."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    ids = _ids(128, device)

    kept = torch.zeros_like(ids, dtype=torch.int32)
    for ep_rank in range(EP_SIZE):
        w = _Weights(EP_SIZE, ep_rank, LOCAL_EXPERTS)
        kept += (_localize_topk_ids_for_ep(w, ids) >= 0).to(torch.int32)

    assert bool((kept == 1).all()), "each (token, slot) must land on one rank"


def test_disabled_when_ep_is_off() -> None:
    device = torch.device("cuda")
    ids = _ids(16, device)
    w = _Weights(1, 0, GLOBAL_EXPERTS)
    assert _localize_topk_ids_for_ep(w, ids) is ids


def test_router_drops_non_local_ids() -> None:
    """The remap is only sound because the router skips out-of-range ids."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    num_tokens = 8
    ids = torch.randint(
        0, LOCAL_EXPERTS, (num_tokens, TOP_K), device=device, dtype=torch.int32
    )
    weights = torch.rand(num_tokens, TOP_K, device=device, dtype=torch.bfloat16)

    def routed(expert_ids: torch.Tensor) -> int:
        meta, _, _, _ = gluon_precomputed_topk_fused_route(
            weights, expert_ids.contiguous(), LOCAL_EXPERTS, dtype=torch.bfloat16
        )
        torch.cuda.synchronize()
        return int(meta.slice_sizes.sum().item())

    assert routed(ids) == num_tokens * TOP_K

    masked = ids.clone()
    masked[:, 3:] = -1
    assert routed(masked) == num_tokens * 3

    assert routed(torch.full_like(ids, -1)) == 0


def test_localized_ids_route_to_the_expected_count() -> None:
    """End to end: remapped ids dispatch exactly this rank's share."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    # The fused route's small-batch path requires num_tokens * top_k <= 64.
    num_tokens = 8
    ids = _ids(num_tokens, device)
    weights = torch.rand(num_tokens, TOP_K, device=device, dtype=torch.bfloat16)

    total = 0
    for ep_rank in range(EP_SIZE):
        w = _Weights(EP_SIZE, ep_rank, LOCAL_EXPERTS)
        local = _localize_topk_ids_for_ep(w, ids)
        meta, _, _, _ = gluon_precomputed_topk_fused_route(
            weights, local.contiguous(), LOCAL_EXPERTS, dtype=torch.bfloat16
        )
        torch.cuda.synchronize()
        routed = int(meta.slice_sizes.sum().item())
        assert routed == int((local >= 0).sum().item())
        total += routed

    assert total == num_tokens * TOP_K


def _mxfp4_module(num_experts: int, latent: int, inter: int, top_k: int):
    from utils import make_mxfp4_moe_weights

    g = torch.Generator(device="cuda").manual_seed(11)
    raw = make_mxfp4_moe_weights(num_experts, latent, inter, g)
    m = torch.nn.Module()
    m.w13_weight = torch.nn.Parameter(raw["w13_weight"], requires_grad=False)
    m.w13_weight_scale = torch.nn.Parameter(raw["w13_scale"], requires_grad=False)
    m.w2_weight = torch.nn.Parameter(raw["w2_weight"], requires_grad=False)
    m.w2_weight_scale = torch.nn.Parameter(raw["w2_scale"], requires_grad=False)
    m.top_k = top_k
    m.num_experts = num_experts
    m.num_local_experts = num_experts
    m.ep_size = 1
    m.ep_rank = 0
    return m, g


@pytest.mark.parametrize("unrouted", ["all", "partial", "all_but_one"])
def test_unrouted_pairs_contribute_zero(unrouted: str) -> None:
    """Rows for unrouted (token, slot) pairs must be zero, not stale memory.

    The combine only stores rows whose expert id was routed, so the output
    buffer has to start zeroed. Before that fix these cases returned whatever
    the allocator held -- values around 1e35 -- which made expert parallelism
    unusable, since under EP most pairs are owned by other ranks.
    """
    import tokenspeed_kernel

    num_experts, latent, inter, top_k, tokens = 8, 512, 512, 6, 8
    module, g = _mxfp4_module(num_experts, latent, inter, top_k)
    x = (
        torch.randn((tokens, latent), dtype=torch.bfloat16, device="cuda", generator=g)
        * 0.1
    )
    weights = torch.softmax(
        torch.randn((tokens, top_k), dtype=torch.float32, device="cuda", generator=g),
        dim=-1,
    )
    logits = torch.zeros((tokens, num_experts), dtype=torch.float32, device="cuda")

    ids = torch.full((tokens, top_k), -1, dtype=torch.int32, device="cuda")
    if unrouted == "partial":
        ids[:, :3] = torch.randint(
            0, num_experts, (tokens, 3), device="cuda", dtype=torch.int32
        )
    elif unrouted == "all_but_one":
        ids[0, 0] = 0

    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="swiglu",
        routing_mode="precomputed_topk",
        ispp=inter,
        internal_activation_dtype="input",
        solution="gluon",
    )
    tokenspeed_kernel.moe_process_weights(plan, module)
    out = tokenspeed_kernel.moe_apply(
        plan, x, module, logits, topk_weights=weights, topk_ids=ids
    ).float()

    assert torch.isfinite(out).all(), "unrouted rows produced non-finite values"
    # Activations are ~0.1 scale, so anything past 1.0 is stale memory rather
    # than a plausible expert output.
    assert (
        out.abs().max().item() < 1.0
    ), f"unrouted rows leaked stale memory: |max|={out.abs().max().item():.6g}"
    if unrouted == "all":
        assert out.abs().max().item() == 0.0, "nothing routed must give exactly zero"


@pytest.mark.parametrize("num_tokens", [8, 64, 512])
def test_ep_partials_sum_to_the_all_expert_reference(num_tokens: int) -> None:
    """Summed rank partials must equal one call holding every expert.

    ``num_tokens`` spans the routing split: the fused small-M route handles
    roughly ten tokens or fewer, everything above it takes the ragged route.
    Both must place unrouted pairs outside the per-expert slices, which is what
    makes the sum over ranks equal the whole.
    """
    import tokenspeed_kernel

    latent, inter, top_k = 512, 512, TOP_K
    num_experts, ep_size = 64, 8
    local = num_experts // ep_size

    g = torch.Generator(device="cuda").manual_seed(3)
    from utils import make_mxfp4_moe_weights

    raw = make_mxfp4_moe_weights(num_experts, latent, inter, g)
    x = (
        torch.randn(
            (num_tokens, latent), dtype=torch.bfloat16, device="cuda", generator=g
        )
        * 0.1
    )
    ids = torch.stack(
        [
            torch.randperm(num_experts, device="cuda", generator=g)[:top_k].to(
                torch.int32
            )
            for _ in range(num_tokens)
        ]
    )
    weights = torch.softmax(
        torch.randn(
            (num_tokens, top_k), dtype=torch.float32, device="cuda", generator=g
        ),
        dim=-1,
    )
    logits = torch.zeros((num_tokens, num_experts), dtype=torch.float32, device="cuda")

    def build(lo: int, hi: int, ep: int, rank: int):
        m = torch.nn.Module()
        m.w13_weight = torch.nn.Parameter(
            raw["w13_weight"][lo:hi].clone(), requires_grad=False
        )
        m.w13_weight_scale = torch.nn.Parameter(
            raw["w13_scale"][lo:hi].clone(), requires_grad=False
        )
        m.w2_weight = torch.nn.Parameter(
            raw["w2_weight"][lo:hi].clone(), requires_grad=False
        )
        m.w2_weight_scale = torch.nn.Parameter(
            raw["w2_scale"][lo:hi].clone(), requires_grad=False
        )
        m.top_k = top_k
        m.num_experts = num_experts
        m.num_local_experts = hi - lo
        m.ep_size = ep
        m.ep_rank = rank
        return m

    def apply(module, ep: int):
        plan = tokenspeed_kernel.moe_plan(
            "mxfp4",
            input_dtype=torch.bfloat16,
            activation="swiglu",
            routing_mode="precomputed_topk",
            ep_size=(ep if ep > 1 else None),
            ispp=inter,
            internal_activation_dtype="input",
            solution="gluon",
        )
        tokenspeed_kernel.moe_process_weights(plan, module)
        return tokenspeed_kernel.moe_apply(
            plan, x, module, logits, topk_weights=weights, topk_ids=ids
        ).float()

    reference = apply(build(0, num_experts, 1, 0), 1)
    total = torch.zeros_like(reference)
    for rank in range(ep_size):
        total += apply(build(rank * local, (rank + 1) * local, ep_size, rank), ep_size)

    peak = reference.abs().max().item()
    err = (total - reference).abs().max().item()
    assert err <= 2e-2 * peak + 1e-4, f"max_abs={err} peak={peak}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
