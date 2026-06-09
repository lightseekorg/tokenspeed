from __future__ import annotations

import pytest
import tokenspeed_kernel  # noqa: F401  (registers moe kernels)
import tokenspeed_kernel.ops.moe.gluon as gluon_mod
import torch
from tokenspeed_kernel.ops.moe import moe_route
from tokenspeed_kernel.ops.moe.gluon import (
    SMALLM_MAX_M,
    gluon_fused_route,
    gluon_route_supported,
)
from tokenspeed_kernel.platform import current_platform

requires_gfx950 = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform().is_cdna4),
    reason="gluon decode routing kernel is gfx950 (CDNA4) only",
)

E = 128
TOPK = 4
# The small-M fused route regime (M <= SMALLM_MAX_M, the single-block collapse).
SMALL_M = [1, 2, 4, 8, 16]


def _route(logits):
    return moe_route(
        logits,
        TOPK,
        sm_first=False,
        dtype=logits.dtype,
        traits={"output_type": "ragged_metadata"},
        expected_kernel_name="gluon_decode_routing_gfx950",
    )


def _route_generic(logits):
    """Reference: force the generic pipeline by disabling the small-M bound.

    The gfx950 route kernel is still selected; setting its bound below 1 makes
    it fall back to the registered triton_kernels_routing generic pipeline.
    """
    saved = gluon_mod.SMALLM_MAX_M
    gluon_mod.SMALLM_MAX_M = -1
    try:
        return _route(logits)
    finally:
        gluon_mod.SMALLM_MAX_M = saved


def _assert_metadata_exact(rg, rg_ref):
    assert torch.equal(rg.slice_sizes, rg_ref.slice_sizes)
    assert torch.equal(rg.slice_offs, rg_ref.slice_offs)
    assert torch.equal(rg.block_offs_data, rg_ref.block_offs_data)
    assert torch.equal(rg.block_schedule_data, rg_ref.block_schedule_data)


@requires_gfx950
@pytest.mark.parametrize("M", SMALL_M)
def test_small_m_routing_matches_generic(M):
    """Small-M Gluon route == generic pipeline, bit-for-bit.

    For M <= SMALLM_MAX_M the placement is stable, so not only the
    order-independent RaggedTensorMetadata but also the gather/scatter/gate
    index tensors are bit-identical to the generic pipeline.
    """
    gen = torch.Generator(device="cuda").manual_seed(100 + M)
    logits = torch.randn(M, E, device="cuda", dtype=torch.bfloat16, generator=gen)

    rg_ref, ga_ref, sc_ref, gs_ref = _route_generic(logits)
    rg, ga, sc, gs = _route(logits)  # M <= SMALLM_MAX_M -> gluon fast path

    _assert_metadata_exact(rg, rg_ref)
    assert int(rg.slice_sizes.sum()) == M * TOPK
    assert torch.equal(ga, ga_ref)
    assert torch.equal(sc, sc_ref)
    assert torch.allclose(gs.float(), gs_ref.float(), atol=1e-3)


@requires_gfx950
def test_large_m_uses_generic_pipeline():
    """M > SMALLM_MAX_M falls through to the generic pipeline unchanged."""
    M = SMALLM_MAX_M + 16
    gen = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.randn(M, E, device="cuda", dtype=torch.bfloat16, generator=gen)

    rg, ga, _sc, _gs = _route(logits)
    rg_ref, ga_ref, _sc_ref, _gs_ref = _route_generic(logits)
    _assert_metadata_exact(rg, rg_ref)
    assert torch.equal(ga, ga_ref)
    assert int(rg.slice_sizes.sum()) == M * TOPK


@requires_gfx950
@pytest.mark.parametrize("M", SMALL_M)
def test_gluon_fused_route_direct(M):
    """gluon_fused_route returns a well-formed routing result for small M."""
    logits = torch.randn(M, E, device="cuda", dtype=torch.bfloat16)
    rg, ga, sc, gs = gluon_fused_route(logits, TOPK)
    assert int(rg.slice_sizes.sum()) == M * TOPK
    assert ga.numel() == M * TOPK == sc.numel() == gs.numel()


@requires_gfx950
def test_gluon_route_supported_guards():
    """Unsupported configs report False so callers fall back safely."""
    logits = torch.randn(16, E, dtype=torch.bfloat16)
    assert gluon_route_supported(logits, TOPK)
    # unsupported dtype
    assert not gluon_route_supported(logits.to(torch.float64), TOPK)
    # non-2D
    assert not gluon_route_supported(logits.reshape(1, 16, E), TOPK)
    # nonsensical topk
    assert not gluon_route_supported(logits, E + 1)
