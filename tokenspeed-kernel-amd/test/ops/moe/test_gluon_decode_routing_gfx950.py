from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel_amd.ops.moe import fused_mxfp_gfx950


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


requires_gfx950 = pytest.mark.skipif(
    not _is_gfx950(),
    reason="gluon decode routing kernel is gfx950 (CDNA4) only",
)

E = 128
TOPK = 4
# The small-M fused route regime (M <= SMALLM_MAX_M, the single-block collapse).
SMALL_M = [1, 2, 4, 8, 16]


def _route(logits):
    return fused_mxfp_gfx950.gluon_decode_routing_gfx950(
        logits,
        TOPK,
        sm_first=False,
        dtype=logits.dtype,
    )


# Keep the reference in torch. Forcing the generic routing pipeline here has
# crashed on gfx950 ROCm CI before this test reaches the Gluon output check.
def _reference_route(logits):
    M, E = logits.shape
    device = logits.device
    scores = logits.float().clone()
    rows = torch.arange(M, device=device)
    expert_ids = torch.arange(E, device=device, dtype=torch.int64)
    selected_indices = []
    selected_values = []

    for _ in range(TOPK):
        row_max = scores.max(dim=1, keepdim=True).values
        selected = (
            torch.where(
                scores == row_max,
                expert_ids.expand(M, E),
                torch.full((M, E), E, device=device, dtype=torch.int64),
            )
            .min(dim=1)
            .values
        )
        selected_indices.append(selected)
        selected_values.append(logits[rows, selected].float())
        scores[rows, selected] = -float("inf")

    expert_matrix = torch.stack(selected_indices, dim=1).to(torch.int32)
    gate_matrix = torch.softmax(torch.stack(selected_values, dim=1), dim=1).to(
        logits.dtype
    )
    flat_experts = expert_matrix.reshape(-1).long()
    flat_tokens = torch.arange(M, device=device, dtype=torch.int32).repeat_interleave(
        TOPK
    )
    flat_scatter = torch.arange(M * TOPK, device=device, dtype=torch.int32)
    flat_gates = gate_matrix.reshape(-1)

    slice_sizes = torch.bincount(flat_experts, minlength=E).to(torch.int32)
    slice_offs = torch.empty(E + 1, device=device, dtype=torch.int32)
    slice_offs[0] = 0
    slice_offs[1:] = torch.cumsum(slice_sizes, dim=0)

    gather = torch.empty(M * TOPK, device=device, dtype=torch.int32)
    scatter = torch.empty_like(gather)
    gate_scal = torch.empty(M * TOPK, device=device, dtype=logits.dtype)
    ranks = torch.zeros(E, device=device, dtype=torch.int32)
    for g in range(M * TOPK):
        expert = int(flat_experts[g].item())
        pos = int((slice_offs[expert] + ranks[expert]).item())
        gather[pos] = flat_tokens[g]
        scatter[pos] = flat_scatter[g]
        gate_scal[pos] = flat_gates[g]
        ranks[expert] += 1

    return slice_sizes, slice_offs, gather, scatter, gate_scal


def _assert_routing_matches_reference(rg, gather, scatter, gate_scal, logits):
    slice_sizes, slice_offs, gather_ref, scatter_ref, gate_ref = _reference_route(
        logits
    )
    assert torch.equal(rg.slice_sizes, slice_sizes)
    assert torch.equal(rg.slice_offs, slice_offs)

    active = slice_sizes > 0
    active_i32 = active.to(torch.int32)
    active_count = int(active_i32.sum().item())
    block_row = torch.empty_like(slice_offs)
    block_row[:-1] = torch.cumsum(active_i32, dim=0) - active_i32
    block_row[-1] = active_count
    block_offs_ref = block_row.expand_as(rg.block_offs_data)
    assert torch.equal(rg.block_offs_data, block_offs_ref)

    schedule_row = torch.full(
        (rg.block_schedule_data.shape[1],),
        -1,
        device=logits.device,
        dtype=torch.int32,
    )
    schedule_row[:active_count] = (
        torch.nonzero(active, as_tuple=False).flatten().to(torch.int32)
    )
    block_schedule_ref = schedule_row.expand_as(rg.block_schedule_data)
    assert torch.equal(rg.block_schedule_data, block_schedule_ref)

    assert torch.equal(gather, gather_ref)
    assert torch.equal(scatter, scatter_ref)
    torch.testing.assert_close(gate_scal.float(), gate_ref.float(), atol=1e-3, rtol=0)


@requires_gfx950
@pytest.mark.parametrize("M", SMALL_M)
def test_small_m_routing_matches_reference(M):
    gen = torch.Generator(device="cuda").manual_seed(100 + M)
    logits = torch.randn(M, E, device="cuda", dtype=torch.bfloat16, generator=gen)

    rg, gather, scatter, gate_scal = _route(
        logits
    )  # M <= SMALLM_MAX_M -> gluon fast path

    _assert_routing_matches_reference(rg, gather, scatter, gate_scal, logits)
    assert int(rg.slice_sizes.sum()) == M * TOPK


@requires_gfx950
def test_direct_large_m_rejects_fallback_shape():
    M = fused_mxfp_gfx950.SMALLM_MAX_M + 16
    gen = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.randn(M, E, device="cuda", dtype=torch.bfloat16, generator=gen)

    with pytest.raises(ValueError, match="fallback routing is handled"):
        fused_mxfp_gfx950.gluon_decode_routing_gfx950(
            logits,
            TOPK,
            sm_first=False,
            dtype=logits.dtype,
        )


@requires_gfx950
@pytest.mark.parametrize("M", SMALL_M)
def test_gluon_fused_route_direct(M):
    """gluon_fused_route returns a well-formed routing result for small M."""
    logits = torch.randn(M, E, device="cuda", dtype=torch.bfloat16)
    rg, gather, scatter, gate_scal = fused_mxfp_gfx950.gluon_fused_route(logits, TOPK)
    _assert_routing_matches_reference(rg, gather, scatter, gate_scal, logits)
    assert int(rg.slice_sizes.sum()) == M * TOPK
    assert gather.numel() == M * TOPK == scatter.numel() == gate_scal.numel()


@requires_gfx950
def test_gluon_route_supported_guards():
    """Unsupported configs report False so callers fall back safely."""
    logits = torch.randn(16, E, dtype=torch.bfloat16)
    assert fused_mxfp_gfx950.gluon_route_supported(logits, TOPK)
    # unsupported dtype
    assert not fused_mxfp_gfx950.gluon_route_supported(logits.to(torch.float64), TOPK)
    # non-2D
    assert not fused_mxfp_gfx950.gluon_route_supported(logits.reshape(1, 16, E), TOPK)
    # nonsensical topk
    assert not fused_mxfp_gfx950.gluon_route_supported(logits, E + 1)
