# Copyright (c) 2026 LightSeek Foundation

"""Parity + stream-safety tests for the in-house MoE block-aligned sorter.

These tests assert ``gluon_moe_sorting`` produces the block-aligned sorted
routing metadata expected by the gfx950 A4W4 package-prefill stage kernels
(byte-for-byte against an independent reference) and that it is CUDA-graph
capturable (i.e. performs no device-to-host synchronization), which is what
lets the package-prefill path run under graph capture.
"""

from __future__ import annotations

import pytest
import torch


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx950" in arch


if not _is_gfx950():
    pytest.skip(
        "AMD GFX950 is required for MoE sorting tests",
        allow_module_level=True,
    )


from tokenspeed_kernel_amd.ops.moe.gluon_a4w4_gfx950.moe_sorting import (  # noqa: E402
    gluon_moe_sorting,
)


def _ref_moe_sorting_native(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent reference for the block-aligned sort contract.

    Builds the same block-padded, per-expert sorted layout the package stage
    kernels consume, in plain PyTorch, so the kernel output can be checked
    byte-for-byte.
    """
    device = topk_ids.device
    M, topk = topk_ids.shape
    max_num_tokens_padded = topk_ids.numel() + num_experts * block_size - topk
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    init_val = topk << 24 | M
    sorted_ids = torch.full(
        (max_num_tokens_padded,), init_val, dtype=torch.int32, device=device
    )
    sorted_weights = torch.zeros(
        (max_num_tokens_padded,), dtype=torch.float32, device=device
    )
    sorted_expert_ids = torch.full(
        (max_num_m_blocks,), -1, dtype=torch.int32, device=device
    )
    num_valid = torch.empty((2,), dtype=torch.int32, device=device)

    slot = 0
    block = 0
    for expert in range(num_experts):
        token_id, topk_id = torch.where(topk_ids == expert)
        n = token_id.numel()
        n_blocks = (n + block_size - 1) // block_size
        n_pad = n_blocks * block_size
        sorted_ids[slot : slot + n] = (topk_id.to(torch.int32) << 24) | token_id.to(
            torch.int32
        )
        sorted_weights[slot : slot + n] = topk_weights[token_id, topk_id]
        slot += n_pad
        sorted_expert_ids[block : block + n_blocks] = expert
        block += n_blocks
    num_valid[0] = slot
    num_valid[1] = M
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid


def _random_topk(M, E, topk, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(M, E, device="cuda", generator=gen)
    weights, ids = torch.topk(logits.softmax(-1), topk, dim=-1)
    return ids.to(torch.int32).contiguous(), weights.to(torch.float32).contiguous()


@pytest.mark.parametrize(
    "M,E,topk,B",
    [
        (128, 8, 2, 128),  # small dense
        (4096, 32, 8, 128),  # medium expert count
        (4096, 384, 8, 128),  # Kimi prefill shape
        (1, 8, 2, 128),  # single token
        (37, 16, 3, 128),  # ragged / M not block-aligned
        (512, 256, 6, 64),  # large expert count, small block
        (2048, 128, 4, 32),  # small block
        (4096, 384, 8, 32),  # Kimi expert count, small block
    ],
)
def test_matches_ck_reference(M, E, topk, B):
    topk_ids, topk_weights = _random_topk(M, E, topk, seed=M * 31 + E)
    r_ids, r_w, r_eids, r_nv = _ref_moe_sorting_native(topk_ids, topk_weights, E, B)
    g_ids, g_w, g_eids, g_nv, g_out = gluon_moe_sorting(
        topk_ids, topk_weights, E, 32, torch.bfloat16, B
    )

    assert torch.equal(g_ids, r_ids), "sorted_ids mismatch"
    assert torch.equal(g_eids, r_eids), "sorted_expert_ids mismatch"
    assert torch.equal(g_nv, r_nv), "num_valid_ids mismatch"
    assert torch.allclose(g_w, r_w), "sorted_weights mismatch"
    assert g_out.shape == (M, 32) and g_out.dtype == torch.bfloat16


def test_unrouted_expert_ids_are_skipped():
    """``-1`` (unrouted) entries must be ignored, matching the CK reference."""
    topk_ids, topk_weights = _random_topk(256, 64, 4, seed=7)
    # Mark a handful of assignments as unrouted.
    topk_ids[::17, 0] = -1
    r_ids, r_w, r_eids, r_nv = _ref_moe_sorting_native(topk_ids, topk_weights, 64, 128)
    g_ids, g_w, g_eids, g_nv, _ = gluon_moe_sorting(
        topk_ids, topk_weights, 64, 32, torch.bfloat16, 128
    )
    assert torch.equal(g_ids, r_ids)
    assert torch.equal(g_eids, r_eids)
    assert torch.equal(g_nv, r_nv)
    assert torch.allclose(g_w, r_w)


def test_cuda_graph_capturable():
    """The sorter must run with no device-to-host sync (graph-capturable)."""
    M, E, topk, B = 4096, 384, 8, 128
    topk_ids, topk_weights = _random_topk(M, E, topk, seed=123)

    # Reference values (eager).
    e_ids, e_w, e_eids, e_nv, _ = gluon_moe_sorting(
        topk_ids, topk_weights, E, 32, torch.bfloat16, B
    )

    # Warmup on a side stream, then capture.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            gluon_moe_sorting(topk_ids, topk_weights, E, 32, torch.bfloat16, B)
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        c_ids, c_w, c_eids, c_nv, _ = gluon_moe_sorting(
            topk_ids, topk_weights, E, 32, torch.bfloat16, B
        )
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(c_ids, e_ids)
    assert torch.equal(c_eids, e_eids)
    assert torch.equal(c_nv, e_nv)
    assert torch.allclose(c_w, e_w)
