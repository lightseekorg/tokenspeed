"""Capture-safe indexed KDA recurrent-state writes."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention import kda_state_scatter


def _expected_write(
    pool: torch.Tensor,
    updates: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    expected = pool.clone()
    for row, slot in enumerate(indices.cpu().tolist()):
        if 0 <= slot < expected.shape[0]:
            expected[slot] = updates[row]
    return expected


def test_kda_state_scatter_skips_invalid_indices_and_supports_cuda_graph() -> None:
    """Graph capture must preserve ``-1`` sentinel rows and replay live writes."""

    device = "cuda"
    torch.manual_seed(7)
    pool = torch.randn(5, 3, 8, 8, device=device, dtype=torch.float32)
    capture_updates = torch.randn(3, 3, 8, 8, device=device, dtype=torch.float32)
    replay_updates = torch.randn_like(capture_updates)
    indices = torch.full((3,), -1, device=device, dtype=torch.int32)

    # Compile the Triton kernel outside capture, then restore the pool so the
    # captured all-invalid input can prove that no slot is dirtied.
    kda_state_scatter(pool, capture_updates, indices)
    torch.cuda.synchronize()
    initial_pool = pool.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kda_state_scatter(pool, capture_updates, indices)
    torch.cuda.synchronize()
    torch.testing.assert_close(pool, initial_pool)

    indices.copy_(torch.tensor([3, -1, 1], device=device, dtype=torch.int32))
    capture_updates.copy_(replay_updates)
    expected = _expected_write(initial_pool, replay_updates, indices)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(pool, expected)
