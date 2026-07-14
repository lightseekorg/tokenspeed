# Copyright (c) 2026 LightSeek Foundation
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton
from tokenspeed_kernel.ops.moe.triton.mxfp4 import _routing_from_topk
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.triton.minimax_m3 import (
    minimax_m3_reduce_topk,
    minimax_m3_route_counts,
    minimax_m3_route_order,
)

with redirect_triton_to_tokenspeed_triton():
    import triton_kernels.tensor  # noqa: F401

from triton_kernels.tensor import make_ragged_tensor_metadata


pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia,
    reason="MiniMax-M3 native routing targets NVIDIA GPUs.",
)


@pytest.mark.parametrize("num_tokens", [0, 1, 33, 129, 512])
def test_minimax_m3_route_metadata_matches_stable_reference(
    device: str,
    num_tokens: int,
) -> None:
    torch.manual_seed(20260714)
    num_experts = 128
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, 4),
        dtype=torch.int32,
        device=device,
    )
    topk_weights = torch.rand(
        (num_tokens, 4),
        dtype=torch.float32,
        device=device,
    )
    if num_tokens >= 33:
        topk_ids[::31, -1] = -1

    expected_metadata, expected_gather, expected_scatter, expected_gamma = (
        _routing_from_topk(
            topk_weights,
            topk_ids,
            num_experts,
            dtype=torch.float32,
        )
    )
    counts = minimax_m3_route_counts(topk_weights, topk_ids, num_experts)
    metadata = make_ragged_tensor_metadata(counts, topk_ids.numel())
    gather, scatter, gamma = minimax_m3_route_order(
        topk_weights,
        topk_ids,
        metadata.slice_offs,
        num_experts,
    )

    assert torch.equal(counts, expected_metadata.slice_sizes)
    assert torch.equal(metadata.slice_offs, expected_metadata.slice_offs)
    assert torch.equal(gather, expected_gather)
    assert torch.equal(scatter, expected_scatter)
    assert torch.equal(gamma, expected_gamma)


@pytest.mark.parametrize("num_tokens", [1, 129, 512])
def test_minimax_m3_reduce_topk_matches_bf16_sum(
    device: str,
    num_tokens: int,
) -> None:
    torch.manual_seed(20260714)
    expert_output = torch.randn(
        (num_tokens, 4, 6144),
        dtype=torch.bfloat16,
        device=device,
    )

    actual = minimax_m3_reduce_topk(expert_output)

    assert torch.equal(actual, expert_output.sum(dim=1))
