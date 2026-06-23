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
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.ops.moe.triton.mxfp4 import (
    _routing_from_topk,
    _routing_from_topk_medium_m,
    _routing_from_topk_small_m,
    _sorted_moe_metadata_from_routing,
)

pytestmark = pytest.mark.skipif(
    not current_platform().is_amd,
    reason="small-M fused routing is currently AMD-only",
)


def _make_topk(
    num_tokens: int,
    num_experts: int,
    topk: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(
        3100 + num_tokens * 17 + num_experts
    )
    scores = torch.randn(
        (num_tokens, num_experts),
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=True)[1].to(torch.int32)
    weights = (
        torch.arange(num_tokens * topk, device=device, dtype=torch.float32)
        .reshape(num_tokens, topk)
        .add_(1.0)
        .div_(num_tokens * topk + 1 if num_tokens else 1)
    )
    return weights, topk_ids


def _assert_routing_equal(actual, expected) -> None:
    actual_metadata, actual_gather, actual_scatter, actual_gate = actual
    expected_metadata, expected_gather, expected_scatter, expected_gate = expected

    assert torch.equal(actual_metadata.slice_sizes, expected_metadata.slice_sizes)
    assert torch.equal(actual_metadata.slice_offs, expected_metadata.slice_offs)
    assert torch.equal(
        actual_metadata.block_offs_data,
        expected_metadata.block_offs_data,
    )
    assert torch.equal(
        actual_metadata.block_schedule_data,
        expected_metadata.block_schedule_data,
    )
    assert torch.equal(actual_gather, expected_gather)
    assert torch.equal(actual_scatter, expected_scatter)
    torch.testing.assert_close(actual_gate, expected_gate, rtol=0.0, atol=0.0)


def _assert_metadata_equal(actual, expected) -> None:
    actual_metadata = actual[0]
    expected_metadata = expected[0]

    assert torch.equal(actual_metadata.slice_sizes, expected_metadata.slice_sizes)
    assert torch.equal(actual_metadata.slice_offs, expected_metadata.slice_offs)
    assert torch.equal(
        actual_metadata.block_offs_data,
        expected_metadata.block_offs_data,
    )
    assert torch.equal(
        actual_metadata.block_schedule_data,
        expected_metadata.block_schedule_data,
    )


def _assert_valid_routing_permutation(
    actual,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    gate_dtype: torch.dtype,
) -> None:
    metadata, gather, scatter, gate = actual
    total_rows = topk_ids.numel()
    assert torch.equal(
        torch.sort(scatter).values,
        torch.arange(total_rows, device=topk_ids.device, dtype=torch.int32),
    )
    assert torch.equal(gather, scatter // topk_ids.shape[1])

    flat_ids = topk_ids.reshape(-1)
    flat_weights = topk_weights.reshape(-1).to(gate_dtype)
    routed_ids = flat_ids[scatter.to(torch.long)]
    routed_weights = flat_weights[scatter.to(torch.long)]
    torch.testing.assert_close(gate, routed_weights, rtol=0.0, atol=0.0)

    expected_counts = torch.bincount(
        topk_ids.reshape(-1).to(torch.long),
        minlength=num_experts,
    ).to(torch.int32)
    assert torch.equal(metadata.slice_sizes, expected_counts)

    for expert in range(num_experts):
        start = int(metadata.slice_offs[expert].item())
        stop = int(metadata.slice_offs[expert + 1].item())
        if start == stop:
            continue
        assert torch.equal(
            routed_ids[start:stop],
            torch.full(
                (stop - start,),
                expert,
                device=topk_ids.device,
                dtype=routed_ids.dtype,
            ),
        )


def _routing_from_topk_fast_path(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    gate_dtype: torch.dtype,
):
    routed = _routing_from_topk_small_m(
        topk_weights,
        topk_ids,
        num_experts=num_experts,
        dtype=gate_dtype,
    )
    if routed is None:
        routed = _routing_from_topk_medium_m(
            topk_weights,
            topk_ids,
            num_experts=num_experts,
            dtype=gate_dtype,
        )
    assert routed is not None
    return routed


@pytest.mark.parametrize("num_tokens", [0, 1, 8, 16])
@pytest.mark.parametrize("num_experts", [64, 384])
@pytest.mark.parametrize("gate_dtype", [torch.float32, torch.bfloat16])
def test_small_m_routing_from_topk_matches_reference(
    device: str,
    num_tokens: int,
    num_experts: int,
    gate_dtype: torch.dtype,
) -> None:
    topk_weights, topk_ids = _make_topk(
        num_tokens,
        num_experts,
        topk=8,
        device=device,
    )

    actual = _routing_from_topk_small_m(
        topk_weights,
        topk_ids,
        num_experts=num_experts,
        dtype=gate_dtype,
    )
    assert actual is not None
    expected = _routing_from_topk(
        topk_weights,
        topk_ids,
        num_experts=num_experts,
        dtype=gate_dtype,
    )
    torch.cuda.synchronize()

    _assert_routing_equal(actual, expected)


@pytest.mark.parametrize("num_tokens", [32, 128, 512])
@pytest.mark.parametrize("num_experts", [64, 384])
@pytest.mark.parametrize("gate_dtype", [torch.float32, torch.bfloat16])
def test_medium_m_routing_from_topk_matches_reference_metadata(
    device: str,
    num_tokens: int,
    num_experts: int,
    gate_dtype: torch.dtype,
) -> None:
    topk_weights, topk_ids = _make_topk(
        num_tokens,
        num_experts,
        topk=8,
        device=device,
    )

    actual = _routing_from_topk_medium_m(
        topk_weights,
        topk_ids,
        num_experts=num_experts,
        dtype=gate_dtype,
    )
    assert actual is not None
    expected = _routing_from_topk(
        topk_weights,
        topk_ids,
        num_experts=num_experts,
        dtype=gate_dtype,
    )
    torch.cuda.synchronize()

    _assert_metadata_equal(actual, expected)
    _assert_valid_routing_permutation(
        actual,
        topk_weights,
        topk_ids,
        num_experts,
        gate_dtype,
    )


def test_medium_m_routing_can_schedule_only_block16(device: str) -> None:
    topk_weights, topk_ids = _make_topk(
        num_tokens=128,
        num_experts=384,
        topk=8,
        device=device,
    )

    actual = _routing_from_topk_medium_m(
        topk_weights,
        topk_ids,
        num_experts=384,
        dtype=torch.float32,
        schedule_block16_only=True,
    )
    assert actual is not None
    expected = _routing_from_topk(
        topk_weights,
        topk_ids,
        num_experts=384,
        dtype=torch.float32,
    )
    torch.cuda.synchronize()

    actual_metadata, *_ = actual
    expected_metadata, *_ = expected
    torch.testing.assert_close(
        actual_metadata.slice_sizes,
        expected_metadata.slice_sizes,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual_metadata.slice_offs,
        expected_metadata.slice_offs,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual_metadata.block_offs_data,
        expected_metadata.block_offs_data,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        actual_metadata.block_schedule(16),
        expected_metadata.block_schedule(16),
        rtol=0.0,
        atol=0.0,
    )
    for block_size in actual_metadata.block_sizes()[1:]:
        assert torch.equal(
            actual_metadata.block_schedule(block_size),
            torch.full_like(actual_metadata.block_schedule(block_size), -1),
        )
    _assert_valid_routing_permutation(
        actual,
        topk_weights,
        topk_ids,
        384,
        torch.float32,
    )


@pytest.mark.parametrize(
    ("num_tokens", "block_size"),
    [
        (4, 16),
        (128, 16),
        (128, 32),
    ],
)
def test_sorted_moe_metadata_view_matches_routing(
    device: str,
    num_tokens: int,
    block_size: int,
) -> None:
    topk_weights, topk_ids = _make_topk(
        num_tokens=num_tokens,
        num_experts=384,
        topk=8,
        device=device,
    )
    metadata, gather, scatter, gate = _routing_from_topk_fast_path(
        topk_weights,
        topk_ids,
        num_experts=384,
        gate_dtype=torch.float32,
    )
    sorted_metadata = _sorted_moe_metadata_from_routing(
        metadata,
        gather,
        scatter,
        gate,
        block_size=block_size,
    )
    torch.cuda.synchronize()

    assert sorted_metadata.block_size == block_size
    assert torch.equal(sorted_metadata.sorted_token_ids, scatter)
    assert torch.equal(sorted_metadata.scatter_indices, scatter)
    assert torch.equal(sorted_metadata.source_token_ids, gather)
    torch.testing.assert_close(
        sorted_metadata.sorted_weights,
        gate,
        rtol=0.0,
        atol=0.0,
    )
    assert torch.equal(
        sorted_metadata.sorted_expert_ids,
        metadata.block_schedule(block_size),
    )
    assert torch.equal(sorted_metadata.slice_sizes, metadata.slice_sizes)
    assert torch.equal(sorted_metadata.slice_offsets, metadata.slice_offs)
    assert torch.equal(
        sorted_metadata.num_valid_ids,
        torch.full(
            (2,),
            topk_ids.numel(),
            device=device,
            dtype=torch.int32,
        ),
    )
    assert torch.equal(
        sorted_metadata.source_token_ids,
        sorted_metadata.sorted_token_ids // topk_ids.shape[1],
    )

    flat_ids = topk_ids.reshape(-1)
    routed_ids = flat_ids[sorted_metadata.sorted_token_ids.to(torch.long)]
    for expert in range(384):
        start = int(sorted_metadata.slice_offsets[expert].item())
        stop = int(sorted_metadata.slice_offsets[expert + 1].item())
        if start == stop:
            continue
        assert torch.equal(
            routed_ids[start:stop],
            torch.full(
                (stop - start,),
                expert,
                device=device,
                dtype=routed_ids.dtype,
            ),
        )


def test_sorted_moe_metadata_rejects_unsupported_block_size(device: str) -> None:
    topk_weights, topk_ids = _make_topk(
        num_tokens=4,
        num_experts=384,
        topk=8,
        device=device,
    )
    metadata, gather, scatter, gate = _routing_from_topk_fast_path(
        topk_weights,
        topk_ids,
        num_experts=384,
        gate_dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="block_size must be one of"):
        _sorted_moe_metadata_from_routing(
            metadata,
            gather,
            scatter,
            gate,
            block_size=24,
        )


def test_small_m_routing_from_topk_rejects_large_work(device: str) -> None:
    topk_weights, topk_ids = _make_topk(
        num_tokens=17,
        num_experts=384,
        topk=8,
        device=device,
    )

    assert (
        _routing_from_topk_small_m(
            topk_weights,
            topk_ids,
            num_experts=384,
            dtype=torch.bfloat16,
        )
        is None
    )


def test_medium_m_routing_from_topk_rejects_large_work(device: str) -> None:
    topk_weights, topk_ids = _make_topk(
        num_tokens=513,
        num_experts=384,
        topk=8,
        device=device,
    )

    assert (
        _routing_from_topk_medium_m(
            topk_weights,
            topk_ids,
            num_experts=384,
            dtype=torch.bfloat16,
        )
        is None
    )
