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

"""Frozen checkpoint-free input construction and independent tail reference."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

TP_SIZE, TOP_K, LATENT, HIDDEN, SHARD, EPS = 8, 16, 3584, 7168, 896, 1e-5


@dataclass(slots=True)
class LayerInputs:
    """Pointer-stable inputs for one synthetic layer."""

    gemm2_output: torch.Tensor
    expert_weights: torch.Tensor
    expanded_idx: torch.Tensor
    shared_partial: torch.Tensor
    prefix: torch.Tensor
    gamma: torch.Tensor
    up_weight: torch.Tensor


def _rmsnorm_reference(value: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    value_float = value.float()
    variance = value_float.square().mean(dim=-1, keepdim=True)
    return (value_float * torch.rsqrt(variance + EPS) * gamma.float()).to(
        torch.bfloat16
    )


def _local_finalize_reference(layer: LayerInputs) -> torch.Tensor:
    """FP32 top-k accumulation with explicit zero semantics for ``-1``."""

    m = layer.expert_weights.shape[0]
    flat_idx = layer.expanded_idx.view(-1)
    valid = flat_idx >= 0
    safe_idx = flat_idx.clamp_min(0).to(torch.int64)
    gathered = layer.gemm2_output.index_select(0, safe_idx).view(m, TOP_K, LATENT)
    weighted = gathered.float() * layer.expert_weights.float().unsqueeze(-1)
    weighted.masked_fill_(~valid.view(m, TOP_K, 1), 0.0)
    return weighted.sum(dim=1).to(torch.bfloat16)


def _rank_ordered_sum(local: torch.Tensor) -> torch.Tensor:
    """Render BT's rank-ordered BF16 contribution reduction."""

    gathered = [torch.empty_like(local) for _ in range(TP_SIZE)]
    dist.all_gather(gathered, local)
    total = gathered[0].float()
    for peer in gathered[1:]:
        total.add_(peer.float())
    return total.to(torch.bfloat16)


def _semantic_reference(layer: LayerInputs, rank: int) -> torch.Tensor:
    local_finalized = _local_finalize_reference(layer)
    routed_sum = _rank_ordered_sum(local_finalized)
    routed_norm = _rmsnorm_reference(routed_sum, layer.gamma)

    result = layer.shared_partial.clone()
    start = rank * SHARD
    target = result[:, start : start + SHARD]
    target.add_(layer.prefix[:, start : start + SHARD])
    target.addmm_(routed_norm, layer.up_weight.t())
    # The independent reference intentionally uses NCCL for this last sum at
    # every M.  At medium M the production multimem result may differ only by
    # BF16 reduction ordering, which the strict error bounds cover.
    dist.all_reduce(result)
    return result


def _make_layer_inputs(
    rank: int,
    m: int,
    layer_id: int,
    device: torch.device,
) -> LayerInputs:
    rows = m * TOP_K
    routed_generator = torch.Generator(device=device).manual_seed(
        10_000 + rank * 101 + m * 7 + layer_id
    )
    weights_generator = torch.Generator(device=device).manual_seed(
        20_000 + m * 7 + layer_id
    )
    indices_generator = torch.Generator(device=device).manual_seed(
        30_000 + m * 7 + layer_id
    )
    shared_generator = torch.Generator(device=device).manual_seed(
        40_000 + rank * 101 + m * 7 + layer_id
    )
    prefix_generator = torch.Generator(device=device).manual_seed(
        50_000 + m * 7 + layer_id
    )
    gamma_generator = torch.Generator(device=device).manual_seed(60_000 + layer_id)
    weight_generator = torch.Generator(device=device).manual_seed(
        70_000 + rank * 101 + layer_id
    )

    gemm2_output = (
        torch.randn(
            rows,
            LATENT,
            dtype=torch.bfloat16,
            device=device,
            generator=routed_generator,
        )
        * 0.08
    ).contiguous()
    expert_weights = torch.softmax(
        torch.randn(
            m,
            TOP_K,
            dtype=torch.float32,
            device=device,
            generator=weights_generator,
        ),
        dim=-1,
    ).to(torch.bfloat16)
    expanded_idx = torch.randperm(
        rows, dtype=torch.int64, device=device, generator=indices_generator
    ).to(torch.int32)
    slot_ids = torch.arange(rows, dtype=torch.int32, device=device)
    sentinel_mask = (slot_ids + 3 * layer_id + m) % 11 == 0
    expanded_idx.masked_fill_(sentinel_mask, -1)
    if not bool((expanded_idx == -1).any()):
        raise AssertionError("the synthetic map must exercise -1 sentinel slots")

    shared_partial = (
        torch.randn(
            m,
            HIDDEN,
            dtype=torch.bfloat16,
            device=device,
            generator=shared_generator,
        )
        * 0.02
    ).contiguous()
    prefix = (
        torch.randn(
            m,
            HIDDEN,
            dtype=torch.bfloat16,
            device=device,
            generator=prefix_generator,
        )
        * 0.02
    ).contiguous()
    gamma = (
        1.0
        + torch.randn(
            LATENT,
            dtype=torch.bfloat16,
            device=device,
            generator=gamma_generator,
        )
        * 0.02
    ).contiguous()
    up_weight = (
        torch.randn(
            SHARD,
            LATENT,
            dtype=torch.bfloat16,
            device=device,
            generator=weight_generator,
        )
        * 0.015
    ).contiguous()
    return LayerInputs(
        gemm2_output=gemm2_output,
        expert_weights=expert_weights.contiguous(),
        expanded_idx=expanded_idx.contiguous(),
        shared_partial=shared_partial,
        prefix=prefix,
        gamma=gamma,
        up_weight=up_weight,
    )
