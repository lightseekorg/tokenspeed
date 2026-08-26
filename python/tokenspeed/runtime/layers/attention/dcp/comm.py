# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Collective building blocks for exact MLA DCP reconstruction."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.distributed.mapping import Group
else:
    Group = tuple[int, ...]


def gather_fp8_query_heads(query: torch.Tensor, group: Group) -> torch.Tensor:
    """All-gather absorbed query heads without requiring FP8 NCCL support."""
    if len(group) == 1:
        return query
    if query.element_size() != 1:
        raise ValueError(
            f"DCP query byte transport requires a one-byte dtype, got {query.dtype}"
        )
    from tokenspeed.runtime.distributed.comm_ops import all_gather

    with nvtx_range("dcp_query_pack", category="dcp"):
        head_major = query.movedim(-2, 0).contiguous()
    with nvtx_range("dcp_query_all_gather", category="dcp"):
        gathered_bytes = all_gather(head_major.view(torch.uint8), group, dim=0)
    return gathered_bytes.view(query.dtype).movedim(0, -2)


def gather_query_heads(query: torch.Tensor, group: Group) -> torch.Tensor:
    """All-gather the head axis for ordinary NCCL-supported query dtypes."""
    if len(group) == 1:
        return query
    from tokenspeed.runtime.distributed.comm_ops import all_gather

    with nvtx_range("dcp_query_pack", category="dcp"):
        head_major = query.movedim(-2, 0).contiguous()
    with nvtx_range("dcp_query_all_gather", category="dcp"):
        gathered = all_gather(head_major, group, dim=0)
    return gathered.movedim(0, -2)


def merge_partial_outputs(
    partial_outputs: torch.Tensor,
    partial_lse2: torch.Tensor,
) -> torch.Tensor:
    """Reference exact base-2 softmax merge for stacked DCP partials.

    Args:
        partial_outputs: ``[D, ..., value_dim]``.
        partial_lse2: ``[D, ...]`` in base-2 units.
    """
    if partial_outputs.shape[:-1] != partial_lse2.shape:
        raise ValueError(
            "partial output/LSE shapes disagree: "
            f"{tuple(partial_outputs.shape)} and {tuple(partial_lse2.shape)}"
        )
    lse = partial_lse2.float()
    valid = torch.isfinite(lse)
    sanitized = torch.where(valid, lse, torch.full_like(lse, -torch.inf))
    maximum = sanitized.amax(dim=0)
    has_value = torch.isfinite(maximum)
    shifted = torch.where(
        valid & has_value.unsqueeze(0),
        sanitized - maximum.unsqueeze(0),
        torch.full_like(sanitized, -torch.inf),
    )
    masses = torch.exp2(shifted)
    denominator = masses.sum(dim=0)
    weights = torch.where(
        denominator.unsqueeze(0) > 0,
        masses / denominator.clamp_min(torch.finfo(masses.dtype).tiny).unsqueeze(0),
        torch.zeros_like(masses),
    )
    safe_outputs = torch.where(
        torch.isfinite(partial_outputs.float()),
        partial_outputs.float(),
        torch.zeros_like(partial_outputs, dtype=torch.float32),
    )
    return (safe_outputs * weights.unsqueeze(-1)).sum(dim=0)


def reconstruct_and_reduce_scatter(
    local_output: torch.Tensor,
    local_lse2: torch.Tensor,
    *,
    dcp_rank: int,
    group: Group,
    local_nonempty: torch.Tensor | None = None,
    lse_base: float = 2.0,
) -> torch.Tensor:
    """Weight one local partial and reduce-scatter original TP head slices."""
    if len(group) == 1:
        return local_output
    if not 0 <= dcp_rank < len(group):
        raise ValueError(f"dcp_rank={dcp_rank} is invalid for group {group}")
    if local_output.shape[:-1] != local_lse2.shape:
        raise ValueError("local output and LSE shapes disagree")
    if local_nonempty is not None:
        if local_nonempty.shape != local_lse2.shape[:-1]:
            raise ValueError("local nonempty mask must cover batch/query axes")
        if os.environ.get("TOKENSPEED_CACHE_DEBUG") == "1":
            expected = local_nonempty.unsqueeze(-1).expand_as(local_lse2)
            unexpected = expected & ~torch.isfinite(local_lse2)
            if bool(unexpected.any().item()):
                raise RuntimeError("non-finite DCP LSE on a nonempty local shard")

    from tokenspeed.runtime.distributed.comm_ops import all_gather, reduce_scatter

    gathered_lse = all_gather(local_lse2.float().unsqueeze(0), group, dim=0)
    valid = torch.isfinite(gathered_lse)
    sanitized = torch.where(
        valid, gathered_lse, torch.full_like(gathered_lse, -torch.inf)
    )
    maximum = sanitized.amax(dim=0)
    has_value = torch.isfinite(maximum)
    shifted = torch.where(
        valid & has_value.unsqueeze(0),
        sanitized - maximum.unsqueeze(0),
        torch.full_like(sanitized, -torch.inf),
    )
    if lse_base == 2.0:
        masses = torch.exp2(shifted)
    elif lse_base == math.e:
        masses = torch.exp(shifted)
    else:
        masses = torch.pow(torch.as_tensor(lse_base, device=shifted.device), shifted)
    denominator = masses.sum(dim=0)
    local_weight = torch.where(
        denominator > 0,
        masses[dcp_rank] / denominator.clamp_min(torch.finfo(masses.dtype).tiny),
        torch.zeros_like(denominator),
    )
    safe_output = torch.where(
        torch.isfinite(local_output.float()),
        local_output.float(),
        torch.zeros_like(local_output, dtype=torch.float32),
    )
    corrected_head_major = (
        (safe_output * local_weight.unsqueeze(-1)).movedim(-2, 0).contiguous()
    )
    local_head_major = reduce_scatter(corrected_head_major, group)
    return local_head_major.movedim(0, -2).to(local_output.dtype)


__all__ = [
    "gather_fp8_query_heads",
    "gather_query_heads",
    "merge_partial_outputs",
    "reconstruct_and_reduce_scatter",
]
