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

"""Packed output/LSE all-to-all combine for decode context parallelism."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel import pack_dcp_output_lse

from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:
    from tokenspeed.runtime.distributed.mapping import Group
else:
    Group = tuple[int, ...]


def _merge_received_partials(
    recv: torch.Tensor,
    *,
    head_dim: int,
    lse_base: float,
) -> torch.Tensor:
    """Merge packed A2A partials with the AG-RS FP32 arithmetic contract."""
    words = recv.view(torch.uint16)[..., head_dim:]
    low = words[..., 0].to(torch.int32)
    high = words[..., 1].to(torch.int32)
    lse = (low | (high << 16)).contiguous().view(torch.float32)

    valid = torch.isfinite(lse)
    sanitized = torch.where(valid, lse, torch.full_like(lse, -torch.inf))
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
        masses = torch.pow(
            torch.as_tensor(lse_base, device=shifted.device),
            shifted,
        )
    denominator = masses.sum(dim=0)
    weights = torch.where(
        denominator.unsqueeze(0) > 0,
        masses / denominator.clamp_min(torch.finfo(masses.dtype).tiny).unsqueeze(0),
        torch.zeros_like(masses),
    )
    partials = recv[..., :head_dim].float()
    safe_partials = torch.where(
        torch.isfinite(partials),
        partials,
        torch.zeros_like(partials),
    )
    return (safe_partials * weights.unsqueeze(-1)).sum(dim=0).to(recv.dtype)


def reconstruct_with_all_to_all(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    *,
    group: Group,
    lse_base: float = 2.0,
) -> torch.Tensor:
    """Exchange head slices once, then exactly merge every context partial."""
    world_size = len(group)
    if world_size == 1:
        return local_output
    if local_output.ndim < 3 or local_lse.shape != local_output.shape[:-1]:
        raise ValueError("DCP A2A output/LSE shapes disagree")
    if local_output.element_size() != 2:
        raise ValueError(
            "DCP A2A packs FP32 LSE into two 16-bit output elements; got "
            f"{local_output.dtype}"
        )
    leading_shape = local_output.shape[:-2]
    heads, head_dim = local_output.shape[-2:]
    flat_output = local_output.reshape(-1, heads, head_dim)
    flat_lse = local_lse.reshape(-1, heads)
    if heads % world_size:
        raise ValueError(
            f"DCP A2A head count {heads} is not divisible by group size {world_size}"
        )
    heads_per_rank = heads // world_size
    with nvtx_range("dcp_output_lse_pack", category="dcp"):
        send = pack_dcp_output_lse(
            flat_output,
            flat_lse,
            world_size=world_size,
        )
    recv = torch.empty_like(send)

    from tokenspeed.runtime.distributed.comm_ops import all_to_all_single

    with nvtx_range("dcp_output_lse_all_to_all", category="dcp"):
        all_to_all_single(recv.view(-1), send.view(-1), group)
    with nvtx_range("dcp_output_lse_unpack_merge", category="dcp"):
        output = _merge_received_partials(
            recv,
            head_dim=head_dim,
            lse_base=lse_base,
        )
    return output.reshape(*leading_shape, heads_per_rank, head_dim)


__all__ = ["reconstruct_with_all_to_all"]
