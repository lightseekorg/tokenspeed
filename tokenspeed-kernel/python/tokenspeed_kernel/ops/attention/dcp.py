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

import torch

__all__ = ["pack_dcp_output_lse"]


def pack_dcp_output_lse(
    output: torch.Tensor,
    lse: torch.Tensor,
    *,
    world_size: int,
) -> torch.Tensor:
    """Pack output head slices and lossless FP32 LSE words for DCP A2A."""
    if output.ndim != 3 or lse.shape != output.shape[:-1]:
        raise ValueError("DCP output/LSE shapes disagree")
    if output.device != lse.device or not output.is_cuda:
        raise ValueError("DCP output and LSE must be on the same GPU")
    if output.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("DCP output must use BF16 or FP16 elements")
    if lse.dtype != torch.float32:
        raise ValueError("DCP LSE must use FP32 elements")
    if world_size < 1:
        raise ValueError("DCP world size must be positive")

    batch, heads, head_dim = output.shape
    if heads % world_size:
        raise ValueError(
            f"DCP head count {heads} is not divisible by group size {world_size}"
        )
    if batch == 0 or heads == 0 or head_dim == 0:
        raise ValueError("DCP output dimensions must be nonzero")

    heads_per_rank = heads // world_size
    packed = torch.empty(
        (world_size, batch, heads_per_rank, head_dim + 2),
        dtype=output.dtype,
        device=output.device,
    )

    from tokenspeed_kernel.ops.attention.triton.dcp import _pack_dcp_output_lse

    _pack_dcp_output_lse(
        output,
        lse,
        packed,
        world_size=world_size,
        heads_per_rank=heads_per_rank,
        head_dim=head_dim,
    )
    return packed
