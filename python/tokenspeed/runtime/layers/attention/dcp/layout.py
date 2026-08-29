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

"""Cyclic DCP sequence-length helpers."""

from __future__ import annotations

import torch


def _validate_rank(dcp_rank: int, dcp_size: int) -> None:
    if dcp_size <= 0:
        raise ValueError(f"dcp_size must be positive, got {dcp_size}")
    if not 0 <= dcp_rank < dcp_size:
        raise ValueError(f"dcp_rank must be in [0, {dcp_size}), got {dcp_rank}")


def local_lengths(
    global_lengths: torch.Tensor,
    dcp_rank: int,
    dcp_size: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return graph-safe rank-local lengths with optional stable output."""
    _validate_rank(dcp_rank, dcp_size)
    if (
        not torch.is_floating_point(global_lengths)
        and global_lengths.dtype != torch.bool
    ):
        result = torch.div(global_lengths, dcp_size, rounding_mode="floor")
        result = result + (dcp_rank < torch.remainder(global_lengths, dcp_size))
    else:
        raise TypeError(
            f"global_lengths must have an integer dtype, got {global_lengths.dtype}"
        )
    result = result.clamp_min(0).to(global_lengths.dtype)
    if out is not None:
        if out.shape != global_lengths.shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} != input shape {tuple(global_lengths.shape)}"
            )
        out.copy_(result)
        return out
    return result


def visible_local_lengths(
    final_global_lengths: torch.Tensor,
    query_width: int,
    dcp_rank: int,
    dcp_size: int,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Visible local rows for each trailing multi-token decode query.

    ``final_global_lengths`` includes all ``query_width`` trailing tokens.
    Query column ``j`` therefore has the global exclusive causal bound
    ``L - query_width + 1 + j``.
    """
    _validate_rank(dcp_rank, dcp_size)
    if query_width <= 0:
        raise ValueError(f"query_width must be positive, got {query_width}")
    steps = torch.arange(
        1 - query_width,
        1,
        dtype=final_global_lengths.dtype,
        device=final_global_lengths.device,
    )
    bounds = (final_global_lengths.unsqueeze(-1) + steps).clamp_min(0)
    result = local_lengths(bounds, dcp_rank, dcp_size)
    if out is not None:
        if out.shape != result.shape:
            raise ValueError(
                f"out shape {tuple(out.shape)} != result shape {tuple(result.shape)}"
            )
        out.copy_(result)
        return out
    return result
