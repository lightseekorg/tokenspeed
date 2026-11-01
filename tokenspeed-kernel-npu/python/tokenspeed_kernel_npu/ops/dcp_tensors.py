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

"""Tensor layouts for the explicitly selected collective DCP execution path."""

import torch


def select_local_topk(indices, positions, *, rank):
    """Pack this owner's selected IDs in score order and count 128-ID chunks.

    ``indices[T,1,K]`` holds local IDs. ``positions[T,K]`` indexes the
    owner-major global score array. Negative IDs are excluded. Returned INT32
    IDs keep the same shape as indices and use -1 for their unused suffix.
    """
    if (
        indices.ndim != 3
        or indices.shape[1] != 1
        or positions.shape != (indices.shape[0], indices.shape[2])
        or indices.dtype != torch.int32
        or positions.dtype not in (torch.int32, torch.int64)
        or indices.device != positions.device
        or rank < 0
        or indices.shape[2] < 1
    ):
        raise ValueError("Invalid local DCP TopK layout")
    width = indices.shape[2]
    positions = positions.to(torch.int64)
    selected = indices[:, 0].gather(1, positions.remainder(width))
    valid = (positions >= 0) & (positions // width == rank) & (selected >= 0)
    prefix = valid.cumsum(1, dtype=torch.int32)
    columns = torch.arange(width, device=indices.device)
    destinations = torch.where(valid, prefix - 1, prefix[:, -1:] + columns - prefix)
    result = torch.empty_like(selected)
    result.scatter_(1, destinations.to(torch.int64), torch.where(valid, selected, -1))
    return result[:, None], ((prefix[:, -1] + 127) // 128).to(torch.int32)


def merge_normalized_partials(output, lse):
    """Merge gathered normalized O[P,N,D] using FP32 natural-log LSE[P,N].

    This local tensor reduction follows the already completed framework
    collective. Empty-shard values are ignored and all-empty rows return zero.
    Inputs remain unchanged; the result is BF16 [N,D].
    """
    if (
        output.ndim != 3
        or lse.shape != output.shape[:2]
        or lse.dtype != torch.float32
        or output.dtype not in (torch.bfloat16, torch.float32)
        or output.device != lse.device
    ):
        raise ValueError("Invalid normalized DCP partial layout")
    empty = torch.isneginf(lse)
    values = torch.where(empty[..., None], 0, output.float())
    scores = torch.where(empty.all(0)[None, :], 0, lse)
    weights = torch.softmax(scores, dim=0)
    return (values * weights[..., None]).sum(0).to(torch.bfloat16)
