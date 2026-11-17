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

"""Verify outputs of a synthetic acceptance length."""

import torch


def write_synthetic_verify_outputs(
    candidates: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    accept_length: torch.Tensor,
) -> None:
    """Commit each row's draft prefix followed by its target token.

    Row ``r`` accepts ``L = min(lengths[r], width)`` tokens: the drafts
    ``candidates[r, 1:L]`` and then the target token, with zeros after them.

    Args:
        candidates: ``[rows, width]`` verify candidates; column 0 is the
            round's anchor, the drafts follow it. Unit column stride.
        targets: ``[rows]`` target token of each row, or ``[rows, width]``
            target rows whose column ``L - 1`` is committed. Unit last stride.
        lengths: ``[rows]`` contiguous accept lengths, each at least 1.
        predict: Contiguous ``[rows * width]`` committed tokens.
        accept_index: Contiguous ``[rows, width]`` flat candidate index of each
            accepted token, ``-1`` past ``L``.
        accept_length: Contiguous ``[rows]``, written ``L - 1``.

    Raises:
        ValueError: A tensor is not the shape, layout, or device described.
    """
    rows, width = candidates.shape
    if targets.shape not in ((rows,), (rows, width)):
        raise ValueError(
            f"targets must be [{rows}] or [{rows}, {width}], got {list(targets.shape)}"
        )
    if (
        lengths.shape != (rows,)
        or predict.numel() != rows * width
        or accept_index.shape != (rows, width)
        or accept_length.shape != (rows,)
    ):
        raise ValueError("synthetic verify outputs must match the candidate rows")
    if candidates.stride(-1) != 1 or targets.stride(-1) != 1:
        raise ValueError("candidates and targets need a unit last stride")
    if not all(
        t.is_contiguous() for t in (lengths, predict, accept_index, accept_length)
    ):
        raise ValueError("synthetic lengths and verify outputs must be contiguous")
    if any(
        t.device != candidates.device
        for t in (targets, lengths, predict, accept_index, accept_length)
    ):
        raise ValueError("synthetic verify tensors must share the candidates' device")
    if rows == 0:
        return
    if candidates.is_cuda:
        from tokenspeed_kernel.ops.sampling.triton.synthetic import (
            write_synthetic_outputs,
        )

        write_synthetic_outputs(
            candidates, targets, lengths, predict, accept_index, accept_length
        )
        return
    # CPU reference for the output contract; CUDA always uses the fused kernel.
    length = lengths.clamp(max=width)
    target = (
        targets.gather(1, (length - 1).long()[:, None]).squeeze(1)
        if targets.ndim == 2
        else targets
    )
    positions = torch.arange(width, device=candidates.device).expand(rows, width)
    output = predict.view(rows, width)
    output[:, : width - 1].copy_(candidates[:, 1:])
    output.masked_fill_(positions >= length[:, None] - 1, 0)
    output.scatter_(1, (length - 1).long()[:, None], target.to(predict.dtype)[:, None])
    accept_index.copy_(
        torch.where(
            positions < length[:, None],
            positions + torch.arange(rows, device=candidates.device)[:, None] * width,
            -1,
        )
    )
    accept_length.copy_(length - 1)
