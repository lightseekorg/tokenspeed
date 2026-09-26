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

"""The per-forward view a model receives of request-token history."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RequestTokenHistoryView:
    """Graph-stable history tensors for the current packed batch.

    A model whose profile declares ``request_token_history`` receives this as
    its ``request_token_history`` forward argument. Every tensor is a view of
    a persistent buffer, so a captured graph reads the same storage on every
    replay.

    Attributes:
        history_token_ids: ``[pool + 1, capacity]`` int32. Row ``s`` holds
            slot ``s``'s committed tokens in ``[0, committed_lengths[s])``;
            the model's kernels append the forward's inputs past that
            frontier. The last row serves graph padding.
        committed_lengths: ``[pool + 1]`` int32 committed token count per
            slot; read-only for the model.
        req_pool_indices: ``[bs]`` int64 slot of every request row.
        input_start_offsets: ``[bs + 1]`` int32 packed offsets of every
            request's inputs in the forward's ``input_ids``.
        active_request_mask: ``[bs]`` bool; False rows are graph padding and
            must neither read nor append history.
    """

    history_token_ids: torch.Tensor
    committed_lengths: torch.Tensor
    req_pool_indices: torch.Tensor
    input_start_offsets: torch.Tensor
    active_request_mask: torch.Tensor


__all__ = ["RequestTokenHistoryView"]
