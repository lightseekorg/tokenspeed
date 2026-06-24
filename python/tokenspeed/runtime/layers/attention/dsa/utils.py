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


def workspace_indices_to_kv_slots(
    workspace_indices: torch.Tensor,
    kv_workspace_slots: torch.Tensor | None,
) -> torch.Tensor:
    """Map DSA workspace-local top-k indices to global KV cache slot ids.

    Args:
        workspace_indices: Top-k indices in the compact DSA prefill workspace.
            Negative entries are treated as invalid sentinels and preserved.
        kv_workspace_slots: Lookup table mapping workspace rows to KV cache slots.

    Returns:
        A tensor with the same shape as ``workspace_indices`` containing int32 KV
        cache slot ids, or int32 ``workspace_indices`` when no lookup is provided.
    """
    if kv_workspace_slots is None or workspace_indices.numel() == 0:
        return workspace_indices.to(torch.int32)

    flat_indices = workspace_indices.reshape(-1)
    valid = flat_indices >= 0
    flat_slots = flat_indices.to(torch.int64)
    if valid.any():
        flat_slots[valid] = kv_workspace_slots.to(
            device=workspace_indices.device,
            dtype=torch.int64,
        ).index_select(0, flat_slots[valid])
    return flat_slots.view_as(workspace_indices).to(torch.int32)
