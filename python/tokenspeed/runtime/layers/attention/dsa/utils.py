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

from dataclasses import replace
from typing import Any

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


def _prepare_dsa_topk_for_mtp_decode(
    dsa_topk: tuple[Any | None, Any | None],
    gather_ids: torch.Tensor,
    *,
    num_prefill_rows: int = 0,
) -> tuple[Any | None, Any | None]:
    """Gather accepted DSA prefill/decode rows for the next MTP step."""
    prefill_topk, decode_topk = dsa_topk
    if decode_topk is None or decode_topk.topk_indices.shape[0] == 0:
        return dsa_topk

    topk_indices = decode_topk.topk_indices
    topk_lens = decode_topk.topk_lens
    if num_prefill_rows <= 0 and topk_indices.shape[0] <= gather_ids.numel():
        return dsa_topk
    if num_prefill_rows <= 0:
        selected_indices = topk_indices.index_select(0, gather_ids)
        selected_lens = topk_lens.index_select(0, gather_ids)
    else:
        if prefill_topk is None:
            return dsa_topk
        num_prefill_rows = min(int(num_prefill_rows), gather_ids.numel())
        prefill_rows = gather_ids[:num_prefill_rows]
        decode_rows = gather_ids[num_prefill_rows:]
        selected_indices = workspace_indices_to_kv_slots(
            prefill_topk.workspace_indices.index_select(0, prefill_rows),
            prefill_topk.kv_workspace_slots,
        ).to(device=topk_indices.device, dtype=topk_indices.dtype)
        selected_lens = prefill_topk.topk_lens.index_select(0, prefill_rows).to(
            device=topk_lens.device, dtype=topk_lens.dtype
        )
        if decode_rows.numel() > 0:
            selected_indices = torch.cat(
                (selected_indices, topk_indices.index_select(0, decode_rows)),
                dim=0,
            )
            selected_lens = torch.cat(
                (selected_lens, topk_lens.index_select(0, decode_rows)),
                dim=0,
            )

    return prefill_topk, replace(
        decode_topk,
        topk_indices=selected_indices,
        topk_lens=selected_lens,
    )
