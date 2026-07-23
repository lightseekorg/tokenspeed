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

"""Batch metadata structures for segmented LoRA kernels."""

from __future__ import annotations

from dataclasses import dataclass

import torch

NO_LORA_SLOT = -1


@dataclass
class LoraBatchInfo:
    """Per-step segment metadata read by the LoRA kernels.

    All tensors live on the LoRA device.  When the captured CUDA graph needs
    persistent storage, :class:`LoraManager` pre-allocates these tensors with
    maximum sizes; runtime fills the prefix and updates ``bs`` / ``max_len``.
    """

    bs: int
    num_segments: int
    max_len: int
    seg_lens: torch.Tensor  # (num_segments,) int32
    seg_indptr: torch.Tensor  # (num_segments + 1,) int32
    weight_indices: torch.Tensor  # (num_segments,) int32
    lora_ranks: torch.Tensor  # (n_slots,) int32; NO_LORA_SLOT means base model
    scalings: torch.Tensor  # (n_slots,) float32
    permutation: torch.Tensor | None = None  # unused (no sort by adapter yet)
    # Adapter-group metadata for lora_expand_grouped_v2_fwd (decode path only).
    # Populated by prepare_loras when max_len == 1.
    sort_order: torch.Tensor | None = None  # (bs,) int64
    group_slots: torch.Tensor | None = None  # (num_groups,) int32
    group_starts: torch.Tensor | None = None  # (num_groups,) int32
    group_sizes: torch.Tensor | None = None  # (num_groups,) int32
    num_groups: int = 0
    # Largest group size; pre-computed on CPU so the kernel grid avoids a
    # GPU-CPU sync.  Equals max(group_sizes) when num_groups > 0, else 0.
    max_group_size: int = 0
    # Host-only fast-path metadata. Non-negative iff every segment in this step
    # uses the same real adapter slot; NO_LORA_SLOT means mixed/base-only.
    single_lora_slot: int = NO_LORA_SLOT
    # Host-only active rank for ``single_lora_slot``. Zero when no single
    # nonzero adapter slot is active.
    single_lora_rank: int = 0
    # Host-only metadata for the multi-adapter batched CuTeDSL fast path.
    # Non-negative iff segments are equal-length, slots are consecutive, and
    # all participating slots share rank/scaling.
    multi_lora_start_slot: int = NO_LORA_SLOT
    multi_lora_count: int = 0
    multi_lora_segment_len: int = 0
    multi_lora_rank: int = 0


def build_decode_lora_groups(
    per_request_slots: list[int],
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Group decode requests by adapter slot for the grouped expand kernel.

    Returns ``(sort_order, group_slots, group_starts, group_sizes)``.
    ``group_starts`` are offsets into ``sort_order``.
    """
    sort_order = sorted(
        (i for i, slot in enumerate(per_request_slots) if slot != NO_LORA_SLOT),
        key=lambda i: per_request_slots[i],
    )
    group_slots: list[int] = []
    group_starts: list[int] = []
    group_sizes: list[int] = []
    for pos, orig in enumerate(sort_order):
        slot = per_request_slots[orig]
        if not group_slots or group_slots[-1] != slot:
            group_slots.append(slot)
            group_starts.append(pos)
            group_sizes.append(1)
        else:
            group_sizes[-1] += 1
    return sort_order, group_slots, group_starts, group_sizes
