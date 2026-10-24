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

"""Allocate the symmetric selection region used by the fused A0 engines.

The indexer and worker engines own publication, tag polling and generation
advancement. Each attention layer keeps its own region and initializes it
before the first fused launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention.selection_exchange import (
    INDEX_K_WORDS,
    NUM_RANKS,
    NUM_TOKENS,
    TOPK,
    payload_words,
    region_words,
)


@dataclass
class SelectionExchangeState:
    """Live symmetric allocation and its payload layout.

    Attributes:
        symmetric_buffer: This rank's two payload slots and generation flag.
        peer_bases: int64 [8] byte addresses of the peer regions.
        words: One payload slot's width in int32 words.
        handle: Owns the symmetric mapping for the lifetime of this state.
    """

    symmetric_buffer: torch.Tensor
    peer_bases: torch.Tensor
    words: int
    handle: Any


def create_selection_exchange_state(
    group: Any,
    *,
    rank: int,
    device: torch.device,
) -> SelectionExchangeState:
    """Collectively allocate the TP8 selection workspace.

    Args:
        group: Process group spanning exactly eight ranks.
        rank: This rank's index in the group.
        device: CUDA device for the workspace, requiring B300 SM103.

    Returns:
        The symmetric region, peer addresses, payload width and mapping owner.
    """
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    device = torch.device(device)
    if dist.get_world_size(group) != NUM_RANKS:
        raise ValueError(
            f"selection exchange requires world size {NUM_RANKS}, "
            f"got {dist.get_world_size(group)}"
        )
    if device.type != "cuda":
        raise ValueError("selection exchange requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise ValueError("selection exchange requires B300 SM103")
    if type(rank) is not int or not 0 <= rank < NUM_RANKS:
        raise ValueError(f"rank must be a Python int in [0, {NUM_RANKS}), got {rank}")
    # Round the region up to a 16-byte word multiple so the symmetric
    # allocation keeps comfortable alignment for the vectorized copies.
    alloc_words = region_words(INDEX_K_WORDS)
    alloc_words += -alloc_words % 4

    symmetric_buffer = symm_mem.empty(alloc_words, dtype=torch.int32, device=device)
    handle = symm_mem.rendezvous(symmetric_buffer, group)
    if handle.buffer_size < alloc_words * 4:
        raise ValueError("symmetric allocation is too small")
    if len(handle.buffer_ptrs) != NUM_RANKS:
        raise ValueError("symmetric mapping must expose eight ranks")

    peer_bases = torch.tensor(
        [int(pointer) for pointer in handle.buffer_ptrs],
        dtype=torch.int64,
        device=device,
    )
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    return SelectionExchangeState(
        symmetric_buffer=symmetric_buffer,
        peer_bases=peer_bases,
        words=payload_words(INDEX_K_WORDS),
        handle=handle,
    )


__all__ = [
    "INDEX_K_WORDS",
    "NUM_RANKS",
    "NUM_TOKENS",
    "TOPK",
    "SelectionExchangeState",
    "create_selection_exchange_state",
]
