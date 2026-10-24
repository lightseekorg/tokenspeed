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

"""Fixed TP8 verify-8 Lamport workspace geometry for DownReduce."""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch


class LongcatDownReduceOneShotError(ValueError):
    """Raised when a fixed one-shot coordinate or transition is invalid."""


def _coordinate(name: str, value: object, extent: int) -> int:
    if type(value) is not int or not 0 <= value < extent:
        raise LongcatDownReduceOneShotError(
            f"{name} must be a Python int in [0, {extent})"
        )
    return value


@dataclass(frozen=True)
class LongcatDownReduceOneShotLayout:
    """Private ``[slot, token, rank, hidden]`` BF16 replica layout.

    Each aligned BF16x8 fragment publishes eight contiguous hidden values.
    """

    token_count: ClassVar[int] = 8
    rank_count: ClassVar[int] = 8
    hidden_dim: ClassVar[int] = 8192
    element_bytes: ClassVar[int] = 2
    fragment_elements: ClassVar[int] = 8
    slot_count: ClassVar[int] = 3

    @property
    def slot_bytes(self) -> int:
        return self.token_count * self.rank_count * self.hidden_dim * self.element_bytes

    @property
    def allocation_bytes(self) -> int:
        return self.slot_count * self.slot_bytes


@dataclass(frozen=True)
class LongcatDownReduceOneShotWorkspace:
    """Own the private fixed-shape MNNVL replica and rotation state."""

    rank: int
    device: torch.device
    layout: LongcatDownReduceOneShotLayout
    allocation_bytes: int
    symmetric_buffer: torch.Tensor
    local_buffer: torch.Tensor
    local_ptr: int
    multicast_ptr: int
    transport_bases: torch.Tensor
    rotation_flags: torch.Tensor
    refs: tuple[Any, ...]


def create_longcat_downreduce_oneshot_workspace(
    group: Any, device: torch.device
) -> LongcatDownReduceOneShotWorkspace:
    """Collectively allocate the private TP8 verify-8 one-shot workspace."""
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    layout = LongcatDownReduceOneShotLayout()
    device = torch.device(device)
    if dist.get_world_size(group) != layout.rank_count:
        raise LongcatDownReduceOneShotError("one-shot workspace requires TP8")
    if device.type != "cuda":
        raise LongcatDownReduceOneShotError("one-shot workspace requires CUDA")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise LongcatDownReduceOneShotError("one-shot workspace requires B300 SM103")
    rank = dist.get_rank(group)
    _coordinate("rank", rank, layout.rank_count)

    symmetric_buffer = symm_mem.empty(
        layout.allocation_bytes // 4, dtype=torch.float32, device=device
    )
    handle = symm_mem.rendezvous(symmetric_buffer, group)
    if handle.buffer_size < layout.allocation_bytes:
        raise LongcatDownReduceOneShotError("symmetric allocation is too small")
    if len(handle.buffer_ptrs) != layout.rank_count:
        raise LongcatDownReduceOneShotError("symmetric mapping must expose eight ranks")
    if handle.multicast_ptr is None or int(handle.multicast_ptr) == 0:
        raise LongcatDownReduceOneShotError("NVLS multicast mapping is unavailable")

    local_buffer = handle.get_buffer(rank, (handle.buffer_size // 4,), torch.float32)
    local_buffer.fill_(-0.0)
    transport_bases = torch.tensor(
        [int(handle.multicast_ptr), int(handle.buffer_ptrs[rank]), 0, 0, 0, 0, 0, 0],
        dtype=torch.int64,
        device=device,
    )
    rotation_flags = torch.tensor(
        [0, 2, layout.slot_bytes, 0, 0, 0, 0, 0, 0],
        dtype=torch.uint32,
        device=device,
    )
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    return LongcatDownReduceOneShotWorkspace(
        rank=rank,
        device=device,
        layout=layout,
        allocation_bytes=layout.allocation_bytes,
        symmetric_buffer=symmetric_buffer,
        local_buffer=local_buffer,
        local_ptr=int(handle.buffer_ptrs[rank]),
        multicast_ptr=int(handle.multicast_ptr),
        transport_bases=transport_bases,
        rotation_flags=rotation_flags,
        refs=(
            symmetric_buffer,
            local_buffer,
            handle,
        ),
    )


__all__ = [
    "LongcatDownReduceOneShotError",
    "LongcatDownReduceOneShotLayout",
    "LongcatDownReduceOneShotWorkspace",
    "create_longcat_downreduce_oneshot_workspace",
]
