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

"""Symmetric input ownership for fused shared-RS/up-projection/AG kernels."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from tokenspeed_kernel.ops.communication.triton import (
    _RSAG_MIN_BLOCKS,
    TritonCommState,
    _alloc_symm,
    nvidia_create_rsag_state,
)
from tokenspeed_kernel.platform import current_platform

# Keep the existing reservation while removing the standalone RS implementation.
_SHARED_INPUT_SIGNAL_BLOCKS = 128


def _vote(group: dist.ProcessGroup, identity: Any, error: str | None) -> None:
    records = [None] * dist.get_world_size(group)
    dist.all_gather_object(records, (identity, error), group=group)
    if any(r[1] is not None or r[0] != records[0][0] for r in records):
        raise ValueError(f"collective fused shared-RS validation failed: {records}")


@dataclass
class SharedRsWorkspace:
    """Own fused shared-RS input, layout metadata and synchronization storage.

    Allocate outside capture. Independent graphs own separate workspaces;
    sequential layers of one graph may reuse a workspace. The fused tail's exit
    protects peer reads before the next producer overwrites the input. The
    shard allocation supplies layout metadata only; no standalone RS runs.
    """

    state: TritonCommState

    @classmethod
    def allocate(cls, group: dist.ProcessGroup, max_tokens: int, device: torch.device):
        """Collectively allocate TP8 BF16 input and metadata outside capture.

        Args:
            group: Eight-rank NVLS-capable process group.
            max_tokens: Capacity in [1,8192], identical on every rank.
            device: This rank's CUDA device.

        Returns:
            Independently owned workspace; never shared by concurrent graphs.
        """
        error = None
        if dist.get_world_size(group) != 8 or max_tokens < 1 or max_tokens > 8192:
            error = "requires TP8 and capacity in [1,8192]"
        if device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            error = "allocation requires CUDA outside graph capture"
        _vote(group, max_tokens, error)
        return cls(
            _create_shared_input_state(
                group,
                dist.get_rank(group),
                max_tokens,
                7168,
                device,
            )
        )


def _create_shared_input_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_tokens: int,
    hidden_size: int,
    device: torch.device | None,
) -> TritonCommState:
    """Create raw symmetric input and retained fused-GEMM layout metadata.

    Args:
        group: Process group spanning the hidden shards.
        rank_in_group: This process's rank within ``group``.
        max_tokens: Maximum captured token bucket.
        hidden_size: Full hidden width before the even rank split.
        device: Optional CUDA device; defaults to the current device.

    Returns:
        A symmetric full-width communication state with a second symmetric
        local-shard metadata buffer; no standalone RS writes that buffer.
        Both allocations and their rendezvous handles are stable across
        CUDA Graph replays.
    """
    platform = current_platform()
    assert (
        platform.is_nvidia
    ), f"_create_shared_input_state only supports NVIDIA, got {platform}"
    assert hidden_size % group.size() == 0, (
        f"hidden_size ({hidden_size}) must be divisible by world size "
        f"({group.size()})"
    )
    device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
    local_sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    rank_uniform_sm_count = torch.tensor(
        [local_sm_count], dtype=torch.int32, device=device
    )
    dist.all_reduce(rank_uniform_sm_count, op=dist.ReduceOp.MIN, group=group)
    rank_uniform_block_limit = min(
        _SHARED_INPUT_SIGNAL_BLOCKS, int(rank_uniform_sm_count.item())
    )
    assert rank_uniform_block_limit >= _RSAG_MIN_BLOCKS, (
        "fused shared-RS workspace needs at least "
        f"{_RSAG_MIN_BLOCKS} SMs on every rank, got rank-uniform floor "
        f"{rank_uniform_block_limit}"
    )
    # Preserve the established signal-pad reservation and allocation order.
    # Grow the process-global floor before either persistent allocation;
    # nvidia_create_rsag_state only applies max(), so it cannot shrink this.
    hidden_pad_bytes = _SHARED_INPUT_SIGNAL_BLOCKS * group.size() * 4
    symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), hidden_pad_bytes))
    state = nvidia_create_rsag_state(
        group=group,
        rank_in_group=rank_in_group,
        max_tokens=max_tokens,
        hidden_size=hidden_size,
        device=device,
    )
    state.symm_mem_hdl = symm_mem.rendezvous(state.comm_buff, group=group)
    local_hidden = hidden_size // group.size()
    local_shape = (max_tokens, local_hidden)
    state.local_buff, state.local_symm_mem_hdl = _alloc_symm(
        local_shape, torch.bfloat16, state.device, group
    )
    assert state.local_symm_mem_hdl.rank == rank_in_group, "Mismatched local rank id"
    assert (
        state.local_symm_mem_hdl.world_size == group.size()
    ), "Mismatched local world size"
    # Preserve the allocation-time peer-mapping checks. This host-only query
    # adds no collective or rank-dependent allocation.
    for peer in range(group.size()):
        peer_buffer = state.local_symm_mem_hdl.get_buffer(
            peer,
            local_shape,
            torch.bfloat16,
            storage_offset=0,
        )
        assert peer_buffer.data_ptr() != 0, f"Missing local-buffer mapping for {peer=}"
        assert (
            peer_buffer.data_ptr() % 16 == 0
        ), f"Local-buffer mapping for {peer=} is not 16-byte aligned"
        if peer == rank_in_group:
            assert (
                peer_buffer.data_ptr() == state.local_buff.data_ptr()
            ), "The local symmetric-memory handle does not map state.local_buff"
    return state
