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

import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform

logger = logging.getLogger(__file__)

__all__ = [
    "create_state",
    "get_token_dist",
    "reduce_scatter",
    "all_gather",
    "all_reduce_can_run",
    "all_reduce",
    "allreduce_residual_rmsnorm",
]


allreduce_residual_rmsnorm_states = {}


@dataclass
class TritonCommState:
    group: dist.ProcessGroup
    rank_in_group: int
    world_size: int
    device: torch.device
    max_numel: int = 0
    max_token_num: int = 0
    hidden_dim: int = 0
    comm_buff: torch.Tensor | None = None
    symm_mem_hdl: object | None = None


# ------------------------------------------------------------------------------
# Low-level PTX helpers
# ------------------------------------------------------------------------------


@triton.jit
def multimem_ld_reduce_128(multicast_ptrs, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $5, 1;
            @!%p0 bra end;
            multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 {$0, $1, $2, $3}, [$4];
            end:
        }
        """,
        "=r,=r,=r,=r,l,r",
        args=[multicast_ptrs, mask.to(tl.int32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def multimem_st_128(multicast_ptrs, x, y, z, w, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $6, 1;
            @!%p0 bra end;
            multimem.st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};
            end:
        }
        """,
        "=r,l,r,r,r,r,r",
        args=[multicast_ptrs, x, y, z, w, mask.to(tl.int32)],
        dtype=(tl.uint32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def local_ld_128(in_ptr, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $5, 1;
            @!%p0 bra end;
            ld.relaxed.sys.global.v4.b32 {$0, $1, $2, $3}, [$4];
            end:
        }
        """,
        "=r,=r,=r,=r,l,r",
        args=[in_ptr, mask.to(tl.int32)],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def local_st_128(out_put, x, y, z, w, mask):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $6, 1;
            @!%p0 bra end;
            st.relaxed.sys.global.v4.f32 [$1], {$2, $3, $4, $5};
            end:
        }
        """,
        "=r,l,r,r,r,r,r",
        args=[out_put, x, y, z, w, mask.to(tl.int32)],
        dtype=(tl.uint32),
        is_pure=False,
        pack=1,
    )


@triton.jit
def get_tid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %tid.x;
        mov.u32 $1, %tid.y;
        mov.u32 $2, %tid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def get_ntid():
    return tl.inline_asm_elementwise(
        """
        mov.u32 $0, %ntid.x;
        mov.u32 $1, %ntid.y;
        mov.u32 $2, %ntid.z;
        """,
        "=r,=r,=r",
        [],
        dtype=(tl.uint32, tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def get_flat_tid():
    tid_x, tid_y, tid_z = get_tid()
    ntid_x, ntid_y, _ = get_ntid()
    return tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x


@triton.jit
def get_flat_bid():
    return (
        tl.program_id(2) * tl.num_programs(1) * tl.num_programs(0)
        + tl.program_id(1) * tl.num_programs(0)
        + tl.program_id(0)
    )


@triton.jit
def sync_threads():
    tl.inline_asm_elementwise(
        "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    )


# ------------------------------------------------------------------------------
# Signal barriers
# ------------------------------------------------------------------------------


@triton.jit
def send_signal(addrs, sem: tl.constexpr):
    if sem == "relaxed":
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                send_signal:
                    atom.global.relaxed.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                    setp.eq.u32 %p0, %tmp32_0, 0;
                    @!%p0 bra send_signal;
            }
            """,
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    elif sem == "acq_rel":
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                send_signal:
                    atom.global.release.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                    setp.eq.u32 %p0, %tmp32_0, 0;
                    @!%p0 bra send_signal;
            }
            """,
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        raise RuntimeError(f"Unrecognized sem: {sem}")


@triton.jit
def wait_signal(addrs, sem: tl.constexpr):
    if sem == "relaxed":
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                wait_signal:
                    atom.global.sys.relaxed.cas.b32 %tmp32_0, [$1], 1, 0;
                    setp.eq.u32 %p0, %tmp32_0, 1;
                    @!%p0 bra wait_signal;
            }
            """,
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    elif sem == "acq_rel":
        tl.inline_asm_elementwise(
            """
            {
                .reg .u32   %tmp32_<1>;
                .reg .pred  %p<1>;

                wait_signal:
                    atom.global.sys.acquire.cas.b32 %tmp32_0, [$1], 1, 0;
                    setp.eq.u32 %p0, %tmp32_0, 1;
                    @!%p0 bra wait_signal;
            }
            """,
            "=r, l",
            [addrs],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        raise RuntimeError(f"Unrecognized sem: {sem}")


@triton.jit
def blockwise_barrier(
    signal_pad_ptrs,
    block_id,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    sem: tl.constexpr,
):
    if block_id is None:
        block_id = get_flat_bid()
    flat_tid = get_flat_tid()

    remote_ranks = tl.arange(0, world_size)
    signal_pad_ptrs = signal_pad_ptrs.to(tl.pointer_type(tl.uint64))
    remote_signal_pad_addrs = tl.load(signal_pad_ptrs + remote_ranks).to(
        tl.pointer_type(tl.uint32)
    )
    send_addrs = remote_signal_pad_addrs + block_id * world_size + rank

    local_signal_pad_addr = tl.load(signal_pad_ptrs + rank).to(
        tl.pointer_type(tl.uint32)
    )
    wait_addrs = local_signal_pad_addr + block_id * world_size + remote_ranks

    if flat_tid < world_size:
        send_signal(send_addrs, sem)
        wait_signal(wait_addrs, sem)


@triton.jit
def send_signal_to_peers(
    signal_ptrs,
    block_id,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    for peer in tl.static_range(0, world_size):
        remote_signal = tl.load(signal_ptrs + peer).to(tl.pointer_type(tl.uint32))
        send_addr = remote_signal + block_id * world_size + rank
        send_old = tl.full((), 1, tl.int32)
        while send_old != 0:
            send_old = tl.atomic_cas(send_addr, 0, 1, sem="release", scope="sys")


@triton.jit
def wait_signal_from_peers(
    local_signal,
    block_id,
    world_size: tl.constexpr,
):
    for peer in tl.static_range(0, world_size):
        wait_addr = local_signal + block_id * world_size + peer
        wait_old = tl.full((), 0, tl.int32)
        while wait_old != 1:
            wait_old = tl.atomic_cas(wait_addr, 1, 0, sem="acquire", scope="sys")


@triton.jit
def symm_mem_barrier(
    signal_pad_ptrs_dev,
    block_id,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    signal_ptrs = signal_pad_ptrs_dev.to(tl.pointer_type(tl.uint64))
    local_signal = tl.load(signal_ptrs + rank).to(tl.pointer_type(tl.uint32))
    send_signal_to_peers(signal_ptrs, block_id, rank, world_size)
    wait_signal_from_peers(local_signal, block_id, world_size)


# ------------------------------------------------------------------------------
# Shared utilities
# ------------------------------------------------------------------------------


def _get_available_gpu_memory(gpu_id: int, empty_cache: bool = True) -> float:
    if torch.cuda.current_device() != gpu_id:
        logger.warning(
            "current device is not %s, but %s, which may cause useless memory allocation for torch CUDA context.",
            gpu_id,
            torch.cuda.current_device(),
        )
    if empty_cache:
        torch.cuda.empty_cache()
    free_gpu_memory, _ = torch.cuda.mem_get_info(gpu_id)
    return free_gpu_memory / (1 << 30)


# ------------------------------------------------------------------------------
# RS/AG helpers
# ------------------------------------------------------------------------------


def rsag_get_token_dist(state: TritonCommState, total_tokens_in_group: int) -> list:
    token_list_in_group = []
    for rank in range(state.world_size):
        num_tokens_per_rank = total_tokens_in_group // state.world_size + (
            1 if (rank < total_tokens_in_group % state.world_size) else 0
        )
        token_list_in_group.append(num_tokens_per_rank)
    return token_list_in_group


def rsag_get_context(
    state: TritonCommState, token_list_in_group: list
) -> Tuple[int, int, int]:
    total_num_tokens = sum(token_list_in_group)
    assert (
        total_num_tokens <= state.max_token_num
    ), f"The inner comm buffer is too small: {total_num_tokens=} is not <= {state.max_token_num=}"
    local_num_tokens = token_list_in_group[state.rank_in_group]
    local_token_offset = sum(token_list_in_group[: state.rank_in_group])
    return total_num_tokens, local_num_tokens, local_token_offset


def rsag_resize_hidden_if_needed(state: TritonCommState, hidden_size: int):
    hidden_size_bak, comm_buff_bak = state.hidden_dim, state.comm_buff
    if hidden_size < hidden_size_bak:
        state.hidden_dim = hidden_size
        state.comm_buff = comm_buff_bak.reshape(-1)[
            : state.max_token_num * state.hidden_dim
        ].reshape(state.max_token_num, state.hidden_dim)
    return hidden_size_bak, comm_buff_bak


def rsag_restore_hidden(
    state: TritonCommState, hidden_size_bak: int, comm_buff_bak: torch.Tensor
) -> None:
    if state.hidden_dim != hidden_size_bak:
        state.hidden_dim = hidden_size_bak
        state.comm_buff = comm_buff_bak


# ------------------------------------------------------------------------------
# NVIDIA Triton RS/AG
# ------------------------------------------------------------------------------


def nvidia_rsag_get_launch_config(local_numel: int) -> Tuple[int, int, int, int]:
    warp_size = 32
    max_num_blocks = 4
    max_block_size = 1024
    bytes_per_thread = 16
    numel_per_thread = 8
    assert (
        local_numel % numel_per_thread == 0
    ), f"The number of elements must be {bytes_per_thread} bytes aligned"
    block_size = max_block_size
    num_warps = max_block_size // warp_size
    num_blocks = max_num_blocks
    return num_blocks, block_size, num_warps, numel_per_thread


@triton.jit
def nvidia_rsag_reduce_scatter_kernel(
    output_ptr,
    multicast_ptr,
    signal_pad_ptr,
    numel,
    offset,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
) -> None:
    blockwise_barrier(signal_pad_ptr, None, RANK, WORLD_SIZE, sem="relaxed")
    sync_threads()

    numel = numel // NUMEL_PER_THREAD
    pid = tl.program_id(axis=0)
    tid = get_flat_tid()
    block_start = pid * BLOCK_SIZE

    while block_start < numel:
        thread_offset = block_start + tid
        mask = thread_offset < numel
        in_ptr = (
            multicast_ptr.to(tl.int64).to(tl.pointer_type(tl.uint64))
            + (offset // NUMEL_PER_THREAD + thread_offset) * 2
        )
        out_ptr = (
            output_ptr.to(tl.pointer_type(tl.uint64))
            + (offset // NUMEL_PER_THREAD + thread_offset) * 2
        )
        x, y, z, w = multimem_ld_reduce_128(in_ptr, mask)
        local_st_128(out_ptr, x, y, z, w, mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    sync_threads()
    blockwise_barrier(signal_pad_ptr, None, RANK, WORLD_SIZE, sem="acq_rel")


@triton.jit
def nvidia_rsag_all_gather_kernel(
    input_ptr,
    multicast_ptr,
    signal_pad_ptr,
    numel,
    offset,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
) -> None:
    blockwise_barrier(signal_pad_ptr, None, RANK, WORLD_SIZE, sem="relaxed")
    sync_threads()

    numel = numel // NUMEL_PER_THREAD
    pid = tl.program_id(axis=0)
    tid = get_flat_tid()
    block_start = pid * BLOCK_SIZE

    while block_start < numel:
        thread_offset = block_start + tid
        mask = thread_offset < numel
        in_ptr = (
            input_ptr.to(tl.pointer_type(tl.uint64))
            + (offset // NUMEL_PER_THREAD + thread_offset) * 2
        )
        out_ptr = (
            multicast_ptr.to(tl.int64).to(tl.pointer_type(tl.uint64))
            + (offset // NUMEL_PER_THREAD + thread_offset) * 2
        )
        x, y, z, w = local_ld_128(in_ptr, mask)
        multimem_st_128(out_ptr, x, y, z, w, mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    sync_threads()
    blockwise_barrier(signal_pad_ptr, None, RANK, WORLD_SIZE, sem="acq_rel")


def nvidia_create_rsag_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_tokens: int,
    hidden_size: int,
    device: torch.device = None,
) -> TritonCommState:
    assert (
        type(group) == dist.ProcessGroup
    ), f"Expected dist.ProcessGroup, got {type(group)}"
    device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
    free_gpu_memory_begin = _get_available_gpu_memory(torch.cuda.current_device())
    # Allocate outside inference_mode so the persistent comm buffer is not
    # an inference tensor; this class is often lazily constructed during
    # forward (which runs under @maybe_inference_mode). Pair with no_grad
    # so we don't accidentally re-enable autograd just to escape inference.
    with torch.inference_mode(False), torch.no_grad():
        comm_buff = symm_mem.empty(
            (max_tokens, hidden_size), dtype=torch.bfloat16, device=device
        )
    free_gpu_memory_after = _get_available_gpu_memory(torch.cuda.current_device())
    logger.info(
        "Custom Triton RSAG buffer allocated: %s GB",
        free_gpu_memory_begin - free_gpu_memory_after,
    )
    symm_mem.rendezvous(comm_buff, group=group)
    return TritonCommState(
        group=group,
        rank_in_group=rank_in_group,
        world_size=group.size(),
        device=device,
        max_token_num=max_tokens,
        hidden_dim=hidden_size,
        comm_buff=comm_buff,
    )


def nvidia_rsag_multimem_reduce_scatter(
    state: TritonCommState, local_num_tokens: int, local_token_offset: int
) -> None:
    num_elts = local_num_tokens * state.hidden_dim
    num_blocks, block_size, num_warps, numel_per_thread = nvidia_rsag_get_launch_config(
        num_elts
    )
    symm_mem_hdl = symm_mem.rendezvous(state.comm_buff, group=state.group)
    assert state.rank_in_group == symm_mem_hdl.rank, "Mismatched rank id"
    grid = (num_blocks, 1, 1)
    nvidia_rsag_reduce_scatter_kernel[grid](
        output_ptr=state.comm_buff,
        multicast_ptr=symm_mem_hdl.multicast_ptr,
        signal_pad_ptr=symm_mem_hdl.signal_pad_ptrs_dev,
        numel=local_num_tokens * state.hidden_dim,
        offset=local_token_offset * state.hidden_dim,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=num_warps,
    )


def nvidia_rsag_multimem_all_gather(
    state: TritonCommState, local_num_tokens: int, local_token_offset: int
) -> None:
    num_elts = local_num_tokens * state.hidden_dim
    num_blocks, block_size, num_warps, numel_per_thread = nvidia_rsag_get_launch_config(
        num_elts
    )
    symm_mem_hdl = symm_mem.rendezvous(state.comm_buff, group=state.group)
    assert state.rank_in_group == symm_mem_hdl.rank, "Mismatched rank id"
    grid = (num_blocks, 1, 1)
    nvidia_rsag_all_gather_kernel[grid](
        input_ptr=state.comm_buff,
        multicast_ptr=symm_mem_hdl.multicast_ptr,
        signal_pad_ptr=symm_mem_hdl.signal_pad_ptrs_dev,
        numel=local_num_tokens * state.hidden_dim,
        offset=local_token_offset * state.hidden_dim,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=num_warps,
    )


def nvidia_rsag_reduce_scatter(
    state: TritonCommState,
    hidden_states: torch.Tensor,
    tp_num_tokens: int = None,
    token_list_in_group: List[int] = None,
    safe=True,
) -> torch.Tensor:
    assert (
        tp_num_tokens is not None or token_list_in_group is not None
    ), "Either tp_num_tokens or token_list_in_group must be provided"
    if token_list_in_group is None:
        token_list_in_group = rsag_get_token_dist(state, tp_num_tokens)
    assert hidden_states.dtype == torch.bfloat16, "Only bfloat16 is supported for now"
    total_num_tokens, local_num_tokens, local_token_offset = rsag_get_context(
        state, token_list_in_group
    )
    assert (hidden_states.shape[0] == total_num_tokens) and (
        hidden_states.shape[-1] == state.hidden_dim
    ), f"Mismatched shape, {hidden_states.shape[0]=} != {total_num_tokens=} or {hidden_states.shape[-1]=} != {state.hidden_dim=} {hidden_states.shape=}"
    state.comm_buff[:total_num_tokens, :].copy_(hidden_states)
    nvidia_rsag_multimem_reduce_scatter(state, local_num_tokens, local_token_offset)
    output = state.comm_buff[
        local_token_offset : (local_token_offset + local_num_tokens), :
    ]
    return output.clone() if safe else output


def nvidia_rsag_all_gather(
    state: TritonCommState,
    hidden_states: torch.Tensor,
    tp_num_tokens: int = None,
    token_list_in_group: List[int] = None,
    safe=True,
) -> torch.Tensor:
    assert (
        tp_num_tokens is not None or token_list_in_group is not None
    ), "Either tp_num_tokens or token_list_in_group must be provided"
    if token_list_in_group is None:
        token_list_in_group = rsag_get_token_dist(state, tp_num_tokens)
    assert hidden_states.dtype == torch.bfloat16, "Only bfloat16 is supported for now"
    total_num_tokens, local_num_tokens, local_token_offset = rsag_get_context(
        state, token_list_in_group
    )
    assert (hidden_states.shape[0] == local_num_tokens) and (
        hidden_states.shape[-1] <= state.hidden_dim
    ), f"{hidden_states.shape=}|{local_num_tokens=}|{hidden_states.device=} Mismatched shape"
    hidden_size_bak, comm_buff_bak = rsag_resize_hidden_if_needed(
        state, hidden_states.shape[-1]
    )
    try:
        state.comm_buff[
            local_token_offset : (local_token_offset + local_num_tokens), :
        ].copy_(hidden_states)
        nvidia_rsag_multimem_all_gather(state, local_num_tokens, local_token_offset)
        output = state.comm_buff[:total_num_tokens, :]
        return output.clone() if safe else output
    finally:
        rsag_restore_hidden(state, hidden_size_bak, comm_buff_bak)


# ------------------------------------------------------------------------------
# AMD Triton RS/AG
# ------------------------------------------------------------------------------


@triton.jit
def amd_rsag_all_gather_kernel(
    input_ptr,
    buffer_ptrs_dev,
    signal_pad_ptrs_dev,
    LOCAL_NUMEL: tl.constexpr,
    GLOBAL_OFFSET: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < LOCAL_NUMEL
    vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    buffer_ptrs = buffer_ptrs_dev.to(tl.pointer_type(tl.uint64))

    for peer in tl.static_range(0, WORLD_SIZE):
        peer_base = tl.load(buffer_ptrs + peer).to(tl.pointer_type(tl.bfloat16))
        tl.store(peer_base + GLOBAL_OFFSET + offsets, vals, mask=mask)

    symm_mem_barrier(signal_pad_ptrs_dev, tl.program_id(0), RANK, WORLD_SIZE)


@triton.jit
def amd_rsag_reduce_scatter_kernel(
    buffer_ptrs_dev,
    signal_pad_ptrs_dev,
    output_ptr,
    LOCAL_NUMEL: tl.constexpr,
    GLOBAL_OFFSET: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    symm_mem_barrier(signal_pad_ptrs_dev, block_id, RANK, WORLD_SIZE)

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < LOCAL_NUMEL
    buffer_ptrs = buffer_ptrs_dev.to(tl.pointer_type(tl.uint64))
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for peer in tl.static_range(0, WORLD_SIZE):
        peer_base = tl.load(buffer_ptrs + peer).to(tl.pointer_type(tl.bfloat16))
        acc += tl.load(peer_base + GLOBAL_OFFSET + offsets, mask=mask, other=0.0).to(
            tl.float32
        )

    tl.store(output_ptr + offsets, acc, mask=mask)
    symm_mem_barrier(signal_pad_ptrs_dev, block_id, RANK, WORLD_SIZE)


def amd_rsag_num_blocks(token_list_in_group: list[int], hidden_size: int) -> int:
    max_local_numel = max(token_list_in_group) * hidden_size
    return max(1, triton.cdiv(max_local_numel, 1024))


def amd_create_rsag_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_tokens: int,
    hidden_size: int,
    device: torch.device = None,
) -> TritonCommState:
    assert (
        type(group) == dist.ProcessGroup
    ), f"Expected dist.ProcessGroup, got {type(group)}"
    device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
    world_size = group.size()
    max_blocks = max(1, triton.cdiv(max_tokens * hidden_size, 1024))
    pad_bytes = max_blocks * world_size * 4
    symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), pad_bytes))

    free_gpu_memory_begin = _get_available_gpu_memory(torch.cuda.current_device())
    comm_buff = symm_mem.empty(
        (max_tokens, hidden_size), dtype=torch.bfloat16, device=device
    )
    symm_mem_hdl = symm_mem.rendezvous(comm_buff, group=group)
    free_gpu_memory_after = _get_available_gpu_memory(torch.cuda.current_device())
    logger.info(
        "Custom Triton RSAG AMD symmetric-memory buffer allocated: %s GB",
        free_gpu_memory_begin - free_gpu_memory_after,
    )
    assert rank_in_group == symm_mem_hdl.rank, "Mismatched rank id"
    return TritonCommState(
        group=group,
        rank_in_group=rank_in_group,
        world_size=world_size,
        device=device,
        max_token_num=max_tokens,
        hidden_dim=hidden_size,
        comm_buff=comm_buff,
        symm_mem_hdl=symm_mem_hdl,
    )


def amd_rsag_reduce_scatter(
    state: TritonCommState,
    hidden_states: torch.Tensor,
    tp_num_tokens: int = None,
    token_list_in_group: List[int] = None,
    safe=True,
) -> torch.Tensor:
    assert (
        tp_num_tokens is not None or token_list_in_group is not None
    ), "Either tp_num_tokens or token_list_in_group must be provided"
    if token_list_in_group is None:
        token_list_in_group = rsag_get_token_dist(state, tp_num_tokens)
    assert hidden_states.dtype == torch.bfloat16, "Only bfloat16 is supported for now"
    total_num_tokens, local_num_tokens, local_token_offset = rsag_get_context(
        state, token_list_in_group
    )
    assert (hidden_states.shape[0] == total_num_tokens) and (
        hidden_states.shape[-1] == state.hidden_dim
    ), f"Mismatched shape, {hidden_states.shape[0]=} != {total_num_tokens=} or {hidden_states.shape[-1]=} != {state.hidden_dim=} {hidden_states.shape=}"

    local_numel = local_num_tokens * state.hidden_dim
    global_offset = local_token_offset * state.hidden_dim
    state.comm_buff[:total_num_tokens, :].copy_(hidden_states)
    output = torch.empty(
        (local_num_tokens, state.hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    grid = (amd_rsag_num_blocks(token_list_in_group, state.hidden_dim),)
    amd_rsag_reduce_scatter_kernel[grid](
        state.symm_mem_hdl.buffer_ptrs_dev,
        state.symm_mem_hdl.signal_pad_ptrs_dev,
        output,
        LOCAL_NUMEL=local_numel,
        GLOBAL_OFFSET=global_offset,
        RANK=state.symm_mem_hdl.rank,
        WORLD_SIZE=state.symm_mem_hdl.world_size,
        BLOCK_SIZE=1024,
        num_warps=4,
    )
    return output.clone() if safe else output


def amd_rsag_all_gather(
    state: TritonCommState,
    hidden_states: torch.Tensor,
    tp_num_tokens: int = None,
    token_list_in_group: List[int] = None,
    safe=True,
) -> torch.Tensor:
    assert (
        tp_num_tokens is not None or token_list_in_group is not None
    ), "Either tp_num_tokens or token_list_in_group must be provided"
    if token_list_in_group is None:
        token_list_in_group = rsag_get_token_dist(state, tp_num_tokens)
    assert hidden_states.dtype == torch.bfloat16, "Only bfloat16 is supported for now"

    hidden_size_bak, comm_buff_bak = rsag_resize_hidden_if_needed(
        state, hidden_states.shape[-1]
    )
    try:
        total_num_tokens, local_num_tokens, local_token_offset = rsag_get_context(
            state, token_list_in_group
        )
        assert (hidden_states.shape[0] == local_num_tokens) and (
            hidden_states.shape[-1] <= state.hidden_dim
        ), f"{hidden_states.shape=}|{local_num_tokens=}|{hidden_states.device=} Mismatched shape"
        local_numel = local_num_tokens * state.hidden_dim
        global_offset = local_token_offset * state.hidden_dim
        grid = (amd_rsag_num_blocks(token_list_in_group, state.hidden_dim),)
        amd_rsag_all_gather_kernel[grid](
            hidden_states,
            state.symm_mem_hdl.buffer_ptrs_dev,
            state.symm_mem_hdl.signal_pad_ptrs_dev,
            LOCAL_NUMEL=local_numel,
            GLOBAL_OFFSET=global_offset,
            RANK=state.symm_mem_hdl.rank,
            WORLD_SIZE=state.symm_mem_hdl.world_size,
            BLOCK_SIZE=1024,
            num_warps=4,
        )
        output = state.comm_buff[:total_num_tokens, :]
        return output.clone() if safe else output
    finally:
        rsag_restore_hidden(state, hidden_size_bak, comm_buff_bak)


# ------------------------------------------------------------------------------
# AMD Triton All-Reduce
# ------------------------------------------------------------------------------


@triton.jit
def amd_all_reduce_kernel(
    buffer_ptrs_dev,
    signal_pad_ptrs_dev,
    output_ptr,
    NUMEL,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    symm_mem_barrier(signal_pad_ptrs_dev, block_id, RANK, WORLD_SIZE)

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUMEL
    buffer_ptrs = buffer_ptrs_dev.to(tl.pointer_type(tl.uint64))
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for peer in tl.static_range(0, WORLD_SIZE):
        peer_base = tl.load(buffer_ptrs + peer).to(tl.pointer_type(tl.bfloat16))
        acc += tl.load(peer_base + offsets, mask=mask, other=0.0).to(tl.float32)

    tl.store(output_ptr + offsets, acc, mask=mask)

    symm_mem_barrier(signal_pad_ptrs_dev, block_id, RANK, WORLD_SIZE)


# ------------------------------------------------------------------------------
# AMD Triton All-Reduce + RMSNorm
# ------------------------------------------------------------------------------


@triton.jit
def amd_allreduce_residual_rmsnorm_kernel(
    buffer_ptrs_dev,
    signal_pad_ptrs_dev,
    residual_ptr,
    weight_ptr,
    norm_out_ptr,
    residual_out_ptr,
    HIDDEN_SIZE: tl.constexpr,
    EPS: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    symm_mem_barrier(signal_pad_ptrs_dev, row, RANK, WORLD_SIZE)

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HIDDEN_SIZE
    row_offsets = row * HIDDEN_SIZE + offsets
    buffer_ptrs = buffer_ptrs_dev.to(tl.pointer_type(tl.uint64))
    reduced = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for peer in tl.static_range(0, WORLD_SIZE):
        peer_base = tl.load(buffer_ptrs + peer).to(tl.pointer_type(tl.bfloat16))
        reduced += tl.load(peer_base + row_offsets, mask=mask, other=0.0).to(tl.float32)

    residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    residual_out = reduced + residual
    tl.store(residual_out_ptr + row_offsets, residual_out, mask=mask)

    variance = tl.sum(residual_out * residual_out, axis=0) / HIDDEN_SIZE
    scale = tl.rsqrt(variance + EPS)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(norm_out_ptr + row_offsets, residual_out * scale * weight, mask=mask)

    symm_mem_barrier(signal_pad_ptrs_dev, row, RANK, WORLD_SIZE)


def create_allreduce_residual_rmsnorm_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_token_num: int,
    hidden_dim: int,
    device: torch.device = None,
) -> TritonCommState:
    assert (
        type(group) == dist.ProcessGroup
    ), f"Expected dist.ProcessGroup, got {type(group)}"
    device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
    world_size = group.size()
    comm_buff = None
    symm_mem_hdl = None

    platform = current_platform()
    if platform.is_amd:
        pad_bytes = max_token_num * world_size * 4
        symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), pad_bytes))
        free_gpu_memory_begin = _get_available_gpu_memory(torch.cuda.current_device())
        comm_buff = symm_mem.empty(
            (max_token_num, hidden_dim), dtype=torch.bfloat16, device=device
        )
        symm_mem_hdl = symm_mem.rendezvous(comm_buff, group=group)
        free_gpu_memory_after = _get_available_gpu_memory(torch.cuda.current_device())
        logger.info(
            "Triton AR+RMSNorm AMD symmetric-memory buffer allocated: %s GB",
            free_gpu_memory_begin - free_gpu_memory_after,
        )
        assert rank_in_group == symm_mem_hdl.rank, "Mismatched rank id"
    else:
        assert platform.is_nvidia, f"Unsupported platform: {platform}"

    return TritonCommState(
        group=group,
        rank_in_group=rank_in_group,
        world_size=world_size,
        device=device,
        max_token_num=max_token_num,
        hidden_dim=hidden_dim,
        comm_buff=comm_buff,
        symm_mem_hdl=symm_mem_hdl,
    )


def allreduce_residual_rmsnorm_get_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_token_num: int,
    hidden_dim: int,
    device: torch.device = None,
) -> TritonCommState:
    key = (id(group), max_token_num, hidden_dim)
    state = allreduce_residual_rmsnorm_states.get(key)
    if state is None:
        state = create_allreduce_residual_rmsnorm_state(
            group=group,
            rank_in_group=rank_in_group,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            device=device,
        )
        allreduce_residual_rmsnorm_states[key] = state
    return state


def allreduce_residual_rmsnorm_can_run(
    state: TritonCommState,
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    platform = current_platform()
    return (
        platform.is_amd
        and state.symm_mem_hdl is not None
        and input_tensor.is_cuda
        and residual.is_cuda
        and weight.is_cuda
        and input_tensor.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
        and input_tensor.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and input_tensor.shape == residual.shape
        and input_tensor.dim() == 2
        and input_tensor.shape[0] <= state.max_token_num
        and input_tensor.shape[1] == state.hidden_dim
        and weight.shape[0] == state.hidden_dim
        and state.world_size > 1
    )


def allreduce_residual_rmsnorm(
    input_tensor: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    rank: int,
    group: dist.ProcessGroup,
    eps: float = 1e-6,
    max_token_num: int = 2048,
    use_oneshot: bool | None = None,
    trigger_completion_at_end: bool = False,
    fp32_acc: bool = False,
    block_quant_fp8: bool = False,
    residual_reduce_scattered: bool = False,
    has_partial_norm_out: bool = False,
    max_sm_to_use: int | None = None,
    launch_with_pdl: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
    platform = current_platform()
    if platform.is_amd:
        if (
            block_quant_fp8
            or residual_reduce_scattered
            or has_partial_norm_out
            or input_tensor.dim() != 2
            or residual is None
        ):
            return None, None, None, None

        token_num, hidden_dim = input_tensor.shape

        from . import iris as _iris_mod

        if (
            input_tensor.is_cuda
            and residual.is_cuda
            and weight.is_cuda
            and input_tensor.is_contiguous()
            and residual.is_contiguous()
            and weight.is_contiguous()
            and input_tensor.dtype == torch.bfloat16
            and residual.dtype == torch.bfloat16
            and input_tensor.shape == residual.shape
            and weight.shape == (hidden_dim,)
            and group.size() > 1
            and token_num <= max_token_num
        ):
            key = (id(group), max_token_num, hidden_dim, input_tensor.dtype)
            iris_state = _iris_mod.IRIS_AR_RMSNORM_STATES.get(key)
            if iris_state is None:
                iris_state = _iris_mod.create_iris_ar_rmsnorm_state(
                    group=group,
                    rank_in_group=rank,
                    max_token_num=max_token_num,
                    hidden_dim=hidden_dim,
                    dtype=input_tensor.dtype,
                )
                _iris_mod.IRIS_AR_RMSNORM_STATES[key] = iris_state
            norm_out, residual_out = _iris_mod.iris_allreduce_residual_rmsnorm(
                iris_state,
                input_tensor=input_tensor,
                residual=residual,
                weight=weight,
                eps=eps,
            )
            return norm_out, residual_out, None, None

        state = allreduce_residual_rmsnorm_get_state(
            group=group,
            rank_in_group=rank,
            max_token_num=max_token_num,
            hidden_dim=hidden_dim,
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        )
        if not allreduce_residual_rmsnorm_can_run(
            state, input_tensor, residual, weight
        ):
            return None, None, None, None

        state.comm_buff[:token_num, :].copy_(input_tensor)
        norm_out = torch.empty_like(input_tensor)
        residual_out = torch.empty_like(residual)
        amd_allreduce_residual_rmsnorm_kernel[(token_num,)](
            state.symm_mem_hdl.buffer_ptrs_dev,
            state.symm_mem_hdl.signal_pad_ptrs_dev,
            residual,
            weight,
            norm_out,
            residual_out,
            HIDDEN_SIZE=hidden_dim,
            EPS=eps,
            RANK=state.symm_mem_hdl.rank,
            WORLD_SIZE=state.symm_mem_hdl.world_size,
            BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
            num_warps=8,
        )
        return norm_out, residual_out, None, None
    else:
        assert platform.is_nvidia, f"Unsupported platform: {platform}"
        return None, None, None, None


# ------------------------------------------------------------------------------
# Public interface
# ------------------------------------------------------------------------------


def create_state(
    group: dist.ProcessGroup,
    rank_in_group: int,
    max_tokens: int = 0,
    hidden_size: int = 0,
    device: torch.device = None,
    max_numel: int = 0,
) -> TritonCommState:
    assert (
        type(group) == dist.ProcessGroup
    ), f"Expected dist.ProcessGroup, got {type(group)}"
    if max_numel:
        device = device or torch.device(f"cuda:{torch.cuda.current_device()}")
        world_size = group.size()
        comm_buff = None
        symm_mem_hdl = None

        platform = current_platform()
        if platform.is_amd:
            max_blocks = max(1, triton.cdiv(max_numel, 1024))
            pad_bytes = max_blocks * world_size * 4
            symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), pad_bytes))
            free_gpu_memory_begin = _get_available_gpu_memory(
                torch.cuda.current_device()
            )
            comm_buff = symm_mem.empty(
                (max_numel,), dtype=torch.bfloat16, device=device
            )
            symm_mem_hdl = symm_mem.rendezvous(comm_buff, group=group)
            free_gpu_memory_after = _get_available_gpu_memory(
                torch.cuda.current_device()
            )
            logger.info(
                "Triton all-reduce AMD symmetric-memory buffer allocated: %s GB",
                free_gpu_memory_begin - free_gpu_memory_after,
            )
            assert rank_in_group == symm_mem_hdl.rank, "Mismatched rank id"
        else:
            assert platform.is_nvidia, f"Unsupported platform: {platform}"

        return TritonCommState(
            group=group,
            rank_in_group=rank_in_group,
            world_size=world_size,
            device=device,
            max_numel=max_numel,
            comm_buff=comm_buff,
            symm_mem_hdl=symm_mem_hdl,
        )

    assert max_tokens > 0, "max_tokens must be specified for RS/AG state"
    assert hidden_size > 0, "hidden_size must be specified for RS/AG state"
    platform = current_platform()
    if platform.is_amd:
        return amd_create_rsag_state(
            group=group,
            rank_in_group=rank_in_group,
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            device=device,
        )
    else:
        assert platform.is_nvidia, f"Unsupported platform: {platform}"
        return nvidia_create_rsag_state(
            group=group,
            rank_in_group=rank_in_group,
            max_tokens=max_tokens,
            hidden_size=hidden_size,
            device=device,
        )


def all_reduce_can_run(state: TritonCommState, tensor: torch.Tensor, op=None) -> bool:
    if op is None:
        op = torch.distributed.ReduceOp.SUM
    platform = current_platform()
    return (
        platform.is_amd
        and state.symm_mem_hdl is not None
        and op == torch.distributed.ReduceOp.SUM
        and tensor.is_cuda
        and tensor.is_contiguous()
        and tensor.dtype == torch.bfloat16
        and 0 < tensor.numel() <= state.max_numel
        and state.world_size > 1
    )


def all_reduce(state: TritonCommState, tensor: torch.Tensor, op=None) -> torch.Tensor:
    assert all_reduce_can_run(state, tensor, op=op)
    numel = tensor.numel()
    state.comm_buff[:numel].copy_(tensor.reshape(-1))
    grid = (triton.cdiv(numel, 1024),)
    amd_all_reduce_kernel[grid](
        state.symm_mem_hdl.buffer_ptrs_dev,
        state.symm_mem_hdl.signal_pad_ptrs_dev,
        tensor,
        numel,
        RANK=state.symm_mem_hdl.rank,
        WORLD_SIZE=state.symm_mem_hdl.world_size,
        BLOCK_SIZE=1024,
        num_warps=4,
    )
    return tensor


def get_token_dist(state: TritonCommState, total_tokens_in_group: int) -> list:
    return rsag_get_token_dist(state, total_tokens_in_group)


def reduce_scatter(
    state: TritonCommState,
    hidden_states: torch.Tensor,
    tp_num_tokens: int = None,
    token_list_in_group: List[int] = None,
    safe=True,
) -> torch.Tensor:
    platform = current_platform()
    if platform.is_amd:
        return amd_rsag_reduce_scatter(
            state,
            hidden_states,
            tp_num_tokens=tp_num_tokens,
            token_list_in_group=token_list_in_group,
            safe=safe,
        )
    else:
        assert platform.is_nvidia, f"Unsupported platform: {platform}"
        return nvidia_rsag_reduce_scatter(
            state,
            hidden_states,
            tp_num_tokens=tp_num_tokens,
            token_list_in_group=token_list_in_group,
            safe=safe,
        )


def all_gather(
    state: TritonCommState,
    hidden_states: torch.Tensor,
    tp_num_tokens: int = None,
    token_list_in_group: List[int] = None,
    safe=True,
) -> torch.Tensor:
    platform = current_platform()
    if platform.is_amd:
        return amd_rsag_all_gather(
            state,
            hidden_states,
            tp_num_tokens=tp_num_tokens,
            token_list_in_group=token_list_in_group,
            safe=safe,
        )
    else:
        assert platform.is_nvidia, f"Unsupported platform: {platform}"
        return nvidia_rsag_all_gather(
            state,
            hidden_states,
            tp_num_tokens=tp_num_tokens,
            token_list_in_group=token_list_in_group,
            safe=safe,
        )
