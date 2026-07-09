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
from typing import Any, List, Tuple

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
    "all_gather_inner",
    "all_reduce_can_run",
    "all_reduce",
    "allreduce_residual_rmsnorm",
    "create_dp_sampling_state",
    "dp_sampling_gather",
    "dp_sampling_swap",
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


@dataclass
class DpSamplingState:
    """Symmetric-memory workspace reused across graph replays.

    recv_logits stores this rank's request shard as [max_reqs_per_rank, N, V].
    Verify buffers store full padded-batch outputs:
    recv_predict[max_pad_bs, N], recv_accept_idx[max_pad_bs, N], and
    recv_accept_len[max_pad_bs].
    """

    group: dist.ProcessGroup
    rank_in_group: int
    tp_size: int
    device: torch.device

    max_pad_bs: int
    num_tokens_per_req: int
    vocab_size: int
    logits_dtype: torch.dtype

    recv_logits: torch.Tensor | None = None
    recv_predict: torch.Tensor | None = None
    recv_accept_idx: torch.Tensor | None = None
    recv_accept_len: torch.Tensor | None = None

    # Keep handles alive; kernels use their peer pointers and signal pads.
    recv_logits_hdl: Any | None = None
    recv_predict_hdl: Any | None = None
    recv_accept_idx_hdl: Any | None = None
    recv_accept_len_hdl: Any | None = None

    recv_logits_peer_ptrs: torch.Tensor | None = None
    recv_predict_peer_ptrs: torch.Tensor | None = None
    recv_accept_idx_peer_ptrs: torch.Tensor | None = None
    recv_accept_len_peer_ptrs: torch.Tensor | None = None
    flags_peer_ptrs: torch.Tensor | None = None


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
# Batch-DP speculative verify helpers
# ------------------------------------------------------------------------------


@triton.jit
def _dp_sampling_swap_kernel(
    local_logits,
    recv_logits_ptrs_dev,
    REQS_PER_RANK: tl.constexpr,
    N: tl.constexpr,
    V_LOCAL: tl.constexpr,
    V: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LOGITS_DTYPE_CODE: tl.constexpr,
):
    pid = tl.program_id(0)
    vocab_blocks = tl.cdiv(V_LOCAL, BLOCK_SIZE)

    vocab_block = pid % vocab_blocks
    tmp = pid // vocab_blocks
    draft_pos = tmp % N
    tmp = tmp // N
    local_req = tmp % REQS_PER_RANK
    dst_rank = tmp // REQS_PER_RANK

    offsets = vocab_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < V_LOCAL

    src_row = dst_rank * REQS_PER_RANK * N + local_req * N + draft_pos
    vals = tl.load(local_logits + src_row * V_LOCAL + offsets, mask=mask)

    peer_ptrs = recv_logits_ptrs_dev.to(tl.pointer_type(tl.uint64))
    if LOGITS_DTYPE_CODE == 0:
        peer_base = tl.load(peer_ptrs + dst_rank).to(tl.pointer_type(tl.bfloat16))
    elif LOGITS_DTYPE_CODE == 1:
        peer_base = tl.load(peer_ptrs + dst_rank).to(tl.pointer_type(tl.float16))
    else:
        peer_base = tl.load(peer_ptrs + dst_rank).to(tl.pointer_type(tl.float32))

    dst_offset = local_req * N * V + draft_pos * V + RANK * V_LOCAL + offsets
    tl.store(peer_base + dst_offset, vals, mask=mask)


@triton.jit
def _dp_sampling_swap_barrier_kernel(
    signal_pad_ptrs_dev,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    symm_mem_barrier(signal_pad_ptrs_dev, 0, RANK, WORLD_SIZE)


@triton.jit
def _dp_sampling_gather_kernel(
    predict_local,
    accept_index_local,
    accept_length_local,
    recv_predict_ptrs_dev,
    recv_accept_idx_ptrs_dev,
    recv_accept_len_ptrs_dev,
    REQS_PER_RANK: tl.constexpr,
    N: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    local_req = pid % REQS_PER_RANK
    dst_rank = pid // REQS_PER_RANK

    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N

    src_base = local_req * N
    pred_vals = tl.load(predict_local + src_base + offsets, mask=mask)
    accept_idx_vals = tl.load(accept_index_local + src_base + offsets, mask=mask)
    accept_len_val = tl.load(accept_length_local + local_req)

    pred_ptrs = recv_predict_ptrs_dev.to(tl.pointer_type(tl.uint64))
    accept_idx_ptrs = recv_accept_idx_ptrs_dev.to(tl.pointer_type(tl.uint64))
    accept_len_ptrs = recv_accept_len_ptrs_dev.to(tl.pointer_type(tl.uint64))

    pred_peer = tl.load(pred_ptrs + dst_rank).to(tl.pointer_type(tl.int32))
    accept_idx_peer = tl.load(accept_idx_ptrs + dst_rank).to(tl.pointer_type(tl.int32))
    accept_len_peer = tl.load(accept_len_ptrs + dst_rank).to(tl.pointer_type(tl.int32))

    dst_row = RANK * REQS_PER_RANK + local_req
    dst_base = dst_row * N
    tl.store(pred_peer + dst_base + offsets, pred_vals, mask=mask)
    tl.store(accept_idx_peer + dst_base + offsets, accept_idx_vals, mask=mask)
    tl.store(accept_len_peer + dst_row, accept_len_val)


@triton.jit
def _dp_sampling_gather_barrier_kernel(
    signal_pad_ptrs_dev,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    symm_mem_barrier(signal_pad_ptrs_dev, 0, RANK, WORLD_SIZE)


def _logits_dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.float32:
        return "fp32"
    raise AssertionError(f"Unsupported dp-sampling logits dtype: {dtype}")


def _logits_dtype_code(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 0
    if dtype == torch.float16:
        return 1
    if dtype == torch.float32:
        return 2
    raise AssertionError(f"Unsupported dp-sampling logits dtype: {dtype}")


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _alloc_symm(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    group: dist.ProcessGroup,
):
    with torch.inference_mode(False), torch.no_grad():
        tensor = symm_mem.empty(shape, dtype=dtype, device=device)
    handle = symm_mem.rendezvous(tensor, group=group)
    return tensor, handle


def _peer_ptrs_dev(
    handle: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    ptrs = [
        handle.get_buffer(peer, shape, dtype, storage_offset=0).data_ptr()
        for peer in range(world_size)
    ]
    return torch.tensor(ptrs, dtype=torch.uint64, device=device)


def create_dp_sampling_state(
    *,
    group: dist.ProcessGroup,
    rank_in_group: int,
    tp_size: int,
    max_pad_bs: int,
    num_tokens_per_req: int,
    vocab_size: int,
    logits_dtype: torch.dtype,
    device: torch.device,
) -> DpSamplingState:
    """Allocate symmetric-memory buffers and peer pointer tables.

    Logits storage is [max_reqs_per_rank, N, V] per rank, where
    max_reqs_per_rank=max_pad_bs/TP. Verify-output storage is full-batch:
    predict[max_pad_bs, N], accept_index[max_pad_bs, N], and
    accept_length[max_pad_bs].
    """
    assert isinstance(
        group, dist.ProcessGroup
    ), f"Expected ProcessGroup, got {type(group)}"
    assert rank_in_group == dist.get_rank(group), (
        f"rank_in_group={rank_in_group} does not match process-group rank "
        f"{dist.get_rank(group)}"
    )
    assert tp_size == group.size(), f"tp_size={tp_size} != group.size()={group.size()}"
    assert max_pad_bs % tp_size == 0
    assert vocab_size % tp_size == 0
    assert num_tokens_per_req >= 1
    _logits_dtype_name(logits_dtype)

    max_reqs_per_rank = max_pad_bs // tp_size
    v_local = vocab_size // tp_size
    swap_block_size = min(1024, _next_power_of_2(v_local))
    gather_block_n = min(1024, _next_power_of_2(num_tokens_per_req))
    swap_max_blocks = (
        tp_size
        * max_reqs_per_rank
        * num_tokens_per_req
        * triton.cdiv(v_local, swap_block_size)
    )
    gather_max_blocks = tp_size * max_reqs_per_rank
    signal_pad_bytes = max(swap_max_blocks, gather_max_blocks) * tp_size * 4
    symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), signal_pad_bytes))

    recv_logits, recv_logits_hdl = _alloc_symm(
        (max_reqs_per_rank, num_tokens_per_req, vocab_size), logits_dtype, device, group
    )
    recv_predict, recv_predict_hdl = _alloc_symm(
        (max_pad_bs, num_tokens_per_req), torch.int32, device, group
    )
    recv_accept_idx, recv_accept_idx_hdl = _alloc_symm(
        (max_pad_bs, num_tokens_per_req), torch.int32, device, group
    )
    recv_accept_len, recv_accept_len_hdl = _alloc_symm(
        (max_pad_bs,), torch.int32, device, group
    )

    return DpSamplingState(
        group=group,
        rank_in_group=rank_in_group,
        tp_size=tp_size,
        device=device,
        max_pad_bs=max_pad_bs,
        num_tokens_per_req=num_tokens_per_req,
        vocab_size=vocab_size,
        logits_dtype=logits_dtype,
        recv_logits=recv_logits,
        recv_predict=recv_predict,
        recv_accept_idx=recv_accept_idx,
        recv_accept_len=recv_accept_len,
        recv_logits_hdl=recv_logits_hdl,
        recv_predict_hdl=recv_predict_hdl,
        recv_accept_idx_hdl=recv_accept_idx_hdl,
        recv_accept_len_hdl=recv_accept_len_hdl,
        recv_logits_peer_ptrs=_peer_ptrs_dev(
            recv_logits_hdl, recv_logits.shape, recv_logits.dtype, tp_size, device
        ),
        recv_predict_peer_ptrs=_peer_ptrs_dev(
            recv_predict_hdl, recv_predict.shape, recv_predict.dtype, tp_size, device
        ),
        recv_accept_idx_peer_ptrs=_peer_ptrs_dev(
            recv_accept_idx_hdl,
            recv_accept_idx.shape,
            recv_accept_idx.dtype,
            tp_size,
            device,
        ),
        recv_accept_len_peer_ptrs=_peer_ptrs_dev(
            recv_accept_len_hdl,
            recv_accept_len.shape,
            recv_accept_len.dtype,
            tp_size,
            device,
        ),
        flags_peer_ptrs=recv_logits_hdl.signal_pad_ptrs_dev,
    )


def dp_sampling_swap(
    state: DpSamplingState,
    local_logits: torch.Tensor,
    *,
    pad_bs: int,
) -> torch.Tensor:
    """Move logits from vocab shards to request shards.

    Input is local_logits[pad_bs * N, V_local] on each rank, where
    V_local=V/TP. Output is a view of state.recv_logits with shape
    [reqs_per_rank * N, V] for this rank's reqs_per_rank=pad_bs/TP
    requests.
    Returned row local_req * N + d is global request
    rank * reqs_per_rank + local_req at draft position d.
    """
    tp_size = state.tp_size
    n = state.num_tokens_per_req
    vocab_size = state.vocab_size
    assert pad_bs <= state.max_pad_bs
    assert pad_bs % tp_size == 0
    assert vocab_size % tp_size == 0
    assert local_logits.is_cuda and local_logits.is_contiguous()
    assert local_logits.dtype == state.logits_dtype

    reqs_per_rank = pad_bs // tp_size
    v_local = vocab_size // tp_size
    expected_shape = (pad_bs * n, v_local)
    assert (
        tuple(local_logits.shape) == expected_shape
    ), f"local_logits shape {tuple(local_logits.shape)} != {expected_shape}"
    assert state.recv_logits is not None
    assert state.recv_logits_peer_ptrs is not None
    assert state.flags_peer_ptrs is not None

    block_size = min(1024, _next_power_of_2(v_local))
    grid = (tp_size * reqs_per_rank * n * triton.cdiv(v_local, block_size),)
    _dp_sampling_swap_kernel[grid](
        local_logits,
        state.recv_logits_peer_ptrs,
        REQS_PER_RANK=reqs_per_rank,
        N=n,
        V_LOCAL=v_local,
        V=vocab_size,
        RANK=state.rank_in_group,
        WORLD_SIZE=tp_size,
        BLOCK_SIZE=block_size,
        LOGITS_DTYPE_CODE=_logits_dtype_code(state.logits_dtype),
        num_warps=4,
    )
    _dp_sampling_swap_barrier_kernel[(1,)](
        state.flags_peer_ptrs,
        RANK=state.rank_in_group,
        WORLD_SIZE=tp_size,
        num_warps=1,
    )
    return state.recv_logits[:reqs_per_rank].view(reqs_per_rank * n, vocab_size)


def dp_sampling_gather(
    state: DpSamplingState,
    predict_local: torch.Tensor,
    accept_index_local: torch.Tensor,
    accept_length_local: torch.Tensor,
    *,
    pad_bs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather per-rank verify outputs into full padded-batch buffers.

    Inputs are predict_local[reqs_per_rank, N],
    accept_index_local[reqs_per_rank, N], and
    accept_length_local[reqs_per_rank].
    Returns views predict[pad_bs, N], accept_index[pad_bs, N], and
    accept_length[pad_bs] from symmetric memory.
    Row r from source rank src lands at src * reqs_per_rank + r.
    """
    tp_size = state.tp_size
    n = state.num_tokens_per_req
    assert pad_bs <= state.max_pad_bs
    assert pad_bs % tp_size == 0

    reqs_per_rank = pad_bs // tp_size
    assert tuple(predict_local.shape) == (reqs_per_rank, n)
    assert tuple(accept_index_local.shape) == (reqs_per_rank, n)
    assert tuple(accept_length_local.shape) == (reqs_per_rank,)
    assert predict_local.is_cuda and predict_local.is_contiguous()
    assert accept_index_local.is_cuda and accept_index_local.is_contiguous()
    assert accept_length_local.is_cuda and accept_length_local.is_contiguous()
    assert predict_local.dtype == torch.int32
    assert accept_index_local.dtype == torch.int32
    assert accept_length_local.dtype == torch.int32
    assert state.recv_predict is not None
    assert state.recv_accept_idx is not None
    assert state.recv_accept_len is not None
    assert state.recv_predict_peer_ptrs is not None
    assert state.recv_accept_idx_peer_ptrs is not None
    assert state.recv_accept_len_peer_ptrs is not None
    assert state.flags_peer_ptrs is not None

    block_n = min(1024, _next_power_of_2(n))
    grid = (tp_size * reqs_per_rank,)
    _dp_sampling_gather_kernel[grid](
        predict_local,
        accept_index_local,
        accept_length_local,
        state.recv_predict_peer_ptrs,
        state.recv_accept_idx_peer_ptrs,
        state.recv_accept_len_peer_ptrs,
        REQS_PER_RANK=reqs_per_rank,
        N=n,
        RANK=state.rank_in_group,
        WORLD_SIZE=tp_size,
        BLOCK_N=block_n,
        num_warps=1,
    )
    _dp_sampling_gather_barrier_kernel[(1,)](
        state.flags_peer_ptrs,
        RANK=state.rank_in_group,
        WORLD_SIZE=tp_size,
        num_warps=1,
    )
    return (
        state.recv_predict[:pad_bs],
        state.recv_accept_idx[:pad_bs],
        state.recv_accept_len[:pad_bs],
    )


# ------------------------------------------------------------------------------
# Shared utilities
# ------------------------------------------------------------------------------


def _get_available_gpu_memory(gpu_id: int, empty_cache: bool = True) -> float:
    if torch.cuda.current_device() != gpu_id:
        logger.warning(
            f"current device is not {gpu_id}, but {torch.cuda.current_device()}, which may cause useless memory allocation for torch CUDA context."
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

# multimem reduce-scatter / all-gather launch geometry. A CTA runs
# _RSAG_BLOCK_THREADS threads; each thread moves _RSAG_NUMEL_PER_THREAD bf16
# elements with one 128-bit multimem op (16 bytes / 2 bytes per bf16). So a CTA
# sweeps _RSAG_BLOCK_THREADS * _RSAG_NUMEL_PER_THREAD elements per grid-stride
# step. get_launch_config and the reduce-scatter block-count heuristic share
# these constants so the two can never drift apart.
_RSAG_BLOCK_THREADS = 1024
_RSAG_NUMEL_PER_THREAD = 8
# The multimem kernels grid-stride, so the CTA count is a free tuning knob, not
# fixed by the data. Reduce-scatter scales it with the payload between these
# bounds: _RSAG_MIN_BLOCKS is the smallest grid we ever launch (and the
# all-gather fallback); _RSAG_MAX_BLOCKS caps it at a count that still saturates
# NVLink while bounding the signal-pad slots nvidia_create_rsag_state reserves.
_RSAG_MIN_BLOCKS = 4
_RSAG_MAX_BLOCKS = 32


def nvidia_rsag_get_launch_config(
    local_numel: int, num_blocks: int | None = None
) -> Tuple[int, int, int, int]:
    warp_size = 32
    max_block_size = _RSAG_BLOCK_THREADS
    bytes_per_thread = 16
    numel_per_thread = _RSAG_NUMEL_PER_THREAD
    assert (
        local_numel % numel_per_thread == 0
    ), f"The number of elements must be {bytes_per_thread} bytes aligned"
    block_size = max_block_size
    num_warps = max_block_size // warp_size
    # Reduce-scatter passes a payload-scaled count; the all-gather paths leave it
    # None and fall back to the minimum grid.
    num_blocks = _RSAG_MIN_BLOCKS if num_blocks is None else num_blocks
    return num_blocks, block_size, num_warps, numel_per_thread


def nvidia_rsag_reduce_scatter_num_blocks(
    token_list_in_group: list[int], hidden_size: int
) -> int:
    """Choose how many CTAs (grid blocks) the reduce-scatter kernel launches.

    The multimem kernel is a grid-stride loop, so the block count is a free
    tuning knob rather than dictated by the data: more CTAs expose more
    parallelism on a large payload, fewer avoid cross-CTA barrier cost on a small
    one. The count is sized from the busiest rank in the group
    (``max(token_list_in_group)``) because a collective must launch an identical
    grid on every rank for the kernel's per-CTA cross-rank barrier to pair up.

    Args:
        token_list_in_group: Per-rank token counts participating in this
            collective (identical on every rank).
        hidden_size: Hidden dimension, i.e. elements per token.

    Returns:
        A power of two in ``[_RSAG_MIN_BLOCKS, _RSAG_MAX_BLOCKS]``: enough CTAs
        for roughly one grid-stride pass over the busiest rank's payload, floored
        and capped, then rounded up to a power of two. The cap is exactly what
        ``nvidia_create_rsag_state`` reserves signal-pad slots for.
    """
    # Elements one CTA sweeps per grid-stride step (threads * elems-per-thread).
    numel_per_program = _RSAG_BLOCK_THREADS * _RSAG_NUMEL_PER_THREAD
    max_local_numel = max(token_list_in_group) * hidden_size
    needed_blocks = max(
        _RSAG_MIN_BLOCKS, triton.cdiv(max_local_numel, numel_per_program)
    )
    return min(_RSAG_MAX_BLOCKS, triton.next_power_of_2(needed_blocks))


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
    # Reserve the symmetric-memory signal pad for the largest grid the
    # reduce-scatter launcher can pick. blockwise_barrier indexes the pad at
    # block_id * world_size + rank, so an _RSAG_MAX_BLOCKS-CTA grid needs
    # _RSAG_MAX_BLOCKS * world_size uint32 (4-byte) slots. Otherwise this path
    # silently relies on PyTorch's default pad size, which a smaller-payload
    # module could have set below what 32 CTAs need. max() only grows the pad, so
    # we never shrink one another module enlarged. Must precede symm_mem.empty()
    # below, which bakes the pad size into the allocation.
    pad_bytes = _RSAG_MAX_BLOCKS * group.size() * 4
    symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), pad_bytes))
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
        f"Custom Triton RSAG symmetric-memory buffer allocated: {free_gpu_memory_begin - free_gpu_memory_after} GB"
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
    state: TritonCommState,
    local_num_tokens: int,
    local_token_offset: int,
    num_blocks: int | None = None,
) -> None:
    num_elts = local_num_tokens * state.hidden_dim
    num_blocks, block_size, num_warps, numel_per_thread = nvidia_rsag_get_launch_config(
        num_elts, num_blocks=num_blocks
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
    num_blocks = nvidia_rsag_reduce_scatter_num_blocks(
        token_list_in_group, state.hidden_dim
    )
    nvidia_rsag_multimem_reduce_scatter(
        state, local_num_tokens, local_token_offset, num_blocks=num_blocks
    )
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
        f"Custom Triton RSAG AMD symmetric-memory buffer allocated: {free_gpu_memory_begin - free_gpu_memory_after} GB"
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
            f"Triton AR+RMSNorm AMD symmetric-memory buffer allocated: {free_gpu_memory_begin - free_gpu_memory_after} GB"
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
                f"Triton all-reduce AMD symmetric-memory buffer allocated: {free_gpu_memory_begin - free_gpu_memory_after} GB"
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


INNER_AG_NUMEL_PER_THREAD = 8


@triton.jit
def nvidia_rsag_all_gather_kernel_inner(
    input_ptr,
    multicast_ptr,
    signal_pad_ptr,
    total_tokens,
    hidden_offset,
    LOCAL_HIDDEN: tl.constexpr,
    TOTAL_HIDDEN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUMEL_PER_THREAD: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    SKIP_ENTRY_SYNC: tl.constexpr,
) -> None:
    if SKIP_ENTRY_SYNC == 0:
        blockwise_barrier(signal_pad_ptr, None, RANK, WORLD_SIZE, sem="relaxed")
        sync_threads()

    chunks_per_row: tl.constexpr = LOCAL_HIDDEN // NUMEL_PER_THREAD
    total_hidden_chunks: tl.constexpr = TOTAL_HIDDEN // NUMEL_PER_THREAD
    hidden_offset_chunks = hidden_offset // NUMEL_PER_THREAD
    total_chunks = total_tokens * chunks_per_row

    pid = tl.program_id(axis=0)
    tid = get_flat_tid()
    block_start = pid * BLOCK_SIZE

    while block_start < total_chunks:
        chunk = block_start + tid
        mask = chunk < total_chunks
        row = chunk // chunks_per_row
        col_chunk = chunk % chunks_per_row

        in_ptr = input_ptr.to(tl.pointer_type(tl.uint64)) + chunk * 2
        out_chunk = row * total_hidden_chunks + hidden_offset_chunks + col_chunk
        out_ptr = (
            multicast_ptr.to(tl.int64).to(tl.pointer_type(tl.uint64)) + out_chunk * 2
        )
        x, y, z, w = local_ld_128(in_ptr, mask)
        multimem_st_128(out_ptr, x, y, z, w, mask)
        block_start += tl.num_programs(axis=0) * BLOCK_SIZE

    sync_threads()
    blockwise_barrier(signal_pad_ptr, None, RANK, WORLD_SIZE, sem="acq_rel")


def nvidia_rsag_multimem_all_gather_inner(
    state: TritonCommState,
    hidden_states: torch.Tensor,
    total_tokens: int,
    local_hidden: int,
    hidden_offset: int,
    skip_entry_sync: bool,
) -> None:
    num_elts = total_tokens * local_hidden
    num_blocks, block_size, num_warps, numel_per_thread = nvidia_rsag_get_launch_config(
        num_elts
    )
    symm_mem_hdl = symm_mem.rendezvous(state.comm_buff, group=state.group)
    assert state.rank_in_group == symm_mem_hdl.rank, "Mismatched rank id"
    grid = (num_blocks, 1, 1)
    nvidia_rsag_all_gather_kernel_inner[grid](
        input_ptr=hidden_states,
        multicast_ptr=symm_mem_hdl.multicast_ptr,
        signal_pad_ptr=symm_mem_hdl.signal_pad_ptrs_dev,
        total_tokens=total_tokens,
        hidden_offset=hidden_offset,
        LOCAL_HIDDEN=local_hidden,
        TOTAL_HIDDEN=state.hidden_dim,
        BLOCK_SIZE=block_size,
        NUMEL_PER_THREAD=numel_per_thread,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        SKIP_ENTRY_SYNC=1 if skip_entry_sync else 0,
        num_warps=num_warps,
    )


def nvidia_rsag_all_gather_inner(
    state: TritonCommState,
    hidden_states: torch.Tensor,
    tp_hidden_dim: int = None,
    hidden_list_in_group: List[int] = None,
    skip_entry_sync: bool = False,
    safe: bool = True,
) -> torch.Tensor:
    assert (
        tp_hidden_dim is not None or hidden_list_in_group is not None
    ), "Either tp_hidden_dim or hidden_list_in_group must be provided"
    if hidden_list_in_group is None:
        # Strict even split: refuse to distribute remainder because 128-bit
        # multimem.st needs each per-rank slice to be a multiple of 8 bf16, and
        # remainder distribution would yield non-aligned widths.
        assert tp_hidden_dim % state.world_size == 0, (
            f"For automatic even hidden split, tp_hidden_dim ({tp_hidden_dim}) "
            f"must be divisible by world_size ({state.world_size}); otherwise "
            f"pass hidden_list_in_group explicitly."
        )
        hidden_list_in_group = [tp_hidden_dim // state.world_size] * state.world_size
    for r, h in enumerate(hidden_list_in_group):
        assert h > 0, (
            f"hidden_list_in_group[{r}]={h} must be > 0; a zero-width shard "
            f"would make the kernel's chunks_per_row constexpr collapse and "
            f"trigger a div-by-zero at JIT time while peers hang in the barrier"
        )
        assert h % INNER_AG_NUMEL_PER_THREAD == 0, (
            f"hidden_list_in_group[{r}]={h} must be a multiple of "
            f"{INNER_AG_NUMEL_PER_THREAD} bf16 (16-byte multimem.st alignment); "
            f"pad in the producer if needed"
        )
    total_hidden = sum(hidden_list_in_group)
    assert total_hidden <= state.hidden_dim, (
        f"The inner comm buffer is too narrow: {total_hidden=} is not <= "
        f"{state.hidden_dim=}"
    )
    local_hidden = hidden_list_in_group[state.rank_in_group]
    hidden_offset = sum(hidden_list_in_group[: state.rank_in_group])

    assert hidden_states.dtype == torch.bfloat16, "Only bfloat16 is supported"
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    # is_contiguous() does not imply 16-byte data_ptr alignment — e.g. a
    # contiguous slice of a larger tensor (outer[i] on a 3D tensor) can land
    # at a 2-byte offset. local_ld_128 in the kernel issues unaligned loads
    # in that case, so reject early.
    assert hidden_states.data_ptr() % 16 == 0, (
        f"hidden_states.data_ptr()={hex(hidden_states.data_ptr())} must be "
        f"16-byte aligned for 128-bit multimem.st loads; copy/contiguous "
        f"the input through a fresh allocation if needed"
    )
    assert state.hidden_dim % INNER_AG_NUMEL_PER_THREAD == 0, (
        f"state.hidden_dim={state.hidden_dim} must be a multiple of "
        f"{INNER_AG_NUMEL_PER_THREAD} bf16 (16-byte multimem.st row stride alignment)"
    )
    total_tokens, in_hidden = hidden_states.shape
    assert in_hidden == local_hidden, (
        f"input hidden ({in_hidden}) does not match this rank's "
        f"hidden_list_in_group[{state.rank_in_group}]={local_hidden}"
    )
    assert (
        total_tokens <= state.max_token_num
    ), f"{total_tokens=} exceeds {state.max_token_num=}"

    hidden_size_bak, comm_buff_bak = rsag_resize_hidden_if_needed(state, total_hidden)
    try:
        nvidia_rsag_multimem_all_gather_inner(
            state,
            hidden_states,
            total_tokens,
            local_hidden,
            hidden_offset,
            skip_entry_sync,
        )
        output = state.comm_buff[:total_tokens, :]
        return output.clone() if safe else output
    finally:
        rsag_restore_hidden(state, hidden_size_bak, comm_buff_bak)


def all_gather_inner(
    state: TritonCommState,
    hidden_states: torch.Tensor,
    tp_hidden_dim: int = None,
    hidden_list_in_group: List[int] = None,
    skip_entry_sync: bool = False,
    safe: bool = True,
) -> torch.Tensor:
    """Inner all-gather — NVIDIA-only, concatenates along the hidden dim.

    ``skip_entry_sync=True`` removes the entry CAS barrier via a compile-time
    constexpr. Safe only when the caller has externally guaranteed that *all
    ranks* have finished reading ``state.comm_buff`` before this call enters;
    otherwise a faster rank may multicast new data into a slower peer's
    comm-buf while that peer is still consuming the previous result (clone,
    matmul, etc.). An adjacent tokenspeed collective's acq_rel exit barrier
    is NOT sufficient on its own — it only synchronizes the end of the kernel,
    not the end of consumers queued after it. Typical safe patterns: an
    explicit ``dist.barrier`` since the last buffer read, or back-to-back
    skip-entry calls where the consumer is the next kernel's multimem store.
    """
    platform = current_platform()
    assert platform.is_nvidia, f"all_gather_inner only supports NVIDIA, got {platform}"
    return nvidia_rsag_all_gather_inner(
        state,
        hidden_states,
        tp_hidden_dim=tp_hidden_dim,
        hidden_list_in_group=hidden_list_in_group,
        skip_entry_sync=skip_entry_sync,
        safe=safe,
    )
