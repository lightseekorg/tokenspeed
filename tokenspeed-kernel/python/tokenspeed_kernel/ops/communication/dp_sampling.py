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

"""One-sided NVLink collectives for Batch-DP speculative verify."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from tokenspeed_kernel._triton import tl, triton

from .triton import symm_mem_barrier


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
    assert type(group) == dist.ProcessGroup, f"Expected ProcessGroup, got {type(group)}"
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
    symm_mem.set_signal_pad_size(
        max(symm_mem.get_signal_pad_size(), signal_pad_bytes)
    )

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
    assert tuple(local_logits.shape) == expected_shape, (
        f"local_logits shape {tuple(local_logits.shape)} != {expected_shape}"
    )
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
