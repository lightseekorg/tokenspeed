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

"""One-sided NVLink kernels for the Batch-DP spec-sampling pipeline.

This module is the kernel-layer companion to
``tokenspeed.runtime.distributed.dp_sampling_comm.DpSamplingComm``. It
exposes two symmetric-memory one-sided collectives plus the workspace
they share:

  * ``dp_sampling_swap`` (stage 4):
        ``[pad_bs * N, V/TP]``  ->  ``[K_req * N, V]``
    Each rank writes the K_req requests destined for every peer into the
    peer's symmetric-memory recv buffer at the right ``[src_rank, ...,
    vocab_slice]`` offset. Because every peer's stripe is contiguous along
    a different vocab range and the K_req x N tile is contiguous along
    the request axis, the recv buffer is *natively* shaped
    ``[K_req, N, V]`` with no ``.permute(1, 2, 0, 3).contiguous()`` reshape.

  * ``dp_sampling_gather`` (stage 6):
        per-rank ``predict[K_req, N]`` + ``accept_index[K_req, N]`` +
        ``accept_length[K_req]``  ->  three ``[pad_bs, *]`` tensors on
        every rank.
    All three payloads are pushed in one kernel launch + one
    release/acquire barrier so the trailing 3 NCCL ``all_gather`` calls
    in ``bench/dp_sampling_flow.html`` stage 6 collapse to a single
    fused op. The full ``[pad_bs, ...]`` recv buffers are pre-allocated
    in ``create_dp_sampling_state``; per-call returns are zero-copy
    views into them.

CUDA-graph lifecycle (mirrors ``NvlinkAllReduceFusion`` and the TRT-LLM
NVLinkOneSided MoeAlltoAll):

    state = create_dp_sampling_state(...)   # host-side rendezvous; pre-capture
    # ... CUDA graph capture starts here ...
    swapped = dp_sampling_swap(state, local_logits, pad_bs=...)
    # ... sampling kernel ...
    p, ai, al = dp_sampling_gather(state, predict_local, ..., pad_bs=...)
    # ... CUDA graph capture ends ...

The per-step entrypoints are pure CUDA / Triton kernel launches over
``state``'s persistent symm-mem buffers. ``state`` itself is allocated
once and never resized; ``pad_bs`` is a runtime arg bounded by
``max_pad_bs`` at construction time.

============================================================================
STATUS: stubbed pending implementation. See ``DpSamplingComm`` runtime-side
wiring; ``DpSamplingComm._onesided_available()`` returns False so callers
default to the NCCL fallback until this lands. The stubs below define the
exact surface ``DpSamplingComm._init_onesided`` / ``_swap_..._onesided`` /
``_gather_..._onesided`` will bind to.
============================================================================

Implementation references when wiring this up:

  * Symmetric-memory rendezvous: see ``flashinfer.comm.torch_symmetric_memory.
    _alloc_symm_buffer_bytes`` for a 30-line wrapper around
    ``symm_mem.empty`` + ``symm_mem.rendezvous`` that returns the per-peer
    pointer table -- exactly the shape we need.

  * Release/acquire barrier (CTA-scoped or block-wise): see
    ``send_signal_to_peers`` / ``wait_signal_from_peers`` /
    ``symm_mem_barrier`` in ``ops/communication/triton.py``.

  * NVLinkOneSided design (rank-major recv buffer, dispatch kernel
    structure, top_k dedup -- the dedup does NOT apply here since
    uniform a2a has no routing): ``bench/dp_sampling_flow.html``
    summary plus the TRT-LLM blog at
    docs/source/blogs/tech_blog/blog18_Optimizing_MoE_Communication_with_One_Sided_AlltoAll_Over_NVLink.md.

  * Graph capture footguns (specifically the PyTorch
    ``mode='reduce-overhead'`` issue that mangles P2P pool mapping):
    pytorch/pytorch#175450 and #178138. Tokenspeed uses raw
    ``torch.cuda.graph()``, which is unaffected, but the comment in
    ``DpSamplingComm.__init__`` documents the constraint for posterity.
"""

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
    """Persistent workspace. Allocated once at engine init; reused across
    replays. All tensor fields below are symmetric-memory
    backed (peer-importable VMM pages).

    Sizing: the prefix [:pad_bs * tp_size * ...] of each buffer.

    Field overview:
        recv_logits:     [max_K_req, N, V]
        recv_predict:    [max_pad_bs, N] int32
        recv_accept_idx: [max_pad_bs, N] int32
        recv_accept_len: [max_pad_bs] int32
        flags_peer_ptrs: Device-side signal-pad pointer table supplied by
                         symmetric memory. The Triton barrier flips each
                         per-CTA flag from 0 -> 1 -> 0, so no host-side
                         epoch is needed for graph replay.
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

    # Symmetric-memory handles. Keep these alive for the lifetime of
    # the state; kernels use their device-side peer-pointer tables and
    # signal-pad pointer table on every call.
    recv_logits_hdl: Any | None = None
    recv_predict_hdl: Any | None = None
    recv_accept_idx_hdl: Any | None = None
    recv_accept_len_hdl: Any | None = None

    # Per-peer pointer tables -- one entry per rank in ``group``. These
    # are raw device pointers (int64); Triton kernels dereference them
    # via ``tl.tensor(..., pointer_type)``. See
    # ``flashinfer.comm.torch_symmetric_memory._alloc_symm_buffer_bytes``
    # for the canonical way to extract these from a symm-mem handle.
    recv_logits_peer_ptrs: torch.Tensor | None = None
    recv_predict_peer_ptrs: torch.Tensor | None = None
    recv_accept_idx_peer_ptrs: torch.Tensor | None = None
    recv_accept_len_peer_ptrs: torch.Tensor | None = None
    flags_peer_ptrs: torch.Tensor | None = None


# ----------------------------------------------------------------------------
# Triton kernels
# ----------------------------------------------------------------------------


@triton.jit
def _dp_sampling_swap_kernel(
    local_logits,
    recv_logits_ptrs_dev,
    signal_pad_ptrs_dev,
    K_REQ: tl.constexpr,
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
    req_local = tmp % K_REQ
    dst_rank = tmp // K_REQ

    offsets = vocab_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < V_LOCAL

    src_row = dst_rank * K_REQ * N + req_local * N + draft_pos
    vals = tl.load(local_logits + src_row * V_LOCAL + offsets, mask=mask)

    peer_ptrs = recv_logits_ptrs_dev.to(tl.pointer_type(tl.uint64))
    if LOGITS_DTYPE_CODE == 0:
        peer_base = tl.load(peer_ptrs + dst_rank).to(tl.pointer_type(tl.bfloat16))
    elif LOGITS_DTYPE_CODE == 1:
        peer_base = tl.load(peer_ptrs + dst_rank).to(tl.pointer_type(tl.float16))
    else:
        peer_base = tl.load(peer_ptrs + dst_rank).to(tl.pointer_type(tl.float32))

    dst_offset = req_local * N * V + draft_pos * V + RANK * V_LOCAL + offsets
    tl.store(peer_base + dst_offset, vals, mask=mask)

    symm_mem_barrier(signal_pad_ptrs_dev, pid, RANK, WORLD_SIZE)


@triton.jit
def _dp_sampling_gather_kernel(
    predict_local,
    accept_index_local,
    accept_length_local,
    recv_predict_ptrs_dev,
    recv_accept_idx_ptrs_dev,
    recv_accept_len_ptrs_dev,
    signal_pad_ptrs_dev,
    K_REQ: tl.constexpr,
    N: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    req_local = pid % K_REQ
    dst_rank = pid // K_REQ

    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N

    src_base = req_local * N
    pred_vals = tl.load(predict_local + src_base + offsets, mask=mask)
    accept_idx_vals = tl.load(accept_index_local + src_base + offsets, mask=mask)
    accept_len_val = tl.load(accept_length_local + req_local)

    pred_ptrs = recv_predict_ptrs_dev.to(tl.pointer_type(tl.uint64))
    accept_idx_ptrs = recv_accept_idx_ptrs_dev.to(tl.pointer_type(tl.uint64))
    accept_len_ptrs = recv_accept_len_ptrs_dev.to(tl.pointer_type(tl.uint64))

    pred_peer = tl.load(pred_ptrs + dst_rank).to(tl.pointer_type(tl.int32))
    accept_idx_peer = tl.load(accept_idx_ptrs + dst_rank).to(tl.pointer_type(tl.int32))
    accept_len_peer = tl.load(accept_len_ptrs + dst_rank).to(tl.pointer_type(tl.int32))

    dst_row = RANK * K_REQ + req_local
    dst_base = dst_row * N
    tl.store(pred_peer + dst_base + offsets, pred_vals, mask=mask)
    tl.store(accept_idx_peer + dst_base + offsets, accept_idx_vals, mask=mask)
    tl.store(accept_len_peer + dst_row, accept_len_val)

    symm_mem_barrier(signal_pad_ptrs_dev, pid, RANK, WORLD_SIZE)


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


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
    """Allocate symmetric-memory workspaces and rendezvous with peers.

    Outline:
      1. ``symm_mem.empty(...)`` for each of recv_logits, recv_predict,
         recv_accept_idx, recv_accept_len, flags, epoch.
      2. ``symm_mem.rendezvous(buf, group=group.group_name)`` to get the
         peer handle for each buffer.
      3. ``handle.get_buffer(peer, shape, dtype)`` for every peer to
         build the peer-pointer tables.
      4. Keep handles alive so their signal-pad pointer table remains valid.
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

    max_k_req = max_pad_bs // tp_size
    v_local = vocab_size // tp_size
    swap_block_size = min(1024, _next_power_of_2(v_local))
    gather_block_n = min(1024, _next_power_of_2(num_tokens_per_req))
    swap_max_blocks = (
        tp_size
        * max_k_req
        * num_tokens_per_req
        * triton.cdiv(v_local, swap_block_size)
    )
    gather_max_blocks = tp_size * max_k_req
    signal_pad_bytes = max(swap_max_blocks, gather_max_blocks) * tp_size * 4
    symm_mem.set_signal_pad_size(
        max(symm_mem.get_signal_pad_size(), signal_pad_bytes)
    )

    recv_logits, recv_logits_hdl = _alloc_symm(
        (max_k_req, num_tokens_per_req, vocab_size), logits_dtype, device, group
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
    """Stage 4 one-sided swap. Returns a view into ``state.recv_logits``.

    Kernel structure (one Triton kernel launched on the local stream):
      1. Compute ``(target_rank, target_index)`` for each input row from
         ``rank_in_group``, ``tp_size``, ``num_tokens_per_req``, and
         ``pad_bs`` -- purely arithmetic, no routing dependency.
      2. Issue vectorized stores into every peer's
         ``recv_logits[my_rank, target_index_within_my_stripe, :]`` at
         the corresponding vocab-slice offset.
      3. ``__threadfence_system()`` (release).
      4. Stamp my flag slot on every peer's ``flags`` array with the
         current epoch.
      5. Spin-wait on local ``flags`` until every peer has stamped.
      6. ``__threadfence_system()`` (acquire).

    Returns the prefix view ``state.recv_logits[:K_req].view(K_req * N, V)``
    where ``K_req = pad_bs // tp_size``.
    """
    tp_size = state.tp_size
    n = state.num_tokens_per_req
    vocab_size = state.vocab_size
    assert pad_bs <= state.max_pad_bs
    assert pad_bs % tp_size == 0
    assert vocab_size % tp_size == 0
    assert local_logits.is_cuda and local_logits.is_contiguous()
    assert local_logits.dtype == state.logits_dtype

    k_req = pad_bs // tp_size
    v_local = vocab_size // tp_size
    expected_shape = (pad_bs * n, v_local)
    assert tuple(local_logits.shape) == expected_shape, (
        f"local_logits shape {tuple(local_logits.shape)} != {expected_shape}"
    )
    assert state.recv_logits is not None
    assert state.recv_logits_peer_ptrs is not None
    assert state.flags_peer_ptrs is not None

    block_size = min(1024, _next_power_of_2(v_local))
    grid = (tp_size * k_req * n * triton.cdiv(v_local, block_size),)
    _dp_sampling_swap_kernel[grid](
        local_logits,
        state.recv_logits_peer_ptrs,
        state.flags_peer_ptrs,
        K_REQ=k_req,
        N=n,
        V_LOCAL=v_local,
        V=vocab_size,
        RANK=state.rank_in_group,
        WORLD_SIZE=tp_size,
        BLOCK_SIZE=block_size,
        LOGITS_DTYPE_CODE=_logits_dtype_code(state.logits_dtype),
        num_warps=4,
    )
    return state.recv_logits[:k_req].view(k_req * n, vocab_size)


def dp_sampling_gather(
    state: DpSamplingState,
    predict_local: torch.Tensor,
    accept_index_local: torch.Tensor,
    accept_length_local: torch.Tensor,
    *,
    pad_bs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-sided gather:
      - per-rank predict[K_req, N]
      - accept_index[K_req, N]
      - accept_length[K_req]

    Replace three all_gather_into_tensor calls with a single kernel.

    Returns (predict_full, accept_index_full, accept_length_full)
    """
    tp_size = state.tp_size
    n = state.num_tokens_per_req
    assert pad_bs <= state.max_pad_bs
    assert pad_bs % tp_size == 0

    k_req = pad_bs // tp_size
    assert tuple(predict_local.shape) == (k_req, n)
    assert tuple(accept_index_local.shape) == (k_req, n)
    assert tuple(accept_length_local.shape) == (k_req,)
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
    grid = (tp_size * k_req,)
    _dp_sampling_gather_kernel[grid](
        predict_local,
        accept_index_local,
        accept_length_local,
        state.recv_predict_peer_ptrs,
        state.recv_accept_idx_peer_ptrs,
        state.recv_accept_len_peer_ptrs,
        state.flags_peer_ptrs,
        K_REQ=k_req,
        N=n,
        RANK=state.rank_in_group,
        WORLD_SIZE=tp_size,
        BLOCK_N=block_n,
        num_warps=1,
    )
    return (
        state.recv_predict[:pad_bs],
        state.recv_accept_idx[:pad_bs],
        state.recv_accept_len[:pad_bs],
    )
