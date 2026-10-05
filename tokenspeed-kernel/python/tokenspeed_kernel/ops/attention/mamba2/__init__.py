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

"""Mamba2 selective state space (SSD) scans.

The recurrence per head ``h`` with ``A = -exp(A_log)`` and
``dt' = softplus(dt + dt_bias)`` clamped to ``dt_limit`` is
``s_t = exp(dt' * A) * s_{t-1} + dt' * x_t (x) B_t`` and ``y_t = C_t . s_t + D * x_t``. States are ``[heads, head_dim, d_state]`` with
``d_state`` last, the layout of the runtime's recurrent-state pool. ``B`` and ``C``
are shared by ``heads / groups`` consecutive heads.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.attention.gdn import validate_replay_commit_args
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

# Kernels widen only an operand's leading index; offsets within one row stay int32.
_MAX_ROW_SPAN = 2**31 - 1


@dataclass(frozen=True)
class Mamba2ChunkMetadata:
    """Logical chunks of a packed varlen batch for the chunked scan.

    A logical chunk never crosses a sequence boundary or a physical
    ``chunk_size`` boundary of the packed token axis.

    Attributes:
        chunk_size: Physical chunk length the metadata was built for.
        cu_chunk_seqlens: ``[num_chunks + 1]`` int32 token offsets of the chunks.
        last_chunk_indices: ``[batch]`` int32 index of each sequence's last chunk.
        seq_idx: ``[num_chunks]`` int32 sequence of each chunk.
    """

    chunk_size: int
    cu_chunk_seqlens: torch.Tensor
    last_chunk_indices: torch.Tensor
    seq_idx: torch.Tensor

    def __post_init__(self) -> None:
        vectors = (self.cu_chunk_seqlens, self.last_chunk_indices, self.seq_idx)
        if self.cu_chunk_seqlens.shape != (self.seq_idx.shape[0] + 1,) or any(
            t.dim() != 1 or t.dtype != torch.int32 or t.stride(0) != 1 for t in vectors
        ):
            raise ValueError(
                "chunk metadata must be dense int32 vectors with num_chunks + 1 offsets"
            )


def build_mamba2_chunk_metadata(
    cu_seqlens_cpu: torch.Tensor, chunk_size: int, device: torch.device
) -> Mamba2ChunkMetadata:
    """Split a packed varlen batch into logical chunks on the host.

    Args:
        cu_seqlens_cpu: Host ``[batch + 1]`` strictly increasing cumulative
            token offsets starting at 0; the scan has no empty sequences.
        chunk_size: Physical chunk length of the scan.
        device: Device the returned tensors live on.

    Returns:
        The chunk metadata, copied to ``device`` without blocking.
    """
    if cu_seqlens_cpu.device.type != "cpu" or cu_seqlens_cpu.dim() != 1:
        raise ValueError("cu_seqlens_cpu must be a 1-D host tensor")
    bounds = cu_seqlens_cpu.to(torch.int64).tolist()
    if bounds[0] != 0:
        raise ValueError(f"cu_seqlens_cpu must start at 0, got {bounds[0]}")
    chunk_lens: list[int] = []
    seq_idx: list[int] = []
    last_chunk_indices: list[int] = []
    for seq, (start, end) in enumerate(itertools.pairwise(bounds)):
        if end <= start:
            raise ValueError(f"sequence {seq} is empty; every sequence needs a token")
        pos = start
        while pos < end:
            take = min(chunk_size - pos % chunk_size, end - pos)
            chunk_lens.append(take)
            seq_idx.append(seq)
            pos += take
        last_chunk_indices.append(len(chunk_lens) - 1)
    cu_chunk_seqlens = [0, *itertools.accumulate(chunk_lens)]

    def upload(values: list[int]) -> torch.Tensor:
        host = torch.tensor(values, dtype=torch.int32, pin_memory=True)
        return host.to(device, non_blocking=True)

    return Mamba2ChunkMetadata(
        chunk_size=chunk_size,
        cu_chunk_seqlens=upload(cu_chunk_seqlens),
        last_chunk_indices=upload(last_chunk_indices),
        seq_idx=upload(seq_idx),
    )


def _signature(x: torch.Tensor):
    return format_signature(x=dense_tensor_format(x.dtype))


def _check_ssd_operands(
    x: torch.Tensor,
    dt: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Validate the token-major operands every Mamba2 kernel indexes directly."""
    *lead, heads, _ = x.shape
    if dt.shape != (*lead, heads) or out.shape != x.shape:
        raise ValueError("dt must be [..., heads] and out shaped like x")
    if B.dim() != x.dim() or B.shape[:-2] != tuple(lead) or C.shape != B.shape:
        raise ValueError("B and C must be [..., groups, d_state] alongside x")
    if heads % B.shape[-2]:
        raise ValueError("heads must be a multiple of groups")
    if any(t.stride(-1) != 1 for t in (x, dt, B, C)) or not out.is_contiguous():
        raise ValueError(
            "x, dt, B and C must be contiguous in their last dim and out contiguous"
        )
    if any(t.shape != (heads,) or t.stride(0) != 1 for t in (A_log, D, dt_bias)):
        raise ValueError("A_log, D and dt_bias must be dense [heads] vectors")


def _span(t: torch.Tensor, first_dim: int) -> int:
    """Elements from ``t``'s first to its last address over dims from ``first_dim``."""
    return sum((n - 1) * s for n, s in zip(t.shape[first_dim:], t.stride()[first_dim:]))


def _check_memory(
    written: tuple[torch.Tensor, ...], read: tuple[torch.Tensor, ...]
) -> None:
    """Reject rows int32 offsets cannot index and outputs sharing another operand's storage."""
    storages = []
    for t in (*written, *read):
        storage = t.untyped_storage()
        # A storage of at most _MAX_ROW_SPAN elements bounds every row it holds.
        if storage.nbytes() > _MAX_ROW_SPAN * t.element_size():
            span = _span(t, 1)
            if span > _MAX_ROW_SPAN:
                raise ValueError(f"an operand row spans {span} elements, over int32")
        storages.append(storage.data_ptr() if t.numel() else None)
    for i, storage in enumerate(storages[: len(written)]):
        if storage is not None and storage in storages[:i] + storages[i + 1 :]:
            raise ValueError("outputs must not share storage with another operand")


def _check_state_pool(
    state: torch.Tensor, x: torch.Tensor, B: torch.Tensor, state_indices: torch.Tensor
) -> None:
    """Validate a paged state pool and the per-request slots read from it."""
    heads, head_dim = x.shape[-2:]
    d_state = B.shape[-1]
    if state.shape[1:] != (heads, head_dim, d_state) or state.stride()[1:] != (
        head_dim * d_state,
        d_state,
        1,
    ):
        raise ValueError(
            "state must be [slots, heads, head_dim, d_state], dense within a slot"
        )
    if state.shape[0] > 1 and state.stride(0) < heads * head_dim * d_state:
        raise ValueError("state slots must not overlap")
    if state_indices.shape != (x.shape[0],) or state_indices.stride(0) != 1:
        raise ValueError("state_indices must be a dense [batch] vector")


def mamba2_chunk_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    dt_limit: tuple[float, float],
    initial_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_metadata: Mamba2ChunkMetadata,
    out: torch.Tensor,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Scan a packed varlen prefill batch from per-sequence initial states.

    Args:
        x: ``[tokens, heads, head_dim]`` conv'd input.
        dt: ``[tokens, heads]`` raw time step (before bias and softplus).
        A_log: ``[heads]`` fp32 log of the negated decay rate.
        B: ``[tokens, groups, d_state]`` input projection.
        C: ``[tokens, groups, d_state]`` output projection.
        D: ``[heads]`` skip coefficient.
        dt_bias: ``[heads]`` time-step bias.
        dt_limit: ``(low, high)`` clamp applied after softplus.
        initial_states: ``[batch, heads, head_dim, d_state]`` state each sequence
            starts from; zero rows for sequences without history. Its dtype is
            the dtype of the returned final states.
        cu_seqlens: ``[batch + 1]`` int32 device token offsets.
        chunk_metadata: Logical chunks built from the same offsets.
        out: ``[tokens, heads, head_dim]`` output, written in place.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        ``[batch, heads, head_dim, d_state]`` final states.
    """
    _check_ssd_operands(x, dt, A_log, B, C, D, dt_bias, out)
    _, heads, head_dim = x.shape
    d_state = B.shape[-1]
    if initial_states.shape != (
        cu_seqlens.shape[0] - 1,
        heads,
        head_dim,
        d_state,
    ) or initial_states.stride()[1:] != (head_dim * d_state, d_state, 1):
        raise ValueError(
            "initial_states must be [batch, heads, head_dim, d_state] with dense states"
        )
    _check_memory(
        (out,),
        (
            x,
            dt,
            A_log,
            B,
            C,
            D,
            dt_bias,
            initial_states,
            cu_seqlens,
            chunk_metadata.cu_chunk_seqlens,
            chunk_metadata.last_chunk_indices,
            chunk_metadata.seq_idx,
        ),
    )
    if chunk_metadata.last_chunk_indices.shape[0] != cu_seqlens.shape[0] - 1:
        raise ValueError("chunk_metadata was built for a different batch")
    chunk = chunk_metadata.chunk_size
    if chunk < 16 or chunk & (chunk - 1):
        raise ValueError(
            f"chunk_size must be a power of two of at least 16, got {chunk}"
        )
    kernel = select_kernel(
        "attention",
        "mamba2_chunk_scan",
        _signature(x),
        traits={},
        solution=solution,
        override=override,
    )
    shape_params = {
        "batch_size": cu_seqlens.shape[0] - 1,
        "total_tokens": x.shape[0],
        "num_heads": x.shape[1],
        "head_dim": x.shape[2],
        "d_state": B.shape[2],
    }
    ShapeCapture.get().record(
        "attention", "mamba2_chunk_scan", kernel.name, x.dtype, shape_params
    )
    with kernel_scope(
        "attention",
        "mamba2_chunk_scan",
        x.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            x,
            dt,
            A_log,
            B,
            C,
            D,
            dt_bias,
            dt_limit=dt_limit,
            initial_states=initial_states,
            cu_seqlens=cu_seqlens,
            chunk_metadata=chunk_metadata,
            out=out,
        )


def mamba2_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_indices: torch.Tensor,
    dst_state_indices: torch.Tensor,
    null_slot: int,
    out: torch.Tensor,
    override: str | None = None,
    solution: str | None = None,
) -> None:
    """Advance paged recurrent states by one token per request.

    Args:
        state: ``[slots, heads, head_dim, d_state]`` state pool.
        x: ``[batch, heads, head_dim]`` conv'd input.
        dt: ``[batch, heads]`` raw time step.
        A_log: ``[heads]`` fp32 log of the negated decay rate.
        B: ``[batch, groups, d_state]``.
        C: ``[batch, groups, d_state]``.
        D: ``[heads]`` skip coefficient.
        dt_bias: ``[heads]`` time-step bias.
        state_indices: ``[batch]`` slot each request reads its state from.
        dst_state_indices: ``[batch]`` slot each request writes its new state to.
        null_slot: Index marking padded rows: their state is neither read nor
            written.
        out: ``[batch, heads, head_dim]`` output, written in place.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.
    """
    _check_ssd_operands(x, dt, A_log, B, C, D, dt_bias, out)
    _check_state_pool(state, x, B, state_indices)
    if (
        dst_state_indices.shape != state_indices.shape
        or dst_state_indices.stride(0) != 1
    ):
        raise ValueError("dst_state_indices must be a dense [batch] vector")
    _check_memory(
        (out, state),
        (x, dt, A_log, B, C, D, dt_bias, state_indices, dst_state_indices),
    )
    kernel = select_kernel(
        "attention",
        "mamba2_state_update",
        _signature(x),
        traits={},
        solution=solution,
        override=override,
    )
    shape_params = {
        "batch_size": x.shape[0],
        "num_heads": x.shape[1],
        "head_dim": x.shape[2],
        "d_state": B.shape[2],
    }
    ShapeCapture.get().record(
        "attention", "mamba2_state_update", kernel.name, x.dtype, shape_params
    )
    with kernel_scope(
        "attention",
        "mamba2_state_update",
        x.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        kernel(
            state,
            x,
            dt,
            A_log,
            B,
            C,
            D,
            dt_bias,
            state_indices=state_indices,
            dst_state_indices=dst_state_indices,
            null_slot=null_slot,
            out=out,
        )


def _check_tree_parents(x: torch.Tensor, parent_indices: torch.Tensor | None) -> None:
    """Validate a draft tree's ``[batch, T]`` parent table."""
    if parent_indices is None:
        return
    batch, steps = x.shape[:2]
    if steps > 64:
        raise ValueError(
            f"a draft tree holds at most 64 tokens (a 64-bit ancestor mask), got {steps}"
        )
    if (
        parent_indices.shape != (batch, steps)
        or parent_indices.dtype != torch.int32
        or not parent_indices.is_contiguous()
    ):
        raise ValueError(
            f"parent_indices must be contiguous int32 {(batch, steps)}, got "
            f"{parent_indices.dtype} {tuple(parent_indices.shape)}"
        )


def mamba2_verify_scan(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_indices: torch.Tensor,
    dst_state_indices: torch.Tensor | None,
    parent_indices: torch.Tensor | None,
    null_slot: int,
    out: torch.Tensor,
    override: str | None = None,
    solution: str | None = None,
) -> None:
    """Advance paged states through each request's ``T`` verify tokens.

    Every request starts from its own slot and steps through its tokens in
    order, exactly as ``T`` consecutive ``mamba2_state_update`` calls would.
    With ``parent_indices`` the tokens form a draft tree: token ``t``
    continues from the state after its parent token instead of token ``t - 1``.

    Args:
        state: ``[slots, heads, head_dim, d_state]`` state pool, ``d_state``
            contiguous.
        x: ``[batch, T, heads, head_dim]`` conv'd input, ``head_dim``
            contiguous.
        dt: ``[batch, T, heads]`` raw time step.
        A_log: ``[heads]`` fp32 log of the negated decay rate.
        B: ``[batch, T, groups, d_state]``, ``d_state`` contiguous.
        C: ``[batch, T, groups, d_state]``, ``d_state`` contiguous.
        D: ``[heads]`` skip coefficient.
        dt_bias: ``[heads]`` time-step bias.
        state_indices: ``[batch]`` slot each request reads its state from.
        dst_state_indices: ``[batch, T]`` int32 slot receiving the state after
            each token, or None to leave the pool untouched (the caller
            replays the accepted tokens later).
        parent_indices: Contiguous int32 ``[batch, T]`` draft-tree parents:
            token ``t`` continues from the state after token
            ``parent_indices[i, t]``, or from the read slot when negative;
            parents precede their children, ``T`` is at most 64, and
            destinations must not alias the read slot or each other. None
            for a chain.
        null_slot: Index marking padding: a request whose read slot is
            ``null_slot`` starts from zeros, and a ``null_slot`` destination
            is skipped.
        out: ``[batch, T, heads, head_dim]`` output, written in place.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.
    """
    batch, steps, heads, head_dim = x.shape
    d_state = state.shape[-1]
    _check_ssd_operands(x, dt, A_log, B, C, D, dt_bias, out)
    _check_state_pool(state, x, B, state_indices)
    if dst_state_indices is not None and (
        dst_state_indices.shape != (batch, steps)
        or dst_state_indices.dtype != torch.int32
        or dst_state_indices.stride(-1) != 1
    ):
        raise ValueError("dst_state_indices must be a dense int32 [batch, T] table")
    _check_tree_parents(x, parent_indices)
    destinations = () if dst_state_indices is None else (dst_state_indices,)
    parents = () if parent_indices is None else (parent_indices,)
    _check_memory(
        (out, state),
        (x, dt, A_log, B, C, D, dt_bias, state_indices, *destinations, *parents),
    )
    if batch == 0 or steps == 0:
        return
    kernel = select_kernel(
        "attention",
        "mamba2_verify_scan",
        _signature(x),
        traits={},
        solution=solution,
        override=override,
    )
    shape_params = {
        "batch_size": batch,
        "seq_len": steps,
        "num_heads": heads,
        "head_dim": head_dim,
        "d_state": d_state,
    }
    ShapeCapture.get().record(
        "attention", "mamba2_verify_scan", kernel.name, x.dtype, shape_params
    )
    with kernel_scope(
        "attention",
        "mamba2_verify_scan",
        x.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        kernel(
            state,
            x,
            dt,
            A_log,
            B,
            C,
            D,
            dt_bias,
            state_indices=state_indices,
            dst_state_indices=dst_state_indices,
            parent_indices=parent_indices,
            null_slot=null_slot,
            out=out,
        )


def mamba2_replay_commit(
    payload: torch.Tensor,
    parameters: torch.Tensor,
    *,
    state_addresses: torch.Tensor,
    state_row_strides: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    draft_token_num: int,
    geometry: tuple[int, int, int, int],
    state_dtype: torch.dtype,
    override: str | None = None,
    solution: str | None = None,
) -> None:
    """Replay every Mamba2 layer's accepted verify tokens in one launch.

    The payload uses the recurrent replay layout ``[K | V | a | b]`` that the
    verify split writes: ``B`` as the key, ``x`` as the value and the raw
    ``dt`` as ``a``; the SSD recurrence has no ``b`` and ignores that slot.
    Each program starts from its layer's committed state, advances it by the
    request's accepted tokens and writes only the final state.

    Args:
        payload: Contiguous ``[L, rows, G*N + H*P + 2*H]`` storage whose first
            ``batch * T`` rows per layer are the request-major verify window.
        parameters: Contiguous fp32 ``[L, 2, H]`` table of ``A_log`` and
            ``dt_bias``.
        state_addresses: uint64 ``[L]`` base addresses of the state pools.
        state_row_strides: int64 ``[L]`` pool row strides in elements; each
            row holds a dense ``[H, P, N]`` state.
        read_indices: int32 ``[L, batch]`` committed-state pages; a negative
            page starts from zeros.
        write_indices: int32 ``[L, batch]`` destination pages; a negative page
            is skipped.
        accepted_length: int32 ``[batch]`` tokens to replay, clamped to ``T``.
        draft_token_num: Verify window length ``T``.
        geometry: ``(groups, heads, d_state, head_dim)``.
        state_dtype: Element dtype shared by every state pool.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.
    """
    if not validate_replay_commit_args(
        payload,
        parameters,
        state_addresses=state_addresses,
        state_row_strides=state_row_strides,
        read_indices=read_indices,
        write_indices=write_indices,
        accepted_length=accepted_length,
        draft_token_num=draft_token_num,
        geometry=geometry,
        state_dtype=state_dtype,
    ):
        return
    _, heads, d_state, head_dim = geometry
    num_layers = payload.shape[0]
    batch = accepted_length.numel()
    kernel = select_kernel(
        "attention",
        "mamba2_replay_commit",
        format_signature(payload=dense_tensor_format(payload.dtype)),
        traits={},
        solution=solution,
        override=override,
    )
    with kernel_scope(
        "attention",
        "mamba2_replay_commit",
        payload.dtype,
        kernel_name=kernel.name,
        batch_size=batch,
        seq_len=draft_token_num,
        num_layers=num_layers,
        num_heads=heads,
        head_dim=head_dim,
        d_state=d_state,
    ):
        kernel(
            payload,
            parameters,
            state_addresses=state_addresses,
            state_row_strides=state_row_strides,
            read_indices=read_indices,
            write_indices=write_indices,
            accepted_length=accepted_length,
            draft_token_num=draft_token_num,
            geometry=geometry,
            state_dtype=state_dtype,
        )


def mamba2_replay_commit_supported(dtype: torch.dtype) -> bool:
    """Whether a Mamba2 replay kernel is registered for ``dtype`` payloads here.

    Args:
        dtype: Activation dtype of the verify pass, which the payload stores.

    Returns:
        ``True`` when ``mamba2_replay_commit`` can run on this platform.
    """
    try:
        select_kernel(
            "attention",
            "mamba2_replay_commit",
            format_signature(payload=dense_tensor_format(dtype)),
            traits={},
        )
    except NoKernelFoundError:
        return False
    return True


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.mamba2.triton  # noqa: E402,F401

# isort: on

__all__ = [
    "Mamba2ChunkMetadata",
    "build_mamba2_chunk_metadata",
    "mamba2_chunk_scan",
    "mamba2_replay_commit",
    "mamba2_replay_commit_supported",
    "mamba2_state_update",
    "mamba2_verify_scan",
]
