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

"""Apply every request's token stop conditions to a forward's outputs.

A verify round may accept tokens past a stop token or past the request's
token budget. Everything that follows the round -- the recurrent-state
commit, the accepted frontier, the drafter and the tokens reported to the
scheduler -- must see the request end exactly where the control plane ends
it, so the accepted width is cut on the device, in place, before any of them
reads it. A request that ends this round is marked stopped; the forward the
overlap schedule already dispatched behind it then keeps nothing.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)

__all__ = ["apply_stop_conditions"]

_BLOCK = 64


@triton.jit(do_not_specialize=["batch_size", "num_extends", "num_prefill_outputs"])
def _apply_stop_conditions_kernel(
    req_pool_indices_ptr,
    output_tokens_ptr,
    accept_lengths_ptr,
    input_lengths_ptr,
    prompt_lengths_ptr,
    valid_cache_lengths_ptr,
    stop_token_ids_ptr,
    token_limits_ptr,
    stopped_ptr,
    batch_size,
    num_extends,
    num_prefill_outputs,
    padding_index,
    BLOCK: tl.constexpr,
    DECODE_WIDTH: tl.constexpr,
    WIDTH_PAD: tl.constexpr,
    NUM_STOP_IDS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row_mask = rows < batch_size
    slots = tl.load(req_pool_indices_ptr + rows, mask=row_mask, other=padding_index)
    live = row_mask & (slots != padding_index)
    is_extend = rows < num_extends
    valid = tl.load(valid_cache_lengths_ptr + slots, mask=live, other=0)
    limits = tl.load(token_limits_ptr + slots, mask=live, other=0)

    # Every output token of the row: the prefill sample at its request index,
    # a decode row's verify window at its fixed offset.
    columns = tl.arange(0, WIDTH_PAD)
    width = tl.where(is_extend, 1, DECODE_WIDTH)
    first_token = tl.where(
        is_extend,
        rows,
        num_prefill_outputs + (rows - num_extends) * DECODE_WIDTH,
    )
    has_output = live & (~is_extend | (rows < num_prefill_outputs))
    token_mask = has_output[:, None] & (columns[None, :] < width[:, None])
    tokens = tl.load(
        output_tokens_ptr + first_token[:, None] + columns[None, :],
        mask=token_mask,
        other=-1,
    )
    is_stop = tl.zeros((BLOCK, WIDTH_PAD), dtype=tl.int1)
    for k in tl.static_range(NUM_STOP_IDS):
        stop_id = tl.load(
            stop_token_ids_ptr + slots * NUM_STOP_IDS + k, mask=live, other=-1
        )
        is_stop = is_stop | ((tokens == stop_id[:, None]) & (stop_id[:, None] >= 0))

    # Extend rows: only the chunk that completes the prompt owes a token.
    input_lengths = tl.load(input_lengths_ptr + rows, mask=row_mask, other=0)
    prompt_lengths = tl.load(
        prompt_lengths_ptr + rows, mask=row_mask & is_extend, other=-1
    )
    completes_prompt = valid + input_lengths == prompt_lengths
    extend_stops = (
        completes_prompt
        & (rows < num_prefill_outputs)
        & ((tl.sum(is_stop.to(tl.int32), axis=1) > 0) | (prompt_lengths + 1 >= limits))
    )

    # Decode rows: keep through the first accepted stop token and within the
    # budget, nothing once the request has stopped.
    accepted = tl.load(accept_lengths_ptr + rows, mask=row_mask, other=0).to(tl.int32)
    first_stop = tl.min(
        tl.where(
            is_stop & (columns[None, :] < accepted[:, None]),
            columns[None, :],
            WIDTH_PAD,
        ),
        axis=1,
    )
    # A live request always has room for one more token; the floor keeps a
    # broken invariant from walking the cache length backwards.
    budget = tl.maximum(limits - valid - 1, 0)
    kept = tl.minimum(tl.minimum(accepted, first_stop + 1), budget)
    already_stopped = tl.load(stopped_ptr + slots, mask=live, other=0) != 0
    decode_stops = already_stopped | (first_stop < kept) | (kept >= budget)
    kept = tl.where(already_stopped, 0, kept)

    decode = live & ~is_extend
    tl.store(accept_lengths_ptr + rows, kept, mask=decode)
    stops = tl.where(is_extend, extend_stops, decode_stops)
    tl.store(stopped_ptr + slots, stops.to(tl.int1), mask=live)


def _apply_stop_conditions_torch(
    req_pool_indices: torch.Tensor,
    output_tokens: torch.Tensor,
    accept_lengths: torch.Tensor,
    input_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    valid_cache_lengths: torch.Tensor,
    stop_token_ids: torch.Tensor,
    token_limits: torch.Tensor,
    stopped: torch.Tensor,
    num_extends: int,
    num_prefill_outputs: int,
    decode_width: int,
    padding_index: int,
) -> None:
    """Scalar reference of the kernel, for devices without it (and the
    kernel's parity tests): cut each decode row's accepted width at its stop
    conditions and record ``stopped``, in place, as ``apply_stop_conditions``
    documents."""
    for row in range(req_pool_indices.shape[0]):
        slot = int(req_pool_indices[row])
        if slot == padding_index:
            continue
        stop_ids = {int(x) for x in stop_token_ids[slot].tolist() if x >= 0}
        valid = int(valid_cache_lengths[slot])
        limit = int(token_limits[slot])
        if row < num_extends:
            stops = (
                row < num_prefill_outputs
                and valid + int(input_lengths[row]) == int(prompt_lengths[row])
                and (
                    int(output_tokens[row]) in stop_ids
                    or int(prompt_lengths[row]) + 1 >= limit
                )
            )
            stopped[slot] = stops
            continue
        start = num_prefill_outputs + (row - num_extends) * decode_width
        accepted = int(accept_lengths[row])
        window = output_tokens[start : start + accepted].tolist()
        first_stop = next(
            (j for j, token in enumerate(window) if token in stop_ids), decode_width
        )
        budget = max(limit - valid - 1, 0)
        # The kernel's no-stop sentinel is its padded width; any value at
        # least ``accepted`` gives the same ``kept`` and stop decision.
        kept = min(accepted, first_stop + 1, budget)
        if bool(stopped[slot]):
            accept_lengths[row] = 0
            continue
        accept_lengths[row] = kept
        stopped[slot] = first_stop < kept or kept >= budget


def apply_stop_conditions(
    req_pool_indices: torch.Tensor,
    output_tokens: torch.Tensor,
    accept_lengths: torch.Tensor,
    input_lengths: torch.Tensor,
    prompt_lengths: torch.Tensor,
    valid_cache_lengths: torch.Tensor,
    stop_token_ids: torch.Tensor,
    token_limits: torch.Tensor,
    stopped: torch.Tensor,
    *,
    num_extends: int,
    num_prefill_outputs: int,
    decode_width: int,
    padding_index: int,
) -> None:
    """Cut each decode row's accepted width at its stop conditions.

    A request stops on the first accepted token in its stop set, or when its
    token count reaches its limit (prompt plus ``max_new_tokens``). A decode
    row keeps its accepted tokens through that point; a request already
    stopped by an earlier round keeps none. ``stopped`` records which
    requests ended, for the forwards dispatched after this one.

    Args:
        req_pool_indices: ``[batch]`` int64 request slots; ``padding_index``
            marks graph padding rows, which are left alone.
        output_tokens: ``[outputs]`` int32 sampled tokens: one per completing
            prefill (the first ``num_prefill_outputs`` extend requests), then
            ``decode_width`` per decode request.
        accept_lengths: ``[batch]`` int32 output counts; the decode entries
            are cut in place.
        input_lengths: ``[batch]`` integer input rows of every request.
        prompt_lengths: ``[>=num_extends]`` int32 whole-prompt lengths of the
            extend requests.
        valid_cache_lengths: ``[pool]`` int32 cache length of every slot
            before this forward (an extend request's prefix).
        stop_token_ids: ``[pool, K]`` int32 stop tokens per slot, ``-1``
            padded.
        token_limits: ``[pool]`` int32 token count at which a slot's request
            ends.
        stopped: ``[pool]`` bool; read for decode requests, written for every
            live request.
        num_extends: Leading requests that are extends.
        num_prefill_outputs: Leading extend requests that sampled a token.
        decode_width: Output tokens per decode request.
        padding_index: Slot value of a padding row.

    Returns:
        None. ``accept_lengths`` and ``stopped`` are updated in place.

    Raises:
        ValueError: A tensor is not the shape, dtype or layout the update
            indexes with.
    """
    batch_size = req_pool_indices.shape[0]
    if req_pool_indices.ndim != 1 or req_pool_indices.dtype != torch.int64:
        raise ValueError("req_pool_indices must be a 1D int64 tensor")
    if accept_lengths.shape != (batch_size,) or accept_lengths.dtype != torch.int32:
        raise ValueError(f"accept_lengths must be [{batch_size}] int32")
    if input_lengths.shape != (batch_size,):
        raise ValueError(f"input_lengths must be [{batch_size}]")
    if not 0 <= num_prefill_outputs <= num_extends <= batch_size:
        raise ValueError(
            "need 0 <= num_prefill_outputs <= num_extends <= batch, got "
            f"{num_prefill_outputs}, {num_extends}, {batch_size}"
        )
    if prompt_lengths.shape[0] < num_extends:
        raise ValueError(f"prompt_lengths must cover {num_extends} extend requests")
    if decode_width < 1:
        raise ValueError(f"decode_width must be positive, got {decode_width}")
    num_outputs = num_prefill_outputs + (batch_size - num_extends) * decode_width
    if output_tokens.ndim != 1 or output_tokens.shape[0] < num_outputs:
        raise ValueError(f"output_tokens must hold {num_outputs} tokens")
    pool = valid_cache_lengths.shape[0]
    if (
        stop_token_ids.ndim != 2
        or stop_token_ids.shape[0] != pool
        or stop_token_ids.dtype != torch.int32
    ):
        raise ValueError(f"stop_token_ids must be [{pool}, K] int32")
    if token_limits.shape != (pool,) or stopped.shape != (pool,):
        raise ValueError(f"token_limits and stopped must be [{pool}]")
    if stopped.dtype != torch.bool:
        raise ValueError("stopped must be a bool tensor")
    tensors = (
        req_pool_indices,
        output_tokens,
        accept_lengths,
        input_lengths,
        prompt_lengths,
        valid_cache_lengths,
        stop_token_ids,
        token_limits,
        stopped,
    )
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("apply_stop_conditions needs contiguous tensors")
    if any(t.device != req_pool_indices.device for t in tensors):
        raise ValueError("apply_stop_conditions needs colocated tensors")
    if batch_size == 0:
        return
    if not req_pool_indices.is_cuda and req_pool_indices.device.type != "npu":
        _apply_stop_conditions_torch(
            req_pool_indices,
            output_tokens,
            accept_lengths,
            input_lengths,
            prompt_lengths,
            valid_cache_lengths,
            stop_token_ids,
            token_limits,
            stopped,
            num_extends,
            num_prefill_outputs,
            decode_width,
            padding_index,
        )
        return

    kernel = select_kernel(
        "metadata",
        "apply_stop_conditions",
        format_signature(req_pool_indices=dense_tensor_format(req_pool_indices.dtype)),
    )
    kernel(
        req_pool_indices,
        output_tokens,
        accept_lengths,
        input_lengths,
        prompt_lengths,
        valid_cache_lengths,
        stop_token_ids,
        token_limits,
        stopped,
        num_extends,
        num_prefill_outputs,
        decode_width,
        padding_index,
    )


@register_kernel(
    "metadata",
    "apply_stop_conditions",
    name="triton_apply_stop_conditions",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd", "ascend"})),
    signatures=format_signatures("req_pool_indices", "dense", {torch.int64}),
    priority=Priority.PORTABLE,
)
def triton_apply_stop_conditions(
    req_pool_indices,
    output_tokens,
    accept_lengths,
    input_lengths,
    prompt_lengths,
    valid_cache_lengths,
    stop_token_ids,
    token_limits,
    stopped,
    num_extends,
    num_prefill_outputs,
    decode_width,
    padding_index,
) -> None:
    """Apply the portable device update after public input validation.

    Registry leaves accept all 13 arguments positionally in this order:
    ``req_pool_indices``, ``output_tokens``, ``accept_lengths``,
    ``input_lengths``, ``prompt_lengths``, ``valid_cache_lengths``,
    ``stop_token_ids``, ``token_limits``, ``stopped``, ``num_extends``,
    ``num_prefill_outputs``, ``decode_width``, ``padding_index``.
    Shapes, dtypes and in-place semantics follow :func:`apply_stop_conditions`.
    """
    batch_size = req_pool_indices.shape[0]
    _apply_stop_conditions_kernel[(triton.cdiv(batch_size, _BLOCK),)](
        req_pool_indices,
        # An unfinished prefill batch can owe no sample. Keep a valid pointer
        # even when every token load is masked (also required by Ascend DMA).
        output_tokens if output_tokens.numel() else accept_lengths,
        accept_lengths,
        input_lengths,
        # An empty extend span still needs a valid pointer.
        prompt_lengths if prompt_lengths.numel() else accept_lengths,
        valid_cache_lengths,
        stop_token_ids,
        token_limits,
        stopped,
        batch_size,
        num_extends,
        num_prefill_outputs,
        padding_index,
        BLOCK=_BLOCK,
        DECODE_WIDTH=decode_width,
        WIDTH_PAD=max(2, triton.next_power_of_2(decode_width)),
        NUM_STOP_IDS=stop_token_ids.shape[1],
        num_warps=4,
    )
