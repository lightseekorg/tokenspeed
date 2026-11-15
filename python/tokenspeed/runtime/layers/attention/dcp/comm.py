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

"""DCP attention collectives; all probability arithmetic uses natural-log LSE."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.dsv4._triton.dcp import (
    combine_dcp_attention_partials,
    dcp_apply_sink,
    dcp_weight_for_reduce_scatter,
    dispatch_dcp_attention_partials,
)

from tokenspeed.runtime.distributed.comm_ops import (
    acquire_exchange_input,
    all_gather,
    all_gather_single,
    all_reduce,
    all_to_all_single,
    reduce_scatter,
    try_combine_records,
)


def _query_payload(query: torch.Tensor) -> torch.Tensor:
    """``[tokens, heads, dim]`` as contiguous ``[tokens, heads * dim]`` rows.

    NCCL wrappers need not expose FP8 dtypes for a byte-preserving gather, so
    FP8 is reinterpreted as bytes rather than cast: quantized values must
    retain their bits.
    """
    tokens, heads, dim = query.shape
    payload = query.reshape(tokens, heads * dim).contiguous()
    if query.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        payload = payload.view(torch.uint8)
    return payload


def gather_query_heads(query: torch.Tensor, group: tuple[int, ...]) -> torch.Tensor:
    """Gather only actual TP query heads after QNorm/RoPE; padding stays local."""
    if len(group) == 1:
        return query
    tokens, heads, dim = query.shape
    # The 2-D inner-dimension collective uses the existing low-latency
    # backend where supported, with its topology/dtype/NCCL fallbacks. Its
    # symmetric buffer is sized for the prefill token budget although decode
    # only ever gathers max_decode_bs * spec_tokens rows; a per-collective
    # capacity needs the symmetric buffers managed in one place first.
    gathered = all_gather(_query_payload(query), group, dim=-1).view(query.dtype)
    return gathered.reshape(tokens, heads * len(group), dim)


def gather_query_heads_lent(
    query: torch.Tensor, group: tuple[int, ...]
) -> torch.Tensor:
    """:func:`gather_query_heads` through the group's exchange input.

    The rows are copied into ``acquire_exchange_input``'s storage, gathered
    rank-major (pulled in place where the backend lends its buffer) and laid
    out token-major in 8-byte words: the same bytes at the same places.
    Only for callers whose ``query`` does not live in that storage.
    """
    if len(group) == 1:
        return query
    tokens, heads, dim = query.shape
    payload = _query_payload(query)
    send = acquire_exchange_input(payload.nbytes, query, group)
    send = send.view(payload.dtype).view(payload.shape)
    send.copy_(payload)
    gathered = payload.new_empty((len(group) * tokens, payload.shape[1]))
    all_gather_single(gathered, send, group)
    rows = gathered.view(torch.uint8).view(len(group), tokens, -1)
    if rows.shape[-1] % 8 == 0:
        # Copy whole 8-byte words: a strided byte copy of an FP8 query is far slower.
        rows = rows.view(torch.int64)
    token_major = rows.transpose(0, 1).contiguous().view(torch.uint8)
    return token_major.view(query.dtype).view(tokens, heads * len(group), dim)


def gather_query_heads_rank_major(
    query: torch.Tensor, group: tuple[int, ...]
) -> torch.Tensor:
    """Gather the group's TP query heads rank-major, as the collective lands them.

    :func:`gather_query_heads` returns gathered head ``r * heads + h`` of a
    token at ``[token, r * heads + h]``, which costs a relayout copy of the
    gathered rows; here it stays at ``[r, token, h]``, for a core that reads
    that layout (``dsa_prefill_dcp_records``).

    Args:
        query: This rank's ``[tokens, heads, dim]`` TP heads.
        group: The KVP subgroup gathering them; every rank holds the same
            tokens.

    Returns:
        ``[len(group), tokens, heads, dim]`` in ``query``'s dtype, a view of
        the gathered buffer (no copy).
    """
    tokens, heads, dim = query.shape
    if len(group) == 1:
        return query.unsqueeze(0)
    payload = _query_payload(query)
    gathered = payload.new_empty((len(group) * tokens, heads * dim))
    all_gather_single(gathered, payload, group)
    return gathered.view(query.dtype).view(len(group), tokens, heads, dim)


def gather_query_heads_padded(
    query: torch.Tensor, group: tuple[int, ...], padded_heads: int
) -> torch.Tensor:
    """Gather the group's TP query heads token-major into a zero-padded query.

    A core that runs only some head counts (trtllm-gen sparse MLA: 64 or 128)
    pads :func:`gather_query_heads`' result with a second copy of every row;
    here the rank-major gather lands in the leading heads with one copy and
    only the padding heads are zeroed.

    Args:
        query: This rank's ``[tokens, heads, dim]`` TP heads.
        group: The KVP subgroup gathering them; every rank holds the same
            tokens.
        padded_heads: The core's head count, at least the gathered heads.

    Returns:
        ``[tokens, padded_heads, dim]`` in ``query``'s dtype: gathered head
        ``r * heads + h`` at ``[token, r * heads + h]`` as
        :func:`gather_query_heads` places it, zeros after.
    """
    tokens, heads, dim = query.shape
    total = heads * len(group)
    if padded_heads < total:
        raise ValueError("padded query heads must hold every gathered head")
    padded = query.new_empty((tokens, padded_heads, dim))
    gathered = gather_query_heads_rank_major(query, group)
    rows = padded.view(torch.uint8).view(tokens, padded_heads * dim * query.itemsize)
    src = gathered.view(torch.uint8).view(len(group), tokens, -1)
    width = src.shape[-1]
    if width % 8 == 0 and rows.shape[-1] % 8 == 0:
        # Copy whole 8-byte words: a strided byte copy of an FP8 query is far slower.
        rows, src, width = rows.view(torch.int64), src.view(torch.int64), width // 8
    rows[:, len(group) * width :].zero_()
    rows[:, : len(group) * width].view(tokens, len(group), width).copy_(
        src.transpose(0, 1)
    )
    return padded


def combine_attention_records(
    records: torch.Tensor,
    *,
    group: tuple[int, ...],
    head_dim: int,
    partial_dtype: torch.dtype,
    output_dtype: torch.dtype,
    sink: torch.Tensor | None,
) -> torch.Tensor:
    """Send packed partial records to their head owners and combine them there.

    The head-sharded exchange of :func:`combine_attention_partials`, from
    records already packed (``dispatch_dcp_attention_partials``, or a core
    writing that layout, ``dsa_prefill_dcp_records``).

    Args:
        records: Contiguous int32 ``[len(group), tokens, local_heads, words + 1]``:
            row ``r`` goes to group rank ``r``.
        group: The KVP subgroup exchanging them.
        head_dim: Unpadded partial head dimension.
        partial_dtype: The dtype the records pack (FP16, BF16 or FP32).
        output_dtype: Desired output dtype; weighting and summation use FP32.
        sink: Required keyword: TP-local sink logits, or explicitly None.

    Returns:
        Contiguous ``[tokens, local_heads, head_dim]`` of this rank's heads.
    """
    if sink is None:
        combined = try_combine_records(
            records,
            group,
            head_dim=head_dim,
            partial_dtype=partial_dtype,
            output_dtype=output_dtype,
        )
        if combined is not None:
            return combined
    received = torch.empty_like(records)
    all_to_all_single(received.flatten(), records.flatten(), group)
    return combine_dcp_attention_partials(
        received,
        head_dim=head_dim,
        partial_dtype=partial_dtype,
        output_dtype=output_dtype,
        sink=sink,
    )


def _lent_records(
    partials: torch.Tensor, degree: int, group: tuple[int, ...]
) -> torch.Tensor:
    """Storage for ``dispatch_dcp_attention_partials``' records of
    ``partials``, borrowed from the group's exchange input."""
    tokens, heads, dim = partials.shape
    shape = (
        degree,
        tokens,
        heads // degree,
        -(-dim * partials.element_size() // 4) + 1,
    )
    nbytes = shape[0] * shape[1] * shape[2] * shape[3] * 4
    lent = acquire_exchange_input(nbytes, partials, group)
    return lent.view(torch.int32).view(shape)


def combine_attention_partials(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    *,
    group: tuple[int, ...],
    rank: int,
    sink: torch.Tensor | None,
    keep_all_heads: bool,
    lend_records: bool = False,
) -> torch.Tensor:
    """Combine partials across context shards, returning the requested heads.

    Args:
        local_output: CUDA local context output [tokens, gathered_heads, head_dim].
        local_lse: Natural-log FP32 LSE [tokens, gathered_heads].
        group: The KVP subgroup (consecutive attention-TP ranks) whose
            partials are merged.
        rank: This process's position in group.
        sink: Required keyword: TP-local sink logits, or explicitly None for
            attention without a sink. There is no implicit sink policy.
        keep_all_heads: Required keyword. ``False`` is the head-sharded form:
            the query heads were gathered from every rank of the group and the
            original partials and LSE are sent to their head owner, which
            weights and sums them in FP32. ``True`` is the head-replicated form
            (a layer holding every head, as under query context parallelism): every
            rank attended all heads over its own pages and the weighted
            partials are all-reduced, so every rank returns every head; no
            sink yet. Both forms split heads by the group's degree, so
            ``heads % len(group) == 0`` holds
            for either.
        lend_records: Pack the head-sharded form's records into the group's
            exchange input (``acquire_exchange_input``), so a backend that
            pulls it in place exchanges and combines them in one step
            (``try_combine_records``). Only for callers whose inputs do not
            live in that storage and who start no other exchange in between.

    Returns:
        Original-dtype output [tokens, heads, head_dim] -- the TP-local heads,
        or every head with ``keep_all_heads`` -- with the sink applied once
        after combining every context shard.
    """
    degree = len(group)
    if not 0 <= rank < degree or local_output.shape[1] % degree:
        raise ValueError("DCP combine topology does not partition query heads")
    if local_lse.shape != local_output.shape[:-1]:
        raise ValueError("DCP combine output and LSE shapes disagree")
    if keep_all_heads and sink is not None:
        raise ValueError("the head-replicated DCP combine takes no sink")
    heads = local_output.shape[1] // degree
    if sink is not None and sink.numel() < heads:
        raise ValueError("DCP sink must cover the TP-local heads")
    if (
        not keep_all_heads
        and local_output.is_cuda
        and local_output.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and (sink is None or local_output.dtype != torch.float32)
    ):
        packed = dispatch_dcp_attention_partials(
            local_output,
            local_lse.float(),
            degree=degree,
            out=_lent_records(local_output, degree, group) if lend_records else None,
        )
        return combine_attention_records(
            packed,
            group=group,
            head_dim=local_output.shape[-1],
            partial_dtype=local_output.dtype,
            output_dtype=local_output.dtype,
            sink=sink,
        )
    # Gather FP32 bits as 16-bit rows so the shared backend can use its
    # low-latency inner gather. This is a view, never a BF16 conversion:
    # infinities and NaN payloads retain their bits. Other backends fall back.
    tokens, gathered_heads = local_lse.shape
    packed_lse = local_lse.float().contiguous().view(torch.bfloat16)
    gathered_lse = all_gather(packed_lse, group, dim=-1)
    gathered_lse = (
        gathered_lse.view(torch.float32)
        .reshape(tokens, degree, gathered_heads)
        .permute(1, 0, 2)
    )
    weighted, lse = dcp_weight_for_reduce_scatter(local_output, gathered_lse, rank)
    if keep_all_heads:
        # Every head's partial is weighted; the sum over owners is the full
        # attention of every head on every rank.
        return (
            all_reduce(weighted, group)
            .movedim(0, 1)
            .to(dtype=local_output.dtype, memory_format=torch.contiguous_format)
            .contiguous()
        )
    output = reduce_scatter(weighted, group).movedim(0, 1)
    if sink is None:
        # Request token-major storage during the cast to avoid copying twice.
        # Keep contiguous() for FP32, where to() can return the original view.
        return output.to(
            dtype=local_output.dtype, memory_format=torch.contiguous_format
        ).contiguous()
    return dcp_apply_sink(output, lse, sink, dtype=local_output.dtype)
