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

"""PyTorch references for ``tokenspeed_kernel.ops.attention.kpool``: pooled
index keys (KPool) and the top-k pool selection that expands to KV slots."""

from __future__ import annotations

import torch
from tokenspeed_kernel.numerics.torch.transform import sylvester_hadamard
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_FP8_MAX = 448.0


def kpool_compress(
    keys: torch.Tensor, gates: torch.Tensor, ape: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool each group of ``pool`` index keys into one FP8 row.

    The keys mix by the softmax over the pool of ``gates + ape``; the BF16
    mix is Hadamard-rotated (``/ sqrt(dim)``), rounded to BF16 again, and
    quantized to FP8 E4M3 with one ``amax / 448`` scale (``amax >= 1e-4``).

    Args:
        keys: ``[pools, pool, dim]``.
        gates: Per-channel pool scores shaped like ``keys``.
        ape: ``[pool, dim]`` intra-pool position bias.

    Returns:
        ``(values [pools, dim] FP8, scales [pools] FP32)``.
    """
    dim = keys.shape[-1]
    logits = gates.float() + ape.unsqueeze(0)
    pooled = (torch.softmax(logits, dim=1) * keys.float()).sum(dim=1)
    pooled = pooled.to(torch.bfloat16).float()
    hadamard = sylvester_hadamard(dim, keys.device)
    pooled = ((pooled @ hadamard) / dim**0.5).to(torch.bfloat16).float()
    scale = pooled.abs().amax(dim=-1).clamp(min=1e-4) / _FP8_MAX
    values = (pooled / scale[:, None]).clamp(-_FP8_MAX, _FP8_MAX)
    return values.to(torch.float8_e4m3fn), scale


@register_kernel(
    "attention",
    "kpool_decode_append",
    name="torch_kpool_decode_append",
    solution="torch",
    signatures=format_signatures("k", "dense", {torch.bfloat16}),
    priority=Priority.REFERENCE,
)
def torch_kpool_decode_append(
    *,
    k: torch.Tensor,
    gate: torch.Tensor,
    tail_k: torch.Tensor,
    tail_gate: torch.Tensor,
    seq_lens: torch.Tensor,
    request_slots: torch.Tensor,
    index_block_table: torch.Tensor,
    index_values: torch.Tensor,
    index_scales: torch.Tensor,
    ape: torch.Tensor,
) -> None:
    """Append a decode window to request tails, writing each completed pool.

    Step ``s`` of request ``r`` is position ``seq_lens[r] - steps + s``; it
    lands in tail row ``position % tail_size`` of ``request_slots[r]``. A
    position closing a pool (``position % pool == pool - 1``) compresses the
    pool's tail rows (:func:`kpool_compress`) into index row ``pool_id %
    rows_per_page`` of page ``index_block_table[r, pool_id //
    rows_per_page]``. Every cache is updated in place.
    """
    pool = ape.shape[0]
    tail_size = tail_k.shape[1]
    rows_per_page = index_values.shape[1]
    steps = k.shape[1]
    for req in range(k.shape[0]):
        request_slot = int(request_slots[req])
        first_position = int(seq_lens[req]) - steps
        for step in range(steps):
            position = first_position + step
            tail_k[request_slot, position % tail_size] = k[req, step]
            tail_gate[request_slot, position % tail_size] = gate[req, step]
            if position % pool != pool - 1:
                continue
            tail_rows = torch.arange(position - (pool - 1), position + 1)
            tail_rows = tail_rows.remainder(tail_size).to(k.device)
            pooled, pooled_scale = kpool_compress(
                tail_k[request_slot, tail_rows].unsqueeze(0),
                tail_gate[request_slot, tail_rows].unsqueeze(0),
                ape,
            )
            pool_id = position // pool
            page = int(index_block_table[req, pool_id // rows_per_page])
            index_values[page, pool_id % rows_per_page] = pooled[0]
            index_scales[page, pool_id % rows_per_page, 0] = pooled_scale[0]


@register_kernel(
    "attention",
    "kpool_prefill_tail_write",
    name="torch_kpool_prefill_tail_write",
    solution="torch",
    signatures=format_signatures("k", "dense", {torch.bfloat16}),
    priority=Priority.REFERENCE,
)
def torch_kpool_prefill_tail_write(
    *,
    k: torch.Tensor,
    gate: torch.Tensor,
    tail_k: torch.Tensor,
    tail_gate: torch.Tensor,
    source_starts: torch.Tensor,
    destination_slots: torch.Tensor,
    destination_positions: torch.Tensor,
    valid_counts: torch.Tensor,
    pool_size: int,
) -> None:
    """Copy each request's incomplete-pool rows into its tail, in place.

    Entry ``i`` copies ``min(valid_counts[i], pool_size)`` rows from
    ``source_starts[i]`` to tail ``destination_slots[i]`` at rows
    ``(destination_positions[i] + offset) % tail_size``. Entries with an
    out-of-range slot or a negative position, and source rows outside ``k``,
    are skipped.
    """
    metadata = zip(
        source_starts.tolist(),
        destination_slots.tolist(),
        destination_positions.tolist(),
        valid_counts.tolist(),
    )
    for source, slot, position, count in metadata:
        if not (0 <= slot < tail_k.shape[0]) or position < 0:
            continue
        for offset in range(min(count, pool_size)):
            row = source + offset
            if not 0 <= row < k.shape[0]:
                continue
            tail_row = (position + offset) % tail_k.shape[1]
            tail_k[slot, tail_row] = k[row]
            tail_gate[slot, tail_row] = gate[row]


def kpool_scores(
    q: torch.Tensor,
    keys: torch.Tensor,
    weights: torch.Tensor,
    scale: float,
    *,
    apply_relu: bool,
) -> torch.Tensor:
    """Pool scores of one query row: ``sum_h w_h * relu(scale * q_h . k)``.

    Args:
        q: ``[heads, dim]``.
        keys: ``[pools, dim]`` pooled index keys.
        weights: ``[heads]``.
        scale: Logit scale, applied before the ReLU.
        apply_relu: Clamp each head's score at zero.

    Returns:
        ``[pools]`` FP32 scores.
    """
    head_scores = torch.einsum("hd,pd->hp", q.float(), keys.float()) * scale
    if apply_relu:
        head_scores = head_scores.relu()
    return (head_scores * weights.float()[:, None]).sum(0)


@register_kernel(
    "attention",
    "kpool_select",
    name="torch_kpool_select",
    solution="torch",
    signatures=format_signatures("q", "dense", {torch.bfloat16, torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_kpool_select(
    q: torch.Tensor,
    pooled_k: torch.Tensor,
    weights: torch.Tensor,
    causal_lens: torch.Tensor,
    req_ids: torch.Tensor,
    index_table: torch.Tensor,
    kv_table: torch.Tensor,
    *,
    topk: int,
    pool_size: int,
    index_page_size: int,
    kv_page_size: int,
    scale: float,
    apply_relu: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KV slots of each token's top-k pools and its incomplete tail.

    A token with ``L`` causal positions has ``L // pool_size`` complete
    pools; with at most ``topk`` of them all are kept, otherwise the top-k
    by :func:`kpool_scores` (order unspecified). The kept pools' positions,
    then the tail ``[pools * pool_size, L)``, map through ``kv_table``.

    Args:
        q: ``[tokens, heads, dim]``.
        pooled_k: Dequantized pooled keys, ``[index_pages * index_page_size,
            dim]`` rows.
        weights: ``[tokens, heads]``.
        causal_lens: ``[tokens]`` visible positions.
        req_ids: ``[tokens]`` request of each token.
        index_table: ``[requests, index_pages]`` pool pages.
        kv_table: ``[requests, kv_pages]`` KV pages.

    Returns:
        ``(slots [tokens, topk * pool_size + pool_size - 1] int32, lens
        [tokens] int32)``, ``-1`` padded.
    """
    width = topk * pool_size + pool_size - 1
    slots = torch.full((q.shape[0], width), -1, dtype=torch.int32, device=q.device)
    lens = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    pooled_k = pooled_k.view(-1, index_page_size, pooled_k.shape[-1])
    for token in range(q.shape[0]):
        req = int(req_ids[token])
        causal = max(int(causal_lens[token]), 0)
        num_pools = causal // pool_size
        if num_pools <= topk:
            selected = torch.arange(num_pools, device=q.device)
        else:
            pools = torch.arange(num_pools, device=q.device)
            pages = index_table[req, pools // index_page_size].long()
            keys = pooled_k[pages, pools % index_page_size]
            scores = kpool_scores(
                q[token], keys, weights[token], scale, apply_relu=apply_relu
            )
            selected = torch.topk(scores, k=topk, sorted=False).indices
        raw_slots = (
            selected[:, None] * pool_size + torch.arange(pool_size, device=q.device)
        ).flatten()
        raw_slots = torch.cat(
            (raw_slots, torch.arange(num_pools * pool_size, causal, device=q.device))
        ).long()
        pages = kv_table[req, raw_slots // kv_page_size].long()
        global_slots = pages * kv_page_size + raw_slots % kv_page_size
        slots[token, : raw_slots.numel()] = global_slots.to(torch.int32)
        lens[token] = raw_slots.numel()
    return slots, lens


@register_kernel(
    "attention",
    "kpool_expand",
    name="torch_kpool_expand",
    solution="torch",
    signatures=format_signatures("pool_indices", "dense", {torch.int32, torch.int64}),
    priority=Priority.REFERENCE,
)
def torch_kpool_expand(
    pool_indices: torch.Tensor,
    causal_lens: torch.Tensor,
    req_ids: torch.Tensor,
    block_table: torch.Tensor,
    *,
    pool_size: int,
    block_size: int,
    append_tail: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand selected pools to flat KV slots.

    A token with no more complete pools than ``pool_indices`` columns keeps
    all of them, otherwise its listed pools; at most the complete-pool
    history is expanded, a listed pool outside the history expands to
    ``-1`` columns, and ``append_tail`` adds the incomplete tail. Positions
    past the causal length or the block table stay ``-1`` and do not count.

    Returns:
        ``(slots [tokens, width] int32, lens [tokens] int32)`` where width is
        ``topk_pools * pool_size`` (``+ pool_size - 1`` with the tail).
    """
    num_tokens, topk_pools = pool_indices.shape
    width = topk_pools * pool_size + (pool_size - 1 if append_tail else 0)
    device = pool_indices.device
    slots = torch.full((num_tokens, width), -1, dtype=torch.int32, device=device)
    lens = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    for token in range(num_tokens):
        seq_len = max(int(causal_lens[token]), 0)
        num_pools = seq_len // pool_size
        history_len = min(num_pools * pool_size, topk_pools * pool_size)
        selected = (
            list(range(num_pools))
            if num_pools <= topk_pools
            else pool_indices[token].tolist()
        )
        raw_slots: list[int] = []
        for pool in selected[:topk_pools]:
            if len(raw_slots) >= history_len:
                break
            if 0 <= pool < num_pools:
                raw_slots.extend(range(pool * pool_size, (pool + 1) * pool_size))
            else:
                raw_slots.extend([-1] * pool_size)
        if append_tail:
            raw_slots.extend(range(num_pools * pool_size, seq_len))
        count = 0
        req = int(req_ids[token])
        for column, raw_slot in enumerate(raw_slots[:width]):
            if not 0 <= raw_slot < seq_len:
                continue
            page_idx = raw_slot // block_size
            if page_idx >= block_table.shape[1]:
                continue
            page = int(block_table[req, page_idx])
            slots[token, column] = page * block_size + raw_slot % block_size
            count += 1
        lens[token] = count
    return slots, lens
