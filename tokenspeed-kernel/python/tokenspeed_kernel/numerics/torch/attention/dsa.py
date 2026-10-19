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

"""PyTorch references for ``tokenspeed_kernel.ops.attention.dsa``
(DeepSeek sparse attention): the indexer's top-k key selection, and MLA over
the selected keys."""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

_INDEX_SIGNATURES = format_signatures(
    "q", "dense", {torch.bfloat16, torch.float32, torch.float8_e4m3fn}
)
# Candidates of one query row: (ids, FP32 scores), for tie-aware checks.
Candidates = tuple[torch.Tensor, torch.Tensor]


def dsa_index_scores(
    q: torch.Tensor, weights: torch.Tensor, keys: torch.Tensor, scale: float
) -> torch.Tensor:
    """Indexer scores of one query row: ``scale * sum_h w_h * relu(k . q_h)``.

    Args:
        q: ``[heads, dim]`` index query.
        weights: ``[heads]`` head weights.
        keys: ``[candidates, dim]`` index keys.
        scale: Score scale.

    Returns:
        ``[candidates]`` FP32 scores.
    """
    per_head = keys.float() @ q.float().transpose(0, 1)
    return (per_head.relu() * weights.float()).sum(dim=1) * scale


def _select(
    scores: torch.Tensor, ids: torch.Tensor, topk: int
) -> tuple[torch.Tensor, int]:
    count = min(scores.numel(), topk)
    return ids.index_select(0, torch.topk(scores, count).indices), count


@register_kernel(
    "attention",
    "dsa_paged_index_topk",
    name="torch_dsa_paged_index_topk",
    solution="torch",
    signatures=_INDEX_SIGNATURES,
    priority=Priority.REFERENCE,
)
def torch_dsa_paged_index_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    *,
    page_size: int,
    topk: int,
    softmax_scale: float,
    q_len_per_req: int,
) -> tuple[torch.Tensor, torch.Tensor, list[Candidates]]:
    """Top-k cache slots of each decode query row by indexer score.

    Row ``t`` is query ``t % q_len_per_req`` of request ``t //
    q_len_per_req`` and sees the request's first ``seq_len - q_len_per_req +
    1 + offset`` positions, paged through ``block_table``.

    Args:
        q: ``[rows, heads, dim]`` index queries.
        weights: ``[rows, heads]`` head weights.
        index_k: ``[slots, dim]`` index keys by cache slot.
        seq_lens: ``[requests]`` cached lengths including the queries.
        block_table: ``[requests, max_pages]``.

    Returns:
        ``(slots [rows, topk] int32, lens [rows] int32, candidates)``: the
        selected slots best first, ``-1`` padded; per row, all candidate
        slots and their scores.
    """
    out = torch.full((q.shape[0], topk), -1, device=q.device, dtype=torch.int32)
    lens = torch.zeros((q.shape[0],), device=q.device, dtype=torch.int32)
    candidates: list[Candidates] = []
    for row in range(q.shape[0]):
        request, offset = divmod(row, q_len_per_req)
        visible = max(int(seq_lens[request]) - q_len_per_req + 1 + offset, 0)
        positions = torch.arange(visible, device=q.device)
        pages = block_table[request].long().index_select(0, positions // page_size)
        slots = pages * page_size + positions.remainder(page_size)
        scores = dsa_index_scores(
            q[row], weights[row], index_k.index_select(0, slots), softmax_scale
        )
        selected, lens[row] = _select(scores, slots, topk)
        out[row, : selected.numel()] = selected.to(torch.int32)
        candidates.append((slots, scores))
    return out, lens, candidates


@register_kernel(
    "attention",
    "dsa_workspace_index_topk",
    name="torch_dsa_workspace_index_topk",
    solution="torch",
    signatures=_INDEX_SIGNATURES,
    priority=Priority.REFERENCE,
)
def torch_dsa_workspace_index_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k: torch.Tensor,
    kv_workspace_slots: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    topk: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, list[Candidates]]:
    """Top-k workspace rows of each prefill query row by indexer score.

    Row ``t`` sees workspace rows ``[row_starts[t], row_ends[t])`` (clamped to
    the workspace), whose cache slots ``kv_workspace_slots`` names.

    Returns:
        ``(rows [tokens, topk] int32, lens [tokens] int32, candidates)``: the
        selected workspace rows best first, ``-1`` padded; per token, all
        candidate rows and their scores.
    """
    out = torch.full((q.shape[0], topk), -1, device=q.device, dtype=torch.int32)
    lens = torch.zeros((q.shape[0],), device=q.device, dtype=torch.int32)
    candidates: list[Candidates] = []
    for row in range(q.shape[0]):
        start = max(int(row_starts[row]), 0)
        end = max(min(int(row_ends[row]), kv_workspace_slots.numel()), start)
        rows = torch.arange(start, end, device=q.device)
        slots = kv_workspace_slots.index_select(0, rows).long()
        scores = dsa_index_scores(
            q[row], weights[row], index_k.index_select(0, slots), softmax_scale
        )
        selected, lens[row] = _select(scores, rows, topk)
        out[row, : selected.numel()] = selected.to(torch.int32)
        candidates.append((rows, scores))
    return out, lens, candidates


@register_kernel(
    "attention",
    "dsa_logits_topk",
    name="torch_dsa_logits_topk",
    solution="torch",
    signatures=format_signatures("logits", "dense", {torch.float32}),
    priority=Priority.REFERENCE,
)
def torch_dsa_logits_topk(
    logits: torch.Tensor, seq_lens: torch.Tensor, topk: int, next_n: int
) -> torch.Tensor:
    """Top-k columns of each row of indexer logits inside its causal window.

    Row ``r`` is query ``r % next_n`` of request ``r // next_n`` and sees
    columns ``[0, seq_lens[request] - next_n + 1 + r % next_n)``. Ties go to
    the lowest column.

    Returns:
        ``[rows, topk]`` int32 columns best first, ``-1`` padded.
    """
    rows, cols = logits.shape
    row_ids = torch.arange(rows, device=logits.device)
    window = seq_lens.long()[row_ids // next_n] - next_n + 1 + row_ids % next_n
    visible = torch.arange(cols, device=logits.device)[None, :] < window[:, None]
    masked = logits.masked_fill(~visible, float("-inf"))
    indices = torch.argsort(masked, dim=1, descending=True, stable=True)[:, :topk]
    values = masked.gather(1, indices)
    out = torch.full((rows, topk), -1, dtype=torch.int32, device=logits.device)
    out[:, : indices.shape[1]] = torch.where(values != float("-inf"), indices, -1).to(
        torch.int32
    )
    return out


@register_kernel(
    "attention",
    "dsa_sparse_attention",
    name="torch_dsa_sparse_attention",
    solution="torch",
    signatures=format_signatures("q", "dense", {torch.bfloat16}),
    priority=Priority.REFERENCE,
)
def torch_dsa_sparse_attention(
    q: torch.Tensor,
    latent: torch.Tensor,
    rope: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """MLA of each query row over its selected latent cache slots.

    Logits are ``q[:, :rank] . latent + q[:, rank:] . rope`` (no rope part
    when ``rope`` is None); the latent is also the value. Negative slots and
    slots past ``topk_lens`` are ignored; a row with none returns zeros.

    Args:
        q: ``[tokens, heads, rank (+ rope_dim)]``.
        latent: ``[slots, rank]``.
        rope: ``[slots, rope_dim]`` or None.
        topk_slots: ``[tokens, topk]``.
        topk_lens: ``[tokens]``.

    Returns:
        ``[tokens, heads, rank]`` BF16.
    """
    rank = latent.shape[1]
    refs = []
    for token in range(q.shape[0]):
        slots = topk_slots[token, : int(topk_lens[token])].long()
        slots = slots[slots >= 0]
        q_nope = q[token, :, :rank].float()
        if slots.numel() == 0:
            refs.append(torch.zeros_like(q_nope))
            continue
        k_nope = latent.index_select(0, slots).float()
        scores = torch.einsum("hd,kd->hk", q_nope, k_nope)
        if rope is not None:
            k_rope = rope.index_select(0, slots).float()
            scores += torch.einsum("hd,kd->hk", q[token, :, rank:].float(), k_rope)
        probs = torch.softmax(scores * softmax_scale, dim=-1)
        refs.append(probs @ k_nope)
    return torch.stack(refs).to(torch.bfloat16)


@register_kernel(
    "attention",
    "dsa_compact_index_pages",
    name="torch_dsa_compact_index_pages",
    solution="torch",
    signatures=format_signatures("table", "dense", {torch.int32}),
    priority=Priority.REFERENCE,
)
def torch_dsa_compact_index_pages(
    table: torch.Tensor, requests: torch.Tensor, causal: torch.Tensor, page: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Each row's mapped index pages inside its causal length, packed left.

    A table column counts when its page id is positive and it starts before
    the row's causal length; rows with an out-of-range request are empty.

    Returns:
        On the host: ``(pages [rows, cols], columns [rows, cols], lengths
        [rows, 1])`` int32: the kept page ids, their table columns, and the
        positions they cover.
    """
    rows, cols = requests.numel(), table.shape[1]
    pages = torch.zeros(rows, cols, dtype=torch.int32)
    positions = torch.zeros_like(pages)
    lengths = torch.zeros(rows, 1, dtype=torch.int32)
    for row, (req, length) in enumerate(zip(requests.tolist(), causal.tolist())):
        if not 0 <= req < table.shape[0]:
            continue
        valid = [
            col
            for col, value in enumerate(table[req].tolist())
            if value > 0 and col * page < length
        ]
        pages[row, : len(valid)] = table[req, valid].cpu()
        positions[row, : len(valid)] = torch.tensor(valid, dtype=torch.int32)
        lengths[row] = sum(min(page, length - col * page) for col in valid)
    return pages, positions, lengths
