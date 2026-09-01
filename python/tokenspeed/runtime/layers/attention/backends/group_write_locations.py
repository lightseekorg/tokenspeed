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

"""Cache write-location math, as pure functions.

Two families share the invariant ``slot = table[req, pos // P] * P + pos % P``
but differ in table shape: the per-group family (MHA/TRTLLM/MSA) computes a
dict of locations over per-group tables with heterogeneous granularities,
the MLA family computes one location tensor over the single full-history
table. Everything here is stateless — callers pass the geometry explicitly,
so the math is testable without a backend instance and can never read stale
backend state.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.triton.mla_write_locations import (
    mla_write_locations,
)

# ---------------------------------------------------------------------------
# Per-group family (MHA / TRTLLM / MSA): dict-of-groups locations
# ---------------------------------------------------------------------------


def decode_group_out_cache_locs(
    page_tables: dict[str, torch.Tensor],
    seq_lens: torch.Tensor,
    granularity_of,
    num_tokens_per_req: int = 1,
):
    """Per-group decode write locs, gathered from the group's own read table.

    Plain decode writes one token per request at ``seq_len-1``; spec verify
    writes ``num_tokens_per_req`` at ``seq_len-N..seq_len-1``, flattened
    token-major per request (``[bs*N]``, single-table verify layout).
    Positions clamp at 0 for graph-padded rows (seq_len 1 < N), which
    dereference the dummy page harmlessly. The tail page is never a hole
    (SWA holes sit only at the window front).

    Args:
        page_tables: ``group_id -> [bs, cols]`` kernel-page tables.
        seq_lens: ``[bs]`` live lengths.
        granularity_of: ``group_id -> page size`` callable (the group's own
            granularity; heterogeneous groups divide by their own).
        num_tokens_per_req: Locations per request (1, or the verify width).

    Returns:
        ``group_id -> [bs * num_tokens_per_req]`` absolute slot locations.
    """
    n = num_tokens_per_req
    if n == 1:
        pos = (seq_lens - 1).to(torch.int64)
    else:
        steps = torch.arange(n, device=seq_lens.device, dtype=torch.int64)
        pos = (seq_lens.to(torch.int64).unsqueeze(1) - n + steps).clamp_min(0)
        pos = pos.reshape(-1)
    out = {}
    for gid, table in page_tables.items():
        ps = granularity_of(gid)
        page_idx = pos // ps
        off = (pos % ps).to(torch.int32)
        if n == 1:
            pages = table.gather(1, page_idx.unsqueeze(1)).squeeze(1)
        else:
            pages = table.gather(1, page_idx.view(-1, n)).reshape(-1)
        # Mirror the graph-path kernel's clamp: -1 pads/holes route to dummy page 0.
        out[gid] = pages.clamp_min(0) * ps + off
    return out


def extend_group_out_cache_locs(
    page_tables: dict[str, torch.Tensor],
    extend_prefix_lens_cpu: torch.Tensor,
    extend_seq_lens_cpu: torch.Tensor,
    granularity_of,
):
    """Per-group extend write locs: positions ``[prefix_len, seq_len)`` per
    request, flattened in q/k/v token order (cu_extend_seq_lens). Bounds come
    from the CPU mirrors — no per-request GPU sync.
    TODO(cache-perf): batch the per-request loop via repeat_interleave.
    """
    device = next(iter(page_tables.values())).device
    prefix_lens = [int(x) for x in extend_prefix_lens_cpu.tolist()]
    extend_lens = [int(x) for x in extend_seq_lens_cpu.tolist()]
    out = {gid: [] for gid in page_tables}
    for i, (start, num_new) in enumerate(zip(prefix_lens, extend_lens)):
        pos = torch.arange(start, start + num_new, dtype=torch.int64, device=device)
        for gid, table in page_tables.items():
            ps = granularity_of(gid)
            max_col = (start + num_new - 1) // ps
            if max_col >= table.shape[1]:
                raise RuntimeError(
                    f"extend write locations out of table bounds: group "
                    f"{gid!r} table {tuple(table.shape)} req={i} "
                    f"prefix={start} new={num_new} page_size={ps} needs "
                    f"col {max_col}"
                )
            pages = table[i].gather(0, pos // ps)
            out[gid].append(pages * ps + (pos % ps).to(torch.int32))
    return {
        gid: (
            torch.cat(chunks)
            if chunks
            else torch.empty(0, dtype=torch.int32, device=device)
        )
        for gid, chunks in out.items()
    }


def check_group_write_locs(
    page_tables: dict[str, torch.Tensor],
    out_cache_locs: dict[str, torch.Tensor],
    granularity_of,
) -> None:
    """TOKENSPEED_CACHE_DEBUG assertion (eager only, GPU sync): write pages
    must be real and inside the group's table. Not for graph-padded batches —
    dummy rows would trip the non-hole assert (see the padding contract in
    GroupGraphBuffers.fill)."""
    for gid, locs in out_cache_locs.items():
        pages = (locs // granularity_of(gid)).to(torch.int32)
        table = page_tables[gid]
        assert (
            pages != 0
        ).all(), f"cache write location in null page 0 for group {gid!r}"
        real = table[table > 0]
        assert torch.isin(
            pages, real
        ).all(), f"cache write pages escape group {gid!r}'s table"


# ---------------------------------------------------------------------------
# MLA family: single full-history latent table
# ---------------------------------------------------------------------------


def verify_q_len(spec_num_tokens: int, is_draft: bool, forward_mode) -> int:
    """KV write locations each request needs this decode step.

    The target's verify decode writes the whole speculative window
    (``spec_num_tokens`` trailing positions); plain decode and any draft
    write a single location.
    """
    if spec_num_tokens <= 1:
        return 1
    if (
        not is_draft
        and forward_mode is not None
        and (forward_mode.is_decode() or forward_mode.is_mixed())
    ):
        return spec_num_tokens
    return 1


def graph_verify_q_len(spec_num_tokens: int, is_draft: bool) -> int:
    """Verify-window width baked into captured decode-graph buffers.

    Graphs only record decode, so there is no forward mode to consult;
    capture and replay must agree on this width exactly.
    """
    if spec_num_tokens > 1 and not is_draft:
        return spec_num_tokens
    return 1


def mla_decode_out_cache_loc(
    table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    page_size: int,
    batch_size: int,
    validate_pages: bool = False,
    out: torch.Tensor | None = None,
    q_len_per_req: int = 1,
) -> torch.Tensor:
    """Absolute latent write locations for decoded tokens.

    Plain decode writes one location per request (position ``seq-1``).
    Speculative target verify decodes ``q_len_per_req`` tokens per request
    and must write every one of them, at the trailing positions
    ``seq-q_len .. seq-1``, flattened request-major to match the query
    layout the verify read path builds.
    """
    locations = mla_write_locations(
        seq_lens,
        table,
        page_size=page_size,
        q_len_per_req=q_len_per_req,
        batch_size=batch_size,
        out=out,
    )
    if validate_pages and locations.numel():
        # Page 0 is the null page, so a write there lands below one page.
        if not bool((locations >= page_size).all().item()):
            raise RuntimeError(
                "MLA write location resolves to the null page 0 or a " "-1 table hole"
            )
    return locations


def mla_extend_out_cache_loc(
    table: torch.Tensor,
    extend_prefix_lens_cpu: torch.Tensor,
    extend_seq_lens_cpu: torch.Tensor,
    *,
    page_size: int,
    validate_pages: bool = False,
) -> torch.Tensor:
    """Packed latent extend-write locations in query order."""
    chunks: list[torch.Tensor] = []
    pages_for_validation: list[torch.Tensor] = []
    for row, (start, num_new) in enumerate(
        zip(
            extend_prefix_lens_cpu.tolist(),
            extend_seq_lens_cpu.tolist(),
            strict=True,
        )
    ):
        start, num_new = int(start), int(num_new)
        if num_new <= 0:
            continue
        max_column = (start + num_new - 1) // page_size
        if max_column >= table.shape[1]:
            raise RuntimeError(
                "extend write locations exceed the full-attention "
                f"table: row={row}, prefix={start}, new={num_new}, "
                f"page_size={page_size}, columns={table.shape[1]}"
            )
        positions = torch.arange(
            start, start + num_new, dtype=torch.int64, device=table.device
        )
        pages = table[row].gather(0, positions // page_size)
        pages_for_validation.append(pages)
        chunks.append(pages.to(torch.int64) * page_size + positions % page_size)
    if not chunks:
        return torch.empty(0, dtype=torch.int64, device=table.device)
    if validate_pages and not bool((torch.cat(pages_for_validation) > 0).all().item()):
        raise RuntimeError(
            "MLA write location resolves to the null page 0 or a -1 table hole"
        )
    return torch.cat(chunks)


def mla_per_token_slot_table(
    table: torch.Tensor,
    *,
    batch_size: int,
    page_size: int,
    max_context_len: int,
) -> torch.Tensor:
    """Per-token absolute latent slots from a kernel-page table.

    flashinfer's paged prefill (``plan(page_size=1)``) reads a
    ``[bs, max_context]`` table indexed per token: slot(req, t) =
    ``table[req, t // p] * p + t % p``. Columns past a request's live range
    resolve through the table's null pages and are never read (the kernel
    walks only ``seq_len`` tokens per request).
    """
    table = table[:batch_size]
    num_columns = table.shape[1]
    columns = torch.arange(max_context_len, device=table.device)
    page_index = torch.div(columns, page_size, rounding_mode="floor").clamp_max(
        num_columns - 1
    )
    offset = columns % page_size
    pages = table[:, page_index].clamp_min(0).to(torch.int64)
    return pages * page_size + offset
