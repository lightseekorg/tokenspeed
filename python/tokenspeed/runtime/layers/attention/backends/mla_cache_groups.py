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

"""Shared cache-group (LCM full-history) helpers for MLA backends.

Every MLA backend consumes the cache-group full-attention table through the
same block-table route every other backend uses: the wrapper distributes the
scheduler's per-group ``block_tables`` (block-granularity page ids), and the
backend expands its history group's table into its own KERNEL pages. All the
location math here then uses ``self.kernel_page_size``. This mixin holds that
logic so ``MLAAttnBackend``, ``FlashMLABackend``, ``TRTLLMMLABackend`` and
``CuteDSLMLABackend`` share one implementation rather than four copies.

Host-class requirements: ``self.kernel_page_size`` (kernel page size in tokens),
``self.max_num_pages`` (kernel page-table width), and ``self.device``. The host
must also define ``self._cache_contract_bound`` / ``self._cache_groups_bound``
(the mixin's :meth:`mark_cache_contract` sets the former).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.triton.mla_write_locations import (
    mla_write_locations,
)

from tokenspeed.runtime.layers.attention.page_table import expand_page_table


class MlaCacheGroupMixin:
    """Full-history table resolution + latent write-location math for MLA."""

    # MLA backends consume only the history (full-attention) cache family.
    cache_consumer_families = frozenset({"history"})
    # The wrapper hands raw scheduler tables and this mixin expands them
    # (self-padding: the refresh nulls its own dummy rows, so the wrapper
    # must not F.pad).
    tables_self_padding = True

    def bind_decode_views(self, bs: int, cache_group_ids: tuple[str, ...] = ()) -> None:
        """Pre-build the per-bs views for the base default capture. The MLA
        group binding is a separate latch owned by the backends whose
        recorded write-loc branch reads it (see MLAAttnBackend)."""
        del cache_group_ids
        self._decode_views(bs)

    # Learned from the pool's published specs at set_cache_pool; None until a
    # pool binds (unit fixtures may run without one).
    _full_history_group_id: str | None = None
    _history_block_granularity: int | None = None

    def set_cache_pool(self, cache_pool) -> None:
        """Bind the pool and learn the full-history group's geometry.

        The same selection rule as ``ModelExecutor``: the first published
        group with ``family="history"`` and ``retention="full_history"``.
        """
        super().set_cache_pool(cache_pool)
        for spec in getattr(cache_pool.arena, "cache_group_specs", ()):
            if spec.family == "history" and spec.retention == "full_history":
                self._full_history_group_id = str(spec.group_id)
                self._history_block_granularity = int(spec.block_granularity)
                break

    def mark_cache_contract(self) -> None:
        """Flag this backend as an LCM cache-group contract sub-backend.

        Called by the registry before graph-state allocation. Eager forwards
        bind the group tables automatically once they arrive; this flag lets
        CUDA-graph capture size its per-group write-location buffer up front.
        """
        self._cache_contract_bound = True

    def _resolve_full_history_table(
        self, block_tables, bs: int, out: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """This forward's full-attention table in this backend's kernel pages.

        Expands the wrapper-delivered raw group table (block-granularity page
        ids, batch-ordered rows) into kernel pages of width
        ``self.max_num_pages``; -1 holes clamp into the null page 0's kernel
        range. Returns None when no table was delivered (warmup placeholders,
        idle before binding) — the caller falls back to ``page_table``.
        ``out`` writes the expansion into a persistent buffer (its rows past
        the raw table's are the caller's to null).
        """
        if not block_tables or self._full_history_group_id is None:
            return None
        raw = block_tables.get(self._full_history_group_id)
        if raw is None:
            return None
        if raw.shape[0] < bs:
            raise RuntimeError(
                f"full-attention table has {raw.shape[0]} rows but the "
                f"batch has {bs} requests"
            )
        return self._expand_history_table(raw, out=out)

    def _expand_history_table(
        self, raw: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Expand a batch-ordered raw table (scheduler pages) into this
        backend's kernel pages, ``self.max_num_pages`` wide. The staged draft
        page table and the wrapper's group tables share this one mapping."""
        return expand_page_table(
            raw,
            block_granularity=self._history_block_granularity or self.kernel_page_size,
            kernel_page_size=self.kernel_page_size,
            max_kernel_pages=self.max_num_pages,
            out=out,
        )

    @staticmethod
    def _group_per_token_slot_table(
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

    def _cache_decode_out_cache_loc(
        self,
        table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        batch_size: int,
        validate_pages: bool = False,
        out: torch.Tensor | None = None,
        q_len_per_req: int = 1,
    ) -> torch.Tensor:
        """Absolute latent write locations for decoded tokens in the full-attention group.

        Plain decode writes one location per request (position ``seq-1``).
        Speculative target verify decodes ``q_len_per_req`` tokens per request
        and must write every one of them, at the trailing positions
        ``seq-q_len .. seq-1``, flattened request-major to match the query
        layout the verify read path builds.
        """
        page_size = self.kernel_page_size
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
                    "MLA write location resolves to the null page 0 or a "
                    "-1 table hole"
                )
        return locations

    def _verify_q_len(self, forward_mode) -> int:
        """KV write locations each request needs this decode step.

        The target's verify decode writes the whole speculative window
        (``spec_num_tokens`` trailing positions); plain decode and any draft
        write a single location.
        """
        if self.spec_num_tokens <= 1:
            return 1
        if (
            not self.is_draft
            and forward_mode is not None
            and (forward_mode.is_decode() or forward_mode.is_mixed())
        ):
            return self.spec_num_tokens
        return 1

    def _graph_verify_q_len(self) -> int:
        """Verify-window width baked into captured decode-graph buffers.

        Graphs only record decode, so there is no forward mode to consult;
        capture and replay must agree on this width exactly.
        """
        if self.spec_num_tokens > 1 and not self.is_draft:
            return self.spec_num_tokens
        return 1

    def _extend_out_cache_loc(
        self,
        table: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        *,
        validate_pages: bool = False,
    ) -> torch.Tensor:
        """Return packed cache-group extend-write locations in query order."""
        page_size = self.kernel_page_size
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
        if validate_pages and not bool(
            (torch.cat(pages_for_validation) > 0).all().item()
        ):
            raise RuntimeError(
                "MLA write location resolves to the null page 0 or a " "-1 table hole"
            )
        return torch.cat(chunks)
