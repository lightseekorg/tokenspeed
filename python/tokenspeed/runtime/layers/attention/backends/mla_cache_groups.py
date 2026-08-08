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

Every MLA backend that consumes the paged-cache full-attention table needs the
same primitives: resolve the scheduler's full-history table, expand its logical
(scheduler) pages into the backend's kernel pages, and turn per-request sequence
lengths into absolute latent write locations. This mixin holds that logic so
``MLAAttnBackend``, ``FlashMLABackend`` and ``TRTLLMMLABackend`` share one
implementation rather than three copies.

Host-class requirements: ``self.page_size`` (kernel page size in tokens),
``self.max_num_pages`` (kernel page-table width), and ``self.device``. The host
must also define ``self._cache_contract_bound`` / ``self._cache_groups_bound``
(the mixin's :meth:`mark_cache_contract` sets the former).
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.layers.attention.page_table import expand_page_table


class MlaCacheGroupMixin:
    """Full-history table resolution + latent write-location math for MLA."""

    # MLA backends consume only the history (full-attention) cache family.
    cache_consumer_families = frozenset({"history"})

    # A draft MLA backend always reads the batch-ordered draft page table that
    # DraftPageStaging publishes (single history group, already in kernel
    # pages), never the wrapper's per-group table dispatch. This flag tells the
    # CUDA-graph wrapper to skip that dispatch for MLA drafts.
    reads_staged_draft_page_table = True

    def mark_cache_contract(self, logical_page_size: int | None = None) -> None:
        """Flag this backend as an LCM cache-group contract sub-backend.

        Called by the registry before graph-state allocation. Eager forwards
        bind the group tables automatically once cache metadata arrives; this
        flag lets CUDA-graph capture size its per-group write-location buffer up
        front. ``logical_page_size`` is accepted for call-site uniformity with
        other backends but unused: every MLA draft reads the batch-ordered draft
        page table published by ``DraftPageStaging`` (already in kernel pages),
        so no logical size ever needs to be recorded here.
        """
        del logical_page_size
        self._cache_contract_bound = True

    def _draft_reads_batch_pages(self, bs: int, forward_mode) -> bool:
        """True when this draft reads the published draft page table directly.

        A contract-bound MLA draft is always driven this way: the drafter runs
        without cache_metadata, and ``ModelExecutor`` publishes the target's
        full-history table into the batch-ordered draft page table (row i ==
        batch position i), expanding scheduler pages into draft kernel pages
        exactly once at publish time. The backend then reads those ids as-is
        (identity expand); it never needs the scheduler's logical page size.
        """
        return (
            self.is_draft
            and self._cache_contract_bound
            and bs > 0
            and not forward_mode.is_idle()
        )

    def _resolve_full_history_table(
        self, cache_metadata, forward_batch, bs: int
    ) -> tuple[torch.Tensor, int]:
        table = cache_metadata.require_full_attention_table(
            active_forward_op=forward_batch
        )
        if table.shape[0] < bs:
            raise RuntimeError(
                f"full-attention table has {table.shape[0]} rows but the "
                f"batch has {bs} requests"
            )
        logical_page_size = int(cache_metadata.block_size)
        if logical_page_size <= 0 or logical_page_size % self.page_size:
            raise RuntimeError(
                f"logical page size {logical_page_size} is not a positive multiple "
                f"of the MLA kernel page size {self.page_size}"
            )
        if table.stride(0) != table.shape[1] and table.shape[0] > 1:
            table = table.contiguous()
        return table, logical_page_size

    @staticmethod
    def _validate_live_pages(
        table: torch.Tensor, seq_lens: torch.Tensor, logical_page_size: int
    ) -> None:
        """Reject null or missing Paged cache pages inside each request's live range."""
        if table.numel() == 0 or seq_lens.numel() == 0:
            return
        batch_size = seq_lens.shape[0]
        live_pages = (
            (seq_lens.to(torch.int64) + logical_page_size - 1) // logical_page_size
        ).clamp_max_(table.shape[1])
        columns = torch.arange(table.shape[1], device=table.device)
        live_entries = table[:batch_size][
            columns.unsqueeze(0) < live_pages.unsqueeze(1)
        ]
        if not bool((live_entries > 0).all().item()):
            raise RuntimeError(
                "full-attention table contains -1 or the null page 0 "
                "inside a live range"
            )

    def _expand_group_page_table(
        self,
        table: torch.Tensor,
        *,
        batch_size: int,
        logical_page_size: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Expand scheduler pages for this backend's MLA kernel pages."""
        return expand_page_table(
            table[:batch_size],
            logical_page_size=logical_page_size,
            kernel_page_size=self.page_size,
            max_kernel_pages=self.max_num_pages,
            out=out,
        )

    @staticmethod
    def _group_per_token_slot_table(
        table: torch.Tensor,
        *,
        batch_size: int,
        logical_page_size: int,
        max_context_len: int,
    ) -> torch.Tensor:
        """Per-token absolute latent slots from a logical full-history table.

        flashinfer's paged prefill (``plan(page_size=1)``) reads a
        ``[bs, max_context]`` table indexed per token: slot(req, t) =
        ``table[req, t // P] * P + t % P``. Columns past a request's live range
        resolve through the table's null/-1 pages and are never read (the kernel
        walks only ``seq_len`` tokens per request).
        """
        table = table[:batch_size]
        num_columns = table.shape[1]
        columns = torch.arange(max_context_len, device=table.device)
        page_index = torch.div(
            columns, logical_page_size, rounding_mode="floor"
        ).clamp_max(num_columns - 1)
        offset = columns % logical_page_size
        pages = table[:, page_index].clamp_min(0).to(torch.int64)
        return pages * logical_page_size + offset

    @staticmethod
    def _cache_decode_out_cache_loc(
        table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        batch_size: int,
        logical_page_size: int,
        validate_pages: bool = False,
        out: torch.Tensor | None = None,
        q_len_per_req: int = 1,
    ) -> torch.Tensor:
        """Absolute latent write locations for decoded tokens in Paged cache.

        Plain decode writes one location per request (position ``seq-1``).
        Speculative target verify decodes ``q_len_per_req`` tokens per request
        and must write every one of them, at the trailing positions
        ``seq-q_len .. seq-1``, flattened request-major to match the query
        layout the verify read path builds.
        """
        last = (seq_lens[:batch_size].to(torch.int64) - 1).clamp_min(0)
        if q_len_per_req == 1:
            positions = last.unsqueeze(1)
        else:
            steps = torch.arange(
                1 - q_len_per_req, 1, device=seq_lens.device, dtype=torch.int64
            )
            positions = (last.unsqueeze(1) + steps).clamp_min(0)
        page_indices = torch.div(positions, logical_page_size, rounding_mode="floor")
        pages = table[:batch_size].gather(1, page_indices)
        if validate_pages and pages.numel() and not bool((pages > 0).all().item()):
            raise RuntimeError(
                "MLA write location resolves to the null page 0 or a " "-1 table hole"
            )
        locations = (
            pages.clamp_min(0).to(torch.int64) * logical_page_size
            + (positions % logical_page_size)
        ).reshape(-1)
        if out is not None:
            out[: batch_size * q_len_per_req].copy_(locations)
            return out
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

    @staticmethod
    def _extend_out_cache_loc(
        table: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        *,
        logical_page_size: int,
        validate_pages: bool = False,
    ) -> torch.Tensor:
        """Return packed Paged cache extend-write locations in query order."""
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
            max_column = (start + num_new - 1) // logical_page_size
            if max_column >= table.shape[1]:
                raise RuntimeError(
                    "extend write locations exceed the full-attention "
                    f"table: row={row}, prefix={start}, new={num_new}, "
                    f"logical_page_size={logical_page_size}, columns={table.shape[1]}"
                )
            positions = torch.arange(
                start, start + num_new, dtype=torch.int64, device=table.device
            )
            pages = table[row].gather(0, positions // logical_page_size)
            pages_for_validation.append(pages)
            chunks.append(
                pages.to(torch.int64) * logical_page_size
                + positions % logical_page_size
            )
        if not chunks:
            return torch.empty(0, dtype=torch.int64, device=table.device)
        if validate_pages and not bool(
            (torch.cat(pages_for_validation) > 0).all().item()
        ):
            raise RuntimeError(
                "MLA write location resolves to the null page 0 or a " "-1 table hole"
            )
        return torch.cat(chunks)
