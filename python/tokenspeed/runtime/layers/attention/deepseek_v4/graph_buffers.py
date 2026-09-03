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

"""DeepSeek V4 persistent CUDA-graph decode buffers, as one composed object.

A captured decode graph holds the addresses of every tensor its kernels read
forever, so the buffers backing V4's decode metadata must be allocated once
and refreshed in place (docs/design/unified_path.md, "Pointer-stable per-bs
views"). This object owns that storage and the per-``(bs, tokens_per_req)``
metadata views over it — one keyed builder shared by capture, replay refresh
and the above-ladder eager path, so the three arms cannot bind different
objects (the class of bug this extraction exists to prevent).

Deliberately storage-only: cache-group *contract* validation (which groups a
capture placeholder may name, live-page geometry checks) stays on the
backend; this object answers "where do the rows live", never "are they
legal".
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.layers.attention.deepseek_v4.metadata import (
    DeepseekV4ForwardMetadata,
)
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_deepseek_v4 import (
    DeepseekV4CacheMetadata,
)


class DeepseekV4GraphBuffers:
    """Storage plus per-shape views for V4's unified decode path.

    Attributes:
        max_bs: Row capacity — the runner's max decode bs, never the capture
            ladder (docs/design/unified_path.md, "Buffer sizing").
        max_tokens_per_req: Widest packed verify width (spec window).
        page_table: ``[max_bs, max_num_pages]`` base full-history table.
        seq_lens / query_lens: ``[max_bs]`` row state.
        query_start_loc: ``[max_bs + 1]`` cumulative query offsets.
        token_to_req: ``[max_bs * max_tokens_per_req]`` token-major owner map.
        is_valid_token: same shape; the CUDA-graph padding mask
            (True = live token).
        query_start_by_width / token_to_req_by_width: per-width constant
            tables, precomputed so a packed refresh is copy-only.
        block_tables: per-group persistent tables (allocated by the
            backend's contract configuration through
            :meth:`allocate_group_tables`).
    """

    def __init__(
        self,
        *,
        max_bs: int,
        max_tokens_per_req: int,
        max_num_pages: int,
        device: torch.device | str,
    ) -> None:
        self.max_bs = max_bs
        self.max_tokens_per_req = max(1, int(max_tokens_per_req))
        max_tokens = max_bs * self.max_tokens_per_req
        self.device = device
        self.page_table = torch.zeros(
            (max_bs, max_num_pages), dtype=torch.int32, device=device
        )
        self.seq_lens = torch.ones((max_bs,), dtype=torch.int32, device=device)
        self.query_lens = torch.ones((max_bs,), dtype=torch.int32, device=device)
        self.query_start_loc = torch.arange(
            max_bs + 1, dtype=torch.int32, device=device
        )
        self.token_to_req = torch.arange(max_tokens, dtype=torch.int32, device=device)
        self.is_valid_token = torch.ones(max_tokens, dtype=torch.bool, device=device)
        query_start_base = torch.arange(max_bs + 1, dtype=torch.int32, device=device)
        token_to_req_base = torch.arange(max_bs, dtype=torch.int32, device=device)
        self.query_start_by_width: dict[int, torch.Tensor] = {}
        self.token_to_req_by_width: dict[int, torch.Tensor] = {}
        for width in range(1, self.max_tokens_per_req + 1):
            self.query_start_by_width[width] = query_start_base * width
            self.token_to_req_by_width[width] = token_to_req_base.repeat_interleave(
                width
            )
        self.block_tables: dict[str, torch.Tensor] = {}
        # Per-(bs, tokens_per_req) metadata views — the single builder's cache.
        self._views: dict[tuple[int, int], DeepseekV4ForwardMetadata] = {}

    def allocate_group_tables(self, group_widths: dict[str, int]) -> None:
        """Allocate the per-group persistent tables from the backend-resolved
        widths (the backend owns the contract math; this owns the storage)."""
        for gid, max_pages in group_widths.items():
            self.block_tables[gid] = torch.zeros(
                (self.max_bs, max_pages), dtype=torch.int32, device=self.device
            )
        # Views slice the group tables at construction; any built over the
        # previous set must rebuild.
        self._views.clear()

    # ------------------------------------------------------------------
    # The single per-shape view builder
    # ------------------------------------------------------------------

    def views(
        self,
        bs: int,
        tokens_per_req: int,
        *,
        kernel_page_size: int,
        max_num_pages: int,
        forward_mode,
    ) -> DeepseekV4ForwardMetadata:
        """The pointer-stable metadata views for ``(bs, tokens_per_req)``.

        Built once per shape and cached; every tensor field — the ``cache``
        slot's group tables included — is a fixed slice of the persistent
        buffers, assigned exactly at construction. Capture, replay refresh and
        lazy eager building all receive the same object and only fill the
        buffers underneath, so a refresh can never bind storage the captured
        graph did not record. Callers own the per-round ``forward_mode``
        update.
        """
        total_tokens = bs * tokens_per_req
        metadata = self._views.get((bs, tokens_per_req))
        if metadata is not None:
            metadata.forward_mode = forward_mode
            return metadata
        metadata = DeepseekV4ForwardMetadata(
            seq_lens=self.seq_lens[:bs],
            query_lens=self.query_lens[:bs],
            query_start_loc=self.query_start_loc[: bs + 1],
            token_to_req_indices=self.token_to_req[:total_tokens],
            cache=DeepseekV4CacheMetadata.from_group_tables(
                page_size=kernel_page_size,
                page_table=self.page_table[:bs, :max_num_pages],
                block_tables={gid: buf[:bs] for gid, buf in self.block_tables.items()},
            ),
            is_valid_token=self.is_valid_token[:total_tokens],
            seq_lens_cpu=None,
            query_lens_cpu=None,
            forward_mode=forward_mode,
        )
        self._views[(bs, tokens_per_req)] = metadata
        return metadata

    # ------------------------------------------------------------------
    # In-place refreshes
    # ------------------------------------------------------------------

    def tokens_per_req(self, bs: int, num_tokens: int) -> int:
        """Packed verify width for this refresh; validates uniform packing."""
        if num_tokens != bs:
            if bs == 0:
                return self.max_tokens_per_req
            if num_tokens % bs != 0:
                raise RuntimeError(
                    "DeepSeek V4 packed CUDA graph metadata expects uniformly "
                    f"packed tokens per request, got num_tokens={num_tokens}, "
                    f"bs={bs}"
                )
            width = num_tokens // bs
            if width > self.max_tokens_per_req:
                raise RuntimeError(
                    "DeepSeek V4 packed CUDA graph metadata was initialized "
                    f"for at most {self.max_tokens_per_req} tokens "
                    f"per request, got {width}"
                )
            return max(1, width)
        return 1

    def write_batch(self, bs: int, seq_lens: torch.Tensor) -> None:
        """Copy this round's row lengths into the persistent buffer."""
        self.seq_lens[:bs].copy_(seq_lens[:bs].to(torch.int32))

    def fill_packed_rows(self, *, bs: int, actual_bs: int, tokens_per_req: int) -> int:
        """Refresh the packed row machinery (query starts, owner map, padding
        mask) from the precomputed per-width tables. Returns total tokens."""
        total_tokens = bs * tokens_per_req
        actual_tokens = actual_bs * tokens_per_req
        query_start = self.query_start_by_width.get(tokens_per_req)
        token_to_req = self.token_to_req_by_width.get(tokens_per_req)
        if query_start is None or token_to_req is None:
            raise RuntimeError(
                "DeepSeek V4 CUDA graph packed metadata was not precomputed "
                f"for tokens_per_req={tokens_per_req}"
            )
        self.query_lens[:bs].fill_(tokens_per_req)
        self.query_start_loc[: bs + 1].copy_(query_start[: bs + 1])
        self.token_to_req[:total_tokens].copy_(token_to_req[:total_tokens])
        self.is_valid_token[:actual_tokens].fill_(True)
        if actual_tokens < total_tokens:
            self.is_valid_token[actual_tokens:total_tokens].fill_(False)
        return total_tokens

    def refresh_block_tables(
        self,
        bs: int,
        block_tables: dict[str, torch.Tensor],
        *,
        actual_bs: int | None = None,
        pad_value: int = 0,
    ) -> None:
        """Refresh the persistent per-group tables in place (the views built
        over them see the new rows). Missing groups keep the pad fill."""
        active_rows = bs if actual_bs is None else actual_bs
        for group_id, buf in self.block_tables.items():
            table = block_tables.get(group_id)
            buf[:bs].fill_(pad_value)
            if table is None:
                continue
            if int(table.shape[0]) < active_rows:
                raise RuntimeError(
                    "DeepSeek V4 CUDA graph cache-group table row count "
                    f"mismatch for {group_id!r}: got {int(table.shape[0])}, "
                    f"expected at least actual_bs {active_rows}"
                )
            cols = int(table.shape[1])
            if cols > int(buf.shape[1]):
                raise RuntimeError(
                    "DeepSeek V4 CUDA graph cache-group table width "
                    f"mismatch for {group_id!r}: got {cols}, capture "
                    f"buffer has {int(buf.shape[1])}"
                )
            if active_rows > 0 and cols > 0:
                buf[:active_rows, :cols].copy_(
                    table[:active_rows, :cols].to(torch.int32)
                )
