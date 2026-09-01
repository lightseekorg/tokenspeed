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

"""Persistent per-group CUDA-graph table/write-location buffers.

One composed object owns the stacked buffers a captured decode graph reads:
attention-consumed groups get views into ONE stacked table/loc pair
(``[G, max_bs, Wmax]`` / ``[G, max_bs * spec_num_tokens]``) so the
replay-time write-loc math ALWAYS runs as a single fused triton launch over
all groups — the per-group python chains (~4 tiny elementwise launches per
group per step, the nsys inter-step band) are gone, on the spec-verify path
too.

Constructed once by ``AttentionBackend._init_group_graph_buffers`` (geometry
is frozen by then — the registry binds the pool before graph-state
allocation); capture records views of these buffers, replay refreshes them
in place at the same addresses.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.kvcache.triton import (
    compute_group_decode_locs,
    unpack_group_tables,
)

from tokenspeed.runtime.layers.attention.backends.cache_group_geometry import (
    CacheGroupGeometry,
)
from tokenspeed.runtime.layers.attention.page_table import expand_page_table
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.common import ceil_div

logger = get_colorful_logger(__name__)


class GroupGraphBuffers:
    """The stacked per-group graph buffers and their capture/replay ops.

    Attributes:
        page_tables: ``group_id -> [max_bs, width]`` consumer-page-grain
            views into the stacked table buffer — attention-consumed (paged
            KV) groups only, the one place page vocabulary applies here.
        owned_block_tables: ``group_id -> [max_bs, width]`` block-granularity
            views for the wrapper-owned (Inkling conv) groups riding the
            stack tail; the packed unpack fills them, no location views.
        out_cache_locs: ``group_id -> [max_bs * spec_n]`` views into the
            stacked location buffer (attention groups only).
    """

    def __init__(
        self,
        geometry: CacheGroupGeometry,
        *,
        consumed_group_ids: frozenset[str] | None = None,
        engine_owned_group_ids: frozenset[str],
        consumer_page_size_of,
        max_bs: int,
        max_num_pages: int,
        kernel_page_size: int,
        spec_num_tokens: int,
        device,
    ) -> None:
        """Size and allocate the stacks from the frozen geometry.

        Args:
            geometry: The pool's learned group geometry.
            consumed_group_ids: The groups this backend consumes — its
                positive claim (``AttentionBackend._consumed_group_ids``);
                they form the stack's attention prefix. ``None`` claims
                every row-geometry group that is not wrapper-owned.
            engine_owned_group_ids: Wrapper-owned (Inkling conv) groups —
                their tables ride in the stack tail but get no location
                views (the wrapper keeps its own write-loc machinery).
            consumer_page_size_of: ``group_id -> page size`` callable giving
                the size this backend VIEWS the group's cache with (mha:
                the group's own granularity; trtllm/msa: the kernel page
                size).
            max_bs: Persistent-buffer row capacity (max decode bs).
            max_num_pages: Kernel page-table width at kernel_page_size.
            kernel_page_size: The backend's kernel page size in tokens.
            spec_num_tokens: Verify width; sizes the location stack rows.
            device: Buffer device.
        """
        self.page_tables: dict[str, torch.Tensor] = {}
        self.owned_block_tables: dict[str, torch.Tensor] = {}
        self.out_cache_locs: dict[str, torch.Tensor] = {}
        self._max_bs = max_bs
        self._locs_stack = None
        self._tables_stack = None
        self._source_widths: dict[str, int] = {}
        self._grain_ratios: tuple[int, ...] = ()
        self._group_ids: list[str] = []
        self._attention_group_count = 0
        self._geometry = geometry
        self._kernel_page_size = kernel_page_size
        self._engine_owned_group_ids = engine_owned_group_ids

        if consumed_group_ids is None:
            consumed_group_ids = (
                frozenset(geometry.granularities) - engine_owned_group_ids
            )
        att_gids = sorted(
            gid for gid in geometry.granularities if gid in consumed_group_ids
        )
        owned_gids = sorted(
            gid for gid in geometry.granularities if gid in engine_owned_group_ids
        )
        gids = att_gids + owned_gids  # attention prefix, wrapper-owned tail
        if not gids:
            return
        granularity = geometry.granularity_of
        source_widths = {
            gid: ceil_div(max_num_pages * kernel_page_size, granularity(gid))
            for gid in gids
        }
        consumer_page_sizes = {gid: consumer_page_size_of(gid) for gid in att_gids}
        ratios = {
            gid: (
                granularity(gid) // consumer_page_sizes[gid] if gid in att_gids else 1
            )
            for gid in gids
        }
        if any(
            ratio <= 0 or granularity(gid) % consumer_page_sizes[gid]
            for gid, ratio in ratios.items()
            if gid in att_gids
        ):
            raise ValueError(
                "cache group page sizes must be positive multiples of their "
                "consumer page sizes"
            )
        widths = {gid: source_widths[gid] * ratios[gid] for gid in gids}
        logger.debug(
            "cache graph buffers: max_num_pages=%d page_size=%d max_bs=%d "
            "source_widths=%s widths=%s",
            max_num_pages,
            kernel_page_size,
            max_bs,
            source_widths,
            widths,
        )
        wmax = max(widths.values())
        g = len(gids)
        self._group_ids = gids
        self._attention_group_count = len(att_gids)
        self._tables_stack = torch.zeros(
            (g, max_bs, wmax), dtype=torch.int32, device=device
        )
        # Spec verify: graphs read [max_bs*N] loc views of the stack, so size it up front
        spec_n = max(int(spec_num_tokens or 1), 1)
        self._locs_stack = torch.zeros(
            (len(att_gids), max_bs * spec_n), dtype=torch.int32, device=device
        )
        self._source_widths = source_widths
        self._grain_ratios = tuple(ratios[gid] for gid in gids)
        self._consumer_page_sizes_tensor = torch.tensor(
            [consumer_page_sizes[gid] for gid in att_gids],
            dtype=torch.int32,
            device=device,
        )
        self._unpack_metadata_device = torch.zeros(
            (g, 3), dtype=torch.int32, device=device
        )
        for i, gid in enumerate(gids):
            view = self._tables_stack[i, :, : widths[gid]]
            if i < len(att_gids):
                self.page_tables[gid] = view
                self.out_cache_locs[gid] = self._locs_stack[i]
            else:
                self.owned_block_tables[gid] = view

    def capture_views(
        self,
        bs: int,
        cache_group_ids,
        tokens_per_req: int = 1,
        *,
        consumed_group_ids: frozenset[str] | None = None,
    ):
        """Capture-time (page_tables, out_cache_locs) per-group views.

        Real tables only arrive at replay, which copies fresh data to these
        graph-recorded addresses. Verify (tokens_per_req = spec_num_tokens)
        keeps [bs]-row tables but records [bs*N] write-loc views (token-major,
        single-table verify layout). ``consumed_group_ids`` is the backend's
        positive claim; delivered groups outside it ride to their own
        consumers (state tables ride to the mamba backend, engine-owned
        conv groups keep the wrapper's own capture buffers). Defaults to
        the construction-time attention prefix. Returns (None, None) when
        no consumed group (or none at all) is delivered.
        """
        if not cache_group_ids:
            return None, None
        if consumed_group_ids is None:
            consumed_group_ids = frozenset(
                self._group_ids[: self._attention_group_count]
            )
        page_tables = {}
        out_cache_locs = {}
        for gid in cache_group_ids:
            if gid not in consumed_group_ids:
                continue
            buf = self.page_tables.get(gid)
            if buf is None:
                # Replay write locs are ALWAYS the fused triton launch over
                # the stacked buffers; a group without an attention view could
                # never get its locs filled. Every capture-visible group must
                # be known (set_cache_pool) before init_cuda_graph_state.
                raise RuntimeError(
                    f"cache group {gid!r} has no attention view in the "
                    "stacked CUDA-graph buffers (attention stack: "
                    f"{self._group_ids[: self._attention_group_count]}); "
                    "declare every capture-visible group's page size before "
                    "graph init."
                )
            loc_buf = self.out_cache_locs.get(gid)
            need = self._max_bs * tokens_per_req
            if loc_buf is None or loc_buf.shape[0] < need:
                raise RuntimeError(
                    f"location stack too small for group {gid!r}: capture "
                    f"needs {need} rows, have "
                    f"{0 if loc_buf is None else loc_buf.shape[0]}; the "
                    "stack is sized max_bs * spec_num_tokens at init, so "
                    f"tokens_per_req={tokens_per_req} must not exceed "
                    "spec_num_tokens."
                )
            page_tables[gid] = buf[:bs, :]
            out_cache_locs[gid] = loc_buf[: bs * tokens_per_req]
        if not page_tables:
            # Only foreign groups delivered: nothing for this backend.
            return None, None
        return page_tables, out_cache_locs

    def _try_packed_unpack(self, bs: int, block_tables, tail_pad: int) -> bool:
        """One-launch fill of the stacked graph tables from the bridge's
        packed upload. Requires the stack to cover every delivered
        non-state group and all sources to share one storage (the packed
        bridge guarantees both); returns False to take the per-group
        fallback otherwise."""
        stack = self._tables_stack
        if stack is None or stack.device.type != "cuda":
            # CPU unit tests take the per-group torch fallback.
            return False
        gids = self._group_ids
        # Fresh pinned alloc each step: a persistent pinned buffer would race with overlap scheduling
        meta = torch.empty((len(gids), 3), dtype=torch.int32, pin_memory=True)
        base_ptr = None
        actual = None
        for i, gid in enumerate(gids):
            src = block_tables.get(gid)
            if src is None or src.shape[1] == 0:
                return False
            ptr = src.untyped_storage().data_ptr()
            if base_ptr is None:
                base_ptr = ptr
                actual = src.shape[0]
            elif ptr != base_ptr or src.shape[0] != actual:
                return False
            meta[i, 0] = src.storage_offset()
            meta[i, 1] = src.shape[1]
            meta[i, 2] = self._grain_ratios[i]
        if base_ptr is None:
            return False
        self._unpack_metadata_device.copy_(meta, non_blocking=True)
        src0 = block_tables[gids[0]]
        packed = torch.as_strided(
            src0,
            (src0.untyped_storage().nbytes() // 4,),
            (1,),
            storage_offset=0,
        )
        unpack_group_tables(
            packed,
            self._unpack_metadata_device,
            stack,
            bs,
            actual_bs=min(actual, bs),
            tail_pad=tail_pad,
        )
        return True

    def fill(
        self,
        bs: int,
        block_tables,
        seq_lens,
        *,
        tokens_per_req: int = 1,
        tail_pad: int = -1,
        engine_owned_group_ids: frozenset[str] | None = None,
    ) -> bool:
        """Copy this replay's tables into the captured buffers and recompute
        the per-group write locs from the live seq_lens (tokens_per_req locs
        per request on the spec-verify path).

        Padding contract (canonical; bs is the padded bs): dummy ROWS pad
        with 0 — replayed at seq_lens=1 they dereference exactly col 0,
        the zero-init dummy page. Column tails pad with ``tail_pad``, never
        read past cache_seqlens.

        Returns:
            Whether the packed one-launch unpack path ran (telemetry).
        """
        if self._locs_stack is None or self._locs_stack.shape[1] < bs * tokens_per_req:
            raise RuntimeError(
                "replay write locations need the stacked location buffer "
                f"(bs={bs}, tokens_per_req={tokens_per_req}, stack="
                f"{None if self._locs_stack is None else tuple(self._locs_stack.shape)}); "
                "the stack is sized max_bs * spec_num_tokens at graph init "
                "and there is no python fallback."
            )
        # The wrapper may register its owned (conv) groups after buffer
        # construction; honor the live set when the caller passes one.
        owned = (
            engine_owned_group_ids
            if engine_owned_group_ids is not None
            else self._engine_owned_group_ids
        )
        packed_ran = self._try_packed_unpack(bs, block_tables, tail_pad)
        if not packed_ran:
            for i, gid in enumerate(self._group_ids):
                src = block_tables.get(gid)
                if src is None:
                    continue
                if gid in owned:
                    # The wrapper fills its own block-granularity tables.
                    continue
                buf = self.page_tables.get(gid)
                if buf is None:
                    # Construction-owned but live-unowned: still a stack
                    # resident, copied plainly at block granularity (ratio 1).
                    buf = self.owned_block_tables[gid]
                # Clamp: scheduler may send extra reservation columns; kernels never read past cache_seqlens
                source_cols = min(src.shape[1], self._source_widths[gid])
                # cols >= 1: a zero-width table would leave dummy rows' col 0 unwritten
                assert source_cols >= 1, f"table for group {gid!r}: zero-width table"
                rows = min(src.shape[0], bs)
                ratio = self._grain_ratios[i]
                if ratio != 1:
                    expand_page_table(
                        src[:rows, :source_cols],
                        block_granularity=self._geometry.granularity_of(gid),
                        kernel_page_size=int(self._consumer_page_sizes_tensor[i]),
                        max_kernel_pages=buf.shape[1],
                        out=buf[:rows],
                    )
                    if rows < bs:
                        buf[rows:bs].zero_()
                    continue
                cols = min(source_cols, buf.shape[1])
                buf[:rows, :cols].copy_(src[:rows, :cols])
                if cols < buf.shape[1]:
                    buf[:rows, cols:].fill_(tail_pad)
                if rows < bs:
                    # Dummy rows pad with 0 (the zero-init dummy page).
                    buf[rows:bs].fill_(0)

        # One fused launch writes every group's locs into the stacked buffer the graphs read
        if self._locs_stack.device.type != "cuda":
            # CPU unit tests: same math in torch (triton needs a GPU).
            self._compute_decode_locs_torch(bs, seq_lens, tokens_per_req)
        else:
            compute_group_decode_locs(
                self._tables_stack[: self._attention_group_count],
                self._consumer_page_sizes_tensor,
                seq_lens[:bs],
                self._locs_stack,
                bs,
                tokens_per_req,
            )
        return packed_ran

    def _compute_decode_locs_torch(self, bs: int, seq_lens, tokens_per_req: int):
        n = tokens_per_req
        if n == 1:
            pos = (seq_lens[:bs].to(torch.int64) - 1).clamp_min(0)
        else:
            steps = torch.arange(n, device=seq_lens.device, dtype=torch.int64)
            pos = (
                (seq_lens[:bs].to(torch.int64).unsqueeze(1) - n + steps)
                .clamp_min(0)
                .reshape(-1)
            )
        for i in range(self._attention_group_count):
            ps = int(self._consumer_page_sizes_tensor[i])
            table = self._tables_stack[i, :bs]
            page_idx = pos // ps
            off = (pos % ps).to(torch.int32)
            if n == 1:
                pages = table.gather(1, page_idx.unsqueeze(1)).squeeze(1)
            else:
                pages = table.gather(1, page_idx.view(bs, n)).reshape(-1)
            self._locs_stack[i, : bs * n].copy_(pages.clamp_min(0) * ps + off)
