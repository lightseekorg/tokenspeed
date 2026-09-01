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

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel import (
    mla_decode_with_kvcache,
    mla_extend_with_kvcache,
    mla_prefill,
    mla_use_absorbed_extend,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.cache_group_geometry import (
    resolve_full_history_table,
)
from tokenspeed.runtime.layers.attention.backends.group_write_locations import (
    graph_verify_q_len,
    mla_decode_out_cache_loc,
    mla_extend_out_cache_loc,
)
from tokenspeed.runtime.layers.attention.chunk import (
    build_chunked_prefill_metadata_arrays,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    MLA_PAGE_SIZE,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    cache_debug_enabled,
)
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.attention.utils import build_page_table
from tokenspeed.runtime.utils.common import ceil_div

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


@dataclass(kw_only=True)
class MLAPrefillMetadata:
    # Device-side metadata for explicit Q/K/V MLA prefill and prefix replay.
    seq_lens: torch.Tensor
    req_pool_indices: torch.Tensor
    extend_prefix_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    cum_extend_seq_lens: torch.Tensor
    cum_seq_lens_kv: torch.Tensor | None
    page_table: torch.Tensor | None
    # Host-side metadata.
    extend_seq_lens_cpu: list[int]
    max_extend_seq_len: int
    max_extend_prefix_len: int
    use_absorbed_cached_extend: bool
    # Per-prefix-chunk arrays consumed by DeepSeek's chunked prefix replay.
    chunked_loop_num: int
    chunk_kv_indices_list: list[torch.Tensor]
    chunked_seq_len: torch.Tensor
    cu_chunked_seq_len: torch.Tensor
    max_chunk_len_per_loop: list[int]
    # Cache-group path only: absolute latent locations for model-owned extend writes.
    out_cache_loc: torch.Tensor | None = None


@dataclass(kw_only=True)
class MLADecodeMetadata:
    # num_extends lets mixed batches slice decode requests after extend requests.
    num_extends: int
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    # Cache-group path only: absolute latent write locations, request-major, with
    # ``q_len_per_req`` entries per batch row (1 outside target verify).
    out_cache_loc: torch.Tensor | None = None
    q_len_per_req: int = 1

    @property
    def seq_lens_k(self) -> torch.Tensor:
        return self.seq_lens


class MLAAttnBackend(AttentionBackend):
    """Unified MLA backend routed through tokenspeed_kernel MLA APIs."""

    # The refresh nulls its own dummy table rows, so the wrapper must pass
    # UNPADDED tables (no per-step F.pad).
    tables_self_padding = True

    supports_mla_projected_value_decode = True

    def __init__(self, config: AttnConfig, spec: MLAConfig):
        super().__init__(config, spec)

        self._cache_groups_bound = False
        self.max_context_len = config.context_len
        self.kernel_page_size = (
            config.kernel_page_size
            if config.kernel_page_size is not None
            else MLA_PAGE_SIZE
        )
        self.max_num_pages = ceil_div(self.max_context_len, self.kernel_page_size)

        self.kv_lora_rank = spec.kv_lora_rank
        self.qk_nope_head_dim = spec.qk_nope_head_dim
        self.qk_rope_head_dim = spec.qk_rope_head_dim
        self.v_head_dim = spec.v_head_dim
        self.kv_cache_dim = spec.kv_cache_dim
        self.scaling = spec.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype
        self.num_local_heads = spec.num_attention_heads // spec.attn_tp_size

        # DFLASH/DSpark draft: the drafter proposes a whole block in one decode
        # forward and needs the block to be non-causal. Rather than a mask, each
        # request expands into spec_num_tokens single-query rows sharing the
        # block-end seq_len, so every block query sees the whole block including
        # its own future. Mirrors the MHA/MSA/TRTLLM draft_block_decode path;
        # target verify and ordinary decode are untouched.
        self.draft_block_decode = bool(config.draft_block_decode)

        backend_name = spec.backend_name or "mla"
        self.kernel_solution = {"mla": None, "gluon": "gluon"}[backend_name]

        self.forward_decode_metadata: MLADecodeMetadata | None = None
        self.forward_prefill_metadata: MLAPrefillMetadata | None = None
        self.chunked_prefill_metadata: MLAPrefillMetadata | None = None
        self.decode_cuda_graph_metadata: dict[int, MLADecodeMetadata] = {}
        self.cuda_graph_page_table: torch.Tensor | None = None
        self.cuda_graph_seq_lens: torch.Tensor | None = None
        self.decode_cuda_graph_out_cache_loc: torch.Tensor | None = None

    def _should_use_absorbed_cached_extend(
        self, *, max_extend_seq_len: int, max_extend_prefix_len: int
    ) -> bool:
        return max_extend_prefix_len > 0 and mla_use_absorbed_extend(
            q_dtype=self.q_data_type,
            kv_dtype=self.data_type,
            num_q_heads=self.num_local_heads,
            page_size=self.kernel_page_size,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            max_seqlen_q=max_extend_seq_len,
            solution=self.kernel_solution,
        )

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        **kwargs,
    ):
        kwargs.pop("cache_metadata", None)
        kwargs.pop("forward_batch", None)
        full_history_table = resolve_full_history_table(
            kwargs.pop("block_tables", None),
            self._geometry,
            bs,
            kernel_page_size=self.kernel_page_size,
            max_kernel_pages=self.max_num_pages,
        )
        if full_history_table is not None:
            self._cache_groups_bound = True
        elif self.is_draft and bs > 0:
            # The drafter drives the draft backend directly with no group
            # tables; the staged batch-ordered draft page table (raw
            # scheduler pages) stands in, expanded like any full-history table.
            self._cache_groups_bound = True
        elif self._cache_groups_bound and bs > 0 and not forward_mode.is_idle():
            raise RuntimeError(
                "MLAAttnBackend is bound to cache groups but received no "
                "group tables; refusing the legacy page_table path"
            )

        if not (forward_mode.is_extend_or_mixed() or forward_mode.is_idle()):
            raise RuntimeError(
                "MLA decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend/mixed ({forward_mode})"
            )
        if forward_mode.is_extend_or_mixed():
            self._init_prefill_metadata(
                seq_lens=seq_lens[:num_extends],
                req_pool_indices=req_pool_indices[:num_extends],
                page_table=page_table,
                extend_prefix_lens=extend_prefix_lens[:num_extends],
                extend_prefix_lens_cpu=extend_prefix_lens_cpu[:num_extends],
                extend_seq_lens=extend_seq_lens[:num_extends],
                extend_seq_lens_cpu=extend_seq_lens_cpu[:num_extends],
                full_history_table=full_history_table,
            )

        # Target mixed/idle batches carry decode rows whose metadata this init
        # must cover (verify decodes q_len tokens per request, so the write
        # locations span every one of them). A draft's decode metadata instead
        # comes from the wrapper's refresh_decode_metadata after this init
        # (the unified draft contract).
        if forward_mode.is_idle() or (forward_mode.is_mixed() and not self.is_draft):
            self._init_decode_metadata(
                bs=bs,
                num_extends=num_extends,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                page_table=page_table,
                full_history_table=full_history_table,
                q_len_per_req=self._verify_q_len(forward_mode),
            )

    @contextmanager
    def override_num_extends(self, num_extends: int):
        assert self.forward_decode_metadata is not None
        prev = self.forward_decode_metadata.num_extends
        self.forward_decode_metadata.num_extends = num_extends
        try:
            yield
        finally:
            self.forward_decode_metadata.num_extends = prev

    def _init_prefill_metadata(
        self,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        page_table: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        full_history_table: torch.Tensor | None = None,
    ):
        extend_seq_lens_cpu_list = [int(x) for x in extend_seq_lens_cpu.tolist()]
        cum_extend_seq_lens = torch.zeros(
            extend_seq_lens.shape[0] + 1,
            device=self.device,
            dtype=torch.int32,
        )
        torch.cumsum(extend_seq_lens, dim=0, out=cum_extend_seq_lens[1:])

        max_extend_seq_len = max(extend_seq_lens_cpu_list, default=0)
        max_extend_prefix_len = int(extend_prefix_lens_cpu.max().item())
        use_absorbed_cached_extend = self._should_use_absorbed_cached_extend(
            max_extend_seq_len=max_extend_seq_len,
            max_extend_prefix_len=max_extend_prefix_len,
        )

        cum_seq_lens_kv = None
        absorbed_page_table = None
        if use_absorbed_cached_extend:
            cum_seq_lens_kv = torch.zeros_like(cum_extend_seq_lens)
            torch.cumsum(seq_lens, dim=0, out=cum_seq_lens_kv[1:])

        if full_history_table is not None:
            out_cache_loc = mla_extend_out_cache_loc(
                full_history_table[: seq_lens.shape[0]],
                extend_prefix_lens_cpu,
                extend_seq_lens_cpu,
                page_size=self.kernel_page_size,
                validate_pages=cache_debug_enabled(),
            )
            chunk_page_table = full_history_table[: seq_lens.shape[0]]
            chunk_req_pool_indices = torch.arange(
                seq_lens.shape[0], dtype=torch.int64, device=full_history_table.device
            )
            chunk_page_size = self.kernel_page_size
            if use_absorbed_cached_extend:
                absorbed_page_table = full_history_table[: seq_lens.shape[0]]
        else:
            # Idle/warmup placeholder: page_table is batch-ordered (row i ==
            # batch position i), so identity row indices apply.
            out_cache_loc = None
            chunk_page_table = page_table
            chunk_req_pool_indices = torch.arange(
                seq_lens.shape[0], dtype=torch.int64, device=page_table.device
            )
            chunk_page_size = self.kernel_page_size
            if use_absorbed_cached_extend:
                absorbed_page_table = build_page_table(
                    req_pool_indices,
                    page_table,
                    self.kernel_page_size,
                    self.max_context_len,
                )

        (
            chunked_loop_num,
            chunk_kv_indices_list,
            chunked_seq_len,
            cu_chunked_seq_len,
            max_chunk_len_per_loop,
        ) = build_chunked_prefill_metadata_arrays(
            extend_prefix_lens,
            extend_prefix_lens_cpu,
            chunk_page_table,
            chunk_req_pool_indices,
            chunk_page_size,
        )

        metadata = MLAPrefillMetadata(
            seq_lens=seq_lens,
            req_pool_indices=req_pool_indices,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            cum_extend_seq_lens=cum_extend_seq_lens,
            cum_seq_lens_kv=cum_seq_lens_kv,
            page_table=absorbed_page_table,
            extend_seq_lens_cpu=extend_seq_lens_cpu_list,
            max_extend_seq_len=max_extend_seq_len,
            max_extend_prefix_len=max_extend_prefix_len,
            use_absorbed_cached_extend=use_absorbed_cached_extend,
            chunked_loop_num=chunked_loop_num,
            chunk_kv_indices_list=chunk_kv_indices_list,
            chunked_seq_len=chunked_seq_len,
            cu_chunked_seq_len=cu_chunked_seq_len,
            max_chunk_len_per_loop=max_chunk_len_per_loop,
            out_cache_loc=out_cache_loc,
        )
        self.forward_prefill_metadata = metadata
        self.chunked_prefill_metadata = metadata

    def _verify_q_len(self, forward_mode: ForwardMode) -> int:
        """KV write locations each request needs this step.

        The target's verify decode writes a whole window, and so does a
        block-decode draft: it proposes the entire block in one forward, and its
        seq_lens already run to the block end, so the trailing-window positions
        are the block's own. A chaining draft owns its per-step locations, and
        prefill goes through the extend path.
        """
        if self.spec_num_tokens <= 1 or self._block_decode_active:
            return 1
        if not self.is_draft and (forward_mode.is_decode() or forward_mode.is_mixed()):
            return self.spec_num_tokens
        return 1

    def _init_decode_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        full_history_table: torch.Tensor | None = None,
        q_len_per_req: int = 1,
    ):
        if full_history_table is not None:
            page_table = full_history_table[:bs]
            out_cache_loc = mla_decode_out_cache_loc(
                full_history_table,
                seq_lens,
                page_size=self.kernel_page_size,
                batch_size=bs,
                validate_pages=cache_debug_enabled(),
                q_len_per_req=q_len_per_req,
            )
        elif self.is_draft:
            # The executor publishes draft rows in batch order and in raw
            # scheduler pages; expand into this backend's kernel pages. Do
            # not index them by request-pool id.
            page_table = self._expand_history_table(page_table[:bs])
            out_cache_loc = None
        else:
            page_table = build_page_table(
                req_pool_indices[:bs],
                page_table,
                self.kernel_page_size,
                self.max_context_len,
            )
            out_cache_loc = None
        if self._block_decode_active:
            page_table, block_seq_lens = self._expand_block_decode_metadata(
                page_table, seq_lens[:bs], bs
            )
            self.forward_decode_metadata = MLADecodeMetadata(
                # The drafter calls in with num_extends == bs (its rows are
                # "extend" by its own convention) but every one of them is a
                # block row this metadata describes, so nothing is skipped.
                # Carrying its value here would slice the whole block away.
                num_extends=0,
                page_table=page_table,
                seq_lens=block_seq_lens,
                # The drafter owns block write locations: it recomputes them
                # in-graph from the live draft length, which is not knowable
                # here. Only the read path takes the group page table.
                out_cache_loc=None,
                q_len_per_req=1,
            )
            return
        self.forward_decode_metadata = MLADecodeMetadata(
            num_extends=num_extends,
            page_table=page_table,
            seq_lens=seq_lens[:bs],
            out_cache_loc=out_cache_loc,
            q_len_per_req=q_len_per_req,
        )

    # ------------------------------------------------------------------
    # Non-causal block decode (DFLASH / DSpark draft)
    # ------------------------------------------------------------------

    @property
    def _block_decode_active(self) -> bool:
        return self.draft_block_decode and self.spec_num_tokens > 1

    def _expand_block_decode_metadata(
        self,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        bs: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One decode row per block position, all sharing the block-end length.

        The MLA decode kernel derives its mask from each row's ``cache_seqlens``.
        Giving all ``spec_num_tokens`` rows of a request the same length makes
        every block query attend over the whole block -- including positions
        after it -- which is exactly the block-diffusion draft's semantics.
        """
        spec = self.spec_num_tokens
        expanded_page_table = page_table.repeat_interleave(spec, dim=0)
        # Clamp so a request near the context limit cannot ask the kernel for
        # more page-table columns than exist (mirrors the MHA path, where the
        # unclamped block-end length caused out-of-bounds page reads).
        expanded_seq_lens = (
            seq_lens.clamp(spec, self.max_context_len)
            .repeat_interleave(spec)
            .contiguous()
        )
        return expanded_page_table, expanded_seq_lens

    def select_out_cache_loc(self, layer, out_cache_loc, forward_mode=None):
        # A draft owns its per-step write locations (computed from the staged
        # batch-ordered draft page table); only the target routes writes
        # through the table-derived location buffer in the metadata.
        if (
            not self._cache_groups_bound
            or forward_mode is None
            or forward_mode.is_idle()
            or self.is_draft
        ):
            return out_cache_loc
        if self._block_decode_active:
            # The drafter computes the block's locations from its page table and
            # rewrites them in-graph each replay; overriding them here would
            # pin every replay to the capture-time draft length.
            return out_cache_loc
        if forward_mode.is_decode():
            metadata = self.forward_decode_metadata
            if metadata is None or metadata.out_cache_loc is None:
                raise RuntimeError("MLA decode write locations are missing")
            # Locations are request-major with q_len_per_req entries per
            # row, so a mixed batch skips whole windows, not single rows.
            locs = metadata.out_cache_loc[
                metadata.num_extends * metadata.q_len_per_req :
            ]
        else:
            metadata = self.forward_prefill_metadata
            if metadata is None or metadata.out_cache_loc is None:
                raise RuntimeError("MLA prefill write locations are missing")
            locs = metadata.out_cache_loc
        if out_cache_loc is not None and locs.shape[0] != out_cache_loc.shape[0]:
            raise RuntimeError(
                f"MLA write locations cover {locs.shape[0]} tokens but "
                f"the caller provided {out_cache_loc.shape[0]}"
            )
        return locs

    def init_cuda_graph_state(self, max_bs: int, **kwargs):
        # Block decode records spec_num_tokens rows per request.
        graph_rows = max_bs * (self.spec_num_tokens if self._block_decode_active else 1)
        self.cuda_graph_page_table = torch.zeros(
            (graph_rows, self.max_num_pages), dtype=torch.int32, device=self.device
        )
        # Own the cache-seqlens buffer; replay copies the live lengths in, so
        # graph state does not depend on the controller mutating a shared tensor.
        self.cuda_graph_seq_lens = torch.zeros(
            graph_rows, dtype=torch.int32, device=self.device
        )
        self.decode_cuda_graph_metadata = {}
        if not self.is_draft:
            # Target verify records spec_num_tokens write locations per
            # request. A draft owns its per-step write locations and never
            # reads this buffer (every LCM pool publishes a cache contract,
            # so there is no separate contract gate).
            self.decode_cuda_graph_out_cache_loc = torch.zeros(
                max_bs * max(1, self.spec_num_tokens),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self.decode_cuda_graph_out_cache_loc = None

    # Capture is inherited (base default: bind_decode_views + idle refresh).

    def bind_decode_views(self, bs: int, cache_group_ids: tuple[str, ...] = ()) -> None:
        # Every LCM pool publishes cache groups, so capture always latches
        # the grouped path; the latch must be set BEFORE the views are built
        # (it selects the write-location view) and the recorded
        # select_out_cache_loc branch reads it. A draft binds too — its
        # refresh consumes the wrapper-dispatched full-history table — but its
        # write locations stay drafter-owned (see _decode_views).
        del cache_group_ids
        self._cache_groups_bound = True
        self._decode_views(bs)

    def _replay_block_decode_page_table(
        self,
        bs: int,
        page_table: torch.Tensor | None,
        block_tables,
    ) -> None:
        """Refresh the block rows' page table, broadcast from one row/request.

        Reads page ids the same way the eager path does: through the group
        tables when delivered, otherwise from the batch-ordered draft page
        table (raw scheduler pages, padded rows zeroed at publish time).
        """
        spec = self.spec_num_tokens
        width = self.max_num_pages
        rows = self.cuda_graph_page_table[: bs * spec, :width].view(bs, spec, width)
        table = resolve_full_history_table(
            block_tables,
            self._geometry,
            0,
            kernel_page_size=self.kernel_page_size,
            max_kernel_pages=self.max_num_pages,
        )
        if table is None:
            if page_table is None:
                raise RuntimeError(
                    "MLA block-decode replay has neither group tables nor a "
                    "draft page table to read page ids from"
                )
            table = self._expand_history_table(page_table[:bs])
        real_bs = min(int(table.shape[0]), bs)
        if real_bs > 0:
            rows[:real_bs].copy_(table[:real_bs, None, :width])
        if real_bs < bs:
            rows[real_bs:].zero_()

    def _decode_views(self, bs: int) -> MLADecodeMetadata:
        """Per-bs decode metadata views over the persistent buffers.

        One builder for capture and refresh; cached per bs — pointer-stable,
        no storage allocated.
        """
        metadata = self.decode_cuda_graph_metadata.get(bs)
        if metadata is not None:
            return metadata
        if self._block_decode_active:
            expanded_bs = bs * self.spec_num_tokens
            metadata = MLADecodeMetadata(
                num_extends=0,
                page_table=self.cuda_graph_page_table[:expanded_bs, :],
                seq_lens=self.cuda_graph_seq_lens[:expanded_bs],
                out_cache_loc=None,
            )
        else:
            capture_q_len = graph_verify_q_len(self.spec_num_tokens, self.is_draft)
            # A draft owns its per-step write locations (it reads the
            # published batch-ordered table); only the target routes writes
            # through the persistent loc buffer (allocated iff not draft).
            if self.decode_cuda_graph_out_cache_loc is not None:
                out_cache_loc = self.decode_cuda_graph_out_cache_loc[
                    : bs * capture_q_len
                ]
            else:
                out_cache_loc = None
            metadata = MLADecodeMetadata(
                num_extends=0,
                page_table=self.cuda_graph_page_table[:bs, :],
                seq_lens=self.cuda_graph_seq_lens[:bs],
                out_cache_loc=out_cache_loc,
                q_len_per_req=capture_q_len,
            )
        self.decode_cuda_graph_metadata[bs] = metadata
        return metadata

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        forward_mode: ForwardMode,
        page_table: torch.Tensor | None = None,
        num_extends: int = 0,
        for_graph_replay: bool = False,
        **kwargs,
    ) -> None:
        if forward_mode is not None and forward_mode.is_extend_or_mixed():
            raise NotImplementedError(
                f"mla decode refresh not supported for {forward_mode}"
            )

        metadata = self._decode_views(bs)
        # The cached view bakes num_extends=0; a mixed round's decode rows
        # start after the extend rows, so publish this round's split.
        metadata.num_extends = num_extends
        block_tables = kwargs.get("block_tables")
        if self._block_decode_active:
            # Replicate each request's page table across its block rows. Under
            # replay the seq_lens are re-derived in-graph from the live draft
            # length; eager has no in-graph writer, so fill them here.
            self._replay_block_decode_page_table(bs, page_table, block_tables)
            # Eager has no in-graph writer; the capture default's idle
            # refresh (actual_bs == 0) must seed the same safe baseline the
            # recorded fill_block_decode_seq_lens overwrites on replay.
            if not for_graph_replay or actual_bs == 0:
                self.fill_block_decode_seq_lens(bs, seq_lens)
            self.forward_decode_metadata = metadata
            return
        # Copy the live lengths into our own cache-seqlens buffer
        # (metadata.seq_lens views it); every metadata path reads it.
        q_len = metadata.q_len_per_req
        # clamp_min(1) is the identity, so the verify clamp is unconditional.
        self.cuda_graph_seq_lens[:bs].copy_(seq_lens[:bs].clamp_min(q_len))
        has_full_history_table = bool(block_tables) and (
            self._geometry.full_history_group_id in block_tables
        )
        if metadata.out_cache_loc is not None and (
            has_full_history_table or for_graph_replay
        ):
            if has_full_history_table:
                # Latch the group binding so select_out_cache_loc serves the
                # table-derived write locations (parity with the extend path).
                self._cache_groups_bound = True
            # The idle replay (actual_bs == 0) carries synthesized placeholder
            # tables; every row is a dummy row, so zero the buffers instead of
            # resolving write locations against the placeholders.
            self._replay_refresh_decode(
                bs, seq_lens, metadata, block_tables if actual_bs > 0 else None
            )
        elif self.is_draft:
            # Draft: expand the staged batch-ordered draft page table (raw
            # scheduler pages) into the persistent buffer.
            self._expand_history_table(
                page_table[:bs], out=self.cuda_graph_page_table[:bs]
            )
        else:
            # Idle/warmup before the backend binds: page_table is an
            # empty/dummy batch-ordered placeholder.
            self.cuda_graph_page_table[:bs, : self.max_num_pages].copy_(
                page_table[:bs, : self.max_num_pages]
            )
        self.forward_decode_metadata = metadata

    def _replay_refresh_decode(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        metadata: MLADecodeMetadata,
        block_tables,
    ) -> None:
        if metadata.out_cache_loc is None:
            raise RuntimeError("MLA graph metadata has no write-location buffer")
        # Must match the width baked into the captured buffer view.
        q_len = metadata.q_len_per_req
        real_bs = 0
        table = resolve_full_history_table(
            block_tables,
            self._geometry,
            0,
            kernel_page_size=self.kernel_page_size,
            max_kernel_pages=self.max_num_pages,
        )
        if table is not None:
            real_bs = min(int(table.shape[0]), bs)
            if real_bs > 0:
                metadata.page_table[:real_bs, : table.shape[1]].copy_(table[:real_bs])
                metadata.page_table[:real_bs, table.shape[1] :].zero_()
                mla_decode_out_cache_loc(
                    table,
                    # The clamped copy: a request shorter than the verify
                    # window would otherwise resolve locations before its start.
                    self.cuda_graph_seq_lens,
                    page_size=self.kernel_page_size,
                    batch_size=real_bs,
                    validate_pages=cache_debug_enabled(),
                    out=metadata.out_cache_loc,
                    q_len_per_req=q_len,
                )
        # Padded rows resolve to the null page 0 so they never touch a live page.
        metadata.page_table[real_bs:bs].zero_()
        metadata.out_cache_loc[real_bs * q_len : bs * q_len].zero_()

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # q is absorbed MLA query [T, H, R + D_rope]; k is compressed KV
        # [T, 1, R + D_rope]. DeepSeek normally writes cache before this call.
        if save_kv_cache:
            assert k is not None
            out_cache_loc = self.select_out_cache_loc(
                layer,
                out_cache_loc,
                kwargs.get("forward_mode", ForwardMode.DECODE),
            )
            token_to_kv_pool.set_mla_kv_buffer(
                layer,
                out_cache_loc,
                k[..., : self.kv_lora_rank],
                k[..., self.kv_lora_rank :],
            )

        metadata = self.forward_decode_metadata
        assert metadata is not None
        num_extends = metadata.num_extends
        q_len_per_req = q.shape[0] // bs if bs > 0 else 1

        if self._block_decode_active:
            # Metadata already carries one row per block position, each with the
            # block-end length, so the block is non-causal. Adding the causal
            # offsets below would re-impose exactly the ordering the draft must
            # not have, and re-expanding the rows would square the batch.
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim).unsqueeze(1)
            rows = num_extends * q_len_per_req
            page_table = metadata.page_table[rows:]
            cache_seqlens = metadata.seq_lens[rows:]
            max_seqlen_k = self.max_context_len
        elif q_len_per_req > 1:
            query = q.view(-1, layer.tp_q_head_num, layer.head_dim).unsqueeze(1)
            page_table = metadata.page_table[num_extends:].repeat_interleave(
                q_len_per_req, dim=0
            )
            cache_seqlens = metadata.seq_lens[num_extends:].repeat_interleave(
                q_len_per_req
            )
            # Draft catch-up starts from the current draft KV length; target
            # verify starts from the final target KV length and backs up.
            offset_start = 0 if self.is_draft else 1 - q_len_per_req
            offsets = torch.arange(
                offset_start,
                offset_start + q_len_per_req,
                device=cache_seqlens.device,
                dtype=cache_seqlens.dtype,
            ).repeat(bs)
            cache_seqlens = cache_seqlens + offsets
            max_seqlen_k = self.max_context_len
        else:
            query = q.view(bs, -1, layer.tp_q_head_num, layer.head_dim)
            page_table = metadata.page_table[num_extends:]
            cache_seqlens = metadata.seq_lens[num_extends:]
            max_seqlen_k = self.max_context_len

        softmax_scale = layer.scaling
        if self.data_type in (torch.float8_e4m3fn, torch.float8_e5m2):
            k_scale = (
                layer.k_scale_float
                if getattr(layer, "k_scale_float", None) is not None
                else 1.0
            )
            softmax_scale = k_scale * softmax_scale

        kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        if self.data_type != kv_cache.dtype:
            kv_cache = kv_cache.to(self.data_type)
        kv_cache = kv_cache.view(-1, self.kernel_page_size, 1, self.kv_cache_dim)

        value_weight = kwargs.get("value_weight")
        gate = kwargs.get("output_gate")
        projected_out = kwargs.get("projected_output")
        if value_weight is not None:
            # Fuse projection and gate into decode to avoid materializing latent output.
            result = mla_decode_with_kvcache(
                query,
                kv_cache,
                page_table,
                cache_seqlens,
                max_seqlen_k,
                self.qk_nope_head_dim,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                softmax_scale,
                value_weight=value_weight,
                gate=gate,
                out=projected_out,
                logit_cap=layer.logit_cap,
            )
        else:
            result = mla_decode_with_kvcache(
                q=query,
                kv_cache=kv_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                max_seqlen_k=max_seqlen_k,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                softmax_scale=softmax_scale,
                logit_cap=layer.logit_cap,
                solution=self.kernel_solution,
            )
        output = self._unwrap_output(result)
        if value_weight is not None:
            return output.reshape(-1, value_weight.shape[0] * value_weight.shape[2])
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if save_kv_cache:
            raise NotImplementedError(
                "MLA forward_extend cannot derive compressed cache rows from "
                "materialized K/V; DeepSeek writes MLA cache in the model path"
            )

        metadata = self.forward_prefill_metadata
        assert metadata is not None
        if metadata.use_absorbed_cached_extend:
            assert metadata.page_table is not None
            assert metadata.cum_seq_lens_kv is not None
            q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
            kv_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
            if self.data_type != kv_cache.dtype:
                kv_cache = kv_cache.to(self.data_type)
            kv_cache = kv_cache.view(-1, self.kernel_page_size, 1, self.kv_cache_dim)
            result = mla_extend_with_kvcache(
                q=q,
                kv_cache=kv_cache,
                page_table=metadata.page_table,
                cache_seqlens=metadata.seq_lens,
                cu_seqlens_q=metadata.cum_extend_seq_lens,
                cu_seqlens_kv=metadata.cum_seq_lens_kv,
                max_seqlen_q=metadata.max_extend_seq_len,
                max_seqlen_k=self.max_context_len,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                softmax_scale=layer.scaling,
                is_causal=True,
                logit_cap=layer.logit_cap,
                solution=self.kernel_solution,
            )
            output = self._unwrap_output(result)
            return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

        if metadata.max_extend_prefix_len > 0:
            raise NotImplementedError(
                "MLA prefix-cache extend is handled by DeepSeek's chunked "
                "prefix replay path via forward_extend_chunked"
            )

        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        result = mla_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=metadata.cum_extend_seq_lens,
            cu_seqlens_kv=metadata.cum_extend_seq_lens,
            max_seqlen_q=metadata.max_extend_seq_len,
            max_seqlen_kv=metadata.max_extend_seq_len,
            softmax_scale=layer.scaling,
            seq_lens_kv=metadata.extend_seq_lens,
            is_causal=True,
            logit_cap=layer.logit_cap,
            solution=self.kernel_solution,
        )
        output = self._unwrap_output(result)
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend_chunked(
        self,
        q,
        k,
        v,
        scaling,
        logits_soft_cap=None,
        *,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        max_q_len,
        max_kv_len,
        seq_lens,
        batch_size,
        causal,
        out: torch.Tensor | None = None,
    ):
        if causal:
            step_counter = self.step_counter
            if step_counter is not None:
                step_counter.record_cache()

        head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        q = q.reshape(-1, self.num_local_heads, head_dim)
        k = k.reshape(-1, self.num_local_heads, head_dim)
        v = v.reshape(-1, self.num_local_heads, self.v_head_dim)

        if q.dtype == torch.float8_e4m3fn:
            k = k.to(torch.float8_e4m3fn)
            v = v.to(torch.float8_e4m3fn)

        result = mla_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cum_seq_lens_q,
            cu_seqlens_kv=cum_seq_lens_kv,
            max_seqlen_q=max_q_len,
            max_seqlen_kv=max_kv_len,
            softmax_scale=scaling,
            seq_lens_kv=seq_lens,
            is_causal=causal,
            logit_cap=logits_soft_cap or 0.0,
            return_lse=True,
            out=out,
            solution=self.kernel_solution,
        )

        if isinstance(result, tuple):
            return result[0], result[1]
        return result, None

    def _unwrap_output(self, result):
        if isinstance(result, tuple):
            return result[0]
        return result


for _backend_name in ("mla", "gluon"):
    register_backend(_backend_name, {AttentionArch.MLA}, MLAAttnBackend)
