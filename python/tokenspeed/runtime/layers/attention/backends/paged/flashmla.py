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
from tokenspeed_kernel.ops.attention.flash_attn import flash_attn_varlen_func
from tokenspeed_kernel.ops.attention.flash_mla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from tokenspeed_kernel.ops.attention.flashinfer import (
    BatchMLAPagedAttentionWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.base import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.chunk import (
    build_chunked_prefill_metadata_arrays,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    FLASH_MLA_PAGE_SIZE as PAGE_SIZE,
)
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.layers.attention.utils import (
    create_flashinfer_kv_indices_triton,
)
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.flashinfer_config import get_flashinfer_workspace_size

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


@dataclass
class FlashMLADecodeMetadata:
    num_extends: int = 0
    page_table: torch.Tensor | None = None
    seq_lens_k: torch.Tensor | None = None
    # Verify window width baked into the graph views (1 outside target verify).
    q_len_per_req: int = 1


@dataclass
class _PrefillMetadata:
    prefill_wrapper: BatchMLAPagedAttentionWrapper
    use_ragged: bool


@dataclass
class _ChunkedPrefillMetadata:
    extend_prefix_lens: torch.Tensor
    extend_prefix_lens_cpu: torch.Tensor
    extend_seq_lens: torch.Tensor
    extend_seq_lens_cpu: torch.Tensor
    cum_extend_seq_lens: torch.Tensor
    max_extend_seq_len: int
    chunked_loop_num: int
    chunk_kv_indices_list: list
    chunked_seq_len: torch.Tensor
    cu_chunked_seq_len: torch.Tensor
    max_chunk_len_per_loop: list


def _per_token_slot_table(
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


# Shared across all flashinfer prefill wrappers used by FlashMLABackend.
_global_workspace_buffer = None


class FlashMLABackend(PagedAttentionBackend):
    """FlashMLA leaf for TokenSpeed scheduling.

    Uses the FlashMLA kernel for decode (any q_len) and FlashInfer's MLA
    prefill wrappers for the EXTEND path. The FlashMLA kernel walks pages at
    a fixed ``PAGE_SIZE`` stride, so that is the leaf's kernel page size.
    """

    default_kernel_page_size = PAGE_SIZE

    def __init__(self, config: AttnConfig, spec: MLAConfig, *, kernel_page_size: int):
        if kernel_page_size != PAGE_SIZE:
            raise ValueError(
                f"FlashMLA walks pages at a fixed {PAGE_SIZE}-token stride, "
                f"got kernel_page_size={kernel_page_size}"
            )
        super().__init__(config, spec, kernel_page_size=kernel_page_size)

        self.kv_cache_quant_method = config.kv_cache_quant_method
        self.cache_dtype = config.kv_cache_dtype

        # MLA-specific dimensions
        self.kv_lora_rank = spec.kv_lora_rank
        self.qk_nope_head_dim = spec.qk_nope_head_dim
        self.qk_rope_head_dim = spec.qk_rope_head_dim
        self.v_head_dim = spec.v_head_dim
        self.kv_cache_dim = spec.kv_lora_rank + spec.qk_rope_head_dim
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype
        self.num_local_heads = spec.num_attention_heads // spec.attn_tp_size
        self.num_q_heads = spec.num_attention_heads // spec.attn_tp_size

        # A block drafter (DFLASH/DSpark) drafts a whole block per pass; its
        # captured decode graph seeds block-end lengths via
        # ``fill_block_decode_seq_lens`` (see the drafter's is_capturing path).
        self.draft_block_decode = bool(config.draft_block_decode)

        if self.kv_cache_quant_method == "per_token_head":
            raise NotImplementedError(
                "FlashMLABackend no longer supports "
                "kv_cache_quant_method='per_token_head'."
            )
        if self.cache_dtype == torch.float8_e4m3fn:
            raise NotImplementedError(
                "FlashMLABackend no longer supports dense FP8 KV cache. "
                "Use a non-FP8 KV cache."
            )

        # Workspace buffer + flashinfer prefill wrappers (EXTEND path only).
        global _global_workspace_buffer
        if _global_workspace_buffer is None:
            _global_workspace_buffer = torch.empty(
                get_flashinfer_workspace_size(),
                dtype=torch.uint8,
                device=config.device,
            )
        self.workspace_buffer = _global_workspace_buffer

        max_bs = config.max_bs
        self.kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=config.device
        )
        self.qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=config.device
        )

        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )
        self.prefill_wrapper_paged = BatchMLAPagedAttentionWrapper(
            self.workspace_buffer,
            backend="auto",
        )
        self.indices_updater_prefill = _PrefillIndicesUpdater(config, spec, self)

        # Metadata state. Decode and prefill metadata are split so MIXED batches
        # can carry both simultaneously (decode-half + prefill-half sub-contexts
        # dispatch to their respective metadata).
        self.forward_decode_metadata: FlashMLADecodeMetadata | None = None
        self.forward_prefill_metadata: _PrefillMetadata | None = None
        self.chunked_prefill_metadata: _ChunkedPrefillMetadata | None = None
        # FlashMLA builds its tile schedule lazily inside the FIRST
        # flash_mla_with_kvcache call (from that call's cache_seqlens) and then
        # freezes it on the FlashMLASchedMeta object: later calls keep the
        # frozen page range, and a request whose seq_len has since crossed a
        # page boundary silently loses its newest page. So the object is bound
        # to one seq_lens value: every write into seq_lens_buf that a kernel
        # call will follow renews it (_renew_decode_tile_metadata). It lives
        # here, on the backend, not on the decode views: kernel-owned per-step
        # scratch, not an address the refresh keeps for a graph. Under capture
        # the recorded schedule-build re-runs from the live buffer on every
        # replay (which never reads this slot); strong refs keep every sched a
        # graph recorded alive for the graph's lifetime.
        self._decode_tile_metadata: object | None = None
        self._decode_tile_metadata_keepalive: list[object] = []

    # ------------------------------------------------------------------
    # Metadata init
    # ------------------------------------------------------------------

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_with_prefix: bool,
        **kwargs,
    ):
        if not (forward_mode.is_extend_or_mixed() or forward_mode.is_idle()):
            raise RuntimeError(
                "FlashMLA decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend/mixed ({forward_mode})"
            )
        if forward_mode.is_extend_or_mixed():
            self._init_prefill_metadata(
                seq_lens=seq_lens[:num_extends],
                extend_with_prefix=extend_with_prefix,
                extend_prefix_lens=extend_prefix_lens[:num_extends],
                extend_prefix_lens_cpu=extend_prefix_lens_cpu[:num_extends],
                extend_seq_lens=extend_seq_lens[:num_extends],
                extend_seq_lens_cpu=extend_seq_lens_cpu[:num_extends],
                page_table=page_table[:num_extends],
            )
        # Target mixed/idle batches carry decode rows whose metadata this
        # init must cover; the same in-place refresh serves them. A draft's
        # decode metadata instead comes from the wrapper's refresh after this
        # init (the unified draft contract).
        if forward_mode.is_idle() or (forward_mode.is_mixed() and not self.is_draft):
            self.refresh_decode_metadata(
                bs, bs, seq_lens, page_table, num_extends=num_extends
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

    def _renew_decode_tile_metadata(self, *, for_graph: bool) -> None:
        """Bind a fresh (uninitialized) FlashMLASchedMeta for the seq_lens just
        written; the next kernel call builds its schedule from them.

        Args:
            for_graph: The next kernel call is recorded into a CUDA graph
                (capture seeding, or an in-graph seq_lens edit under capture).
                The graph replays the recorded schedule-build against this
                object's tensors, so it is kept alive for the graph's lifetime.
                Eager objects are dropped with the step.
        """
        tile_metadata = get_mla_metadata()[0]
        if for_graph:
            self._decode_tile_metadata_keepalive.append(tile_metadata)
        self._decode_tile_metadata = tile_metadata

    def _write_decode_seq_lens(self, bs: int, seq_lens: torch.Tensor) -> None:
        """An in-graph-capable seq_lens rewrite (drafter hooks): copy the rows
        into the persistent buffer and renew the schedule they invalidate."""
        self.seq_lens_buf[:bs].copy_(seq_lens[:bs])
        self._renew_decode_tile_metadata(
            for_graph=torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        )

    def _init_prefill_metadata(
        self,
        seq_lens: torch.Tensor,
        extend_with_prefix: bool,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        page_table: torch.Tensor,
    ):
        # EXTEND path — flashinfer ragged/paged prefill.
        seq_lens_cpu = seq_lens.cpu()
        seq_lens_sum = seq_lens_cpu.sum().item()

        extend_no_prefix = not extend_with_prefix
        if extend_no_prefix and bool(extend_prefix_lens_cpu.any()):
            # The ragged plan sizes its paged kv_indices by the prefix sum
            # (zero here), so a prefixed row would overrun it in-kernel.
            raise RuntimeError(
                "FlashMLABackend: extend_with_prefix is False but "
                f"extend_prefix_lens={extend_prefix_lens_cpu.tolist()} has a "
                "cached/chunked prefix"
            )
        use_ragged = (
            not global_server_args_dict["mla_disable_ragged"] and extend_no_prefix
        )

        # Two differently-shaped views of the kernel page table:
        #   * flashinfer paged prefill (plan page_size=1) walks a PER-TOKEN slot
        #     table, so expand each token to its absolute latent slot;
        #   * chunked prefix replay (create_chunked_cache_kv_indices_paged) walks
        #     the PAGE table directly, deriving slot = page_id*p + pos%p
        #     in-kernel.
        prefill_table = _per_token_slot_table(
            page_table,
            batch_size=seq_lens.shape[0],
            page_size=self.kernel_page_size,
            max_context_len=self.max_context_len,
        ).to(torch.int32)

        self.indices_updater_prefill.update(
            seq_lens,
            seq_lens_sum,
            extend_prefix_lens,
            page_table=prefill_table,
            prefill_wrapper_paged=self.prefill_wrapper_paged,
            use_ragged=use_ragged,
        )
        self.forward_prefill_metadata = _PrefillMetadata(
            self.prefill_wrapper_paged, use_ragged
        )

        num_extends = extend_seq_lens.shape[0]
        cum_extend_seq_lens = torch.zeros(
            num_extends + 1, device=self.device, dtype=torch.int32
        )
        torch.cumsum(extend_seq_lens, dim=0, out=cum_extend_seq_lens[1:])
        max_extend_seq_len = extend_seq_lens_cpu.max().item()
        (
            chunked_loop_num,
            chunk_kv_indices_list,
            chunked_seq_len,
            cu_chunked_seq_len,
            max_chunk_len_per_loop,
        ) = build_chunked_prefill_metadata_arrays(
            extend_prefix_lens,
            extend_prefix_lens_cpu,
            page_table[: seq_lens.shape[0]],
            self.kernel_page_size,
        )
        self.chunked_prefill_metadata = _ChunkedPrefillMetadata(
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            cum_extend_seq_lens=cum_extend_seq_lens,
            max_extend_seq_len=max_extend_seq_len,
            chunked_loop_num=chunked_loop_num,
            chunk_kv_indices_list=chunk_kv_indices_list,
            chunked_seq_len=chunked_seq_len,
            cu_chunked_seq_len=cu_chunked_seq_len,
            max_chunk_len_per_loop=max_chunk_len_per_loop,
        )

    # ------------------------------------------------------------------
    # CUDA graph (decode only, any q_len)
    # ------------------------------------------------------------------

    def _decode_views(self, bs: int) -> FlashMLADecodeMetadata:
        """Per-bs decode metadata views over the persistent buffers.

        One builder for capture and refresh; cached per bs — pointer-stable,
        no storage allocated. The step's tile schedule is not a view field
        (see ``_decode_tile_metadata``).
        """
        metadata = self._decode_views_by_bs.get(bs)
        if metadata is not None:
            return metadata
        metadata = FlashMLADecodeMetadata(
            num_extends=0,
            page_table=self.page_table_buf[:bs],
            seq_lens_k=self.seq_lens_buf[:bs],
            q_len_per_req=self.verify_floor,
        )
        self._decode_views_by_bs[bs] = metadata
        return metadata

    def init_forward_metadata_capture_cuda_graph(
        self, bs: int, seq_lens: torch.Tensor, page_table: torch.Tensor
    ) -> None:
        # The one sanctioned capture-only asymmetry: the refresh super() runs
        # seeds the seq_lens, and the schedule the graph records is built
        # against a dedicated object kept alive for the graph's lifetime;
        # eager refresh renews per step instead.
        super().init_forward_metadata_capture_cuda_graph(bs, seq_lens, page_table)
        self._renew_decode_tile_metadata(for_graph=True)

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        num_extends: int = 0,
        for_graph_replay: bool = False,
    ) -> None:
        metadata = self._decode_views(bs)
        # Verify rows span seq-N..seq-1; clamp so a request shorter than the
        # window does not resolve positions before its start (identity for
        # floor 1). Padding rows replay at seq_len 1 and clamp the same way.
        q_len = metadata.q_len_per_req
        self.seq_lens_buf[:bs].copy_(seq_lens[:bs].clamp_min(q_len))
        # Copy the router-resolved kernel page table into the persistent buffer.
        self.page_table_buf[:bs, : page_table.shape[1]].copy_(page_table[:bs])
        metadata.num_extends = num_extends
        # Replay leaves the schedule slot alone: the graph re-runs its recorded
        # schedule-build against the live seq_lens and never reads the slot
        # from Python. Eager renews for the seq_lens just written.
        if not for_graph_replay:
            self._renew_decode_tile_metadata(for_graph=False)
        self.forward_decode_metadata = metadata

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        """A drafter's per-step seq_lens edit (chain step, step-0 accept
        correction, MTP re-anchor): republish the rows and renew the schedule
        — the step's kernel call must not inherit a schedule frozen on the
        previous step's lengths."""
        self._write_decode_seq_lens(seq_lens.shape[0], seq_lens)

    @property
    def block_decode_expansion(self) -> int:
        """One entry per request even under block decode: ``forward_decode``
        repeats it across the block's queries at forward time."""
        return 1

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        """Publish block-end cache lengths inside a captured draft graph.

        A block drafter runs its multi-token pass inside the captured graph and
        writes the per-request block-end length here (one entry per request;
        the block's ``draft_query_width`` queries share it, non-causal).
        ``forward_decode`` repeats each entry across the block's queries, so
        the graph keeps ``bs`` entries — mirrors ``TRTLLMMLABackend``.
        """
        if not self.draft_block_decode:
            raise RuntimeError("Block decode sequence lengths require DFLASH mode.")
        self._write_decode_seq_lens(
            bs, block_seq_lens.clamp(self.spec_num_tokens, self.max_context_len)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool,
        forward_mode: ForwardMode,
        **kwargs,
    ):
        assert forward_mode is not None and forward_mode.is_extend()

        # Prefill: dispatch to ragged (MHA-style) or absorbed (MQA) path.
        if self.forward_prefill_metadata.use_ragged:
            return self._forward_normal_extend(q, k, v, layer, save_kv_cache)
        else:
            return self._forward_absorbed_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                save_kv_cache,
            )

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
        # flash_attn_varlen_func has no `out=` parameter; copy into the
        # caller-provided buffer at the end when requested.
        output, lse, *_ = flash_attn_varlen_func(
            q=q.view(-1, self.num_local_heads, head_dim),
            k=k.view(-1, self.num_local_heads, head_dim).to(q.dtype),
            v=v.view(-1, self.num_local_heads, self.v_head_dim).to(q.dtype),
            cu_seqlens_q=cum_seq_lens_q,
            cu_seqlens_k=cum_seq_lens_kv,
            max_seqlen_q=max_q_len,
            max_seqlen_k=max_kv_len,
            softmax_scale=scaling,
            causal=causal,
            return_attn_probs=True,
        )
        if out is not None:
            out.copy_(output.view(out.shape))
            output = out
        # lse must be transposed when using fa3.
        return output, lse.T.contiguous()

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool,
        **kwargs,
    ) -> torch.Tensor:
        # Multi-token decode (target verify or drafter compound) runs the same
        # FlashMLA decode kernel with q_len > 1 rows per request.
        q_len_per_req = q.shape[0] // bs if bs > 0 else 1
        if q_len_per_req > 1:
            metadata = self.forward_decode_metadata
            num_extends = metadata.num_extends
            bs = (
                q.shape[0]
                if self.is_draft
                else metadata.page_table.shape[0] - num_extends
            )

        o, _ = self._run_flash_mla_decode(
            q,
            k,
            v,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            bs,
            save_kv_cache=save_kv_cache,
        )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------
    # EXTEND prefill helpers
    # ------------------------------------------------------------------

    def _forward_normal_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        save_kv_cache: bool = True,
    ):
        assert not save_kv_cache

        o = self.prefill_wrapper_ragged.forward(
            q,
            k.view(-1, layer.tp_k_head_num, layer.head_dim),
            v.view(-1, layer.tp_k_head_num, layer.v_head_dim),
            causal=True,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_absorbed_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        save_kv_cache: bool = True,
    ):
        # q is whole Q [T, H, head_dim]; k is whole latent [T, 1, head_dim].
        # flashinfer prefill_wrapper.run() requires q_nope / q_pe split, so
        # slice views here (free) before handing off to the kernel.
        assert k is not None

        if save_kv_cache:
            token_to_kv_pool.set_mla_kv_buffer(
                layer,
                out_cache_loc,
                k[..., : layer.v_head_dim],
                k[..., layer.v_head_dim :],
            )

        q = q.view(-1, layer.tp_q_head_num, layer.head_dim)
        q_nope = q[..., : layer.v_head_dim]
        q_pe = q[..., layer.v_head_dim :]
        o = q_nope.new_empty(q_nope.shape)

        k_buf = token_to_kv_pool.get_key_buffer(layer.layer_id).to(q_nope.dtype)
        o = self.forward_prefill_metadata.prefill_wrapper.run(
            q_nope,
            q_pe,
            k_buf[:, :, : layer.v_head_dim],
            k_buf[:, :, layer.v_head_dim :],
            out=o,
        )
        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _run_flash_mla_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        *,
        save_kv_cache: bool,
    ):
        if k is not None:
            assert v is not None
            if save_kv_cache:
                token_to_kv_pool.set_kv_buffer(layer, out_cache_loc, k, v)

        metadata = self.forward_decode_metadata
        num_extends = metadata.num_extends
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        assert (
            layer.tp_q_head_num == self.num_q_heads
        ), f"{layer.tp_q_head_num=} != {self.num_q_heads=}"
        reshape_q = q.view(bs, -1, self.num_q_heads, layer.head_dim)

        page_table = metadata.page_table[num_extends : num_extends + bs]
        cache_seqlens = metadata.seq_lens_k.to(torch.int32)
        # Draft block-decode: forward_decode flattened q to one kernel row per
        # drafted block position (bs == bs_orig * draft_query_width), but the
        # page table and seq_lens carry one entry per request. Repeat each
        # request's row across its block positions so every block query attends
        # the whole block (block-diffusion); the FlashMLA kernel requires
        # cache_seqlens to be shape (num_kernel_rows).
        src_rows = page_table.shape[0]
        if self.is_draft and 0 < src_rows < bs and bs % src_rows == 0:
            width = bs // src_rows
            page_table = page_table.repeat_interleave(width, dim=0)
            cache_seqlens = cache_seqlens.repeat_interleave(width)

        return flash_mla_with_kvcache(
            q=reshape_q,
            k_cache=k_cache.view(-1, PAGE_SIZE, 1, self.kv_cache_dim),
            block_table=page_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=self._decode_tile_metadata,
            softmax_scale=layer.scaling,
            causal=True,
        )


class _PrefillIndicesUpdater:
    """Plans FlashInfer MLA prefill wrappers for the EXTEND path."""

    def __init__(
        self, config: AttnConfig, spec: MLAConfig, attn_backend: FlashMLABackend
    ):
        self.num_local_heads = spec.num_attention_heads // spec.attn_tp_size
        self.kv_lora_rank = spec.kv_lora_rank
        self.qk_nope_head_dim = spec.qk_nope_head_dim
        self.qk_rope_head_dim = spec.qk_rope_head_dim
        self.v_head_dim = spec.v_head_dim
        self.scaling = spec.scaling
        self.data_type = config.kv_cache_dtype
        self.q_data_type = config.dtype

        self.kv_indptr = attn_backend.kv_indptr
        self.qo_indptr = attn_backend.qo_indptr
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

    def update(
        self,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        prefix_lens: torch.Tensor,
        *,
        page_table: torch.Tensor,
        prefill_wrapper_paged: BatchMLAPagedAttentionWrapper,
        use_ragged: bool,
    ):
        if use_ragged:
            paged_kernel_lens = prefix_lens
            paged_kernel_lens_sum = 0
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self._call_begin_forward(
            self.prefill_wrapper_ragged,
            prefill_wrapper_paged,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            self.kv_indptr,
            self.qo_indptr,
            use_ragged,
            page_table,
        )

    def _call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchMLAPagedAttentionWrapper,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        page_table: torch.Tensor,
    ):
        bs = len(seq_lens)
        sm_scale = self.scaling

        torch.cumsum(paged_kernel_lens, dim=0, out=kv_indptr[1 : bs + 1])
        kv_indptr = kv_indptr[: bs + 1]
        kv_indices = torch.empty(
            paged_kernel_lens_sum,
            dtype=torch.int32,
            device=seq_lens.device,
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            page_table,
            paged_kernel_lens,
            kv_indptr,
            kv_indices,
            page_table.shape[1],
        )
        torch.cumsum(seq_lens - prefix_lens, dim=0, out=qo_indptr[1 : bs + 1])
        qo_indptr = qo_indptr[: bs + 1]

        if use_ragged:
            wrapper_ragged.begin_forward(
                qo_indptr=qo_indptr,
                kv_indptr=qo_indptr,
                num_qo_heads=self.num_local_heads,
                num_kv_heads=self.num_local_heads,
                head_dim_qk=self.qk_nope_head_dim + self.qk_rope_head_dim,
                head_dim_vo=self.v_head_dim,
                q_data_type=self.q_data_type,
            )
        else:
            kv_len_arr = kv_indptr[1:] - kv_indptr[:-1]
            wrapper_paged.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_len_arr,
                self.num_local_heads,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
                1,
                True,
                sm_scale,
                self.q_data_type,
                self.data_type,
            )


register_backend("flashmla", {AttentionArch.MLA}, FlashMLABackend)
