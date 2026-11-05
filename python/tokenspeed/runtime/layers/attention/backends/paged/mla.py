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
from tokenspeed_kernel.ops.attention.dsv4._triton.dcp import normalize_dcp_partials
from tokenspeed_kernel.ops.attention.mla import (
    mla_decode_with_kvcache,
    mla_extend_with_kvcache,
    mla_prefill,
    mla_use_absorbed_extend,
    supports_mla_decode_query_blocks,
)
from tokenspeed_kernel.selection import KernelPlan
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.configs.numerics import BITWISE_ENVELOPES
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.layers.attention.backends.base import reject_query_shard
from tokenspeed.runtime.layers.attention.backends.paged.base import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.support import CudaGraphSupport
from tokenspeed.runtime.layers.attention.chunk import (
    build_chunked_prefill_metadata_arrays,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.dcp.comm import (
    combine_attention_partials,
    gather_query_heads,
)
from tokenspeed.runtime.layers.attention.dcp.decode import DCPDecodeState
from tokenspeed.runtime.layers.attention.dcp.metadata import CompactDCPMetadata
from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    MLA_PAGE_SIZE,
)
from tokenspeed.runtime.layers.attention.registry import register_backend
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import global_server_args_dict

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention

logger = get_colorful_logger(__name__)


@dataclass(kw_only=True)
class MLAPrefillMetadata:
    # Device-side metadata for explicit Q/K/V MLA prefill and prefix replay.
    seq_lens: torch.Tensor
    extend_prefix_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    cum_extend_seq_lens: torch.Tensor
    cum_seq_lens_kv: torch.Tensor | None
    page_table: torch.Tensor | None
    # Host-side metadata.
    extend_prefix_lens_cpu: torch.Tensor
    extend_seq_lens_cpu: torch.Tensor
    max_extend_seq_len: int
    max_extend_prefix_len: int
    use_absorbed_cached_extend: bool
    # Per-prefix-chunk arrays consumed by DeepSeek's chunked prefix replay.
    chunked_loop_num: int
    chunk_kv_indices_list: list[torch.Tensor]
    chunked_seq_len: torch.Tensor
    cu_chunked_seq_len: torch.Tensor
    max_chunk_len_per_loop: list[int]
    # Host bounds are optional for leaves that do not gather prefix history.
    chunk_bounds_cpu: torch.Tensor | None = None


@dataclass(kw_only=True)
class MLADecodeMetadata:
    # num_extends lets mixed batches slice decode requests after extend requests.
    num_extends: int
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    # Verify window width baked into the graph views (1 outside target verify).
    q_len_per_req: int = 1
    # Block decode only: the same rows before expansion, one per request.
    # The query-axis fold reads these instead of re-deriving them once
    # per layer.
    block_page_table: torch.Tensor | None = None
    block_seq_lens: torch.Tensor | None = None
    # Sharded read metadata; page_table retains global virtual addresses.
    dcp: CompactDCPMetadata | None = None
    # DSA wrapper's per-token indexer rows (``[bs * spec_num_tokens, 1]``
    # context lengths) and their opaque ``dsa_plan`` (None when the selected
    # kernel needs none); the dense leaf itself never reads them.
    _dsa_seq_lens_2d: torch.Tensor | None = None
    _dsa_plan: object | None = None

    @property
    def seq_lens_k(self) -> torch.Tensor:
        return self.seq_lens


class MLAAttnBackend(PagedAttentionBackend):
    """Unified MLA leaf routed through tokenspeed_kernel MLA APIs."""

    supports_mla_projected_value_decode = True
    default_kernel_page_size = MLA_PAGE_SIZE
    # Decode forwards layer.sliding_window_size as window_left.
    supports_layer_sliding_window: bool = True

    def __init__(self, config: AttnConfig, spec: MLAConfig, *, kernel_page_size: int):
        super().__init__(config, spec, kernel_page_size=kernel_page_size)

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
        # request materializes one single-query decode metadata entry per
        # block position (block_decode_expansion), all sharing the block-end
        # seq_len, so every block query sees the whole block including its own
        # future. Target verify and ordinary decode are untouched.
        self.draft_block_decode = bool(config.draft_block_decode)

        backend_name = spec.backend_name or "mla"
        self.kernel_solution = {"mla": None, "gluon": "gluon"}[backend_name]
        self.batch_invariant = global_server_args_dict["numerics"] in BITWISE_ENVELOPES
        self.extend_kernel_plan = None
        self.decode_kernel_plan = None
        if self.batch_invariant:
            if spec.sliding_window_tokens is not None:
                raise ValueError(
                    "Absorbed MLA extend does not yet carry a sliding-window mask"
                )
            signature = format_signature(
                q=dense_tensor_format(self.q_data_type),
                kv_cache=dense_tensor_format(self.data_type),
            )
            traits = {
                "num_q_heads": self.num_local_heads,
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "page_size": self.kernel_page_size,
                "logit_cap": False,
                "return_lse": False,
            }
            variable_traits = frozenset({"batch_size", "max_seqlen_q", "max_seqlen_k"})
            self.extend_kernel_plan = KernelPlan.select(
                "attention",
                "mla_extend_with_kvcache",
                signature,
                features=frozenset({"batch_invariant"}),
                static_traits={**traits, "is_causal": True},
                variable_traits=variable_traits,
                solution=self.kernel_solution,
                declared_traits=frozenset(),
                ignore_layout=False,
            )
            self.decode_kernel_plan = KernelPlan.select(
                "attention",
                "mla_decode_with_kvcache",
                signature,
                features=frozenset({"batch_invariant"}),
                static_traits={
                    **traits,
                    "q_len": 1,
                    "sliding_window": False,
                    "noncausal_block_size": 1,
                },
                variable_traits=variable_traits,
                solution=self.kernel_solution,
                declared_traits=frozenset(),
                ignore_layout=False,
            )
        # The strict arithmetic materializes latent output before value projection.
        self.supports_mla_projected_value_decode = not self.batch_invariant
        self.supports_layer_sliding_window = not self.batch_invariant
        self.cuda_graph_support = CudaGraphSupport.from_plans(
            prefill=self.extend_kernel_plan, decode=self.decode_kernel_plan
        )

        self.forward_decode_metadata: MLADecodeMetadata | None = None
        self.forward_prefill_metadata: MLAPrefillMetadata | None = None
        self.chunked_prefill_metadata: MLAPrefillMetadata | None = None
        self._query_block_decode: dict[tuple[int, int, bool], bool] = {}
        self._block_page_table_buf: torch.Tensor | None = None
        self._block_seq_lens_buf: torch.Tensor | None = None

    def _takes_query_blocks(
        self, num_q_heads: int, q_len: int, sliding_window: bool
    ) -> bool:
        """Whether a decode kernel wants this block on the query axis.

        Only the shapes this backend was built for and the layer's mask decide
        it, so it is resolved once per combination and reused. False keeps the
        flattened one-row-per-block-position metadata the portable kernel
        reads, which is what every other configuration has always sent.
        """
        if self.decode_kernel_plan is not None:
            return False
        key = (num_q_heads, q_len, sliding_window)
        answer = self._query_block_decode.get(key)
        if answer is None:
            answer = supports_mla_decode_query_blocks(
                q_dtype=self.data_type,
                kv_dtype=self.data_type,
                page_size=self.kernel_page_size,
                num_q_heads=num_q_heads,
                q_len=q_len,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                sliding_window=sliding_window,
                noncausal_block_size=q_len if self.block_decode_active else 1,
                solution=self.kernel_solution,
            )
            self._query_block_decode[key] = answer
            logger.info(
                "MLA block decode uses the "
                f"{('query-axis' if answer else 'flattened')!s} layout "
                f"(heads={num_q_heads:d}, block={q_len:d}, page="
                f"{self.kernel_page_size:d}, dtype={self.data_type!s}, window="
                f"{sliding_window!s}).",
            )
        return answer

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        super()._publish_cache_pool(cache_pool)
        self.forward_decode_metadata = None
        self.forward_prefill_metadata = None
        self.chunked_prefill_metadata = None
        self._block_page_table_buf = None
        self._block_seq_lens_buf = None

    def _should_use_absorbed_cached_extend(
        self, *, max_extend_seq_len: int, max_extend_prefix_len: int
    ) -> bool:
        if self.batch_invariant:
            # Initial and cached prefill need the same query representation
            # as decode, independent of prefix length or batch geometry.
            return True
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

    # ------------------------------------------------------------------
    # Metadata
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
        query_shard: QueryShardPlan | None,
        page_table_cpu: torch.Tensor | None,
        **kwargs,
    ) -> None:
        reject_query_shard(query_shard, "MLAAttnBackend")
        del page_table_cpu
        if not (forward_mode.is_extend_or_mixed() or forward_mode.is_idle()):
            raise RuntimeError(
                "MLA decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend/mixed ({forward_mode})"
            )
        if forward_mode.is_extend_or_mixed():
            self._init_prefill_metadata(
                seq_lens=seq_lens[:num_extends],
                extend_prefix_lens=extend_prefix_lens[:num_extends],
                extend_prefix_lens_cpu=extend_prefix_lens_cpu[:num_extends],
                extend_seq_lens=extend_seq_lens[:num_extends],
                extend_seq_lens_cpu=extend_seq_lens_cpu[:num_extends],
                page_table=page_table[:num_extends],
            )

        # Target mixed/idle batches carry decode requests whose metadata this
        # init must cover (verify decodes q_len tokens per request); the same
        # in-place refresh serves them. A draft's decode metadata instead
        # comes from the wrapper's refresh after this init (the unified draft
        # contract).
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

    def _init_prefill_metadata(
        self,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        page_table: torch.Tensor,
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
        if use_absorbed_cached_extend:
            cum_seq_lens_kv = torch.zeros_like(cum_extend_seq_lens)
            torch.cumsum(seq_lens, dim=0, out=cum_seq_lens_kv[1:])

        (
            chunked_loop_num,
            chunk_kv_indices_list,
            chunked_seq_len,
            cu_chunked_seq_len,
            max_chunk_len_per_loop,
            chunk_bounds_cpu,
        ) = build_chunked_prefill_metadata_arrays(
            extend_prefix_lens,
            extend_prefix_lens_cpu,
            page_table,
            self.kernel_page_size,
        )

        metadata = MLAPrefillMetadata(
            seq_lens=seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            cum_extend_seq_lens=cum_extend_seq_lens,
            cum_seq_lens_kv=cum_seq_lens_kv,
            page_table=page_table if use_absorbed_cached_extend else None,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            max_extend_seq_len=max_extend_seq_len,
            max_extend_prefix_len=max_extend_prefix_len,
            use_absorbed_cached_extend=use_absorbed_cached_extend,
            chunked_loop_num=chunked_loop_num,
            chunk_kv_indices_list=chunk_kv_indices_list,
            chunked_seq_len=chunked_seq_len,
            cu_chunked_seq_len=cu_chunked_seq_len,
            max_chunk_len_per_loop=max_chunk_len_per_loop,
            chunk_bounds_cpu=chunk_bounds_cpu,
        )
        self.forward_prefill_metadata = metadata
        self.chunked_prefill_metadata = metadata

    def _ensure_block_row_buffers(self, bs: int) -> None:
        """Resident per-request rows behind the block entries, sized once.

        Allocated against the graph buffers' capacity so a captured graph
        records a block that outlives it.
        """
        if self._block_page_table_buf is not None:
            return
        max_bs = self.page_table_buf.shape[0] // max(self.block_decode_expansion, 1)
        self._block_page_table_buf = torch.zeros(
            (max_bs, self.max_num_pages),
            dtype=self.page_table_buf.dtype,
            device=self.page_table_buf.device,
        )
        self._block_seq_lens_buf = torch.zeros(
            max_bs, dtype=self.seq_lens_buf.dtype, device=self.seq_lens_buf.device
        )

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        """Broadcast, keeping the per-request row the query-axis fold reads.

        The drafter calls this inside the captured graph, so the row has to be
        written there too rather than derived from the expanded view per layer.
        """
        self._ensure_block_row_buffers(bs)
        rows = self._block_seq_lens_buf[:bs]
        torch.clamp(
            block_seq_lens[:bs], self.spec_num_tokens, self.max_context_len, out=rows
        )
        expansion = self.block_decode_expansion
        self.decode_seq_lens_buffer[: bs * expansion].view(bs, expansion).copy_(
            rows.unsqueeze(1)
        )

    def _decode_views(self, bs: int) -> MLADecodeMetadata:
        """Per-bs decode metadata views over the persistent buffers.

        One builder for capture and refresh; cached per bs — pointer-stable,
        no storage allocated.
        """
        metadata = self._decode_views_by_bs.get(bs)
        if metadata is not None:
            return metadata
        if self.block_decode_active:
            expanded_bs = bs * self.block_decode_expansion
            self._ensure_block_row_buffers(bs)
            metadata = MLADecodeMetadata(
                num_extends=0,
                page_table=self.page_table_buf[:expanded_bs],
                seq_lens=self.seq_lens_buf[:expanded_bs],
                # The rows the block entries were expanded from. A kernel that
                # reads the block on the query axis wants exactly these, and
                # re-deriving them from the expanded view costs a strided copy
                # per layer instead of one write per forward.
                block_page_table=self._block_page_table_buf[:bs],
                block_seq_lens=self._block_seq_lens_buf[:bs],
            )
        else:
            metadata = MLADecodeMetadata(
                num_extends=0,
                page_table=self.page_table_buf[:bs],
                seq_lens=self.seq_lens_buf[:bs],
                q_len_per_req=self.verify_floor,
            )
        self._decode_views_by_bs[bs] = metadata
        return metadata

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
        # The cached view bakes num_extends=0; a mixed round's decode requests
        # start after the extend requests, so publish this round's split.
        metadata.num_extends = num_extends
        if self.block_decode_active:
            # Replicate each request's page table across its block positions.
            # Under replay the seq_lens are re-derived in-graph from the live
            # draft length; eager has no in-graph writer, and the capture
            # seeding needs the same safe baseline the recorded
            # fill_block_decode_seq_lens overwrites on replay.
            spec = self.spec_num_tokens
            self._ensure_block_row_buffers(bs)
            rows = self._block_page_table_buf[:bs]
            rows.copy_(page_table[:bs])
            self.page_table_buf[: bs * spec].view(bs, spec, self.max_num_pages).copy_(
                rows[:, None, :]
            )
            if not for_graph_replay or actual_bs == 0:
                self.fill_block_decode_seq_lens(bs, seq_lens)
            self.forward_decode_metadata = metadata
            return
        # clamp_min(1) is the identity, so the verify clamp is unconditional.
        self.seq_lens_buf[:bs].copy_(seq_lens[:bs].clamp_min(metadata.q_len_per_req))
        self.page_table_buf[:bs].copy_(page_table[:bs])
        self.forward_decode_metadata = metadata

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        **kwargs,
    ) -> torch.Tensor:
        # q is the absorbed MLA query [T, H, R + D_rope]; the prologue wrote the latent cache.
        if (
            kwargs.get("value_weight") is not None
            and not self.supports_mla_projected_value_decode
        ):
            raise ValueError("Batch-invariant MLA requires a separate value projection")
        metadata = self.forward_decode_metadata
        assert metadata is not None
        num_extends = metadata.num_extends
        q_len_per_req = q.shape[0] // bs if bs > 0 else 1

        window_left = int(getattr(layer, "sliding_window_size", -1) or -1)
        noncausal_block_size = self.spec_num_tokens if self.block_decode_active else 1

        if self.block_decode_active:
            if q_len_per_req == noncausal_block_size and self._takes_query_blocks(
                layer.tp_q_head_num, q_len_per_req, window_left >= 0
            ):
                # One page table row and one cache length per request, both
                # built once per forward. Valid only at the stride the
                # metadata was expanded with, which the width test above
                # pins: a narrower forward would read the next request's row.
                query = q.view(bs, q_len_per_req, layer.tp_q_head_num, layer.head_dim)
                page_table = metadata.block_page_table
                cache_seqlens = metadata.block_seq_lens
            else:
                # One metadata row per block position already, each at the
                # block-end length, so the causal offsets below would re-impose
                # the ordering the draft must not have. No extend offset
                # either: refresh_decode_metadata expanded rows [0, bs), which
                # is exactly what this query covers.
                query = q.view(-1, layer.tp_q_head_num, layer.head_dim).unsqueeze(1)
                page_table = metadata.page_table
                cache_seqlens = metadata.seq_lens
            max_seqlen_k = self.max_context_len
        elif (
            q_len_per_req > 1
            and q.dtype == self.data_type
            and layer.logit_cap == 0.0
            and self._takes_query_blocks(
                layer.tp_q_head_num, q_len_per_req, window_left >= 0
            )
        ):
            query = q.view(bs, q_len_per_req, layer.tp_q_head_num, layer.head_dim)
            page_table = metadata.page_table[num_extends:]
            cache_seqlens = metadata.seq_lens[num_extends:]
            if self.is_draft:
                # Draft catch-up stores the first query's visible KV length;
                # this kernel expects the last query's visible KV length.
                cache_seqlens = cache_seqlens + (q_len_per_req - 1)
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
                window_left=window_left,
                noncausal_block_size=noncausal_block_size,
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
                plan=self.decode_kernel_plan,
                window_left=window_left,
                noncausal_block_size=noncausal_block_size,
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
        **kwargs,
    ) -> torch.Tensor:
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
                plan=self.extend_kernel_plan,
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


class GluonMLAAttnBackend(MLAAttnBackend):
    """Gluon MLA with the shared cache placement, DCP metadata and LSE merge."""

    supports_mla_dcp = True

    def __init__(self, config: AttnConfig, spec: MLAConfig, *, kernel_page_size: int):
        self.kvp_group = config.kvp_group
        self.kvp_rank = config.kvp_rank
        self._dcp: DCPDecodeState | None = None
        if len(self.kvp_group) > 1:
            if global_server_args_dict["numerics"] in BITWISE_ENVELOPES:
                raise ValueError("Gluon MLA DCP does not support bitwise numerics")
            if config.kv_cache_dtype != torch.float8_e4m3fn or kernel_page_size != 64:
                raise ValueError("Gluon MLA DCP requires E4M3 KV and 64-token pages")
            if config.kv_cache_quant_method not in (None, "none"):
                raise ValueError("Gluon MLA DCP requires plain E4M3 cache rows")
            if (spec.kv_lora_rank, spec.qk_rope_head_dim) != (512, 64):
                raise ValueError("Gluon MLA DCP requires latent512 + RoPE64")
            if spec.sliding_window_tokens is not None:
                raise ValueError("Gluon MLA DCP requires full-history attention")
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        if len(self.kvp_group) > 1:
            # Value projection and its gate consume the globally merged latent.
            self.supports_mla_projected_value_decode = False
            self.supports_layer_sliding_window = False

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        super()._publish_cache_pool(cache_pool)
        self._dcp = None

    def configure_runtime(
        self,
        *,
        block_granularity: int,
        virtual_block_count: int,
        shard_count: int,
        **kwargs,
    ) -> None:
        super().configure_runtime(**kwargs)
        if shard_count != len(self.kvp_group):
            raise ValueError("MLA cache and KVP topology disagree")
        self._dcp = (
            DCPDecodeState(
                CachePlacement(
                    block_granularity,
                    virtual_block_count,
                    self.kvp_group,
                    self.kvp_rank,
                )
            )
            if shard_count > 1
            else None
        )

    def cache_placement(self, layer: PagedAttention) -> CachePlacement | None:
        return self._dcp.placement if self._dcp is not None else None

    def init_cuda_graph_state(self, max_bs: int) -> None:
        super().init_cuda_graph_state(max_bs)
        self._block_page_table_buf = None
        self._block_seq_lens_buf = None
        if self._dcp is not None:
            if self.block_decode_active:
                self._ensure_block_row_buffers(max_bs)
                table, lengths = self._block_page_table_buf, self._block_seq_lens_buf
            else:
                table, lengths = self.page_table_buf, self.seq_lens_buf
            self._dcp.initialize(
                table,
                lengths,
                page_size=self.kernel_page_size,
                query_count=self.spec_num_tokens,
                causal=not self.block_decode_active,
            )

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        super().advance_draft_forward_metadata(seq_lens)
        if self._dcp is not None:
            self._dcp.refresh_lengths(
                self._decode_views(seq_lens.shape[0]).dcp, seq_lens
            )

    def _should_use_absorbed_cached_extend(
        self, *, max_extend_seq_len: int, max_extend_prefix_len: int
    ) -> bool:
        if len(self.kvp_group) > 1:
            return False
        return super()._should_use_absorbed_cached_extend(
            max_extend_seq_len=max_extend_seq_len,
            max_extend_prefix_len=max_extend_prefix_len,
        )

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
        query_shard: QueryShardPlan | None,
        page_table_cpu: torch.Tensor | None,
        **kwargs,
    ) -> None:
        super().init_forward_metadata(
            bs,
            num_extends,
            seq_lens,
            page_table,
            forward_mode,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_with_prefix=extend_with_prefix,
            query_shard=query_shard,
            page_table_cpu=page_table_cpu,
            **kwargs,
        )
        if forward_mode.is_extend_or_mixed():
            metadata = self.chunked_prefill_metadata
            assert metadata.chunk_bounds_cpu is not None
            self._prepare_prefix_history(
                metadata.chunk_kv_indices_list,
                page_table_cpu=page_table_cpu,
                chunk_bounds_cpu=metadata.chunk_bounds_cpu,
            )

    def _decode_views(self, bs: int) -> MLADecodeMetadata:
        metadata = super()._decode_views(bs)
        if self._dcp is not None and metadata.dcp is None:
            metadata.dcp = self._dcp.metadata.slice_requests(0, bs)
        return metadata

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
        super().refresh_decode_metadata(
            bs,
            actual_bs,
            seq_lens,
            page_table,
            num_extends=num_extends,
            for_graph_replay=for_graph_replay,
        )
        if self._dcp is not None:
            metadata = self.forward_decode_metadata
            lengths = (
                metadata.block_seq_lens
                if self.block_decode_active
                else metadata.seq_lens
            )
            self._dcp.refresh(metadata.dcp, lengths)

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        super().fill_block_decode_seq_lens(bs, block_seq_lens)
        if self._dcp is not None:
            self._dcp.refresh_lengths(
                self._decode_views(bs).dcp, self._block_seq_lens_buf[:bs]
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        **kwargs,
    ) -> torch.Tensor:
        if self._dcp is None:
            return super().forward_decode(
                q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
            )
        if any(
            kwargs.get(name) is not None
            for name in ("value_weight", "output_gate", "projected_output")
        ):
            raise ValueError(
                "Gluon MLA DCP requires value projection after the LSE merge"
            )
        if bs == 0:
            return torch.empty(
                (0, layer.tp_q_head_num * layer.v_head_dim),
                dtype=self.dtype,
                device=q.device,
            )
        return self._forward_dcp_decode(q, layer, token_to_kv_pool, bs)

    def _forward_dcp_decode(
        self,
        q: torch.Tensor,
        layer: PagedAttention,
        token_to_kv_pool,
        bs: int,
    ) -> torch.Tensor:
        """Attend local pages using single-query/head chunks, then merge by LSE.

        This initial implementation deliberately rereads KV for each query and
        16-head chunk. Per-query local lengths preserve causal boundaries even
        when successive global positions belong to different page owners.
        """
        if layer.sliding_window_size >= 0:
            raise ValueError("Gluon MLA DCP requires full-history attention")
        metadata = self.forward_decode_metadata
        start = 0 if self.block_decode_active else metadata.num_extends
        local = metadata.dcp.slice_requests(start, start + bs)
        width = q.shape[0] // bs
        if width not in (1, self.spec_num_tokens):
            raise ValueError(
                "Gluon MLA DCP requires one query or the full verify window"
            )
        lengths = (
            local.local_seq_lens[:, None] if width == 1 else local.local_visible_lens
        ).reshape(-1)
        table = local.local_page_table.repeat_interleave(width, dim=0)
        query = gather_query_heads(
            q.reshape(-1, layer.tp_q_head_num, layer.head_dim), self.kvp_group
        )
        if query.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # FP8 probability rounding depends on each shard's local maximum.
            # Preserve the quantized query values exactly while using the
            # existing BF16-query kernel for this correctness-first composition.
            query = query.to(torch.bfloat16)
        cache = token_to_kv_pool.get_key_buffer(layer.layer_id)
        if cache.dtype != self.data_type:
            cache = cache.to(self.data_type)
        cache = cache.view(-1, self.kernel_page_size, 1, self.kv_cache_dim)
        partials, lses = [], []
        for head in range(0, query.shape[1], 16):
            output, lse = mla_decode_with_kvcache(
                q=query[:, None, head : head + 16].contiguous(),
                kv_cache=cache,
                page_table=table,
                cache_seqlens=lengths,
                max_seqlen_k=self.max_context_len,
                qk_nope_head_dim=self.qk_nope_head_dim,
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
                softmax_scale=layer.scaling,
                logit_cap=layer.logit_cap,
                return_lse=True,
                solution=self.kernel_solution,
            )
            partials.append(
                output.reshape(-1, min(16, query.shape[1] - head), self.kv_lora_rank)
            )
            lses.append(lse.reshape(-1, min(16, query.shape[1] - head)))
        partial, lse = normalize_dcp_partials(
            torch.cat(partials, dim=1), torch.cat(lses, dim=1), lengths, None
        )
        output = combine_attention_partials(
            partial,
            lse,
            group=self.kvp_group,
            rank=self.kvp_rank,
            sink=None,
            keep_all_heads=False,
        )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)


register_backend("mla", {AttentionArch.MLA}, MLAAttnBackend)
register_backend("gluon", {AttentionArch.MLA}, GluonMLAAttnBackend)
