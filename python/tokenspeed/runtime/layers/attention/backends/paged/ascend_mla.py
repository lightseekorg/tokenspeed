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

"""Ascend dense MLA over the host's owner-sharded KV cache."""

from __future__ import annotations

from dataclasses import replace

import torch
from tokenspeed_kernel.ops.attention.mla.ascend import (
    ascend_mla_dcp_kernels,
    prepare_ascend_mla_prefill,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.layers.attention.backends.paged.mla import (
    MLAAttnBackend,
    MLAPrefillMetadata,
)
from tokenspeed.runtime.layers.attention.dcp.ascend import (
    AscendDCPContext,
    AscendPrefillCache,
    AscendPrefillPlan,
)
from tokenspeed.runtime.layers.attention.dcp.metadata import (
    CompactDCPLayout,
    refresh_dcp_page_table_metadata,
)
from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement
from tokenspeed.runtime.layers.attention.registry import register_backend


def ascend_mla_prefill_backend(backend):
    """Find the MLA leaf without adding prefetch hooks to shared routers."""
    pending = [backend]
    leaves = []
    while pending:
        node = pending.pop()
        if isinstance(node, AscendMLABackend):
            leaves.append(node)
        else:
            pending.extend(node.child_backends())
    if len(leaves) != 1:
        raise ValueError("MLA Prefill needs exactly one MLA cache leaf")
    return leaves[0]


def mla_prefill_spans(extends, prefixes, *, first, rows):
    """Return local query slices and their causal starting KV positions."""
    if len(extends) != len(prefixes) or min([first, rows, *extends, *prefixes]) < 0:
        raise ValueError("MLA Prefill lengths and query partition are invalid")
    if first + rows > sum(extends):
        raise ValueError("MLA Prefill query partition exceeds the extend rows")
    offset = 0
    spans = []
    for extent, prefix in zip(extends, prefixes, strict=True):
        begin, end = max(first, offset), min(first + rows, offset + extent)
        if begin < end:
            spans.append((begin - first, end - first, prefix + begin - offset))
        offset += extent
    return tuple(spans)


class AscendMLABackend(MLAAttnBackend):
    supports_mla_dcp = True
    supports_mla_projected_value_decode = False
    supports_layer_sliding_window = False
    default_kernel_page_size = 128

    def __init__(self, config, spec, *, kernel_page_size):
        kernels = ascend_mla_dcp_kernels()
        self._lengths_class = kernels.LocalLengths
        self._merge_partials = kernels.merge_dcp_partials
        self._mla_partial = kernels.mla_partial
        if config.is_draft or config.draft_block_decode:
            raise NotImplementedError(
                "NPU MLA DCP currently supports target decode/chain verify"
            )
        if (
            config.kv_cache_dtype != torch.bfloat16
            or config.kv_cache_quant_method not in (None, "none")
        ):
            raise ValueError("NPU MLA DCP requires an unquantized BF16 cache")
        if (spec.kv_lora_rank, spec.qk_rope_head_dim) != (512, 64):
            raise ValueError("NPU MLA DCP requires latent512 + auxiliary64")
        if kernel_page_size not in (64, 128):
            raise ValueError("NPU MLA DCP requires kernel pages of 64 or 128 tokens")
        if config.kvp_size > 1 and config.kvp_size != spec.attn_tp_size:
            raise ValueError("NPU MLA DCP must span the full attention TP group")
        super().__init__(
            config, replace(spec, backend_name=None), kernel_page_size=kernel_page_size
        )
        if config.kvp_size not in (1, 8):
            raise ValueError("Ascend MLA native merge supports DCP1/8")
        self._dcp = AscendDCPContext(
            degree=int(config.kvp_size),
            rank=int(config.kvp_rank),
            ranks=tuple(config.kvp_group),
            auxiliary_namespace="mla_dcp_aux",
        )
        self._placement = None
        self._local_views = {}
        self._page_metadata = {}
        self._current_local = None
        self._prefill_runtime = None
        self._prefill_plan: AscendPrefillPlan | None = None
        self._prefill_spans: tuple[tuple[int, int, int], ...] = ()
        self._prefill_pages: torch.Tensor | None = None
        self._prefill_options = None

    @property
    def native_prefill_enabled(self):
        return self._prefill_options is not None

    def _publish_cache_pool(self, cache_pool):
        self.close_prefill()
        super()._publish_cache_pool(cache_pool)
        self._prefill_options = (
            cache_pool.prefill_options
            if isinstance(cache_pool, AscendPrefillCache)
            else None
        )
        self._page_metadata.clear()
        self._local_views.clear()
        self._current_local = None
        self._placement = None
        self._prefill_plan = None
        self._prefill_pages = None
        self._prefill_spans = ()
        self.prepare_prefill_workspace()

    def prepare_prefill_workspace(self):
        if not self.native_prefill_enabled:
            return 0
        if self._prefill_runtime is not None:
            return self._prefill_runtime.workspace_bytes
        if self.cache_pool is None:
            raise RuntimeError("Bind the MLA cache before preparing Prefill")
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager,
        )

        layer_id = min(self.cache_pool.history_group_by_layer())
        kv = self.cache_pool.get_key_buffer(layer_id)
        self._prefill_options.check_workspace(
            local_rows=kv.numel() // 576, degree=self._dcp.degree, index_width=0
        )
        self._prefill_runtime = prepare_ascend_mla_prefill(
            kv,
            ranks=self._dcp.ranks,
            bootstrap_group=process_group_manager.get_process_group(
                "gloo", self._dcp.ranks
            ),
            max_current_rows=self._prefill_options.max_current_rows,
        )
        return self._prefill_runtime.workspace_bytes

    def _init_prefill_metadata(self, **kwargs):
        if not self.native_prefill_enabled:
            return super()._init_prefill_metadata(**kwargs)
        extends = kwargs["extend_seq_lens_cpu"].tolist()
        prefixes = kwargs["extend_prefix_lens_cpu"].tolist()
        cumulative = torch.zeros(
            len(extends) + 1, dtype=torch.int32, device=kwargs["seq_lens"].device
        )
        torch.cumsum(kwargs["extend_seq_lens"], 0, out=cumulative[1:])
        metadata = MLAPrefillMetadata(
            **kwargs,
            cum_extend_seq_lens=cumulative,
            cum_seq_lens_kv=None,
            max_extend_seq_len=max(extends, default=0),
            max_extend_prefix_len=max(prefixes, default=0),
            use_absorbed_cached_extend=False,
            chunked_loop_num=0,
            chunk_kv_indices_list=[],
            chunked_seq_len=kwargs["seq_lens"][:0],
            cu_chunked_seq_len=cumulative[:0],
            max_chunk_len_per_loop=[],
        )
        if self._prefill_runtime is None or self._placement is None:
            raise RuntimeError("Prepare MLA Prefill cache and geometry before metadata")
        plan = AscendPrefillPlan.build(
            metadata,
            self._placement,
            degree=self._dcp.degree,
            rank=self._dcp.rank,
            page_size=self.kernel_page_size,
            topk=1,
            local_cache_pages=self._prefill_runtime.local_rows // self.kernel_page_size,
            query_partition=(0, sum(extends)),
        )
        self._prefill_plan = plan
        self._prefill_pages = plan.page_table.long()
        self._prefill_spans = mla_prefill_spans(
            extends, prefixes, first=plan.row_start, rows=plan.row_end - plan.row_start
        )
        self.forward_prefill_metadata = metadata
        self.chunked_prefill_metadata = metadata

    def prefetch_prefill_layer(self, layer_id):
        if self._prefill_runtime is None or self._prefill_plan is None:
            raise RuntimeError("Prepare MLA Prefill workspace and metadata first")
        self._prefill_runtime.prefetch(
            layer_id,
            self.cache_pool.get_key_buffer(layer_id),
            None,
            has_history=not self._prefill_plan.cold,
        )

    def start_prefill_kv(self, layer_id, latent):
        if self._prefill_runtime is None or self._prefill_plan is None:
            raise RuntimeError("Prepare MLA Prefill before producing current KV")
        self._prefill_runtime.gather_current(
            layer_id, latent, width=576, total_rows=sum(self._prefill_plan.row_counts)
        )

    def consume_prefill_kv(self, layer_id, slots):
        if self._prefill_runtime is None or self._prefill_plan is None:
            raise RuntimeError("Prepare MLA Prefill before consuming KV")
        self._prefill_runtime.patch(
            layer_id,
            slots,
            self._prefill_plan.current_direct,
            width=576,
            block_rows=self._placement.block_granularity,
            virtual_blocks=self._placement.virtual_block_count,
        )

    def read_prefill_rows(self, sequence, begin, end):
        """Read a bounded logical span; the mapped table is prepared on device."""
        pages = self._prefill_pages[
            sequence, begin // self.kernel_page_size : -(-end // self.kernel_page_size)
        ]
        cache = self._prefill_runtime.kv.view(-1, self.kernel_page_size, 576)
        rows = cache.index_select(0, pages).flatten(0, 1)
        return rows[
            begin % self.kernel_page_size : begin % self.kernel_page_size + end - begin
        ]

    def finish_prefill_layer(self, layer_id):
        self._prefill_runtime.finish(layer_id)

    def close_prefill(self):
        if self._prefill_runtime is not None:
            self._prefill_runtime.close()
            self._prefill_runtime = None

    def configure_runtime(
        self, *, block_granularity, virtual_block_count, shard_count, **kwargs
    ):
        super().configure_runtime(**kwargs)
        if shard_count != self._dcp.degree:
            raise ValueError("MLA cache sharding and DCP group disagree")
        self._placement = CachePlacement(
            block_granularity, virtual_block_count, self._dcp.ranks, self._dcp.rank
        )

    def cache_placement(self, layer):
        if self._dcp.degree == 1:
            return None
        if self._placement is None:
            raise RuntimeError("MLA DCP cache geometry was not configured")
        return self._placement

    def _takes_query_blocks(self, *args):
        return False

    def _should_use_absorbed_cached_extend(self, **kwargs):
        # Expanded prefill reads sharded history through read_prefix_history.
        return False

    def read_prefix_history(self, pool, layer, chunk_kv_indices_list, *, dst_dtype):
        if not chunk_kv_indices_list:
            return iter(())
        if self._dcp.degree == 1:
            return super().read_prefix_history(
                pool, layer, chunk_kv_indices_list, dst_dtype=dst_dtype
            )
        placement = self.cache_placement(layer)
        rows = self._dcp.gather_cache_rows(
            pool.get_key_buffer(layer.layer_id),
            chunk_kv_indices_list,
            block_granularity=placement.block_granularity,
            virtual_block_count=placement.virtual_block_count,
        )
        return (
            (
                kv[..., : self.kv_lora_rank].to(dst_dtype).contiguous(),
                kv[..., self.kv_lora_rank :].to(dst_dtype).contiguous(),
            )
            for kv in rows
        )

    def refresh_decode_metadata(
        self,
        bs,
        actual_bs,
        seq_lens,
        page_table,
        *,
        num_extends=0,
        for_graph_replay=False,
    ):
        super().refresh_decode_metadata(
            bs,
            actual_bs,
            seq_lens,
            page_table,
            num_extends=num_extends,
            for_graph_replay=for_graph_replay,
        )
        if self._placement is None:
            raise RuntimeError("MLA DCP cache geometry was not configured")
        metadata = self.forward_decode_metadata
        width = metadata.q_len_per_req
        # Padding rows must have no history even though the host clamps lengths
        # to the verify floor. Their page table is already all-null.
        key = (bs, num_extends, width)
        seq_lens = metadata.seq_lens[num_extends:]
        endpoints = (
            seq_lens[:, None].to(torch.int64)
            + torch.arange(1 - width, 1, device=seq_lens.device)
        ).clamp_min(0)
        local = refresh_dcp_page_table_metadata(
            page_table=metadata.page_table[num_extends:],
            virtual_block_count=self._placement.virtual_block_count,
            degree=self._dcp.degree,
            rank=self._dcp.rank,
            layout=CompactDCPLayout(
                seq_lens=seq_lens,
                page_size=self.kernel_page_size,
                block_granularity=self._placement.block_granularity,
                visible_lens=endpoints,
            ),
            previous=self._page_metadata.get(key),
        )
        self._page_metadata[key] = local
        previous = self._local_views.get(key)
        table = local.local_page_table.repeat_interleave(width, dim=0)
        if previous is None:
            previous = (
                table,
                self._lengths_class(local.local_visible_lens),
                local.local_visible_lens,
            )
            self._local_views[key] = previous
        else:
            previous[0].copy_(table)
            previous[1].refresh(local.local_visible_lens)
        self._current_local = previous

    def forward_decode(
        self, q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
    ):
        if layer.logit_cap or int(getattr(layer, "sliding_window_size", -1) or -1) >= 0:
            raise NotImplementedError("NPU MLA DCP does not support logit caps/windows")
        if kwargs.get("value_weight") is not None:
            raise ValueError("MLA value projection must follow the global DCP merge")
        if self._current_local is None:
            raise RuntimeError("MLA DCP decode metadata was not refreshed")
        table, lengths, visible = self._current_local
        if q.shape[0] != table.shape[0]:
            raise ValueError("MLA query rows disagree with refreshed verify width")
        query = self._dcp.gather_query(q)
        cache = token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1, self.kernel_page_size, 1, self.kv_cache_dim
        )
        output, lse = self._mla_partial(query, cache, table, lengths, layer.scaling)
        empty = visible.flatten()[:, None] == 0
        lse = torch.where(empty, -torch.inf, lse)
        return self._dcp.combine(output, lse, merge=self._merge_partials).flatten(1)


if current_platform().is_npu:
    register_backend("tokenspeed_mla", {AttentionArch.MLA}, AscendMLABackend)
