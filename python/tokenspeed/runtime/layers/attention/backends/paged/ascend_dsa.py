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

"""Ascend implementation of the paged DSA backend."""

from __future__ import annotations

import dataclasses

import torch
from tokenspeed_kernel.ops.attention.dsa.ascend import (
    ascend_dsa_kernels,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.ascend_qcp_prefill import (
    AscendQCPPrefill,
    AscendQCPPrefillBackend,
    prefill_history_group,
)
from tokenspeed.runtime.layers.attention.backends.paged.ascend_dsa_local_decode import (
    AscendDSALocalDecode,
)
from tokenspeed.runtime.layers.attention.backends.paged.ascend_dsa_tp_kvp_decode import (
    AscendDSATPKVPDecode,
    _AscendCachePlacement,
)
from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.registry import register_backend

# Fixed page geometry of Ascend SparseFlashAttentionDecode.
ASCEND_SFAD_PAGE_SIZE = 128


class AscendDSABackend(AscendQCPPrefillBackend, DSABackend):
    """Ascend DSA execution backed by Ascend indexer and sparse-attention ops."""

    # The decode path owns its DCP page placement and query exchange; the
    # generic MLA gate must accept this backend when attn_tp == dcp_size.
    supports_mla_dcp = True
    default_kernel_page_size = ASCEND_SFAD_PAGE_SIZE

    @classmethod
    def resolve_kernel_page_size(cls, config, block_granularity: int) -> int:
        del block_granularity
        if config.kernel_page_size not in (None, ASCEND_SFAD_PAGE_SIZE):
            raise ValueError(
                "Ascend SparseFlashAttentionDecode requires "
                f"kernel_page_size={ASCEND_SFAD_PAGE_SIZE}, got "
                f"{config.kernel_page_size}"
            )
        return ASCEND_SFAD_PAGE_SIZE

    @staticmethod
    def _make_dense_leaf(config, spec, platform, kernel_page_size):
        del platform
        dense_spec = dataclasses.replace(spec, backend_name=None)
        return MLAAttnBackend(config, dense_spec, kernel_page_size=kernel_page_size)

    def __init__(
        self,
        config,
        spec,
        *,
        kernel_page_size: int,
    ) -> None:
        if not (spec.is_dsa and spec.index_k_format == "bf16"):
            raise NotImplementedError("Ascend DSA currently requires BF16 index keys")
        if kernel_page_size != ASCEND_SFAD_PAGE_SIZE:
            raise ValueError(
                "Ascend SparseFlashAttentionDecode requires "
                f"kernel_page_size={ASCEND_SFAD_PAGE_SIZE}, got {kernel_page_size}"
            )
        if config.kv_cache_dtype != torch.bfloat16:
            raise ValueError("Ascend DSA requires BF16 KV cache")
        if config.kv_cache_quant_method not in (None, "none"):
            raise ValueError("Ascend DSA cache format is selected by kv_cache_dtype")
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        if self.is_draft:
            raise ValueError("Ascend DSA draft attention is disabled")
        if self.spec_num_tokens > 8:
            raise ValueError("Ascend DSA verify supports at most 8 tokens")
        self._indexer_spec = spec
        self._kvp_degree = int(config.kvp_size)
        self._kvp_ranks = tuple(config.kvp_group)
        self._ascend_placement = None
        if self._kvp_degree == 1:
            self._decode = AscendDSALocalDecode(
                config, spec, page_size=kernel_page_size
            )
        else:
            if spec.attn_tp_size != self._kvp_degree:
                raise ValueError("DSA KV sharding must span its attention group")
            self._decode = AscendDSATPKVPDecode(
                config, spec, page_size=kernel_page_size
            )
        self._indexer_kernels = ascend_dsa_kernels()
        self._prefill: AscendQCPPrefill | None = None
        self._prefill_layout = None

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        self._prefill = None
        self._ascend_placement = None
        super()._publish_cache_pool(cache_pool)
        self._prefill_layout = None

    def _create_qcp_prefill(self, layout, layer_id):
        kv = self.cache_pool.get_key_buffer(layer_id)
        index = self.cache_pool.get_component(layer_id, "dsa_index_k")
        runtime = self._indexer_kernels.prepare_qcp_prefill(
            kv,
            index,
            ranks=self._kvp_ranks,
            history_group=prefill_history_group(self._kvp_ranks),
            layout=layout,
        )
        return AscendQCPPrefill(
            layout=layout,
            page_size=self.kernel_page_size,
            topk=self._indexer_spec.index_topk,
            runtime=runtime,
        )

    def configure_runtime(self, **kwargs):
        super().configure_runtime(**kwargs)
        if self._kvp_degree > 1:
            self._decode.bind_cache(super().cache_placement(None))

    def cache_placement(self, layer) -> CachePlacement | None:
        if self._ascend_placement is None:
            placement = super().cache_placement(layer)
            if placement is not None:
                self._ascend_placement = _AscendCachePlacement(
                    placement.block_granularity,
                    placement.virtual_block_count,
                    placement.group,
                    placement.rank,
                    self.prepare_prolog_metadata,
                )
        return self._ascend_placement

    def init_cuda_graph_state(self, max_bs: int) -> None:
        super().init_cuda_graph_state(max_bs)
        self._decode.allocate_decode_buffers(
            max_bs * self.spec_num_tokens, self.max_num_pages, self.device
        )

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
        # MLA refreshes stable page-table/length buffers for eager and graph
        # Decode. DSA then consumes page_table, seq_lens, num_extends and
        # q_len_per_req to build its local or TP/KVP query metadata.
        self._dense_backend.refresh_decode_metadata(
            bs,
            actual_bs,
            seq_lens,
            page_table,
            num_extends=num_extends,
            for_graph_replay=for_graph_replay,
        )
        self._decode.refresh(self.forward_decode_metadata)

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
                "DSA decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend/mixed ({forward_mode})"
            )
        # QCP preparation consumes host extend/prefix lengths for partitioning
        # and gather planning, plus device cum_extend_seq_lens, prefix lengths
        # and page_table for attention addresses and causal query/KV boundaries.
        self._dense_backend.init_forward_metadata(
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
            **kwargs,
        )
        self._prefill_page_table = None
        if num_extends > 0 and forward_mode.is_extend_or_mixed():
            cmeta = self._dense_backend.chunked_prefill_metadata
            if cmeta is not None:
                # MLA may leave this unset when it expects dense gathered KV.
                # DSA needs the original request pages to plan QCP workspace KV.
                self._prefill_page_table = page_table[:num_extends]
                cmeta.page_table = self._prefill_page_table
        if bs > num_extends:
            self._decode.refresh(self.forward_decode_metadata)
        if num_extends > 0 and self.prefill_enabled:
            self._prefill.prepare_batch_metadata(
                self.forward_prefill_metadata,
                self.cache_placement(None),
                page_table_cpu=kwargs.get("page_table_cpu"),
            )

    def prepare_prolog_metadata(self, locations, *, init_tokens, local_tokens):
        return self._decode.prepare_prolog_metadata(
            locations,
            self.forward_decode_metadata,
            init_tokens=init_tokens,
            local_tokens=local_tokens,
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        *,
        fused: bool,
        **kwargs,
    ) -> torch.Tensor:
        self._validate_logit_cap(layer.logit_cap)
        if fused:
            raise ValueError("Native fused decode queries are not Prefill inputs")
        initial, window = kwargs["index_init_tokens"], kwargs["index_local_tokens"]
        if (
            initial < 0
            or window < 0
            or initial + window > self._indexer_spec.index_topk
        ):
            raise ValueError("Prefill init/local token counts must fit index_topk")
        head_major_output = kwargs.pop("head_major_output", False)
        spec, prefill = self._indexer_spec, self._prefill
        if prefill is None or not prefill.row_counts:
            raise ValueError(
                "DSA Prefill requires initialized workspace and batch metadata"
            )
        if q.shape[:2] != (prefill.row_counts[prefill.rank], spec.num_attention_heads):
            raise ValueError(
                "Prefill needs token-local queries with all attention heads"
            )
        prefill_kv, prefill_index = prefill.prepare_layer_kv(
            self.cache_pool.get_key_buffer(layer.layer_id),
            self.cache_pool.get_component(layer.layer_id, "dsa_index_k"),
            kwargs["prefill_latent"].contiguous(),
            kwargs["index_key"].contiguous(),
            out_cache_loc,
        )
        if self.step_counter is not None:
            self.step_counter.record_cache()
        if q.shape[0]:
            index_pages = prefill_index.view(
                -1, prefill.page_size, 1, spec.index_head_dim
            )
            indices = self._indexer_kernels.prefill_index(
                kwargs["index_query"],
                index_pages,
                kwargs["index_weights"],
                prefill.query_cu,
                prefill.key_cu,
                prefill.page_table,
                spec.index_topk,
                kwargs["index_init_tokens"],
                kwargs["index_local_tokens"],
            )
            kv_pages = prefill_kv.view(-1, prefill.page_size, spec.kv_cache_dim)
            output = self._indexer_kernels.prefill_attention(
                q,
                kv_pages,
                indices,
                prefill.scan_count,
                prefill.page_table,
                prefill.query_seq,
                layer.scaling,
            )
        else:
            output = q.new_empty((q.shape[1], 0, spec.kv_lora_rank))
        return (
            output
            if head_major_output
            else output.transpose(0, 1).contiguous().flatten(1)
        )

    def forward_extend_chunked(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "Ascend DSA does not implement chunked prefix-replay prefill"
        )

    def forward_sparse_prefill(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "Ascend DSA does not accept externally selected sparse prefill"
        )

    def forward_sparse_decode(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "Ascend DSA does not accept externally selected sparse decode"
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        topk_indices: torch.Tensor | None = None,
        topk_lens: torch.Tensor | None = None,
        *,
        fused: bool,
        **kwargs,
    ) -> torch.Tensor:
        self._validate_logit_cap(layer.logit_cap)
        if self._kvp_degree == 1 and self._indexer_spec.attn_tp_size != 1:
            raise ValueError("Local DSA Decode requires TP1/KVP1")
        if bs <= 0 or q.shape[0] % bs:
            raise ValueError("Ascend DSA decode tokens must divide by requests")
        tokens_per_request = q.shape[0] // bs
        if tokens_per_request not in (1, self.spec_num_tokens):
            raise ValueError(
                "Ascend DSA decode requires one token or a complete verify "
                "window per request"
            )
        if topk_indices is not None or topk_lens is not None:
            raise ValueError(
                "Ascend DSA expects indexer projections, not global-slot TopK"
            )
        initial = kwargs["index_init_tokens"]
        window = kwargs["index_local_tokens"]
        if initial < 0 or window < 0:
            raise ValueError("DSA initial/local token counts must be nonnegative")
        if initial + window > self._indexer_spec.index_topk:
            raise ValueError("DSA initial/local tokens exceed index_topk")
        output = self._decode.forward(
            q,
            k,
            v,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            bs,
            fused=fused,
            metadata=self.forward_decode_metadata,
            step_counter=self.step_counter,
            **kwargs,
        )
        return output


if current_platform().is_npu:
    register_backend(
        "dsa",
        {AttentionArch.DSA},
        AscendDSABackend,
    )
