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
from collections.abc import Callable
from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.attention.dsa.ascend import (
    ascend_dsa_kernels,
)
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.ascend_prefill import (
    AscendPrefill,
    AscendPrefillCache,
)
from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
)
from tokenspeed.runtime.layers.attention.dsa.ascend_kvp_decode import (
    AscendDSATPKVPDecode,
)
from tokenspeed.runtime.layers.attention.dsa.ascend_local_decode import (
    AscendDSALocalDecode,
)
from tokenspeed.runtime.layers.attention.dsa.ascend_prefill import run_prefill_attention
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.registry import register_backend

# Fixed page geometry of Ascend SparseFlashAttentionDecode.
ASCEND_SFAD_PAGE_SIZE = 128


def ascend_dsa_prefill_backend(backend):
    """Find the DSA leaf without adding model hooks to shared router nodes."""
    if isinstance(backend, DSABackend):
        return backend
    leaves = []
    pending = list(backend.child_backends())
    while pending:
        node = pending.pop()
        if isinstance(node, DSABackend):
            leaves.append(node)
        else:
            pending.extend(node.child_backends())
    if len(leaves) != 1:
        raise ValueError("DSA Prefill needs exactly one DSA cache leaf")
    return leaves[0]


@dataclass(frozen=True)
class _AscendCachePlacement(CachePlacement):
    # The placement owner prepares native DSA inputs before the first Prolog.
    prepare_decode_metadata: Callable | None = None


def prepare_ascend_dsa_cp_metadata(
    locations, placement, *, initial_tokens, window_tokens
):
    """Prepare the leaf's CP metadata before Prolog through its placement owner."""
    if (
        not isinstance(placement, _AscendCachePlacement)
        or placement.prepare_decode_metadata is None
    ):
        raise ValueError("Native DSA requires Ascend CP cache placement")
    slots = placement.prepare_decode_metadata(
        locations,
        init_tokens=initial_tokens,
        local_tokens=window_tokens,
    )
    return slots


class AscendDSABackend(DSABackend):
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
            if spec.attn_tp_size != 1:
                raise ValueError("Local DSA Decode requires TP1/KVP1")
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
        self._prefill: AscendPrefill | None = None
        self._prefill_group = None
        self._prefill_options = None

    @property
    def prefill_enabled(self):
        return self._prefill_options is not None

    @property
    def prefill_needs_host_page_table(self):
        return self.prefill_enabled

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        self._prefill = None
        self._ascend_placement = None
        super()._publish_cache_pool(cache_pool)
        self._prefill_options = (
            cache_pool.prefill_options
            if isinstance(cache_pool, AscendPrefillCache)
            else None
        )
        # Every world rank creates the cache-owner groups before any forward.
        if self.prefill_enabled and self._kvp_degree > 1:
            from tokenspeed.runtime.distributed.process_group_manager import (
                process_group_manager,
            )

            self._prefill_group = process_group_manager.get_dedicated_device_group(
                self._kvp_ranks, "prefill-history"
            )
        else:
            self._prefill_group = None

    @property
    def prefill_kv_capacity(self) -> int:
        return self._prefill_options.max_history_rows if self.prefill_enabled else 0

    def preallocate_prefill_workspace(self) -> int:
        if not self.prefill_enabled:
            return 0
        if self._prefill is not None:
            return self._prefill.runtime.workspace_bytes
        if self.cache_pool is None:
            raise RuntimeError("Bind the DSA cache before preparing Prefill")
        layer_id = min(self.cache_pool.history_group_by_layer())
        kv = self.cache_pool.get_key_buffer(layer_id)
        index = self.cache_pool.get_component(layer_id, "dsa_index_k")
        runtime = self._indexer_kernels.prepare_prefill(
            kv,
            index,
            ranks=self._kvp_ranks,
            history_group=self._prefill_group,
            max_current_rows=self._prefill_options.max_current_rows,
            max_history_rows=self._prefill_options.max_history_rows,
            history_null_rows=self._prefill_options.kv_page_size,
        )
        self._prefill = AscendPrefill(
            options=self._prefill_options,
            page_size=self.kernel_page_size,
            topk=self._indexer_spec.index_topk,
            runtime=runtime,
        )
        return runtime.workspace_bytes

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
                self._prefill_page_table = page_table[:num_extends]
                cmeta.page_table = self._prefill_page_table
        if bs > num_extends:
            self._decode.refresh(self.forward_decode_metadata)
        if num_extends > 0 and self.prefill_enabled:
            self._prefill.prepare(
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
        if q.shape[:2] != (prefill.row_counts[prefill.rank], spec.num_attention_heads):
            raise ValueError(
                "Prefill needs token-local queries with all attention heads"
            )
        prefill_kv, prefill_index = prefill.prepare_layer(
            self.cache_pool.get_key_buffer(layer.layer_id),
            self.cache_pool.get_component(layer.layer_id, "dsa_index_k"),
            kwargs["prefill_latent"].contiguous(),
            kwargs["index_key"].contiguous(),
            out_cache_loc,
        )
        if self.step_counter is not None:
            self.step_counter.record_cache()
        output = run_prefill_attention(
            q,
            layer,
            prefill,
            prefill_kv,
            prefill_index,
            kernels=self._indexer_kernels,
            spec=spec,
            **kwargs,
        )
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
