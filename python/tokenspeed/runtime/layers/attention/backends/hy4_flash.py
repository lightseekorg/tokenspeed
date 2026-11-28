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

"""Route Hy4-flash compressed-DSA and SWA-MLA groups through paged leaves."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from tokenspeed_kernel.ops.attention import apply_attention_sink
from tokenspeed_kernel.ops.attention.mla import mla_prefill
from tokenspeed_kernel.platform import ArchVersion, current_platform

from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
from tokenspeed.runtime.layers.attention.backends.paged.mla import (
    MLAAttnBackend,
    MLAPrefillMetadata,
)
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.backends.support import CudaGraphSupport
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.hy4_flash import (
    Hy4FlashAttentionConfig,
)
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_hy4_flash import (
    HybridHy4FlashTokenToKVPool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.hy4_flash import (
    HY4_FLASH_LOGICAL_BLOCK_TOKENS,
)


class Hy4FlashCacheGroupRouter(CacheGroupRouter):
    """Cache-group router for Hy4-flash DSA + SWA layers plus DFly2 index sharing."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._hy4_shared_topk_slots: dict[int, torch.Tensor] = {}

    def leaf_for(self, layer) -> Hy4FlashDSABackend | Hy4FlashSWABackend:
        """Return the paged leaf serving ``layer``'s bound cache group."""
        return self._leaf_for(layer)

    def dsa_prefill_max_context_len(self, num_extends: int) -> int | None:
        """Return the host-known max DSA context length across ``num_extends`` requests."""
        if num_extends <= 0:
            return 0
        dsa_leaf = self.leaves.get("dsa")
        metadata = getattr(dsa_leaf, "chunked_prefill_metadata", None)
        if metadata is None:
            return None
        prefix_cpu = getattr(metadata, "extend_prefix_lens_cpu", None)
        extend_cpu = getattr(metadata, "extend_seq_lens_cpu", None)
        if prefix_cpu is None or extend_cpu is None:
            return None
        return max(
            (
                int(p) + int(e)
                for p, e in zip(
                    prefix_cpu[:num_extends].tolist(),
                    extend_cpu[:num_extends].tolist(),
                    strict=True,
                )
            ),
            default=0,
        )

    def record_shared_topk_slots(self, layer_id: int, slots: torch.Tensor) -> None:
        """Record the latest global KV slots selected by DSA layer ``layer_id``."""
        self._hy4_shared_topk_slots[int(layer_id)] = slots

    def get_shared_topk_slots(self, layer_id: int) -> torch.Tensor | None:
        """Return the most recently recorded global KV slots for ``layer_id``."""
        return self._hy4_shared_topk_slots.get(int(layer_id))

    def clear_shared_topk_slots(self) -> None:
        self._hy4_shared_topk_slots.clear()

    def clear_shared_topk_slots_for_layer(self, layer_id: int) -> None:
        self._hy4_shared_topk_slots.pop(int(layer_id), None)


class Hy4FlashDSABackend(DSABackend):
    """DSA leaf for Hy4-flash: Hopper-safe dense delegate + Hy4 tail slot export."""

    def __init__(
        self,
        config: AttnConfig,
        spec: Hy4FlashAttentionConfig,
        *,
        kernel_page_size: int,
    ) -> None:
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        # Keep decode CUDA graphs enabled while running dynamic 4:1 prefill pooling eagerly.
        self.cuda_graph_support = CudaGraphSupport(prefill_graph=False)

    @staticmethod
    def _make_dense_leaf(config, spec, platform, kernel_page_size):
        # TRTLLM dense MLA requires Blackwell (SM100+); Hopper (SM90) uses Triton MLA.
        if platform.is_nvidia and platform.arch_version < ArchVersion(10, 0):
            leaf = MLAAttnBackend(
                config,
                replace(spec, backend_name="mla"),
                kernel_page_size=kernel_page_size,
            )
            leaf.kernel_solution = "triton"
            return leaf
        return DSABackend._make_dense_leaf(config, spec, platform, kernel_page_size)

    def slot_state_rows(self, slot: int) -> list[torch.Tensor]:
        if self.is_draft:
            return []
        if isinstance(self.cache_pool, HybridHy4FlashTokenToKVPool):
            return self.cache_pool.kpool_tail_slot_rows(slot)
        return super().slot_state_rows(slot)


@dataclass(kw_only=True)
class Hy4FlashSWAPrefillMetadata(MLAPrefillMetadata):
    prefix_query_start_offsets: tuple[torch.Tensor, ...]


class Hy4FlashSWABackend(MLAAttnBackend):
    """64-head SWA-MLA leaf with bounded prefix replay, sink support, and SM90 q_len>=1 verify."""

    default_kernel_page_size = HY4_FLASH_LOGICAL_BLOCK_TOKENS

    def __init__(
        self,
        config: AttnConfig,
        spec: MLAConfig,
        *,
        kernel_page_size: int,
    ) -> None:
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        if (
            not isinstance(spec.sliding_window_tokens, int)
            or spec.sliding_window_tokens < 1
        ):
            raise ValueError("Hy4-flash SWA requires a positive sliding_window_tokens")
        self.window_left = spec.sliding_window_tokens - 1
        platform = current_platform()
        if platform.is_nvidia and platform.arch_version < ArchVersion(10, 0):
            # Triton MLA decode supports sliding_window + q_len >= 1 + return_lse on SM90.
            self.kernel_solution = "triton"

    def _should_use_absorbed_cached_extend(
        self, *, max_extend_seq_len: int, max_extend_prefix_len: int
    ) -> bool:
        del max_extend_seq_len, max_extend_prefix_len
        return False

    def _init_prefill_metadata(
        self,
        seq_lens,
        extend_prefix_lens,
        extend_prefix_lens_cpu,
        extend_seq_lens,
        extend_seq_lens_cpu,
        page_table,
        page_table_cpu,
        *,
        extend_with_prefix,
    ):
        del page_table_cpu
        page_size = self.kernel_page_size
        origin_cpu = (
            (extend_prefix_lens_cpu - self.window_left).clamp(min=0) // page_size
        ) * page_size
        origin = (
            (extend_prefix_lens - self.window_left).clamp(min=0) // page_size
        ) * page_size
        prefix_cpu = extend_prefix_lens_cpu - origin_cpu
        prefix = extend_prefix_lens - origin
        num_pages = (self.window_left + page_size - 1) // page_size + 1
        columns = origin[:, None] // page_size + torch.arange(
            num_pages, device=self.device
        )
        prefix_table = page_table.gather(
            1, columns.clamp(max=page_table.shape[1] - 1).long()
        )
        super()._init_prefill_metadata(
            seq_lens - origin,
            prefix,
            prefix_cpu,
            extend_seq_lens,
            extend_seq_lens_cpu,
            prefix_table,
            page_table_cpu=None,
            extend_with_prefix=extend_with_prefix,
        )
        metadata = self.chunked_prefill_metadata
        lengths = metadata.chunked_seq_len
        offsets = prefix[None, :] - (lengths.cumsum(0) - lengths)
        metadata = Hy4FlashSWAPrefillMetadata(
            **vars(metadata), prefix_query_start_offsets=tuple(offsets.unbind(0))
        )
        self.forward_prefill_metadata = metadata
        self.chunked_prefill_metadata = metadata

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        out_cache_loc,
        token_to_kv_pool,
        bs,
        **kwargs,
    ):
        del out_cache_loc, token_to_kv_pool
        metadata = self.forward_prefill_metadata
        assert metadata is not None
        if metadata.max_extend_prefix_len:
            raise NotImplementedError(
                "Hy4-flash SWA prefix extend uses bounded chunk replay"
            )
        if layer.sliding_window_size != self.window_left:
            raise ValueError(
                f"Hy4-flash SWA layer window ({layer.sliding_window_size}) "
                f"!= backend window ({self.window_left})"
            )
        sinks = kwargs.get("sinks")
        output, lse = self.forward_extend_chunked(
            q,
            k,
            v,
            layer.scaling,
            layer.logit_cap,
            cum_seq_lens_q=metadata.cum_extend_seq_lens,
            cum_seq_lens_kv=metadata.cum_extend_seq_lens,
            max_q_len=metadata.max_extend_seq_len,
            max_kv_len=metadata.max_extend_seq_len,
            seq_lens=metadata.extend_seq_lens,
            batch_size=bs,
            causal=True,
        )
        if sinks is not None:
            output = apply_attention_sink(output, lse, sinks, inplace=True)
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
        out=None,
        window_left: int = -1,
        query_start_offsets: torch.Tensor | None = None,
        sinks: torch.Tensor | None = None,
    ):
        del batch_size
        effective_window = self.window_left if window_left < 0 else window_left
        if not causal and query_start_offsets is None:
            raise ValueError("Hy4-flash SWA prefix chunks require query_start_offsets")
        if causal and self.step_counter is not None:
            self.step_counter.record_cache()
        head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        q = q.reshape(-1, self.num_local_heads, head_dim)
        k = k.reshape(-1, self.num_local_heads, head_dim)
        v = v.reshape(-1, self.num_local_heads, self.v_head_dim)
        if q.dtype == torch.float8_e4m3fn:
            k, v = k.to(q.dtype), v.to(q.dtype)
        output, lse = mla_prefill(
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
            window_left=effective_window,
            query_start_offsets=query_start_offsets,
        )
        if sinks is not None:
            output = apply_attention_sink(output, lse, sinks, inplace=True)
        return output, lse


def create_hy4_flash_router(config: AttnConfig) -> Hy4FlashCacheGroupRouter:
    """Create the Hy4-flash CacheGroupRouter serving both DSA and SWA-MLA groups."""
    if config.kernel_page_size is not None:
        raise ValueError(
            "Hy4-flash uses 64-token paged blocks for both DSA and SWA; "
            "kernel_page_size must be unset"
        )
    spec = config.component(Hy4FlashAttentionConfig)
    if spec is None:
        raise TypeError("create_hy4_flash_router requires Hy4FlashAttentionConfig")
    platform = current_platform()

    def leaf_factory(group_id: str, block_granularity: int):
        if group_id == "dsa":
            leaf_spec, leaf_cls = spec, Hy4FlashDSABackend
        elif group_id == "swa":
            leaf_spec, leaf_cls = spec.swa, Hy4FlashSWABackend
        else:
            raise ValueError(f"Unknown Hy4-flash cache group {group_id!r}")
        if block_granularity != HY4_FLASH_LOGICAL_BLOCK_TOKENS:
            raise ValueError(
                f"Hy4-flash group {group_id!r} requires "
                f"{HY4_FLASH_LOGICAL_BLOCK_TOKENS}-token pages, got {block_granularity}"
            )
        leaf_config = replace(
            config,
            components=(leaf_spec,),
            kernel_page_size=HY4_FLASH_LOGICAL_BLOCK_TOKENS,
        )
        leaf = leaf_cls(
            leaf_config,
            leaf_spec,
            kernel_page_size=HY4_FLASH_LOGICAL_BLOCK_TOKENS,
        )
        if (
            isinstance(leaf, DSABackend)
            and leaf.kernel_solution is None
            and (not platform.is_nvidia or platform.arch_version < ArchVersion(10, 0))
        ):
            # Strided flatkv planes on Hopper use the Triton sparse MLA kernel.
            leaf.kernel_solution = "triton"
        return leaf

    return Hy4FlashCacheGroupRouter(
        leaf_factory,
        is_draft=bool(config.is_draft),
        spec_num_tokens=config.speculative_num_draft_tokens or 1,
        device=config.device,
        consumed_group_ids=None,
    )


__all__ = [
    "Hy4FlashCacheGroupRouter",
    "Hy4FlashDSABackend",
    "Hy4FlashSWABackend",
    "Hy4FlashSWAPrefillMetadata",
    "create_hy4_flash_router",
]
