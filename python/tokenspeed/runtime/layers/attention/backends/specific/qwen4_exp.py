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

"""Qwen4-Exp composition of attention, PLE and QSA cache consumers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (
    HybridLinearAttnBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import SoftmaxAttnConfig

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.base import SparseTopKShare
    from tokenspeed.runtime.layers.attention.backends.specific.qsa_indexer import (
        QSAIndexerBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.specific.qwen4_exp_ple import (
        Qwen4ExpPLEBackend,
    )
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention
    from tokenspeed.runtime.pd.utils import StepCounter


def qwen4_exp_backend(attn_backend: AttentionBackend) -> Qwen4ExpBackend:
    """Resolve the model's composite without depending on an attention leaf."""
    if not isinstance(attn_backend, Qwen4ExpBackend):
        raise RuntimeError("Qwen4-Exp layers require their model's composite backend")
    return attn_backend


class Qwen4ExpBackend(AttentionBackend):
    """Broadcast cache lifecycle calls; model layers retain their compute order."""

    def __init__(
        self,
        config: AttnConfig,
        attention_backend: AttentionBackend,
        ple_backend: Qwen4ExpPLEBackend | None,
        indexer_backend: QSAIndexerBackend | None,
    ) -> None:
        super().__init__(config, config.component(SoftmaxAttnConfig))
        self.attention_backend = attention_backend
        self.ple_backend = ple_backend
        self.indexer_backend = indexer_backend

    def child_backends(self) -> tuple[AttentionBackend, ...]:
        return (self.attention_backend,) + tuple(
            backend
            for backend in (self.ple_backend, self.indexer_backend)
            if backend is not None
        )

    @property
    def cache_consumer_families(self) -> frozenset[str]:
        return frozenset().union(
            *(backend.cache_consumer_families for backend in self.child_backends())
        )

    def set_cache_pool(self, cache_pool: CachePool) -> None:
        self.cache_pool = cache_pool
        for backend in self.child_backends():
            backend.set_cache_pool(cache_pool)

    def preallocate_verify_workspace(self, max_bs: int, draft_token_num: int) -> int:
        """Preallocate target verify consumers and return their total byte count."""
        if self.is_draft or self.spec_num_tokens <= 1:
            return 0
        gdn = (
            self.attention_backend.linear_attn_backend
            if isinstance(self.attention_backend, HybridLinearAttnBackend)
            else None
        )
        return sum(
            consumer.preallocate_verify_workspace(max_bs, draft_token_num)
            for consumer in (gdn, self.ple_backend, self.indexer_backend)
            if consumer is not None
        )

    def init_cuda_graph_state(self, max_bs: int, **kwargs) -> None:
        for backend in self.child_backends():
            backend.init_cuda_graph_state(max_bs, **kwargs)

    def init_forward_metadata(self, *args, **kwargs) -> None:
        for backend in self.child_backends():
            backend.init_forward_metadata(*args, **kwargs)

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs) -> None:
        for backend in self.child_backends():
            backend.init_forward_metadata_capture_cuda_graph(*args, **kwargs)

    def refresh_decode_metadata(self, *args, **kwargs) -> None:
        for backend in self.child_backends():
            backend.refresh_decode_metadata(*args, **kwargs)

    def configure_runtime(self, **kwargs) -> None:
        self._full_attn_backend.configure_runtime(**kwargs)

    def init_prefill_graph_state(self, max_num_tokens: int, max_bs: int) -> None:
        self._full_attn_backend.init_prefill_graph_state(max_num_tokens, max_bs)

    def register_step_counter(self, step_counter: StepCounter) -> None:
        self.attention_backend.register_step_counter(step_counter)

    def forward(self, *args, **kwargs):
        # The attention child owns the break point and the single PD cache step.
        return self.attention_backend.forward(*args, **kwargs)

    @property
    def sparse_topk(self) -> SparseTopKShare:
        return self.attention_backend.sparse_topk

    @property
    def supports_layer_sliding_window(self) -> bool:
        return self._full_attn_backend.supports_layer_sliding_window

    def support_kv_cache_prewrite(self, forward_mode: ForwardMode | None) -> bool:
        return self.attention_backend.support_kv_cache_prewrite(forward_mode)

    def write_locations(
        self, layer: PagedAttention, forward_mode: ForwardMode
    ) -> torch.Tensor:
        return self.attention_backend.write_locations(layer, forward_mode)

    def publish_draft_step_locations(
        self, cache_start: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        return self.attention_backend.publish_draft_step_locations(
            cache_start, num_tokens
        )

    def draft_write_locations_uniform(
        self, out: torch.Tensor, cache_start: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        return self.attention_backend.draft_write_locations_uniform(
            out, cache_start, num_tokens
        )

    def draft_history_view(self):
        return self.attention_backend.draft_history_view()

    def decode_window_locations(self) -> torch.Tensor:
        return self.attention_backend.decode_window_locations()

    def extend_span_locations(self) -> torch.Tensor:
        return self.attention_backend.extend_span_locations()

    def override_num_extends(self, num_extends: int):
        return self.attention_backend.override_num_extends(num_extends)

    @property
    def _full_attn_backend(self) -> AttentionBackend:
        if isinstance(self.attention_backend, HybridLinearAttnBackend):
            return self.attention_backend.full_attn_backend
        return self.attention_backend

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        self._full_attn_backend.advance_draft_forward_metadata(seq_lens)
        if self.indexer_backend is not None:
            self.indexer_backend.advance_draft_forward_metadata(seq_lens)

    def update_draft_forward_metadata(self, frontier: torch.Tensor) -> None:
        # Hybrid's base implementation is a no-op; publish to the router itself.
        self._full_attn_backend.update_draft_forward_metadata(frontier)
        if self.indexer_backend is not None:
            self.indexer_backend.update_draft_forward_metadata(frontier)

    def fill_block_decode_seq_lens(self, bs: int, block_seq_lens: torch.Tensor) -> None:
        self._full_attn_backend.fill_block_decode_seq_lens(bs, block_seq_lens)
        if self.indexer_backend is not None:
            self.indexer_backend.fill_block_decode_seq_lens(bs, block_seq_lens)

    def commit_speculative_state_after_verify(
        self, accepted_lengths: torch.Tensor, *, num_extends: int
    ) -> None:
        self.attention_backend.commit_speculative_state_after_verify(
            accepted_lengths, num_extends=num_extends
        )
        if num_extends == 0 and self.ple_backend is not None:
            self.ple_backend.commit_verified_state(accepted_lengths)
        if self.indexer_backend is not None:
            self.indexer_backend.commit_after_mtp_verify(
                accepted_lengths, num_extends=num_extends
            )


__all__ = ["Qwen4ExpBackend", "qwen4_exp_backend"]
