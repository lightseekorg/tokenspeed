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

"""Route dots3-note's DSA and SWA groups through the existing paged leaves."""

from dataclasses import dataclass, replace

import torch
from tokenspeed_kernel.ops.attention.mla import mla_prefill

from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend
from tokenspeed.runtime.layers.attention.backends.paged.mla import (
    MLAAttnBackend,
    MLAPrefillMetadata,
)
from tokenspeed.runtime.layers.attention.backends.paged.router import CacheGroupRouter
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.dots3_note import (
    Dots3NoteAttentionConfig,
)
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig


class Dots3NoteCacheGroupRouter(CacheGroupRouter):
    def register_step_counter(self, step_counter) -> None:
        # Dots attention publishes once after all latent and index cache writes.
        self.step_counter = step_counter


class Dots3NoteDSABackend(DSABackend):
    @staticmethod
    def _make_dense_leaf(config, spec, platform, kernel_page_size):
        # The default TRTLLM dense delegate cannot run on Hopper.
        leaf = MLAAttnBackend(
            config,
            replace(spec, backend_name="mla"),
            kernel_page_size=kernel_page_size,
        )
        leaf.kernel_solution = "triton"
        return leaf


@dataclass(kw_only=True)
class Dots3NoteSWAPrefillMetadata(MLAPrefillMetadata):
    prefix_query_start_offsets: tuple[torch.Tensor, ...]


class Dots3NoteSWABackend(MLAAttnBackend):
    """MLA with bounded prefix gathers and a window on every prefill pass."""

    default_kernel_page_size = 32

    def __init__(self, config: AttnConfig, spec: MLAConfig, *, kernel_page_size: int):
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        if (
            not isinstance(spec.sliding_window_tokens, int)
            or spec.sliding_window_tokens < 1
        ):
            raise ValueError("dots3-note SWA needs one positive retention window")
        self.window_left = spec.sliding_window_tokens - 1

    def _should_use_absorbed_cached_extend(
        self, *, max_extend_seq_len: int, max_extend_prefix_len: int
    ) -> bool:
        # The rank-1024 window path expands only bounded prefix chunks.
        return False

    def _init_prefill_metadata(
        self,
        seq_lens,
        extend_prefix_lens,
        extend_prefix_lens_cpu,
        extend_seq_lens,
        extend_seq_lens_cpu,
        page_table,
    ):
        page_size = self.kernel_page_size
        origin_cpu = (
            (extend_prefix_lens_cpu - self.window_left).clamp(min=0) // page_size
        ) * page_size
        origin = (
            (extend_prefix_lens - self.window_left).clamp(min=0) // page_size
        ) * page_size
        prefix_cpu = extend_prefix_lens_cpu - origin_cpu
        prefix = extend_prefix_lens - origin
        # A fixed table width avoids recompiling metadata for each prefix length.
        num_pages = (self.window_left + page_size - 1) // page_size + 1
        columns = origin[:, None] // page_size + torch.arange(
            num_pages, device=self.device
        )
        # Shorter requests do not consume the extra gathered columns.
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
        )
        metadata = self.chunked_prefill_metadata
        lengths = metadata.chunked_seq_len
        offsets = prefix[None, :] - (lengths.cumsum(0) - lengths)
        metadata = Dots3NoteSWAPrefillMetadata(
            **vars(metadata), prefix_query_start_offsets=tuple(offsets.unbind(0))
        )
        self.forward_prefill_metadata = metadata
        self.chunked_prefill_metadata = metadata

    def forward_extend(
        self, q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
    ):
        metadata = self.forward_prefill_metadata
        assert metadata is not None
        if metadata.max_extend_prefix_len:
            raise NotImplementedError(
                "dots3-note SWA prefix extend uses bounded chunk replay"
            )
        if layer.sliding_window_size != self.window_left:
            raise ValueError("dots3-note SWA layer and cache windows must match")
        output, _ = self.forward_extend_chunked(
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
        query_start_offsets=None,
    ):
        if not causal and query_start_offsets is None:
            raise ValueError("SWA prefix chunks require query_start_offsets")
        if causal and self.step_counter is not None:
            self.step_counter.record_cache()
        head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        q = q.reshape(-1, self.num_local_heads, head_dim)
        k = k.reshape(-1, self.num_local_heads, head_dim)
        v = v.reshape(-1, self.num_local_heads, self.v_head_dim)
        if q.dtype == torch.float8_e4m3fn:
            k, v = k.to(q.dtype), v.to(q.dtype)
        return mla_prefill(
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
            window_left=self.window_left,
            query_start_offsets=query_start_offsets,
        )


def create_dots3_note_router(config: AttnConfig) -> CacheGroupRouter:
    if config.kernel_page_size is not None:
        raise ValueError(
            "dots3-note uses DSA page64 / SWA page32; kernel_page_size must be unset"
        )
    spec = config.component(Dots3NoteAttentionConfig)

    def leaf_factory(group_id: str, block_granularity: int):
        if group_id == "dsa":
            leaf_spec, leaf_cls, page_size = spec, Dots3NoteDSABackend, 64
        elif group_id == "swa":
            leaf_spec, leaf_cls, page_size = spec.swa, Dots3NoteSWABackend, 32
        else:
            raise ValueError(f"Unknown dots3-note cache group {group_id!r}")
        if block_granularity != page_size:
            raise ValueError(
                f"dots3-note group {group_id!r} requires {page_size}-token pages"
            )
        leaf_config = replace(
            config, components=(leaf_spec,), kernel_page_size=page_size
        )
        leaf = leaf_cls(leaf_config, leaf_spec, kernel_page_size=page_size)
        if isinstance(leaf, DSABackend) and leaf.kernel_solution is None:
            # Other sparse implementations can materialize the whole strided cache.
            leaf.kernel_solution = "triton"
        return leaf

    return Dots3NoteCacheGroupRouter(
        leaf_factory,
        is_draft=config.is_draft,
        spec_num_tokens=config.speculative_num_draft_tokens,
        device=config.device,
        consumed_group_ids=None,
    )
