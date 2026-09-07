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

"""The hybrid linear-attention composite: full-attention router + state leaf.

One backend node serves a model that interleaves paged full-attention layers
with recurrent (GDN/KDA) layers: metadata calls broadcast to both children
(each consumes its own groups out of the delivered ``block_tables``), and
forwards dispatch per layer id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    current_forward_ctx,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.backends.state.mamba import MambaAttnBackend

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


class HybridLinearAttnBackend(AttentionBackend):
    """Hybrid backend that routes between full attention and linear attention by layer ID."""

    # Both sub-backends consume per-group tables (MHA: KV pages; Mamba:
    # dual-index state pages). Target verify publishes only the accepted
    # position.

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: MambaAttnBackend,
        full_attn_layers: list[int],
    ):
        self.device = full_attn_backend.device
        self.full_attn_layers = set(full_attn_layers)
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend

    # The MLA full-attention sub-backend owns the spec-decode token width and
    # the chunked-prefill machinery. The DeepseekV3-style MLA layer forward
    # (reused by Kimi-K3) reads these off ``ctx.attn_backend`` -- which is this
    # hybrid wrapper -- so route them to the full-attention backend.
    @property
    def spec_num_tokens(self) -> int:
        return self.full_attn_backend.spec_num_tokens

    @property
    def chunked_prefill_metadata(self):
        return self.full_attn_backend.chunked_prefill_metadata

    @property
    def data_type(self):
        return self.full_attn_backend.data_type

    @property
    def supports_mla_projected_value_decode(self) -> bool:
        return self.full_attn_backend.supports_mla_projected_value_decode

    @property
    def sparse_topk(self):
        # The sparse layers (QSA) index the full-attention child's groups.
        return self.full_attn_backend.sparse_topk

    def override_num_extends(self, num_extends: int):
        return self.full_attn_backend.override_num_extends(num_extends)

    def forward_extend_chunked(self, *args, **kwargs):
        return self.full_attn_backend.forward_extend_chunked(*args, **kwargs)

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor) -> None:
        # Composite: the full-attention child owns the seq_lens the draft reads.
        self.full_attn_backend.advance_draft_forward_metadata(seq_lens)

    def draft_history_view(self):
        return self.full_attn_backend.draft_history_view()

    def draft_write_locations_uniform(self, out, cache_start, num_tokens):
        return self.full_attn_backend.draft_write_locations_uniform(
            out, cache_start, num_tokens
        )

    def publish_draft_step_locations(self, cache_start, num_tokens):
        return self.full_attn_backend.publish_draft_step_locations(
            cache_start, num_tokens
        )

    def decode_window_locations(self):
        return self.full_attn_backend.decode_window_locations()

    def extend_span_locations(self):
        return self.full_attn_backend.extend_span_locations()

    def write_locations(self, layer, forward_mode):
        return self._backend_for_layer(layer.layer_id).write_locations(
            layer, forward_mode
        )

    @property
    def cache_consumer_families(self) -> frozenset[str]:
        """Cache families consumed by the two child backends."""
        return (
            self.full_attn_backend.cache_consumer_families
            | self.linear_attn_backend.cache_consumer_families
        )

    def child_backends(self):
        return (self.full_attn_backend, self.linear_attn_backend)

    def set_cache_pool(self, cache_pool) -> None:
        self.cache_pool = cache_pool
        for backend in self.child_backends():
            backend.set_cache_pool(cache_pool)

    def _backend_for_layer(self, layer_id: int) -> AttentionBackend:
        if layer_id in self.full_attn_layers:
            return self.full_attn_backend
        return self.linear_attn_backend

    # ---- Metadata delegation ----

    def init_forward_metadata(self, *args, **kwargs):
        self.full_attn_backend.init_forward_metadata(*args, **kwargs)
        self.linear_attn_backend.init_forward_metadata(*args, **kwargs)

    def init_cuda_graph_state(self, max_bs: int, **kwargs):
        # Both children are runner-facing nodes whose init_cuda_graph_state
        # absorbs the runner extras (cache_group_specs, ...) through **kwargs.
        self.full_attn_backend.init_cuda_graph_state(max_bs, **kwargs)
        self.linear_attn_backend.init_cuda_graph_state(max_bs, **kwargs)

    def register_step_counter(self, step_counter):
        # Hybrid layerwise transfer needs one global step per model layer,
        # including both full-attention and mamba layers. Normal attention
        # dispatch records in this wrapper; model-owned chunked prefill bypasses
        # that dispatch, so its full-attention child needs the same counter.
        self.step_counter = step_counter
        self.full_attn_backend.register_step_counter(step_counter)

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        self.full_attn_backend.init_forward_metadata_capture_cuda_graph(*args, **kwargs)
        self.linear_attn_backend.init_forward_metadata_capture_cuda_graph(
            *args, **kwargs
        )

    def refresh_decode_metadata(self, *args, **kwargs) -> None:
        self.full_attn_backend.refresh_decode_metadata(*args, **kwargs)
        self.linear_attn_backend.refresh_decode_metadata(*args, **kwargs)

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        return self.full_attn_backend.support_kv_cache_prewrite(forward_mode)

    # ---- Forward dispatch ----

    @break_point
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        token_to_kv_pool,
        forward_mode: ForwardMode,
        bs: int,
        save_kv_cache: bool = True,
        record_kv_cache: bool | None = None,
        **kwargs,
    ):
        """Dispatch one layer to its full-attention or GDN backend (the break point).

        Overrides the base forward, so it carries its own ``@break_point``;
        the frozen capture-time scalars (forward_mode/bs) are re-read from the
        ambient ctx (semantics: see breakable_cuda_graph). The GDN scan's
        batched [1, T, Hv, D] output is collapsed to z-shaped [T, Hv, D].
        """
        # Frozen capture-time scalars, re-read live (see docstring); no-op in eager.
        amb = current_forward_ctx()
        if amb is not None:
            forward_mode = amb.forward_mode
            bs = amb.bs

        if forward_mode.is_idle():
            if layer is None:
                return torch.empty_like(kwargs["z"])
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        backend = self._backend_for_layer(layer_id)

        # See AttentionBackend.forward for the record_kv_cache contract; the step
        # is recorded in this wrapper (not the child backends) to keep one step
        # per model layer across full-attn + mamba. Idle already returned above.
        with self.record_pd_cache_step(forward_mode, save_kv_cache, record_kv_cache):
            if forward_mode.is_decode():
                ret = backend.forward_decode(
                    q,
                    k,
                    v,
                    layer,
                    token_to_kv_pool,
                    bs,
                    save_kv_cache=save_kv_cache,
                    **kwargs,
                )
            else:
                ret = backend.forward_extend(
                    q,
                    k,
                    v,
                    layer,
                    token_to_kv_pool,
                    bs,
                    save_kv_cache=save_kv_cache,
                    forward_mode=forward_mode,
                    **kwargs,
                )
        # Collapse the GDN scan's batched [1, T, Hv, D] to z-shaped (see docstring).
        if ret is not None and ret.dim() == 4:
            # Strictly [1, T, Hv, D]: a genuine B>1 must fail loud, not corrupt the handoff.
            assert (
                ret.shape[0] == 1
            ), f"GDN scan batched rank expected leading 1, got {ret.shape}"
            ret = ret.flatten(0, 1)
        return ret

    def update_mamba_state_after_mtp_verify(self, accepted_lengths):
        self.linear_attn_backend.commit_verified_state(accepted_lengths)
