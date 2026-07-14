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

import inspect
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.execution.breakable_cuda_graph import break_point

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
    from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention
    from tokenspeed.runtime.pd.utils import StepCounter


def init_backend_cuda_graph_state(
    backend: "AttentionBackend",
    max_bs: int,
    seq_lens_buf: torch.Tensor,
    **extras,
) -> None:
    """Call ``backend.init_cuda_graph_state`` with only the kwargs its
    signature accepts (VAR_KEYWORD accepts all of them).

    Signature-probe instead of try/except TypeError: paged_cache_group_specs
    is load-bearing for the state shed, so a TypeError raised from inside the
    backend's body must propagate rather than silently retry without specs.

    Shared by the cuda-graph wrapper and by composite backends (hybrid) that
    forward to user-selectable sub-backends with possibly narrow signatures.
    """
    params = inspect.signature(backend.init_cuda_graph_state).parameters
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        extras = {k: v for k, v in extras.items() if k in params}
    backend.init_cuda_graph_state(max_bs, seq_lens_buf, **extras)


class AttentionBackend(ABC):
    """The base class of attention backends"""

    uses_paged_cache_groups: bool = False
    # Flat KV-cache per-group block tables (absolute index, null hole = 0). A
    # separate flag from uses_paged_cache_groups because the two mechanisms have
    # different hole/index semantics; a group-aware flat backend (Phase 4) sets
    # this True. Default False keeps every existing backend on today's path.
    uses_flat_cache_groups: bool = False
    # False for flat-capable backends whose spec-verify path is not wired yet.
    flat_spec_capable: bool = True
    uses_padded_decode_token_mask: bool = False

    def __init__(self, config: BaseAttnConfig) -> None:
        self.device = config.device
        self.num_qo_heads = config.num_attention_heads // config.attn_tp_size
        self.num_kv_heads = max(config.num_kv_heads // config.attn_tp_size, 1)
        self.dtype = config.dtype
        self.head_dim = config.head_dim
        self.is_draft = config.is_draft
        self.spec_num_tokens = config.speculative_num_draft_tokens
        # True when this backend's CUDA-graph block-table (kv_indices) buffer is
        # aliased to a peer backend's (e.g. a drafter sharing the target's), so
        # the replay path skips rebuilding it — the peer already populates it.
        self._block_table_aliased = False

    @contextmanager
    def override_num_extends(self, num_extends: int):
        """Temporarily override the decode-metadata slice discriminator for the
        wrapped block. Used by MLA backends to flip between drafter step 0
        (slice = [num_extends:]) and step 1+ (slice = [0:]).

        Default no-op for backends that fill separate prefill/decode metadata
        at init time.
        """
        yield

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        return False

    def select_out_cache_loc(self, layer, out_cache_loc, forward_mode=None):
        """Flat per-group write-location hook for out-of-backend KV writers
        (fused RoPE prewrite); identity for backends without flat cache
        groups (see uses_flat_cache_groups). ``forward_mode`` picks the
        metadata slot for backends that prewrite on extend as well."""
        return out_cache_loc

    @property
    def sinks_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @abstractmethod
    def init_forward_metadata(self, *args, **kwargs):
        """Init the metadata for a forward pass.

        When use_cuda_graph=True the backend should use its pre-allocated
        cuda-graph buffers instead of the normal eager buffers.
        """
        raise NotImplementedError()

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor):
        """Init the global shared states for cuda graph. `seq_lens_buf` is
        the controller-owned per-request seq_lens; backends should reference
        (alias) it rather than copy, and must not mutate the contents."""
        raise NotImplementedError()

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        flat_cache_group_ids: tuple[str, ...] = (),
        **kwargs,
    ):
        """Init the metadata for a forward pass for capturing a cuda graph.

        ``flat_cache_group_ids`` names the flat KV-cache groups whose page
        tables arrive at replay; a flat-capable backend (uses_flat_cache_groups)
        allocates its persistent per-group buffers from these ids — no table
        data exists at capture time. Empty tuple for non-flat backends.
        """
        raise NotImplementedError()

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        flat_block_tables: dict[str, torch.Tensor] | None = None,
        **kwargs,
    ):
        """Update pre-allocated CUDA-graph metadata buffers in-place before replay.

        Called instead of init_forward_metadata when use_cuda_graph=True, so
        that the captured kernels (which hold pointers into the pre-allocated
        buffers) see the current batch's data without any new allocations.
        ``flat_block_tables`` carries the per-group flat page tables
        (group_id -> [>=bs, cols]) for flat-capable backends; a backend that
        captured flat buffers must be handed non-empty tables whenever bs > 0.
        Default: fall back to init_forward_metadata (correct but may not work
        for all backends that use separate cuda-graph buffer pools).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement init_forward_metadata_replay_cuda_graph "
            "for CUDA graph support"
        )

    def configure_runtime(self, **kwargs) -> None:
        """Configure runtime state after model loading (e.g. sliding_window_size).

        Called once during ModelExecutor initialization with information that is
        not available at backend construction time.  Default: no-op.
        """
        pass

    def register_step_counter(self, step_counter: StepCounter):
        self.step_counter = step_counter

    @contextmanager
    def record_pd_cache_step(
        self,
        forward_mode: ForwardMode,
        save_kv_cache: bool,
        record_kv_cache: bool | None,
    ):
        """Anchor the PD layerwise cache-step record to the wrapped KV write.

        Records the ``StepCounter`` step before the attention call when the KV
        was pre-written (``save_kv_cache=False``) and after it otherwise, so a
        layerwise cache transfer always observes a fully written layer. See
        ``forward`` for the ``record_kv_cache`` override contract. No-op when no
        step counter is registered. Backends that own the record (e.g. the
        hybrid wrapper, which counts once per model layer across full-attn +
        mamba children) reuse this to avoid duplicating the gate logic.
        """
        if record_kv_cache is None:
            record_cache = not forward_mode.is_decode() and not forward_mode.is_idle()
        else:
            record_cache = record_kv_cache
        record_cache = record_cache and getattr(self, "step_counter", None) is not None

        if record_cache and not save_kv_cache:
            self.step_counter.record_cache()
        yield
        if record_cache and save_kv_cache:
            self.step_counter.record_cache()

    @break_point
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: BaseTokenToKVPool,
        forward_mode: ForwardMode,
        bs: int,
        save_kv_cache: bool = True,
        record_kv_cache: bool | None = None,
        **kwargs,
    ):
        """Run forward on an attention layer with explicit scheduler metadata.

        ``record_kv_cache`` overrides the PD layerwise cache-step recording:
        ``None`` keeps the default (record on the EXTEND-side path), an explicit
        bool forces it so a DECODE-dispatched draft catch-up can still record.
        """
        with self.record_pd_cache_step(forward_mode, save_kv_cache, record_kv_cache):
            if forward_mode.is_decode():
                ret = self.forward_decode(
                    q,
                    k,
                    v,
                    layer,
                    out_cache_loc,
                    token_to_kv_pool,
                    bs,
                    save_kv_cache=save_kv_cache,
                    **kwargs,
                )
            else:
                ret = self.forward_extend(
                    q,
                    k,
                    v,
                    layer,
                    out_cache_loc,
                    token_to_kv_pool,
                    bs,
                    save_kv_cache=save_kv_cache,
                    forward_mode=forward_mode,
                    **kwargs,
                )
        return ret

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: BaseTokenToKVPool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run a forward for decode."""
        raise NotImplementedError()

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: BaseTokenToKVPool,
        bs: int,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run a forward for extend."""
        raise NotImplementedError()
