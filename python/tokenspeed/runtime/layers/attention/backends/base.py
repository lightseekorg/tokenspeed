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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
    from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention
    from tokenspeed.runtime.pd.utils import StepCounter


class AttentionBackend(ABC):
    """The base class of attention backends"""

    def __init__(self, config: BaseAttnConfig) -> None:
        self.device = config.device
        self.num_qo_heads = config.num_attention_heads // config.attn_tp_size
        self.num_kv_heads = max(config.num_kv_heads // config.attn_tp_size, 1)
        self.dtype = config.dtype
        self.head_dim = config.head_dim
        self.is_draft = config.is_draft

    @property
    def support_kv_cache_prewrite(self) -> bool:
        return False

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
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
    ):
        """Init the metadata for a forward pass for capturing a cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        **kwargs,
    ):
        """Update pre-allocated CUDA-graph metadata buffers in-place before replay.

        Called instead of init_forward_metadata when use_cuda_graph=True, so
        that the captured kernels (which hold pointers into the pre-allocated
        buffers) see the current batch's data without any new allocations.
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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: BaseTokenToKVPool,
        forward_mode: ForwardMode,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run forward on an attention layer with explicit scheduler metadata."""
        if forward_mode.is_decode():
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        else:
            if (
                not forward_mode.is_idle()
                and getattr(self, "step_counter", None)
                and not save_kv_cache
            ):
                self.step_counter.record_cache()

            ret = self.forward_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                save_kv_cache=save_kv_cache,
                forward_mode=forward_mode,
                **kwargs,
            )

            if (
                not forward_mode.is_idle()
                and getattr(self, "step_counter", None)
                and save_kv_cache
            ):
                self.step_counter.record_cache()
            return ret

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool: BaseTokenToKVPool,
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
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run a forward for extend."""
        raise NotImplementedError()
