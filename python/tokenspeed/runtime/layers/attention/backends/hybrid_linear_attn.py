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

"""Hybrid linear attention backend for Qwen3.5 GDN models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from tokenspeed.runtime.layers.attention.linear.chunk import chunk_gated_delta_rule
from tokenspeed.runtime.layers.attention.linear.chunk_delta_h import (
    CHUNK_SIZE as FLA_CHUNK_SIZE,
)
from tokenspeed.runtime.layers.attention.linear.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from tokenspeed.runtime.layers.attention.linear.gdn import fused_gdn_gating
from tokenspeed.runtime.layers.attention.linear.mamba_state_scatter_triton import (
    fused_mamba_state_scatter_with_mask,
)

_USE_BLADE_GDN = os.environ.get("BLADE_GDN", "0") == "1"
_USE_FUSED_BLADE_GDN = os.environ.get("FUSED_BLADE_GDN", "0") == "1"
_USE_FLASHINFER_GDN = os.environ.get("FLASHINFER_GDN", "0") == "1"
if _USE_BLADE_GDN or _USE_FUSED_BLADE_GDN:
    import tiny_gdn

    _blade_gdn_workspace = None

if _USE_FLASHINFER_GDN:
    try:
        from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

        _flashinfer_gdn_decode = gated_delta_rule_decode_pretranspose
        _flashinfer_gdn_use_pool = torch.cuda.get_device_capability()[0] != 9
    except ImportError:
        import logging

        logging.getLogger(__name__).warning(
            "FLASHINFER_GDN=1 but flashinfer.gdn_decode not available, falling back to triton"
        )
        _USE_FLASHINFER_GDN = False

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
    from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


@dataclass
class MambaForwardMetadata:
    query_start_loc: torch.Tensor | None
    mamba_cache_indices: torch.Tensor
    extend_prefix_lens: Optional[torch.Tensor] = None
    # Pre-computed src/dst indices for extracting Mamba prefix-cache snapshots.
    track_ssm_h_src: Optional[torch.Tensor] = None
    track_ssm_h_dst: Optional[torch.Tensor] = None
    track_conv_indices: Optional[torch.Tensor] = None
    track_ssm_final_src: Optional[torch.Tensor] = None
    track_ssm_final_dst: Optional[torch.Tensor] = None


class LayerMappedKVPool:
    """Wraps a KV pool to map global layer IDs to internal pool indices.

    For hybrid models, only full attention layers have KV cache. This wrapper
    translates global layer IDs (e.g., 3, 7, 11) to pool indices (0, 1, 2).
    """

    def __init__(
        self, inner_pool: BaseTokenToKVPool, full_attention_layer_ids: list[int]
    ):
        self.inner = inner_pool
        self.layer_ids = list(full_attention_layer_ids)
        self.layer_map = {
            global_id: pool_idx
            for pool_idx, global_id in enumerate(full_attention_layer_ids)
        }
        # Expose page_size from inner pool for the scheduler
        self.page_size = getattr(inner_pool, "page_size", 1)

    def _map(self, layer_id: int) -> int:
        return self.layer_map.get(layer_id, layer_id)

    def set_kv_buffer(
        self,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor | None,
        k_scale: torch.Tensor | None = None,
        v_scale: torch.Tensor | None = None,
    ):
        orig = layer.layer_id
        layer.layer_id = self._map(orig)
        self.inner.set_kv_buffer(layer, out_cache_loc, k, v, k_scale, v_scale)
        layer.layer_id = orig

    def get_kv_buffer(self, layer_id: int):
        return self.inner.get_kv_buffer(self._map(layer_id))

    def get_key_buffer(self, layer_id: int):
        return self.inner.get_key_buffer(self._map(layer_id))

    def get_value_buffer(self, layer_id: int):
        return self.inner.get_value_buffer(self._map(layer_id))

    def __getattr__(self, name):
        return getattr(self.inner, name)


class SimpleMambaPool:
    """Mamba state pool indexed by scheduler-assigned cache slots."""

    def __init__(
        self,
        size: int,
        num_mamba_layers: int,
        conv_state_shape: tuple,
        temporal_state_shape: tuple,
        conv_dtype: torch.dtype,
        ssm_dtype: torch.dtype,
        mamba_layer_ids: list[int],
        device: str,
        page_size: int = 1,
        speculative_num_draft_tokens: int = 0,
    ):
        self.size = size
        self.device = device
        self.mamba_layer_ids = list(mamba_layer_ids)
        self.page_size = page_size
        self.mamba_map = {layer_id: i for i, layer_id in enumerate(mamba_layer_ids)}
        self.is_kda_cache = False

        # Allocate conv state: (num_mamba_layers, size+1, conv_dim, state_len)
        self.conv_state = torch.zeros(
            num_mamba_layers,
            size + 1,
            *conv_state_shape,
            dtype=conv_dtype,
            device=device,
        )
        # Allocate temporal/SSM state: (num_mamba_layers, size+1, heads, key_dim, val_dim)
        self.ssm_state = torch.zeros(
            num_mamba_layers,
            size + 1,
            *temporal_state_shape,
            dtype=ssm_dtype,
            device=device,
        )

        # Speculative decoding intermediate caches
        if speculative_num_draft_tokens > 0:
            self.intermediate_ssm_state_cache = torch.empty(
                num_mamba_layers,
                size + 1,
                speculative_num_draft_tokens,
                *temporal_state_shape,
                dtype=ssm_dtype,
                device=device,
            )
            self.intermediate_conv_window_cache = torch.zeros(
                num_mamba_layers,
                size + 1,
                speculative_num_draft_tokens,
                *conv_state_shape,
                dtype=conv_dtype,
                device=device,
            )
            self.mamba_cache = (
                self.conv_state,
                self.ssm_state,
                self.intermediate_ssm_state_cache,
                self.intermediate_conv_window_cache,
            )
        else:
            self.mamba_cache = (self.conv_state, self.ssm_state)

    def get_mamba_indices(self, mamba_pool_indices: torch.Tensor) -> torch.Tensor:
        """Return mamba cache indices directly (allocated by C++ scheduler)."""
        return mamba_pool_indices.to(torch.int32)

    def get_mamba_params(self, layer_id: int):
        """Return per-layer cache slices."""
        internal_idx = self.mamba_map[layer_id]
        return [self.mamba_cache[i][internal_idx] for i in range(len(self.mamba_cache))]

    def get_mamba_params_all_layers(self):
        """Return all layers for all cache components."""
        return [self.mamba_cache[i] for i in range(len(self.mamba_cache))]

    def get_contiguous_buf_infos(self):
        """Return per-layer mamba cache buffers for disaggregated transfer."""
        data_ptrs = []
        data_lens = []
        item_lens = []
        for cache in self.mamba_cache:
            for layer_id in range(cache.shape[0]):
                layer_cache = cache[layer_id]
                data_ptrs.append(layer_cache.data_ptr())
                data_lens.append(layer_cache.nbytes)
                item_lens.append(layer_cache[0].nbytes)
        return data_ptrs, data_lens, item_lens

    def get_contiguous_buf_layer_ids(self):
        """Return global layer ids aligned with get_contiguous_buf_infos()."""
        return self.mamba_layer_ids * len(self.mamba_cache)


class MambaAttnBackend(AttentionBackend):
    """Attention backend for Mamba/GDN linear attention layers."""

    def __init__(self, config: BaseAttnConfig):
        super().__init__(config)
        self.pad_slot_id = -1
        self.forward_metadata: MambaForwardMetadata = None
        self.state_indices_list = []
        self.query_start_loc_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None
        self.speculative_num_draft_tokens = getattr(
            config, "speculative_num_draft_tokens", 0
        )
        self.pool: SimpleMambaPool = None

    def set_pool(self, pool: SimpleMambaPool):
        self.pool = pool

    def init_forward_metadata(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = ForwardMode.DECODE,
        **kwargs,
    ):
        mamba_pool_indices = kwargs.get("mamba_pool_indices")
        if mamba_pool_indices is not None:
            mamba_cache_indices = self.pool.get_mamba_indices(mamba_pool_indices[:bs])
        else:
            mamba_cache_indices = self.pool.get_mamba_indices(req_pool_indices[:bs])

        if forward_mode.is_decode_or_idle():
            query_start_loc = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
        elif forward_mode.is_extend():
            if forward_mode.is_target_verify() or forward_mode.is_draft_extend():
                tokens_per_req = kwargs.get(
                    "tokens_per_req", self.speculative_num_draft_tokens
                )
                query_start_loc = torch.arange(
                    0,
                    num_tokens + 1,
                    step=tokens_per_req,
                    dtype=torch.int32,
                    device=self.device,
                )
            else:
                extend_start_loc = kwargs.get("extend_start_loc")
                extend_seq_lens = kwargs.get("extend_seq_lens")
                if extend_start_loc is not None and extend_seq_lens is not None:
                    query_start_loc = torch.empty(
                        (bs + 1,), dtype=torch.int32, device=self.device
                    )
                    query_start_loc[:bs] = extend_start_loc
                    query_start_loc[bs] = extend_start_loc[-1] + extend_seq_lens[-1]
                else:
                    extend_prefix_lens = kwargs.get("extend_prefix_lens")
                    if extend_prefix_lens is not None:
                        extend_lens = (seq_lens[:bs] - extend_prefix_lens[:bs]).to(
                            torch.int32
                        )
                    else:
                        # No prefix: all tokens are new
                        extend_lens = seq_lens[:bs].to(torch.int32)
                    query_start_loc = torch.zeros(
                        bs + 1, dtype=torch.int32, device=self.device
                    )
                    torch.cumsum(extend_lens, dim=0, out=query_start_loc[1:])
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        track_ssm_h_src = None
        track_ssm_h_dst = None
        track_conv_indices = None
        track_ssm_final_src = None
        track_ssm_final_dst = None
        if forward_mode.is_extend() and not forward_mode.is_target_verify():
            mamba_branching_seqlens = kwargs.get("mamba_branching_seqlens")
            extend_prefix_lens_kw = kwargs.get("extend_prefix_lens")
            mamba_track_pool_indices = kwargs.get("mamba_track_pool_indices")
            if (
                mamba_branching_seqlens is not None
                and extend_prefix_lens_kw is not None
                and mamba_track_pool_indices is not None
            ):
                branching = mamba_branching_seqlens[:bs].to(
                    dtype=torch.int32, device=self.device
                )
                prefix = extend_prefix_lens_kw[:bs].to(
                    dtype=torch.int32, device=self.device
                )
                track_indices = mamba_track_pool_indices[:bs].to(
                    dtype=torch.int32, device=self.device
                )
                extend_lens = (seq_lens[:bs] - prefix).to(torch.int32)
                checkpoint_mask = (track_indices >= 0) & (mamba_cache_indices >= 0)

                branch_lens = branching - prefix
                branch_inside = (
                    (branching > 0)
                    & checkpoint_mask
                    & (branch_lens > 0)
                    & (branch_lens < extend_lens)
                )
                track_mask = branch_inside & ((branch_lens % FLA_CHUNK_SIZE) == 0)
                if (branch_inside & ~track_mask).any():
                    raise RuntimeError(
                        "Mamba branching seqlen must be aligned to FLA chunk size"
                    )

                page_size = getattr(self.pool, "page_size", 1)
                final_lens = prefix + extend_lens
                final_mask = (
                    checkpoint_mask
                    & ~branch_inside
                    & (final_lens >= page_size)
                    & ((final_lens % page_size) == 0)
                )
                if final_mask.any():
                    track_ssm_final_src = mamba_cache_indices[final_mask]
                    track_ssm_final_dst = track_indices[final_mask]

                if track_mask.any():
                    (
                        track_ssm_h_src,
                        track_ssm_h_dst,
                    ) = self._compute_track_ssm_indices(
                        branch_lens,
                        track_mask,
                        track_indices,
                        seq_lens[:bs] - prefix,  # extend_seq_lens
                    )
                    track_conv_indices = self._compute_track_conv_indices(
                        query_start_loc,
                        branch_lens,
                        track_mask,
                    )

        self.forward_metadata = MambaForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            extend_prefix_lens=kwargs.get("extend_prefix_lens"),
            track_ssm_h_src=track_ssm_h_src,
            track_ssm_h_dst=track_ssm_h_dst,
            track_conv_indices=track_conv_indices,
            track_ssm_final_src=track_ssm_final_src,
            track_ssm_final_dst=track_ssm_final_dst,
        )

    def _compute_track_conv_indices(
        self,
        query_start_loc: torch.Tensor,
        branch_lens: torch.Tensor,
        track_mask: torch.Tensor,
    ):
        """Compute packed input indices for conv windows at tracked boundaries."""
        conv_state_len = self.pool.conv_state.shape[-1]
        lens_m = branch_lens[track_mask]
        start = query_start_loc[:-1][track_mask] + lens_m - conv_state_len
        indices = start.unsqueeze(-1) + torch.arange(
            conv_state_len,
            device=self.device,
            dtype=start.dtype,
        )
        return indices.clamp(0, query_start_loc[-1] - 1)

    def _compute_track_ssm_indices(
        self,
        branch_lens: torch.Tensor,
        track_mask: torch.Tensor,
        mamba_track_indices: torch.Tensor,
        extend_seq_lens: torch.Tensor,
    ):
        """Compute src/dst indices for extracting intermediate SSM states.

        Matching conv windows are gathered separately from packed pre-conv inputs.
        """
        num_h_states = (extend_seq_lens - 1) // FLA_CHUNK_SIZE + 1
        offset = torch.zeros_like(num_h_states)
        offset[1:] = torch.cumsum(num_h_states[:-1], dim=0)

        lens_m = branch_lens[track_mask]
        offset_m = offset[track_mask]
        dst_m = mamba_track_indices[track_mask]  # write to TRACKING slots

        # h[i] is the state before chunk i, so an aligned lens maps directly to
        # lens // FLA_CHUNK_SIZE.
        track_ssm_h_src = offset_m + (lens_m // FLA_CHUNK_SIZE)
        track_ssm_h_dst = dst_m

        return (
            track_ssm_h_src,
            track_ssm_h_dst,
        )

    # ---- CUDA graph state ----

    def init_cuda_graph_state(
        self, max_num_tokens: int, seq_lens_buf: torch.Tensor = None
    ):
        del seq_lens_buf  # mamba doesn't use seq_lens_buf.
        for i in range(max_num_tokens):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.query_start_loc_list.append(
                torch.empty((i + 2,), dtype=torch.int32, device=self.device)
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_num_tokens + 1, dtype=torch.int32, device=self.device
        )
        if self.speculative_num_draft_tokens > 0:
            # Need max_num_tokens+1 entries (one per request + sentinel).
            # Each entry is request_index * spec_num_draft_tokens.
            self.cached_cuda_graph_verify_query_start_loc = torch.arange(
                0,
                (max_num_tokens + 1) * self.speculative_num_draft_tokens,
                step=self.speculative_num_draft_tokens,
                dtype=torch.int32,
                device=self.device,
            )
        self._qsl_dirty = [False] * max_num_tokens
        self._qsl_last_mode = [None] * max_num_tokens

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        **kwargs,
    ):
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif forward_mode.is_target_verify() or forward_mode.is_draft_extend():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        mamba_pool_indices = kwargs.get("mamba_pool_indices")
        if mamba_pool_indices is not None:
            mamba_indices = self.pool.get_mamba_indices(mamba_pool_indices[:bs])
        else:
            mamba_indices = self.pool.get_mamba_indices(req_pool_indices[:bs])
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        self._qsl_dirty[bs - 1] = False
        self._qsl_last_mode[bs - 1] = forward_mode
        self.forward_metadata = MambaForwardMetadata(
            query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_cache_indices=self.state_indices_list[bs - 1],
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        req_to_page: torch.Tensor = None,
        **kwargs,
    ):
        num_padding = kwargs.get("num_padding", 0)
        mamba_pool_indices = kwargs.get("mamba_pool_indices")

        real_bs = bs - num_padding
        req_pool_indices = req_pool_indices[:bs]
        if mamba_pool_indices is not None:
            mamba_indices = self.pool.get_mamba_indices(mamba_pool_indices[:real_bs])
        else:
            mamba_indices = self.pool.get_mamba_indices(req_pool_indices[:real_bs])

        self.state_indices_list[bs - 1][:real_bs].copy_(mamba_indices)
        if num_padding > 0:
            self.state_indices_list[bs - 1][real_bs:].fill_(self.pad_slot_id)

        if num_padding == 0:
            need_copy = (
                self._qsl_dirty[bs - 1] or self._qsl_last_mode[bs - 1] != forward_mode
            )
            if need_copy:
                if forward_mode.is_decode_or_idle():
                    self.query_start_loc_list[bs - 1].copy_(
                        self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                    )
                elif forward_mode.is_target_verify() or forward_mode.is_draft_extend():
                    self.query_start_loc_list[bs - 1].copy_(
                        self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                    )
                self._qsl_dirty[bs - 1] = False
                self._qsl_last_mode[bs - 1] = forward_mode
        else:
            if forward_mode.is_decode_or_idle():
                self.query_start_loc_list[bs - 1][:real_bs].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[:real_bs]
                )
                self.query_start_loc_list[bs - 1][real_bs:].fill_(real_bs)
            elif forward_mode.is_target_verify() or forward_mode.is_draft_extend():
                self.query_start_loc_list[bs - 1][:real_bs].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[:real_bs]
                )
                self.query_start_loc_list[bs - 1][real_bs:].fill_(
                    real_bs * self.speculative_num_draft_tokens
                )
            else:
                raise ValueError(f"Invalid forward mode: {forward_mode=}")
            self._qsl_dirty[bs - 1] = True
            self._qsl_last_mode[bs - 1] = forward_mode

        self.forward_metadata = MambaForwardMetadata(
            query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_cache_indices=self.state_indices_list[bs - 1],
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ---- Forward ----

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        a = kwargs["a"]
        b = kwargs["b"]
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]

        conv_states, ssm_states, *rest = self.pool.get_mamba_params(layer_id)
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            conv_weights,
            bias,
            activation,
            conv_state_indices=cache_indices,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [
                key_dim // attn_tp_size,
                key_dim // attn_tp_size,
                value_dim // attn_tp_size,
            ],
            dim=-1,
        )
        seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        query = query.view(1, seq_len, num_heads, head_k_dim)
        key = key.view(1, seq_len, num_heads, head_k_dim)
        value = value.view(1, seq_len, value.shape[1] // head_v_dim, head_v_dim)

        global _blade_gdn_workspace
        # Four branches: FUSED_BLADE_GDN > BLADE_GDN > FLASHINFER_GDN > default triton
        if _USE_FUSED_BLADE_GDN:
            # ---- blade_gdn fused_gating_recur: single kernel, no gating pre-compute ---- 
            if _blade_gdn_workspace is None:
                _blade_gdn_workspace = torch.zeros(
                    (1000,), dtype=torch.int32, device=query.device
                )

            # 4D format: [batch=seq_len, S=1, H, D]
            q_4d = query.transpose(0, 1)   # [1, seq_len, H, D] -> [seq_len, 1, H, D]
            k_4d = key.transpose(0, 1)     # [seq_len, 1, H, D]
            v_4d = value.transpose(0, 1)   # [seq_len, 1, HV, DV]

            blade_output = torch.empty_like(v_4d)  # [seq_len, 1, HV, DV]

            tiny_gdn.gdn_main_recur_fused_gating(
                q_4d, k_4d, v_4d, A_log, a, dt_bias, b,
                blade_output, ssm_states, ssm_states, _blade_gdn_workspace,
                scale=None,
                req_ids=cache_indices,
                use_qk_l2_norm=True,
            )

            core_attn_out = blade_output.transpose(0, 1)  # [1, seq_len, HV, DV]

        elif _USE_BLADE_GDN:
            # ---- blade_gdn two-kernel: fused_gdn_gating + gdn_main_recur ----
            if _blade_gdn_workspace is None:
                _blade_gdn_workspace = torch.zeros(
                    (1000,), dtype=torch.int32, device=query.device
                )

            # Fused gating + beta computation (one triton kernel launch)
            # g: [1, seq_len, HV] fp32 log-space, beta: [1, seq_len, HV] sigmoid(b)
            g, beta_out = fused_gdn_gating(A_log, a, dt_bias, b=b)

            # 4D format: [batch=seq_len, S=1, H, D]
            q_4d = query.transpose(0, 1)
            k_4d = key.transpose(0, 1)
            v_4d = value.transpose(0, 1)
            g_4d = g.transpose(0, 1)
            beta_4d = beta_out.transpose(0, 1)

            blade_output = torch.empty_like(v_4d)

            tiny_gdn.gdn_main_recur(
                q_4d, k_4d, v_4d, g_4d, beta_4d, blade_output,
                ssm_states, ssm_states, _blade_gdn_workspace,
                scale=None,
                req_ids=cache_indices,
                use_qk_l2_norm=True,
            )

            core_attn_out = blade_output.transpose(0, 1)

        elif _USE_FLASHINFER_GDN:
            # ---- flashinfer decode (CUDA CuTe kernel) ----
            batch_size = cache_indices.shape[0]
            num_v_heads = value.shape[2]

            # flashinfer expects [B, 1, H, D]
            query_fi = query.view(batch_size, 1, num_heads, head_k_dim)
            key_fi = key.view(batch_size, 1, num_heads, head_k_dim)
            value_fi = value.view(batch_size, 1, num_v_heads, head_v_dim)
            a_fi = a.view(batch_size, 1, num_v_heads)
            b_fi = b.view(batch_size, 1, num_v_heads)

            if _flashinfer_gdn_use_pool:
                # SM100+ (Blackwell): pool API with indices
                output_fi, _ = _flashinfer_gdn_decode(
                    q=query_fi,
                    k=key_fi,
                    v=value_fi,
                    state=None,
                    A_log=A_log.float(),
                    a=a_fi,
                    dt_bias=dt_bias,
                    b=b_fi,
                    use_qk_l2norm=True,
                    initial_state=ssm_states,
                    initial_state_indices=cache_indices,
                )
            else:
                # SM90 (Hopper): gather/scatter
                state_batch = ssm_states[cache_indices]
                output_fi, new_state = _flashinfer_gdn_decode(
                    q=query_fi,
                    k=key_fi,
                    v=value_fi,
                    state=state_batch,
                    A_log=A_log.float(),
                    a=a_fi,
                    dt_bias=dt_bias,
                    b=b_fi,
                    scale=None,
                    use_qk_l2norm=True,
                )
                ssm_states[cache_indices] = new_state

            core_attn_out = output_fi.view(1, batch_size, num_v_heads, head_v_dim)

        else:
            # ---- Default: tokenspeed triton recurrent kernel ----
            core_attn_out = fused_sigmoid_gating_delta_rule_update(
                A_log=A_log,
                dt_bias=dt_bias,
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                cu_seqlens=query_start_loc,
                use_qk_l2norm_in_kernel=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
            )

        return core_attn_out

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        forward_mode: ForwardMode,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        a = kwargs["a"]
        b = kwargs["b"]
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]
        seq_len = kwargs["seq_len"]

        is_target_verify = forward_mode.is_target_verify()

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        if is_target_verify:
            draft_token_num = kwargs.get(
                "draft_token_num", self.speculative_num_draft_tokens
            )
            (
                conv_states,
                ssm_states,
                intermediate_state_cache,
                intermediate_conv_window_cache,
            ) = self.pool.get_mamba_params(layer_id)

            batch_size = seq_len // draft_token_num
            mixed_qkv_reshaped = (
                mixed_qkv.view(batch_size, draft_token_num, -1)
                .transpose(1, 2)
                .contiguous()
            )
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                conv_weights,
                bias,
                activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
            )
            mixed_qkv = (
                mixed_qkv_processed.transpose(1, 2).contiguous().view(seq_len, -1)
            )
        else:
            conv_states, ssm_states, *rest = self.pool.get_mamba_params(layer_id)
            extend_prefix_lens = kwargs.get("extend_prefix_lens")
            if extend_prefix_lens is None:
                extend_prefix_lens = self.forward_metadata.extend_prefix_lens
            extend_seq_lens_cpu = kwargs.get("extend_seq_lens_cpu")
            has_initial_states = (
                extend_prefix_lens > 0 if extend_prefix_lens is not None else None
            )
            need_h_track = (
                self.forward_metadata.track_ssm_h_src is not None
                and self.forward_metadata.track_ssm_h_src.numel() > 0
            )

            mixed_qkv_t = mixed_qkv.transpose(0, 1)
            if need_h_track:
                if self.forward_metadata.track_conv_indices is None:
                    raise RuntimeError(
                        "Missing conv indices for intermediate mamba track"
                    )
                conv_states[self.forward_metadata.track_ssm_h_dst] = mixed_qkv_t[
                    :, self.forward_metadata.track_conv_indices
                ].transpose(0, 1)

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv_t,
                conv_weights,
                bias,
                activation=activation,
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        key_split_dim = key_dim // attn_tp_size
        value_split_dim = value_dim // attn_tp_size

        query, key, value = torch.split(
            mixed_qkv,
            [key_split_dim, key_split_dim, value_split_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        num_value_heads = value.shape[1] // head_v_dim

        query = query.view(1, actual_seq_len, num_heads, head_k_dim)
        key = key.view(1, actual_seq_len, num_heads, head_k_dim)
        value = value.view(1, actual_seq_len, num_value_heads, head_v_dim)

        if is_target_verify:
            draft_token_num = kwargs.get(
                "draft_token_num", self.speculative_num_draft_tokens
            )
            global _blade_gdn_workspace
            if _USE_FUSED_BLADE_GDN:
                # ---- blade_gdn fused gating MTP: single kernel ----
                if _blade_gdn_workspace is None:
                    _blade_gdn_workspace = torch.zeros(
                        (1000,), dtype=torch.int32, device=query.device
                    )
                # 3D layout: rearrange("1 l h d -> l h d")
                q_3d = query.squeeze(0)   # [seq_len, H, D]
                k_3d = key.squeeze(0)
                v_3d = value.squeeze(0)
                core_attn_out = torch.empty(
                    value.shape, dtype=value.dtype, device=value.device
                )  # [1, seq_len, HV, DV]
                tiny_gdn.gdn_main_recur_fused_gating(
                    q_3d, k_3d, v_3d, A_log, a, dt_bias, b,
                    core_attn_out, ssm_states, ssm_states, _blade_gdn_workspace,
                    scale=None,
                    req_ids=cache_indices,
                    token_verify_ids=kwargs.get("num_accepted_tokens"),
                    query_start_loc=query_start_loc,
                    use_qk_l2_norm=True,
                )

            elif _USE_BLADE_GDN:
                # ---- blade_gdn two-kernel MTP: fused_gdn_gating + gdn_main_recur ----
                if _blade_gdn_workspace is None:
                    _blade_gdn_workspace = torch.zeros(
                        (1000,), dtype=torch.int32, device=query.device
                    )
                # Compute gating
                g, beta_out = fused_gdn_gating(A_log, a, dt_bias, b=b)
                # 3D layout: rearrange("1 l h d -> l h d")
                q_3d = query.squeeze(0)
                k_3d = key.squeeze(0)
                v_3d = value.squeeze(0)
                g_2d = g.squeeze(0)       # [seq_len, HV]
                beta_2d = beta_out.squeeze(0)
                core_attn_out = torch.empty(
                    value.shape, dtype=value.dtype, device=value.device
                )
                tiny_gdn.gdn_main_recur(
                    q_3d, k_3d, v_3d, g_2d, beta_2d, core_attn_out,
                    ssm_states, ssm_states, _blade_gdn_workspace,
                    scale=None,
                    req_ids=cache_indices,
                    token_verify_ids=kwargs.get("num_accepted_tokens"),
                    query_start_loc=query_start_loc,
                    use_qk_l2_norm=True,
                )

            else:
                # ---- Default: tokenspeed triton recurrent kernel ----
                core_attn_out = fused_sigmoid_gating_delta_rule_update(
                    A_log=A_log,
                    dt_bias=dt_bias,
                    q=query,
                    k=key,
                    v=value,
                    a=a,
                    b=b,
                    initial_state_source=ssm_states,
                    initial_state_indices=cache_indices,
                    cu_seqlens=query_start_loc,
                    use_qk_l2norm_in_kernel=True,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                    # target_verify specific parameters
                    disable_state_update=True,
                    intermediate_states_buffer=intermediate_state_cache,
                    cache_steps=draft_token_num,
                )
        else:
            beta = b.sigmoid()
            g = fused_gdn_gating(A_log, a, dt_bias)
            g = g.unsqueeze(0)
            beta = beta.unsqueeze(0)

            recurrent_state = ssm_states[cache_indices]
            need_final_track = (
                self.forward_metadata.track_ssm_final_src is not None
                and self.forward_metadata.track_ssm_final_src.numel() > 0
            )
            if need_h_track:
                core_attn_out, last_recurrent_state, h = chunk_gated_delta_rule(
                    q=query,
                    k=key,
                    v=value,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=True,
                    cu_seqlens=query_start_loc,
                    head_first=False,
                    use_qk_l2norm_in_kernel=True,
                    output_h=True,
                )
            else:
                core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                    q=query,
                    k=key,
                    v=value,
                    g=g,
                    beta=beta,
                    initial_state=recurrent_state,
                    output_final_state=True,
                    cu_seqlens=query_start_loc,
                    head_first=False,
                    use_qk_l2norm_in_kernel=True,
                )
            last_recurrent_state = last_recurrent_state.to(ssm_states.dtype, copy=False)
            ssm_states[cache_indices] = last_recurrent_state

            if need_h_track:
                if h is None:
                    raise RuntimeError(
                        "Missing intermediate mamba states for branching track"
                    )
                ssm_states[self.forward_metadata.track_ssm_h_dst] = h.squeeze(0)[
                    self.forward_metadata.track_ssm_h_src
                ].to(ssm_states.dtype, copy=False)

            if need_final_track:
                conv_states[self.forward_metadata.track_ssm_final_dst] = conv_states[
                    self.forward_metadata.track_ssm_final_src
                ]
                ssm_states[self.forward_metadata.track_ssm_final_dst] = ssm_states[
                    self.forward_metadata.track_ssm_final_src
                ]

        return core_attn_out


class HybridLinearAttnBackend(AttentionBackend):
    """Hybrid backend that routes between full attention and linear attention by layer ID."""

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

    def _backends(self):
        return [self.full_attn_backend, self.linear_attn_backend]

    def _backend_for_layer(self, layer_id: int) -> AttentionBackend:
        if self.linear_attn_backend is None or layer_id in self.full_attn_layers:
            return self.full_attn_backend
        return self.linear_attn_backend

    _MAMBA_KWARGS = frozenset(
        {
            "mamba_pool_indices",
            "mamba_cow_src_indices",
            "mamba_branching_seqlens",
            "mamba_track_pool_indices",
            "mamba_cache_chunk_size",
        }
    )

    @staticmethod
    def _split_mamba_kwargs(kwargs: dict) -> tuple[dict, dict]:
        mamba_kw = {}
        common_kw = {}
        for k, v in kwargs.items():
            if k in HybridLinearAttnBackend._MAMBA_KWARGS:
                mamba_kw[k] = v
            else:
                common_kw[k] = v
        return common_kw, mamba_kw

    # ---- Metadata delegation ----

    def init_forward_metadata(self, *args, **kwargs):
        common_kw, mamba_kw = self._split_mamba_kwargs(kwargs)
        self.full_attn_backend.init_forward_metadata(*args, **common_kw)
        self.linear_attn_backend.init_forward_metadata(*args, **common_kw, **mamba_kw)

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor):
        for backend in self._backends():
            backend.init_cuda_graph_state(max_bs, seq_lens_buf)

    def register_step_counter(self, step_counter):
        # Hybrid layerwise transfer needs one global step per model layer,
        # including both full-attention and mamba layers. Record steps in this
        # wrapper instead of in child backends to avoid double counting.
        self.step_counter = step_counter

    def init_forward_metadata_capture_cuda_graph(self, *args, **kwargs):
        common_kw, mamba_kw = self._split_mamba_kwargs(kwargs)
        self.full_attn_backend.init_forward_metadata_capture_cuda_graph(
            *args, **common_kw
        )
        self.linear_attn_backend.init_forward_metadata_capture_cuda_graph(
            *args, **common_kw, **mamba_kw
        )

    def init_forward_metadata_replay_cuda_graph(self, *args, **kwargs):
        common_kw, mamba_kw = self._split_mamba_kwargs(kwargs)
        self.full_attn_backend.init_forward_metadata_replay_cuda_graph(
            *args, **common_kw
        )
        self.linear_attn_backend.init_forward_metadata_replay_cuda_graph(
            *args, **common_kw, **mamba_kw
        )

    @property
    def support_kv_cache_prewrite(self) -> bool:
        return self.full_attn_backend.support_kv_cache_prewrite

    # ---- Forward dispatch ----

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc,
        token_to_kv_pool,
        forward_mode: ForwardMode = None,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        if forward_mode is None:
            return super().forward(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                forward_mode,
                save_kv_cache,
                **kwargs,
            )

        if forward_mode.is_idle():
            if layer is None:
                return torch.empty_like(kwargs["z"])
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        backend = self._backend_for_layer(layer_id)

        if forward_mode.is_decode():
            return backend.forward_decode(
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
            if backend is self.linear_attn_backend and forward_mode.is_target_verify():
                kwargs["is_target_verify"] = True
            step_counter = getattr(self, "step_counter", None)
            if (
                not forward_mode.is_idle()
                and step_counter is not None
                and not save_kv_cache
            ):
                step_counter.record_cache()
            ret = backend.forward_extend(
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
                and step_counter is not None
                and save_kv_cache
            ):
                step_counter.record_cache()
            return ret

    def forward_decode(self, q, k, v, layer, out_cache_loc, token_to_kv_pool, **kwargs):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        return self._backend_for_layer(layer_id).forward_decode(
            q, k, v, layer, out_cache_loc, token_to_kv_pool, **kwargs
        )

    def forward_extend(self, q, k, v, layer, out_cache_loc, token_to_kv_pool, **kwargs):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        return self._backend_for_layer(layer_id).forward_extend(
            q, k, v, layer, out_cache_loc, token_to_kv_pool, **kwargs
        )

    def update_mamba_state_after_mtp_verify(self, accepted_length, model):
        request_number = accepted_length.shape[0]
        state_indices_tensor = (
            self.linear_attn_backend.forward_metadata.mamba_cache_indices[
                :request_number
            ]
        )
        mamba_caches = self.linear_attn_backend.pool.get_mamba_params_all_layers()
        (
            conv_states,
            ssm_states,
            intermediate_state_cache,
            intermediate_conv_window_cache,
        ) = mamba_caches

        # blade_gdn already writes accepted state back to ssm_states internally
        # via token_verify_ids, skip ssm scatter to avoid overwriting with stale data
        if not (_USE_FUSED_BLADE_GDN or _USE_BLADE_GDN):
            fused_mamba_state_scatter_with_mask(
                ssm_states,
                intermediate_state_cache,
                state_indices_tensor,
                accepted_length,
            )
        fused_mamba_state_scatter_with_mask(
            conv_states,
            intermediate_conv_window_cache,
            state_indices_tensor,
            accepted_length,
        )
