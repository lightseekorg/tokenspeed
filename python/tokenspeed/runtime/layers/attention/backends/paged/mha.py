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

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_plan,
    mha_prefill,
)
from tokenspeed_kernel.ops.kvcache.triton import (
    fused_fp8_set_kv_buffer,
)
from tokenspeed_kernel.ops.quantization import quantize_mxfp8

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.breakable_cuda_graph import slice_to_real_tokens
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.paged.base import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


_KERNEL_SOLUTION_BY_BACKEND = {
    "mha": None,
    "fa3": "fa3",
    "fa4": "fa4",
    "triton": "triton",
    "flashinfer": "flashinfer",
}


def _slice_extend_inputs(metadata, q, k, v):
    """Remove prefill-graph padding rows before calling an attention kernel.

    The live cu-seqlens describe only real tokens. Some kernels tolerate extra
    zero rows, but others still derive work from the tensor shape. Use the
    pinned CPU mirror (sync-free) so every solution receives the same exact-row
    contract. No-op on normal unpadded forwards.
    """
    return slice_to_real_tokens(metadata.cu_extend_seq_lens_cpu[-1], q, k, v)


def trim_kv_to_locs(out_cache_loc, k, v):
    """Slice a padded KV write down to the write-loc count.

    Prefill-graph replay pads k/v rows to the bucket while the write
    locations cover only the real (leading) rows. Trimming beats padding the
    locs with the null page: kernels that don't scrub tail rows would write
    garbage into page 0, breaking its stays-zero invariant. No-op off the
    padded path.
    """
    n = out_cache_loc.shape[0]
    if k is not None and k.shape[0] > n:
        return k[:n], v[:n]
    return k, v


@dataclass(kw_only=True)
class MHAExtendMetadata:
    # Device-side metadata:
    # - seq_lens: total length after this step
    # - extend_seq_lens: length of new tokens
    #   cu_extend_seq_lens: the cumsum version of extend_seq_lens
    #   cu_seqlens_kv: the cumsum version of seq_lens
    # - extend_prefix_lens: length of the cached prefix tokens
    # seq_lens[i] = extend_prefix_lens[i] + extend_seq_lens[i]
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    extend_seq_lens: torch.Tensor
    cu_extend_seq_lens: torch.Tensor
    cu_seqlens_kv: torch.Tensor
    extend_prefix_lens: torch.Tensor
    extend_seq_lens_cpu: list[int]
    cu_extend_seq_lens_cpu: list[int]
    max_extend_seq_len: int
    max_extend_prefix_len: int = 0


@dataclass(kw_only=True)
class MHADecodeMetadata:
    page_table: torch.Tensor
    seq_lens: torch.Tensor


class MHAAttnBackend(PagedAttentionBackend):
    """Standard MHA leaf routed through tokenspeed_kernel attention APIs."""

    # Every kernel call site forwards layer.sliding_window_size.
    supports_layer_sliding_window: bool = True

    def __init__(self, config: AttnConfig, spec: MHAConfig, *, kernel_page_size: int):
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        # Map the selected backend to the corresponding kernel solution string.
        backend_name = spec.backend_name or "mha"
        self.kernel_solution = _KERNEL_SOLUTION_BY_BACKEND[backend_name]

        self.tp_q_head_num = max(spec.num_attention_heads // spec.attn_tp_size, 1)
        self.tp_kv_head_num = max(spec.num_kv_heads // spec.attn_tp_size, 1)
        self.qkv_dtype = config.dtype
        self.kv_cache_dtype = config.kv_cache_dtype
        self.is_mxfp8 = bool(config.kv_cache_mxfp8)
        # mxfp8 shares the fp8 storage dtype but uses block scales; keep it off
        # the per-tensor casts
        self.is_fp8 = (
            self.kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            and not self.is_mxfp8
        )
        self.plan = partial(
            mha_plan,
            dtype=(
                torch.float8_e4m3fn
                if self.is_mxfp8
                else (self.kv_cache_dtype if self.is_fp8 else self.qkv_dtype)
            ),
            head_dim=self.head_dim,
            return_lse=False,
            solution=self.kernel_solution,
        )
        # DFLASH draft: the whole block in one decode forward, with one
        # decode metadata entry per block position and uniform non-causal
        # seq_lens (block_decode_expansion).
        self.draft_block_decode = bool(config.draft_block_decode)

        self.forward_decode_metadata: MHADecodeMetadata | None = None
        self.forward_extend_metadata: MHAExtendMetadata | None = None

    def support_kv_cache_prewrite(
        self, forward_mode: ForwardMode | None = None
    ) -> bool:
        return forward_mode is not None and forward_mode.is_decode()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

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
    ) -> None:
        assert not forward_mode.is_mixed(), "mha backend does not support mixed batch"
        if not forward_mode.is_extend_or_mixed():
            raise RuntimeError(
                "MHA decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend ({forward_mode})"
            )

        seq_lens = seq_lens[:bs]
        extend_seq_lens = extend_seq_lens[:bs]
        extend_seq_lens_cpu = [int(x) for x in extend_seq_lens_cpu[:bs].tolist()]
        cu_extend_seq_lens = torch.nn.functional.pad(
            torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32),
            (1, 0),
        )
        cu_extend_seq_lens_cpu = [0]
        for length in extend_seq_lens_cpu:
            cu_extend_seq_lens_cpu.append(cu_extend_seq_lens_cpu[-1] + length)
        cu_seqlens_kv = torch.nn.functional.pad(
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
            (1, 0),
        )
        self.forward_extend_metadata = MHAExtendMetadata(
            page_table=page_table[:bs],
            seq_lens=seq_lens,
            extend_seq_lens=extend_seq_lens,
            cu_extend_seq_lens=cu_extend_seq_lens,
            cu_seqlens_kv=cu_seqlens_kv,
            extend_prefix_lens=extend_prefix_lens[:bs],
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            cu_extend_seq_lens_cpu=cu_extend_seq_lens_cpu,
            max_extend_seq_len=max(extend_seq_lens_cpu),
            max_extend_prefix_len=int(extend_prefix_lens_cpu[:bs].max().item()),
        )

    def _decode_views(self, bs: int) -> MHADecodeMetadata:
        """Per-bs decode metadata views over the persistent buffers: one
        builder for capture and refresh, cached per bs — pointer-stable."""
        metadata = self._decode_views_by_bs.get(bs)
        if metadata is None:
            expanded_bs = bs * self.block_decode_expansion
            metadata = MHADecodeMetadata(
                page_table=self.page_table_buf[:expanded_bs],
                seq_lens=self.seq_lens_buf[:expanded_bs],
            )
            self._decode_views_by_bs[bs] = metadata
        return metadata

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
        del num_extends
        if self.block_decode_active:
            # DFLASH draft: replicate each request's page table to its
            # spec_num_tokens block positions. Under replay the block-end
            # seq_lens are re-derived inside the captured graph
            # (fill_block_decode_seq_lens); eager has no in-graph writer, and
            # the capture seeding needs the same safe baseline.
            spec = self.spec_num_tokens
            self.page_table_buf[: bs * spec].view(bs, spec, self.max_num_pages).copy_(
                page_table[:bs, None, :]
            )
            if not for_graph_replay or actual_bs == 0:
                self.fill_block_decode_seq_lens(bs, seq_lens)
        else:
            # Clamp short requests (padded requests replay at seq_len 1) to
            # the verify floor: verify derives per-token lengths as
            # seq - N + t + 1, which must stay positive. Plain decode and
            # drafts have floor 1 (identity).
            torch.clamp_min(
                seq_lens[:bs], self.verify_floor, out=self.seq_lens_buf[:bs]
            )
            self.page_table_buf[:bs].copy_(page_table[:bs])
        self.forward_decode_metadata = self._decode_views(bs)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        assert layer.qk_head_dim == layer.v_head_dim
        assert (k is None) == (v is None)

        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        if k is not None:
            k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
            v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        metadata = self.forward_decode_metadata
        if save_kv_cache:
            self._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)

        scale_kwargs = {}
        if self.is_mxfp8:
            q, q_sf = self._quantize_mxfp8_tokens(q)
            k_sf, v_sf = token_to_kv_pool.get_kv_scale_buffer(layer.layer_id)
            scale_kwargs = dict(q_scale=q_sf, k_scale=k_sf, v_scale=v_sf)
        elif self.is_fp8:
            q = q.to(self.kv_cache_dtype)

        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        max_seqlen_q = q.shape[0] // metadata.seq_lens.shape[0]
        output = mha_decode_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=metadata.page_table,
            cache_seqlens=metadata.seq_lens,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=kwargs.get("sinks"),
            max_seqlen_k=self.max_context_len,
            max_seqlen_q=max_seqlen_q,
            solution=self.kernel_solution,
            **scale_kwargs,
        )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        bs: int,
        save_kv_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        assert layer.qk_head_dim == layer.v_head_dim
        assert (k is None) == (v is None)
        assert k is not None

        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)

        metadata = self.forward_extend_metadata
        sinks = kwargs.get("sinks")
        plan = self.plan(
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
        )
        extend_mode = plan.get("extend_mode", "prewrite")
        if metadata.max_extend_prefix_len == 0 and extend_mode == "postwrite":
            return self._forward_prefill(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                metadata,
                save_kv_cache,
                sinks,
            )
        return self._forward_extend(
            q,
            k,
            v,
            layer,
            out_cache_loc,
            token_to_kv_pool,
            metadata,
            save_kv_cache,
            sinks,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHAExtendMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        q, k, v = _slice_extend_inputs(metadata, q, k, v)
        # TODO: use a custom kernel to do downcast
        if self.is_fp8:
            q = q.to(self.kv_cache_dtype)
            k = k.to(self.kv_cache_dtype)
            v = v.to(self.kv_cache_dtype)

        output = mha_prefill(
            q=q,
            k=k,
            v=v,
            cu_seqlens=metadata.cu_extend_seq_lens,
            cu_seqlens_cpu=metadata.cu_extend_seq_lens_cpu,
            max_seqlen=metadata.max_extend_seq_len,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            solution=self.kernel_solution,
        )
        output = output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
        if save_kv_cache:
            self._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)
        return output

    def _forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        metadata: MHAExtendMetadata,
        save_kv_cache: bool,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        q, k, v = _slice_extend_inputs(metadata, q, k, v)
        if save_kv_cache:
            # KV store (incl. the mxfp8 quantize-on-store path) lives solely
            # in _save_kv_cache.
            self._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)

        scale_kwargs = {}
        if self.is_mxfp8:
            q, q_sf = self._quantize_mxfp8_tokens(q)
            k_sf, v_sf = token_to_kv_pool.get_kv_scale_buffer(layer.layer_id)
            scale_kwargs = dict(q_scale=q_sf, k_scale=k_sf, v_scale=v_sf)
        elif self.is_fp8:
            q = q.to(self.kv_cache_dtype)

        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        output = mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=metadata.cu_extend_seq_lens,
            cu_seqlens_kv=metadata.cu_seqlens_kv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=metadata.page_table,
            cache_seqlens=metadata.seq_lens,
            max_seqlen_q=metadata.max_extend_seq_len,
            max_seqlen_k=self.max_context_len,
            is_causal=True,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            solution=self.kernel_solution,
            **scale_kwargs,
        )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------
    # KV store helpers
    # ------------------------------------------------------------------

    def _store_kv_mxfp8(self, layer, loc, token_to_kv_pool, k, v) -> None:
        """MXFP8 quantize-on-store: one fused launch when the pool supports
        it (bit-identical, PDL-chained), else the split 5-launch path.
        The sole caller (_save_kv_cache) has already trimmed k/v to loc."""
        fused = getattr(token_to_kv_pool, "quantize_and_set_kv_buffer", None)
        if fused is not None and fused(layer, loc, k, v):
            return
        k_q, k_sf = self._quantize_mxfp8_tokens(k)
        v_q, v_sf = self._quantize_mxfp8_tokens(v)
        token_to_kv_pool.set_kv_buffer(layer, loc, k_q, v_q, k_scale=k_sf, v_scale=v_sf)

    def _quantize_mxfp8_tokens(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token MXFP8: (fp8-e4m3 [T, H, D], UE8M0 scales [T, H, D // 32]).

        Accepts [T, H, D] or [T, H * D]; H is inferred so the same helper
        serves q (tp_q_head_num) and k/v (tp_kv_head_num).
        """
        t, d = x.shape[0], self.head_dim
        h = x.numel() // (t * d)
        # (A PDL triton variant measured 0.07 ms slower e2e at decode Q shapes; flashinfer stays)
        data, sf = quantize_mxfp8(x.reshape(t * h, d))
        return (
            data.view(t, h, d),
            sf.view(torch.float8_e8m0fnu).view(t, h, d // 32),
        )

    def _save_kv_cache(
        self,
        layer: PagedAttention,
        out_cache_loc: torch.Tensor,
        token_to_kv_pool,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
    ) -> None:
        if k is None:
            return
        k, v = trim_kv_to_locs(out_cache_loc, k, v)

        if self.is_mxfp8:
            # Quantize-on-store: fp8 data + per-token e8m0 scales into the
            # paged interleaved layout
            self._store_kv_mxfp8(layer, out_cache_loc, token_to_kv_pool, k, v)
        elif self.is_fp8:
            k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
            fused_fp8_set_kv_buffer(
                k=k,
                v=v,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_loc=out_cache_loc,
                k_scale=layer.k_scale,
                v_scale=layer.v_scale,
                page_size=self.kernel_page_size,
            )
        else:
            token_to_kv_pool.set_kv_buffer(
                layer,
                out_cache_loc,
                k,
                v,
                layer.k_scale,
                layer.v_scale,
            )

    def _get_kv_cache(self, layer: PagedAttention, token_to_kv_pool):
        # Page ids are in this leaf's kernel-page units (the router expanded
        # them), so view the cache at that size.
        k_cache = token_to_kv_pool.get_key_buffer(layer.layer_id).view(
            -1,
            self.kernel_page_size,
            layer.tp_k_head_num,
            layer.qk_head_dim,
        )
        v_cache = token_to_kv_pool.get_value_buffer(layer.layer_id).view(
            -1,
            self.kernel_page_size,
            layer.tp_v_head_num,
            layer.v_head_dim,
        )
        return k_cache, v_cache


for _backend_name in _KERNEL_SOLUTION_BY_BACKEND:
    register_backend(_backend_name, {AttentionArch.MHA}, MHAAttnBackend)
