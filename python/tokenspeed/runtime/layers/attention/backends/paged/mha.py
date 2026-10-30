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
from tokenspeed_kernel.ops.attention.mha import (
    mha_decode_schedule,
    mha_decode_schedule_policy,
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_plan,
    mha_prefill,
)
from tokenspeed_kernel.ops.quantization import quantize_mxfp8
from tokenspeed_kernel.ops.tuning import is_warming_up

from tokenspeed.runtime.configs.model_config import AttentionArch
from tokenspeed.runtime.execution.breakable_cuda_graph import slice_to_real_tokens
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.query_shard import QueryShardPlan
from tokenspeed.runtime.layers.attention.backends.base import reject_query_shard
from tokenspeed.runtime.layers.attention.backends.paged.base import (
    PagedAttentionBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.registry import register_backend

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.paged_attention import PagedAttention


_KERNEL_SOLUTION_BY_BACKEND = {
    "mha": None,
    "fa3": "fa3",
    "fa4": "fa4",
    "triton": "triton",
    "flashinfer": "flashinfer",
}


# The decode kernel variant whose full-attention layers split each request's
# keys by a schedule balanced over the step's lengths (mha_decode_schedule)
# instead of the default's even split of every request.
DECODE_VARIANT_BALANCED = "balanced_splits"


def _fp8_scale_kwargs(layer: PagedAttention) -> dict:
    """The per-tensor scales an FP8 cache was written with, for the kernel's read."""
    scales = layer.fp8_kv_scales
    return {} if scales is None else {"fp8_kv_scales": (scales.k, scales.v)}


def _slice_extend_inputs(metadata, *tensors):
    """Remove prefill-graph padding rows before calling an attention kernel.

    The live cu-seqlens describe only real tokens. Some kernels tolerate extra
    zero rows, but others still derive work from the tensor shape. Use the
    pinned CPU mirror (sync-free) so every solution receives the same exact-row
    contract. No-op on normal unpadded forwards.
    """
    return slice_to_real_tokens(metadata.cu_extend_seq_lens_cpu[-1], *tensors)


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
    # Bound on seq_lens for the cached-prefix kernel, which may size its work
    # (e.g. split counts) from it: the longest request when every length is
    # known on the host, else the context limit.
    max_seqlen_k: int
    # Mixed target queries are cache-backed even when the leading prefills
    # have no prefix; the fresh-prefill path must not discard decode history.
    is_mixed: bool
    max_extend_prefix_len: int = 0


@dataclass(kw_only=True)
class MHADecodeMetadata:
    page_table: torch.Tensor
    seq_lens: torch.Tensor


class MHAAttnBackend(PagedAttentionBackend):
    """Standard MHA leaf routed through tokenspeed_kernel attention APIs.

    Decode kernel variant ``balanced_splits``: where the kernel library
    offers a split schedule for this leaf's full-attention calls (gfx950)
    and the context limit can reach where it pays, the target's decode and
    verify can split each request's keys by a schedule balanced over the
    step's lengths. ``mha_decode_schedule_policy`` picks it per step from
    the requests' page counts; the refresh then computes the schedule
    (``mha_decode_schedule``) in place into ``decode_schedule_buf``, which
    every full-attention layer of the leaf's shape reads.
    """

    # Every kernel call site forwards layer.sliding_window_size.
    supports_layer_sliding_window: bool = True

    def __init__(self, config: AttnConfig, spec: MHAConfig, *, kernel_page_size: int):
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        # Map the selected backend to the corresponding kernel solution string.
        backend_name = spec.backend_name or "mha"
        self.kernel_solution = _KERNEL_SOLUTION_BY_BACKEND[backend_name]

        self.skip_softmax_threshold = spec.skip_softmax_threshold

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
        # The balanced variant: its host policy (resolved once, lazily;
        # False when there is none), whether the current step runs it, the
        # schedule its refresh computes in place, and which layers read it.
        self._balanced_policy = None
        self._balanced = False
        self.decode_schedule_buf: torch.Tensor | None = None
        self._schedule_calls: dict[int, dict] = {}
        self._layer_reads_schedule: dict[tuple[int, int], bool] = {}

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        super()._publish_cache_pool(cache_pool)
        self.forward_decode_metadata = None
        self.forward_extend_metadata = None
        self.decode_schedule_buf = None

    def supports_narrowed_draft_decode(self, forward_mode: ForwardMode) -> bool:
        # Decode metadata exists only in decode rounds.
        return forward_mode.is_decode()

    def reads_fp8_kv_scales(self, layer: PagedAttention) -> bool:
        # Both reads hand them to the kernel (_fp8_scale_kwargs).
        return True

    def query_dtype(
        self, layer: PagedAttention, forward_mode: ForwardMode, activation: torch.dtype
    ) -> torch.dtype:
        # Decode reads an FP8 cache with FP8 queries. A fresh prefill attends
        # the activation-dtype K/V rows instead, so extend converts its own.
        if self.is_fp8 and forward_mode.is_decode():
            return self.kv_cache_dtype
        return activation

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
        query_shard: QueryShardPlan | None,
        page_table_cpu: torch.Tensor | None,
        **kwargs,
    ) -> None:
        reject_query_shard(query_shard, "MHAAttnBackend")
        del page_table_cpu
        if not forward_mode.is_extend_or_mixed():
            raise RuntimeError(
                "MHA decode metadata goes through refresh_decode_metadata; "
                f"init_forward_metadata only serves extend/mixed ({forward_mode})"
            )
        is_mixed = forward_mode.is_mixed()
        if is_mixed and self.is_draft and not self.draft_block_decode:
            raise RuntimeError("MHA mixed draft metadata requires block decoding")
        # Block drafters share the target's mixed metadata initialization.
        # Before their model forward, they refresh decode metadata and publish
        # a uniform block window starting at each request's accepted prefix.
        # This does not enable a mixed forward for ordinary token drafters.
        if is_mixed and not 0 < num_extends < bs:
            raise ValueError(
                "MHA mixed metadata requires both prefill and decode requests"
            )

        seq_lens = seq_lens[:bs]
        prefill_prefix_lens = extend_prefix_lens[:num_extends]
        query_lens_cpu = [int(x) for x in extend_seq_lens_cpu[:num_extends].tolist()]
        if is_mixed:
            # The runner supplies extend lengths only for the leading prefill
            # requests. Decode/verify queries follow them in request order and
            # attend a causal suffix of each request's already-written cache.
            query_lens = seq_lens.new_full((bs,), self.spec_num_tokens)
            query_lens[:num_extends] = extend_seq_lens[:num_extends]
            query_lens_cpu.extend([self.spec_num_tokens] * (bs - num_extends))
            prefix_lens = seq_lens - query_lens
        else:
            query_lens = extend_seq_lens[:bs]
            prefix_lens = prefill_prefix_lens
        cu_extend_seq_lens = torch.nn.functional.pad(
            torch.cumsum(query_lens, dim=0, dtype=torch.int32),
            (1, 0),
        )
        cu_extend_seq_lens_cpu = [0]
        for length in query_lens_cpu:
            cu_extend_seq_lens_cpu.append(cu_extend_seq_lens_cpu[-1] + length)
        cu_seqlens_kv = torch.nn.functional.pad(
            torch.cumsum(seq_lens, dim=0, dtype=torch.int32),
            (1, 0),
        )
        # Mixed batches keep the context limit: their decode lengths live only
        # on the device.
        max_seqlen_k = self.max_context_len
        if not is_mixed:
            max_seqlen_k = int(
                (
                    extend_prefix_lens_cpu[:num_extends]
                    + extend_seq_lens_cpu[:num_extends]
                )
                .max()
                .item()
            )
        self.forward_extend_metadata = MHAExtendMetadata(
            page_table=page_table[:bs],
            seq_lens=seq_lens,
            extend_seq_lens=query_lens,
            cu_extend_seq_lens=cu_extend_seq_lens,
            cu_seqlens_kv=cu_seqlens_kv,
            extend_prefix_lens=prefix_lens,
            extend_seq_lens_cpu=query_lens_cpu,
            cu_extend_seq_lens_cpu=cu_extend_seq_lens_cpu,
            max_extend_seq_len=max(query_lens_cpu),
            max_seqlen_k=max_seqlen_k,
            is_mixed=is_mixed,
            # Only prefills have a host prefix mirror. Mixed batches always
            # use cached attention, without reading decode lengths back to CPU.
            max_extend_prefix_len=int(
                extend_prefix_lens_cpu[:num_extends].max().item()
            ),
        )

    def init_cuda_graph_state(self, max_bs: int) -> None:
        super().init_cuda_graph_state(max_bs)
        # One header row and one row per request (mha_decode_schedule).
        self.decode_schedule_buf = torch.zeros(
            (self.seq_lens_buf.shape[0] + 1, 2), dtype=torch.int32, device=self.device
        )

    # ------------------------------------------------------------------
    # Balanced decode variant
    # ------------------------------------------------------------------

    def _schedule_call(self, rows: int) -> dict:
        """Meta tensors with the shapes and dtypes of this leaf's
        full-attention decode calls over ``rows`` query rows, and their
        options, for the schedule API (it reads only metadata); cached per
        row count."""
        call = self._schedule_calls.get(rows)
        if call is None:
            query_dtype = self.kv_cache_dtype if self.is_fp8 else self.qkv_dtype
            kv = torch.empty(
                (1, self.kernel_page_size, self.tp_kv_head_num, self.head_dim),
                dtype=self.kv_cache_dtype,
                device="meta",
            )
            call = dict(
                q=torch.empty(
                    (rows, self.tp_q_head_num, self.head_dim),
                    dtype=query_dtype,
                    device="meta",
                ),
                k_cache=kv,
                v_cache=kv,
                max_seqlen_q=self.verify_floor,
                solution=self.kernel_solution,
            )
            self._schedule_calls[rows] = call
        return call

    def _policy(self):
        """The balanced variant's host policy for this leaf's calls, resolved
        once; None when it has none. The target's decode or verify only:
        drafters keep the default (their round graphs are captured once per
        batch size)."""
        if self._balanced_policy is None:
            policy = None
            if not (self.is_draft or self.block_decode_active or self.is_mxfp8):
                policy = mha_decode_schedule_policy(
                    max_seqlen_k=self.max_context_len,
                    **self._schedule_call(self.verify_floor),
                )
            self._balanced_policy = policy or False
        return self._balanced_policy or None

    def decode_variant_min_pages(self) -> int:
        policy = self._policy()
        return 0 if policy is None else policy.min_pages

    def decode_kernel_variants(self, bs: int) -> tuple[str, ...]:
        # Past max_requests every request's row blocks already fill the
        # schedule's work items, so the variant cannot win there.
        policy = self._policy()
        if policy is None or bs > policy.max_requests:
            return ()
        return (DECODE_VARIANT_BALANCED,)

    def prefer_decode_kernel_variant(self, pages) -> str | None:
        if not self.decode_kernel_variants(len(pages)):
            return None
        return DECODE_VARIANT_BALANCED if self._policy().prefers(pages) else None

    def use_decode_kernel_variant(self, variant: str | None) -> None:
        if variant is not None and (
            variant != DECODE_VARIANT_BALANCED or self._policy() is None
        ):
            raise ValueError(f"MHAAttnBackend has no decode variant {variant!r}")
        self._balanced = variant is not None

    def _refresh_decode_schedule(self, bs: int) -> None:
        """Under the balanced variant, compute the step's schedule from the
        lengths the refresh just wrote, in place, before any layer (or a
        graph replay) reads it."""
        if not self._balanced:
            return
        schedule = mha_decode_schedule(
            cache_seqlens=self.seq_lens_buf[:bs],
            out=self.decode_schedule_buf,
            **self._schedule_call(bs * self.verify_floor),
        )
        assert schedule is not None, "the balanced variant was offered"

    def _decode_schedule(
        self,
        layer: PagedAttention,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        max_seqlen_q: int,
        sinks: torch.Tensor | None,
        scale_kwargs: dict,
    ) -> torch.Tensor | None:
        """The step's schedule for a layer call that reads one: the balanced
        variant's full-attention layers of the shape it was computed for,
        whose arguments select a schedule-reading kernel."""
        if not self._balanced or layer.sliding_window_size >= 0:
            return None
        key = (layer.layer_id, max_seqlen_q)
        reads = self._layer_reads_schedule.get(key)
        if reads is None:
            reads = (
                q.shape[1] == self.tp_q_head_num
                and k_cache.shape[2] == self.tp_kv_head_num
                and max_seqlen_q == self.verify_floor
                and mha_decode_schedule_policy(
                    q=q,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=self.max_context_len,
                    window_left=-1,
                    logit_cap=layer.logit_cap,
                    sinks=sinks,
                    return_lse=False,
                    q_scale=None,
                    k_scale=None,
                    v_scale=None,
                    fp8_kv_scales=scale_kwargs.get("fp8_kv_scales"),
                    solution=self.kernel_solution,
                )
                is not None
            )
            self._layer_reads_schedule[key] = reads
        return self.decode_schedule_buf if reads else None

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
        self._refresh_decode_schedule(bs)

    def decode_buffer_targets(
        self, bs: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.block_decode_active:
            return None
        return self.seq_lens_buf[:bs], self.page_table_buf[:bs]

    @property
    def decode_seq_lens_floor(self) -> int:
        return self.verify_floor

    def publish_decode_metadata(
        self, bs: int, seq_lens: torch.Tensor, *, num_extends: int
    ) -> None:
        del seq_lens, num_extends
        self.forward_decode_metadata = self._decode_views(bs)
        self._refresh_decode_schedule(bs)

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
        **kwargs,
    ) -> torch.Tensor:
        assert layer.qk_head_dim == layer.v_head_dim

        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        metadata = self.forward_decode_metadata

        scale_kwargs = {}
        if self.is_mxfp8:
            q, q_sf = self._quantize_mxfp8_tokens(q)
            k_sf, v_sf = token_to_kv_pool.get_kv_scale_buffer(layer.layer_id)
            scale_kwargs = dict(q_scale=q_sf, k_scale=k_sf, v_scale=v_sf)
        elif self.is_fp8:
            # A no-op on the prologue's decode query (see query_dtype).
            q = q.to(self.kv_cache_dtype)
            scale_kwargs = _fp8_scale_kwargs(layer)

        k_cache, v_cache = self._get_kv_cache(layer, token_to_kv_pool)
        max_seqlen_q = q.shape[0] // metadata.seq_lens.shape[0]
        sinks = kwargs.get("sinks")
        schedule = self._decode_schedule(
            layer, q, k_cache, v_cache, max_seqlen_q, sinks, scale_kwargs
        )
        output = mha_decode_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=metadata.page_table,
            cache_seqlens=metadata.seq_lens,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            max_seqlen_k=self.max_context_len,
            max_seqlen_q=max_seqlen_q,
            schedule=schedule,
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
        **kwargs,
    ) -> torch.Tensor:
        assert layer.qk_head_dim == layer.v_head_dim
        assert k is not None and v is not None

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
        if (
            not metadata.is_mixed
            and metadata.max_extend_prefix_len == 0
            and extend_mode == "postwrite"
        ):
            if is_warming_up():
                # Startup warms kernels with a fresh prefill, which skips the
                # cached-prefix kernel a later prefix-cache hit runs. Run that
                # too, on the same rows, so it compiles its variants now; its
                # output is not used.
                self._forward_extend(q, layer, token_to_kv_pool, metadata, sinks)
            return self._forward_prefill(q, k, v, layer, metadata, sinks)
        return self._forward_extend(q, layer, token_to_kv_pool, metadata, sinks)

    def _forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: PagedAttention,
        metadata: MHAExtendMetadata,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        q, k, v = _slice_extend_inputs(metadata, q, k, v)
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
            skip_softmax_threshold=self.skip_softmax_threshold,
            solution=self.kernel_solution,
        )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _forward_extend(
        self,
        q: torch.Tensor,
        layer: PagedAttention,
        token_to_kv_pool,
        metadata: MHAExtendMetadata,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        (q,) = _slice_extend_inputs(metadata, q)

        scale_kwargs = {}
        if self.is_mxfp8:
            q, q_sf = self._quantize_mxfp8_tokens(q)
            k_sf, v_sf = token_to_kv_pool.get_kv_scale_buffer(layer.layer_id)
            scale_kwargs = dict(q_scale=q_sf, k_scale=k_sf, v_scale=v_sf)
        elif self.is_fp8:
            q = q.to(self.kv_cache_dtype)
            scale_kwargs = _fp8_scale_kwargs(layer)

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
            max_seqlen_k=metadata.max_seqlen_k,
            is_causal=True,
            window_left=layer.sliding_window_size,
            logit_cap=layer.logit_cap,
            sinks=sinks,
            solution=self.kernel_solution,
            **scale_kwargs,
        )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------
    # MXFP8 helpers
    # ------------------------------------------------------------------

    def _quantize_mxfp8_tokens(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token MXFP8: (fp8-e4m3 [T, H, D], UE8M0 scales [T, H, D // 32]).

        Accepts query rows as [T, H, D] or [T, H * D].
        """
        t, d = x.shape[0], self.head_dim
        h = x.numel() // (t * d)
        # (A PDL triton variant measured 0.07 ms slower e2e at decode Q shapes; flashinfer stays)
        data, sf = quantize_mxfp8(x.reshape(t * h, d))
        return (
            data.view(t, h, d),
            sf.view(torch.float8_e8m0fnu).view(t, h, d // 32),
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
