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

"""Inkling attention backend wrapper: dense MHA + engine-side sconv state.

The C++ scheduler sees Inkling as a plain dense GQA model (KV pages only). The
sconv working state — four short-causal-conv streams per decoder block, a
ring of the last ``R`` input rows per request — is managed entirely
engine-side. The ring row of absolute position ``p`` is ``p % R``; positions
derive from the through-chunk ``seq_lens``, so there is no stored cursor and
rejected speculative rows are overwritten when their positions recur.

* ``InklingConvStatePool`` holds one channel-concatenated conv buffer per layer,
  sized by the request-pool capacity and indexed by ``req_pool_indices``
  (rank-local, 1-based, stable for a request's lifetime, reused only after
  completion — the same indices the dense KV path already uses).
* ``InklingAttnBackend`` wraps the plain ``MHAAttnBackend``: every attention
  call is delegated unchanged, while ``init_forward_metadata`` additionally
  derives the conv metadata (``InklingConvMetadata``) the model's sconv modules
  consume.

Prefix caching is supported when the conv state is fully paged (kvconv +
hiddenconv groups): cache-hit restores replay the conv columns from the
layers' own K/V slots. A fresh prefill still runs with
``has_initial_state=False`` so a reused slot's stale rolling state is ignored
and overwritten.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from tokenspeed_kernel import (
    rel_mha_decode_with_kvcache,
    rel_mha_extend_with_kvcache,
    rel_mha_plan,
    rel_mha_prefill,
)
from tokenspeed_kernel.ops.conv import seq_idx_from_cu_seqlens

from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    scrub_padding_tail,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import (
    AttentionBackend,
    init_backend_cuda_graph_state,
)
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.pdl import pdl_enabled

logger = get_colorful_logger(__name__)

# Matches the runtime causal_conv1d kernels' padded-slot sentinel.
PAD_SLOT_ID = -1


@dataclass
class InklingConvMetadata:
    """Per-forward metadata for the sconv state kernels.

    Attributes:
        query_start_loc: ``[bs + 1]`` int32 cumulative token offsets of the
            batch's sequences (decode: ``arange(bs + 1)``).
        cache_indices: ``[bs]`` int32 conv-pool slot per request
            (``req_pool_indices``; ``PAD_SLOT_ID`` marks padded rows).
        has_initial_state: ``[bs]`` bool; False for fresh prefills so stale
            slot contents are ignored.
        seq_idx: ``[total_tokens]`` int32 sequence id per token (decode:
            the cached arange — token t belongs to request t).
        seq_lens: ``[bs]`` int32 lengths THROUGH the chunk; the source of
            every ring position (chunk token ``t`` of request ``si`` sits at
            absolute position ``seq_lens[si] - (eos - t)``).
    """

    query_start_loc: torch.Tensor
    cache_indices: torch.Tensor
    has_initial_state: torch.Tensor
    seq_idx: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    # Extend rows in the batch: 0 (decode-family rounds) or the batch's
    # extend count. Selects the kernel (> 0 -> prefill: checkpoint taps,
    # gated ring writes; 0 -> decode: ring taps, write-all). True mixes are
    # rejected at metadata init; a future MIXED implementation would
    # partition two launches at this count.
    num_extends: int = 0
    # Target decode graphs keep endpoint restoration captured. This mask is
    # armed for exactly the first decode after a successful remote transfer.
    remote_restore_mask: torch.Tensor | None = None
    # Checkpoint groups: per-group tables {group: [bs, max_conv_blocks]}; None -> no paged groups.
    col_block_table: dict[str, torch.Tensor] | None = None


class InklingConvStatePool:
    """Engine-side working state (ring) for all sconv streams of all layers.

    Memory layout: ``[num_layers, num_slots, R, conv_dim]`` — the feature dim
    is contiguous (the ``tokenspeed_kernel.ops.conv`` kernels' contract). Ring
    row of absolute position ``p`` is ``p % R``; ``R >= (W-1) + K``
    so a round's pre-chunk tap reads and chunk-row writes never alias. The
    four streams of a block live at fixed channel offsets given by
    ``inkling_conv_stream_layout``; modules take channel slices.
    """

    def __init__(
        self,
        num_layers: int,
        num_slots: int,
        conv_dim: int,
        kernel_size: int,
        ring_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.conv_dim = conv_dim
        self.kernel_size = kernel_size
        self.ring_size = ring_size
        self.conv_state = torch.zeros(
            (num_layers, num_slots, ring_size, conv_dim),
            dtype=dtype,
            device=device,
        )
        # Request slots are reused. Admission clears this bit before RDMA and
        # a successful transfer arms it for the first target decode only.
        self.remote_restore_pending = torch.zeros(
            num_slots,
            dtype=torch.bool,
            device=device,
        )

    def layer_state_wd(self, layer_id: int) -> torch.Tensor:
        """One layer's ring in the native ``[num_slots, R, conv_dim]``
        layout (the tokenspeed_kernel ops/conv sconv kernels' contract)."""
        return self.conv_state[layer_id]

    def mem_usage_bytes(self) -> int:
        return self.conv_state.nbytes + self.remote_restore_pending.nbytes


class InklingAttnBackend(AttentionBackend):
    """Thin wrapper over the dense MHA backend adding conv metadata.

    All attention forwards and CUDA-graph hooks delegate to the wrapped
    backend; this class only derives ``InklingConvMetadata`` from the same
    arguments the dense path already receives, so the scheduler and executor
    are unaware anything beyond dense attention exists.
    """

    # Ask the graph wrapper for actual_bs at replay so padded rows can be marked PAD_SLOT_ID.
    uses_padded_decode_token_mask = True

    def __init__(
        self,
        inner: AttentionBackend,
        conv_pool: InklingConvStatePool,
        *,
        conv_columns: dict,
        spec_num_tokens: int = 1,
        is_draft: bool = False,
        enable_layerwise_cache_ready: bool = False,
    ):
        # Deliberately skip AttentionBackend.__init__: the wrapper mirrors inner via __getattr__.
        self.inner = inner
        self.conv_pool = conv_pool
        # Paged conv geometry (see _inkling_conv_columns). Mandatory: the
        # sconv state always has its paged bridges; there is no rolling mode.
        self.conv_columns = conv_columns
        # The conv groups are wrapper-owned: the inner mixin must skip their
        # write-loc math and capture buffers (see cache_groups.py).
        inner.engine_owned_group_ids = frozenset(conv_columns["group_block_tokens"])
        self.conv_metadata: InklingConvMetadata | None = None
        # Spec decoding: >1 means decode rounds carry this many tokens/request (verify / catch-up).
        self.conv_spec_num_tokens = max(1, int(spec_num_tokens))
        self.conv_is_draft = is_draft
        self.enable_layerwise_cache_ready = enable_layerwise_cache_ready
        # Persistent spec conv metadata buffers for CUDA graphs; sized in init_cuda_graph_state.
        self._graph_spec_qsl: torch.Tensor | None = None
        self._graph_spec_seq_idx: torch.Tensor | None = None
        # Persistent decode qsl (arange) keeps metadata CUDA-graph-capturable; grown to largest bs.
        self._decode_qsl: torch.Tensor | None = None
        # Persistent CUDA-graph conv metadata buffers; sized in init_cuda_graph_state.
        self._graph_cache_indices: torch.Tensor | None = None
        self._graph_has_initial_state: torch.Tensor | None = None
        self._graph_remote_restore_mask: torch.Tensor | None = None
        # Breakable-prefill-graph static conv metadata; None keeps the plain per-step path.
        self._pfg_seq_idx: torch.Tensor | None = None
        self._pfg_qsl: torch.Tensor | None = None
        self._pfg_prefix_lens: torch.Tensor | None = None
        self._pfg_seq_lens: torch.Tensor | None = None
        self._pfg_col_tables: dict[str, torch.Tensor] | None = None
        self._pfg_cache_indices: torch.Tensor | None = None
        self._pfg_has_initial_state: torch.Tensor | None = None
        self._pfg_max_bs = 0
        # Registered lazily by the model's four ShortConv sites. The buffers
        # are fixed LCM field views; target verify publishes them only after
        # accepted-length selection.
        self._checkpoint_streams: dict[
            tuple[int, int, int, str], tuple[torch.Tensor, ...]
        ] = {}

    def __getattr__(self, name):
        # Guard `inner` so a half-constructed wrapper raises AttributeError instead of recursing.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    # Class-level flags on AttentionBackend would shadow __getattr__; mirror inner's explicitly.
    @property
    def uses_paged_cache_groups(self):
        return self.inner.uses_paged_cache_groups

    @property
    def uses_cache_groups(self):
        return self.inner.uses_cache_groups

    @property
    def cache_consumer_families(self):
        return frozenset(getattr(self.inner, "cache_consumer_families", ())) | {"state"}

    def prepare_remote_cache_slots(self, slot_indices: list[int]) -> None:
        """Clear stale hydration state before publishing RDMA destinations."""
        self.conv_pool.remote_restore_pending[slot_indices] = False

    def mark_remote_cache_ready(self, slot_index: int) -> None:
        """Arm endpoint hydration after the complete remote transfer succeeds."""
        self.conv_pool.remote_restore_pending[slot_index] = True

    def _consume_remote_restore_mask(
        self,
        cache_indices: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        slots = cache_indices.to(torch.int64)
        valid = (slots > 0) & (slots < self.conv_pool.num_slots)
        safe_slots = slots.clamp(min=0, max=self.conv_pool.num_slots - 1)
        pending = self.conv_pool.remote_restore_pending[safe_slots]
        restore = pending & valid
        # Invalid/padded rows clamp to a sentinel slot but must not consume it.
        self.conv_pool.remote_restore_pending[safe_slots] = pending & ~valid
        if out is None:
            return restore
        out.copy_(restore)
        return out

    @staticmethod
    def _checkpoint_blocks_at_endpoints(
        table: torch.Tensor,
        lengths: torch.Tensor,
        checkpoint_granularity: int,
    ) -> torch.Tensor:
        """Resolve each positive endpoint to its latest-snapshot block."""
        lengths = lengths.to(torch.int64)
        valid = lengths > 0
        slots = torch.div(lengths - 1, checkpoint_granularity, rounding_mode="floor")
        slots = slots.clamp(min=0, max=table.shape[1] - 1)
        blocks = table.gather(1, slots.unsqueeze(1)).squeeze(1).to(torch.int32)
        torch._assert_async(
            ((~valid) | (blocks > 0)).all(),
            "ShortConv endpoint checkpoint is a hole or pad",
        )
        return torch.where(valid, blocks, torch.zeros_like(blocks))

    def restore_shortconv_endpoint(
        self,
        state: torch.Tensor,
        checkpoint_buffers: tuple[torch.Tensor, ...],
        metadata: InklingConvMetadata,
        group_id: str,
    ) -> None:
        """Restore a transferred prompt endpoint into a request's ring."""
        restore_mask = metadata.remote_restore_mask
        if restore_mask is None:
            return
        n = restore_mask.shape[0]
        qsl = metadata.query_start_loc[: n + 1].to(torch.int64)
        boundary = metadata.seq_lens[:n].to(torch.int64) - (qsl[1:] - qsl[:-1])
        masked_boundary = torch.where(
            restore_mask,
            boundary,
            torch.zeros_like(boundary),
        )
        pages = self._checkpoint_blocks_at_endpoints(
            metadata.col_block_table[group_id][:n],
            masked_boundary,
            int(self.conv_columns["block_tokens"]),
        )
        slots = metadata.cache_indices[:n].to(torch.int64)
        valid = restore_mask & (pages > 0) & (slots >= 0)
        pages = pages.to(torch.int64).clamp_min(0)
        slots = slots.clamp_min(0)
        restored = torch.cat(
            [buffer[pages].to(state.dtype) for buffer in checkpoint_buffers],
            dim=-1,
        )
        if restored.shape[-1] != state.shape[-1]:
            raise ValueError(
                f"ShortConv checkpoint width {restored.shape[-1]} does not "
                f"match ring width {state.shape[-1]}"
            )
        state_rows = restored.shape[1]
        rows = (
            boundary.view(n, 1)
            - state_rows
            + torch.arange(state_rows, device=state.device).view(1, state_rows)
        ).remainder(state.shape[1])
        current = state[slots.view(n, 1), rows]
        state[slots.view(n, 1), rows] = torch.where(
            valid.view(n, 1, 1),
            restored,
            current,
        )

    def publish_shortconv_endpoint(
        self,
        state: torch.Tensor,
        checkpoint_buffers: tuple[torch.Tensor, ...],
        metadata: InklingConvMetadata,
        group_id: str,
    ) -> None:
        """Publish the final ``W - 1`` ring rows to the PD snapshot block."""
        n = metadata.cache_indices.shape[0]
        lengths = metadata.seq_lens[:n].to(torch.int64)
        pages = self._checkpoint_blocks_at_endpoints(
            metadata.col_block_table[group_id][:n],
            lengths,
            int(self.conv_columns["block_tokens"]),
        )
        slots = metadata.cache_indices[:n].to(torch.int64)
        valid = (pages > 0) & (slots >= 0) & (lengths > 0)
        pages = pages.to(torch.int64).clamp_min(0)
        slots = slots.clamp_min(0)
        widths = tuple(int(buffer.shape[-1]) for buffer in checkpoint_buffers)
        state_rows = int(checkpoint_buffers[0].shape[1])
        absolute_rows = (
            lengths.view(n, 1)
            - state_rows
            + torch.arange(state_rows, device=state.device).view(1, state_rows)
        )
        rows = absolute_rows.remainder(state.shape[1])
        endpoint = state[slots.view(n, 1), rows]
        endpoint = torch.where(
            (absolute_rows >= 0).unsqueeze(-1),
            endpoint,
            torch.zeros_like(endpoint),
        )
        for buffer, values in zip(
            checkpoint_buffers,
            endpoint.split(widths, dim=-1),
            strict=True,
        ):
            current = buffer[pages]
            buffer[pages] = torch.where(
                valid.view(n, 1, 1),
                values.to(buffer.dtype),
                current,
            )

    # ------------------------------------------------------------------
    # Conv metadata
    # ------------------------------------------------------------------

    def _decode_query_start_loc(self, bs: int, device) -> torch.Tensor:
        if self._decode_qsl is None or self._decode_qsl.shape[0] < bs + 1:
            size = max(bs + 1, 256)
            self._decode_qsl = torch.arange(size, dtype=torch.int32, device=device)
        return self._decode_qsl[: bs + 1]

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        page_table: torch.Tensor,
        forward_mode: ForwardMode,
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        **kwargs,
    ):
        if forward_mode.is_mixed():
            raise RuntimeError(
                "Inkling sconv does not support MIXED batches: the prefill "
                "kernel hard-codes checkpoint taps at aligned chunk starts "
                "and decode rows have none (run without --enable-mixed-batch)"
            )
        # Paged sconv: conv groups ride block_tables, which the inner backend sheds — grab here.
        group_tables = kwargs.get("block_tables") or {}
        extend_total = (
            int(sum(extend_seq_lens_cpu[:bs]))
            if forward_mode.is_extend_or_mixed() and extend_seq_lens_cpu is not None
            else None
        )
        # In-bucket extends must use armed PFG statics: captured sconv kernels baked their addresses.
        pfg_total = -1
        if (
            self._pfg_seq_idx is not None
            and extend_total is not None
            and extend_total <= self._pfg_seq_idx.shape[0]
            and bs <= self._pfg_max_bs
        ):
            pfg_total = extend_total
        groups = set(self.conv_columns["group_block_tokens"])
        found = {g: group_tables.get(g) for g in groups}
        missing = sorted(g for g, t in found.items() if t is None)
        if missing:
            raise RuntimeError(
                f"paged sconv: block_tables is missing conv groups {missing}; "
                "the paged conv bridges are mandatory (no rolling fallback)"
            )
        if pfg_total >= 0:
            # The stream-ordered copy into the statics doubles as the plain path's clone() snapshot.
            col_block_table = self._pfg_refresh_col_tables(found, bs)
        else:
            # clone(): the scheduler can recycle these live tables while extend kernels are in flight.
            col_block_table = {g: t.clone() for g, t in found.items()}
        self.inner.init_forward_metadata(
            bs,
            num_extends,
            req_pool_indices,
            seq_lens,
            page_table,
            forward_mode,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            **kwargs,
        )

        cache_indices = req_pool_indices[:bs].to(torch.int32)
        seq_idx = None
        if forward_mode.is_extend_or_mixed():
            assert extend_seq_lens is not None and extend_prefix_lens is not None
            # Reuse the cumsum the inner backend just computed for this batch.
            inner_md = getattr(self.inner, "forward_extend_metadata", None)
            if inner_md is not None:
                query_start_loc = inner_md.cu_extend_seq_lens
            else:
                query_start_loc = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens[:bs], dim=0, dtype=torch.int32),
                    (1, 0),
                )
            has_initial_state = extend_prefix_lens[:bs] > 0
            if extend_total is not None:
                seq_idx = seq_idx_from_cu_seqlens(query_start_loc, extend_total)
            if pfg_total >= 0:
                # PFG statics: tail qsl closes the PAD request's empty chunk; tail seq_idx marks pads PAD.
                self._pfg_qsl[: bs + 1].copy_(query_start_loc)
                self._pfg_qsl[bs + 1 :].fill_(pfg_total)
                self._pfg_seq_idx[:pfg_total].copy_(seq_idx)
                self._pfg_seq_idx[pfg_total:].fill_(self._pfg_max_bs)
                self._pfg_prefix_lens[:bs].copy_(extend_prefix_lens[:bs])
                self._pfg_prefix_lens[bs:].zero_()
                self._pfg_seq_lens[:bs].copy_(seq_lens[:bs])
                self._pfg_seq_lens[bs:].zero_()
                self._pfg_cache_indices[:bs].copy_(cache_indices)
                self._pfg_cache_indices[bs:].fill_(PAD_SLOT_ID)
                self._pfg_has_initial_state[:bs].copy_(has_initial_state)
                self._pfg_has_initial_state[bs:].zero_()
                query_start_loc = self._pfg_qsl
                seq_idx = self._pfg_seq_idx
                cache_indices = self._pfg_cache_indices
                has_initial_state = self._pfg_has_initial_state
        elif forward_mode.is_decode() and self.conv_spec_num_tokens > 1:
            # Multi-token decode: target verify / draft catch-up, k rows per
            # request written speculatively at their ring positions.
            k = self.conv_spec_num_tokens
            device = req_pool_indices.device
            query_start_loc = torch.arange(
                0, bs * k + 1, step=k, dtype=torch.int32, device=device
            )
            seq_idx = seq_idx_from_cu_seqlens(query_start_loc, bs * k)
            has_initial_state = torch.ones(bs, dtype=torch.bool, device=device)
        else:
            query_start_loc = self._decode_query_start_loc(bs, req_pool_indices.device)
            # Decode: token t belongs to request t, so seq_idx is the same
            # cached arange, one element shorter.
            seq_idx = query_start_loc[:bs]
            has_initial_state = torch.ones(
                bs, dtype=torch.bool, device=req_pool_indices.device
            )
        remote_restore_mask = None
        if (
            forward_mode.is_decode()
            and self.conv_spec_num_tokens == 1
            and self.conv_columns.get("pd_endpoint_snapshots", False)
        ):
            remote_restore_mask = self._consume_remote_restore_mask(cache_indices[:bs])
        self.conv_metadata = InklingConvMetadata(
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            seq_idx=seq_idx,
            seq_lens=(
                self._pfg_seq_lens if pfg_total >= 0 else seq_lens[:bs].to(torch.int32)
            ),
            col_block_table=col_block_table,
            num_extends=num_extends,
            remote_restore_mask=remote_restore_mask,
        )

    # ------------------------------------------------------------------
    # Speculative-decoding conv metadata
    # ------------------------------------------------------------------

    def fixed_workspace_bytes(self) -> int:
        """Return persistent ShortConv state owned outside the LCM arenas."""
        return self.conv_pool.mem_usage_bytes()

    def _spec_conv_metadata(self, bs: int) -> InklingConvMetadata:
        """Multi-token decode conv metadata over the persistent CUDA-graph
        buffers (target verify / draft catch-up)."""
        k = self.conv_spec_num_tokens
        return InklingConvMetadata(
            query_start_loc=self._graph_spec_qsl[: bs + 1],
            cache_indices=self._graph_cache_indices[:bs],
            has_initial_state=self._graph_has_initial_state[:bs],
            seq_idx=self._graph_spec_seq_idx[: bs * k],
            seq_lens=self._graph_seq_lens[:bs],
            col_block_table={
                g: table[:bs] for g, table in self._graph_col_tables.items()
            },
        )

    def _graph_decode_conv_metadata(self, bs: int) -> InklingConvMetadata:
        """Single-token decode conv metadata over the persistent CUDA-graph
        buffers (shared by graph capture and replay)."""
        return InklingConvMetadata(
            query_start_loc=self._decode_qsl[: bs + 1],
            cache_indices=self._graph_cache_indices[:bs],
            has_initial_state=self._graph_has_initial_state[:bs],
            seq_idx=self._decode_qsl[:bs],
            seq_lens=self._graph_seq_lens[:bs],
            col_block_table={g: t[:bs] for g, t in self._graph_col_tables.items()},
            remote_restore_mask=(
                self._graph_remote_restore_mask[:bs]
                if self.conv_columns.get("pd_endpoint_snapshots", False)
                else None
            ),
        )

    def update_draft_forward_metadata(self, frontier: torch.Tensor) -> None:
        """Re-anchor the k-row conv metadata and the inner backend's
        seq_lens/write locs to end at ``frontier`` ([bs] int32). Accept-
        dependent, so pure tensor ops — recomputed per graph replay; the
        next round's ``init_forward_metadata`` resets."""
        self.inner.update_draft_forward_metadata(frontier)
        # Paged bridges ride through: the in-kernel publish resolves pages
        # by position, so boundaries rewritten by the re-anchored rows are
        # re-published with committed content.
        self.conv_metadata = replace(self.conv_metadata, seq_lens=frontier)

    def register_shortconv_checkpoint_stream(
        self,
        *,
        layer_id: int,
        channel_offset: int,
        dim: int,
        group_id: str,
        buffers: tuple[torch.Tensor, ...],
    ) -> None:
        """Record one fixed checkpoint view for post-verify publication."""
        key = (layer_id, channel_offset, dim, group_id)
        existing = self._checkpoint_streams.setdefault(key, buffers)
        if len(existing) != len(buffers) or any(
            lhs.data_ptr() != rhs.data_ptr()
            or lhs.storage_offset() != rhs.storage_offset()
            or lhs.shape != rhs.shape
            or lhs.stride() != rhs.stride()
            for lhs, rhs in zip(existing, buffers)
        ):
            raise RuntimeError(
                f"ShortConv checkpoint stream {key!r} changed storage buffer"
            )

    # ------------------------------------------------------------------
    # Attention delegation
    # ------------------------------------------------------------------

    # forward is NOT overridden: base dispatch sends rel_logits layers to the rel_mha overrides.

    def _rel_decode_cu_seqlens_q(
        self, bs: int, max_seqlen_q: int, device
    ) -> torch.Tensor:
        """Cached ``arange(bs + 1) * max_seqlen_q`` for rel decode.

        Cached PER ``max_seqlen_q``: one wrapper instance can see several
        row-per-request shapes (non-spec decode at 1, spec windows at
        ``spec_num_tokens``), and a single keyed-on-last-step buffer would be
        reallocated on every switch — invalidating the pointer captured CUDA
        graphs hold.
        Grown buffers are retained (never freed): their static contents stay
        correct for any graph that recorded them.
        """
        cache = getattr(self, "_rel_qsl_cache", None)
        if cache is None:
            cache = self._rel_qsl_cache = {}
            self._rel_qsl_retired = []
        buf = cache.get(max_seqlen_q)
        if buf is None or buf.shape[0] < bs + 1:
            if buf is not None:
                self._rel_qsl_retired.append(buf)
            size = max(bs + 1, 256)
            buf = torch.arange(size, dtype=torch.int32, device=device) * max_seqlen_q
            cache[max_seqlen_q] = buf
        return buf[: bs + 1]

    def forward_decode(
        self,
        q,
        k,
        v,
        layer,
        out_cache_loc,
        token_to_kv_pool,
        bs,
        save_kv_cache=True,
        **kwargs,
    ):
        rel_logits = kwargs.pop("rel_logits", None)
        tau = kwargs.pop("log_scaling_tau", None)
        if rel_logits is None:
            return self.inner.forward_decode(
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
        inner = self.inner
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        if k is not None:
            k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
            v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        metadata = inner.forward_decode_metadata
        out_cache_loc = inner._select_out_cache_loc(layer, metadata, out_cache_loc)
        if save_kv_cache:
            # Decode-side rows and write locs must agree exactly: a shorter
            # loc vector would make _save_kv_cache silently TRIM the rows
            # (dropping most of a multi-token window's KV — the grouped-cache
            # draft accept regression), a longer one would crash the store.
            assert k is None or out_cache_loc.shape[0] == k.shape[0], (
                f"Inkling decode KV write: {k.shape[0]} rows vs "
                f"{out_cache_loc.shape[0]} write locs (layer "
                f"{layer.layer_id}, group {layer.group_id!r}); a chaining "
                "one-row-per-step draft loop is unsupported with grouped cache."
            )
            inner._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)
        scale_kwargs = {}
        if inner.is_mxfp8:
            q, q_sf = inner._quantize_mxfp8_tokens(q)
            k_sf, v_sf = token_to_kv_pool.get_kv_scale_buffer(layer.layer_id)
            scale_kwargs = dict(q_scale=q_sf, k_scale=k_sf, v_scale=v_sf)
        elif inner.is_fp8:
            q = q.to(torch.float8_e4m3fn)
        k_cache, v_cache = inner._get_kv_cache(layer, token_to_kv_pool)
        n_reqs = metadata.seq_lens.shape[0]
        max_seqlen_q = q.shape[0] // n_reqs
        output = rel_mha_decode_with_kvcache(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=inner._select_page_table(layer, metadata),
            cache_seqlens=metadata.seq_lens,
            max_seqlen_k=inner.max_context_len,
            rel_logits=rel_logits,
            cu_seqlens_q=self._rel_decode_cu_seqlens_q(n_reqs, max_seqlen_q, q.device),
            max_seqlen_q=max_seqlen_q,
            window_left=layer.sliding_window_size,
            softmax_scale=layer.scaling,
            tau=tau,
            enable_pdl=pdl_enabled(),
            solution=inner.kernel_solution,
            **scale_kwargs,
        )
        return output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        out_cache_loc,
        token_to_kv_pool,
        bs,
        save_kv_cache=False,
        **kwargs,
    ):
        rel_logits = kwargs.pop("rel_logits", None)
        tau = kwargs.pop("log_scaling_tau", None)
        if rel_logits is None:
            return self.inner.forward_extend(
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
        inner = self.inner
        q = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        metadata = inner.forward_extend_metadata
        _num_real = metadata.cu_extend_seq_lens_cpu[-1]
        # Relative attention keeps bucket-shaped inputs because rel_logits and
        # its handoff are bucket-shaped. Scrub the padded rows instead of using
        # the plain MHA path's exact-row kernel contract.
        scrub_padding_tail(_num_real, q, k, v)
        out_cache_loc = inner._select_out_cache_loc(layer, metadata, out_cache_loc)
        plan = rel_mha_plan(
            dtype=torch.float8_e4m3fn if inner.is_fp8 else inner.qkv_dtype,
            head_dim=inner.head_dim,
            window_left=layer.sliding_window_size,
            return_lse=False,
            solution=inner.kernel_solution,
        )
        if metadata.max_extend_prefix_len == 0 and plan["extend_mode"] == "postwrite":
            if inner.is_fp8:
                q = q.to(torch.float8_e4m3fn)
                k = k.to(torch.float8_e4m3fn)
                v = v.to(torch.float8_e4m3fn)
            output = rel_mha_prefill(
                q=q,
                k=k,
                v=v,
                rel_logits=rel_logits,
                cu_seqlens=metadata.cu_extend_seq_lens,
                cu_seqlens_cpu=metadata.cu_extend_seq_lens_cpu,
                max_seqlen=metadata.max_extend_seq_len,
                window_left=layer.sliding_window_size,
                softmax_scale=layer.scaling,
                tau=tau,
                enable_pdl=pdl_enabled(),
                solution=inner.kernel_solution,
            )
            output = output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
            if output.shape[0] > _num_real:
                output[_num_real:].zero_()
            if save_kv_cache:
                inner._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)
            return output
        if save_kv_cache:
            inner._save_kv_cache(layer, out_cache_loc, token_to_kv_pool, k, v)
        scale_kwargs = {}
        if inner.is_mxfp8:
            q, q_sf = inner._quantize_mxfp8_tokens(q)
            k_sf, v_sf = token_to_kv_pool.get_kv_scale_buffer(layer.layer_id)
            scale_kwargs = dict(q_scale=q_sf, k_scale=k_sf, v_scale=v_sf)
        elif inner.is_fp8:
            q = q.to(torch.float8_e4m3fn)
        k_cache, v_cache = inner._get_kv_cache(layer, token_to_kv_pool)
        output = rel_mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=metadata.cu_extend_seq_lens,
            cu_seqlens_kv=metadata.cu_seqlens_kv,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=inner._select_page_table(layer, metadata),
            cache_seqlens=metadata.seq_lens,
            max_seqlen_q=metadata.max_extend_seq_len,
            max_seqlen_k=inner.max_context_len,
            rel_logits=rel_logits,
            window_left=layer.sliding_window_size,
            softmax_scale=layer.scaling,
            tau=tau,
            enable_pdl=pdl_enabled(),
            solution=inner.kernel_solution,
            **scale_kwargs,
        )
        output = output.reshape(-1, layer.tp_q_head_num * layer.v_head_dim)
        if output.shape[0] > _num_real:
            output[_num_real:].zero_()
        return output

    def support_kv_cache_prewrite(self, forward_mode: ForwardMode | None = None):
        return self.inner.support_kv_cache_prewrite(forward_mode)

    def configure_runtime(self, **kwargs) -> None:
        self.inner.configure_runtime(**kwargs)

    def register_step_counter(self, step_counter):
        # The cache layer includes ShortConv fields written after wrapped
        # attention, so InklingDecoderLayer owns the layer-ready boundary.
        self.step_counter = step_counter

    @break_point
    def record_layer_cache_ready(
        self,
        hidden_states: torch.Tensor,
        forward_mode: ForwardMode,
    ) -> torch.Tensor:
        if not forward_mode.is_decode() and not forward_mode.is_idle():
            self.step_counter.record_cache()
        return hidden_states

    # ------------------------------------------------------------------
    # CUDA graph hooks (decode-only, like the inner backend's)
    # ------------------------------------------------------------------

    def init_prefill_graph_state(self, max_num_tokens: int, max_bs: int) -> None:
        """Allocate the static conv metadata the breakable prefill graphs bake.

        Captured sconv prefill kernels hold capture-time device addresses, so
        once this is called EVERY extend (eager, capture or replayed) routes
        its conv metadata through these persistent buffers, refreshed by
        stream-ordered device copies in :meth:`init_forward_metadata` (which
        also makes the per-step table ``clone()`` snapshot unnecessary).
        Replay pads the token count up to the captured bucket; padded tokens
        carry ``seq_idx == max_bs`` — the PAD request row: an empty chunk
        (``cu_seqlens[max_bs:]`` holds the step's real token count), zero
        prefix and an all ``-1`` (hole) table row, so pad tokens read only
        in-bounds x rows, write garbage into discarded ``y`` rows and persist
        nothing (the pool store is masked on ``block >= 0``).

        Args:
            max_num_tokens: Largest captured token bucket (sizes ``seq_idx``;
                extends beyond it run eager and skip the static route).
            max_bs: Request capacity; also the PAD request row index.

        Raises:
            RuntimeError: When any conv site still runs the rolling-state
                path — its cache-update grid is batch-shaped, which a
                token-bucket graph cannot serve for other batch sizes. The
                caller treats this as capture failure and (world-agreed)
                degrades to eager prefill.
        """
        geo = self.conv_columns
        device = self.conv_pool.conv_state.device
        self._pfg_max_bs = min(max_bs, self.conv_pool.num_slots - 2)
        self._pfg_seq_idx = torch.full(
            (max_num_tokens,), self._pfg_max_bs, dtype=torch.int32, device=device
        )
        self._pfg_qsl = torch.zeros(
            self._pfg_max_bs + 2, dtype=torch.int32, device=device
        )
        self._pfg_prefix_lens = torch.zeros(
            self._pfg_max_bs + 1, dtype=torch.int32, device=device
        )
        self._pfg_seq_lens = torch.zeros(
            self._pfg_max_bs + 1, dtype=torch.int32, device=device
        )
        self._pfg_cache_indices = torch.full(
            (self._pfg_max_bs + 1,),
            PAD_SLOT_ID,
            dtype=torch.int32,
            device=device,
        )
        self._pfg_has_initial_state = torch.zeros(
            self._pfg_max_bs + 1, dtype=torch.bool, device=device
        )
        self._pfg_col_tables = {
            g: torch.full(
                (self._pfg_max_bs + 1, -(-self.inner.max_context_len // bt)),
                -1,
                dtype=torch.int32,
                device=device,
            )
            for g, bt in geo["group_block_tokens"].items()
        }

    def _pfg_refresh_col_tables(
        self, found: dict[str, torch.Tensor | None], bs: int
    ) -> dict[str, torch.Tensor]:
        """Copy this step's live conv tables into the prefill-graph statics.

        Only ``[0:bs, 0:live_width]`` needs refreshing: a request's prefix
        taps and persist columns stay under ``ceil(seq_len / BT)`` <= the
        live table width, rows in ``(bs, max_bs)`` are pointed at by no
        ``seq_idx``, and the PAD row (``max_bs``) has been all ``-1`` since
        init. The device-side copy is stream-ordered, so it doubles as the
        snapshot the eager path otherwise takes with ``clone()``.
        """
        tables = {}
        for g, src in found.items():
            buf = self._pfg_col_tables[g]
            if src is not None and bs > 0:
                rows = min(src.shape[0], bs)
                cols = min(src.shape[1], buf.shape[1])
                buf[:rows, :cols].copy_(src[:rows, :cols])
            tables[g] = buf
        return tables

    def init_cuda_graph_state(self, max_bs: int, **kwargs):
        init_backend_cuda_graph_state(self.inner, max_bs, **kwargs)
        device = self.conv_pool.conv_state.device
        self._decode_qsl = torch.arange(max_bs + 1, dtype=torch.int32, device=device)
        # Own the cache-seqlens buffer instead of aliasing the controller's
        # seq_lens_buf; replay copies the live lengths in, so graph state does
        # not depend on the controller mutating a shared tensor in place.
        self._graph_seq_lens = torch.zeros(max_bs, dtype=torch.int32, device=device)
        # Adopted stacked views are filled by the mixin's packed unpack; pad rows hit dummy slot 0.
        inner_tabs = getattr(self.inner, "cuda_graph_page_tables", {})
        groups = self.conv_columns["group_block_tokens"]
        self._graph_col_tables_adopted = all(g in inner_tabs for g in groups)
        if self._graph_col_tables_adopted:
            self._graph_col_tables = {g: inner_tabs[g] for g in groups}
        else:
            self._graph_col_tables = {
                g: torch.full(
                    (max_bs, -(-self.inner.max_context_len // bt)),
                    1,
                    dtype=torch.int32,
                    device=device,
                )
                for g, bt in groups.items()
            }
        self._graph_cache_indices = torch.full(
            (max_bs,), PAD_SLOT_ID, dtype=torch.int32, device=device
        )
        self._graph_has_initial_state = torch.ones(
            max_bs, dtype=torch.bool, device=device
        )
        self._graph_remote_restore_mask = torch.zeros(
            max_bs,
            dtype=torch.bool,
            device=device,
        )
        if self.conv_spec_num_tokens > 1:
            k = self.conv_spec_num_tokens
            # Static-content spec buffers at fixed addresses; recorded kernels slice per-bs views.
            self._graph_spec_qsl = torch.arange(
                0, max_bs * k + 1, k, dtype=torch.int32, device=device
            )
            self._graph_spec_seq_idx = torch.repeat_interleave(
                torch.arange(max_bs, dtype=torch.int32, device=device), k
            )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        **kwargs,
    ):
        self.inner.init_forward_metadata_capture_cuda_graph(
            bs, req_pool_indices, seq_lens, forward_mode, **kwargs
        )
        assert self._graph_cache_indices is not None
        # Seed the owned buffer: paged conv reads pos = seq_len - 1, so an
        # unseeded (zero) length would address position -1 during capture.
        self._graph_seq_lens[:bs].copy_(seq_lens[:bs])
        if self.conv_spec_num_tokens > 1:
            # k-token spec chunk (target verify / draft window).
            self.conv_metadata = self._spec_conv_metadata(bs)
            return
        if self.conv_columns.get("pd_endpoint_snapshots", False):
            self._graph_remote_restore_mask[:bs].zero_()
        self.conv_metadata = self._graph_decode_conv_metadata(bs)

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode = None,
        page_table: torch.Tensor = None,
        **kwargs,
    ):
        actual_bs = kwargs.pop("actual_bs", None)
        self.inner.init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            forward_mode=forward_mode,
            page_table=page_table,
            **kwargs,
        )
        assert self._graph_cache_indices is not None
        self._graph_seq_lens[:bs].copy_(seq_lens[:bs])
        self._graph_cache_indices[:bs].copy_(req_pool_indices[:bs].to(torch.int32))
        if actual_bs is not None and actual_bs < bs:
            # Pad rows may carry stale indices aliasing LIVE slots; PAD_SLOT_ID keeps writes off them.
            self._graph_cache_indices[actual_bs:bs].fill_(PAD_SLOT_ID)
        group_tables = kwargs.get("block_tables") or {}
        adopted_filled = getattr(self, "_graph_col_tables_adopted", False) and getattr(
            self.inner, "_packed_group_unpack_ran", False
        )
        for g, buf in self._graph_col_tables.items():
            src = group_tables.get(g)
            if src is None:
                raise RuntimeError(
                    f"paged sconv replay: no {g!r} table in " "block_tables"
                )
            if adopted_filled:
                # The inner mixin's packed unpack already filled the shared stack rows this step.
                continue
            cols = min(src.shape[1], buf.shape[1])
            rows = min(src.shape[0], bs)
            buf[:rows, :cols].copy_(src[:rows, :cols])
            if cols < buf.shape[1]:
                buf[:rows, cols:].fill_(-1)
            if rows < bs:
                buf[rows:bs].fill_(-1)
            if actual_bs is not None and actual_bs < min(bs, rows):
                buf[actual_bs:bs].fill_(-1)
        if self.conv_spec_num_tokens > 1:
            # Rebuild so the eager post-verify hook (outside the graph) sees this round's bs and mode.
            self.conv_metadata = self._spec_conv_metadata(bs)
            return
        if self.conv_columns.get("pd_endpoint_snapshots", False):
            self._consume_remote_restore_mask(
                self._graph_cache_indices[:bs],
                out=self._graph_remote_restore_mask[:bs],
            )
        self.conv_metadata = self._graph_decode_conv_metadata(bs)
