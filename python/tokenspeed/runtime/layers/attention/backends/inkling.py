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

from dataclasses import dataclass

import torch
from tokenspeed_kernel import (
    rel_mha_decode_with_kvcache,
    rel_mha_extend_with_kvcache,
    rel_mha_plan,
    rel_mha_prefill,
)
from tokenspeed_kernel.ops.conv import inkling_ring_restore, seq_idx_from_cu_seqlens

from tokenspeed.runtime.execution.breakable_cuda_graph import scrub_padding_tail
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
        is_decode: True when this is a single-token-per-request decode batch.
        seq_idx: ``[total_tokens]`` int32 sequence id per token (decode:
            the cached arange — token t belongs to request t).
        seq_lens: ``[bs]`` int32 lengths THROUGH the chunk; the source of
            every ring position (chunk token ``t`` of request ``si`` sits at
            absolute position ``seq_lens[si] - (eos - t)``).
    """

    query_start_loc: torch.Tensor
    cache_indices: torch.Tensor
    has_initial_state: torch.Tensor
    is_decode: bool
    seq_idx: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    # Extend/mixed chunks can exceed the compute kernel's in-kernel
    # persistence bound (R - (W-1)); True runs inkling_ring_sconv_update after it.
    needs_ring_update: bool = False
    # Checkpoint restore is admission-only: True only on extend/mixed batches
    # where some request has an aligned prefix hit (host-checked), so decode
    # rounds and cold prefills skip the restore ops entirely.
    needs_restore: bool = False
    # Checkpoint groups: per-group tables {group: [bs, max_conv_blocks]}; None -> no paged groups.
    col_page_table: dict[str, torch.Tensor] | None = None


class InklingConvStatePool:
    """Engine-side working state (ring) for all sconv streams of all layers.

    Memory layout: ``[num_layers, num_slots, R, conv_dim]`` — the feature dim
    is contiguous (the ``tokenspeed_kernel.ops.conv`` kernels' contract). Ring
    row of absolute position ``p`` is ``p % R``; ``R >= (W-1) + K + lookback``
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

    def layer_state_wd(self, layer_id: int) -> torch.Tensor:
        """One layer's ring in the native ``[num_slots, R, conv_dim]``
        layout (the tokenspeed_kernel ops/conv sconv kernels' contract)."""
        return self.conv_state[layer_id]

    def mem_usage_bytes(self) -> int:
        return self.conv_state.numel() * self.conv_state.element_size()


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
        # Draft decode-window lookback: D > 0 makes the catch-up chunk carry
        # D extra leading rows that re-run committed positions (ring reads go
        # D deeper). Configured by the drafter via configure_draft_lookback.
        self._draft_lookback = 0
        # Persistent spec conv metadata buffers for CUDA graphs; sized in init_cuda_graph_state.
        self._graph_spec_qsl: torch.Tensor | None = None
        self._graph_spec_seq_idx: torch.Tensor | None = None
        # Persistent decode qsl (arange) keeps metadata CUDA-graph-capturable; grown to largest bs.
        self._decode_qsl: torch.Tensor | None = None
        # Persistent CUDA-graph conv metadata buffers; sized in init_cuda_graph_state.
        self._graph_cache_indices: torch.Tensor | None = None
        self._graph_has_initial_state: torch.Tensor | None = None
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
            col_page_table = self._pfg_refresh_col_tables(found, bs)
        else:
            # clone(): the scheduler can recycle these live tables while extend kernels are in flight.
            col_page_table = {g: t.clone() for g, t in found.items()}
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
            is_decode = False
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
            is_decode = False
        else:
            query_start_loc = self._decode_query_start_loc(bs, req_pool_indices.device)
            # Decode: token t belongs to request t, so seq_idx is the same
            # cached arange, one element shorter.
            seq_idx = query_start_loc[:bs]
            has_initial_state = torch.ones(
                bs, dtype=torch.bool, device=req_pool_indices.device
            )
            is_decode = True
        needs_restore = False
        if forward_mode.is_extend_or_mixed() and extend_prefix_lens_cpu is not None:
            # Restore only ever has a source page on an aligned prefix (the
            # publish gate); the kernel re-checks per request on device.
            page_size = int(self.conv_columns["block_tokens"])
            prefix = extend_prefix_lens_cpu[:bs]
            needs_restore = bool(((prefix > 0) & (prefix % page_size == 0)).any())
        self.conv_metadata = InklingConvMetadata(
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            is_decode=is_decode,
            seq_idx=seq_idx,
            seq_lens=(
                self._pfg_seq_lens if pfg_total >= 0 else seq_lens[:bs].to(torch.int32)
            ),
            col_page_table=col_page_table,
            needs_ring_update=forward_mode.is_extend_or_mixed(),
            needs_restore=needs_restore,
        )

    # ------------------------------------------------------------------
    # Speculative-decoding conv metadata
    # ------------------------------------------------------------------

    def fixed_workspace_bytes(self) -> int:
        """Return persistent ShortConv state owned outside the LCM arenas."""
        return self.conv_pool.conv_state.nbytes

    def _spec_conv_metadata(self, bs: int) -> InklingConvMetadata:
        """Multi-token decode conv metadata over the persistent CUDA-graph
        buffers (target verify / draft catch-up)."""
        k = self.conv_spec_num_tokens
        return InklingConvMetadata(
            query_start_loc=self._graph_spec_qsl[: bs + 1],
            cache_indices=self._graph_cache_indices[:bs],
            has_initial_state=self._graph_has_initial_state[:bs],
            is_decode=False,
            seq_idx=self._graph_spec_seq_idx[: bs * k],
            seq_lens=self._graph_seq_lens[:bs],
            col_page_table={
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
            is_decode=True,
            seq_idx=self._decode_qsl[:bs],
            seq_lens=self._graph_seq_lens[:bs],
            col_page_table={g: t[:bs] for g, t in self._graph_col_tables.items()},
        )

    def configure_draft_lookback(self, lookback: int) -> bool:
        """Drafter hook (draft wrapper only): arm decode-window lookback.

        Lookback rows are ring reads ``lookback`` positions behind the
        committed frontier, so arming only widens the catch-up chunk; no
        extra state is allocated. Paged conv columns coexist with it: the
        ring stays the working store, the columns are only the
        publish/restore bridge. Returns True when armed; False when this
        backend cannot support it (target wrapper).
        """
        if not self.conv_is_draft or lookback <= 0:
            return False
        self._draft_lookback = int(lookback)
        # Arm the inner backend's grouped lookback-location stack (sized at graph
        # init, which runs after this): the lookback pass writes N + D rows
        # per request, so its cache write locations need their own variant.
        self.inner.draft_lookback = int(lookback)
        return True

    def enter_draft_lookback_window(self, bs: int) -> bool:
        """Drafter hook before the lookback window loop: rebuild the
        catch-up conv metadata for ``k + D`` rows per request. The next
        round's ``init_forward_metadata`` restores the plain shape."""
        lookback = self._draft_lookback
        md = self.conv_metadata
        if lookback <= 0 or md is None or md.is_decode or not self.conv_is_draft:
            return False
        # Cache write locations must widen to the lookback rows too; refusal falls
        # back to the plain window pass before any metadata is mutated.
        inner_enter = getattr(self.inner, "enter_draft_lookback", None)
        if inner_enter is not None and not inner_enter(bs):
            return False
        tokens = self.conv_spec_num_tokens + lookback
        device = md.cache_indices.device
        qsl = torch.arange(
            0, bs * tokens + 1, step=tokens, dtype=torch.int32, device=device
        )
        self.conv_metadata = InklingConvMetadata(
            query_start_loc=qsl,
            cache_indices=md.cache_indices[:bs],
            has_initial_state=md.has_initial_state[:bs],
            is_decode=False,
            seq_idx=seq_idx_from_cu_seqlens(qsl, bs * tokens),
            # Same through-chunk lengths: the wider chunk keeps its end, the
            # lookback rows extend it backwards.
            seq_lens=md.seq_lens[:bs] if md.seq_lens is not None else None,
            # Paged draft conv rides through: the in-kernel publish resolves
            # pages from the table by position, so a boundary rewritten by
            # the lookback rows is re-published with committed content.
            col_page_table=md.col_page_table,
        )
        return True

    def restore_shortconv_checkpoint(
        self,
        state: torch.Tensor,
        checkpoint_buffers: tuple[torch.Tensor, ...],
        metadata: InklingConvMetadata,
        group_id: str,
    ) -> None:
        """Restore an aligned cached boundary into request ring slots.

        The checkpoint holds the ``W - 1`` inputs before the boundary; they
        land at their positions' ring rows (``(boundary - W + 1 + j) % R``),
        with the boundary derived per request as ``seq_lens - chunk_len``.
        One kernel launch; requests without an aligned boundary, with a hole
        page or a PAD slot are skipped in-kernel.
        """
        n = metadata.cache_indices.shape[0]
        inkling_ring_restore(
            state,
            checkpoint_buffers[0],
            checkpoint_buffers[1] if len(checkpoint_buffers) == 2 else None,
            metadata.col_page_table[group_id][:n],
            metadata.query_start_loc[: n + 1],
            metadata.seq_lens[:n],
            metadata.cache_indices,
            page_size=int(self.conv_columns["block_tokens"]),
        )

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

    def advance_draft_forward_metadata(self, seq_lens: torch.Tensor | None = None):
        """Drafter hook before each multi-step decode step: the catch-up
        (k tokens/request) metadata becomes single-token decode metadata."""
        inner_advance = getattr(self.inner, "advance_draft_forward_metadata", None)
        if inner_advance is not None:
            inner_advance(seq_lens)
        md = self.conv_metadata
        if md is None or md.is_decode:
            return
        bs = md.cache_indices.shape[0]
        if self._graph_has_initial_state is not None:
            has_initial = self._graph_has_initial_state[:bs]
        else:
            has_initial = torch.ones(
                bs, dtype=torch.bool, device=md.cache_indices.device
            )
        query_start_loc = self._decode_query_start_loc(bs, md.cache_indices.device)
        self.conv_metadata = InklingConvMetadata(
            query_start_loc=query_start_loc,
            cache_indices=md.cache_indices,
            has_initial_state=has_initial,
            is_decode=True,
            seq_idx=query_start_loc[:bs],
            seq_lens=(
                seq_lens[:bs].to(torch.int32) if seq_lens is not None else md.seq_lens
            ),
            # Paged conv rides through every per-step rebuild: dropping the
            # table silently disables the T=1 landing publish for the step.
            col_page_table=md.col_page_table,
        )

    # ------------------------------------------------------------------
    # Attention delegation
    # ------------------------------------------------------------------

    # forward is NOT overridden: base dispatch sends rel_logits layers to the rel_mha overrides.

    def _rel_decode_cu_seqlens_q(
        self, bs: int, max_seqlen_q: int, device
    ) -> torch.Tensor:
        """Cached ``arange(bs + 1) * max_seqlen_q`` for rel decode.

        Cached PER ``max_seqlen_q``: under MTP the draft backend alternates
        between the catch-up chunk (``spec_num_tokens``) and single-token
        steps, and a single keyed-on-last-step buffer would be reallocated on
        every switch — invalidating the pointer captured CUDA graphs hold.
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
        self.inner.register_step_counter(step_counter)

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
            # k-token spec chunk; drafter capture swaps to 1-token steps via advance_draft_forward_metadata.
            self.conv_metadata = self._spec_conv_metadata(bs)
            return
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
        self.conv_metadata = self._graph_decode_conv_metadata(bs)
