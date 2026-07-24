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
sconv rolling state — four short-causal-conv streams per decoder block,
window ``W-1`` states per request — is managed entirely engine-side:

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
from tokenspeed_kernel.ops.conv import sconv_cache_update, seq_idx_from_cu_seqlens

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import (
    AttentionBackend,
    init_backend_cuda_graph_state,
)
from tokenspeed.runtime.layers.attention.backends.mha import (
    _scrub_extend_padding,
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
        seq_idx: ``[total_tokens]`` int32 sequence id per token (extend
            batches only; None on decode).
        update_mode: How the conv window is persisted after the forward.
            ``inplace``: kernel-native update (normal decode/extend).
            ``stash``: target verify — some chunk tokens may be REJECTED, so
            stash the pre-conv chunk activations and defer the window write
            to ``update_mamba_state_after_mtp_verify`` (post-verify hook).
            ``valid_len``: draft catch-up — the chunk's valid prefix length
            is already known (``ctx.accept_lengths``); write the window
            ending at the accepted position inline.
        tokens_per_req: Uniform tokens per request for the multi-token
            decode modes (``stash``/``valid_len``).
        lookback: Draft decode-window lookback rows (``valid_len`` only).
            When > 0 the catch-up chunk carries ``lookback`` extra leading
            rows that re-run the last committed positions of the previous
            round, so the conv compute reads the LAGGED window (see
            ``draft_lag_conv_state_wd``) instead of the main one.
    """

    query_start_loc: torch.Tensor
    cache_indices: torch.Tensor
    has_initial_state: torch.Tensor
    is_decode: bool
    seq_idx: torch.Tensor | None = None
    update_mode: str = "inplace"
    tokens_per_req: int = 1
    lookback: int = 0
    # Paged sconv: per-group tables {group: [bs, max_conv_blocks]} + lengths; None -> rolling state.
    col_page_table: dict[str, torch.Tensor] | None = None
    col_seq_lens: torch.Tensor | None = None
    col_prefix_lens: torch.Tensor | None = None
    # Lazy cache of ``_write_lag_extend``'s md-only index math (one forward's
    # streams share it; md is rebuilt every forward).
    lag_extend_cache: tuple | None = None


class InklingConvStatePool:
    """Engine-side rolling conv state for all sconv streams of all layers.

    Memory layout: ``[num_layers, num_slots, W-1, conv_dim]`` — the feature
    dim is contiguous, which the runtime ``causal_conv1d`` kernels require
    (``conv_state.stride(-2) == 1`` after the transposed view) and which
    matches the ``tokenspeed_kernel.ops.conv`` sconv kernels' native
    ``[slots, W-1, D]`` layout for the P2 swap-in. The four streams of a
    block live at fixed channel offsets given by ``inkling_conv_stream_layout``;
    modules take channel slices.
    """

    def __init__(
        self,
        num_layers: int,
        num_slots: int,
        conv_dim: int,
        kernel_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.conv_dim = conv_dim
        self.kernel_size = kernel_size
        self.conv_state = torch.zeros(
            (num_layers, num_slots, kernel_size - 1, conv_dim),
            dtype=dtype,
            device=device,
        )

    def layer_state(self, layer_id: int) -> torch.Tensor:
        """One layer's state, viewed ``[num_slots, conv_dim, W-1]`` with the
        feature dim contiguous (the causal_conv1d kernels' contract)."""
        return self.conv_state[layer_id].transpose(1, 2)

    def layer_state_wd(self, layer_id: int) -> torch.Tensor:
        """One layer's state in the native ``[num_slots, W-1, conv_dim]``
        layout (the tokenspeed_kernel ops/conv sconv kernels' contract)."""
        return self.conv_state[layer_id]

    def mem_usage_bytes(self) -> int:
        return self.conv_state.numel() * self.conv_state.element_size()


class InklingAttnBackend(AttentionBackend):
    """Thin wrapper over the dense MHA backend adding conv metadata.

    All attention forwards and CUDA-graph hooks delegate to the wrapped
    backend. This class derives ``InklingConvMetadata`` from the same arguments
    the dense path already receives and declares that Flat execution must keep
    its heterogeneous cache locations group-keyed; the scheduler and executor
    remain unaware of the model-specific reason for that capability.
    """

    # Ask the graph wrapper for actual_bs at replay so padded rows can be marked PAD_SLOT_ID.
    uses_padded_decode_token_mask = True
    # Inkling's heterogeneous groups reinterpret pool-local page ids with
    # different token spans. A scalar req_to_page/out_cache_loc drops the
    # required group identity, so Flat execution must use the existing
    # per-group table and write-location metadata for both target and draft.
    requires_group_keyed_cache_locs = True

    def __init__(
        self,
        inner: AttentionBackend,
        conv_pool: InklingConvStatePool,
        *,
        spec_num_tokens: int = 1,
        is_draft: bool = False,
    ):
        # Deliberately skip AttentionBackend.__init__: the wrapper mirrors inner via __getattr__.
        self.inner = inner
        self.conv_pool = conv_pool
        self.conv_metadata: InklingConvMetadata | None = None
        # Spec decoding: >1 means decode rounds carry this many tokens/request (verify / catch-up).
        self.conv_spec_num_tokens = max(1, int(spec_num_tokens))
        self.conv_is_draft = is_draft
        # Target-verify stash [num_layers, tokens, conv_dim]; pinned once CUDA graphs record it.
        self._verify_stash: torch.Tensor | None = None
        self._stash_pinned = False
        # Draft decode-window lookback: D > 0 arms
        # a second per-layer conv window that lags the committed frontier by
        # D rows, so lookback rows convolve against the state they actually
        # follow. Configured by the drafter via configure_draft_lookback.
        self._draft_lookback = 0
        self._draft_lag_conv_state: torch.Tensor | None = None
        self._warned_mixed_spec = False
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
        self._pfg_col_tables: dict[str, torch.Tensor] | None = None
        self._pfg_max_bs = 0

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
    def uses_flat_cache_groups(self):
        return self.inner.uses_flat_cache_groups

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
        req_to_page: torch.Tensor,
        forward_mode: ForwardMode,
        extend_seq_lens: torch.Tensor | None = None,
        extend_seq_lens_cpu: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        extend_prefix_lens_cpu: torch.Tensor | None = None,
        **kwargs,
    ):
        # Paged sconv: conv groups ride flat_block_tables, which the inner backend sheds — grab here.
        flat_tables = kwargs.get("flat_block_tables") or {}
        col_page_table = None
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
        ):
            pfg_total = extend_total
        if getattr(self, "conv_columns", None) is not None:
            groups = set(self.conv_columns["group_block_tokens"])
            found = {g: flat_tables.get(g) for g in groups}
            missing = sorted(g for g, t in found.items() if t is None)
            if pfg_total >= 0:
                if missing:
                    raise RuntimeError(
                        f"paged sconv: prefill-graph statics are armed but "
                        f"flat_block_tables is missing conv groups {missing}"
                    )
                # The stream-ordered copy into the statics doubles as the plain path's clone() snapshot.
                col_page_table = self._pfg_refresh_col_tables(found, bs)
            elif not missing:
                # clone(): the scheduler can recycle these live tables while extend kernels are in flight.
                col_page_table = {g: t.clone() for g, t in found.items()}
            elif len(missing) < len(found):
                raise RuntimeError(
                    f"paged sconv: flat_block_tables delivered only part of "
                    f"the conv groups (missing {missing}); refusing to mix "
                    "paged and rolling conv state in one step"
                )
        self.inner.init_forward_metadata(
            bs,
            num_extends,
            req_pool_indices,
            seq_lens,
            req_to_page,
            forward_mode,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            **kwargs,
        )

        cache_indices = req_pool_indices[:bs].to(torch.int32)
        seq_idx = None
        update_mode = "inplace"
        tokens_per_req = 1
        col_prefix_lens = (
            extend_prefix_lens[:bs]
            if col_page_table is not None and extend_prefix_lens is not None
            else None
        )
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
            if pfg_total >= 0 and col_page_table is not None:
                # PFG statics: tail qsl closes the PAD request's empty chunk; tail seq_idx marks pads PAD.
                self._pfg_qsl[: bs + 1].copy_(query_start_loc)
                self._pfg_qsl[self._pfg_max_bs :].fill_(pfg_total)
                self._pfg_seq_idx[:pfg_total].copy_(seq_idx)
                self._pfg_seq_idx[pfg_total:].fill_(self._pfg_max_bs)
                self._pfg_prefix_lens[:bs].copy_(extend_prefix_lens[:bs])
                query_start_loc = self._pfg_qsl
                seq_idx = self._pfg_seq_idx
                col_prefix_lens = self._pfg_prefix_lens
            if (
                forward_mode.is_mixed()
                and self.conv_spec_num_tokens > 1
                and not self._warned_mixed_spec
            ):
                # MIXED spec round decode rows keep the in-place update: rejected tails land in windows.
                self._warned_mixed_spec = True
                logger.warning(
                    "Inkling sconv: MIXED batch during speculative decoding — "
                    "decode-row conv windows are not rolled back this round."
                )
        elif forward_mode.is_decode() and self.conv_spec_num_tokens > 1:
            # Multi-token decode: verify -> stash (accept unknown); draft catch-up -> valid_len write.
            k = self.conv_spec_num_tokens
            tokens_per_req = k
            device = req_pool_indices.device
            query_start_loc = torch.arange(
                0, bs * k + 1, step=k, dtype=torch.int32, device=device
            )
            seq_idx = seq_idx_from_cu_seqlens(query_start_loc, bs * k)
            has_initial_state = torch.ones(bs, dtype=torch.bool, device=device)
            is_decode = False
            update_mode = "valid_len" if self.conv_is_draft else "stash"
            if update_mode == "stash":
                self._ensure_verify_stash(bs * k, device)
        else:
            query_start_loc = self._decode_query_start_loc(bs, req_pool_indices.device)
            has_initial_state = torch.ones(
                bs, dtype=torch.bool, device=req_pool_indices.device
            )
            is_decode = True
        self.conv_metadata = InklingConvMetadata(
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            is_decode=is_decode,
            seq_idx=seq_idx,
            update_mode=update_mode,
            tokens_per_req=tokens_per_req,
            col_page_table=col_page_table,
            col_seq_lens=seq_lens[:bs] if col_page_table is not None else None,
            col_prefix_lens=col_prefix_lens,
        )

    # ------------------------------------------------------------------
    # Speculative-decoding conv state (eager path; CUDA graphs off for MTP)
    # ------------------------------------------------------------------

    def _ensure_verify_stash(self, num_tokens: int, device) -> None:
        pool = self.conv_pool
        if self._verify_stash is None or self._verify_stash.shape[1] < num_tokens:
            if self._stash_pinned:
                # Growing would leave captured graphs writing freed memory; pinned size = pool capacity.
                raise RuntimeError(
                    f"Inkling verify stash needs {num_tokens} rows but is "
                    f"pinned at {self._verify_stash.shape[1]} by CUDA graphs."
                )
            self._verify_stash = torch.empty(
                (pool.num_layers, num_tokens, pool.conv_dim),
                dtype=pool.conv_state.dtype,
                device=device,
            )

    def _spec_conv_metadata(self, bs: int) -> InklingConvMetadata:
        """Multi-token decode conv metadata over the persistent CUDA-graph
        buffers (target verify: stash; draft catch-up: valid_len)."""
        k = self.conv_spec_num_tokens
        return InklingConvMetadata(
            query_start_loc=self._graph_spec_qsl[: bs + 1],
            cache_indices=self._graph_cache_indices[:bs],
            has_initial_state=self._graph_has_initial_state[:bs],
            is_decode=False,
            seq_idx=self._graph_spec_seq_idx[: bs * k],
            update_mode="valid_len" if self.conv_is_draft else "stash",
            tokens_per_req=k,
        )

    def _graph_decode_conv_metadata(self, bs: int) -> InklingConvMetadata:
        """Single-token decode conv metadata over the persistent CUDA-graph
        buffers (shared by graph capture and replay)."""
        paged = getattr(self, "conv_columns", None) is not None
        return InklingConvMetadata(
            query_start_loc=self._decode_qsl[: bs + 1],
            cache_indices=self._graph_cache_indices[:bs],
            has_initial_state=self._graph_has_initial_state[:bs],
            is_decode=True,
            col_page_table=(
                {g: t[:bs] for g, t in self._graph_col_tables.items()}
                if paged
                else None
            ),
            col_seq_lens=self._graph_seq_lens[:bs] if paged else None,
        )

    def configure_draft_lookback(self, lookback: int) -> bool:
        """Drafter hook (draft wrapper only): arm decode-window lookback.

        Allocates the per-layer lagged conv window and makes every extend
        chunk advance it (see ``_write_lag_extend``), so the first decode
        round's lookback has a well-defined conv init state. Returns True
        when armed; False when this backend cannot support it (target
        wrapper, or paged draft conv).
        """
        if not self.conv_is_draft or lookback <= 0:
            return False
        if getattr(self, "conv_columns", None) is not None:
            # Rolling state only: the lag recurrence has no paged variant.
            return False
        self._draft_lookback = int(lookback)
        # Arm the inner backend's flat lookback loc stack (sized at graph
        # init, which runs after this): the lookback pass writes N + D rows
        # per request, so its flat write locs need their own variant.
        self.inner.flat_draft_lookback = int(lookback)
        if self._draft_lag_conv_state is None:
            self._draft_lag_conv_state = torch.zeros_like(self.conv_pool.conv_state)
        return True

    def enter_draft_lookback_window(self, bs: int) -> bool:
        """Drafter hook before the lookback window loop: rebuild the
        catch-up conv metadata for ``k + D`` rows per request. The next
        round's ``init_forward_metadata`` restores the plain shape."""
        lookback = self._draft_lookback
        md = self.conv_metadata
        if (
            lookback <= 0
            or md is None
            or md.update_mode != "valid_len"
            or md.col_page_table is not None
        ):
            return False
        # Flat write locs must widen to the lookback rows too; refusal falls
        # back to the plain window pass before any metadata is mutated.
        inner_enter = getattr(self.inner, "flat_enter_draft_lookback", None)
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
            update_mode="valid_len",
            tokens_per_req=tokens,
            lookback=lookback,
        )
        return True

    def draft_lag_conv_state_wd(self, layer_id: int) -> torch.Tensor:
        """One layer's lagged conv window, ``[num_slots, W-1, conv_dim]``.

        Ends ``lookback + 1`` positions behind the committed frontier — the
        state a decode-window lookback row actually follows.
        """
        return self._draft_lag_conv_state[layer_id]

    def apply_conv_state_update(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        md: InklingConvMetadata,
        layer_id: int,
        channel_offset: int,
        dim: int,
        accept_lengths: torch.Tensor | None = None,
    ) -> None:
        """Persist one stream's conv window after a non-decode forward.

        ``x`` is the stream's pre-conv chunk activations ``[tokens, dim]``;
        ``state`` is the channel slice ``[slots, W-1, dim]`` of the layer's
        pool buffer that the compute kernel just read (the LAGGED buffer on
        lookback window passes).
        """
        if md.update_mode == "inplace":
            if self._draft_lookback > 0 and md.col_page_table is None:
                # Advance the lag window BEFORE the kernel overwrites the
                # main one: short chunks borrow its pre-update rows.
                self._write_lag_extend(state, x, md, layer_id, channel_offset, dim)
            sconv_cache_update(
                x,
                state,
                md.query_start_loc,
                md.cache_indices,
                md.has_initial_state,
            )
            return
        if md.update_mode == "stash":
            # Rejects unknown until verify: stash; the post-verify hook writes the final window.
            self._verify_stash[
                layer_id, : x.shape[0], channel_offset : channel_offset + dim
            ].copy_(x)
            return
        assert md.update_mode == "valid_len"
        if accept_lengths is None:
            raise RuntimeError(
                "Inkling draft catch-up conv update needs ctx.accept_lengths"
            )
        if md.lookback > 0:
            # Lookback window pass: ``state`` is the LAG slice, whose old
            # window ends exactly one row before the chunk's first row, so
            # both targets gather off ONE shared pre-update ``[lag_old ||
            # chunk]`` extension:
            #   main window (ends at accept)      -> valid = accept + D
            #   lag window (ends at accept - D)   -> valid = accept
            main = self.conv_pool.layer_state_wd(layer_id)[
                :, :, channel_offset : channel_offset + dim
            ]
            bs = md.cache_indices.shape[0]
            accept = accept_lengths[:bs].to(torch.int64)
            # PAD_SLOT_ID rows route to slot 0: the 1-based request pool
            # reserves it (never live).
            idx = md.cache_indices.long().clamp_min(0)
            ext = torch.cat([state[idx], x.view(bs, md.tokens_per_req, dim)], dim=1)
            w1 = state.shape[1]
            main[idx] = self._gather_ext_window(
                ext, accept + md.lookback, md.tokens_per_req, w1
            )
            state[idx] = self._gather_ext_window(ext, accept, md.tokens_per_req, w1)
            return
        self._write_window_at(
            state, x, md.cache_indices, md.tokens_per_req, accept_lengths
        )

    @staticmethod
    def _gather_ext_window(
        ext: torch.Tensor,
        valid_lens: torch.Tensor,
        tokens_per_req: int,
        w1: int,
    ) -> torch.Tensor:
        """Rows ``valid .. valid+W-2`` of the ``[old window || chunk]``
        extension ``ext`` ``[bs, W-1+tokens_per_req, dim]`` — the window
        after accepting ``valid`` chunk tokens per request (clamped to
        ``[1, tokens_per_req]``)."""
        bs, _, dim = ext.shape
        a = valid_lens.long().clamp(1, tokens_per_req)
        rows = a.view(bs, 1) + torch.arange(w1, device=ext.device).view(1, w1)
        return ext.gather(1, rows.unsqueeze(-1).expand(bs, w1, dim))

    @staticmethod
    def _write_window_from(
        dst: torch.Tensor,
        src: torch.Tensor,
        chunk: torch.Tensor,
        cache_indices: torch.Tensor,
        tokens_per_req: int,
        valid_lens: torch.Tensor,
    ) -> None:
        """dst window <- last W-1 of ``[src window || chunk[:valid]]``.

        ``dst``/``src``: ``[slots, W-1, dim]`` channel slices (``src``'s old
        window must end one row before the chunk's first row); ``chunk``:
        ``[bs*tokens_per_req, dim]``; ``valid_lens``: per-request count of
        valid chunk positions (clamped to ``[1, tokens_per_req]``). The blend
        handles ``valid < W-1`` by borrowing rows from the src window.
        """
        bs = cache_indices.shape[0]
        # PAD_SLOT_ID rows route to slot 0: the 1-based request pool reserves it (never live).
        idx = cache_indices.long().clamp_min(0)
        ext = torch.cat([src[idx], chunk.view(bs, tokens_per_req, src.shape[2])], dim=1)
        dst[idx] = InklingAttnBackend._gather_ext_window(
            ext, valid_lens[:bs], tokens_per_req, src.shape[1]
        )

    @classmethod
    def _write_window_at(
        cls,
        state: torch.Tensor,
        chunk: torch.Tensor,
        cache_indices: torch.Tensor,
        tokens_per_req: int,
        valid_lens: torch.Tensor,
    ) -> None:
        """working window <- last W-1 of ``[old window || chunk[:valid]]``."""
        cls._write_window_from(
            state, state, chunk, cache_indices, tokens_per_req, valid_lens
        )

    def _write_lag_extend(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        md: InklingConvMetadata,
        layer_id: int,
        channel_offset: int,
        dim: int,
    ) -> None:
        """Advance the lag window across an EXTEND/MIXED chunk.

        The lag target ends ``D`` rows behind the chunk end; chunks shorter
        than ``D + W-1`` borrow trailing rows from the pre-update MAIN window
        (``state``), which is contiguous with the chunk's first row. Fresh
        prefills (``has_initial_state`` False) clamp into the chunk instead —
        exact whenever the first chunk is at least ``D + W-1`` tokens.

        MIXED decode rows share the extend chunk's inplace path, so their lag
        (like their main window) is not rolled back to the accepted length
        this round — same known wart, same bounded blast radius (the first
        W-1 rows of the next lookback window's conv).
        """
        lag = self._draft_lag_conv_state[layer_id][
            :, :, channel_offset : channel_offset + dim
        ]
        bs = md.cache_indices.shape[0]
        w1 = state.shape[1]
        idx, rows_old, chunk_rows, use_old = self._lag_extend_index_math(
            md, w1, x.shape[0], x.device
        )
        old = state[idx]  # pre-update main window [bs, W-1, dim]
        gathered_old = old.gather(1, rows_old.unsqueeze(-1).expand(bs, w1, dim))
        gathered_chunk = x[chunk_rows].view(bs, w1, dim)
        lag[idx] = torch.where(use_old, gathered_old, gathered_chunk)

    def _lag_extend_index_math(
        self,
        md: InklingConvMetadata,
        w1: int,
        total: int,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """md-only index math of ``_write_lag_extend``, computed once per
        forward and cached on the metadata object (rebuilt every forward),
        so all ~4 streams x layers of the step reuse it.

        Returns:
            ``(idx, rows_old, chunk_rows, use_old)``: the [bs] pool slots,
            the clamped [bs, w1] old-window gather rows, the flat [bs*w1]
            chunk gather rows, and the [bs, w1, 1] old-vs-chunk blend mask.
        """
        cached = getattr(md, "lag_extend_cache", None)
        if cached is not None:
            return cached
        lookback = self._draft_lookback
        bs = md.cache_indices.shape[0]
        qsl = md.query_start_loc[: bs + 1].long()
        lens = qsl[1:] - qsl[:-1]
        idx = md.cache_indices.long().clamp_min(0)
        valid = (lens - lookback).clamp_min(0)
        # Rows over ext = [main_old(W-1) || chunk_r]: valid .. valid+W-2.
        rows = valid.view(bs, 1) + torch.arange(w1, device=device).view(1, w1)
        from_old = rows < w1
        chunk_rows = (
            (qsl[:-1].view(bs, 1) + (rows - w1).clamp_min(0))
            .clamp(max=max(total - 1, 0))
            .reshape(-1)
        )
        use_old = (from_old & md.has_initial_state[:bs].view(bs, 1)).unsqueeze(-1)
        md.lag_extend_cache = (idx, rows.clamp(max=w1 - 1), chunk_rows, use_old)
        return md.lag_extend_cache

    def update_mamba_state_after_mtp_verify(self, accept_lengths, model) -> None:
        """Post-verify hook (duck-typed from CudaGraphWrapper): select each
        request's conv window at its accepted length from the stashed
        activations. Name matches the generic executor hook.

        Runs outside the captured graph once per MTP round, so it is batched
        over all layers in one shot (the per-layer ``_write_window_at`` loop
        is host-launch-bound). The blend is cat-free — two gathers selected
        by ``where`` — to avoid materializing a ``[L, n, W-1+k, D]`` extension
        buffer at full batch.
        """
        del model
        md = self.conv_metadata
        if md is None or md.update_mode != "stash":
            return
        pool = self.conv_pool
        k = md.tokens_per_req
        # Graph replay pads the batch; accept_lengths covers only the real, leading stash rows.
        n = min(md.cache_indices.shape[0], accept_lengths.shape[0])
        if n == 0:
            return
        state = pool.conv_state  # [L, slots, W-1, D]
        num_layers, _, w1, dim = state.shape
        # Padded rows carry PAD_SLOT_ID (-1): route their writes to slot 0,
        # which the 1-based request pool reserves (never a live request).
        idx = md.cache_indices[:n].long().clamp_min(0)
        chunk = self._verify_stash[:, : n * k].unflatten(1, (n, k))
        old = state[:, idx]  # [L, n, W-1, D]
        a = accept_lengths[:n].long().clamp(1, k)
        # Window after accepting `a` chunk tokens = rows a .. a+W-2 of the
        # virtual [old || chunk] extension; row r reads old[r] when r < W-1,
        # else chunk[r - (W-1)] (always an accepted position: r-(W-1) < a).
        rows = a.view(n, 1) + self._cached_arange_w1(w1, state.device).view(1, w1)
        from_old = (rows < w1).view(1, n, w1, 1)
        rows_old = (
            rows.clamp(max=w1 - 1).view(1, n, w1, 1).expand(num_layers, n, w1, dim)
        )
        rows_new = (
            (rows - w1).clamp(min=0).view(1, n, w1, 1).expand(num_layers, n, w1, dim)
        )
        state[:, idx] = torch.where(
            from_old, old.gather(2, rows_old), chunk.gather(2, rows_new)
        )

    def _cached_arange_w1(self, w1: int, device) -> torch.Tensor:
        buf = getattr(self, "_arange_w1_buf", None)
        if buf is None or buf.shape[0] != w1:
            buf = self._arange_w1_buf = torch.arange(w1, device=device)
        return buf

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
        self.conv_metadata = InklingConvMetadata(
            query_start_loc=self._decode_query_start_loc(bs, md.cache_indices.device),
            cache_indices=md.cache_indices,
            has_initial_state=has_initial,
            is_decode=True,
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
            # (dropping most of a multi-token window's KV — the flat-arm
            # draft accept regression), a longer one would crash the store.
            assert k is None or out_cache_loc.shape[0] == k.shape[0], (
                f"Inkling decode KV write: {k.shape[0]} rows vs "
                f"{out_cache_loc.shape[0]} write locs (layer "
                f"{layer.layer_id}, group {layer.group_id!r}); a chaining "
                "one-row-per-step draft loop is unsupported on the flat arm."
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
        # Rel path skips base pad hygiene: zero q/k/v + OUTPUT pad rows, else uninit NaNs REAL rows.
        _scrub_extend_padding(metadata, q, k, v)
        _num_real = metadata.cu_extend_seq_lens_cpu[-1]
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
        geo = getattr(self, "conv_columns", None)
        if geo is None:
            raise RuntimeError(
                "Inkling prefill graph needs fully-paged sconv; rolling conv "
                "state is per-batch-shaped and cannot be graphed by token bucket"
            )
        if geo.get("hidden_group_of_layer") is None:
            raise RuntimeError(
                "Inkling prefill graph needs paged hidden conv; the ATTN/MLP "
                "sconv sites still run the rolling-state path"
            )
        device = self.conv_pool.conv_state.device
        self._pfg_max_bs = max_bs
        self._pfg_seq_idx = torch.full(
            (max_num_tokens,), max_bs, dtype=torch.int32, device=device
        )
        self._pfg_qsl = torch.zeros(max_bs + 2, dtype=torch.int32, device=device)
        self._pfg_prefix_lens = torch.zeros(
            max_bs + 1, dtype=torch.int32, device=device
        )
        self._pfg_col_tables = {
            g: torch.full(
                (max_bs + 1, -(-self.inner.max_context_len // bt)),
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

    def init_cuda_graph_state(self, max_bs: int, seq_lens_buf: torch.Tensor, **kwargs):
        init_backend_cuda_graph_state(self.inner, max_bs, seq_lens_buf, **kwargs)
        device = self.conv_pool.conv_state.device
        self._decode_qsl = torch.arange(max_bs + 1, dtype=torch.int32, device=device)
        self._graph_seq_lens = seq_lens_buf
        if getattr(self, "conv_columns", None) is not None:
            # Adopted stacked views are filled by the mixin's packed unpack; pad rows hit dummy slot 0.
            inner_tabs = getattr(self.inner, "cuda_graph_flat_page_tables", {})
            groups = self.conv_columns["group_block_tokens"]
            self._graph_col_tables_adopted = all(g in inner_tabs for g in groups)
            if self._graph_col_tables_adopted:
                self._graph_col_tables = {g: inner_tabs[g] for g in groups}
            else:
                self._graph_col_tables = {
                    g: torch.full(
                        (max_bs, -(-self.inner.max_context_len // bt)),
                        -1,
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
            if not self.conv_is_draft:
                # Pin stash at pool capacity: eager bs > graph max_bs must not realloc a graph-bound buffer.
                self._ensure_verify_stash((self.conv_pool.num_slots - 2) * k, device)
                self._stash_pinned = True

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
        req_to_page: torch.Tensor = None,
        **kwargs,
    ):
        actual_bs = kwargs.pop("actual_bs", None)
        self.inner.init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            forward_mode=forward_mode,
            req_to_page=req_to_page,
            **kwargs,
        )
        assert self._graph_cache_indices is not None
        self._graph_cache_indices[:bs].copy_(req_pool_indices[:bs].to(torch.int32))
        if actual_bs is not None and actual_bs < bs:
            # Pad rows may carry stale indices aliasing LIVE slots; PAD_SLOT_ID keeps writes off them.
            self._graph_cache_indices[actual_bs:bs].fill_(PAD_SLOT_ID)
        if getattr(self, "conv_columns", None) is not None:
            flat_tables = kwargs.get("flat_block_tables") or {}
            adopted_filled = getattr(
                self, "_graph_col_tables_adopted", False
            ) and getattr(self.inner, "_flat_packed_unpack_ran", False)
            for g, buf in self._graph_col_tables.items():
                src = flat_tables.get(g)
                if src is None:
                    raise RuntimeError(
                        f"paged sconv replay: no {g!r} table in " "flat_block_tables"
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
