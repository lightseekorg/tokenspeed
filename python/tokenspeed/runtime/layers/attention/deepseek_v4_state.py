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

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import TypeVar

import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_deepseek_v4 import (
    DeepseekV4CacheMetadata,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
)


@dataclass
class DeepseekV4IndexerPrefillChunkPlan:
    token_start: int
    token_end: int
    request_start: int
    request_end: int
    slot_start: int
    slot_end: int
    gather_row_start: int
    gather_row_end: int
    max_seq_len_k: int
    cu_seq_lens_start: int
    cu_seq_lens_end: int
    skip_kv_gather: bool = False


@dataclass
class DeepseekV4IndexerPrefillMetadata:
    chunks: tuple[DeepseekV4IndexerPrefillChunkPlan, ...]
    chunk_specs: torch.Tensor
    chunk_offsets: torch.Tensor
    slots: torch.Tensor
    cu_seq_lens: torch.Tensor
    cu_seqlen_k_start: torch.Tensor
    cu_seqlen_k_end: torch.Tensor
    seq_lens_k: torch.Tensor

    def max_gather_rows(self) -> int:
        if not self.chunks:
            return 0
        return max(max(0, chunk.slot_end - chunk.slot_start) for chunk in self.chunks)

    @classmethod
    def empty(cls, device: torch.device) -> "DeepseekV4IndexerPrefillMetadata":
        return cls(
            chunks=(),
            chunk_specs=torch.empty((0, 5), dtype=torch.int64, device="cpu"),
            chunk_offsets=torch.empty((0, 7), dtype=torch.int64, device="cpu"),
            slots=torch.empty(0, dtype=torch.int64, device=device),
            cu_seq_lens=torch.empty(0, dtype=torch.int32, device=device),
            cu_seqlen_k_start=torch.empty(0, dtype=torch.int32, device=device),
            cu_seqlen_k_end=torch.empty(0, dtype=torch.int32, device=device),
            seq_lens_k=torch.empty(0, dtype=torch.int32, device=device),
        )


@dataclass
class DeepseekV4IndexerDecodePlan:
    context_lens: torch.Tensor
    page_table: torch.Tensor
    max_context_len: int


@dataclass
class DeepseekV4IndexerBatchMetadata:
    positions: torch.Tensor
    token_to_req_indices: torch.Tensor
    seq_lens_cpu: torch.Tensor
    query_lens_cpu: torch.Tensor
    num_prefill_tokens: int
    num_decode_tokens: int


@dataclass
class DeepseekV4AttentionMetadata:
    swa_indices: torch.Tensor | None = None
    swa_lens: torch.Tensor | None = None
    # All-zero lengths shaped like ``swa_lens``: the SWA cache is replicated
    # across a KVP group, so every rank but the first attends to none of it.
    swa_lens_none: torch.Tensor | None = None
    swa_window_size: int = 0
    swa_block_size: int = 0
    # Cache for dense compressed decode attention indices/lens. CSA decode uses
    # dynamic top-k indices and does not populate this cache.
    # (indices, scan lens, owned-row counts) per dense compressed decode key.
    decode_dense_compressed_indices_cache: dict[
        tuple[int, int, int, int],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ] = field(default_factory=dict)
    decode_dense_compressed_indices_capture_safe_keys: set[
        tuple[int, int, int, int]
    ] = field(default_factory=set)


@dataclass
class DeepseekV4IndexerMetadata:
    decode_schedule_metadata_cache: dict[tuple[int, int, int], torch.Tensor] = field(
        default_factory=dict
    )
    decode_plan_cache: dict[tuple[int, int, int], DeepseekV4IndexerDecodePlan] = field(
        default_factory=dict
    )
    decode_plan_refreshed_keys: set[tuple[int, int, int]] = field(default_factory=set)
    decode_schedule_metadata_refreshed_keys: set[tuple[int, int, int]] = field(
        default_factory=set
    )
    prefill_plan_cache: dict[tuple[int, ...], DeepseekV4IndexerPrefillMetadata] = field(
        default_factory=dict
    )


@dataclass
class DeepseekV4SparseIndexerMetadata:
    batch_metadata: DeepseekV4IndexerBatchMetadata | None = None
    prefill_metadata: DeepseekV4IndexerPrefillMetadata | None = None
    decode_plan: DeepseekV4IndexerDecodePlan | None = None
    decode_schedule_metadata: torch.Tensor | None = None


@dataclass
class DeepseekV4KvpPrefillChunk:
    local_destinations: torch.Tensor
    counts: list[int]
    destinations: torch.Tensor
    workspace_width: int


@dataclass
class DeepseekV4ForwardMetadata:
    seq_lens: torch.Tensor
    query_lens: torch.Tensor
    query_start_loc: torch.Tensor
    token_to_req_indices: torch.Tensor
    cache: DeepseekV4CacheMetadata
    attention: DeepseekV4AttentionMetadata = field(
        default_factory=DeepseekV4AttentionMetadata
    )
    indexer: DeepseekV4IndexerMetadata = field(
        default_factory=DeepseekV4IndexerMetadata
    )
    forward_mode: ForwardMode | None = None
    # The CUDA-graph padding mask, one mission: True = live token, False =
    # a padded replay row (never mixed-batch state; prefill rows are always
    # live).
    is_valid_token: torch.Tensor | None = None
    # CPU lens are retained for sparse prefill/indexer planning without
    # forcing another device-to-host sync in the model path.
    seq_lens_cpu: torch.Tensor | None = None
    query_lens_cpu: torch.Tensor | None = None
    # Cached split boundary derived from scheduler num_extends/query_lens.
    num_prefill_reqs: int = 0
    num_prefill_tokens: int = 0
    # Prepared once for a live forward, shared by its request slices and layers.
    kvp_prefill: dict[int, dict[tuple[int, int], DeepseekV4KvpPrefillChunk]] = field(
        default_factory=dict
    )
    prefill_req_offset: int = 0

    def decode_req_count(self) -> int:
        return max(0, int(self.seq_lens.shape[0]) - int(self.num_prefill_reqs))

    def decode_token_count(self) -> int:
        return max(
            0,
            int(self.token_to_req_indices.shape[0]) - int(self.num_prefill_tokens),
        )


class DeepseekV4GraphBuffers:
    """Storage plus per-shape views for V4's unified decode path.

    DeepSeek V4 persistent CUDA-graph decode buffers, as one composed object.

    A captured decode graph holds the addresses of every tensor its kernels read
    forever, so the buffers backing V4's decode metadata must be allocated once
    and refreshed in place (docs/design/unified_path.md, "Pointer-stable per-bs
    views"). This object owns that storage and the per-``(bs, tokens_per_req)``
    metadata views over it — one keyed builder shared by capture, replay refresh
    and the above-ladder eager path, so the three arms cannot bind different
    objects (the class of bug this extraction exists to prevent).

    Deliberately storage-only: cache-group *contract* validation (which groups a
    capture placeholder may name, live-page geometry checks) stays on the
    backend; this object answers "where do the rows live", never "are they
    legal".

    Attributes:
        max_bs: Row capacity — the runner's max decode bs, never the capture
            ladder (docs/design/unified_path.md, "Buffer sizing").
        max_tokens_per_req: Widest packed verify width (spec window).
        page_table: ``[max_bs, max_num_pages]`` base full-history table.
        seq_lens / query_lens: ``[max_bs]`` row state.
        query_start_loc: ``[max_bs + 1]`` cumulative query offsets.
        token_to_req: ``[max_bs * max_tokens_per_req]`` token-major owner map.
        is_valid_token: same shape; the CUDA-graph padding mask
            (True = live token).
        query_start_by_width / token_to_req_by_width: per-width constant
            tables, precomputed so a packed refresh is copy-only.
        block_tables: per-group persistent tables (allocated by the
            backend's contract configuration through
            :meth:`allocate_group_tables`).
        kvp_size / kvp_rank / runtime_contract: the virtual-block placement
            every cache view translates through.
    """

    def __init__(
        self,
        *,
        max_bs: int,
        max_tokens_per_req: int,
        max_num_pages: int,
        device: torch.device | str,
        kvp_size: int,
        kvp_rank: int,
        runtime_contract: CacheRuntimeContract,
    ) -> None:
        self.max_bs = max_bs
        self.kvp_size = kvp_size
        self.kvp_rank = kvp_rank
        self.runtime_contract = runtime_contract
        self.max_tokens_per_req = max(1, int(max_tokens_per_req))
        max_tokens = max_bs * self.max_tokens_per_req
        self.device = device
        self.page_table = torch.zeros(
            (max_bs, max_num_pages), dtype=torch.int32, device=device
        )
        self.seq_lens = torch.ones((max_bs,), dtype=torch.int32, device=device)
        self.query_lens = torch.ones((max_bs,), dtype=torch.int32, device=device)
        self.query_start_loc = torch.arange(
            max_bs + 1, dtype=torch.int32, device=device
        )
        self.token_to_req = torch.arange(max_tokens, dtype=torch.int32, device=device)
        self.is_valid_token = torch.ones(max_tokens, dtype=torch.bool, device=device)
        query_start_base = torch.arange(max_bs + 1, dtype=torch.int32, device=device)
        token_to_req_base = torch.arange(max_bs, dtype=torch.int32, device=device)
        self.query_start_by_width: dict[int, torch.Tensor] = {}
        self.token_to_req_by_width: dict[int, torch.Tensor] = {}
        for width in range(1, self.max_tokens_per_req + 1):
            self.query_start_by_width[width] = query_start_base * width
            self.token_to_req_by_width[width] = token_to_req_base.repeat_interleave(
                width
            )
        self.block_tables: dict[str, torch.Tensor] = {}
        # Per-(bs, tokens_per_req) metadata views — the single builder's cache.
        self._views: dict[tuple[int, int], DeepseekV4ForwardMetadata] = {}

    def allocate_group_tables(self, group_widths: dict[str, int]) -> None:
        """Allocate the per-group persistent tables from the backend-resolved
        widths (the backend owns the contract math; this owns the storage)."""
        for gid, max_pages in group_widths.items():
            self.block_tables[gid] = torch.zeros(
                (self.max_bs, max_pages), dtype=torch.int32, device=self.device
            )
        # Views slice the group tables at construction; any built over the
        # previous set must rebuild.
        self._views.clear()

    # ------------------------------------------------------------------
    # The single per-shape view builder
    # ------------------------------------------------------------------

    def views(
        self,
        bs: int,
        tokens_per_req: int,
        *,
        kernel_page_size: int,
        max_num_pages: int,
        forward_mode,
    ) -> DeepseekV4ForwardMetadata:
        """The pointer-stable metadata views for ``(bs, tokens_per_req)``.

        Built once per shape and cached; every tensor field — the ``cache``
        slot's group tables included — is a fixed slice of the persistent
        buffers, assigned exactly at construction. Capture, replay refresh and
        lazy eager building all receive the same object and only fill the
        buffers underneath, so a refresh can never bind storage the captured
        graph did not record. Callers own the per-round ``forward_mode``
        update.
        """
        total_tokens = bs * tokens_per_req
        metadata = self._views.get((bs, tokens_per_req))
        if metadata is not None:
            metadata.forward_mode = forward_mode
            return metadata
        metadata = DeepseekV4ForwardMetadata(
            seq_lens=self.seq_lens[:bs],
            query_lens=self.query_lens[:bs],
            query_start_loc=self.query_start_loc[: bs + 1],
            token_to_req_indices=self.token_to_req[:total_tokens],
            cache=DeepseekV4CacheMetadata.from_group_tables(
                page_size=kernel_page_size,
                page_table=self.page_table[:bs, :max_num_pages],
                block_tables={gid: buf[:bs] for gid, buf in self.block_tables.items()},
                kvp_size=self.kvp_size,
                kvp_rank=self.kvp_rank,
                runtime_contract=self.runtime_contract,
            ),
            is_valid_token=self.is_valid_token[:total_tokens],
            seq_lens_cpu=None,
            query_lens_cpu=None,
            forward_mode=forward_mode,
        )
        self._views[(bs, tokens_per_req)] = metadata
        return metadata

    # ------------------------------------------------------------------
    # In-place refreshes
    # ------------------------------------------------------------------

    def tokens_per_req(self, bs: int, num_tokens: int) -> int:
        """Packed verify width for this refresh; validates uniform packing."""
        if num_tokens != bs:
            if bs == 0:
                return self.max_tokens_per_req
            if num_tokens % bs != 0:
                raise RuntimeError(
                    "DeepSeek V4 packed CUDA graph metadata expects uniformly "
                    f"packed tokens per request, got num_tokens={num_tokens}, "
                    f"bs={bs}"
                )
            width = num_tokens // bs
            if width > self.max_tokens_per_req:
                raise RuntimeError(
                    "DeepSeek V4 packed CUDA graph metadata was initialized "
                    f"for at most {self.max_tokens_per_req} tokens "
                    f"per request, got {width}"
                )
            return max(1, width)
        return 1

    def write_batch(self, bs: int, seq_lens: torch.Tensor) -> None:
        """Copy this round's row lengths into the persistent buffer."""
        self.seq_lens[:bs].copy_(seq_lens[:bs].to(torch.int32))

    def fill_packed_rows(self, *, bs: int, actual_bs: int, tokens_per_req: int) -> int:
        """Refresh the packed row machinery (query starts, owner map, padding
        mask) from the precomputed per-width tables. Returns total tokens."""
        total_tokens = bs * tokens_per_req
        actual_tokens = actual_bs * tokens_per_req
        query_start = self.query_start_by_width.get(tokens_per_req)
        token_to_req = self.token_to_req_by_width.get(tokens_per_req)
        if query_start is None or token_to_req is None:
            raise RuntimeError(
                "DeepSeek V4 CUDA graph packed metadata was not precomputed "
                f"for tokens_per_req={tokens_per_req}"
            )
        self.query_lens[:bs].fill_(tokens_per_req)
        self.query_start_loc[: bs + 1].copy_(query_start[: bs + 1])
        self.token_to_req[:total_tokens].copy_(token_to_req[:total_tokens])
        self.is_valid_token[:actual_tokens].fill_(True)
        if actual_tokens < total_tokens:
            self.is_valid_token[actual_tokens:total_tokens].fill_(False)
        return total_tokens

    def refresh_block_tables(
        self,
        bs: int,
        block_tables: dict[str, torch.Tensor],
        *,
        actual_bs: int | None = None,
        pad_value: int = 0,
    ) -> None:
        """Refresh the persistent per-group tables in place (the views built
        over them see the new rows). Missing groups keep the pad fill."""
        active_rows = bs if actual_bs is None else actual_bs
        for group_id, buf in self.block_tables.items():
            table = block_tables.get(group_id)
            buf[:bs].fill_(pad_value)
            if table is None:
                continue
            if int(table.shape[0]) < active_rows:
                raise RuntimeError(
                    "DeepSeek V4 CUDA graph cache-group table row count "
                    f"mismatch for {group_id!r}: got {int(table.shape[0])}, "
                    f"expected at least actual_bs {active_rows}"
                )
            cols = int(table.shape[1])
            if cols > int(buf.shape[1]):
                raise RuntimeError(
                    "DeepSeek V4 CUDA graph cache-group table width "
                    f"mismatch for {group_id!r}: got {cols}, capture "
                    f"buffer has {int(buf.shape[1])}"
                )
            if active_rows > 0 and cols > 0:
                buf[:active_rows, :cols].copy_(
                    table[:active_rows, :cols].to(torch.int32)
                )


class DeepseekV4DraftRounds:
    """Owner of the draft's plain-step (bs-row) decode metadata views.

    The V4 draft's per-step decode metadata, as pointer-stable per-bs views.

    A V4 MTP draft alternates two row shapes each round: the packed bs*N verify
    step (step 0, served by the GraphBuffers views) and the plain bs-row steps
    1+ that this object serves. The plain-step metadata is one pointer-stable
    view per bs, exactly like the unified decode path's per-bs views. ``prepare``
    and ``advance`` are copy-only — no arm ever rebinds a tensor field, so there
    is nothing a captured graph could have recorded that a later round can
    invalidate, on the graph path and the eager path alike (they are the same
    object). The one non-tensor slot, ``cache``, is borrowed from the round's
    source metadata: the packed GraphBuffers views on the graph path (whose
    group tables are themselves fixed slices of the persistent buffers), the
    round's fresh prefill metadata on the eager extend path.

    Storage-sharing rules with :class:`DeepseekV4GraphBuffers` (request-major
    row state is safe to share, token-major packed state is NOT):

    * ``query_start_loc`` / ``token_to_req_indices`` — shared width-1 constant
      arange tables (``graph.query_start_by_width[1]`` etc.), never mutated.
    * ``seq_lens`` — OWN buffer: ``advance`` records a ``copy_`` of the
      drafter's lengths into it inside the graph; ``graph.seq_lens`` carries the
      packed step's lengths and must not move underneath it.
    * ``is_valid_token`` — OWN buffer: the per-request gather of the packed
      token-major mask; sharing ``graph.is_valid_token`` would alias rows with
      wrong values.
    """

    def __init__(self, graph: DeepseekV4GraphBuffers) -> None:
        self._graph = graph
        device = graph.seq_lens.device
        max_bs = graph.max_bs
        # Own row state (see the class docstring's sharing rules).
        self.step_seq_lens = torch.zeros(max_bs, dtype=torch.int32, device=device)
        self.step_valid = torch.ones(max_bs, dtype=torch.bool, device=device)
        # Width-1 query lens are a constant the graph never mutates; own a
        # dedicated ones table so packed refreshes (which fill
        # graph.query_lens with N) cannot disturb the recorded step views.
        self._step_query_lens = torch.ones(max_bs, dtype=torch.int32, device=device)
        self._views: dict[int, DeepseekV4ForwardMetadata] = {}
        # The round's live step metadata; None before the first prepare.
        self.current: DeepseekV4ForwardMetadata | None = None

    def step_views(self, bs: int) -> DeepseekV4ForwardMetadata:
        """The pointer-stable plain-step views for ``bs`` (built once)."""
        metadata = self._views.get(bs)
        if metadata is not None:
            return metadata
        graph = self._graph
        metadata = DeepseekV4ForwardMetadata(
            seq_lens=self.step_seq_lens[:bs],
            query_lens=self._step_query_lens[:bs],
            query_start_loc=graph.query_start_by_width[1][: bs + 1],
            token_to_req_indices=graph.token_to_req_by_width[1][:bs],
            cache=None,
            is_valid_token=self.step_valid[:bs],
            forward_mode=ForwardMode.DECODE,
        )
        self._views[bs] = metadata
        return metadata

    def prepare(
        self,
        prefill_metadata: DeepseekV4ForwardMetadata,
        base_seq_lens: torch.Tensor,
    ) -> DeepseekV4ForwardMetadata:
        """Bind this round's step metadata from the packed/prefill state.

        Copy-only over the per-bs views; the ``cache`` slot borrows the
        source's (see the class docstring). Returns the views for the
        backend to run the indexer/slot-mapping refresh hooks over and publish.
        """
        bs = prefill_metadata.seq_lens.numel()
        metadata = self.step_views(bs)
        metadata.cache = prefill_metadata.cache
        self.step_seq_lens[:bs].copy_(base_seq_lens[:bs])
        packed_mask = prefill_metadata.is_valid_token
        if packed_mask is None:
            self.step_valid[:bs].fill_(True)
        else:
            self.step_valid[:bs].copy_(
                packed_mask[prefill_metadata.query_start_loc[:bs].to(torch.int64)]
            )
        metadata.num_prefill_reqs = 0
        metadata.num_prefill_tokens = 0
        metadata.forward_mode = ForwardMode.DECODE
        self.current = metadata
        return metadata

    def advance(self, seq_lens: torch.Tensor) -> DeepseekV4ForwardMetadata:
        """Advance the live step metadata to the next draft step in place."""
        metadata = self.current
        if metadata is None:
            raise RuntimeError("DeepSeek V4 draft metadata was not initialized")
        metadata.seq_lens.copy_(seq_lens[: metadata.seq_lens.numel()])
        metadata.forward_mode = ForwardMode.DECODE
        return metadata


T = TypeVar("T")


class DeepseekV4ForwardSlotMappings:
    """The token-shaped write-slot mappings one forward's layers share.

    The SWA mapping, the compressor state / compressed-cache mappings per
    compress ratio and the indexer-state mapping depend only on the forward's
    rows and the published metadata, not on the layer, so the first layer
    that needs one computes it and every later layer reuses it (each is a
    chain of small elementwise kernels — per layer they once baked ~7 kernels
    x 61 layers into the captured decode graph).

    This is backend scratch, deliberately outside the metadata slots:
    ``graph_ptr_guard`` snapshots every tensor a slot reaches, and a mapping
    computed under capture must not be pinned there. The backend clears the
    memo in every slot publisher, and every forward is preceded by a publish
    (extend init, decode refresh, the drafter's per-step advance, capture
    seeding), so no mapping outlives the forward it was computed for.
    """

    def __init__(self) -> None:
        self._entries: dict[Hashable, object] = {}

    def clear(self) -> None:
        self._entries.clear()

    def first_use(self, key: Hashable) -> bool:
        """Return whether ``key`` is observed for the first time this forward."""
        if key in self._entries:
            return False
        self._entries[key] = True
        return True

    def get_or_compute(self, key: Hashable, compute: Callable[[], T]) -> T:
        """Return the mapping memoized under ``key``, computing it on the
        first request of the forward.

        Args:
            key: The mapping's identity, e.g. ``"swa"`` or
                ``("state", compress_ratio)``.
            compute: Builds the mapping when the memo has no entry.

        Returns:
            The memoized value; the same object for every later call within
            the forward.
        """
        try:
            return self._entries[key]  # type: ignore[return-value]
        except KeyError:
            value = compute()
            self._entries[key] = value
            return value
