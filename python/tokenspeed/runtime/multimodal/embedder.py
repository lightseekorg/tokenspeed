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

"""Assemble LM input embeddings with multimodal encoder tokens spliced in.

Three sequential phases:

  1. ``_plan`` walks the active multimodal inputs in the current forward
     batch and emits an :class:`EncodePlan` listing (a) the unique items
     that still need to be encoded this iteration and (b) every flat
     position in ``input_ids`` that should be filled from an encoder token,
     along with the source range inside the owning item's encoded tensor.

  2. ``_encode`` invokes the model-supplied encoder once per modality, then
     writes each item's output back onto the item itself (``item.encoded`` /
     ``item.encoded_deepstack``). In weight-TP mode every rank encodes the
     full miss list together. In item-DP mode each rank encodes a deterministic
     subset of whole items and collects the variable-length results with
     exact-size broadcasts.

  3. ``_assemble`` runs the text-token embedding lookup and slices the
     encoder-token ranges into the right positions using the plan's
     :class:`ScatterRange` records.

Per-item encoded tensors live on the :class:`MultimodalDataItem` itself,
not in an engine-global cache. Lifetime tracks the owning request: when
the request finishes and its ``RequestState`` is dropped, the tensors are
released by GC. Across chunked-prefill iterations of the same request the
item is identical Python object, so the second chunk sees ``item.encoded``
already set and skips re-encoding.

Within a single forward batch we still de-duplicate by modality and
``item.hash``: if two requests reference the same media content using
the same modality, only the first item is fed to the encoder; the second
request's scatter ranges read from the first item's ``encoded`` tensor.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import nn

from tokenspeed.runtime.distributed.mapping import VisionTowerMapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalForwardContext,
    MultimodalInputs,
)
from tokenspeed.runtime.multimodal.shm_transport import ShmTensorHandle
from tokenspeed.runtime.utils.env import envs

EncoderFn = Callable[[list[MultimodalDataItem]], torch.Tensor]

logger = logging.getLogger(__name__)
LOG_MM_TIMING = envs.TOKENSPEED_LOG_MM_TIMING.get()


@dataclass
class EncoderSpec:
    """Per-modality encoder registration.

    Bundles the encoder callable with whether its output needs to be
    split into a main + deepstack pair via the model's
    ``separate_deepstack_embeds`` hook.
    """

    fn: EncoderFn
    deepstack: bool = False


# ---------------------------------------------------------------------------
# Input-id padding helper
# ---------------------------------------------------------------------------


def pad_input_tokens(input_ids: list[int], mm_inputs: MultimodalInputs) -> list[int]:
    """Substitute placeholder token IDs with each item's ``pad_value``.

    The gateway produces ``input_ids`` with a single placeholder token
    repeated across every multimodal-token position (e.g. ``<image>``
    repeated 1024 times for a 1024-token image). The prefix cache needs
    each placeholder run to carry a content-derived ID so two different
    images compare unequal. We rewrite each ``offsets`` range to the
    item's pre-computed ``pad_value`` here.
    """
    if not input_ids or not mm_inputs.mm_items:
        return input_ids

    out = None
    for item in mm_inputs.mm_items:
        if item.pad_value is None or not item.offsets:
            continue
        if out is None:
            out = list(input_ids)
        pad_value = int(item.pad_value)
        for offset_start, offset_end in item.offsets:
            out[offset_start : offset_end + 1] = [pad_value] * (
                offset_end - offset_start + 1
            )
    return input_ids if out is None else out


# ---------------------------------------------------------------------------
# Plan structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScatterRange:
    """One contiguous range to fill with multimodal encoder tokens.

    ``flat_dst_*`` are positions in the batch-flat ``input_ids`` tensor
    (inclusive on both ends). ``item_src_*`` are positions within
    ``item.encoded`` (also inclusive). ``item`` is the *canonical* item
    holding the encoded tensor — for within-batch dedup'd entries it may
    differ from the request-local item that produced the offsets.
    """

    flat_dst_start: int
    flat_dst_end: int
    item: MultimodalDataItem
    item_src_start: int
    item_src_end: int


@dataclass
class EncodePlan:
    """Work to do this prefill iteration.

    ``misses_by_modality`` lists the canonical items the encoder needs to
    process; each modality/content-hash pair appears at most once.
    ``scatter_ranges`` describes every place an encoder token must land.
    """

    misses_by_modality: dict[Modality, list[MultimodalDataItem]] = field(
        default_factory=lambda: defaultdict(list)
    )
    scatter_ranges: list[ScatterRange] = field(default_factory=list)
    aliases_by_canonical: dict[MultimodalDataItem, list[MultimodalDataItem]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def __bool__(self) -> bool:
        return bool(self.scatter_ranges)


def _item_token_count(item: MultimodalDataItem) -> int:
    """Total encoded tokens for an item. One offset per subgrid; the
    encoder concatenates subgrid tokens in offsets order."""
    if not item.offsets:
        return 0
    return sum(end - start + 1 for start, end in item.offsets)


@dataclass(frozen=True)
class EncoderDPAssignment:
    """Deterministic ownership and rank-major output sizes for encoder items."""

    item_indices_by_rank: tuple[tuple[int, ...], ...]
    token_counts_by_rank: tuple[int, ...]


def _assign_encoder_items(
    item_token_counts: Sequence[int], dp_size: int
) -> EncoderDPAssignment:
    """Assign whole items with deterministic longest-processing-time packing.

    Encoder DP intentionally never splits one multimodal item. The encoded
    token count is available before execution from the authored placeholder
    offsets and is a stable proxy for encoder work across every rank.
    """
    rank_loads = [0] * dp_size
    item_indices_by_rank: list[list[int]] = [[] for _ in range(dp_size)]

    # Stable tie breaks are part of the collective protocol: all ranks must
    # derive exactly the same owner without exchanging Python objects.
    for item_index in sorted(
        range(len(item_token_counts)),
        key=lambda index: (-item_token_counts[index], index),
    ):
        owner = min(range(dp_size), key=lambda rank: (rank_loads[rank], rank))
        item_indices_by_rank[owner].append(item_index)
        rank_loads[owner] += item_token_counts[item_index]

    # The encoder concatenates outputs in input order. Keep each rank's local
    # subset in that order so rank-major gathered rows can be reconstructed.
    for indices in item_indices_by_rank:
        indices.sort()
    return EncoderDPAssignment(
        item_indices_by_rank=tuple(tuple(indices) for indices in item_indices_by_rank),
        token_counts_by_rank=tuple(rank_loads),
    )


# ---------------------------------------------------------------------------
# MultimodalEmbedder
# ---------------------------------------------------------------------------


class MultimodalEmbedder:
    """Multimodal input embedding pipeline for one model executor."""

    def __init__(
        self,
        encoder_mapping: VisionTowerMapping | None = None,
    ) -> None:
        self._encoder_dp_group = (
            encoder_mapping.dp_group if encoder_mapping is not None else None
        )
        self._encoder_dp_rank = (
            encoder_mapping.dp_rank if encoder_mapping is not None else 0
        )
        self._h2d_stream: torch.cuda.Stream | None = None

    @property
    def has_encoder_dp(self) -> bool:
        return self._encoder_dp_group is not None and len(self._encoder_dp_group) > 1

    # --- public entry point ------------------------------------------------

    def apply(
        self,
        input_ids: torch.Tensor,
        text_embedding: nn.Embedding,
        ctx: MultimodalForwardContext | None,
        encoders: dict[Modality, EncoderSpec],
        multimodal_model: nn.Module,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """Compose LM input embeddings with encoder tokens scattered in.

        Returns ``(None, {})`` when there is no active multimodal overlap in
        this forward. The caller falls back to the regular text-only path on
        that signal. With multimodal timing enabled, CUDA encode time is
        measured with events on the current stream.
        """
        if ctx is None or not ctx.has_extend_inputs():
            return None, {}

        total_started = time.perf_counter() if LOG_MM_TIMING else None
        plan_started = time.perf_counter() if LOG_MM_TIMING else None
        plan = self._plan(ctx)
        plan_elapsed_ms = (
            (time.perf_counter() - plan_started) * 1000
            if plan_started is not None
            else None
        )
        if not plan:
            return None, {}

        encode_started: float | None = None
        encode_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
        encode_elapsed_ms: float | None = None
        if LOG_MM_TIMING:
            if input_ids.device.type == "cuda":
                encode_events = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                encode_events[0].record(torch.cuda.current_stream(input_ids.device))
            else:
                encode_started = time.perf_counter()
        self._encode(
            plan,
            encoders,
            multimodal_model,
            input_ids.device,
            text_embedding.embedding_dim,
            text_embedding.weight.dtype,
        )
        if LOG_MM_TIMING:
            if encode_events is not None:
                encode_events[1].record(torch.cuda.current_stream(input_ids.device))
                encode_events[1].synchronize()
                encode_elapsed_ms = encode_events[0].elapsed_time(encode_events[1])
            elif encode_started is not None:
                encode_elapsed_ms = (time.perf_counter() - encode_started) * 1000

        alias_started = time.perf_counter() if LOG_MM_TIMING else None
        released_alias_features = self._share_encoded_aliases(plan)
        alias_elapsed_ms = (
            (time.perf_counter() - alias_started) * 1000
            if alias_started is not None
            else None
        )

        assemble_started = time.perf_counter() if LOG_MM_TIMING else None
        input_embeds, kwargs = self._assemble(
            input_ids, text_embedding, plan, encoders, multimodal_model
        )
        assemble_elapsed_ms = (
            (time.perf_counter() - assemble_started) * 1000
            if assemble_started is not None
            else None
        )

        cleanup_started = time.perf_counter() if LOG_MM_TIMING else None
        released_encoded_features = self._drop_encoded_features(ctx)
        cleanup_elapsed_ms = (
            (time.perf_counter() - cleanup_started) * 1000
            if cleanup_started is not None
            else None
        )
        if LOG_MM_TIMING and total_started is not None:
            misses = {
                modality.name: len(items)
                for modality, items in plan.misses_by_modality.items()
                if items
            }
            logger.info(
                "mm_timing multimodal_embedder_apply_ms total=%.3f plan=%.3f "
                "encode=%.3f alias=%.3f assemble=%.3f feature_cleanup=%.3f "
                "scatter_ranges=%d misses=%s input_rows=%d aliases=%d "
                "released_alias_features=%d released_encoded_features=%d",
                (time.perf_counter() - total_started) * 1000,
                plan_elapsed_ms,
                encode_elapsed_ms,
                alias_elapsed_ms,
                assemble_elapsed_ms,
                cleanup_elapsed_ms,
                len(plan.scatter_ranges),
                misses,
                int(input_ids.numel()),
                sum(len(items) for items in plan.aliases_by_canonical.values()),
                released_alias_features,
                released_encoded_features,
            )
        return input_embeds, kwargs

    # --- phase 1: plan -----------------------------------------------------

    def _plan(self, ctx: MultimodalForwardContext) -> EncodePlan:
        plan = EncodePlan()
        if not ctx.mm_inputs:
            return plan

        # Within-batch dedup: first item per modality and content hash is
        # canonical; duplicates reuse its encoded tensor.
        canonical_by_key: dict[tuple[Modality, int], MultimodalDataItem] = {}
        scheduled: set[MultimodalDataItem] = set()

        # Walk the FULL batch (including text-only / decode requests)
        # so base offsets line up with the flat input_ids tensor that
        # the caller hands us. Requests without mm input contribute
        # nothing but still advance ``base``.
        base = 0
        for req_idx, mm_inputs in enumerate(ctx.mm_inputs):
            if req_idx >= len(ctx.extend_seq_lens) or req_idx >= len(
                ctx.extend_prefix_lens
            ):
                break
            seq = ctx.extend_seq_lens[req_idx]
            if mm_inputs is None or seq <= 0:
                base += max(seq, 0)
                continue

            prefix = ctx.extend_prefix_lens[req_idx]
            chunk_start = prefix
            chunk_end_inc = prefix + seq - 1

            for item in mm_inputs.mm_items:
                if item is None or not item.offsets:
                    continue

                if item.encoded is not None:
                    canonical = item
                elif (
                    item.hash is not None
                    and (item.modality, item.hash) in canonical_by_key
                ):
                    canonical = canonical_by_key[(item.modality, item.hash)]
                else:
                    canonical = item
                    if item.hash is not None:
                        canonical_by_key[(item.modality, item.hash)] = item

                if canonical is not item:
                    plan.aliases_by_canonical[canonical].append(item)

                # src_cursor: start of current subgrid inside item.encoded.
                src_cursor = 0
                for offset_start, offset_end in item.offsets:
                    span = offset_end - offset_start + 1
                    overlap_start = max(offset_start, chunk_start)
                    overlap_end = min(offset_end, chunk_end_inc)
                    if overlap_start > overlap_end:
                        src_cursor += span
                        continue

                    plan.scatter_ranges.append(
                        ScatterRange(
                            flat_dst_start=base + (overlap_start - prefix),
                            flat_dst_end=base + (overlap_end - prefix),
                            item=canonical,
                            item_src_start=src_cursor + (overlap_start - offset_start),
                            item_src_end=src_cursor + (overlap_end - offset_start),
                        )
                    )
                    if canonical.encoded is None and canonical not in scheduled:
                        scheduled.add(canonical)
                        plan.misses_by_modality[canonical.modality].append(canonical)
                    src_cursor += span

            base += seq

        return plan

    # --- phase 2: encode ---------------------------------------------------

    def _encode(
        self,
        plan: EncodePlan,
        encoders: dict[Modality, EncoderSpec],
        multimodal_model: nn.Module,
        device: torch.device,
        embedding_width: int,
        embedding_dtype: torch.dtype,
    ) -> None:
        for modality, items in plan.misses_by_modality.items():
            if not items:
                continue
            spec = encoders.get(modality)
            if spec is None:
                raise RuntimeError(
                    f"MultimodalEmbedder: no encoder registered for {modality}"
                )

            if self.has_encoder_dp:
                output_width = embedding_width
                if spec.deepstack:
                    output_width *= 1 + len(multimodal_model.deepstack_visual_indexes)
                per_item_embs = self._encode_data_parallel(
                    items,
                    spec,
                    device,
                    output_width,
                    embedding_dtype,
                )
            else:
                output = self._run_encoder(items, spec, device)
                per_item_lens = [_item_token_count(it) for it in items]
                output = output.reshape(-1, output.shape[-1])
                per_item_embs = list(torch.split(output, per_item_lens, dim=0))

            self._store_encoder_outputs(items, per_item_embs, spec, multimodal_model)

    def _run_encoder(
        self,
        items: list[MultimodalDataItem],
        spec: EncoderSpec,
        device: torch.device,
    ) -> torch.Tensor:
        self._move_features_to_device(items, device)
        return spec.fn(items)

    def _encode_data_parallel(
        self,
        items: list[MultimodalDataItem],
        spec: EncoderSpec,
        device: torch.device,
        output_width: int,
        output_dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        assert self._encoder_dp_group is not None
        per_item_lens = tuple(_item_token_count(item) for item in items)
        assignment = _assign_encoder_items(per_item_lens, len(self._encoder_dp_group))
        local_indices = assignment.item_indices_by_rank[self._encoder_dp_rank]
        local_items = [items[index] for index in local_indices]
        local_rows = assignment.token_counts_by_rank[self._encoder_dp_rank]

        local_output = torch.empty((0, output_width), dtype=output_dtype, device=device)
        if local_items:
            local_output = self._run_encoder(local_items, spec, device)
            local_output = local_output.reshape(local_rows, output_width).to(
                device=device, dtype=output_dtype
            )

        gathered = self._gather_encoder_outputs(
            local_output, assignment.token_counts_by_rank
        )
        per_item_embs: list[torch.Tensor | None] = [None] * len(items)
        cursor = 0
        for indices, rank_rows in zip(
            assignment.item_indices_by_rank, assignment.token_counts_by_rank
        ):
            rank_output = gathered[cursor : cursor + rank_rows]
            cursor += rank_rows
            rank_cursor = 0
            for item_index in indices:
                item_rows = per_item_lens[item_index]
                per_item_embs[item_index] = rank_output[
                    rank_cursor : rank_cursor + item_rows
                ]
                rank_cursor += item_rows
        return cast(list[torch.Tensor], per_item_embs)

    def _gather_encoder_outputs(
        self, local_output: torch.Tensor, token_counts_by_rank: Sequence[int]
    ) -> torch.Tensor:
        assert self._encoder_dp_group is not None
        total_rows = sum(token_counts_by_rank)
        output_width = local_output.shape[-1]
        gathered = torch.empty(
            (total_rows, output_width),
            dtype=local_output.dtype,
            device=local_output.device,
        )
        process_group = process_group_manager.get_process_group(
            "nccl", self._encoder_dp_group
        )
        cursor = 0
        for owner_rank, rows in enumerate(token_counts_by_rank):
            if rows == 0:
                continue
            rank_output = gathered[cursor : cursor + rows]
            if owner_rank == self._encoder_dp_rank:
                rank_output.copy_(local_output)
            # A sequence of exact-size broadcasts avoids max-row padding. In
            # the common single-large-item case this reduces the temporary
            # DP8 embedding buffer from 8x the result size to exactly 1x.
            torch.distributed.broadcast(
                rank_output,
                src=self._encoder_dp_group[owner_rank],
                group=process_group,
            )
            cursor += rows
        return gathered

    @staticmethod
    def _store_encoder_outputs(
        items: list[MultimodalDataItem],
        per_item_embs: list[torch.Tensor],
        spec: EncoderSpec,
        multimodal_model: nn.Module,
    ) -> None:
        if spec.deepstack:
            for item, emb in zip(items, per_item_embs):
                main, deep = multimodal_model.separate_deepstack_embeds(emb)
                item.encoded = main
                item.encoded_deepstack = deep
        else:
            for item, emb in zip(items, per_item_embs):
                item.encoded = emb

    def _share_encoded_aliases(self, plan: EncodePlan) -> int:
        released = 0
        for canonical, aliases in plan.aliases_by_canonical.items():
            if canonical.encoded is None:
                continue
            for alias in aliases:
                alias.encoded = canonical.encoded
                alias.encoded_deepstack = canonical.encoded_deepstack
                if self._drop_raw_feature(alias):
                    released += 1
        return released

    # --- phase 3: assemble -------------------------------------------------

    def _assemble(
        self,
        input_ids: torch.Tensor,
        text_embedding: nn.Embedding,
        plan: EncodePlan,
        encoders: dict[Modality, EncoderSpec],
        multimodal_model: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Placeholder positions hold large content-derived IDs that exceed
        # vocab_size; the lookup we run here is overwritten for those rows
        # by the scatter below, but the lookup still needs valid indices.
        vocab_size = text_embedding.num_embeddings
        safe_ids = input_ids.clamp(min=0, max=vocab_size - 1)
        input_embeds = text_embedding(safe_ids)

        kwargs: dict[str, Any] = {}
        deepstack_buffer: torch.Tensor | None = None
        deepstack_modalities = {
            modality for modality, spec in encoders.items() if spec.deepstack
        }
        if any(r.item.modality in deepstack_modalities for r in plan.scatter_ranges):
            num_deepstack = len(multimodal_model.deepstack_visual_indexes)
            shape = input_embeds.shape[:-1] + (input_embeds.shape[-1] * num_deepstack,)
            deepstack_buffer = torch.zeros(
                shape, dtype=input_embeds.dtype, device=input_embeds.device
            )
            kwargs["input_deepstack_embeds"] = deepstack_buffer

        for r in plan.scatter_ranges:
            main = r.item.encoded
            if main is None:
                raise RuntimeError(
                    "MultimodalEmbedder: item scheduled for encode has no "
                    "encoded tensor after _encode; this is a bug"
                )
            src = main[r.item_src_start : r.item_src_end + 1]
            input_embeds[r.flat_dst_start : r.flat_dst_end + 1] = src.to(
                dtype=input_embeds.dtype, device=input_embeds.device
            )

            if deepstack_buffer is not None and r.item.encoded_deepstack is not None:
                deep_src = r.item.encoded_deepstack[
                    r.item_src_start : r.item_src_end + 1
                ]
                deepstack_buffer[r.flat_dst_start : r.flat_dst_end + 1] = deep_src.to(
                    dtype=input_embeds.dtype, device=input_embeds.device
                )

        return input_embeds, kwargs

    # --- device helpers ----------------------------------------------------

    def _h2d_stream_on(self, device: torch.device) -> torch.cuda.Stream:
        if self._h2d_stream is None:
            self._h2d_stream = torch.cuda.Stream(device=device)
        return self._h2d_stream

    def _move_features_to_device(
        self, items: list[MultimodalDataItem], device: torch.device
    ) -> None:
        """Stage encoder features onto ``device`` on a dedicated H2D stream.

        Inputs that originate from the SHM transport are pinned, so the
        H2D copy can actually run async with respect to the LM kernels
        already queued on the current stream. We synchronise the current
        stream with the H2D stream so the encoder sees the moved tensors,
        then record that consumer stream for allocator-safe reuse.
        """
        pending = [
            it
            for it in items
            if isinstance(it.feature, (torch.Tensor, ShmTensorHandle))
            and (isinstance(it.feature, ShmTensorHandle) or it.feature.device != device)
        ]
        if not pending:
            return

        for it in pending:
            if isinstance(it.feature, ShmTensorHandle):
                it.feature = it.feature.consume()

        if device.type != "cuda":
            for it in pending:
                if isinstance(it.feature, torch.Tensor):
                    it.feature = it.feature.to(device, non_blocking=True)
            return

        h2d = self._h2d_stream_on(device)
        current = torch.cuda.current_stream(device)
        with torch.cuda.stream(h2d):
            for it in pending:
                if isinstance(it.feature, torch.Tensor):
                    it.feature = it.feature.to(device, non_blocking=True)
        current.wait_stream(h2d)
        for it in pending:
            if isinstance(it.feature, torch.Tensor):
                it.feature.record_stream(current)

    @staticmethod
    def _drop_raw_feature(item: MultimodalDataItem) -> bool:
        if item.feature is None:
            return False
        if isinstance(item.feature, ShmTensorHandle):
            item.feature.release()
        item.feature = None
        return True

    @staticmethod
    def _drop_encoded_features(ctx: MultimodalForwardContext) -> int:
        released = 0
        for mm in ctx.mm_inputs:
            if mm is None:
                continue
            for it in mm.mm_items:
                if it.encoded is not None and MultimodalEmbedder._drop_raw_feature(it):
                    released += 1
        return released


# Compatibility alias for model implementations that predate audio support.
VisionEmbedder = MultimodalEmbedder
