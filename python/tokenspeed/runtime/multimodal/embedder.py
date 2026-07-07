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

  2. ``_encode`` invokes the model-supplied encoder once per modality with
     every miss in the batch in a single call, then writes each item's
     output back onto the item itself (``item.encoded`` /
     ``item.encoded_deepstack``).

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
import math
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

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
# Small transfers are faster after staging the whole batch; larger transfers
# benefit from overlapping each H2D enqueue with the next SHM-to-pinned copy.
_INTERLEAVED_H2D_MIN_AVERAGE_BYTES = 1024 * 1024
# One rank can stage the input and distribute it faster than every TP rank
# repeating the host copy once the payload reaches this TP-scaled threshold.
# Local B200 measurements put the break-even points near 128 MiB for TP2 and
# 64 MiB for TP4.
_TP_BROADCAST_BASE_MIN_BYTES = 256 * 1024 * 1024


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


# ---------------------------------------------------------------------------
# MultimodalEmbedder
# ---------------------------------------------------------------------------


class MultimodalEmbedder:
    """Multimodal input embedding pipeline for one model executor."""

    def __init__(self, vision_tp_group: tuple[int, ...] | None = None) -> None:
        self._h2d_stream: torch.cuda.Stream | None = None
        self._vision_tp_group = vision_tp_group
        self._vision_tp_process_group = None
        self._vision_tp_src_rank: int | None = None
        if vision_tp_group is not None and len(vision_tp_group) > 1:
            from tokenspeed.runtime.distributed.process_group_manager import (
                process_group_manager as pg_manager,
            )

            self._vision_tp_process_group = pg_manager.get_process_group(
                "nccl", vision_tp_group
            )
            self._vision_tp_src_rank = vision_tp_group[0]

    # --- public entry point ------------------------------------------------

    def apply(
        self,
        input_ids: torch.Tensor,
        text_embedding: nn.Embedding,
        ctx: MultimodalForwardContext | None,
        encoders: dict[Modality, EncoderSpec],
        multimodal_model: nn.Module,
        is_decode_or_idle: bool = False,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        """Compose LM input embeddings with encoder tokens scattered in.

        Returns ``(None, {})`` when there is nothing multimodal to do this
        forward (decode iteration, or no active multimodal inputs). The
        caller falls back to the regular text-only path on that signal.
        """
        if is_decode_or_idle or ctx is None or not ctx.has_extend_inputs():
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

        encode_started = time.perf_counter() if LOG_MM_TIMING else None
        self._encode(plan, encoders, multimodal_model, input_ids.device)
        encode_elapsed_ms = (
            (time.perf_counter() - encode_started) * 1000
            if encode_started is not None
            else None
        )

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
    ) -> None:
        for modality, items in plan.misses_by_modality.items():
            if not items:
                continue
            spec = encoders.get(modality)
            if spec is None:
                raise RuntimeError(
                    f"MultimodalEmbedder: no encoder registered for {modality}"
                )

            move_started = time.perf_counter() if LOG_MM_TIMING else None
            self._move_features_to_device(items, device)
            move_elapsed_ms = (
                (time.perf_counter() - move_started) * 1000
                if move_started is not None
                else None
            )
            encoder_started = time.perf_counter() if LOG_MM_TIMING else None
            output = spec.fn(items)
            if LOG_MM_TIMING and device.type == "cuda":
                torch.cuda.synchronize(device)
            encoder_elapsed_ms = (
                (time.perf_counter() - encoder_started) * 1000
                if encoder_started is not None
                else None
            )
            output = output.reshape(-1, output.shape[-1])

            per_item_lens = [_item_token_count(it) for it in items]
            per_item_embs = torch.split(output, per_item_lens, dim=0)

            if spec.deepstack:
                for item, emb in zip(items, per_item_embs):
                    main, deep = multimodal_model.separate_deepstack_embeds(emb)
                    item.encoded = main
                    item.encoded_deepstack = deep
            else:
                for item, emb in zip(items, per_item_embs):
                    item.encoded = emb
            if LOG_MM_TIMING:
                logger.info(
                    "mm_timing encoder_ms modality=%s items=%d "
                    "encoder_output_tokens=%d move_h2d=%.3f encode=%.3f "
                    "per_item_tokens=%s",
                    modality.name,
                    len(items),
                    int(output.shape[0]),
                    move_elapsed_ms,
                    encoder_elapsed_ms,
                    per_item_lens,
                )

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
        stream with the H2D stream before returning so the encode call
        sees the moved tensors.
        """
        pending = [
            it
            for it in items
            if isinstance(it.feature, (torch.Tensor, ShmTensorHandle))
            and (isinstance(it.feature, ShmTensorHandle) or it.feature.device != device)
        ]
        if not pending:
            return

        if device.type != "cuda":
            for it in pending:
                if isinstance(it.feature, ShmTensorHandle):
                    it.feature = it.feature.consume()
                if isinstance(it.feature, torch.Tensor):
                    it.feature = it.feature.to(device, non_blocking=True)
            return

        shm_count = 0
        shm_nbytes = 0
        for item in pending:
            if isinstance(item.feature, ShmTensorHandle):
                shm_count += 1
                shm_nbytes += item.feature.nbytes
        use_tp_broadcast = self._should_move_shm_via_tp_broadcast(pending)
        interleave_h2d = shm_nbytes > shm_count * _INTERLEAVED_H2D_MIN_AVERAGE_BYTES
        if not use_tp_broadcast and not interleave_h2d:
            for item in pending:
                if isinstance(item.feature, ShmTensorHandle):
                    item.feature = item.feature.consume()

        h2d = self._h2d_stream_on(device)
        current = torch.cuda.current_stream(device)
        with torch.cuda.stream(h2d):
            if use_tp_broadcast:
                self._move_shm_via_tp_broadcast(pending, device)
                current.wait_stream(h2d)
                return
            for it in pending:
                if isinstance(it.feature, ShmTensorHandle):
                    it.feature = it.feature.consume()
                if isinstance(it.feature, torch.Tensor):
                    it.feature = it.feature.to(device, non_blocking=True)
        current.wait_stream(h2d)

    def _should_move_shm_via_tp_broadcast(
        self, items: list[MultimodalDataItem]
    ) -> bool:
        tp_group = self._vision_tp_group
        if (
            tp_group is None
            or self._vision_tp_process_group is None
            or self._vision_tp_src_rank is None
            or not items
            or not all(isinstance(item.feature, ShmTensorHandle) for item in items)
        ):
            return False

        handles = [item.feature for item in items]
        dtype = handles[0].dtype
        if any(handle.dtype != dtype for handle in handles):
            return False
        total_nbytes = sum(handle.nbytes for handle in handles)
        return total_nbytes >= _TP_BROADCAST_BASE_MIN_BYTES // len(tp_group)

    def _move_shm_via_tp_broadcast(
        self,
        items: list[MultimodalDataItem],
        device: torch.device,
    ) -> None:
        tp_group = self._vision_tp_group
        process_group = self._vision_tp_process_group
        src_rank = self._vision_tp_src_rank
        assert tp_group is not None
        assert process_group is not None
        assert src_rank is not None

        handles = [item.feature for item in items]
        dtype = handles[0].dtype

        element_lengths = [math.prod(handle.shape) for handle in handles]
        base = torch.empty(sum(element_lengths), dtype=dtype, device=device)
        is_source = torch.distributed.get_rank() == src_rank
        offset = 0
        if is_source:
            for handle, length in zip(handles, element_lengths, strict=True):
                source = handle.consume().reshape(-1)
                base.narrow(0, offset, length).copy_(source, non_blocking=True)
                offset += length
        else:
            for handle in handles:
                handle.release()

        torch.distributed.broadcast(base, src=src_rank, group=process_group)
        offset = 0
        for item, handle, length in zip(items, handles, element_lengths, strict=True):
            item.feature = base.narrow(0, offset, length).view(handle.shape)
            offset += length

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
