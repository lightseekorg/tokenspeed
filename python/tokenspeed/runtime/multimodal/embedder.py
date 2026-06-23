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

"""VisionEmbedder: assemble LM input embeddings with vision tokens spliced in.

Three sequential phases:

  1. ``_plan`` walks the active multimodal inputs in the current forward
     batch and emits an :class:`EncodePlan` listing (a) the unique items
     that still need to be encoded this iteration and (b) every flat
     position in ``input_ids`` that should be filled from a vision token,
     along with the source range inside the owning item's encoded tensor.

  2. ``_encode`` invokes the model-supplied encoder once per modality with
     every miss in the batch in a single call, then writes each item's
     output back onto the item itself (``item.encoded`` /
     ``item.encoded_deepstack``).

  3. ``_assemble`` runs the text-token embedding lookup and slices the
     vision-token ranges into the right positions using the plan's
     :class:`ScatterRange` records.

Per-item encoded tensors live on the :class:`MultimodalDataItem` itself,
not in an engine-global cache. Lifetime tracks the owning request: when
the request finishes and its ``RequestState`` is dropped, the tensors are
released by GC. Across chunked-prefill iterations of the same request the
item is identical Python object, so the second chunk sees ``item.encoded``
already set and skips re-encoding.

Within a single forward batch we still de-duplicate by ``item.hash``: if
two requests reference the same image content, only the first item is
fed to the encoder; the second request's scatter ranges read from the
first item's ``encoded`` tensor.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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

EncoderFn = Callable[[List[MultimodalDataItem]], torch.Tensor]

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


def pad_input_tokens(input_ids: List[int], mm_inputs: MultimodalInputs) -> List[int]:
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
    """One contiguous range to fill with vision tokens.

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
    process; each unique content hash appears at most once.
    ``scatter_ranges`` describes every place a vision token must land.
    """

    misses_by_modality: Dict[Modality, List[MultimodalDataItem]] = field(
        default_factory=lambda: defaultdict(list)
    )
    scatter_ranges: List[ScatterRange] = field(default_factory=list)
    aliases_by_canonical: Dict[MultimodalDataItem, List[MultimodalDataItem]] = field(
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
# VisionEmbedder
# ---------------------------------------------------------------------------


class VisionEmbedder:
    """Vision-aware input embedding pipeline for one model executor."""

    def __init__(self) -> None:
        self._h2d_stream: Optional[torch.cuda.Stream] = None

    # --- public entry point ------------------------------------------------

    def apply(
        self,
        input_ids: torch.Tensor,
        text_embedding: nn.Embedding,
        ctx: Optional[MultimodalForwardContext],
        encoders: Dict[Modality, EncoderSpec],
        multimodal_model: nn.Module,
        is_decode_or_idle: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """Compose LM input embeddings with vision tokens scattered in.

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
        released_encoded_features = self._drop_encoded_pixel_features(ctx)
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
                "mm_timing vision_embedder_apply_ms total=%.3f plan=%.3f "
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

        # Within-batch dedup: first item per content hash is canonical;
        # duplicates reuse its encoded tensor.
        canonical_by_hash: Dict[int, MultimodalDataItem] = {}
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
                elif item.hash is not None and item.hash in canonical_by_hash:
                    canonical = canonical_by_hash[item.hash]
                else:
                    canonical = item
                    if item.hash is not None:
                        canonical_by_hash[item.hash] = item

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
        encoders: Dict[Modality, EncoderSpec],
        multimodal_model: nn.Module,
        device: torch.device,
    ) -> None:
        for modality, items in plan.misses_by_modality.items():
            if not items:
                continue
            spec = encoders.get(modality)
            if spec is None:
                raise RuntimeError(
                    f"VisionEmbedder: no encoder registered for {modality}"
                )

            move_started = time.perf_counter() if LOG_MM_TIMING else None
            self._move_pixel_features_to_device(items, device)
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
        encoders: Dict[Modality, EncoderSpec],
        multimodal_model: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Placeholder positions hold large content-derived IDs that exceed
        # vocab_size; the lookup we run here is overwritten for those rows
        # by the scatter below, but the lookup still needs valid indices.
        vocab_size = text_embedding.num_embeddings
        safe_ids = input_ids.clamp(min=0, max=vocab_size - 1)
        input_embeds = text_embedding(safe_ids)

        kwargs: Dict[str, Any] = {}
        deepstack_buffer: Optional[torch.Tensor] = None
        if any(spec.deepstack for spec in encoders.values()):
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
                    "VisionEmbedder: item scheduled for encode has no "
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

    def _move_pixel_features_to_device(
        self, items: List[MultimodalDataItem], device: torch.device
    ) -> None:
        """Stage pixel features onto ``device`` on a dedicated H2D stream.

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

    @staticmethod
    def _drop_raw_feature(item: MultimodalDataItem) -> bool:
        if item.feature is None:
            return False
        if isinstance(item.feature, ShmTensorHandle):
            item.feature.release()
        item.feature = None
        return True

    @staticmethod
    def _drop_encoded_pixel_features(ctx: MultimodalForwardContext) -> int:
        released = 0
        for mm in ctx.mm_inputs:
            if mm is None:
                continue
            for it in mm.mm_items:
                if it.encoded is not None and VisionEmbedder._drop_raw_feature(it):
                    released += 1
        return released
