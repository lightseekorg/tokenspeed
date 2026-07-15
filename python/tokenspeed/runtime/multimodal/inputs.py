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

"""Multimodal request data structures used across processors and model adapters."""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Mapping
from enum import Enum, auto
from typing import Any

import numpy as np
import torch

from tokenspeed.runtime.multimodal.hash import hash_feature
from tokenspeed.runtime.multimodal.shm_transport import ShmTensorHandle

# Multimodal pad-value substitute IDs: a placeholder mm token's id is rewritten
# to a content-derived value so duplicate features share the same substitute and
# prefix-match in the text-only prefix cache. The signed-int32 space above the
# text vocabulary is split into one equally sized interval per modality;
# speculative draft models use that interval tag to replace image/audio/video
# positions with the corresponding in-vocab placeholder before embedding lookup.
# Interval encoding preserves substantially more content-hash entropy than
# packing a two-bit modality tag into the old 30-bit payload.
_MM_PAD_BASE = 1_000_000
_MM_PAD_INT32_MAX = (1 << 31) - 1
_MM_PAD_MODALITY_COUNT = 3
_MM_PAD_HASH_SLOTS = (_MM_PAD_INT32_MAX - _MM_PAD_BASE + 1) // _MM_PAD_MODALITY_COUNT
_MM_PAD_MAX = _MM_PAD_BASE + _MM_PAD_HASH_SLOTS * _MM_PAD_MODALITY_COUNT - 1


def is_mm_pad_value(token_ids: torch.Tensor) -> torch.Tensor:
    """Bool mask of positions rewritten to a hash-derived multimodal pad id."""
    return (token_ids >= _MM_PAD_BASE) & (token_ids <= _MM_PAD_MAX)


def _modality_pad_tag(modality: "Modality") -> int:
    return {
        Modality.IMAGE: 0,
        Modality.VIDEO: 1,
        Modality.AUDIO: 2,
    }[modality]


def is_mm_pad_value_for(token_ids: torch.Tensor, modality: "Modality") -> torch.Tensor:
    """Bool mask for hash-derived pad IDs belonging to ``modality``."""
    relative = token_ids - _MM_PAD_BASE
    return is_mm_pad_value(token_ids) & (
        torch.div(relative, _MM_PAD_HASH_SLOTS, rounding_mode="floor")
        == _modality_pad_tag(modality)
    )


def maybe_substitute_mm_pad(
    input_ids: torch.Tensor,
    substitute_ids: int | Mapping["Modality", int] | None,
) -> torch.Tensor:
    """Replace hash MM-pad positions with in-vocab draft token IDs.

    A scalar keeps the legacy behavior and substitutes every modality with one
    token. A mapping preserves modality-specific semantics, which is required
    by mixed image+audio prompts such as TML/Inkling.
    """
    if substitute_ids is None:
        return input_ids
    if isinstance(substitute_ids, int):
        return input_ids.masked_fill(is_mm_pad_value(input_ids), substitute_ids)

    output = input_ids
    for modality, substitute_id in substitute_ids.items():
        if output is input_ids:
            output = input_ids.clone()
        output.masked_fill_(is_mm_pad_value_for(input_ids, modality), substitute_id)
    return output


class Modality(Enum):
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()


def resolve_mm_pad_substitute_ids(config: Any) -> dict[Modality, int]:
    """Resolve in-vocab speculative-draft tokens for each media modality.

    Model families use different configuration names. Prefer a modality's
    explicit token, then its transport placeholder, and finally a shared media
    placeholder. Returning only configured modalities lets text-only and
    partially multimodal models keep their existing behavior.
    """

    sources = [config]
    # Composite multimodal configs do not consistently forward their media
    # token IDs. Qwen3-Omni, for example, keeps all three on thinker_config.
    for nested_name in ("thinker_config", "text_config"):
        nested = getattr(config, nested_name, None)
        if nested is not None and all(nested is not source for source in sources):
            sources.append(nested)

    def first_configured(*names: str) -> int | None:
        for name in names:
            for source in sources:
                value = getattr(source, name, None)
                if value is not None:
                    return int(value)
        return None

    shared = first_configured("media_placeholder_token_id")
    candidates = {
        Modality.IMAGE: first_configured(
            "image_token_id", "image_placeholder_token_id"
        ),
        Modality.VIDEO: first_configured(
            "video_token_id", "video_placeholder_token_id"
        ),
        Modality.AUDIO: first_configured(
            "audio_token_id", "audio_placeholder_token_id"
        ),
    }
    return {
        modality: token_id if token_id is not None else shared
        for modality, token_id in candidates.items()
        if token_id is not None or shared is not None
    }


# ``eq=False`` on every dataclass below: tensor-valued fields crash the
# default element-wise ``__eq__`` and force ``__hash__`` to None.
@dataclasses.dataclass(eq=False)
class MultimodalDataItem:
    modality: Modality
    hash: int | None = None
    pad_value: int | None = None
    offsets: list | None = None
    feature: torch.Tensor | np.ndarray | ShmTensorHandle | None = None
    model_specific_data: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Encoder output for this item, populated on first encoder pass and reused
    # across chunked-prefill iterations of the owning request. Lifetime is
    # tied to the request: when the request finishes the item is GC'd and
    # these tensors are released. ``encoded_deepstack`` is set only for
    # deepstack-enabled modalities.
    encoded: torch.Tensor | None = None
    encoded_deepstack: torch.Tensor | None = None
    # EPD (encode-prefill-decode): when set, this item's embedding is received
    # from an encode worker over Mooncake into ``encoded`` instead of running the
    # vision tower. A dict ``{bootstrap_room, bootstrap_host, bootstrap_port}``
    # naming the encode worker's rendezvous for this item's image (one room per
    # item: the gateway splits the mm payload one item per image and the encode
    # worker row-splits the concatenated-subgrid embedding per item). None for
    # non-EPD items (left to the vision tower).
    encode_handshake: dict | None = None

    def __getattr__(self, name: str):
        if (
            "model_specific_data" in self.__dict__
            and name in self.__dict__["model_specific_data"]
        ):
            return self.__dict__["model_specific_data"][name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def ensure_hash(self, *, skip_compute_hash: bool = False) -> None:
        """Resolve ``self.hash`` to a concrete content id, lazily.

        The hash is resolved on demand rather than at construction because it
        is usually supplied by the caller, a SHM-backed feature cannot be
        hashed here without reading shared memory, and hashing inline bytes is
        only worth doing once the value is actually needed.

        Resolution order:
          * ``skip_compute_hash=True`` -> a random id (dedup disabled);
          * an already-set hash (e.g. the gateway-provided ``content_hash`` for
            image/video) is kept as-is, no recompute;
          * inline features the gateway does not hash (e.g. audio) are hashed
            in-engine via ``hash_feature``;
          * SHM-backed features must carry a caller-provided hash, else raise --
            we cannot hash a handle without reading shared memory.
        """
        if skip_compute_hash:
            self.hash = uuid.uuid4().int
        elif self.hash is None:
            if isinstance(self.feature, ShmTensorHandle):
                raise ValueError(
                    "SHM-backed multimodal items must carry content hash or "
                    "pad_value before TokenSpeed consumes them"
                )
            self.hash = hash_feature(self.feature)
        if self.hash is None:
            raise RuntimeError("Failed to resolve multimodal item hash.")

    def set_pad_value(self, *, skip_compute_hash: bool = False) -> None:
        if self.pad_value is not None:
            return
        self.ensure_hash(skip_compute_hash=skip_compute_hash)
        modality_offset = _modality_pad_tag(self.modality) * _MM_PAD_HASH_SLOTS
        self.pad_value = (
            _MM_PAD_BASE + modality_offset + (self.hash % _MM_PAD_HASH_SLOTS)
        )

    def is_modality(self, modality: Modality) -> bool:
        return self.modality == modality


@dataclasses.dataclass(eq=False)
class MultimodalInputs:
    mm_items: list[MultimodalDataItem]
    im_token_id: int | None = None
    video_token_id: int | None = None
    mrope_positions: torch.Tensor | None = None
    mrope_position_delta: torch.Tensor | None = None
    mrope_position_delta_scalar: int | None = None
    mrope_position_delta_repeated_cache: torch.Tensor | None = None

    def ensure_pad_values(self, *, skip_compute_hash: bool = False) -> None:
        for item in self.mm_items:
            item.set_pad_value(skip_compute_hash=skip_compute_hash)

    def publish_shm_features(self) -> None:
        for item in self.mm_items:
            if isinstance(item.feature, torch.Tensor):
                item.feature = ShmTensorHandle.publish(item.feature)

    def attach_shm_features(self) -> None:
        """Open every pending handle on this rank. Must run before the
        cross-rank barrier in ``request_handler.recv_reqs``.
        """
        for item in self.mm_items:
            if isinstance(item.feature, ShmTensorHandle):
                item.feature.attach()

    def release_shm_features(self, *, log_timing: bool = False) -> None:
        for item in self.mm_items:
            if isinstance(item.feature, ShmTensorHandle):
                item.feature.release(log_timing=log_timing)
                item.feature = None

    def has_pending_shm_features(self) -> bool:
        return any(isinstance(item.feature, ShmTensorHandle) for item in self.mm_items)


@dataclasses.dataclass(eq=False)
class MultimodalForwardContext:
    """Per-forward multimodal metadata for prefill embedding replacement."""

    mm_inputs: list[MultimodalInputs | None]
    extend_prefix_lens: list[int]
    extend_seq_lens: list[int]

    def has_inputs(self) -> bool:
        return bool(self.mm_inputs and any(x is not None for x in self.mm_inputs))

    def has_extend_inputs(self) -> bool:
        return any(
            mm_input is not None and index < len(self.extend_seq_lens)
            for index, mm_input in enumerate(self.mm_inputs)
        )
