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
from enum import Enum, auto
from typing import Any, List, Optional, Union

import numpy as np
import torch

from tokenspeed.runtime.multimodal.hash import hash_feature
from tokenspeed.runtime.multimodal.shm_transport import ShmTensorHandle
from tokenspeed.runtime.utils.env import envs

# Multimodal pad-value substitute IDs: a placeholder mm token's id is rewritten
# to ``_MM_PAD_BASE + (hash & _MM_PAD_HASH_MASK)`` so duplicate features share
# the same substitute and prefix-match in the text-only prefix cache. The base
# sits well above any text vocab; the 30-bit mask keeps cross-hash collisions
# rare enough for long-running servers (~10^9 slots).
_MM_PAD_BASE = 1_000_000
_MM_PAD_HASH_MASK = (1 << 30) - 1


def is_mm_pad_value(token_ids: torch.Tensor) -> torch.Tensor:
    """Bool mask of positions rewritten to a hash-derived multimodal pad id."""
    return (token_ids >= _MM_PAD_BASE) & (token_ids <= _MM_PAD_BASE + _MM_PAD_HASH_MASK)


def maybe_substitute_mm_pad(
    input_ids: torch.Tensor, substitute_id: int | None
) -> torch.Tensor:
    """Replace hash mm-pad positions with ``substitute_id``; no-op if None."""
    if substitute_id is None:
        return input_ids
    return input_ids.masked_fill(is_mm_pad_value(input_ids), substitute_id)


class Modality(Enum):
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()


# ``eq=False`` on every dataclass below: tensor-valued fields crash the
# default element-wise ``__eq__`` and force ``__hash__`` to None.
@dataclasses.dataclass(eq=False)
class MultimodalDataItem:
    modality: Modality
    hash: Optional[int] = None
    pad_value: Optional[int] = None
    offsets: Optional[list] = None
    feature: Optional[Union[torch.Tensor, np.ndarray, ShmTensorHandle]] = None
    model_specific_data: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Encoder output for this item, populated on first encoder pass and reused
    # across chunked-prefill iterations of the owning request. Lifetime is
    # tied to the request: when the request finishes the item is GC'd and
    # these tensors are released. ``encoded_deepstack`` is set only for
    # deepstack-enabled modalities.
    encoded: Optional[torch.Tensor] = None
    encoded_deepstack: Optional[torch.Tensor] = None

    def __getattr__(self, name: str):
        if (
            "model_specific_data" in self.__dict__
            and name in self.__dict__["model_specific_data"]
        ):
            return self.__dict__["model_specific_data"][name]
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def ensure_hash(self):
        """Resolve ``self.hash`` to a concrete content id, lazily.

        The hash is resolved on demand rather than at construction because it
        is usually supplied by the caller, a SHM-backed feature cannot be
        hashed here without reading shared memory, and hashing inline bytes is
        only worth doing once the value is actually needed.

        Resolution order:
          * ``TOKENSPEED_MM_SKIP_COMPUTE_HASH`` -> a random id (dedup disabled);
          * an already-set hash (e.g. the gateway-provided ``content_hash`` for
            image/video) is kept as-is, no recompute;
          * inline features the gateway does not hash (e.g. audio) are hashed
            in-engine via ``hash_feature``;
          * SHM-backed features must carry a caller-provided hash, else raise --
            we cannot hash a handle without reading shared memory.
        """
        if envs.TOKENSPEED_MM_SKIP_COMPUTE_HASH.get():
            self.hash = uuid.uuid4().int
        elif self.hash is None:
            if isinstance(self.feature, ShmTensorHandle):
                raise ValueError(
                    "SHM-backed multimodal items must carry content hash or "
                    "pad_value before TokenSpeed consumes them"
                )
            self.hash = hash_feature(self.feature)
        assert self.hash is not None

    def set_pad_value(self):
        if self.pad_value is not None:
            return
        self.ensure_hash()
        self.pad_value = _MM_PAD_BASE + (self.hash & _MM_PAD_HASH_MASK)

    def is_modality(self, modality: Modality) -> bool:
        return self.modality == modality


@dataclasses.dataclass(eq=False)
class MultimodalInputs:
    mm_items: List[MultimodalDataItem]
    im_token_id: Optional[int] = None
    video_token_id: Optional[int] = None
    mrope_positions: Optional[torch.Tensor] = None
    mrope_position_delta: Optional[torch.Tensor] = None
    mrope_position_delta_scalar: Optional[int] = None
    mrope_position_delta_repeated_cache: Optional[torch.Tensor] = None

    def ensure_pad_values(self) -> None:
        for item in self.mm_items:
            item.set_pad_value()

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

    def release_shm_features(self) -> None:
        for item in self.mm_items:
            if isinstance(item.feature, ShmTensorHandle):
                item.feature.release()
                item.feature = None

    def has_pending_shm_features(self) -> bool:
        return any(isinstance(item.feature, ShmTensorHandle) for item in self.mm_items)


@dataclasses.dataclass(eq=False)
class MultimodalForwardContext:
    """Per-forward multimodal metadata for prefill embedding replacement."""

    mm_inputs: List[Optional[MultimodalInputs]]
    extend_prefix_lens: List[int]
    extend_seq_lens: List[int]

    def has_inputs(self) -> bool:
        return bool(self.mm_inputs and any(x is not None for x in self.mm_inputs))

    def has_extend_inputs(self) -> bool:
        return any(
            mm_input is not None and index < len(self.extend_seq_lens)
            for index, mm_input in enumerate(self.mm_inputs)
        )
