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

import msgspec
import torch

from tokenspeed.runtime.multimodal.hash import hash_feature
from tokenspeed.runtime.multimodal.shm_transport import ShmTensorHandle
from tokenspeed.runtime.utils.env import envs

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


def substitute_mm_pad_(
    token_ids: torch.Tensor,
    substitute_ids: Mapping["Modality", int],
) -> torch.Tensor:
    """Replace hash MM-pad positions with in-vocab draft token IDs, in place.

    Rewrites every hash-derived multimodal pad id to its modality's in-vocab
    substitute token, directly in ``token_ids``. Returns ``token_ids``.
    Modality-specific substitutes are required by mixed image+audio prompts
    such as TML/Inkling.
    """
    for modality, substitute_id in substitute_ids.items():
        token_ids.masked_fill_(is_mm_pad_value_for(token_ids, modality), substitute_id)
    return token_ids


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


# ``eq=False`` on every IPC struct below: tensor-valued fields crash the
# default element-wise ``__eq__`` and force ``__hash__`` to None.
#
# ``msgspec.Struct``: these types cross engine IPC nested inside the tokenized
# request, so they encode as typed msgpack (tensors ride as ext-typed raw
# buffers via the io_struct codec hooks). Field ORDER is the wire contract —
# append, never reorder.
class MultimodalDataItem(msgspec.Struct, eq=False, kw_only=True, array_like=True):
    modality: Modality
    hash: int | None = None
    pad_value: int | None = None
    offsets: list[tuple[int, int]] | None = None
    # Inline feature tensor (pixels / mel features / ...). Exactly one of
    # ``feature`` / ``feature_shm`` is set while the payload is in flight;
    # msgpack cannot union two hook-decoded types in one field, so the SHM
    # handle gets its own slot instead of overloading ``feature``.
    feature: torch.Tensor | None = None
    feature_shm: ShmTensorHandle | None = None
    model_specific_data: dict[str, torch.Tensor] = {}
    # Encoder output for this item, populated on first encoder pass and reused
    # across chunked-prefill iterations of the owning request. Lifetime is
    # tied to the request: when the request finishes the item is GC'd and
    # these tensors are released. ``encoded_deepstack`` is set only for
    # deepstack-enabled modalities. Scheduler-local: always None on the wire.
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

    def __post_init__(self) -> None:
        # Compatibility shim for callers that predate the feature/feature_shm
        # split and pass a SHM handle via ``feature``: re-slot it so the
        # transport fields stay single-typed.
        if isinstance(self.feature, ShmTensorHandle):
            self.feature_shm = self.feature
            self.feature = None

    def __getattr__(self, name: str):
        # msgspec structs have no __dict__; read the declared field directly.
        # Guard underscored names so pickle/copy protocol probes fall through.
        if not name.startswith("_"):
            try:
                model_specific_data = object.__getattribute__(
                    self, "model_specific_data"
                )
            except AttributeError:
                model_specific_data = None
            if model_specific_data is not None and name in model_specific_data:
                return model_specific_data[name]
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
            if self.feature_shm is not None:
                raise ValueError(
                    "SHM-backed multimodal items must carry content hash or "
                    "pad_value before TokenSpeed consumes them"
                )
            self.hash = hash_feature(self.feature)
        if self.hash is None:
            raise RuntimeError("Failed to resolve multimodal item hash.")

    def set_pad_value(self):
        if self.pad_value is not None:
            return
        self.ensure_hash()
        modality_offset = _modality_pad_tag(self.modality) * _MM_PAD_HASH_SLOTS
        self.pad_value = (
            _MM_PAD_BASE + modality_offset + (self.hash % _MM_PAD_HASH_SLOTS)
        )

    def is_modality(self, modality: Modality) -> bool:
        return self.modality == modality


class MultimodalInputs(msgspec.Struct, eq=False, kw_only=True, array_like=True):
    mm_items: list[MultimodalDataItem]
    im_token_id: int | None = None
    video_token_id: int | None = None
    mrope_positions: torch.Tensor | None = None
    mrope_position_delta: torch.Tensor | None = None
    mrope_position_delta_scalar: int | None = None
    # Scheduler-local expansion cache; always None on the wire.
    mrope_position_delta_repeated_cache: torch.Tensor | None = None

    def ensure_pad_values(self) -> None:
        for item in self.mm_items:
            item.set_pad_value()

    def publish_shm_features(self) -> None:
        for item in self.mm_items:
            if isinstance(item.feature, torch.Tensor):
                item.feature_shm = ShmTensorHandle.publish(item.feature)
                item.feature = None

    def attach_shm_features(self) -> None:
        """Open every pending handle before peers may consume and unlink it."""
        for item in self.mm_items:
            if item.feature_shm is not None:
                item.feature_shm.attach()

    def release_shm_features(self) -> None:
        for item in self.mm_items:
            if item.feature_shm is not None:
                item.feature_shm.release()
                item.feature_shm = None

    def has_pending_shm_features(self) -> bool:
        return any(item.feature_shm is not None for item in self.mm_items)


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


def multimodal_context_for_forward(forward_op, rid_to_state):
    """Assemble the batch's :class:`MultimodalForwardContext` from per-request
    state.

    Args:
        forward_op: The forward op about to dispatch; supplies the batch's
            request ids and extend geometry.
        rid_to_state: The output processor's request-id -> state map, whose
            states carry each request's ``multimodal_inputs``.

    Returns:
        A ``MultimodalForwardContext`` aligned to the batch order, or ``None``
        when no request in the batch carries multimodal inputs.
    """
    num_extends = forward_op.num_extends()
    mm_inputs = []
    has_mm = False
    for index, rid in enumerate(forward_op.request_ids):
        state = rid_to_state.get(rid)
        if state is not None and index < num_extends:
            state.maybe_extend_multimodal_mrope_positions()
        item = getattr(state, "multimodal_inputs", None) if state else None
        mm_inputs.append(item)
        has_mm = has_mm or item is not None
    if not has_mm:
        return None

    return MultimodalForwardContext(
        mm_inputs=mm_inputs,
        extend_prefix_lens=list(forward_op.extend_prefix_lens),
        extend_seq_lens=list(forward_op.input_lengths[:num_extends]),
    )
