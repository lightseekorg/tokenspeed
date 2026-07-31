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

"""Model-neutral Paged cache prefill/decode metadata contract.

Paged cache groups share a page-ID namespace and bind to aliased raw slabs.
Prefill and decode allocate their pages independently, so each side publishes a
typed layout and a per-request page manifest. The existing Mooncake backend
then transfers each group's selected pages across its bound raw slabs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

CACHE_PD_PROTOCOL_VERSION = 1
MAX_CACHE_LAYOUT_WIRE_BYTES = 256 << 10
MAX_CACHE_MANIFEST_WIRE_BYTES = 2 << 20
MAX_CACHE_GROUPS = 64
MAX_CACHE_PHYSICAL_SLOTS = 256
MAX_CACHE_MANIFEST_PAGES = 1 << 15

_UINT64_LIMIT = 1 << 64
_FAMILIES = frozenset(("history", "state"))
_TRANSFER_POLICIES = frozenset(("full_suffix", "latest_snapshot"))
_PEER_LAYOUT_KEYS = frozenset(("version", "layout_fingerprint", "num_pages_with_null"))
_MANIFEST_KEYS = frozenset(("version", "prefix_len", "prompt_len", "groups"))
_MANIFEST_GROUP_KEYS = frozenset(("group_id", "page_ids"))

Family = Literal["history", "state"]
TransferPolicy = Literal["full_suffix", "latest_snapshot"]


class CachePDProtocolError(ValueError):
    """A peer, descriptor, or scheduler violated the Paged cache PD contract."""


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CachePDProtocolError(
            f"{name} must be an integer >= {minimum}, got {value!r}"
        )
    return value


def _string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise CachePDProtocolError(
            f"{name} must be a non-empty bounded string without control characters"
        )
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CachePDProtocolError(f"{name} is not valid Unicode") from exc
    if len(encoded) > 256 or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise CachePDProtocolError(
            f"{name} must be a non-empty bounded string without control characters"
        )
    return value


def _choice(name: str, value: object, choices: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise CachePDProtocolError(
            f"{name} must be one of {sorted(choices)}, got {value!r}"
        )
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CachePDProtocolError(message)


def _sequence(
    value: object, *, name: str, maximum: int, nonempty: bool = True
) -> tuple[Any, ...]:
    if not isinstance(value, (tuple, list)):
        raise CachePDProtocolError(f"{name} must be a tuple or list")
    minimum = 1 if nonempty else 0
    if not minimum <= len(value) <= maximum:
        raise CachePDProtocolError(f"{name} must contain {minimum}..{maximum} entries")
    return tuple(value)


def _integer_sequence(
    value: object,
    *,
    name: str,
    maximum: int,
    minimum: int = 0,
    ordered_unique: bool = False,
) -> tuple[int, ...]:
    result = tuple(
        _integer(f"{name} entry", item, minimum)
        for item in _sequence(value, name=name, maximum=maximum)
    )
    if ordered_unique:
        _require(
            result == tuple(sorted(set(result))),
            f"{name} must be sorted and unique",
        )
    return result


def _exact_keys(value: dict[str, Any], expected: frozenset[str], name: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise CachePDProtocolError(
            f"{name} schema keys disagree: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _encode(payload: dict[str, Any], *, name: str, maximum: int) -> bytes:
    try:
        result = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError, OverflowError) as exc:
        raise CachePDProtocolError(f"{name} cannot be encoded") from exc
    if len(result) > maximum:
        raise CachePDProtocolError(f"{name} exceeds {maximum} wire bytes")
    return result


def _decode(raw: bytes, *, name: str, maximum: int) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > maximum:
        raise CachePDProtocolError(f"{name} payload must be 1..{maximum} bytes")

    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CachePDProtocolError(
                    f"{name} JSON contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise CachePDProtocolError(f"{name} JSON contains non-finite number {value}")

    try:
        value = json.loads(
            raw.decode(),
            object_pairs_hook=object_hook,
            parse_constant=reject_constant,
        )
    except CachePDProtocolError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
        OverflowError,
    ) as exc:
        raise CachePDProtocolError(f"invalid {name} JSON") from exc
    if not isinstance(value, dict):
        raise CachePDProtocolError(f"{name} payload must be a JSON object")
    return value


def _canonical_round_trip(
    raw: bytes, value: object, *, name: str, maximum: int
) -> None:
    if raw != _encode(asdict(value), name=name, maximum=maximum):  # type: ignore[arg-type]
        raise CachePDProtocolError(f"{name} must use canonical JSON encoding")


@dataclass(frozen=True, slots=True)
class CachePDTransferSegment:
    """One strided field copied for every selected page in a cache group."""

    physical_slot: int
    field_id: str
    dtype: str
    page_zero_offset: int
    page_stride_bytes: int
    payload_bytes: int

    def __post_init__(self) -> None:
        _integer("segment physical_slot", self.physical_slot)
        _string("segment field_id", self.field_id)
        _string("segment dtype", self.dtype)
        _integer("segment page_zero_offset", self.page_zero_offset)
        _integer("segment page_stride_bytes", self.page_stride_bytes, 1)
        _integer("segment payload_bytes", self.payload_bytes, 1)
        _require(
            self.payload_bytes <= self.page_stride_bytes,
            f"segment {self.field_id!r} payload exceeds its page stride",
        )


@dataclass(frozen=True, slots=True)
class CachePDGroup:
    group_id: str
    family: Family
    transfer_policy: TransferPolicy
    physical_slots: tuple[int, ...]
    cache_blocks_per_lcm_block: int = 1
    transfer_segments: tuple[CachePDTransferSegment, ...] = ()

    def __post_init__(self) -> None:
        _string("group_id", self.group_id)
        _choice(f"group {self.group_id!r} family", self.family, _FAMILIES)
        _choice(
            f"group {self.group_id!r} transfer_policy",
            self.transfer_policy,
            _TRANSFER_POLICIES,
        )
        _integer_sequence(
            self.physical_slots,
            name=f"group {self.group_id!r} physical_slots",
            maximum=MAX_CACHE_PHYSICAL_SLOTS,
            ordered_unique=True,
        )
        _integer(
            f"group {self.group_id!r} cache_blocks_per_lcm_block",
            self.cache_blocks_per_lcm_block,
            1,
        )
        segments = _sequence(
            self.transfer_segments,
            name=f"group {self.group_id!r} transfer_segments",
            maximum=MAX_CACHE_PHYSICAL_SLOTS,
            nonempty=False,
        )
        if not all(isinstance(segment, CachePDTransferSegment) for segment in segments):
            raise CachePDProtocolError(
                f"group {self.group_id!r} transfer_segments must contain "
                "CachePDTransferSegment values"
            )
        field_ids = tuple(segment.field_id for segment in segments)
        _require(
            len(field_ids) == len(set(field_ids)),
            f"group {self.group_id!r} repeats a transfer field",
        )


@dataclass(frozen=True, slots=True)
class CachePDSlabRegistration:
    physical_slot: int
    buffer_id: str
    base_addr: int
    length: int

    def __post_init__(self) -> None:
        _integer("slab physical_slot", self.physical_slot)
        _string("slab buffer_id", self.buffer_id)
        _integer("slab base_addr", self.base_addr, 1)
        _integer("slab length", self.length, 1)
        _require(self.base_addr < _UINT64_LIMIT, "slab base_addr exceeds uint64")
        _require(
            self.base_addr + self.length <= _UINT64_LIMIT,
            "slab registered extent exceeds uint64",
        )


def _layout_fingerprint(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CachePDProtocolError(
            "layout_fingerprint must be a 64-character lowercase SHA-256 hex string"
        )
    return value


@dataclass(frozen=True, slots=True)
class CachePDPeerLayout:
    """Peer-local fields that are not committed by the semantic fingerprint."""

    version: int
    layout_fingerprint: str
    num_pages_with_null: int

    def __post_init__(self) -> None:
        if _integer("layout version", self.version, 1) != CACHE_PD_PROTOCOL_VERSION:
            raise CachePDProtocolError(
                f"unsupported Paged cache layout version {self.version}"
            )
        _layout_fingerprint(self.layout_fingerprint)
        _integer("num_pages_with_null", self.num_pages_with_null, 2)

    def to_wire_bytes(self) -> bytes:
        return _encode(
            asdict(self),
            name="Paged cache peer layout",
            maximum=MAX_CACHE_LAYOUT_WIRE_BYTES,
        )

    @classmethod
    def from_wire_bytes(cls, raw: bytes) -> "CachePDPeerLayout":
        payload = _decode(
            raw, name="Paged cache peer layout", maximum=MAX_CACHE_LAYOUT_WIRE_BYTES
        )
        _exact_keys(payload, _PEER_LAYOUT_KEYS, "Paged cache peer layout")
        layout = cls(
            version=payload["version"],
            layout_fingerprint=payload["layout_fingerprint"],
            num_pages_with_null=payload["num_pages_with_null"],
        )
        _canonical_round_trip(
            raw,
            layout,
            name="Paged cache peer layout",
            maximum=MAX_CACHE_LAYOUT_WIRE_BYTES,
        )
        return layout


@dataclass(frozen=True, slots=True)
class CachePDLayout:
    """One local, typed Paged cache ABI plus its local capacity."""

    version: int
    layout_fingerprint: str
    block_size: int
    num_pages_with_null: int
    physical_buffer_ids: tuple[str, ...]
    physical_page_bytes: int
    groups: tuple[CachePDGroup, ...]

    def __post_init__(self) -> None:
        if _integer("layout version", self.version, 1) != CACHE_PD_PROTOCOL_VERSION:
            raise CachePDProtocolError(
                f"unsupported Paged cache layout version {self.version}"
            )
        _layout_fingerprint(self.layout_fingerprint)
        _integer("block_size", self.block_size, 1)
        _integer("num_pages_with_null", self.num_pages_with_null, 2)
        _integer("physical_page_bytes", self.physical_page_bytes, 1)
        buffer_ids = _sequence(
            self.physical_buffer_ids,
            name="physical_buffer_ids",
            maximum=MAX_CACHE_PHYSICAL_SLOTS,
        )
        for buffer_id in buffer_ids:
            _string("physical buffer_id", buffer_id)
        _require(
            len(buffer_ids) == len(set(buffer_ids)),
            "layout contains duplicate physical buffer IDs",
        )
        if self.num_pages_with_null * self.physical_page_bytes >= _UINT64_LIMIT:
            raise CachePDProtocolError("raw slab extent exceeds uint64")
        groups = _sequence(self.groups, name="layout groups", maximum=MAX_CACHE_GROUPS)
        if not all(isinstance(group, CachePDGroup) for group in groups):
            raise CachePDProtocolError("layout groups must contain CachePDGroup values")
        group_ids = tuple(group.group_id for group in groups)
        if len(group_ids) != len(set(group_ids)):
            raise CachePDProtocolError("layout contains duplicate group IDs")
        covered: set[int] = set()
        for group in groups:
            for slot in group.physical_slots:
                if slot >= self.physical_slot_count:
                    raise CachePDProtocolError(
                        f"group {group.group_id!r} slot {slot} exceeds "
                        f"physical_slot_count={self.physical_slot_count}"
                    )
                covered.add(slot)
            for segment in group.transfer_segments:
                if segment.physical_slot >= self.physical_slot_count:
                    raise CachePDProtocolError(
                        f"group {group.group_id!r} segment "
                        f"{segment.field_id!r} slot {segment.physical_slot} "
                        f"exceeds physical_slot_count={self.physical_slot_count}"
                    )
                registered_extent = self.num_pages_with_null * self.physical_page_bytes
                group_page_count = (
                    1
                    + (self.num_pages_with_null - 1) * group.cache_blocks_per_lcm_block
                )
                segment_extent = (
                    segment.page_zero_offset
                    + (group_page_count - 1) * segment.page_stride_bytes
                    + segment.payload_bytes
                )
                if segment_extent > registered_extent:
                    raise CachePDProtocolError(
                        f"group {group.group_id!r} segment "
                        f"{segment.field_id!r} exceeds its registered buffer"
                    )
                covered.add(segment.physical_slot)
        missing = set(range(self.physical_slot_count)) - covered
        if missing:
            raise CachePDProtocolError(
                f"layout leaves physical slots unbound: {sorted(missing)}"
            )

    @property
    def physical_slot_count(self) -> int:
        return len(self.physical_buffer_ids)

    @property
    def peer(self) -> CachePDPeerLayout:
        return CachePDPeerLayout(
            version=self.version,
            layout_fingerprint=self.layout_fingerprint,
            num_pages_with_null=self.num_pages_with_null,
        )


@dataclass(frozen=True, slots=True)
class CachePDGroupPages:
    group_id: str
    page_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _string("manifest group_id", self.group_id)
        pages = _integer_sequence(
            self.page_ids,
            name=f"manifest group {self.group_id!r} page_ids",
            maximum=MAX_CACHE_MANIFEST_PAGES,
            minimum=1,
        )
        _require(
            len(pages) == len(set(pages)),
            f"manifest group {self.group_id!r} repeats a page ID",
        )


@dataclass(frozen=True, slots=True)
class CachePDPageManifest:
    groups: tuple[CachePDGroupPages, ...]
    prefix_len: int
    prompt_len: int
    version: int = CACHE_PD_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if _integer("manifest version", self.version, 1) != CACHE_PD_PROTOCOL_VERSION:
            raise CachePDProtocolError(
                f"unsupported Paged cache manifest version {self.version}"
            )
        _integer("manifest prefix_len", self.prefix_len)
        _integer("manifest prompt_len", self.prompt_len, 1)
        if self.prefix_len >= self.prompt_len:
            raise CachePDProtocolError("manifest requires prefix_len < prompt_len")
        groups = _sequence(
            self.groups, name="manifest groups", maximum=MAX_CACHE_GROUPS
        )
        if not all(isinstance(group, CachePDGroupPages) for group in groups):
            raise CachePDProtocolError(
                "manifest groups must contain CachePDGroupPages values"
            )
        group_ids = tuple(group.group_id for group in groups)
        if len(group_ids) != len(set(group_ids)):
            raise CachePDProtocolError("manifest contains duplicate group IDs")
        page_count = sum(len(group.page_ids) for group in groups)
        if page_count > MAX_CACHE_MANIFEST_PAGES:
            raise CachePDProtocolError("manifest contains too many pages")

    def to_wire_bytes(self) -> bytes:
        return _encode(
            asdict(self),
            name="Paged cache manifest",
            maximum=MAX_CACHE_MANIFEST_WIRE_BYTES,
        )

    @classmethod
    def from_wire_bytes(cls, raw: bytes) -> "CachePDPageManifest":
        payload = _decode(
            raw, name="Paged cache manifest", maximum=MAX_CACHE_MANIFEST_WIRE_BYTES
        )
        _exact_keys(payload, _MANIFEST_KEYS, "Paged cache manifest")
        groups = []
        total = 0
        for position, group in enumerate(
            _sequence(
                payload["groups"],
                name="Paged cache manifest groups",
                maximum=MAX_CACHE_GROUPS,
            )
        ):
            if not isinstance(group, dict):
                raise CachePDProtocolError(
                    f"Paged cache manifest group {position} must be an object"
                )
            _exact_keys(
                group, _MANIFEST_GROUP_KEYS, f"Paged cache manifest group {position}"
            )
            pages = _sequence(
                group["page_ids"],
                name=f"Paged cache manifest group {position} page_ids",
                maximum=MAX_CACHE_MANIFEST_PAGES,
            )
            total += len(pages)
            if total > MAX_CACHE_MANIFEST_PAGES:
                raise CachePDProtocolError("manifest contains too many pages")
            groups.append(
                CachePDGroupPages(
                    group["group_id"],
                    tuple(pages),
                )
            )
        manifest = cls(
            groups=tuple(groups),
            prefix_len=payload["prefix_len"],
            prompt_len=payload["prompt_len"],
            version=payload["version"],
        )
        _canonical_round_trip(
            raw,
            manifest,
            name="Paged cache manifest",
            maximum=MAX_CACHE_MANIFEST_WIRE_BYTES,
        )
        return manifest


def _logical_slots(
    policy: TransferPolicy, prefix_len: int, prompt_len: int, block_size: int
) -> tuple[int, ...]:
    if policy == "full_suffix":
        begin = prefix_len // block_size
        end = (prompt_len + block_size - 1) // block_size
        count = end - begin
        if count > MAX_CACHE_MANIFEST_PAGES:
            raise CachePDProtocolError("manifest logical suffix exceeds the page limit")
        return tuple(range(begin, end))
    if policy == "latest_snapshot":
        return ((prompt_len - 1) // block_size,)
    raise CachePDProtocolError(f"unsupported transfer policy {policy!r}")


def _group_page_count(group: CachePDGroup, num_pages_with_null: int) -> int:
    """Resolve one group's child-page capacity from the registered parents."""
    return 1 + (num_pages_with_null - 1) * group.cache_blocks_per_lcm_block


def validate_cache_peer_layout(
    layout: CachePDLayout, peer_layout: CachePDPeerLayout
) -> None:
    if not isinstance(layout, CachePDLayout) or not isinstance(
        peer_layout, CachePDPeerLayout
    ):
        raise CachePDProtocolError(
            "layout compatibility requires local and peer Paged cache layouts"
        )
    if layout.version != peer_layout.version:
        raise CachePDProtocolError("Paged cache P/D layout ABI mismatch: version")
    if layout.layout_fingerprint != peer_layout.layout_fingerprint:
        raise CachePDProtocolError(
            "Paged cache P/D layout ABI mismatch: layout_fingerprint"
        )


def validate_cache_manifest(
    manifest: CachePDPageManifest,
    *,
    layout: CachePDLayout,
    num_pages_with_null: int,
    peer: str,
) -> None:
    if not isinstance(manifest, CachePDPageManifest):
        raise CachePDProtocolError(f"{peer} manifest has the wrong type")
    capacity = _integer(f"{peer} num_pages_with_null", num_pages_with_null, 2)
    expected = tuple(group.group_id for group in layout.groups)
    actual = tuple(group.group_id for group in manifest.groups)
    if actual != expected:
        raise CachePDProtocolError(f"{peer} manifest group order disagrees with layout")
    if manifest.prefix_len % layout.block_size:
        raise CachePDProtocolError(f"{peer} manifest prefix_len is not page aligned")
    for group, layout_group in zip(manifest.groups, layout.groups, strict=True):
        required = _logical_slots(
            layout_group.transfer_policy,
            manifest.prefix_len,
            manifest.prompt_len,
            layout.block_size,
        )
        if len(group.page_ids) != len(required):
            raise CachePDProtocolError(
                f"{peer} manifest group {group.group_id!r} page count disagrees "
                "with its transfer policy"
            )
        group_capacity = _group_page_count(layout_group, capacity)
        if any(page >= group_capacity for page in group.page_ids):
            raise CachePDProtocolError(
                f"{peer} manifest group {group.group_id!r} has an out-of-bounds page"
            )


def validate_cache_manifest_pair(
    src_manifest: CachePDPageManifest,
    dst_manifest: CachePDPageManifest,
    layout: CachePDLayout,
    *,
    dst_num_pages_with_null: int,
) -> None:
    validate_cache_manifest(
        src_manifest,
        layout=layout,
        num_pages_with_null=layout.num_pages_with_null,
        peer="source",
    )
    validate_cache_manifest(
        dst_manifest,
        layout=layout,
        num_pages_with_null=dst_num_pages_with_null,
        peer="destination",
    )
    if (
        src_manifest.prefix_len != dst_manifest.prefix_len
        or src_manifest.prompt_len != dst_manifest.prompt_len
    ):
        raise CachePDProtocolError(
            "source/destination prefix_len or prompt_len disagree"
        )


def cache_manifest_page_ids(
    manifest: CachePDPageManifest,
    *,
    layout: CachePDLayout,
    num_pages_with_null: int | None = None,
    peer: str = "local",
) -> tuple[int, ...]:
    """Flatten a validated manifest into the legacy Mooncake page-vector order."""
    capacity = (
        layout.num_pages_with_null
        if num_pages_with_null is None
        else num_pages_with_null
    )
    validate_cache_manifest(
        manifest,
        layout=layout,
        num_pages_with_null=capacity,
        peer=peer,
    )
    return tuple(page_id for group in manifest.groups for page_id in group.page_ids)


def build_cache_page_manifest(
    forward_op: object,
    *,
    layout: CachePDLayout,
    request_row: int,
    prefix_len: int,
    prompt_len: int,
) -> CachePDPageManifest:
    """Select each group's pages according to its explicit transfer policy."""
    if not isinstance(layout, CachePDLayout):
        raise CachePDProtocolError("layout must be a CachePDLayout")
    request_row = _integer("request_row", request_row)
    prefix_len = _integer("prefix_len", prefix_len)
    prompt_len = _integer("prompt_len", prompt_len, 1)
    if prefix_len >= prompt_len:
        raise CachePDProtocolError("Paged cache PD requires prefix_len < prompt_len")
    if prefix_len % layout.block_size:
        raise CachePDProtocolError("Paged cache PD prefix_len must be page aligned")
    arrays_fn = getattr(forward_op, "block_tables_arrays", None)
    if not callable(arrays_fn):
        raise CachePDProtocolError("Paged cache PD requires block_tables_arrays()")
    mapping = arrays_fn()
    if not isinstance(mapping, Mapping):
        raise CachePDProtocolError("block_tables_arrays() must return a mapping")
    actual_ids = tuple(mapping)
    for group_id in actual_ids:
        _string("scheduler group_id", group_id)
    _require(
        len(actual_ids) == len(set(actual_ids)),
        "scheduler returned duplicate group IDs",
    )
    expected_ids = tuple(group.group_id for group in layout.groups)
    layout_groups_by_id = {group.group_id: group for group in layout.groups}
    if set(actual_ids) != set(expected_ids):
        raise CachePDProtocolError(
            "scheduler group IDs disagree with the Paged cache layout: "
            f"missing={sorted(set(expected_ids) - set(actual_ids))}, "
            f"extra={sorted(set(actual_ids) - set(expected_ids))}"
        )

    groups: list[CachePDGroupPages] = []
    inspected: list[tuple[str, Any, frozenset[int]]] = []
    for layout_group in layout.groups:
        table = mapping[layout_group.group_id]
        if (
            getattr(table, "ndim", None) != 2
            or getattr(getattr(table, "dtype", None), "kind", None) != "i"
            or getattr(getattr(table, "dtype", None), "itemsize", None) not in (4, 8)
        ):
            raise CachePDProtocolError(
                f"table {layout_group.group_id!r} must be a 2-D int32/int64 array"
            )
        if request_row >= table.shape[0]:
            raise CachePDProtocolError(
                f"request_row exceeds table {layout_group.group_id!r}"
            )
        logical_slots = _logical_slots(
            layout_group.transfer_policy,
            prefix_len,
            prompt_len,
            layout.block_size,
        )
        if logical_slots[-1] >= table.shape[1]:
            raise CachePDProtocolError(
                f"table {layout_group.group_id!r} misses logical slot "
                f"{logical_slots[-1]}"
            )
        page_ids = tuple(
            int(table[request_row, logical_slot]) for logical_slot in logical_slots
        )
        for logical_slot, page_id in zip(logical_slots, page_ids, strict=True):
            group_capacity = _group_page_count(layout_group, layout.num_pages_with_null)
            if page_id <= 0 or page_id >= group_capacity:
                raise CachePDProtocolError(
                    f"table {layout_group.group_id!r} logical slot {logical_slot} "
                    f"has invalid page ID {page_id}"
                )
        groups.append(
            CachePDGroupPages(
                layout_group.group_id,
                page_ids,
            )
        )
        inspected.append((layout_group.group_id, table, frozenset(logical_slots)))
    manifest = CachePDPageManifest(
        groups=tuple(groups), prefix_len=prefix_len, prompt_len=prompt_len
    )

    selected = {
        (group.group_id, page_id): logical_slot
        for group, layout_group in zip(manifest.groups, layout.groups, strict=True)
        for page_id, logical_slot in zip(
            group.page_ids,
            _logical_slots(
                layout_group.transfer_policy,
                manifest.prefix_len,
                manifest.prompt_len,
                layout.block_size,
            ),
            strict=True,
        )
    }
    for group_id, table, selected_slots in inspected:
        for logical_slot, raw_page in enumerate(table[request_row]):
            page_id = int(raw_page)
            layout_group = layout_groups_by_id[group_id]
            group_capacity = _group_page_count(layout_group, layout.num_pages_with_null)
            if (
                logical_slot not in selected_slots
                and 0 < page_id < group_capacity
                and (group_id, page_id) in selected
            ):
                raise CachePDProtocolError(
                    f"selected page {page_id} for {group_id!r} slot "
                    f"{selected[(group_id, page_id)]} aliases live unselected slot "
                    f"{logical_slot}"
                )
    return manifest


def validate_cache_slab_registrations(
    registrations: object,
    *,
    layout: CachePDLayout,
    peer: str,
    num_pages_with_null: int | None = None,
) -> tuple[CachePDSlabRegistration, ...]:
    _string("registration peer", peer)
    values = _sequence(
        registrations,
        name=f"{peer} slab registrations",
        maximum=MAX_CACHE_PHYSICAL_SLOTS,
    )
    if len(values) != layout.physical_slot_count:
        raise CachePDProtocolError(
            f"{peer} slab registration count disagrees with layout"
        )
    if not all(isinstance(value, CachePDSlabRegistration) for value in values):
        raise CachePDProtocolError(
            f"{peer} registrations must contain CachePDSlabRegistration values"
        )
    result = tuple(values)
    if tuple(value.physical_slot for value in result) != tuple(
        range(layout.physical_slot_count)
    ):
        raise CachePDProtocolError(
            f"{peer} registrations must use exact physical-slot order"
        )
    if tuple(value.buffer_id for value in result) != layout.physical_buffer_ids:
        raise CachePDProtocolError(
            f"{peer} registration buffer IDs disagree with layout"
        )
    capacity = (
        layout.num_pages_with_null
        if num_pages_with_null is None
        else _integer(f"{peer} num_pages_with_null", num_pages_with_null, 2)
    )
    extent = capacity * layout.physical_page_bytes
    if extent >= _UINT64_LIMIT:
        raise CachePDProtocolError(f"{peer} registered extent exceeds uint64")
    if any(value.length != extent for value in result):
        raise CachePDProtocolError(
            f"{peer} registered extent disagrees with layout capacity"
        )
    by_address = sorted(result, key=lambda value: value.base_addr)
    if any(
        right.base_addr < left.base_addr + left.length
        for left, right in zip(by_address, by_address[1:])
    ):
        raise CachePDProtocolError(f"{peer} registered slab extents overlap")
    return result


def build_lcm_pd_cache_contract(
    *,
    plan: object,
    backing: object,
    group_specs: object,
    field_dtypes: Mapping[str, str],
) -> tuple[CachePDLayout, tuple[CachePDSlabRegistration, ...]]:
    """Describe one LCM arena without copying or flattening its backing."""
    groups = tuple(getattr(plan, "groups", ()))
    fields = tuple(getattr(plan, "fields", ()))
    planes = tuple(getattr(plan, "planes", ()))
    specs = tuple(group_specs)
    plan_group_ids = tuple(group.group_id for group in groups)
    spec_group_ids = tuple(spec.group_id for spec in specs)
    if set(plan_group_ids) != set(spec_group_ids):
        raise CachePDProtocolError(
            "LCM plan and scheduler group IDs disagree: "
            f"missing={sorted(set(plan_group_ids) - set(spec_group_ids))}, "
            f"extra={sorted(set(spec_group_ids) - set(plan_group_ids))}"
        )
    if set(field_dtypes) != {field.field_id for field in fields}:
        raise CachePDProtocolError(
            "LCM field dtype map must contain exactly the planned fields"
        )

    plan_groups = {group.group_id: group for group in groups}
    plan_planes = {plane.plane_id: plane for plane in planes}
    transfer_groups = []
    for spec in specs:
        group = plan_groups[spec.group_id]
        transfer_policy = getattr(spec, "transfer_policy", None)
        if transfer_policy is None:
            raise CachePDProtocolError(
                f"LCM PD group {spec.group_id!r} requires a transfer policy"
            )
        group_fields = tuple(
            field for field in fields if field.group_id == spec.group_id
        )
        if not group_fields:
            raise CachePDProtocolError(
                f"LCM PD group {spec.group_id!r} has no planned fields"
            )
        segments = []
        for field in group_fields:
            plane = plan_planes[field.plane_id]
            segments.append(
                CachePDTransferSegment(
                    physical_slot=0,
                    field_id=field.field_id,
                    dtype=field_dtypes[field.field_id],
                    page_zero_offset=(
                        plane.arena_offset_bytes
                        + plane.bytes_per_lcm_block
                        - field.page_stride_bytes
                        + field.field_offset_bytes
                    ),
                    page_stride_bytes=field.page_stride_bytes,
                    payload_bytes=field.payload_bytes,
                )
            )
        transfer_groups.append(
            CachePDGroup(
                group_id=spec.group_id,
                family=spec.family,
                transfer_policy=transfer_policy,
                physical_slots=(0,),
                cache_blocks_per_lcm_block=group.cache_blocks_per_lcm_block,
                transfer_segments=tuple(segments),
            )
        )

    fingerprint_payload = {
        "schema_version": CACHE_PD_PROTOCOL_VERSION,
        "logical_block_tokens": int(plan.logical_block_tokens),
        "lcm_block_bytes": int(plan.lcm_block_bytes),
        "groups": [
            {
                "order": order,
                "group_id": spec.group_id,
                "family": spec.family,
                "transfer_policy": spec.transfer_policy,
                "retention": spec.retention,
                "sliding_window_tokens": spec.sliding_window_tokens,
                "cache_blocks_per_lcm_block": plan_groups[
                    spec.group_id
                ].cache_blocks_per_lcm_block,
            }
            for order, spec in enumerate(specs)
        ],
        "planes": [
            {
                "plane_id": plane.plane_id,
                "bytes_per_lcm_block": plane.bytes_per_lcm_block,
                "arena_offset_bytes": plane.arena_offset_bytes,
            }
            for plane in planes
        ],
        "fields": [
            {
                "group_id": field.group_id,
                "field_id": field.field_id,
                "plane_id": field.plane_id,
                "shape": field.shape,
                "dtype": field_dtypes[field.field_id],
                "element_size": field.element_size,
                "field_offset_bytes": field.field_offset_bytes,
                "page_stride_bytes": field.page_stride_bytes,
            }
            for field in fields
        ],
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    layout = CachePDLayout(
        version=CACHE_PD_PROTOCOL_VERSION,
        layout_fingerprint=fingerprint,
        block_size=int(plan.logical_block_tokens),
        num_pages_with_null=int(plan.num_lcm_blocks) + 1,
        physical_buffer_ids=("lcm_arena",),
        physical_page_bytes=int(plan.lcm_block_bytes),
        groups=tuple(transfer_groups),
    )

    if (
        getattr(backing, "dtype", None) is None
        or str(backing.dtype) != "torch.uint8"
        or not backing.is_contiguous()
        or backing.storage_offset() != 0
        or backing.data_ptr() != backing.untyped_storage().data_ptr()
        or int(backing.nbytes) != int(plan.arena_bytes)
    ):
        raise CachePDProtocolError(
            "LCM PD backing must be the contiguous uint8 arena owner"
        )
    registrations = (
        CachePDSlabRegistration(
            physical_slot=0,
            buffer_id="lcm_arena",
            base_addr=backing.data_ptr(),
            length=backing.nbytes,
        ),
    )
    return layout, validate_cache_slab_registrations(
        registrations, layout=layout, peer="local"
    )
