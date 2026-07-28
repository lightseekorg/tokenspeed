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

"""Model-neutral FlatKV prefill/decode metadata contract.

Flat cache groups share a page-ID namespace and bind to aliased raw slabs.
Prefill and decode allocate their pages independently, so each side publishes a
typed layout and a per-request page manifest. The existing Mooncake backend
then transfers each group's selected pages across its bound raw slabs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

FLATKV_PD_PROTOCOL_VERSION = 1
MAX_FLATKV_LAYOUT_WIRE_BYTES = 256 << 10
MAX_FLATKV_MANIFEST_WIRE_BYTES = 2 << 20
MAX_FLATKV_GROUPS = 64
MAX_FLATKV_PHYSICAL_SLOTS = 256
MAX_FLATKV_MANIFEST_PAGES = 1 << 15

_UINT64_LIMIT = 1 << 64
_FAMILIES = frozenset(("history", "state"))
_TRANSFER_POLICIES = frozenset(("full_suffix", "latest_snapshot"))
_PEER_LAYOUT_KEYS = frozenset(("version", "layout_fingerprint", "num_pages_with_null"))
_MANIFEST_KEYS = frozenset(("version", "prefix_len", "prompt_len", "groups"))
_MANIFEST_GROUP_KEYS = frozenset(("group_id", "page_ids"))

Family = Literal["history", "state"]
TransferPolicy = Literal["full_suffix", "latest_snapshot"]


class FlatKVPDProtocolError(ValueError):
    """A peer, descriptor, or scheduler violated the FlatKV PD contract."""


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FlatKVPDProtocolError(
            f"{name} must be an integer >= {minimum}, got {value!r}"
        )
    return value


def _string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise FlatKVPDProtocolError(
            f"{name} must be a non-empty bounded string without control characters"
        )
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise FlatKVPDProtocolError(f"{name} is not valid Unicode") from exc
    if len(encoded) > 256 or any(
        ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise FlatKVPDProtocolError(
            f"{name} must be a non-empty bounded string without control characters"
        )
    return value


def _choice(name: str, value: object, choices: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise FlatKVPDProtocolError(
            f"{name} must be one of {sorted(choices)}, got {value!r}"
        )
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FlatKVPDProtocolError(message)


def _sequence(
    value: object, *, name: str, maximum: int, nonempty: bool = True
) -> tuple[Any, ...]:
    if not isinstance(value, (tuple, list)):
        raise FlatKVPDProtocolError(f"{name} must be a tuple or list")
    minimum = 1 if nonempty else 0
    if not minimum <= len(value) <= maximum:
        raise FlatKVPDProtocolError(f"{name} must contain {minimum}..{maximum} entries")
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
        raise FlatKVPDProtocolError(
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
        raise FlatKVPDProtocolError(f"{name} cannot be encoded") from exc
    if len(result) > maximum:
        raise FlatKVPDProtocolError(f"{name} exceeds {maximum} wire bytes")
    return result


def _decode(raw: bytes, *, name: str, maximum: int) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > maximum:
        raise FlatKVPDProtocolError(f"{name} payload must be 1..{maximum} bytes")

    def object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FlatKVPDProtocolError(
                    f"{name} JSON contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise FlatKVPDProtocolError(f"{name} JSON contains non-finite number {value}")

    try:
        value = json.loads(
            raw.decode(),
            object_pairs_hook=object_hook,
            parse_constant=reject_constant,
        )
    except FlatKVPDProtocolError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
        OverflowError,
    ) as exc:
        raise FlatKVPDProtocolError(f"invalid {name} JSON") from exc
    if not isinstance(value, dict):
        raise FlatKVPDProtocolError(f"{name} payload must be a JSON object")
    return value


def _canonical_round_trip(
    raw: bytes, value: object, *, name: str, maximum: int
) -> None:
    if raw != _encode(asdict(value), name=name, maximum=maximum):  # type: ignore[arg-type]
        raise FlatKVPDProtocolError(f"{name} must use canonical JSON encoding")


@dataclass(frozen=True, slots=True)
class FlatKVPDGroup:
    group_id: str
    family: Family
    transfer_policy: TransferPolicy
    physical_slots: tuple[int, ...]

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
            maximum=MAX_FLATKV_PHYSICAL_SLOTS,
            ordered_unique=True,
        )


@dataclass(frozen=True, slots=True)
class FlatKVPDSLabRegistration:
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
        raise FlatKVPDProtocolError(
            "layout_fingerprint must be a 64-character lowercase SHA-256 hex string"
        )
    return value


@dataclass(frozen=True, slots=True)
class FlatKVPDPeerLayout:
    """Peer-local fields that are not committed by the semantic fingerprint."""

    version: int
    layout_fingerprint: str
    num_pages_with_null: int

    def __post_init__(self) -> None:
        if _integer("layout version", self.version, 1) != FLATKV_PD_PROTOCOL_VERSION:
            raise FlatKVPDProtocolError(
                f"unsupported FlatKV layout version {self.version}"
            )
        _layout_fingerprint(self.layout_fingerprint)
        _integer("num_pages_with_null", self.num_pages_with_null, 2)

    def to_wire_bytes(self) -> bytes:
        return _encode(
            asdict(self),
            name="FlatKV peer layout",
            maximum=MAX_FLATKV_LAYOUT_WIRE_BYTES,
        )

    @classmethod
    def from_wire_bytes(cls, raw: bytes) -> "FlatKVPDPeerLayout":
        payload = _decode(
            raw, name="FlatKV peer layout", maximum=MAX_FLATKV_LAYOUT_WIRE_BYTES
        )
        _exact_keys(payload, _PEER_LAYOUT_KEYS, "FlatKV peer layout")
        layout = cls(
            version=payload["version"],
            layout_fingerprint=payload["layout_fingerprint"],
            num_pages_with_null=payload["num_pages_with_null"],
        )
        _canonical_round_trip(
            raw,
            layout,
            name="FlatKV peer layout",
            maximum=MAX_FLATKV_LAYOUT_WIRE_BYTES,
        )
        return layout


@dataclass(frozen=True, slots=True)
class FlatKVPDLayout:
    """One local, typed Flat cache ABI plus its local capacity."""

    version: int
    layout_fingerprint: str
    block_size: int
    num_pages_with_null: int
    physical_buffer_ids: tuple[str, ...]
    physical_page_bytes: int
    groups: tuple[FlatKVPDGroup, ...]

    def __post_init__(self) -> None:
        if _integer("layout version", self.version, 1) != FLATKV_PD_PROTOCOL_VERSION:
            raise FlatKVPDProtocolError(
                f"unsupported FlatKV layout version {self.version}"
            )
        _layout_fingerprint(self.layout_fingerprint)
        _integer("block_size", self.block_size, 1)
        _integer("num_pages_with_null", self.num_pages_with_null, 2)
        _integer("physical_page_bytes", self.physical_page_bytes, 1)
        buffer_ids = _sequence(
            self.physical_buffer_ids,
            name="physical_buffer_ids",
            maximum=MAX_FLATKV_PHYSICAL_SLOTS,
        )
        for buffer_id in buffer_ids:
            _string("physical buffer_id", buffer_id)
        _require(
            len(buffer_ids) == len(set(buffer_ids)),
            "layout contains duplicate physical buffer IDs",
        )
        if self.num_pages_with_null * self.physical_page_bytes >= _UINT64_LIMIT:
            raise FlatKVPDProtocolError("raw slab extent exceeds uint64")
        groups = _sequence(self.groups, name="layout groups", maximum=MAX_FLATKV_GROUPS)
        if not all(isinstance(group, FlatKVPDGroup) for group in groups):
            raise FlatKVPDProtocolError(
                "layout groups must contain FlatKVPDGroup values"
            )
        group_ids = tuple(group.group_id for group in groups)
        if len(group_ids) != len(set(group_ids)):
            raise FlatKVPDProtocolError("layout contains duplicate group IDs")
        covered: set[int] = set()
        for group in groups:
            for slot in group.physical_slots:
                if slot >= self.physical_slot_count:
                    raise FlatKVPDProtocolError(
                        f"group {group.group_id!r} slot {slot} exceeds "
                        f"physical_slot_count={self.physical_slot_count}"
                    )
                covered.add(slot)
        missing = set(range(self.physical_slot_count)) - covered
        if missing:
            raise FlatKVPDProtocolError(
                f"layout leaves physical slots unbound: {sorted(missing)}"
            )

    @property
    def physical_slot_count(self) -> int:
        return len(self.physical_buffer_ids)

    @property
    def peer(self) -> FlatKVPDPeerLayout:
        return FlatKVPDPeerLayout(
            version=self.version,
            layout_fingerprint=self.layout_fingerprint,
            num_pages_with_null=self.num_pages_with_null,
        )


@dataclass(frozen=True, slots=True)
class FlatKVPDGroupPages:
    group_id: str
    page_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _string("manifest group_id", self.group_id)
        pages = _integer_sequence(
            self.page_ids,
            name=f"manifest group {self.group_id!r} page_ids",
            maximum=MAX_FLATKV_MANIFEST_PAGES,
            minimum=1,
        )
        _require(
            len(pages) == len(set(pages)),
            f"manifest group {self.group_id!r} repeats a page ID",
        )


@dataclass(frozen=True, slots=True)
class FlatKVPDPageManifest:
    groups: tuple[FlatKVPDGroupPages, ...]
    prefix_len: int
    prompt_len: int
    version: int = FLATKV_PD_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if _integer("manifest version", self.version, 1) != FLATKV_PD_PROTOCOL_VERSION:
            raise FlatKVPDProtocolError(
                f"unsupported FlatKV manifest version {self.version}"
            )
        _integer("manifest prefix_len", self.prefix_len)
        _integer("manifest prompt_len", self.prompt_len, 1)
        if self.prefix_len >= self.prompt_len:
            raise FlatKVPDProtocolError("manifest requires prefix_len < prompt_len")
        groups = _sequence(
            self.groups, name="manifest groups", maximum=MAX_FLATKV_GROUPS
        )
        if not all(isinstance(group, FlatKVPDGroupPages) for group in groups):
            raise FlatKVPDProtocolError(
                "manifest groups must contain FlatKVPDGroupPages values"
            )
        group_ids = tuple(group.group_id for group in groups)
        if len(group_ids) != len(set(group_ids)):
            raise FlatKVPDProtocolError("manifest contains duplicate group IDs")
        pages = [page for group in groups for page in group.page_ids]
        if len(pages) > MAX_FLATKV_MANIFEST_PAGES:
            raise FlatKVPDProtocolError("manifest contains too many pages")
        if len(pages) != len(set(pages)):
            raise FlatKVPDProtocolError("manifest repeats a globally owned page ID")

    def to_wire_bytes(self) -> bytes:
        return _encode(
            asdict(self),
            name="FlatKV manifest",
            maximum=MAX_FLATKV_MANIFEST_WIRE_BYTES,
        )

    @classmethod
    def from_wire_bytes(cls, raw: bytes) -> "FlatKVPDPageManifest":
        payload = _decode(
            raw, name="FlatKV manifest", maximum=MAX_FLATKV_MANIFEST_WIRE_BYTES
        )
        _exact_keys(payload, _MANIFEST_KEYS, "FlatKV manifest")
        groups = []
        total = 0
        for position, group in enumerate(
            _sequence(
                payload["groups"],
                name="FlatKV manifest groups",
                maximum=MAX_FLATKV_GROUPS,
            )
        ):
            if not isinstance(group, dict):
                raise FlatKVPDProtocolError(
                    f"FlatKV manifest group {position} must be an object"
                )
            _exact_keys(
                group, _MANIFEST_GROUP_KEYS, f"FlatKV manifest group {position}"
            )
            pages = _sequence(
                group["page_ids"],
                name=f"FlatKV manifest group {position} page_ids",
                maximum=MAX_FLATKV_MANIFEST_PAGES,
            )
            total += len(pages)
            if total > MAX_FLATKV_MANIFEST_PAGES:
                raise FlatKVPDProtocolError("manifest contains too many pages")
            groups.append(
                FlatKVPDGroupPages(
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
            name="FlatKV manifest",
            maximum=MAX_FLATKV_MANIFEST_WIRE_BYTES,
        )
        return manifest


def _logical_slots(
    policy: TransferPolicy, prefix_len: int, prompt_len: int, block_size: int
) -> tuple[int, ...]:
    if policy == "full_suffix":
        begin = prefix_len // block_size
        end = (prompt_len + block_size - 1) // block_size
        count = end - begin
        if count > MAX_FLATKV_MANIFEST_PAGES:
            raise FlatKVPDProtocolError(
                "manifest logical suffix exceeds the page limit"
            )
        return tuple(range(begin, end))
    if policy == "latest_snapshot":
        return ((prompt_len - 1) // block_size,)
    raise FlatKVPDProtocolError(f"unsupported transfer policy {policy!r}")


def validate_flatkv_peer_layout(
    layout: FlatKVPDLayout, peer_layout: FlatKVPDPeerLayout
) -> None:
    if not isinstance(layout, FlatKVPDLayout) or not isinstance(
        peer_layout, FlatKVPDPeerLayout
    ):
        raise FlatKVPDProtocolError(
            "layout compatibility requires local and peer FlatKV layouts"
        )
    if layout.version != peer_layout.version:
        raise FlatKVPDProtocolError("FlatKV P/D layout ABI mismatch: version")
    if layout.layout_fingerprint != peer_layout.layout_fingerprint:
        raise FlatKVPDProtocolError(
            "FlatKV P/D layout ABI mismatch: layout_fingerprint"
        )


def validate_flatkv_manifest(
    manifest: FlatKVPDPageManifest,
    *,
    layout: FlatKVPDLayout,
    num_pages_with_null: int,
    peer: str,
) -> None:
    if not isinstance(manifest, FlatKVPDPageManifest):
        raise FlatKVPDProtocolError(f"{peer} manifest has the wrong type")
    capacity = _integer(f"{peer} num_pages_with_null", num_pages_with_null, 2)
    expected = tuple(group.group_id for group in layout.groups)
    actual = tuple(group.group_id for group in manifest.groups)
    if actual != expected:
        raise FlatKVPDProtocolError(
            f"{peer} manifest group order disagrees with layout"
        )
    if manifest.prefix_len % layout.block_size:
        raise FlatKVPDProtocolError(f"{peer} manifest prefix_len is not page aligned")
    for group, layout_group in zip(manifest.groups, layout.groups, strict=True):
        required = _logical_slots(
            layout_group.transfer_policy,
            manifest.prefix_len,
            manifest.prompt_len,
            layout.block_size,
        )
        if len(group.page_ids) != len(required):
            raise FlatKVPDProtocolError(
                f"{peer} manifest group {group.group_id!r} page count disagrees "
                "with its transfer policy"
            )
        if any(page >= capacity for page in group.page_ids):
            raise FlatKVPDProtocolError(
                f"{peer} manifest group {group.group_id!r} has an out-of-bounds page"
            )


def validate_flatkv_manifest_pair(
    src_manifest: FlatKVPDPageManifest,
    dst_manifest: FlatKVPDPageManifest,
    layout: FlatKVPDLayout,
    *,
    dst_num_pages_with_null: int,
) -> None:
    validate_flatkv_manifest(
        src_manifest,
        layout=layout,
        num_pages_with_null=layout.num_pages_with_null,
        peer="source",
    )
    validate_flatkv_manifest(
        dst_manifest,
        layout=layout,
        num_pages_with_null=dst_num_pages_with_null,
        peer="destination",
    )
    if (
        src_manifest.prefix_len != dst_manifest.prefix_len
        or src_manifest.prompt_len != dst_manifest.prompt_len
    ):
        raise FlatKVPDProtocolError(
            "source/destination prefix_len or prompt_len disagree"
        )


def flatkv_manifest_page_ids(
    manifest: FlatKVPDPageManifest,
    *,
    layout: FlatKVPDLayout,
    num_pages_with_null: int | None = None,
    peer: str = "local",
) -> tuple[int, ...]:
    """Flatten a validated manifest into the legacy Mooncake page-vector order."""
    capacity = (
        layout.num_pages_with_null
        if num_pages_with_null is None
        else num_pages_with_null
    )
    validate_flatkv_manifest(
        manifest,
        layout=layout,
        num_pages_with_null=capacity,
        peer=peer,
    )
    return tuple(page_id for group in manifest.groups for page_id in group.page_ids)


def build_flatkv_page_manifest(
    forward_op: object,
    *,
    layout: FlatKVPDLayout,
    request_row: int,
    prefix_len: int,
    prompt_len: int,
) -> FlatKVPDPageManifest:
    """Select each group's pages according to its explicit transfer policy."""
    if not isinstance(layout, FlatKVPDLayout):
        raise FlatKVPDProtocolError("layout must be a FlatKVPDLayout")
    request_row = _integer("request_row", request_row)
    prefix_len = _integer("prefix_len", prefix_len)
    prompt_len = _integer("prompt_len", prompt_len, 1)
    if prefix_len >= prompt_len:
        raise FlatKVPDProtocolError("FlatKV PD requires prefix_len < prompt_len")
    if prefix_len % layout.block_size:
        raise FlatKVPDProtocolError("FlatKV PD prefix_len must be page aligned")
    arrays_fn = getattr(forward_op, "flat_block_tables_arrays", None)
    if not callable(arrays_fn):
        raise FlatKVPDProtocolError("FlatKV PD requires flat_block_tables_arrays()")
    mapping = arrays_fn()
    if not isinstance(mapping, Mapping):
        raise FlatKVPDProtocolError("flat_block_tables_arrays() must return a mapping")
    actual_ids = tuple(mapping)
    for group_id in actual_ids:
        _string("scheduler group_id", group_id)
    _require(
        len(actual_ids) == len(set(actual_ids)),
        "scheduler returned duplicate group IDs",
    )
    expected_ids = tuple(group.group_id for group in layout.groups)
    if set(actual_ids) != set(expected_ids):
        raise FlatKVPDProtocolError(
            "scheduler group IDs disagree with the FlatKV layout: "
            f"missing={sorted(set(expected_ids) - set(actual_ids))}, "
            f"extra={sorted(set(actual_ids) - set(expected_ids))}"
        )

    groups: list[FlatKVPDGroupPages] = []
    inspected: list[tuple[str, Any, frozenset[int]]] = []
    for layout_group in layout.groups:
        table = mapping[layout_group.group_id]
        if (
            getattr(table, "ndim", None) != 2
            or getattr(getattr(table, "dtype", None), "kind", None) != "i"
            or getattr(getattr(table, "dtype", None), "itemsize", None) not in (4, 8)
        ):
            raise FlatKVPDProtocolError(
                f"table {layout_group.group_id!r} must be a 2-D int32/int64 array"
            )
        if request_row >= table.shape[0]:
            raise FlatKVPDProtocolError(
                f"request_row exceeds table {layout_group.group_id!r}"
            )
        logical_slots = _logical_slots(
            layout_group.transfer_policy,
            prefix_len,
            prompt_len,
            layout.block_size,
        )
        if logical_slots[-1] >= table.shape[1]:
            raise FlatKVPDProtocolError(
                f"table {layout_group.group_id!r} misses logical slot "
                f"{logical_slots[-1]}"
            )
        page_ids = tuple(
            int(table[request_row, logical_slot]) for logical_slot in logical_slots
        )
        for logical_slot, page_id in zip(logical_slots, page_ids, strict=True):
            if page_id <= 0 or page_id >= layout.num_pages_with_null:
                raise FlatKVPDProtocolError(
                    f"table {layout_group.group_id!r} logical slot {logical_slot} "
                    f"has invalid page ID {page_id}"
                )
        groups.append(
            FlatKVPDGroupPages(
                layout_group.group_id,
                page_ids,
            )
        )
        inspected.append((layout_group.group_id, table, frozenset(logical_slots)))
    manifest = FlatKVPDPageManifest(
        groups=tuple(groups), prefix_len=prefix_len, prompt_len=prompt_len
    )

    selected = {
        page_id: (group.group_id, logical_slot)
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
            if (
                logical_slot not in selected_slots
                and 0 < page_id < layout.num_pages_with_null
                and page_id in selected
            ):
                owner_group, owner_slot = selected[page_id]
                raise FlatKVPDProtocolError(
                    f"selected page {page_id} for {owner_group!r} slot "
                    f"{owner_slot} aliases live unselected {group_id!r} slot "
                    f"{logical_slot}"
                )
    return manifest


def validate_flatkv_slab_registrations(
    registrations: object,
    *,
    layout: FlatKVPDLayout,
    peer: str,
    num_pages_with_null: int | None = None,
) -> tuple[FlatKVPDSLabRegistration, ...]:
    _string("registration peer", peer)
    values = _sequence(
        registrations,
        name=f"{peer} slab registrations",
        maximum=MAX_FLATKV_PHYSICAL_SLOTS,
    )
    if len(values) != layout.physical_slot_count:
        raise FlatKVPDProtocolError(
            f"{peer} slab registration count disagrees with layout"
        )
    if not all(isinstance(value, FlatKVPDSLabRegistration) for value in values):
        raise FlatKVPDProtocolError(
            f"{peer} registrations must contain FlatKVPDSLabRegistration values"
        )
    result = tuple(values)
    if tuple(value.physical_slot for value in result) != tuple(
        range(layout.physical_slot_count)
    ):
        raise FlatKVPDProtocolError(
            f"{peer} registrations must use exact physical-slot order"
        )
    if tuple(value.buffer_id for value in result) != layout.physical_buffer_ids:
        raise FlatKVPDProtocolError(
            f"{peer} registration buffer IDs disagree with layout"
        )
    capacity = (
        layout.num_pages_with_null
        if num_pages_with_null is None
        else _integer(f"{peer} num_pages_with_null", num_pages_with_null, 2)
    )
    extent = capacity * layout.physical_page_bytes
    if extent >= _UINT64_LIMIT:
        raise FlatKVPDProtocolError(f"{peer} registered extent exceeds uint64")
    if any(value.length != extent for value in result):
        raise FlatKVPDProtocolError(
            f"{peer} registered extent disagrees with layout capacity"
        )
    by_address = sorted(result, key=lambda value: value.base_addr)
    if any(
        right.base_addr < left.base_addr + left.length
        for left, right in zip(by_address, by_address[1:])
    ):
        raise FlatKVPDProtocolError(f"{peer} registered slab extents overlap")
    return result
