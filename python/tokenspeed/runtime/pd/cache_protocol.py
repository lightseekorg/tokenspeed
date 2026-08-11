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

"""Paged cache transfer contract and per-request PD protocol.

The cache recipe owns semantic group specs and the memory planner owns physical
geometry. PD transports those objects directly instead of maintaining a second
flattened layout schema. The same module also owns request page manifests and
peer validation, keeping the lockstep PD wire surface in one place.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldLayout,
    CacheGroupLayout,
    CacheMemoryPlan,
    CachePlaneLayout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    PagedCacheGroupSpec,
    Retention,
    TransferPolicy,
)

MAX_CACHE_CONTRACT_WIRE_BYTES = 256 << 10
MAX_CACHE_MANIFEST_WIRE_BYTES = 2 << 20


class CacheContractError(ValueError):
    """A cache pool, peer, or scheduler violated the runtime cache contract."""


def _dump_wire_json(value: object, *, name: str, maximum: int) -> bytes:
    try:
        result = json.dumps(
            asdict(value),  # type: ignore[arg-type]
            separators=(",", ":"),
        ).encode()
    except (TypeError, ValueError, OverflowError) as exc:
        raise CacheContractError(f"{name} cannot be encoded") from exc
    if len(result) > maximum:
        raise CacheContractError(f"{name} exceeds {maximum} wire bytes")
    return result


def _load_wire_json(raw: bytes, *, name: str, maximum: int) -> dict:
    if not raw or len(raw) > maximum:
        raise CacheContractError(f"{name} payload must be 1..{maximum} bytes")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheContractError(f"invalid {name} JSON") from exc
    if not isinstance(value, dict):
        raise CacheContractError(f"{name} payload must be a JSON object")
    return value


@dataclass(frozen=True, slots=True)
class CacheTransferContract:
    """Thin PD wire envelope around the cache-owned plan and group specs."""

    plan: CacheMemoryPlan
    group_specs: tuple[PagedCacheGroupSpec, ...]
    # Dtypes align positionally with plan.fields; field IDs stay plan-owned.
    field_dtypes: tuple[str, ...]

    def fields_for_group(self, group_id: str) -> tuple[CacheFieldLayout, ...]:
        return tuple(
            sorted(
                (field for field in self.plan.fields if field.group_id == group_id),
                key=lambda field: field.field_id,
            )
        )

    def field_dtype(self, field_id: str) -> str:
        for field, dtype in zip(self.plan.fields, self.field_dtypes, strict=True):
            if field.field_id == field_id:
                return dtype
        raise KeyError(field_id)

    def to_wire_bytes(self) -> bytes:
        return _dump_wire_json(
            self,
            name="cache transfer contract",
            maximum=MAX_CACHE_CONTRACT_WIRE_BYTES,
        )

    @classmethod
    def from_wire_bytes(cls, raw: bytes) -> "CacheTransferContract":
        payload = _load_wire_json(
            raw,
            name="cache transfer contract",
            maximum=MAX_CACHE_CONTRACT_WIRE_BYTES,
        )
        try:
            plan_payload = payload["plan"]
            plan = CacheMemoryPlan(
                logical_block_tokens=plan_payload["logical_block_tokens"],
                lcm_block_bytes=plan_payload["lcm_block_bytes"],
                num_lcm_blocks=plan_payload["num_lcm_blocks"],
                groups=tuple(
                    CacheGroupLayout(**group) for group in plan_payload["groups"]
                ),
                planes=tuple(
                    CachePlaneLayout(**plane) for plane in plan_payload["planes"]
                ),
                fields=tuple(
                    CacheFieldLayout(
                        **{
                            **field,
                            "shape": tuple(field["shape"]),
                        }
                    )
                    for field in plan_payload["fields"]
                ),
            )
            return cls(
                plan=plan,
                group_specs=tuple(
                    PagedCacheGroupSpec(**spec) for spec in payload["group_specs"]
                ),
                field_dtypes=tuple(payload["field_dtypes"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheContractError("invalid cache transfer contract") from exc


def build_cache_transfer_contract(
    *,
    plan: CacheMemoryPlan,
    buffer: object,
    group_specs: Sequence[PagedCacheGroupSpec],
    field_dtypes: Mapping[str, str],
) -> tuple[CacheTransferContract, int]:
    """Bind one cache memory plan to its semantics, dtypes, and raw slab."""
    specs = tuple(group_specs)
    plan_group_ids = tuple(group.group_id for group in plan.groups)
    spec_group_ids = tuple(spec.group_id for spec in specs)
    if set(plan_group_ids) != set(spec_group_ids):
        raise CacheContractError(
            "cache plan and scheduler group IDs disagree: "
            f"missing={sorted(set(plan_group_ids) - set(spec_group_ids))}, "
            f"extra={sorted(set(spec_group_ids) - set(plan_group_ids))}"
        )
    expected_field_ids = {field.field_id for field in plan.fields}
    if set(field_dtypes) != expected_field_ids:
        raise CacheContractError(
            "cache field dtype map must contain exactly the planned fields"
        )
    contract = CacheTransferContract(
        plan=plan,
        group_specs=specs,
        field_dtypes=tuple(field_dtypes[field.field_id] for field in plan.fields),
    )
    if (
        str(buffer.dtype) != "torch.uint8"
        or not buffer.is_contiguous()
        or buffer.storage_offset() != 0
        or buffer.data_ptr() != buffer.untyped_storage().data_ptr()
        or int(buffer.nbytes) != plan.arena_bytes
    ):
        raise CacheContractError(
            "cache transfer buffer must be the contiguous uint8 arena owner"
        )
    return contract, buffer.data_ptr()


def build_pool_cache_transfer_contract(
    pool: object,
) -> tuple[CacheTransferContract, int]:
    """Build the PD wire envelope from the cache-owned arena binding."""
    buffer, field_dtypes = pool.cache_contract_binding()
    return build_cache_transfer_contract(
        plan=pool.plan,
        buffer=buffer,
        group_specs=pool.paged_cache_group_specs,
        field_dtypes=field_dtypes,
    )


@dataclass(frozen=True, slots=True)
class CachePDGroupPages:
    group_id: str
    page_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CachePDPageManifest:
    groups: tuple[CachePDGroupPages, ...]
    prefix_len: int
    prompt_len: int

    def to_wire_bytes(self) -> bytes:
        return _dump_wire_json(
            self,
            name="Paged cache manifest",
            maximum=MAX_CACHE_MANIFEST_WIRE_BYTES,
        )

    @classmethod
    def from_wire_bytes(cls, raw: bytes) -> "CachePDPageManifest":
        payload = _load_wire_json(
            raw, name="Paged cache manifest", maximum=MAX_CACHE_MANIFEST_WIRE_BYTES
        )
        return cls(
            groups=tuple(
                CachePDGroupPages(group["group_id"], tuple(group["page_ids"]))
                for group in payload["groups"]
            ),
            prefix_len=payload["prefix_len"],
            prompt_len=payload["prompt_len"],
        )


def _logical_slots(
    policy: TransferPolicy,
    prefix_len: int,
    prompt_len: int,
    block_size: int,
    retention: Retention,
    sliding_window_tokens: int | None,
) -> tuple[int, ...]:
    if policy == "full_suffix":
        begin = prefix_len // block_size
        if retention == "sliding_window":
            # The next decode token can attend the preceding window - 1 raw
            # tokens. Include every group page intersecting that retained tail.
            retained_begin = max(0, prompt_len - sliding_window_tokens + 1)
            begin = max(begin, retained_begin // block_size)
        end = (prompt_len + block_size - 1) // block_size
        return tuple(range(begin, end))
    return ((prompt_len - 1) // block_size,)


def validate_cache_peer_layout(
    layout: CacheTransferContract, peer_layout: CacheTransferContract
) -> None:
    if layout.plan.logical_block_tokens != peer_layout.plan.logical_block_tokens:
        raise CacheContractError(
            "Paged cache P/D contract mismatch: scheduler_block_tokens"
        )
    local_group_ids = tuple(spec.group_id for spec in layout.group_specs)
    peer_group_ids = tuple(spec.group_id for spec in peer_layout.group_specs)
    if local_group_ids != peer_group_ids:
        raise CacheContractError("Paged cache P/D contract mismatch: group order")
    for local_spec, peer_spec in zip(
        layout.group_specs, peer_layout.group_specs, strict=True
    ):
        if (
            local_spec.family != peer_spec.family
            or local_spec.cache_block_tokens != peer_spec.cache_block_tokens
            or local_spec.retention != peer_spec.retention
            or local_spec.sliding_window_tokens != peer_spec.sliding_window_tokens
            or local_spec.transfer_policy != peer_spec.transfer_policy
        ):
            raise CacheContractError(
                f"Paged cache P/D contract mismatch: group {local_spec.group_id!r} "
                "semantics"
            )
        local_fields = layout.fields_for_group(local_spec.group_id)
        peer_fields = peer_layout.fields_for_group(peer_spec.group_id)
        if tuple(field.field_id for field in local_fields) != tuple(
            field.field_id for field in peer_fields
        ):
            raise CacheContractError(
                f"Paged cache P/D contract mismatch: group "
                f"{local_spec.group_id!r} transfer field order"
            )
        for local_field, peer_field in zip(local_fields, peer_fields, strict=True):
            if (
                layout.field_dtype(local_field.field_id)
                != peer_layout.field_dtype(peer_field.field_id)
                or local_field.shape != peer_field.shape
                or local_field.element_size != peer_field.element_size
            ):
                raise CacheContractError(
                    f"Paged cache P/D contract mismatch: field "
                    f"{local_field.field_id!r} semantics"
                )


def validate_cache_manifest(
    manifest: CachePDPageManifest,
    *,
    layout: CacheTransferContract,
    peer: str,
) -> None:
    if manifest.prefix_len >= manifest.prompt_len:
        raise CacheContractError(f"{peer} manifest requires prefix_len < prompt_len")
    expected = tuple(spec.group_id for spec in layout.group_specs)
    actual = tuple(group.group_id for group in manifest.groups)
    if actual != expected:
        raise CacheContractError(f"{peer} manifest group order disagrees with layout")
    if manifest.prefix_len % layout.plan.logical_block_tokens:
        raise CacheContractError(f"{peer} manifest prefix_len is not page aligned")
    for group, spec in zip(manifest.groups, layout.group_specs, strict=True):
        required = _logical_slots(
            spec.transfer_policy,
            manifest.prefix_len,
            manifest.prompt_len,
            spec.cache_block_tokens,
            spec.retention,
            spec.sliding_window_tokens,
        )
        if len(group.page_ids) != len(required):
            raise CacheContractError(
                f"{peer} manifest group {group.group_id!r} page count disagrees "
                "with its transfer policy"
            )
        group_capacity = layout.plan.group(spec.group_id).page_count
        if any(page <= 0 or page >= group_capacity for page in group.page_ids):
            raise CacheContractError(
                f"{peer} manifest group {group.group_id!r} has an out-of-bounds page"
            )


def validate_cache_manifest_pair(
    src_manifest: CachePDPageManifest,
    dst_manifest: CachePDPageManifest,
    src_layout: CacheTransferContract,
    dst_layout: CacheTransferContract,
) -> None:
    validate_cache_manifest(src_manifest, layout=src_layout, peer="source")
    validate_cache_manifest(dst_manifest, layout=dst_layout, peer="destination")
    if (
        src_manifest.prefix_len != dst_manifest.prefix_len
        or src_manifest.prompt_len != dst_manifest.prompt_len
    ):
        raise CacheContractError("source/destination prefix_len or prompt_len disagree")


def cache_manifest_page_ids(
    manifest: CachePDPageManifest,
    *,
    layout: CacheTransferContract,
    peer: str = "local",
) -> tuple[int, ...]:
    """Flatten a validated manifest into the legacy Mooncake page-vector order."""
    validate_cache_manifest(manifest, layout=layout, peer=peer)
    return tuple(page_id for group in manifest.groups for page_id in group.page_ids)


def build_cache_page_manifest(
    forward_op: object,
    *,
    layout: CacheTransferContract,
    request_row: int,
    prefix_len: int,
    prompt_len: int,
) -> CachePDPageManifest:
    """Select each group's pages according to its explicit transfer policy."""
    if prefix_len >= prompt_len:
        raise CacheContractError("Paged cache PD requires prefix_len < prompt_len")
    if prefix_len % layout.plan.logical_block_tokens:
        raise CacheContractError("Paged cache PD prefix_len must be page aligned")
    mapping = forward_op.block_tables_arrays()  # type: ignore[attr-defined]
    expected_ids = {group.group_id for group in layout.group_specs}
    if set(mapping) != expected_ids:
        raise CacheContractError(
            "scheduler group IDs disagree with the Paged cache layout: "
            f"missing={sorted(expected_ids - set(mapping))}, "
            f"extra={sorted(set(mapping) - expected_ids)}"
        )

    groups: list[CachePDGroupPages] = []
    for spec in layout.group_specs:
        table = mapping[spec.group_id]
        logical_slots = _logical_slots(
            spec.transfer_policy,
            prefix_len,
            prompt_len,
            spec.cache_block_tokens,
            spec.retention,
            spec.sliding_window_tokens,
        )
        if logical_slots and logical_slots[-1] >= table.shape[1]:
            raise CacheContractError(
                f"table {spec.group_id!r} misses logical slot " f"{logical_slots[-1]}"
            )
        page_ids = tuple(
            int(table[request_row, logical_slot]) for logical_slot in logical_slots
        )
        for logical_slot, page_id in zip(logical_slots, page_ids, strict=True):
            group_capacity = layout.plan.group(spec.group_id).page_count
            if page_id <= 0 or page_id >= group_capacity:
                raise CacheContractError(
                    f"table {spec.group_id!r} logical slot {logical_slot} "
                    f"has invalid page ID {page_id}"
                )
        groups.append(
            CachePDGroupPages(
                spec.group_id,
                page_ids,
            )
        )
    return CachePDPageManifest(
        groups=tuple(groups), prefix_len=prefix_len, prompt_len=prompt_len
    )


__all__ = [
    "CacheContractError",
    "CachePDGroupPages",
    "CachePDPageManifest",
    "CacheTransferContract",
    "build_cache_page_manifest",
    "build_cache_transfer_contract",
    "build_pool_cache_transfer_contract",
    "cache_manifest_page_ids",
    "validate_cache_manifest",
    "validate_cache_manifest_pair",
    "validate_cache_peer_layout",
]
