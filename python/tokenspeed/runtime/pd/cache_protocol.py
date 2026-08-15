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

The cache recipe owns semantic group specs and the transfer schema, while the
memory planner owns physical geometry. PD transports those objects directly
instead of maintaining a second physical layout schema. This module owns the
wire contract, request block manifests, producer-step projection, and peer
validation.
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
    cache_field_layer_id,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    PagedCacheGroupSpec,
    Retention,
    TransferPolicy,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.transfer import (
    CacheFieldPartition,
    CacheFieldTransferSpec,
    CacheTransferSchema,
    build_cache_transfer_schema,
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
    transfer_schema: CacheTransferSchema = CacheTransferSchema()

    def __post_init__(self) -> None:
        self.transfer_schema.validate(self.plan)

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
                prefix_granularity=plan_payload["prefix_granularity"],
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
            schema_payload = payload["transfer_schema"]
            return cls(
                plan=plan,
                group_specs=tuple(
                    PagedCacheGroupSpec(**spec) for spec in payload["group_specs"]
                ),
                field_dtypes=tuple(payload["field_dtypes"]),
                transfer_schema=CacheTransferSchema(
                    tuple(
                        CacheFieldTransferSpec(
                            field_id=field["field_id"],
                            partition=CacheFieldPartition(
                                axis=field["partition"]["axis"],
                                global_extent=field["partition"]["global_extent"],
                                global_parts=tuple(field["partition"]["global_parts"]),
                            ),
                        )
                        for field in schema_payload["fields"]
                    )
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CacheContractError("invalid cache transfer contract") from exc


def build_cache_transfer_contract(
    *,
    plan: CacheMemoryPlan,
    buffer: object,
    group_specs: Sequence[PagedCacheGroupSpec],
    field_dtypes: Mapping[str, str],
    transfer_schema: CacheTransferSchema = CacheTransferSchema(),
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
        transfer_schema=transfer_schema,
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
    *,
    transfer_schema: CacheTransferSchema = CacheTransferSchema(),
) -> tuple[CacheTransferContract, int]:
    """Build the PD wire envelope from the cache-owned arena binding."""
    buffer, field_dtypes = pool.cache_contract_binding()
    return build_cache_transfer_contract(
        plan=pool.plan,
        buffer=buffer,
        group_specs=pool.paged_cache_group_specs,
        field_dtypes=field_dtypes,
        transfer_schema=transfer_schema,
    )


@dataclass(frozen=True, slots=True)
class CacheProducerSchedule:
    """Prefill-local field readiness metadata, intentionally absent from wire."""

    fields_by_step: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not self.fields_by_step:
            raise ValueError("cache producer schedule must contain at least one step")
        fields = []
        for step_fields in self.fields_by_step:
            if any(not field_id for field_id in step_fields):
                raise ValueError("cache producer field IDs must be non-empty")
            fields.extend(step_fields)
        if len(fields) != len(set(fields)):
            raise ValueError("a cache field cannot be produced by multiple steps")

    @property
    def step_count(self) -> int:
        return len(self.fields_by_step)

    def fields_in_range(self, begin_step: int, end_step: int) -> frozenset[str]:
        if not 0 <= begin_step <= end_step <= self.step_count:
            raise ValueError(
                f"cache producer step range [{begin_step}, {end_step}) is outside "
                f"[0, {self.step_count})"
            )
        return frozenset(
            field_id
            for step_fields in self.fields_by_step[begin_step:end_step]
            for field_id in step_fields
        )


def build_cache_fields_by_producer_step(
    plan: CacheMemoryPlan,
    *,
    num_target_layers: int,
) -> CacheProducerSchedule:
    """Group cache fields by the Prefill barrier that makes them transferable."""

    fields_by_layer: dict[int, list[str]] = {}
    for field in plan.fields:
        layer_id = cache_field_layer_id(field.field_id)
        fields_by_layer.setdefault(layer_id, []).append(field.field_id)

    if not fields_by_layer:
        raise ValueError("layerwise PD requires at least one cache field")
    merged_layers = max(fields_by_layer) + 1
    if (
        isinstance(num_target_layers, bool)
        or not isinstance(num_target_layers, int)
        or not 0 < num_target_layers <= merged_layers
    ):
        raise ValueError("PD target layer count is outside the cache plan")

    fields_by_step = [
        tuple(fields_by_layer.get(layer_id, ()))
        for layer_id in range(num_target_layers)
    ]
    if merged_layers > num_target_layers:
        # A speculative drafter may execute its physical layers repeatedly;
        # all draft cache fields become transferable at one final barrier.
        fields_by_step.append(
            tuple(
                field_id
                for layer_id in range(num_target_layers, merged_layers)
                for field_id in fields_by_layer.get(layer_id, ())
            )
        )

    return CacheProducerSchedule(tuple(fields_by_step))


@dataclass(frozen=True, slots=True)
class CachePDGroupBlocks:
    group_id: str
    block_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CachePDBlockManifest:
    groups: tuple[CachePDGroupBlocks, ...]
    prefix_len: int
    prompt_len: int

    def to_wire_bytes(self) -> bytes:
        return _dump_wire_json(
            self,
            name="Cache block manifest",
            maximum=MAX_CACHE_MANIFEST_WIRE_BYTES,
        )

    @classmethod
    def from_wire_bytes(cls, raw: bytes) -> "CachePDBlockManifest":
        payload = _load_wire_json(
            raw, name="Cache block manifest", maximum=MAX_CACHE_MANIFEST_WIRE_BYTES
        )
        return cls(
            groups=tuple(
                CachePDGroupBlocks(group["group_id"], tuple(group["block_ids"]))
                for group in payload["groups"]
            ),
            prefix_len=payload["prefix_len"],
            prompt_len=payload["prompt_len"],
        )


@dataclass(frozen=True, slots=True)
class CachePDLayerwiseGroupSelection:
    """Source blocks for one group and their positions in Decode's manifest."""

    source_block_ids: tuple[int, ...]
    destination_positions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CachePDLayerwiseBlockSelection:
    """One prompt chunk's group-aware source-to-destination block selection.

    This object stays process-local on Prefill. Decode publishes the full
    request manifest once; each queued layerwise chunk references positions in
    that immutable destination manifest instead of inventing a partial wire
    manifest with ambiguous state-group semantics.
    """

    groups: tuple[CachePDLayerwiseGroupSelection, ...]


def _logical_slots(
    policy: TransferPolicy,
    prefix_len: int,
    prompt_len: int,
    block_granularity: int,
    retention: Retention,
    sliding_window_tokens: int | None,
) -> tuple[int, ...]:
    if policy == "full_suffix":
        begin = prefix_len // block_granularity
        if retention == "sliding_window":
            # The next decode token can attend the preceding window - 1 raw
            # tokens. Include every group block intersecting that retained tail.
            retained_begin = max(0, prompt_len - sliding_window_tokens + 1)
            begin = max(begin, retained_begin // block_granularity)
        end = (prompt_len + block_granularity - 1) // block_granularity
        return tuple(range(begin, end))
    return ((prompt_len - 1) // block_granularity,)


def validate_cache_peer_layout(
    layout: CacheTransferContract, peer_layout: CacheTransferContract
) -> None:
    if layout.plan.prefix_granularity != peer_layout.plan.prefix_granularity:
        raise CacheContractError(
            "Paged cache P/D contract mismatch: prefix_granularity"
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
            or local_spec.rows_per_page != peer_spec.rows_per_page
            or local_spec.entry_stride_tokens != peer_spec.entry_stride_tokens
            or local_spec.checkpoint_granularity != peer_spec.checkpoint_granularity
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
        local_field_ids = tuple(field.field_id for field in local_fields)
        peer_field_ids = tuple(field.field_id for field in peer_fields)
        if local_field_ids != peer_field_ids:
            raise CacheContractError(
                f"Paged cache P/D contract mismatch: group "
                f"{local_spec.group_id!r} transfer field order"
            )
        for local_field, peer_field in zip(local_fields, peer_fields, strict=True):
            local_global_shape = list(local_field.shape)
            peer_global_shape = list(peer_field.shape)
            local_partition = layout.transfer_schema.partition_for(local_field.field_id)
            peer_partition = peer_layout.transfer_schema.partition_for(
                peer_field.field_id
            )
            if local_partition is not None:
                local_global_shape[local_partition.axis] = local_partition.global_extent
            if peer_partition is not None:
                peer_global_shape[peer_partition.axis] = peer_partition.global_extent
            if (
                layout.field_dtype(local_field.field_id)
                != peer_layout.field_dtype(peer_field.field_id)
                or local_field.element_size != peer_field.element_size
                or local_partition != peer_partition
                or tuple(local_global_shape) != tuple(peer_global_shape)
            ):
                raise CacheContractError(
                    f"Paged cache P/D contract mismatch: field "
                    f"{local_field.field_id!r} semantics"
                )


def validate_cache_manifest(
    manifest: CachePDBlockManifest,
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
    if manifest.prefix_len % layout.plan.prefix_granularity:
        raise CacheContractError(
            f"{peer} manifest prefix_len is not aligned to prefix_granularity"
        )
    for group, spec in zip(manifest.groups, layout.group_specs, strict=True):
        required = _logical_slots(
            spec.transfer_policy,
            manifest.prefix_len,
            manifest.prompt_len,
            spec.block_granularity,
            spec.retention,
            spec.sliding_window_tokens,
        )
        if len(group.block_ids) != len(required):
            raise CacheContractError(
                f"{peer} manifest group {group.group_id!r} block count disagrees "
                "with its transfer policy"
            )
        group_capacity = layout.plan.group(spec.group_id).page_count
        if any(block <= 0 or block >= group_capacity for block in group.block_ids):
            raise CacheContractError(
                f"{peer} manifest group {group.group_id!r} has an out-of-bounds block"
            )


def build_cache_block_manifest(
    forward_op: object,
    *,
    layout: CacheTransferContract,
    request_row: int,
    prefix_len: int,
    prompt_len: int,
) -> CachePDBlockManifest:
    """Select each group's blocks according to its explicit transfer policy."""
    if prefix_len >= prompt_len:
        raise CacheContractError("Paged cache PD requires prefix_len < prompt_len")
    if prefix_len % layout.plan.prefix_granularity:
        raise CacheContractError(
            "Paged cache PD prefix_len must be aligned to prefix_granularity"
        )
    mapping = forward_op.block_tables_arrays()  # type: ignore[attr-defined]
    expected_ids = {group.group_id for group in layout.group_specs}
    if set(mapping) != expected_ids:
        raise CacheContractError(
            "scheduler group IDs disagree with the Paged cache layout: "
            f"missing={sorted(expected_ids - set(mapping))}, "
            f"extra={sorted(set(mapping) - expected_ids)}"
        )

    groups: list[CachePDGroupBlocks] = []
    for spec in layout.group_specs:
        table = mapping[spec.group_id]
        logical_slots = _logical_slots(
            spec.transfer_policy,
            prefix_len,
            prompt_len,
            spec.block_granularity,
            spec.retention,
            spec.sliding_window_tokens,
        )
        if logical_slots and logical_slots[-1] >= table.shape[1]:
            raise CacheContractError(
                f"table {spec.group_id!r} misses logical slot {logical_slots[-1]}"
            )
        block_ids = tuple(
            int(table[request_row, logical_slot]) for logical_slot in logical_slots
        )
        for logical_slot, block_id in zip(logical_slots, block_ids, strict=True):
            group_capacity = layout.plan.group(spec.group_id).page_count
            if block_id <= 0 or block_id >= group_capacity:
                raise CacheContractError(
                    f"table {spec.group_id!r} logical slot {logical_slot} "
                    f"has invalid block ID {block_id}"
                )
        groups.append(
            CachePDGroupBlocks(
                spec.group_id,
                block_ids,
            )
        )
    return CachePDBlockManifest(
        groups=tuple(groups), prefix_len=prefix_len, prompt_len=prompt_len
    )


def build_cache_layerwise_block_selection(
    forward_op: object,
    *,
    layout: CacheTransferContract,
    request_row: int,
    prefix_len: int,
    prompt_len: int,
    chunk_start: int,
    chunk_end: int,
) -> CachePDLayerwiseBlockSelection:
    """Select blocks that became transferable during one Prefill chunk.

    Full-suffix groups publish newly completed retained blocks (plus the final
    partial block). Latest-snapshot groups publish only the prompt's final
    snapshot. The returned destination positions refer to Decode's immutable
    full manifest.
    """
    if prefix_len % layout.plan.prefix_granularity:
        raise CacheContractError(
            "Paged cache PD prefix_len must be aligned to prefix_granularity"
        )
    if prefix_len >= prompt_len:
        raise CacheContractError("layerwise selection requires prefix_len < prompt_len")
    if chunk_start >= chunk_end:
        raise CacheContractError("layerwise selection requires chunk_start < chunk_end")
    if chunk_end > prompt_len:
        raise CacheContractError("layerwise selection requires chunk_end <= prompt_len")
    is_final = chunk_end == prompt_len

    mapping = forward_op.block_tables_arrays()  # type: ignore[attr-defined]
    expected_ids = {group.group_id for group in layout.group_specs}
    if set(mapping) != expected_ids:
        raise CacheContractError(
            "scheduler group IDs disagree with the Paged cache layout: "
            f"missing={sorted(expected_ids - set(mapping))}, "
            f"extra={sorted(set(mapping) - expected_ids)}"
        )

    selections = []
    for spec in layout.group_specs:
        table = mapping[spec.group_id]
        block_granularity = spec.block_granularity
        final_slots = _logical_slots(
            spec.transfer_policy,
            prefix_len,
            prompt_len,
            block_granularity,
            spec.retention,
            spec.sliding_window_tokens,
        )
        if spec.transfer_policy == "full_suffix":
            group_start_position = (
                max(
                    0,
                    min(
                        len(final_slots),
                        chunk_start // block_granularity - final_slots[0],
                    ),
                )
                if final_slots
                else 0
            )
            ready_end = (
                (chunk_end + block_granularity - 1) // block_granularity
                if is_final
                else chunk_end // block_granularity
            )
            ready_count = (
                max(0, min(len(final_slots), ready_end - final_slots[0]))
                if final_slots
                else 0
            )
            destination_positions = tuple(range(group_start_position, ready_count))
            logical_slots = tuple(
                final_slots[position] for position in destination_positions
            )
        elif spec.transfer_policy == "latest_snapshot":
            destination_positions = (0,) if is_final else ()
            logical_slots = (final_slots[0],) if is_final else ()
        if logical_slots and logical_slots[-1] >= table.shape[1]:
            raise CacheContractError(
                f"table {spec.group_id!r} misses logical slot {logical_slots[-1]}"
            )
        source_block_ids = tuple(
            int(table[request_row, logical_slot]) for logical_slot in logical_slots
        )
        group_capacity = layout.plan.group(spec.group_id).page_count
        for logical_slot, block_id in zip(logical_slots, source_block_ids, strict=True):
            if block_id <= 0 or block_id >= group_capacity:
                raise CacheContractError(
                    f"table {spec.group_id!r} logical slot {logical_slot} "
                    f"has invalid block ID {block_id}"
                )
        selections.append(
            CachePDLayerwiseGroupSelection(
                source_block_ids=source_block_ids,
                destination_positions=destination_positions,
            )
        )

    return CachePDLayerwiseBlockSelection(groups=tuple(selections))


__all__ = [
    "CacheContractError",
    "CacheFieldPartition",
    "CacheFieldTransferSpec",
    "CachePDBlockManifest",
    "CachePDGroupBlocks",
    "CachePDLayerwiseGroupSelection",
    "CachePDLayerwiseBlockSelection",
    "CacheProducerSchedule",
    "CacheTransferContract",
    "CacheTransferSchema",
    "build_cache_fields_by_producer_step",
    "build_cache_block_manifest",
    "build_cache_layerwise_block_selection",
    "build_cache_transfer_schema",
    "build_cache_transfer_contract",
    "build_pool_cache_transfer_contract",
    "validate_cache_manifest",
    "validate_cache_peer_layout",
]
