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

"""The cache transfer contract a PD rank publishes.

The cache recipe owns semantic group specs and the transfer schema, while the
memory planner owns physical geometry. PD transports those objects directly
instead of maintaining a second physical layout schema: this module binds
them into the wire contract, and the producer schedule layerwise streaming
follows. Everything a transfer reads off the contract -- routes, block
manifests, copies -- is planned by the engine host (``ts-pd``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheMemoryPlan,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.transfer import (
    CacheFieldPartition,
    CacheFieldTransferSpec,
    CacheTransferSchema,
    build_cache_transfer_schema,
)

MAX_CACHE_CONTRACT_WIRE_BYTES = 256 << 10


class CacheContractError(ValueError):
    """A cache pool or its plan violated the runtime cache contract."""


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


@dataclass(frozen=True)
class CacheTransferContract:
    """Thin PD wire envelope around the cache-owned plan and group specs."""

    plan: CacheMemoryPlan
    group_specs: tuple[CacheGroupSpec, ...]
    transfer_schema: CacheTransferSchema = CacheTransferSchema()
    # Endpoint bindings; absent in the rank-independent bootstrap layout.
    field_addresses: dict[str, int] | None = None

    def __post_init__(self) -> None:
        plan_group_ids = {group.group_id for group in self.plan.groups}
        spec_group_ids = {spec.group_id for spec in self.group_specs}
        if plan_group_ids != spec_group_ids:
            raise CacheContractError(
                "cache plan and scheduler group IDs disagree: "
                f"missing={sorted(plan_group_ids - spec_group_ids)}, "
                f"extra={sorted(spec_group_ids - plan_group_ids)}"
            )
        self.transfer_schema.validate(self.plan)
        if self.field_addresses is not None:
            if not isinstance(self.field_addresses, dict):
                raise CacheContractError("field addresses must be a mapping")
            if set(self.field_addresses) != {f.field_id for f in self.plan.fields}:
                raise CacheContractError("field addresses must cover the complete plan")
            if any(
                type(p) is not int or not 0 < p < 2**64
                for p in self.field_addresses.values()
            ):
                raise CacheContractError("field addresses must be nonzero uint64")

    def to_wire_bytes(self) -> bytes:
        return _dump_wire_json(
            self,
            name="cache transfer contract",
            maximum=MAX_CACHE_CONTRACT_WIRE_BYTES,
        )


def build_arena_cache_transfer_contract(
    arena: object,
    *,
    transfer_schema: CacheTransferSchema = CacheTransferSchema(),
) -> tuple[CacheTransferContract, int]:
    """Build the PD wire envelope from the cache arena it transfers."""
    if not arena.supports_disaggregation:
        raise RuntimeError("cache arena has no transfer policy in its runtime contract")
    return (
        CacheTransferContract(
            plan=arena.plan,
            group_specs=tuple(arena.cache_group_specs),
            transfer_schema=transfer_schema,
            field_addresses={
                f.field_id: arena.field_address(f.field_id) for f in arena.plan.fields
            },
        ),
        arena.registration_regions()[0][0],
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
    producer_fields_by_step: tuple[tuple[str, ...], ...],
) -> CacheProducerSchedule:
    """Validate construction-provided readiness against the resident field plan.

    Args:
        plan: This rank's physical cache plan.
        producer_fields_by_step: Field IDs becoming ready at each local barrier.

    Returns:
        A model-independent producer schedule covering every resident field once.
    """
    schedule = CacheProducerSchedule(fields_by_step=producer_fields_by_step)
    if schedule.fields_in_range(0, schedule.step_count) != frozenset(
        field.field_id for field in plan.fields
    ):
        raise ValueError("cache producer schedule must cover every resident field")
    return schedule


__all__ = [
    "CacheContractError",
    "CacheFieldPartition",
    "CacheFieldTransferSpec",
    "CacheProducerSchedule",
    "CacheTransferContract",
    "CacheTransferSchema",
    "build_cache_fields_by_producer_step",
    "build_cache_transfer_schema",
    "build_arena_cache_transfer_contract",
]
