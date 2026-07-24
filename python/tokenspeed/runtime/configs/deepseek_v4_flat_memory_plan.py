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

"""DeepSeek V4 owner union, storage binding, and TP plan agreement."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace

from tokenspeed.runtime.configs.flat_kv_contract import (
    CACHE_OWNER_DRAFT,
    CACHE_OWNER_TARGET,
)
from tokenspeed.runtime.configs.flat_memory_plan import (
    FlatBlockPoolPlan,
    FlatCacheOwner,
    FlatComponentTensorPlan,
    FlatRuntimeMetadataPlan,
    require_sha256_hexdigest,
)
from tokenspeed.runtime.configs.paged_cache_spec import PagedCacheGroupSpec


@dataclass(frozen=True)
class V4FlatMemoryPlan:
    """Canonical device-side V4 pool plan and scheduler/owner group views."""

    max_total_tokens: int
    pools: tuple[FlatBlockPoolPlan, ...]
    scheduler_group_specs: tuple[PagedCacheGroupSpec, ...]
    runtime_metadata: FlatRuntimeMetadataPlan
    plan_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "pools", tuple(self.pools))
        object.__setattr__(
            self, "scheduler_group_specs", tuple(self.scheduler_group_specs)
        )
        if not isinstance(self.runtime_metadata, FlatRuntimeMetadataPlan):
            raise TypeError("runtime_metadata must be FlatRuntimeMetadataPlan")
        if (
            isinstance(self.max_total_tokens, bool)
            or not isinstance(self.max_total_tokens, int)
            or self.max_total_tokens < 0
        ):
            raise ValueError("V4 flat memory max_total_tokens must be >= 0")
        if not self.pools:
            raise ValueError("V4 flat memory plan must contain at least one pool")
        scheduler_group_ids = [spec.group_id for spec in self.scheduler_group_specs]
        if len(set(scheduler_group_ids)) != len(scheduler_group_ids):
            raise ValueError("scheduler_group_specs must have unique group ids")
        if any(spec.owner_mask == 0 for spec in self.scheduler_group_specs):
            raise ValueError(
                "scheduler_group_specs must assign every group to an owner"
            )
        table_group_ids = {
            plan.group_id for plan in self.runtime_metadata.group_table_plans
        }
        if table_group_ids != set(scheduler_group_ids):
            raise ValueError(
                "group_table_plans must cover the scheduler group union exactly"
            )
        require_sha256_hexdigest(self.plan_fingerprint, field_name="plan_fingerprint")

    def group_specs_for_owner(self, owner_mask: int) -> tuple[PagedCacheGroupSpec, ...]:
        """Return a read-only owner view over the canonical scheduler schema."""

        if isinstance(owner_mask, bool) or owner_mask not in (
            CACHE_OWNER_TARGET,
            CACHE_OWNER_DRAFT,
        ):
            raise ValueError(
                "owner_mask must be exactly CACHE_OWNER_TARGET or "
                f"CACHE_OWNER_DRAFT, got {owner_mask!r}"
            )
        return tuple(
            spec for spec in self.scheduler_group_specs if spec.owner_mask & owner_mask
        )

    @property
    def target_owner_group_specs(self) -> tuple[PagedCacheGroupSpec, ...]:
        """Target-owned entries from the canonical scheduler schema."""

        return self.group_specs_for_owner(CACHE_OWNER_TARGET)

    @property
    def draft_owner_group_specs(self) -> tuple[PagedCacheGroupSpec, ...]:
        """Draft-owned entries from the canonical scheduler schema."""

        return self.group_specs_for_owner(CACHE_OWNER_DRAFT)

    @property
    def payload_bytes(self) -> int:
        """Exact device payload bytes implied by the physical pools."""

        return sum(pool.total_blocks * pool.bytes_per_block for pool in self.pools)

    @property
    def device_cache_total_bytes(self) -> int:
        """Exact payload plus graph and forward-input metadata bytes."""

        return (
            self.payload_bytes
            + self.runtime_metadata.graph_metadata_bytes
            + self.runtime_metadata.forward_input_bytes
        )


@dataclass(frozen=True)
class V4FlatPlanAgreementRecord:
    """Minimal rank-local record used by the normal TP agreement path."""

    rank: int
    plan_fingerprint: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 0
        ):
            raise ValueError(f"rank must be an integer >= 0, got {self.rank!r}")
        require_sha256_hexdigest(
            self.plan_fingerprint,
            field_name="plan_fingerprint",
        )


_SCHEDULING_SCHEMA_FIELDS = (
    "pool_id",
    "block_size",
    "rows_per_page",
    "entry_stride_tokens",
    "retention",
    "family",
    "sliding_window_tokens",
    "prefix_role",
    "table_layout",
)


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_v4_flat_memory_plan(plan: V4FlatMemoryPlan) -> dict[str, object]:
    """Return the JSON-safe correctness fields covered by the fingerprint."""

    payload = asdict(plan)
    del payload["plan_fingerprint"]
    return payload


def make_v4_flat_plan_agreement_record(
    plan: V4FlatMemoryPlan,
    *,
    rank: int,
) -> V4FlatPlanAgreementRecord:
    """Build the CPU-serializable record gathered by attention-TP peers."""

    return V4FlatPlanAgreementRecord(
        rank=rank,
        plan_fingerprint=plan.plan_fingerprint,
    )


def _first_plan_difference(
    reference: object,
    candidate: object,
    *,
    path: str = "plan",
) -> tuple[str, object, object] | None:
    """Return the first deterministic leaf difference for mismatch diagnostics."""

    if type(reference) is not type(candidate):
        return path, reference, candidate
    if isinstance(reference, Mapping):
        reference_keys = set(reference)
        candidate_keys = set(candidate)
        if reference_keys != candidate_keys:
            return f"{path}.keys", sorted(reference_keys), sorted(candidate_keys)
        for key in sorted(reference_keys, key=str):
            difference = _first_plan_difference(
                reference[key],
                candidate[key],
                path=f"{path}.{key}",
            )
            if difference is not None:
                return difference
        return None
    if isinstance(reference, Sequence) and not isinstance(
        reference, (str, bytes, bytearray)
    ):
        if len(reference) != len(candidate):
            return f"{path}.length", len(reference), len(candidate)
        for index, (reference_item, candidate_item) in enumerate(
            zip(reference, candidate)
        ):
            difference = _first_plan_difference(
                reference_item,
                candidate_item,
                path=f"{path}[{index}]",
            )
            if difference is not None:
                return difference
        return None
    if reference != candidate:
        return path, reference, candidate
    return None


def assert_v4_flat_plan_agreement(
    records: Sequence[V4FlatPlanAgreementRecord],
    *,
    canonical_plans: Mapping[int, Mapping[str, object]] | None = None,
) -> None:
    """Fail on TP drift, optionally including an on-demand field diagnostic."""

    if len(records) <= 1:
        return
    reference = records[0]
    for candidate in records[1:]:
        if candidate.plan_fingerprint == reference.plan_fingerprint:
            continue
        detail = ""
        if canonical_plans is not None:
            reference_plan = canonical_plans.get(reference.rank)
            candidate_plan = canonical_plans.get(candidate.rank)
            if reference_plan is not None and candidate_plan is not None:
                difference = _first_plan_difference(reference_plan, candidate_plan)
                if difference is not None:
                    path, reference_value, candidate_value = difference
                    detail = (
                        f"; first difference at {path}: rank {reference.rank}="
                        f"{reference_value!r}, rank {candidate.rank}="
                        f"{candidate_value!r}"
                    )
        raise RuntimeError(
            "DeepSeek V4 flat plan differs across attention-TP ranks: "
            f"rank {candidate.rank} fingerprint={candidate.plan_fingerprint} "
            f"differs from rank {reference.rank} "
            f"fingerprint={reference.plan_fingerprint}{detail}"
        )


def _normalize_owner_specs(
    specs: Sequence[PagedCacheGroupSpec],
    *,
    owner: FlatCacheOwner,
    owner_bit: int,
) -> tuple[PagedCacheGroupSpec, ...]:
    by_group: dict[str, PagedCacheGroupSpec] = {}
    for spec in specs:
        if not spec.group_id:
            raise ValueError(f"{owner} group_id must be non-empty")
        rows_per_page = spec.rows_per_page
        entry_stride_tokens = spec.entry_stride_tokens
        if (
            isinstance(rows_per_page, bool)
            or not isinstance(rows_per_page, int)
            or rows_per_page <= 0
            or isinstance(entry_stride_tokens, bool)
            or not isinstance(entry_stride_tokens, int)
            or entry_stride_tokens <= 0
        ):
            raise ValueError(
                f"{owner} group {spec.group_id}: rows_per_page and "
                "entry_stride_tokens must be positive integers"
            )
        if spec.owner_mask not in (0, owner_bit):
            raise ValueError(
                f"{owner} group {spec.group_id}: owner_mask must be 0 or "
                f"{owner_bit}, got {spec.owner_mask}"
            )
        normalized = replace(spec, owner_mask=owner_bit)
        previous = by_group.get(spec.group_id)
        if previous is not None and previous != normalized:
            raise ValueError(
                f"{owner} group {spec.group_id}: duplicate group schema drift"
            )
        by_group[spec.group_id] = normalized
    return tuple(by_group[group_id] for group_id in sorted(by_group))


def _normalize_page_counts(
    specs: Sequence[PagedCacheGroupSpec],
    counts: Mapping[str, int] | None,
    *,
    owner: FlatCacheOwner,
) -> dict[str, int]:
    expected = {spec.group_id for spec in specs}
    if counts is None:
        if expected:
            raise ValueError(f"{owner}_group_page_counts are required")
        return {}
    actual = set(counts)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{owner}_group_page_counts keys do not match specs: "
            f"missing={missing}, extra={extra}"
        )
    normalized: dict[str, int] = {}
    for group_id in sorted(expected):
        count = counts[group_id]
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError(
                f"{owner} group {group_id}: total block count must be >= 2, "
                f"got {count!r}"
            )
        normalized[group_id] = count
    return normalized


def _merge_scheduler_specs(
    target_specs: Sequence[PagedCacheGroupSpec],
    draft_specs: Sequence[PagedCacheGroupSpec],
) -> tuple[PagedCacheGroupSpec, ...]:
    target_by_group = {spec.group_id: spec for spec in target_specs}
    draft_by_group = {spec.group_id: spec for spec in draft_specs}
    merged: list[PagedCacheGroupSpec] = []
    for group_id in sorted(target_by_group.keys() | draft_by_group.keys()):
        target = target_by_group.get(group_id)
        draft = draft_by_group.get(group_id)
        if target is not None and draft is not None:
            for field in _SCHEDULING_SCHEMA_FIELDS:
                target_value = getattr(target, field)
                draft_value = getattr(draft, field)
                if target_value != draft_value:
                    raise ValueError(
                        f"group {group_id} scheduling schema mismatch at {field}: "
                        f"target={target_value!r}, draft={draft_value!r}"
                    )
        base = target if target is not None else draft
        assert base is not None
        merged.append(
            replace(
                base,
                owner_mask=(target.owner_mask if target is not None else 0)
                | (draft.owner_mask if draft is not None else 0),
            )
        )
    return tuple(merged)


def _normalize_components(
    target_components: Sequence[FlatComponentTensorPlan],
    draft_components: Sequence[FlatComponentTensorPlan],
    *,
    target_specs: Sequence[PagedCacheGroupSpec],
    draft_specs: Sequence[PagedCacheGroupSpec],
) -> tuple[tuple[str, FlatComponentTensorPlan], ...]:
    owner_specs = {
        "target": {spec.group_id: spec for spec in target_specs},
        "draft": {spec.group_id: spec for spec in draft_specs},
    }
    by_identity: dict[
        tuple[str, FlatCacheOwner, int, str],
        tuple[str, FlatComponentTensorPlan],
    ] = {}
    groups_with_components: dict[FlatCacheOwner, set[str]] = {
        "target": set(),
        "draft": set(),
    }
    for expected_owner, components in (
        ("target", target_components),
        ("draft", draft_components),
    ):
        for component in components:
            if component.owner != expected_owner:
                raise ValueError(
                    f"{expected_owner}_components contains {component.owner!r} "
                    f"component {component.component!r}"
                )
            spec = owner_specs[expected_owner].get(component.group_id)
            if spec is None:
                raise ValueError(
                    f"{expected_owner} component {component.component!r} refers "
                    f"to unknown group {component.group_id!r}"
                )
            groups_with_components[expected_owner].add(component.group_id)
            identity = (
                spec.pool_id,
                component.owner,
                component.layer,
                component.component,
            )
            previous = by_identity.get(identity)
            current = (spec.pool_id, component)
            if previous is not None and previous != current:
                raise ValueError(
                    "duplicate component identity has storage schema drift: "
                    f"pool={spec.pool_id!r}, owner={component.owner!r}, "
                    f"layer={component.layer}, component={component.component!r}"
                )
            by_identity[identity] = current

    for owner, specs in owner_specs.items():
        missing = sorted(set(specs) - groups_with_components[owner])
        if missing:
            raise ValueError(f"{owner} groups missing component tensors: {missing}")

    owner_order = {"target": 0, "draft": 1}
    return tuple(
        sorted(
            by_identity.values(),
            key=lambda item: (
                item[0],
                owner_order[item[1].owner],
                item[1].layer,
                item[1].component,
                item[1].group_id,
                item[1].dtype,
                item[1].shape_per_block,
                item[1].stride_bytes,
                item[1].alignment_bytes,
                item[1].bytes_per_block,
            ),
        )
    )


def build_v4_flat_memory_plan(
    *,
    max_total_tokens: int = 0,
    target_group_specs: Sequence[PagedCacheGroupSpec],
    target_group_page_counts: Mapping[str, int],
    target_components: Sequence[FlatComponentTensorPlan],
    runtime_metadata: FlatRuntimeMetadataPlan,
    draft_group_specs: Sequence[PagedCacheGroupSpec] = (),
    draft_group_page_counts: Mapping[str, int] | None = None,
    draft_components: Sequence[FlatComponentTensorPlan] = (),
) -> V4FlatMemoryPlan:
    """Build one canonical physical plan from target/draft owner-local inputs.

    Args:
        max_total_tokens: Token capacity represented by the plan.
        target_group_specs: Target model's owner-local logical group specs.
        target_group_page_counts: Target capacity demand including null page 0.
        target_components: Target component-plane storage schemas.
        draft_group_specs: Optional draft model owner-local group specs.
        draft_group_page_counts: Draft capacity demand including null page 0.
        draft_components: Draft component-plane storage schemas.
        runtime_metadata: Canonical table shapes and graph/staging bounds.

    Returns:
        An immutable plan whose scheduler specs are the canonical target/draft
        union.  A shared group's capacity is the maximum owner demand while its
        bytes per block include every owner-namespaced component plane.

    Raises:
        ValueError: If logical scheduling schemas, counts, owner membership, or
        duplicate component storage schemas disagree.
    """
    if (
        isinstance(max_total_tokens, bool)
        or not isinstance(max_total_tokens, int)
        or max_total_tokens < 0
    ):
        raise ValueError("max_total_tokens must be an integer >= 0")
    if not isinstance(runtime_metadata, FlatRuntimeMetadataPlan):
        raise TypeError("runtime_metadata must be FlatRuntimeMetadataPlan")
    target_specs = _normalize_owner_specs(
        target_group_specs,
        owner="target",
        owner_bit=CACHE_OWNER_TARGET,
    )
    draft_specs = _normalize_owner_specs(
        draft_group_specs,
        owner="draft",
        owner_bit=CACHE_OWNER_DRAFT,
    )
    scheduler_specs = _merge_scheduler_specs(target_specs, draft_specs)
    if not scheduler_specs:
        raise ValueError("V4 flat memory plan requires at least one cache group")
    metadata_group_ids = {plan.group_id for plan in runtime_metadata.group_table_plans}
    scheduler_group_ids = {spec.group_id for spec in scheduler_specs}
    if metadata_group_ids != scheduler_group_ids:
        missing = sorted(scheduler_group_ids - metadata_group_ids)
        extra = sorted(metadata_group_ids - scheduler_group_ids)
        raise ValueError(
            "metadata group plans do not match scheduler group union: "
            f"missing={missing}, extra={extra}"
        )
    target_group_ids = {spec.group_id for spec in target_specs}
    draft_group_ids = {spec.group_id for spec in draft_specs}
    for table_plan in runtime_metadata.group_table_plans:
        target_has_capture = table_plan.target_capture_cols > 0
        draft_has_capture = table_plan.draft_capture_cols > 0
        target_owns_group = table_plan.group_id in target_group_ids
        draft_owns_group = table_plan.group_id in draft_group_ids
        if (
            target_has_capture != target_owns_group
            or draft_has_capture != draft_owns_group
        ):
            raise ValueError(
                "flat table capture owner membership does not match group "
                f"specs for {table_plan.group_id!r}: "
                f"capture=(target={target_has_capture}, "
                f"draft={draft_has_capture}), "
                f"owners=(target={target_owns_group}, "
                f"draft={draft_owns_group})"
            )
    groups_by_pool: dict[str, str] = {}
    for spec in scheduler_specs:
        previous_group = groups_by_pool.setdefault(spec.pool_id, spec.group_id)
        if previous_group != spec.group_id:
            raise ValueError(
                f"V4 flat pool {spec.pool_id!r} must bind exactly one group, "
                f"got {previous_group!r} and {spec.group_id!r}"
            )

    target_counts = _normalize_page_counts(
        target_specs,
        target_group_page_counts,
        owner="target",
    )
    draft_counts = _normalize_page_counts(
        draft_specs,
        draft_group_page_counts,
        owner="draft",
    )
    components_with_pools = _normalize_components(
        target_components,
        draft_components,
        target_specs=target_specs,
        draft_specs=draft_specs,
    )

    pool_ids = sorted({spec.pool_id for spec in scheduler_specs})
    pools: list[FlatBlockPoolPlan] = []
    for pool_id in pool_ids:
        group_ids = {
            spec.group_id for spec in scheduler_specs if spec.pool_id == pool_id
        }
        capacity_demands = [
            counts[group_id]
            for counts in (target_counts, draft_counts)
            for group_id in group_ids
            if group_id in counts
        ]
        if not capacity_demands:
            raise ValueError(f"pool {pool_id!r} has no capacity demand")
        tensors = tuple(
            component
            for component_pool_id, component in components_with_pools
            if component_pool_id == pool_id
        )
        if not tensors:
            raise ValueError(f"pool {pool_id!r} has no component tensors")
        bytes_per_block = sum(tensor.bytes_per_block for tensor in tensors)
        storage_schema_hash = _canonical_hash(
            {
                "pool_id": pool_id,
                "tensors": [asdict(tensor) for tensor in tensors],
            }
        )
        pools.append(
            FlatBlockPoolPlan(
                pool_id=pool_id,
                total_blocks=max(capacity_demands),
                bytes_per_block=bytes_per_block,
                storage_schema_hash=storage_schema_hash,
                tensors=tensors,
            )
        )

    plan_without_fingerprint = V4FlatMemoryPlan(
        max_total_tokens=max_total_tokens,
        pools=tuple(pools),
        scheduler_group_specs=scheduler_specs,
        runtime_metadata=runtime_metadata,
        plan_fingerprint="0" * 64,
    )
    return replace(
        plan_without_fingerprint,
        plan_fingerprint=_canonical_hash(
            canonical_v4_flat_memory_plan(plan_without_fingerprint)
        ),
    )


__all__ = [
    "V4FlatMemoryPlan",
    "V4FlatPlanAgreementRecord",
    "assert_v4_flat_plan_agreement",
    "build_v4_flat_memory_plan",
    "canonical_v4_flat_memory_plan",
    "make_v4_flat_plan_agreement_record",
]
