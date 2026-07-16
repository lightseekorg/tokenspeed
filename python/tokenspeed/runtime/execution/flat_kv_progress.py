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

"""Static flat-KV producer schema and per-forward execution evidence.

The dispatch ABI carries only identity and raw ranges.  Producer domains come
from the active runtime cache plan.  A successful model orchestration proves
that the deterministic full-layer write path was enqueued; existing table/slot
guards prove that path cannot silently skip a required allocation.  No
per-layer progress tensors or hot-path writer instrumentation are needed.

The evidence remains internal until ``ModelExecutionResult.sync()``.  Only then
are conservative raw-exclusive ends derived: target domains round down the
dispatched end to the group's entry stride and draft domains round down the
accepted end.  An owner is complete only when every planned layer plane and
auxiliary-stream write in that domain was enqueued.  A required owner path that
did not establish that continuous prefix fails closed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tokenspeed.runtime.configs.flat_kv_contract import (
    V4_PRODUCER_DRAFT_INDEXER,
    V4_PRODUCER_DRAFT_MAIN,
    V4_PRODUCER_TARGET_INDEXER,
    V4_PRODUCER_TARGET_MAIN,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.flat_memory_plan import V4FlatMemoryPlan
    from tokenspeed.runtime.execution.types import (
        FlatKVCompletionInput,
        FlatKVGroupCompletion,
    )

_UINT32_MAX = 0xFFFFFFFF
_TARGET_DOMAIN_MASK = V4_PRODUCER_TARGET_MAIN | V4_PRODUCER_TARGET_INDEXER
_DRAFT_DOMAIN_MASK = V4_PRODUCER_DRAFT_MAIN | V4_PRODUCER_DRAFT_INDEXER
_V4_DOMAIN_MASK = _TARGET_DOMAIN_MASK | _DRAFT_DOMAIN_MASK
_MAIN_COMPONENTS = frozenset({"swa_kv", "compressed_kv", "compressor_state"})
_INDEXER_COMPONENTS = frozenset({"indexer_kv", "indexer_state"})


def _domain_bits(mask: int) -> tuple[int, ...]:
    return tuple(1 << bit for bit in range(32) if mask & (1 << bit))


def _required_mask(raw_mask: Any, *, group_id: str) -> int:
    if type(raw_mask) is not int:
        raise TypeError(
            f"{group_id}.required_producer_domain_mask must be a host Python int"
        )
    if raw_mask < 0 or raw_mask > _UINT32_MAX:
        raise ValueError(f"{group_id}.required_producer_domain_mask is outside uint32")
    # Legacy/generic flat specs predate producer domains.  Both scheduler and
    # runtime canonicalize zero to the single target producer bit.
    return raw_mask or V4_PRODUCER_TARGET_MAIN


def _required_uint32(value: Any, *, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be a host Python int")
    if value < 0 or value > _UINT32_MAX:
        raise ValueError(f"{field_name} is outside uint32")
    return value


def _positive_stride(value: Any, *, group_id: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{group_id}.entry_stride_tokens must be a positive int")
    return value


def _round_down(raw_end: int, stride: int) -> int:
    return raw_end // stride * stride


@dataclass(frozen=True, slots=True)
class FlatKVGroupProgressSchema:
    """One scheduler group and its runtime-owned producer domains."""

    group_id: str
    required_domain_mask: int
    target_domain_mask: int
    draft_domain_mask: int
    entry_stride_tokens: int
    domain_bits: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("flat KV progress group_id must be non-empty")
        if self.required_domain_mask == 0:
            raise ValueError("flat KV progress required mask must be non-zero")
        if self.target_domain_mask & self.draft_domain_mask:
            raise ValueError("flat KV target and draft producer masks overlap")
        if (
            self.target_domain_mask | self.draft_domain_mask
        ) != self.required_domain_mask:
            raise ValueError("flat KV producer ownership does not cover required mask")
        _positive_stride(self.entry_stride_tokens, group_id=self.group_id)
        # Cache the ABI packing order once with the static schema. Completion
        # materialization must not rescan all 32 possible bits for every row.
        object.__setattr__(self, "domain_bits", _domain_bits(self.required_domain_mask))


@dataclass(frozen=True, slots=True)
class FlatKVProgressSchema:
    """Static producer-domain schema derived from the active runtime plan."""

    groups: tuple[FlatKVGroupProgressSchema, ...]
    target_domain_mask: int = field(init=False, repr=False)
    draft_domain_mask: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("flat KV progress schema must contain groups")
        group_ids = [group.group_id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("flat KV progress schema contains duplicate groups")
        target_mask = 0
        draft_mask = 0
        for group in self.groups:
            target_mask |= group.target_domain_mask
            draft_mask |= group.draft_domain_mask
        object.__setattr__(self, "target_domain_mask", target_mask)
        object.__setattr__(self, "draft_domain_mask", draft_mask)

    @classmethod
    def from_runtime_pool(cls, pool: Any) -> FlatKVProgressSchema:
        plan = getattr(pool, "flat_memory_plan", None)
        if plan is not None:
            return cls.from_v4_plan(plan)
        specs = getattr(pool, "scheduler_group_specs", None)
        if not specs:
            specs = getattr(pool, "paged_cache_group_specs", None)
        if not specs:
            raise ValueError("flat KV progress requires runtime group specs")
        groups = []
        for spec in specs:
            group_id = str(spec.group_id)
            mask = _required_mask(
                getattr(spec, "required_producer_domain_mask", 0),
                group_id=group_id,
            )
            if len(_domain_bits(mask)) != 1:
                raise ValueError(
                    "generic flat KV groups support one producer domain, got "
                    f"{group_id!r} mask={mask:#x}"
                )
            groups.append(
                FlatKVGroupProgressSchema(
                    group_id=group_id,
                    required_domain_mask=mask,
                    target_domain_mask=mask,
                    draft_domain_mask=0,
                    entry_stride_tokens=_positive_stride(
                        spec.entry_stride_tokens,
                        group_id=group_id,
                    ),
                )
            )
        return cls(tuple(groups))

    @classmethod
    def from_v4_plan(cls, plan: V4FlatMemoryPlan) -> FlatKVProgressSchema:
        specs = tuple(plan.scheduler_group_specs)
        components = tuple(
            component for pool_plan in plan.pools for component in pool_plan.tensors
        )
        observed_domains: dict[str, int] = {}
        draft_layers = set()
        for component in components:
            owner = str(component.owner)
            name = str(component.component)
            if owner == "target":
                main_bit = V4_PRODUCER_TARGET_MAIN
                indexer_bit = V4_PRODUCER_TARGET_INDEXER
            elif owner == "draft":
                main_bit = V4_PRODUCER_DRAFT_MAIN
                indexer_bit = V4_PRODUCER_DRAFT_INDEXER
                draft_layers.add(int(component.layer))
            else:
                raise ValueError(f"unsupported flat KV owner {owner!r}")
            if name in _MAIN_COMPONENTS:
                bit = main_bit
            elif name in _INDEXER_COMPONENTS:
                bit = indexer_bit
            else:
                raise ValueError(f"unknown V4 flat component {name!r}")
            group_id = str(component.group_id)
            observed_domains[group_id] = observed_domains.get(group_id, 0) | bit

        groups = []
        for spec in specs:
            group_id = str(spec.group_id)
            mask = _required_mask(
                spec.required_producer_domain_mask,
                group_id=group_id,
            )
            if mask & ~_V4_DOMAIN_MASK:
                raise ValueError(
                    f"V4 flat group {group_id!r} has unknown domains {mask:#x}"
                )
            observed = observed_domains.get(group_id, 0)
            # The plan builder validates every component plane for every
            # layer. Here the completion bridge only needs to prove that each
            # required owner/domain has a producer in this group.
            if observed & mask != mask:
                raise ValueError(
                    f"V4 flat group {group_id!r} plan is missing required "
                    f"producer planes: required={mask:#x}, observed={observed:#x}"
                )
            groups.append(
                FlatKVGroupProgressSchema(
                    group_id=group_id,
                    required_domain_mask=mask,
                    target_domain_mask=mask & _TARGET_DOMAIN_MASK,
                    draft_domain_mask=mask & _DRAFT_DOMAIN_MASK,
                    entry_stride_tokens=_positive_stride(
                        spec.entry_stride_tokens,
                        group_id=group_id,
                    ),
                )
            )
        if draft_layers and draft_layers != set(range(len(draft_layers))):
            raise ValueError("V4 flat draft layer ids must be contiguous from zero")
        if any(group.draft_domain_mask for group in groups) and not draft_layers:
            raise ValueError("V4 flat draft domains require draft model layers")
        return cls(tuple(groups))


@dataclass(frozen=True, slots=True)
class FlatKVExecutionEvidence:
    """Host-only proof that required continuous producer prefixes were enqueued."""

    schema: FlatKVProgressSchema
    inputs: tuple[FlatKVCompletionInput, ...]
    completed_domain_mask: int

    def __post_init__(self) -> None:
        completed = _required_uint32(
            self.completed_domain_mask,
            field_name="completed_domain_mask",
        )
        known = self.schema.target_domain_mask | self.schema.draft_domain_mask
        if completed & ~known:
            raise ValueError(
                "completed flat KV domains are absent from the active schema"
            )

    def materialize_group_completions(
        self,
        accepted_raw_ends: Sequence[int],
    ) -> tuple[tuple[FlatKVGroupCompletion, ...], ...]:
        from tokenspeed.runtime.execution.types import FlatKVGroupCompletion

        inputs = self.inputs
        if len(accepted_raw_ends) != len(inputs):
            raise ValueError("flat KV accepted-end rows differ from dispatch")
        output = []
        for completion_input, accepted_end in zip(
            inputs,
            accepted_raw_ends,
        ):
            groups = []
            no_op_dispatch = (
                completion_input.dispatch_raw_start == completion_input.dispatch_raw_end
            )
            for group in self.schema.groups:
                missing = group.required_domain_mask & ~self.completed_domain_mask
                if missing and not no_op_dispatch:
                    raise RuntimeError(
                        f"flat KV completion for {completion_input.request_id!r} "
                        f"is missing continuous producer domains {missing:#x} for "
                        f"group {group.group_id!r}"
                    )
                packed_ends = []
                for bit in group.domain_bits:
                    if no_op_dispatch:
                        # The C++ ledger seeds each domain at this same aligned
                        # dispatch start. No owner needs to run when no token is
                        # dispatched, but the schema-complete payload remains
                        # safe to merge.
                        raw_end = completion_input.dispatch_raw_end
                    elif bit & group.draft_domain_mask:
                        raw_end = accepted_end
                    else:
                        raw_end = completion_input.dispatch_raw_end
                    packed_ends.append(_round_down(raw_end, group.entry_stride_tokens))
                groups.append(
                    FlatKVGroupCompletion(
                        group_id=group.group_id,
                        completed_domain_mask=group.required_domain_mask,
                        domain_valid_ends=tuple(packed_ends),
                    )
                )
            output.append(tuple(groups))
        return tuple(output)


class FlatKVExecutionTracker:
    """Small per-forward host state; it never enters model writer hot paths."""

    def __init__(self, schema: FlatKVProgressSchema) -> None:
        self.schema = schema
        self._inputs: tuple[FlatKVCompletionInput, ...] = ()

    def begin_dispatch(
        self,
        raw_inputs: Any,
        *,
        request_ids: Sequence[str],
    ) -> None:
        from tokenspeed.runtime.execution.types import FlatKVCompletionInput

        if raw_inputs is None or len(raw_inputs) == 0:
            self._inputs = ()
            return
        if isinstance(raw_inputs, (str, bytes)) or not isinstance(raw_inputs, Sequence):
            raise TypeError("flat_kv_completion_inputs must be a host sequence")
        if len(raw_inputs) != len(request_ids):
            raise ValueError("flat KV completion input rows differ from requests")
        inputs = tuple(FlatKVCompletionInput.from_raw(raw) for raw in raw_inputs)
        for row, (request_id, completion_input) in enumerate(zip(request_ids, inputs)):
            if completion_input.request_id != request_id:
                raise ValueError(
                    "flat KV completion input row/request mismatch: "
                    f"row {row} is {request_id!r}, payload is "
                    f"{completion_input.request_id!r}"
                )
        self._inputs = inputs

    def finish_dispatch(
        self,
        *,
        target_forward_enqueued: bool,
        draft_continuous_prefix_enqueued: bool,
    ) -> FlatKVExecutionEvidence | None:
        """Freeze owner-domain evidence after all producer enqueues return.

        ``draft_continuous_prefix_enqueued`` is an orchestration fact, not a
        device result.  It may become true only after every planned draft
        layer/domain has enqueued its accepted-prefix writes and any auxiliary
        producer stream has joined the execution-result fence.  The existing
        synchronized accepted-length copy later supplies each row's exact end.
        """
        if not self._inputs:
            return None
        for field_name, value in (
            ("target_forward_enqueued", target_forward_enqueued),
            (
                "draft_continuous_prefix_enqueued",
                draft_continuous_prefix_enqueued,
            ),
        ):
            if type(value) is not bool:
                raise TypeError(f"{field_name} must be a host Python bool")
        completed_mask = 0
        if target_forward_enqueued:
            completed_mask |= self.schema.target_domain_mask
        if draft_continuous_prefix_enqueued:
            completed_mask |= self.schema.draft_domain_mask
        evidence = FlatKVExecutionEvidence(
            schema=self.schema,
            inputs=self._inputs,
            completed_domain_mask=completed_mask,
        )
        self._inputs = ()
        return evidence


__all__ = [
    "FlatKVExecutionEvidence",
    "FlatKVExecutionTracker",
    "FlatKVGroupProgressSchema",
    "FlatKVProgressSchema",
]
