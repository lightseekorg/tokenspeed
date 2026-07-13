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

"""Shared result and enum types for model execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.flat_kv_progress import FlatKVExecutionEvidence
    from tokenspeed.runtime.grammar.capturable_grammar import (
        GrammarStepCompletion,
    )


def _require_pod_int(
    value: Any,
    *,
    field_name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    """Accept only a host Python int, never a tensor/device scalar.

    In particular, do not use ``int(value)`` here: converting a CUDA tensor
    would synchronize and would smuggle a device-backed value across the
    scheduler completion boundary.
    """
    if type(value) is not int:
        raise TypeError(
            f"{field_name} must be a host Python int, got {type(value).__name__}"
        )
    if value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}, got {value}")
    return value


def _pod_field(raw: Any, name: str) -> Any:
    if isinstance(raw, Mapping):
        if name not in raw:
            raise ValueError(f"flat KV completion input is missing {name!r}")
        return raw[name]
    if not hasattr(raw, name):
        raise ValueError(f"flat KV completion input is missing {name!r}")
    return getattr(raw, name)


@dataclass(frozen=True, slots=True)
class FlatKVCompletionInput:
    """Schema-free per-request dispatch metadata awaiting the result fence.

    This is not a completion: it contains no accepted end and must never be
    sent to the scheduler.  In particular it contains no producer domain mask
    and no group watermark to echo.  The active runtime cache plan plus
    sync-gated execution evidence are the producer-progress authority.
    """

    request_id: str
    table_generation: int
    dispatch_seq: int
    dispatch_raw_start: int
    dispatch_raw_end: int
    protected_raw_end: int

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError("flat KV completion request_id must be a non-empty str")
        _require_pod_int(
            self.table_generation,
            field_name="table_generation",
            maximum=0xFFFFFFFFFFFFFFFF,
        )
        _require_pod_int(
            self.dispatch_seq,
            field_name="dispatch_seq",
            maximum=0xFFFFFFFFFFFFFFFF,
        )
        _require_pod_int(
            self.dispatch_raw_start,
            field_name="dispatch_raw_start",
            maximum=0x7FFFFFFF,
        )
        _require_pod_int(
            self.dispatch_raw_end,
            field_name="dispatch_raw_end",
            maximum=0x7FFFFFFF,
        )
        _require_pod_int(
            self.protected_raw_end,
            field_name="protected_raw_end",
            maximum=0x7FFFFFFF,
        )
        if self.dispatch_raw_start > self.dispatch_raw_end:
            raise ValueError(
                "dispatch_raw_start must not exceed dispatch_raw_end: "
                f"{self.dispatch_raw_start} > {self.dispatch_raw_end}"
            )
        if self.protected_raw_end < self.dispatch_raw_end:
            raise ValueError(
                "protected_raw_end must not precede dispatch_raw_end: "
                f"{self.protected_raw_end} < {self.dispatch_raw_end}"
            )

    @classmethod
    def from_raw(cls, raw: Any) -> FlatKVCompletionInput:
        if type(raw) is cls:
            return raw
        if (isinstance(raw, Mapping) and "groups" in raw) or (
            not isinstance(raw, Mapping) and hasattr(raw, "groups")
        ):
            raise ValueError(
                "flat KV completion dispatch input must not contain group schema"
            )
        return cls(
            request_id=_pod_field(raw, "request_id"),
            table_generation=_pod_field(raw, "table_generation"),
            dispatch_seq=_pod_field(raw, "dispatch_seq"),
            dispatch_raw_start=_pod_field(raw, "dispatch_raw_start"),
            dispatch_raw_end=_pod_field(raw, "dispatch_raw_end"),
            protected_raw_end=_pod_field(raw, "protected_raw_end"),
        )


@dataclass(frozen=True, slots=True)
class FlatKVGroupCompletion:
    """Ready-only packed producer-domain progress for one cache group."""

    group_id: str
    completed_domain_mask: int
    domain_valid_ends: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.group_id) is not str or not self.group_id:
            raise ValueError("flat KV completion group_id must be a non-empty str")
        mask = _require_pod_int(
            self.completed_domain_mask,
            field_name=f"{self.group_id}.completed_domain_mask",
            maximum=0xFFFFFFFF,
        )
        if mask == 0:
            raise ValueError("flat KV completed_domain_mask must be non-zero")
        if type(self.domain_valid_ends) is not tuple:
            raise TypeError("flat KV domain_valid_ends must be a tuple")
        expected_size = mask.bit_count()
        if len(self.domain_valid_ends) != expected_size:
            raise ValueError(
                f"{self.group_id}.domain_valid_ends must contain one value per "
                f"set domain bit: {len(self.domain_valid_ends)} != {expected_size}"
            )
        for index, valid_end in enumerate(self.domain_valid_ends):
            _require_pod_int(
                valid_end,
                field_name=f"{self.group_id}.domain_valid_ends[{index}]",
                maximum=0x7FFFFFFF,
            )

    def to_pod(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "completed_domain_mask": self.completed_domain_mask,
            "domain_valid_ends": list(self.domain_valid_ends),
        }


@dataclass(frozen=True, slots=True)
class FlatKVCompletion:
    """Fence-ready scheduler payload containing host primitives only."""

    request_id: str
    table_generation: int
    dispatch_seq: int
    accepted_raw_end: int
    protected_raw_end: int
    groups: tuple[FlatKVGroupCompletion, ...]

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError("flat KV completion request_id must be a non-empty str")
        _require_pod_int(
            self.table_generation,
            field_name="table_generation",
            maximum=0xFFFFFFFFFFFFFFFF,
        )
        _require_pod_int(
            self.dispatch_seq,
            field_name="dispatch_seq",
            maximum=0xFFFFFFFFFFFFFFFF,
        )
        _require_pod_int(
            self.accepted_raw_end,
            field_name="accepted_raw_end",
            maximum=0x7FFFFFFF,
        )
        _require_pod_int(
            self.protected_raw_end,
            field_name="protected_raw_end",
            maximum=0x7FFFFFFF,
        )
        if self.accepted_raw_end > self.protected_raw_end:
            raise ValueError(
                "accepted_raw_end must not exceed protected_raw_end: "
                f"{self.accepted_raw_end} > {self.protected_raw_end}"
            )
        if not self.groups:
            raise ValueError("flat KV completion must contain at least one group")
        if type(self.groups) is not tuple:
            raise TypeError("flat KV completion groups must be a tuple")
        if any(type(group) is not FlatKVGroupCompletion for group in self.groups):
            raise TypeError("flat KV completion groups must be POD group completions")
        group_ids = [group.group_id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError(f"flat KV completion has duplicate groups: {group_ids}")

    def to_pod(self) -> dict[str, Any]:
        """Return the exact primitive container expected by the event bridge."""
        return {
            "request_id": self.request_id,
            "table_generation": self.table_generation,
            "dispatch_seq": self.dispatch_seq,
            "accepted_raw_end": self.accepted_raw_end,
            "protected_raw_end": self.protected_raw_end,
            "groups": [group.to_pod() for group in self.groups],
        }

    def rewind_to_host_accepted_tokens(
        self, *, device_accepted_tokens: int, host_accepted_tokens: int
    ) -> FlatKVCompletion:
        """Return progress aligned with tokens retained by host termination.

        Decode completion materialization advances ``accepted_raw_end`` by the
        device-reported accepted count. Host EOS, stop, max-length, or grammar
        handling can later retain a shorter prefix. Group watermarks continue
        to describe the producer's true writes; C++ clamps them to this revised
        accepted end before publication.

        Args:
            device_accepted_tokens: Accepted token count used when this
                completion was materialized.
            host_accepted_tokens: Prefix of that count retained after host
                terminal filtering.

        Returns:
            This immutable completion when the counts agree, otherwise a copy
            whose accepted raw end is rewound to the host-retained prefix.
        """
        device_accepted_tokens = _require_pod_int(
            device_accepted_tokens,
            field_name="device_accepted_tokens",
            maximum=0x7FFFFFFF,
        )
        host_accepted_tokens = _require_pod_int(
            host_accepted_tokens,
            field_name="host_accepted_tokens",
            maximum=device_accepted_tokens,
        )
        dispatch_raw_start = self.accepted_raw_end - device_accepted_tokens
        if dispatch_raw_start < 0:
            raise ValueError(
                "device accepted token count exceeds the completion accepted end"
            )
        if host_accepted_tokens == device_accepted_tokens:
            return self
        return replace(
            self,
            accepted_raw_end=dispatch_raw_start + host_accepted_tokens,
        )


@dataclass
class ModelExecutionResult:
    """
    Result of model execution returned to scheduler.

    This is the output from the Python executor back to the C++ scheduler.

    Attributes:
        output_tokens: Sampled token IDs
        output_logits: Output logits (if requested)
        output_lengths: Number of tokens generated per request (for spec decoding)
    """

    output_tokens: torch.Tensor
    copy_event: torch.cuda.Event | None = None
    output_logits: torch.Tensor | None = None
    output_lengths: torch.Tensor | None = None
    grammar_completion: GrammarStepCompletion | None = None
    # Per-position logprob of the sampled token, same layout as output_tokens.
    # Populated unconditionally by the sampling backend so it's always
    # available if any request asks for it.
    output_logprobs: torch.Tensor | None = None
    # Optional next-round input rows captured for PD prefill data-plane handoff.
    next_input_ids: torch.Tensor | None = None
    # Per-request NaN-guard flags (int32, [bs]); None when the guard is disabled.
    output_nan_flags: torch.Tensor | None = None
    # Host orchestration proof for the enqueued producer forwards.  It is
    # deliberately private: only sync-gated materialization may expose PODs.
    _flat_kv_execution_evidence: FlatKVExecutionEvidence | None = field(
        default=None, repr=False
    )
    # Ready-only host-POD payloads. init=False prevents executor code from
    # constructing a completion before copy_event and auxiliary KV producers
    # have crossed the execution-result fence.
    flat_kv_completions: tuple[FlatKVCompletion, ...] | None = field(
        default=None, init=False
    )
    _is_synchronized: bool = field(default=False, init=False, repr=False)

    def sync(self) -> None:
        if self.copy_event is None:
            raise RuntimeError("copy_event is required before synchronizing results.")
        self.copy_event.synchronize()
        self._is_synchronized = True

    @property
    def is_synchronized(self) -> bool:
        return self._is_synchronized

    def materialize_flat_kv_completions(
        self, forward_op: Any
    ) -> tuple[FlatKVCompletion, ...]:
        """Build ready-only flat KV completions from dispatch metadata.

        The execution evidence owns the already-parsed schema-free dispatch
        identity/ranges captured before the forward. Group/domain ends come
        from the active plan plus that evidence after the complete target/draft
        orchestration returned. Extend rows accept every input/KV-writing token,
        while decode rows add the synchronized accepted count to
        ``dispatch_raw_start``.
        """
        if not self._is_synchronized:
            raise RuntimeError(
                "flat KV completions are ready only after ModelExecutionResult.sync()"
            )
        if self.flat_kv_completions is not None:
            return self.flat_kv_completions

        evidence = self._flat_kv_execution_evidence
        if evidence is None:
            self.flat_kv_completions = ()
            return self.flat_kv_completions
        completion_inputs = evidence.inputs
        if not completion_inputs:
            self.flat_kv_completions = ()
            return self.flat_kv_completions

        request_ids = list(forward_op.request_ids)
        if len(completion_inputs) != len(request_ids):
            raise ValueError(
                "flat_kv_completion_inputs row count differs from forward op: "
                f"{len(completion_inputs)} != {len(request_ids)}"
            )

        if self.output_lengths is None:
            raise RuntimeError(
                "output_lengths is required to materialize flat KV completions"
            )
        accepted_lengths = self.output_lengths.tolist()
        if len(accepted_lengths) != len(request_ids):
            raise ValueError(
                "output_lengths row count differs from forward op: "
                f"{len(accepted_lengths)} != {len(request_ids)}"
            )
        input_lengths = list(forward_op.input_lengths)
        if len(input_lengths) != len(request_ids):
            raise ValueError(
                "input_lengths row count differs from forward op: "
                f"{len(input_lengths)} != {len(request_ids)}"
            )
        num_extends = _require_pod_int(
            forward_op.num_extends(), field_name="num_extends"
        )
        if num_extends > len(request_ids):
            raise ValueError(
                f"num_extends={num_extends} exceeds batch size {len(request_ids)}"
            )

        accepted_raw_ends: list[int] = []
        for row, (request_id, completion_input) in enumerate(
            zip(request_ids, completion_inputs)
        ):
            if completion_input.request_id != request_id:
                raise ValueError(
                    "flat KV completion input row/request mismatch: "
                    f"row {row} is {request_id!r}, payload is "
                    f"{completion_input.request_id!r}"
                )
            input_length = _require_pod_int(
                input_lengths[row], field_name=f"input_length[{request_id}]"
            )
            dispatch_width = (
                completion_input.dispatch_raw_end - completion_input.dispatch_raw_start
            )
            if input_length != dispatch_width:
                raise ValueError(
                    f"flat KV completion for {request_id!r} has input length "
                    f"{input_length}, but dispatch interval width is "
                    f"{dispatch_width}"
                )
            if row < num_extends:
                accepted_raw_end = completion_input.dispatch_raw_end
            else:
                accepted_length = _require_pod_int(
                    accepted_lengths[row],
                    field_name=f"accepted_length[{request_id}]",
                )
                accepted_raw_end = completion_input.dispatch_raw_start + accepted_length
            if not (
                completion_input.dispatch_raw_start
                <= accepted_raw_end
                <= completion_input.dispatch_raw_end
            ):
                raise ValueError(
                    f"flat KV completion for {request_id!r} accepts through "
                    f"{accepted_raw_end}, outside dispatch interval "
                    f"[{completion_input.dispatch_raw_start}, "
                    f"{completion_input.dispatch_raw_end}]"
                )
            accepted_raw_ends.append(accepted_raw_end)

        groups_by_row = evidence.materialize_group_completions(accepted_raw_ends)
        if len(groups_by_row) != len(completion_inputs):
            raise ValueError("flat KV execution evidence returned the wrong row count")

        completions: list[FlatKVCompletion] = []
        for request_id, completion_input, accepted_raw_end, groups in zip(
            request_ids,
            completion_inputs,
            accepted_raw_ends,
            groups_by_row,
        ):
            completions.append(
                FlatKVCompletion(
                    request_id=request_id,
                    table_generation=completion_input.table_generation,
                    dispatch_seq=completion_input.dispatch_seq,
                    accepted_raw_end=accepted_raw_end,
                    protected_raw_end=completion_input.protected_raw_end,
                    groups=groups,
                )
            )

        self.flat_kv_completions = tuple(completions)
        return self.flat_kv_completions
