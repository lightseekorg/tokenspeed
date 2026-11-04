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

"""Decode monokernels: one persistent launch per decoder layer.

A monokernel runs one decoder layer's decode step as one persistent launch
that keeps one workgroup on every compute unit. What a leaf's launch covers
beyond its layer (such as the next layer's query and K/V write) is in its
leaf's README.

The registry entry is ``("monokernel", "prepare")``, selected by the
``recipe`` trait. A leaf is a class: its static ``prepare_state(...)``
allocates the group's shared state, and its constructor
``(plan_keys=, weights=, state=)`` compiles every plan variant it serves and
returns a :class:`MonokernelLayer` that takes the per-step calls. Vendor
kernels live in private, solution-named packages (``_gluon/`` for gfx950).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    tensor_format,
)

LAYER_KINDS = ("sliding", "full")


@dataclass(frozen=True)
class PlanKey:
    """The compile-time shape of one monokernel launch.

    Attributes:
        requests: Requests in the step (rows = requests x verify width).
        layer_kind: This layer's attention kind, one of :data:`LAYER_KINDS`.
        next_kind: The next layer's attention kind, or ``"regular"`` when the
            next layer runs on the regular path (the launch then ends at the
            FFN boundary and writes the regular boundary rows).
    """

    requests: int
    layer_kind: str
    next_kind: str

    def __post_init__(self) -> None:
        if self.layer_kind not in LAYER_KINDS:
            raise ValueError(
                f"layer_kind must be one of {LAYER_KINDS}, got {self.layer_kind!r}"
            )
        if self.next_kind not in LAYER_KINDS + ("regular",):
            raise ValueError(
                f"next_kind must be one of {LAYER_KINDS + ('regular',)}, got {self.next_kind!r}"
            )


# BF16 dense weights and activations, MXFP4 experts read in place from the
# MoE banks, FP8 KV cache.
MONOKERNEL_SIGNATURE = format_signature(
    dense=dense_tensor_format(torch.bfloat16),
    experts=tensor_format("mxfp4", torch.uint8),
    kv_cache=dense_tensor_format(torch.float8_e4m3fn),
)


class MonokernelLayer(Protocol):
    """A prepared monokernel layer (one per decoder layer of a model)."""

    def decline_reasons(self, requests: int) -> list[str]:
        """Rank-local reasons this layer cannot serve ``requests`` requests
        (compile, residency, memory checks); empty when it can. The caller
        agrees on them across the group before capture."""

    def forward(self, prepared_query, residual, *, positions, cache, next_prologue):
        """Run this layer and the next layer's QKV projection and prologue.

        Args:
            prepared_query: ``(q, output_gate)`` from the attention prologue.
            residual: FP32 residual rows, updated in place.
            positions: Positions of the step's rows.
            cache: This layer's decode KV view.
            next_prologue: The next layer's prologue target (KV cache, norms,
                rotary, query dtype).

        Returns:
            ``(next_prepared_query, residual)``.
        """

    def forward_last(self, prepared_query, residual, *, positions, cache, next_norm):
        """Run the last monokernel layer of a range, whose next layer is regular.

        Args:
            next_norm: ``(weight, eps, weight_offset)`` of the next layer's
                input norm. The other arguments are as in :meth:`forward`.

        Returns:
            ``(normed_rows, residual)`` in the regular boundary layout.
        """


def _select(recipe: str, traits: dict[str, Any]):
    try:
        return select_kernel(
            "monokernel",
            "prepare",
            MONOKERNEL_SIGNATURE,
            traits={"recipe": recipe, **traits},
        )
    except NoKernelFoundError:
        return None


def prepare_monokernel_state(
    *, recipe, group, rank_in_group, max_requests, failure_record_path, device
):
    """Allocate the per-group state the monokernel layers of one model share.

    It holds the launch state (per-workgroup epochs, role claims, the uncached
    flag arena, the host copy of the failure record) and the cross-GPU
    regions. Every rank calls it once per group, before capture. Selection
    uses the recipe alone, since layers whose shape traits differ share the
    state.

    Args:
        recipe: Model recipe, such as ``"muse_spark"``.
        group: The attention tensor-parallel group the layers exchange over.
        rank_in_group: This rank's index in ``group``.
        max_requests: The most requests a step serves.
        failure_record_path: File that backs the host copy of the failure
            record, so the record outlives the process.
        device: This rank's device.

    Returns:
        The state, or ``None`` when no leaf serves ``recipe`` on this platform.
    """
    kernel = _select(recipe, {})
    if kernel is None:
        return None
    return kernel.impl.prepare_state(
        group=group,
        rank_in_group=rank_in_group,
        max_requests=max_requests,
        failure_record_path=failure_record_path,
        device=device,
    )


def prepare_monokernel_layer(
    *, recipe, plan_keys, traits, weights, state
) -> MonokernelLayer | None:
    """Prepare one monokernel layer: compile every plan variant in
    ``plan_keys`` and check it before capture.

    Args:
        recipe: Model recipe, such as ``"muse_spark"``.
        plan_keys: The :class:`PlanKey` values this layer may serve.
        traits: Shape traits for selection (verify width, TP and EP sizes,
            KV cache dtype).
        weights: The layer's weights, read in place (a recipe-defined dict).
        state: From :func:`prepare_monokernel_state`.

    Returns:
        The prepared layer, or ``None`` when no leaf serves the recipe.
    """
    kernel = _select(recipe, traits)
    if kernel is None or state is None:
        return None
    return kernel.impl(plan_keys=plan_keys, weights=weights, state=state)


def monokernel_state_heap_bytes(*, recipe, world_size, max_requests) -> int:
    """Communication-heap bytes :func:`prepare_monokernel_state` allocates.

    Args:
        recipe: Model recipe, such as ``"muse_spark"``.
        world_size: Ranks in the group.
        max_requests: The most requests a step serves.

    Returns:
        The bytes to reserve when the group's heap is created, or 0 when no
        leaf serves ``recipe``.
    """
    kernel = _select(recipe, {})
    if kernel is None:
        return 0
    return kernel.impl.heap_bytes(world_size=world_size, max_requests=max_requests)


# Backend registration (side-effect import).
import tokenspeed_kernel.ops.monokernel.gluon  # noqa: E402,F401
