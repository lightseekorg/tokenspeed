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

"""The Inkling2 layer decode monokernel (``docs/design/monokernel.md``).

The runtime packs a local layer (a mid one, or the first of an
attention-residual block) once after loading
(:func:`prepare_inkling2_local_layer`), or a full attention layer
(:func:`prepare_inkling2_full_layer`), and runs its verify step as one
persistent launch a rank (:func:`inkling2_local_layer_decode`), at any of the
row counts it packed the layer for: one plan a row count, the same packed
weights. The prepare function returns None when no monokernel serves the
layer on this platform.

A layer of a looped core runs once a pass. Its first pass's visit is one
launch, as any layer's. A later pass gates the previous pass's message into
the layer's inputs, which the regular path computes between two launches of
the visit (``part``): the attention part takes the normalized Q, K and V
inputs and writes ``A``, the MoE part takes the normalized MoE input and
writes ``M``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

# The launches of a visit: the whole layer, or a later pass's two parts.
PARTS = ("layer", "attention", "mlp")

__all__ = [
    "PARTS",
    "LocalLayer",
    "inkling2_full_layer_splits",
    "inkling2_local_layer_decode",
    "prepare_inkling2_full_layer",
    "prepare_inkling2_local_layer",
]


@dataclass(frozen=True)
class LocalLayer:
    """A local or full attention layer packed for its monokernel.

    Attributes:
        decode: The selected kernel.
        experts: The routed experts' format, ``"nvfp4"`` or ``"mxfp8"``.
        boundary: Whether the layer starts an attention-residual block: it
            stores the block and computes the new block set's partials.
        first: Whether the layer is layer 0: a boundary layer whose block
            starts from the embedding, with no block stored before it.
        full: Whether the layer is a full attention layer.
        later_passes: Whether the layer runs the later passes of a looped
            core (its launches may be the ``attention`` and ``mlp`` parts).
        weights: The packed weights (opaque to the caller).
        scratch_pairs: int64 mailbox pairs the largest launch needs. One
            buffer serves every layer of a step, at every row count.
    """

    decode: Any
    experts: str
    boundary: bool
    first: bool
    full: bool
    later_passes: bool
    weights: Any
    scratch_pairs: int


def _select(mode: str, rows: int, experts: str, override: str | None):
    signature = format_signature(h=dense_tensor_format(torch.bfloat16))
    return select_kernel(
        "monokernel",
        mode,
        signature,
        traits={"rows": rows, "experts": experts},
        override=override,
    )


def prepare_inkling2_local_layer(
    *,
    rows: tuple[int, ...],
    experts: str,
    boundary: bool,
    first: bool,
    later_passes: bool,
    tensors: dict[str, Any],
    override: str | None = None,
) -> LocalLayer | None:
    """Pack an Inkling2 local layer for its monokernel.

    Args:
        rows: The token rows of the launches, one count a verify batch size
            (the requests' verify blocks, one after another). One kernel
            must serve every count.
        experts: The routed experts' format, ``"nvfp4"`` or ``"mxfp8"``.
        boundary: Whether the layer is the first of an attention-residual
            block (layer ``6 k``), not a mid local layer.
        first: Whether the layer is layer 0 (a boundary layer whose
            attention input is the embedding's RMSNorm, not a mix).
        later_passes: Whether the layer is in a looped core, whose later
            passes project separate K and V sources.
        tensors: The layer's weights by role: ``qkvr``, ``wo``, ``router``
            (row-major BF16), ``router_bias``, ``head`` (the fields of the
            kernel's head weights: Q and rel clip scales, the rel logits
            projection, K and V conv taps), ``attn_taps``, ``mlp_taps``, and
            ``w13`` / ``w2``: each ``(codes, scales, amax, outer
            denominator)`` of the rank's NVFP4 experts, or ``(codes,
            exponents)`` of its MXFP8 experts.
        override: Optional registered kernel name to force.

    Returns:
        The packed layer, or None when no monokernel serves it here.
    """
    return _prepare(
        rows, experts, boundary, first, False, later_passes, tensors, override
    )


def prepare_inkling2_full_layer(
    *,
    rows: tuple[int, ...],
    experts: str,
    later_passes: bool,
    tensors: dict[str, Any],
    override: str | None = None,
) -> LocalLayer | None:
    """Pack an Inkling2 full attention layer for its monokernel, as
    :func:`prepare_inkling2_local_layer` (its ``qkvr`` holds the K and V
    conv tap rows after the local layer's, its ``head`` K and V taps are
    the static ones)."""
    return _prepare(rows, experts, False, False, True, later_passes, tensors, override)


def inkling2_full_layer_splits(max_context_len: int) -> int:
    """The KV splits of a full layer's launches at ``max_context_len``: the
    regular Triton decode's (``rel_mha_decode_with_kvcache``) count, which
    the launch follows bit for bit; the server fixes it at startup."""
    from tokenspeed_kernel.ops.attention.rmha.triton import _decode_split_count

    return _decode_split_count(max_context_len, -1)


def _prepare(
    rows: tuple[int, ...],
    experts: str,
    boundary: bool,
    first: bool,
    full: bool,
    later_passes: bool,
    tensors: dict[str, Any],
    override: str | None,
) -> LocalLayer | None:
    try:
        prepares = {
            _select("inkling2_local_layer_prepare", r, experts, override) for r in rows
        }
        decodes = {
            _select("inkling2_local_layer_decode", r, experts, override) for r in rows
        }
    except NoKernelFoundError:
        if override is not None:
            raise
        return None
    if len(prepares) != 1 or len(decodes) != 1:
        # One packing serves every row count only within one kernel.
        return None
    (prepare,), (decode,) = prepares, decodes
    if first and not boundary:
        raise ValueError("layer 0 starts an attention-residual block")
    packed = prepare(
        tensors=tensors,
        rows=rows,
        experts=experts,
        boundary=boundary,
        first=first,
        full=full,
        later_passes=later_passes,
    )
    if packed is None:
        return None
    weights, scratch_pairs = packed
    return LocalLayer(
        decode=decode,
        experts=experts,
        boundary=boundary,
        first=first,
        full=full,
        later_passes=later_passes,
        weights=weights,
        scratch_pairs=scratch_pairs,
    )


def inkling2_local_layer_decode(
    layer: LocalLayer,
    *,
    part: str,
    inputs: tuple[torch.Tensor, ...],
    rank: int,
    rows: int,
    chain: int,
    scratch: torch.Tensor,
    step: torch.Tensor,
    layer_index: int,
    parity: int,
    bases: torch.Tensor,
    terms: tuple[torch.Tensor, ...],
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    queries: tuple[tuple, tuple],
    sweep: tuple,
    full_splits: int,
    tau: torch.Tensor | None,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    locations: torch.Tensor,
    positions: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    ring: torch.Tensor,
    ring_slots: torch.Tensor,
) -> None:
    """One verify step of a packed layer on rank ``rank``.

    Args:
        layer: From :func:`prepare_inkling2_local_layer`.
        part: The launch (``PARTS``): ``layer`` for a whole visit, or a
            later pass's ``attention`` or ``mlp`` part (a layer packed with
            ``later_passes``).
        inputs: The part's normalized BF16 ``[rows, hidden]`` inputs: the
            Q, K and V sources of the attention part, the MoE input of the
            MoE part; ``()`` for a whole layer.
        rank, rows, chain: The attention TP rank; the launch's token rows (one
            of the counts the layer was packed for) and rows a request.
        scratch, step: The mailbox scratch on the TP group's symmetric heap
            and the int32 step counter (device; it changes between steps).
        layer_index, parity: The layer (tag) and the peer-region copy, which
            alternates from one launch to the next on every rank.
        bases: int64 ``[world]`` symmetric-heap bases of the TP group.
        terms: The previous layer's pending BF16 rows ``[rows, hidden]``;
            layer 0's embedding rows.
        outputs: ``(P, A, M)`` BF16 rows the layer writes; a boundary
            layer's ``P`` is its block's row of the stored blocks. The
            attention part writes only ``A``, the MoE part only ``M``.
        queries: ``(attention, mlp)`` depth-mix queries: ``(score weights,
            gain, acc, m, s)`` of the set partials, rows of this layer. A
            boundary layer's MLP query reads its partial from the sweep.
        sweep: A boundary layer's new set partials sweep: ``(blocks [n + 1,
            rows, hidden] BF16, int32 [1] n, the set's FP32 score weights
            [queries, hidden], and the outputs m, s [queries, rows], acc
            [queries, rows, hidden] FP32)``, ``n`` the blocks stored before
            the layer's; ``()`` for a mid layer.
        full_splits: A full layer's KV splits
            (:func:`inkling2_full_layer_splits`); 0 for a local layer.
        tau: A full layer's FP32 ``[rows]`` attention logit scales (the
            model's ``log_scaling_tau`` of the rows); None for a local layer.
        k_cache, v_cache: The layer's KV cache ``[slots, kv heads, head]``.
        locations: int32 ``[rows]`` cache slot of each row's token.
        positions: int64 ``[rows]`` positions.
        page_table, page_size: int32 ``[requests, pages]`` KV page table.
        ring, ring_slots: The layer's conv ring and each request's slot.
    """
    if layer.full != (full_splits > 0):
        raise ValueError("full_splits must be given for a full layer only")
    if part not in PARTS:
        raise ValueError(f"no launch part {part!r}")
    if part != "layer" and not layer.later_passes:
        raise ValueError("a later pass's part of a layer outside a looped core")
    if len(inputs) != {"layer": 0, "attention": 3, "mlp": 1}[part]:
        raise ValueError(f"the {part} launch takes {len(inputs)} inputs")
    layer.decode(
        part=part,
        inputs=inputs,
        weights=layer.weights,
        experts=layer.experts,
        rank=rank,
        rows=rows,
        chain=chain,
        scratch=scratch,
        step=step,
        layer_index=layer_index,
        parity=parity,
        bases=bases,
        terms=terms,
        outputs=outputs,
        queries=queries,
        boundary=layer.boundary,
        first=layer.first,
        full_splits=full_splits,
        sweep=sweep,
        tau=tau,
        k_cache=k_cache,
        v_cache=v_cache,
        locations=locations,
        positions=positions,
        page_table=page_table,
        page_size=page_size,
        ring=ring,
        ring_slots=ring_slots,
    )


import tokenspeed_kernel.ops.model.inkling2.monokernel.gluon  # noqa: E402,F401
