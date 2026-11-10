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

"""Muse Spark's decode monokernels: the target's decoder layers and its
DFlash2 draft's layers, draft head, and context write.

Each launch runs on every rank of the TP8 group at once, and the ranks
exchange values through the Iris symmetric heap. See
``docs/design/monokernel.md`` for the design and ``README.md`` in this
package for the ops.

The runtime packs each layer once after weight loading
(:func:`prepare_muse_spark_layer` and the draft's ``prepare_*``) and then
runs it per step (:func:`muse_spark_layer_decode` and the draft's
``*_decode``). A prepare function returns None when the platform has no
monokernel for the layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "FusedDecodeLayer",
    "MUSE_SPARK_PROJ_WORDS",
    "muse_spark_draft_context_decode",
    "muse_spark_draft_head_decode",
    "muse_spark_draft_head_scratch_pairs",
    "muse_spark_draft_layer_decode",
    "muse_spark_layer_decode",
    "muse_spark_stuck_record",
    "prepare_muse_spark_draft_context",
    "prepare_muse_spark_draft_head",
    "prepare_muse_spark_draft_layer",
    "prepare_muse_spark_layer",
]


@dataclass(frozen=True)
class FusedDecodeLayer:
    """A decoder layer packed for its monokernel.

    Attributes:
        decode: The selected kernel. The ``*_decode`` function calls it.
        weights: The kernel's packed weights (opaque to the caller).
        scratch_pairs: int64 mailbox pairs the kernel needs. One scratch
            buffer serves every layer of a step.
        late_weights: int32 words of the packed weights that the layer's
            launch reads last, for the launch before it to warm, or None.
    """

    decode: Any
    weights: Any
    scratch_pairs: int
    late_weights: torch.Tensor | None


def _select(mode: str, max_rows: int, solution: str | None):
    # One forced solution selects both the ``*_prepare`` and the ``*_decode``
    # op of a layer, each among that solution's kernels of its own mode; a
    # kernel name matches no solution and raises.
    signature = format_signature(h=dense_tensor_format(torch.bfloat16))
    return select_kernel(
        "monokernel", mode, signature, traits={"rows": max_rows}, solution=solution
    )


def _prepare(
    kind: str, max_rows: int, solution: str | None, **prepare_args: Any
) -> FusedDecodeLayer | None:
    """Select ``kind``'s prepare and decode kernels and pack the layer with
    ``prepare_args``; None when no kernel serves ``max_rows`` here and no
    solution was forced."""
    try:
        prepare = _select(f"{kind}_prepare", max_rows, solution)
        decode = _select(f"{kind}_decode", max_rows, solution)
    except NoKernelFoundError:
        if solution is not None:
            raise
        return None
    weights, scratch_pairs, late_weights = prepare(max_rows=max_rows, **prepare_args)
    return FusedDecodeLayer(
        decode=decode,
        weights=weights,
        scratch_pairs=scratch_pairs,
        late_weights=late_weights,
    )


def prepare_muse_spark_layer(
    *, tensors: dict[str, Any], max_rows: int, solution: str | None = None
) -> FusedDecodeLayer | None:
    """Pack one TP8 rank's Muse Spark decoder layer for its monokernel.

    Args:
        tensors: The rank's weights by role (see
            ``tokenspeed_kernel_amd...muse_spark.weights.pack_layer``).
            ``experts`` holds the routed experts' ``(w13, w13 scales, w2, w2
            scales)`` as the regular gluon MXFP4 MoE prepared them; the
            kernel reads their storage in place. ``experts_kernel`` names
            the regular MoE's apply kernel, whose weight layout that is.
        max_rows: The most token rows a launch will serve: up to two
            requests' rows, each 1 (decode) or ``k + 1`` (a speculative
            verify of ``k`` draft tokens), one request's after the other's.
            The scratch is sized for it.
        solution: Optional registered solution (such as ``"gluon"``) to
            force; its kernels must serve ``max_rows`` here, else this
            raises ``NoKernelFoundError``.

    Returns:
        The packed layer, or None when no monokernel serves ``max_rows``
        here.

    Raises:
        ValueError: A tensor, or the experts' layout, is not the kernel's.
    """
    return _prepare("muse_spark_layer", max_rows, solution, tensors=tensors)


# int32 words of one token row's q | k | v projection that a Muse Spark
# launch hands to the next (``muse_spark_layer_decode``'s ``proj_in`` /
# ``proj_out``): FP32 partial sums of two K halves of 1280 rows.
MUSE_SPARK_PROJ_WORDS = 2560


def muse_spark_layer_decode(
    layer: FusedDecodeLayer,
    *,
    h: torch.Tensor,
    a: torch.Tensor,
    h_out: torch.Tensor,
    a_out: torch.Tensor,
    seq_len: torch.Tensor,
    slot: torch.Tensor,
    page_table: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cos_sin: torch.Tensor,
    scratch: torch.Tensor,
    step: torch.Tensor,
    peer_bases: torch.Tensor,
    layer_index: int,
    chain: int,
    whole_rows: bool,
    tap: torch.Tensor | None,
    tap_slot: int | None,
    warm_next: FusedDecodeLayer | None,
    proj_in: torch.Tensor | None,
    proj_out: torch.Tensor | None,
    stuck_rec: torch.Tensor | None,
    compile_only: bool,
) -> None:
    """One step of a packed Muse Spark layer for the rows of one or more
    requests (a decode row each, or a speculative verify's consecutive
    tokens), on every rank of the TP8 group at once (the launches exchange
    values).

    Args:
        layer: From :func:`prepare_muse_spark_layer`.
        h: FP32 ``[rows, 1024]``: the rank's owned residual columns
            (``1024 rank ..``) before the layer.
        h_out: FP32: the residual after the layer, whole ``[rows, 8192]``
            rows on every rank with ``whole_rows``, else the owned columns
            ``[rows, 1024]``.
        a, a_out: BF16 ``[rows, 8192]``: the layer's attention input (normed
            with its input norm) and the next layer's. Outputs alias no
            input.
        seq_len, slot: int32 ``[rows]``: each row's length including its
            token, and the token's KV cache slot. Row ``t`` of request ``b``
            is the token at position ``seq_len[b chain] - 1 + t``.
        page_table: int32 ``[requests, pages]`` (or ``[pages]`` for one
            request): each request's page table of the layer's cache group
            (64 tokens a page, absolute positions).
        k_cache, v_cache: The layer's BF16 ``[slots, 2, 64]`` caches.
        cos_sin: FP32 ``[positions, 64]`` rotary table (cos, then sin).
        scratch: int64 mailbox scratch of at least ``layer.scratch_pairs``,
            on the Iris symmetric heap (the same offset on every rank).
        step: int32 ``[1]`` counter that changes once per forward (CUDA graph
            safe: the kernel reads it on device).
        peer_bases: int64 ``[8]`` Iris heap bases of the group's ranks.
        layer_index: Decoder layer index, below 64. The launches of one
            forward alternate its parity.
        chain: Tokens a request: 1 for decode, ``k + 1`` for a verify of
            ``k`` draft tokens. The rows are whole requests.
        whole_rows: Store whole residual rows on every rank (for a next
            layer on the regular path, which reads whole rows), not the
            owned columns (for a next fused layer).
        tap: None, or a BF16 ``[rows+, slots, 8192]`` buffer on the Iris
            symmetric heap: the launch also stores the residual after the
            layer (BF16) into slot ``tap_slot`` of every rank's copy, the
            tap a draft reads.
        tap_slot: The tap's slot, or None without ``tap``.
        warm_next: None, or the next layer's prepared launch (from
            :func:`prepare_muse_spark_layer`) that follows this one: this
            launch reads its QKVG weights into the memory-side cache after
            the experts, so the next launch's projections read the MALL.
        proj_in: None (the launch projects its own q | k | v at its start),
            or int32 ``[rows+, MUSE_SPARK_PROJ_WORDS]``: the projection that
            the previous launch stored with ``proj_out``. Its attention then
            starts at once.
        proj_out: None, or (past one row, with ``warm_next``) an int32
            ``[rows+, MUSE_SPARK_PROJ_WORDS]`` buffer other than ``proj_in``:
            the launch ends with ``warm_next``'s q | k | v projection of the
            rows it outputs, for the next launch's ``proj_in`` (in place of
            warming its weights). The results are the same bits either way.
        stuck_rec: None, or the ``tensor`` of a
            :func:`muse_spark_stuck_record` record (host-visible int64, a
            record per wave of the launch's CTAs, then a header): a poll that
            gives up writes what it waited for there before the launch
            traps.
        compile_only: Only compile the launch for these operands, without
            running it. A launch polls its peers' values for about a minute
            before it traps, so a rank that compiles much longer than its
            peers stalls them: compile on every rank, synchronize the ranks,
            then launch.
    """
    layer.decode(
        weights=layer.weights,
        h=h,
        a=a,
        h_out=h_out,
        a_out=a_out,
        seq_len=seq_len,
        slot=slot,
        page_table=page_table,
        k_cache=k_cache,
        v_cache=v_cache,
        cos_sin=cos_sin,
        scratch=scratch,
        step=step,
        peer_bases=peer_bases,
        layer_index=layer_index,
        chain=chain,
        whole_rows=whole_rows,
        tap=tap,
        tap_slot=tap_slot,
        warm_next=None if warm_next is None else warm_next.weights,
        proj_in=proj_in,
        proj_out=proj_out,
        stuck_rec=stuck_rec,
        compile_only=compile_only,
    )


def prepare_muse_spark_draft_context(
    *,
    tensors: dict[str, torch.Tensor],
    rank: int,
    max_rows: int,
    solution: str | None = None,
) -> FusedDecodeLayer | None:
    """Pack one TP8 rank's DFlash2 context write (``fc`` and ``hidden_norm``
    of the target's taps, then every draft layer's K and V into the draft
    cache) for its monokernel.

    Args:
        tensors: The rank's weights by role (see
            ``tokenspeed_kernel_amd...muse_spark.draft_context
            .pack_draft_context``).
        rank: The rank in the TP8 group.
        max_rows: The most verify rows of one launch: those of up to two
            requests, 8 each.
        solution: Optional registered solution (such as ``"gluon"``) to
            force; its kernels must serve ``max_rows`` here, else this
            raises ``NoKernelFoundError``.

    Returns:
        The packed write, or None when no monokernel serves ``max_rows``
        here.
    """
    return _prepare(
        "muse_spark_draft_context", max_rows, solution, tensors=tensors, rank=rank
    )


def muse_spark_draft_context_decode(
    context: FusedDecodeLayer,
    *,
    taps: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    k_caches: tuple[torch.Tensor, ...],
    v_caches: tuple[torch.Tensor, ...],
    cos_sin: torch.Tensor,
    scratch: torch.Tensor,
    step: torch.Tensor,
    peer_bases: torch.Tensor,
    compile_only: bool,
) -> None:
    """The DFlash2 context write of one or two requests' verify rows (8
    each), on every rank of the TP8 group at once (the launches exchange
    the norm).

    Args:
        context: From :func:`prepare_muse_spark_draft_context`.
        taps: BF16 ``[rows, 40960]``: the target's tapped hidden states.
        positions: int32 or int64 ``[rows]``: the rows' positions.
        slots: int32 or int64 ``[rows]``: the rows' draft cache slots.
        k_caches, v_caches: The draft layers' BF16 ``[slots, 1, 128]``
            caches of the rank's KV head.
        cos_sin: FP32 ``[positions, 128]`` RoPE table, cos then sin.
        scratch: int64 mailbox scratch of at least ``context.scratch_pairs``
            on the Iris symmetric heap.
        step: int32 ``[1]`` counter that changes from launch to launch.
        peer_bases: int64 ``[8]`` Iris heap bases of the group's ranks.
        compile_only: Only compile the launch for these operands (compile on
            every rank, synchronize the ranks, then launch).
    """
    context.decode(
        weights=context.weights,
        taps=taps,
        positions=positions,
        slots=slots,
        k_caches=k_caches,
        v_caches=v_caches,
        cos_sin=cos_sin,
        scratch=scratch,
        step=step,
        peer_bases=peer_bases,
        compile_only=compile_only,
    )


def prepare_muse_spark_draft_head(
    *,
    tensors: dict[str, torch.Tensor],
    rank: int,
    max_rows: int,
    scale: float,
    softcap: float,
    solution: str | None = None,
) -> FusedDecodeLayer | None:
    """Pack one TP8 rank's DFlash2 draft head (the vocabulary-parallel top 16
    of each draft step and the candidate selector's scores) for its
    monokernel.

    Args:
        tensors: The rank's weights by role (see
            ``tokenspeed_kernel_amd...muse_spark.draft_head.pack_draft_head``).
        rank: The rank in the TP8 group.
        max_rows: The most draft steps of one launch: those of up to two
            requests' blocks, 7 each.
        scale: The logits' multiplier.
        softcap: The final logit softcap.
        solution: Optional registered solution (such as ``"gluon"``) to
            force; its kernels must serve ``max_rows`` here, else this
            raises ``NoKernelFoundError``.

    Returns:
        The packed head, or None when no monokernel serves ``max_rows`` here.
    """
    return _prepare(
        "muse_spark_draft_head",
        max_rows,
        solution,
        tensors=tensors,
        rank=rank,
        scale=scale,
        softcap=softcap,
    )


def muse_spark_draft_head_scratch_pairs(
    *, max_rows: int, solution: str | None = None
) -> int | None:
    """The int64 mailbox pairs that :func:`prepare_muse_spark_draft_head`'s
    head of ``max_rows`` draft steps needs, without packing it: for reserving
    the Iris heap before the heap exists.

    Args:
        max_rows: The most draft steps of one launch: those of up to two
            requests' blocks, 7 each.
        solution: Optional registered solution (such as ``"gluon"``) to
            force; its kernels must serve ``max_rows`` here, else this
            raises ``NoKernelFoundError``.

    Returns:
        The pairs, or None when no monokernel serves ``max_rows`` here.
    """
    try:
        scratch = _select("muse_spark_draft_head_scratch", max_rows, solution)
    except NoKernelFoundError:
        if solution is not None:
            raise
        return None
    return scratch(max_rows=max_rows)


def muse_spark_draft_head_decode(
    head: FusedDecodeLayer,
    *,
    x: torch.Tensor,
    first_token: torch.Tensor,
    anchor: torch.Tensor,
    cand_ids: torch.Tensor,
    unary: torch.Tensor,
    scores: torch.Tensor,
    scratch: torch.Tensor,
    step: torch.Tensor,
    peer_bases: torch.Tensor,
    compile_only: bool,
) -> None:
    """The DFlash2 draft head of one or two requests' draft steps, on every
    rank of the TP8 group at once (the launches exchange candidates).

    Args:
        head: From :func:`prepare_muse_spark_draft_head`.
        x: BF16 ``[steps, 8192]``: the draft's final-norm output of each
            request's block rows 1 .. 7, request by request.
        first_token: int32 ``[requests]``: each block's first token.
        anchor: int64 ``[requests]`` out: those tokens clamped to the
            vocabulary (each request's step 0 predecessor, for the walk).
        cand_ids: int64 ``[steps, 16]`` out: each step's candidates, by
            descending unary score (the lower token id first on a tie).
        unary: FP32 ``[steps, 16]`` out: their softcapped logits.
        scores: FP32 ``[steps, 16, 16]`` out: the selector's edge scores
            from the previous step's candidate ``p`` to this step's ``c``.
        scratch: int64 mailbox scratch of at least ``head.scratch_pairs`` on
            the Iris symmetric heap.
        step: int32 ``[1]`` counter that changes from launch to launch
            (CUDA graph safe).
        peer_bases: int64 ``[8]`` Iris heap bases of the group's ranks.
        compile_only: Only compile the launch for these operands (compile on
            every rank, synchronize the ranks, then launch).
    """
    head.decode(
        weights=head.weights,
        x=x,
        first_token=first_token,
        anchor=anchor,
        cand_ids=cand_ids,
        unary=unary,
        scores=scores,
        scratch=scratch,
        step=step,
        peer_bases=peer_bases,
        compile_only=compile_only,
    )


def prepare_muse_spark_draft_layer(
    *,
    tensors: dict[str, torch.Tensor],
    rank: int,
    max_rows: int,
    solution: str | None = None,
) -> FusedDecodeLayer | None:
    """Pack one TP8 rank's layer of Muse Spark's DFlash2 draft for its
    monokernel.

    Args:
        tensors: The rank's weights by role (see
            ``tokenspeed_kernel_amd...muse_spark.draft_layer.pack_draft_layer``).
        rank: The rank in the TP8 group.
        max_rows: The most rows of one draft block forward: the blocks of
            up to two requests, 8 rows each.
        solution: Optional registered solution (such as ``"gluon"``) to
            force; its kernels must serve ``max_rows`` here, else this
            raises ``NoKernelFoundError``.

    Returns:
        The packed layer, or None when no monokernel serves ``max_rows``
        here.
    """
    return _prepare(
        "muse_spark_draft_layer", max_rows, solution, tensors=tensors, rank=rank
    )


def muse_spark_draft_layer_decode(
    layer: FusedDecodeLayer,
    *,
    x: torch.Tensor,
    residual: torch.Tensor,
    residual_out: torch.Tensor,
    out: torch.Tensor | None,
    final_gain: torch.Tensor | None,
    normed: torch.Tensor | None,
    seq_len: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    page_table: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cos_sin: torch.Tensor,
    scratch: torch.Tensor,
    step: torch.Tensor,
    peer_bases: torch.Tensor,
    layer_index: int,
    compile_only: bool,
) -> None:
    """One step of a packed DFlash2 draft layer for the blocks of one or two
    requests (8 rows each, request by request), on every rank of the TP8
    group at once (the launches exchange values).

    Args:
        layer: From :func:`prepare_muse_spark_draft_layer`.
        x: BF16 ``[rows, 8192]``: this rank's partial of the layer input.
        residual: BF16 ``[rows, 8192]``: the residual before the layer.
        residual_out: BF16 ``[rows, 8192]``: the rank stores its owned
            columns (``1024 rank ..``) of the residual after the attention
            sublayer; the other columns are left as they are.
        out: BF16 ``[rows, 8192]``: this rank's partial of the layer
            output; None for the draft's last layer, whose launch also runs
            the draft's final norm.
        final_gain: None, or (the last layer) FP32 ``[1024]``: the final
            norm's gain of the rank's owned columns.
        normed: None, or (the last layer) BF16 ``[rows, 8192]``: the final
            norm's output, the whole rows on every rank.
        seq_len: int32 ``[requests]``: each request's length after its
            block.
        positions: int32 ``[rows]``: the block rows' positions.
        slots: int32 or int64 ``[rows]``: the block rows' cache slots.
        page_table: int32 ``[>= requests, pages]``: request ``q``'s page
            table in row ``q`` (64 tokens a page).
        k_cache, v_cache: BF16 ``[slots, 1, 128]``: the layer's cache.
        cos_sin: FP32 ``[positions, 128]`` rotary table, cos then sin.
        scratch: int64 mailbox scratch of at least ``layer.scratch_pairs``
            on the Iris symmetric heap (the same offset on every rank).
        step: int32 ``[1]`` counter that changes once per draft forward
            (CUDA graph safe: the kernel reads it on device).
        peer_bases: int64 ``[8]`` Iris heap bases of the group's ranks.
        layer_index: Draft layer index, below 5.
        compile_only: Only compile the launch for these operands, without
            running it: compile on every rank, synchronize the ranks, then
            launch (a launch polls its peers for about a minute before it
            traps).
    """
    layer.decode(
        weights=layer.weights,
        x=x,
        residual=residual,
        residual_out=residual_out,
        out=out,
        final_gain=final_gain,
        normed=normed,
        seq_len=seq_len,
        positions=positions,
        slots=slots,
        page_table=page_table,
        k_cache=k_cache,
        v_cache=v_cache,
        cos_sin=cos_sin,
        scratch=scratch,
        step=step,
        peer_bases=peer_bases,
        layer_index=layer_index,
        compile_only=compile_only,
    )


# Register the backends (side-effect imports).
import tokenspeed_kernel.ops.model.muse_spark.monokernel.gluon  # noqa: E402,F401


def muse_spark_stuck_record(
    *,
    grid: int,
    waves: int,
    path: str | None,
    scratch: torch.Tensor,
    rank: int,
    solution: str | None = None,
) -> Any:
    """A host-visible stuck-poll record buffer for Muse Spark layer launches:
    a poll that gives up writes what it waited for there before the launch
    traps.

    Args:
        grid: The launches' CTAs (at most).
        waves: Waves per CTA (the launches' num_warps).
        path: None (pinned memory, readable while the process lives), or a
            file to map, which keeps the records after the process dies.
        scratch: The launches' mailbox scratch (its address goes into the
            buffer's header, for decoding).
        rank: This process's rank.
        solution: Optional registered solution (such as ``"gluon"``) to
            force; without a kernel here, this raises ``NoKernelFoundError``.

    Returns:
        The record: ``tensor`` (the launches' ``stuck_rec``), ``path``,
        ``any()`` (whether some poll wrote a record), ``close(remove)``,
        ``layer_kinds(layers)`` (each served layer's kind, for a sidecar),
        and ``describe(layers, launches)`` (one line per stuck poll; ``layers``
        the served layers by index, ``launches`` the ``(rows, chain)`` of the
        launches they ran).
    """
    # One record serves launches of every row count; selection matches the
    # kernel by a row count, and every layer kernel serves a single row.
    make = _select("muse_spark_stuck_record", 1, solution)
    return make(grid=grid, waves=waves, path=path, scratch=scratch, rank=rank)
