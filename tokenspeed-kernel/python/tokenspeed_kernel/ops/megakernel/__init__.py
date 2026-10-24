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

"""Whole-layer decode megakernels.

A megakernel runs one decoder layer's decode step as one persistent launch.
Every stage of the layer (norms, projections, attention, MoE) is a set of
tasks placed on fixed CTAs. The stages hand data to each other through tagged
mailboxes in a scratch buffer, not through kernel boundaries.
See ``docs/design/megakernel.md`` for the design, and ``README.md`` in this
package for adding a model's megakernel.

The runtime packs a layer once after weight loading
(:func:`prepare_qwen4_exp_delta_layer`) and then runs it per step
(:func:`qwen4_exp_delta_layer_decode`). The prepare function returns None
when the platform has no megakernel for the layer, so the caller keeps its
regular path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = [
    "FusedDecodeLayer",
    "qwen4_exp_mtp_front",
    "select_qwen4_exp_mtp_front",
    "prepare_qwen4_exp_delta_layer",
    "prepare_qwen4_exp_qsa_layer",
    "qwen4_exp_delta_layer_decode",
    "qwen4_exp_qsa_layer_decode",
]


@dataclass(frozen=True)
class FusedDecodeLayer:
    """A decoder layer packed for its megakernel.

    Attributes:
        decode: The selected kernel. The ``*_decode`` function calls it.
        weights: The kernel's packed weights (opaque to the caller).
        scratch_pairs: int64 mailbox pairs the kernel needs. One scratch
            buffer serves every layer of a step.
    """

    decode: Any
    weights: Any
    scratch_pairs: int


def _select(mode: str, max_rows: int, override: str | None):
    signature = format_signature(h=dense_tensor_format(torch.bfloat16))
    return select_kernel(
        "megakernel", mode, signature, traits={"rows": max_rows}, override=override
    )


def _prepare(kind: str, max_rows: int, tensors, dense_tiled: bool, override):
    try:
        prepare = _select(f"{kind}_prepare", max_rows, override)
        decode = _select(f"{kind}_decode", max_rows, override)
    except NoKernelFoundError:
        if override is not None:
            raise
        return None
    weights, scratch_pairs = prepare(
        tensors=tensors, dense_tiled=dense_tiled, max_rows=max_rows
    )
    return FusedDecodeLayer(decode=decode, weights=weights, scratch_pairs=scratch_pairs)


def prepare_qwen4_exp_delta_layer(
    *,
    max_rows: int,
    tensors: dict[str, Any],
    dense_tiled: bool,
    override: str | None = None,
) -> FusedDecodeLayer | None:
    """Pack a Qwen3.8-Flash-Next Gated DeltaNet layer for its megakernel.

    Args:
        max_rows: The most request rows (one token each) a decode step will
            run the layer with. A step may run fewer.
        tensors: The layer's weights by role (see
            ``tokenspeed_kernel_amd...qwen4_exp.weights.pack_delta_layer``).
            ``tensors["experts"]`` names the MoE block's format: ``"mxfp4"``
            (``pack_moe``'s tensors) or ``"bf16"`` (``pack_moe_bf16``'s).
        dense_tiled: Whether the dense MXFP4 weights are already in N16 tile
            order (the MFMA linear layout) rather than checkpoint order.
        override: Optional registered kernel name to force.

    Returns:
        The packed layer, or None when no megakernel serves ``max_rows`` here.
    """
    return _prepare("qwen4_exp_delta_layer", max_rows, tensors, dense_tiled, override)


def qwen4_exp_delta_layer_decode(
    layer: FusedDecodeLayer,
    h: torch.Tensor,
    h_out: torch.Tensor,
    *,
    chain: int,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    state_in: torch.Tensor,
    state_out: torch.Tensor,
    conv_out: torch.Tensor,
    conv_out_rows: torch.Tensor,
    replay: torch.Tensor | None,
    scratch: torch.Tensor,
    step: torch.Tensor,
    layer_index: int,
    ple: dict[str, torch.Tensor] | None,
) -> None:
    """One decode step, or one speculative verify, of a packed Gated
    DeltaNet layer.

    Args:
        layer: From :func:`prepare_qwen4_exp_delta_layer`.
        h: Hyper-connection input streams of ``rows`` request rows (at most
            the packed ``max_rows``). The shape is BF16 ``[rows, 4 * hidden]``,
            or ``[rows, hidden]``: one row per request that stands for all
            four streams (the first layer's embedding).
        h_out: Output streams, same shape. ``h_out`` must not alias ``h``.
        chain: Consecutive rows per request: 1 for decode, ``k + 1`` for a
            verify of ``k`` draft tokens (the rows chain the layer's state).
        conv_state, ssm_state: The layer's state pools (pages first).
        state_in, state_out: int32 ``[requests]`` input and output state
            pages, -1 for padding requests. The state after a request's last
            token goes to its output page (-1 in a verify).
        conv_out, conv_out_rows: Each row's conv window after its token goes
            to row ``conv_out_rows[row]`` (int32 ``[rows]``, -1: none) of
            ``conv_out``. In decode, these are ``conv_state`` and
            ``state_out``. In a verify, these are the verify scratch and its
            rows.
        replay: None, or the layer's replay payload a verify fills (each
            row's ``[k | v | a | b]``).
        scratch: int64 mailbox scratch of at least ``layer.scratch_pairs``.
        step: int32 counter (``[1]``, or ``[2]`` for the launch that
            advances it) that changes between steps (CUDA graph safe: the
            kernel reads it on device).
        layer_index: Decoder layer index, below 64.
        ple: For the layer with a PLE block (packed with ``tensors["ple"]``),
            its per-step operands by name:

            * ``token``: each row's token ``[rows]``.
            * ``context``: the int64 context pool ``[pages, 2]``.
            * ``conv_state``: the conv state pool
              ``[pages, hc_count * hidden, 9]``.
            * ``page_in``: int32 ``[requests]``, each request's input page.
            * ``page_out``: int32 ``[rows]``, each row's output row.
            * ``context_out``: ``[*, 2]``.
            * ``conv_out``: ``[*, hc_count * hidden, 9]``.

            The state after each row's token goes to its output row of the
            two outputs. In decode, these are the pools and the output pages.
            In a verify, these are the verify scratch and its rows. ``ple``
            is None for any other layer. The kernel does the n-gram lookup
            itself.
    """
    layer.decode(
        weights=layer.weights,
        h=h,
        h_out=h_out,
        chain=chain,
        conv_state=conv_state,
        ssm_state=ssm_state,
        state_in=state_in,
        state_out=state_out,
        conv_out=conv_out,
        conv_out_rows=conv_out_rows,
        replay=replay,
        scratch=scratch,
        step=step,
        layer_index=layer_index,
        ple=ple,
    )


def prepare_qwen4_exp_qsa_layer(
    *,
    max_rows: int,
    tensors: dict[str, Any],
    dense_tiled: bool,
    override: str | None = None,
) -> FusedDecodeLayer | None:
    """Pack a Qwen3.8-Flash-Next QSA attention layer for its megakernel.

    Args and returns as in :func:`prepare_qwen4_exp_delta_layer`. ``tensors``
    follows ``...qwen4_exp.weights.pack_qsa_layer``.
    """
    return _prepare("qwen4_exp_qsa_layer", max_rows, tensors, dense_tiled, override)


def qwen4_exp_qsa_layer_decode(
    layer: FusedDecodeLayer,
    h: torch.Tensor,
    h_out: torch.Tensor,
    *,
    chain: int,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    seq_len: torch.Tensor,
    full_table: torch.Tensor,
    page_size: int,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qsa_table: torch.Tensor,
    recent_table: torch.Tensor,
    compressed: torch.Tensor,
    raw_key: torch.Tensor,
    rope_position: torch.Tensor,
    staging: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None,
    draft: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    accepted: torch.Tensor | None,
    scratch: torch.Tensor,
    step: torch.Tensor,
    layer_index: int,
    mixed_out: torch.Tensor | None,
    advance_step: bool,
) -> None:
    """One decode step, one speculative verify, or one MTP draft step of a
    packed QSA attention layer.

    Args:
        layer, h, h_out, chain, scratch, step, layer_index: As in
            :func:`qwen4_exp_delta_layer_decode`.
        cos_sin: Rotary cache, FP32 ``[max_position, 64]``.
        positions: int64 ``[3, rows]`` MRoPE positions of each row's token.
        seq_len: int32 ``[requests]``: each request's length including its
            last token (token ``t`` of a request's ``chain`` rows is at
            ``seq_len - chain + t``).
        full_table, page_size: The requests' KV page tables ``[requests,
            pages]`` and the page size.
        k_cache, v_cache: The layer's ``[slots, kv_heads, head_dim]`` caches.
        qsa_table, recent_table: The requests' QSA compressed and recent page
            tables ``[requests, pages]``.
        compressed, raw_key, rope_position: The layer's QSA indexer caches.
        staging: None in decode. In a verify, the rows of TokenSpeed's
            ``QSAVerifyState`` ``(this layer's raw index keys [rows, dim],
            positions [rows, 3], logical positions [rows], recent locations
            [rows])`` that the kernel writes instead of the recent ring.
        draft: None, except in an MTP draft step (``chain`` 1): the requests'
            QSA draft scratch ``(raw keys [rows, 4, 1, dim], group-start
            positions [rows, 3], position tags [rows, 4])``. The kernel stages
            the index keys there instead of the recent ring, and its
            compression reads the members that earlier draft steps staged.
        accepted: None, except in an MTP draft catch-up step (the draft's
            first step over one request's verified rows): the request's int32
            accepted length ``[1]``. Rows at or past it write K / V but no
            index keys. As TokenSpeed's draft narrowing, only the live row
            (the last accepted token's) runs the attention and the rest of
            the layer, so ``h_out`` and ``mixed_out`` hold that one row.
        mixed_out: When :func:`prepare_qwen4_exp_qsa_layer` packed the layer
            with the model's final mixer (``tensors["final_mixer"]``, the last
            layer), the BF16 mixer output (the LM head's input), with the
            rows of ``h_out``. Else ``None``.
        advance_step: Whether the launch moves ``step`` on when it ends (the
            last launch of a forward); ``step`` is then int32 ``[2]`` (the
            step and a count that stays 0 between launches).
    """
    layer.decode(
        weights=layer.weights,
        h=h,
        h_out=h_out,
        chain=chain,
        cos_sin=cos_sin,
        positions=positions,
        seq_len=seq_len,
        full_table=full_table,
        page_size=page_size,
        k_cache=k_cache,
        v_cache=v_cache,
        qsa_table=qsa_table,
        recent_table=recent_table,
        compressed=compressed,
        raw_key=raw_key,
        rope_position=rope_position,
        staging=staging,
        draft=draft,
        accepted=accepted,
        scratch=scratch,
        step=step,
        layer_index=layer_index,
        mixed_out=mixed_out,
        advance_step=advance_step,
    )


def select_qwen4_exp_mtp_front(max_rows: int, override: str | None = None) -> Any:
    """The kernel of :func:`qwen4_exp_mtp_front` for up to ``max_rows``
    rows, or None when no kernel serves them here."""
    signature = format_signature(hidden=dense_tensor_format(torch.bfloat16))
    try:
        return select_kernel(
            "megakernel",
            "qwen4_exp_mtp_front",
            signature,
            traits={"rows": max_rows},
            override=override,
        )
    except NoKernelFoundError:
        if override is not None:
            raise
        return None


def qwen4_exp_mtp_front(
    kernel: Any,
    input_ids: torch.Tensor,
    embedding: torch.Tensor,
    hidden: torch.Tensor,
    norm_embedding: torch.Tensor,
    norm_hidden: torch.Tensor,
    fc_embedding: torch.Tensor,
    fc_hidden: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> None:
    """The Qwen3.8-Flash-Next MTP draft input of a decode step's rows, in one
    launch: ``fc_hidden(norm(hidden)) + fc_embedding(norm(embedding[ids]))``
    for each of the 4 hyper-connection streams.

    Args:
        kernel: From :func:`select_qwen4_exp_mtp_front`.
        input_ids: int ``[rows]`` tokens (``rows`` at most the selected
            ``max_rows``).
        embedding: BF16 ``[vocab, H]`` token embedding (unsharded).
        hidden: BF16 ``[rows, 4 H]`` captured hidden streams.
        norm_embedding, norm_hidden: Gemma RMSNorm weights ``[H]`` and
            ``[4 H]`` (the multiplier is ``1 + weight``); the hidden norm
            spans all 4 streams.
        fc_embedding, fc_hidden: BF16 ``[H, H]`` linear weights
            (``[out, in]``), applied to the embedding and to each stream.
        eps: The norms' epsilon.
        out: BF16 ``[rows, 4 H]`` result, stream major like ``hidden``.
    """
    kernel(
        input_ids,
        embedding,
        hidden,
        norm_embedding,
        norm_hidden,
        fc_embedding,
        fc_hidden,
        eps,
        out,
    )


# Register the backends (side-effect imports).
import tokenspeed_kernel.ops.megakernel.gluon  # noqa: E402,F401
import tokenspeed_kernel.ops.megakernel.triton  # noqa: E402,F401
