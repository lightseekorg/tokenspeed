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
        state_in, state_out: int32 ``[rows]`` input and output state pages,
            -1 for padding rows. A request's rows share its input page. The
            state after its last token goes to that row's output page (-1
            in a verify).
        conv_out, conv_out_rows: Each row's conv window after its token goes
            to row ``conv_out_rows[row]`` (int32 ``[rows]``, -1: none) of
            ``conv_out``. In decode, these are ``conv_state`` and
            ``state_out``. In a verify, these are the verify scratch and its
            rows.
        replay: None, or the layer's replay payload a verify fills (each
            row's ``[k | v | a | b]``).
        scratch: int64 mailbox scratch of at least ``layer.scratch_pairs``.
        step: int32 ``[1]`` counter that changes before every step (CUDA
            graph safe: the kernel reads it on device).
        layer_index: Decoder layer index, below 64.
        ple: For the layer with a PLE block (packed with ``tensors["ple"]``),
            its per-step operands by name:

            * ``token``: each row's token ``[rows]``.
            * ``context``: the int64 context pool ``[pages, 2]``.
            * ``conv_state``: the conv state pool
              ``[pages, hc_count * hidden, 9]``.
            * ``page_in`` and ``page_out``: int32 ``[rows]``, each row's
              input page and output row.
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
    scratch: torch.Tensor,
    step: torch.Tensor,
    layer_index: int,
    mixed_out: torch.Tensor | None,
) -> None:
    """One decode step, or one speculative verify, of a packed QSA attention
    layer.

    Args:
        layer, h, h_out, chain, scratch, step, layer_index: As in
            :func:`qwen4_exp_delta_layer_decode`.
        cos_sin: Rotary cache, FP32 ``[max_position, 64]``.
        positions: int64 ``[3, rows]`` MRoPE positions of each row's token.
        seq_len: int32 ``[rows]``: each row's length including its token.
        full_table, page_size: The rows' KV page tables ``[rows, pages]`` and
            the page size.
        k_cache, v_cache: The layer's ``[slots, kv_heads, head_dim]`` caches.
        qsa_table, recent_table: The rows' QSA compressed and recent page
            tables ``[rows, pages]``.
        compressed, raw_key, rope_position: The layer's QSA indexer caches.
        staging: None in decode. In a verify, the rows of TokenSpeed's
            ``QSAVerifyState`` ``(this layer's raw index keys [rows, dim],
            positions [rows, 3], logical positions [rows], recent locations
            [rows])`` that the kernel writes instead of the recent ring.
        mixed_out: When :func:`prepare_qwen4_exp_qsa_layer` packed the layer
            with the model's final mixer (``tensors["final_mixer"]``, the last
            layer), the BF16 ``[rows, hidden]`` mixer output (the LM head's
            input). Else ``None``.
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
        scratch=scratch,
        step=step,
        layer_index=layer_index,
        mixed_out=mixed_out,
    )


# Register the backends (side-effect imports).
import tokenspeed_kernel.ops.megakernel.gluon  # noqa: E402,F401
