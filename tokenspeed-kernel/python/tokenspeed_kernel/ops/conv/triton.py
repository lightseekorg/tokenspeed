# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2024, Tri Dao
#
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


"""Triton convolution solutions.

``triton_causal_conv1d_prefill`` runs varlen prefill chunks and
``triton_causal_conv1d_update`` decode, chain verify and draft-tree steps;
both are registered leaves of the ``conv`` family's
``causal_conv1d_prefill`` / ``causal_conv1d_update`` modes, selected through
``tokenspeed_kernel.ops.conv``. The DFlash2 grouped conv and the sconv
kernels are re-exported for the conv package entry points.
"""

import functools

import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.attention.gdn.triton import (
    CausalConv1dPrefillMetadata,
)
from tokenspeed_kernel.ops.conv._triton.causal_conv1d import (
    PAD_SLOT_ID,
    _causal_conv1d_fwd_kernel,
    _causal_conv1d_tree_update,
    _causal_conv1d_update_kernel,
)
from tokenspeed_kernel.ops.conv._triton.grouped import dflash2_grouped_conv
from tokenspeed_kernel.ops.conv._triton.sconv import (
    _inkling_ring_commit_kernel,
    _inkling_ring_sconv_decode_kernel,
    _inkling_ring_sconv_prefill_kernel,
    select_prefill_config,
)
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

__all__ = [
    "PAD_SLOT_ID",
    "_inkling_ring_commit_kernel",
    "_inkling_ring_sconv_decode_kernel",
    "_inkling_ring_sconv_prefill_kernel",
    "dflash2_grouped_conv",
    "select_prefill_config",
    "triton_causal_conv1d_prefill",
    "triton_causal_conv1d_update",
]

_CONV_DTYPES = {torch.bfloat16, torch.float16, torch.float32}
# Channels per program of the sequential update, one per thread.
_UPDATE_BLOCK_CHANNELS = 128


@functools.cache
def _update_num_warps() -> int:
    """Warps holding one channel per thread (32- or 64-lane warps)."""
    warp_size = triton.runtime.driver.active.get_current_target().warp_size
    return max(_UPDATE_BLOCK_CHANNELS // warp_size, 1)


@register_kernel(
    "conv",
    "causal_conv1d_prefill",
    name="triton_causal_conv1d_prefill",
    solution="triton",
    signatures=format_signatures("x", "dense", _CONV_DTYPES),
    priority=Priority.PORTABLE,
)
def triton_causal_conv1d_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data=False,
    *,
    prefill_metadata: CausalConv1dPrefillMetadata,
):
    """support varlen + continuous batching when x is 2D tensor

    x: (dim,cu_seq_len)
        cu_seq_len = total tokens of all seqs in that batch
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
        [it use `cache_indices` to get the index to the cache of conv_state for that sequence

        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
             and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
        ]
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        if
        x = [5, 1, 1, 1] <- continuous batching (batch=4)
        then
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
           the ending index of the last sequence
        [length(query_start_loc)-1 == batch]
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
        [single boolean for each sequence in the batch: True or False]
    bias: (dim,)
    activation: either None or "silu" or "swish" or True
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3
    prefill_metadata: CausalConv1dPrefillMetadata
        Read-only program mapping built once with this forward's sequence
        boundaries and shared by all recurrent-attention layers.

    out: same shape as `x`
    """
    enable_pdl = pdl_enabled()
    if isinstance(activation, bool) and activation:
        activation = "silu"

    out = torch.empty_like(x)
    batch_ptr = prefill_metadata.batch_indices
    token_chunk_offset_ptr = prefill_metadata.chunk_offsets
    if batch_ptr.device != x.device or token_chunk_offset_ptr.device != x.device:
        raise ValueError("convolution metadata must be on the input device")
    if batch_ptr.numel() == 0:
        return out

    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
    dim, cu_seqlen = x.shape
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    padded_batch = query_start_loc.size(0) - 1
    stride_x_seq = 0
    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    num_cache_lines = 0
    if conv_states is not None:
        # 1. conv_states is used to replaced initial_states
        # 2. conv_states serve as a cache with num cache lines can be larger than batch size
        # 3. mapping from sequence x[idx] to a cache line at index as specified via cache_indices[idx]
        # 4. computation can be skipped if cache_indices[idx] == pad_slot_id
        num_cache_lines = conv_states.size(0)
        assert (
            num_cache_lines == conv_states.shape[0]
            and dim == conv_states.shape[1]
            and width - 1 <= conv_states.shape[2]
        )
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)
    if out.dim() == 2:
        stride_o_seq = 0
        stride_o_dim = out.stride(0)
        stride_o_token = out.stride(1)
    else:
        stride_o_seq = out.stride(0)
        stride_o_dim = out.stride(1)
        stride_o_token = out.stride(2)

    if validate_data:
        assert x.dim() == 2
        assert query_start_loc is not None
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1 or x.stride(1) == 1
        if bias is not None:
            assert bias.dim() == 1
            assert dim == bias.size(0)
        if cache_indices is not None:
            assert cache_indices.dim() == 1
            assert padded_batch == cache_indices.size(0)
        if has_initial_state is not None:
            assert has_initial_state.size() == (padded_batch,)
            assert (
                conv_states is not None
            ), "ERROR: `has_initial_state` is used, which needs also `conv_states`"
        assert weight.stride(1) == 1
        assert (dim, width) == weight.shape
        assert is_channel_last, "Need to run in channel-last layout"

    grid = (batch_ptr.numel(), triton.cdiv(dim, 256))
    _causal_conv1d_fwd_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        batch_ptr,
        token_chunk_offset_ptr,
        out,
        # Matrix dimensions
        dim,
        cu_seqlen,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_INITIAL_STATES=has_initial_state is not None,
        HAS_CACHE=conv_states is not None,
        IS_CONTINUOUS_BATCHING=cache_indices is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        BLOCK_M=prefill_metadata.block_m,
        BLOCK_N=256,
        num_stages=2,
        ENABLE_PDL=enable_pdl,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    return out


@register_kernel(
    "conv",
    "causal_conv1d_update",
    name="triton_causal_conv1d_update",
    solution="triton",
    signatures=format_signatures("x", "dense", _CONV_DTYPES),
    priority=Priority.PORTABLE,
)
def triton_causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    cache_seqlens: torch.Tensor | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    intermediate_conv_window: torch.Tensor | None = None,
    output_state_indices: torch.Tensor | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
    validate_data=False,
    *,
    parent_indices: torch.Tensor | None,
    out: torch.Tensor | None = None,
    replay_payload: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    parent_indices: (batch, seqlen), dtype int32; requires
        conv_state_indices and int32 output_state_indices. Draft trees: token t's
        window ends at token parent_indices[b, t] (the initial window when
        negative) instead of token t - 1.
    out: destination shaped and typed like x, not overlapping it; None writes over x.
    replay_payload: (payload, a, b) to fill, or None; row i * seqlen + t: [k/v channels | a | b].
    returns: the output, (batch, dim) or (batch, dim, seqlen)
    """
    enable_pdl = pdl_enabled()
    if parent_indices is not None:
        tree_tables_ok = (
            conv_state_indices is not None
            and output_state_indices is not None
            and parent_indices.shape == output_state_indices.shape
            and conv_state_indices.shape == parent_indices.shape[:1]
            and {
                parent_indices.dtype,
                output_state_indices.dtype,
                conv_state_indices.dtype,
            }
            == {torch.int32}
        )
        chain_only_options = (
            num_accepted_tokens is not None
            or cache_seqlens is not None
            or intermediate_conv_window is not None
            or pad_slot_id != PAD_SLOT_ID
            or validate_data
        )
        if not tree_tables_ok or chain_only_options:
            raise ValueError(
                "parent_indices needs int32 conv_state_indices (batch,) and "
                "output_state_indices of its shape, and none of num_accepted_tokens, "
                "cache_seqlens, intermediate_conv_window, pad_slot_id, validate_data"
            )
    if validate_data:
        assert cache_seqlens is None
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]
    if parent_indices is not None:
        if out is not None or replay_payload is not None:
            raise ValueError("the tree conv writes its own output, without payload")
        return _causal_conv1d_tree_update(
            x,
            conv_state,
            weight,
            bias,
            activation,
            conv_state_indices,
            output_state_indices,
            parent_indices,
        )
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert (
            conv_state.stride(-2) == 1
        ), f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None

    if out is None:
        out = x
    elif out.dim() == 2:
        out = out.unsqueeze(-1)
    if out.shape != x.shape or out.dtype != x.dtype:
        raise ValueError(
            f"out {out.dtype} {tuple(out.shape)} must match x {x.dtype} {x.shape}"
        )
    stride_w_dim, stride_w_width = weight.stride()

    stride_x_seq, stride_x_dim, stride_x_token = x.stride()  # X (batch, dim, seqlen)

    stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    if output_state_indices is not None or intermediate_conv_window is not None:
        state_len = width - 1
    else:
        state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    # prepare intermediate buffer strides if provided
    if intermediate_conv_window is not None:
        stride_inter_seq, stride_inter_step, stride_inter_dim, stride_inter_win = (
            intermediate_conv_window.stride(0),
            intermediate_conv_window.stride(1),
            intermediate_conv_window.stride(2),
            intermediate_conv_window.stride(3),
        )
    else:
        stride_inter_seq = stride_inter_step = stride_inter_dim = stride_inter_win = 0
    if output_state_indices is not None:
        stride_output_state_indices_seq, stride_output_state_indices_step = (
            output_state_indices.stride(0),
            output_state_indices.stride(1),
        )
    else:
        stride_output_state_indices_seq = stride_output_state_indices_step = 0
    # One channel per thread for multi-token windows, two for single-token decode (faster each).
    block_n, num_warps = (
        (_UPDATE_BLOCK_CHANNELS, _update_num_warps()) if seqlen > 1 else (256, 4)
    )
    if replay_payload is not None:
        payload, replay_a, replay_b = replay_payload
        gate_heads = replay_a.shape[-1]
        rows = batch * seqlen
        # The payload saves the channels from this one on.
        replay_first_channel = dim + 2 * gate_heads - payload.shape[-1]
        if (
            payload.shape[0] != rows
            or payload.stride(-1) != 1
            or replay_a.shape != (rows, gate_heads)
            or replay_b.shape != (rows, gate_heads)
            or replay_a.stride(-1) != 1
            or replay_b.stride(-1) != 1
            or not 0 <= replay_first_channel < dim
            or any(t.dtype != x.dtype for t in replay_payload)
        ):
            raise ValueError(
                "replay_payload needs a [batch * seqlen, width] payload saving "
                "1 to dim channels plus two [batch * seqlen, heads] gates, all of "
                "x's dtype with unit inner strides"
            )
        replay_strides = (payload.stride(0), replay_a.stride(0), replay_b.stride(0))
    else:
        payload = replay_a = replay_b = out
        gate_heads = replay_first_channel = 0
        replay_strides = (0, 0, 0)

    _causal_conv1d_update_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_state,
        cache_seqlens,
        conv_state_indices,
        num_accepted_tokens,
        intermediate_conv_window if intermediate_conv_window is not None else x,
        output_state_indices if output_state_indices is not None else x,
        out,
        payload,
        replay_a,
        replay_b,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_inter_seq,
        stride_inter_step,
        stride_inter_dim,
        stride_inter_win,
        stride_output_state_indices_seq,
        stride_output_state_indices_step,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        *replay_strides,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=block_n,
        num_warps=num_warps,
        SAVE_INTERMEDIATE=intermediate_conv_window is not None,
        HAS_OUTPUT_STATE_INDICES=output_state_indices is not None,
        HAS_REPLAY_PAYLOAD=replay_payload is not None,
        REPLAY_FIRST_CHANNEL=replay_first_channel,
        GATE_HEADS=gate_heads,
        GATE_BLOCK=triton.next_power_of_2(max(gate_heads, 1)),
        NP2_SEQLEN=triton.next_power_of_2(seqlen),
        ENABLE_PDL=enable_pdl,
        **({"launch_pdl": True} if enable_pdl else {}),
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out
