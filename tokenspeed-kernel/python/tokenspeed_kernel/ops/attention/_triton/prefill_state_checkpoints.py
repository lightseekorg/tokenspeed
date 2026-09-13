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

"""Shared Triton checkpoint preparation for GDN and KDA recurrent prefill."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel._triton import tl, triton


@dataclass(frozen=True)
class PackedPrefillCheckpointInputs:
    """Token-packed inputs and row-packed initial state for one prefix scan."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    recurrent_state: torch.Tensor
    a: torch.Tensor | None
    b: torch.Tensor | None
    g_raw: torch.Tensor | None
    f_a_out: torch.Tensor | None
    beta_raw: torch.Tensor | None


@triton.jit(do_not_specialize=["num_rows", "num_channels", "state_len"])
def _write_prefill_conv_checkpoints_kernel(
    raw_inputs,
    conv_states,
    state_in_blocks,
    state_out_blocks,
    checkpoint_blocks,
    rows,
    sequence_starts,
    checkpoint_seq_lens,
    num_rows,
    num_channels,
    state_len,
    raw_token_stride,
    raw_channel_stride,
    state_page_stride,
    state_channel_stride,
    state_position_stride,
    STATE_BLOCK: tl.constexpr,
):
    checkpoint_row = tl.program_id(0)
    channel = tl.program_id(1)
    positions = tl.arange(0, STATE_BLOCK)
    live = (checkpoint_row < num_rows) & (channel < num_channels)
    position_live = live & (positions < state_len)

    request_row = tl.load(rows + checkpoint_row, mask=live, other=0)
    sequence_start = tl.load(sequence_starts + checkpoint_row, mask=live, other=0)
    checkpoint_len = tl.load(checkpoint_seq_lens + checkpoint_row, mask=live, other=0)
    # Page IDs fit int32, but their element offsets in large pools may not.
    state_in = tl.load(state_in_blocks + request_row, mask=live, other=0).to(tl.int64)
    state_out = tl.load(state_out_blocks + request_row, mask=live, other=0).to(tl.int64)
    source = tl.where(state_in > 0, state_in, state_out)
    destination = tl.load(checkpoint_blocks + request_row, mask=live, other=0).to(
        tl.int64
    )

    relative_token = checkpoint_len - state_len + positions
    from_raw = relative_token >= 0
    raw_offsets = (
        sequence_start + relative_token
    ) * raw_token_stride + channel * raw_channel_stride
    source_offsets = (
        source * state_page_stride
        + channel * state_channel_stride
        + (state_len + relative_token) * state_position_stride
    )
    # Load the complete old-state window before any store. This makes the
    # operation safe even if a caller aliases source and destination pages.
    raw_value = tl.load(raw_inputs + raw_offsets, mask=position_live & from_raw)
    source_value = tl.load(conv_states + source_offsets, mask=position_live & ~from_raw)
    value = tl.where(from_raw, raw_value, source_value)
    destination_offsets = (
        destination * state_page_stride
        + channel * state_channel_stride
        + positions * state_position_stride
    )
    tl.store(conv_states + destination_offsets, value, mask=position_live)


def _torch_write_prefill_conv_checkpoints(
    raw_inputs: torch.Tensor,
    conv_states: torch.Tensor,
    state_in_blocks: torch.Tensor,
    state_out_blocks: torch.Tensor,
    checkpoint_blocks: torch.Tensor,
    rows: torch.Tensor,
    sequence_starts: torch.Tensor,
    checkpoint_seq_lens: torch.Tensor,
) -> None:
    state_len = conv_states.shape[-1]
    destinations = checkpoint_blocks.index_select(0, rows).to(torch.int64)
    state_in = state_in_blocks.index_select(0, rows)
    state_out = state_out_blocks.index_select(0, rows)
    sources = torch.where(state_in > 0, state_in, state_out).to(torch.int64)
    source_state = conv_states.index_select(0, sources)
    positions = torch.arange(state_len, dtype=torch.int64, device=raw_inputs.device)
    relative = checkpoint_seq_lens.unsqueeze(1) - state_len + positions.unsqueeze(0)
    raw_indices = sequence_starts.unsqueeze(1) + relative
    raw_indices.clamp_(min=0)
    raw_window = raw_inputs.index_select(0, raw_indices.flatten()).view(
        rows.numel(), state_len, raw_inputs.shape[1]
    )
    raw_window = raw_window.transpose(1, 2)
    source_positions = (state_len + relative).clamp_(min=0, max=state_len - 1)
    source_window = source_state.gather(
        2, source_positions.unsqueeze(1).expand(-1, source_state.shape[1], -1)
    )
    checkpoint_state = torch.where(
        (relative >= 0).unsqueeze(1), raw_window, source_window
    )
    conv_states.index_copy_(0, destinations, checkpoint_state)


def write_prefill_conv_checkpoints(
    raw_inputs: torch.Tensor,
    conv_states: torch.Tensor,
    state_in_blocks: torch.Tensor,
    state_out_blocks: torch.Tensor,
    checkpoint_blocks: torch.Tensor,
    rows: torch.Tensor,
    sequence_starts: torch.Tensor,
    checkpoint_seq_lens: torch.Tensor,
) -> None:
    """Write sparse convolution checkpoints with one launch.

    Args:
        raw_inputs: Token-major raw convolution inputs ``[tokens, channels]``.
        conv_states: State pool ``[pages, channels, state_len]``, updated in place.
        state_in_blocks: Per-request input state page ids.
        state_out_blocks: Per-request final output state page ids.
        checkpoint_blocks: Per-request internal-checkpoint destination page ids.
        rows: Request rows that own an internal checkpoint.
        sequence_starts: Token offset of each selected request in ``raw_inputs``.
        checkpoint_seq_lens: Request-local length of each selected checkpoint.

    Returns:
        None. ``conv_states`` is updated in place.
    """
    if rows.numel() == 0 or conv_states.shape[-1] == 0:
        return
    if raw_inputs.ndim != 2 or conv_states.ndim != 3:
        raise ValueError("raw_inputs must be 2D and conv_states must be 3D")
    if not (rows.numel() == sequence_starts.numel() == checkpoint_seq_lens.numel()):
        raise ValueError("checkpoint row, start, and length counts must agree")
    if not conv_states.is_cuda:
        _torch_write_prefill_conv_checkpoints(
            raw_inputs,
            conv_states,
            state_in_blocks,
            state_out_blocks,
            checkpoint_blocks,
            rows,
            sequence_starts,
            checkpoint_seq_lens,
        )
        return
    state_len = conv_states.shape[-1]
    state_block = triton.next_power_of_2(state_len)
    _write_prefill_conv_checkpoints_kernel[(rows.numel(), conv_states.shape[1])](
        raw_inputs,
        conv_states,
        state_in_blocks,
        state_out_blocks,
        checkpoint_blocks,
        rows,
        sequence_starts,
        checkpoint_seq_lens,
        rows.numel(),
        conv_states.shape[1],
        state_len,
        raw_inputs.stride(0),
        raw_inputs.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        STATE_BLOCK=state_block,
    )


@triton.jit(do_not_specialize=["num_tokens", "num_rows"])
def _pack_prefill_recurrent_inputs_kernel(
    token_indices,
    rows,
    query,
    query_out,
    key,
    key_out,
    value,
    value_out,
    recurrent_state,
    recurrent_state_out,
    a,
    a_out,
    b,
    b_out,
    g_raw,
    g_raw_out,
    f_a_out,
    f_a_packed,
    beta_raw,
    beta_raw_out,
    num_tokens,
    num_rows,
    query_stride: tl.constexpr,
    key_stride: tl.constexpr,
    value_stride: tl.constexpr,
    state_stride: tl.constexpr,
    state_shape: tl.constexpr,
    state_feature_strides: tl.constexpr,
    a_stride: tl.constexpr,
    b_stride: tl.constexpr,
    g_stride: tl.constexpr,
    f_a_stride: tl.constexpr,
    beta_stride: tl.constexpr,
    query_width: tl.constexpr,
    key_width: tl.constexpr,
    value_width: tl.constexpr,
    state_width: tl.constexpr,
    a_width: tl.constexpr,
    b_width: tl.constexpr,
    g_width: tl.constexpr,
    f_a_width: tl.constexpr,
    beta_width: tl.constexpr,
    HAS_A: tl.constexpr,
    HAS_B: tl.constexpr,
    HAS_G: tl.constexpr,
    HAS_F_A: tl.constexpr,
    HAS_BETA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token = offsets // query_width
    feature = offsets % query_width
    mask = (token < num_tokens) & (feature < query_width)
    source_token = tl.load(token_indices + token, mask=mask, other=0)
    tl.store(
        query_out + token * query_width + feature,
        tl.load(query + source_token * query_stride + feature, mask=mask),
        mask=mask,
    )

    token = offsets // key_width
    feature = offsets % key_width
    mask = (token < num_tokens) & (feature < key_width)
    source_token = tl.load(token_indices + token, mask=mask, other=0)
    tl.store(
        key_out + token * key_width + feature,
        tl.load(key + source_token * key_stride + feature, mask=mask),
        mask=mask,
    )

    token = offsets // value_width
    feature = offsets % value_width
    mask = (token < num_tokens) & (feature < value_width)
    source_token = tl.load(token_indices + token, mask=mask, other=0)
    tl.store(
        value_out + token * value_width + feature,
        tl.load(value + source_token * value_stride + feature, mask=mask),
        mask=mask,
    )

    row = offsets // state_width
    feature = offsets % state_width
    mask = (row < num_rows) & (feature < state_width)
    source_row = tl.load(rows + row, mask=mask, other=0)
    state_feature_offset = tl.full((BLOCK,), 0, tl.int32)
    remaining_feature = feature
    for dim in tl.static_range(len(state_shape) - 1, -1, -1):
        state_feature_offset += (
            remaining_feature % state_shape[dim]
        ) * state_feature_strides[dim]
        remaining_feature = remaining_feature // state_shape[dim]
    tl.store(
        recurrent_state_out + row * state_width + feature,
        tl.load(
            recurrent_state + source_row * state_stride + state_feature_offset,
            mask=mask,
        ),
        mask=mask,
    )

    if HAS_A:
        token = offsets // a_width
        feature = offsets % a_width
        mask = (token < num_tokens) & (feature < a_width)
        source_token = tl.load(token_indices + token, mask=mask, other=0)
        tl.store(
            a_out + token * a_width + feature,
            tl.load(a + source_token * a_stride + feature, mask=mask),
            mask=mask,
        )
    if HAS_B:
        token = offsets // b_width
        feature = offsets % b_width
        mask = (token < num_tokens) & (feature < b_width)
        source_token = tl.load(token_indices + token, mask=mask, other=0)
        tl.store(
            b_out + token * b_width + feature,
            tl.load(b + source_token * b_stride + feature, mask=mask),
            mask=mask,
        )
    if HAS_G:
        token = offsets // g_width
        feature = offsets % g_width
        mask = (token < num_tokens) & (feature < g_width)
        source_token = tl.load(token_indices + token, mask=mask, other=0)
        tl.store(
            g_raw_out + token * g_width + feature,
            tl.load(g_raw + source_token * g_stride + feature, mask=mask),
            mask=mask,
        )
    if HAS_F_A:
        token = offsets // f_a_width
        feature = offsets % f_a_width
        mask = (token < num_tokens) & (feature < f_a_width)
        source_token = tl.load(token_indices + token, mask=mask, other=0)
        tl.store(
            f_a_packed + token * f_a_width + feature,
            tl.load(f_a_out + source_token * f_a_stride + feature, mask=mask),
            mask=mask,
        )
    if HAS_BETA:
        token = offsets // beta_width
        feature = offsets % beta_width
        mask = (token < num_tokens) & (feature < beta_width)
        source_token = tl.load(token_indices + token, mask=mask, other=0)
        tl.store(
            beta_raw_out + token * beta_width + feature,
            tl.load(beta_raw + source_token * beta_stride + feature, mask=mask),
            mask=mask,
        )


def _token_width(tensor: torch.Tensor, token_dim: int) -> int:
    return tensor.numel() // tensor.shape[token_dim]


def _allocate_token_pack(
    tensor: torch.Tensor | None, token_dim: int, num_tokens: int
) -> torch.Tensor | None:
    if tensor is None:
        return None
    shape = list(tensor.shape)
    shape[token_dim] = num_tokens
    return torch.empty(shape, dtype=tensor.dtype, device=tensor.device)


def pack_prefill_recurrent_checkpoint_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    recurrent_state: torch.Tensor,
    rows: torch.Tensor,
    token_indices: torch.Tensor,
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    g_raw: torch.Tensor | None,
    f_a_out: torch.Tensor | None,
    beta_raw: torch.Tensor | None,
) -> PackedPrefillCheckpointInputs:
    """Pack every checkpoint prefix and its initial state with one GPU launch.

    Token dimension is 1 for query/key/value (with batch dimension 1) and 0
    for gate tensors. Features within each token or state row must be dense,
    but rows may have noncontiguous strides, as with fused projection slices.
    The returned tensors are contiguous and preserve the corresponding input
    shapes with packed token/request counts.

    Args:
        query: Query tensor with token dimension 1.
        key: Key tensor with token dimension 1.
        value: Value tensor with token dimension 1.
        recurrent_state: Per-request initial recurrent states. Row and feature
            strides may be noncontiguous, including transposed scan results.
        rows: Request rows selected for checkpoint scans.
        token_indices: Source-token indices for all selected checkpoint prefixes.
        a: Optional token-major GDN scan input.
        b: Optional token-major GDN scan input.
        g_raw: Optional token-major KDA gate input.
        f_a_out: Optional token-major KDA gate input.
        beta_raw: Optional token-major KDA beta input.
    Returns:
        Packed scan inputs backed by newly allocated storage.
    """
    num_tokens = token_indices.numel()
    num_rows = rows.numel()
    optional_inputs = (a, b, g_raw, f_a_out, beta_raw)
    if not query.is_cuda:
        return PackedPrefillCheckpointInputs(
            query=query.index_select(1, token_indices),
            key=key.index_select(1, token_indices),
            value=value.index_select(1, token_indices),
            recurrent_state=recurrent_state.index_select(0, rows),
            a=None if a is None else a.index_select(0, token_indices),
            b=None if b is None else b.index_select(0, token_indices),
            g_raw=None if g_raw is None else g_raw.index_select(0, token_indices),
            f_a_out=(
                None if f_a_out is None else f_a_out.index_select(0, token_indices)
            ),
            beta_raw=(
                None if beta_raw is None else beta_raw.index_select(0, token_indices)
            ),
        )

    query_out = _allocate_token_pack(query, 1, num_tokens)
    key_out = _allocate_token_pack(key, 1, num_tokens)
    value_out = _allocate_token_pack(value, 1, num_tokens)
    assert query_out is not None and key_out is not None and value_out is not None
    state_out = torch.empty(
        (num_rows, *recurrent_state.shape[1:]),
        dtype=recurrent_state.dtype,
        device=recurrent_state.device,
    )
    optional_outputs = tuple(
        _allocate_token_pack(tensor, 0, num_tokens) for tensor in optional_inputs
    )

    tensors = (query, key, value, recurrent_state, *optional_inputs)
    token_dims = (1, 1, 1, 0, 0, 0, 0, 0, 0)
    for index, (tensor, token_dim) in enumerate(zip(tensors, token_dims)):
        if tensor is None:
            continue
        if token_dim == 1 and tensor.shape[0] != 1:
            raise ValueError("checkpoint pack query/key/value batch size must be 1")
        if index != 3 and not tensor.select(token_dim, 0).is_contiguous():
            raise ValueError("checkpoint pack inputs must have dense row features")
    strides = tuple(
        1 if tensor is None else tensor.stride(token_dim)
        for tensor, token_dim in zip(tensors, token_dims)
    )
    widths = (
        _token_width(query, 1),
        _token_width(key, 1),
        _token_width(value, 1),
        recurrent_state[0].numel(),
        *(
            1 if tensor is None else _token_width(tensor, 0)
            for tensor in optional_inputs
        ),
    )
    max_elements = max(
        num_tokens * max(widths[0], widths[1], widths[2], *widths[4:]),
        num_rows * widths[3],
    )
    block = 256
    dummy = query
    kernel_outputs = tuple(
        dummy if tensor is None else tensor for tensor in optional_outputs
    )
    kernel_inputs = tuple(
        dummy if tensor is None else tensor for tensor in optional_inputs
    )
    _pack_prefill_recurrent_inputs_kernel[(triton.cdiv(max_elements, block),)](
        token_indices,
        rows,
        query,
        query_out,
        key,
        key_out,
        value,
        value_out,
        recurrent_state,
        state_out,
        kernel_inputs[0],
        kernel_outputs[0],
        kernel_inputs[1],
        kernel_outputs[1],
        kernel_inputs[2],
        kernel_outputs[2],
        kernel_inputs[3],
        kernel_outputs[3],
        kernel_inputs[4],
        kernel_outputs[4],
        num_tokens,
        num_rows,
        query_stride=strides[0],
        key_stride=strides[1],
        value_stride=strides[2],
        state_stride=strides[3],
        state_shape=tuple(recurrent_state.shape[1:]),
        state_feature_strides=tuple(recurrent_state.stride()[1:]),
        a_stride=strides[4],
        b_stride=strides[5],
        g_stride=strides[6],
        f_a_stride=strides[7],
        beta_stride=strides[8],
        query_width=widths[0],
        key_width=widths[1],
        value_width=widths[2],
        state_width=widths[3],
        a_width=widths[4],
        b_width=widths[5],
        g_width=widths[6],
        f_a_width=widths[7],
        beta_width=widths[8],
        HAS_A=a is not None,
        HAS_B=b is not None,
        HAS_G=g_raw is not None,
        HAS_F_A=f_a_out is not None,
        HAS_BETA=beta_raw is not None,
        BLOCK=block,
    )
    return PackedPrefillCheckpointInputs(
        query=query_out,
        key=key_out,
        value=value_out,
        recurrent_state=state_out,
        a=optional_outputs[0],
        b=optional_outputs[1],
        g_raw=optional_outputs[2],
        f_a_out=optional_outputs[3],
        beta_raw=optional_outputs[4],
    )


@triton.jit(do_not_specialize=["num_rows", "state_width"])
def _write_prefill_recurrent_checkpoints_kernel(
    checkpoint_state,
    ssm_states,
    checkpoint_blocks,
    rows,
    num_rows,
    state_width,
    state_dim_2,
    state_dim_3,
    checkpoint_stride_0,
    checkpoint_stride_1,
    checkpoint_stride_2,
    checkpoint_stride_3,
    pool_stride_0,
    pool_stride_1,
    pool_stride_2,
    pool_stride_3,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // state_width
    feature = offsets % state_width
    live = (row < num_rows) & (feature < state_width)
    request_row = tl.load(rows + row, mask=live, other=0)
    destination = tl.load(checkpoint_blocks + request_row, mask=live, other=0).to(
        tl.int64
    )
    dim_1_index = feature // (state_dim_2 * state_dim_3)
    remainder = feature % (state_dim_2 * state_dim_3)
    dim_2_index = remainder // state_dim_3
    dim_3_index = remainder % state_dim_3
    checkpoint_offset = (
        row * checkpoint_stride_0
        + dim_1_index * checkpoint_stride_1
        + dim_2_index * checkpoint_stride_2
        + dim_3_index * checkpoint_stride_3
    )
    pool_offset = (
        destination * pool_stride_0
        + dim_1_index * pool_stride_1
        + dim_2_index * pool_stride_2
        + dim_3_index * pool_stride_3
    )
    value = tl.load(checkpoint_state + checkpoint_offset, mask=live)
    tl.store(ssm_states + pool_offset, value, mask=live)


def write_prefill_recurrent_checkpoints(
    checkpoint_state: torch.Tensor,
    ssm_states: torch.Tensor,
    checkpoint_blocks: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Scatter packed recurrent checkpoint states into the state pool.

    Args:
        checkpoint_state: Dense scan results, one state per selected row.
        ssm_states: Dense recurrent-state pool, updated in place.
        checkpoint_blocks: Per-request checkpoint destination ids.
        rows: Request rows corresponding to ``checkpoint_state``.

    Returns:
        None. ``ssm_states`` is updated in place.
    """
    if rows.numel() == 0:
        return
    if checkpoint_state.ndim != 4 or ssm_states.ndim != 4:
        raise ValueError("recurrent checkpoint states must be four-dimensional")
    if checkpoint_state.shape[1:] != ssm_states.shape[1:]:
        raise ValueError("checkpoint and pool recurrent-state shapes must agree")
    if not checkpoint_state.is_cuda:
        destinations = checkpoint_blocks.index_select(0, rows).to(torch.int64)
        ssm_states.index_copy_(0, destinations, checkpoint_state.to(ssm_states.dtype))
        return
    state_width = checkpoint_state[0].numel()
    block = 256
    _write_prefill_recurrent_checkpoints_kernel[
        (triton.cdiv(rows.numel() * state_width, block),)
    ](
        checkpoint_state,
        ssm_states,
        checkpoint_blocks,
        rows,
        rows.numel(),
        state_width,
        checkpoint_state.shape[2],
        checkpoint_state.shape[3],
        checkpoint_state.stride(0),
        checkpoint_state.stride(1),
        checkpoint_state.stride(2),
        checkpoint_state.stride(3),
        ssm_states.stride(0),
        ssm_states.stride(1),
        ssm_states.stride(2),
        ssm_states.stride(3),
        BLOCK=block,
    )
