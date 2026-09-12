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

"""Fused preparation of convolution and recurrent inputs for state prefill."""

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit
def _prepare_prefill_state_inputs_kernel(
    conv_states,
    ssm_states,
    state_in,
    state_out,
    recurrent,
    has_initial_state,
    conv_width: tl.constexpr,
    conv_state_len: tl.constexpr,
    recurrent_width: tl.constexpr,
    key_dim: tl.constexpr,
    value_dim: tl.constexpr,
    conv_block_stride: tl.constexpr,
    conv_channel_stride: tl.constexpr,
    conv_position_stride: tl.constexpr,
    ssm_block_stride: tl.constexpr,
    ssm_head_stride: tl.constexpr,
    ssm_key_stride: tl.constexpr,
    ssm_value_stride: tl.constexpr,
    in_stride: tl.constexpr,
    out_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    source = tl.load(state_in + row * in_stride).to(tl.int64)
    destination = tl.load(state_out + row * out_stride).to(tl.int64)
    has_history = source > 0
    if tile == 0:
        tl.store(has_initial_state + row, has_history)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)

    # Fresh conv windows stay on the allocator-zeroed working block; in-place
    # evolution needs no copy. Shared input snapshots are never written.
    conv_offset = (
        offsets // conv_state_len * conv_channel_stride
        + offsets % conv_state_len * conv_position_stride
    )
    copy_conv = has_history & (source != destination) & (offsets < conv_width)
    window = tl.load(
        conv_states + source * conv_block_stride + conv_offset,
        mask=copy_conv,
        other=0,
    )
    tl.store(
        conv_states + destination * conv_block_stride + conv_offset,
        window,
        mask=copy_conv,
    )

    state_offset = (
        offsets // (key_dim * value_dim) * ssm_head_stride
        + offsets // value_dim % key_dim * ssm_key_stride
        + offsets % value_dim * ssm_value_stride
    )
    # Never read the null block or stale working-block bytes for a fresh row,
    # even if they contain NaNs. Materialize its logical zero directly.
    state = tl.load(
        ssm_states + source * ssm_block_stride + state_offset,
        mask=has_history & (offsets < recurrent_width),
        other=0,
    )
    tl.store(
        recurrent + row * recurrent_width + offsets,
        state,
        mask=offsets < recurrent_width,
    )


def prepare_prefill_state_inputs(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    state_in_blocks: torch.Tensor,
    state_out_blocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy resumed conv windows and gather/zero recurrent inputs in one launch.

    Args:
        conv_states: Strided [blocks, channels, window] convolution cache.
        ssm_states: Strided [blocks, heads, key, value] recurrent cache.
        state_in_blocks: int32/int64 [requests] source ids; <=0 means fresh.
        state_out_blocks: int32/int64 [requests] unique, positive working ids.
            As required by cache ownership, a destination cannot overwrite
            another request's input snapshot; source==destination is allowed
            for that request's private in-place evolution.

    Returns:
        Contiguous [requests, heads, key, value] recurrent inputs and bool
        [requests] history flags. Fresh recurrent rows are exactly zero;
        fresh conv windows are unchanged. The recurrent cache and null block
        are untouched. GPU supports arbitrary cache strides without index
        casts or scratch gathers; CPU retains the PyTorch reference.
    """
    if conv_states.ndim != 3 or ssm_states.ndim != 4:
        raise ValueError("prefill caches must be 3D conv and 4D recurrent tensors")
    if (
        state_in_blocks.ndim != 1
        or state_out_blocks.shape != state_in_blocks.shape
        or state_in_blocks.dtype not in (torch.int32, torch.int64)
        or state_out_blocks.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("state indices must be equally sized integer vectors")
    if any(
        tensor.device != conv_states.device
        for tensor in (ssm_states, state_in_blocks, state_out_blocks)
    ):
        raise ValueError("prefill caches and indices must be on the same device")
    if not conv_states.is_cuda:
        has_initial_state = state_in_blocks > 0
        safe_inputs = torch.where(
            has_initial_state, state_in_blocks, state_out_blocks
        ).to(torch.int64)
        conv_states[state_out_blocks.long()] = conv_states[safe_inputs]
        recurrent = ssm_states[safe_inputs]
        recurrent.masked_fill_(~has_initial_state[:, None, None, None], 0)
        return recurrent, has_initial_state

    rows = state_in_blocks.numel()
    recurrent = torch.empty(
        (rows, *ssm_states.shape[1:]),
        device=ssm_states.device,
        dtype=ssm_states.dtype,
    )
    has_initial_state = torch.empty(
        rows, dtype=torch.bool, device=state_in_blocks.device
    )
    if rows:
        conv_width = conv_states.shape[1] * conv_states.shape[2]
        recurrent_width = (
            ssm_states.shape[1] * ssm_states.shape[2] * ssm_states.shape[3]
        )
        _prepare_prefill_state_inputs_kernel[
            (rows, triton.cdiv(max(conv_width, recurrent_width), 512))
        ](
            conv_states,
            ssm_states,
            state_in_blocks,
            state_out_blocks,
            recurrent,
            has_initial_state,
            conv_width,
            conv_states.shape[2],
            recurrent_width,
            ssm_states.shape[2],
            ssm_states.shape[3],
            *conv_states.stride(),
            *ssm_states.stride(),
            state_in_blocks.stride(0),
            state_out_blocks.stride(0),
            BLOCK=512,
        )
    return recurrent, has_initial_state
