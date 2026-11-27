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


# Pool block counts change with the pool size and the index vectors are often
# offset slices of larger buffers, so neither may key a new specialization.
@triton.jit(
    do_not_specialize=["conv_num_blocks", "ssm_num_blocks"],
    do_not_specialize_on_alignment=["state_in", "state_out"],
)
def _prepare_prefill_state_inputs_kernel(
    conv_states,
    ssm_states,
    state_in,
    state_out,
    recurrent,
    has_initial_state,
    conv_num_blocks,
    ssm_num_blocks,
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
    # The request count scales row * recurrent_width past int32.
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    source = tl.load(state_in + row * in_stride).to(tl.int64)
    destination = tl.load(state_out + row * out_stride).to(tl.int64)
    # Block 0 is the null block and ids <= 0 or past either pool are not
    # pages of that pool, so each pool bounds source and destination itself.
    valid_conv_src = (source > 0) & (source < conv_num_blocks)
    valid_conv_dst = (destination > 0) & (destination < conv_num_blocks)
    valid_ssm_src = (source > 0) & (source < ssm_num_blocks)
    if tile == 0:
        tl.store(has_initial_state + row, valid_ssm_src)
    tile_start = tile * BLOCK
    offsets = tile_start + tl.arange(0, BLOCK)

    # Fresh conv windows stay on the allocator-zeroed working block; in-place
    # evolution needs no copy. Shared input snapshots are never written. The
    # branch is uniform per program, so tiles past the conv window, which are
    # most of the grid, skip the index math and the masked copy entirely.
    if (
        valid_conv_src
        & valid_conv_dst
        & (source != destination)
        & (tile_start < conv_width)
    ):
        conv_offset = (
            offsets // conv_state_len * conv_channel_stride
            + offsets % conv_state_len * conv_position_stride
        )
        copy_conv = offsets < conv_width
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

    if tile_start < recurrent_width:
        in_state = offsets < recurrent_width
        if valid_ssm_src:
            state_offset = (
                offsets // (key_dim * value_dim) * ssm_head_stride
                + offsets // value_dim % key_dim * ssm_key_stride
                + offsets % value_dim * ssm_value_stride
            )
            state = tl.load(
                ssm_states + source * ssm_block_stride + state_offset,
                mask=in_state,
                other=0,
            )
            tl.store(recurrent + row * recurrent_width + offsets, state, mask=in_state)
        else:
            # Never read the null block, stale working-block bytes, or memory
            # past the pool for a row without history, even if they contain
            # NaNs. Materialize its logical zero directly.
            tl.store(
                recurrent + row * recurrent_width + offsets,
                tl.zeros((BLOCK,), dtype=recurrent.dtype.element_ty),
                mask=in_state,
            )


def prepare_prefill_state_inputs(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    state_in_blocks: torch.Tensor,
    state_out_blocks: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy resumed conv windows and gather/zero recurrent inputs in one launch.

    Block ids are pages of each cache pool. Only ids in the open interval
    ``(0, num_blocks)`` of a pool are read or written there: block 0 is the
    reserved null block, and ids ``<= 0`` (including ``-1`` padding) or
    ``>= num_blocks`` are skipped. Each id is checked against each pool on its
    own, so the conv and recurrent pools may differ in size.

    Args:
        conv_states: Strided [conv_num_blocks, channels, window] convolution
            cache.
        ssm_states: Strided [ssm_num_blocks, heads, key, value] recurrent cache.
        state_in_blocks: int32/int64 [requests] source ids. A row has history
            only when its source lies in ``(0, ssm_num_blocks)``.
        state_out_blocks: int32/int64 [requests] working ids. A row's conv
            window is copied only when its source and destination both lie in
            ``(0, conv_num_blocks)`` and differ. As required by cache
            ownership, a copied destination cannot overwrite another request's
            input snapshot; source==destination is allowed for that request's
            private in-place evolution.
        dtype: Recurrent input dtype; conversion is fused into the gather.

    Returns:
        Contiguous [requests, heads, key, value] recurrent inputs and bool
        [requests] history flags. Rows without history are exactly zero; conv
        windows of rows without a copy are unchanged. The recurrent cache, the
        null block of each cache, and memory outside both caches are untouched.
        CUDA, ROCm, and Ascend support arbitrary cache strides without index
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
    if conv_states.device.type not in ("cuda", "npu"):
        source = state_in_blocks.long()
        destination = state_out_blocks.long()
        has_initial_state = (source > 0) & (source < ssm_states.shape[0])
        copy_conv = (
            (source > 0)
            & (source < conv_states.shape[0])
            & (destination > 0)
            & (destination < conv_states.shape[0])
            & (source != destination)
        )
        conv_states[destination[copy_conv]] = conv_states[source[copy_conv]]
        recurrent = torch.zeros(
            (source.numel(), *ssm_states.shape[1:]),
            device=ssm_states.device,
            dtype=dtype,
        )
        recurrent[has_initial_state] = ssm_states[source[has_initial_state]].to(dtype)
        return recurrent, has_initial_state

    rows = state_in_blocks.numel()
    recurrent = torch.empty(
        (rows, *ssm_states.shape[1:]),
        device=ssm_states.device,
        dtype=dtype,
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
            conv_states.shape[0],
            ssm_states.shape[0],
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
