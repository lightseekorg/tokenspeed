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

"""Validation and launch boundary for page-owned KDA history."""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform


def _kernels():
    if current_platform().is_amd:
        from tokenspeed_kernel_amd.ops import _kda_records

        return _kda_records, {"WAVE": 64 if current_platform().is_cdna4 else 32}
    from tokenspeed_kernel.ops.attention.kda._triton import retained

    return retained, {}


def materialize(recurrent, *, history, source_indices, recurrent_layout):
    pages, heads, capacity, values = history.correction.shape
    keys = history.key.shape[-1]
    expected = (values, keys) if recurrent_layout == "v_major" else (keys, values)
    if recurrent_layout not in ("v_major", "k_major"):
        raise ValueError("unsupported recurrent layout")
    if (
        recurrent.shape != (source_indices.numel(), heads, *expected)
        or recurrent.dtype not in (torch.float32, torch.bfloat16)
        or source_indices.ndim != 1
        or source_indices.dtype not in (torch.int32, torch.int64)
        or source_indices.device != recurrent.device
        or history.correction.device != recurrent.device
    ):
        raise ValueError(
            "gathered recurrent state and source indices must match retained history"
        )
    if source_indices.numel() == 0:
        return
    kernels, options = _kernels()
    kernels.retained_materialize_kernel[
        (source_indices.numel(), heads, triton.cdiv(keys * values, 256))
    ](
        recurrent,
        history.correction,
        history.key,
        history.decay,
        history.length,
        source_indices,
        recurrent.stride(),
        history.correction.stride(),
        history.key.stride(),
        history.decay.stride(),
        history.length.stride(0),
        source_indices.stride(0),
        pages,
        K=keys,
        V=values,
        CAPACITY=capacity,
        V_MAJOR=recurrent_layout == "v_major",
        BLOCK=256,
        num_warps=4,
        **options,
    )


def reset_length(length, *, write_indices):
    if (
        length.ndim != 2
        or length.shape[1] != 1
        or length.dtype != torch.int32
        or write_indices.ndim != 1
        or write_indices.dtype not in (torch.int32, torch.int64)
        or write_indices.device != length.device
    ):
        raise ValueError("length must be INT32 [page, 1] and indices an integer vector")
    if write_indices.numel() == 0:
        return
    kernels, options = _kernels()
    kernels.retained_reset_length_kernel[(triton.cdiv(write_indices.numel(), 256),)](
        length,
        write_indices,
        write_indices.numel(),
        length.shape[0],
        length.stride(0),
        write_indices.stride(0),
        BLOCK=256,
        num_warps=4,
        **options,
    )


def batched_commit(
    *,
    descriptors,
    group_indices,
    read_indices,
    write_indices,
    accepted_length,
    query_start_loc,
    record_metadata,
    record_stamps,
    draft_token_num,
    history_capacity,
    num_heads,
    head_dim,
    conv_width,
    activation_dtype,
    conv_state_dtype,
):
    """Update all layers' convolution, then publish their accepted lengths.

    descriptors is contiguous UINT64 [layers, 9]: original QKV, convolution,
    length pointers; page count; QKV row, convolution page, length page,
    convolution channel, convolution tap element strides. QKV channels have
    unit stride. group_indices maps layers into [groups, requests] page tables.
    Each live destination is private and cannot overlap another active source.
    Per-layer descriptor values are validated by the cache owner when binding
    its arena; the launch validates shared tensor geometry without GPU reads.

    query_start_loc is the packed [requests + 1] boundary vector. Acceptance is
    clamped to the actual interval and draft_token_num. metadata is contiguous
    INT64 [requests, 3]; stamps is INT64 [layers, slots, ring capacity, 2],
    including strided layer subsets. Every accepted stamp must match before
    either phase writes. history_capacity and ring capacity are independent.
    No dense checkpoints or history vectors are copied by this operation.
    """
    if (
        num_heads <= 0
        or head_dim <= 0
        or conv_width != 4
        or draft_token_num <= 0
        or history_capacity < draft_token_num
    ):
        raise ValueError(
            "retained commit requires positive heads/window, positive head_dim, conv_width=4 and sufficient capacity"
        )
    types = {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }
    if (
        activation_dtype not in (torch.bfloat16, torch.float16)
        or conv_state_dtype not in types
    ):
        raise ValueError("unsupported retained activation or convolution dtype")
    if (
        descriptors.ndim != 2
        or descriptors.shape[1] != 9
        or descriptors.dtype != torch.uint64
        or not descriptors.is_contiguous()
    ):
        raise ValueError("retained descriptors must be contiguous UINT64 [layers, 9]")
    layers = descriptors.shape[0]
    batch = accepted_length.numel()
    if (
        group_indices.shape != (layers,)
        or not group_indices.is_contiguous()
        or read_indices.ndim != 2
        or read_indices.shape[1] != batch
        or write_indices.shape != read_indices.shape
        or accepted_length.shape != (batch,)
        or query_start_loc.shape != (batch + 1,)
    ):
        raise ValueError(
            "retained commit requires layer groups, [groups, batch] pages and packed boundaries"
        )
    if any(
        x.dtype not in (torch.int32, torch.int64)
        for x in (
            group_indices,
            read_indices,
            write_indices,
            accepted_length,
            query_start_loc,
        )
    ):
        raise ValueError(
            "retained routing and accepted lengths must be integer tensors"
        )
    if (
        record_metadata.shape != (batch, 3)
        or not record_metadata.is_contiguous()
        or record_stamps.ndim != 4
        or record_stamps.shape[0] != layers
        or record_stamps.shape[-1] != 2
        or record_stamps.shape[2] < draft_token_num
        or record_metadata.dtype != torch.int64
        or record_stamps.dtype != torch.int64
    ):
        raise ValueError(
            "retained metadata/stamps must be INT64 [batch,3] and [layers,slots,capacity,2]"
        )
    if any(
        x.device != descriptors.device
        for x in (
            group_indices,
            read_indices,
            write_indices,
            accepted_length,
            query_start_loc,
            record_metadata,
            record_stamps,
        )
    ):
        raise ValueError("retained commit tensors must share a device")
    if not layers or not batch:
        return
    kernels, options = _kernels()
    args = (
        descriptors,
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        query_start_loc,
        record_metadata,
        record_stamps,
        read_indices.shape[0],
        read_indices.stride(),
        write_indices.stride(),
        accepted_length.stride(0),
        query_start_loc.stride(0),
        record_stamps.stride(),
        record_stamps.shape[1],
        record_stamps.shape[2],
    )
    kernels.retained_batched_conv_kernel[
        (layers, batch, triton.cdiv(3 * num_heads * head_dim, 256))
    ](
        *args,
        CAPACITY=history_capacity,
        WINDOW=draft_token_num,
        CHANNELS=3 * num_heads * head_dim,
        ACTIVATION=types[activation_dtype],
        CONV=types[conv_state_dtype],
        BLOCK=256,
        num_warps=4,
        **options,
    )
    kernels.retained_batched_publish_kernel[(layers, batch)](
        *args,
        CAPACITY=history_capacity,
        WINDOW=draft_token_num,
        num_warps=4,
        **options,
    )
