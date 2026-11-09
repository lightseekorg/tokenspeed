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

"""Experimental NVIDIA GDN replay with a bounded grid and asynchronous prefetch.

The launcher also has an opt-in registry adapter for end-to-end profiling.
One work item owns a complete value head; no V split or extra request state is
needed. All source rows must be independent of other work items' destinations,
as in the ordinary replay commit (in-place updates within one item are allowed).
"""

from typing import Any

import torch
from tokenspeed_kernel._triton import (
    gl,
    gluon,
)
from tokenspeed_kernel._triton import gluon_nvidia_async_copy as cp
from tokenspeed_kernel._triton import (
    tl,
    triton,
)
from tokenspeed_kernel.ops.attention.gdn import (
    validate_replay_commit_args,
    validate_replay_commit_setup,
)
from tokenspeed_kernel.platform import CapabilityRequirement, pdl_enabled
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

mb = gl.nvidia.ampere.mbarrier


@gluon.jit
def _load_replay_descriptor(
    parameters,
    state_addresses,
    state_row_strides,
    layer,
    head,
    HV: gl.constexpr,
    STATE_DTYPE_CODE: gl.constexpr,
):
    address = gl.load(state_addresses + layer)
    stride = gl.load(state_row_strides + layer).to(gl.int64)
    if STATE_DTYPE_CODE == 0:
        pool = address.to(gl.pointer_type(gl.bfloat16))
    elif STATE_DTYPE_CODE == 1:
        pool = address.to(gl.pointer_type(gl.float16))
    else:
        pool = address.to(gl.pointer_type(gl.float32))
    raw_a_log = gl.load(parameters + layer * 2 * HV + head)
    dt_bias = gl.load(parameters + layer * 2 * HV + HV + head)
    return pool, stride, raw_a_log, dt_bias


@gluon.jit
def _prefetch_replay_work(
    payload,
    pool,
    stride,
    read_indices,
    write_indices,
    accepted_length,
    layer,
    request,
    head,
    buffer_index,
    state_buffers,
    key_buffers,
    value_buffers,
    gate_buffers,
    B,
    T,
    PAYLOAD_LAYER_STRIDE,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    TOKEN_BLOCK: gl.constexpr,
    GATE_BLOCK: gl.constexpr,
    STATE_DTYPE_CODE: gl.constexpr,
    SPT_K: gl.constexpr,
):
    key_head = head // (HV // H)
    read = gl.load(read_indices + layer * B + request).to(gl.int64)
    write = gl.load(write_indices + layer * B + request).to(gl.int64)
    steps = gl.load(accepted_length + request)
    steps = gl.where(write >= 0, gl.minimum(gl.maximum(steps, 0), T), 0)
    storage_dtype: gl.constexpr = gl.uint32 if STATE_DTYPE_CODE == 2 else gl.uint16

    copy_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [4, 8], [gl.num_warps(), 1], [1, 0]
    )
    tokens = gl.arange(0, TOKEN_BLOCK, layout=gl.SliceLayout(1, copy_layout))
    key_offsets = gl.arange(0, K, layout=gl.SliceLayout(0, copy_layout))
    value_offsets = gl.arange(0, V, layout=gl.SliceLayout(0, copy_layout))
    gate_offsets = gl.arange(0, GATE_BLOCK, layout=gl.SliceLayout(0, copy_layout))
    width: gl.constexpr = H * K + HV * V + 2 * HV
    # Every contiguous layer contains whole payload rows, even when its row
    # capacity changes. Retain that alignment without specializing its stride.
    payload_alignment: gl.constexpr = width & -width
    layer_offset = gl.multiple_of(
        layer.to(gl.int64) * PAYLOAD_LAYER_STRIDE, payload_alignment
    )
    rows = (
        payload
        + layer_offset
        + (request.to(gl.int64) * T + tokens.to(gl.int64)) * width
    )
    cp.async_load(
        key_buffers.index(buffer_index),
        rows[:, None] + key_head * K + key_offsets[None, :],
        (tokens < steps)[:, None],
    )
    cp.async_load(
        value_buffers.index(buffer_index),
        rows[:, None] + H * K + head * V + value_offsets[None, :],
        (tokens < steps)[:, None],
    )
    # Copy the contiguous gates together, so even half-precision gates have
    # four-byte copy granularity. The consumer selects this head's two columns.
    cp.async_load(
        gate_buffers.index(buffer_index),
        rows[:, None] + H * K + HV * V + gate_offsets[None, :],
        (tokens < steps)[:, None] & (gate_offsets < 2 * HV)[None, :],
    )
    cp.commit_group()
    state_layout: gl.constexpr = gl.BlockedLayout(
        [SPT_K, 1], [8, 4], [1, gl.num_warps()], [0, 1]
    )
    keys = gl.arange(0, K, layout=gl.SliceLayout(1, state_layout))
    values = gl.arange(0, V, layout=gl.SliceLayout(0, state_layout))
    base = pool + gl.maximum(read, 0) * stride + head * V * K
    state_mask = gl.full([K, V], (read >= 0) & (write >= 0), gl.int1, state_layout)
    # Odd pool row strides can leave BF16/FP16 sources only two-byte aligned.
    # cp.async needs at least four bytes; retain the ordinary load fallback.
    if (base.to(gl.uint64) & 15) == 0:
        aligned = gl.multiple_of(base.to(gl.pointer_type(storage_dtype)), 16)
        cp.async_load(
            state_buffers.index(buffer_index),
            aligned + values[None, :] * K + keys[:, None],
            state_mask,
        )
    else:
        state_buffers.index(buffer_index).store(
            gl.load(
                base + values[None, :] * K + keys[:, None],
                state_mask,
                other=0,
            ).to(storage_dtype, bitcast=True)
        )

    cp.commit_group()
    return write, steps


@gluon.jit
def _replay_producer(
    inputs,
    buffers,
    barriers,
    sizes,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    TOKEN_BLOCK: gl.constexpr,
    GATE_BLOCK: gl.constexpr,
    STATE_DTYPE_CODE: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    SPT_K: gl.constexpr,
    PRODUCER_INDEX: gl.constexpr,
    PRODUCER_COUNT: gl.constexpr,
):
    (
        payload,
        parameters,
        state_addresses,
        state_row_strides,
        read_indices,
        write_indices,
        accepted_length,
    ) = inputs
    (
        state_buffers,
        key_buffers,
        value_buffers,
        gate_buffers,
        metadata_buffers,
        normalized_buffers,
        factor_buffers,
    ) = buffers
    full_bars, empty_bars = barriers
    B, T, L, PAYLOAD_LAYER_STRIDE = sizes
    first = gl.program_id(0) + PRODUCER_INDEX * gl.num_programs(0)
    grid = gl.num_programs(0) * PRODUCER_COUNT
    total = L * B * HV
    if first < total:
        layer = first // (B * HV)
        request = first // HV % B
        head = first % HV
        layer_step = grid // (B * HV)
        request_step = grid // HV % B
        head_step = grid % HV
        request_rotation = layer % B
        rotation_step = layer_step % B
        pool, stride, raw_a_log, dt_bias = _load_replay_descriptor(
            parameters,
            state_addresses,
            state_row_strides,
            layer,
            head,
            HV,
            STATE_DTYPE_CODE,
        )
        metadata_layout: gl.constexpr = gl.BlockedLayout(
            [1], [32], [gl.num_warps()], [0]
        )
        columns = gl.arange(0, 8, layout=metadata_layout)
        key_layout: gl.constexpr = gl.BlockedLayout(
            [1, 2 if STATE_DTYPE_CODE == 2 else 4], [4, 8], [gl.num_warps(), 1], [1, 0]
        )
        gate_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
        gate_column_layout: gl.constexpr = gl.BlockedLayout(
            [1, 1], [32, 1], [gl.num_warps(), 1], [0, 1]
        )
        iteration = PRODUCER_INDEX
        for work in range(first, total, grid):
            slot = iteration % NUM_BUFFERS
            phase = iteration // NUM_BUFFERS & 1
            mb.wait(empty_bars.index(slot), phase ^ 1)
            # Rotate the per-layer permutation to avoid keeping one CTA on
            # only the long or only the short accepted prefixes.
            physical_request = request + request_rotation
            physical_request = gl.where(
                physical_request >= B, physical_request - B, physical_request
            )
            write, steps = _prefetch_replay_work(
                payload,
                pool,
                stride,
                read_indices,
                write_indices,
                accepted_length,
                layer,
                physical_request,
                head,
                slot,
                state_buffers,
                key_buffers,
                value_buffers,
                gate_buffers,
                B,
                T,
                PAYLOAD_LAYER_STRIDE,
                H,
                HV,
                K,
                V,
                TOKEN_BLOCK,
                GATE_BLOCK,
                STATE_DTYPE_CODE,
                SPT_K,
            )
            metadata = gl.where(
                columns == 0,
                pool.to(gl.uint64),
                gl.where(
                    columns == 1,
                    stride.to(gl.uint64),
                    gl.where(
                        columns == 2,
                        write.to(gl.uint64),
                        gl.where(
                            columns == 3,
                            steps.to(gl.uint64),
                            gl.where(
                                columns == 4,
                                head.to(gl.uint64),
                                gl.where(
                                    columns == 5,
                                    raw_a_log.to(gl.uint32, bitcast=True).to(gl.uint64),
                                    dt_bias.to(gl.uint32, bitcast=True).to(gl.uint64),
                                ),
                            ),
                        ),
                    ),
                ),
            )
            metadata_buffers.index(slot).store(metadata)
            cp.wait_group(1)
            gl.barrier()
            key = key_buffers.index(slot).load(key_layout).to(gl.float32)
            # Retain the exact consumer normalization expression and reduction map.
            key = key / gl.sqrt(gl.sum(key * key, 1) + 1e-6)[:, None]
            normalized_buffers.index(slot).store(key)
            gates = gate_buffers.index(slot)
            raw_a = gl.convert_layout(
                gl.reshape(
                    gates.gather(
                        gl.full([TOKEN_BLOCK, 1], head, gl.int32, gate_column_layout), 1
                    ),
                    [TOKEN_BLOCK],
                ),
                gate_layout,
            ).to(gl.float32)
            raw_b = gl.convert_layout(
                gl.reshape(
                    gates.gather(
                        gl.full(
                            [TOKEN_BLOCK, 1], HV + head, gl.int32, gate_column_layout
                        ),
                        1,
                    ),
                    [TOKEN_BLOCK],
                ),
                gate_layout,
            ).to(gl.float32)
            x = raw_a + dt_bias
            softplus = gl.where(x <= 20.0, gl.log(1.0 + gl.exp(x)), x)
            decay = gl.exp(-gl.exp(raw_a_log) * softplus)
            beta = 1.0 / (1.0 + gl.exp(-raw_b))
            factor_buffers.index(slot).slice(0, TOKEN_BLOCK).store(decay)
            factor_buffers.index(slot).slice(TOKEN_BLOCK, TOKEN_BLOCK).store(beta)
            cp.wait_group(0)
            gl.barrier()
            mb.arrive(full_bars.index(slot))
            next_head = head + head_step
            head_carry = next_head >= HV
            next_head = gl.where(head_carry, next_head - HV, next_head)
            next_request = request + request_step + head_carry.to(gl.int32)
            request_carry = next_request >= B
            next_request = gl.where(request_carry, next_request - B, next_request)
            next_layer = layer + layer_step + request_carry.to(gl.int32)
            if work + grid < total:
                if (next_layer != layer) | (next_head != head):
                    pool, stride, raw_a_log, dt_bias = _load_replay_descriptor(
                        parameters,
                        state_addresses,
                        state_row_strides,
                        next_layer,
                        next_head,
                        HV,
                        STATE_DTYPE_CODE,
                    )
            request_rotation = (
                request_rotation + rotation_step + request_carry.to(gl.int32)
            )
            request_rotation = gl.where(
                request_rotation >= B, request_rotation - B, request_rotation
            )
            layer, request, head = next_layer, next_request, next_head
            iteration += PRODUCER_COUNT


@gluon.jit
def _metadata_scalar(metadata, column: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    return gl.sum(metadata.slice(column, 1).load(layout), 0)


@gluon.jit
def _replay_consumer(
    inputs,
    buffers,
    barriers,
    sizes,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    TOKEN_BLOCK: gl.constexpr,
    GATE_BLOCK: gl.constexpr,
    STATE_DTYPE_CODE: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    SPT_K: gl.constexpr,
):
    (
        payload,
        parameters,
        state_addresses,
        state_row_strides,
        read_indices,
        write_indices,
        accepted_length,
    ) = inputs
    (
        state_buffers,
        key_buffers,
        value_buffers,
        gate_buffers,
        metadata_buffers,
        normalized_buffers,
        factor_buffers,
    ) = buffers
    full_bars, empty_bars = barriers
    B, T, L, PAYLOAD_LAYER_STRIDE = sizes
    if STATE_DTYPE_CODE == 0:
        state_dtype: gl.constexpr = gl.bfloat16
        storage_dtype: gl.constexpr = gl.uint16
    elif STATE_DTYPE_CODE == 1:
        state_dtype: gl.constexpr = gl.float16
        storage_dtype: gl.constexpr = gl.uint16
    else:
        state_dtype: gl.constexpr = gl.float32
        storage_dtype: gl.constexpr = gl.uint32
    key_register_layout: gl.constexpr = gl.BlockedLayout(
        [TOKEN_BLOCK, SPT_K], [4, 8], [gl.num_warps(), 1], [1, 0]
    )
    value_register_layout: gl.constexpr = gl.BlockedLayout(
        [TOKEN_BLOCK, 1], [8, 4], [1, gl.num_warps()], [1, 0]
    )
    state_layout: gl.constexpr = gl.BlockedLayout(
        [SPT_K, 1], [8, 4], [1, gl.num_warps()], [0, 1]
    )
    key_layout: gl.constexpr = gl.BlockedLayout(
        [1, 2 if STATE_DTYPE_CODE == 2 else 4],
        [4, 8],
        [gl.num_warps(), 1],
        [1, 0],
    )
    key_read_layout: gl.constexpr = gl.BlockedLayout(
        [1, SPT_K], [4, 8], [gl.num_warps(), 1], [1, 0]
    )
    gate_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    gate_column_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [32, 1], [gl.num_warps(), 1], [0, 1]
    )
    offsets_k = gl.arange(0, K, layout=gl.SliceLayout(1, state_layout))
    offsets_v = gl.arange(0, V, layout=gl.SliceLayout(0, state_layout))
    iteration = 0
    for work in range(gl.program_id(0), L * B * HV, gl.num_programs(0)):
        buffer_index = iteration % NUM_BUFFERS
        phase = iteration // NUM_BUFFERS & 1
        mb.wait(full_bars.index(buffer_index), phase)
        metadata = metadata_buffers.index(buffer_index)
        address = _metadata_scalar(metadata, 0)
        if STATE_DTYPE_CODE == 0:
            pool = address.to(gl.pointer_type(gl.bfloat16))
        elif STATE_DTYPE_CODE == 1:
            pool = address.to(gl.pointer_type(gl.float16))
        else:
            pool = address.to(gl.pointer_type(gl.float32))
        stride = _metadata_scalar(metadata, 1).to(gl.int64)
        write = _metadata_scalar(metadata, 2).to(gl.int64)
        steps = _metadata_scalar(metadata, 3).to(gl.int32)
        head = _metadata_scalar(metadata, 4).to(gl.int32)
        raw_a_log = (
            _metadata_scalar(metadata, 5).to(gl.uint32).to(gl.float32, bitcast=True)
        )
        dt_bias = (
            _metadata_scalar(metadata, 6).to(gl.uint32).to(gl.float32, bitcast=True)
        )
        key_registers = normalized_buffers.index(buffer_index).load(key_register_layout)
        decay = (
            factor_buffers.index(buffer_index).slice(0, TOKEN_BLOCK).load(gate_layout)
        )
        beta = (
            factor_buffers.index(buffer_index)
            .slice(TOKEN_BLOCK, TOKEN_BLOCK)
            .load(gate_layout)
        )
        state = (
            state_buffers.index(buffer_index)
            .load(state_layout)
            .to(state_dtype, bitcast=True)
            .to(gl.float32)
        )
        value_registers = (
            value_buffers.index(buffer_index).load(value_register_layout).to(gl.float32)
        )
        # Every input has now been cached in registers by all consumer warps.
        # The producer can recycle the slot while the recurrence computes.
        gl.barrier()
        mb.arrive(empty_bars.index(buffer_index))
        if steps == TOKEN_BLOCK:
            for step in gl.static_range(TOKEN_BLOCK):
                index = gl.full([1], step, gl.int32, gate_layout)
                step_decay = gl.sum(gl.gather(decay, index, 0), 0)
                step_beta = gl.sum(gl.gather(beta, index, 0), 0)
                k = gl.convert_layout(
                    gl.reshape(
                        gl.gather(
                            key_registers,
                            gl.full([1, K], step, gl.int32, key_register_layout),
                            0,
                        ),
                        [K],
                    ),
                    gl.SliceLayout(1, state_layout),
                )
                v = gl.convert_layout(
                    gl.reshape(
                        gl.gather(
                            value_registers,
                            gl.full([1, V], step, gl.int32, value_register_layout),
                            0,
                        ),
                        [V],
                    ),
                    gl.SliceLayout(0, state_layout),
                )
                decayed = state * step_decay
                correction = v - gl.sum(decayed * k[:, None], 0)
                state = decayed + k[:, None] * (correction * step_beta)[None, :]
        else:
            for step in gl.static_range(TOKEN_BLOCK):
                if step < steps:
                    index = gl.full([1], step, gl.int32, gate_layout)
                    step_decay = gl.sum(gl.gather(decay, index, 0), 0)
                    step_beta = gl.sum(gl.gather(beta, index, 0), 0)
                    k = gl.convert_layout(
                        gl.reshape(
                            gl.gather(
                                key_registers,
                                gl.full([1, K], step, gl.int32, key_register_layout),
                                0,
                            ),
                            [K],
                        ),
                        gl.SliceLayout(1, state_layout),
                    )
                    v = gl.convert_layout(
                        gl.reshape(
                            gl.gather(
                                value_registers,
                                gl.full([1, V], step, gl.int32, value_register_layout),
                                0,
                            ),
                            [V],
                        ),
                        gl.SliceLayout(0, state_layout),
                    )
                    decayed = state * step_decay
                    correction = v - gl.sum(decayed * k[:, None], 0)
                    state = decayed + k[:, None] * (correction * step_beta)[None, :]
        if write >= 0:
            destination = pool + write * stride + head * V * K
            if (destination.to(gl.uint64) & 15) == 0:
                aligned = gl.multiple_of(
                    destination.to(gl.pointer_type(storage_dtype)), 16
                )
                bits = state.to(state_dtype).to(storage_dtype, bitcast=True)
                gl.store(aligned + offsets_v[None, :] * K + offsets_k[:, None], bits)
            else:
                gl.store(
                    destination + offsets_v[None, :] * K + offsets_k[:, None], state
                )

        iteration += 1


@gluon.jit(do_not_specialize=["B", "T", "L", "PAYLOAD_LAYER_STRIDE"])
def _gdn_replay_commit_persistent_kernel(
    payload,
    parameters,
    state_addresses,
    state_row_strides,
    read_indices,
    write_indices,
    accepted_length,
    B,
    T,
    L,
    PAYLOAD_LAYER_STRIDE,
    H: gl.constexpr,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    TOKEN_BLOCK: gl.constexpr,
    GATE_BLOCK: gl.constexpr,
    STATE_DTYPE_CODE: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    SPT_K: gl.constexpr,
    ENABLE_PDL: gl.constexpr,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    storage_dtype: gl.constexpr = gl.uint32 if STATE_DTYPE_CODE == 2 else gl.uint16
    state_buffers = gl.allocate_shared_memory(
        storage_dtype, [NUM_BUFFERS, K, V], gl.SwizzledSharedLayout(1, 1, 1, [0, 1])
    )
    key_buffers = gl.allocate_shared_memory(
        payload.dtype.element_ty,
        [NUM_BUFFERS, TOKEN_BLOCK, K],
        gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
    )
    value_buffers = gl.allocate_shared_memory(
        payload.dtype.element_ty,
        [NUM_BUFFERS, TOKEN_BLOCK, V],
        gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
    )
    gate_buffers = gl.allocate_shared_memory(
        payload.dtype.element_ty,
        [NUM_BUFFERS, TOKEN_BLOCK, GATE_BLOCK],
        gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
    )
    metadata_buffers = gl.allocate_shared_memory(
        gl.uint64, [NUM_BUFFERS, 8], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    # Four FP32 padding elements per eight keys remove the stride-induced
    # bank conflict of the consumer's 128-bit normalized-key reads.
    normalized_buffers = gl.allocate_shared_memory(
        gl.float32,
        [NUM_BUFFERS, TOKEN_BLOCK, K],
        gl.PaddedSharedLayout.with_identity_for([[8, 4]], [TOKEN_BLOCK, K], [1, 0]),
    )
    factor_buffers = gl.allocate_shared_memory(
        gl.float32,
        [NUM_BUFFERS, 2 * TOKEN_BLOCK],
        gl.SwizzledSharedLayout(1, 1, 1, [0]),
    )
    full_bars = gl.allocate_shared_memory(
        gl.int64, [NUM_BUFFERS, 1], mb.MBarrierLayout()
    )
    empty_bars = gl.allocate_shared_memory(
        gl.int64, [NUM_BUFFERS, 1], mb.MBarrierLayout()
    )
    for slot in gl.static_range(NUM_BUFFERS):
        mb.init(full_bars.index(slot), count=1)
        mb.init(empty_bars.index(slot), count=1)
    inputs = (
        payload,
        parameters,
        state_addresses,
        state_row_strides,
        read_indices,
        write_indices,
        accepted_length,
    )
    buffers = (
        state_buffers,
        key_buffers,
        value_buffers,
        gate_buffers,
        metadata_buffers,
        normalized_buffers,
        factor_buffers,
    )
    barriers = (full_bars, empty_bars)
    sizes = (B, T, L, PAYLOAD_LAYER_STRIDE)
    if NUM_BUFFERS == 2:
        gl.warp_specialize(
            [
                (
                    _replay_consumer,
                    (
                        inputs,
                        buffers,
                        barriers,
                        sizes,
                        H,
                        HV,
                        K,
                        V,
                        TOKEN_BLOCK,
                        GATE_BLOCK,
                        STATE_DTYPE_CODE,
                        NUM_BUFFERS,
                        SPT_K,
                    ),
                ),
                (
                    _replay_producer,
                    (
                        inputs,
                        buffers,
                        barriers,
                        sizes,
                        H,
                        HV,
                        K,
                        V,
                        TOKEN_BLOCK,
                        GATE_BLOCK,
                        STATE_DTYPE_CODE,
                        NUM_BUFFERS,
                        SPT_K,
                        0,
                        2,
                    ),
                ),
                (
                    _replay_producer,
                    (
                        inputs,
                        buffers,
                        barriers,
                        sizes,
                        H,
                        HV,
                        K,
                        V,
                        TOKEN_BLOCK,
                        GATE_BLOCK,
                        STATE_DTYPE_CODE,
                        NUM_BUFFERS,
                        SPT_K,
                        1,
                        2,
                    ),
                ),
            ],
            [1, 1],
            [64, 64],
        )
    else:
        gl.warp_specialize(
            [
                (
                    _replay_consumer,
                    (
                        inputs,
                        buffers,
                        barriers,
                        sizes,
                        H,
                        HV,
                        K,
                        V,
                        TOKEN_BLOCK,
                        GATE_BLOCK,
                        STATE_DTYPE_CODE,
                        NUM_BUFFERS,
                        SPT_K,
                    ),
                ),
                (
                    _replay_producer,
                    (
                        inputs,
                        buffers,
                        barriers,
                        sizes,
                        H,
                        HV,
                        K,
                        V,
                        TOKEN_BLOCK,
                        GATE_BLOCK,
                        STATE_DTYPE_CODE,
                        NUM_BUFFERS,
                        SPT_K,
                        0,
                        1,
                    ),
                ),
            ],
            [1],
            [64],
        )
    for slot in gl.static_range(NUM_BUFFERS):
        mb.invalidate(full_bars.index(slot))
        mb.invalidate(empty_bars.index(slot))


def launch_gdn_replay_commit_persistent(
    payload: torch.Tensor,
    parameters: torch.Tensor,
    *,
    state_addresses: torch.Tensor,
    state_row_strides: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    draft_token_num: int,
    geometry: tuple[int, int, int, int],
    state_dtype: torch.dtype,
    max_ctas: int,
    num_buffers: int,
    num_warps: int,
    max_registers: int,
) -> Any:
    """Launch the experimental full-head NVIDIA replay kernel.

    Args:
        payload: Layer-major raw K/V/a/b records, with the ordinary replay layout.
        parameters: FP32 ``[layers, 2, value_heads]`` A_log/dt_bias table.
        state_addresses: Uint64 ``[layers]`` recurrent-state pool addresses.
        state_row_strides: Int64 ``[layers]`` pool row strides in elements.
        read_indices: Int32 ``[layers, batch]`` input rows; negative means zero state.
        write_indices: Int32 ``[layers, batch]`` output rows; negative skips a row.
        accepted_length: Int32 ``[batch]`` accepted counts, clamped to the window.
        draft_token_num: Runtime replay-window length, from 1 through 16.
        geometry: ``(key_heads, value_heads, key_dim, value_dim)``; dimensions are 128.
        state_dtype: BF16, FP16, or FP32 pool element dtype.
        max_ctas: Positive grid bound, hence an upper bound on occupied SMs.
        num_buffers: One slot with one producer, or two slots with two producers.
        num_warps: Compute warps; producers and alignment add CTA threads.
        max_registers: Explicit per-thread register limit for compilation.

    Returns:
        The compiled kernel, allowing the benchmark to report its resource use;
        or ``None`` for an empty batch. State is written through the pool addresses.
    """
    validate_replay_commit_setup(
        payload, parameters, geometry=geometry, state_dtype=state_dtype
    )
    if not validate_replay_commit_args(
        payload,
        parameters,
        state_addresses=state_addresses,
        state_row_strides=state_row_strides,
        read_indices=read_indices,
        write_indices=write_indices,
        accepted_length=accepted_length,
        draft_token_num=draft_token_num,
        geometry=geometry,
        state_dtype=state_dtype,
    ):
        return None
    h, hv, k, v = geometry
    if (k, v) != (128, 128):
        raise ValueError("the persistent replay experiment requires K=V=128")
    if not 1 <= draft_token_num <= 16:
        raise ValueError("the persistent replay experiment requires 1 <= T <= 16")
    if max_ctas <= 0 or num_buffers not in (1, 2):
        raise ValueError("max_ctas must be positive and num_buffers must be 1 or 2")
    if num_warps not in (4, 8, 16):
        raise ValueError(
            "num_warps must be 4, 8, or 16; producer warps need extra threads"
        )
    if max_registers <= 0:
        raise ValueError("max_registers must be positive")
    enable_pdl = pdl_enabled()
    codes = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}
    return _gdn_replay_commit_persistent_kernel[
        (min(max_ctas, payload.shape[0] * accepted_length.numel() * hv),)
    ](
        payload,
        parameters,
        state_addresses,
        state_row_strides,
        read_indices,
        write_indices,
        accepted_length,
        accepted_length.numel(),
        draft_token_num,
        payload.shape[0],
        payload.stride(0),
        H=h,
        HV=hv,
        K=k,
        V=v,
        TOKEN_BLOCK=triton.next_power_of_2(draft_token_num),
        GATE_BLOCK=triton.next_power_of_2(2 * hv),
        STATE_DTYPE_CODE=codes[state_dtype],
        NUM_BUFFERS=num_buffers,
        SPT_K=4 if state_dtype == torch.float32 else 8,
        ENABLE_PDL=enable_pdl,
        num_warps=num_warps,
        maxnreg=max_registers,
        **({"launch_pdl": True} if enable_pdl else {}),
    )


@register_kernel(
    "attention",
    "gdn_replay_commit",
    name="gluon_nvidia_gdn_replay_commit",
    solution="gluon_nvidia",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia"})),
    signatures=format_signatures(
        ("q", "k", "v"), "dense", {torch.float16, torch.bfloat16}
    ),
    # Remain opt-in while end-to-end overlap is being validated. The ordinary
    # portable implementation has higher selection priority.
    priority=Priority.REFERENCE,
    traits={"flat_state": frozenset({True})},
)
def gluon_nvidia_gdn_replay_commit(
    payload: torch.Tensor,
    parameters: torch.Tensor,
    *,
    state_addresses: torch.Tensor,
    state_row_strides: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    draft_token_num: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    state_dtype: torch.dtype,
) -> None:
    """Commit accepted GDN states with the saved 24-CTA experiment settings.

    Arguments and in-place outputs follow ``gdn_replay_commit``. Key and value
    dimensions must both be 128, and the verify window must be at most 16.
    This opt-in solution uses two producer warps, eight compute warps, two
    shared-memory slots, and a 168-register compilation limit.
    """
    launch_gdn_replay_commit_persistent(
        payload,
        parameters,
        state_addresses=state_addresses,
        state_row_strides=state_row_strides,
        read_indices=read_indices,
        write_indices=write_indices,
        accepted_length=accepted_length,
        draft_token_num=draft_token_num,
        geometry=(num_k_heads, num_v_heads, head_k_dim, head_v_dim),
        state_dtype=state_dtype,
        max_ctas=24,
        num_buffers=2,
        num_warps=8,
        max_registers=168,
    )
