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

"""Experimental 16-CTA replay with asynchronous state and streamed operands.

One to three state slots with twice as many input slots allow prefetch while
consumers replay the accepted prefix. This module has no registry adapter.
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
from tokenspeed_kernel.platform import pdl_enabled

mb = gl.nvidia.ampere.mbarrier


@gluon.jit
def _arrive_all(barrier):
    # Keep the side effect lane-wise: every producer or consumer thread
    # contributes one arrival to its corresponding barrier count.
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    thread = gl.arange(0, 32 * gl.num_warps(), layout=layout)
    gl.inline_asm(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [$0];",
        ("r", "r"),
        (barrier, thread),
        result_types=(),
        is_pure=False,
    )


@gluon.jit
def _product(a, b):
    return a * b


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
    value_buffers,
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
    state_slot,
    state_bar,
    PRODUCER_ROLE: gl.constexpr,
):
    key_head = head // (HV // H)
    read = gl.load(read_indices + layer * B + request).to(gl.int64)
    write = gl.load(write_indices + layer * B + request).to(gl.int64)
    steps = gl.load(accepted_length + request)
    steps = gl.where(write >= 0, gl.minimum(gl.maximum(steps, 0), T), 0)
    storage_dtype: gl.constexpr = gl.uint32 if STATE_DTYPE_CODE == 2 else gl.uint16

    if PRODUCER_ROLE == 1:
        copy_layout: gl.constexpr = gl.BlockedLayout(
            [1, 4], [4, 8], [gl.num_warps(), 1], [1, 0]
        )
        tokens = gl.arange(0, TOKEN_BLOCK, layout=gl.SliceLayout(1, copy_layout))
        value_offsets = gl.arange(0, V, layout=gl.SliceLayout(0, copy_layout))
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
        if TOKEN_BLOCK <= 4:
            cp.async_load(
                value_buffers.index(buffer_index).slice(0, V, dim=1),
                rows[:, None] + H * K + head * V + value_offsets[None, :],
                (tokens < steps)[:, None],
            )
        else:
            cp.async_load(
                value_buffers.index(buffer_index),
                rows[:, None] + H * K + head * V + value_offsets[None, :],
                (tokens < steps)[:, None],
            )
        cp.commit_group()
    else:
        state_layout: gl.constexpr = gl.BlockedLayout(
            [SPT_K, 1], [8, 4], [1, gl.num_warps()], [0, 1]
        )
        keys = gl.arange(0, K, layout=gl.SliceLayout(1, state_layout))
        values = gl.arange(0, V, layout=gl.SliceLayout(0, state_layout))
        base = pool + gl.maximum(read, 0) * stride + head * V * K
        state_mask = gl.full([K, V], (read >= 0) & (write >= 0), gl.int1, state_layout)
        # The full barrier tracks both the DMA bytes and producer preparation.
        # Odd row strides can violate bulk-copy alignment; keep synchronous loads.
        if (read >= 0) & (write >= 0) & ((base.to(gl.uint64) & 15) == 0):
            threads_layout: gl.constexpr = gl.BlockedLayout(
                [1], [32], [gl.num_warps()], [0]
            )
            thread = gl.arange(0, 32 * gl.num_warps(), layout=threads_layout)
            nbytes: gl.constexpr = K * V * (4 if STATE_DTYPE_CODE == 2 else 2)
            gl.inline_asm(
                """{
                .reg .pred leader;
                setp.eq.u32 leader, $4, 0;
                fence.proxy.async.shared::cta;
                @leader mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [$1], $3;
                @leader cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$0], [$2], $3, [$1];
                }""",
                ("r", "r", "l", "r", "r"),
                (state_buffers.index(state_slot), state_bar, base, nbytes, thread),
                result_types=(),
                is_pure=False,
            )
        else:
            state_buffers.index(state_slot).store(
                gl.load(
                    base + values[None, :] * K + keys[:, None], state_mask, other=0
                ).to(storage_dtype, bitcast=True)
            )

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
    PRODUCER_ROLE: gl.constexpr,
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
        value_buffers,
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
        if PRODUCER_ROLE == 1:
            gl.static_assert(PRODUCER_COUNT == NUM_BUFFERS)
        else:
            gl.static_assert(PRODUCER_COUNT == 1)
        phase = 0
        iteration = 0
        for work in range(first, total, grid):
            if PRODUCER_ROLE == 0:
                state_slot = iteration % NUM_BUFFERS
                phase = iteration // NUM_BUFFERS & 1
                slot = 0
            else:
                state_slot = PRODUCER_INDEX
                slot = PRODUCER_INDEX + NUM_BUFFERS * phase
            mb.wait(empty_bars.index(state_slot), phase ^ 1)
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
                value_buffers,
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
                state_slot,
                full_bars.index(state_slot),
                PRODUCER_ROLE,
            )
            if PRODUCER_ROLE == 1:
                destination = pool + gl.maximum(write, 0) * stride + head * V * K
                destination = gl.where(write >= 0, destination.to(gl.uint64), 0)
                metadata_buffers.index(slot).slice(0, 1).store(
                    gl.full([1], destination, gl.uint64, metadata_layout)
                )
                gate_tokens = gl.arange(0, TOKEN_BLOCK, layout=gate_layout)
                width: gl.constexpr = H * K + HV * V + 2 * HV
                gate_base = (
                    payload
                    + layer.to(gl.int64) * PAYLOAD_LAYER_STRIDE
                    + physical_request.to(gl.int64) * T * width
                    + H * K
                    + HV * V
                    + head
                )
                raw_a = gl.load(
                    gate_base + gate_tokens.to(gl.int64) * width,
                    gate_tokens < steps,
                    other=0,
                ).to(gl.float32)
                raw_b = gl.load(
                    gate_base + gate_tokens.to(gl.int64) * width + HV,
                    gate_tokens < steps,
                    other=0,
                ).to(gl.float32)
                key_tokens = gl.arange(
                    0, TOKEN_BLOCK, layout=gl.SliceLayout(1, key_layout)
                )
                key_columns = gl.arange(0, K, layout=gl.SliceLayout(0, key_layout))
                key_layer_offset = gl.multiple_of(
                    layer.to(gl.int64) * PAYLOAD_LAYER_STRIDE, width & -width
                )
                key_rows = (
                    payload
                    + key_layer_offset
                    + (physical_request.to(gl.int64) * T + key_tokens.to(gl.int64))
                    * width
                    + (head // (HV // H)) * K
                )
                key = gl.load(
                    key_rows[:, None] + key_columns[None, :],
                    (key_tokens < steps)[:, None],
                    other=0,
                ).to(gl.float32)
                # Normalize in FP32 with the existing epsilon and reduction map.
                key = key * gl.rsqrt(gl.sum(key * key, 1) + 1e-6)[:, None]
                normalized_buffers.index(slot).store(key)
                x = raw_a + dt_bias
                softplus = gl.where(x <= 20.0, gl.log(1.0 + gl.exp(x)), x)
                decay = gl.exp(-gl.exp(raw_a_log) * softplus)
                beta = 1.0 / (1.0 + gl.exp(-raw_b))
                factor_buffers.index(slot).slice(0, TOKEN_BLOCK).store(decay)
                factor_buffers.index(slot).slice(TOKEN_BLOCK, TOKEN_BLOCK).store(beta)
                prefix = gl.associative_scan(decay, 0, _product)
                inverse = 1.0 / prefix
                factor_buffers.index(slot).slice(2 * TOKEN_BLOCK, TOKEN_BLOCK).store(
                    inverse
                )
                last = gl.full([1], gl.maximum(steps - 1, 0), gl.int32, gate_layout)
                final = gl.where(steps > 0, gl.sum(gl.gather(prefix, last, 0), 0), 1.0)
                packed_word = (
                    final.to(gl.uint32, bitcast=True).to(gl.uint64) << 32
                ) | steps.to(gl.uint32).to(gl.uint64)
                packed = gl.full([1], packed_word, gl.uint64, metadata_layout)
                metadata_buffers.index(slot).slice(1, 1).store(packed)
                if TOKEN_BLOCK <= 4:
                    if (steps >= 3) & (final > 1.0e-12):
                        cp.wait_group(0)
                        gl.barrier()
                        raw_values = (
                            value_buffers.index(slot)
                            .slice(0, V, dim=1)
                            .load(key_layout)
                            .to(gl.float32)
                        )
                        coefficient = gl.convert_layout(
                            beta * inverse, gl.SliceLayout(1, key_layout)
                        )
                        prepared_values = raw_values * coefficient[:, None]
                        gl.barrier()
                        value_buffers.index(slot).reinterpret(
                            gl.float32,
                            [TOKEN_BLOCK, V],
                            gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
                        ).store(prepared_values)
                cp.wait_group(0)
            _arrive_all(full_bars.index(state_slot))
            next_head = head + head_step
            head_carry = next_head >= HV
            next_head = gl.where(head_carry, next_head - HV, next_head)
            next_request = request + request_step + head_carry.to(gl.int32)
            request_carry = next_request >= B
            next_request = gl.where(request_carry, next_request - B, next_request)
            next_layer = layer + layer_step + request_carry.to(gl.int32)
            if work + grid < total:
                if (next_layer != layer) | ((PRODUCER_ROLE == 1) & (next_head != head)):
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
            if PRODUCER_ROLE == 1:
                phase = phase ^ 1
            iteration += 1


@gluon.jit
def _metadata_scalar(metadata, column: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    return gl.sum(metadata.slice(column, 1).load(layout), 0)


@gluon.jit
def _metadata_scalar32(metadata, column: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    return gl.sum(metadata.slice(column, 1).load(layout), 0)


@gluon.jit
def _load_replay_value(
    value_buffers,
    input_slot,
    step: gl.constexpr,
    V: gl.constexpr,
    TOKEN_BLOCK: gl.constexpr,
    LAYOUT: gl.constexpr,
    prepared,
):
    if TOKEN_BLOCK <= 4:
        if prepared:
            fp32_values = value_buffers.index(input_slot).reinterpret(
                gl.float32, [TOKEN_BLOCK, V], gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
            )
            result = gl.reshape(fp32_values.slice(step, 1).load(LAYOUT), [V])
        else:
            raw_values = value_buffers.index(input_slot).slice(0, V, dim=1)
            result = gl.reshape(raw_values.slice(step, 1).load(LAYOUT), [V]).to(
                gl.float32
            )
    else:
        result = gl.reshape(
            value_buffers.index(input_slot).slice(step, 1).load(LAYOUT), [V]
        ).to(gl.float32)
    return result


@gluon.jit
def _scaled_token_replay(
    state,
    normalized,
    values,
    factors,
    steps,
    final_scale,
    K: gl.constexpr,
    V: gl.constexpr,
    TB: gl.constexpr,
    SPT: gl.constexpr,
    PREPARED: gl.constexpr,
):
    state_layout: gl.constexpr = gl.BlockedLayout(
        [SPT, 1], [8, 4], [1, gl.num_warps()], [0, 1]
    )
    key_layout: gl.constexpr = gl.BlockedLayout(
        [1, SPT], [4, 8], [gl.num_warps(), 1], [1, 0]
    )
    value_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [8, 4], [1, gl.num_warps()], [0, 1]
    )
    if steps == TB:
        for step in gl.static_range(TB):
            k = gl.convert_layout(
                gl.reshape(normalized.slice(step, 1).load(key_layout), [K]),
                gl.SliceLayout(1, state_layout),
            )
            v = gl.convert_layout(
                gl.reshape(values.slice(step, 1).load(value_layout), [V]),
                gl.SliceLayout(0, state_layout),
            ).to(gl.float32)
            beta = _metadata_scalar32(factors, TB + step)
            projection = gl.sum(state * k[:, None], 0)
            if PREPARED:
                correction = gl.fma(-projection, beta, v)
            else:
                inverse = _metadata_scalar32(factors, 2 * TB + step)
                correction = (v * inverse - projection) * beta
            state = state + k[:, None] * correction[None, :]
    else:
        for step in gl.static_range(TB):
            if step < steps:
                k = gl.convert_layout(
                    gl.reshape(normalized.slice(step, 1).load(key_layout), [K]),
                    gl.SliceLayout(1, state_layout),
                )
                v = gl.convert_layout(
                    gl.reshape(values.slice(step, 1).load(value_layout), [V]),
                    gl.SliceLayout(0, state_layout),
                ).to(gl.float32)
                beta = _metadata_scalar32(factors, TB + step)
                projection = gl.sum(state * k[:, None], 0)
                if PREPARED:
                    correction = gl.fma(-projection, beta, v)
                else:
                    inverse = _metadata_scalar32(factors, 2 * TB + step)
                    correction = (v * inverse - projection) * beta
                state = state + k[:, None] * correction[None, :]
    return state * final_scale


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
        value_buffers,
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
        [4, SPT_K], [4, 8], [gl.num_warps(), 1], [1, 0]
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
    stream_value_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [8, 4], [1, gl.num_warps()], [0, 1]
    )
    iteration = 0
    for work in range(gl.program_id(0), L * B * HV, gl.num_programs(0)):
        buffer_index = iteration % NUM_BUFFERS
        input_slot = iteration % (NUM_BUFFERS * 2)
        phase = iteration // NUM_BUFFERS & 1
        mb.wait(full_bars.index(buffer_index), phase)
        metadata = metadata_buffers.index(input_slot)
        address = _metadata_scalar(metadata, 0)
        if STATE_DTYPE_CODE == 0:
            pool = address.to(gl.pointer_type(gl.bfloat16))
        elif STATE_DTYPE_CODE == 1:
            pool = address.to(gl.pointer_type(gl.float16))
        else:
            pool = address.to(gl.pointer_type(gl.float32))
        packed_word = _metadata_scalar(metadata, 1)
        steps = packed_word.to(gl.uint32).to(gl.int32)
        final_scale = (packed_word >> 32).to(gl.uint32).to(gl.float32, bitcast=True)
        state = (
            state_buffers.index(buffer_index)
            .load(state_layout)
            .to(state_dtype, bitcast=True)
            .to(gl.float32)
        )
        # Each compute thread releases only after caching its state values.
        # Inputs use twice as many slots: overwriting input i requires state
        # i + NUM_BUFFERS to be consumed, which follows all reads for item i.
        _arrive_all(empty_bars.index(buffer_index))
        if final_scale > 1.0e-12:
            if TOKEN_BLOCK <= 4 and steps >= 3:
                state = _scaled_token_replay(
                    state,
                    normalized_buffers.index(input_slot),
                    value_buffers.index(input_slot).reinterpret(
                        gl.float32,
                        [TOKEN_BLOCK, V],
                        gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
                    ),
                    factor_buffers.index(input_slot),
                    steps,
                    final_scale,
                    K,
                    V,
                    TOKEN_BLOCK,
                    SPT_K,
                    True,
                )
            else:
                if TOKEN_BLOCK <= 4:
                    raw_view = value_buffers.index(input_slot).slice(0, V, dim=1)
                else:
                    raw_view = value_buffers.index(input_slot)
                state = _scaled_token_replay(
                    state,
                    normalized_buffers.index(input_slot),
                    raw_view,
                    factor_buffers.index(input_slot),
                    steps,
                    final_scale,
                    K,
                    V,
                    TOKEN_BLOCK,
                    SPT_K,
                    False,
                )
        else:
            if TOKEN_BLOCK <= 4:
                fallback_view = value_buffers.index(input_slot).slice(0, V, dim=1)
            else:
                fallback_view = value_buffers.index(input_slot)
            for step in gl.static_range(TOKEN_BLOCK):
                if step < steps:
                    k = gl.convert_layout(
                        gl.reshape(
                            normalized_buffers.index(input_slot)
                            .slice(step, 1)
                            .load(key_read_layout),
                            [K],
                        ),
                        gl.SliceLayout(1, state_layout),
                    )
                    v = gl.convert_layout(
                        gl.reshape(
                            fallback_view.slice(step, 1).load(stream_value_layout), [V]
                        ),
                        gl.SliceLayout(0, state_layout),
                    ).to(gl.float32)
                    step_decay = _metadata_scalar32(
                        factor_buffers.index(input_slot), step
                    )
                    step_beta = _metadata_scalar32(
                        factor_buffers.index(input_slot), TOKEN_BLOCK + step
                    )
                    state = state * step_decay
                    correction = v - gl.sum(state * k[:, None], 0)
                    state = state + k[:, None] * (correction * step_beta)[None, :]
        if address != 0:
            destination = pool
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
def _gdn_replay_commit_legacy_kernel(
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
    gl.static_assert(NUM_BUFFERS >= 1, "at least one state buffer is required")
    gl.static_assert(NUM_BUFFERS <= 3, "at most three state buffers are supported")
    gl.static_assert(
        NUM_BUFFERS != 3 or STATE_DTYPE_CODE != 2 or TOKEN_BLOCK <= 4,
        "three FP32 state buffers require a window bucket at most four",
    )
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    storage_dtype: gl.constexpr = gl.uint32 if STATE_DTYPE_CODE == 2 else gl.uint16
    state_buffers = gl.allocate_shared_memory(
        storage_dtype, [NUM_BUFFERS, K, V], gl.SwizzledSharedLayout(1, 1, 1, [0, 1])
    )
    value_buffers = gl.allocate_shared_memory(
        payload.dtype.element_ty,
        [NUM_BUFFERS * 2, TOKEN_BLOCK, 2 * V if TOKEN_BLOCK <= 4 else V],
        gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
    )
    metadata_buffers = gl.allocate_shared_memory(
        gl.uint64, [NUM_BUFFERS * 2, 2], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    # Four FP32 padding elements per eight keys remove the stride-induced
    # bank conflict of the consumer's 128-bit normalized-key reads.
    normalized_buffers = gl.allocate_shared_memory(
        gl.float32,
        [NUM_BUFFERS * 2, TOKEN_BLOCK, K],
        gl.PaddedSharedLayout.with_identity_for([[8, 4]], [TOKEN_BLOCK, K], [1, 0]),
    )
    factor_buffers = gl.allocate_shared_memory(
        gl.float32,
        [NUM_BUFFERS * 2, 4 * TOKEN_BLOCK],
        gl.SwizzledSharedLayout(1, 1, 1, [0]),
    )
    full_bars = gl.allocate_shared_memory(
        gl.int64, [NUM_BUFFERS, 1], mb.MBarrierLayout()
    )
    empty_bars = gl.allocate_shared_memory(
        gl.int64, [NUM_BUFFERS, 1], mb.MBarrierLayout()
    )
    for slot in gl.static_range(NUM_BUFFERS):
        mb.init(full_bars.index(slot), count=64)
        mb.init(empty_bars.index(slot), count=32 * gl.num_warps())
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
        value_buffers,
        metadata_buffers,
        normalized_buffers,
        factor_buffers,
    )
    barriers = (full_bars, empty_bars)
    sizes = (B, T, L, PAYLOAD_LAYER_STRIDE)
    if NUM_BUFFERS == 3:
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
                        0,
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
                        3,
                        1,
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
                        3,
                        1,
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
                        2,
                        3,
                        1,
                    ),
                ),
            ],
            [1, 1, 1, 1],
            [64, 64, 64, 64],
        )
    elif NUM_BUFFERS == 2:
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
                        0,
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
                        1,
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
                        1,
                    ),
                ),
            ],
            [1, 1, 1],
            [64, 64, 64],
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
                        0,
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
                        1,
                    ),
                ),
            ],
            [1, 1],
            [64, 64],
        )
    for slot in gl.static_range(NUM_BUFFERS):
        mb.invalidate(full_bars.index(slot))
        mb.invalidate(empty_bars.index(slot))


@gluon.jit
def _group_prefetch_states(
    pool,
    stride,
    read_indices,
    write_indices,
    layer,
    request,
    key_head,
    state_slot,
    states,
    full_bar,
    B,
    HV: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    SPT: gl.constexpr,
):
    read = gl.load(read_indices + layer * B + request).to(gl.int64)
    write = gl.load(write_indices + layer * B + request)
    base = pool + gl.maximum(read, 0) * stride + key_head * 3 * V * K
    if (read >= 0) & (write >= 0) & ((base.to(gl.uint64) & 15) == 0):
        threads: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
        thread = gl.arange(0, 32, layout=threads)
        # The descriptor covers all six state cells. This96KiBcopy stays
        # within it at offsets0/96KiB and covers three consecutive value heads.
        gl.inline_asm(
            """{
            .reg .pred leader;
            .reg .b32 destination;
            add.u32 destination,$0,$4;
            setp.eq.u32 leader,$5,0;
            fence.proxy.async.shared::cta;
            @leader mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [$1],$3;
            @leader cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [destination],[$2],$3,[$1];
            }""",
            ("r", "r", "l", "r", "r", "r"),
            (states, full_bar, base, 3 * V * K * 2, state_slot * 3 * V * K * 2, thread),
            result_types=(),
            is_pure=False,
        )
    else:
        layout: gl.constexpr = gl.BlockedLayout([SPT, 1], [8, 4], [1, 1], [0, 1])
        keys = gl.arange(0, K, layout=gl.SliceLayout(1, layout))
        values = gl.arange(0, V, layout=gl.SliceLayout(0, layout))
        mask = gl.full([K, V], (read >= 0) & (write >= 0), gl.int1, layout)
        for local_head in gl.static_range(3):
            raw = gl.load(
                base + local_head * V * K + values[None, :] * K + keys[:, None],
                mask,
                other=0,
            )
            states.index(state_slot * 3 + local_head).store(
                raw.to(gl.uint16, bitcast=True)
            )


@gluon.jit
def _group_producer(
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
    NUM_BUFFERS: gl.constexpr,
    SPT_K: gl.constexpr,
    ROLE: gl.constexpr,
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
        value_buffers,
        metadata_buffers,
        normalized_buffers,
        factor_buffers,
        gram_buffers,
    ) = buffers
    full_bars, empty_bars = barriers
    B, T, L, PAYLOAD_LAYER_STRIDE = sizes
    first = gl.program_id(0)
    grid = gl.num_programs(0)
    total = L * B * H
    if first < total:
        layer = first // (B * H)
        request = first // H % B
        key_head = first % H
        local_head: gl.constexpr = 0 if ROLE == 0 else ROLE - 1
        head = key_head * 3 + local_head
        layer_step = grid // (B * H)
        request_step = grid // H % B
        head_step = grid % H
        request_rotation = layer % B
        rotation_step = layer_step % B
        pool, stride, raw_a_log, dt_bias = _load_replay_descriptor(
            parameters, state_addresses, state_row_strides, layer, head, HV, 0
        )
        metadata_layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
        key_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [1, 1], [1, 0])
        gate_layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
        iteration = 0
        for work in range(first, total, grid):
            state_slot = iteration % NUM_BUFFERS
            phase = iteration // NUM_BUFFERS & 1
            mb.wait(empty_bars.index(state_slot), phase ^ 1)
            physical_request = request + request_rotation
            physical_request = gl.where(
                physical_request >= B, physical_request - B, physical_request
            )
            if ROLE == 0:
                _group_prefetch_states(
                    pool,
                    stride,
                    read_indices,
                    write_indices,
                    layer,
                    physical_request,
                    key_head,
                    state_slot,
                    state_buffers,
                    full_bars.index(state_slot),
                    B,
                    HV,
                    K,
                    V,
                    SPT_K,
                )
            else:
                slot = state_slot * 3 + local_head
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
                    value_buffers,
                    B,
                    T,
                    PAYLOAD_LAYER_STRIDE,
                    H,
                    HV,
                    K,
                    V,
                    TOKEN_BLOCK,
                    GATE_BLOCK,
                    0,
                    SPT_K,
                    state_slot,
                    full_bars.index(state_slot),
                    1,
                )
                destination = pool + gl.maximum(write, 0) * stride + head * V * K
                destination = gl.where(write >= 0, destination.to(gl.uint64), 0)
                metadata_buffers.index(slot).slice(0, 1).store(
                    gl.full([1], destination, gl.uint64, metadata_layout)
                )
                gate_tokens = gl.arange(0, TOKEN_BLOCK, layout=gate_layout)
                width: gl.constexpr = H * K + HV * V + 2 * HV
                gate_base = (
                    payload
                    + layer.to(gl.int64) * PAYLOAD_LAYER_STRIDE
                    + physical_request.to(gl.int64) * T * width
                    + H * K
                    + HV * V
                    + head
                )
                raw_a = gl.load(
                    gate_base + gate_tokens.to(gl.int64) * width,
                    gate_tokens < steps,
                    other=0,
                ).to(gl.float32)
                raw_b = gl.load(
                    gate_base + gate_tokens.to(gl.int64) * width + HV,
                    gate_tokens < steps,
                    other=0,
                ).to(gl.float32)
                if ROLE == 1:
                    key_tokens = gl.arange(
                        0, TOKEN_BLOCK, layout=gl.SliceLayout(1, key_layout)
                    )
                    key_columns = gl.arange(0, K, layout=gl.SliceLayout(0, key_layout))
                    key_layer_offset = gl.multiple_of(
                        layer.to(gl.int64) * PAYLOAD_LAYER_STRIDE, width & -width
                    )
                    key_rows = (
                        payload
                        + key_layer_offset
                        + (physical_request.to(gl.int64) * T + key_tokens.to(gl.int64))
                        * width
                        + (head // (HV // H)) * K
                    )
                    key = gl.load(
                        key_rows[:, None] + key_columns[None, :],
                        (key_tokens < steps)[:, None],
                        other=0,
                    ).to(gl.float32)
                    # Normalize in FP32 with the existing epsilon and reduction map.
                    key = key * gl.rsqrt(gl.sum(key * key, 1) + 1e-6)[:, None]
                    normalized_buffers.index(state_slot).store(key)
                    if steps >= 3:
                        # Reuse FP32 normalized K for all three value heads.
                        # Rejected rows do not participate in the Gram solve.
                        gram_layout: gl.constexpr = gl.BlockedLayout(
                            [1, 1], [1, 32], [1, 1], [1, 0]
                        )
                        gram_key = gl.convert_layout(key, gram_layout)
                        gram_tokens = gl.arange(
                            0, TOKEN_BLOCK, layout=gl.SliceLayout(1, gram_layout)
                        )
                        for gram_row in gl.static_range(1, TOKEN_BLOCK):
                            if gram_row < steps:
                                row_key = gl.sum(
                                    gl.where(
                                        (gram_tokens == gram_row)[:, None],
                                        gram_key,
                                        0.0,
                                    ),
                                    0,
                                )
                                for gram_col in gl.static_range(gram_row):
                                    col_key = gl.sum(
                                        gl.where(
                                            (gram_tokens == gram_col)[:, None],
                                            gram_key,
                                            0.0,
                                        ),
                                        0,
                                    )
                                    product = gl.sum(row_key * col_key, 0)
                                    gram_buffers.index(state_slot).slice(
                                        gram_row * TOKEN_BLOCK + gram_col, 1
                                    ).store(
                                        gl.full(
                                            [1], product, gl.float32, metadata_layout
                                        )
                                    )

                x = raw_a + dt_bias
                softplus = gl.where(x <= 20.0, gl.log(1.0 + gl.exp(x)), x)
                decay = gl.exp(-gl.exp(raw_a_log) * softplus)
                beta = 1.0 / (1.0 + gl.exp(-raw_b))
                factor_buffers.index(slot).slice(0, TOKEN_BLOCK).store(decay)
                factor_buffers.index(slot).slice(TOKEN_BLOCK, TOKEN_BLOCK).store(beta)
                prefix = gl.associative_scan(decay, 0, _product)
                inverse = 1.0 / prefix
                factor_buffers.index(slot).slice(2 * TOKEN_BLOCK, TOKEN_BLOCK).store(
                    inverse
                )
                last = gl.full([1], gl.maximum(steps - 1, 0), gl.int32, gate_layout)
                final = gl.where(steps > 0, gl.sum(gl.gather(prefix, last, 0), 0), 1.0)
                packed_word = (
                    final.to(gl.uint32, bitcast=True).to(gl.uint64) << 32
                ) | steps.to(gl.uint32).to(gl.uint64)
                packed = gl.full([1], packed_word, gl.uint64, metadata_layout)
                metadata_buffers.index(slot).slice(1, 1).store(packed)
                if TOKEN_BLOCK <= 4:
                    if (steps >= 3) & (final > 1.0e-12):
                        cp.wait_group(0)
                        gl.barrier()
                        raw_values = (
                            value_buffers.index(slot)
                            .slice(0, V, dim=1)
                            .load(key_layout)
                            .to(gl.float32)
                        )
                        coefficient = gl.convert_layout(
                            beta * inverse, gl.SliceLayout(1, key_layout)
                        )
                        prepared_values = raw_values * coefficient[:, None]
                        gl.barrier()
                        value_buffers.index(slot).reinterpret(
                            gl.float32,
                            [TOKEN_BLOCK, V],
                            gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
                        ).store(prepared_values)
                cp.wait_group(0)
            _arrive_all(full_bars.index(state_slot))
            next_key_head = key_head + head_step
            head_carry = next_key_head >= H
            next_key_head = gl.where(head_carry, next_key_head - H, next_key_head)
            next_request = request + request_step + head_carry.to(gl.int32)
            request_carry = next_request >= B
            next_request = gl.where(request_carry, next_request - B, next_request)
            next_layer = layer + layer_step + request_carry.to(gl.int32)
            next_head = next_key_head * 3 + local_head
            if work + grid < total:
                if (next_layer != layer) | ((ROLE != 0) & (next_key_head != key_head)):
                    pool, stride, raw_a_log, dt_bias = _load_replay_descriptor(
                        parameters,
                        state_addresses,
                        state_row_strides,
                        next_layer,
                        next_head,
                        HV,
                        0,
                    )
            request_rotation = (
                request_rotation + rotation_step + request_carry.to(gl.int32)
            )
            request_rotation = gl.where(
                request_rotation >= B, request_rotation - B, request_rotation
            )
            layer, request, key_head, head = (
                next_layer,
                next_request,
                next_key_head,
                next_head,
            )
            iteration += 1


@gluon.jit
def _simt_key(normalized, step: gl.constexpr, K: gl.constexpr, SPT: gl.constexpr):
    state_layout: gl.constexpr = gl.BlockedLayout(
        [SPT, 1], [8, 4], [1, gl.num_warps()], [0, 1]
    )
    key_layout: gl.constexpr = gl.BlockedLayout(
        [1, SPT], [4, 8], [gl.num_warps(), 1], [1, 0]
    )
    return gl.convert_layout(
        gl.reshape(normalized.slice(step, 1).load(key_layout), [K]),
        gl.SliceLayout(1, state_layout),
    )


@gluon.jit
def _simt_value(values, step: gl.constexpr, V: gl.constexpr, SPT: gl.constexpr):
    state_layout: gl.constexpr = gl.BlockedLayout(
        [SPT, 1], [8, 4], [1, gl.num_warps()], [0, 1]
    )
    value_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [8, 4], [1, gl.num_warps()], [0, 1]
    )
    return gl.convert_layout(
        gl.reshape(values.slice(step, 1).load(value_layout), [V]),
        gl.SliceLayout(0, state_layout),
    ).to(gl.float32)


@gluon.constexpr_function
def _paired_dot_asm(outputs, inputs):
    (out,) = outputs
    state, key = inputs
    assert len(state) == len(out) * len(key)
    assert len(key) % 2 == 0
    code = [
        "{",
        f".reg .b64 keys<{len(key)//2}>;",
        f".reg .b64 accum<{len(out)}>;",
        ".reg .b64 pair;",
        ".reg .f32 low,high;",
    ]
    for offset in range(len(key) // 2):
        code.append(f"mov.b64 keys{offset},{{{key[2*offset]},{key[2*offset+1]}}};")
    for value in range(len(out)):
        code.append(f"mov.b64 accum{value},0;")
    # Four independent V accumulators hide the paired-FMA dependency.
    for offset in range(len(key) // 2):
        for value in range(len(out)):
            first = value * len(key) + 2 * offset
            code.append(f"mov.b64 pair,{{{state[first]},{state[first+1]}}};")
            code.append(f"fma.rn.f32x2 accum{value},pair,keys{offset},accum{value};")
    for value, destination in enumerate(out):
        code.append(f"mov.b64 {{low,high}},accum{value};")
        code.append(f"add.f32 {destination},low,high;")
    return "\n".join(code + ["}"])


@gluon.constexpr_function
def _paired_values_dot_asm(outputs, inputs):
    (out,) = outputs
    state, key = inputs
    assert len(state) == len(out) * len(key)
    assert len(out) % 2 == 0
    code = ["{", f".reg .b64 accum<{len(out)//2}>;", ".reg .b64 pair,keypair;"]
    for value in range(len(out) // 2):
        code.append(f"mov.b64 accum{value},0;")
    for offset, kreg in enumerate(key):
        code.append(f"mov.b64 keypair,{{{kreg},{kreg}}};")
        for value in range(len(out) // 2):
            lo = 2 * value * len(key) + offset
            hi = (2 * value + 1) * len(key) + offset
            code.append(f"mov.b64 pair,{{{state[lo]},{state[hi]}}};")
            code.append(f"fma.rn.f32x2 accum{value},pair,keypair,accum{value};")
    for value in range(len(out) // 2):
        code.append(f"mov.b64 {{{out[2*value]},{out[2*value+1]}}},accum{value};")
    return "\n".join(code + ["}"])


@gluon.jit
def _paired_projection(state, key, K: gl.constexpr, V: gl.constexpr):
    # K has eight participating lanes; all remaining terms are thread-local.
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [8, 4], [1, gl.num_warps()], [0, 1])
    result: gl.constexpr = gl.distributed_type(gl.float32, [8, V], layout)
    partial = gl.inline_asm(
        _paired_dot_asm, ("=&f", "f", "f"), (state, key), result, is_pure=True
    )
    return gl.convert_layout(gl.sum(partial, 0), gl.SliceLayout(0, state.type.layout))


@gluon.jit
def _simt_initial_projection(
    state, normalized, step: gl.constexpr, K: gl.constexpr, SPT: gl.constexpr
):
    key = _simt_key(normalized, step, K, SPT)
    if SPT >= 8:
        return _paired_projection(state, key, K, state.shape[1])
    else:
        return gl.sum(state * key[:, None], 0)


@gluon.jit
def _simt_rank_replay(
    state,
    normalized,
    values,
    factors,
    gram,
    steps,
    final_scale,
    K: gl.constexpr,
    V: gl.constexpr,
    TB: gl.constexpr,
    SPT: gl.constexpr,
):
    # Each projection sees the unchanged input state. Only the small V-vector
    # solve is recurrent; complete state matrices are not updated between dots.
    p0 = _simt_initial_projection(state, normalized, 0, K, SPT)
    p1 = _simt_initial_projection(state, normalized, 1, K, SPT)
    p2 = _simt_initial_projection(state, normalized, 2, K, SPT)
    b0 = _metadata_scalar32(factors, TB)
    b1 = _metadata_scalar32(factors, TB + 1)
    b2 = _metadata_scalar32(factors, TB + 2)
    q0 = gl.fma(-p0, b0, _simt_value(values, 0, V, SPT))
    q1 = gl.fma(-p1, b1, _simt_value(values, 1, V, SPT))
    q1 = gl.fma(-b1 * _metadata_scalar32(gram, TB), q0, q1)
    q2 = gl.fma(-p2, b2, _simt_value(values, 2, V, SPT))
    q2 = gl.fma(-b2 * _metadata_scalar32(gram, 2 * TB), q0, q2)
    q2 = gl.fma(-b2 * _metadata_scalar32(gram, 2 * TB + 1), q1, q2)
    state_layout: gl.constexpr = gl.BlockedLayout(
        [SPT, 1], [8, 4], [1, gl.num_warps()], [0, 1]
    )
    q3 = gl.full([V], 0.0, gl.float32, gl.SliceLayout(0, state_layout))
    if steps == 4:
        p3 = _simt_initial_projection(state, normalized, 3, K, SPT)
        b3 = _metadata_scalar32(factors, TB + 3)
        q3 = gl.fma(-p3, b3, _simt_value(values, 3, V, SPT))
        q3 = gl.fma(-b3 * _metadata_scalar32(gram, 3 * TB), q0, q3)
        q3 = gl.fma(-b3 * _metadata_scalar32(gram, 3 * TB + 1), q1, q3)
        q3 = gl.fma(-b3 * _metadata_scalar32(gram, 3 * TB + 2), q2, q3)
    state = state + _simt_key(normalized, 0, K, SPT)[:, None] * q0[None, :]
    state = state + _simt_key(normalized, 1, K, SPT)[:, None] * q1[None, :]
    state = state + _simt_key(normalized, 2, K, SPT)[:, None] * q2[None, :]
    if steps == 4:
        state = state + _simt_key(normalized, 3, K, SPT)[:, None] * q3[None, :]
    return state * final_scale


@gluon.jit
def _wait_group_reader(barrier, phase):
    # Each reader acquires the same completed phase. Empty still counts every
    # reader, so no producer can recycle a slot until all warps finish it.
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0])
    thread = gl.arange(0, 32 * gl.num_warps(), layout=layout)
    gl.inline_asm(
        """{
        .reg .pred complete;
        reader_wait:
        mbarrier.try_wait.parity.shared::cta.b64 complete,[$0],$1;
        @!complete bra.uni reader_wait;
        }""",
        "r,r,r,~{memory}",
        (barrier, phase, thread),
        result_types=(),
        is_pure=False,
    )


@gluon.jit
def _group_consumer(
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
        value_buffers,
        metadata_buffers,
        normalized_buffers,
        factor_buffers,
        gram_buffers,
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
        [4, SPT_K], [4, 8], [gl.num_warps(), 1], [1, 0]
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
    stream_value_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1], [8, 4], [1, gl.num_warps()], [0, 1]
    )
    iteration = 0
    for work in range(gl.program_id(0), L * B * H, gl.num_programs(0)):
        buffer_index = iteration % NUM_BUFFERS
        phase = iteration // NUM_BUFFERS & 1
        _wait_group_reader(full_bars.index(buffer_index), phase)
        for local_head in tl.range(0, 3, loop_unroll_factor=1):
            input_slot = buffer_index * 3 + local_head
            metadata = metadata_buffers.index(input_slot)
            address = _metadata_scalar(metadata, 0)
            if STATE_DTYPE_CODE == 0:
                pool = address.to(gl.pointer_type(gl.bfloat16))
            elif STATE_DTYPE_CODE == 1:
                pool = address.to(gl.pointer_type(gl.float16))
            else:
                pool = address.to(gl.pointer_type(gl.float32))
            packed_word = _metadata_scalar(metadata, 1)
            steps = packed_word.to(gl.uint32).to(gl.int32)
            final_scale = (packed_word >> 32).to(gl.uint32).to(gl.float32, bitcast=True)
            state = (
                state_buffers.index(input_slot)
                .load(state_layout)
                .to(state_dtype, bitcast=True)
                .to(gl.float32)
            )
            if final_scale > 1.0e-12:
                if TOKEN_BLOCK <= 4 and steps >= 3:
                    state = _simt_rank_replay(
                        state,
                        normalized_buffers.index(buffer_index),
                        value_buffers.index(input_slot).reinterpret(
                            gl.float32,
                            [TOKEN_BLOCK, V],
                            gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
                        ),
                        factor_buffers.index(input_slot),
                        gram_buffers.index(buffer_index),
                        steps,
                        final_scale,
                        K,
                        V,
                        TOKEN_BLOCK,
                        SPT_K,
                    )
                else:
                    if TOKEN_BLOCK <= 4:
                        raw_view = value_buffers.index(input_slot).slice(0, V, dim=1)
                    else:
                        raw_view = value_buffers.index(input_slot)
                    state = _scaled_token_replay(
                        state,
                        normalized_buffers.index(buffer_index),
                        raw_view,
                        factor_buffers.index(input_slot),
                        steps,
                        final_scale,
                        K,
                        V,
                        TOKEN_BLOCK,
                        SPT_K,
                        False,
                    )
            else:
                if TOKEN_BLOCK <= 4:
                    fallback_view = value_buffers.index(input_slot).slice(0, V, dim=1)
                else:
                    fallback_view = value_buffers.index(input_slot)
                for step in gl.static_range(TOKEN_BLOCK):
                    if step < steps:
                        k = gl.convert_layout(
                            gl.reshape(
                                normalized_buffers.index(buffer_index)
                                .slice(step, 1)
                                .load(key_read_layout),
                                [K],
                            ),
                            gl.SliceLayout(1, state_layout),
                        )
                        v = gl.convert_layout(
                            gl.reshape(
                                fallback_view.slice(step, 1).load(stream_value_layout),
                                [V],
                            ),
                            gl.SliceLayout(0, state_layout),
                        ).to(gl.float32)
                        step_decay = _metadata_scalar32(
                            factor_buffers.index(input_slot), step
                        )
                        step_beta = _metadata_scalar32(
                            factor_buffers.index(input_slot), TOKEN_BLOCK + step
                        )
                        state = state * step_decay
                        correction = v - gl.sum(state * k[:, None], 0)
                        state = state + k[:, None] * (correction * step_beta)[None, :]
            if address != 0:
                destination = pool
                if (destination.to(gl.uint64) & 15) == 0:
                    aligned = gl.multiple_of(
                        destination.to(gl.pointer_type(storage_dtype)), 16
                    )
                    bits = state.to(state_dtype).to(storage_dtype, bitcast=True)
                    gl.store(
                        aligned + offsets_v[None, :] * K + offsets_k[:, None], bits
                    )
                else:
                    gl.store(
                        destination + offsets_v[None, :] * K + offsets_k[:, None], state
                    )

        _arrive_all(empty_bars.index(buffer_index))
        iteration += 1


@gluon.jit(do_not_specialize=["B", "T", "L", "PAYLOAD_LAYER_STRIDE"])
def _group_main(
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
    gl.static_assert(NUM_BUFFERS == 2)
    gl.static_assert(STATE_DTYPE_CODE == 0)
    gl.static_assert(HV == 3 * H)
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    state_buffers = gl.allocate_shared_memory(
        gl.uint16, [NUM_BUFFERS * 3, K, V], gl.SwizzledSharedLayout(1, 1, 1, [0, 1])
    )
    value_buffers = gl.allocate_shared_memory(
        payload.dtype.element_ty,
        [NUM_BUFFERS * 3, TOKEN_BLOCK, 2 * V],
        gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
    )
    metadata_buffers = gl.allocate_shared_memory(
        gl.uint64, [NUM_BUFFERS * 3, 2], gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    normalized_buffers = gl.allocate_shared_memory(
        gl.float32,
        [NUM_BUFFERS, TOKEN_BLOCK, K],
        gl.PaddedSharedLayout.with_identity_for([[8, 4]], [TOKEN_BLOCK, K], [1, 0]),
    )
    factor_buffers = gl.allocate_shared_memory(
        gl.float32,
        [NUM_BUFFERS * 3, 4 * TOKEN_BLOCK],
        gl.SwizzledSharedLayout(1, 1, 1, [0]),
    )
    gram_buffers = gl.allocate_shared_memory(
        gl.float32,
        [NUM_BUFFERS, TOKEN_BLOCK * TOKEN_BLOCK],
        gl.SwizzledSharedLayout(1, 1, 1, [0]),
    )
    full_bars = gl.allocate_shared_memory(
        gl.int64, [NUM_BUFFERS, 1], mb.MBarrierLayout()
    )
    empty_bars = gl.allocate_shared_memory(
        gl.int64, [NUM_BUFFERS, 1], mb.MBarrierLayout()
    )
    for slot in gl.static_range(NUM_BUFFERS):
        mb.init(full_bars.index(slot), count=128)
        mb.init(empty_bars.index(slot), count=32 * gl.num_warps())
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
        value_buffers,
        metadata_buffers,
        normalized_buffers,
        factor_buffers,
        gram_buffers,
    )
    barriers = (full_bars, empty_bars)
    sizes = (B, T, L, PAYLOAD_LAYER_STRIDE)
    gl.warp_specialize(
        [
            (
                _group_consumer,
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
                _group_producer,
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
                    NUM_BUFFERS,
                    SPT_K,
                    0,
                ),
            ),
            (
                _group_producer,
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
                    NUM_BUFFERS,
                    SPT_K,
                    1,
                ),
            ),
            (
                _group_producer,
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
                    NUM_BUFFERS,
                    SPT_K,
                    2,
                ),
            ),
            (
                _group_producer,
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
                    NUM_BUFFERS,
                    SPT_K,
                    3,
                ),
            ),
        ],
        [1, 1, 1, 1],
        [64, 96, 64, 64],
    )
    for slot in gl.static_range(NUM_BUFFERS):
        mb.invalidate(full_bars.index(slot))
        mb.invalidate(empty_bars.index(slot))


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
    if STATE_DTYPE_CODE == 0 and TOKEN_BLOCK == 4 and NUM_BUFFERS == 2 and HV == 3 * H:
        _group_main(
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
            H,
            HV,
            K,
            V,
            TOKEN_BLOCK,
            GATE_BLOCK,
            STATE_DTYPE_CODE,
            NUM_BUFFERS,
            SPT_K,
            ENABLE_PDL,
        )
    else:
        _gdn_replay_commit_legacy_kernel(
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
            H,
            HV,
            K,
            V,
            TOKEN_BLOCK,
            GATE_BLOCK,
            STATE_DTYPE_CODE,
            NUM_BUFFERS,
            SPT_K,
            ENABLE_PDL,
        )


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
        num_buffers: One to three state slots, each with one producer warp.
            Three slots support FP32 state only for windows up to four tokens.
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
    if not 0 < max_ctas <= 16 or num_buffers not in (1, 2, 3):
        raise ValueError(
            "max_ctas must be between 1 and 16; num_buffers must be 1, 2, or 3"
        )
    if num_buffers == 3 and state_dtype == torch.float32 and draft_token_num > 4:
        raise ValueError(
            "three buffers support FP32 state windows up to 4; use one or two buffers for longer windows"
        )
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
