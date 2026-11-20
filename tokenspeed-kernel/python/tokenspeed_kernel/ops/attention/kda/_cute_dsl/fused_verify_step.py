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

"""One-token KDA verify with its convolution and gate producers, in CuTe DSL.

Contract: the split producers (the gate projection ``f_a @ f_b.T`` and
``kda_verify_conv_update``) followed by ``kda_recoverssm_verify`` for windows
of one token, in one launch. Outputs and records match theirs, the
pre-convolution rows are retained for commit, and the checkpoint and the
convolution state are read only.

For one request and head, let S0 be the [V, K] state after the accepted prefix:

    q, k, v = silu(conv(history, row))      rounded to BF16, as the producer stores them
    g       = f_a @ f_b[head].T             rounded to BF16
    e^G     = decay(g)                      one value per key
    rk = rsqrt(|k|^2 + 1e-6), rq = rsqrt(|q|^2 + 1e-6) / sqrt(K)
    c   = beta (v - rk S0 (k e^G))          the correction record
    out = rq S0 (q e^G) + rq rk (q . k) c

With the layer's output norm the kernel stores, instead of ``out``, its gated
RMSNorm ``rmsnorm(bf16(out)) * w * sigmoid(gate)`` over the head (the
o-projection input), as the separate norm kernel computes it from the stored
BF16 output; only the order of the sum of squares differs.

For an accepted prefix, write each retained update as U times a transposed
key. Both checkpoint products then use the base with the cumulative decay.
Each accepted U contributes its key's dot product with the new k or q,
weighted by subsequent decays. Up to 16 accepted tokens use this projection.
For longer prefixes the kernel replays into a private shared-memory tile.

One CTA of four warps covers one (request, head). It bulk-copies the
checkpoint into shared memory while thread t convolves channel t of q, k and
v. For the gate projection and the checkpoint product, warp w takes 32 keys
or 32 checkpoint rows and lane l elements [4l, 4l + 4) of each row, so every
warp-wide load reads one contiguous row, and a transposing butterfly leaves
each row's total in one lane. Weights, the checkpoint and the convolution
history are read before the programmatic dependency wait (none is written by
the previous launch); the projection's rows after it.
"""

from __future__ import annotations

import functools

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Boolean, Float16, Float32, Int32, Int64, const_expr, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor
from tokenspeed_kernel.ops.attention.kda import KdaOutputNorm, KdaRetainedHistory
from tokenspeed_kernel.ops.attention.kda._cute_dsl.recoverssm_verify import (
    GATE_FLOOR,
    _asm,
    _bulk_load,
    _div,
    _exp,
    _lds_f32x4,
    _load,
    _rsqrt,
    _store_if,
)

__all__ = ["fused_verify_step", "fused_verify_step_supported"]

THREADS = 128
WARPS = THREADS // 32
CONV_TAPS = 4
_MAX_PROJECTED_HISTORY = 16
_DTYPES = {
    torch.bfloat16: BFloat16,
    torch.float16: Float16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


from tokenspeed_kernel.ops.attention.kda._cute_dsl.commit_state import (
    consolidate_history,
)
from tokenspeed_kernel.ops.attention.kda._cute_dsl.retained import (
    fake_retained_arguments,
    load_element,
    retained_arguments,
    retained_length,
    retained_page,
)
from tokenspeed_kernel.thirdparty.vllm_recoverssm.kda import reset_retained_history


@dsl_user_op
def _ldg_bf16x4(byte_addr: Int64, *, loc=None, ip=None):
    """Four BF16 values from an 8-byte aligned global address, as FP32."""
    value = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int64(byte_addr)],
        "{ .reg .b32 a, b; ld.global.v2.b32 {a, b}, [$4]; "
        "shl.b32 $0, a, 16; and.b32 $1, a, 0xffff0000; "
        "shl.b32 $2, b, 16; and.b32 $3, b, 0xffff0000; }",
        "=f,=f,=f,=f,l",
        True,
        loc,
        ip,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), value, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def _lds_bf16x4(byte_addr: Int32, *, loc=None, ip=None):
    """Four BF16 values from an 8-byte aligned shared address, as FP32."""
    value = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int32(byte_addr)],
        "{ .reg .b32 a, b; ld.shared.v2.b32 {a, b}, [$4]; "
        "shl.b32 $0, a, 16; and.b32 $1, a, 0xffff0000; "
        "shl.b32 $2, b, 16; and.b32 $3, b, 0xffff0000; }",
        "=f,=f,=f,=f,r",
        True,
        loc,
        ip,
    )
    return [
        Float32(llvm.extractvalue(T.f32(), value, [i], loc=loc, ip=ip))
        for i in range(4)
    ]


@dsl_user_op
def _prefetch_l2(byte_addr: Int64, size: int, *, loc=None, ip=None):
    """Prefetch ``size`` bytes (a multiple of 16) at a 16-byte aligned address into L2."""
    _asm(
        None,
        [Int64(byte_addr), Int32(size)],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _store_bf16_if(live: Int32, byte_addr: Int64, value: Float32, *, loc=None, ip=None):
    """Store an FP32 value that is already a BF16 number as BF16, when live."""
    _store_if(live, byte_addr, value, True, loc=loc, ip=ip)


@cute.jit
def _bf16(x: Float32) -> Float32:
    return x.to(BFloat16).to(Float32)


@cute.jit
def _silu(x: Float32) -> Float32:
    return x * _div(Float32(1.0), Float32(1.0) + _exp(-x))


@cute.jit
def _transpose_sum(values, lane: Int32):
    """Sum each of 32 per-lane values over the warp; lane l returns value l's total.

    Each of the five steps hands half of the values still held to the lane
    ``offset`` away and keeps the other half, adding what it receives.
    """
    for stage in cutlass.range_constexpr(5):
        offset = 16 >> stage
        half = 16 >> stage
        upper = (lane & offset) != 0
        for i in cutlass.range_constexpr(half):
            send = values[i] if upper else values[i + half]
            keep = values[i + half] if upper else values[i]
            values[i] = keep + cute.arch.shuffle_sync_bfly(send, offset=offset)
    return values[0]


class FusedVerifyStep:
    """One-token verify with fused producers for one head shape."""

    def __init__(
        self,
        heads: int,
        capacity: int,
        bounded_gate: bool,
        has_metadata: bool,
        gated_norm: bool,
        fp32_checkpoint: bool,
        packed_history: bool,
    ) -> None:
        self.key_dim = self.value_dim = self.gate_rank = THREADS
        self.heads = heads
        self.capacity = capacity
        self.bounded_gate = bounded_gate
        self.has_metadata = has_metadata
        self.gated_norm = gated_norm
        self.fp32_checkpoint = fp32_checkpoint
        self.packed_history = packed_history
        self.width = heads * THREADS
        # The checkpoint, then raw keys, raw queries, decays and values, then
        # the four cross-warp sum partials; 128 bytes cover the mbarrier.
        self.smem_bytes = 4 * (THREADS * THREADS + 4 * THREADS + 4 * WARPS) + 128
        # U, K and decay prefixes, cumulative decay, and the two dot products
        # for each record. Allow for alignment between shared allocations.
        self.smem_bytes += (
            4
            * (
                3 * _MAX_PROJECTED_HISTORY * THREADS
                + THREADS
                + 2 * _MAX_PROJECTED_HISTORY
            )
            + 384
        )

    @cute.jit
    def __call__(
        self,
        qkv: cute.Tensor,
        conv_weights: cute.Tensor,
        conv_states: cute.Tensor,
        f_a: cute.Tensor,
        f_b: cute.Tensor,
        beta: cute.Tensor,
        a_log: cute.Tensor,
        dt_bias: cute.Tensor,
        state: cute.Tensor,
        retained: tuple,
        ring_capacity: Int64,
        correction: cute.Tensor,
        normalized_key: cute.Tensor,
        original: cute.Tensor,
        out: cute.Tensor,
        query_start_loc: cute.Tensor,
        state_indices: cute.Tensor,
        metadata: cute.Tensor | None,
        stamps: cute.Tensor | None,
        gate: cute.Tensor | None,
        norm_weight: cute.Tensor | None,
        lower_bound: Float32,
        norm_eps: Float32,
        stream: CUstream,
    ):
        metadata_ptr = None
        stamps_ptr = None
        if const_expr(self.has_metadata):
            metadata_ptr = metadata.iterator
            stamps_ptr = stamps.iterator
        gate_ptr = None
        norm_weight_ptr = None
        gate_token_stride = Int64(0)
        if const_expr(self.gated_norm):
            gate_ptr = gate.iterator
            norm_weight_ptr = norm_weight.iterator
            gate_token_stride = gate.stride[0]
        self.kernel(
            qkv.iterator,
            conv_weights.iterator,
            conv_states.iterator,
            f_a.iterator,
            f_b.iterator,
            beta.iterator,
            a_log.iterator,
            dt_bias.iterator,
            state.iterator,
            retained,
            ring_capacity,
            correction.iterator,
            normalized_key.iterator,
            original.iterator,
            out.iterator,
            query_start_loc.iterator,
            state_indices.iterator,
            metadata_ptr,
            stamps_ptr,
            gate_ptr,
            norm_weight_ptr,
            lower_bound,
            norm_eps,
            qkv.stride[0],
            conv_states.stride[0],
            f_a.stride[0],
            beta.stride[0],
            state.stride[0],
            state.stride[1],
            correction.stride[0],
            correction.stride[1],
            correction.stride[2],
            normalized_key.stride[0],
            normalized_key.stride[1],
            normalized_key.stride[2],
            original.stride[0],
            out.stride[0],
            gate_token_stride,
        ).launch(
            grid=[state_indices.shape[0], self.heads, 1],
            block=[THREADS, 1, 1],
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        qkv,
        conv_weights,
        conv_states,
        f_a,
        f_b,
        beta,
        a_log,
        dt_bias,
        state,
        retained: tuple,
        ring_capacity: Int64,
        correction,
        normalized_key,
        original,
        out,
        query_start_loc,
        state_indices,
        metadata,
        stamps,
        norm_gate,
        norm_weight,
        lower_bound: Float32,
        norm_eps: Float32,
        qkv_token_stride: Int64,
        conv_page_stride: Int64,
        f_a_token_stride: Int64,
        beta_token_stride: Int64,
        state_page_stride: Int64,
        state_head_stride: Int64,
        correction_slot_stride: Int64,
        correction_head_stride: Int64,
        correction_position_stride: Int64,
        normalized_key_slot_stride: Int64,
        normalized_key_head_stride: Int64,
        normalized_key_position_stride: Int64,
        original_token_stride: Int64,
        out_token_stride: Int64,
        gate_token_stride: Int64,
    ):
        K = self.key_dim
        V = self.value_dim
        P = self.width
        tidx, _, _ = cute.arch.thread_idx()
        request, head, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32

        smem = cutlass.memory.SmemAllocator()
        tile = smem.allocate_tensor(
            Float32, cute.make_layout((V * K,)), byte_alignment=128
        )
        raw_keys = smem.allocate_tensor(
            Float32, cute.make_layout((K,)), byte_alignment=16
        )
        raw_queries = smem.allocate_tensor(
            Float32, cute.make_layout((K,)), byte_alignment=16
        )
        decays = smem.allocate_tensor(
            Float32, cute.make_layout((K,)), byte_alignment=16
        )
        values = smem.allocate_tensor(
            Float32, cute.make_layout((V,)), byte_alignment=16
        )
        partials = smem.allocate_tensor(
            Float32, cute.make_layout((4 * WARPS,)), byte_alignment=16
        )
        barrier = smem.allocate_array(Int64, num_elems=1)
        history_layout = cute.make_layout((_MAX_PROJECTED_HISTORY * THREADS,))
        history_decay = smem.allocate_tensor(
            Float32, cute.make_layout((THREADS,)), byte_alignment=128
        )
        query_coefficients = smem.allocate_tensor(
            Float32, cute.make_layout((_MAX_PROJECTED_HISTORY,)), byte_alignment=16
        )
        weighted_history_keys = smem.allocate_tensor(
            Float32, history_layout, byte_alignment=128
        )
        history_u = smem.allocate_tensor(Float32, history_layout, byte_alignment=16)
        record_decay = smem.allocate_tensor(Float32, history_layout, byte_alignment=16)
        key_coefficients = smem.allocate_tensor(
            Float32, cute.make_layout((_MAX_PROJECTED_HISTORY,)), byte_alignment=16
        )
        tile_addr = Int32(tile.iterator.toint())
        raw_keys_addr = Int32(raw_keys.iterator.toint())
        raw_queries_addr = Int32(raw_queries.iterator.toint())
        decays_addr = Int32(decays.iterator.toint())

        # Weights first: they do not depend on the previous launch.
        q_channel = head * K + tidx
        k_channel = P + q_channel
        v_channel = 2 * P + head * V + tidx
        q_taps = _ldg_bf16x4((conv_weights + q_channel * CONV_TAPS).toint())
        k_taps = _ldg_bf16x4((conv_weights + k_channel * CONV_TAPS).toint())
        v_taps = _ldg_bf16x4((conv_weights + v_channel * CONV_TAPS).toint())
        # Warp w projects gate keys [32 w, 32 w + 32), lane l reading elements
        # [4l, 4l + 4) of each f_b row; after the butterfly lane l holds key
        # 32 w + l, which is thread tidx's own key.
        f_b_rows = f_b + (head * K + 32 * warp) * self.gate_rank + 4 * lane
        if tidx == 0:
            head_f_b = f_b + head * K * self.gate_rank
            _prefetch_l2(head_f_b.toint(), K * self.gate_rank * 2)
        gate_scale = _exp(_load(a_log, head).to(Float32))
        gate_bias = _load(dt_bias, head * K + tidx).to(Float32)
        norm_scale = Float32(0.0)
        if const_expr(self.gated_norm):
            norm_scale = _load(norm_weight, tidx).to(Float32)

        # The batch layout, the checkpoint and the convolution history were
        # written before this forward, not by the previous launch: the
        # checkpoint copy and the history loads overlap the projection too.
        token_start = Int64(_load(query_start_loc, request))
        valid = Int32(Int64(_load(query_start_loc, request + 1)) - token_start)
        page = Int64(_load(state_indices, request))
        live_page = retained_page(retained, request)
        live = (page > 0) & (live_page > 0) & (valid > 0)

        accepted = Int32(0)
        packed_checkpoint = Boolean(False)
        if live:
            if tidx == 0:
                cute.arch.mbarrier_init(barrier, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.barrier()
            accepted = retained_length(retained, live_page)
            projected_history = (accepted > 0) & (accepted <= _MAX_PROJECTED_HISTORY)
            if (accepted == 0) | projected_history:
                tile_src = state + (
                    live_page * state_page_stride + Int64(head) * state_head_stride
                )
                record_bytes = Int32(0)
                if const_expr(self.packed_history):
                    if projected_history:
                        record_bytes = accepted * K * 4
                # The checkpoint and three accepted prefixes share one completion barrier.
                history_bytes = 3 * record_bytes
                if const_expr(self.fp32_checkpoint):
                    if tidx == 0:
                        size = V * K * 4
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            barrier, size + history_bytes
                        )
                        _bulk_load(
                            tile_addr, tile_src.toint(), size, Int32(barrier.toint())
                        )
                else:
                    packed_checkpoint = (tile_src.toint() & Int64(15)) == Int64(0)
                    if packed_checkpoint:
                        if tidx == 0:
                            size = V * K * 2
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                barrier, size + history_bytes
                            )
                            _bulk_load(
                                tile_addr,
                                tile_src.toint(),
                                size,
                                Int32(barrier.toint()),
                            )
                    elif const_expr(self.packed_history):
                        if projected_history and tidx == 0:
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                barrier, history_bytes
                            )
                if projected_history:
                    if const_expr(self.packed_history):
                        if tidx == 0:
                            destinations = (
                                history_u,
                                weighted_history_keys,
                                record_decay,
                            )
                            for i in cutlass.range_constexpr(3):
                                field = retained[i + 1]
                                base = field.iterator + (
                                    live_page * field.stride[0]
                                    + Int64(head) * field.stride[1]
                                )
                                _bulk_load(
                                    Int32(destinations[i].iterator.toint()),
                                    base.toint(),
                                    record_bytes,
                                    Int32(barrier.toint()),
                                )
                    else:
                        for rec in cutlass.range(accepted, unroll=2):
                            history_u[rec * THREADS + tidx] = load_element(
                                retained[1], live_page, head, rec, tidx
                            )
                            weighted_history_keys[rec * THREADS + tidx] = load_element(
                                retained[2], live_page, head, rec, tidx
                            )
                            record_decay[rec * THREADS + tidx] = load_element(
                                retained[3], live_page, head, rec, tidx
                            )
                if const_expr(not self.fp32_checkpoint):
                    if not packed_checkpoint:
                        _load_empty_step_tile(
                            tile, retained[0], live_page, head, tidx, V, K
                        )
                        if (tidx == 0) & (record_bytes == 0):
                            cute.arch.mbarrier_arrive(barrier)
            else:
                _replay_step_tile(tile, retained, live_page, head, tidx, accepted, V, K)
                if tidx == 0:
                    cute.arch.mbarrier_arrive(barrier)
        # Padding pages see a zero history, as the split producer's do.
        history = conv_states + page * conv_page_stride
        channels = [q_channel, k_channel, v_channel]
        pasts = []
        for c in cutlass.range_constexpr(3):
            for i in cutlass.range_constexpr(CONV_TAPS - 1):
                past = Float32(0.0)
                if (valid > 0) & (page >= 0):
                    past = _load(history, channels[c] * 3 + i).to(Float32)
                pasts.append(past)

        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()

        if valid > 0:
            # Convolution over the committed window and this row.
            row = qkv + token_start * qkv_token_stride
            raw = [_load(row, channel) for channel in channels]
            taps = [q_taps, k_taps, v_taps]
            convolved = []
            for c in cutlass.range_constexpr(3):
                acc = Float32(0.0)
                for i in cutlass.range_constexpr(CONV_TAPS - 1):
                    acc = acc + pasts[c * (CONV_TAPS - 1) + i] * taps[c][i]
                acc = acc + raw[c].to(Float32) * taps[c][CONV_TAPS - 1]
                convolved.append(_bf16(_silu(acc)))
            query, key_value, value = convolved
            kept = original + token_start * original_token_stride
            for c in cutlass.range_constexpr(3):
                _store_bf16_if(
                    Int32(1), (kept + channels[c]).toint(), raw[c].to(Float32)
                )

            if live:
                raw_keys[tidx] = key_value
                raw_queries[tidx] = query
                values[tidx] = value
                sums = [key_value * key_value, query * query, query * key_value]
                for i in cutlass.range_constexpr(3):
                    sums[i] = cute.arch.warp_reduction_sum(sums[i])
                if lane == 0:
                    for i in cutlass.range_constexpr(3):
                        partials[WARPS * i + warp] = sums[i]
                slot = Int64(request)
                generation = Int64(0)
                start = Int64(0)
                if const_expr(self.has_metadata):
                    slot = Int64(_load(metadata, request * 3))
                    generation = Int64(_load(metadata, request * 3 + 1))
                    start = Int64(_load(metadata, request * 3 + 2))
                position = Int64(start % ring_capacity)
                if const_expr(self.has_metadata):
                    if (head == 0) & (tidx == 0):
                        stamp = cute.make_tensor(
                            stamps + (slot * ring_capacity + position) * 2,
                            cute.make_layout((2,)),
                        )
                        stamp[0] = start
                        stamp[1] = generation

                # Gate projection, one contiguous f_b row per warp-wide load.
                f_a_part = _ldg_bf16x4(
                    (f_a + token_start * f_a_token_stride + 4 * lane).toint()
                )
                gate_parts = []
                for j in cutlass.range_constexpr(32):
                    weights = _ldg_bf16x4((f_b_rows + j * self.gate_rank).toint())
                    part = Float32(0.0)
                    for e in cutlass.range_constexpr(4):
                        part = part + f_a_part[e] * weights[e]
                    gate_parts.append(part)
                gate = _bf16(_transpose_sum(gate_parts, lane))
                x = gate + gate_bias
                step = Float32(0.0)
                if const_expr(self.bounded_gate):
                    step = lower_bound * _div(
                        Float32(1.0), Float32(1.0) + _exp(-(gate_scale * x))
                    )
                else:
                    softplus = x
                    if x <= Float32(20.0):
                        softplus = cute.math.log(Float32(1.0) + _exp(x), fastmath=True)
                    step = -gate_scale * softplus
                if step < Float32(GATE_FLOOR):
                    step = Float32(GATE_FLOOR)
                decays[tidx] = _exp(step)
                beta_raw = _load(beta, token_start * beta_token_stride + head).to(
                    Float32
                )
                beta_gate = _div(Float32(1.0), Float32(1.0) + _exp(-beta_raw))
                cute.arch.mbarrier_wait(barrier, 0)
                if (accepted > 0) & (accepted <= _MAX_PROJECTED_HISTORY):
                    # Each lane weights its K channel in place, from newest to oldest.
                    if accepted <= 2:
                        decay1 = Float32(1.0)
                        k1 = Float32(0.0)
                        if accepted == 2:
                            decay1 = record_decay[THREADS + tidx]
                            k1 = weighted_history_keys[THREADS + tidx]
                        history_decay[tidx] = record_decay[tidx] * decay1
                        weighted_history_keys[tidx] = (
                            weighted_history_keys[tidx] * decay1
                        )
                        weighted_history_keys[THREADS + tidx] = k1
                    else:
                        suffix = Float32(1.0)
                        for i in cutlass.range(accepted, unroll=4):
                            suffix_record = Int32(accepted - 1 - i)
                            weighted_history_keys[suffix_record * THREADS + tidx] = (
                                weighted_history_keys[suffix_record * THREADS + tidx]
                                * suffix
                            )
                            suffix = (
                                suffix * record_decay[suffix_record * THREADS + tidx]
                            )
                        history_decay[tidx] = suffix
                cute.arch.barrier()

                totals = [
                    partials[WARPS * i]
                    + partials[WARPS * i + 1]
                    + partials[WARPS * i + 2]
                    + partials[WARPS * i + 3]
                    for i in range(3)
                ]
                key_scale = _rsqrt(totals[0] + Float32(1e-6))
                query_scale = _rsqrt(totals[1] + Float32(1e-6)) * Float32(K**-0.5)
                for field, value in (
                    (retained[2], key_value * key_scale),
                    (retained[3], decays[tidx]),
                ):
                    dst_offset = (
                        live_page * field.stride[0]
                        + Int64(head) * field.stride[1]
                        + Int64(accepted) * field.stride[2]
                        + Int64(tidx) * field.stride[3]
                    )
                    cute.make_tensor(
                        field.iterator + dst_offset, cute.make_layout((1,))
                    )[0] = value

                output_coeff = query_scale * key_scale * totals[2]
                lane_decays = _lds_f32x4(decays_addr + 16 * lane)
                lane_keys = _lds_f32x4(raw_keys_addr + 16 * lane)
                lane_queries = _lds_f32x4(raw_queries_addr + 16 * lane)
                decayed_keys = [lane_keys[i] * lane_decays[i] for i in range(4)]
                decayed_queries = [lane_queries[i] * lane_decays[i] for i in range(4)]

                projected_history = (accepted > 0) & (
                    accepted <= _MAX_PROJECTED_HISTORY
                )
                if projected_history:
                    history_addr = (
                        Int32(weighted_history_keys.iterator.toint()) + 16 * lane
                    )
                    if accepted <= WARPS:
                        record = Int32(warp)
                        if record < accepted:
                            _store_history_projection(
                                record,
                                history_addr,
                                lane,
                                decayed_keys,
                                decayed_queries,
                                key_coefficients,
                                query_coefficients,
                            )
                    else:
                        for group in cutlass.range(
                            (accepted + WARPS - 1) // WARPS, unroll=1
                        ):
                            record = Int32(group) * WARPS + Int32(warp)
                            if record < accepted:
                                _store_history_projection(
                                    record,
                                    history_addr,
                                    lane,
                                    decayed_keys,
                                    decayed_queries,
                                    key_coefficients,
                                    query_coefficients,
                                )
                    suffix = _lds_f32x4(
                        Int32(history_decay.iterator.toint()) + 16 * lane
                    )
                    for e in cutlass.range_constexpr(4):
                        decayed_keys[e] = decayed_keys[e] * suffix[e]
                        decayed_queries[e] = decayed_queries[e] * suffix[e]

                # S0 times the decayed keys and queries: warp w takes rows
                # [32 w, 32 w + 32), one contiguous row per warp-wide load.
                key_total = Float32(0.0)
                query_total = Float32(0.0)
                if const_expr(self.fp32_checkpoint):
                    key_total, query_total = _project_state_tile(
                        tile_addr, warp, lane, K, decayed_keys, decayed_queries, False
                    )
                else:
                    if packed_checkpoint:
                        key_total, query_total = _project_state_tile(
                            tile_addr,
                            warp,
                            lane,
                            K,
                            decayed_keys,
                            decayed_queries,
                            True,
                        )
                    else:
                        key_total, query_total = _project_state_tile(
                            tile_addr,
                            warp,
                            lane,
                            K,
                            decayed_keys,
                            decayed_queries,
                            False,
                        )
                # Lane l now holds value row 32 w + l, thread tidx's own channel.
                if projected_history:
                    # All warps can complete the dense products while each retained
                    # record's owner publishes its two projection coefficients.
                    cute.arch.barrier()
                    if accepted <= 2:
                        u0 = history_u[tidx]
                        key_total = key_total + u0 * key_coefficients[0]
                        query_total = query_total + u0 * query_coefficients[0]
                        if accepted == 2:
                            u1 = history_u[THREADS + tidx]
                            key_total = key_total + u1 * key_coefficients[1]
                            query_total = query_total + u1 * query_coefficients[1]
                    else:
                        for value_record in cutlass.range(accepted, unroll=2):
                            value_index = Int32(value_record)
                            value_update = history_u[value_index * THREADS + tidx]
                            key_total = (
                                key_total + value_update * key_coefficients[value_index]
                            )
                            query_total = (
                                query_total
                                + value_update * query_coefficients[value_index]
                            )
                corrected = (values[tidx] - key_scale * key_total) * beta_gate
                output = query_scale * query_total + output_coeff * corrected
                if const_expr(self.gated_norm):
                    # The norm kernel's math on the BF16 output: x rsqrt(mean
                    # x^2 + eps) w sigmoid(gate), multiplied left to right.
                    output = _bf16(output)
                    logit = _load(
                        norm_gate, token_start * gate_token_stride + head * V + tidx
                    ).to(Float32)
                    square = cute.arch.warp_reduction_sum(output * output)
                    if lane == 0:
                        partials[3 * WARPS + warp] = square
                    cute.arch.barrier()
                    total = (
                        partials[3 * WARPS]
                        + partials[3 * WARPS + 1]
                        + partials[3 * WARPS + 2]
                        + partials[3 * WARPS + 3]
                    )
                    inverse_rms = _rsqrt(_div(total, Float32(V)) + norm_eps)
                    sigmoid = _div(Float32(1.0), Float32(1.0) + _exp(-logit))
                    output = output * inverse_rms * norm_scale * sigmoid
                dst = out + (token_start * out_token_stride + head * V + tidx)
                _store_if(Int32(1), dst.toint(), output, True)
                entry = correction + (
                    live_page * correction_slot_stride
                    + head * correction_head_stride
                    + Int64(accepted) * correction_position_stride
                    + tidx
                )
                _store_if(Int32(1), entry.toint(), corrected, False)
            else:
                # Padding pages: zero outputs, no records.
                dst = out + (token_start * out_token_stride + head * V + tidx)
                _store_if(Int32(1), dst.toint(), Float32(0.0), True)


@cute.jit
def _project_state_tile(
    tile_addr, warp, lane, width, decayed_keys, decayed_queries, packed
):
    key_parts = []
    query_parts = []
    for i in cutlass.range_constexpr(32):
        offset = (32 * warp + i) * width + 4 * lane
        if const_expr(packed):
            s = _lds_bf16x4(tile_addr + offset * 2)
        else:
            s = _lds_f32x4(tile_addr + offset * 4)
        key_part = Float32(0.0)
        query_part = Float32(0.0)
        for e in cutlass.range_constexpr(4):
            key_part = key_part + s[e] * decayed_keys[e]
            query_part = query_part + s[e] * decayed_queries[e]
        key_parts.append(key_part)
        query_parts.append(query_part)
    key_total = _transpose_sum(key_parts, lane)
    query_total = _transpose_sum(query_parts, lane)
    return key_total, query_total


@cute.jit
def _store_history_projection(
    record, address, lane, keys, queries, key_coefficients, query_coefficients
):
    """One warp computes and publishes both dot products for a retained key."""
    weights = _lds_f32x4(address + record * (4 * THREADS))
    key_part = Float32(0.0)
    query_part = Float32(0.0)
    for e in cutlass.range_constexpr(4):
        key_part = key_part + weights[e] * keys[e]
        query_part = query_part + weights[e] * queries[e]
    key_sum = cute.arch.warp_reduction_sum(key_part)
    query_sum = cute.arch.warp_reduction_sum(query_part)
    if lane == 0:
        key_coefficients[record] = key_sum
        query_coefficients[record] = query_sum


@cute.jit
def _load_empty_step_tile(
    tile,
    state,
    page,
    head,
    tidx,
    rows: cutlass.Constexpr[int],
    keys: cutlass.Constexpr[int],
):
    for value in cutlass.range_constexpr(rows):
        tile[value * keys + tidx] = load_element(state, page, head, value, tidx)
    cute.arch.barrier()


@dsl_user_op
def _update_step_state(current, decay, u, key, *, loc=None, ip=None):
    # Match the original replay's rounding: round the decay multiply, then
    # accumulate the correction with one FP32 FMA.
    return Float32(
        _asm(
            T.f32(),
            [Float32(current), Float32(decay), Float32(u), Float32(key)],
            "{ .reg .f32 decayed; mul.rn.f32 decayed, $1, $2; "
            "fma.rn.f32 $0, $3, $4, decayed; }",
            "=f,f,f,f,f",
            False,
            loc,
            ip,
        )
    )


@cute.jit
def _replay_step_record(
    history, page, head, position, first, key, decay, s0, s1, s2, s3
):
    u0 = load_element(history[1], page, head, position, first)
    u1 = load_element(history[1], page, head, position, first + 1)
    u2 = load_element(history[1], page, head, position, first + 2)
    u3 = load_element(history[1], page, head, position, first + 3)
    return (
        _update_step_state(s0, decay, u0, key),
        _update_step_state(s1, decay, u1, key),
        _update_step_state(s2, decay, u2, key),
        _update_step_state(s3, decay, u3, key),
    )


@cute.jit
def _replay_step_tile(
    tile,
    history,
    page,
    head,
    tidx,
    count,
    rows: cutlass.Constexpr[int],
    keys: cutlass.Constexpr[int],
):
    # One thread keeps four value rows for the same key channel.
    for group in cutlass.range(rows // 4, unroll=1):
        first = group * 4
        s0 = load_element(history[0], page, head, first, tidx)
        s1 = load_element(history[0], page, head, first + 1, tidx)
        s2 = load_element(history[0], page, head, first + 2, tidx)
        s3 = load_element(history[0], page, head, first + 3, tidx)
        for position in cutlass.range(count, unroll=1):
            key = load_element(history[2], page, head, position, tidx)
            decay = load_element(history[3], page, head, position, tidx)
            s0, s1, s2, s3 = _replay_step_record(
                history, page, head, position, first, key, decay, s0, s1, s2, s3
            )
        tile[first * keys + tidx] = s0
        tile[(first + 1) * keys + tidx] = s1
        tile[(first + 2) * keys + tidx] = s2
        tile[(first + 3) * keys + tidx] = s3
    cute.arch.barrier()


@functools.cache
def _compile(
    heads: int,
    capacity: int,
    state_dtype: torch.dtype,
    live_dtype: torch.dtype,
    packed_history: bool,
    bounded_gate: bool,
    has_metadata: bool,
    norm_dtype: torch.dtype | None,
    beta_dtype: torch.dtype,
    param_dtype: torch.dtype,
    bias_dtype: torch.dtype,
    start_dtype: torch.dtype,
    index_dtype: torch.dtype,
):
    key_dim = value_dim = gate_rank = THREADS
    tokens = cute.sym_int()
    slots = cute.sym_int()
    batch = cute.sym_int()
    width = heads * key_dim
    qkv = make_fake_tensor(BFloat16, (tokens, 3 * width), divisibility=8)
    conv_weights = make_fake_tensor(BFloat16, (3 * width, 4), divisibility=4)
    conv_states = make_fake_tensor(
        BFloat16, (cute.sym_int(), cute.sym_int(), 3), divisibility=1
    )
    f_a = make_fake_tensor(BFloat16, (tokens, gate_rank), divisibility=8)
    f_b = make_fake_tensor(BFloat16, (width, gate_rank), divisibility=8)
    beta = make_fake_tensor(_DTYPES[beta_dtype], (tokens, heads))
    a_log = make_fake_tensor(_DTYPES[param_dtype], (heads,))
    dt_bias = make_fake_tensor(_DTYPES[bias_dtype], (heads * key_dim,))
    state = make_fake_tensor(
        _DTYPES[state_dtype],
        (cute.sym_int(), heads, value_dim, key_dim),
        divisibility=4,
    )
    correction = make_fake_tensor(Float32, (slots, heads, capacity, value_dim))
    normalized_key = make_fake_tensor(
        Float32, (slots, heads, capacity, key_dim), leading_dim=None
    )
    original = make_fake_tensor(BFloat16, (tokens, 3 * width))
    out = make_fake_tensor(BFloat16, (tokens, heads * value_dim))
    query_start_loc = make_fake_tensor(_DTYPES[start_dtype], (cute.sym_int(),))
    state_indices = make_fake_tensor(_DTYPES[index_dtype], (batch,))
    metadata = stamps = None
    if has_metadata:
        metadata = make_fake_tensor(Int64, (batch, 3))
        stamps = make_fake_tensor(Int64, (cute.sym_int(), cute.sym_int(), 2))
    gate = norm_weight = None
    if norm_dtype is not None:
        gate = make_fake_tensor(BFloat16, (tokens, heads * value_dim))
        norm_weight = make_fake_tensor(_DTYPES[norm_dtype], (value_dim,))
    retained = fake_retained_arguments(
        state,
        correction,
        normalized_key,
        slots=slots,
        batch=batch,
        live_dtype=_DTYPES[live_dtype],
    )
    kernel = FusedVerifyStep(
        heads,
        capacity,
        bounded_gate,
        has_metadata,
        norm_dtype is not None,
        state_dtype == torch.float32,
        packed_history,
    )
    return cute.compile(
        kernel,
        qkv,
        conv_weights,
        conv_states,
        f_a,
        f_b,
        beta,
        a_log,
        dt_bias,
        state,
        retained,
        Int64(0),
        correction,
        normalized_key,
        original,
        out,
        query_start_loc,
        state_indices,
        metadata,
        stamps,
        gate,
        norm_weight,
        Float32(0.0),
        Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def fused_verify_step_supported(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint_state: torch.Tensor,
    retained_history: KdaRetainedHistory,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    lower_bound: float | None,
) -> bool:
    """Whether this kernel can address these inputs.

    Args:
        mixed_qkv: ``[tokens, 3 * num_heads * head_dim]`` BF16 q|k|v rows
            before the convolution.
        conv_weights: ``[3 * num_heads * head_dim, 4]`` BF16 convolution taps.
        conv_states: ``[pages, channels, 3]`` BF16 convolution history.
        f_a_out: ``[tokens, rank]`` BF16 gate projection input.
        f_b_weight: ``[num_heads * head_dim, rank]`` BF16 gate projection.
        beta_logits: ``[tokens, num_heads]`` beta logits.
        A_log: ``[num_heads]`` gate decay logarithms.
        dt_bias: ``[num_heads * head_dim]`` gate bias.
        checkpoint_state: ``[pages, num_heads, head_dim, head_dim]`` FP32/BF16
            checkpoint in value-major layout.
        retained_history: Accepted records and per-request live pages.
        query_start_loc: Packed request token boundaries.
        state_indices: Per-request source pages.
        num_heads: Per-rank head count.
        head_dim: Head width.
        lower_bound: Bounded-gate lower bound, or None for the softplus gate.

    Returns:
        True when the shapes, dtypes, strides and base alignments fit the
        kernel: head width and gate rank 128, BF16 activations, 16-byte aligned
        projection rows, dense request vectors and unit-stride correction channels.
    """
    width = num_heads * head_dim
    rank = f_a_out.shape[-1]
    return (
        head_dim == THREADS
        and mixed_qkv.dtype == torch.bfloat16
        and conv_weights.dtype == torch.bfloat16
        and conv_states.dtype == torch.bfloat16
        and f_a_out.dtype == f_b_weight.dtype == torch.bfloat16
        and checkpoint_state.dtype in (torch.float32, torch.bfloat16)
        and mixed_qkv.shape[-1] == 3 * width
        and conv_weights.shape == (3 * width, CONV_TAPS)
        and conv_weights.is_contiguous()
        and conv_states.ndim == 3
        and conv_states.shape[1] >= 3 * width
        and conv_states.stride()[1:] == (3, 1)
        and f_b_weight.shape == (width, rank)
        and f_b_weight.is_contiguous()
        and rank == 128
        and (lower_bound is None or lower_bound <= 0)
        and mixed_qkv.stride(-1) == f_a_out.stride(-1) == 1
        and mixed_qkv.stride(0) % 8 == 0
        and f_a_out.stride(0) % 8 == 0
        and conv_weights.data_ptr() % 8 == 0
        and mixed_qkv.data_ptr() % 16 == 0
        and f_a_out.data_ptr() % 16 == 0
        and f_b_weight.data_ptr() % 16 == 0
        and beta_logits.stride(-1) == 1
        and A_log.stride(0) == 1
        and dt_bias.is_contiguous()
        and checkpoint_state.shape[1:] == (num_heads, head_dim, head_dim)
        and checkpoint_state.stride()[2:] == (head_dim, 1)
        and checkpoint_state.stride(0) % 4 == 0
        and checkpoint_state.stride(1) % 4 == 0
        and checkpoint_state.data_ptr() % 16 == 0
        and retained_history.correction.stride(-1) == 1
        and all(
            x.ndim == 1 and x.dtype in (torch.int32, torch.int64) and x.stride(0) == 1
            for x in (
                query_start_loc,
                state_indices,
                retained_history.live_state_indices,
            )
        )
    )


def _check_records(
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
    batch: int,
    correction_cache: torch.Tensor,
) -> None:
    """Raise ValueError unless the replay records are laid out as the kernel reads them.

    The metadata must be dense INT64 ``[batch, 3]`` and the stamps dense INT64
    ``[slots, capacity, 2]``, both on the cache's device.
    """
    if (record_metadata is None) != (record_stamps is None):
        raise ValueError("record metadata and stamps must be supplied together")
    if record_metadata is None:
        return
    if (
        record_metadata.shape != (batch, 3)
        or record_stamps.ndim != 3
        or record_stamps.shape[-1] != 2
    ):
        raise ValueError(
            "retained metadata needs [batch,3] and stamps [slots,capacity,2]"
        )
    if any(
        x.dtype != torch.int64
        or not x.is_contiguous()
        or x.device != correction_cache.device
        for x in (record_metadata, record_stamps)
    ):
        raise ValueError(
            "replay metadata and stamps must be contiguous INT64 on the input device"
        )


def _check_output_norm(
    output_norm: KdaOutputNorm | None, out: torch.Tensor, head_dim: int
) -> None:
    """Raise ValueError unless the output norm is laid out as the kernel reads it.

    The gate must be BF16 ``[tokens, heads * head_dim]`` like ``out`` with unit
    inner stride, and the weight a dense ``[head_dim]`` vector, both on the
    output's device.
    """
    if output_norm is None:
        return
    gate, weight = output_norm.gate, output_norm.weight
    if (
        gate.shape != out.shape
        or gate.dtype != torch.bfloat16
        or gate.stride(-1) != 1
        or weight.shape != (head_dim,)
        or not weight.is_contiguous()
        or weight.dtype not in _DTYPES
        or gate.device != out.device
        or weight.device != out.device
    ):
        raise ValueError(
            "the output norm needs a BF16 [tokens, heads * head_dim] gate with "
            "unit inner stride and a dense [head_dim] weight"
        )


def fused_verify_step(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    checkpoint_state: torch.Tensor,
    retained_history,
    original_qkv: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    out: torch.Tensor,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
    output_norm: KdaOutputNorm | None,
) -> torch.Tensor:
    """Run one-token record verify with the gate and convolution producers fused.

    Args:
        mixed_qkv: ``[tokens, 3 * num_heads * head_dim]`` BF16 q|k|v rows
            before the convolution, one token per request.
        conv_weights: ``[3 * num_heads * head_dim, 4]`` BF16 convolution taps.
        conv_states: ``[pages, channels, 3]`` BF16 committed convolution
            history; read only.
        f_a_out: ``[tokens, 128]`` BF16 gate projection input.
        f_b_weight: ``[num_heads * head_dim, 128]`` BF16 gate projection.
        beta_logits: ``[tokens, num_heads]`` beta logits.
        A_log: ``[num_heads]`` gate decay logarithms.
        dt_bias: ``[num_heads * head_dim]`` gate bias.
        lower_bound: Bounded-gate lower bound, or None for the softplus gate.
        checkpoint_state: ``[pages, num_heads, head_dim, head_dim]`` FP32/BF16
            committed checkpoint in value-major layout; read only.
        retained_history: Page-owned FP32 U/normalized-K/decay and accepted
            lengths. Candidates append after each live page's accepted prefix.
        original_qkv: ``[tokens, 3 * num_heads * head_dim]`` BF16; receives
            the pre-convolution rows for commit.
        query_start_loc: ``[batch + 1]`` token offset of each request.
        state_indices: ``[batch]`` checkpoint and convolution page of each
            request. Pages at or below 0 are padding: zero outputs and no
            records, though their pre-convolution rows are still retained.
        num_heads: Per-rank head count.
        head_dim: Head width; 128.
        out: ``[tokens, num_heads * head_dim]`` BF16 verify output, or with
            ``output_norm`` its gated RMSNorm; written.
        record_metadata: ``[batch, 3]`` int64 record slot, generation and
            start position per request, or None to record request r at slot
            r, position 0.
        record_stamps: ``[slots, capacity, 2]`` int64 start and generation of
            each record position, written with ``record_metadata``; else None.
        output_norm: The layer's gated output RMSNorm to apply before the
            store, or None to store the verify output.

    Returns:
        ``out``.

    Raises:
        ValueError: If ``fused_verify_step_supported`` rejects the inputs, the
            record metadata and stamps are not dense INT64 views of the record
            layout, or the output norm's gate or weight are laid out otherwise.
    """
    correction_cache = retained_history.correction
    if mixed_qkv.shape[0] == 0:
        return out
    if not fused_verify_step_supported(
        mixed_qkv,
        conv_weights,
        conv_states,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        checkpoint_state,
        retained_history,
        query_start_loc,
        state_indices,
        num_heads,
        head_dim,
        lower_bound,
    ):
        raise ValueError("CuTe DSL fused KDA verify step cannot address these inputs")
    _check_records(
        record_metadata, record_stamps, state_indices.shape[0], correction_cache
    )
    _check_output_norm(output_norm, out, head_dim)
    bounded = lower_bound is not None
    kernel = _compile(
        num_heads,
        correction_cache.shape[2],
        checkpoint_state.dtype,
        retained_history.live_state_indices.dtype,
        retained_history.packed_record_width == head_dim,
        bounded,
        record_metadata is not None,
        None if output_norm is None else output_norm.weight.dtype,
        beta_logits.dtype,
        A_log.dtype,
        dt_bias.dtype,
        query_start_loc.dtype,
        state_indices.dtype,
    )
    consolidate_history(
        checkpoint_state, retained_history, state_indices, query_start_loc
    )
    reset_retained_history(retained_history, state_indices, query_start_loc)
    kernel(
        mixed_qkv,
        conv_weights,
        conv_states,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias.reshape(-1),
        checkpoint_state,
        retained_arguments(checkpoint_state, retained_history),
        (
            record_stamps.shape[1]
            if record_stamps is not None
            else correction_cache.shape[2]
        ),
        correction_cache,
        retained_history.key,
        original_qkv,
        out,
        query_start_loc,
        state_indices,
        record_metadata,
        record_stamps,
        None if output_norm is None else output_norm.gate,
        None if output_norm is None else output_norm.weight,
        float(lower_bound) if bounded else 0.0,
        0.0 if output_norm is None else float(output_norm.eps),
    )
    return out
