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

"""Batched KDA RecoverSSM commit of FP32 states in CuTe DSL.

Contract: the recurrent-state commit of ``batched_recoverssm_commit`` on its
CUDA-core path (windows of up to 15 tokens) for a bounded gate, FP32 state
pages and BF16 records, bit for bit. Each (layer, head, request) whose
interval passes the record checks folds its ``n`` accepted tokens into the
source page and writes the destination page,

    S[v, k] <- fma(u_t[v], kn_t[k], S[v, k] * d_t[k])      for t < n,

where ``kn_t`` is the normalized key and ``d_t`` the decay, each formed with
the Triton commit's instructions and in its order.

Each (layer x head, request) gets 2 CTAs of 256 threads, one per half
of the value rows. Each thread first issues the loads of its 8 value rows x 4 keys of
the source state, which a positive source page allows whatever the record
checks find. Warp w then forms tokens w and w + 8: lane l holds keys l, l + 32,
l + 64 and l + 96, as the Triton commit's one-warp layout does, so the key's
sum of squares adds in the same order, and stores kn_t, d_t, and the token's
corrections to shared memory. After one barrier each thread folds its values
and, if the interval checks out, stores them. Every state byte is read once
and written once, where the Triton commit splits a head over 8 or 32
programs that each rebuild the token operands.
"""

from __future__ import annotations

import functools

import cutlass
import torch
from cutlass import Float32, Int32, Int64, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor
from tokenspeed_kernel.ops.attention.kda._cute_dsl.recoverssm_verify import (
    _asm,
    _div,
    _load,
    _rsqrt,
)

__all__ = ["MAX_WINDOW", "commit_states", "commit_states_supported"]

DIM = 128
THREADS = 256
WARPS = THREADS // 32
# The Triton commit takes CUDA cores below 16 tokens, so these windows match it.
MAX_WINDOW = 15
LOG2E = 1.4426950408889634
# At 76 registers, 2 CTAs per head fit 3 per SM without spills (GB300 W = 8:
# 28 us at 1 request, 179 at 8, against the Triton commit's 50 and 228).
VALUE_TILES = 2
MIN_CTAS = 3
NORM_EPS = 1e-6
# Descriptor columns: FP32 corrections, BF16 key / raw-gate records, A_log, dt_bias, state.
_CORRECTION, _KEY_GATE, _A_LOG, _DT_BIAS, _STATE = 3, 5, 6, 7, 8


@dsl_user_op
def _mul(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(a), Float32(b)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _add(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(a), Float32(b)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _fma(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(a), Float32(b), Float32(c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ex2(x: Float32, *, loc=None, ip=None) -> Float32:
    """Triton's FP32 exp2 (``tl.exp`` scales by log2(e) first)."""
    return Float32(
        _asm(T.f32(), [Float32(x)], "ex2.approx.f32 $0, $1;", "=f,f", False, loc, ip)
    )


@dsl_user_op
def _ldg_f32(byte_addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Int64(byte_addr)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ldg_bf16(byte_addr: Int64, *, loc=None, ip=None) -> Float32:
    """A BF16 value from global memory, widened to FP32."""
    return Float32(
        _asm(
            T.f32(),
            [Int64(byte_addr)],
            "{ .reg .b16 h; ld.global.b16 h, [$1]; cvt.f32.bf16 $0, h; }",
            "=f,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ldg_f32x4(byte_addr: Int64, *, loc=None, ip=None):
    value = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int64(byte_addr)],
        "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
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
def _stg_f32x4(byte_addr: Int64, values, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(byte_addr), *(Float32(v) for v in values)],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _lds_f32x4(byte_addr: Int32, *, loc=None, ip=None):
    value = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int32(byte_addr)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
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
def _sts_f32(byte_addr: Int32, value: Float32, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(byte_addr), Float32(value)],
        "st.shared.f32 [$0], $1;",
        "r,f,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _sts_f32x4(byte_addr: Int32, values, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(byte_addr), *(Float32(v) for v in values)],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f,~{memory}",
        True,
        loc,
        ip,
    )


class KdaCommitState:
    """Compile-time shape of one batched commit: heads, window, and record geometry."""

    def __init__(
        self,
        heads: int,
        window: int,
        capacity: int,
        slots: int,
        state_stride: int,
        correction_stride: int,
        key_gate_stride: int,
        value_tiles: int,
        min_ctas: int,
    ) -> None:
        self.heads = heads
        self.value_tiles = value_tiles
        self.min_ctas = min_ctas
        # Value rows of one thread; it holds keys [4 l, 4 l + 4) of each.
        self.rows = DIM // value_tiles // WARPS
        self.window = window
        self.capacity = capacity
        self.slots = slots
        self.state_stride = state_stride
        self.correction_stride = correction_stride
        self.key_gate_stride = key_gate_stride
        # kn_t, d_t, and u_t of every token, one FP32 row of DIM each.
        self.smem_bytes = 3 * window * DIM * 4 + 128

    @cute.jit
    def __call__(
        self,
        descriptors: cute.Tensor,
        group_indices: cute.Tensor,
        read_indices: cute.Tensor,
        write_indices: cute.Tensor,
        accepted_length: cute.Tensor,
        metadata: cute.Tensor,
        stamps: cute.Tensor,
        lower_bound: Float32,
        stream,
    ):
        self.kernel(
            descriptors.iterator,
            group_indices.iterator,
            read_indices.iterator,
            write_indices.iterator,
            accepted_length.iterator,
            metadata.iterator,
            stamps.iterator,
            lower_bound,
            read_indices.stride[0],
        ).launch(
            grid=[
                descriptors.shape[0] * self.heads * self.value_tiles,
                accepted_length.shape[0],
                1,
            ],
            block=[THREADS, 1, 1],
            smem=self.smem_bytes,
            stream=stream,
            min_blocks_per_mp=self.min_ctas,
        )

    @cute.kernel
    def kernel(
        self,
        descriptors,
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        metadata,
        stamps,
        lower_bound: Float32,
        group_stride: Int64,
    ):
        K = DIM
        W = self.window
        CAP = self.capacity
        tidx, _, _ = cute.arch.thread_idx()
        block, request, _ = cute.arch.block_idx()
        layer_head = block // self.value_tiles
        layer = layer_head // self.heads
        head = layer_head % self.heads
        ROWS = self.rows
        warp = tidx // 32
        lane = tidx % 32
        smem = cutlass.memory.SmemAllocator()
        keys = smem.allocate_tensor(
            Float32, cute.make_layout((W * K,)), byte_alignment=128
        )
        decays = smem.allocate_tensor(
            Float32, cute.make_layout((W * K,)), byte_alignment=128
        )
        corrections = smem.allocate_tensor(
            Float32, cute.make_layout((W * K,)), byte_alignment=128
        )
        keys_addr = Int32(keys.iterator.toint())
        decays_addr = Int32(decays.iterator.toint())
        corrections_addr = Int32(corrections.iterator.toint())

        row = Int64(layer) * 10
        state_base = Int64(_load(descriptors, row + _STATE))
        group = Int64(_load(group_indices, layer))
        source = Int64(_load(read_indices, group * group_stride + Int64(request)))
        target = Int64(_load(write_indices, group * group_stride + Int64(request)))
        n = Int32(_load(accepted_length, request))
        slot = Int64(_load(metadata, request * 3))
        generation = Int64(_load(metadata, request * 3 + 1))
        start = Int64(_load(metadata, request * 3 + 2))
        head_offset = Int64(head) * (K * K)
        first_row = (block % self.value_tiles) * (DIM // self.value_tiles) + warp * ROWS
        values = cute.make_rmem_tensor((ROWS * 4,), Float32)
        # A positive source page is readable whatever the records say; only the store waits.
        if source > 0:
            source_addr = state_base + (source * self.state_stride + head_offset) * 4
            for i in cutlass.range_constexpr(ROWS):
                offset = ((first_row + i) * K + lane * 4) * 4
                loaded = _ldg_f32x4(source_addr + offset)
                for j in cutlass.range_constexpr(4):
                    values[i * 4 + j] = loaded[j]
        # The weights and record bases do not depend on this round's checks.
        log_scale = _ldg_f32(Int64(_load(descriptors, row + _A_LOG)) + Int64(head) * 4)
        bias_addr = Int64(_load(descriptors, row + _DT_BIAS)) + Int64(head) * K * 4
        bias = [_ldg_f32(bias_addr + (lane + 32 * j) * 4) for j in range(4)]
        correction_layer = Int64(_load(descriptors, row + _CORRECTION))
        key_gate_layer = Int64(_load(descriptors, row + _KEY_GATE))
        gate_scale = _ex2(_mul(log_scale, LOG2E))
        # The Triton commit's early returns, none of which writes anything.
        valid = (source > 0) & (target > 0) & (slot >= 0) & (slot < self.slots)
        # Shared memory holds W tokens; the planner never accepts more (W <= CAP).
        valid = valid & (n >= 0) & (n <= W) & (start >= 0) & (generation >= 0)
        # Every accepted record must carry this round's position and generation.
        stamp_ok = Int32(1)
        if valid & (lane < n):
            position = start + lane
            stamp = (Int64(layer) * self.slots + slot) * CAP + position % CAP
            stamped_position = Int64(_load(stamps, stamp * 2))
            stamped_generation = Int64(_load(stamps, stamp * 2 + 1))
            if (stamped_position != position) | (stamped_generation != generation):
                stamp_ok = Int32(0)
        commit = valid & cute.arch.vote_all_sync(stamp_ok != 0)
        # Records of a valid interval are addressable; the stamps gate only the store.
        if valid:
            correction_base = (
                correction_layer
                + (slot * CAP * self.correction_stride + Int64(head) * K) * 4
            )
            key_gate_base = (
                key_gate_layer
                + (slot * CAP * self.key_gate_stride + Int64(head) * 2 * K) * 2
            )
            for r in cutlass.range_constexpr((W + WARPS - 1) // WARPS):
                t = warp + r * WARPS
                if t < n:
                    position = (start + t) % CAP
                    record = key_gate_base + position * self.key_gate_stride * 2
                    key = [_ldg_bf16(record + (lane + 32 * j) * 2) for j in range(4)]
                    raw = [
                        _ldg_bf16(record + (K + lane + 32 * j) * 2) for j in range(4)
                    ]
                    # The one-warp layout's sum of squares: two FMA pairs, then lanes.
                    squares = _add(
                        _fma(key[0], key[0], _mul(key[2], key[2])),
                        _fma(key[1], key[1], _mul(key[3], key[3])),
                    )
                    for offset in (16, 8, 4, 2, 1):
                        squares = _add(
                            squares,
                            cute.arch.shuffle_sync_bfly(
                                squares, offset=offset, mask=-1, mask_and_clamp=31
                            ),
                        )
                    inverse_norm = _rsqrt(_add(squares, NORM_EPS))
                    for j in cutlass.range_constexpr(4):
                        column = (t * K + lane + 32 * j) * 4
                        _sts_f32(keys_addr + column, _mul(inverse_norm, key[j]))
                        # lower_bound * sigmoid(A * (raw + bias)), then exp, as tl.exp.
                        scaled = _mul(gate_scale, -_add(bias[j], raw[j]))
                        sigmoid = _div(
                            Float32(1.0), _add(_ex2(_mul(scaled, LOG2E)), Float32(1.0))
                        )
                        decay = _ex2(_mul(_mul(lower_bound, sigmoid), LOG2E))
                        _sts_f32(decays_addr + column, decay)
                    correction = _ldg_f32x4(
                        correction_base
                        + (position * self.correction_stride + lane * 4) * 4
                    )
                    _sts_f32x4(corrections_addr + (t * K + lane * 4) * 4, correction)
        cute.arch.barrier()
        if commit:
            target_addr = state_base + (target * self.state_stride + head_offset) * 4
            for t in cutlass.range(n, unroll=1):
                decay = _lds_f32x4(decays_addr + (t * K + lane * 4) * 4)
                key = _lds_f32x4(keys_addr + (t * K + lane * 4) * 4)
                for q in cutlass.range_constexpr(ROWS // 4):
                    update = _lds_f32x4(
                        corrections_addr + (t * K + first_row + q * 4) * 4
                    )
                    for i in cutlass.range_constexpr(4):
                        for j in cutlass.range_constexpr(4):
                            index = (q * 4 + i) * 4 + j
                            values[index] = _fma(
                                update[i], key[j], _mul(values[index], decay[j])
                            )
            for i in cutlass.range_constexpr(ROWS):
                offset = ((first_row + i) * K + lane * 4) * 4
                _stg_f32x4(target_addr + offset, [values[i * 4 + j] for j in range(4)])


@functools.cache
def _compile(
    heads: int,
    window: int,
    capacity: int,
    slots: int,
    state_stride: int,
    correction_stride: int,
    key_gate_stride: int,
    value_tiles: int,
    min_ctas: int,
):
    layers = cute.sym_int()
    groups = cute.sym_int()
    batch = cute.sym_int()
    descriptors = make_fake_tensor(Int64, (layers, 10))
    group_indices = make_fake_tensor(Int32, (layers,))
    read_indices = make_fake_tensor(Int32, (groups, batch))
    write_indices = make_fake_tensor(Int32, (groups, batch))
    accepted_length = make_fake_tensor(Int32, (batch,))
    metadata = make_fake_tensor(Int64, (batch, 3))
    stamps = make_fake_tensor(Int64, (layers, slots, capacity, 2))
    kernel = KdaCommitState(
        heads,
        window,
        capacity,
        slots,
        state_stride,
        correction_stride,
        key_gate_stride,
        value_tiles,
        min_ctas,
    )
    return cute.compile(
        kernel,
        descriptors,
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        metadata,
        stamps,
        Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def commit_states_supported(
    *,
    activation_dtype: torch.dtype,
    state_dtype: torch.dtype,
    draft_token_num: int,
    head_dim: int,
    lower_bound: float | None,
    record_metadata: torch.Tensor | None,
) -> bool:
    """Whether ``commit_states`` takes this batched commit.

    It takes BF16 records, FP32 state, 128-wide heads, a bounded gate, record
    metadata, and windows of at most ``MAX_WINDOW`` tokens (the Triton
    commit's CUDA-core path).
    """
    return (
        activation_dtype == torch.bfloat16
        and state_dtype == torch.float32
        and head_dim == DIM
        and lower_bound is not None
        and record_metadata is not None
        and 1 <= draft_token_num <= MAX_WINDOW
    )


def commit_states(
    descriptors: torch.Tensor,
    group_indices: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    *,
    draft_token_num: int,
    num_heads: int,
    correction_stride: int,
    key_gate_stride: int,
    state_stride: int,
    lower_bound: float,
    record_metadata: torch.Tensor,
    record_stamps: torch.Tensor,
) -> None:
    """Commit every layer's accepted tokens into its FP32 state pages.

    The arguments are ``batched_recoverssm_commit``'s, already checked by it
    (``check_batched_commit``) and admitted by ``commit_states_supported``;
    the convolution history is committed separately.

    Args:
        descriptors: ``[layers, 10]`` per-layer pointer table; columns 3, 5,
            6, 7 and 8 hold the FP32 corrections, BF16 key / raw-gate records,
            FP32 A_log, FP32 dt_bias and FP32 state pages.
        group_indices: INT32 ``[layers]`` row of each layer's pages.
        read_indices: INT32 ``[groups, batch]`` source pages (non-positive skips).
        write_indices: INT32 ``[groups, batch]`` destination pages.
        accepted_length: INT32 ``[batch]`` tokens to fold, in ``[0, window]``;
            a request outside that range is not written.
        draft_token_num: The verify window.
        num_heads: Local heads per layer.
        correction_stride: Token stride of the corrections, in elements.
        key_gate_stride: Token stride of the key / raw-gate records.
        state_stride: Page stride of the state pools, in elements.
        lower_bound: The gate's lower bound.
        record_metadata: INT64 ``[batch, 3]`` (slot, generation, position).
        record_stamps: INT64 ``[layers, slots, capacity, 2]`` record stamps.
    """
    kernel = _compile(
        num_heads,
        draft_token_num,
        record_stamps.shape[2],
        record_stamps.shape[1],
        state_stride,
        correction_stride,
        key_gate_stride,
        VALUE_TILES,
        MIN_CTAS,
    )
    kernel(
        descriptors.view(torch.int64),
        group_indices,
        read_indices,
        write_indices,
        accepted_length,
        record_metadata,
        record_stamps,
        float(lower_bound),
    )
