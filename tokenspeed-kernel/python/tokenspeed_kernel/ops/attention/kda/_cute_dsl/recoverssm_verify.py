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

"""RecoverSSM KDA verify in CuTe DSL, one CTA per (value tile, request, head).

Contract: inputs, outputs and records match ``kda_recoverssm_verify``, and the
checkpoint is read only.

Chunked form. For one request and head, let S0 be the [V, K] checkpoint, G_t
the cumulative gate (one value per key), k_t and q_t the raw keys and queries,
rk_t = rsqrt(|k_t|^2 + 1e-6) and rq_t = rsqrt(|q_t|^2 + 1e-6) / sqrt(K). Then

    P     = S0 @ [k_0 e^G_0, ..., k_{W-1} e^G_{W-1} | q_0 e^G_0, ...]
    a_tj  = rk_t rk_j sum_k k_t k_j e^(G_t - G_j)      for j < t
    b_tj  = rq_t rk_j sum_k q_t k_j e^(G_t - G_j)      for j <= t
    c_t   = beta_t (v_t - rk_t P[:, t] - sum_{j<t} a_tj c_j)
    out_t = rq_t P[:, W + t] + sum_{j<=t} b_tj c_j

where c_t is the correction record and out_t the verify output.

A CTA of 128 threads bulk-copies the window's token rows and its [rows, K]
checkpoint tile into shared memory, forms the gates and the decayed columns
(one thread per key), forms the pair sums, multiplies the tile by the columns
on CUDA cores with K split over eight lanes, finalizes a_tj and b_tj, and
solves the window for c_t and out_t (one thread per value channel).

Pair sums. With a bounded gate and spec_query_len * |lower_bound| <=
FACTORED_DECAY, e^-G stays a normal FP32 number, so extra tile rows
k_j e^-G_j and q_j e^-G_j ride along in the tile product and yield the pair
sums and norms. Otherwise each pair adds up the gates between j and t.

Numerical limits. Products use raw keys and queries, so |k| e^48 and
|state| |k| K must stay below 3e38 (model activations are far below that), and
e^G is formed before the checkpoint product, so a checkpoint decayed past e^-87
contributes zero (an absolute error below |state| x 1.2e-38).
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float16, Float32, Int32, Int64, const_expr, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.compile_utils import make_fake_tensor
from tokenspeed_kernel.thirdparty.vllm_recoverssm.kda import (
    check_recoverssm_verify_inputs,
)

__all__ = ["FACTORED_DECAY", "recoverssm_verify", "recoverssm_verify_supported"]

THREADS = 128
LOG2E = 1.4426950408889634
# Largest window decay for which the factored rows' e^-G is a normal FP32 number.
FACTORED_DECAY = 48.0
# Gates are clamped here so sums stay finite; e^-1e30 is already exactly zero.
GATE_FLOOR = -1e30
# The phase-2 product splits K over this many lanes of a warp.
KEY_SPLITS = 8
_DTYPES = {
    torch.bfloat16: BFloat16,
    torch.float16: Float16,
    torch.float32: Float32,
    torch.int32: Int32,
    torch.int64: Int64,
}


def _asm(ret, args, text, constraints, side_effects, loc, ip):
    return llvm.inline_asm(
        ret,
        [arg.ir_value(loc=loc, ip=ip) for arg in args],
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _unpack(value, count, loc, ip):
    return [
        Float32(llvm.extractvalue(T.f32(), value, [i], loc=loc, ip=ip))
        for i in range(count)
    ]


@dsl_user_op
def _lds_f32x4(byte_addr: Int32, *, loc=None, ip=None):
    """Four FP32 values from a 16-byte aligned shared-memory address."""
    value = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int32(byte_addr)],
        "ld.shared.v4.f32 {$0,$1,$2,$3}, [$4];",
        "=f,=f,=f,=f,r",
        True,
        loc,
        ip,
    )
    return _unpack(value, 4, loc, ip)


@dsl_user_op
def _lds_bf16x4(byte_addr: Int32, *, loc=None, ip=None):
    """Four BF16 values from an 8-byte aligned shared-memory address, as FP32."""
    value = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int32(byte_addr)],
        "{ .reg .b32 a, b; ld.shared.v2.b32 {a, b}, [$4]; shl.b32 $0, a, 16; "
        "and.b32 $1, a, 0xffff0000; shl.b32 $2, b, 16; and.b32 $3, b, 0xffff0000; }",
        "=f,=f,=f,=f,r",
        True,
        loc,
        ip,
    )
    return _unpack(value, 4, loc, ip)


@dsl_user_op
def _sts_f32x4(byte_addr: Int32, values, *, loc=None, ip=None):
    """Store four FP32 values to a 16-byte aligned shared-memory address."""
    _asm(
        None,
        [Int32(byte_addr)] + [Float32(x) for x in values],
        "st.shared.v4.f32 [$0], {$1,$2,$3,$4};",
        "r,f,f,f,f,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _bulk_load(
    dst_byte_addr: Int32,
    src_byte_addr: Int64,
    size: int,
    barrier_byte_addr: Int32,
    *,
    loc=None,
    ip=None,
):
    """Copy global memory to shared memory, completing bytes on an mbarrier."""
    _asm(
        None,
        [
            Int32(dst_byte_addr),
            Int64(src_byte_addr),
            Int32(size),
            Int32(barrier_byte_addr),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[$0], [$1], $2, [$3];",
        "r,l,r,r,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _store_if(
    live: Int32, byte_addr: Int64, value: Float32, bf16: bool, *, loc=None, ip=None
):
    """Store to global memory when live; a predicate keeps the code branch-free."""
    if bf16:
        text = (
            "{ .reg .pred p; .reg .b16 x; setp.ne.b32 p, $0, 0; "
            "cvt.rn.bf16.f32 x, $2; @p st.global.b16 [$1], x; }"
        )
    else:
        text = "{ .reg .pred p; setp.ne.b32 p, $0, 0; @p st.global.f32 [$1], $2; }"
    _asm(
        None,
        [Int32(live), Int64(byte_addr), Float32(value)],
        text,
        "r,l,f",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _rsqrt(x: Float32, *, loc=None, ip=None) -> Float32:
    """Triton's rsqrt: rsqrt.approx.ftz."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(x)],
            "rsqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _div(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Triton's FP32 division: div.full."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(a), Float32(b)],
            "div.full.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@cute.jit
def _exp(x: Float32) -> Float32:
    return cute.math.exp2(x * LOG2E, fastmath=True)


def _load(ptr, offset):
    return cute.make_tensor(ptr + offset, cute.make_layout((1,)))[0]


def _round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


class _SumsTable:
    """Word offsets of the per-CTA FP32 table of norms, pair sums and coefficients."""

    def __init__(self, window: int) -> None:
        square = window * window
        # Raw sums: |k_t|^2, |q_t|^2, [t][j] key and query-key pair sums.
        self.key_norm = 0
        self.query_norm = window
        self.key_pairs = 2 * window
        self.query_key_pairs = self.key_pairs + square
        # Finalized: rk_t, rq_t, a_tj and b_tj as [t][j], and beta_t.
        self.key_scale = self.query_key_pairs + square
        self.query_scale = self.key_scale + window
        self.correction_coeffs = self.query_scale + window
        self.output_coeffs = self.correction_coeffs + square
        self.beta = self.output_coeffs + square
        self.words = _round_up(self.beta + window, 4)


class _Shared(NamedTuple):
    """Shared-memory buffers of one CTA and the byte addresses PTX helpers take."""

    tile: cute.Tensor
    tile_addr: Int32
    columns: cute.Tensor
    columns_addr: Int32
    sums: cute.Tensor
    tokens: cute.Tensor
    tokens_addr: Int32
    gate_steps: cute.Tensor | None
    norm: cute.Tensor | None
    token_barrier: object
    state_barrier: object
    weight_barrier: object
    norm_barrier: object


class RecoverSsmVerify:
    """Verify kernel for one head shape, window, value tile and gate form.

    Shared memory per CTA:
      tile: the [tile_rows, K] FP32 checkpoint tile, then (factored) the extra
        rows k_j e^-G_j and q_j e^-G_j; after the product it is reused for the
        [KEY_SPLITS, tile_rows, partial_row_stride] FP32 partial products.
      columns: M, the [K, padded_columns] decayed keys then queries, stored in
        the order the product loads them (see ``_column_offset``).
      sums: norms, pair sums and coefficients (see ``_SumsTable``).
      tokens: BF16 rows q, k, gate as [W, K] each, then this tile's v as [W, VT].
      gate_steps: (pairwise only) each token's FP32 gate as [W, K].
      norm: (subclasses that set ``norm_words``) FP32 output-norm sums.
    """

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        window: int,
        value_tile_rows: int,
        capacity: int,
        bounded_gate: bool,
        factored: bool,
        has_metadata: bool,
    ) -> None:
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.window = window
        self.value_tile_rows = value_tile_rows
        self.capacity = capacity
        self.bounded_gate = bounded_gate
        self.factored = factored
        self.has_metadata = has_metadata
        self.padded_columns = _round_up(2 * window, 4)
        self.column_vectors = self.padded_columns // 4
        # Factored rows come in multiples of 16 so every lane group owns whole rows.
        self.extra_rows = _round_up(2 * window, 16) if factored else 0
        self.tile_rows = value_tile_rows + self.extra_rows
        self.rows_per_thread = self.tile_rows // 16
        # The +4 words shift consecutive rows and splits across shared-memory banks.
        self.partial_row_stride = self.padded_columns + 4
        self.partial_split_stride = self.tile_rows * self.partial_row_stride + 4
        self.tile_words = max(
            self.tile_rows * key_dim, KEY_SPLITS * self.partial_split_stride
        )
        self.sums = _SumsTable(window)
        self.token_q = 0
        self.token_k = window * key_dim
        self.token_gate = 2 * window * key_dim
        self.token_v = 3 * window * key_dim
        self.token_elems = window * (3 * key_dim + value_tile_rows)
        self.gate_step_words = 0 if factored else window * key_dim
        self.norm_words = 0
        # 512 bytes cover alignment padding between regions and the four mbarriers.
        self.smem_bytes = (
            4 * self.tile_words
            + 4 * key_dim * self.padded_columns
            + 4 * self.gate_step_words
            + 4 * self.sums.words
            + 2 * self.token_elems
            + 512
        )

    # Shared-memory index maps, each used by both the producer and the consumer.

    def _column_offset(self, chunk, within, split, column):
        """Element offset of M[32 * chunk + 4 * split + within, column].

        A lane of key split ``split`` loads columns 4c..4c+3 of one key with a
        single 16-byte load, and the eight splits of a chunk sit side by side,
        so a warp's load is one contiguous 128-byte wavefront.
        """
        return (
            ((chunk * 4 + within) * self.column_vectors + column // 4) * 32
            + split * 4
            + column % 4
        )

    def _partial_offset(self, split, row, column):
        """Element offset of the partial product of tile row ``row`` over key split ``split``."""
        return (
            split * self.partial_split_stride + row * self.partial_row_stride + column
        )

    def _tile_row(self, warp, row_group, iteration):
        """Tile row that a lane of ``row_group`` in ``warp`` handles in ``iteration``."""
        return warp * (self.tile_rows // 4) + row_group + 4 * iteration

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        gate: cute.Tensor,
        beta: cute.Tensor,
        a_log: cute.Tensor,
        dt_bias: cute.Tensor,
        state: cute.Tensor,
        correction: cute.Tensor,
        key_gate: cute.Tensor,
        out: cute.Tensor,
        query_start_loc: cute.Tensor,
        state_indices: cute.Tensor,
        metadata: cute.Tensor | None,
        stamps: cute.Tensor | None,
        lower_bound: Float32,
        stream: CUstream,
    ):
        metadata_ptr = None
        stamps_ptr = None
        if const_expr(self.has_metadata):
            metadata_ptr = metadata.iterator
            stamps_ptr = stamps.iterator
        self.kernel(
            q.iterator,
            k.iterator,
            v.iterator,
            gate.iterator,
            beta.iterator,
            a_log.iterator,
            dt_bias.iterator,
            state.iterator,
            correction.iterator,
            key_gate.iterator,
            out.iterator,
            query_start_loc.iterator,
            state_indices.iterator,
            metadata_ptr,
            stamps_ptr,
            lower_bound,
            q.stride[0],
            k.stride[0],
            v.stride[0],
            gate.stride[0],
            beta.stride[0],
            out.stride[0],
            state.stride[0],
            state.stride[1],
            correction.stride[0],
            correction.stride[1],
            correction.stride[2],
            key_gate.stride[0],
            key_gate.stride[1],
            key_gate.stride[2],
        ).launch(
            grid=[
                self.value_dim // self.value_tile_rows,
                state_indices.shape[0],
                q.shape[1],
            ],
            block=[THREADS, 1, 1],
            smem=self.smem_bytes,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        dt_bias,
        state,
        correction,
        key_gate,
        out,
        query_start_loc,
        state_indices,
        metadata,
        stamps,
        lower_bound: Float32,
        q_token_stride: Int64,
        k_token_stride: Int64,
        v_token_stride: Int64,
        gate_token_stride: Int64,
        beta_token_stride: Int64,
        out_token_stride: Int64,
        state_page_stride: Int64,
        state_head_stride: Int64,
        correction_slot_stride: Int64,
        correction_head_stride: Int64,
        correction_position_stride: Int64,
        key_gate_slot_stride: Int64,
        key_gate_head_stride: Int64,
        key_gate_position_stride: Int64,
    ):
        K = self.key_dim
        W = self.window
        VT = self.value_tile_rows
        tidx, _, _ = cute.arch.thread_idx()
        value_tile, request, head = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        shared = self._allocate_shared()

        token_start = Int64(_load(query_start_loc, request))
        valid_tokens = Int32(Int64(_load(query_start_loc, request + 1)) - token_start)
        page = Int64(_load(state_indices, request))
        value_start = value_tile * VT

        if (page <= 0) | (valid_tokens <= 0):
            # Padding requests: zero outputs for their tokens, no records.
            if tidx < VT:
                for t in cutlass.range_constexpr(W):
                    row = out + (
                        (token_start + t) * out_token_stride
                        + head * self.value_dim
                        + value_start
                        + tidx
                    )
                    _store_if(Int32(t < valid_tokens), row.toint(), Float32(0.0), True)
        else:
            if tidx == 0:
                cute.arch.mbarrier_init(shared.token_barrier, 1)
                cute.arch.mbarrier_init(shared.state_barrier, 1)
            cute.arch.mbarrier_init_fence()
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    shared.token_barrier, W * (3 * K + VT) * 2
                )
                cute.arch.mbarrier_arrive_and_expect_tx(
                    shared.state_barrier, VT * K * 4
                )
            cute.arch.barrier()
            self._copy_token_rows(
                shared,
                q,
                k,
                v,
                gate,
                q_token_stride,
                k_token_stride,
                v_token_stride,
                gate_token_stride,
                token_start,
                valid_tokens,
                head,
                value_start,
                warp,
                lane,
            )
            if tidx == 0:
                tile_src = state + (
                    page * state_page_stride
                    + head * state_head_stride
                    + value_start * K
                )
                _bulk_load(
                    shared.tile_addr,
                    tile_src.toint(),
                    VT * K * 4,
                    Int32(shared.state_barrier.toint()),
                )

            slot = Int64(request)
            generation = Int64(0)
            start = Int64(0)
            if const_expr(self.has_metadata):
                slot = Int64(_load(metadata, request * 3))
                generation = Int64(_load(metadata, request * 3 + 1))
                start = Int64(_load(metadata, request * 3 + 2))
            # Reduce the 64-bit start before narrowing so ring positions stay exact.
            ring_start = Int32(start % self.capacity)
            gate_scale = _exp(_load(a_log, head).to(Float32))
            gate_bias = Float32(0.0)
            if tidx < K:
                gate_bias = _load(dt_bias, head * K + tidx).to(Float32)
            if tidx < W:
                token = token_start + cutlass.min(tidx, valid_tokens - 1)
                raw_beta = _load(beta, token * beta_token_stride + head).to(Float32)
                shared.sums[self.sums.beta + tidx] = _div(
                    Float32(1.0), Float32(1.0) + _exp(-raw_beta)
                )
            if const_expr(self.has_metadata):
                # One CTA per request stamps the positions this window records.
                if (value_tile == 0) & (head == 0) & (tidx < W) & (tidx < valid_tokens):
                    stamp = cute.make_tensor(
                        stamps
                        + (slot * self.capacity + (ring_start + tidx) % self.capacity)
                        * 2,
                        cute.make_layout((2,)),
                    )
                    stamp[0] = start + tidx
                    stamp[1] = generation
            cute.arch.mbarrier_wait(shared.token_barrier, 0)
            self._project_window(
                shared,
                tidx,
                warp,
                lane,
                gate_bias,
                gate_scale,
                lower_bound,
                key_gate + (slot * key_gate_slot_stride + head * key_gate_head_stride),
                key_gate_position_stride,
                Int32(value_tile == 0),
                valid_tokens,
                ring_start,
            )

            # Phase 3, one thread per value channel.
            if tidx < VT:
                corrections, outputs = self._solve_window(shared, tidx)
                out_row = out + (
                    token_start * out_token_stride
                    + head * self.value_dim
                    + value_start
                    + tidx
                )
                correction_row = correction + (
                    slot * correction_slot_stride
                    + head * correction_head_stride
                    + value_start
                    + tidx
                )
                for t in cutlass.range_constexpr(W):
                    live = Int32(t < valid_tokens)
                    position = (ring_start + t) % self.capacity
                    _store_if(
                        live, (out_row + t * out_token_stride).toint(), outputs[t], True
                    )
                    _store_if(
                        live,
                        (
                            correction_row + position * correction_position_stride
                        ).toint(),
                        corrections[t],
                        False,
                    )

    @cute.jit
    def _project_window(
        self,
        shared: _Shared,
        tidx: Int32,
        warp: Int32,
        lane: Int32,
        gate_bias: Float32,
        gate_scale: Float32,
        lower_bound: Float32,
        key_record,
        key_record_position_stride: Int64,
        records_keys: Int32,
        valid_tokens: Int32,
        ring_start: Int32,
    ):
        """Phases 1 and 2 on the token rows in shared memory.

        Leaves P's partials in the tile and the coefficients in the sums table
        for ``_solve_window``. ``key_record`` is this (slot, head)'s key and
        raw-gate record; a CTA stores its window positions when
        ``records_keys`` is set.
        """
        K = self.key_dim
        W = self.window
        # Phase 1, one thread per key; every load precedes every store.
        if tidx < K:
            raw_gates, raw_keys, raw_queries = self._load_key_channel(shared, tidx)
            steps, prefix_hi, prefix_lo = self._gate_prefix(
                raw_gates, gate_bias, gate_scale, lower_bound
            )
            self._store_key_channel(
                shared, tidx, raw_keys, raw_queries, steps, prefix_hi, prefix_lo
            )
            for t in cutlass.range_constexpr(W):
                live = Int32((records_keys != 0) & (t < valid_tokens))
                position = (ring_start + t) % self.capacity
                entry = key_record + (position * key_record_position_stride + tidx)
                _store_if(live, entry.toint(), raw_keys[t], True)
                _store_if(live, (entry + K).toint(), raw_gates[t], True)
        cute.arch.barrier()

        if const_expr(not self.factored):
            self._pairwise_sums(shared, tidx, lane)
            cute.arch.barrier()
            self._finalize_coefficients(shared.sums, tidx)

        # Phase 2: tile @ M; the barrier after it frees the tile for partials.
        cute.arch.mbarrier_wait(shared.state_barrier, 0)
        products = self._tile_times_columns(shared, warp, lane)
        cute.arch.barrier()
        self._store_partials(shared, products, warp, lane)
        cute.arch.barrier()

        if const_expr(self.factored):
            self._factored_sums(shared, tidx)
            cute.arch.barrier()
            self._finalize_coefficients(shared.sums, tidx)
            cute.arch.barrier()

    def _allocate_shared(self) -> _Shared:
        smem = cutlass.memory.SmemAllocator()
        tile = smem.allocate_tensor(
            Float32, cute.make_layout((self.tile_words,)), byte_alignment=128
        )
        columns = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.key_dim * self.padded_columns,)),
            byte_alignment=128,
        )
        sums = smem.allocate_tensor(
            Float32, cute.make_layout((self.sums.words,)), byte_alignment=16
        )
        tokens = smem.allocate_tensor(
            BFloat16, cute.make_layout((self.token_elems,)), byte_alignment=128
        )
        gate_steps = None
        if const_expr(not self.factored):
            gate_steps = smem.allocate_tensor(
                Float32, cute.make_layout((self.gate_step_words,)), byte_alignment=128
            )
        norm = None
        if const_expr(self.norm_words > 0):
            norm = smem.allocate_tensor(
                Float32, cute.make_layout((self.norm_words,)), byte_alignment=16
            )
        barriers = smem.allocate_array(Int64, num_elems=4)
        return _Shared(
            tile=tile,
            tile_addr=Int32(tile.iterator.toint()),
            columns=columns,
            columns_addr=Int32(columns.iterator.toint()),
            sums=sums,
            tokens=tokens,
            tokens_addr=Int32(tokens.iterator.toint()),
            gate_steps=gate_steps,
            norm=norm,
            token_barrier=barriers,
            state_barrier=barriers + 1,
            weight_barrier=barriers + 2,
            norm_barrier=barriers + 3,
        )

    @cute.jit
    def _copy_token_rows(
        self,
        shared: _Shared,
        q,
        k,
        v,
        gate,
        q_token_stride: Int64,
        k_token_stride: Int64,
        v_token_stride: Int64,
        gate_token_stride: Int64,
        token_start: Int64,
        valid_tokens: Int32,
        head: Int32,
        value_start: Int32,
        warp: Int32,
        lane: Int32,
    ):
        """Bulk-copy the window's q, k, gate rows and this tile's v rows, spread over warps."""
        K = self.key_dim
        W = self.window
        copy = lane * 4 + warp
        if copy < 4 * W:
            kind = copy // W
            t = copy % W
            # Past the window's end the last token repeats; its results are not stored.
            token = token_start + cutlass.min(t, valid_tokens - 1)
            barrier = Int32(shared.token_barrier.toint())
            if kind == 0:
                src = q + (token * q_token_stride + head * K)
                self._copy_row(shared, self.token_q + t * K, src, K, barrier)
            elif kind == 1:
                src = k + (token * k_token_stride + head * K)
                self._copy_row(shared, self.token_k + t * K, src, K, barrier)
            elif kind == 2:
                src = gate + (token * gate_token_stride + head * K)
                self._copy_row(shared, self.token_gate + t * K, src, K, barrier)
            else:
                src = v + (token * v_token_stride + head * self.value_dim + value_start)
                self._copy_row(
                    shared,
                    self.token_v + t * self.value_tile_rows,
                    src,
                    self.value_tile_rows,
                    barrier,
                )

    def _copy_row(self, shared: _Shared, token_offset, src, elems: int, barrier: Int32):
        _bulk_load(
            shared.tokens_addr + token_offset * 2, src.toint(), elems * 2, barrier
        )

    @cute.jit
    def _load_key_channel(self, shared: _Shared, key: Int32):
        """Raw gates, keys and queries of one key channel for every window token."""
        K = self.key_dim
        raw_gates = []
        raw_keys = []
        raw_queries = []
        for t in cutlass.range_constexpr(self.window):
            raw_gates.append(shared.tokens[self.token_gate + t * K + key].to(Float32))
            raw_keys.append(shared.tokens[self.token_k + t * K + key].to(Float32))
            raw_queries.append(shared.tokens[self.token_q + t * K + key].to(Float32))
        return raw_gates, raw_keys, raw_queries

    @cute.jit
    def _gate_prefix(
        self, raw_gates, gate_bias: Float32, gate_scale: Float32, lower_bound: Float32
    ):
        """Per-step gates and their compensated prefix sums G_t = hi_t + lo_t."""
        steps = []
        prefix_hi = []
        prefix_lo = []
        hi = Float32(0.0)
        lo = Float32(0.0)
        for t in cutlass.range_constexpr(self.window):
            x = raw_gates[t] + gate_bias
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
            # TwoSum: lo collects what rounding hi + step dropped.
            total = hi + step
            spill = total - hi
            lo = lo + ((hi - (total - spill)) + (step - spill))
            hi = total
            steps.append(step)
            prefix_hi.append(hi)
            prefix_lo.append(lo)
        return steps, prefix_hi, prefix_lo

    @cute.jit
    def _store_key_channel(
        self,
        shared: _Shared,
        key: Int32,
        raw_keys,
        raw_queries,
        steps,
        prefix_hi,
        prefix_lo,
    ):
        """Write M's row for this key, plus the factored rows or the gate steps."""
        K = self.key_dim
        W = self.window
        quad = key // 4
        chunk = quad // KEY_SPLITS
        split = quad % KEY_SPLITS
        within = key % 4
        for column in cutlass.range_constexpr(2 * W, self.padded_columns):
            shared.columns[self._column_offset(chunk, within, split, column)] = Float32(
                0.0
            )
        for t in cutlass.range_constexpr(W):
            # e^(hi + lo) to first order in lo.
            decay = _exp(prefix_hi[t])
            decay = decay + decay * prefix_lo[t]
            shared.columns[self._column_offset(chunk, within, split, t)] = (
                raw_keys[t] * decay
            )
            shared.columns[self._column_offset(chunk, within, split, W + t)] = (
                raw_queries[t] * decay
            )
            if const_expr(self.factored):
                growth = _exp(-prefix_hi[t])
                growth = growth - growth * prefix_lo[t]
                tile_row = self.value_tile_rows + t
                shared.tile[tile_row * K + key] = raw_keys[t] * growth
                shared.tile[(tile_row + W) * K + key] = raw_queries[t] * growth
            else:
                shared.gate_steps[t * K + key] = steps[t]
        if const_expr(self.factored):
            for row in cutlass.range_constexpr(2 * W, self.extra_rows):
                shared.tile[(self.value_tile_rows + row) * K + key] = Float32(0.0)

    @cute.jit
    def _pairwise_sums(self, shared: _Shared, tidx: Int32, lane: Int32):
        """Norms and decayed pair sums, one (t, j <= t) item per eight-lane group."""
        K = self.key_dim
        W = self.window
        table = self.sums
        group = tidx // 8
        group_lane = lane % 8
        pairs = W * (W - 1) // 2
        items = pairs + W
        groups = THREADS // 8
        gate_steps_addr = Int32(shared.gate_steps.iterator.toint())
        for it in cutlass.range((items + groups - 1) // groups, unroll=1):
            item = it * groups + group
            # Surplus groups redo the last item so the shuffles below see full warps.
            clamped = cutlass.min(item, items - 1)
            is_pair = clamped < pairs
            t = clamped - pairs
            j = t
            if is_pair:
                # Items 0..pairs-1 enumerate j < t row by row.
                t = Int32(1)
                for row in cutlass.range_constexpr(2, W):
                    if clamped >= row * (row - 1) // 2:
                        t = Int32(row)
                j = clamped - t * (t - 1) // 2
            key_pair = Float32(0.0)
            query_key_pair = Float32(0.0)
            query_norm = Float32(0.0)
            for chunk in cutlass.range_constexpr(K // 32):
                key = 4 * (group_lane + 8 * chunk)
                kt = _lds_bf16x4(shared.tokens_addr + (self.token_k + t * K + key) * 2)
                qt = _lds_bf16x4(shared.tokens_addr + (self.token_q + t * K + key) * 2)
                if is_pair:
                    kj = _lds_bf16x4(
                        shared.tokens_addr + (self.token_k + j * K + key) * 2
                    )
                    # G_t - G_j summed over the gates in (j, t], not prefix sums.
                    d0 = Float32(0.0)
                    d1 = Float32(0.0)
                    d2 = Float32(0.0)
                    d3 = Float32(0.0)
                    for s in cutlass.range_constexpr(1, W):
                        if (s > j) & (s <= t):
                            g = _lds_f32x4(gate_steps_addr + (s * K + key) * 4)
                            d0 = d0 + g[0]
                            d1 = d1 + g[1]
                            d2 = d2 + g[2]
                            d3 = d3 + g[3]
                    decays = (d0, d1, d2, d3)
                    for r in cutlass.range_constexpr(4):
                        decayed_key = kj[r] * _exp(decays[r])
                        key_pair = key_pair + kt[r] * decayed_key
                        query_key_pair = query_key_pair + qt[r] * decayed_key
                else:
                    for r in cutlass.range_constexpr(4):
                        key_pair = key_pair + kt[r] * kt[r]
                        query_key_pair = query_key_pair + qt[r] * kt[r]
                        query_norm = query_norm + qt[r] * qt[r]
            for offset in (4, 2, 1):
                key_pair = key_pair + cute.arch.shuffle_sync_bfly(
                    key_pair, offset=offset, mask=-1, mask_and_clamp=31
                )
                query_key_pair = query_key_pair + cute.arch.shuffle_sync_bfly(
                    query_key_pair, offset=offset, mask=-1, mask_and_clamp=31
                )
                query_norm = query_norm + cute.arch.shuffle_sync_bfly(
                    query_norm, offset=offset, mask=-1, mask_and_clamp=31
                )
            if (group_lane == 0) & (item < items):
                shared.sums[table.query_key_pairs + t * W + j] = query_key_pair
                if is_pair:
                    shared.sums[table.key_pairs + t * W + j] = key_pair
                else:
                    shared.sums[table.key_norm + t] = key_pair
                    shared.sums[table.query_norm + t] = query_norm

    @cute.jit
    def _tile_times_columns(self, shared: _Shared, warp: Int32, lane: Int32):
        """This lane's partial of tile @ M over its key split, as [rows, padded_columns]."""
        K = self.key_dim
        columns = self.padded_columns
        split = lane % KEY_SPLITS
        row_group = lane // KEY_SPLITS
        products = cute.make_rmem_tensor((self.rows_per_thread * columns,), Float32)
        products.fill(0.0)
        for chunk in cutlass.range(K // 32, unroll=1):
            key0 = 4 * (split + KEY_SPLITS * chunk)
            tile_values = []
            for i in cutlass.range_constexpr(self.rows_per_thread):
                row = self._tile_row(warp, row_group, i)
                tile_values.append(_lds_f32x4(shared.tile_addr + (row * K + key0) * 4))
            for within in cutlass.range_constexpr(4):
                m_row = []
                for vector in cutlass.range_constexpr(self.column_vectors):
                    offset = self._column_offset(chunk, within, split, 4 * vector)
                    m_row += _lds_f32x4(shared.columns_addr + offset * 4)
                for i in cutlass.range_constexpr(self.rows_per_thread):
                    s = tile_values[i][within]
                    for n in cutlass.range_constexpr(0, columns, 2):
                        products[i * columns + n], products[i * columns + n + 1] = (
                            cute.arch.fma_packed_f32x2(
                                (s, s),
                                (m_row[n], m_row[n + 1]),
                                (
                                    products[i * columns + n],
                                    products[i * columns + n + 1],
                                ),
                            )
                        )
        return products

    @cute.jit
    def _store_partials(self, shared: _Shared, products, warp: Int32, lane: Int32):
        """Write this lane's partials over the tile, which the product no longer reads."""
        columns = self.padded_columns
        split = lane % KEY_SPLITS
        row_group = lane // KEY_SPLITS
        for i in cutlass.range_constexpr(self.rows_per_thread):
            row = self._tile_row(warp, row_group, i)
            for vector in cutlass.range_constexpr(self.column_vectors):
                n = 4 * vector
                offset = self._partial_offset(split, row, n)
                _sts_f32x4(
                    shared.tile_addr + offset * 4,
                    [products[i * columns + n + r] for r in range(4)],
                )

    @cute.jit
    def _partial_sum(self, shared: _Shared, row, column):
        total = Float32(0.0)
        for split in cutlass.range_constexpr(KEY_SPLITS):
            total = total + shared.tile[self._partial_offset(split, row, column)]
        return total

    @cute.jit
    def _factored_sums(self, shared: _Shared, tidx: Int32):
        """Norms and pair sums from the products of the extra rows k_j e^-G_j and q_j e^-G_j."""
        W = self.window
        table = self.sums
        key_rows = self.value_tile_rows
        query_rows = self.value_tile_rows + W
        for base in cutlass.range_constexpr(0, W * W, THREADS):
            entry = tidx + base
            if entry < W * W:
                t = entry // W
                j = entry % W
                if j <= t:
                    key_pair = self._partial_sum(shared, key_rows + j, t)
                    shared.sums[table.query_key_pairs + entry] = self._partial_sum(
                        shared, key_rows + j, W + t
                    )
                    if j < t:
                        shared.sums[table.key_pairs + entry] = key_pair
                    else:
                        shared.sums[table.key_norm + t] = key_pair
                        shared.sums[table.query_norm + t] = self._partial_sum(
                            shared, query_rows + t, W + t
                        )

    @cute.jit
    def _finalize_coefficients(self, sums: cute.Tensor, tidx: Int32):
        """rk_t, rq_t, a_tj and b_tj from the raw norms and pair sums."""
        W = self.window
        table = self.sums
        for base in cutlass.range_constexpr(0, W * W, THREADS):
            entry = tidx + base
            if entry < W * W:
                t = entry // W
                j = entry % W
                key_scale_t = _rsqrt(sums[table.key_norm + t] + Float32(1e-6))
                key_scale_j = _rsqrt(sums[table.key_norm + j] + Float32(1e-6))
                query_scale_t = _rsqrt(
                    sums[table.query_norm + t] + Float32(1e-6)
                ) * Float32(self.key_dim**-0.5)
                if j < t:
                    sums[table.correction_coeffs + entry] = (
                        key_scale_t * key_scale_j * sums[table.key_pairs + entry]
                    )
                if j <= t:
                    sums[table.output_coeffs + entry] = (
                        query_scale_t
                        * key_scale_j
                        * sums[table.query_key_pairs + entry]
                    )
                if j == 0:
                    sums[table.key_scale + t] = key_scale_t
                    sums[table.query_scale + t] = query_scale_t

    @cute.jit
    def _solve_window(self, shared: _Shared, channel: Int32):
        """Corrections c_t and outputs out_t of one value channel, by forward substitution."""
        W = self.window
        table = self.sums
        # P[channel, :]: the checkpoint seen through every decayed key and query.
        projections = [Float32(0.0) for _ in range(self.padded_columns)]
        for split in cutlass.range_constexpr(KEY_SPLITS):
            for vector in cutlass.range_constexpr(self.column_vectors):
                offset = self._partial_offset(split, channel, 4 * vector)
                partial = _lds_f32x4(shared.tile_addr + offset * 4)
                for r in cutlass.range_constexpr(4):
                    projections[4 * vector + r] = (
                        projections[4 * vector + r] + partial[r]
                    )
        # Every load precedes the solve so the dependent chain runs without stalls.
        values = []
        key_scales = []
        query_scales = []
        betas = []
        correction_coeffs = []
        output_coeffs = []
        for t in cutlass.range_constexpr(W):
            values.append(
                shared.tokens[self.token_v + t * self.value_tile_rows + channel].to(
                    Float32
                )
            )
            key_scales.append(shared.sums[table.key_scale + t])
            query_scales.append(shared.sums[table.query_scale + t])
            betas.append(shared.sums[table.beta + t])
            correction_coeffs.append(
                [shared.sums[table.correction_coeffs + t * W + j] for j in range(t)]
            )
            output_coeffs.append(
                [shared.sums[table.output_coeffs + t * W + j] for j in range(t + 1)]
            )
        corrections = []
        outputs = []
        for t in cutlass.range_constexpr(W):
            residual = values[t] - key_scales[t] * projections[t]
            for j in cutlass.range_constexpr(t):
                residual = residual - correction_coeffs[t][j] * corrections[j]
            corrections.append(residual * betas[t])
            output = query_scales[t] * projections[W + t]
            for j in cutlass.range_constexpr(t + 1):
                output = output + output_coeffs[t][j] * corrections[j]
            outputs.append(output)
        return corrections, outputs


@functools.cache
def _compile(
    key_dim: int,
    value_dim: int,
    heads: int,
    window: int,
    value_tile_rows: int,
    capacity: int,
    bounded_gate: bool,
    factored: bool,
    has_metadata: bool,
    beta_dtype: torch.dtype,
    param_dtype: torch.dtype,
    bias_dtype: torch.dtype,
    start_dtype: torch.dtype,
    index_dtype: torch.dtype,
):
    tokens = cute.sym_int()
    slots = cute.sym_int()
    batch = cute.sym_int()

    # Separate descriptors keep each token-row tensor's stride independent.
    def token_rows(dim):
        return make_fake_tensor(BFloat16, (tokens, heads, dim), divisibility=8)

    q = token_rows(key_dim)
    k = token_rows(key_dim)
    v = token_rows(value_dim)
    gate = token_rows(key_dim)
    out = token_rows(value_dim)
    beta = make_fake_tensor(_DTYPES[beta_dtype], (tokens, heads))
    a_log = make_fake_tensor(_DTYPES[param_dtype], (heads,))
    dt_bias = make_fake_tensor(_DTYPES[bias_dtype], (heads * key_dim,))
    state = make_fake_tensor(
        Float32, (cute.sym_int(), heads, value_dim, key_dim), divisibility=4
    )
    correction = make_fake_tensor(Float32, (slots, heads, capacity, value_dim))
    key_gate = make_fake_tensor(BFloat16, (slots, heads, capacity, 2 * key_dim))
    query_start_loc = make_fake_tensor(_DTYPES[start_dtype], (cute.sym_int(),))
    state_indices = make_fake_tensor(_DTYPES[index_dtype], (batch,))
    metadata = stamps = None
    if has_metadata:
        metadata = make_fake_tensor(Int64, (batch, 3))
        stamps = make_fake_tensor(Int64, (slots, capacity, 2))
    kernel = RecoverSsmVerify(
        key_dim,
        value_dim,
        window,
        value_tile_rows,
        capacity,
        bounded_gate,
        factored,
        has_metadata,
    )
    return cute.compile(
        kernel,
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        dt_bias,
        state,
        correction,
        key_gate,
        out,
        query_start_loc,
        state_indices,
        metadata,
        stamps,
        Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _value_tile_rows(batch: int, heads: int, window: int, value_dim: int) -> int:
    """GB300 choice: small grids want more CTAs, large ones less repeated token work."""
    pairs = batch * heads
    if window > 8:
        rows = 32
    elif pairs <= 36:
        rows = 32
    elif pairs <= 192:
        rows = 64
    else:
        rows = 128
    while value_dim % rows:
        rows //= 2
    return rows


def recoverssm_verify_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    kg_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    lower_bound: float | None,
    out: torch.Tensor,
) -> bool:
    """Whether this kernel can run inputs that already pass the shared validation.

    It checks only what this implementation adds: dtypes, shape limits, the
    gate's sign, and the alignment and strides its bulk copies and fixed
    descriptors need.
    """
    key_dim, value_dim = q.shape[-1], v.shape[-1]
    floats = (torch.bfloat16, torch.float16, torch.float32)
    rows = (q, k, v, raw_g, out)
    return (
        all(x.dtype == torch.bfloat16 for x in (*rows, kg_cache))
        and checkpoint_state.dtype == correction_cache.dtype == torch.float32
        and all(x.dtype in floats for x in (raw_beta, A_log, dt_bias))
        and A_log.stride(0) == 1
        and key_dim % 32 == 0
        and key_dim <= THREADS
        and value_dim % 32 == 0
        and 1 <= spec_query_len <= 16
        # Cumulative gates assume decay; a positive bound would overflow e^G.
        and (lower_bound is None or lower_bound <= 0)
        # Bulk copies need 16-byte aligned token rows and dense state tiles.
        and all(x.stride(1) % 8 == 0 and x.data_ptr() % 16 == 0 for x in rows)
        and checkpoint_state.stride()[2:] == (key_dim, 1)
        and checkpoint_state.stride(0) % 4 == 0
        and checkpoint_state.stride(1) % 4 == 0
        and checkpoint_state.data_ptr() % 16 == 0
        and all(
            x.dtype in (torch.int32, torch.int64) and x.stride(0) == 1
            for x in (query_start_loc, state_indices)
        )
        and correction_cache.stride(3) == 1
        and kg_cache.stride(3) == 1
    )


def recoverssm_verify(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    checkpoint_state: torch.Tensor,
    correction_cache: torch.Tensor,
    kg_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    out: torch.Tensor,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
) -> torch.Tensor:
    """Run the CuTe DSL verify; arguments and results match ``kda_recoverssm_verify``."""
    check_recoverssm_verify_inputs(
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        checkpoint_state,
        correction_cache,
        kg_cache,
        query_start_loc,
        state_indices,
        spec_query_len,
        out,
        record_metadata,
        record_stamps,
    )
    _, total_tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if total_tokens == 0:
        return out
    if not recoverssm_verify_supported(
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        checkpoint_state,
        correction_cache,
        kg_cache,
        query_start_loc,
        state_indices,
        spec_query_len,
        lower_bound,
        out,
    ):
        raise ValueError("CuTe DSL KDA verify cannot address these inputs")
    bounded = lower_bound is not None
    kernel = _compile(
        key_dim,
        value_dim,
        heads,
        spec_query_len,
        _value_tile_rows(state_indices.shape[0], heads, spec_query_len, value_dim),
        correction_cache.shape[2],
        bounded,
        bounded and spec_query_len * abs(lower_bound) <= FACTORED_DECAY,
        record_metadata is not None,
        raw_beta.dtype,
        A_log.dtype,
        dt_bias.dtype,
        query_start_loc.dtype,
        state_indices.dtype,
    )
    kernel(
        q[0],
        k[0],
        v[0],
        raw_g[0],
        raw_beta[0],
        A_log,
        dt_bias.reshape(-1),
        checkpoint_state,
        correction_cache,
        kg_cache,
        out[0],
        query_start_loc,
        state_indices,
        record_metadata,
        record_stamps,
        float(lower_bound) if bounded else 0.0,
    )
    return out
