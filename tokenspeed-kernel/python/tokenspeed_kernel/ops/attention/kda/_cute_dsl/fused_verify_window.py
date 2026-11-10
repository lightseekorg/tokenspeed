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

"""Multi-token KDA verify with its producers and output norm, in CuTe DSL.

Contract: the split producers (the gate projection ``f_a @ f_b.T`` and
``kda_verify_conv_update``) followed by ``kda_recoverssm_verify`` for windows
of 2 to 16 tokens, in one launch, and with the layer's output norm the gated
RMSNorm after it. Outputs and records match theirs, the pre-convolution rows
are retained for commit, and the checkpoint and the convolution state are
read only.

The window math and its shared-memory layout are RecoverSsmVerify's: one CTA
per (value tile, request, head). The C = 128 / VT tiles of a head form a
cluster that shares the producers. CTA r convolves q and k channels
[r VT, (r + 1) VT) and its own v tile, projects the gate for keys
[r VT, (r + 1) VT), and stores the BF16 rows, rounded as the split producers
store them, into the token rows of every CTA of the cluster. With the output
norm, each warp of 32 value channels stores its tokens' sums of squares to
every CTA of the cluster, and each CTA adds the head's four sums in order,
as the one-token kernel does.

Before the programmatic dependency wait a CTA reads only what the previous
launch does not write: the batch layout, its checkpoint tile (bulk copy),
the convolution taps and history, and the gate and norm parameters. The
projection rows are read after it.
"""

from __future__ import annotations

import functools

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, const_expr, cute
from cutlass.cutlass_dsl import dsl_user_op
from quack.compile_utils import make_fake_tensor
from tokenspeed_kernel.ops.attention.kda import KdaOutputNorm
from tokenspeed_kernel.ops.attention.kda._cute_dsl.fused_verify_step import (
    _DTYPES,
    CONV_TAPS,
    _bf16,
    _check_output_norm,
    _check_records,
    _ldg_bf16x4,
    _prefetch_l2,
    _silu,
    _transpose_sum,
    fused_verify_step_supported,
)
from tokenspeed_kernel.ops.attention.kda._cute_dsl.recoverssm_verify import (
    FACTORED_DECAY,
    THREADS,
    RecoverSsmVerify,
    _asm,
    _bulk_load,
    _div,
    _exp,
    _load,
    _round_up,
    _rsqrt,
    _store_if,
    _value_tile_rows,
)

__all__ = ["fused_verify_window", "fused_verify_window_supported"]

DIM = 128
GATE_RANK = 128
MAX_WINDOW = 16
# The head's value channels in groups of one warp; the norm adds their sums in order.
NORM_GROUPS = DIM // 32


@dsl_user_op
def _store_shared_bf16(byte_addr: Int32, value: Float32, *, loc=None, ip=None):
    """Store an FP32 value that is already a BF16 number as BF16 in shared memory."""
    _asm(
        None,
        [Int32(byte_addr), Float32(value)],
        "{ .reg .b16 x; cvt.rn.bf16.f32 x, $1; st.shared.b16 [$0], x; }",
        "r,f,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _store_peer_bf16(
    byte_addr: Int32, peer: Int32, value: Float32, *, loc=None, ip=None
):
    """Store a BF16 number at this shared-memory address of cluster CTA ``peer``."""
    _asm(
        None,
        [Int32(byte_addr), Int32(peer), Float32(value)],
        "{ .reg .b32 a; .reg .b16 x; mapa.shared::cluster.u32 a, $0, $1; "
        "cvt.rn.bf16.f32 x, $2; st.shared::cluster.b16 [a], x; }",
        "r,r,f,~{memory}",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _store_shared_f32(byte_addr: Int32, value: Float32, *, loc=None, ip=None):
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
def _store_peer_f32(
    byte_addr: Int32, peer: Int32, value: Float32, *, loc=None, ip=None
):
    """Store an FP32 value at this shared-memory address of cluster CTA ``peer``."""
    _asm(
        None,
        [Int32(byte_addr), Int32(peer), Float32(value)],
        "{ .reg .b32 a; mapa.shared::cluster.u32 a, $0, $1; "
        "st.shared::cluster.f32 [a], $2; }",
        "r,r,f,~{memory}",
        True,
        loc,
        ip,
    )


class FusedVerifyWindow(RecoverSsmVerify):
    """Multi-token verify with fused producers for one head count, window and value tile."""

    def __init__(
        self,
        heads: int,
        window: int,
        value_tile_rows: int,
        capacity: int,
        bounded_gate: bool,
        factored: bool,
        has_metadata: bool,
        gated_norm: bool,
    ) -> None:
        super().__init__(
            DIM,
            DIM,
            window,
            value_tile_rows,
            capacity,
            bounded_gate,
            factored,
            has_metadata,
        )
        self.heads = heads
        self.width = heads * DIM
        self.cluster = DIM // value_tile_rows
        self.gated_norm = gated_norm
        # Channel c < 3 VT of a CTA: q, k, then v channels value_start + c % VT.
        self.conv_items = (3 * value_tile_rows + THREADS - 1) // THREADS
        if gated_norm:
            self.norm_words = _round_up(NORM_GROUPS * window, 4)
            self.smem_bytes += 4 * self.norm_words + 16

    @cute.jit
    def _publish_bf16(self, byte_addr: Int32, value: Float32):
        """Store a BF16 number at this shared-memory address of every CTA in the cluster."""
        if const_expr(self.cluster == 1):
            _store_shared_bf16(byte_addr, value)
        else:
            for peer in cutlass.range_constexpr(self.cluster):
                _store_peer_bf16(byte_addr, Int32(peer), value)

    @cute.jit
    def _publish_f32(self, byte_addr: Int32, value: Float32):
        """Store an FP32 value at this shared-memory address of every CTA in the cluster."""
        if const_expr(self.cluster == 1):
            _store_shared_f32(byte_addr, value)
        else:
            for peer in cutlass.range_constexpr(self.cluster):
                _store_peer_f32(byte_addr, Int32(peer), value)

    @cute.jit
    def _cluster_sync(self):
        """Make every CTA's shared-memory stores so far visible to the whole cluster."""
        if const_expr(self.cluster == 1):
            cute.arch.barrier()
        else:
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

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
        correction: cute.Tensor,
        key_gate: cute.Tensor,
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
            correction.iterator,
            key_gate.iterator,
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
            key_gate.stride[0],
            key_gate.stride[1],
            key_gate.stride[2],
            original.stride[0],
            out.stride[0],
            gate_token_stride,
        ).launch(
            grid=[self.cluster, state_indices.shape[0], self.heads],
            block=[THREADS, 1, 1],
            cluster=[self.cluster, 1, 1],
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
        correction,
        key_gate,
        original,
        out,
        query_start_loc,
        state_indices,
        metadata,
        stamps,
        gate,
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
        key_gate_slot_stride: Int64,
        key_gate_head_stride: Int64,
        key_gate_position_stride: Int64,
        original_token_stride: Int64,
        out_token_stride: Int64,
        gate_token_stride: Int64,
    ):
        K = self.key_dim
        V = self.value_dim
        W = self.window
        VT = self.value_tile_rows
        P = self.width
        tidx, _, _ = cute.arch.thread_idx()
        value_tile, request, head = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        shared = self._allocate_shared()
        value_start = value_tile * VT
        if const_expr(self.cluster > 1):
            # Pairs with the wait before the first store into a peer.
            cute.arch.cluster_arrive_relaxed()

        # The batch layout, the checkpoint, the convolution history and the
        # weights were written before this forward, not by the previous launch.
        token_start = Int64(_load(query_start_loc, request))
        valid_tokens = Int32(Int64(_load(query_start_loc, request + 1)) - token_start)
        page = Int64(_load(state_indices, request))
        live = (page > 0) & (valid_tokens > 0)
        if tidx == 0:
            cute.arch.mbarrier_init(shared.state_barrier, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        if live:
            if tidx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    shared.state_barrier, VT * K * 4
                )
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
                f_b_slice = f_b + (head * K + value_start) * GATE_RANK
                _prefetch_l2(f_b_slice.toint(), VT * GATE_RANK * 2)

        channels = []
        taps = []
        pasts = []
        history = conv_states + page * conv_page_stride
        for item in cutlass.range_constexpr(self.conv_items):
            c = cutlass.min(item * THREADS + tidx, 3 * VT - 1)
            kind = c // VT
            channel = kind * P + head * K + value_start + c % VT
            channels.append(channel)
            taps.append(_ldg_bf16x4((conv_weights + channel * CONV_TAPS).toint()))
            # Padding pages see a zero history, as the split producer's do.
            for i in cutlass.range_constexpr(CONV_TAPS - 1):
                past = Float32(0.0)
                if (valid_tokens > 0) & (page >= 0):
                    past = _load(history, channel * 3 + i).to(Float32)
                pasts.append(past)
        gate_scale = _exp(_load(a_log, head).to(Float32))
        gate_bias = _load(dt_bias, head * K + tidx).to(Float32)
        channel = cutlass.min(tidx, VT - 1)
        norm_scale = Float32(0.0)
        if const_expr(self.gated_norm):
            norm_scale = _load(norm_weight, value_start + channel).to(Float32)

        cute.arch.griddepcontrol_wait()
        if const_expr(self.cluster > 1):
            cute.arch.cluster_wait()
        cute.arch.griddepcontrol_launch_dependents()

        if valid_tokens > 0:
            self._convolve(
                shared,
                qkv,
                original,
                qkv_token_stride,
                original_token_stride,
                token_start,
                valid_tokens,
                live,
                tidx,
                value_start,
                channels,
                taps,
                pasts,
            )
            if live:
                self._project_gate(
                    shared,
                    f_a,
                    f_b,
                    f_a_token_stride,
                    token_start,
                    valid_tokens,
                    head,
                    value_start,
                    warp,
                    lane,
                )

        if live:
            self._cluster_sync()
            slot = Int64(request)
            generation = Int64(0)
            start = Int64(0)
            if const_expr(self.has_metadata):
                slot = Int64(_load(metadata, request * 3))
                generation = Int64(_load(metadata, request * 3 + 1))
                start = Int64(_load(metadata, request * 3 + 2))
            # Reduce the 64-bit start before narrowing so ring positions stay exact.
            ring_start = Int32(start % self.capacity)
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

            # Phase 3, one thread per value channel; the surplus warps repeat
            # the last channel so the norm's warp sums see full warps.
            corrections, outputs = self._solve_window(shared, channel)
            owner = Int32(tidx < VT)
            if const_expr(self.gated_norm):
                outputs = self._normalize(
                    shared,
                    outputs,
                    gate,
                    gate_token_stride,
                    token_start,
                    valid_tokens,
                    head,
                    value_start + channel,
                    value_tile,
                    warp,
                    lane,
                    owner,
                    norm_scale,
                    norm_eps,
                )
            out_row = out + (
                token_start * out_token_stride + head * V + value_start + channel
            )
            correction_row = correction + (
                slot * correction_slot_stride
                + head * correction_head_stride
                + value_start
                + channel
            )
            for t in cutlass.range_constexpr(W):
                stored = owner & Int32(t < valid_tokens)
                position = (ring_start + t) % self.capacity
                _store_if(
                    stored, (out_row + t * out_token_stride).toint(), outputs[t], True
                )
                _store_if(
                    stored,
                    (correction_row + position * correction_position_stride).toint(),
                    corrections[t],
                    False,
                )
        else:
            # Padding requests: zero outputs for their tokens, no records.
            if tidx < VT:
                for t in cutlass.range_constexpr(W):
                    row = out + (
                        (token_start + t) * out_token_stride
                        + head * V
                        + value_start
                        + tidx
                    )
                    _store_if(Int32(t < valid_tokens), row.toint(), Float32(0.0), True)

    @cute.jit
    def _convolve(
        self,
        shared,
        qkv,
        original,
        qkv_token_stride: Int64,
        original_token_stride: Int64,
        token_start: Int64,
        valid_tokens: Int32,
        live,
        tidx: Int32,
        value_start: Int32,
        channels,
        taps,
        pasts,
    ):
        """Retain this CTA's pre-convolution channels and publish their convolved rows.

        The q and k channels go to every CTA of the cluster, the v tile to this
        one; the window's last valid row stands in for the rows past it.
        """
        K = self.key_dim
        W = self.window
        VT = self.value_tile_rows
        for item in cutlass.range_constexpr(self.conv_items):
            c = item * THREADS + tidx
            in_range = Int32(c < 3 * VT)
            kind = cutlass.min(c, 3 * VT - 1) // VT
            within = cutlass.min(c, 3 * VT - 1) % VT
            raws = []
            for t in cutlass.range_constexpr(W):
                token = token_start + cutlass.min(t, valid_tokens - 1)
                raws.append(
                    _load(qkv, token * qkv_token_stride + channels[item]).to(Float32)
                )
            kept = original + (token_start * original_token_stride + channels[item])
            for t in cutlass.range_constexpr(W):
                _store_if(
                    in_range & Int32(t < valid_tokens),
                    (kept + t * original_token_stride).toint(),
                    raws[t],
                    True,
                )
            if live & (in_range != 0):
                for t in cutlass.range_constexpr(W):
                    # Taps t .. t + 3 of the history followed by the window.
                    acc = Float32(0.0)
                    for i in cutlass.range_constexpr(CONV_TAPS):
                        if const_expr(t + i < CONV_TAPS - 1):
                            term = pasts[item * (CONV_TAPS - 1) + t + i]
                        else:
                            term = raws[t + i - (CONV_TAPS - 1)]
                        acc = acc + term * taps[item][i]
                    convolved = _bf16(_silu(acc))
                    if kind < 2:
                        offset = kind * W * K + t * K + value_start + within
                        self._publish_bf16(shared.tokens_addr + offset * 2, convolved)
                    else:
                        offset = self.token_v + t * VT + within
                        _store_shared_bf16(shared.tokens_addr + offset * 2, convolved)

    @cute.jit
    def _project_gate(
        self,
        shared,
        f_a,
        f_b,
        f_a_token_stride: Int64,
        token_start: Int64,
        valid_tokens: Int32,
        head: Int32,
        value_start: Int32,
        warp: Int32,
        lane: Int32,
    ):
        """Project the raw gate of this CTA's keys and publish it to the cluster.

        An item is 32 keys and two tokens; warp w takes items w, w + 4, ...
        and lane l elements [4l, 4l + 4) of each f_a and f_b row, so a
        transposing butterfly leaves key 32 g + l in lane l, as in the
        one-token kernel.
        """
        K = self.key_dim
        W = self.window
        pairs = (W + 1) // 2
        items = (self.value_tile_rows // 32) * pairs
        for step in cutlass.range_constexpr((items + 3) // 4):
            item = step * 4 + warp
            if item < items:
                group = item // pairs
                first = 2 * (item % pairs)
                rows = []
                for r in cutlass.range_constexpr(2):
                    token = token_start + cutlass.min(first + r, valid_tokens - 1)
                    rows.append(
                        _ldg_bf16x4((f_a + token * f_a_token_stride + 4 * lane).toint())
                    )
                weights = f_b + (head * K + value_start + 32 * group) * GATE_RANK
                parts = [[], []]
                for j in cutlass.range_constexpr(32):
                    w = _ldg_bf16x4((weights + j * GATE_RANK + 4 * lane).toint())
                    for r in cutlass.range_constexpr(2):
                        part = Float32(0.0)
                        for e in cutlass.range_constexpr(4):
                            part = part + rows[r][e] * w[e]
                        parts[r].append(part)
                key = value_start + 32 * group + lane
                for r in cutlass.range_constexpr(2):
                    raw_gate = _bf16(_transpose_sum(parts[r], lane))
                    t = first + r
                    if t < W:
                        offset = self.token_gate + t * K + key
                        self._publish_bf16(shared.tokens_addr + offset * 2, raw_gate)

    @cute.jit
    def _normalize(
        self,
        shared,
        outputs,
        gate,
        gate_token_stride: Int64,
        token_start: Int64,
        valid_tokens: Int32,
        head: Int32,
        channel: Int32,
        value_tile: Int32,
        warp: Int32,
        lane: Int32,
        owner: Int32,
        norm_scale: Float32,
        norm_eps: Float32,
    ):
        """The norm kernel's x rsqrt(mean x^2 + eps) w sigmoid(gate) on the BF16 outputs."""
        W = self.window
        V = self.value_dim
        logits = []
        for t in cutlass.range_constexpr(W):
            token = token_start + cutlass.min(t, valid_tokens - 1)
            logits.append(
                _load(gate, token * gate_token_stride + head * V + channel).to(Float32)
            )
        norm_addr = Int32(shared.norm.iterator.toint())
        group = value_tile * (self.value_tile_rows // 32) + warp
        rounded = []
        for t in cutlass.range_constexpr(W):
            output = _bf16(outputs[t])
            rounded.append(output)
            square = cute.arch.warp_reduction_sum(output * output)
            if (lane == 0) & (owner != 0):
                self._publish_f32(norm_addr + (group * W + t) * 4, square)
        self._cluster_sync()
        normed = []
        for t in cutlass.range_constexpr(W):
            total = shared.norm[t]
            for g in cutlass.range_constexpr(1, NORM_GROUPS):
                total = total + shared.norm[g * W + t]
            inverse_rms = _rsqrt(_div(total, Float32(V)) + norm_eps)
            sigmoid = _div(Float32(1.0), Float32(1.0) + _exp(-logits[t]))
            normed.append(rounded[t] * inverse_rms * norm_scale * sigmoid)
        return normed


@functools.cache
def _compile(
    heads: int,
    window: int,
    value_tile_rows: int,
    capacity: int,
    bounded_gate: bool,
    factored: bool,
    has_metadata: bool,
    norm_dtype: torch.dtype | None,
    beta_dtype: torch.dtype,
    param_dtype: torch.dtype,
    bias_dtype: torch.dtype,
    start_dtype: torch.dtype,
    index_dtype: torch.dtype,
):
    tokens = cute.sym_int()
    slots = cute.sym_int()
    batch = cute.sym_int()
    width = heads * DIM
    qkv = make_fake_tensor(BFloat16, (tokens, 3 * width), divisibility=8)
    conv_weights = make_fake_tensor(BFloat16, (3 * width, 4), divisibility=4)
    conv_states = make_fake_tensor(
        BFloat16, (cute.sym_int(), cute.sym_int(), 3), divisibility=1
    )
    f_a = make_fake_tensor(BFloat16, (tokens, GATE_RANK), divisibility=8)
    f_b = make_fake_tensor(BFloat16, (width, GATE_RANK), divisibility=8)
    beta = make_fake_tensor(_DTYPES[beta_dtype], (tokens, heads))
    a_log = make_fake_tensor(_DTYPES[param_dtype], (heads,))
    dt_bias = make_fake_tensor(_DTYPES[bias_dtype], (heads * DIM,))
    state = make_fake_tensor(Float32, (cute.sym_int(), heads, DIM, DIM), divisibility=4)
    correction = make_fake_tensor(Float32, (slots, heads, capacity, DIM))
    key_gate = make_fake_tensor(BFloat16, (slots, heads, capacity, 2 * DIM))
    original = make_fake_tensor(BFloat16, (tokens, 3 * width))
    out = make_fake_tensor(BFloat16, (tokens, width))
    query_start_loc = make_fake_tensor(_DTYPES[start_dtype], (cute.sym_int(),))
    state_indices = make_fake_tensor(_DTYPES[index_dtype], (batch,))
    metadata = stamps = None
    if has_metadata:
        metadata = make_fake_tensor(Int64, (batch, 3))
        stamps = make_fake_tensor(Int64, (slots, capacity, 2))
    gate = norm_weight = None
    if norm_dtype is not None:
        gate = make_fake_tensor(BFloat16, (tokens, width))
        norm_weight = make_fake_tensor(_DTYPES[norm_dtype], (DIM,))
    kernel = FusedVerifyWindow(
        heads,
        window,
        value_tile_rows,
        capacity,
        bounded_gate,
        factored,
        has_metadata,
        norm_dtype is not None,
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
        correction,
        key_gate,
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


def fused_verify_window_supported(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    window: int,
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
        checkpoint_state: ``[pages, num_heads, head_dim, head_dim]`` FP32
            checkpoint in value-major layout.
        query_start_loc: ``[batch + 1]`` token offset of each request.
        state_indices: ``[batch]`` checkpoint and convolution page of each
            request.
        num_heads: Per-rank head count.
        head_dim: Head width.
        window: Tokens per request window.
        lower_bound: Bounded-gate lower bound, or None for the softplus gate.

    Returns:
        True when the one-token kernel's conditions hold, the window has 2 to
        16 tokens and the batch layout is dense INT32 or INT64.
    """
    return (
        2 <= window <= MAX_WINDOW
        and fused_verify_step_supported(
            mixed_qkv,
            conv_weights,
            conv_states,
            f_a_out,
            f_b_weight,
            beta_logits,
            A_log,
            dt_bias,
            checkpoint_state,
            num_heads,
            head_dim,
            lower_bound,
        )
        and all(
            x.dtype in (torch.int32, torch.int64) and x.stride(0) == 1
            for x in (query_start_loc, state_indices)
        )
    )


def fused_verify_window(
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
    correction_cache: torch.Tensor,
    kg_cache: torch.Tensor,
    original_qkv: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    window: int,
    out: torch.Tensor,
    record_metadata: torch.Tensor | None,
    record_stamps: torch.Tensor | None,
    output_norm: KdaOutputNorm | None,
) -> torch.Tensor:
    """Run multi-token record verify with the gate and convolution producers fused.

    Args:
        mixed_qkv: ``[tokens, 3 * num_heads * head_dim]`` BF16 q|k|v rows
            before the convolution, request after request.
        conv_weights: ``[3 * num_heads * head_dim, 4]`` BF16 convolution taps.
        conv_states: ``[pages, channels, 3]`` BF16 committed convolution
            history; read only.
        f_a_out: ``[tokens, 128]`` BF16 gate projection input.
        f_b_weight: ``[num_heads * head_dim, 128]`` BF16 gate projection.
        beta_logits: ``[tokens, num_heads]`` beta logits.
        A_log: ``[num_heads]`` gate decay logarithms.
        dt_bias: ``[num_heads * head_dim]`` gate bias.
        lower_bound: Bounded-gate lower bound, or None for the softplus gate.
        checkpoint_state: ``[pages, num_heads, head_dim, head_dim]`` FP32
            committed checkpoint in value-major layout; read only.
        correction_cache: ``[slots, num_heads, capacity, head_dim]`` FP32
            correction records; written.
        kg_cache: ``[slots, num_heads, capacity, 2 * head_dim]`` BF16 key and
            raw-gate records; written.
        original_qkv: ``[tokens, 3 * num_heads * head_dim]`` BF16; receives
            the pre-convolution rows for commit.
        query_start_loc: ``[batch + 1]`` token offset of each request; a
            request has at most ``window`` tokens.
        state_indices: ``[batch]`` checkpoint and convolution page of each
            request. Pages at or below 0 are padding: zero outputs and no
            records, though their pre-convolution rows are still retained.
        num_heads: Per-rank head count.
        head_dim: Head width; 128.
        window: Tokens per request window, 2 to 16.
        out: ``[tokens, num_heads * head_dim]`` BF16 verify output, or with
            ``output_norm`` its gated RMSNorm; written.
        record_metadata: ``[batch, 3]`` int64 record slot, generation and
            start position per request, or None to record request r at slot
            r from position 0.
        record_stamps: ``[slots, capacity, 2]`` int64 start and generation of
            each record position, written with ``record_metadata``; else None.
        output_norm: The layer's gated output RMSNorm to apply before the
            store, or None to store the verify output.

    Returns:
        ``out``.

    Raises:
        ValueError: If ``fused_verify_window_supported`` rejects the inputs,
            the record metadata and stamps are not dense INT64 views of the
            record layout, or the output norm's gate or weight are laid out
            otherwise.
    """
    if mixed_qkv.shape[0] == 0:
        return out
    if not fused_verify_window_supported(
        mixed_qkv,
        conv_weights,
        conv_states,
        f_a_out,
        f_b_weight,
        beta_logits,
        A_log,
        dt_bias,
        checkpoint_state,
        query_start_loc,
        state_indices,
        num_heads,
        head_dim,
        window,
        lower_bound,
    ):
        raise ValueError("CuTe DSL fused KDA verify window cannot address these inputs")
    _check_records(
        record_metadata, record_stamps, state_indices.shape[0], correction_cache
    )
    _check_output_norm(output_norm, out, head_dim)
    bounded = lower_bound is not None
    kernel = _compile(
        num_heads,
        window,
        _value_tile_rows(state_indices.shape[0], num_heads, window, head_dim),
        correction_cache.shape[2],
        bounded,
        bounded and window * abs(lower_bound) <= FACTORED_DECAY,
        record_metadata is not None,
        None if output_norm is None else output_norm.weight.dtype,
        beta_logits.dtype,
        A_log.dtype,
        dt_bias.dtype,
        query_start_loc.dtype,
        state_indices.dtype,
    )
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
        correction_cache,
        kg_cache,
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
