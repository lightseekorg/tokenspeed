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

"""Single-launch native FP8 mixer for prequantized activation and weights."""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import cutlass.utils as cute_utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cuda.bindings.driver import CUstream
from cutlass._mlir.dialects import arith, llvm, nvvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import CopyUniversalOp, cpasync, tcgen05
from cutlass.cutlass_dsl import T
from cutlass.experimental.cuda import (
    TensorMap,
    TensorMapL2Promotion,
    TensorMapSwizzle,
    create_tensor_map_tiled,
)
from cutlass.experimental.primitives import tcgen05_fence
from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.common import (
    _acquire,
    _commit_residual,
    _copy_residual_16,
    _decode_residual,
    _load_f32,
    _make_down_operands,
    _mul_hidden_float_pair,
    _operand_at,
    _reduce_split_partials,
    _release,
    _store_private_epoch,
    _wait_residual,
    _x_scale,
)
from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.resident import (
    MixerOperands,
    QuantizationConfig,
    ResidentMixer,
    _mixer_cta,
    _mixer_shared,
    _quant_scale,
    _quantized_value,
)
from tokenspeed_kernel.thirdparty.cute_dsl.hc_fp8 import (
    Fp8GatedResidualGemm,
    _epilogue_barrier,
    _up_sigmoid_tanh,
)
from tokenspeed_kernel.thirdparty.cute_dsl.hc_fused import (
    _publish_activation_proxy,
)


@cute.experimental.jit
def _tanh_half(value):
    half = value * 0.5
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [half.ir_value()],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            asm_dialect=0,
        )
    )


@cute.experimental.jit
def _up_sigmoid_pair(value0, value1):
    tanh0, tanh1 = _tanh_half(value0), _tanh_half(value1)
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [tanh0.ir_value(), tanh1.ir_value()],
        "{ .reg .b64 x,h,y; mov.b64 x, {$2,$3}; mov.b64 h, 0x3f0000003f000000; "
        "fma.rn.f32x2 y,x,h,h; mov.b64 {$0,$1},y; }",
        "=f,=f,f,f",
        has_side_effects=False,
        asm_dialect=0,
    )
    return cutlass.Float32(llvm.extractvalue(T.f32(), result, [0])), cutlass.Float32(
        llvm.extractvalue(T.f32(), result, [1])
    )


@cute.experimental.jit
def _copy_entry_epoch(destination, source):
    # The private ordinal is 8-byte aligned. It is consumed only after the
    # first Down handoff has waited for this thread's asynchronous group.
    llvm.inline_asm(
        None,
        [cutlass.Int32(destination.toint()).ir_value(), source.toint().ir_value()],
        "cp.async.ca.shared.global [$0], [$1], 8;",
        "r,l,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _load_relaxed_counter(ptr):
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [ptr.toint().ir_value()],
            "ld.global.relaxed.gpu.u64 $0, [$1];",
            "=l,l,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )
    )


@cute.experimental.jit
def _store_activation_pair(destination, first, second):
    llvm.inline_asm(
        None,
        [destination.toint().ir_value(), first.ir_value(), second.ir_value()],
        "{ .reg .b16 packet; cvt.rn.satfinite.e4m3x2.f32 packet, $2, $1; "
        "st.global.b16 [$0], packet; }",
        "l,f,f,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _store_compact_activation(
    destination,
    first,
    second,
    ordinal,
    lane,
    first_plane,
    activation_scale,
    input_scale,
):
    # Each even lane forms four consecutive codes. Six lanes make one
    # record, so the last record of a 64-byte row plane has only four codes.
    packet = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 3),
        [first.ir_value(), second.ir_value(), lane.ir_value()],
        """{
        .reg .b16 pair;
        .reg .b32 lo, peer, middle, high;
        cvt.rn.satfinite.e4m3x2.f32 pair, $4, $3;
        cvt.u32.u16 lo, pair;
        shfl.sync.bfly.b32 peer, lo, 1, 31, 0xffffffff;
        shl.b32 peer, peer, 16;
        or.b32 lo, lo, peer;
        add.u32 middle, $5, 2;
        add.u32 high, $5, 4;
        min.u32 middle, middle, 30;
        min.u32 high, high, 30;
        mov.b32 $0, lo;
        shfl.sync.idx.b32 $1, lo, middle, 31, 0xffffffff;
        shfl.sync.idx.b32 $2, lo, high, 31, 0xffffffff;
        }""",
        "=r,=r,=r,f,f,r",
        has_side_effects=True,
        asm_dialect=0,
    )
    if lane % 6 == 0:
        code = cutlass.Uint32(llvm.extractvalue(T.i32(), packet, [0]))
        middle = cutlass.Uint32(llvm.extractvalue(T.i32(), packet, [1]))
        high = cutlass.Uint32(llvm.extractvalue(T.i32(), packet, [2]))
        # The final record of the first K64 plane has eight spare bytes.
        if first_plane and lane == 30:
            middle = cutlass.Uint32(llvm.bitcast(T.i32(), activation_scale.ir_value()))
            high = cutlass.Uint32(llvm.bitcast(T.i32(), input_scale.ir_value()))
        llvm.inline_asm(
            None,
            [
                destination.toint().ir_value(),
                code.ir_value(),
                middle.ir_value(),
                high.ir_value(),
                ordinal.to(cutlass.Uint32).ir_value(),
            ],
            "{ .reg .b128 record; mov.b128 record, {$1,$2,$3,$4}; "
            "st.global.relaxed.gpu.b128 [$0], record; }",
            "l,r,r,r,r,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )


@cute.experimental.jit
def _store_stamped_alpha(destination, value, ordinal):
    llvm.inline_asm(
        None,
        [destination.toint().ir_value(), value.ir_value(), ordinal.ir_value()],
        """{
        .reg .b32 bits;
        .reg .b64 payload;
        .reg .b128 word;
        mov.b32 bits, $1;
        cvt.u64.u32 payload, bits;
        mov.b128 word, {payload,$2};
        st.global.relaxed.gpu.b128 [$0], word;
        }""",
        "l,f,l,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _load_compact_activation(address):
    record = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.toint().ir_value()],
        "{ .reg .b128 record; ld.global.relaxed.gpu.b128 record, [$4]; "
        "mov.b128 {$0,$1,$2,$3}, record; }",
        "=r,=r,=r,=r,l,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )
    return (
        cutlass.Uint32(llvm.extractvalue(T.i32(), record, [0])),
        cutlass.Uint32(llvm.extractvalue(T.i32(), record, [1])),
        cutlass.Uint32(llvm.extractvalue(T.i32(), record, [2])),
        cutlass.Uint32(llvm.extractvalue(T.i32(), record, [3])),
    )


@cute.experimental.jit
def _wait_compact_activation(address, target):
    a, b, c, ordinal = _load_compact_activation(address)
    # A slot can only contain the current or preceding epoch. The initial
    # zero differs from the first epoch, and every epoch advances by at most
    # one resident grid for Down plus one for reuse.
    while ordinal != target:
        llvm.inline_asm(
            None,
            [],
            "nanosleep.u32 64;",
            "~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )
        a, b, c, ordinal = _load_compact_activation(address)
    return a, b, c


@cute.experimental.jit
def _write_compact_plane_word(arena, linear, bits):
    # SW64 operates above the four low byte bits. Each word is aligned and
    # stays in a single 16-byte swizzle segment.
    offset = linear ^ (((linear >> 7) & 3) << 4)
    llvm.inline_asm(
        None,
        [cutlass.Uint32((arena.iterator + offset).toint()).ir_value(), bits.ir_value()],
        "st.shared.b32 [$0], $1;",
        "r,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _load_private_epoch(ptr):
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [ptr.toint().ir_value()],
            "ld.global.cg.u64 $0, [$1];",
            "=l,l,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )
    )


@cute.experimental.jit
def _arrive_phase_release(ptr, asynchronous: cutlass.Constexpr):
    # Acquire polling observes every preceding release through the atomic RMW
    # chain. Arrival itself neither consumes peer data nor needs the old value.
    instruction = "red.async" if asynchronous else "red"
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value()],
        f"{instruction}.release.gpu.global.add.u64 [$0], 1;",
        "l,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _wait_activation(counter, expected):
    while _acquire(counter) < expected:
        pass


@cute.experimental.jit
def _wait_activation_with_delay(counter, expected, delay_ns: cutlass.Constexpr):
    while _acquire(counter) < expected:
        llvm.inline_asm(
            None,
            [],
            f"nanosleep.u32 {delay_ns};",
            "~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )


@cute.experimental.jit
def _arrive_activation_row(counter, increment):
    llvm.inline_asm(
        None,
        [counter.toint().ir_value(), increment.ir_value()],
        "red.async.release.gpu.global.add.u64 [$0], $1;",
        "l,l,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _wait_activation_rows(
    tags, expected, rows: cutlass.Constexpr, retry_ns: cutlass.Constexpr
):
    lane = cute.arch.lane_idx()
    ready = cutlass.Boolean(False)
    while not ready:
        local_ready = cutlass.Boolean(True)
        if lane < rows:
            local_ready = _acquire(tags + cutlass.Int64(lane)) >= expected
        ready = cute.arch.vote_all_sync(local_ready)
        if cutlass.const_expr(retry_ns > 0):
            if not ready:
                llvm.inline_asm(
                    None,
                    [],
                    f"nanosleep.u32 {retry_ns};",
                    "~{memory}",
                    has_side_effects=True,
                    asm_dialect=0,
                )
    # The TMA is issued by one lane after all row acquires.
    cute.arch.sync_warp()


@cute.experimental.jit
def _grid_barrier(epochs, expected, wait_for_all, async_arrival: cutlass.Constexpr):
    # Cooperative launch and measured capacity keep every participant resident.
    cute.arch.sync_threads()
    if cute.arch.thread_idx()[0] == 0:
        _arrive_phase_release(epochs.iterator, async_arrival)
        if wait_for_all:
            while _acquire(epochs.iterator) < expected:
                pass
    cute.arch.sync_threads()


@cute.experimental.jit
def _down_weight_cache_policy():
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            asm_dialect=0,
        )
    )


@cute.experimental.jit
def _activation_bulk_copy(destination, source, full, size: cutlass.Constexpr):
    llvm.inline_asm(
        None,
        [
            cutlass.Uint32(destination.toint()).ir_value(),
            source.toint().ir_value(),
            cutlass.Uint32(size).ir_value(),
            cutlass.Uint32(full.toint()).ir_value(),
        ],
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.experimental.jit
def _copy_activation_bytes(destination, source, full, size: cutlass.Constexpr):
    with cute.arch.elect_one():
        _activation_bulk_copy(destination, source, full, size)


class _DownPhase(Fp8GatedResidualGemm):
    def __init__(
        self,
        *,
        producer_warp,
        mma_warp,
        use_cta_reader_handoff,
        static_partial_store,
        evict_weights_last,
        omit_down_empty,
        scratch_token_tile,
        entry_input_ring_first,
        aligned_partial_rows,
        input_refill_hint,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.producer_warp = producer_warp
        self.mma_warp = mma_warp
        self.use_cta_reader_handoff = use_cta_reader_handoff
        self.static_partial_store = static_partial_store
        self.aligned_partial_rows = aligned_partial_rows
        self.scratch_token_tile = scratch_token_tile
        self.entry_input_ring_first = entry_input_ring_first
        self.token_parts = scratch_token_tile // self.n
        self.evict_weights_last = evict_weights_last
        if omit_down_empty and self.tiles != self.stages:
            raise ValueError("Omitting Down empty events requires a resident split")
        self.omit_down_empty = omit_down_empty
        self.empty_stages = 0 if omit_down_empty else self.stages
        self.input_refill_hint = input_refill_hint

    @cute.experimental.jit
    def _partial_tile_origin(self, output_tile, rank):
        # Full projection tiles share one pitch. The injection values follow
        # them in the same allocation, with a separate four-element pitch.
        row_count = self.scratch_token_tile
        tile_elements = row_count * self.split_k * self.m
        activation_origin = cutlass.Int64(output_tile) * tile_elements + rank * self.m
        injection_width = self.projection_rows - self.lowrank
        injection_origin = (
            cutlass.Int64(row_count * self.split_k * self.lowrank)
            + rank * injection_width
        )
        injection_tile = output_tile == self.lowrank // self.m
        origin = cutlass.Int64(
            arith.select(
                injection_tile.ir_value(),
                injection_origin.ir_value(),
                activation_origin.ir_value(),
            )
        )
        activation_stride = cutlass.Int32(self.split_k * self.m)
        injection_stride = cutlass.Int32(self.split_k * injection_width)
        row_stride = cutlass.Int32(
            arith.select(
                injection_tile.ir_value(),
                injection_stride.ir_value(),
                activation_stride.ir_value(),
            )
        )
        return origin, row_stride

    @cute.experimental.jit
    def _hint_refill_input(self, descriptor, output_tile, split, token_tile):
        # All projection tiles for this token tile and split use these bytes.
        # One producer issues the hint. The demand copies retain their waits.
        row = cutlass.Int64(token_tile) * self.n
        if output_tile == 0 and row < cutlass.Int64(1 << 31):
            with cute.arch.elect_one():
                for index in cutlass.range_constexpr(len(self.input_refill_hint)):
                    tile = self.input_refill_hint[index]
                    k_segment = (split * self.tiles + tile) * (self.k // 64)
                    nvvm.cp_async_bulk_tensor_prefetch(
                        descriptor.get_ptr().ir_value(),
                        [
                            cutlass.Int32(0).ir_value(),
                            cutlass.Int32(row).ir_value(),
                            cutlass.Int32(k_segment).ir_value(),
                        ],
                        [],
                    )

    @cute.experimental.jit
    def _weight_view(self, weight):
        return cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (self.projection_rows, (64, self.inner // 64), 1),
                stride=(self.inner, (1, 64), 0),
            ),
        )

    @cute.experimental.jit
    def make_weight_atom(self, weight):
        dtype = cutlass.Float8E4M3FN
        tiler = (self.m, self.n, self.k)
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            (self.m, self.n),
        )
        layout = sm100_utils.make_smem_layout_a(mma, tiler, dtype, self.stages)
        return cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            self._weight_view(weight),
            cute.slice_(layout, (None, None, None, 0)),
            tiler,
            mma,
        )

    @cute.experimental.jit
    def _partition_weight(self, atom, coordinates, sw, output_tile):
        tiles = cute.local_tile(coordinates, (self.m, self.k), (output_tile, None, 0))
        return cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sw, 0, 3),
            cute.group_modes(tiles, 0, 2),
        )

    @cute.experimental.jit
    def _load_weight(
        self,
        gw,
        sw,
        full,
        stage,
        input_tile,
        map_w,
        atom,
        source,
        target,
        cache_policy,
    ):
        cute.copy(
            atom,
            source[None, input_tile],
            target[None, stage],
            tma_bar_ptr=full + stage,
            cache_policy=cache_policy,
        )

    @cute.experimental.jit
    def project(
        self,
        x,
        weight,
        partials,
        arena,
        barriers,
        tmem_base,
        output_tile,
        split,
        token_tile,
        entry_prefilled,
        weight_atom,
        weight_coords,
        input_hint_map,
    ):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        tiler = (self.m, self.n, self.k)
        mma, sw, sx = _make_down_operands(arena, tiler, self.stages)
        full, empty = barriers, barriers + self.stages
        done, read_done = empty + self.empty_stages, empty + self.empty_stages + 1
        if not entry_prefilled:
            if warp == self.mma_warp:
                with cute.arch.elect_one():
                    for stage in cutlass.range_constexpr(self.stages):
                        cute.arch.mbarrier_init(full + stage, 1)
                        if cutlass.const_expr(not self.omit_down_empty):
                            cute.arch.mbarrier_init(empty + stage, 1)
                    cute.arch.mbarrier_init(done, 1)
                    if cutlass.const_expr(not self.use_cta_reader_handoff):
                        cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
        tcgen05_fence("after_thread_sync")
        w_view = self._weight_view(weight)
        x_view = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1),
                stride=(self.inner, (1, 64), 0),
            ),
        )
        gw = cute.local_tile(w_view, (self.m, self.k), (output_tile, None, 0))
        gx = cute.local_tile(x_view, (self.n, self.k), (token_tile, None, 0))
        map_w = cute_ext.get_cta_v_map_ab(w_view, tiler, mma, "A")
        map_x = cute_ext.get_cta_v_map_ab(x_view, tiler, mma, "B")
        stage_bytes = (self.m + self.n) * self.k
        if warp == self.producer_warp:
            weight_source, weight_target, cache_policy = None, None, None
            weight_target, weight_source = self._partition_weight(
                weight_atom, weight_coords, sw, output_tile
            )
            cache_policy = _down_weight_cache_policy()
            if not entry_prefilled:
                # The enclosing kernel already waited for input and weight producers.
                # Pair each stage's operands so its MMA can start before later loads.
                for stage in cutlass.range_constexpr(self.stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + stage, stage_bytes
                        )
                    self._load_weight(
                        gw,
                        sw,
                        full,
                        stage,
                        split * self.tiles + stage,
                        map_w,
                        weight_atom,
                        weight_source,
                        weight_target,
                        cache_policy,
                    )
                    cute_ext.tma_load(
                        gx[None, None, split * self.tiles + stage],
                        sx[None, None, None, stage],
                        (full + stage).value,
                        cta_v_map=map_x,
                        update_expect_tx=False,
                    )
            if cutlass.const_expr(len(self.input_refill_hint) > 0):
                self._hint_refill_input(input_hint_map, output_tile, split, token_tile)
            for tile in cutlass.range(self.stages, self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(empty + stage, (tile // self.stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full + stage, stage_bytes)
                self._load_weight(
                    gw,
                    sw,
                    full,
                    stage,
                    split * self.tiles + tile,
                    map_w,
                    weight_atom,
                    weight_source,
                    weight_target,
                    cache_policy,
                )
                cute_ext.tma_load(
                    gx[None, None, split * self.tiles + tile],
                    sx[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_x,
                    update_expect_tx=False,
                )
        elif warp == self.mma_warp:
            thread_mma = mma.get_slice(0)
            a_desc = thread_mma.make_fragment_A(sw)
            b_desc = thread_mma.make_fragment_B(sx)
            fragment = thread_mma.make_fragment_C(
                mma.partition_shape_C((self.m, self.n))
            )
            acc = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                fragment.layout,
            )
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                for block in cutlass.range_constexpr(cute.size(a_desc, mode=[2])):
                    cute.gemm(
                        mma,
                        acc,
                        a_desc[None, None, block, stage],
                        b_desc[None, None, block, stage],
                        acc,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                if cutlass.const_expr(not self.omit_down_empty):
                    with cute.arch.elect_one():
                        tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            if cutlass.const_expr(not self.use_cta_reader_handoff):
                cute.arch.mbarrier_wait(read_done, 0)
        elif tid < self.epilogue_threads:
            cute.arch.mbarrier_wait(done, 0)
            fragment = mma.get_slice(0).make_fragment_C(
                mma.partition_shape_C((self.m, self.n))
            )
            acc = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                fragment.layout,
            )[((None, None), 0, 0)]
            reader_n = self.n // self.epilogue_groups
            tile_shape = (self.m, reader_n)
            group, lane = tid // 128, tid % 128
            acc = cute.local_tile(acc, tile_shape, (0, group))
            coordinates = cute.local_tile(
                cute.make_identity_tensor((self.m, self.n)),
                tile_shape,
                (0, group),
            )
            copy = tcgen05.make_tmem_copy(
                sm100_utils.get_tmem_load_op(
                    (self.m, reader_n, self.k),
                    cute_utils.LayoutEnum.COL_MAJOR,
                    cutlass.Float32,
                    cutlass.Float32,
                    tile_shape,
                    False,
                ),
                acc,
            )
            reader = copy.get_slice(lane)
            dummy = cute.make_tensor(
                partials.iterator, cute.make_layout(tile_shape, stride=(1, self.m))
            )
            values = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.rmem,
                cute_ext.make_t2r_rmem_layout(
                    copy, cute.flat_divide(dummy, tile_shape), lane
                ),
                alignment=32,
            )
            coords = reader.partition_D(coordinates)
            cute_ext.partition_and_copy(reader, acc, values)
            cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(self.aligned_partial_rows):
                origin, row_stride = self._partial_tile_origin(output_tile, split)
                tile_base = partials.iterator + origin
            for index in cutlass.range_constexpr(cute.size(values)):
                col, row = coords[index]
                local_row = cutlass.Int64(row)
                row = local_row + token_tile * self.n
                if cutlass.const_expr(self.token_parts > 1):
                    # Partials for each token part share the round's workspace.
                    local_row = local_row + (token_tile % self.token_parts) * self.n
                global_col = output_tile * self.m + col
                if row < x.shape[0] and global_col < self.projection_rows:
                    if cutlass.const_expr(self.aligned_partial_rows):
                        offset = cutlass.Int32(local_row) * row_stride + col
                        target = cute.make_tensor(
                            tile_base + offset, cute.make_layout(1)
                        )
                        target[0] = values[index]
                    elif cutlass.const_expr(self.static_partial_store):
                        # The launcher owns this fixed startup workspace shape.
                        offset = (
                            cutlass.Int64(split) * self.scratch_token_tile + local_row
                        ) * self.projection_rows + global_col
                        target = cute.make_tensor(
                            partials.iterator + offset, cute.make_layout(1)
                        )
                        target[0] = values[index]
                    else:
                        partials[split, local_row, global_col] = values[index]
            tcgen05_fence("before_thread_sync")
            if cutlass.const_expr(not self.use_cta_reader_handoff):
                cute.arch.mbarrier_arrive(read_done)

    @cute.experimental.jit
    def _load_entry_input(self, source, target, full, stage, tile, mapping):
        cute_ext.tma_load(
            source[None, None, tile],
            target[None, None, None, stage],
            (full + stage).value,
            cta_v_map=mapping,
            update_expect_tx=False,
        )

    @cute.experimental.jit
    def prepare_entry(
        self,
        x,
        weight,
        arena,
        barriers,
        output_tile,
        split,
        token_tile,
        weight_atom,
        weight_coords,
    ):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        tiler = (self.m, self.n, self.k)
        mma, sw, sx = _make_down_operands(arena, tiler, self.stages)
        full, empty = barriers, barriers + self.stages
        done, read_done = empty + self.empty_stages, empty + self.empty_stages + 1
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(full + stage, 1)
                    if cutlass.const_expr(not self.omit_down_empty):
                        cute.arch.mbarrier_init(empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                if cutlass.const_expr(not self.use_cta_reader_handoff):
                    cute.arch.mbarrier_init(read_done, self.epilogue_threads)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        w_view = self._weight_view(weight)
        x_view = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.inner // 64), 1),
                stride=(self.inner, (1, 64), 0),
            ),
        )
        gw = cute.local_tile(w_view, (self.m, self.k), (output_tile, None, 0))
        gx = cute.local_tile(x_view, (self.n, self.k), (token_tile, None, 0))
        map_w = cute_ext.get_cta_v_map_ab(w_view, tiler, mma, "A")
        map_x = cute_ext.get_cta_v_map_ab(x_view, tiler, mma, "B")
        stage_bytes = (self.m + self.n) * self.k
        if warp == self.producer_warp:
            weight_source, weight_target, cache_policy = None, None, None
            weight_target, weight_source = self._partition_weight(
                weight_atom, weight_coords, sw, output_tile
            )
            cache_policy = _down_weight_cache_policy()
            # The enclosing kernel already waited for input and weight producers.
            if cutlass.const_expr(self.entry_input_ring_first):
                # Fetch the reused input across the resident ring before
                # streaming weights. Each stage still waits for both operands.
                for stage in cutlass.range_constexpr(self.stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + stage, stage_bytes
                        )
                    self._load_entry_input(
                        gx, sx, full, stage, split * self.tiles + stage, map_x
                    )
            for stage in cutlass.range_constexpr(self.stages):
                tile = split * self.tiles + stage
                if cutlass.const_expr(not self.entry_input_ring_first):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + stage, stage_bytes
                        )
                self._load_weight(
                    gw,
                    sw,
                    full,
                    stage,
                    tile,
                    map_w,
                    weight_atom,
                    weight_source,
                    weight_target,
                    cache_policy,
                )
                if cutlass.const_expr(not self.entry_input_ring_first):
                    self._load_entry_input(gx, sx, full, stage, tile, map_x)


class _CompactCta(NamedTuple):
    first_quant_group: cutlass.Int32
    down_actor: cutlass.Boolean
    down_arrivals: cutlass.Int64
    activation_counter: cute.Pointer | None
    cached_scales: cute.Tensor
    prepared_residual: cute.Tensor | None


class _CompactMaps(NamedTuple):
    weight_atom: cute.CopyAtom | None
    weight_coords: cute.Tensor | None
    residual_atom: cute.CopyAtom | None
    residual_coords: cute.Tensor | None
    activation_atom: cute.CopyAtom | None
    activation_coords: cute.Tensor | None
    input_hint_map: TensorMap
    up_weight_map: TensorMap


class Fp8FusedGatedResidualKernel(ResidentMixer):
    """Resident phases reuse one operand arena and one TMEM allocation."""

    def __init__(
        self,
        *,
        g,
        h,
        r,
        p,
        token_bucket,
        projection_scale,
        use_pdl,
        weights_independent,
        down_per_channel,
        up_per_channel,
        down_tactic,
        down_token_tile,
        up_tactic,
        use_cta_reader_handoff,
        static_partial_store,
        static_partial_stride,
        quant_max_join,
        quant_warp_handoff,
        down_release,
        down_writer_release,
        async_phase_arrivals,
        omit_down_empty,
        up_register_rows,
        omit_up_empty,
        private_epoch_order,
        entry_transport,
        minimum_blocks_per_sm,
        entry_down_prefill,
        entry_down_input_ring_first,
        evict_down_weights_last,
        residual_issue,
        residual_transfer,
        pad_residual_rows,
        up_hidden_columns,
        up_gate_exchange,
        warp_owned_gates,
        quant_copy_warp,
        paired_up_sigmoid,
        up_scale_issue,
        prefetch_up_weights_on_down_retire,
        up_retirement,
        activation_handoff,
        split_activation_counter,
        dedicated_quant_ctas,
        quant_down_overlap,
        quant_down_poll,
        quant_activation_store,
        up_activation_prefix,
        stamped_activation,
        aligned_partial_rows,
        up_activation_transport,
        activation_retry_ns,
        input_refill_hint,
        up_activation_delivery,
    ):
        if type(entry_down_input_ring_first) is not bool or (
            entry_down_input_ring_first
            and (
                token_bucket not in (16, 32, 64)
                or not entry_down_prefill
                or down_tactic[3] != 2
            )
        ):
            raise ValueError(
                "Input-first ring requires two resident small narrow Down stages"
            )
        self.entry_down_input_ring_first = entry_down_input_ring_first
        if up_activation_delivery not in (
            "native",
            "private_deferred",
            "bundle_deferred",
        ):
            raise ValueError("Select the Up activation delivery explicitly")
        if up_activation_delivery != "native" and up_activation_transport != "bulk":
            raise ValueError("The Up delivery mode requires a packed activation")
        self.up_activation_delivery = up_activation_delivery
        self.copied_alpha = up_activation_delivery != "native"
        self.bundle_alpha = up_activation_delivery == "bundle_deferred"
        if up_activation_transport not in ("native", "packed_tma", "bulk"):
            raise ValueError("Select the Up activation transport explicitly")
        self.up_activation_transport = up_activation_transport
        if activation_retry_ns not in (0, 64, 128, 256):
            raise ValueError("Select an explicit activation retry interval")
        if activation_retry_ns and token_bucket not in (16, 32, 64):
            raise ValueError(
                "Activation retry delay requires a narrow 16, 32 or 64 bucket"
            )
        self.activation_retry_ns = activation_retry_ns
        if private_epoch_order not in ("ordered", "load", "store", "coherent"):
            raise ValueError(
                "Private epoch order must explicitly select ordered, load, store or coherent"
            )
        self.private_epoch_order = private_epoch_order
        if entry_transport not in ("native", "deferred"):
            raise ValueError(
                "Entry transport must explicitly select native or deferred"
            )
        if entry_transport == "deferred" and private_epoch_order not in (
            "load",
            "coherent",
        ):
            raise ValueError(
                "Deferred entry requires the explicitly relaxed initial-load choices"
            )
        if type(minimum_blocks_per_sm) is not int or minimum_blocks_per_sm not in (
            0,
            1,
        ):
            raise ValueError("Minimum blocks per SM must explicitly select 0 or 1")
        self.entry_transport = entry_transport
        if type(entry_down_prefill) is not bool:
            raise ValueError("Entry Down prefill must be an explicit boolean")
        self.entry_down_prefill = entry_down_prefill
        if type(evict_down_weights_last) is not bool:
            raise TypeError("Down weight cache policy requires an explicit boolean")
        self.evict_down_weights_last = evict_down_weights_last
        self.minimum_blocks_per_sm = minimum_blocks_per_sm
        if not isinstance(use_cta_reader_handoff, bool):
            raise TypeError("Reader handoff selection must be an explicit boolean")
        self.use_cta_reader_handoff = use_cta_reader_handoff
        if not isinstance(static_partial_store, bool) or not isinstance(
            static_partial_stride, bool
        ):
            raise TypeError("Workspace geometry selections must be explicit booleans")
        self.static_partial_store = static_partial_store
        self.static_partial_stride = static_partial_stride
        self.g = g
        self.h = h
        self.r = r
        self.p = p
        self.n = token_bucket
        self.scale = projection_scale
        self.pdl = use_pdl
        self.weights_independent = weights_independent
        self.down_per_channel = down_per_channel
        self.up_per_channel = up_per_channel
        if type(omit_down_empty) is not bool or type(up_register_rows) is not bool:
            raise ValueError(
                "Select terminal Down events and Up row scale storage explicitly"
            )
        if up_register_rows and (token_bucket != 256 or activation_handoff != "grid"):
            raise ValueError("Register row scales require the narrow256 M128 Up reader")
        self.up_register_rows = up_register_rows
        self.prefill_scales = token_bucket == 16
        if up_scale_issue not in ("native", "up_async", "handoff_async"):
            raise ValueError("Select the Up weight scale issue point explicitly")
        if up_scale_issue != "native" and token_bucket != 128:
            raise ValueError("Async Up weight scales require narrow128")
        self.up_scale_issue = up_scale_issue
        if up_retirement not in ("native", "skip_final", "skip_final_and_tail_join"):
            raise ValueError("Final Up retirement uses the narrow persistent tile loop")
        if up_retirement == "skip_final_and_tail_join" and (
            p != r + g
            or token_bucket not in (16, 32, 64)
            or (not use_cta_reader_handoff)
        ):
            raise ValueError(
                "Independent warp exit requires the local narrow TMEM handoff"
            )
        self.up_skip_final = up_retirement != "native"
        self.final_cta_join = up_retirement != "skip_final_and_tail_join"
        self.async_up_scales = up_scale_issue != "native" and up_per_channel
        # Two reader groups cover the small narrow tiles without idle warps.
        self.compute_threads = 256 if token_bucket <= 32 else 512
        self.reader_groups = self.compute_threads // 128
        self.producer_warp = self.compute_threads // 32
        self.mma_warp = self.producer_warp + 1
        self.threads = self.compute_threads + 64
        # Independent row groups avoid serial quantization passes in large buckets.
        self.quant_rows = 4 if token_bucket == 256 else 2 if token_bucket == 128 else 1
        self.quant_threads = self.compute_threads // self.quant_rows
        self.quant_columns = max(
            2,
            1
            << (((r + self.quant_threads - 1) // self.quant_threads - 1).bit_length()),
        )
        if quant_activation_store not in (
            "native",
            "pack",
            "pack_sat",
            "pack_max",
            "scalar_max",
        ):
            raise ValueError("Select the FP8 activation write policy explicitly")
        if quant_activation_store != "native" and (
            p != r + g or token_bucket not in (16, 32, 64) or self.quant_columns != 2
        ):
            raise ValueError("The two-column FP8 store requires a small narrow bucket")
        self.quant_activation_store = quant_activation_store
        if quant_max_join not in ("double", "single_full", "single_active"):
            raise ValueError("Select the quantization maximum handoff explicitly")
        if quant_max_join != "double" and (
            token_bucket > 128
            or r % (32 * self.quant_columns)
            or r // self.quant_columns > self.quant_threads
        ):
            raise ValueError(
                "Single quantization join requires full active warps in every row"
            )
        self.quant_max_threads = (
            r // self.quant_columns
            if quant_max_join == "single_active"
            else self.quant_threads
        )
        if type(quant_warp_handoff) is not bool:
            raise TypeError("Select the quantization handoff explicitly")
        if quant_warp_handoff and (
            token_bucket != 64
            or quant_max_join != "single_active"
            or activation_handoff != "warp"
            or (
                residual_issue
                not in ("handoff", "register_before_mma", "register_after_mma")
            )
        ):
            raise ValueError("Warp-local quantization requires the narrow64 handoff")
        self.quant_warp_handoff = quant_warp_handoff
        # The warp after the FP8 writers handles the injection columns.
        self.quant_handoff_threads = self.quant_max_threads + 32
        self.up_hidden = 32
        self.up_channels = self.up_hidden * g
        self.up_parts = 2 if token_bucket == 256 else 1
        # Small narrow tiles avoid padding tokens to the native M128 reader.
        self.token_major_up = token_bucket == 256
        self.up_rows = (
            min(128, max(64, token_bucket)) if self.token_major_up else token_bucket
        )
        self.up_mn = (
            (self.up_rows, self.up_channels)
            if self.token_major_up
            else (self.up_channels, self.up_rows)
        )
        self.hidden_reader_columns = min(32, token_bucket // self.reader_groups)
        self.hidden_reader_passes = 1
        self.reader_columns = self.up_hidden // self.reader_groups
        self.up_k, self.up_stages = up_tactic
        if up_activation_transport != "native" and (
            token_bucket not in (16, 32, 64)
            or self.token_major_up
            or self.up_parts != 1
            or (self.up_stages != 1)
            or (self.up_k != r)
        ):
            raise ValueError("Packed activation requires one resident narrow Up tile")
        if self.up_k not in (64, r) or r % self.up_k:
            raise ValueError("Up K tiles must evenly cover the reduction")
        self.up_tiles = r // self.up_k
        if not 1 <= self.up_stages <= self.up_tiles:
            raise ValueError("Up stages must fit the reduction")
        if type(omit_up_empty) is not bool or (
            omit_up_empty and self.up_tiles != self.up_stages
        ):
            raise ValueError(
                "Omitting Up empty events requires a resident narrow reduction"
            )
        self.omit_up_empty = omit_up_empty
        self.up_empty_stages = 0 if omit_up_empty else self.up_stages
        self.up_tail_tiles = (
            self.up_stages if self.up_parts == 2 and self.up_tiles > 1 else 0
        )
        self.early_up_weights = True
        if residual_issue not in (
            "up",
            "handoff",
            "activation",
            "down_entry",
            "register_before_mma",
            "register_after_mma",
        ):
            raise ValueError("Invalid residual copy issue point")
        if residual_issue != "up" and (
            self.token_major_up or not self.early_up_weights
        ):
            raise ValueError(
                "Early residual issue requires prepared hidden-major Up weights"
            )
        self.residual_issue = residual_issue
        self.residual_in_registers = residual_issue in (
            "register_before_mma",
            "register_after_mma",
        )
        if self.residual_in_registers and (
            token_bucket not in (16, 32, 64)
            or p != r + g
            or residual_transfer != "lane"
        ):
            raise ValueError(
                "The Up register residual path uses the small narrow hidden-major reader"
            )
        if residual_transfer not in ("lane", "aligned_lane", "tma"):
            raise ValueError("Select the residual transfer explicitly")
        if residual_transfer != "lane" and (
            token_bucket != 128 or residual_issue == "activation"
        ):
            raise ValueError("Aligned or TMA residual transfer requires narrow128")
        self.residual_tma = residual_transfer == "tma"
        if type(pad_residual_rows) is not bool or (
            pad_residual_rows
            and (
                (p != r + g or token_bucket != 64)
                or residual_issue != "handoff"
                or residual_transfer != "lane"
                or (up_hidden_columns != 4)
            )
        ):
            raise ValueError(
                "The alternate residual layout requires the narrow64 four-column reader"
            )
        # Put the four rows read by a mix warp on different residual banks.
        self.residual_row_stride = 160 if pad_residual_rows else 128
        if up_hidden_columns not in (1, 2, 4):
            raise ValueError("Select the contiguous Up output width explicitly")
        self.vector_hidden_up = up_hidden_columns > 1
        if self.vector_hidden_up and (
            p != r + g
            or (
                token_bucket == 16
                and (up_hidden_columns != 2 or not self.residual_in_registers)
                or (
                    token_bucket in (32, 64)
                    and (up_hidden_columns != 4 or self.residual_in_registers)
                )
                or token_bucket not in (16, 32, 64)
            )
            or self.token_major_up
            or self.residual_tma
        ):
            raise ValueError("The output width requires a supported narrow Up reader")
        self.up_hidden_columns = up_hidden_columns
        if up_gate_exchange not in (
            "native",
            "row_pairs_scalar",
            "row_pairs_vector",
        ) or (
            up_gate_exchange != "native"
            and (
                (p != r + g or token_bucket != 64)
                or self.up_rows != 64
                or self.up_channels != 128
                or (not self.vector_hidden_up)
            )
        ):
            raise ValueError(
                "Row-pair gate exchange requires the narrow64 hidden-major tile"
            )
        self.up_gate_exchange = up_gate_exchange
        if type(warp_owned_gates) is not bool or (
            warp_owned_gates
            and (
                (p != r + g or token_bucket != 32)
                or self.up_hidden_columns != 4
                or self.hidden_reader_columns != 16
                or (self.up_stages != 1)
                or (self.up_k != r)
                or (residual_issue != "handoff")
                or (not self.use_cta_reader_handoff)
            )
        ):
            raise ValueError("Warp-owned mixing requires the resident narrow32 Up plan")
        if quant_copy_warp not in (-1, 5) or (
            quant_copy_warp >= 0
            and (
                (p != r + g or token_bucket != 32)
                or residual_issue != "handoff"
                or residual_transfer != "lane"
                or (activation_handoff != "row_tags_quant")
                or (self.quant_rows != 1)
                or (32 * quant_copy_warp * self.quant_columns < r)
                or (32 * quant_copy_warp >= self.compute_threads)
            )
        ):
            raise ValueError("A residual copy warp requires the narrow32 row-tag plan")
        self.quant_copy_warp = quant_copy_warp
        self.warp_owned_gates = warp_owned_gates
        if warp_owned_gates:
            # Pad the row to preserve copy alignment and distribute shared accesses.
            self.residual_row_stride = 144
        if type(paired_up_sigmoid) is not bool or (
            paired_up_sigmoid
            and ((p != r + g or token_bucket != 32) or up_hidden_columns != 4)
        ):
            raise ValueError(
                "Paired sigmoid requires the four-column narrow 32-token plan"
            )
        self.paired_up_sigmoid = paired_up_sigmoid
        if warp_owned_gates and not paired_up_sigmoid:
            raise ValueError("Warp-owned mixing requires paired sigmoid arithmetic")
        self.up_output_threads = (
            self.up_rows * (self.up_hidden // up_hidden_columns)
            if self.vector_hidden_up
            else self.compute_threads
        )
        if self.up_output_threads > self.compute_threads:
            raise ValueError(
                "The Up output packets must fit in the original reader threads"
            )
        self.separate_up_scale_warp = (
            self.vector_hidden_up and token_bucket == 64 and up_per_channel
        )
        if residual_issue == "down_entry" and not self.residual_tma:
            raise ValueError("Down entry residual issue requires TMA")
        self.residual_alignment = 16 if residual_transfer == "lane" else 128
        if activation_handoff not in (
            "grid",
            "warp",
            "row_async_bulk",
            "row_count_quant",
            "row_tags_quant",
        ):
            raise ValueError("Invalid activation handoff")
        if activation_handoff != "grid" and (
            self.token_major_up or self.quant_rows != 1 or not self.early_up_weights
        ):
            raise ValueError(
                "Row publication requires one quantized row and hidden-major Up"
            )
        self.activation_handoff = activation_handoff
        if type(split_activation_counter) is not bool or (
            split_activation_counter
            and (
                (p != r + g or token_bucket != 64)
                or activation_handoff != "warp"
                or up_activation_delivery != "native"
                or dedicated_quant_ctas
                or (async_phase_arrivals != "native")
            )
        ):
            raise ValueError(
                "Two activation counters require the narrow 64-token warp handoff"
            )
        self.split_activation_counter = split_activation_counter
        self.activation_rows_per_counter = token_bucket // (
            2 if split_activation_counter else 1
        )
        self.activation_row_tags = activation_handoff in (
            "row_async_bulk",
            "row_tags_quant",
        )
        self.activation_row_count = activation_handoff == "row_count_quant"
        self.quant_only_down_acquire = activation_handoff in (
            "row_count_quant",
            "row_tags_quant",
        )
        self.activation_row_gate = self.activation_row_tags or self.activation_row_count
        # Reserve up to 15 words for the aligned row tags or counter.
        ready_words = self.n if self.activation_row_tags else 1
        self.extra_epoch_slots = (
            ready_words + 15
            if self.activation_row_gate
            else (
                1 + int(split_activation_counter) if activation_handoff == "warp" else 0
            )
        )
        self.paired_activation_alpha = self.vector_hidden_up and token_bucket == 64
        self.up_scale_warps = (
            1 if self.paired_activation_alpha else (self.up_rows + 31) // 32
        )
        if type(dedicated_quant_ctas) is not bool:
            raise ValueError("Quantization CTA placement requires an explicit boolean")
        if dedicated_quant_ctas and (
            activation_handoff == "warp" or not self.early_up_weights
        ):
            raise ValueError("Dedicated quantization CTAs require the early-Up handoff")
        if self.activation_row_gate:
            narrow16 = (
                (p == r + g and token_bucket == 16)
                and self.activation_row_tags
                and dedicated_quant_ctas
                and (up_activation_delivery == "native")
                and (up_activation_transport == "bulk")
                and (async_phase_arrivals == "both")
                and (self.up_stages == 1)
                and (self.up_tiles == 1)
            )
            narrow32 = (
                (p == r + g and token_bucket == 32)
                and self.quant_only_down_acquire
                and (not dedicated_quant_ctas)
                and (up_activation_delivery in ("private_deferred", "bundle_deferred"))
                and (up_activation_transport == "bulk")
                and (async_phase_arrivals == "down")
            )
            if not (narrow16 or narrow32):
                raise ValueError(
                    "The row gate requires the async narrow16 or narrow32 plan"
                )
        self.dedicated_quant_ctas = dedicated_quant_ctas
        if down_release not in ("native", "early_arrive"):
            raise ValueError("Select the Down release point explicitly")
        if down_release == "early_arrive" and (
            token_bucket > 128 or not use_cta_reader_handoff
        ):
            raise ValueError("Early Down release requires the narrow CTA reader join")
        self.down_release = down_release
        if async_phase_arrivals not in ("native", "down", "activation", "both") or (
            async_phase_arrivals != "native" and token_bucket not in (16, 32, 64)
        ):
            raise ValueError("Async release is scoped to small narrow startup buckets")
        self.async_down_arrival = async_phase_arrivals in ("down", "both")
        self.async_activation_arrival = async_phase_arrivals in ("activation", "both")
        self.quant_groups = (self.n + self.quant_rows - 1) // self.quant_rows
        if entry_transport == "deferred" and not self.early_up_weights:
            raise ValueError(
                "Deferred entry requires the existing early-Up handoff join"
            )
        self.hidden_tiles = h // self.up_hidden
        self.up_operand_bytes = (
            (self.up_channels + self.up_parts * self.up_rows)
            * self.up_k
            * self.up_stages
        )
        self.up_barrier_count = (
            self.up_stages
            + self.up_empty_stages
            + self.up_parts
            + int(not self.use_cta_reader_handoff or self.warp_owned_gates)
            + int(self.residual_tma)
        )
        self.down_m, self.splits, down_k, down_stages = down_tactic
        if down_token_tile != token_bucket and (token_bucket, down_token_tile) not in (
            (64, 32),
            (128, 64),
            (256, 64),
        ):
            raise ValueError("Unsupported Down token tile")
        self.down_token_parts = token_bucket // down_token_tile
        if g * h % (self.splits * down_k) or not (
            1 <= down_stages <= g * h // (self.splits * down_k)
        ):
            raise ValueError("Down stages must fit an evenly partitioned reduction")
        down_epilogue_threads = min(
            256 if token_bucket == 16 else 512, self.compute_threads
        )
        group_count, before_cta_join, separate_row_halves = down_writer_release
        if group_count not in (1, 2) or (separate_row_halves and group_count != 2):
            raise ValueError("Invalid Down writer release groups")
        if down_writer_release != (1, False, False) and (
            (p != r + g or token_bucket != 16)
            or down_token_tile != token_bucket
            or down_epilogue_threads != 256
            or (not dedicated_quant_ctas)
            or (not self.activation_row_tags)
            or (not self.async_down_arrival)
            or (down_release != "early_arrive")
        ):
            raise ValueError(
                "Down writer release groups require the narrow16 reader groups"
            )
        self.down_publish_groups = group_count
        self.down_publish_before_join = before_cta_join
        self.down_publish_row_halves = separate_row_halves
        if self.down_publish_row_halves:
            # Reserve a separate line after the sixteen activation row tags.
            self.extra_epoch_slots += 16
        if type(aligned_partial_rows) is not bool or (
            aligned_partial_rows
            and (
                (p != r + g or token_bucket != 16)
                or (self.down_m, self.splits, down_token_tile) != (64, 20, 16)
            )
        ):
            raise ValueError("Aligned partial rows require the narrow16 Down layout")
        self.aligned_partial_rows = aligned_partial_rows
        if input_refill_hint not in ((), (3,), (4,), (3, 4)):
            raise ValueError("Invalid Down input refill hint")
        if input_refill_hint and (
            (p != r + g or token_bucket != 256)
            or (down_k, down_stages, down_token_tile) != (512, 3, 64)
        ):
            raise ValueError("Down input refill hints require the narrow256 Down ring")
        self.input_refill_hint = input_refill_hint
        self.down = _DownPhase(
            aligned_partial_rows=aligned_partial_rows,
            input_refill_hint=input_refill_hint,
            evict_weights_last=evict_down_weights_last,
            omit_down_empty=omit_down_empty,
            producer_warp=self.producer_warp,
            mma_warp=self.mma_warp,
            use_cta_reader_handoff=self.use_cta_reader_handoff,
            static_partial_store=self.static_partial_store,
            scratch_token_tile=token_bucket,
            entry_input_ring_first=entry_down_input_ring_first,
            role="projection",
            hc_count=g,
            hidden_size=h,
            lowrank=r,
            projection_rows=p,
            token_tile=down_token_tile,
            projection_tile=self.down_m,
            k_tile=down_k,
            stages=down_stages,
            split_k=self.splits,
            use_pdl=False,
            weights_independent=weights_independent,
            weight_scale_per_channel=down_per_channel,
            token_major_up=False,
            weight_scale_vectorized=False,
            up_hidden_tile=32,
            up_token_rows=self.up_rows,
            separate_operand_ready=False,
            epilogue_threads=down_epilogue_threads,
            shared_up_scales=False,
            interleave_tma=False,
        )
        self.down_tiles = (p + self.down_m - 1) // self.down_m
        self.down_tasks = self.down_token_parts * self.down_tiles * self.splits
        if quant_down_overlap not in (
            "native",
            "quant_cta",
            "warp",
            "early_scalar",
            "early_vector",
        ):
            raise ValueError("Select the quantization Down overlap explicitly")
        if quant_down_overlap != "native" and (
            (p != r + g or token_bucket != 16)
            or activation_handoff != "row_async_bulk"
            or self.quant_columns != 2
        ):
            raise ValueError(
                "Quantization Down overlap requires the narrow 16-token row plan"
            )
        self.quant_down_overlap = quant_down_overlap
        self.quant_warp_down = quant_down_overlap in (
            "warp",
            "early_scalar",
            "early_vector",
        )
        self.quant_early_scales = quant_down_overlap in ("early_scalar", "early_vector")
        if quant_down_poll not in ("native", "acquire_fence", "confirm_acquire"):
            raise ValueError("Select a valid Down polling mode")
        if quant_down_poll != "native" and (
            not self.quant_warp_down
            or down_release != "early_arrive"
            or not self.async_down_arrival
        ):
            raise ValueError(
                "Down polling requires the narrow 16-token warp acquire and async release"
            )
        self.quant_down_poll = quant_down_poll
        self.up_tasks = self.hidden_tiles
        if type(prefetch_up_weights_on_down_retire) is not bool or (
            prefetch_up_weights_on_down_retire
            and (
                (p != r + g or token_bucket != 64)
                or not self.early_up_weights
                or self.up_stages != 1
            )
        ):
            raise ValueError(
                "The Up weight hint requires the narrow64 single-stage Up tile"
            )
        self.prefetch_up_weights_on_down_retire = prefetch_up_weights_on_down_retire
        self.up_full_bytes = (
            self.up_channels + self.up_rows * self.up_parts
        ) * self.up_k
        if self.copied_alpha:
            self.up_full_bytes += self.up_rows * 4
        prefix_planes, begin_mma_early = up_activation_prefix
        if (
            prefix_planes not in (0, 1, 3)
            or type(begin_mma_early) is not bool
            or (begin_mma_early and not prefix_planes)
        ):
            raise ValueError("Select a supported Up activation split")
        if prefix_planes != 0 and (
            p != r + g
            or token_bucket not in (16, 64)
            or (self.up_k, self.up_tiles, self.up_stages, self.up_parts) != (r, 1, 1, 1)
            or self.token_major_up
            or self.residual_tma
            or (up_activation_delivery != "native")
            or (up_activation_transport != "bulk")
        ):
            raise ValueError(
                "The activation split requires the local resident narrow K320 layout"
            )
        # Packed SW64 activation stores one contiguous segment per K64 plane.
        self.up_activation_prefix = prefix_planes * 64 * self.up_rows
        self.up_activation_suffix = self.up_rows * self.up_k - self.up_activation_prefix
        self.up_mma_before_suffix = begin_mma_early
        self.up_first_tx_bytes = self.up_full_bytes
        if type(stamped_activation) is not bool:
            raise ValueError("Select the stamped activation plan explicitly")
        self.stamped_activation = stamped_activation
        self.stamped_prefix_threads = 96 if stamped_activation else 32
        self.stamped_suffix_threads = 96
        self.activation_copy_threads = (
            self.stamped_prefix_threads + self.stamped_suffix_threads
        )
        self.stamped_prefix_records = self.up_rows * 6
        self.activation_payload_records = 0
        self.up_full_arrivals = 1
        if stamped_activation:
            if (
                (p != r + g or token_bucket != 16)
                or up_activation_transport != "bulk"
                or activation_handoff != "row_async_bulk"
                or (self.up_rows != self.n)
                or (self.up_k != self.r)
                or (self.up_tiles != 1)
                or (self.up_parts != 1)
                or self.token_major_up
                or (up_activation_delivery != "native")
                or prefix_planes
                or (quant_activation_store not in ("pack", "pack_sat", "pack_max"))
                or (not self.quant_early_scales)
                or (quant_max_join != "single_full")
            ):
                raise ValueError(
                    "Stamped activation requires the narrow 16-token packed plan"
                )
            self.up_first_tx_bytes -= self.up_rows * self.up_k
            self.up_full_arrivals += self.stamped_prefix_threads // 32
            self.activation_payload_records = self.up_rows * (self.up_k // 64) * 6
            self.up_barrier_count += 1
        fp8_threads = (
            self.quant_max_threads if self.quant_warp_handoff else self.compute_threads
        )
        packed_store = (
            up_activation_transport != "native"
            or quant_activation_store
            in (
                "pack",
                "pack_sat",
                "pack_max",
            )
        )
        self.quantization = QuantizationConfig(
            threads=fp8_threads,
            maximum_threads=self.quant_max_threads,
            injection_start=fp8_threads,
            column_stride=1,
            warp_max="redux",
            replicate_maximum=quant_max_join != "double",
            refined_scales=False,
            clamp=(
                "ptx_lower"
                if quant_activation_store in ("pack_max", "scalar_max")
                else "lower" if quant_activation_store == "pack_sat" else "both"
            ),
            activation_store=(
                "stamped"
                if stamped_activation
                else "packed" if packed_store else "native"
            ),
            release_writes=activation_handoff != "grid" and not stamped_activation,
        )
        if begin_mma_early:
            self.up_barrier_count += 1
            self.up_first_tx_bytes -= self.up_activation_suffix
        if self.bundle_alpha:
            self.up_operand_bytes += self.up_rows * 4
        self.columns = max(128, token_bucket, self.up_channels * self.up_parts)
        self.arena_bytes = max(
            (self.down_m + down_token_tile) * down_k * down_stages,
            self.up_operand_bytes,
            0 if self.token_major_up else token_bucket * self.up_channels * 4,
        )
        self.down_barrier_count = (
            down_stages
            + self.down.empty_stages
            + (1 if self.use_cta_reader_handoff else 2)
        )
        if (
            residual_issue == "down_entry"
            and self.up_barrier_count - 1 < self.down_barrier_count
        ):
            raise ValueError(
                "The entry residual barrier must be outside the Down barriers"
            )
        self.barrier_count = max(self.down_barrier_count, self.up_barrier_count)
        # A conservative occupancy query includes alignment and epilogue scratch.
        self.shared_bytes = (
            self.arena_bytes
            + (0 if self.token_major_up else token_bucket * self.up_channels)
            + self.up_rows * (self.residual_row_stride - 128)
            + (2 * token_bucket + self.up_channels) * 4
            + self.barrier_count * 8
            + 2048
        )

    @cute.experimental.jit
    def __call__(
        self,
        x,
        xs,
        down,
        ds,
        up,
        us,
        partials,
        activation,
        alpha,
        epochs,
        out,
        inject,
        grid_ctas: cutlass.Int32,
        stream: CUstream,
    ):
        weight_atom, weight_coords = None, None
        weight_atom, weight_coords = self.down.make_weight_atom(down)
        residual_atom, residual_coords = None, None
        if cutlass.const_expr(self.residual_tma):
            residual_source = cute.make_tensor(
                x.iterator,
                cute.make_layout(
                    (self.h, self.g, x.shape[0]), stride=(1, self.h, self.g * self.h)
                ),
            )
            residual_atom, residual_coords = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                residual_source,
                self._residual_tma_layout(),
                cta_tiler=(self.up_hidden, self.g, self.up_rows),
            )
        activation_atom, activation_coords = None, None
        if cutlass.const_expr(self.up_activation_transport == "packed_tma"):
            layout = self._packed_activation_tma_layout()
            source = cute.make_tensor(activation.iterator, layout)
            activation_atom, activation_coords = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                source,
                layout,
                cta_tiler=layout.shape,
            )
        if cutlass.const_expr(len(self.input_refill_hint) > 0):
            input_hint_map = create_tensor_map_tiled(
                x.iterator.toint(),
                cutlass.Float8E4M3FN,
                [64, x.shape[0], self.g * self.h // 64],
                [self.g * self.h // 16, 4],
                [64, self.down.n, self.down.k // 64],
                swizzle=TensorMapSwizzle.s64b,
                l2_promotion=TensorMapL2Promotion.none,
            )
        else:
            input_hint_map = create_tensor_map_tiled(
                x.iterator.toint(),
                cutlass.Float8E4M3FN,
                [64],
                [],
                [64],
                swizzle=TensorMapSwizzle.none,
                l2_promotion=TensorMapL2Promotion.none,
            )
        if cutlass.const_expr(self.warp_owned_gates):
            # Shared M is [eight hidden channels, branch, hidden group].
            # The canonical global weight tensor keeps the hidden-first strides.
            up_weight_map = create_tensor_map_tiled(
                up.iterator.toint(),
                cutlass.Float8E4M3FN,
                [64, 8, self.g, self.h // 8, self.r // 64],
                [self.r // 16, self.h * self.r // 16, 8 * self.r // 16, 4],
                [64, 8, self.g, self.up_hidden // 8, self.up_k // 64],
                swizzle=TensorMapSwizzle.s64b,
                l2_promotion=TensorMapL2Promotion.none,
            )
        else:
            up_weight_map = create_tensor_map_tiled(
                up.iterator.toint(),
                cutlass.Float8E4M3FN,
                [64],
                [],
                [64],
                swizzle=TensorMapSwizzle.none,
                l2_promotion=TensorMapL2Promotion.none,
            )
        self.kernel(
            x,
            xs,
            down,
            ds,
            up,
            us,
            partials,
            activation,
            alpha,
            epochs,
            out,
            inject,
            weight_atom,
            weight_coords,
            residual_atom,
            residual_coords,
            activation_atom,
            activation_coords,
            input_hint_map,
            up_weight_map,
        ).launch(
            grid=(grid_ctas, 1, 1),
            block=(self.threads, 1, 1),
            cluster=None,
            stream=stream,
            cooperative=True,
            use_pdl=self.pdl,
            min_blocks_per_mp=self.minimum_blocks_per_sm,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        x,
        xs,
        down,
        ds,
        up,
        us,
        partials,
        activation,
        alpha,
        epochs,
        out,
        inject,
        weight_atom,
        weight_coords,
        residual_atom,
        residual_coords,
        activation_atom,
        activation_coords,
        input_hint_map: cutlass.GridConstant[TensorMap],
        up_weight_map: cutlass.GridConstant[TensorMap],
    ):
        cta = _mixer_cta()
        tid, warp, pid, ctas = cta
        first_quant_group = cutlass.Int32(pid)
        down_actor = cutlass.Boolean(True)
        down_arrivals = ctas
        allocate_tmem = cutlass.Boolean(True)
        if cutlass.const_expr(self.dedicated_quant_ctas):
            down_actor = pid < self.down_tasks
            if ctas > self.down_tasks:
                down_arrivals = cutlass.Int64(self.down_tasks)
            mma_workers = max(self.down_tasks, self.up_tasks)
            allocate_tmem = pid < mma_workers
            if ctas > mma_workers:
                spares = cute.math.min(
                    cutlass.Int32(ctas - mma_workers), cutlass.Int32(self.quant_groups)
                )
                # Move the first quantization groups to otherwise idle CTAs.
                if pid < mma_workers:
                    first_quant_group = cutlass.Int32(pid) + spares
                elif pid < mma_workers + spares:
                    first_quant_group = cutlass.Int32(pid) - mma_workers
        activation_counter = None
        if cutlass.const_expr(self.activation_row_gate):
            # The launcher allocates enough padding after the CTA ledger.
            activation_counter = epochs.iterator + ((ctas + 16) & -16)
        elif cutlass.const_expr(self.activation_handoff != "grid"):
            activation_counter = epochs.iterator + ctas + 1
        elif cutlass.const_expr(self.copied_alpha):
            activation_counter = epochs.iterator
        shared = _mixer_shared(self.arena_bytes, self.barrier_count, 0)
        cached_scales = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(2 * self.n + self.up_channels),
            alignment=16,
        )
        prepared_residual = None
        if cutlass.const_expr(
            self.residual_issue != "up" and not self.residual_in_registers
        ):
            prepared_residual = self._residual_tile()
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
        if pid < self.down_tasks:
            # Keep the original two joins; overlap actual first-ring TMA
            # with TMEM allocation and the deferred private epoch copy.
            output_tile, split, input_tile = self._down_tile_coordinates(
                pid, cutlass.Int64(0)
            )
            self.down.prepare_entry(
                x,
                down,
                shared.arena,
                shared.down_barriers,
                output_tile,
                split,
                input_tile,
                weight_atom,
                weight_coords,
            )
        if tid == 0:
            # This slot has one writer per call and is private to this stream.
            # Stream completion or the PDL wait above orders the previous call.
            _copy_entry_epoch(
                shared.epoch.iterator, epochs.iterator + cutlass.Int64(pid) + 1
            )
            _commit_residual()
        if allocate_tmem and warp == self.mma_warp:
            cute.arch.alloc_tmem(self.columns, shared.tmem, is_two_cta=False)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.sync_threads()
        phase_base = cutlass.Int64(0)
        self._run_mixer_rounds(
            MixerOperands(
                x,
                xs,
                down,
                ds,
                up,
                us,
                partials,
                activation,
                alpha,
                epochs,
                out,
                inject,
            ),
            cta,
            shared,
            _CompactMaps(
                weight_atom,
                weight_coords,
                residual_atom,
                residual_coords,
                activation_atom,
                activation_coords,
                input_hint_map,
                up_weight_map,
            ),
            _CompactCta(
                first_quant_group,
                down_actor,
                down_arrivals,
                activation_counter,
                cached_scales,
                prepared_residual,
            ),
            phase_base,
            entry_prefills_down=True,
            prefetch_residual_before_down=self.residual_issue == "down_entry",
            skip_last_up_join=True,
            track_last_up_task=True,
        )
        if cutlass.const_expr(self.use_cta_reader_handoff):
            # Readers drained/fenced their TMEM loads before the owner CTA joins.
            tcgen05_fence("after_thread_sync")
        if allocate_tmem and warp == self.mma_warp:
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, shared.tmem),
                self.columns,
            )
        if cutlass.const_expr(self.final_cta_join):
            cute.arch.sync_threads()
        if cutlass.const_expr(self.pdl):
            if tid == 0:
                # The next launch waits for completion before reading output or epochs.
                cute.arch.griddepcontrol_launch_dependents()

    @cute.experimental.jit
    def _prefetch_round_residual(self, io, cta, shared, maps, state, global_base):
        if cta.pid < self.up_tasks and cta.warp == self.producer_warp:
            # Down uses earlier events in the same array while this copy runs.
            ready = shared.down_barriers + self.up_barrier_count - 1
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(ready, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_warp()
            self._issue_residual_tma(
                maps.residual_atom,
                maps.residual_coords,
                state.prepared_residual,
                ready,
                cta.pid,
                global_base,
            )

    @cute.experimental.jit
    def _project_down_tile(self, io, cta, shared, maps, state, tile, global_base):
        self.down.project(
            io.x,
            io.down,
            io.partials,
            shared.arena,
            shared.down_barriers,
            shared.tmem,
            tile.output_tile,
            tile.split,
            tile.token_tile,
            entry_prefilled=tile.entry_prefilled,
            weight_atom=maps.weight_atom,
            weight_coords=maps.weight_coords,
            input_hint_map=maps.input_hint_map,
        )
        if cutlass.const_expr(self.prefetch_up_weights_on_down_retire):
            if (
                (global_base == 0)
                & (cta.pid < self.up_tasks)
                & (cta.warp == self.producer_warp)
                & (tile.last)
            ):
                # The final Down input and weight requests are already issued on this warp.
                self._hint_up_weight(io.up, cta.pid)
        if cutlass.const_expr(self.down_publish_before_join):
            if tile.last:
                self._publish_down_readers(cta.tid, io.epochs.iterator)
        cute.arch.sync_threads()
        if cutlass.const_expr(
            self.down_release == "early_arrive" and not self.down_publish_before_join
        ):
            if cta.tid == 0 and tile.last:
                # The last Down task's partial stores precede the reader join.
                _arrive_phase_release(io.epochs.iterator, self.async_down_arrival)

    @cute.experimental.jit
    def _begin_quantization(
        self, io, cta, shared, maps, state, phase_base, global_base
    ):
        # The ordinal is not used by Down. Complete its initial copy
        # before this existing CTA join, then read it exactly once.
        if global_base == 0 and cta.tid == 0:
            _wait_residual()
        # Publish Down before reusing its operand arena for Up weights.
        cute.arch.sync_threads()
        if global_base == 0:
            phase_base = shared.epoch[0]
        if cta.tid == 0 and state.down_actor:
            if cutlass.const_expr(self.down_release == "early_arrive"):
                # CTAs with no Down task still participate in the full-grid protocol.
                if cta.pid >= self.down_tasks:
                    _arrive_phase_release(io.epochs.iterator, self.async_down_arrival)
            else:
                _arrive_phase_release(io.epochs.iterator, self.async_down_arrival)
        if cta.pid < self.up_tasks:
            self._prefetch_up_weights(
                io.up,
                io.us,
                state.cached_scales,
                shared.arena,
                shared.down_barriers,
                cta.pid,
                maps.up_weight_map,
            )
            if cutlass.const_expr(self.residual_issue == "handoff"):
                if cutlass.const_expr(self.residual_tma):
                    if cta.warp == self.producer_warp:
                        self._issue_residual_tma(
                            maps.residual_atom,
                            maps.residual_coords,
                            state.prepared_residual,
                            shared.down_barriers + self.up_barrier_count - 1,
                            cta.pid,
                            global_base,
                        )
                elif cutlass.const_expr(self.quant_warp_handoff):
                    if self.quant_handoff_threads <= cta.tid < self.compute_threads:
                        self._issue_residual_copy(
                            io.x,
                            state.prepared_residual,
                            cta.pid,
                            global_base,
                            self.quant_handoff_threads,
                            self.compute_threads - self.quant_handoff_threads,
                        )
                elif cutlass.const_expr(self.quant_copy_warp >= 0):
                    # The copy warp finishes its group before the Up reader-input join.
                    first_copy_thread = self.quant_copy_warp * 32
                    if state.first_quant_group < self.quant_groups:
                        if first_copy_thread <= cta.tid < first_copy_thread + 32:
                            self._issue_residual_copy(
                                io.x,
                                state.prepared_residual,
                                cta.pid,
                                global_base,
                                first_copy_thread,
                                32,
                            )
                    elif cta.tid < self.compute_threads:
                        self._issue_residual_copy(
                            io.x,
                            state.prepared_residual,
                            cta.pid,
                            global_base,
                            0,
                            self.compute_threads,
                        )
                else:
                    if cta.tid < self.compute_threads:
                        self._issue_residual_copy(
                            io.x,
                            state.prepared_residual,
                            cta.pid,
                            global_base,
                            0,
                            self.compute_threads,
                        )
        if cutlass.const_expr(self.quant_warp_handoff):
            if (
                state.first_quant_group < self.quant_groups
                and cta.tid < self.quant_handoff_threads
            ):
                if cta.tid == 0:
                    _wait_activation(
                        io.epochs.iterator, (2 * phase_base + 1) * cta.count
                    )
                _epilogue_barrier(self.quant_handoff_threads, cutlass.Int32(14))
        else:
            wait_down = cutlass.Boolean(True)
            if cutlass.const_expr(
                self.quant_only_down_acquire or self.quant_down_overlap == "quant_cta"
            ):
                wait_down = state.first_quant_group < self.quant_groups
            if cutlass.const_expr(self.quant_warp_down):
                wait_down = cutlass.Boolean(False)
            if cta.tid == 0 and wait_down:
                down_target = (phase_base + 1) * cta.count
                if cutlass.const_expr(self.dedicated_quant_ctas):
                    down_target = phase_base + state.down_arrivals
                elif cutlass.const_expr(self.activation_handoff == "warp"):
                    down_target = (2 * phase_base + 1) * cta.count
                while _acquire(io.epochs.iterator) < down_target:
                    pass
            cute.arch.sync_threads()
        activation_target = cutlass.Int64(0)
        tag_increment = cutlass.Int64(0)
        if cutlass.const_expr(self.activation_row_gate):
            # The dedicated plan counts individual Down arrivals. The
            # other plan counts whole-grid epochs.
            if cutlass.const_expr(self.dedicated_quant_ctas):
                tag_increment, reuse_increment = state.down_arrivals, cta.count
            else:
                tag_increment = reuse_increment = cutlass.Int64(1)
            if global_base + self.n < io.x.shape[0]:
                tag_increment = tag_increment + reuse_increment
            # Each row joins the next phase base, including the upcoming
            # reuse step. The root processes that step separately.
            activation_target = phase_base + tag_increment
        row = cutlass.Int64(state.first_quant_group) * self.quant_rows
        if cutlass.const_expr(self.quant_warp_handoff):
            if cta.tid >= self.quant_handoff_threads:
                row = cutlass.Int64(self.n)
        return phase_base, row, (tag_increment, activation_target)

    @cute.experimental.jit
    def _quantize_row(self, io, cta, state, quant, phase_base, global_base, row):
        tag_increment, activation_target = quant
        tid = cta.tid
        preloaded_scales = None
        if cutlass.const_expr(self.quant_early_scales):
            # Entry already orders the external scales, so load them before Down acquire.
            input_scale = cutlass.Float32(0.0)
            weight_scales = cute.make_rmem_tensor(2, cutlass.Float32)
            weight_scales.fill(0.0)
            first_col = cutlass.Int32(-1)
            first_inject = self.quantization.injection_start
            if tid < self.r // 2:
                first_col = tid * 2
            elif first_inject <= tid < first_inject + self.g // 2:
                first_col = self.r + (tid - first_inject) * 2
            global_row = global_base + row
            if global_row < io.x.shape[0] and first_col >= 0:
                input_scale, loaded_pair = self._read_quant_scales(
                    io.xs, io.ds, global_row, first_col, 2, 1
                )
                for i in cutlass.range_constexpr(2):
                    weight_scales[i] = loaded_pair[i]
            preloaded_scales = (input_scale, weight_scales)
        if cutlass.const_expr(self.quant_warp_down):
            self._wait_quant_down(
                tid, io.epochs.iterator, phase_base + state.down_arrivals
            )

        activation_layout = None
        if cutlass.const_expr(self.up_activation_transport != "native"):
            activation_layout = sm100_utils.make_smem_layout_b(
                self._make_up_mma(),
                (*self.up_mn, self.up_k),
                cutlass.Float8E4M3FN,
                self.up_stages,
            )
        self._quantize_down_row(
            io,
            tid,
            global_base,
            row,
            preloaded_scales,
            activation_layout,
            activation_target,
        )

        if cutlass.const_expr(self.quant_warp_handoff):
            _epilogue_barrier(self.quant_handoff_threads, cutlass.Int32(14))
        else:
            cute.arch.sync_threads()
        if cutlass.const_expr(self.quantization.release_writes):
            if tid == 0:
                ready = state.activation_counter
                if cutlass.const_expr(self.activation_row_gate):
                    if cutlass.const_expr(self.activation_row_tags):
                        ready = ready + row
                    _arrive_activation_row(ready, tag_increment)
                else:
                    if cutlass.const_expr(self.split_activation_counter):
                        # Even and odd rows publish to separate counters.
                        ready = ready + (row % 2)
                    _arrive_phase_release(ready, self.async_activation_arrival)

    @cute.experimental.jit
    def _finish_quantization(self, io, cta, state, phase_base, quant):
        _, activation_target = quant
        if cutlass.const_expr(self.quant_warp_handoff):
            # The quantization and copy warps reunite before Up consumes the arena.
            cute.arch.sync_threads()
        if cutlass.const_expr(self.activation_handoff == "grid"):
            _publish_activation_proxy()
            activation_target = (phase_base + 2) * cta.count
            if cutlass.const_expr(self.dedicated_quant_ctas):
                activation_target = phase_base + state.down_arrivals + cta.count
            if cutlass.const_expr(self.copied_alpha):
                _grid_barrier(
                    io.epochs, activation_target, False, self.async_activation_arrival
                )
            elif cutlass.const_expr(self.activation_retry_ns > 0):
                cute.arch.sync_threads()
                if cta.tid == 0:
                    _arrive_phase_release(
                        io.epochs.iterator, self.async_activation_arrival
                    )
                    _wait_activation_with_delay(
                        io.epochs.iterator,
                        activation_target,
                        self.activation_retry_ns,
                    )
                cute.arch.sync_threads()
            else:
                _grid_barrier(
                    io.epochs, activation_target, True, self.async_activation_arrival
                )
        elif cutlass.const_expr(not self.activation_row_gate):
            activation_target = (phase_base + 1) * self.activation_rows_per_counter
        return activation_target

    @cute.experimental.jit
    def _project_up_tile(
        self, io, shared, maps, state, tile, global_base, activation_target
    ):
        self._up(
            io.activation,
            io.alpha,
            io.x,
            io.xs,
            io.up,
            io.us,
            io.out,
            shared.arena,
            shared.down_barriers,
            shared.tmem,
            tile.hidden_tile,
            global_base,
            tile.weights_prepared,
            state.cached_scales,
            state.prepared_residual,
            state.activation_counter,
            activation_target,
            maps.residual_atom,
            maps.residual_coords,
            maps.activation_atom,
            maps.activation_coords,
            maps.up_weight_map,
        )

    @cute.experimental.jit
    def _finish_round(
        self, io, cta, shared, state, phase_base, activation_target, more_rounds
    ):
        if cutlass.const_expr(self.up_activation_delivery != "native"):
            if cta.tid == 0 and more_rounds and cta.pid >= self.up_tasks:
                # A non-Up CTA has no local MMA completion to order
                # arrival at the reuse barrier after the earlier phase.
                if cutlass.const_expr(
                    self.copied_alpha and self.activation_handoff == "grid"
                ):
                    _wait_activation(io.epochs.iterator, activation_target)
                elif cutlass.const_expr(self.activation_handoff == "warp"):
                    _wait_activation(
                        io.epochs.iterator, (2 * phase_base + 1) * cta.count
                    )
        if cutlass.const_expr(self.activation_row_gate or self.dedicated_quant_ctas):
            # The row gate includes any upcoming reuse step in the epoch.
            phase_base = activation_target
            if cutlass.const_expr(not self.activation_row_gate):
                if more_rounds:
                    phase_base = phase_base + cta.count
        elif cutlass.const_expr(self.activation_handoff == "grid"):
            phase_base = phase_base + 2
            if more_rounds:
                phase_base = phase_base + 1
        else:
            phase_base = phase_base + 1
        if cta.tid == 0:
            _store_private_epoch(
                io.epochs.iterator + cutlass.Int64(cta.pid) + 1, phase_base
            )
        # Only another token round needs a global handoff before reuse.
        # Stream completion or the successor's PDL wait orders a new call.
        if cutlass.const_expr(self.activation_row_gate):
            if more_rounds:
                if cutlass.const_expr(self.dedicated_quant_ctas):
                    reuse_target = phase_base
                    down_target = phase_base - cta.count
                else:
                    reuse_target = phase_base * cta.count
                    down_target = (phase_base - 1) * cta.count
                if cutlass.const_expr(
                    self.quant_only_down_acquire or self.quant_down_overlap != "native"
                ):
                    if (
                        cta.tid == 0
                        and cta.pid >= self.up_tasks
                        and state.first_quant_group >= self.quant_groups
                    ):
                        # An idle CTA cannot contribute to reuse before
                        # the previous Down count is complete.
                        _wait_activation(io.epochs.iterator, down_target)
                _grid_barrier(io.epochs, reuse_target, True, False)
        elif cutlass.const_expr(self.dedicated_quant_ctas):
            if more_rounds:
                _grid_barrier(io.epochs, phase_base, True, False)
        elif cutlass.const_expr(self.activation_handoff == "grid"):
            if more_rounds:
                _grid_barrier(io.epochs, phase_base * cta.count, True, False)
        else:
            if more_rounds:
                _grid_barrier(io.epochs, 2 * phase_base * cta.count, True, False)
            elif cta.pid == 0 and cta.tid == 0:
                # All Down arrivals retired before this CTA passed its Down wait.
                _release(io.epochs.iterator, 2 * phase_base * cta.count)
        return phase_base

    @cute.experimental.jit
    def _publish_down_readers(self, tid, root):
        # Each 128-thread TMEM reader group owns a disjoint half of the token
        # tile. The full CTA still joins before recycling MMA barriers.
        threads = self.down.epilogue_threads // self.down_publish_groups
        if tid < self.down.epilogue_threads:
            group = tid // threads
            if cutlass.const_expr(self.down_publish_before_join):
                barrier = cutlass.Int32(6) + group
                llvm.inline_asm(
                    None,
                    [barrier.ir_value(), cutlass.Int32(threads).ir_value()],
                    "barrier.cta.sync $0, $1;",
                    "r,r,~{memory}",
                    has_side_effects=True,
                    asm_dialect=0,
                )
            if tid % threads == 0:
                counter = root
                _arrive_phase_release(counter, self.async_down_arrival)

    @cute.experimental.jit
    def _partials(
        self,
        partial,
        row,
        col,
        columns: cutlass.Constexpr,
        injection: cutlass.Constexpr,
    ):
        pieces = cute.make_rmem_tensor((columns, self.splits), cutlass.Float32)
        row = cutlass.Int64(row)
        if cutlass.const_expr(self.aligned_partial_rows):
            if cutlass.const_expr(injection):
                stride = self.g
                base = (
                    partial.iterator
                    + self.n * self.splits * self.r
                    + row * self.splits * self.g
                    + col
                    - self.r
                )
            else:
                tile, tile_col = col // self.down_m, col % self.down_m
                tile_size = self.n * self.splits * self.down_m
                stride = self.down_m
                base = (
                    partial.iterator
                    + tile * tile_size
                    + row * self.splits * stride
                    + tile_col
                )
        else:
            if cutlass.const_expr(self.static_partial_stride):
                # Scratch rows equal the startup bucket, including repeated rounds.
                stride = cutlass.Int64(self.n) * self.p
            else:
                stride = cutlass.Int64(partial.shape[1]) * self.p
            base = partial.iterator + row * self.p + col
        return _reduce_split_partials(pieces, base, stride, row, False)

    @cute.experimental.jit
    def _wait_quant_down(self, tid, root, target):
        warp = cute.arch.make_warp_uniform(tid // 32)
        if warp < self.quant_max_threads // 32 or warp == self.compute_threads // 32:
            if cute.arch.lane_idx() == 0:
                if cutlass.const_expr(self.quant_down_poll == "native"):
                    _wait_activation(root, target)
                else:
                    while _load_relaxed_counter(root) < target:
                        pass
                    if cutlass.const_expr(self.quant_down_poll == "acquire_fence"):
                        # The successful relaxed GPU load and this fence form
                        # an acquire before the warp reads Down partials.
                        llvm.inline_asm(
                            None,
                            [],
                            "fence.acquire.gpu;",
                            "~{memory}",
                            has_side_effects=True,
                            asm_dialect=0,
                        )
                    else:
                        _wait_activation(root, target)
            cute.arch.sync_warp()

    @cute.experimental.jit
    def _store_packed_quantization(
        self,
        io,
        row,
        col,
        local_tid,
        rounded,
        multiplier,
        maximum,
        preloaded_scales,
        activation_layout,
        target,
    ):
        if cutlass.const_expr(self.stamped_activation):
            packed_scale = cutlass.Float32(0.0)
            if col == 60:
                packed_scale = _quant_scale(maximum)
            input_scale, _ = preloaded_scales
            row_plane = cutlass.Int32(row) + (col // 64) * self.n
            word = 6 * row_plane + (col % 64) // 12
            _store_compact_activation(
                io.activation.iterator + word * 16,
                _quantized_value(rounded[0], multiplier, self.quantization.clamp),
                _quantized_value(rounded[1], multiplier, self.quantization.clamp),
                target,
                cute.arch.lane_idx(),
                col < 64,
                packed_scale,
                input_scale,
            )
            if local_tid == 0:
                # Up receives the scale in its first plane as well as in the row header.
                self._store_quant_alpha(io, row, _quant_scale(maximum), target)
        else:
            if cutlass.const_expr(activation_layout is None):
                offset = row * self.r + col
            else:
                # Both adjacent codes stay in the same 16-byte swizzle block.
                offset = cutlass.Int64(
                    cute.crd2idx(((row, col % 32), 0, col // 32, 0), activation_layout)
                )
            destination = io.activation.iterator + offset
            if cutlass.const_expr(
                self.quant_activation_store in ("pack", "pack_sat", "pack_max")
            ):
                _store_activation_pair(
                    destination,
                    _quantized_value(rounded[0], multiplier, self.quantization.clamp),
                    _quantized_value(rounded[1], multiplier, self.quantization.clamp),
                )
            else:
                codes = cute.make_tensor(
                    destination, cute.make_layout(self.quant_columns)
                )
                for i in cutlass.range_constexpr(self.quant_columns):
                    value = _quantized_value(
                        rounded[i], multiplier, self.quantization.clamp
                    )
                    codes[i] = value.to(cutlass.Float8E4M3FN)

    @cute.experimental.jit
    def _store_quant_alpha(self, io, row, scale, target):
        if cutlass.const_expr(self.stamped_activation):
            destination = (
                io.activation.iterator + self.activation_payload_records * 16 + row * 16
            )
            _store_stamped_alpha(destination, scale, target)
        else:
            io.alpha[row] = scale

    @cute.experimental.jit
    def _init_up_barriers(self, bars, preserve_residual: cutlass.Constexpr):
        warp = cute.arch.make_warp_uniform(cute.arch.thread_idx()[0] // 32)
        full, empty = bars, bars + self.up_stages
        done = empty + self.up_empty_stages
        read_done = done + self.up_parts
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.up_stages):
                    cute.arch.mbarrier_init(full + stage, self.up_full_arrivals)
                for part in cutlass.range_constexpr(self.up_parts):
                    cute.arch.mbarrier_init(done + part, 1)
                if cutlass.const_expr(self.up_mma_before_suffix):
                    cute.arch.mbarrier_init(bars + self.up_barrier_count - 1, 1)
                if cutlass.const_expr(self.stamped_activation):
                    cute.arch.mbarrier_init(
                        bars + self.up_barrier_count - 1,
                        self.stamped_suffix_threads // 32,
                    )
                if cutlass.const_expr(self.warp_owned_gates):
                    cute.arch.mbarrier_init(read_done, self.compute_threads // 32)
                elif cutlass.const_expr(not self.use_cta_reader_handoff):
                    cute.arch.mbarrier_init(read_done, self.compute_threads)
                if cutlass.const_expr(self.residual_tma and not preserve_residual):
                    cute.arch.mbarrier_init(bars + self.up_barrier_count - 1, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

    @cute.experimental.jit
    def _hint_up_weight(self, weight, hidden_tile):
        if (weight.iterator.toint() & 15) == 0:
            with cute.arch.elect_one():
                for branch in cutlass.range_constexpr(self.g):
                    channel = (
                        branch * self.h + cutlass.Int64(hidden_tile) * self.up_hidden
                    )
                    base = weight.iterator + channel * self.r
                    llvm.inline_asm(
                        None,
                        [
                            base.toint().ir_value(),
                            cutlass.Int32(self.up_hidden * self.r).ir_value(),
                        ],
                        "cp.async.bulk.prefetch.L2.global [$0], $1;",
                        "l,r,~{memory}",
                        has_side_effects=True,
                        asm_dialect=0,
                    )

    @cute.experimental.jit
    def _load_warp_grouped_weights(self, destination, ready, mapping, hidden_tile):
        with cute.arch.elect_one():
            hidden_group = cutlass.Int32(hidden_tile * (self.up_hidden // 8))
            llvm.inline_asm(
                None,
                [
                    cutlass.Uint32(destination.iterator.toint()).ir_value(),
                    mapping.get_ptr().toint().ir_value(),
                    cutlass.Uint32(ready.toint()).ir_value(),
                    hidden_group.ir_value(),
                ],
                "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes "
                "[$0], [$1, {0, 0, 0, $3, 0}], [$2];",
                "r,l,r,r,~{memory}",
                has_side_effects=True,
                asm_dialect=0,
            )

    @cute.experimental.jit
    def _prefetch_up_weights(
        self, weight, ws, cached, arena, bars, hidden_tile, up_weight_map
    ):
        dtype = cutlass.Float8E4M3FN
        mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            self.up_mn,
        )
        tiler = (*self.up_mn, self.up_k)
        al = sm100_utils.make_smem_layout_a(mma, tiler, dtype, self.up_stages)
        bl = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.up_stages)
        if cutlass.const_expr(self.token_major_up):
            sw = _operand_at(arena, bl, 0)
        else:
            if cutlass.const_expr(self.bundle_alpha):
                sw = _operand_at(arena, al, 0)
            else:
                sw = _operand_at(arena, al, cute.cosize(bl))
        wv = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (
                    (self.up_hidden, self.g, self.h // self.up_hidden),
                    (64, self.r // 64),
                    1,
                ),
                stride=((self.r, self.h * self.r, self.up_hidden * self.r), (1, 64), 0),
            ),
        )
        gw = cute.local_tile(wv, (self.up_channels, self.up_k), (hidden_tile, None, 0))
        if cutlass.const_expr(self.token_major_up):
            mapping = cute_ext.get_cta_v_map_ab(wv, tiler, mma, "B")
        else:
            mapping = cute_ext.get_cta_v_map_ab(wv, tiler, mma, "A")
        self._init_up_barriers(bars, self.residual_issue == "down_entry")
        warp = cute.arch.make_warp_uniform(cute.arch.thread_idx()[0] // 32)
        if warp == self.producer_warp:
            for stage in cutlass.range_constexpr(self.up_stages):
                with cute.arch.elect_one():
                    # The first event always includes the resident weights.
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bars + stage, self.up_first_tx_bytes
                    )
                if cutlass.const_expr(self.warp_owned_gates):
                    self._load_warp_grouped_weights(
                        sw[None, None, None, stage],
                        bars + stage,
                        up_weight_map,
                        hidden_tile,
                    )
                else:
                    cute_ext.tma_load(
                        gw[None, None, stage],
                        sw[None, None, None, stage],
                        (bars + stage).value,
                        cta_v_map=mapping,
                        update_expect_tx=False,
                    )
        tid = cute.arch.thread_idx()[0]
        if cutlass.const_expr(self.prefill_scales):
            if tid < self.up_channels:
                channel = (
                    (tid // self.up_hidden) * self.h
                    + hidden_tile * self.up_hidden
                    + tid % self.up_hidden
                )
                cached[2 * self.n + tid] = _load_f32(
                    ws.iterator + cutlass.Int64(channel if self.up_per_channel else 0)
                )

    @cute.experimental.jit
    def _issue_up_scale_copy(self, scales, cached, hidden_tile):
        tid = cute.arch.thread_idx()[0]
        if tid < self.up_channels // 4:
            col = tid * 4
            channel = (
                (col // self.up_hidden) * self.h
                + hidden_tile * self.up_hidden
                + col % self.up_hidden
            )
            _copy_residual_16(
                cached.iterator + 2 * self.up_rows + col,
                scales.iterator + channel,
                cutlass.Int32(16),
            )
            _commit_residual()

    @cute.experimental.jit
    def _read_hidden_weight_scales(self, scales, cached, hidden_tile, tid):
        if (scales.iterator.toint() & 15) == 0:
            # The first two reader warps join the Up activation producer.
            # A third warp can load the independent weight scales during that wait.
            writer = tid - 64
            if 0 <= writer < self.up_channels // 4:
                column = writer * 4
                channel = (
                    (column // self.up_hidden) * self.h
                    + hidden_tile * self.up_hidden
                    + column % self.up_hidden
                )
                source = cute.make_ptr(
                    cutlass.Float32,
                    (scales.iterator + cutlass.Int64(channel)).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                target = cute.make_ptr(
                    cutlass.Float32,
                    (cached.iterator + 2 * self.up_rows + column).toint(),
                    cute.AddressSpace.smem,
                    assumed_align=16,
                )
                layout = cute.make_layout(4)
                packet = cute.make_rmem_tensor(layout, cutlass.Float32)
                atom = cute.make_copy_atom(
                    CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128
                )
                cute.copy(atom, cute.make_tensor(source, layout), packet)
                cute.copy(atom, packet, cute.make_tensor(target, layout))
        elif tid < self.up_channels:
            channel = (
                (tid // self.up_hidden) * self.h
                + hidden_tile * self.up_hidden
                + tid % self.up_hidden
            )
            cached[2 * self.up_rows + tid] = scales[channel]

    @cute.experimental.jit
    def _residual_tile(self):
        return cute_ext.allocate(
            cutlass.Float8E4M3FN,
            cute.AddressSpace.smem,
            cute.make_layout((self.up_rows, 128), stride=(self.residual_row_stride, 1)),
            alignment=self.residual_alignment,
        )

    @cute.experimental.jit
    def _residual_tma_layout(self):
        return cute.make_layout(
            (self.up_hidden, self.g, self.up_rows),
            stride=(1, self.up_hidden, self.up_channels),
        )

    @cute.experimental.jit
    def _issue_residual_tma(
        self, atom, coordinates, destination, ready, hidden_tile, global_base
    ):
        tile = (self.up_hidden, self.g, self.up_rows)
        source = cute.local_tile(
            coordinates, tile, (hidden_tile, 0, global_base // self.up_rows)
        )
        target = cute.make_tensor(destination.iterator, self._residual_tma_layout())
        target_part, source_part = cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(target, 0, 3),
            cute.group_modes(source, 0, 3),
        )
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(
                ready, self.up_rows * self.up_channels
            )
        # The TMA copy elects a lane internally and needs the producer warp.
        cute.copy(atom, source_part, target_part, tma_bar_ptr=ready)

    @cute.experimental.jit
    def _issue_residual_copy(
        self,
        x,
        residual_tile,
        hidden_tile,
        global_base,
        first_thread: cutlass.Constexpr,
        copy_threads: cutlass.Constexpr,
    ):
        tid = cute.arch.thread_idx()[0] - first_thread
        for copy_slice in cutlass.range_constexpr(
            (self.up_rows * 128 + copy_threads * 16 - 1) // (copy_threads * 16)
        ):
            linear = tid * 16 + copy_slice * copy_threads * 16
            if linear < self.up_rows * 128:
                local_row = linear // 128
                channel = linear % 128
                global_row = global_base + cutlass.Int64(local_row)
                source = (
                    x.iterator
                    + global_row * (self.g * self.h)
                    + (channel // 32) * self.h
                    + hidden_tile * 32
                    + channel % 32
                )
                size = cutlass.Int32(0)
                if global_row < x.shape[0]:
                    size = cutlass.Int32(16)
                destination = linear
                if cutlass.const_expr(self.residual_row_stride != 128):
                    destination = local_row * self.residual_row_stride + channel
                _copy_residual_16(residual_tile.iterator + destination, source, size)
        _commit_residual()

    @cute.experimental.jit
    def _load_up_residual(self, x, residual, hidden_tile, global_base, tid):
        if cutlass.const_expr(self.vector_hidden_up):
            packet = self.up_hidden_columns
            lanes_per_row = self.up_hidden // packet
            global_row = global_base + cutlass.Int64(tid // lanes_per_row)
            if tid < self.up_output_threads and global_row < x.shape[0]:
                hidden = hidden_tile * self.up_hidden + packet * (tid % lanes_per_row)
                for branch in cutlass.range_constexpr(self.g):
                    offset = global_row * (self.g * self.h) + branch * self.h + hidden
                    codes = self._load_hidden_packet(
                        x,
                        offset,
                        cutlass.Float8E4M3FN,
                        packet,
                        cute.AddressSpace.gmem,
                    )
                    for i in cutlass.range_constexpr(packet):
                        residual[i, branch] = codes[i]
        else:
            hidden = hidden_tile * self.up_hidden + tid % self.up_hidden
            residual.fill(0.0)
            for part in cutlass.range_constexpr(cute.size(residual, mode=[0])):
                row = tid // self.up_hidden + part * (
                    self.compute_threads // self.up_hidden
                )
                global_row = global_base + cutlass.Int64(row)
                if global_row < x.shape[0]:
                    for branch in cutlass.range_constexpr(self.g):
                        residual[part, branch] = x[global_row, branch * self.h + hidden]

    @cute.experimental.jit
    def _join_activation_readers(self):
        # The producer's acquire orders the scale readers' generic loads.
        _epilogue_barrier(32 * (1 + self.up_scale_warps), cutlass.Int32(13))

    @cute.experimental.jit
    def _cache_hidden_input_scale(self, xs, cached, x, global_base, row):
        if 0 <= row < self.up_rows:
            global_row = global_base + cutlass.Int64(row)
            scale = cutlass.Float32(0)
            if global_row < x.shape[0]:
                scale = _x_scale(xs, global_row)
            cached[self.up_rows + row] = scale

    @cute.experimental.jit
    def _read_activation_scale_pair(self, alpha, cached, lane):
        if lane < 32:
            row = lane * 2
            # Quantization also writes alpha for padded rows in this scratch tile.
            source = cute.make_ptr(
                cutlass.Float32,
                (alpha.iterator + row).toint(),
                cute.AddressSpace.gmem,
                assumed_align=8,
            )
            target = cute.make_ptr(
                cutlass.Float32,
                (cached.iterator + row).toint(),
                cute.AddressSpace.smem,
                assumed_align=8,
            )
            layout = cute.make_layout(2)
            packet = cute.make_rmem_tensor(layout, cutlass.Float32)
            atom = cute.make_copy_atom(
                CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=64
            )
            cute.copy(atom, cute.make_tensor(source, layout), packet)
            cute.copy(atom, packet, cute.make_tensor(target, layout))

    @cute.experimental.jit
    def _make_up_mma(self):
        dtype = cutlass.Float8E4M3FN
        return sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            tcgen05.CtaGroup.ONE,
            self.up_mn,
        )

    @cute.experimental.jit
    def _packed_activation_tma_layout(self):
        return cute.make_layout((128, self.up_rows * self.up_k // 128), stride=(1, 128))

    @cute.experimental.jit
    def _copy_stamped_activation(
        self, arena, activation, full, target, hidden_tile, tid, cached
    ):
        if tid == 0:
            # Spread the first check across rows. Every record is checked
            # again by the warp that copies it.
            row = hidden_tile % self.n
            _wait_compact_activation(
                activation.iterator + row * 96, target.to(cutlass.Uint32)
            )
        _epilogue_barrier(self.activation_copy_threads, cutlass.Int32(11))

        if tid < self.stamped_prefix_threads:
            # One contiguous 32-record batch per prefix warp.
            copy_warp = cute.arch.make_warp_uniform(tid // 32)
            self._copy_compact_activation(
                arena,
                activation,
                target.to(cutlass.Uint32),
                tid % 32,
                cached,
                copy_warp * 32,
                32,
                32,
                True,
            )
            self._arrive_stamped_activation(full)
        else:
            self._copy_compact_activation(
                arena,
                activation,
                target.to(cutlass.Uint32),
                tid - self.stamped_prefix_threads,
                cached,
                self.stamped_prefix_records,
                self.activation_payload_records - self.stamped_prefix_records,
                self.stamped_suffix_threads,
                False,
            )
            self._arrive_stamped_activation(full + self.up_barrier_count - 1)

    @cute.experimental.jit
    def _arrive_stamped_activation(self, full):
        llvm.inline_asm(
            None,
            [],
            "fence.proxy.async.shared::cta;",
            "~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )
        cute.arch.sync_warp()
        if cute.arch.lane_idx() == 0:
            cute.arch.mbarrier_arrive(full)

    @cute.experimental.jit
    def _copy_compact_activation(
        self,
        arena,
        activation,
        target,
        tid,
        cached,
        first_record,
        records: cutlass.Constexpr,
        threads: cutlass.Constexpr,
        read_row_scales: cutlass.Constexpr,
    ):
        packets_per_thread = (records + threads - 1) // threads
        packets = cute.make_rmem_tensor((4, packets_per_thread), cutlass.Uint32)
        for part in cutlass.range_constexpr(packets_per_thread):
            index = part * threads + tid
            if index < records:
                word = first_record + index
                a, b, c, ordinal = _load_compact_activation(
                    activation.iterator + word * 16
                )
                (
                    packets[0, part],
                    packets[1, part],
                    packets[2, part],
                    packets[3, part],
                ) = (a, b, c, ordinal)
        for part in cutlass.range_constexpr(packets_per_thread):
            index = part * threads + tid
            if index < records:
                word = first_record + index
                a, b, c = packets[0, part], packets[1, part], packets[2, part]
                if packets[3, part] != target:
                    a, b, c = _wait_compact_activation(
                        activation.iterator + word * 16, target
                    )
                packet = word % 6
                row_plane = word // 6
                linear = row_plane * 64 + packet * 12
                _write_compact_plane_word(arena, linear, a)
                if packet < 5:
                    _write_compact_plane_word(arena, linear + 4, b)
                    _write_compact_plane_word(arena, linear + 8, c)
                elif cutlass.const_expr(read_row_scales):
                    # The scale bits belong to the same checked record.
                    cached[row_plane] = cutlass.Float32(
                        llvm.bitcast(T.f32(), b.ir_value())
                    )
                    cached[self.n + row_plane] = cutlass.Float32(
                        llvm.bitcast(T.f32(), c.ir_value())
                    )

    @cute.experimental.jit
    def _issue_packed_activation(
        self, arena, activation, full, atom, coords, size: cutlass.Constexpr
    ):
        if cutlass.const_expr(self.up_activation_transport == "bulk"):
            if cutlass.const_expr(self.up_activation_prefix == 0):
                _copy_activation_bytes(arena.iterator, activation.iterator, full, size)
            else:
                prefix = self.up_activation_prefix
                with cute.arch.elect_one():
                    _activation_bulk_copy(
                        arena.iterator, activation.iterator, full, prefix
                    )
                    suffix_full = full
                    if cutlass.const_expr(self.up_mma_before_suffix):
                        suffix_full = full + self.up_barrier_count - 1
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            suffix_full, self.up_activation_suffix
                        )
                    _activation_bulk_copy(
                        arena.iterator + prefix,
                        activation.iterator + prefix,
                        suffix_full,
                        self.up_activation_suffix,
                    )
        else:
            layout = self._packed_activation_tma_layout()
            destination = cute.make_tensor(arena.iterator, layout)
            dst, src = cpasync.tma_partition(
                atom,
                0,
                cute.make_layout(1),
                cute.group_modes(destination, 0, 2),
                cute.group_modes(coords, 0, 2),
            )
            cute.copy(atom, src, dst, tma_bar_ptr=full)

    @cute.experimental.jit
    def _issue_deferred_activation(self, dst, codes, alpha, alpha_dst, full):
        payload_bytes = self.up_rows * self.up_k
        if cutlass.const_expr(self.bundle_alpha):
            payload_bytes += self.up_rows * 4
        _copy_activation_bytes(dst, codes, full, payload_bytes)
        if cutlass.const_expr(not self.bundle_alpha):
            _copy_activation_bytes(alpha_dst, alpha, full, self.up_rows * 4)

    @cute.experimental.jit
    def _load_hidden_packet(
        self,
        tensor,
        offset,
        dtype: cutlass.Constexpr,
        width: cutlass.Constexpr,
        space: cutlass.Constexpr,
    ):
        pointer = cute.make_ptr(
            dtype,
            (tensor.iterator + offset).toint(),
            space,
            assumed_align=width * dtype.width // 8,
        )
        source = cute.make_tensor(pointer, cute.make_layout(width))
        values = cute.make_rmem_tensor(width, dtype)
        atom = cute.make_copy_atom(
            CopyUniversalOp(), dtype, num_bits_per_copy=width * dtype.width
        )
        cute.copy(atom, source, values)
        return values

    @cute.experimental.jit
    def _up_gate_channel(self, channel, row):
        if cutlass.const_expr(self.warp_owned_gates):
            if cutlass.const_expr(self.up_gate_exchange == "native"):
                return channel ^ (8 * (row % 4))
            return channel ^ (8 * ((row // 2) % 2))
        return channel

    @cute.experimental.jit
    def _mix_hidden_columns(
        self,
        gates,
        residual_tile,
        residual,
        cached,
        out,
        x,
        tid,
        hidden_tile,
        global_base,
    ):
        packet = self.up_hidden_columns
        lanes_per_row = self.up_hidden // packet
        if cutlass.const_expr(self.warp_owned_gates):
            hidden_group = (tid // 32) % (self.up_hidden // 8)
            lane = tid % 32
            row = (tid // 128) * self.hidden_reader_columns + lane // (8 // packet)
            column_in_group = (lane % (8 // packet)) * packet
            column = hidden_group * 8 + column_in_group
        else:
            row = tid // lanes_per_row
            column = packet * (tid % lanes_per_row)
        hidden = hidden_tile * self.up_hidden + column
        global_row = global_base + cutlass.Int64(row)
        if tid < self.up_output_threads and global_row < x.shape[0]:
            activation_scale, input_scale = cached[row], cached[self.up_rows + row]
            totals = cute.make_rmem_tensor(packet, cutlass.Float32)
            totals.fill(0.0)
            for branch in cutlass.range_constexpr(self.g):
                channel = branch * self.up_hidden + column
                offset = row * self.up_channels + channel
                gate_offset = offset
                if cutlass.const_expr(self.warp_owned_gates):
                    gate_channel = (
                        hidden_group * (8 * self.g) + branch * 8 + column_in_group
                    )
                    gate_offset = row * self.up_channels + self._up_gate_channel(
                        gate_channel, row
                    )
                weight_offset = 2 * self.up_rows + channel
                weights = self._load_hidden_packet(
                    cached,
                    weight_offset,
                    cutlass.Float32,
                    packet,
                    cute.AddressSpace.smem,
                )
                if cutlass.const_expr(self.residual_in_registers):
                    codes = residual[None, branch]
                else:
                    codes = self._load_hidden_packet(
                        residual_tile,
                        row * self.residual_row_stride + channel,
                        cutlass.Float8E4M3FN,
                        packet,
                        cute.AddressSpace.smem,
                    )
                decoded = _decode_residual(codes)
                values = self._load_hidden_packet(
                    gates,
                    gate_offset,
                    cutlass.Float32,
                    packet,
                    cute.AddressSpace.smem,
                )
                if cutlass.const_expr(self.paired_up_sigmoid):
                    for pair in cutlass.range_constexpr(packet // 2):
                        lo, hi = 2 * pair, 2 * pair + 1
                        if cutlass.const_expr(self.warp_owned_gates):
                            factor0, factor1 = _mul_hidden_float_pair(
                                activation_scale,
                                activation_scale,
                                weights[lo],
                                weights[hi],
                            )
                            res0, res1 = _mul_hidden_float_pair(
                                decoded[lo],
                                decoded[hi],
                                input_scale,
                                input_scale,
                            )
                        else:
                            factor0 = activation_scale * weights[lo]
                            factor1 = activation_scale * weights[hi]
                            res0, res1 = (
                                decoded[lo] * input_scale,
                                decoded[hi] * input_scale,
                            )
                        # Keep the gate multiply scalar so it can absorb the
                        # half scale used by the tanh input.
                        gate0, gate1 = _up_sigmoid_pair(
                            values[lo] * factor0, values[hi] * factor1
                        )
                        totals[lo] = totals[lo] + res0 * gate0
                        totals[hi] = totals[hi] + res1 * gate1
                else:
                    for element in cutlass.range_constexpr(packet):
                        factor = activation_scale * weights[element]
                        xv = decoded[element] * input_scale
                        totals[element] = totals[element] + xv * _up_sigmoid_tanh(
                            values[element] * factor
                        )
            packed = cute.make_rmem_tensor(packet, out.element_type)
            for element in cutlass.range_constexpr(packet):
                packed[element] = (totals[element] / self.g).to(out.element_type)
            pointer = cute.make_ptr(
                out.element_type,
                (out.iterator + global_row * self.h + hidden).toint(),
                cute.AddressSpace.gmem,
                assumed_align=2 * packet,
            )
            target = cute.make_tensor(pointer, cute.make_layout(packet))
            atom = cute.make_copy_atom(
                CopyUniversalOp(), out.element_type, num_bits_per_copy=16 * packet
            )
            cute.copy(atom, packed, target)

    @cute.experimental.jit
    def _write_up_gate_row_pairs(self, gates, values, coords):
        # The M128xN16 TMEM reader owns one channel and consecutive rows.
        # Pair those rows within each channel so both sides have contiguous packets.
        for pair in cutlass.range_constexpr(cute.size(values) // 2):
            channel, row = coords[2 * pair]
            target = (
                gates.iterator
                + row * self.up_channels
                + self._up_gate_channel(channel, row) * 2
            )
            address = cutlass.Uint32(target.toint())
            inst = "st.shared.f32 [$0], $1; st.shared.f32 [$0+4], $2;"
            llvm.inline_asm(
                None,
                [
                    address.ir_value(),
                    values[2 * pair].ir_value(),
                    values[2 * pair + 1].ir_value(),
                ],
                inst,
                "r,f,f,~{memory}",
                has_side_effects=True,
                asm_dialect=0,
            )

    @cute.experimental.jit
    def _mix_hidden_row_pairs(
        self, gates, residual_tile, cached, out, x, tid, hidden_tile, global_base
    ):
        # A lane owns two adjacent hidden channels in each of two token rows.
        lanes_per_pair = self.up_hidden // 2
        row = 2 * (tid // lanes_per_pair)
        column = 2 * (tid % lanes_per_pair)
        global_row = global_base + cutlass.Int64(row)
        if global_row < x.shape[0]:
            activation_scales = self._load_hidden_packet(
                cached,
                row,
                cutlass.Float32,
                2,
                cute.AddressSpace.smem,
            )
            input_scales = self._load_hidden_packet(
                cached,
                self.up_rows + row,
                cutlass.Float32,
                2,
                cute.AddressSpace.smem,
            )
            totals = cute.make_rmem_tensor((2, 2), cutlass.Float32)
            totals.fill(0.0)
            for branch in cutlass.range_constexpr(self.g):
                channel = branch * self.up_hidden + column
                values = self._load_hidden_packet(
                    gates,
                    row * self.up_channels + 2 * channel,
                    cutlass.Float32,
                    4,
                    cute.AddressSpace.smem,
                )
                weights = self._load_hidden_packet(
                    cached,
                    2 * self.up_rows + channel,
                    cutlass.Float32,
                    2,
                    cute.AddressSpace.smem,
                )
                for row_part in cutlass.range_constexpr(2):
                    offset = (row + row_part) * self.residual_row_stride + channel
                    codes = self._load_hidden_packet(
                        residual_tile,
                        offset,
                        cutlass.Float8E4M3FN,
                        2,
                        cute.AddressSpace.smem,
                    )
                    decoded = _decode_residual(codes)
                    for element in cutlass.range_constexpr(2):
                        factor = activation_scales[row_part] * weights[element]
                        xv = decoded[element] * input_scales[row_part]
                        gate = values[2 * element + row_part]
                        totals[element, row_part] = totals[
                            element, row_part
                        ] + xv * _up_sigmoid_tanh(gate * factor)
            hidden = hidden_tile * self.up_hidden + column
            for row_part in cutlass.range_constexpr(2):
                if global_row + row_part < x.shape[0]:
                    packed = cute.make_rmem_tensor(2, out.element_type)
                    for element in cutlass.range_constexpr(2):
                        packed[element] = (totals[element, row_part] / self.g).to(
                            out.element_type
                        )
                    pointer = cute.make_ptr(
                        out.element_type,
                        (
                            out.iterator + (global_row + row_part) * self.h + hidden
                        ).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    )
                    target = cute.make_tensor(pointer, cute.make_layout(2))
                    atom = cute.make_copy_atom(
                        CopyUniversalOp(), out.element_type, num_bits_per_copy=32
                    )
                    cute.copy(atom, packed, target)

    @cute.experimental.jit
    def _up(
        self,
        activation,
        alpha,
        x,
        xs,
        weight,
        ws,
        out,
        arena,
        bars,
        base,
        hidden_tile,
        global_base,
        weights_prepared,
        cached,
        prepared_residual,
        activation_counter,
        activation_target,
        residual_atom,
        residual_coords,
        activation_atom,
        activation_coords,
        up_weight_map,
    ):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        scales_prepared = cutlass.Boolean(False)
        if cutlass.const_expr(self.prefill_scales):
            scales_prepared = weights_prepared
        residual_tile = prepared_residual
        residual = None
        if cutlass.const_expr(self.residual_in_registers):
            values_per_lane = (
                self.up_hidden_columns
                if self.vector_hidden_up
                else self.up_rows // (self.compute_threads // self.up_hidden)
            )
            residual = cute.make_rmem_tensor(
                (values_per_lane, self.g), cutlass.Float8E4M3FN
            )
        elif cutlass.const_expr(
            not self.token_major_up and self.residual_issue == "up"
        ):
            residual_tile = self._residual_tile()
        dtype = cutlass.Float8E4M3FN
        mma = self._make_up_mma()
        tiler = (*self.up_mn, self.up_k)
        al = sm100_utils.make_smem_layout_a(mma, tiler, dtype, self.up_stages)
        bl = sm100_utils.make_smem_layout_b(mma, tiler, dtype, self.up_stages)
        sw = _operand_at(arena, bl, 0)
        sa0 = _operand_at(arena, al, cute.cosize(bl))
        sa1 = _operand_at(arena, al, cute.cosize(bl) + cute.cosize(al))
        alpha_ptr = None
        if cutlass.const_expr(self.bundle_alpha):
            # The full packed activation follows the aligned weight operand.
            # The row scales occupy the next 16-byte-aligned region.
            sa0 = _operand_at(arena, al, 0)
            sw = _operand_at(arena, bl, cute.cosize(al))
            alpha_ptr = cute.make_ptr(
                cutlass.Float32,
                (sw.iterator + self.up_rows * self.up_k).toint(),
                cute.AddressSpace.smem,
                assumed_align=16,
            )
        full, empty = bars, bars + self.up_stages
        done = empty + self.up_empty_stages
        read_done = done + self.up_parts
        if not weights_prepared:
            self._init_up_barriers(bars, False)
        tcgen05_fence("after_thread_sync")
        av = cute.make_tensor(
            activation.iterator,
            cute.make_layout(
                (activation.shape[0], (64, self.r // 64), 1),
                stride=(self.r, (1, 64), 0),
            ),
        )
        wv = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (
                    (self.up_hidden, self.g, self.h // self.up_hidden),
                    (64, self.r // 64),
                    1,
                ),
                stride=((self.r, self.h * self.r, self.up_hidden * self.r), (1, 64), 0),
            ),
        )
        gx0 = cute.local_tile(av, (self.up_rows, self.up_k), (0, None, 0))
        gx1 = cute.local_tile(av, (self.up_rows, self.up_k), (1, None, 0))
        gw = cute.local_tile(wv, (self.up_channels, self.up_k), (hidden_tile, None, 0))
        if cutlass.const_expr(self.token_major_up):
            ga0, ga1, gb = gx0, gx1, gw
            ma = cute_ext.get_cta_v_map_ab(av, tiler, mma, "A")
            mb = cute_ext.get_cta_v_map_ab(wv, tiler, mma, "B")
        else:
            ga0, ga1, gb = gw, gw, gx0
            ma = cute_ext.get_cta_v_map_ab(wv, tiler, mma, "A")
            mb = cute_ext.get_cta_v_map_ab(av, tiler, mma, "B")
        if warp == self.producer_warp:
            if cutlass.const_expr(
                self.activation_row_tags and not self.stamped_activation
            ):
                _wait_activation_rows(
                    activation_counter,
                    activation_target,
                    self.n,
                    self.activation_retry_ns,
                )
            elif cutlass.const_expr(self.split_activation_counter):
                _wait_activation_rows(
                    activation_counter, activation_target, 2, self.activation_retry_ns
                )
                self._join_activation_readers()
            elif cutlass.const_expr(self.copied_alpha):
                if cute.arch.lane_idx() == 0:
                    if cutlass.const_expr(self.activation_retry_ns > 0):
                        _wait_activation_with_delay(
                            activation_counter,
                            activation_target,
                            self.activation_retry_ns,
                        )
                    else:
                        _wait_activation(activation_counter, activation_target)
                cute.arch.sync_warp()
            elif cutlass.const_expr(self.activation_handoff == "warp"):
                if cute.arch.lane_idx() == 0:
                    if cutlass.const_expr(self.activation_retry_ns > 0):
                        _wait_activation_with_delay(
                            activation_counter,
                            activation_target,
                            self.activation_retry_ns,
                        )
                    else:
                        _wait_activation(activation_counter, activation_target)
                self._join_activation_readers()
            if cutlass.const_expr(self.residual_tma):
                issue_residual = not weights_prepared
                if cutlass.const_expr(self.residual_issue == "up"):
                    issue_residual = cutlass.Boolean(True)
                if issue_residual:
                    self._issue_residual_tma(
                        residual_atom,
                        residual_coords,
                        residual_tile,
                        bars + self.up_barrier_count - 1,
                        hidden_tile,
                        global_base,
                    )
            for tile in cutlass.range_constexpr(self.up_stages):
                slot = tile
                load_a = cutlass.Boolean(True)
                load_b = cutlass.Boolean(not self.stamped_activation)
                if cutlass.const_expr(self.token_major_up):
                    load_b = not weights_prepared
                else:
                    load_a = not weights_prepared
                if not weights_prepared:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + slot,
                            self.up_first_tx_bytes,
                        )
                if load_b:
                    if cutlass.const_expr(self.up_activation_transport == "native"):
                        cute_ext.tma_load(
                            gb[None, None, tile],
                            sw[None, None, None, slot],
                            (full + slot).value,
                            cta_v_map=mb,
                            update_expect_tx=False,
                        )
                    elif cutlass.const_expr(self.up_activation_delivery != "native"):
                        self._issue_deferred_activation(
                            sw.iterator,
                            activation.iterator,
                            alpha.iterator,
                            cached.iterator,
                            full + slot,
                        )
                    else:
                        self._issue_packed_activation(
                            arena,
                            activation,
                            full + slot,
                            activation_atom,
                            activation_coords,
                            self.up_rows * self.up_k,
                        )
                        if cutlass.const_expr(self.activation_row_tags):
                            # Start the bulk transfer before joining the alpha readers.
                            self._join_activation_readers()
                if load_a:
                    if cutlass.const_expr(self.warp_owned_gates):
                        self._load_warp_grouped_weights(
                            sa0[None, None, None, slot],
                            full + slot,
                            up_weight_map,
                            hidden_tile,
                        )
                    else:
                        cute_ext.tma_load(
                            ga0[None, None, tile],
                            sa0[None, None, None, slot],
                            (full + slot).value,
                            cta_v_map=ma,
                            update_expect_tx=False,
                        )
                if cutlass.const_expr(self.up_parts == 2):
                    cute_ext.tma_load(
                        ga1[None, None, tile],
                        sa1[None, None, None, slot],
                        (full + slot).value,
                        cta_v_map=ma,
                        update_expect_tx=False,
                    )
            for tile in cutlass.range(self.up_stages, self.up_tiles, unroll=1):
                slot = tile % self.up_stages
                cute.arch.mbarrier_wait(empty + slot, (tile // self.up_stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + slot,
                        (self.up_channels + self.up_rows * self.up_parts) * self.up_k,
                    )
                cute_ext.tma_load(
                    gb[None, None, tile],
                    sw[None, None, None, slot],
                    (full + slot).value,
                    cta_v_map=mb,
                    update_expect_tx=False,
                )
                cute_ext.tma_load(
                    ga0[None, None, tile],
                    sa0[None, None, None, slot],
                    (full + slot).value,
                    cta_v_map=ma,
                    update_expect_tx=False,
                )
                if cutlass.const_expr(self.up_parts == 2):
                    cute_ext.tma_load(
                        ga1[None, None, tile],
                        sa1[None, None, None, slot],
                        (full + slot).value,
                        cta_v_map=ma,
                        update_expect_tx=False,
                    )
        elif warp == self.mma_warp:
            tm = mma.get_slice(0)
            ad0, bd = tm.make_fragment_A(sa0), tm.make_fragment_B(sw)
            ad1 = ad0
            if cutlass.const_expr(self.up_parts == 2):
                ad1 = tm.make_fragment_A(sa1)
            layout = cute_ext.make_tmem_layout_acc(
                mma, self.up_mn, acc_stage=self.up_parts
            )
            all_acc = cute.make_tensor(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base), layout
            )
            acc0 = all_acc[None, None, None, 0]
            acc1 = acc0
            if cutlass.const_expr(self.up_parts == 2):
                acc1 = all_acc[None, None, None, 1]
            mma.set(tcgen05.Field.ACCUMULATE, False)
            for tile in cutlass.range(self.up_tiles - self.up_tail_tiles, unroll=1):
                slot = tile % self.up_stages
                cute.arch.mbarrier_wait(full + slot, tile // self.up_stages % 2)
                for block in cutlass.range_constexpr(cute.size(ad0, mode=[2])):
                    if cutlass.const_expr(
                        (
                            self.up_mma_before_suffix
                            and block
                            == self.up_activation_prefix
                            * cute.size(ad0, mode=[2])
                            // (self.up_rows * self.up_k)
                        )
                        or (self.stamped_activation and block == 2)
                    ):
                        cute.arch.mbarrier_wait(bars + self.up_barrier_count - 1, 0)
                    cute.gemm(
                        mma,
                        acc0,
                        ad0[None, None, block, slot],
                        bd[None, None, block, slot],
                        acc0,
                    )
                    if cutlass.const_expr(self.up_parts == 2):
                        cute.gemm(
                            mma,
                            acc1,
                            ad1[None, None, block, slot],
                            bd[None, None, block, slot],
                            acc1,
                        )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
            if cutlass.const_expr(self.up_parts == 2):
                for tile in cutlass.range(
                    self.up_tiles - self.up_tail_tiles, self.up_tiles, unroll=1
                ):
                    slot = tile % self.up_stages
                    cute.arch.mbarrier_wait(full + slot, tile // self.up_stages % 2)
                    for block in cutlass.range_constexpr(cute.size(ad0, mode=[2])):
                        cute.gemm(
                            mma,
                            acc0,
                            ad0[None, None, block, slot],
                            bd[None, None, block, slot],
                            acc0,
                        )
                    # Retain these last operand slots for the other token part.
                with cute.arch.elect_one():
                    tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
                for tile in cutlass.range(
                    self.up_tiles - self.up_tail_tiles, self.up_tiles, unroll=1
                ):
                    slot = tile % self.up_stages
                    for block in cutlass.range_constexpr(cute.size(ad1, mode=[2])):
                        cute.gemm(
                            mma,
                            acc1,
                            ad1[None, None, block, slot],
                            bd[None, None, block, slot],
                            acc1,
                        )
                with cute.arch.elect_one():
                    tcgen05.commit(done + 1, None, tcgen05.CtaGroup.ONE)
            else:
                with cute.arch.elect_one():
                    tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            if cutlass.const_expr(
                not self.use_cta_reader_handoff or self.warp_owned_gates
            ):
                cute.arch.mbarrier_wait(read_done, 0)
        elif tid < self.compute_threads:
            if cutlass.const_expr(self.stamped_activation):
                if tid < self.activation_copy_threads:
                    self._copy_stamped_activation(
                        arena,
                        activation,
                        full,
                        activation_target,
                        hidden_tile,
                        tid,
                        cached,
                    )
            if cutlass.const_expr(self.token_major_up):
                if cutlass.const_expr(not self.up_register_rows):
                    if tid < self.n:
                        global_row = global_base + cutlass.Int64(tid)
                        a_scale, x_scale = cutlass.Float32(0.0), cutlass.Float32(0.0)
                        if global_row < x.shape[0]:
                            a_scale = _load_f32(alpha.iterator + cutlass.Int64(tid))
                            x_scale = _x_scale(xs, global_row)
                        cached[tid] = a_scale
                        cached[self.n + tid] = x_scale
                if tid < self.up_channels:
                    if not scales_prepared:
                        channel = (
                            (tid // self.up_hidden) * self.h
                            + hidden_tile * self.up_hidden
                            + tid % self.up_hidden
                        )
                        cached[2 * self.n + tid] = ws[
                            channel if self.up_per_channel else 0
                        ]
                _epilogue_barrier(self.compute_threads, cutlass.Int32(5))
                self._mix_token_major_up(
                    mma,
                    base,
                    x,
                    xs,
                    alpha,
                    out,
                    cached,
                    tid,
                    hidden_tile,
                    global_base,
                    done,
                    read_done,
                    None,
                    row_scale_source=(
                        "global" if self.up_register_rows else "shared_element"
                    ),
                    weight_cache_offset=2 * self.n,
                    vector_weight_scales=False,
                    prefetched_branches=0,
                    wait_once=False,
                    release_after_reads=False,
                )
                tcgen05_fence("before_thread_sync")
                if cutlass.const_expr(not self.use_cta_reader_handoff):
                    cute.arch.mbarrier_arrive(read_done)
            else:
                async_scales = cutlass.Boolean(False)
                if cutlass.const_expr(self.async_up_scales):
                    async_scales = (ws.iterator.toint() & 15) == 0
                    if async_scales:
                        copy_scales = not weights_prepared
                        if cutlass.const_expr(self.up_scale_issue == "up_async"):
                            copy_scales = cutlass.Boolean(True)
                        if copy_scales:
                            self._issue_up_scale_copy(ws, cached, hidden_tile)
                        scales_prepared = cutlass.Boolean(True)
                if cutlass.const_expr(self.residual_issue == "register_before_mma"):
                    self._load_up_residual(x, residual, hidden_tile, global_base, tid)
                if cutlass.const_expr(
                    not self.residual_tma and not self.residual_in_registers
                ):
                    if cutlass.const_expr(self.residual_issue == "up"):
                        self._issue_residual_copy(
                            x,
                            residual_tile,
                            hidden_tile,
                            global_base,
                            0,
                            self.compute_threads,
                        )
                    elif not weights_prepared:
                        # A later task on this CTA needs a different residual tile.
                        self._issue_residual_copy(
                            x,
                            residual_tile,
                            hidden_tile,
                            global_base,
                            0,
                            self.compute_threads,
                        )
                if cutlass.const_expr(
                    self.activation_handoff != "grid"
                    and not self.copied_alpha
                    and not self.stamped_activation
                ):
                    if warp < self.up_scale_warps:
                        self._join_activation_readers()
                if cutlass.const_expr(self.paired_activation_alpha):
                    self._read_activation_scale_pair(alpha, cached, tid)
                    self._cache_hidden_input_scale(xs, cached, x, global_base, tid)
                elif cutlass.const_expr(not self.stamped_activation):
                    if tid < self.up_rows:
                        global_row = global_base + cutlass.Int64(tid)
                        activation_scale = cutlass.Float32(0.0)
                        input_scale = cutlass.Float32(0.0)
                        if global_row < x.shape[0]:
                            if cutlass.const_expr(not self.copied_alpha):
                                activation_scale = _load_f32(
                                    alpha.iterator + cutlass.Int64(tid)
                                )
                            input_scale = _x_scale(xs, global_row)
                        if cutlass.const_expr(not self.copied_alpha):
                            cached[tid] = activation_scale
                        cached[self.up_rows + tid] = input_scale
                if cutlass.const_expr(not self.separate_up_scale_warp):
                    if tid < 128:
                        if not scales_prepared:
                            channel = (tid // 32) * self.h + hidden_tile * 32 + tid % 32
                            cached[2 * self.up_rows + tid] = ws[
                                channel if self.up_per_channel else 0
                            ]
                else:
                    self._read_hidden_weight_scales(ws, cached, hidden_tile, tid)
                if cutlass.const_expr(not self.warp_owned_gates):
                    cute.arch.mbarrier_wait(done, 0)
                if cutlass.const_expr(self.copied_alpha):
                    if tid < self.up_rows:
                        cute.arch.mbarrier_wait(full, 0)
                        if cutlass.const_expr(self.bundle_alpha):
                            scales = cute.make_tensor(
                                alpha_ptr, cute.make_layout(self.up_rows)
                            )
                            cached[tid] = scales[tid]
                if cutlass.const_expr(self.warp_owned_gates):
                    # Residual copies and cached scales are shared across warps.
                    # Complete those inputs while the independent MMA runs.
                    _wait_residual()
                    _epilogue_barrier(self.compute_threads, cutlass.Int32(12))
                    cute.arch.mbarrier_wait(done, 0)
                if cutlass.const_expr(self.warp_owned_gates):
                    tcgen05_fence("after_thread_sync")
                gates = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float32,
                        arena.iterator.toint(),
                        cute.AddressSpace.smem,
                        assumed_align=1024,
                    ),
                    cute.make_layout((self.up_rows, 128), stride=(128, 1)),
                )
                for token_slice in cutlass.range_constexpr(self.hidden_reader_passes):
                    acc, reader, values, coords = self._hidden_reader(
                        mma, base, x, tid, token_slice, None
                    )
                    cute_ext.partition_and_copy(reader, acc, values)
                    cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(self.up_gate_exchange != "native"):
                        self._write_up_gate_row_pairs(gates, values, coords)
                    else:
                        for i in cutlass.range_constexpr(cute.size(values)):
                            channel, row = coords[i]
                            gates[row, self._up_gate_channel(channel, row)] = values[i]
                tcgen05_fence("before_thread_sync")
                if cutlass.const_expr(self.warp_owned_gates):
                    # The gate producer and consumer are the same warp.
                    # The MMA owner can retire TMEM after each warp signals.
                    cute.arch.sync_warp()
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(read_done)
                elif cutlass.const_expr(self.residual_tma):
                    if cutlass.const_expr(self.async_up_scales):
                        if async_scales and tid < self.up_channels // 4:
                            _wait_residual()
                    if tid == 0:
                        cute.arch.mbarrier_wait(bars + self.up_barrier_count - 1, 0)
                elif cutlass.const_expr(not self.residual_in_registers):
                    _wait_residual()
                if cutlass.const_expr(not self.use_cta_reader_handoff):
                    cute.arch.mbarrier_arrive(read_done)
        if cutlass.const_expr(not self.warp_owned_gates):
            cute.arch.sync_threads()
        if cutlass.const_expr(not self.token_major_up):
            if tid < self.compute_threads:
                gates = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float32,
                        arena.iterator.toint(),
                        cute.AddressSpace.smem,
                        assumed_align=1024,
                    ),
                    cute.make_layout((self.up_rows, 128), stride=(128, 1)),
                )
                if cutlass.const_expr(self.up_gate_exchange != "native"):
                    self._mix_hidden_row_pairs(
                        gates,
                        residual_tile,
                        cached,
                        out,
                        x,
                        tid,
                        hidden_tile,
                        global_base,
                    )
                elif cutlass.const_expr(self.vector_hidden_up):
                    self._mix_hidden_columns(
                        gates,
                        residual_tile,
                        residual,
                        cached,
                        out,
                        x,
                        tid,
                        hidden_tile,
                        global_base,
                    )
                else:
                    col = tid % 32
                    hidden = hidden_tile * 32 + col
                    scales = cute.make_rmem_tensor(self.g, cutlass.Float32)
                    for branch in cutlass.range_constexpr(self.g):
                        channel = branch * self.h + hidden
                        scales[branch] = cached[2 * self.up_rows + branch * 32 + col]
                    for row_part in cutlass.range_constexpr(
                        self.up_rows // (self.compute_threads // 32)
                    ):
                        row = tid // 32 + row_part * (self.compute_threads // 32)
                        global_row = global_base + cutlass.Int64(row)
                        if global_row < x.shape[0]:
                            activation_scale = cached[row]
                            input_scale = cached[self.up_rows + row]
                            total = cutlass.Float32(0.0)
                            for branch in cutlass.range_constexpr(self.g):
                                channel = branch * self.h + hidden
                                gate = gates[row, branch * 32 + col]
                                factor = activation_scale * scales[branch]
                                if cutlass.const_expr(self.residual_in_registers):
                                    code = residual[row_part, branch]
                                else:
                                    code = residual_tile[row, branch * 32 + col]
                                res = code.to(cutlass.Float32) * input_scale
                                total = total + res * _up_sigmoid_tanh(gate * factor)
                            out[global_row, hidden] = (total / self.g).to(
                                out.element_type
                            )
