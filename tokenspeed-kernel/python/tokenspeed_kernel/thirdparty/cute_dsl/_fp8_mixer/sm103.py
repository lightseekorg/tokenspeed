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

"""SM103 schedules for the prequantized FP8 mixer."""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
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
    _arrive_phase_release,
    _commit_residual,
    _copy_residual_16,
    _decode_residual,
    _grid_barrier,
    _init_mma_barriers,
    _load_f32,
    _make_down_fragments,
    _make_down_operands,
    _make_k_major_mma,
    _mma_barriers,
    _mul_hidden_float_pair,
    _operand_at,
    _quant_scale,
    _quantized_value,
    _reduce_split_partials,
    _release,
    _run_cute_mma_ring,
    _store_private_epoch,
    _tma_matrix_view,
    _wait_residual,
    _x_scale,
)
from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.raw_issue import (
    fp8_descriptor_at,
    fp8_instruction,
    fp8_mma_encoded,
)
from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.resident import (
    MixerOperands,
    QuantizationConfig,
    ResidentDown,
    ResidentMixer,
    _mixer_cta,
    _mixer_shared,
    _read_down_partials,
)
from tokenspeed_kernel.thirdparty.cute_dsl.hc_fp8 import (
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
def _wait_activation_rows(tags, expected, rows: cutlass.Constexpr):
    lane = cute.arch.lane_idx()
    ready = cutlass.Boolean(False)
    while not ready:
        local_ready = cutlass.Boolean(True)
        if lane < rows:
            local_ready = _acquire(tags + cutlass.Int64(lane)) >= expected
        ready = cute.arch.vote_all_sync(local_ready)
    # The TMA is issued by one lane after all row acquires.
    cute.arch.sync_warp()


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


class _NarrowDownPhase(ResidentDown):
    def __init__(
        self,
        *,
        producer_warp,
        mma_warp,
        use_cta_reader_handoff,
        static_partial_store,
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
        if omit_down_empty and self.tiles != self.stages:
            raise ValueError("Omitting Down empty events requires a resident split")
        self.omit_down_empty = omit_down_empty
        self.empty_stages = 0 if omit_down_empty else self.stages
        self.reader_arrivals = 0 if use_cta_reader_handoff else self.epilogue_threads
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
        return _tma_matrix_view(weight, self.projection_rows, self.inner, self.inner)

    @cute.experimental.jit
    def make_weight_atom(self, weight):
        dtype = cutlass.Float8E4M3FN
        tiler = (self.m, self.n, self.k)
        mma = _make_k_major_mma(dtype, (self.m, self.n))
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
    def _load_weight(self, weights, full, stage, input_tile):
        atom, source, target, cache_policy = weights
        cute.copy(
            atom,
            source[None, input_tile],
            target[None, stage],
            tma_bar_ptr=full + stage,
            cache_policy=cache_policy,
        )

    @cute.experimental.jit
    def _load_down_stage(self, operands, full, stage, tile):
        weight, (gx, sx, map_x) = operands
        self._load_weight(weight, full, stage, tile)
        self._load_input(gx, sx, full, stage, tile, map_x)

    @cute.experimental.jit
    def project(
        self,
        x,
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
        full, empty, done, read_done = _mma_barriers(
            barriers, self.stages, self.empty_stages, 1
        )
        if not entry_prefilled:
            if warp == self.mma_warp:
                with cute.arch.elect_one():
                    _init_mma_barriers(
                        barriers,
                        self.stages,
                        self.empty_stages,
                        1,
                        1,
                        0,
                        self.reader_arrivals,
                    )
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
        tcgen05_fence("after_thread_sync")
        x_view = _tma_matrix_view(x, x.shape[0], self.inner, self.inner)
        gx = cute.local_tile(x_view, (self.n, self.k), (token_tile, None, 0))
        map_x = cute_ext.get_cta_v_map_ab(x_view, tiler, mma, "B")
        if warp == self.producer_warp:
            weight_target, weight_source = self._partition_weight(
                weight_atom, weight_coords, sw, output_tile
            )
            weights = (
                weight_atom,
                weight_source,
                weight_target,
                _down_weight_cache_policy(),
            )
            refill_hint = None
            if cutlass.const_expr(len(self.input_refill_hint) > 0):
                refill_hint = (input_hint_map, output_tile, token_tile)
            self._feed_down_ring(
                (weights, (gx, sx, map_x)),
                full,
                empty,
                split,
                entry_prefilled,
                refill_hint,
            )
        elif warp == self.mma_warp:
            acc, a_desc, b_desc = _make_down_fragments(
                mma.get_slice(0),
                tmem_base,
                sw,
                sx,
                mma.partition_shape_C((self.m, self.n)),
            )
            _run_cute_mma_ring(
                mma,
                acc,
                a_desc,
                b_desc,
                full,
                empty,
                self.tiles,
                self.stages,
                None,
                not self.omit_down_empty,
            )
            with cute.arch.elect_one():
                tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
            if cutlass.const_expr(not self.use_cta_reader_handoff):
                cute.arch.mbarrier_wait(read_done, 0)
        elif tid < self.epilogue_threads:
            cute.arch.mbarrier_wait(done, 0)
            values, coords = _read_down_partials(
                mma,
                tmem_base,
                partials,
                tid,
                (self.m, self.n, self.k),
                self.epilogue_groups,
            )
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
    def _load_input(self, source, target, full, stage, tile, mapping):
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
        full = barriers
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                _init_mma_barriers(
                    barriers,
                    self.stages,
                    self.empty_stages,
                    1,
                    1,
                    0,
                    self.reader_arrivals,
                )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        x_view = _tma_matrix_view(x, x.shape[0], self.inner, self.inner)
        gx = cute.local_tile(x_view, (self.n, self.k), (token_tile, None, 0))
        map_x = cute_ext.get_cta_v_map_ab(x_view, tiler, mma, "B")
        stage_bytes = (self.m + self.n) * self.k
        if warp == self.producer_warp:
            weight_target, weight_source = self._partition_weight(
                weight_atom, weight_coords, sw, output_tile
            )
            weights = (
                weight_atom,
                weight_source,
                weight_target,
                _down_weight_cache_policy(),
            )
            # The enclosing kernel already waited for input and weight producers.
            if cutlass.const_expr(self.entry_input_ring_first):
                # Fetch the reused input across the resident ring before
                # streaming weights. Each stage still waits for both operands.
                for stage in cutlass.range_constexpr(self.stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + stage, stage_bytes
                        )
                    self._load_input(
                        gx, sx, full, stage, split * self.tiles + stage, map_x
                    )
            for stage in cutlass.range_constexpr(self.stages):
                tile = split * self.tiles + stage
                if cutlass.const_expr(not self.entry_input_ring_first):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + stage, stage_bytes
                        )
                self._load_weight(weights, full, stage, tile)
                if cutlass.const_expr(not self.entry_input_ring_first):
                    self._load_input(gx, sx, full, stage, tile, map_x)


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


class NarrowFp8Mixer(ResidentMixer):
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
    ):
        has_injection = p == r + g
        small = token_bucket == 16
        medium = token_bucket == 32
        row_handoff = token_bucket == 64
        long_tile = token_bucket == 128
        paired_tokens = token_bucket == 256
        stamped = small and has_injection
        warp_owned = medium and has_injection
        split_counter = row_handoff and has_injection

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
        self.compute_threads = 256 if token_bucket <= 32 else 512
        self.reader_groups = self.compute_threads // 128
        self.producer_warp = self.compute_threads // 32
        self.mma_warp = self.producer_warp + 1
        self.threads = self.compute_threads + 64
        self.quant_rows = 4 if paired_tokens else 2 if long_tile else 1
        self.quant_threads = self.compute_threads // self.quant_rows
        self.quant_columns = max(
            2,
            1
            << (((r + self.quant_threads - 1) // self.quant_threads - 1).bit_length()),
        )
        self.quant_max_threads = (
            r // self.quant_columns if row_handoff else self.quant_threads
        )
        # The warp after the FP8 writers handles the injection columns.
        self.quant_handoff_threads = self.quant_max_threads + 32
        self.quant_groups = (self.n + self.quant_rows - 1) // self.quant_rows
        self.quant_warp_handoff = row_handoff
        self.quant_copy_warp = 5 if warp_owned else -1
        self.quant_early_scales = stamped
        self.quant_warp_down = stamped
        packed_codes = has_injection and token_bucket <= 64
        self.quant_activation_store = "pack_max" if packed_codes else "native"

        self.up_hidden = 32
        self.up_channels = self.up_hidden * g
        self.up_parts = 2 if paired_tokens else 1
        self.token_major_up = paired_tokens
        self.up_rows = token_bucket // self.up_parts
        self.up_mn = (
            (self.up_rows, self.up_channels)
            if self.token_major_up
            else (self.up_channels, self.up_rows)
        )
        self.hidden_reader_columns = min(32, token_bucket // self.reader_groups)
        self.reader_columns = self.up_hidden // self.reader_groups
        self.up_k = 64 if long_tile else r
        self.up_tiles = self.up_stages = r // self.up_k
        self.use_cta_reader_handoff = not paired_tokens
        self.prefill_scales = small
        self.async_up_scales = long_tile and up_per_channel
        self.final_cta_join = not (warp_owned or split_counter)
        self.residual_issue = (
            "register_before_mma"
            if stamped
            else (
                "down_entry"
                if long_tile
                else "handoff" if medium or row_handoff else "up"
            )
        )
        self.residual_in_registers = stamped
        self.residual_tma = long_tile
        self.residual_alignment = 128 if long_tile else 16
        self.residual_row_stride = self.up_channels + (
            32 if split_counter else 16 if warp_owned else 0
        )
        self.up_hidden_columns = (
            2 if stamped else 4 if warp_owned or split_counter else 1
        )
        self.vector_hidden_up = self.up_hidden_columns > 1
        self.up_gate_exchange = "row_pairs_scalar" if split_counter else "native"
        self.warp_owned_gates = warp_owned
        self.up_output_threads = (
            self.up_rows * (self.up_hidden // self.up_hidden_columns)
            if self.vector_hidden_up
            else self.compute_threads
        )
        self.separate_up_scale_warp = split_counter and up_per_channel
        self.paired_activation_alpha = split_counter
        self.up_scale_warps = 1 if split_counter else (self.up_rows + 31) // 32

        self.activation_handoff = (
            "row_async_bulk"
            if stamped
            else "row_tags_quant" if warp_owned else "warp" if row_handoff else "grid"
        )
        self.split_activation_counter = split_counter
        self.activation_rows_per_counter = token_bucket // (2 if split_counter else 1)
        self.activation_row_gate = stamped or warp_owned
        self.quant_only_down_acquire = warp_owned
        # Align the row tags to a separate epoch line.
        self.extra_epoch_slots = (
            token_bucket + 15
            if self.activation_row_gate
            else 1 + int(split_counter) if row_handoff else 0
        )
        self.dedicated_quant_ctas = small
        self.activation_retry_ns = 64 if small else 0
        self.up_activation_transport = (
            "bulk"
            if stamped or medium or row_handoff
            else "packed_tma" if small else "native"
        )
        self.up_activation_delivery = (
            "bundle_deferred"
            if warp_owned
            else "private_deferred" if medium else "native"
        )
        self.copied_alpha = medium
        self.bundle_alpha = warp_owned
        self.prefetch_up_weights_on_down_retire = split_counter
        self.up_mma_before_suffix = split_counter
        # Packed SW64 activation stores one contiguous segment per K64 plane.
        self.up_activation_prefix = 3 * 64 * self.up_rows if split_counter else 0
        self.up_activation_suffix = self.up_rows * self.up_k - self.up_activation_prefix

        self.stamped_activation = stamped
        self.stamped_prefix_threads = 96 if stamped else 32
        self.stamped_suffix_threads = 96
        self.activation_copy_threads = (
            self.stamped_prefix_threads + self.stamped_suffix_threads
        )
        self.stamped_prefix_records = self.up_rows * 6
        self.activation_payload_records = (
            self.up_rows * (self.up_k // 64) * 6 if stamped else 0
        )
        self.up_full_arrivals = 1 + (
            self.stamped_prefix_threads // 32 if stamped else 0
        )
        fp8_threads = self.quant_max_threads if row_handoff else self.compute_threads
        self.quantization = QuantizationConfig(
            threads=fp8_threads,
            maximum_threads=self.quant_max_threads,
            injection_start=fp8_threads,
            column_stride=1,
            warp_max="redux",
            replicate_maximum=not paired_tokens,
            refined_scales=False,
            clamp="ptx_lower" if packed_codes else "both",
            activation_store=(
                "stamped" if stamped else "packed" if token_bucket <= 64 else "native"
            ),
            release_writes=not stamped and (warp_owned or row_handoff),
        )

        self.hidden_tiles = h // self.up_hidden
        self.up_tasks = self.hidden_tiles
        self.up_operand_bytes = (
            (self.up_channels + self.up_rows * self.up_parts)
            * self.up_k
            * self.up_stages
        )
        self.up_barrier_count = (
            self.up_stages
            + self.up_parts
            + int(paired_tokens or warp_owned)
            + int(self.residual_tma)
            + int(stamped or split_counter)
        )
        self.up_full_bytes = (
            self.up_channels + self.up_rows * self.up_parts
        ) * self.up_k
        if self.copied_alpha:
            self.up_full_bytes += self.up_rows * 4
        self.up_first_tx_bytes = self.up_full_bytes
        if stamped:
            self.up_first_tx_bytes -= self.up_rows * self.up_k
        if split_counter:
            self.up_first_tx_bytes -= self.up_activation_suffix
        if self.bundle_alpha:
            self.up_operand_bytes += self.up_rows * 4

        self.down_m = 64
        self.splits = (
            20 if stamped else 16 if token_bucket <= 32 else 4 if paired_tokens else 8
        )
        down_stages = 3 if paired_tokens else 2
        down_k = 512 if paired_tokens else g * h // (self.splits * down_stages)
        down_token_tile = 32 if row_handoff else min(64, token_bucket)
        self.down_token_parts = token_bucket // down_token_tile
        if g * h % (self.splits * down_k) or not (
            1 <= down_stages <= g * h // (self.splits * down_k)
        ):
            raise ValueError("Down stages must fit an evenly partitioned reduction")
        self.static_partial_store = medium or split_counter
        self.static_partial_stride = small or medium or split_counter
        self.aligned_partial_rows = stamped
        self.entry_down_input_ring_first = token_bucket <= 64
        self.down_release = "early_arrive" if small else "native"
        self.async_down_arrival = small or medium
        self.async_activation_arrival = small
        self.down_publish_before_join = stamped
        self.input_refill_hint = (3, 4) if has_injection and paired_tokens else ()
        self.down = _NarrowDownPhase(
            aligned_partial_rows=stamped,
            input_refill_hint=self.input_refill_hint,
            omit_down_empty=token_bucket <= 64,
            producer_warp=self.producer_warp,
            mma_warp=self.mma_warp,
            use_cta_reader_handoff=self.use_cta_reader_handoff,
            static_partial_store=self.static_partial_store,
            scratch_token_tile=token_bucket,
            entry_input_ring_first=self.entry_down_input_ring_first,
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
            epilogue_threads=self.compute_threads,
            shared_up_scales=False,
            interleave_tma=False,
        )
        self.down_tiles = (p + self.down_m - 1) // self.down_m
        self.down_tasks = self.down_token_parts * self.down_tiles * self.splits
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
        if self.residual_tma and self.up_barrier_count - 1 < self.down_barrier_count:
            raise ValueError(
                "The entry residual barrier must be outside the Down barriers"
            )
        self.barrier_count = max(self.down_barrier_count, self.up_barrier_count)
        # A conservative occupancy query includes alignment and epilogue scratch.
        self.shared_bytes = (
            self.arena_bytes
            + (0 if self.token_major_up else token_bucket * self.up_channels)
            + self.up_rows * (self.residual_row_stride - self.up_channels)
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
            min_blocks_per_mp=1,
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
        cached_scales = self._up_scale_storage(2 * self.n)
        prepared_residual = None
        if cutlass.const_expr(
            self.residual_issue != "up" and not self.residual_in_registers
        ):
            prepared_residual = self._up_residual_tile(
                self.residual_row_stride, self.residual_alignment
            )
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
                if cutlass.const_expr(self.quant_warp_handoff):
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
            if cutlass.const_expr(self.quant_only_down_acquire):
                wait_down = state.first_quant_group < self.quant_groups
            if cutlass.const_expr(self.quant_warp_down):
                wait_down = cutlass.Boolean(False)
            if cta.tid == 0 and wait_down:
                down_target = (phase_base + 1) * cta.count
                if cutlass.const_expr(self.dedicated_quant_ctas):
                    down_target = phase_base + state.down_arrivals
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
                    _arrive_activation_row(ready + row, tag_increment)
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
                if (
                    cta.tid == 0
                    and cta.pid >= self.up_tasks
                    and state.first_quant_group >= self.quant_groups
                ):
                    # An idle CTA waits for the previous Down count.
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
        # Join the Down readers before their leader publishes the partials.
        threads = self.down.epilogue_threads
        if tid < threads:
            llvm.inline_asm(
                None,
                [cutlass.Int32(6).ir_value(), cutlass.Int32(threads).ir_value()],
                "barrier.cta.sync $0, $1;",
                "r,r,~{memory}",
                has_side_effects=True,
                asm_dialect=0,
            )
            if tid == 0:
                _arrive_phase_release(root, self.async_down_arrival)

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
                while _load_relaxed_counter(root) < target:
                    pass
                # Acquire before reading the published Down partials.
                llvm.inline_asm(
                    None,
                    [],
                    "fence.acquire.gpu;",
                    "~{memory}",
                    has_side_effects=True,
                    asm_dialect=0,
                )
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
                packed_scale = _quant_scale(maximum, False)
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
                self._store_quant_alpha(io, row, _quant_scale(maximum, False), target)
        else:
            # Both adjacent codes stay in the same 16-byte swizzle block.
            offset = cutlass.Int64(
                cute.crd2idx(((row, col % 32), 0, col // 32, 0), activation_layout)
            )
            destination = io.activation.iterator + offset
            if cutlass.const_expr(self.quant_activation_store == "pack_max"):
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
        _, _, _, read_done = _mma_barriers(bars, self.up_stages, 0, self.up_parts)
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                _init_mma_barriers(
                    bars, self.up_stages, 0, self.up_parts, self.up_full_arrivals, 0, 0
                )
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
        mma, tiler, al, bl = self._up_operand_layouts()
        if cutlass.const_expr(self.token_major_up):
            sw = _operand_at(arena, bl, 0)
        else:
            if cutlass.const_expr(self.bundle_alpha):
                sw = _operand_at(arena, al, 0)
            else:
                sw = _operand_at(arena, al, cute.cosize(bl))
        wv = self._up_weight_view(weight)
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
                    self._load_up_tma_operand(
                        gw, sw, stage, stage, bars + stage, mapping
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
        self._copy_hidden_residual(
            x,
            residual_tile,
            hidden_tile,
            global_base,
            cute.arch.thread_idx()[0] - first_thread,
            copy_threads,
            self.residual_row_stride,
        )

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
    def _up_gate_channel(self, channel, row):
        if cutlass.const_expr(self.warp_owned_gates):
            return channel ^ (8 * (row % 4))
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
                if cutlass.const_expr(self.warp_owned_gates):
                    for pair in cutlass.range_constexpr(packet // 2):
                        lo, hi = 2 * pair, 2 * pair + 1
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
            residual_tile = self._up_residual_tile(
                self.residual_row_stride, self.residual_alignment
            )
        mma, tiler, al, bl = self._up_operand_layouts()
        sw, sa0, sa1 = self._up_operand_storage(arena, al, bl)
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
        full, _, done, read_done = _mma_barriers(bars, self.up_stages, 0, self.up_parts)
        if not weights_prepared:
            self._init_up_barriers(bars, False)
        tcgen05_fence("after_thread_sync")
        ga0, ga1, gb, ma, mb = self._up_tma_operands(
            mma, tiler, activation, weight, hidden_tile
        )
        if warp == self.producer_warp:
            if cutlass.const_expr(
                self.activation_row_gate and not self.stamped_activation
            ):
                _wait_activation_rows(activation_counter, activation_target, self.n)
            elif cutlass.const_expr(self.split_activation_counter):
                _wait_activation_rows(activation_counter, activation_target, 2)
                self._join_activation_readers()
            elif cutlass.const_expr(self.copied_alpha):
                if cute.arch.lane_idx() == 0:
                    _wait_activation(activation_counter, activation_target)
                cute.arch.sync_warp()
            elif cutlass.const_expr(self.activation_handoff == "warp"):
                if cute.arch.lane_idx() == 0:
                    _wait_activation(activation_counter, activation_target)
                self._join_activation_readers()
            if cutlass.const_expr(self.residual_tma):
                if not weights_prepared:
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
                        self._load_up_tma_operand(gb, sw, tile, slot, full + slot, mb)
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
                        if cutlass.const_expr(self.activation_row_gate):
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
                        self._load_up_tma_operand(ga0, sa0, tile, slot, full + slot, ma)
                if cutlass.const_expr(self.up_parts == 2):
                    self._load_up_tma_operand(ga1, sa1, tile, slot, full + slot, ma)
        elif warp == self.mma_warp:
            fragments = self._up_mma_fragments(mma, base, sa0, sa1, sw)
            mma.set(tcgen05.Field.ACCUMULATE, False)
            suffix = None
            if cutlass.const_expr(self.up_mma_before_suffix or self.stamped_activation):
                suffix_block = (
                    2
                    if self.stamped_activation
                    else self.up_activation_prefix
                    * cute.size(fragments.a[0], mode=[2])
                    // (self.up_rows * self.up_k)
                )
                suffix = (bars + self.up_barrier_count - 1, suffix_block)
            self._run_up_mma_tiles(
                mma, fragments, full, None, suffix, self.up_tiles, False
            )
            self._publish_up_done(done)
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
                    row_scale_source="global",
                    weight_cache_offset=2 * self.n,
                    vector_weight_scales=False,
                    prefetched_branches=0,
                    wait_once=False,
                    release_after_reads=False,
                )
                tcgen05_fence("before_thread_sync")
                cute.arch.mbarrier_arrive(read_done)
            else:
                async_scales = cutlass.Boolean(False)
                if cutlass.const_expr(self.async_up_scales):
                    async_scales = (ws.iterator.toint() & 15) == 0
                    if async_scales:
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
                    self._cache_up_row_scales(
                        alpha,
                        xs,
                        cached,
                        x,
                        global_base,
                        tid,
                        self.up_rows,
                        not self.copied_alpha,
                    )
                if cutlass.const_expr(not self.separate_up_scale_warp):
                    self._cache_up_weight_scales(
                        ws, cached, hidden_tile, tid, 2 * self.up_rows, scales_prepared
                    )
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
                gates = self._hidden_gate_tile(arena.iterator)
                values, coords = self._read_hidden_up(mma, base, x, tid, None)
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
        if cutlass.const_expr(not self.warp_owned_gates):
            cute.arch.sync_threads()
        if cutlass.const_expr(not self.token_major_up):
            if tid < self.compute_threads:
                gates = self._hidden_gate_tile(arena.iterator)
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
                    self._mix_hidden_major_up(
                        gates,
                        residual_tile,
                        cached,
                        out,
                        x.shape[0],
                        tid,
                        hidden_tile,
                        global_base,
                        self.compute_threads,
                    )


# Down projection tile, split count, K tile and stage count by token bucket.
_DOWN_TACTICS = {
    16: (64, 8, 512, 3),
    32: (64, 8, 512, 3),
    64: (64, 8, 256, 6),
    128: (64, 8, 256, 4),
    256: (128, 16, 128, 4),
}


@cute.experimental.jit
def _weight_eviction_policy(priority):
    # CUDA/PTX creates the opaque policy; no cache-policy bits are encoded here.
    value = llvm.inline_asm(
        T.i64(),
        [priority.ir_value()],
        "{ .reg .b64 normal, first, last; .reg .pred pfirst, plast;\n"
        "createpolicy.fractional.L2::evict_normal.b64 normal, 1.0;\n"
        "createpolicy.fractional.L2::evict_first.b64 first, 1.0;\n"
        "createpolicy.fractional.L2::evict_last.b64 last, 1.0;\n"
        "setp.eq.u32 pfirst, $1, 1; setp.eq.u32 plast, $1, 2;\n"
        "selp.b64 $0, first, normal, pfirst;\n"
        "selp.b64 $0, last, $0, plast; }",
        "=l,r",
        has_side_effects=False,
        asm_dialect=0,
    )
    return cutlass.Int64(value)


@cute.experimental.jit
def _explicit_tma_copy(destination, descriptor, mbar, coordinates, cache_policy):
    rank = len(coordinates)
    values = [
        cutlass.Int32(destination.toint()).ir_value(),
        descriptor.get_ptr().toint().ir_value(),
        cutlass.Int32(mbar.toint()).ir_value(),
    ]
    for coordinate in coordinates:
        values.append(cutlass.Int32(coordinate).ir_value())
    indices = ", ".join(f"${i + 3}" for i in range(rank))
    instruction = f"cp.async.bulk.tensor.{rank}d.shared::cluster.global.mbarrier::complete_tx::bytes"
    operands = f"[$0], [$1, {{{indices}}}], [$2]"
    constraints = "r,l,r," + ",".join(["r"] * rank)
    if cutlass.const_expr(cache_policy is not None):
        instruction += ".L2::cache_hint"
        operands += f", ${rank + 3}"
        constraints += ",l"
        values.append(cache_policy.ir_value())
    with cute.arch.elect_one():
        llvm.inline_asm(
            None,
            values,
            f"{instruction} {operands};",
            constraints + ",~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )


@cute.experimental.jit
def _transpose_store_packet(
    values, packet: cutlass.Constexpr, lane_shift: cutlass.Constexpr
):
    # Swap lane/element bits; every lane reaches each full-warp shuffle.
    lane = cute.arch.thread_idx()[0] % 32
    for bit in cutlass.range_constexpr(1 if packet == 2 else 2):
        register_mask = 1 << bit
        lane_mask = register_mask << lane_shift
        for low in cutlass.range_constexpr(packet):
            if cutlass.const_expr((low & register_mask) == 0):
                high = low | register_mask
                send = values[high]
                if (lane & lane_mask) != 0:
                    send = values[low]
                received = cute.arch.shuffle_sync_bfly(
                    send, lane_mask, mask=0xFFFFFFFF, mask_and_clamp=31
                )
                if (lane & lane_mask) != 0:
                    values[low] = received
                else:
                    values[high] = received


@cute.experimental.jit
def _up_weight_coordinates(
    hidden_tile,
    tile,
    channel_offset: cutlass.Constexpr,
    hidden_width: cutlass.Constexpr,
    k_width: cutlass.Constexpr,
    packed_hidden_tiles: cutlass.Constexpr,
):
    if cutlass.const_expr(packed_hidden_tiles > 0):
        return (
            0,
            channel_offset * hidden_width,
            tile * packed_hidden_tiles + hidden_tile,
        )
    return (tile * k_width, hidden_tile * hidden_width, channel_offset)


@cute.experimental.jit
def _issue_up_weight_from_down_arena(
    destination,
    barrier,
    descriptor,
    coordinates,
    cache_priority,
    transaction_bytes: cutlass.Constexpr,
    initialize: cutlass.Constexpr,
):
    if cutlass.const_expr(initialize):
        with cute.arch.elect_one():
            cute.arch.mbarrier_init(barrier, 1)
        cute.arch.mbarrier_init_fence()
        llvm.inline_asm(
            None,
            [],
            "fence.proxy.async.shared::cta;",
            "~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )
        cute.arch.sync_warp()
    with cute.arch.elect_one():
        # The original Up activation transfers complete this same stage.
        cute.arch.mbarrier_arrive_and_expect_tx(barrier, transaction_bytes)
    _explicit_tma_copy(
        destination,
        descriptor,
        barrier,
        coordinates,
        _weight_eviction_policy(cache_priority),
    )


def _down_retirement_events(start, size, weight_bytes, input_bytes, stages, tiles):
    end = start + size
    input_base = weight_bytes * stages
    events = []
    for slot in range(stages):
        weight_start = slot * weight_bytes
        input_start = input_base + slot * input_bytes
        if max(start, weight_start) < min(end, weight_start + weight_bytes) or max(
            start, input_start
        ) < min(end, input_start + input_bytes):
            last_phase = (tiles - 1 - slot) // stages
            events.append((slot, last_phase % 2))
    return tuple(events)


class _WideDownPhase(ResidentDown):
    def __init__(
        self,
        *,
        mma_k_unroll,
        issue_style,
        partial_store_packet,
        native_down_partials,
        up_reuse_stages,
        up_stage_bytes,
        up_barriers_at_entry,
        up_first_weight_copies,
        early_entry_weights,
        packed_up_hidden_tiles,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.partial_store_packet = partial_store_packet
        self.native_down_partials = native_down_partials
        self.mma_k_unroll = mma_k_unroll
        self.issue_style = issue_style
        self.up_reuse_stages = up_reuse_stages
        self.up_stage_bytes = up_stage_bytes
        self.up_barriers_at_entry = up_barriers_at_entry
        self.up_first_weight_copies = up_first_weight_copies
        self.early_entry_weights = early_entry_weights
        self.packed_up_hidden_tiles = packed_up_hidden_tiles
        self.producer_warp = 16
        self.mma_warp = 17

    @cute.experimental.jit
    def _load_down_stage(self, operands, full, stage, tile):
        sw, sx, weight_map, input_map, cache_priority, output_tile, token_tile = (
            operands
        )
        k_segment = tile * (self.k // 128)
        self._operand_load(
            sw[None, None, None, stage],
            full + stage,
            cache_priority=cache_priority,
            is_weight=True,
            descriptor=weight_map,
            coordinates=(0, output_tile * self.m, k_segment),
        )
        self._operand_load(
            sx[None, None, None, stage],
            full + stage,
            cache_priority=cache_priority,
            is_weight=False,
            descriptor=input_map,
            coordinates=(0, cutlass.Int64(token_tile) * self.n, k_segment),
        )

    @cute.experimental.jit
    def project(
        self,
        x,
        partials,
        arena,
        barriers,
        tmem_base,
        output_tile,
        split,
        token_tile,
        down_weight_map,
        down_input_map,
        cache_priority,
        entry_prefilled,
        prepare_up_weights,
        up_bars,
        up_weight_map,
        up_weight_priority,
        up_hidden_tile,
    ):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        tiler = (self.m, self.n, self.k)
        mma, sw, sx = _make_down_operands(arena, tiler, self.stages)
        full, empty, done, read_done = _mma_barriers(
            barriers, self.stages, self.stages, 1
        )
        if not entry_prefilled:
            if warp == self.mma_warp:
                with cute.arch.elect_one():
                    _init_mma_barriers(
                        barriers,
                        self.stages,
                        self.stages,
                        1,
                        1,
                        0,
                        self.epilogue_threads,
                    )
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
        tcgen05_fence("after_thread_sync")
        if warp == self.producer_warp:
            operands = (
                sw,
                sx,
                down_weight_map,
                down_input_map,
                cache_priority,
                output_tile,
                token_tile,
            )
            self._feed_down_ring(operands, full, empty, split, entry_prefilled, None)
            if cutlass.const_expr(self.up_reuse_stages > 0):
                if prepare_up_weights:
                    # A Down weight slot covers 16 KiB for these shapes. Each Up weight
                    # slot covers two Down slots at the same arena offsets.
                    # The final empty events release those bytes after MMA.
                    parity = ((self.tiles - 1) // self.stages) % 2
                    for stage in cutlass.range_constexpr(self.up_reuse_stages):
                        cute.arch.mbarrier_wait(empty + 2 * stage, parity)
                        cute.arch.mbarrier_wait(empty + 2 * stage + 1, parity)
                        _issue_up_weight_from_down_arena(
                            arena.iterator + stage * (2 * self.m * self.k),
                            up_bars + stage,
                            up_weight_map,
                            _up_weight_coordinates(
                                up_hidden_tile,
                                stage,
                                0,
                                64,
                                128,
                                self.packed_up_hidden_tiles,
                            ),
                            up_weight_priority,
                            self.up_stage_bytes,
                            not self.up_barriers_at_entry,
                        )
            if cutlass.const_expr(len(self.up_first_weight_copies) > 0):
                if prepare_up_weights:
                    for copy_index in cutlass.range_constexpr(
                        len(self.up_first_weight_copies)
                    ):
                        channel, offset, events = self.up_first_weight_copies[
                            copy_index
                        ]
                        for event_index in cutlass.range_constexpr(len(events)):
                            slot, parity = events[event_index]
                            cute.arch.mbarrier_wait(empty + slot, parity)
                        coordinates = _up_weight_coordinates(
                            up_hidden_tile,
                            0,
                            channel,
                            64,
                            128,
                            self.packed_up_hidden_tiles,
                        )
                        if cutlass.const_expr(copy_index == 0):
                            # Remaining weight bytes and activation complete the original transaction.
                            _issue_up_weight_from_down_arena(
                                arena.iterator + offset,
                                up_bars,
                                up_weight_map,
                                coordinates,
                                up_weight_priority,
                                self.up_stage_bytes,
                                True,
                            )
                        else:
                            _explicit_tma_copy(
                                arena.iterator + offset,
                                up_weight_map,
                                up_bars,
                                coordinates,
                                _weight_eviction_policy(up_weight_priority),
                            )
        elif warp == self.mma_warp:
            acc, a_desc, b_desc = _make_down_fragments(
                mma.get_slice(0),
                tmem_base,
                sw,
                sx,
                mma.partition_shape_C((self.m, self.n)),
            )
            if cutlass.const_expr(self.issue_style == "cute"):
                _run_cute_mma_ring(
                    mma,
                    acc,
                    a_desc,
                    b_desc,
                    full,
                    empty,
                    self.tiles,
                    self.stages,
                    self.mma_k_unroll,
                    True,
                )
                with cute.arch.elect_one():
                    tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
                cute.arch.mbarrier_wait(read_done, 0)
            else:
                instruction = fp8_instruction(self.m, self.n)
                for tile in cutlass.range_constexpr(self.tiles):
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                    with cute.arch.elect_one():
                        for block in cutlass.range_constexpr(
                            cute.size(a_desc, mode=[2])
                        ):
                            fp8_mma_encoded(
                                acc,
                                fp8_descriptor_at(a_desc, block, stage),
                                fp8_descriptor_at(b_desc, block, stage),
                                instruction,
                                (tile != 0) | (block != 0),
                            )
                        tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
                with cute.arch.elect_one():
                    tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
                cute.arch.mbarrier_wait(read_done, 0)
        elif tid < self.epilogue_threads:
            cute.arch.mbarrier_wait(done, 0)
            values, coords = _read_down_partials(
                mma,
                tmem_base,
                partials,
                tid,
                (self.m, self.n, self.k),
                self.epilogue_groups,
            )
            if cutlass.const_expr(self.native_down_partials):
                rank_offset = cutlass.Int64(split) * cutlass.Int64(partials.stride[0])
                token_base = cutlass.Int64(token_tile) * self.n
                full_columns = (self.projection_rows // self.m) * self.m
                for block in cutlass.range_constexpr(cute.size(values) // 4):
                    col, row = coords[4 * block]
                    local_row = cutlass.Int64(row)
                    global_col = output_tile * self.m + col
                    if (
                        token_base + local_row < x.shape[0]
                        and global_col < self.projection_rows
                    ):
                        # This TMEM load yields two adjacent rows for c, then the
                        # same rows for c+8. Both rows fit in the even-sized bucket.
                        base = (
                            partials.iterator
                            + rank_offset
                            + local_row * self.projection_rows
                        )
                        if global_col < full_columns:
                            offset = 2 * global_col + 2 * (global_col % 8)
                            llvm.inline_asm(
                                None,
                                [(base + offset).toint().ir_value()]
                                + [values[4 * block + i].ir_value() for i in range(4)],
                                "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
                                "l,f,f,f,f,~{memory}",
                                has_side_effects=True,
                                asm_dialect=0,
                            )
                        else:
                            # The final four injection columns have no c+8 partner.
                            llvm.inline_asm(
                                None,
                                [
                                    (base + 2 * global_col).toint().ir_value(),
                                    values[4 * block].ir_value(),
                                    values[4 * block + 1].ir_value(),
                                ],
                                "st.global.v2.f32 [$0], {$1, $2};",
                                "l,f,f,~{memory}",
                                has_side_effects=True,
                                asm_dialect=0,
                            )
            elif cutlass.const_expr(self.partial_store_packet == 1):
                for index in cutlass.range_constexpr(cute.size(values)):
                    col, row = coords[index]
                    local_row = cutlass.Int64(row)
                    row = local_row + token_tile * self.n
                    global_col = output_tile * self.m + col
                    if row < x.shape[0] and global_col < self.projection_rows:
                        partials[split, local_row, global_col] = values[index]
            else:
                packet = self.partial_store_packet
                lane_shift = 2 if self.m == 64 else 0
                lane_row = (tid >> lane_shift) % packet
                for block in cutlass.range_constexpr(cute.size(values) // packet):
                    payload = cute.make_rmem_tensor(packet, cutlass.Float32)
                    for i in cutlass.range_constexpr(packet):
                        payload[i] = values[block * packet + i]
                    _transpose_store_packet(payload, packet, lane_shift)
                    col, row = coords[block * packet]
                    local_row = cutlass.Int64(row + lane_row)
                    global_row = local_row + cutlass.Int64(token_tile) * self.n
                    global_col = output_tile * self.m + col - lane_row
                    if global_row < x.shape[0] and global_col < self.projection_rows:
                        offset = cutlass.Int64(split) * cutlass.Int64(
                            partials.stride[0]
                        )
                        offset = offset + local_row * self.projection_rows + global_col
                        destination = cute.make_tensor(
                            cute.make_ptr(
                                cutlass.Float32,
                                (partials.iterator + offset).toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=packet * 4,
                            ),
                            cute.make_layout(packet),
                        )
                        cute.copy(
                            cute.make_copy_atom(
                                CopyUniversalOp(),
                                cutlass.Float32,
                                num_bits_per_copy=packet * 32,
                            ),
                            payload,
                            destination,
                        )
            tcgen05_fence("before_thread_sync")
            cute.arch.mbarrier_arrive(read_done)

    @cute.experimental.jit
    def _operand_load(
        self,
        destination,
        mbar,
        *,
        descriptor,
        coordinates,
        cache_priority,
        is_weight: cutlass.Constexpr,
    ):
        cache_policy = None
        if cutlass.const_expr(is_weight):
            cache_policy = _weight_eviction_policy(cache_priority)
        _explicit_tma_copy(
            destination.iterator, descriptor, mbar, coordinates, cache_policy
        )

    @cute.experimental.jit
    def _issue_entry_weight(
        self,
        destination,
        full,
        descriptor,
        output_tile,
        k_tile,
        cache_priority,
    ):
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(full, (self.m + self.n) * self.k)
        self._operand_load(
            destination,
            full,
            cache_priority=cache_priority,
            is_weight=True,
            descriptor=descriptor,
            coordinates=(0, output_tile * self.m, k_tile * (self.k // 128)),
        )

    @cute.experimental.jit
    def prepare_entry(
        self,
        arena,
        barriers,
        output_tile,
        split,
        token_tile,
        down_weight_map,
        down_input_map,
        cache_priority,
    ):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        tiler = (self.m, self.n, self.k)
        mma, sw, sx = _make_down_operands(arena, tiler, self.stages)
        full = barriers
        if cutlass.const_expr(self.early_entry_weights > 0):
            if warp == self.producer_warp:
                # The producer owns these full events. The original CTA join
                # still precedes the input copies that complete their bytes.
                with cute.arch.elect_one():
                    for stage in cutlass.range_constexpr(self.early_entry_weights):
                        cute.arch.mbarrier_init(full + stage, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.sync_warp()
                for stage in cutlass.range_constexpr(self.early_entry_weights):
                    k_tile = split * self.tiles + stage
                    self._issue_entry_weight(
                        sw[None, None, None, stage],
                        full + stage,
                        down_weight_map,
                        output_tile,
                        k_tile,
                        cache_priority,
                    )
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                _init_mma_barriers(
                    barriers,
                    self.stages,
                    self.stages,
                    1,
                    1,
                    self.early_entry_weights,
                    self.epilogue_threads,
                )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if warp == self.producer_warp:
            # The enclosing kernel already waited for input and weight producers.
            # Pair each stage's operands so its MMA can start before later loads.
            for stage in cutlass.range_constexpr(self.stages):
                k_tile = split * self.tiles + stage
                if cutlass.const_expr(stage >= self.early_entry_weights):
                    self._issue_entry_weight(
                        sw[None, None, None, stage],
                        full + stage,
                        down_weight_map,
                        output_tile,
                        k_tile,
                        cache_priority,
                    )
                self._operand_load(
                    sx[None, None, None, stage],
                    (full + stage),
                    cache_priority=cache_priority,
                    is_weight=False,
                    descriptor=down_input_map,
                    coordinates=(0, token_tile * self.n, k_tile * (self.k // 128)),
                )


class _WideMaps(NamedTuple):
    down_weight_map: TensorMap
    down_input_map: TensorMap
    up_weight_map: TensorMap
    up_input_map: TensorMap
    down_weight_priority: cutlass.Int32
    up_weight_priority: cutlass.Int32


class WideFp8Mixer(ResidentMixer):
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
        up_weight_is_packed,
    ):
        small = token_bucket == 16
        medium = token_bucket == 32
        raw_mma = token_bucket == 64
        entry_barriers = token_bucket == 128
        paired_tokens = token_bucket == 256
        has_injection = p == r + g
        grouped_readers = small and has_injection
        release_after_reads = entry_barriers and has_injection

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
        self.compute_threads = 512
        self.producer_warp = 16
        self.mma_warp = 17
        self.threads = 576
        self.quant_rows = 2 if paired_tokens else 1
        self.quant_threads = self.compute_threads // self.quant_rows
        self.quant_columns = max(
            2,
            1
            << (((r + self.quant_threads - 1) // self.quant_threads - 1).bit_length()),
        )
        self.native_down_partials = entry_barriers
        self.quantization = QuantizationConfig(
            threads=self.compute_threads,
            maximum_threads=self.quant_threads,
            injection_start=self.compute_threads,
            column_stride=8 if entry_barriers else 1,
            warp_max="redux" if small else "pair",
            replicate_maximum=entry_barriers,
            refined_scales=not small,
            clamp="both" if small else "ptx_lower",
            activation_store="native",
            release_writes=False,
        )
        self.activation_wait = (
            "grid" if small else "warp_sync" if medium else "warp_arrive"
        )
        self.async_down_arrival = paired_tokens
        self.async_activation_arrival = paired_tokens or entry_barriers
        self.prefetch_down_scales = self.prefetch_up_scales = (
            self.prefetch_input_scales
        ) = entry_barriers

        self.up_hidden = 64
        self.up_channels = self.up_hidden * g
        self.up_channel_parts = 2 if token_bucket <= 64 else 1
        self.token_major_up = self.up_channel_parts == 1
        self.up_parts = 2 if not self.token_major_up or paired_tokens else 1
        self.up_rows = 128 if self.token_major_up else token_bucket
        self.up_mn = (
            (self.up_rows, self.up_channels)
            if self.token_major_up
            else (self.up_channels // self.up_channel_parts, self.up_rows)
        )
        self.up_k = 128
        self.up_stages = 5 if small else min(3, r // self.up_k)
        self.up_tiles = (r + self.up_k - 1) // self.up_k
        self.up_tail_tiles = (
            self.up_stages if paired_tokens and self.up_tiles > 1 else 0
        )
        self.packed_up_hidden_tiles = h // self.up_hidden if up_weight_is_packed else 0
        self.up_register_rows = paired_tokens
        self.up_cached_weight_base = 0 if paired_tokens else 2 * token_bucket
        self.up_scale_warps = (
            self.compute_threads // 32
            if paired_tokens
            else ((self.n if self.token_major_up else self.up_rows) + 31) // 32
        )
        self.reader_groups = self.compute_threads // 128
        self.hidden_reader_columns = min(32, token_bucket // self.reader_groups)
        self.reader_columns = self.up_hidden // self.reader_groups
        self.up_hidden_mix = (
            "register_half_group_no_join"
            if grouped_readers
            else "vector_pair" if has_injection and (medium or raw_mma) else "native"
        )
        self.up_grouped_readers = grouped_readers
        self.up_hidden_pair_mul = has_injection and raw_mma
        self.up_early_residual_count = int(release_after_reads)
        self.up_tmem_dealloc_site = (
            "after_reads" if release_after_reads else "kernel_tail"
        )
        self.up_unused_empty = int(release_after_reads)
        self.up_skip_final = not entry_barriers or release_after_reads
        self.up_wait_once = entry_barriers

        self.up_prefetch_stages = (
            self.up_stages if small else 3 if release_after_reads else 2
        )
        self.up_late_prefetch_stages = int(grouped_readers)
        self.up_activation_prefetch = (
            (5, 2, 1) if small else (7, 1, 1) if paired_tokens else (3, 1, 1)
        )
        self.up_followup_prefetch = {
            16: ("off", -1, 1),
            32: ("reader", 5, 1),
            64: ("before_acquire", 5, 1),
            128: ("before_acquire", 5, 1),
            256: ("down_handoff", 6, 1),
        }[token_bucket]
        if up_weight_is_packed and medium:
            self.last_up_weight_prefetch = ("after_release", 16)
        elif release_after_reads:
            self.last_up_weight_prefetch = ("before_invalidate", 16)
        else:
            self.last_up_weight_prefetch = ("off", 0)
        self.up_refill_unroll = paired_tokens
        self.up_tile4_hint_stride = 2 if paired_tokens else 0
        self.up_early_weight_parts = (
            (0, 1) if token_bucket <= 32 else (0,) if raw_mma else ()
        )
        self.up_reuse_stages = 2 if self.token_major_up else 0
        self.up_barriers_at_entry = entry_barriers

        self.hidden_tiles = h // self.up_hidden
        self.up_tasks = self.hidden_tiles
        self.up_stage_bytes = (
            self.up_mn[0] * self.up_parts + self.up_mn[1]
        ) * self.up_k
        self.up_operand_bytes = self.up_stage_bytes * self.up_stages
        self.up_barrier_count = (
            2 * self.up_stages + self.up_parts + 1 + int(self.up_grouped_readers)
        )
        self.down_m, self.splits, down_k, down_stages = _DOWN_TACTICS[token_bucket]
        self.down_issue_style = "raw_unrolled" if raw_mma else "cute"
        self.up_issue_style = (
            "cute_unrolled" if paired_tokens else self.down_issue_style
        )
        self.down_mma_unroll = down_k // 32 if raw_mma else 4
        self.mma_down_arrival = not (small or paired_tokens)
        self.partial_store_packet = 4 if paired_tokens else 1
        self.down_token_parts = 1
        self.down_barrier_count = 2 * down_stages + 2
        self.down_tiles = (p + self.down_m - 1) // self.down_m
        self.down_tasks = self.down_tiles * self.splits
        if g * h % (self.splits * down_k) or not (
            1 <= down_stages <= g * h // (self.splits * down_k)
        ):
            raise ValueError("Down stages must fit an evenly partitioned reduction")
        tiles = g * h // (self.splits * down_k)
        if self.up_reuse_stages and (
            2 * self.down_m * down_k != g * self.up_hidden * self.up_k
            or tiles % down_stages
        ):
            raise ValueError("An Up weight stage must fit two retired Down stages")
        up_first_weight_copies = []
        if self.up_early_weight_parts:
            if self.down_tasks < self.up_tasks:
                raise ValueError(
                    "Every early Up task must follow a Down task on its CTA"
                )
            for part in self.up_early_weight_parts:
                up_weight_bytes = self.up_mn[0] * self.up_k
                offset = (
                    self.up_rows * self.up_k + part * up_weight_bytes
                ) * self.up_stages
                events = _down_retirement_events(
                    offset,
                    up_weight_bytes,
                    self.down_m * down_k,
                    token_bucket * down_k,
                    down_stages,
                    tiles,
                )
                last_reader_tile = max(
                    (
                        slot + ((tiles - 1 - slot) // down_stages) * down_stages
                        for slot, _ in events
                    ),
                    default=-1,
                )
                up_first_weight_copies.append(
                    (last_reader_tile, (part * 2, offset, events))
                )
            up_first_weight_copies.sort()

        self.down = _WideDownPhase(
            up_reuse_stages=self.up_reuse_stages,
            up_stage_bytes=self.up_stage_bytes,
            up_barriers_at_entry=self.up_barriers_at_entry,
            up_first_weight_copies=tuple(copy for _, copy in up_first_weight_copies),
            early_entry_weights=int(small or raw_mma or entry_barriers),
            packed_up_hidden_tiles=self.packed_up_hidden_tiles,
            mma_k_unroll=self.down_mma_unroll,
            partial_store_packet=self.partial_store_packet,
            native_down_partials=self.native_down_partials,
            issue_style=self.down_issue_style,
            role="projection",
            hc_count=g,
            hidden_size=h,
            lowrank=r,
            projection_rows=p,
            token_tile=token_bucket,
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
            epilogue_threads=256 if small else self.compute_threads,
            shared_up_scales=False,
            interleave_tma=False,
        )
        self.columns = max(128, token_bucket, self.up_mn[1] * self.up_parts)
        self.arena_bytes = max(
            (self.down_m + token_bucket) * down_k * down_stages,
            self.up_operand_bytes,
            0 if self.token_major_up else token_bucket * self.up_channels * 4,
        )
        self.barrier_count = self.down_barrier_count + self.up_barrier_count

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
        down_weight_l2: cutlass.Int32,
        down_input_l2: cutlass.Int32,
        up_weight_l2: cutlass.Int32,
        up_input_l2: cutlass.Int32,
        down_weight_priority: cutlass.Int32,
        up_weight_priority: cutlass.Int32,
        stream: CUstream,
    ):
        down_weight_map = create_tensor_map_tiled(
            down.iterator.toint(),
            cutlass.Float8E4M3FN,
            [128, self.p, self.g * self.h // 128],
            [self.g * self.h // 16, 8],
            [128, self.down_m, self.down.k // 128],
            swizzle=TensorMapSwizzle.s128b,
            l2_promotion=down_weight_l2,
        )
        down_input_map = create_tensor_map_tiled(
            x.iterator.toint(),
            cutlass.Float8E4M3FN,
            [128, x.shape[0], self.g * self.h // 128],
            [self.g * self.h // 16, 8],
            [128, self.n, self.down.k // 128],
            swizzle=TensorMapSwizzle.s128b,
            l2_promotion=down_input_l2,
        )
        if cutlass.const_expr(self.packed_up_hidden_tiles > 0):
            up_weight_map = create_tensor_map_tiled(
                up.iterator.toint(),
                cutlass.Float8E4M3FN,
                [
                    self.up_k,
                    self.up_channels,
                    self.packed_up_hidden_tiles * self.up_tiles,
                ],
                [self.up_k // 16, self.up_k * self.up_channels // 16],
                [self.up_k, self.up_channels // self.up_channel_parts, 1],
                swizzle=TensorMapSwizzle.s128b,
                l2_promotion=up_weight_l2,
            )
        else:
            up_weight_map = create_tensor_map_tiled(
                up.iterator.toint(),
                cutlass.Float8E4M3FN,
                [self.r, self.h, self.g],
                [self.r // 16, self.h * self.r // 16],
                [self.up_k, self.up_hidden, self.g // self.up_channel_parts],
                swizzle=TensorMapSwizzle.s128b,
                l2_promotion=up_weight_l2,
            )
        up_input_map = create_tensor_map_tiled(
            activation.iterator.toint(),
            cutlass.Float8E4M3FN,
            [self.r, self.n],
            [self.r // 16],
            [self.up_k, self.up_rows],
            swizzle=TensorMapSwizzle.s128b,
            l2_promotion=up_input_l2,
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
            down_weight_map,
            down_input_map,
            up_weight_map,
            up_input_map,
            down_weight_priority,
            up_weight_priority,
        ).launch(
            grid=(grid_ctas, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            cooperative=True,
            use_pdl=self.pdl,
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
        down_weight_map: cutlass.GridConstant[TensorMap],
        down_input_map: cutlass.GridConstant[TensorMap],
        up_weight_map: cutlass.GridConstant[TensorMap],
        up_input_map: cutlass.GridConstant[TensorMap],
        down_weight_priority: cutlass.Int32,
        up_weight_priority: cutlass.Int32,
    ):
        cta = _mixer_cta()
        tid, warp, pid, ctas = cta
        shared = _mixer_shared(
            self.arena_bytes, self.barrier_count, self.down_barrier_count
        )
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_wait()
        if pid < self.down_tasks:
            # The first operand ring can fill while TMEM and the private
            # invocation ordinal become ready. Reuse the original two joins.
            self.down.prepare_entry(
                shared.arena,
                shared.down_barriers,
                pid // self.splits,
                pid % self.splits,
                cutlass.Int64(0),
                down_weight_map,
                down_input_map,
                down_weight_priority,
            )
        if cutlass.const_expr(self.up_barriers_at_entry):
            if pid < self.up_tasks:
                # The existing entry join publishes Up barriers before any
                # producer can use them. Down's operand copies can run now.
                self._init_up_barriers(
                    shared.up_barriers, cutlass.Int32(0), use_existing_join=True
                )
        if cutlass.const_expr(
            (self.prefetch_down_scales and self.down_per_channel)
            or (self.prefetch_up_scales and self.up_per_channel)
            or self.prefetch_input_scales
        ):
            if warp == 17:
                self._prefetch_scale_lines(ds, us, xs, pid)
        if tid == 0:
            shared.epoch[0] = _acquire(epochs.iterator + cutlass.Int64(pid) + 1)
        if warp == 17:
            cute.arch.alloc_tmem(self.columns, shared.tmem, is_two_cta=False)
            cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.sync_threads()
        phase_base = shared.epoch[0]
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
            _WideMaps(
                down_weight_map,
                down_input_map,
                up_weight_map,
                up_input_map,
                down_weight_priority,
                up_weight_priority,
            ),
            None,
            phase_base,
            entry_prefills_down=True,
            prefetch_residual_before_down=False,
            skip_last_up_join=self.up_skip_final,
            track_last_up_task=self.up_tmem_dealloc_site != "kernel_tail"
            or self.up_skip_final,
        )
        dealloc_at_tail = cutlass.Boolean(True)
        if cutlass.const_expr(self.up_tmem_dealloc_site != "kernel_tail"):
            dealloc_at_tail = (pid >= self.up_tasks) | (x.shape[0] == 0)
        if (warp == 17) & dealloc_at_tail:
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, shared.tmem),
                self.columns,
            )
        cute.arch.sync_threads()
        if cutlass.const_expr(self.pdl):
            if tid == 0:
                cute.arch.griddepcontrol_launch_dependents()

    @cute.experimental.jit
    def _project_down_tile(self, io, cta, shared, maps, state, tile, global_base):
        prepare_up_weights = cutlass.Boolean(False)
        prepare_up_weights = (cta.pid < self.up_tasks) & tile.last
        self.down.project(
            io.x,
            io.partials,
            shared.arena,
            shared.down_barriers,
            shared.tmem,
            tile.output_tile,
            tile.split,
            tile.token_tile,
            maps.down_weight_map,
            maps.down_input_map,
            cache_priority=maps.down_weight_priority,
            entry_prefilled=tile.entry_prefilled,
            prepare_up_weights=prepare_up_weights,
            up_bars=shared.up_barriers,
            up_weight_map=maps.up_weight_map,
            up_weight_priority=maps.up_weight_priority,
            up_hidden_tile=cta.pid,
        )
        if cutlass.const_expr(self.mma_down_arrival):
            if (cta.warp == self.mma_warp) & tile.last:
                # The existing read_done acquire orders the partials.
                # Earlier tasks have already joined this CTA.
                with cute.arch.elect_one():
                    _arrive_phase_release(
                        io.epochs.iterator,
                        self.async_down_arrival,
                    )
        cute.arch.sync_threads()
        if cutlass.const_expr(self.last_up_weight_prefetch[0] == "before_invalidate"):
            if tile.last:
                self._prefetch_last_up_weight_at(
                    "before_invalidate",
                    maps.up_weight_map,
                    cta.pid,
                    cta.count,
                    cta.warp,
                )

    @cute.experimental.jit
    def _begin_quantization(
        self, io, cta, shared, maps, state, phase_base, global_base
    ):
        cute.arch.sync_threads()
        if cta.tid == 0:
            if cutlass.const_expr(self.mma_down_arrival):
                if cta.pid >= self.down_tasks:
                    _arrive_phase_release(
                        io.epochs.iterator,
                        self.async_down_arrival,
                    )
            else:
                _arrive_phase_release(
                    io.epochs.iterator,
                    self.async_down_arrival,
                )
        self._prefetch_last_up_weight_at(
            "after_release", maps.up_weight_map, cta.pid, cta.count, cta.warp
        )
        if cta.pid < self.up_tasks:
            prepared_weight_stages = cutlass.Int32(0)
            if cutlass.const_expr(self.up_reuse_stages > 0):
                if cta.pid < self.down_tasks:
                    prepared_weight_stages = cutlass.Int32(self.up_reuse_stages)
            self._prefetch_up_weights(
                shared.arena,
                shared.up_barriers,
                cta.pid,
                maps.up_weight_map,
                cache_priority=maps.up_weight_priority,
                prepared_weight_stages=prepared_weight_stages,
                after_acquire=False,
            )
        if cta.tid == 0:
            while _acquire(io.epochs.iterator) < (phase_base + 1) * cta.count:
                pass
        cute.arch.sync_threads()
        if cutlass.const_expr(self.up_late_prefetch_stages):
            if cta.pid < self.up_tasks:
                self._prefetch_up_weights(
                    shared.arena,
                    shared.up_barriers,
                    cta.pid,
                    maps.up_weight_map,
                    cache_priority=maps.up_weight_priority,
                    prepared_weight_stages=cutlass.Int32(0),
                    after_acquire=True,
                )
        row = cutlass.Int64(cta.pid) * self.quant_rows
        return phase_base, row, None

    @cute.experimental.jit
    def _finish_quantization(self, io, cta, state, phase_base, quant):
        _publish_activation_proxy()
        activation_target = (phase_base + 2) * cta.count
        if cutlass.const_expr(self.activation_wait == "grid"):
            _grid_barrier(
                io.epochs,
                activation_target,
                True,
                self.async_activation_arrival,
            )
        else:
            # Keep the same release from each CTA. Only real readers acquire.
            cute.arch.sync_threads()
            if cta.tid == 0:
                _arrive_phase_release(
                    io.epochs.iterator,
                    self.async_activation_arrival,
                )
        return activation_target

    @cute.experimental.jit
    def _project_up_tile(
        self, io, shared, maps, state, tile, global_base, activation_target
    ):
        self._up(
            io.alpha,
            io.x,
            io.xs,
            io.us,
            io.out,
            shared.arena,
            shared.up_barriers,
            shared.tmem,
            tile.hidden_tile,
            global_base,
            tile.weights_prepared,
            maps.up_weight_map,
            maps.up_input_map,
            cache_priority=maps.up_weight_priority,
            activation_counter=io.epochs.iterator,
            activation_target=activation_target,
            terminal_up_task=tile.last,
        )

    @cute.experimental.jit
    def _finish_round(
        self, io, cta, shared, state, phase_base, activation_target, more_rounds
    ):
        phase_base = phase_base + 2
        if more_rounds:
            phase_base = phase_base + 1
        if cta.tid == 0:
            # Stream completion or the original PDL wait orders the next
            # call's read of this private per-CTA ledger.
            _store_private_epoch(
                io.epochs.iterator + cutlass.Int64(cta.pid) + 1, phase_base
            )
        # Only another token round needs a global handoff before reuse.
        # Stream completion or the successor's PDL wait orders a new call.
        if more_rounds:
            if cutlass.const_expr(self.up_barriers_at_entry):
                if cta.pid < self.up_tasks:
                    # The last Up task invalidated its barriers. The
                    # existing grid barrier joins the CTA before reuse.
                    self._init_up_barriers(
                        shared.up_barriers, cutlass.Int32(0), use_existing_join=True
                    )
            if cutlass.const_expr(self.activation_wait != "grid"):
                if cta.pid >= self.up_tasks and cta.tid == 0:
                    # No Up task acquired phase two on this CTA. Its reuse
                    # arrival must not satisfy a pending activation acquire.
                    while _acquire(io.epochs.iterator) < activation_target:
                        pass
            _grid_barrier(
                io.epochs,
                phase_base * cta.count,
                True,
                False,
            )
        return phase_base

    @cute.experimental.jit
    def _prefetch_scale_line(self, pointer):
        llvm.inline_asm(
            None,
            [pointer.toint().ir_value()],
            "prefetch.global.L1 [$0];",
            "l,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )

    @cute.experimental.jit
    def _prefetch_scale_lines(self, ds, us, xs, pid):
        lane = cute.arch.lane_idx()
        if cutlass.const_expr(self.prefetch_down_scales and self.down_per_channel):
            if pid < self.n:
                for i in cutlass.range_constexpr((self.p + 255) // 256):
                    offset = i * 256 + lane * 8
                    if offset < self.p:
                        self._prefetch_scale_line(ds.iterator + offset)
        if cutlass.const_expr(self.prefetch_up_scales and self.up_per_channel):
            if pid < self.up_tasks:
                # One 64-channel hidden tile in each of the four branches.
                offset = (lane // 8) * self.h + pid * self.up_hidden + (lane % 8) * 8
                self._prefetch_scale_line(us.iterator + offset)
        if cutlass.const_expr(self.prefetch_input_scales):
            if pid < self.up_tasks:
                offset = lane * 8
                if (offset < self.n) & (offset < xs.shape[0]):
                    self._prefetch_scale_line(xs.iterator + offset)

    @cute.experimental.jit
    def _load_hidden_pair(self, tensor, offset, dtype: cutlass.Constexpr):
        return self._load_hidden_packet(
            tensor, offset, dtype, 2, cute.AddressSpace.smem
        )

    @cute.experimental.jit
    def _write_hidden_pairs(
        self, gates, residual_tile, cached, out, x, tid, hidden_tile, global_base
    ):
        lanes_per_row = self.up_hidden // 2
        rows_per_batch = self.compute_threads // lanes_per_row
        column = 2 * (tid % lanes_per_row)
        hidden = hidden_tile * self.up_hidden + column
        scales = cute.make_rmem_tensor((2, self.g), cutlass.Float32)
        for branch in cutlass.range_constexpr(self.g):
            offset = 2 * self.up_rows + branch * self.up_hidden + column
            weights = self._load_hidden_pair(cached, offset, cutlass.Float32)
            for element in cutlass.range_constexpr(2):
                scales[element, branch] = weights[element]
        for batch in cutlass.range_constexpr(self.up_rows // rows_per_batch):
            row = tid // lanes_per_row + batch * rows_per_batch
            global_row = global_base + cutlass.Int64(row)
            if global_row < x.shape[0]:
                activation_scale, input_scale = cached[row], cached[self.up_rows + row]
                totals = cute.make_rmem_tensor(2, cutlass.Float32)
                totals.fill(0.0)
                for branch in cutlass.range_constexpr(self.g):
                    channel0 = branch * self.up_hidden + column
                    offset = row * self.up_channels + channel0
                    codes = self._load_hidden_pair(
                        residual_tile, offset, cutlass.Float8E4M3FN
                    )
                    decoded = _decode_residual(codes)
                    values = self._load_hidden_pair(gates, offset, cutlass.Float32)
                    if cutlass.const_expr(self.up_hidden_pair_mul):
                        # Scale the two adjacent columns without changing either FMA chain.
                        f0, f1 = _mul_hidden_float_pair(
                            activation_scale,
                            activation_scale,
                            scales[0, branch],
                            scales[1, branch],
                        )
                        g0, g1 = _mul_hidden_float_pair(values[0], values[1], f0, f1)
                        x0, x1 = _mul_hidden_float_pair(
                            decoded[0], decoded[1], input_scale, input_scale
                        )
                        s0, s1 = _up_sigmoid_tanh(g0), _up_sigmoid_tanh(g1)
                        totals[0] = totals[0] + x0 * s0
                        totals[1] = totals[1] + x1 * s1
                    else:
                        for element in cutlass.range_constexpr(2):
                            gate, residual = values[element], decoded[element]
                            factor = activation_scale * scales[element, branch]
                            xv = residual * input_scale
                            totals[element] = totals[element] + xv * _up_sigmoid_tanh(
                                gate * factor
                            )
                packed = cute.make_rmem_tensor(2, out.element_type)
                for element in cutlass.range_constexpr(2):
                    packed[element] = (totals[element] / self.g).to(out.element_type)
                pointer = cute.make_ptr(
                    out.element_type,
                    (out.iterator + global_row * self.h + hidden).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                target = cute.make_tensor(pointer, cute.make_layout(2))
                cute.copy(
                    cute.make_copy_atom(
                        CopyUniversalOp(),
                        out.element_type,
                        num_bits_per_copy=32,
                    ),
                    packed,
                    target,
                )

    @cute.experimental.jit
    def _write_paired_reader_rows(
        self,
        gates,
        kept,
        cached,
        out,
        x,
        reader_channel,
        reader_row,
        hidden_tile,
        global_base,
        branch_group: cutlass.Constexpr,
        residual_codes,
        weight_scales,
    ):
        col = reader_channel % self.up_hidden
        hidden = hidden_tile * self.up_hidden + col
        row0 = reader_row + 2 * branch_group
        row1 = row0 + 1
        global_row0 = global_base + cutlass.Int64(row0)
        global_row1 = global_row0 + 1
        if global_row0 < x.shape[0]:
            # The residual copy and scale writers initialize the padded row.
            activation0, activation1 = cached[row0], cached[row1]
            input0, input1 = cached[self.up_rows + row0], cached[self.up_rows + row1]
            total0, total1 = cutlass.Float32(0.0), cutlass.Float32(0.0)
            for branch in cutlass.range_constexpr(self.g):
                channel = branch * self.up_hidden + col
                weight = weight_scales[branch]
                if cutlass.const_expr(branch % 2 == branch_group):
                    gate0, gate1 = kept[0, branch // 2], kept[1, branch // 2]
                else:
                    gate0, gate1 = gates[row0, channel], gates[row1, channel]
                residual0 = residual_codes[0, branch].to(cutlass.Float32)
                residual1 = residual_codes[1, branch].to(cutlass.Float32)
                gate0 = gate0 * (activation0 * weight)
                gate1 = gate1 * (activation1 * weight)
                residual0 = residual0 * input0
                residual1 = residual1 * input1
                total0 = total0 + residual0 * _up_sigmoid_tanh(gate0)
                total1 = total1 + residual1 * _up_sigmoid_tanh(gate1)
            out[global_row0, hidden] = (total0 / self.g).to(out.element_type)
            if global_row1 < x.shape[0]:
                out[global_row1, hidden] = (total1 / self.g).to(out.element_type)

    @cute.experimental.jit
    def _mix_reader_group(
        self,
        gates,
        kept_gates,
        cached,
        out,
        x,
        reader_channel,
        reader_row,
        hidden_tile,
        global_base,
        residual_codes,
        weight_scales,
    ):
        if reader_channel < self.up_hidden:
            self._write_paired_reader_rows(
                gates,
                kept_gates,
                cached,
                out,
                x,
                reader_channel,
                reader_row,
                hidden_tile,
                global_base,
                0,
                residual_codes,
                weight_scales,
            )
        else:
            self._write_paired_reader_rows(
                gates,
                kept_gates,
                cached,
                out,
                x,
                reader_channel,
                reader_row,
                hidden_tile,
                global_base,
                1,
                residual_codes,
                weight_scales,
            )

    @cute.experimental.jit
    def _read_mix_inputs(self, coords, cached, residual_tile):
        channel, row = coords[0]
        column = channel % self.up_hidden
        row0 = row + 2 * (channel // self.up_hidden)
        residual_codes = cute.make_rmem_tensor((2, self.g), cutlass.Float8E4M3FN)
        for branch in cutlass.range_constexpr(self.g):
            column_in_branch = branch * self.up_hidden + column
            residual_codes[0, branch] = residual_tile[row0, column_in_branch]
            residual_codes[1, branch] = residual_tile[row0 + 1, column_in_branch]
        weight_scales = cute.make_rmem_tensor(self.g, cutlass.Float32)
        for branch in cutlass.range_constexpr(self.g):
            weight_scales[branch] = cached[
                2 * self.up_rows + branch * self.up_hidden + column
            ]
        return residual_codes, weight_scales

    @cute.experimental.jit
    def _init_up_barriers(
        self, bars, prepared_weight_stages, use_existing_join: cutlass.Constexpr
    ):
        warp = cute.arch.make_warp_uniform(cute.arch.thread_idx()[0] // 32)
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                _init_mma_barriers(
                    bars,
                    self.up_stages,
                    self.up_stages,
                    self.up_parts,
                    1,
                    prepared_weight_stages,
                    self.compute_threads,
                )
                if cutlass.const_expr(self.up_grouped_readers):
                    cute.arch.mbarrier_init(bars + self.up_barrier_count - 1, 1)
                if cutlass.const_expr(use_existing_join):
                    cute.arch.mbarrier_init_fence()
                    llvm.inline_asm(
                        None,
                        [],
                        "fence.proxy.async.shared::cta;",
                        "~{memory}",
                        has_side_effects=True,
                        asm_dialect=0,
                    )
        if cutlass.const_expr(not use_existing_join):
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()

    @cute.experimental.jit
    def _prefetch_up_weights(
        self,
        arena,
        bars,
        hidden_tile,
        up_weight_map,
        cache_priority,
        prepared_weight_stages,
        after_acquire: cutlass.Constexpr,
    ):
        _, _, al, bl = self._up_operand_layouts()
        if cutlass.const_expr(self.token_major_up):
            sw = _operand_at(arena, bl, 0)
        else:
            sw = _operand_at(arena, al, cute.cosize(bl))
        extra_sw = None
        if cutlass.const_expr(self.up_channel_parts == 2):
            extra_sw = _operand_at(arena, al, cute.cosize(bl) + cute.cosize(al))
        warp = cute.arch.make_warp_uniform(cute.arch.thread_idx()[0] // 32)
        if cutlass.const_expr(after_acquire):
            # These events were initialized by the original local Down handoff.
            pass
        elif cutlass.const_expr(not self.up_barriers_at_entry):
            initialized_full = prepared_weight_stages
            if cutlass.const_expr(len(self.up_early_weight_parts) > 0):
                initialized_full = cutlass.Int32(1)
            self._init_up_barriers(bars, initialized_full, use_existing_join=False)
        if warp == self.producer_warp:
            handoff_stages = self.up_prefetch_stages
            late_start = handoff_stages - self.up_late_prefetch_stages
            for stage in cutlass.range_constexpr(
                late_start if after_acquire else 0, handoff_stages
            ):
                if stage >= prepared_weight_stages:
                    arm_stage = cutlass.Boolean(not after_acquire)
                    issue_part0 = cutlass.Boolean(after_acquire or stage < late_start)
                    issue_part1 = cutlass.Boolean(not after_acquire)
                    if cutlass.const_expr(
                        len(self.up_early_weight_parts) == 0 or stage != 0
                    ):
                        if arm_stage:
                            with cute.arch.elect_one():
                                # Later transfers complete the same transaction.
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    bars + stage,
                                    self.up_stage_bytes,
                                )
                    if cutlass.const_expr(
                        0 not in self.up_early_weight_parts or stage != 0
                    ):
                        if issue_part0:
                            self._up_operand_load(
                                sw[None, None, None, stage],
                                bars + stage,
                                weight_map=up_weight_map,
                                input_map=up_weight_map,
                                hidden_tile=hidden_tile,
                                tile=stage,
                                operand=("B" if self.token_major_up else "A"),
                                part=0,
                                cache_priority=cache_priority,
                            )
                    if cutlass.const_expr(
                        self.up_channel_parts == 2
                        and (1 not in self.up_early_weight_parts or stage != 0)
                    ):
                        if issue_part1:
                            self._up_operand_load(
                                extra_sw[None, None, None, stage],
                                bars + stage,
                                weight_map=up_weight_map,
                                input_map=up_weight_map,
                                hidden_tile=hidden_tile,
                                tile=stage,
                                operand=("B" if self.token_major_up else "A"),
                                part=1,
                                cache_priority=cache_priority,
                            )
            if cutlass.const_expr(not after_acquire):
                # Initial demand copies are issued before speculating on a refill.
                self._prefetch_followup_at(
                    "down_handoff",
                    up_weight_map,
                    hidden_tile,
                    cutlass.Boolean(True),
                )

    @cute.experimental.jit
    def _up_weight_coords(self, hidden_tile, tile, part: cutlass.Constexpr):
        return _up_weight_coordinates(
            hidden_tile,
            tile,
            part * (self.g // self.up_channel_parts),
            self.up_hidden,
            self.up_k,
            self.packed_up_hidden_tiles,
        )

    @cute.experimental.jit
    def _prefetch_one_up_weight(self, weight_map, hidden_tile, tile: cutlass.Constexpr):
        for part in cutlass.range_constexpr(self.up_channel_parts):
            coordinates = self._up_weight_coords(hidden_tile, tile, part)
            with cute.arch.elect_one():
                nvvm.cp_async_bulk_tensor_prefetch(
                    weight_map.get_ptr().ir_value(),
                    [
                        cutlass.Int32(coordinate).ir_value()
                        for coordinate in coordinates
                    ],
                    [],
                )

    @cute.experimental.jit
    def _prefetch_last_up_weight_at(
        self, site: cutlass.Constexpr, weight_map, pid, ctas, warp
    ):
        if cutlass.const_expr(self.last_up_weight_prefetch[0] == site):
            warps = self.last_up_weight_prefetch[1]
            end = self.up_tasks + self.splits
            if (ctas >= end) & (pid >= self.up_tasks) & (pid < end) & (warp < warps):
                first_tile = (cutlass.Int32(pid) - self.up_tasks) * 16 + warp
                for index in cutlass.range_constexpr(16 // warps):
                    self._prefetch_one_up_weight(
                        weight_map, first_tile + index * warps, self.up_tiles - 1
                    )

    @cute.experimental.jit
    def _prefetch_up_demand_window(self, weight_map, hidden_tile):
        first, count, stride = self.up_activation_prefetch
        if hidden_tile % stride == 0:
            for tile in cutlass.range_constexpr(first, first + count):
                self._prefetch_one_up_weight(weight_map, hidden_tile, tile)
        if cutlass.const_expr(self.up_tile4_hint_stride > 0):
            if hidden_tile % self.up_tile4_hint_stride == 0:
                self._prefetch_one_up_weight(weight_map, hidden_tile, 4)

    @cute.experimental.jit
    def _prefetch_followup_at(
        self, site: cutlass.Constexpr, weight_map, hidden_tile, weights_prepared
    ):
        wanted, tile, _ = self.up_followup_prefetch
        if cutlass.const_expr(wanted == site):
            if weights_prepared:
                self._prefetch_one_up_weight(weight_map, hidden_tile, tile)

    @cute.experimental.jit
    def _up_operand_load(
        self,
        destination,
        mbar,
        *,
        weight_map,
        input_map,
        hidden_tile,
        tile,
        operand: cutlass.Constexpr,
        part: cutlass.Constexpr,
        cache_priority,
    ):
        is_weight = operand == ("B" if self.token_major_up else "A")
        if cutlass.const_expr(is_weight):
            coordinates = self._up_weight_coords(hidden_tile, tile, part)
            descriptor = weight_map
        else:
            coordinates = (tile * self.up_k, part * self.up_rows)
            descriptor = input_map
        cache_policy = None
        if cutlass.const_expr(is_weight):
            cache_policy = _weight_eviction_policy(cache_priority)
        _explicit_tma_copy(
            destination.iterator, descriptor, mbar, coordinates, cache_policy
        )

    @cute.experimental.jit
    def _load_up_stage(
        self,
        operands,
        full,
        maps,
        hidden_tile,
        tile,
        slot,
        cache_priority,
        stage_prepared,
    ):
        sw, sa0, sa1 = operands
        weight_map, input_map = maps
        load_a, load_b = True, True
        if cutlass.const_expr(self.token_major_up):
            load_b = not stage_prepared
        else:
            load_a = not stage_prepared
        if not stage_prepared:
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(full, self.up_stage_bytes)
        if load_b:
            self._up_operand_load(
                sw[None, None, None, slot],
                full,
                weight_map=weight_map,
                input_map=input_map,
                hidden_tile=hidden_tile,
                tile=tile,
                operand="B",
                part=0,
                cache_priority=cache_priority,
            )
        if load_a:
            self._up_operand_load(
                sa0[None, None, None, slot],
                full,
                weight_map=weight_map,
                input_map=input_map,
                hidden_tile=hidden_tile,
                tile=tile,
                operand="A",
                part=0,
                cache_priority=cache_priority,
            )
            if cutlass.const_expr(self.up_parts == 2):
                self._up_operand_load(
                    sa1[None, None, None, slot],
                    full,
                    weight_map=weight_map,
                    input_map=input_map,
                    hidden_tile=hidden_tile,
                    tile=tile,
                    operand="A",
                    part=1,
                    cache_priority=cache_priority,
                )

    @cute.experimental.jit
    def _acquire_activation_for_up(self, counter, expected):
        if cute.arch.lane_idx() == 0:
            while _acquire(counter) < expected:
                pass
        readers = 32 * (1 + self.up_scale_warps)
        if cutlass.const_expr(self.activation_wait == "warp_arrive"):
            # TMA may be issued by a different lane than the one that acquired.
            cute.arch.sync_warp()
            nvvm.barrier_cta_arrive(
                cutlass.Int32(13).ir_value(),
                cutlass.Int32(readers).ir_value(),
                aligned=False,
            )
        else:
            _epilogue_barrier(readers, cutlass.Int32(13))

    @cute.experimental.jit
    def _await_activation_scales(self, warp):
        if cutlass.const_expr(self.activation_wait != "grid"):
            if warp < self.up_scale_warps:
                _epilogue_barrier(32 * (1 + self.up_scale_warps), cutlass.Int32(13))

    @cute.experimental.jit
    def _up(
        self,
        alpha,
        x,
        xs,
        ws,
        out,
        arena,
        bars,
        base,
        hidden_tile,
        global_base,
        weights_prepared,
        up_weight_map,
        up_input_map,
        cache_priority,
        activation_counter,
        activation_target,
        terminal_up_task,
    ):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        cached = self._up_scale_storage(self.up_cached_weight_base)
        residual_tile = None
        reader_channel = cutlass.Int32(0)
        reader_row = cutlass.Int32(0)
        kept_gates = None
        if cutlass.const_expr(self.up_grouped_readers):
            kept_gates = cute.make_rmem_tensor((2, self.up_parts), cutlass.Float32)
        if cutlass.const_expr(not self.token_major_up):
            residual_tile = self._up_residual_tile(self.up_channels, 16)
        mma, _, al, bl = self._up_operand_layouts()
        sw, sa0, sa1 = self._up_operand_storage(arena, al, bl)
        full, empty, done, read_done = _mma_barriers(
            bars, self.up_stages, self.up_stages, self.up_parts
        )
        if not weights_prepared:
            self._init_up_barriers(bars, cutlass.Int32(0), use_existing_join=False)
        tcgen05_fence("after_thread_sync")
        if warp == self.producer_warp:
            if weights_prepared:
                self._prefetch_up_demand_window(up_weight_map, hidden_tile)
            self._prefetch_followup_at(
                "before_acquire", up_weight_map, hidden_tile, weights_prepared
            )
            if cutlass.const_expr(self.activation_wait != "grid"):
                self._acquire_activation_for_up(activation_counter, activation_target)
            for tile in cutlass.range_constexpr(self.up_stages):
                slot = tile
                stage_prepared = weights_prepared
                if cutlass.const_expr(tile >= self.up_prefetch_stages):
                    stage_prepared = cutlass.Boolean(False)
                self._load_up_stage(
                    (sw, sa0, sa1),
                    full + slot,
                    (up_weight_map, up_input_map),
                    hidden_tile,
                    tile,
                    slot,
                    cache_priority,
                    stage_prepared,
                )
            for tile in cutlass.range(
                self.up_stages,
                self.up_tiles - int(self.up_grouped_readers),
                unroll=1,
                unroll_full=self.up_refill_unroll,
            ):
                slot = tile % self.up_stages
                cute.arch.mbarrier_wait(empty + slot, (tile // self.up_stages - 1) % 2)
                self._load_up_stage(
                    (sw, sa0, sa1),
                    full + slot,
                    (up_weight_map, up_input_map),
                    hidden_tile,
                    tile,
                    slot,
                    cache_priority,
                    False,
                )
            if cutlass.const_expr(self.up_grouped_readers):
                tile = self.up_tiles - 1
                slot = tile % self.up_stages
                first_full, second_full = full + slot, read_done + 1
                half_bytes = self.up_mn[0] * self.up_k
                cute.arch.mbarrier_wait(empty + slot, (tile // self.up_stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        first_full, self.up_stage_bytes - half_bytes
                    )
                    cute.arch.mbarrier_arrive_and_expect_tx(second_full, half_bytes)
                self._up_operand_load(
                    sw[None, None, None, slot],
                    first_full,
                    weight_map=up_weight_map,
                    input_map=up_input_map,
                    hidden_tile=hidden_tile,
                    tile=tile,
                    operand="B",
                    part=0,
                    cache_priority=cache_priority,
                )
                self._up_operand_load(
                    sa0[None, None, None, slot],
                    first_full,
                    weight_map=up_weight_map,
                    input_map=up_input_map,
                    hidden_tile=hidden_tile,
                    tile=tile,
                    operand="A",
                    part=0,
                    cache_priority=cache_priority,
                )
                self._up_operand_load(
                    sa1[None, None, None, slot],
                    second_full,
                    weight_map=up_weight_map,
                    input_map=up_input_map,
                    hidden_tile=hidden_tile,
                    tile=tile,
                    operand="A",
                    part=1,
                    cache_priority=cache_priority,
                )
        elif warp == self.mma_warp:
            fragments = self._up_mma_fragments(mma, base, sa0, sa1, sw)
            if cutlass.const_expr(self.up_issue_style != "raw_unrolled"):
                fully_unroll = self.up_issue_style == "cute_unrolled"
                prefix_tiles = (
                    self.up_tiles - self.up_tail_tiles - int(self.up_grouped_readers)
                )
                mma.set(tcgen05.Field.ACCUMULATE, False)
                self._run_up_mma_tiles(
                    mma,
                    fragments,
                    full,
                    (empty, self.up_unused_empty),
                    None,
                    prefix_tiles,
                    fully_unroll,
                )
                if cutlass.const_expr(self.up_grouped_readers):
                    mma.set(tcgen05.Field.ACCUMULATE, prefix_tiles > 0)
                    tile = self.up_tiles - 1
                    slot = tile % self.up_stages
                    cute.arch.mbarrier_wait(full + slot, tile // self.up_stages % 2)
                    self._issue_up_mma_part(mma, fragments, slot, 0)
                    with cute.arch.elect_one():
                        tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
                    cute.arch.mbarrier_wait(read_done + 1, 0)
                    self._issue_up_mma_part(mma, fragments, slot, 1)
                    self._retire_up_slot(empty, slot, tile, self.up_unused_empty)
                    with cute.arch.elect_one():
                        tcgen05.commit(done + 1, None, tcgen05.CtaGroup.ONE)
                elif cutlass.const_expr(self.up_tail_tiles != 0):
                    self._run_paired_up_tail(mma, fragments, full, empty, done, True)
                else:
                    self._publish_up_done(done)
                cute.arch.mbarrier_wait(read_done, 0)
            else:
                instruction = fp8_instruction(self.up_mn[0], self.up_mn[1])
                for tile in cutlass.range_constexpr(self.up_tiles):
                    slot = tile % self.up_stages
                    cute.arch.mbarrier_wait(full + slot, tile // self.up_stages % 2)
                    with cute.arch.elect_one():
                        for block in cutlass.range_constexpr(
                            cute.size(fragments.a[0], mode=[2])
                        ):
                            accumulate = (tile != 0) | (block != 0)
                            fp8_mma_encoded(
                                fragments.acc[0],
                                fp8_descriptor_at(fragments.a[0], block, slot),
                                fp8_descriptor_at(fragments.b, block, slot),
                                instruction,
                                accumulate,
                            )
                            fp8_mma_encoded(
                                fragments.acc[1],
                                fp8_descriptor_at(fragments.a[1], block, slot),
                                fp8_descriptor_at(fragments.b, block, slot),
                                instruction,
                                accumulate,
                            )
                        tcgen05.commit(empty + slot, None, tcgen05.CtaGroup.ONE)
                self._publish_up_done(done)
                cute.arch.mbarrier_wait(read_done, 0)
            if cutlass.const_expr(self.up_tmem_dealloc_site != "kernel_tail"):
                if terminal_up_task:
                    tcgen05_fence("after_thread_sync")
                    cute.arch.dealloc_tmem(
                        cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, base),
                        self.columns,
                    )
        elif tid < self.compute_threads:
            if cutlass.const_expr(self.up_followup_prefetch[0] == "reader"):
                if warp == self.compute_threads // 32 - 1:
                    self._prefetch_followup_at(
                        "reader", up_weight_map, hidden_tile, weights_prepared
                    )
            if cutlass.const_expr(self.token_major_up):
                first_ring = first_template = first_coords = None
                if cutlass.const_expr(self.up_early_residual_count):
                    _, _, first_template, first_coords = self._reader(
                        mma, base, x, tid, 0, 0
                    )
                    first_ring = cute.make_rmem_tensor(
                        (cute.size(first_template), 2), cutlass.Float8E4M3FN
                    )
                    first_ring.fill(0.0)
                    self._prefetch_branch_x(
                        first_ring[None, 0],
                        first_coords,
                        x,
                        hidden_tile,
                        global_base,
                        0,
                        0,
                    )
                self._await_activation_scales(warp)
                if cutlass.const_expr(not self.up_register_rows):
                    self._cache_up_row_scales(
                        alpha, xs, cached, x, global_base, tid, self.n, True
                    )
                self._cache_up_weight_scales(
                    ws, cached, hidden_tile, tid, self.up_cached_weight_base, False
                )
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
                    (first_template, first_coords, first_ring),
                    row_scale_source=(
                        "global" if self.up_register_rows else "shared_row"
                    ),
                    weight_cache_offset=self.up_cached_weight_base,
                    vector_weight_scales=True,
                    prefetched_branches=self.up_early_residual_count,
                    wait_once=self.up_wait_once,
                    release_after_reads=self.up_tmem_dealloc_site == "after_reads",
                )
                if cutlass.const_expr(self.up_tmem_dealloc_site != "after_reads"):
                    tcgen05_fence("before_thread_sync")
                    cute.arch.mbarrier_arrive(read_done)
            else:
                self._copy_hidden_residual(
                    x,
                    residual_tile,
                    hidden_tile,
                    global_base,
                    tid,
                    self.compute_threads,
                    self.up_channels,
                )
                self._await_activation_scales(warp)
                self._cache_up_row_scales(
                    alpha, xs, cached, x, global_base, tid, self.up_rows, True
                )
                self._cache_up_weight_scales(
                    ws, cached, hidden_tile, tid, 2 * self.up_rows, False
                )
                if cutlass.const_expr(self.up_grouped_readers):
                    # Publish these CTA-shared operands while the MMA is
                    # in flight. They are read by every token-row group.
                    _wait_residual()
                    _epilogue_barrier(self.compute_threads, cutlass.Int32(5))
                mix_codes, mix_scales = None, None
                if cutlass.const_expr(self.up_grouped_readers):
                    _, _, _, coordinates = self._hidden_reader(mma, base, x, tid, 0)
                    mix_codes, mix_scales = self._read_mix_inputs(
                        coordinates, cached, residual_tile
                    )
                gate_offset = 0
                if cutlass.const_expr(self.up_grouped_readers):
                    # The first weight slot was last used by K5. At the
                    # first done event only K7's third slot can still be live.
                    gate_offset = cute.cosize(bl)
                else:
                    for part in cutlass.range_constexpr(self.up_parts):
                        cute.arch.mbarrier_wait(done + part, 0)
                gates = self._hidden_gate_tile(arena.iterator + gate_offset)
                for part in cutlass.range_constexpr(self.up_parts):
                    if cutlass.const_expr(self.up_grouped_readers):
                        cute.arch.mbarrier_wait(done + part, 0)
                    values, coords = self._read_hidden_up(mma, base, x, tid, part)
                    part_base = part * (self.up_channels // self.up_channel_parts)
                    if cutlass.const_expr(self.up_grouped_readers):
                        reader_channel, reader_row = coords[0]
                        # Each lane holds four token rows of one channel.
                        # The other branch group consumes the two rows
                        # this lane does not write to the output.
                        if reader_channel < self.up_hidden:
                            for i in cutlass.range_constexpr(2):
                                kept_gates[i, part] = values[i]
                                channel, row = coords[i + 2]
                                gates[row, part_base + channel] = values[i + 2]
                        else:
                            for i in cutlass.range_constexpr(2):
                                kept_gates[i, part] = values[i + 2]
                                channel, row = coords[i]
                                gates[row, part_base + channel] = values[i]
                    else:
                        for i in cutlass.range_constexpr(cute.size(values)):
                            channel, row = coords[i]
                            gates[row, part_base + channel] = values[i]
                tcgen05_fence("before_thread_sync")
                if cutlass.const_expr(not self.up_grouped_readers):
                    _wait_residual()
                cute.arch.mbarrier_arrive(read_done)
                if cutlass.const_expr(self.up_grouped_readers):
                    # Four reader warps exchange the gates for their four rows.
                    _epilogue_barrier(128, cutlass.Int32(6) + tid // 128)
                    self._mix_reader_group(
                        gates,
                        kept_gates,
                        cached,
                        out,
                        x,
                        reader_channel,
                        reader_row,
                        hidden_tile,
                        global_base,
                        mix_codes,
                        mix_scales,
                    )
        # The caller joins all threads before reusing the operand arena and
        # before the final PDL signal. Group-local exchange has already ended.
        if cutlass.const_expr(not self.up_grouped_readers):
            cute.arch.sync_threads()
        if cutlass.const_expr(not self.token_major_up and not self.up_grouped_readers):
            if tid < self.compute_threads:
                gates = self._hidden_gate_tile(arena.iterator)
                if cutlass.const_expr(self.up_hidden_mix != "native"):
                    self._write_hidden_pairs(
                        gates,
                        residual_tile,
                        cached,
                        out,
                        x,
                        tid,
                        hidden_tile,
                        global_base,
                    )
                else:
                    self._mix_hidden_major_up(
                        gates,
                        residual_tile,
                        cached,
                        out,
                        x.shape[0],
                        tid,
                        hidden_tile,
                        global_base,
                        self.compute_threads,
                    )
