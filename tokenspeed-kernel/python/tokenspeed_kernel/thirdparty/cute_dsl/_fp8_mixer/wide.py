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
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import CopyUniversalOp, tcgen05
from cutlass.cutlass_dsl import T
from cutlass.experimental.cuda import (
    TensorMap,
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
    _store_private_epoch,
    _wait_residual,
    _x_scale,
)
from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.raw_issue import (
    fp8_descriptor_at,
    fp8_instruction,
    fp8_mbarrier_wait,
    fp8_mma_encoded,
)
from tokenspeed_kernel.thirdparty.cute_dsl._fp8_mixer.resident import (
    MixerOperands,
    QuantizationConfig,
    ResidentMixer,
    _mixer_cta,
    _mixer_shared,
)
from tokenspeed_kernel.thirdparty.cute_dsl.hc_fp8 import (
    Fp8GatedResidualGemm,
    _epilogue_barrier,
    _up_sigmoid_tanh,
)
from tokenspeed_kernel.thirdparty.cute_dsl.hc_fused import (
    _arrive_counter,
    _publish_activation_proxy,
)


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
def _arrive_phase(
    ptr, release_only: cutlass.Constexpr, asynchronous: cutlass.Constexpr
):
    if cutlass.const_expr(release_only):
        # Arrivals publish data; only the following acquire poll consumes it.
        # The RMW chain keeps each publisher in observation order.
        instruction = "red.async" if asynchronous else "red"
        llvm.inline_asm(
            None,
            [ptr.toint().ir_value()],
            f"{instruction}.release.gpu.global.add.u64 [$0], 1;",
            "l,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )
    else:
        _arrive_counter(ptr)


@cute.experimental.jit
def _grid_barrier(
    epochs,
    expected,
    wait_for_all,
    release_only: cutlass.Constexpr,
    asynchronous: cutlass.Constexpr,
):
    # Cooperative launch and measured capacity keep every participant resident.
    cute.arch.sync_threads()
    if cute.arch.thread_idx()[0] == 0:
        _arrive_phase(epochs.iterator, release_only, asynchronous)
        if wait_for_all:
            while _acquire(epochs.iterator) < expected:
                pass
    cute.arch.sync_threads()


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


class _DownPhase(Fp8GatedResidualGemm):
    def __init__(
        self,
        *,
        mma_k_unroll,
        issue_style,
        partial_store_packet,
        native_down_partials,
        operand_transport,
        weight_cache_hint,
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
        self.operand_transport = operand_transport
        self.weight_cache_hint = weight_cache_hint
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
        full, empty = barriers, barriers + self.stages
        done, read_done = empty + self.stages, empty + self.stages + 1
        if not entry_prefilled:
            if warp == self.mma_warp:
                with cute.arch.elect_one():
                    for stage in cutlass.range_constexpr(self.stages):
                        cute.arch.mbarrier_init(full + stage, 1)
                        cute.arch.mbarrier_init(empty + stage, 1)
                    cute.arch.mbarrier_init(done, 1)
                    cute.arch.mbarrier_init(read_done, self.epilogue_threads)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
        tcgen05_fence("after_thread_sync")
        w_view = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (self.projection_rows, (64, self.inner // 64), 1),
                stride=(self.inner, (1, 64), 0),
            ),
        )
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
            if not entry_prefilled:
                # The enclosing kernel already waited for input and weight producers.
                # Pair each stage's operands so its MMA can start before later loads.
                for stage in cutlass.range_constexpr(self.stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + stage, stage_bytes
                        )
                    self._operand_load(
                        gw[None, None, split * self.tiles + stage],
                        sw[None, None, None, stage],
                        (full + stage),
                        cta_v_map=map_w,
                        update_expect_tx=False,
                        cache_priority=cache_priority,
                        is_weight=True,
                        descriptor=down_weight_map,
                        coordinates=(
                            0,
                            output_tile * self.m,
                            (split * self.tiles + stage) * (self.k // 128),
                        ),
                    )
                    self._operand_load(
                        gx[None, None, split * self.tiles + stage],
                        sx[None, None, None, stage],
                        (full + stage),
                        cta_v_map=map_x,
                        update_expect_tx=False,
                        cache_priority=cache_priority,
                        is_weight=False,
                        descriptor=down_input_map,
                        coordinates=(
                            0,
                            token_tile * self.n,
                            (split * self.tiles + stage) * (self.k // 128),
                        ),
                    )
            for tile in cutlass.range(self.stages, self.tiles, unroll=1):
                stage = tile % self.stages
                cute.arch.mbarrier_wait(empty + stage, (tile // self.stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full + stage, stage_bytes)
                self._operand_load(
                    gw[None, None, split * self.tiles + tile],
                    sw[None, None, None, stage],
                    (full + stage),
                    cta_v_map=map_w,
                    update_expect_tx=False,
                    cache_priority=cache_priority,
                    is_weight=True,
                    descriptor=down_weight_map,
                    coordinates=(
                        0,
                        output_tile * self.m,
                        (split * self.tiles + tile) * (self.k // 128),
                    ),
                )
                self._operand_load(
                    gx[None, None, split * self.tiles + tile],
                    sx[None, None, None, stage],
                    (full + stage),
                    cta_v_map=map_x,
                    update_expect_tx=False,
                    cache_priority=cache_priority,
                    is_weight=False,
                    descriptor=down_input_map,
                    coordinates=(
                        0,
                        token_tile * self.n,
                        (split * self.tiles + tile) * (self.k // 128),
                    ),
                )
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
            if cutlass.const_expr(self.issue_style == "cute"):
                mma.set(tcgen05.Field.ACCUMULATE, False)
                for tile in cutlass.range(self.tiles, unroll=1):
                    stage = tile % self.stages
                    cute.arch.mbarrier_wait(full + stage, tile // self.stages % 2)
                    for block in cutlass.range(
                        cute.size(a_desc, mode=[2]), unroll=self.mma_k_unroll
                    ):
                        cute.gemm(
                            mma,
                            acc,
                            a_desc[None, None, block, stage],
                            b_desc[None, None, block, stage],
                            acc,
                        )
                        mma.set(tcgen05.Field.ACCUMULATE, True)
                    with cute.arch.elect_one():
                        tcgen05.commit(empty + stage, None, tcgen05.CtaGroup.ONE)
                with cute.arch.elect_one():
                    tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
                cute.arch.mbarrier_wait(read_done, 0)
            elif cutlass.const_expr(self.issue_style == "raw_unrolled"):
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
            else:
                instruction = fp8_instruction(self.m, self.n)
                # One elected actor owns every issue, completion and ring release.
                with cute.arch.elect_one():
                    for tile in cutlass.range_constexpr(self.tiles):
                        stage = tile % self.stages
                        fp8_mbarrier_wait(full + stage, tile // self.stages % 2)
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
                    tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
                    fp8_mbarrier_wait(read_done, 0)
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
        source,
        destination,
        mbar,
        *,
        cta_v_map,
        update_expect_tx: cutlass.Constexpr,
        descriptor,
        coordinates,
        cache_priority,
        is_weight: cutlass.Constexpr,
    ):
        if cutlass.const_expr(update_expect_tx):
            raise ValueError(
                "Explicit TMA requires an existing transaction expectation"
            )
        cache_policy = None
        if cutlass.const_expr(self.weight_cache_hint and is_weight):
            cache_policy = _weight_eviction_policy(cache_priority)
        _explicit_tma_copy(
            destination.iterator, descriptor, mbar, coordinates, cache_policy
        )

    @cute.experimental.jit
    def _issue_entry_weight(
        self,
        source,
        destination,
        full,
        mapping,
        descriptor,
        output_tile,
        k_tile,
        cache_priority,
    ):
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(full, (self.m + self.n) * self.k)
        self._operand_load(
            source,
            destination,
            full,
            cta_v_map=mapping,
            update_expect_tx=False,
            cache_priority=cache_priority,
            is_weight=True,
            descriptor=descriptor,
            coordinates=(0, output_tile * self.m, k_tile * (self.k // 128)),
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
        down_weight_map,
        down_input_map,
        cache_priority,
    ):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        tiler = (self.m, self.n, self.k)
        mma, sw, sx = _make_down_operands(arena, tiler, self.stages)
        full, empty = barriers, barriers + self.stages
        done, read_done = empty + self.stages, empty + self.stages + 1
        w_view = cute.make_tensor(
            weight.iterator,
            cute.make_layout(
                (self.projection_rows, (64, self.inner // 64), 1),
                stride=(self.inner, (1, 64), 0),
            ),
        )
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
                        gw[None, None, k_tile],
                        sw[None, None, None, stage],
                        full + stage,
                        map_w,
                        down_weight_map,
                        output_tile,
                        k_tile,
                        cache_priority,
                    )
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.stages):
                    if cutlass.const_expr(stage >= self.early_entry_weights):
                        cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(empty + stage, 1)
                cute.arch.mbarrier_init(done, 1)
                cute.arch.mbarrier_init(read_done, self.epilogue_threads)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if warp == self.producer_warp:
            # The enclosing kernel already waited for input and weight producers.
            # Pair each stage's operands so its MMA can start before later loads.
            for stage in cutlass.range_constexpr(self.stages):
                k_tile = split * self.tiles + stage
                if cutlass.const_expr(stage >= self.early_entry_weights):
                    self._issue_entry_weight(
                        gw[None, None, k_tile],
                        sw[None, None, None, stage],
                        full + stage,
                        map_w,
                        down_weight_map,
                        output_tile,
                        k_tile,
                        cache_priority,
                    )
                self._operand_load(
                    gx[None, None, k_tile],
                    sx[None, None, None, stage],
                    (full + stage),
                    cta_v_map=map_x,
                    update_expect_tx=False,
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
        down_mma_unroll,
        up_tactic,
        up_prefetch_stages,
        up_prefetch_split,
        issue_style,
        partial_store_packet,
        native_down_partials,
        up_scale_cache,
        private_epoch_order,
        release_phase_arrivals,
        async_phase_arrivals,
        operand_transport,
        minimum_blocks_per_sm,
        weight_cache_hint,
        entry_down_prefill,
        early_entry_weights,
        activation_wait,
        quant_scale_mode,
        quant_max_mode,
        up_weight_handoff,
        up_barrier_placement,
        down_arrival_actor,
        up_activation_prefetch,
        up_followup_prefetch,
        up_refill_unroll,
        up_tile4_hint_stride,
        up_skip_final,
        up_wait_once,
        quant_max_join,
        quant_publication,
        entry_scale_prefetches,
        last_up_weight_prefetch,
        up_tmem_dealloc_site,
        up_unused_empty,
        up_residual_order,
        up_hidden_mix,
        up_final_part_mode,
        up_mix_input_prefetch,
        up_reader_pair_arithmetic,
        up_hidden_pair_mul,
        up_w2_issue_site,
        up_weight_is_packed,
    ):
        """Create static geometry; up_tactic selects channel parts, K and stages.

        The split value2 keeps64hidden channels per CTA and uses two M128
        accumulator parts for wide16/32/64-token plans. Value1 retains the
        original geometry. down_mma_unroll controls native Down K unrolling.
        """
        hint_site, hint_warps = last_up_weight_prefetch
        if hint_site == "off":
            if hint_warps:
                raise ValueError(
                    "Disabled final Up weight prefetch requires zero warps"
                )
        else:
            wide32 = (
                token_bucket == 32
                and up_tactic == (2, 128, 3)
                and up_weight_handoff == "retire_first_stage"
            )
            wide128 = (
                token_bucket == 128
                and up_tactic == (1, 128, 3)
                and up_weight_handoff == ("reuse_pair")
            )
            if (
                hint_site
                not in ("entry", "down_epilogue", "before_invalidate", "after_release")
                or hint_warps not in (4, 8, 16)
                or p != r + g
                or (not (wide32 or wide128))
                or (operand_transport[1] != "explicit")
            ):
                raise ValueError(
                    "The final Up weight prefetch requires the wide injection tile"
                )
        self.last_up_weight_prefetch = last_up_weight_prefetch
        if up_tmem_dealloc_site not in (
            "kernel_tail",
            "after_reads",
            "after_writes",
        ) or (
            up_tmem_dealloc_site != "kernel_tail"
            and last_up_weight_prefetch != ("before_invalidate", 16)
        ):
            raise ValueError("Early TMEM deallocation requires the wide injection tile")
        self.up_tmem_dealloc_site = up_tmem_dealloc_site
        if (
            type(up_unused_empty) is not int
            or up_unused_empty not in (0, 1, 2, 3)
            or (
                up_unused_empty
                and (
                    last_up_weight_prefetch != ("before_invalidate", 16)
                    or issue_style[1] not in ("cute", "cute_unrolled")
                )
            )
        ):
            raise ValueError(
                "Unused Up empty events require the wide three-slot producer"
            )
        self.up_unused_empty = up_unused_empty
        if up_scale_cache not in ("native", "row", "vector", "vector_register_rows"):
            raise ValueError("Select the Up scale layout explicitly")
        if up_scale_cache == "vector_register_rows" and (
            token_bucket != 256 or up_tactic[0] != 1
        ):
            raise ValueError(
                "Direct FP32 row scales require the wide256 token-major reader"
            )
        if up_scale_cache != "native" and token_bucket < 128:
            raise ValueError(
                "Register scale cache requires the native M128 token-major reader"
            )
        if type(entry_down_prefill) is not bool:
            raise ValueError("Entry Down prefill must be an explicit boolean")
        self.entry_down_prefill = entry_down_prefill
        if type(early_entry_weights) is not int or early_entry_weights not in (0, 1):
            raise ValueError("Choose zero or one early Down weight slot")
        if early_entry_weights and (
            not entry_down_prefill
            or early_entry_weights > down_tactic[3]
            or operand_transport[0] != "explicit"
        ):
            raise ValueError("Early Down weights require explicit entry prefill")
        self.up_scale_cache = up_scale_cache
        self.up_register_rows = up_scale_cache == "vector_register_rows"
        self.up_cached_weight_base = 0 if self.up_register_rows else 2 * token_bucket
        self.up_vector_weight_scales = up_scale_cache in (
            "vector",
            "vector_register_rows",
        )
        if private_epoch_order not in ("ordered", "store"):
            raise ValueError(
                "Private epoch order must explicitly select ordered or store"
            )
        self.private_epoch_order = private_epoch_order
        if type(release_phase_arrivals) is not bool:
            raise TypeError("Phase arrival ordering must be an explicit boolean")
        self.release_phase_arrivals = release_phase_arrivals
        if async_phase_arrivals not in ("native", "down", "activation", "both") or (
            async_phase_arrivals != "native" and (not release_phase_arrivals)
        ):
            raise ValueError(
                "Async phase publication requires the wide release-only arrivals"
            )
        self.async_down_arrival = async_phase_arrivals in ("down", "both")
        self.async_activation_arrival = async_phase_arrivals in ("activation", "both")
        if type(up_skip_final) is not bool:
            raise TypeError(
                "Up terminal barrier retirement requires an explicit boolean"
            )
        self.up_skip_final = up_skip_final
        if type(up_wait_once) is not bool:
            raise ValueError("Select the Up MMA wait policy explicitly")
        if quant_max_join not in ("native", "replicate", "warp_leader"):
            raise ValueError("Select the quantization maximum handoff explicitly")
        if quant_publication not in ("native", "merged", "writers"):
            raise ValueError("Select the activation publication handoff explicitly")
        if (
            quant_max_join != "native" or quant_publication != "native" or up_wait_once
        ) and token_bucket != 128:
            raise ValueError("The quantization handoff experiment requires wide128")
        self.up_wait_once = up_wait_once
        self.quant_publication = quant_publication
        if len(entry_scale_prefetches) != 3 or any(
            type(item) is not bool for item in entry_scale_prefetches
        ):
            raise ValueError("Select Down, Up and input scale prefetch explicitly")
        if any(entry_scale_prefetches) and token_bucket != 128:
            raise ValueError("Entry scale prefetch policies are calibrated for wide128")
        (
            self.prefetch_down_scales,
            self.prefetch_up_scales,
            self.prefetch_input_scales,
        ) = entry_scale_prefetches
        if len(operand_transport) != 2 or any(
            v not in ("native", "explicit") for v in operand_transport
        ):
            raise ValueError(
                "Down/Up transports must explicitly select native or explicit"
            )
        if "explicit" in operand_transport and (not (g == 4 and up_tactic[1] == 128)):
            raise ValueError("Explicit descriptors require native wide K128 Up")
        if type(minimum_blocks_per_sm) is not int or minimum_blocks_per_sm not in (
            0,
            1,
        ):
            raise ValueError("Minimum blocks per SM must explicitly select0or1")
        self.operand_transport = operand_transport
        if type(weight_cache_hint) is not bool:
            raise ValueError("Weight cache-hint selection must be an explicit bool")
        if weight_cache_hint and operand_transport != ("explicit", "explicit"):
            raise ValueError(
                "Weight eviction controls require both explicit transports"
            )
        self.weight_cache_hint = weight_cache_hint
        self.minimum_blocks_per_sm = minimum_blocks_per_sm
        if type(partial_store_packet) is not int or partial_store_packet not in (
            1,
            2,
            4,
        ):
            raise ValueError("Partial store packet must explicitly be1,2or4")
        if (
            p % partial_store_packet
            or token_bucket % partial_store_packet
            or (partial_store_packet == 4 and down_tactic[0] != 128)
        ):
            raise ValueError(
                "Store packet must match projection alignment and native reader geometry"
            )
        self.partial_store_packet = partial_store_packet
        if type(native_down_partials) is not bool:
            raise TypeError("Native Down partial layout requires an explicit boolean")
        if native_down_partials and (
            token_bucket not in (16, 32, 64, 128)
            or down_tactic[0] != 64
            or p not in (r, r + g)
            or (partial_store_packet != 1)
        ):
            raise ValueError("Native Down partial layout requires the wide M64 reader")
        self.native_down_partials = native_down_partials
        if len(issue_style) != 2 or any(
            style not in ("cute", "cute_unrolled", "raw_unrolled", "raw_loop_unrolled")
            for style in issue_style
        ):
            raise ValueError(
                "Down/Up issue styles must explicitly selectcute,cute_unrolled,raw_unrolledorraw_loop_unrolled"
            )
        self.down_issue_style, self.up_issue_style = issue_style
        up_channel_parts, up_k, up_stages = up_tactic
        if type(up_channel_parts) is not int or up_channel_parts not in (1, 2):
            raise ValueError("Up channel parts must be explicitly1or2")
        if up_channel_parts == 2 and (not (g == 4 and token_bucket <= 64)):
            raise ValueError("Split Up channels require the small wide geometry")
        self.up_channel_parts = up_channel_parts
        if type(up_k) is not int or up_k <= 0 or up_k % 64 or r % up_k:
            raise ValueError("Up K must be a positive multiple of64 dividing rank")
        if type(up_stages) is not int or not (1 <= up_stages <= r // up_k):
            raise ValueError("Up stages must fit the complete reduction")
        self.up_k = up_k
        self.up_stages = up_stages
        if (
            type(up_prefetch_stages) is not int
            or not 0 <= up_prefetch_stages <= up_stages
        ):
            raise ValueError("Up prefetch stages must explicitly fit the operand ring")
        self.up_prefetch_stages = up_prefetch_stages
        late_stages, late_stride, late_part = up_prefetch_split
        if (
            late_stages not in (0, 1, 2)
            or late_stride not in (1, 2, 4)
            or late_part not in (-1, 0, 1)
        ):
            raise ValueError(
                "Select the late Up stages, CTA stride and channel part explicitly"
            )
        if late_stages and (
            (p != r + g or token_bucket != 16)
            or up_tactic != (2, 128, 5)
            or up_prefetch_stages != 5
            or (up_weight_handoff != "retire_first_stage")
            or (up_barrier_placement != "handoff")
        ):
            raise ValueError("Split Up prefetch requires the five-stage wide16 handoff")
        self.up_late_prefetch_stages = late_stages
        self.up_late_prefetch_stride = late_stride
        self.up_late_prefetch_part = late_part
        if type(down_mma_unroll) is not int or down_mma_unroll not in (
            1,
            2,
            4,
            8,
            16,
            32,
            64,
        ):
            raise ValueError(
                "Down MMA unroll factor must be an explicit power of two in1..64"
            )
        self.down_mma_unroll = down_mma_unroll
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
        self.threads = 576
        self.compute_threads = 512
        self.producer_warp = 16
        self.mma_warp = 17
        # Independent row groups avoid serial quantization passes in large buckets.
        self.quant_rows = 2 if token_bucket == 256 else 1
        self.quant_threads = 512 // self.quant_rows
        self.quant_columns = max(
            2,
            1
            << (((r + self.quant_threads - 1) // self.quant_threads - 1).bit_length()),
        )
        if quant_scale_mode not in ("original", "clamp", "scales", "scales_clamp"):
            raise ValueError("Invalid quantization scale policy")
        if quant_max_mode not in ("shuffle", "warp", "warp_local"):
            raise ValueError("An explicit quantization maximum policy is required")
        if quant_max_mode == "warp_local" and self.quant_columns % 2:
            raise ValueError("Local magnitude reduction requires complete pairs")
        self.quantization = QuantizationConfig(
            threads=self.compute_threads,
            maximum_threads=self.quant_threads,
            injection_start=self.compute_threads,
            column_stride=8 if self.native_down_partials else 1,
            warp_max="pair" if quant_max_mode == "warp_local" else "redux",
            replicate_maximum=quant_max_join != "native",
            refined_scales=quant_scale_mode in ("scales", "scales_clamp"),
            clamp=(
                "ptx_lower" if quant_scale_mode in ("clamp", "scales_clamp") else "both"
            ),
            activation_store="native",
            release_writes=False,
        )
        self.up_hidden = 64
        self.up_channels = self.up_hidden * g
        if up_weight_is_packed and (
            self.up_k != 128 or self.operand_transport[1] != "explicit"
        ):
            raise ValueError("Packed Up tiles require the wide explicit FP8 transport")
        self.packed_up_hidden_tiles = h // self.up_hidden if up_weight_is_packed else 0
        self.up_parts = (
            up_channel_parts
            if up_channel_parts == 2
            else (2 if token_bucket == 256 else 1)
        )
        # Small narrow tiles avoid padding tokens to the native M128 reader.
        self.token_major_up = up_channel_parts == 1
        residual_orders = {
            "native": (0, False),
            "before_join": (1, False),
            "two_before_join": (2, False),
            "before_acquire": (1, True),
            "two_before_acquire": (2, True),
        }
        if up_residual_order not in residual_orders:
            raise ValueError("Select the Up residual issue point explicitly")
        if up_residual_order != "native" and (
            p != r + g or token_bucket not in (128, 256) or (not self.token_major_up)
        ):
            raise ValueError(
                "Early register residuals are scoped to the wide M128 reader"
            )
        self.up_residual_order = up_residual_order
        self.up_early_residual_count, self.up_residual_before_acquire = residual_orders[
            up_residual_order
        ]
        self.up_rows = (
            min(128, max(64, token_bucket)) if self.token_major_up else token_bucket
        )
        if activation_wait not in ("grid", "warp_sync", "warp_arrive"):
            raise ValueError("Invalid activation acquire scope")
        self.activation_wait = activation_wait
        scale_rows = self.n if self.token_major_up else self.up_rows
        # A direct alpha reader participates in the producer's acquire.
        self.up_scale_warps = (
            self.compute_threads // 32
            if self.up_register_rows
            else (scale_rows + 31) // 32
        )
        self.up_mn = (
            (self.up_rows, self.up_channels)
            if self.token_major_up
            else (self.up_channels // self.up_channel_parts, self.up_rows)
        )
        self.reader_groups = self.compute_threads // 128
        self.hidden_reader_columns = min(32, token_bucket // self.reader_groups)
        self.hidden_reader_passes = 1
        group_modes = (
            "reader_rows_group",
            "register_half_group",
            "reader_rows_group_no_join",
            "register_half_group_no_join",
        )
        reader_modes = (
            "reader_rows",
            "register_half",
            "register_half_early",
            *group_modes,
        )
        if up_hidden_mix not in ("native", "scalar_pair", "vector_pair", *reader_modes):
            raise ValueError("Select the hidden-major Up mix explicitly")
        if up_hidden_mix != "native" and (
            p != r + g
            or token_bucket not in (16, 32, 64)
            or self.token_major_up
            or (self.up_parts != 2)
        ):
            raise ValueError(
                "Adjacent hidden-column mixing requires the paired wide tile"
            )
        self.up_hidden_mix = up_hidden_mix
        self.up_reader_rows = up_hidden_mix in reader_modes
        self.up_register_gates = up_hidden_mix in (
            "register_half",
            "register_half_early",
            "register_half_group",
            "register_half_group_no_join",
        )
        self.up_group_gate_join = up_hidden_mix in group_modes
        self.up_skip_gate_tail = up_hidden_mix in (
            "reader_rows_group_no_join",
            "register_half_group_no_join",
        )
        self.up_early_reader_inputs = (
            self.up_group_gate_join or up_hidden_mix == "register_half_early"
        )
        if self.up_reader_rows and token_bucket != 16:
            raise ValueError(
                "The TMEM reader row mapping requires the four-row wide16 fragment"
            )
        if up_reader_pair_arithmetic not in (
            "separate",
            "scalar_pair",
            "packed_mul",
        ) or (up_reader_pair_arithmetic != "separate" and not self.up_reader_rows):
            raise ValueError(
                "Paired row arithmetic requires the TMEM reader output mapping"
            )
        self.up_reader_pair_arithmetic = up_reader_pair_arithmetic
        if type(up_hidden_pair_mul) is not bool or (
            up_hidden_pair_mul
            and (up_hidden_mix != "vector_pair" or token_bucket != 64)
        ):
            raise ValueError(
                "Packed FP32 multiplication requires the wide64 vector mix"
            )
        self.up_hidden_pair_mul = up_hidden_pair_mul
        self.reader_columns = self.up_hidden // self.reader_groups
        self.up_tiles = (r + self.up_k - 1) // self.up_k
        first, count, stride = up_activation_prefetch
        if type(count) is not int or count < 0 or stride not in (1, 2, 4):
            raise ValueError("Select an explicit Up activation-window prefetch")
        if count > 0 and (
            operand_transport[1] != "explicit"
            or not self.up_stages <= first < first + count <= self.up_tiles
        ):
            raise ValueError(
                "Activation-window prefetch requires explicit recycled Up tiles"
            )
        if count == 0 and (first, stride) != (0, 1):
            raise ValueError("Disabled Up prefetch requires (0, 0, 1)")
        self.up_activation_prefetch = (first, count, stride)
        followup_site, followup_tile, followup_stride = up_followup_prefetch
        if followup_site not in ("off", "before_acquire", "reader", "down_handoff"):
            raise ValueError("Select an explicit site for the second Up hint")
        if followup_site == "off":
            if (followup_tile, followup_stride) != (-1, 1):
                raise ValueError("Disabled followup prefetch requires (-1, 1)")
        elif (
            operand_transport[1] != "explicit"
            or not self.up_stages <= followup_tile < self.up_tiles
            or first <= followup_tile < first + count
            or (type(followup_stride) is not int)
            or (followup_stride != 1)
        ):
            raise ValueError(
                "The second hint must address a recycled tile outside the existing hints"
            )
        self.up_followup_prefetch = up_followup_prefetch
        if type(up_refill_unroll) is not bool:
            raise ValueError("Select a looped or unrolled Up producer explicitly")
        self.up_refill_unroll = up_refill_unroll
        if (
            type(up_tile4_hint_stride) is not int
            or up_tile4_hint_stride not in (0, 2, 4)
            or (
                up_tile4_hint_stride > 0
                and (
                    token_bucket != 256
                    or up_tactic != (1, 128, 3)
                    or operand_transport[1] != "explicit"
                    or (up_activation_prefetch != (7, 1, 1))
                )
            )
        ):
            raise ValueError("Select a tile 4 hint stride for the wide256 Up schedule")
        self.up_tile4_hint_stride = up_tile4_hint_stride
        # Token-major readers can write each completed token part independently.
        self.up_tail_tiles = (
            self.up_stages
            if self.token_major_up and self.up_parts == 2 and self.up_tiles > 1
            else 0
        )
        if up_final_part_mode not in (
            "native",
            "serial",
            "early_read",
            "split_wait",
            "split",
        ) or (
            up_final_part_mode != "native"
            and (
                (p != r + g or token_bucket != 16)
                or up_tactic != (2, 128, 5)
                or self.up_issue_style != "cute"
                or (not self.up_group_gate_join)
            )
        ):
            raise ValueError(
                "An independent Up tail requires the grouped wide16 readers"
            )
        self.up_final_part_mode = up_final_part_mode
        self.up_final_serial = up_final_part_mode != "native"
        self.up_final_read_early = up_final_part_mode in (
            "early_read",
            "split_wait",
            "split",
        )
        self.up_final_part_split = up_final_part_mode in ("split_wait", "split")
        if up_mix_input_prefetch not in ("none", "residual", "weight", "both") or (
            up_mix_input_prefetch != "none"
            and (
                not self.up_final_read_early
                or up_reader_pair_arithmetic != "scalar_pair"
            )
        ):
            raise ValueError("Mix input prefetch requires the wide16 paired reader")
        self.up_mix_input_prefetch = up_mix_input_prefetch
        self.up_prefetch_mix_residual = up_mix_input_prefetch in ("residual", "both")
        self.up_prefetch_mix_weight = up_mix_input_prefetch in ("weight", "both")
        self.early_up_weights = up_prefetch_stages > 0
        if down_arrival_actor not in ("thread0", "mma"):
            raise ValueError("Select the Down arrival actor explicitly")
        self.mma_down_arrival = down_arrival_actor == "mma"
        if self.mma_down_arrival and (
            not self.early_up_weights or not release_phase_arrivals
        ):
            raise ValueError("Down MMA arrival requires the wide release path")
        self.hidden_tiles = h // self.up_hidden
        self.up_stage_bytes = (
            self.up_mn[0] * self.up_parts + self.up_mn[1]
        ) * self.up_k
        self.up_operand_bytes = self.up_stage_bytes * self.up_stages
        self.up_barrier_count = (
            2 * self.up_stages + self.up_parts + 1 + int(self.up_final_part_split)
        )
        self.down_m, self.splits, down_k, down_stages = down_tactic
        if self.down_issue_style != "cute" and self.down_mma_unroll != down_k // 32:
            raise ValueError(
                "Fully unrolled FP8 issue requires the explicit completeK32block count"
            )
        if g * h % (self.splits * down_k) or not (
            1 <= down_stages <= g * h // (self.splits * down_k)
        ):
            raise ValueError("Down stages must fit an evenly partitioned reduction")
        if up_weight_handoff not in (
            "retire_first_stage",
            "retire_part0",
            "reuse_pair",
        ):
            raise ValueError("Select an explicit Up weight handoff")
        self.up_weight_handoff = up_weight_handoff
        self.up_early_weight_parts = {
            "retire_part0": (0,),
            "retire_first_stage": (0, 1),
            "reuse_pair": (),
        }[up_weight_handoff]
        if self.up_early_weight_parts and (
            self.up_channel_parts != 2 or self.token_major_up or self.up_k != 128
        ):
            raise ValueError("Early Up weight parts require the split-channel layout")
        if up_barrier_placement not in ("handoff", "entry"):
            raise ValueError("Select the Up barrier placement explicitly")
        self.up_barriers_at_entry = up_barrier_placement == "entry"
        if self.up_barriers_at_entry and up_weight_handoff != "reuse_pair":
            raise ValueError("Entry Up barriers require a retired Down weight prefix")
        if up_w2_issue_site not in ("activation_demand", "handoff"):
            raise ValueError("Select the Up W2 issue site explicitly")
        self.up_w2_issue_site = up_w2_issue_site
        expected_prefix = 3 if up_w2_issue_site == "handoff" else 2
        self.up_reuse_stages = 2 if up_weight_handoff == "reuse_pair" else 0
        if self.up_reuse_stages and (
            2 * self.down_m * down_k != g * self.up_hidden * self.up_k
            or g * h // (self.splits * down_k) % down_stages
            or not self.token_major_up
            or self.up_prefetch_stages != expected_prefix
        ):
            raise ValueError("An Up weight stage must fit two retired Down stages")
        if up_w2_issue_site == "handoff" and (
            not self.up_barriers_at_entry or self.up_prefetch_stages != expected_prefix
        ):
            raise ValueError("Issuing Up W2 at the handoff requires the entry barriers")
        self.up_early_weight_stages = self.up_reuse_stages
        self.separate_up_barriers = True
        self.down_token_parts = 1
        self.down_barrier_count = 2 * down_stages + 2
        self.down_tiles = (p + self.down_m - 1) // self.down_m
        self.down_tasks = self.down_tiles * self.splits
        self.up_tasks = self.hidden_tiles
        up_first_weight_copies = []
        if self.up_early_weight_parts:
            if self.down_tasks < self.up_tasks:
                raise ValueError(
                    "Every early Up task must follow a Down task on its CTA"
                )
            tiles = g * h // (self.splits * down_k)
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
        down_epilogue_threads = 256 if token_bucket == 16 else 512
        self.down = _DownPhase(
            up_reuse_stages=self.up_reuse_stages,
            up_stage_bytes=self.up_stage_bytes,
            up_barriers_at_entry=self.up_barriers_at_entry,
            up_first_weight_copies=tuple(copy for _, copy in up_first_weight_copies),
            early_entry_weights=early_entry_weights,
            packed_up_hidden_tiles=self.packed_up_hidden_tiles,
            mma_k_unroll=self.down_mma_unroll,
            partial_store_packet=self.partial_store_packet,
            native_down_partials=self.native_down_partials,
            operand_transport=self.operand_transport[0],
            weight_cache_hint=self.weight_cache_hint,
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
            separate_operand_ready=False,
            epilogue_threads=down_epilogue_threads,
            shared_up_scales=False,
            interleave_tma=False,
        )
        self.columns = max(128, token_bucket, self.up_mn[1] * self.up_parts)
        self.arena_bytes = max(
            (self.down_m + token_bucket) * down_k * down_stages,
            self.up_operand_bytes,
            0 if self.token_major_up else token_bucket * self.up_channels * 4,
        )
        self.barrier_count = max(self.down_barrier_count, self.up_barrier_count)
        if self.separate_up_barriers:
            self.barrier_count = self.down_barrier_count + self.up_barrier_count
        # A conservative occupancy query includes alignment and epilogue scratch.
        self.shared_bytes = (
            self.arena_bytes
            + (0 if self.token_major_up else token_bucket * self.up_channels)
            + (self.up_cached_weight_base + self.up_channels) * 4
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
            min_blocks_per_mp=self.minimum_blocks_per_sm,
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
                x,
                down,
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
        self._prefetch_last_up_weight_at("entry", up_weight_map, pid, ctas, warp)
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
            io.down,
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
                    _arrive_phase(
                        io.epochs.iterator,
                        self.release_phase_arrivals,
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
                    _arrive_phase(
                        io.epochs.iterator,
                        self.release_phase_arrivals,
                        self.async_down_arrival,
                    )
            else:
                _arrive_phase(
                    io.epochs.iterator,
                    self.release_phase_arrivals,
                    self.async_down_arrival,
                )
        self._prefetch_last_up_weight_at(
            "after_release", maps.up_weight_map, cta.pid, cta.count, cta.warp
        )
        if cta.pid < self.up_tasks:
            prepared_weight_stages = cutlass.Int32(0)
            if cutlass.const_expr(self.up_early_weight_stages > 0):
                if cta.pid < self.down_tasks:
                    prepared_weight_stages = cutlass.Int32(self.up_early_weight_stages)
            self._prefetch_up_weights(
                io.up,
                shared.arena,
                shared.up_barriers,
                cta.pid,
                maps.up_weight_map,
                cache_priority=maps.up_weight_priority,
                prepared_weight_stages=prepared_weight_stages,
                during_down=False,
                after_acquire=False,
            )
        if cta.tid == 0:
            while _acquire(io.epochs.iterator) < (phase_base + 1) * cta.count:
                pass
        cute.arch.sync_threads()
        if cutlass.const_expr(self.up_late_prefetch_stages):
            if cta.pid < self.up_tasks and cta.pid % self.up_late_prefetch_stride == 0:
                self._prefetch_up_weights(
                    io.up,
                    shared.arena,
                    shared.up_barriers,
                    cta.pid,
                    maps.up_weight_map,
                    cache_priority=maps.up_weight_priority,
                    prepared_weight_stages=cutlass.Int32(0),
                    during_down=False,
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
                self.release_phase_arrivals,
                self.async_activation_arrival,
            )
        else:
            # Keep the same release from each CTA. Only real readers acquire.
            cute.arch.sync_threads()
            if cta.tid == 0:
                _arrive_phase(
                    io.epochs.iterator,
                    self.release_phase_arrivals,
                    self.async_activation_arrival,
                )
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
                self.release_phase_arrivals,
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
        pointer = cute.make_ptr(
            dtype,
            (tensor.iterator + offset).toint(),
            cute.AddressSpace.smem,
            assumed_align=2 * dtype.width // 8,
        )
        source = cute.make_tensor(pointer, cute.make_layout(2))
        values = cute.make_rmem_tensor(2, dtype)
        cute.copy(
            cute.make_copy_atom(
                CopyUniversalOp(), dtype, num_bits_per_copy=2 * dtype.width
            ),
            source,
            values,
        )
        return values

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
            if cutlass.const_expr(self.up_hidden_mix == "vector_pair"):
                weights = self._load_hidden_pair(cached, offset, cutlass.Float32)
                for element in cutlass.range_constexpr(2):
                    scales[element, branch] = weights[element]
            else:
                for element in cutlass.range_constexpr(2):
                    scales[element, branch] = cached[offset + element]
        for batch in cutlass.range_constexpr(self.up_rows // rows_per_batch):
            row = tid // lanes_per_row + batch * rows_per_batch
            global_row = global_base + cutlass.Int64(row)
            if global_row < x.shape[0]:
                activation_scale, input_scale = cached[row], cached[self.up_rows + row]
                totals = cute.make_rmem_tensor(2, cutlass.Float32)
                totals.fill(0.0)
                for branch in cutlass.range_constexpr(self.g):
                    channel0 = branch * self.up_hidden + column
                    if cutlass.const_expr(self.up_hidden_mix == "vector_pair"):
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
                            if cutlass.const_expr(self.up_hidden_mix == "vector_pair"):
                                gate, residual = values[element], decoded[element]
                            else:
                                gate = gates[row, channel0 + element]
                                residual = residual_tile[row, channel0 + element].to(
                                    cutlass.Float32
                                )
                            factor = activation_scale * scales[element, branch]
                            xv = residual * input_scale
                            totals[element] = totals[element] + xv * _up_sigmoid_tanh(
                                gate * factor
                            )
                if cutlass.const_expr(self.up_hidden_mix == "vector_pair"):
                    packed = cute.make_rmem_tensor(2, out.element_type)
                    for element in cutlass.range_constexpr(2):
                        packed[element] = (totals[element] / self.g).to(
                            out.element_type
                        )
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
                else:
                    for element in cutlass.range_constexpr(2):
                        out[global_row, hidden + element] = (
                            totals[element] / self.g
                        ).to(out.element_type)

    @cute.experimental.jit
    def _write_reader_rows(
        self,
        gates,
        kept,
        residual_tile,
        cached,
        out,
        x,
        reader_channel,
        reader_row,
        hidden_tile,
        global_base,
        branch_group: cutlass.Constexpr,
    ):
        col = reader_channel % self.up_hidden
        hidden = hidden_tile * self.up_hidden + col
        scales = cute.make_rmem_tensor(self.g, cutlass.Float32)
        for branch in cutlass.range_constexpr(self.g):
            scales[branch] = cached[2 * self.up_rows + branch * self.up_hidden + col]
        for row_index in cutlass.range_constexpr(2):
            row = reader_row + 2 * branch_group + row_index
            global_row = global_base + cutlass.Int64(row)
            if global_row < x.shape[0]:
                activation_scale = cached[row]
                input_scale = cached[self.up_rows + row]
                total = cutlass.Float32(0.0)
                for branch in cutlass.range_constexpr(self.g):
                    channel = branch * self.up_hidden + col
                    if cutlass.const_expr(
                        self.up_register_gates and branch % 2 == branch_group
                    ):
                        gate = kept[row_index, branch // 2]
                    else:
                        gate = gates[row, channel]
                    factor = activation_scale * scales[branch]
                    residual = (
                        residual_tile[row, channel].to(cutlass.Float32) * input_scale
                    )
                    total = total + residual * _up_sigmoid_tanh(gate * factor)
                out[global_row, hidden] = (total / self.g).to(out.element_type)

    @cute.experimental.jit
    def _write_paired_reader_rows(
        self,
        gates,
        kept,
        residual_tile,
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
                if cutlass.const_expr(self.up_prefetch_mix_weight):
                    weight = weight_scales[branch]
                else:
                    weight = cached[2 * self.up_rows + channel]
                if cutlass.const_expr(
                    self.up_register_gates and branch % 2 == branch_group
                ):
                    gate0, gate1 = kept[0, branch // 2], kept[1, branch // 2]
                else:
                    gate0, gate1 = gates[row0, channel], gates[row1, channel]
                if cutlass.const_expr(self.up_prefetch_mix_residual):
                    residual0 = residual_codes[0, branch].to(cutlass.Float32)
                    residual1 = residual_codes[1, branch].to(cutlass.Float32)
                else:
                    residual0 = residual_tile[row0, channel].to(cutlass.Float32)
                    residual1 = residual_tile[row1, channel].to(cutlass.Float32)
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
        residual_tile,
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
        write = self._write_reader_rows
        inputs = ()
        if cutlass.const_expr(self.up_reader_pair_arithmetic != "separate"):
            write = self._write_paired_reader_rows
            inputs = (residual_codes, weight_scales)
        if reader_channel < self.up_hidden:
            write(
                gates,
                kept_gates,
                residual_tile,
                cached,
                out,
                x,
                reader_channel,
                reader_row,
                hidden_tile,
                global_base,
                0,
                *inputs,
            )
        else:
            write(
                gates,
                kept_gates,
                residual_tile,
                cached,
                out,
                x,
                reader_channel,
                reader_row,
                hidden_tile,
                global_base,
                1,
                *inputs,
            )

    @cute.experimental.jit
    def _read_mix_inputs(self, coords, cached, residual_tile):
        channel, row = coords[0]
        column = channel % self.up_hidden
        row0 = row + 2 * (channel // self.up_hidden)
        residual_codes, weight_scales = None, None
        if cutlass.const_expr(self.up_prefetch_mix_residual):
            residual_codes = cute.make_rmem_tensor((2, self.g), cutlass.Float8E4M3FN)
            for branch in cutlass.range_constexpr(self.g):
                column_in_branch = branch * self.up_hidden + column
                residual_codes[0, branch] = residual_tile[row0, column_in_branch]
                residual_codes[1, branch] = residual_tile[row0 + 1, column_in_branch]
        if cutlass.const_expr(self.up_prefetch_mix_weight):
            weight_scales = cute.make_rmem_tensor(self.g, cutlass.Float32)
            for branch in cutlass.range_constexpr(self.g):
                weight_scales[branch] = cached[
                    2 * self.up_rows + branch * self.up_hidden + column
                ]
        return residual_codes, weight_scales

    @cute.experimental.jit
    def _prefetch_up_residual_prefix(self, ring, coords, x, hidden_tile, global_base):
        for branch in cutlass.range_constexpr(self.up_early_residual_count):
            self._prefetch_branch_x(
                ring[None, branch], coords, x, hidden_tile, global_base, 0, branch
            )

    @cute.experimental.jit
    def _init_up_barriers(
        self, bars, prepared_weight_stages, use_existing_join: cutlass.Constexpr
    ):
        warp = cute.arch.make_warp_uniform(cute.arch.thread_idx()[0] // 32)
        full, empty = bars, bars + self.up_stages
        done = empty + self.up_stages
        read_done = done + self.up_parts
        if warp == self.mma_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.up_stages):
                    if stage >= prepared_weight_stages:
                        cute.arch.mbarrier_init(full + stage, 1)
                    cute.arch.mbarrier_init(empty + stage, 1)
                for part in cutlass.range_constexpr(self.up_parts):
                    cute.arch.mbarrier_init(done + part, 1)
                cute.arch.mbarrier_init(read_done, self.compute_threads)
                if cutlass.const_expr(self.up_final_part_split):
                    cute.arch.mbarrier_init(read_done + 1, 1)
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
        weight,
        arena,
        bars,
        hidden_tile,
        up_weight_map,
        cache_priority,
        prepared_weight_stages,
        during_down: cutlass.Constexpr,
        after_acquire: cutlass.Constexpr,
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
        gw = cute.local_tile(
            wv,
            (self.up_channels // self.up_channel_parts, self.up_k),
            (hidden_tile * self.up_channel_parts, None, 0),
        )
        extra_gw, extra_sw = None, None
        if cutlass.const_expr(self.up_channel_parts == 2):
            extra_gw = cute.local_tile(
                wv,
                (self.up_channels // self.up_channel_parts, self.up_k),
                (hidden_tile * self.up_channel_parts + 1, None, 0),
            )
            extra_sw = _operand_at(arena, al, cute.cosize(bl) + cute.cosize(al))
        if cutlass.const_expr(self.token_major_up):
            mapping = cute_ext.get_cta_v_map_ab(wv, tiler, mma, "B")
        else:
            mapping = cute_ext.get_cta_v_map_ab(wv, tiler, mma, "A")
        warp = cute.arch.make_warp_uniform(cute.arch.thread_idx()[0] // 32)
        if cutlass.const_expr(during_down):
            # Only the producer warp reaches this call. The remaining
            # barriers are initialized before any Up activation load.
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.up_prefetch_stages):
                    cute.arch.mbarrier_init(bars + stage, 1)
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
        elif cutlass.const_expr(after_acquire):
            # These events were initialized by the original local Down handoff.
            pass
        elif cutlass.const_expr(not self.up_barriers_at_entry):
            initialized_full = prepared_weight_stages
            if cutlass.const_expr(len(self.up_early_weight_parts) > 0):
                initialized_full = cutlass.Int32(1)
            self._init_up_barriers(bars, initialized_full, use_existing_join=False)
        if warp == self.producer_warp:
            handoff_stages = self.up_prefetch_stages - int(
                self.up_w2_issue_site == "before_acquire"
            )
            late_start = handoff_stages - self.up_late_prefetch_stages
            for stage in cutlass.range_constexpr(
                late_start if after_acquire else 0, handoff_stages
            ):
                if stage >= prepared_weight_stages:
                    arm_stage = cutlass.Boolean(True)
                    issue_part0 = cutlass.Boolean(True)
                    issue_part1 = cutlass.Boolean(True)
                    if cutlass.const_expr(after_acquire):
                        # The before-acquire half already armed the full event.
                        arm_stage = cutlass.Boolean(self.up_late_prefetch_part == -1)
                        issue_part0 = cutlass.Boolean(
                            self.up_late_prefetch_part in (-1, 0)
                        )
                        issue_part1 = cutlass.Boolean(
                            self.up_late_prefetch_part in (-1, 1)
                        )
                    elif cutlass.const_expr(stage >= late_start):
                        defer = hidden_tile % self.up_late_prefetch_stride == 0
                        issue_part0 = not defer
                        if cutlass.const_expr(self.up_late_prefetch_part in (-1, 1)):
                            issue_part1 = not defer
                        if cutlass.const_expr(self.up_late_prefetch_part == -1):
                            arm_stage = not defer
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
                                gw[None, None, stage],
                                sw[None, None, None, stage],
                                bars + stage,
                                cta_v_map=mapping,
                                update_expect_tx=False,
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
                                extra_gw[None, None, stage],
                                extra_sw[None, None, None, stage],
                                bars + stage,
                                cta_v_map=mapping,
                                update_expect_tx=False,
                                weight_map=up_weight_map,
                                input_map=up_weight_map,
                                hidden_tile=hidden_tile,
                                tile=stage,
                                operand=("B" if self.token_major_up else "A"),
                                part=1,
                                cache_priority=cache_priority,
                            )
            if cutlass.const_expr(not during_down and not after_acquire):
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
        source,
        destination,
        mbar,
        *,
        cta_v_map,
        update_expect_tx: cutlass.Constexpr,
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
        if cutlass.const_expr(update_expect_tx):
            raise ValueError(
                "Explicit TMA requires an existing transaction expectation"
            )
        cache_policy = None
        if cutlass.const_expr(self.weight_cache_hint and is_weight):
            cache_policy = _weight_eviction_policy(cache_priority)
        _explicit_tma_copy(
            destination.iterator, descriptor, mbar, coordinates, cache_policy
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
    def _retire_up_slot(self, empty, slot, tile):
        notify = cutlass.Boolean(True)
        if cutlass.const_expr(self.up_unused_empty):
            notify = tile < self.up_tiles - self.up_unused_empty
        if notify:
            with cute.arch.elect_one():
                tcgen05.commit(empty + slot, None, tcgen05.CtaGroup.ONE)

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
        up_weight_map,
        up_input_map,
        cache_priority,
        activation_counter,
        activation_target,
        terminal_up_task,
    ):
        tid = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(tid // 32)
        cached = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(self.up_cached_weight_base + self.up_channels),
            alignment=16,
        )
        residual_tile = None
        reader_channel = cutlass.Int32(0)
        reader_row = cutlass.Int32(0)
        kept_gates = None
        if cutlass.const_expr(self.up_register_gates):
            kept_gates = cute.make_rmem_tensor((2, self.up_parts), cutlass.Float32)
        if cutlass.const_expr(not self.token_major_up):
            residual_tile = cute_ext.allocate(
                cutlass.Float8E4M3FN,
                cute.AddressSpace.smem,
                cute.make_layout(
                    (self.up_rows, self.up_channels), stride=(self.up_channels, 1)
                ),
                alignment=16,
            )
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
        sw = _operand_at(arena, bl, 0)
        sa0 = _operand_at(arena, al, cute.cosize(bl))
        sa1 = _operand_at(arena, al, cute.cosize(bl) + cute.cosize(al))
        full, empty = bars, bars + self.up_stages
        done = empty + self.up_stages
        read_done = done + self.up_parts
        if not weights_prepared:
            self._init_up_barriers(bars, cutlass.Int32(0), use_existing_join=False)
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
        gw = cute.local_tile(
            wv,
            (self.up_channels // self.up_channel_parts, self.up_k),
            (hidden_tile * self.up_channel_parts, None, 0),
        )
        gw1 = gw
        if cutlass.const_expr(self.up_channel_parts == 2):
            gw1 = cute.local_tile(
                wv,
                (self.up_channels // self.up_channel_parts, self.up_k),
                (hidden_tile * self.up_channel_parts + 1, None, 0),
            )
        if cutlass.const_expr(self.token_major_up):
            ga0, ga1, gb = gx0, gx1, gw
            ma = cute_ext.get_cta_v_map_ab(av, tiler, mma, "A")
            mb = cute_ext.get_cta_v_map_ab(wv, tiler, mma, "B")
        else:
            ga0, ga1, gb = gw, gw1, gx0
            ma = cute_ext.get_cta_v_map_ab(wv, tiler, mma, "A")
            mb = cute_ext.get_cta_v_map_ab(av, tiler, mma, "B")
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
                load_a = cutlass.Boolean(True)
                load_b = cutlass.Boolean(True)
                if cutlass.const_expr(self.token_major_up):
                    load_b = not stage_prepared
                else:
                    load_a = not stage_prepared
                if not stage_prepared:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            full + slot,
                            self.up_stage_bytes,
                        )
                if load_b:
                    self._up_operand_load(
                        gb[None, None, tile],
                        sw[None, None, None, slot],
                        (full + slot),
                        cta_v_map=mb,
                        update_expect_tx=False,
                        weight_map=up_weight_map,
                        input_map=up_input_map,
                        hidden_tile=hidden_tile,
                        tile=tile,
                        operand="B",
                        part=0,
                        cache_priority=cache_priority,
                    )
                if load_a:
                    self._up_operand_load(
                        ga0[None, None, tile],
                        sa0[None, None, None, slot],
                        (full + slot),
                        cta_v_map=ma,
                        update_expect_tx=False,
                        weight_map=up_weight_map,
                        input_map=up_input_map,
                        hidden_tile=hidden_tile,
                        tile=tile,
                        operand="A",
                        part=0,
                        cache_priority=cache_priority,
                    )
                if cutlass.const_expr(self.up_parts == 2):
                    load_a1 = cutlass.Boolean(True)
                    if cutlass.const_expr(not self.token_major_up):
                        load_a1 = not stage_prepared
                    if load_a1:
                        self._up_operand_load(
                            ga1[None, None, tile],
                            sa1[None, None, None, slot],
                            (full + slot),
                            cta_v_map=ma,
                            update_expect_tx=False,
                            weight_map=up_weight_map,
                            input_map=up_input_map,
                            hidden_tile=hidden_tile,
                            tile=tile,
                            operand="A",
                            part=1,
                            cache_priority=cache_priority,
                        )
            for tile in cutlass.range(
                self.up_stages,
                self.up_tiles - int(self.up_final_part_split),
                unroll=1,
                unroll_full=self.up_refill_unroll,
            ):
                slot = tile % self.up_stages
                cute.arch.mbarrier_wait(empty + slot, (tile // self.up_stages - 1) % 2)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + slot,
                        self.up_stage_bytes,
                    )
                self._up_operand_load(
                    gb[None, None, tile],
                    sw[None, None, None, slot],
                    (full + slot),
                    cta_v_map=mb,
                    update_expect_tx=False,
                    weight_map=up_weight_map,
                    input_map=up_input_map,
                    hidden_tile=hidden_tile,
                    tile=tile,
                    operand="B",
                    part=0,
                    cache_priority=cache_priority,
                )
                self._up_operand_load(
                    ga0[None, None, tile],
                    sa0[None, None, None, slot],
                    (full + slot),
                    cta_v_map=ma,
                    update_expect_tx=False,
                    weight_map=up_weight_map,
                    input_map=up_input_map,
                    hidden_tile=hidden_tile,
                    tile=tile,
                    operand="A",
                    part=0,
                    cache_priority=cache_priority,
                )
                if cutlass.const_expr(self.up_parts == 2):
                    self._up_operand_load(
                        ga1[None, None, tile],
                        sa1[None, None, None, slot],
                        (full + slot),
                        cta_v_map=ma,
                        update_expect_tx=False,
                        weight_map=up_weight_map,
                        input_map=up_input_map,
                        hidden_tile=hidden_tile,
                        tile=tile,
                        operand="A",
                        part=1,
                        cache_priority=cache_priority,
                    )
            if cutlass.const_expr(self.up_final_part_split):
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
                    gb[None, None, tile],
                    sw[None, None, None, slot],
                    first_full,
                    cta_v_map=mb,
                    update_expect_tx=False,
                    weight_map=up_weight_map,
                    input_map=up_input_map,
                    hidden_tile=hidden_tile,
                    tile=tile,
                    operand="B",
                    part=0,
                    cache_priority=cache_priority,
                )
                self._up_operand_load(
                    ga0[None, None, tile],
                    sa0[None, None, None, slot],
                    first_full,
                    cta_v_map=ma,
                    update_expect_tx=False,
                    weight_map=up_weight_map,
                    input_map=up_input_map,
                    hidden_tile=hidden_tile,
                    tile=tile,
                    operand="A",
                    part=0,
                    cache_priority=cache_priority,
                )
                self._up_operand_load(
                    ga1[None, None, tile],
                    sa1[None, None, None, slot],
                    second_full,
                    cta_v_map=ma,
                    update_expect_tx=False,
                    weight_map=up_weight_map,
                    input_map=up_input_map,
                    hidden_tile=hidden_tile,
                    tile=tile,
                    operand="A",
                    part=1,
                    cache_priority=cache_priority,
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
            if cutlass.const_expr(self.up_issue_style == "cute"):
                mma.set(tcgen05.Field.ACCUMULATE, False)
                for tile in cutlass.range(
                    self.up_tiles - self.up_tail_tiles - int(self.up_final_serial),
                    unroll=1,
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
                        if cutlass.const_expr(self.up_parts == 2):
                            cute.gemm(
                                mma,
                                acc1,
                                ad1[None, None, block, slot],
                                bd[None, None, block, slot],
                                acc1,
                            )
                        mma.set(tcgen05.Field.ACCUMULATE, True)
                    self._retire_up_slot(empty, slot, tile)
                if cutlass.const_expr(self.up_final_serial):
                    tile = self.up_tiles - 1
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
                    with cute.arch.elect_one():
                        tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
                    if cutlass.const_expr(self.up_final_part_mode == "split"):
                        cute.arch.mbarrier_wait(read_done + 1, 0)
                    for block in cutlass.range_constexpr(cute.size(ad1, mode=[2])):
                        cute.gemm(
                            mma,
                            acc1,
                            ad1[None, None, block, slot],
                            bd[None, None, block, slot],
                            acc1,
                        )
                    self._retire_up_slot(empty, slot, tile)
                    with cute.arch.elect_one():
                        tcgen05.commit(done + 1, None, tcgen05.CtaGroup.ONE)
                elif cutlass.const_expr(self.up_parts == 2):
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
                            tcgen05.commit(empty + slot, None, tcgen05.CtaGroup.ONE)
                    with cute.arch.elect_one():
                        tcgen05.commit(done + 1, None, tcgen05.CtaGroup.ONE)
                else:
                    with cute.arch.elect_one():
                        tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
                cute.arch.mbarrier_wait(read_done, 0)
            elif cutlass.const_expr(self.up_issue_style == "cute_unrolled"):
                mma.set(tcgen05.Field.ACCUMULATE, False)
                for tile in cutlass.range_constexpr(self.up_tiles - self.up_tail_tiles):
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
                        if cutlass.const_expr(self.up_parts == 2):
                            cute.gemm(
                                mma,
                                acc1,
                                ad1[None, None, block, slot],
                                bd[None, None, block, slot],
                                acc1,
                            )
                        mma.set(tcgen05.Field.ACCUMULATE, True)
                    self._retire_up_slot(empty, slot, tile)
                if cutlass.const_expr(self.up_parts == 2):
                    for tile in cutlass.range_constexpr(
                        self.up_tiles - self.up_tail_tiles, self.up_tiles
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
                    for tile in cutlass.range_constexpr(
                        self.up_tiles - self.up_tail_tiles, self.up_tiles
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
                            tcgen05.commit(empty + slot, None, tcgen05.CtaGroup.ONE)
                    with cute.arch.elect_one():
                        tcgen05.commit(done + 1, None, tcgen05.CtaGroup.ONE)
                else:
                    with cute.arch.elect_one():
                        tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
                cute.arch.mbarrier_wait(read_done, 0)
            elif cutlass.const_expr(self.up_issue_style == "raw_unrolled"):
                instruction = fp8_instruction(self.up_mn[0], self.up_mn[1])
                for tile in cutlass.range_constexpr(self.up_tiles - self.up_tail_tiles):
                    slot = tile % self.up_stages
                    cute.arch.mbarrier_wait(full + slot, tile // self.up_stages % 2)
                    with cute.arch.elect_one():
                        for block in cutlass.range_constexpr(cute.size(ad0, mode=[2])):
                            accumulate = (tile != 0) | (block != 0)
                            fp8_mma_encoded(
                                acc0,
                                fp8_descriptor_at(ad0, block, slot),
                                fp8_descriptor_at(bd, block, slot),
                                instruction,
                                accumulate,
                            )
                            if cutlass.const_expr(self.up_parts == 2):
                                fp8_mma_encoded(
                                    acc1,
                                    fp8_descriptor_at(ad1, block, slot),
                                    fp8_descriptor_at(bd, block, slot),
                                    instruction,
                                    accumulate,
                                )
                        tcgen05.commit(empty + slot, None, tcgen05.CtaGroup.ONE)
                if cutlass.const_expr(self.up_parts == 2):
                    for tile in cutlass.range_constexpr(
                        self.up_tiles - self.up_tail_tiles, self.up_tiles
                    ):
                        slot = tile % self.up_stages
                        cute.arch.mbarrier_wait(full + slot, tile // self.up_stages % 2)
                        with cute.arch.elect_one():
                            for block in cutlass.range_constexpr(
                                cute.size(ad0, mode=[2])
                            ):
                                fp8_mma_encoded(
                                    acc0,
                                    fp8_descriptor_at(ad0, block, slot),
                                    fp8_descriptor_at(bd, block, slot),
                                    instruction,
                                    (tile != 0) | (block != 0),
                                )
                    with cute.arch.elect_one():
                        tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
                    for tile in cutlass.range_constexpr(
                        self.up_tiles - self.up_tail_tiles, self.up_tiles
                    ):
                        slot = tile % self.up_stages
                        with cute.arch.elect_one():
                            for block in cutlass.range_constexpr(
                                cute.size(ad1, mode=[2])
                            ):
                                fp8_mma_encoded(
                                    acc1,
                                    fp8_descriptor_at(ad1, block, slot),
                                    fp8_descriptor_at(bd, block, slot),
                                    instruction,
                                    (tile != 0) | (block != 0),
                                )
                            tcgen05.commit(empty + slot, None, tcgen05.CtaGroup.ONE)
                    with cute.arch.elect_one():
                        tcgen05.commit(done + 1, None, tcgen05.CtaGroup.ONE)
                else:
                    with cute.arch.elect_one():
                        tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
                cute.arch.mbarrier_wait(read_done, 0)
            else:
                instruction = fp8_instruction(self.up_mn[0], self.up_mn[1])
                with cute.arch.elect_one():
                    for tile in cutlass.range_constexpr(
                        self.up_tiles - self.up_tail_tiles
                    ):
                        slot = tile % self.up_stages
                        fp8_mbarrier_wait(full + slot, tile // self.up_stages % 2)
                        for block in cutlass.range_constexpr(cute.size(ad0, mode=[2])):
                            accumulate = (tile != 0) | (block != 0)
                            fp8_mma_encoded(
                                acc0,
                                fp8_descriptor_at(ad0, block, slot),
                                fp8_descriptor_at(bd, block, slot),
                                instruction,
                                accumulate,
                            )
                            if cutlass.const_expr(self.up_parts == 2):
                                fp8_mma_encoded(
                                    acc1,
                                    fp8_descriptor_at(ad1, block, slot),
                                    fp8_descriptor_at(bd, block, slot),
                                    instruction,
                                    accumulate,
                                )
                        tcgen05.commit(empty + slot, None, tcgen05.CtaGroup.ONE)
                    if cutlass.const_expr(self.up_parts == 2):
                        for tile in cutlass.range_constexpr(
                            self.up_tiles - self.up_tail_tiles, self.up_tiles
                        ):
                            slot = tile % self.up_stages
                            fp8_mbarrier_wait(full + slot, tile // self.up_stages % 2)
                            for block in cutlass.range_constexpr(
                                cute.size(ad0, mode=[2])
                            ):
                                fp8_mma_encoded(
                                    acc0,
                                    fp8_descriptor_at(ad0, block, slot),
                                    fp8_descriptor_at(bd, block, slot),
                                    instruction,
                                    (tile != 0) | (block != 0),
                                )
                        tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
                        for tile in cutlass.range_constexpr(
                            self.up_tiles - self.up_tail_tiles, self.up_tiles
                        ):
                            slot = tile % self.up_stages
                            for block in cutlass.range_constexpr(
                                cute.size(ad1, mode=[2])
                            ):
                                fp8_mma_encoded(
                                    acc1,
                                    fp8_descriptor_at(ad1, block, slot),
                                    fp8_descriptor_at(bd, block, slot),
                                    instruction,
                                    (tile != 0) | (block != 0),
                                )
                            tcgen05.commit(empty + slot, None, tcgen05.CtaGroup.ONE)
                        tcgen05.commit(done + 1, None, tcgen05.CtaGroup.ONE)
                    else:
                        tcgen05.commit(done, None, tcgen05.CtaGroup.ONE)
                    fp8_mbarrier_wait(read_done, 0)
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
                    if cutlass.const_expr(self.up_residual_before_acquire):
                        self._prefetch_up_residual_prefix(
                            first_ring, first_coords, x, hidden_tile, global_base
                        )
                self._await_activation_scales(warp)
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
                    channel = (
                        (tid // self.up_hidden) * self.h
                        + hidden_tile * self.up_hidden
                        + tid % self.up_hidden
                    )
                    cached[self.up_cached_weight_base + tid] = ws[
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
                    (first_template, first_coords, first_ring),
                    row_scale_source=(
                        "global"
                        if self.up_register_rows
                        else (
                            "shared_element"
                            if self.up_scale_cache == "native"
                            else "shared_row"
                        )
                    ),
                    weight_cache_offset=self.up_cached_weight_base,
                    vector_weight_scales=self.up_vector_weight_scales,
                    prefetched_branches=self.up_early_residual_count,
                    wait_once=self.up_wait_once,
                    release_after_reads=self.up_tmem_dealloc_site == "after_reads",
                )
                if cutlass.const_expr(self.up_tmem_dealloc_site != "after_reads"):
                    tcgen05_fence("before_thread_sync")
                    cute.arch.mbarrier_arrive(read_done)
            else:
                for copy_slice in cutlass.range_constexpr(
                    (self.up_rows * self.up_channels + self.compute_threads * 16 - 1)
                    // (self.compute_threads * 16)
                ):
                    linear = tid * 16 + copy_slice * self.compute_threads * 16
                    if linear < self.up_rows * self.up_channels:
                        local_row = linear // self.up_channels
                        channel = linear % self.up_channels
                        global_row = global_base + cutlass.Int64(local_row)
                        source = (
                            x.iterator
                            + global_row * (self.g * self.h)
                            + (channel // self.up_hidden) * self.h
                            + hidden_tile * self.up_hidden
                            + channel % self.up_hidden
                        )
                        size = cutlass.Int32(0)
                        if global_row < x.shape[0]:
                            size = cutlass.Int32(16)
                        _copy_residual_16(residual_tile.iterator + linear, source, size)
                _commit_residual()
                self._await_activation_scales(warp)
                if tid < self.up_rows:
                    global_row = global_base + cutlass.Int64(tid)
                    activation_scale = cutlass.Float32(0.0)
                    input_scale = cutlass.Float32(0.0)
                    if global_row < x.shape[0]:
                        activation_scale = _load_f32(
                            alpha.iterator + cutlass.Int64(tid)
                        )
                        input_scale = _x_scale(xs, global_row)
                    cached[tid] = activation_scale
                    cached[self.up_rows + tid] = input_scale
                if tid < self.up_channels:
                    channel = (
                        (tid // self.up_hidden) * self.h
                        + hidden_tile * self.up_hidden
                        + tid % self.up_hidden
                    )
                    cached[2 * self.up_rows + tid] = ws[
                        channel if self.up_per_channel else 0
                    ]
                if cutlass.const_expr(self.up_early_reader_inputs):
                    # Publish these CTA-shared operands while the MMA is
                    # in flight. They are read by every token-row group.
                    _wait_residual()
                    _epilogue_barrier(self.compute_threads, cutlass.Int32(5))
                mix_codes, mix_scales = None, None
                if cutlass.const_expr(
                    self.up_prefetch_mix_residual or self.up_prefetch_mix_weight
                ):
                    _, _, _, coordinates = self._hidden_reader(mma, base, x, tid, 0, 0)
                    mix_codes, mix_scales = self._read_mix_inputs(
                        coordinates, cached, residual_tile
                    )
                gate_offset = 0
                if cutlass.const_expr(self.up_final_read_early):
                    # The first weight slot was last used by K5. At the
                    # first done event only K7's third slot can still be live.
                    gate_offset = cute.cosize(bl)
                else:
                    for part in cutlass.range_constexpr(self.up_parts):
                        cute.arch.mbarrier_wait(done + part, 0)
                gates = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float32,
                        (arena.iterator + gate_offset).toint(),
                        cute.AddressSpace.smem,
                        assumed_align=1024,
                    ),
                    cute.make_layout(
                        (self.up_rows, self.up_channels), stride=(self.up_channels, 1)
                    ),
                )
                for part in cutlass.range_constexpr(self.up_parts):
                    if cutlass.const_expr(self.up_final_read_early):
                        cute.arch.mbarrier_wait(done + part, 0)
                    for token_slice in cutlass.range_constexpr(
                        self.hidden_reader_passes
                    ):
                        acc, reader, values, coords = self._hidden_reader(
                            mma, base, x, tid, token_slice, part
                        )
                        cute_ext.partition_and_copy(reader, acc, values)
                        cute.arch.fence_view_async_tmem_load()
                        part_base = part * (self.up_channels // self.up_channel_parts)
                        if cutlass.const_expr(self.up_reader_rows):
                            reader_channel, reader_row = coords[0]
                        if cutlass.const_expr(self.up_register_gates):
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
                if cutlass.const_expr(not self.up_early_reader_inputs):
                    _wait_residual()
                cute.arch.mbarrier_arrive(read_done)
                if cutlass.const_expr(self.up_group_gate_join):
                    # Four reader warps exchange the gates for their four rows.
                    _epilogue_barrier(128, cutlass.Int32(6) + tid // 128)
                    self._mix_reader_group(
                        gates,
                        kept_gates,
                        residual_tile,
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
        if cutlass.const_expr(not self.up_skip_gate_tail):
            cute.arch.sync_threads()
        if cutlass.const_expr(not self.token_major_up and not self.up_group_gate_join):
            if tid < self.compute_threads:
                gates = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Float32,
                        arena.iterator.toint(),
                        cute.AddressSpace.smem,
                        assumed_align=1024,
                    ),
                    cute.make_layout(
                        (self.up_rows, self.up_channels), stride=(self.up_channels, 1)
                    ),
                )
                if cutlass.const_expr(self.up_reader_rows):
                    self._mix_reader_group(
                        gates,
                        kept_gates,
                        residual_tile,
                        cached,
                        out,
                        x,
                        reader_channel,
                        reader_row,
                        hidden_tile,
                        global_base,
                        None,
                        None,
                    )
                elif cutlass.const_expr(self.up_hidden_mix != "native"):
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
                    col = tid % self.up_hidden
                    hidden = hidden_tile * self.up_hidden + col
                    scales = cute.make_rmem_tensor(self.g, cutlass.Float32)
                    for branch in cutlass.range_constexpr(self.g):
                        scales[branch] = cached[
                            2 * self.up_rows + branch * self.up_hidden + col
                        ]
                    for row_part in cutlass.range_constexpr(
                        self.up_rows // (self.compute_threads // self.up_hidden)
                    ):
                        row = tid // self.up_hidden + row_part * (
                            self.compute_threads // self.up_hidden
                        )
                        global_row = global_base + cutlass.Int64(row)
                        if global_row < x.shape[0]:
                            activation_scale = cached[row]
                            input_scale = cached[self.up_rows + row]
                            total = cutlass.Float32(0.0)
                            for branch in cutlass.range_constexpr(self.g):
                                gate = gates[row, branch * self.up_hidden + col]
                                factor = activation_scale * scales[branch]
                                residual = (
                                    residual_tile[
                                        row, branch * self.up_hidden + col
                                    ].to(cutlass.Float32)
                                    * input_scale
                                )
                                total = total + residual * _up_sigmoid_tanh(
                                    gate * factor
                                )
                            out[global_row, hidden] = (total / self.g).to(
                                out.element_type
                            )
