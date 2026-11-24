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
from cuda.bindings.driver import CUstream
from cutlass._mlir.dialects import llvm, nvvm
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
    _arrive_phase_release,
    _decode_residual,
    _grid_barrier,
    _init_mma_barriers,
    _make_down_fragments,
    _make_down_operands,
    _mma_barriers,
    _mul_hidden_float_pair,
    _operand_at,
    _run_cute_mma_ring,
    _store_private_epoch,
    _wait_residual,
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


class _DownPhase(ResidentDown):
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

        self.down = _DownPhase(
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
