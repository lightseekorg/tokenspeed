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

"""Blackwell fused gated residual with distributed cluster reduction and TMA pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import cutlass.utils as cute_utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cuda.bindings.driver import CUstream
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute import math as cute_math
from cutlass.cute.nvgpu import CopyUniversalOp, cpasync, tcgen05
from cutlass.cutlass_dsl import T
from cutlass.experimental.primitives import Tcgen05InstrDesc, tcgen05_fence


@cute.jit
def _up_sigmoid_tanh(value):
    # Keep gate arithmetic in FP32. One native tanh replaces exp plus reciprocal;
    # the original FP64 mixer oracle remains the numerical acceptance boundary.
    scaled = value * 0.5
    result = cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [scaled.ir_value()],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            asm_dialect=0,
        )
    )
    return result * 0.5 + 0.5


@cute.jit
def _load_partial_fp32(ptr):
    # Workspace contents change between graph replays. Peer values come from
    # coherent L2 after the release/acquire publication, bypassing stale L1.
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [ptr.toint().ir_value()],
            "ld.global.cg.f32 $0, [$1];",
            "=f,l,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )
    )


@cute.jit
def _load_partial_fp32x4(ptr):
    # Callers provide four valid columns at a 16-byte-aligned L2 address.
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [ptr.toint().ir_value()],
        "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )
    return (
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [0])),
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [1])),
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [2])),
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [3])),
    )


@cute.jit
def _up_weight_cache_policy():
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            asm_dialect=0,
        )
    )


@cute.jit
def _down_x_cache_policy():
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


@cute.jit
def _prefetch_up_weight_l2(pointer, byte_count: cutlass.Constexpr):
    # Match the real Up TMA eviction policy; this hint never owns operand SMEM.
    llvm.inline_asm(
        None,
        [
            pointer.toint().ir_value(),
            cutlass.Int32(byte_count).ir_value(),
            _up_weight_cache_policy().ir_value(),
        ],
        "cp.async.bulk.prefetch.L2.global.L2::cache_hint [$0], $1, $2;",
        "l,r,l,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.jit
def _publish_activation_proxy():
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.async.global;",
        "~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.jit
def _epilogue_barrier(threads: cutlass.Constexpr, barrier_id: cutlass.Int32):
    nvvm.barrier_cta_sync(
        barrier_id.ir_value(),
        thread_count=cutlass.Int32(threads).ir_value(),
        aligned=False,
    )


@cute.jit
def _load_tail(ptr, valid, dtype: cutlass.Constexpr):
    """Load a valid half input as FP32, or return zero without accessing memory."""
    # Convert inside the asm instead of returning a partially defined 16-bit
    # register; masked-off loads must not depend on its previous contents.
    conversion = "cvt.f32.bf16" if dtype == cutlass.BFloat16 else "cvt.f32.f16"
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [ptr.toint().ir_value(), cutlass.Int32(valid).ir_value()],
            "{ .reg .pred p; .reg .b16 v; setp.ne.u32 p, $2, 0; mov.b16 v, 0; @p ld.global.b16 v, [$1]; "
            + conversion
            + " $0, v; }",
            "=f,l,r,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )
    )


@cute.jit
def _store_tail(ptr, value, valid):
    llvm.inline_asm(
        None,
        [
            ptr.toint().ir_value(),
            llvm.bitcast(T.i16(), value.ir_value()),
            cutlass.Int32(valid).ir_value(),
        ],
        "{ .reg .pred p; setp.ne.u32 p, $2, 0; @p st.global.b16 [$0], $1; }",
        "l,h,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.jit
def _load_epoch_acquire(ptr):
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [ptr.toint().ir_value()],
            "ld.global.acquire.gpu.u64 $0, [$1];",
            "=l,l",
            has_side_effects=True,
            asm_dialect=0,
        )
    )


@cute.jit
def _arrive_counter(ptr):
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [ptr.toint().ir_value()],
            "atom.global.acq_rel.gpu.add.u64 $0, [$1], 1;",
            "=l,l,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )
    )


@cute.jit
def _arrive_counter_step(ptr, step: cutlass.Constexpr):
    return cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [ptr.toint().ir_value()],
            f"atom.global.acq_rel.gpu.add.u64 $0, [$1], {step};",
            "=l,l,~{memory}",
            has_side_effects=True,
            asm_dialect=0,
        )
    )


@cute.jit
def _store_epoch_release(ptr, value):
    llvm.inline_asm(
        None,
        [ptr.toint().ir_value(), value.ir_value()],
        "st.global.release.gpu.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.jit
def _mma_single_thread(acc, desc_a, desc_b, instruction, accumulate: cutlass.Constexpr):
    """Issue one native MMA; the caller must elect exactly one warp thread."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(acc.iterator.toint()).ir_value(),
            desc_a.ir_value(),
            desc_b.ir_value(),
            cutlass.Int32(instruction).ir_value(),
            cutlass.Int32(accumulate).ir_value(),
        ],
        "{ .reg .pred p; setp.ne.u32 p, $4, 0; "
        "tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, p; }",
        "r,l,l,r,r",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.jit
def _remote_partial(a, b, c, d, ptr, bar, peer):
    mapped = llvm.inline_asm(
        T.i32(),
        [ptr.toint().ir_value(), peer.ir_value()],
        "mapa.shared::cluster.u32 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        asm_dialect=0,
    )
    mapped_bar = llvm.inline_asm(
        T.i32(),
        [bar.toint().ir_value(), peer.ir_value()],
        "mapa.shared::cluster.u32 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        asm_dialect=0,
    )
    llvm.inline_asm(
        None,
        [mapped, a.ir_value(), b.ir_value(), c.ir_value(), d.ir_value(), mapped_bar],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$1, $2, $3, $4}, [$5];",
        "r,f,f,f,f,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.jit
def _cluster_arrive(bar):
    llvm.inline_asm(
        None,
        [bar.toint().ir_value()],
        "{ .reg .b32 dst; mapa.shared::cluster.u32 dst, $0, 0; "
        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [dst]; }",
        "r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@cute.jit
def _cluster_wait(bar, phase):
    llvm.inline_asm(
        None,
        [bar.toint().ir_value(), cutlass.Int32(phase).ir_value()],
        "{ .reg .pred ready; wait_loop: "
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 ready, [$0], $1; "
        "@!ready bra wait_loop; }",
        "r,r,~{memory}",
        has_side_effects=True,
        asm_dialect=0,
    )


@dataclass(frozen=True)
class _SharedStorage:
    gate: int
    down_weight: int
    down_x: int
    up_weight: int
    up_activation: int
    mailbox: int
    partial: int
    mix_accum: int
    barriers: int
    tmem: int
    nbytes: int


class FusedGatedResidualKernel:
    """Full warp groups drain TMEM; one warp loads weights and one issues MMA.

    Projection CTAs reduce M_proj / split_K columns in fixed split-rank order and
    writes post-scale/SiLU activation once. Down can use separate TMA producers
    or a single producer with a shared transaction credit for W and X. The first
    W stage can precede input readiness; later initial stages pair W and X.
    Up recycles a bounded number of K buffers after MMA consumption,
    independently of the model's low rank.
    The native Up K tile is an explicit static parameter (128 or 256).
    Native MMA tiles cover a power-of-two divisor of the branch count at a
    time; the epilogue accumulates branch groups in FP32 before storing.
    A retained activation is loaded once per token tile and stays valid through
    every Up output, independently of the cyclic weight stages. Its readiness
    barrier lets MMA overlap the preceding gate epilogue.
    Extra resident clusters can own only Up output tiles. Their START epochs
    protect invocation-generation sampling, and all grid clusters acknowledge
    CONSUMED before an activation slot is reused.
    Independent weights must be ready and immutable within the forward call.

    Physical clusters reduce a K group through DSM. With multiple reduction
    groups, FP32 partials cross clusters through release/acquire flags; each
    group then finalizes disjoint token rows before the ordinary READY phase.
    Groups are model-static and divide the native Down owner output tile.

    G/H/R are explicit model-static constructor parameters. Native output
    tiles require H divisible by 64; TMA requires R divisible by eight.

    Rows come from the dynamic input layout; only the CTA tactic is compiled.
    ``single_tile`` promises at most one native token tile, so the same pipeline
    can elide its outer scheduling loop. The grid follows the runtime shape,
    capped by the compiled resident capacity.
    Round counts are scheduling buckets: keeping their loop bound static avoids
    excessive register lifetimes in this warp-specialized pipeline.
    ``up_activation_ready_mode`` selects a global READY wait or per-K-tile
    projection readiness. Streamed plans retain the local Down EPI ACK and the
    terminal CONSUMED protocol; they only load activated rank columns.
    Paired streamed plans acknowledge local readers before all cohorts finish
    publication, then join every READY flag after Up's final read ACK. This
    permits useful overlap while covering inject-only and inactive Up actors.
    ``counter`` acquires the completed second counter phase for ACT loads;
    ``stream`` acquires the separate READY flags. ``auto`` uses counters only
    for paired single-iteration plans. READY publication and generation
    sampling remain unchanged in either plan.
    ``down_producer_mode`` selects separate producers, a joint producer, or
    automatic selection from the static pipeline plan. The joint plan requires
    S1, shared operand reuse, one projection step and one token iteration.
    ``full_tiles`` omits row predicates only for native-tile-aligned inputs.
    READY/CONSUMED halves use the workspace extent, never the active grid size.
    """

    def __init__(
        self,
        *,
        hc_count: int,
        hidden_size: int,
        lowrank: int,
        up_k: int,
        down_k_order: str,
        projection_tile: int,
        token_tile: int,
        down_token_parts: int,
        epilogue_threads: int,
        token_major_up: bool,
        up_token_tail_overlap: bool,
        up_output_parts: int,
        up_token_parts: int,
        up_stage_counts: tuple[int, int] | None,
        rotate_up_k: bool,
        projection_rows: int,
        partial_stride: int,
        split_k: int,
        reduction_groups: int,
        projection_tiles: int,
        batch_tiles: int,
        token_tile_batches: int,
        workers: int,
        up_clusters: int,
        stages: int,
        pin_up_activation: bool,
        reuse_down_storage: bool,
        final_tile: int,
        rounds: int,
        use_pdl: bool,
        scale: float,
        weights_independent: bool,
        evict_up_weights_first: bool,
        evict_down_weights_first: bool,
        single_tile: bool,
        full_tiles: bool,
        pipelines_per_cta: int,
        up_columns: int,
        pipeline_shared_budget: int,
        down_pair_reduction: str,
        down_producer_mode: str,
        up_activation_ready_mode: str,
        up_l2_prefetch_k: int,
        up_sigmoid_mode: str,
        down_x_cache_mode: str,
    ):
        if down_k_order not in ("interleaved", "contiguous", "paired"):
            raise ValueError("Down K order must be interleaved, contiguous or paired")
        self.down_k_order = down_k_order
        self.rotate_up_k = rotate_up_k
        if hc_count <= 1 or hidden_size <= 0 or lowrank <= 0:
            raise ValueError("HC must exceed one and H/R must be positive")
        if hidden_size % 64 or lowrank % 8:
            raise ValueError("Fused HC requires H divisible by 64 and R by 8")
        if up_activation_ready_mode not in ("auto", "global", "stream", "counter"):
            raise ValueError(
                "Up activation readiness mode must be auto, global, stream or counter"
            )
        self.up_activation_ready_mode = up_activation_ready_mode
        if up_l2_prefetch_k < 0 or up_l2_prefetch_k % 8:
            raise ValueError(
                "Up L2 prefetch rank extent must be nonnegative and divisible by eight"
            )
        if up_l2_prefetch_k and (
            not reuse_down_storage or split_k != 1 or not evict_up_weights_first
        ):
            raise ValueError(
                "Up L2 prefetch requires S1 shared reuse and evict-first Up weights"
            )
        self.up_l2_prefetch_k = min(up_l2_prefetch_k, max(0, lowrank - up_k))
        if up_sigmoid_mode not in ("exp_rcp", "tanh"):
            raise ValueError("Explicit Up sigmoid mode is required")
        if up_sigmoid_mode == "tanh" and not (
            hc_count == 4
            and hidden_size == 8192
            and lowrank == 1024
            and split_k == 1
            and reduction_groups == 16
            and (
                (
                    token_tile in (64, 192)
                    and projection_tile == 64
                    and pipelines_per_cta == 2
                )
                or (
                    token_tile == 96
                    and projection_tile == 128
                    and pipelines_per_cta == 1
                )
            )
            and workers
            == rounds
            == projection_tiles
            == batch_tiles
            == token_tile_batches
            == 1
            and up_clusters == 0
            and full_tiles
        ):
            raise ValueError(
                "Tanh Up sigmoid requires a full native64, native96 or native192 plan"
            )
        self.up_sigmoid_mode = up_sigmoid_mode
        if down_x_cache_mode not in ("native", "evict_last"):
            raise ValueError("Explicit Down X cache mode is required")
        if down_x_cache_mode == "evict_last" and not (
            hc_count == 4
            and hidden_size == 8192
            and lowrank == 1024
            and token_tile in (64, 96, 128, 192, 256)
            and split_k == 1
            and down_token_parts == 1
            and workers
            == rounds
            == projection_tiles
            == batch_tiles
            == token_tile_batches
            == 1
            and up_clusters == 0
        ):
            raise ValueError("X retention requires a native large-row plan")
        self.down_x_cache_mode = down_x_cache_mode

        if projection_rows not in (lowrank, lowrank + hc_count):
            raise ValueError("Projection rows must be R or R + HC")
        if split_k not in (1, 2, 4, 8, 16):
            raise ValueError("The fused HC tactic requires split-K in 1, 2, 4, 8, 16")
        if projection_tile not in (64, 128) or token_tile not in (
            8,
            16,
            24,
            32,
            64,
            96,
            128,
            192,
            256,
        ):
            raise ValueError("Unsupported native MMA tile")
        if up_k not in (64, 128, 256):
            raise ValueError("Up K tile must be 64, 128 or 256")
        if epilogue_threads not in (128, 256, 512):
            raise ValueError("Epilogues require complete 128-thread warp groups")
        if final_tile not in (16, 32):
            raise ValueError("Final hidden tile must be 16 or 32")
        if down_token_parts not in (1, 2) or token_tile // down_token_parts not in (
            8,
            16,
            24,
            32,
            64,
            96,
            128,
            192,
            256,
        ):
            raise ValueError("Unsupported Down token partition")
        if min(projection_tiles, batch_tiles, workers, stages, rounds) < 1:
            raise ValueError("Tile counts, workers and stages must be positive")
        if token_tile_batches not in (1, 2):
            raise ValueError("Shared-W token tile batches must be one pair at most")
        self.token_tile_batches = token_tile_batches
        if token_tile_batches > 1 and (
            split_k != 1
            or down_token_parts != 1
            or single_tile
            or up_clusters != 0
            or batch_tiles != 1
        ):
            raise ValueError(
                "Shared-W token pairs require S1, whole Down tiles, batch one and no Up-only CTAs"
            )
        if pipelines_per_cta not in (1, 2):
            raise ValueError("Local pipeline count must be1 or2")
        if pipelines_per_cta > 1 and (
            split_k != 1 or up_clusters != 0 or down_token_parts != 1 or single_tile
        ):
            raise ValueError(
                "Multiple local pipelines require S1, no Up-only CTAs, one Down part and early TMEM allocation"
            )
        if up_columns not in (64, 128, 256):
            raise ValueError("Explicit token-major Up column cap must be64/128/256")
        if (
            pipeline_shared_budget < 1
            or pipeline_shared_budget * pipelines_per_cta > 227 * 1024
        ):
            raise ValueError("Per-pipeline shared budget must fit the physical CTA")
        self.pipelines_per_cta = pipelines_per_cta
        self.up_columns = up_columns
        self.pipeline_shared_budget = pipeline_shared_budget
        if up_output_parts not in (1, 2):
            raise ValueError("Up token partitions must be one or two")
        self.up_output_parts = up_output_parts
        if up_token_parts not in (1, 2):
            raise ValueError("Up token partitions must be one or two")
        self.up_token_parts = up_token_parts
        self.up_token_tail_overlap = up_token_tail_overlap
        if up_token_tail_overlap and not (
            hc_count == 4
            and hidden_size == 8192
            and lowrank == 1024
            and token_tile == 256
            and projection_tile == 128
            and split_k == 1
            and reduction_groups == 16
            and pipelines_per_cta == 1
            and token_major_up
            and up_columns == 256
            and workers
            == rounds
            == projection_tiles
            == batch_tiles
            == token_tile_batches
            == 1
            and up_clusters == 0
            and full_tiles
            and down_token_parts == up_output_parts == 1
            and up_token_parts == 2
            and reuse_down_storage
            and not pin_up_activation
            and not rotate_up_k
        ):
            raise ValueError("Token-tail Up overlap requires one full native256 plan")
        self.wide_token_up = token_major_up and token_tile > 64
        if up_token_parts > 1 and (
            not self.wide_token_up or up_output_parts != 1 or rotate_up_k
        ):
            raise ValueError(
                "Token-split Up requires token-major tiles, one output tile and unrotated K"
            )
        self.wide_output_up = (
            not token_major_up and token_tile > 64 and up_output_parts > 1
        )
        if up_output_parts > 1 and not self.wide_output_up:
            raise ValueError(
                "Multiple Up token partitions require native wide token-major tiles"
            )
        self.token_major_up = token_major_up
        self.single_tile = single_tile
        self.full_tiles = full_tiles
        self.hc_count = hc_count
        self.hidden_size = hidden_size
        self.wide = hc_count * hidden_size
        self.p = projection_rows
        if partial_stride < projection_rows or partial_stride % 32:
            raise ValueError(
                "Partial stride must cover all projections and align to32floats"
            )
        self.partial_stride = partial_stride
        self.ps = lowrank
        self.up_k = up_k
        if reduction_groups < 1 or reduction_groups & (reduction_groups - 1):
            raise ValueError("Reduction groups must be a positive power of two")
        self.reduction_groups = reduction_groups
        self.split_k = split_k
        self.down_k = 64 if token_tile > 64 else 128
        self.down_m = projection_tile
        self.activation_projection_clusters = (
            lowrank + projection_tile - 1
        ) // projection_tile
        self.n = token_tile
        self.down_token_parts = down_token_parts
        self.down_n = token_tile // down_token_parts
        self.epilogue_threads = epilogue_threads
        self.epilogue_groups = epilogue_threads // 128
        if (
            self.down_n % self.epilogue_groups
            or self.down_n // self.epilogue_groups < 8
        ):
            raise ValueError(
                "Each epilogue warp group requires at least eight token columns"
            )
        self.epilogue_warps = epilogue_threads // 32
        self.weight_warp = self.epilogue_warps
        self.mma_warp = self.weight_warp + 1
        self.pipeline_threads = (self.mma_warp + 1) * 32
        if pipelines_per_cta > 1:
            # TMEM lane ownership repeats in physical groups of four warps.
            self.pipeline_threads = ((self.pipeline_threads + 127) // 128) * 128
        self.compact_warp_layout = token_tile > 64 and pipelines_per_cta > 1
        self.up_read_slices = max(1, token_tile // 64) if not token_major_up else 1
        self.threads = (
            (epilogue_threads + 64) * pipelines_per_cta
            if self.compact_warp_layout
            else self.pipeline_threads * pipelines_per_cta
        )
        if self.threads > 1024:
            raise ValueError("Local pipelines exceed the physical CTA thread limit")
        if self.down_n * (projection_tile // split_k) % reduction_groups:
            raise ValueError(
                "Reduction groups must divide the native Down owner output count"
            )
        self.slot_rows = max(16, token_tile)
        self.tmem_columns = max(32, 1 << (token_tile - 1).bit_length())
        self.projection_tiles = projection_tiles
        self.batch_tiles = batch_tiles
        self.workers = workers
        self.rounds = rounds
        self.down_stages = stages
        self.branch_tile = min(4, hc_count & -hc_count)
        self.branch_tiles = hc_count // self.branch_tile
        self.final_tile = max(final_tile, 64 // self.branch_tile)
        if token_major_up:
            self.final_tile = min(
                up_columns // self.branch_tile, hidden_size & -hidden_size
            )
        self.up_m = self.branch_tile * self.final_tile
        self.up_token_rows = token_tile // up_token_parts
        self.up_rows = (
            (128 if self.wide_token_up else 64) if token_major_up else self.up_m
        )
        self.up_cols = self.up_m if token_major_up else self.n
        self.up_token_m = self.up_rows if token_major_up else self.n
        self.gate_stride = self.up_m + int(token_major_up)
        self.gate_elements = self.gate_stride * (self.up_token_m // self.up_read_slices)
        # S1 global reductions write partials directly to global memory.
        self.down_mailbox_elements = (
            0 if split_k == 1 and reduction_groups > 1 else projection_tile * token_tile
        )
        self.down_partial_bytes = (
            0 if split_k == 1 else (projection_tile + 4) * self.down_n * 4
        )
        self.tmem_columns = max(
            self.tmem_columns,
            1 << (self.up_cols * up_output_parts * up_token_parts - 1).bit_length(),
        )
        if self.token_tile_batches > 1:
            self.tmem_columns = self.up_cols + self.down_n * self.token_tile_batches
        self.hidden_tiles = hidden_size // self.final_tile
        self.up_tiles = self.hidden_tiles * self.branch_tiles
        self.up_k_tiles = (lowrank + up_k - 1) // up_k
        self.pin_up_activation = pin_up_activation
        self.reuse_down_storage = reuse_down_storage
        if down_producer_mode not in ("auto", "separate", "joint"):
            raise ValueError("Down producer mode must be auto, separate or joint")
        self.down_producer_mode = down_producer_mode
        self.down_producer_eligible = (
            reuse_down_storage
            and split_k == 1
            and projection_tiles == 1
            and down_token_parts == 1
            and rounds * batch_tiles == 1
        )
        if down_producer_mode == "joint" and not self.down_producer_eligible:
            raise ValueError(
                "Joint Down producer requires S1, shared operand reuse, one "
                "projection step, one Down token part and one token iteration"
            )
        self.joint_down_producer = down_producer_mode == "joint" or (
            down_producer_mode == "auto" and self.down_producer_eligible
        )
        self.down_full_arrivals = 1 if self.joint_down_producer else 2
        self.projection_clusters = (
            projection_rows + projection_tile * projection_tiles - 1
        ) // (projection_tile * projection_tiles)
        self.group_clusters = self.projection_clusters * down_token_parts
        self.clusters = self.group_clusters * reduction_groups
        if up_clusters < 0 or (up_clusters and self.clusters == 1):
            raise ValueError("Up-only clusters require multiple projection clusters")
        self.up_clusters = up_clusters
        self.grid_clusters = self.clusters + up_clusters
        if self.grid_clusters % pipelines_per_cta:
            raise ValueError(
                "Logical cluster grid must divide the local pipeline count"
            )
        self.physical_clusters = self.grid_clusters // pipelines_per_cta
        if up_clusters and (self.grid_clusters - 1) * split_k >= self.hidden_tiles:
            raise ValueError("Each Up-only cluster must own a useful output tile")
        self.group_ctas = self.grid_clusters * split_k
        self.up_steps = (
            (self.hidden_tiles + self.group_ctas * up_output_parts - 1)
            // (self.group_ctas * up_output_parts)
        ) * self.branch_tiles
        self.stream_up_activation_eligible = (
            reduction_groups > 1
            and split_k == 1
            and reuse_down_storage
            and not pin_up_activation
            and down_token_parts == 1
            and projection_tiles == 1
            and rounds * batch_tiles == 1
            and up_clusters == 0
            and self.up_steps == 1
        )
        if (
            up_activation_ready_mode in ("stream", "counter")
            and not self.stream_up_activation_eligible
        ):
            raise ValueError(
                "Streamed Up activation requires S1 shared reuse, multiple reduction "
                "groups, one projection/token/Up iteration and no retained activation "
                "or Up-only CTAs"
            )
        self.stream_up_activation = up_activation_ready_mode in (
            "stream",
            "counter",
        ) or (up_activation_ready_mode == "auto" and self.stream_up_activation_eligible)
        # S1/global partial epilogues do not use the shared gate/mailbox.
        # With one Up step, final MMA retirement precedes every gate write
        # and no subsequent operand prefetch can overlap those writes.
        self.alias_gate = (
            (token_major_up or pipelines_per_cta > 1 or up_output_parts > 1)
            and reuse_down_storage
            and split_k == 1
            and reduction_groups > 1
            and down_token_parts == 1
            and self.up_steps == 1
            and up_clusters == 0
        )
        gate_bytes = (
            0
            if self.alias_gate
            else max(self.down_mailbox_elements, self.gate_elements) * 4
        )
        if self.branch_tiles > 1:
            gate_bytes += self.final_tile * token_tile * 4
        large_down_bytes = (
            (projection_tile + self.down_n * self.token_tile_batches) * 256 * 2 * stages
        )
        logical_splits = split_k * reduction_groups
        large_down_tiles = (self.wide + logical_splits * 256 - 1) // (
            logical_splits * 256
        )
        if (
            self.reuse_down_storage
            # The larger K tile must still give every logical split real work.
            # A singleton tiled K mode can fold its stride to zero, so indexing
            # it with another split rank would repeat tile zero.
            and (self.wide + 255) // 256 >= logical_splits
            and large_down_tiles >= stages
            and large_down_bytes + gate_bytes + 2048 <= pipeline_shared_budget
        ):
            self.down_k = 256
        self.k_tiles = (self.wide + logical_splits * self.down_k - 1) // (
            logical_splits * self.down_k
        )
        self.down_weight_post_bytes = (
            (self.down_m + (self.down_n if self.joint_down_producer else 0))
            * self.down_k
            * 2
        )
        if stages > self.k_tiles:
            raise ValueError("Down stages must not exceed the per-rank K tile count")
        # Reserve alignment/barrier overhead within Blackwell's 227 KiB
        # per-CTA budget. Large ranks stream through the stages that fit;
        # smaller ranks can keep all their K tiles resident.
        if self.reuse_down_storage:
            down_bytes = (projection_tile + self.down_n) * self.down_k * 2 * stages
            down_bytes = max(down_bytes, self.down_partial_bytes)
            fixed_bytes = (
                0
                if self.alias_gate
                else max(self.down_mailbox_elements, self.gate_elements) * 4
            )
            if self.branch_tiles > 1:
                fixed_bytes += self.final_tile * token_tile * 4
            up_stages = 0
            for count in range(1, self.up_k_tiles + 1):
                activation_stages = self.up_k_tiles if pin_up_activation else count
                up_bytes = (
                    (
                        self.up_m * count * up_output_parts
                        + self.up_token_m * activation_stages * self.up_token_parts
                    )
                    * up_k
                    * 2
                )
                total_bytes = fixed_bytes + max(down_bytes, up_bytes)
                if total_bytes + 2048 > pipeline_shared_budget:
                    break
                up_stages = count
        else:
            down_weights = projection_tile * self.down_k * 2 * stages
            fixed_bytes = self.down_n * self.down_k * 2 * stages
            mailbox_rows = (
                projection_tile if token_tile <= 16 else max(projection_tile, self.up_m)
            )
            fixed_bytes += mailbox_rows * self.down_n * 4
            fixed_bytes += (projection_tile + 4) * self.down_n * 4
            if token_tile <= 16:
                fixed_bytes += self.gate_elements * 4
            if self.branch_tiles > 1:
                fixed_bytes += self.final_tile * token_tile * 4
            up_stages = 0
            for count in range(1, self.up_k_tiles + 1):
                up_weights = self.up_m * up_k * 2 * count
                weight_bytes = (
                    down_weights + up_weights
                    if token_tile <= 16
                    else max(down_weights, up_weights)
                )
                activation_stages = self.up_k_tiles if pin_up_activation else count
                total_bytes = (
                    fixed_bytes
                    + weight_bytes
                    + self.up_token_m
                    * up_k
                    * 2
                    * activation_stages
                    * self.up_token_parts
                )
                if total_bytes + 2048 > pipeline_shared_budget:
                    break
                up_stages = count
        if up_stages == 0:
            raise ValueError("HC tactic leaves no shared memory for an Up K tile")
        self.independent_up_stages = up_stage_counts is not None
        self.up_stages = up_stages
        self.up_activation_stages = self.up_k_tiles if pin_up_activation else up_stages
        if up_stage_counts is not None:
            if (
                len(up_stage_counts) != 2
                or min(up_stage_counts) < 1
                or max(up_stage_counts) > self.up_k_tiles
                or pin_up_activation
                or split_k != 1
                or not reuse_down_storage
                or token_tile <= 64
                or self.up_steps != 1
                or up_clusters != 0
                or rounds * batch_tiles * token_tile_batches != 1
            ):
                raise ValueError(
                    "Independent Up stages require one streamed native-wide S1 tile"
                )
            self.up_stages, self.up_activation_stages = up_stage_counts
        up_stages = self.up_stages
        self.up_tail_tiles = self.up_stages if up_token_tail_overlap else 0
        if up_token_tail_overlap and (
            self.up_steps != 1
            or self.up_stages != 3
            or self.up_activation_stages != 3
            or self.up_k != 64
            or self.up_token_parts != 2
            or self.independent_up_stages
        ):
            raise ValueError(
                "Token-tail Up overlap requires the original matched three stages"
            )
        self.storage = self._storage_plan(up_stages)
        if self.storage.nbytes > pipeline_shared_budget:
            raise ValueError(
                "Actual pipeline layout exceeds its explicit shared budget"
            )
        self.pipeline_shared_stride = ((self.storage.nbytes + 1023) // 1024) * 1024
        self.cta_shared_bytes = (
            pipelines_per_cta - 1
        ) * self.pipeline_shared_stride + self.storage.nbytes
        self.total_tmem_columns = self.tmem_columns * pipelines_per_cta
        if self.total_tmem_columns > 512:
            raise ValueError("Local pipelines exceed512TMEM columns")
        if (
            self.token_tile_batches > 1
            and self.up_cols + self.down_n * self.token_tile_batches > self.tmem_columns
        ):
            raise ValueError("The Down accumulator pair exceeds the TMEM slot")
        self.owner_cols = projection_tile // split_k
        self.pdl = use_pdl
        self.scale = scale
        self.weights_independent = weights_independent
        self.evict_up_weights_first = evict_up_weights_first
        if evict_down_weights_first and not (
            hc_count == 4
            and hidden_size == 8192
            and lowrank == 1024
            and split_k == 1
            and reduction_groups == 16
            and workers == 1
            and projection_tiles == 1
            and rounds == 1
            and batch_tiles == 1
            and token_tile_batches == 1
            and up_clusters == 0
        ):
            raise ValueError("Down evict-first requires one native single-worker tile")
        self.evict_down_weights_first = evict_down_weights_first
        self.group = tcgen05.CtaGroup.ONE
        if down_pair_reduction not in ("auto", "on", "off"):
            raise ValueError("Explicit Down pair reduction must be auto, on or off")
        pair_supported = (
            pipelines_per_cta == 2
            and split_k == 1
            and reduction_groups > 1
            and projection_tile == 64
            and projection_tiles == 1
            and down_token_parts == 1
            and up_clusters == 0
            and self.up_steps == 1
            and self.branch_tiles == 1
            and rounds == 1
            and batch_tiles == 1
            and self.down_n % (2 * self.epilogue_groups) == 0
            and (
                self.down_n // (2 * self.epilogue_groups) >= 8
                or self.down_n == 8
                and self.epilogue_groups == 1
            )
        )
        if down_pair_reduction == "on" and not pair_supported:
            raise ValueError(
                "Down pair reduction requires complete paired single-iteration native tiles"
            )
        self.down_pair_reduction = down_pair_reduction
        self.local_pair_reduce = pair_supported and down_pair_reduction != "off"
        self.narrow_down_pair = self.local_pair_reduce and self.down_n in (8, 24)
        self.down_merge_groups = (
            reduction_groups // 2 if self.local_pair_reduce else reduction_groups
        )
        self.down_merge_threads = epilogue_threads * (
            2 if self.local_pair_reduce else 1
        )
        self.down_counter_step = 2 if self.local_pair_reduce else 1
        self.counter_activation_ready_eligible = (
            self.local_pair_reduce and self.stream_up_activation_eligible
        )
        if (
            up_activation_ready_mode == "counter"
            and not self.counter_activation_ready_eligible
        ):
            raise ValueError(
                "Counter-based Up activation requires a paired single-iteration plan"
            )
        self.counter_activation_ready = (
            up_activation_ready_mode in ("auto", "counter")
            and self.counter_activation_ready_eligible
        )
        self.vector_down_merge = (
            hc_count == 4
            and hidden_size == 8192
            and lowrank == 1024
            and self.split_k == 1
            and self.reduction_groups == 16
            and self.owner_cols % 4 == 0
            and self.partial_stride % 4 == 0
            and self.p % 4 == 0
            and workers == 1
            and rounds * batch_tiles * token_tile_batches == 1
            and (
                self.down_n == 256
                and self.down_m == 128
                and self.down_merge_groups == 16
                and self.down_merge_threads == 512
                and self.single_tile
                and pipelines_per_cta == 1
                or self.down_n in (128, 192)
                and self.down_m == 64
                and self.local_pair_reduce
                and self.down_merge_groups == 8
                and pipelines_per_cta == 2
            )
        )
        # Full paired tiles use packets; tails retain scalar ACT writeback.
        self.packet_activation_store = self.vector_down_merge and full_tiles

    def _storage_plan(self, up_stages: int) -> _SharedStorage:
        offset = 0

        def allocate(size: int, alignment: int) -> int:
            nonlocal offset
            start = (offset + alignment - 1) // alignment * alignment
            offset = start + size
            return start

        gate = allocate(
            (
                0
                if self.alias_gate
                else (
                    max(self.down_mailbox_elements, self.gate_elements) * 4
                    if self.reuse_down_storage
                    else self.gate_elements * 4
                )
            ),
            128,
        )
        down_weights = self.down_m * self.down_k * self.down_stages * 2
        down_x = self.down_n * self.down_k * self.down_stages * 2
        self.down_x_bytes = down_x
        up_weights = self.up_m * self.up_k * up_stages * self.up_output_parts * 2
        activation_stages = self.up_activation_stages
        up_activation = (
            self.up_token_m * self.up_k * activation_stages * self.up_token_parts * 2
        )
        partial = self.down_partial_bytes
        if self.reuse_down_storage:
            arena = allocate(
                max(
                    down_weights + down_x * self.token_tile_batches,
                    partial,
                    up_weights + up_activation,
                ),
                1024,
            )
            sd, sx = arena, arena + down_weights
            su, sa = arena, arena + up_weights
            mailbox, partial = gate, arena
        else:
            sd = allocate(down_weights, 1024)
            sx = allocate(down_x * self.token_tile_batches, 1024)
            su = allocate(up_weights, 1024)
            sa = allocate(up_activation, 1024)
            mailbox = allocate(self.down_m * self.down_n * 4, 128)
            partial = allocate(partial, 128)
        mix = gate
        if self.branch_tiles > 1:
            mix = allocate(self.final_tile * self.n * 4, 128)
        barrier_count = (
            2 * self.down_stages
            + 7
            + up_stages
            + int(self.pin_up_activation)
            + up_stages * int(self.up_k_tiles > up_stages)
            + int(self.up_steps > 1)
            + int(self.rounds * self.batch_tiles * self.token_tile_batches > 1)
            + 3 * int(self.reduction_groups > 1)
            + 2 * self.up_activation_stages * int(self.independent_up_stages)
            + self.up_token_parts * int(self.up_token_tail_overlap)
        )
        barriers = allocate(barrier_count * 8, 8)
        tmem = allocate(4 + 8 * (self.token_tile_batches - 1), 4)
        return _SharedStorage(
            gate, sd, sx, su, sa, mailbox, partial, mix, barriers, tmem, offset
        )

    @cute.experimental.jit
    def __call__(
        self, x, w, u, activation, partials, epochs, out, inject, stream: CUstream
    ):
        # Preserve the physical row-major weights while making the K64
        # slabs used by native shared layouts explicit to TMA.
        x_view = cute.make_tensor(
            x.iterator,
            cute.make_layout(
                (x.shape[0], (64, self.wide // 64), 1),
                stride=(self.wide, (1, 64), 0),
            ),
        )
        w_view = cute.make_tensor(
            w.iterator,
            cute.make_layout(
                (self.p, (64, self.wide // 64), 1),
                stride=(self.wide, (1, 64), 0),
            ),
        )
        if cutlass.const_expr(self.ps % 64 == 0):
            up = cute.make_tensor(
                u.iterator,
                cute.make_layout(
                    (
                        (self.final_tile, self.hc_count, self.hidden_tiles),
                        (64, self.ps // 64),
                        1,
                    ),
                    stride=(
                        (
                            self.ps,
                            self.hidden_size * self.ps,
                            self.final_tile * self.ps,
                        ),
                        (1, 64),
                        0,
                    ),
                ),
            )
        else:
            up = cute.make_tensor(
                u.iterator,
                cute.make_layout(
                    ((self.final_tile, self.hc_count, self.hidden_tiles), self.ps, 1),
                    stride=(
                        (
                            self.ps,
                            self.hidden_size * self.ps,
                            self.final_tile * self.ps,
                        ),
                        1,
                        0,
                    ),
                ),
            )
        partials_view = partials
        if cutlass.const_expr(self.reduction_groups > 1):
            partials_view = cute.make_tensor(
                partials.iterator,
                cute.make_layout(
                    (partials.shape[0], self.partial_stride * self.reduction_groups),
                    stride=(self.partial_stride * self.reduction_groups, 1),
                ),
            )
        # A native TMA atom exposes L2 priority without repacking weight rows.
        up_atom, up_tma = None, None
        if cutlass.const_expr(self.evict_up_weights_first):
            up_mma = sm100_utils.make_trivial_tiled_mma(
                x.element_type,
                x.element_type,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
                cutlass.Float32,
                self.group,
                (self.up_rows, self.up_cols),
            )
            up_tiler = (self.up_rows, self.up_cols, self.up_k)
            if cutlass.const_expr(self.token_major_up):
                up_layout = sm100_utils.make_smem_layout_b(
                    up_mma, up_tiler, x.element_type, self.up_stages
                )
                make_atom = cute.nvgpu.make_tiled_tma_atom_B
            else:
                up_layout = sm100_utils.make_smem_layout_a(
                    up_mma, up_tiler, x.element_type, self.up_stages
                )
                make_atom = cute.nvgpu.make_tiled_tma_atom_A
            up_atom, up_tma = make_atom(
                cpasync.CopyBulkTensorTileG2SOp(self.group),
                up,
                cute.slice_(up_layout, (None, None, None, 0)),
                up_tiler,
                up_mma,
            )
        down_atom, down_tma = None, None
        if cutlass.const_expr(self.evict_down_weights_first):
            down_mma = sm100_utils.make_trivial_tiled_mma(
                x.element_type,
                x.element_type,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
                cutlass.Float32,
                self.group,
                (self.down_m, self.down_n),
            )
            down_tiler = (self.down_m, self.down_n, self.down_k)
            down_layout = sm100_utils.make_smem_layout_a(
                down_mma, down_tiler, x.element_type, self.down_stages
            )
            down_atom, down_tma = cute.nvgpu.make_tiled_tma_atom_A(
                cpasync.CopyBulkTensorTileG2SOp(self.group),
                w_view,
                cute.slice_(down_layout, (None, None, None, 0)),
                down_tiler,
                down_mma,
            )
        x_atom, x_tma = None, None
        if cutlass.const_expr(self.down_x_cache_mode == "evict_last"):
            x_mma = sm100_utils.make_trivial_tiled_mma(
                x.element_type,
                x.element_type,
                tcgen05.OperandMajorMode.K,
                tcgen05.OperandMajorMode.K,
                cutlass.Float32,
                self.group,
                (self.down_m, self.down_n),
            )
            x_tiler = (self.down_m, self.down_n, self.down_k)
            x_layout = sm100_utils.make_smem_layout_b(
                x_mma, x_tiler, x.element_type, self.down_stages
            )
            x_atom, x_tma = cute.nvgpu.make_tiled_tma_atom_B(
                cpasync.CopyBulkTensorTileG2SOp(self.group),
                x_view,
                cute.slice_(x_layout, (None, None, None, 0)),
                x_tiler,
                x_mma,
            )
        workers = cute.ceil_div(x.shape[0], self.n * self.batch_tiles)
        if cutlass.const_expr(self.clusters > 1):
            workers = min(workers, self.workers)
        self.kernel(
            x_view,
            w_view,
            up,
            activation,
            partials_view,
            epochs,
            out,
            inject,
            up_atom,
            up_tma,
            down_atom,
            down_tma,
            x_atom,
            x_tma,
        ).launch(
            grid=(self.physical_clusters, self.split_k, workers),
            cluster=(1, self.split_k, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            use_pdl=self.pdl,
            cooperative=self.clusters > 1,
            smem=self.cta_shared_bytes,
        )

    @cute.experimental.jit
    def _accumulator_tile(
        self,
        smem_ptr,
        acc_layout,
        tile_m,
        tile_n: cutlass.Constexpr,
        activation,
        epi_tid,
        reader_slices: cutlass.Constexpr,
        reader_slice: cutlass.Constexpr,
        accumulator_stage: cutlass.Constexpr,
    ):
        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, smem_ptr), acc_layout
        )[((None, None), 0, 0, accumulator_stage)]
        reader_n = tile_n // (self.epilogue_groups * reader_slices)
        group = epi_tid // 128 + reader_slice * self.epilogue_groups
        local_tid = epi_tid % 128
        tile = (tile_m, reader_n)
        coordinates = cute.make_identity_tensor((tile_m, tile_n))
        if cutlass.const_expr(self.epilogue_groups * reader_slices > 1):
            # Each aligned warp group reads different N columns. Its warps
            # retain their required TMEM lane windows within the group.
            acc_view = cute.local_tile(acc_view, tile, (0, group))
            coordinates = cute.local_tile(coordinates, tile, (0, group))
        t2r = tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (tile_m, reader_n, 128),
                cute_utils.LayoutEnum.COL_MAJOR,
                cutlass.Float32,
                cutlass.Float32,
                tile,
                False,
            ),
            acc_view,
        )
        dummy = cute.make_tensor(
            activation.iterator, cute.make_layout(tile, stride=(1, self.ps))
        )
        rlayout = cute_ext.make_t2r_rmem_layout(
            t2r, cute.flat_divide(dummy, tile), local_tid
        )
        values = cute_ext.allocate(
            cutlass.Float32, cute.AddressSpace.rmem, rlayout, alignment=32
        )
        thr = t2r.get_slice(local_tid)
        coords = thr.partition_D(coordinates)
        return acc_view, thr, values, coords

    @cute.experimental.jit
    def _paired_down_accumulator_tile(
        self, smem_ptr, acc_layout, activation, epi_tid, pipeline
    ):
        """Read disjoint N slices from both private Down accumulators."""
        peer_ptr = smem_ptr + (1 - 2 * pipeline) * (self.pipeline_shared_stride // 4)
        own = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, smem_ptr), acc_layout
        )[((None, None), 0, 0, 0)]
        peer = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, peer_ptr), acc_layout
        )[((None, None), 0, 0, 0)]
        group = pipeline * self.epilogue_groups + epi_tid // 128
        local_tid = epi_tid % 128
        reader_n = self.down_n // (2 * self.epilogue_groups)
        if cutlass.const_expr(self.narrow_down_pair):
            # N4/N12 do not fit the M64 TMEM load atom. Each reader uses
            # the full N8/N24 tile; pipeline0 alone publishes the paired sum.
            group = epi_tid // 128
            reader_n = self.down_n // self.epilogue_groups
        tile = (self.down_m, reader_n)
        source = cute.local_tile(own, tile, (0, group))
        peer_source = cute.local_tile(peer, tile, (0, group))
        coordinates = cute.local_tile(
            cute.make_identity_tensor((self.down_m, self.down_n)), tile, (0, group)
        )
        copy = tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.down_m, self.down_n, self.down_k),
                cute_utils.LayoutEnum.COL_MAJOR,
                cutlass.Float32,
                cutlass.Float32,
                tile,
                False,
            ),
            source,
        )
        dummy = cute.make_tensor(
            activation.iterator, cute.make_layout(tile, stride=(1, self.ps))
        )
        layout = cute_ext.make_t2r_rmem_layout(
            copy, cute.flat_divide(dummy, tile), local_tid
        )
        values = cute_ext.allocate(
            cutlass.Float32, cute.AddressSpace.rmem, layout, alignment=32
        )
        peer_values = cute_ext.allocate(
            cutlass.Float32, cute.AddressSpace.rmem, layout, alignment=32
        )
        reader = copy.get_slice(local_tid)
        coords = reader.partition_D(coordinates)
        cute_ext.partition_and_copy(reader, source, values)
        cute_ext.partition_and_copy(reader, peer_source, peer_values)
        cute.arch.fence_view_async_tmem_load()
        for i in cutlass.range_constexpr(cute.size(values)):
            values[i] = values[i] + peer_values[i]
        return values, coords

    @cute.experimental.jit
    def _token_major_accumulator_tile(
        self, smem_ptr, acc_layout, activation, epi_tid, output_part: cutlass.Constexpr
    ):
        acc = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, smem_ptr), acc_layout
        )[((None, None), 0, 0, output_part)]
        group = cute.arch.make_warp_uniform(epi_tid // 128)
        local_tid = epi_tid % 128
        reader_n = self.up_cols // self.epilogue_groups
        tile = (self.up_rows, reader_n)
        source = cute.local_tile(acc, tile, (0, group))
        coords_tile = cute.local_tile(
            cute.make_identity_tensor((self.up_rows, self.up_cols)), tile, (0, group)
        )
        copy = tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.up_rows, self.up_cols, self.up_k),
                cute_utils.LayoutEnum.ROW_MAJOR,
                cutlass.Float32,
                cutlass.Float32,
                tile,
                False,
            ),
            source,
        )
        reader = copy.get_slice(local_tid)
        dummy = cute.make_tensor(
            activation.iterator, cute.make_layout(tile, stride=(reader_n, 1))
        )
        rlayout = cute_ext.make_t2r_rmem_layout(
            copy, cute.flat_divide(dummy, tile), local_tid
        )
        values = cute_ext.allocate(
            cutlass.Float32, cute.AddressSpace.rmem, rlayout, alignment=32
        )
        return source, reader, values, reader.partition_D(coords_tile)

    @cute.experimental.jit
    def _prefetch_x(
        self, x_regs, coords, x_values, pid, token, full_tile: cutlass.Constexpr
    ):
        # Compare native row offsets with the tile's remaining rows. Widen the
        # tile address before adding them to preserve linear 64-bit addressing.
        for i in cutlass.range_constexpr(cute.size(x_regs)):
            if cutlass.const_expr(self.token_major_up):
                row, col = coords[i]
            else:
                col, row = coords[i]
            global_row = row + token * self.n
            global_col = (
                (col // self.final_tile + (pid // self.hidden_tiles) * self.branch_tile)
                * self.hidden_size
                + (pid % self.hidden_tiles) * self.final_tile
                + col % self.final_tile
            )
            if cutlass.const_expr(full_tile):
                value = x_values[global_row, global_col]
            else:
                value = _load_tail(
                    x_values.iterator
                    + cutlass.Int64(token) * (self.n * self.wide)
                    + row * self.wide
                    + global_col,
                    row < cutlass.Int32(x_values.shape[0]) - token * self.n,
                    x_values.element_type,
                )
            x_regs[i] = value.to(x_regs.element_type)

    @cute.experimental.jit
    def _read_up_with_prefetched_x(
        self,
        smem_ptr,
        acc_layout,
        activation,
        epi_tid,
        ready_bar,
        x_values,
        pid,
        token,
        phase,
        reader_slice: cutlass.Constexpr,
        output_part: cutlass.Constexpr,
    ):
        if cutlass.const_expr(self.token_major_up):
            acc_view, thr, values, coords = self._token_major_accumulator_tile(
                smem_ptr, acc_layout, activation, epi_tid, output_part
            )
        else:
            acc_view, thr, values, coords = self._accumulator_tile(
                smem_ptr,
                acc_layout,
                self.up_m,
                self.n,
                activation,
                epi_tid,
                self.up_read_slices,
                reader_slice,
                output_part,
            )
        x_regs = cute_ext.allocate(
            (
                x_values.element_type
                if cutlass.const_expr(self.n > 64)
                else cutlass.Float32
            ),
            cute.AddressSpace.rmem,
            values.layout,
            alignment=32,
        )
        # Compute handed PDL readiness to the group before Down. Prefetch the
        # final gate operand while Up MMA proceeds on its separate SMEM inputs.
        if cutlass.const_expr(self.token_major_up):
            active_reader = ((epi_tid % 128) // 32) * 16 < min(
                self.up_token_rows,
                cutlass.Int32(x_values.shape[0])
                - token * self.n
                - output_part * self.up_token_rows,
            )
            if active_reader:
                self._prefetch_x(
                    x_regs,
                    coords,
                    x_values,
                    pid,
                    token,
                    self.full_tiles,
                )
            cute.arch.mbarrier_wait(ready_bar, phase)
            if active_reader:
                cute_ext.partition_and_copy(thr, acc_view, values)
                cute.arch.fence_view_async_tmem_load()
        else:
            self._prefetch_x(
                x_regs,
                coords,
                x_values,
                pid,
                token,
                self.full_tiles,
            )
            cute.arch.mbarrier_wait(ready_bar, phase)
            cute_ext.partition_and_copy(thr, acc_view, values)
            cute.arch.fence_view_async_tmem_load()
        return values, coords, x_regs

    @cute.experimental.jit
    def _down_input_tile(self, tile, logical_rank):
        index = tile * (self.split_k * self.reduction_groups) + logical_rank
        if cutlass.const_expr(self.down_k_order == "contiguous"):
            index = logical_rank * self.k_tiles + tile
        elif cutlass.const_expr(self.down_k_order == "paired"):
            index = (
                (logical_rank // 2) * (2 * self.k_tiles) + 2 * tile + logical_rank % 2
            )
        return index

    @cute.experimental.jit
    def _load_packed_branch_x(
        self,
        xs,
        coords,
        x_values,
        token,
        token_part: cutlass.Constexpr,
        hidden_id,
        branch: cutlass.Constexpr,
    ):
        first_row, first_col = coords[0]
        source_row = token * self.n + token_part * self.up_token_rows + first_row
        source_hidden = (
            hidden_id * self.final_tile + (first_col % self.final_tile // 16) * 16
        )
        if source_row < x_values.shape[0]:
            source_ptr = cute.make_ptr(
                x_values.element_type,
                (
                    x_values.iterator
                    + source_row * self.wide
                    + branch * self.hidden_size
                    + source_hidden
                ).toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            source = cute.make_tensor(source_ptr, cute.make_layout(16))
            copy_atom = cute.make_copy_atom(
                CopyUniversalOp(), x_values.element_type, num_bits_per_copy=128
            )
            cute.copy(copy_atom, source, xs)

    @cute.jit
    def _up_gate_sigmoid(self, value):
        if cutlass.const_expr(self.up_sigmoid_mode == "tanh"):
            return _up_sigmoid_tanh(value)
        return cute.arch.rcp_approx(1.0 + cute.exp(-value, fastmath=True))

    @cute.experimental.jit
    def _token_branch_epilogue(
        self,
        tmem_base,
        acc_layout,
        activation,
        x_values,
        out,
        token,
        hidden_id,
        epi_tid,
        up_done,
        up_phase,
        epi_done,
    ):
        for token_part in cutlass.range_constexpr(self.up_token_parts):
            template_acc, template_thr, template, output_coords = (
                self._accumulator_tile(
                    tmem_base,
                    acc_layout,
                    self.up_rows,
                    self.up_cols,
                    activation,
                    epi_tid,
                    self.branch_tile,
                    0,
                    token_part,
                )
            )
            sums = cute.make_rmem_tensor((cute.size(template),), cutlass.Float32)
            for i in cutlass.range_constexpr(cute.size(template)):
                sums[i] = cutlass.Float32(0)
            xs_ring = cute.make_rmem_tensor((16, 2), x_values.element_type)
            for slot in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(16):
                    xs_ring[j, slot] = x_values.element_type(0)
            # Coordinates have the same token/hidden offsets for every branch.
            self._load_packed_branch_x(
                xs_ring[None, 0],
                output_coords,
                x_values,
                token,
                token_part,
                hidden_id,
                0,
            )
            for branch in cutlass.range_constexpr(self.branch_tile):
                acc, thr, gates, coords = self._accumulator_tile(
                    tmem_base,
                    acc_layout,
                    self.up_rows,
                    self.up_cols,
                    activation,
                    epi_tid,
                    self.branch_tile,
                    branch,
                    token_part,
                )
                if cutlass.const_expr(branch + 1 < self.branch_tile):
                    self._load_packed_branch_x(
                        xs_ring[None, (branch + 1) % 2],
                        coords,
                        x_values,
                        token,
                        token_part,
                        hidden_id,
                        branch + 1,
                    )
                # All readers acquire MMA completion before the TMEM access.
                cute.arch.mbarrier_wait(
                    (
                        up_done + token_part
                        if cutlass.const_expr(self.up_token_tail_overlap)
                        else up_done
                    ),
                    up_phase,
                )
                tcgen05_fence("after_thread_sync")
                cute_ext.partition_and_copy(thr, acc, gates)
                cute.arch.fence_view_async_tmem_load()
                for i in cutlass.range_constexpr(cute.size(gates)):
                    row, col = coords[i]
                    global_row = token * self.n + token_part * self.up_token_rows + row
                    hidden = hidden_id * self.final_tile + col % self.final_tile
                    xv = xs_ring[i, branch % 2].to(cutlass.Float32)
                    if cutlass.const_expr(self.up_sigmoid_mode == "tanh"):
                        sums[i] = sums[i] + xv * self._up_gate_sigmoid(gates[i])
                    else:
                        sums[i] = sums[i] + xv * cute.arch.rcp_approx(
                            1.0 + cute.exp(-gates[i], fastmath=True)
                        )
            if cutlass.const_expr(token_part + 1 == self.up_token_parts):
                tcgen05_fence("before_thread_sync")
                cute.arch.mbarrier_arrive(epi_done)
            first_row, first_col = output_coords[0]
            global_row = token * self.n + token_part * self.up_token_rows + first_row
            packed = cute.make_rmem_tensor((cute.size(sums),), out.element_type)
            for i in cutlass.range_constexpr(cute.size(sums)):
                packed[i] = (sums[i] * (1.0 / self.hc_count)).to(out.element_type)
            if global_row < out.shape[0]:
                pointer = cute.make_ptr(
                    out.element_type,
                    (
                        out.iterator
                        + global_row * self.hidden_size
                        + hidden_id * self.final_tile
                        + first_col
                    ).toint(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                target = cute.make_tensor(pointer, cute.make_layout(cute.size(sums)))
                copy_atom = cute.make_copy_atom(
                    CopyUniversalOp(), out.element_type, num_bits_per_copy=128
                )
                cute.copy(copy_atom, packed, target)

    @cute.experimental.jit
    def _load_down_weight(
        self,
        gd,
        sd,
        stage: cutlass.Constexpr,
        input_tile,
        full,
        map_d,
        down_atom,
        down_tma,
        projection,
    ):
        if cutlass.const_expr(self.evict_down_weights_first):
            g = cute.local_tile(
                down_tma, (self.down_m, self.down_k), (projection, None, 0)
            )
            target, source = cpasync.tma_partition(
                down_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(sd, 0, 3),
                cute.group_modes(g, 0, 2),
            )
            cute.copy(
                down_atom,
                source[None, input_tile],
                target[None, stage],
                tma_bar_ptr=full + stage,
                cache_policy=_up_weight_cache_policy(),
            )
        else:
            cute_ext.tma_load(
                gd[None, None, input_tile],
                sd[None, None, None, stage],
                (full + stage).value,
                cta_v_map=map_d,
                tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                update_expect_tx=False,
            )

    @cute.experimental.jit
    def _load_down_x(
        self,
        gx,
        sx,
        stage,
        input_tile,
        full,
        map_x,
        x_atom,
        x_tma,
        down_token,
    ):
        if cutlass.const_expr(self.down_x_cache_mode == "evict_last"):
            source = cute.local_tile(
                x_tma, (self.down_n, self.down_k), (down_token, None, 0)
            )
            target, source = cpasync.tma_partition(
                x_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(sx, 0, 3),
                cute.group_modes(source, 0, 2),
            )
            cute.copy(
                x_atom,
                source[None, input_tile],
                target[None, stage],
                tma_bar_ptr=full + stage,
                cache_policy=_down_x_cache_policy(),
            )
        else:
            cute_ext.tma_load(
                gx[None, None, input_tile],
                sx[None, None, None, stage],
                (full + stage).value,
                cta_v_map=map_x,
                tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                update_expect_tx=False,
            )

    @cute.experimental.jit
    def _load_up(
        self,
        up,
        su,
        bar,
        map_u,
        pid,
        iteration,
        empty,
        start: cutlass.Constexpr,
        stop: cutlass.Constexpr,
        up_atom,
        up_tma,
        k_offset,
    ):
        for tile in cutlass.range_constexpr(start, stop):
            source_tile = cutlass.Int32(tile)
            if cutlass.const_expr(self.rotate_up_k):
                source_tile = (tile + k_offset) % self.up_k_tiles
            stage = tile % self.up_stages
            if cutlass.const_expr(tile >= self.up_stages):
                uses = (self.up_k_tiles + self.up_stages - 1 - stage) // self.up_stages
                previous = iteration * uses + tile // self.up_stages - 1
                cute.arch.mbarrier_wait(empty + stage, previous % 2)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar + stage, self.up_output_parts * self.up_m * self.up_k * 2
                )
            for output_part in cutlass.range_constexpr(self.up_output_parts):
                part_pid = pid + output_part
                tma_pid = (
                    part_pid % self.hidden_tiles
                ) * self.branch_tiles + part_pid // self.hidden_tiles
                part_su = cute.make_tensor(
                    su.iterator + output_part * self.up_m * self.up_k * self.up_stages,
                    su.layout,
                )
                if cutlass.const_expr(self.evict_up_weights_first):
                    g_tma = cute.local_tile(
                        up_tma, (self.up_m, self.up_k), (tma_pid, None, 0)
                    )
                    su_copy, gu_copy = cpasync.tma_partition(
                        up_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(part_su, 0, 3),
                        cute.group_modes(g_tma, 0, 2),
                    )
                    cute.copy(
                        up_atom,
                        gu_copy[None, source_tile],
                        su_copy[None, stage],
                        tma_bar_ptr=bar + stage,
                        cache_policy=_up_weight_cache_policy(),
                    )
                else:
                    g = cute.local_tile(up, (self.up_m, self.up_k), (tma_pid, None, 0))
                    cute_ext.tma_load(
                        g[None, None, source_tile],
                        part_su[None, None, None, stage],
                        (bar + stage).value,
                        cta_v_map=map_u,
                        tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                        update_expect_tx=False,
                    )

    @cute.jit
    def _prefetch_future_up_weights(self, up, pid):
        # Canonical rows are (branch, hidden, rank), with 16-byte row alignment.
        # The first K tile already travels to SMEM while Down publishes ACT.
        for row_group in cutlass.range_constexpr(
            (self.up_m * self.up_output_parts + 31) // 32
        ):
            row = row_group * 32 + cute.arch.lane_idx()
            if row < self.up_m:
                branch = (
                    pid // self.hidden_tiles * self.branch_tile + row // self.final_tile
                )
                hidden = (
                    pid % self.hidden_tiles * self.final_tile + row % self.final_tile
                )
                pointer = (
                    up.iterator
                    + (branch * self.hidden_size + hidden) * self.ps
                    + self.up_k
                )
                _prefetch_up_weight_l2(pointer, self.up_l2_prefetch_k * 2)

    @cute.experimental.jit
    def _load_activation(
        self,
        activation,
        sa,
        full,
        empty,
        up_mma,
        worker,
        iteration,
        epochs,
        generation,
        k_offset,
    ):
        if cutlass.const_expr(self.wide_token_up):
            self._load_wide_up_activation(
                activation,
                sa,
                full,
                empty,
                up_mma,
                worker,
                iteration,
                epochs,
                generation,
                k_offset,
            )
        else:
            self._load_standard_activation(
                activation,
                sa,
                full,
                empty,
                up_mma,
                worker,
                iteration,
                epochs,
                generation,
                k_offset,
            )

    @cute.experimental.jit
    def _load_wide_up_activation(
        self,
        activation,
        sa,
        full,
        empty,
        up_mma,
        worker,
        iteration,
        epochs,
        generation,
        k_offset,
    ):
        # One ACT arrival covers both token operands; the other arrival is W.
        active = cute.make_tensor(
            activation.iterator,
            cute.make_layout(
                (
                    self.up_token_rows,
                    (64, self.ps // 64),
                    self.workers * self.up_token_parts,
                ),
                stride=(self.ps, (1, 64), self.up_token_rows * self.ps),
            ),
        )
        up_tiler = (self.up_rows, self.up_cols, self.up_k)
        map_a = cute_ext.get_cta_v_map_ab(active, up_tiler, up_mma, "A")
        ga = cute.local_tile(active, (self.up_token_m, self.up_k), (0, None, None))
        ready_base = worker * self.clusters
        all_ready = cutlass.Boolean(False)
        for tile in cutlass.range_constexpr(self.up_k_tiles):
            source_tile = cutlass.Int32(tile)
            if cutlass.const_expr(self.rotate_up_k):
                source_tile = (tile + k_offset) % self.up_k_tiles
            stage = tile % self.up_stages
            if cutlass.const_expr(tile >= self.up_stages):
                uses = (self.up_k_tiles + self.up_stages - 1 - stage) // self.up_stages
                previous = iteration * uses + tile // self.up_stages - 1
                cute.arch.mbarrier_wait(empty + stage, previous % 2)
            if cutlass.const_expr(self.stream_up_activation):
                all_ready = self._wait_activation_epochs(
                    epochs, ready_base, source_tile, generation, all_ready
                )
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    full + stage, self.up_token_parts * self.up_token_m * self.up_k * 2
                )
            for token_part in cutlass.range_constexpr(self.up_token_parts):
                part_sa = cute.make_tensor(
                    sa.iterator
                    + token_part
                    * self.up_token_m
                    * self.up_k
                    * self.up_activation_stages,
                    sa.layout,
                )
                cute_ext.tma_load(
                    ga[
                        None,
                        None,
                        source_tile,
                        worker * self.up_token_parts + token_part,
                    ],
                    part_sa[None, None, None, stage],
                    (full + stage).value,
                    cta_v_map=map_a,
                    tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                    update_expect_tx=False,
                )

    @cute.experimental.jit
    def _load_standard_activation(
        self,
        activation,
        sa,
        full,
        empty,
        up_mma,
        worker,
        iteration,
        epochs,
        generation,
        k_offset,
    ):
        up_tiler = (self.up_rows, self.up_cols, self.up_k)
        workspace_rows = self.workers * self.slot_rows
        if cutlass.const_expr(self.clusters == 1):
            workspace_rows = activation.shape[0]
        if cutlass.const_expr(self.token_major_up):
            # Select workers through TMA coordinates, preserving a static base
            # pointer. Logical N bounds the zero-padded M64 load.
            if cutlass.const_expr(self.ps % 64 == 0):
                active = cute.make_tensor(
                    activation.iterator,
                    cute.make_layout(
                        (self.n, (64, self.ps // 64), workspace_rows // self.slot_rows),
                        stride=(self.ps, (1, 64), self.slot_rows * self.ps),
                    ),
                )
            else:
                active = cute.make_tensor(
                    activation.iterator,
                    cute.make_layout(
                        (self.n, self.ps, workspace_rows // self.slot_rows),
                        stride=(self.ps, 1, self.slot_rows * self.ps),
                    ),
                )
            map_a = cute_ext.get_cta_v_map_ab(active, up_tiler, up_mma, "A")
            ga = cute.local_tile(
                active, (self.up_token_m, self.up_k), (0, None, worker)
            )
        else:
            if cutlass.const_expr(self.ps % 64 == 0):
                active = cute.make_tensor(
                    activation.iterator,
                    cute.make_layout(
                        (workspace_rows, (64, self.ps // 64), 1),
                        stride=(self.ps, (1, 64), 0),
                    ),
                )
            else:
                active = cute.make_tensor(
                    activation.iterator,
                    cute.make_layout(
                        (workspace_rows, self.ps, 1), stride=(self.ps, 1, 0)
                    ),
                )
            map_a = cute_ext.get_cta_v_map_ab(active, up_tiler, up_mma, "B")
            ga = cute.local_tile(
                active,
                (self.n, self.up_k),
                (worker * (self.slot_rows // self.n), None, 0),
            )
        if cutlass.const_expr(self.pin_up_activation):
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    full, self.up_token_m * self.up_k * 2 * self.up_k_tiles
                )
        ready_base = worker * self.clusters
        ready_generation = generation
        if cutlass.const_expr(self.counter_activation_ready):
            # Every ACT store precedes a phase2 acq_rel arrival. An acquire
            # observing its final threshold covers the entire RMW chain, so
            # TMA need not wait for the subsequent separate READY release.
            ready_base = (
                2 * self.workers * self.grid_clusters + worker * self.group_clusters
            )
            ready_generation = generation * (2 * self.reduction_groups * self.split_k)
        all_ready = cutlass.Boolean(False)
        for tile in cutlass.range_constexpr(self.up_k_tiles):
            source_tile = cutlass.Int32(tile)
            if cutlass.const_expr(self.rotate_up_k):
                source_tile = (tile + k_offset) % self.up_k_tiles
            stage = tile if self.pin_up_activation else tile % self.up_activation_stages
            if cutlass.const_expr(not self.pin_up_activation):
                if cutlass.const_expr(tile >= self.up_activation_stages):
                    uses = (
                        self.up_k_tiles + self.up_activation_stages - 1 - stage
                    ) // self.up_activation_stages
                    previous = iteration * uses + tile // self.up_activation_stages - 1
                    cute.arch.mbarrier_wait(empty + stage, previous % 2)
                if cutlass.const_expr(self.stream_up_activation):
                    all_ready = self._wait_activation_epochs(
                        epochs, ready_base, source_tile, ready_generation, all_ready
                    )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        full + stage, self.up_token_m * self.up_k * 2
                    )
            ready = full if self.pin_up_activation else full + stage
            cute_ext.tma_load(
                ga[None, None, source_tile],
                sa[None, None, None, stage],
                ready.value,
                cta_v_map=map_a,
                tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                update_expect_tx=False,
            )

    @cute.experimental.jit
    def _wait_activation_epochs(self, epochs, base, tile, generation, all_ready):
        # A projection READY releases all its activation stores through counter2.
        # Cache the all-ready vote so later tiles need no repeated epoch loads.
        if not all_ready:
            lane = cute.arch.thread_idx()[0] % 32
            first = tile * self.up_k // self.down_m
            stop = ((tile + 1) * self.up_k + self.down_m - 1) // self.down_m
            tile_ready = cutlass.Boolean(False)
            while not tile_ready:
                local_all = cutlass.Boolean(True)
                local_tile = cutlass.Boolean(True)
                for chunk in cutlass.range_constexpr(
                    (self.activation_projection_clusters + 31) // 32
                ):
                    projection = chunk * 32 + lane
                    if projection < self.activation_projection_clusters:
                        observed = _load_epoch_acquire(
                            epochs.iterator + base + projection
                        )
                        ready = observed >= generation
                        local_all = local_all & ready
                        if projection >= first and projection < stop:
                            local_tile = local_tile & ready
                tile_ready = cute.arch.vote_all_sync(local_tile, mask=0xFFFFFFFF)
                all_ready = cute.arch.vote_all_sync(local_all, mask=0xFFFFFFFF)
            # Share the per-lane acquires with the elected TMA issuer. Votes
            # alone do not order memory across the warp.
            cute.arch.sync_warp(mask=0xFFFFFFFF)
        return all_ready

    @cute.experimental.jit
    def _wait_epochs(self, epochs, base, count: cutlass.Constexpr, generation):
        lane = cute.arch.thread_idx()[0] % 32
        ready = cutlass.Boolean(False)
        while not ready:
            local_ready = cutlass.Boolean(True)
            for chunk in cutlass.range_constexpr((count + 31) // 32):
                if lane + chunk * 32 < count:
                    local_ready = local_ready & (
                        _load_epoch_acquire(epochs.iterator + base + lane + chunk * 32)
                        >= generation
                    )
            ready = cute.arch.vote_all_sync(local_ready, mask=0xFFFFFFFF)

    @cute.experimental.jit
    def _up_tile(self, pid, step: cutlass.Constexpr):
        hidden_tile = (
            pid + (step // self.branch_tiles) * self.group_ctas
        ) * self.up_output_parts
        up_tile = hidden_tile + (step % self.branch_tiles) * self.hidden_tiles
        return hidden_tile, up_tile

    @cute.experimental.jit
    def _initialize_barriers(self, barriers, warp):
        """Initialize shared barriers once for all projection and token phases."""
        (
            down_full,
            down_empty,
            down_done,
            epi_done,
            up_full,
            up_activation_ready,
            up_k_empty,
            control_ready,
            projection_ready,
            up_done,
            reduce_ready,
            cluster_done,
            up_empty,
            consumed,
        ) = barriers
        act_full = (
            consumed
            + int(self.rounds * self.batch_tiles * self.token_tile_batches > 1)
            + 3 * int(self.reduction_groups > 1)
        )
        act_empty = act_full + self.up_activation_stages
        token_ready = act_full + 2 * self.up_activation_stages * int(
            self.independent_up_stages
        )
        if warp == self.weight_warp:
            with cute.arch.elect_one():
                for stage in cutlass.range_constexpr(self.down_stages):
                    cute.arch.mbarrier_init(down_full + stage, self.down_full_arrivals)
                if cutlass.const_expr(self.k_tiles > self.down_stages):
                    for stage in cutlass.range_constexpr(self.down_stages):
                        cute.arch.mbarrier_init(down_empty + stage, 1)
                cute.arch.mbarrier_init(down_done, 1)
                cute.arch.mbarrier_init(epi_done, self.epilogue_threads)
                for stage in cutlass.range_constexpr(self.up_stages):
                    cute.arch.mbarrier_init(
                        up_full + stage,
                        (
                            1
                            if self.pin_up_activation or self.independent_up_stages
                            else 2
                        ),
                    )
                    if cutlass.const_expr(self.up_k_tiles > self.up_stages):
                        cute.arch.mbarrier_init(up_k_empty + stage, 1)
                if cutlass.const_expr(self.independent_up_stages):
                    for stage in cutlass.range_constexpr(self.up_activation_stages):
                        cute.arch.mbarrier_init(act_full + stage, 1)
                        cute.arch.mbarrier_init(act_empty + stage, 1)
                if cutlass.const_expr(self.pin_up_activation):
                    cute.arch.mbarrier_init(up_activation_ready, 1)
                cute.arch.mbarrier_init(control_ready, 1)
                cute.arch.mbarrier_init(projection_ready, 1)
                cute.arch.mbarrier_init(up_done, 1)
                if cutlass.const_expr(self.up_token_tail_overlap):
                    for token_part in cutlass.range_constexpr(self.up_token_parts):
                        cute.arch.mbarrier_init(token_ready + token_part, 1)
                cute.arch.mbarrier_init(reduce_ready, 1)
                cute.arch.mbarrier_init(cluster_done, self.split_k)
                if cutlass.const_expr(self.up_steps > 1):
                    cute.arch.mbarrier_init(up_empty, 1)
                if cutlass.const_expr(
                    self.rounds * self.batch_tiles * self.token_tile_batches > 1
                ):
                    cute.arch.mbarrier_init(consumed, self.split_k)
        # Full native24 has one physical CTA per cluster, including both
        # local pipelines. Retain every initialization fence and participant.
        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(
            self.split_k == 1
            and self.hc_count == 4
            and self.hidden_size == 8192
            and self.ps == 1024
            and self.n == 24
            and self.full_tiles
        ):
            cute.arch.sync_threads()
        else:
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

    @cute.experimental.kernel
    def kernel(
        self,
        x,
        w,
        up,
        activation,
        partials,
        epochs,
        out,
        inject,
        up_atom,
        up_tma,
        down_atom,
        down_tma,
        x_atom,
        x_tma,
    ):
        physical_tid = cute.arch.thread_idx()[0]
        if cutlass.const_expr(self.compact_warp_layout):
            epilogue_end = self.epilogue_threads * self.pipelines_per_cta
            pipeline_id = physical_tid // self.epilogue_threads
            local_thread = physical_tid % self.epilogue_threads
            if physical_tid >= epilogue_end:
                pipeline_id = (physical_tid - epilogue_end) // 64
                local_thread = (
                    self.epilogue_threads + (physical_tid - epilogue_end) % 64
                )
            pipeline = cute.arch.make_warp_uniform(pipeline_id)
            tid = local_thread
        else:
            pipeline = cute.arch.make_warp_uniform(
                physical_tid // self.pipeline_threads
            )
            tid = physical_tid % self.pipeline_threads
        warp = cute.arch.make_warp_uniform(tid // 32)
        cluster = cute.arch.block_idx()[0] * self.pipelines_per_cta + pipeline
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        pid = cluster * self.split_k + split_rank
        worker = 0
        if cutlass.const_expr(self.clusters == 1 or self.workers > 1):
            worker = cute.arch.block_idx()[2]
        dtype = x.element_type
        down_mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            self.group,
            (self.down_m, self.down_n),
        )
        up_mma = sm100_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            cutlass.Float32,
            self.group,
            (self.up_rows, self.up_cols),
        )
        down_tiler = (self.down_m, self.down_n, self.down_k)
        up_tiler = (self.up_rows, self.up_cols, self.up_k)
        down_layout = sm100_utils.make_smem_layout_a(
            down_mma, down_tiler, dtype, self.down_stages
        )
        if cutlass.const_expr(self.token_major_up):
            up_layout = sm100_utils.make_smem_layout_b(
                up_mma, up_tiler, dtype, self.up_stages
            )
        else:
            up_layout = sm100_utils.make_smem_layout_a(
                up_mma, up_tiler, dtype, self.up_stages
            )
        x_layout = sm100_utils.make_smem_layout_b(
            down_mma, down_tiler, dtype, self.down_stages
        )
        if cutlass.const_expr(self.token_major_up):
            activation_layout = sm100_utils.make_smem_layout_a(
                up_mma, up_tiler, dtype, self.up_activation_stages
            )
        else:
            activation_layout = sm100_utils.make_smem_layout_b(
                up_mma, up_tiler, dtype, self.up_activation_stages
            )
        # One allocation makes the host byte plan the launch's exact storage.
        cta_storage = cute_ext.allocate(
            cutlass.Int8,
            cute.AddressSpace.smem,
            cute.make_layout(self.cta_shared_bytes),
            alignment=1024,
        )
        storage = cute.make_tensor(
            cute.make_ptr(
                cutlass.Int8,
                (cta_storage.iterator + pipeline * self.pipeline_shared_stride).toint(),
                cute.AddressSpace.smem,
                assumed_align=1024,
            ),
            cute.make_layout(self.storage.nbytes),
        )
        root_tmem_base = cute.recast_ptr(
            cta_storage.iterator + self.storage.tmem, dtype=cutlass.Int32
        )
        assert cutlass.const_expr(
            cute.cosize(down_layout) * 2
            == self.down_m * self.down_k * self.down_stages * 2
        )
        assert cutlass.const_expr(
            cute.cosize(x_layout) * 2
            == self.down_n * self.down_k * self.down_stages * 2
        )
        assert cutlass.const_expr(
            cute.cosize(up_layout) * 2 == self.up_m * self.up_k * self.up_stages * 2
        )
        assert cutlass.const_expr(
            cute.cosize(activation_layout) * 2
            == self.up_token_m * self.up_k * self.up_activation_stages * 2
        )
        gate = cute.make_tensor(
            cute.recast_ptr(
                storage.iterator + self.storage.gate, dtype=cutlass.Float32
            ),
            cute.make_layout(
                (self.up_m, self.up_token_m // self.up_read_slices),
                stride=(1, self.gate_stride),
            ),
        )
        sd = cute.make_tensor(
            cute.recast_ptr(
                storage.iterator + self.storage.down_weight,
                swizzle_=cute.get_swizzle_portion(down_layout),
                dtype=dtype,
            ),
            cute.get_nonswizzle_portion(down_layout),
        )
        sx = cute.make_tensor(
            cute.recast_ptr(
                storage.iterator + self.storage.down_x,
                swizzle_=cute.get_swizzle_portion(x_layout),
                dtype=dtype,
            ),
            cute.get_nonswizzle_portion(x_layout),
        )
        su = cute.make_tensor(
            cute.recast_ptr(
                storage.iterator + self.storage.up_weight,
                swizzle_=cute.get_swizzle_portion(up_layout),
                dtype=dtype,
            ),
            cute.get_nonswizzle_portion(up_layout),
        )
        sa = cute.make_tensor(
            cute.recast_ptr(
                storage.iterator + self.storage.up_activation,
                swizzle_=cute.get_swizzle_portion(activation_layout),
                dtype=dtype,
            ),
            cute.get_nonswizzle_portion(activation_layout),
        )
        mailbox = cute.make_tensor(
            cute.recast_ptr(
                storage.iterator + self.storage.mailbox, dtype=cutlass.Float32
            ),
            cute.make_layout(self.down_m * self.down_n),
        )
        partial_tile = cute.make_tensor(
            cute.recast_ptr(
                storage.iterator + self.storage.partial, dtype=cutlass.Float32
            ),
            cute.make_layout((self.down_m + 4) * self.down_n),
        )
        mix_accum = cute.make_tensor(
            cute.recast_ptr(
                storage.iterator + self.storage.mix_accum, dtype=cutlass.Float32
            ),
            cute.make_layout((self.final_tile, self.n)),
        )
        bars = cute.recast_ptr(
            storage.iterator + self.storage.barriers, dtype=cutlass.Int64
        )
        down_full = bars
        down_empty = down_full + self.down_stages
        down_done = down_empty + self.down_stages
        epi_done = down_done + 1
        up_full = epi_done + 1
        up_activation_ready = up_full + self.up_stages
        up_k_empty = up_activation_ready + int(self.pin_up_activation)
        control_ready = up_k_empty + self.up_stages * int(
            self.up_k_tiles > self.up_stages
        )
        projection_ready = control_ready + 1
        up_done = projection_ready + 1
        reduce_ready = up_done + 1
        cluster_done = reduce_ready + 1
        up_empty = cluster_done + 1
        consumed = up_empty + int(self.up_steps > 1)
        barriers = (
            down_full,
            down_empty,
            down_done,
            epi_done,
            up_full,
            up_activation_ready,
            up_k_empty,
            control_ready,
            projection_ready,
            up_done,
            reduce_ready,
            cluster_done,
            up_empty,
            consumed,
        )
        tmem_base = cute.recast_ptr(
            storage.iterator + self.storage.tmem, dtype=cutlass.Int32
        )
        if (
            pipeline == 0
            and warp == self.mma_warp
            and cutlass.const_expr(not self.single_tile)
        ):
            cute.arch.alloc_tmem(
                self.total_tmem_columns, root_tmem_base, is_two_cta=False
            )
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                root = cute.make_tensor(root_tmem_base, cute.make_layout(1))[0]
                for slot in cutlass.range_constexpr(self.pipelines_per_cta):
                    slot_base = cute.recast_ptr(
                        cta_storage.iterator
                        + slot * self.pipeline_shared_stride
                        + self.storage.tmem,
                        dtype=cutlass.Int32,
                    )
                    cute.make_tensor(slot_base, cute.make_layout(1))[0] = (
                        root + slot * self.tmem_columns
                    )
                    if cutlass.const_expr(self.token_tile_batches == 2):
                        cute.make_tensor(slot_base + 1, cute.make_layout(1))[0] = (
                            root + slot * self.tmem_columns + self.up_cols
                        )
                        cute.make_tensor(slot_base + 2, cute.make_layout(1))[0] = (
                            root + slot * self.tmem_columns + self.up_cols + self.down_n
                        )
            if cutlass.const_expr(self.pdl):
                cute.arch.griddepcontrol_wait()
                cute.arch.griddepcontrol_launch_dependents()
        self._initialize_barriers(barriers, warp)
        token_tiles = cutlass.Int32(cute.ceil_div(x.shape[0], self.n))
        for job_round in cutlass.range(self.rounds, unroll_full=self.rounds == 1):
            if cutlass.const_expr(self.token_tile_batches == 2):
                token = (job_round * self.workers + worker) * 2
                if token < token_tiles:
                    self._process_tile(
                        x,
                        w,
                        up,
                        activation,
                        partials,
                        epochs,
                        out,
                        inject,
                        tid,
                        warp,
                        cluster,
                        split_rank,
                        pid,
                        worker,
                        token,
                        job_round * 2,
                        down_mma,
                        up_mma,
                        sd,
                        sx,
                        su,
                        sa,
                        gate,
                        mix_accum,
                        mailbox,
                        partial_tile,
                        barriers,
                        tmem_base,
                        up_atom,
                        up_tma,
                        down_atom,
                        down_tma,
                        x_atom,
                        x_tma,
                        token + 1,
                        1,
                    )
                if token + 1 < token_tiles:
                    self._process_tile(
                        x,
                        w,
                        up,
                        activation,
                        partials,
                        epochs,
                        out,
                        inject,
                        tid,
                        warp,
                        cluster,
                        split_rank,
                        pid,
                        worker,
                        token + 1,
                        job_round * 2 + 1,
                        down_mma,
                        up_mma,
                        sd,
                        sx,
                        su,
                        sa,
                        gate,
                        mix_accum,
                        mailbox,
                        partial_tile,
                        barriers,
                        tmem_base,
                        up_atom,
                        up_tma,
                        down_atom,
                        down_tma,
                        x_atom,
                        x_tma,
                        -1,
                        2,
                    )
            else:
                for micro in cutlass.range_constexpr(self.batch_tiles):
                    tile_iteration = job_round * self.batch_tiles + micro
                    token = (
                        job_round * self.workers + worker
                    ) * self.batch_tiles + micro
                    if cutlass.const_expr(self.single_tile):
                        token = 0
                    if (
                        cutlass.const_expr(self.rounds == 1 and self.batch_tiles == 1)
                        or token < token_tiles
                    ):
                        self._process_tile(
                            x,
                            w,
                            up,
                            activation,
                            partials,
                            epochs,
                            out,
                            inject,
                            tid,
                            warp,
                            cluster,
                            split_rank,
                            pid,
                            worker,
                            token,
                            tile_iteration,
                            down_mma,
                            up_mma,
                            sd,
                            sx,
                            su,
                            sa,
                            gate,
                            mix_accum,
                            mailbox,
                            partial_tile,
                            barriers,
                            tmem_base,
                            up_atom,
                            up_tma,
                            down_atom,
                            down_tma,
                            x_atom,
                            x_tma,
                            -1,
                            0,
                        )
        if cutlass.const_expr(not self.single_tile):
            # Both pipelines must finish all TMEM reads before the sole free.
            cute.arch.sync_threads()
        if (
            pipeline == 0
            and warp == self.mma_warp
            and cutlass.const_expr(not self.single_tile)
        ):
            tcgen05_fence("after_thread_sync")
            cute.arch.dealloc_tmem(
                cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, root_tmem_base),
                self.total_tmem_columns,
                is_two_cta=False,
            )

    @cute.experimental.jit
    def _store_projection(
        self, value, col, row, activation, inject, x, worker, down_row, down_token
    ):
        value = value * self.scale
        if col < self.ps:
            value = value * cute_math.rcp(
                1.0 + cute.exp(-value, fastmath=True),
                fastmath=False,
                approx=True,
                rounding=None,
                ftz=True,
            )
            activation[worker * self.slot_rows + down_row + row, col] = value.to(
                activation.element_type
            )
            _publish_activation_proxy()
        elif col < self.p:
            if cutlass.const_expr(self.full_tiles):
                inject[down_token * self.down_n + row, col - self.ps] = value.to(
                    activation.element_type
                )
            else:
                _store_tail(
                    inject.iterator
                    + (cutlass.Int64(down_token) * self.down_n + row) * self.hc_count
                    + col
                    - self.ps,
                    value.to(activation.element_type),
                    row < cutlass.Int32(x.shape[0]) - down_token * self.down_n,
                )

    @cute.experimental.jit
    def _store_projection_packet(self, sums, col, row, activation, worker, down_row):
        packed = cute.make_rmem_tensor((4,), activation.element_type)
        for lane in cutlass.range_constexpr(4):
            value = sums[lane] * self.scale
            value = value * cute_math.rcp(
                1.0 + cute.exp(-value, fastmath=True),
                fastmath=False,
                approx=True,
                rounding=None,
                ftz=True,
            )
            packed[lane] = value.to(activation.element_type)
        pointer = cute.make_ptr(
            activation.element_type,
            (
                activation.iterator
                + (worker * self.slot_rows + down_row + row) * activation.stride[0]
                + col
            ).toint(),
            cute.AddressSpace.gmem,
            assumed_align=8,
        )
        target = cute.make_tensor(pointer, cute.make_layout(4))
        atom = cute.make_copy_atom(
            CopyUniversalOp(), activation.element_type, num_bits_per_copy=64
        )
        cute.copy(atom, packed, target)
        # One fence publishes the whole packet before the existing cohort release.
        _publish_activation_proxy()

    @cute.experimental.jit
    def _process_tile(
        self,
        x,
        w,
        up,
        activation,
        partials,
        epochs,
        out,
        inject,
        tid,
        warp,
        cluster,
        split_rank,
        pid,
        worker,
        token,
        tile_iteration,
        down_mma,
        up_mma,
        sd,
        sx,
        su,
        sa,
        gate,
        mix_accum,
        mailbox,
        partial_tile,
        barriers,
        tmem_base,
        up_atom,
        up_tma,
        down_atom,
        down_tma,
        x_atom,
        x_tma,
        pair_token,
        down_partner,
    ):
        dtype = x.element_type
        down_tiler = (self.down_m, self.down_n, self.down_k)
        up_tiler = (self.up_rows, self.up_cols, self.up_k)
        if cutlass.const_expr(self.token_tile_batches == 2):
            pair_valid = cutlass.Boolean(
                down_partner == 1 and pair_token * self.n < x.shape[0]
            )
            sx_b = cute.make_tensor(sx.iterator + self.down_x_bytes // 2, sx.layout)
        else:
            pair_valid = cutlass.Boolean(False)
        down_acc_base = tmem_base
        if cutlass.const_expr(down_partner == 1 and self.token_tile_batches == 2):
            down_acc_base = tmem_base + 1
        if cutlass.const_expr(down_partner == 2):
            down_acc_base = tmem_base + 2
        (
            down_full,
            down_empty,
            down_done,
            epi_done,
            up_full,
            up_activation_ready,
            up_k_empty,
            control_ready,
            projection_ready,
            up_done,
            reduce_ready,
            cluster_done,
            up_empty,
            consumed,
        ) = barriers
        partial_done = consumed + int(
            self.rounds * self.batch_tiles * self.token_tile_batches > 1
        )
        partial_cluster_done = partial_done + 1
        partial_ready = partial_done + 2
        act_full = up_full
        act_empty = up_k_empty
        if cutlass.const_expr(self.independent_up_stages):
            act_full = (
                consumed
                + int(self.rounds * self.batch_tiles * self.token_tile_batches > 1)
                + 3 * int(self.reduction_groups > 1)
            )
            act_empty = act_full + self.up_activation_stages
        partial_generation_storage = cute.make_tensor(partial_done, cute.make_layout(1))
        token_ready = (
            consumed
            + int(self.rounds * self.batch_tiles * self.token_tile_batches > 1)
            + 3 * int(self.reduction_groups > 1)
            + 2 * self.up_activation_stages * int(self.independent_up_stages)
        )
        partial_epoch_base = 2 * self.workers * self.grid_clusters
        reduction_group = cluster // self.group_clusters
        group_cluster = cluster % self.group_clusters
        pipeline = cluster % self.pipelines_per_cta
        k_offset = cutlass.Int32(0)
        if cutlass.const_expr(self.rotate_up_k):
            k_offset = (cluster // self.pipelines_per_cta) % self.up_k_tiles
        if cutlass.const_expr(self.local_pair_reduce):
            # Down pairs adjacent K ranks of one cohort. Up keeps cluster/pid.
            physical = cluster // self.pipelines_per_cta
            group_cluster = physical % self.group_clusters
            reduction_group = 2 * (physical // self.group_clusters) + pipeline
        partial_group = reduction_group
        down_barrier_id = pipeline + 1
        counter_leader = cutlass.Boolean(True)
        if cutlass.const_expr(self.local_pair_reduce):
            partial_group = reduction_group // 2
            down_barrier_id = cutlass.Int32(3)
            counter_leader = pipeline == 0
        logical_split_rank = reduction_group * self.split_k + split_rank
        down_acc_layout = cute_ext.make_tmem_layout_acc(
            down_mma, (self.down_m, self.down_n), acc_stage=1
        )
        up_acc_layout = cute_ext.make_tmem_layout_acc(
            up_mma,
            (self.up_rows, self.up_cols),
            acc_stage=self.up_output_parts * self.up_token_parts,
        )
        epoch_half = self.workers * self.clusters
        if cutlass.const_expr(self.clusters == 1):
            epoch_half = epochs.shape[0] // 2
        generation = cutlass.Int64(0)
        has_projection = cutlass.Boolean(True)
        if cutlass.const_expr(self.up_clusters > 0):
            has_projection = cluster < self.clusters
        start_base = epoch_half + self.workers * self.grid_clusters
        up_steps = self.up_steps
        cta_up_steps = (
            (
                self.hidden_tiles
                + self.group_ctas * self.up_output_parts
                - 1
                - pid * self.up_output_parts
            )
            // (self.group_ctas * self.up_output_parts)
        ) * self.branch_tiles
        tile_phase = tile_iteration % 2
        for projection_step in cutlass.range_constexpr(self.projection_tiles):
            projection = (
                cluster % self.projection_clusters
                + projection_step * self.projection_clusters
            )
            if cutlass.const_expr(self.local_pair_reduce):
                projection = group_cluster
            down_row = (group_cluster // self.projection_clusters) * self.down_n
            down_token = (
                token * self.down_token_parts
                + group_cluster // self.projection_clusters
            )
            if cutlass.const_expr(self.token_tile_batches == 2):
                down_token_pair = (
                    pair_token * self.down_token_parts
                    + group_cluster // self.projection_clusters
                )
            projection_iteration = (
                tile_iteration * self.projection_tiles + projection_step
            )
            projection_phase = projection_iteration % 2
            epi_phase = (
                tile_iteration * (self.projection_tiles + cta_up_steps)
                + projection_step
            ) % 2
            if cutlass.const_expr(self.up_clusters > 0):
                if not has_projection:
                    # Up-only CTAs do not acknowledge projection epilogues.
                    epi_phase = (tile_iteration * cta_up_steps + 1) % 2
            if warp == self.weight_warp:
                # Both weight projections use this producer. The first Down W
                # stage can overlap input readiness in independent mode.
                if cutlass.const_expr(self.pdl and not self.weights_independent):
                    cute.arch.griddepcontrol_wait()
                map_d = cute_ext.get_cta_v_map_ab(w, down_tiler, down_mma, "A")
                map_u = cute_ext.get_cta_v_map_ab(
                    up, up_tiler, up_mma, "B" if self.token_major_up else "A"
                )
                gd = cute.local_tile(
                    w, (self.down_m, self.down_k), (projection, None, 0)
                )
                if cutlass.const_expr(
                    self.token_tile_batches == 2 and self.joint_down_producer
                ):
                    post_bytes = self.down_weight_post_bytes
                    if pair_valid:
                        post_bytes = post_bytes + self.down_n * self.down_k * 2
                else:
                    post_bytes = self.down_weight_post_bytes
                if cutlass.const_expr(down_partner != 2):
                    if has_projection:
                        if cutlass.const_expr(self.joint_down_producer):
                            joint_map_x = cute_ext.get_cta_v_map_ab(
                                x, down_tiler, down_mma, "B"
                            )
                            joint_gx = cute.local_tile(
                                x, (self.down_n, self.down_k), (down_token, None, 0)
                            )
                            if cutlass.const_expr(self.token_tile_batches == 2):
                                joint_gx_pair = cute.local_tile(
                                    x,
                                    (self.down_n, self.down_k),
                                    (down_token_pair, None, 0),
                                )
                        # Logical splits visit adjacent K tiles together. The X
                        # producer below uses the same interleaved global index.
                        for stage in cutlass.range_constexpr(self.down_stages):
                            if cutlass.const_expr(
                                not self.joint_down_producer or stage == 0
                            ):
                                with cute.arch.elect_one():
                                    cute.arch.mbarrier_arrive_and_expect_tx(
                                        down_full + stage, post_bytes
                                    )
                                self._load_down_weight(
                                    gd,
                                    sd,
                                    stage,
                                    self._down_input_tile(stage, logical_split_rank),
                                    down_full,
                                    map_d,
                                    down_atom,
                                    down_tma,
                                    projection,
                                )
                        if cutlass.const_expr(self.joint_down_producer):
                            # Preserve the first W stage ahead of input readiness.
                            # Pair later W/X stages after it, without delaying X0.
                            cute.arch.mbarrier_wait(control_ready, projection_phase)
                            for stage in cutlass.range_constexpr(self.down_stages):
                                if cutlass.const_expr(stage > 0):
                                    with cute.arch.elect_one():
                                        cute.arch.mbarrier_arrive_and_expect_tx(
                                            down_full + stage, post_bytes
                                        )
                                    self._load_down_weight(
                                        gd,
                                        sd,
                                        stage,
                                        self._down_input_tile(
                                            stage, logical_split_rank
                                        ),
                                        down_full,
                                        map_d,
                                        down_atom,
                                        down_tma,
                                        projection,
                                    )
                                if cutlass.const_expr(
                                    self.down_x_cache_mode != "native"
                                ):
                                    self._load_down_x(
                                        joint_gx,
                                        sx,
                                        stage,
                                        self._down_input_tile(
                                            stage, logical_split_rank
                                        ),
                                        down_full,
                                        joint_map_x,
                                        x_atom,
                                        x_tma,
                                        down_token,
                                    )
                                else:
                                    cute_ext.tma_load(
                                        joint_gx[
                                            None,
                                            None,
                                            self._down_input_tile(
                                                stage, logical_split_rank
                                            ),
                                        ],
                                        sx[None, None, None, stage],
                                        (down_full + stage).value,
                                        cta_v_map=joint_map_x,
                                        tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                                        update_expect_tx=False,
                                    )
                                if cutlass.const_expr(self.token_tile_batches == 2):
                                    if pair_valid:
                                        if cutlass.const_expr(
                                            self.down_x_cache_mode != "native"
                                        ):
                                            self._load_down_x(
                                                joint_gx_pair,
                                                sx_b,
                                                stage,
                                                self._down_input_tile(
                                                    stage, logical_split_rank
                                                ),
                                                down_full,
                                                joint_map_x,
                                                x_atom,
                                                x_tma,
                                                down_token_pair,
                                            )
                                        else:
                                            cute_ext.tma_load(
                                                joint_gx_pair[
                                                    None,
                                                    None,
                                                    self._down_input_tile(
                                                        stage, logical_split_rank
                                                    ),
                                                ],
                                                sx_b[None, None, None, stage],
                                                (down_full + stage).value,
                                                cta_v_map=joint_map_x,
                                                tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                                                update_expect_tx=False,
                                            )
                        if cutlass.const_expr(
                            not self.reuse_down_storage
                            and projection_step == self.projection_tiles - 1
                        ):
                            if pid * self.up_output_parts < self.hidden_tiles:
                                self._load_up(
                                    up,
                                    su,
                                    up_full,
                                    map_u,
                                    pid * self.up_output_parts,
                                    tile_iteration * cta_up_steps,
                                    up_k_empty,
                                    0,
                                    self.up_stages,
                                    up_atom,
                                    up_tma,
                                    k_offset,
                                )
                        # Keep stage addresses static without replicating the
                        # producer code for every K tile in a wide projection.
                        for cycle in cutlass.range(
                            1,
                            (self.k_tiles + self.down_stages - 1) // self.down_stages,
                            unroll_full=False,
                        ):
                            for stage in cutlass.range_constexpr(self.down_stages):
                                tile = cycle * self.down_stages + stage
                                if tile < self.k_tiles:
                                    stage_uses = (
                                        self.k_tiles + self.down_stages - 1 - stage
                                    ) // self.down_stages
                                    down_iteration = (
                                        projection_iteration * stage_uses + cycle
                                    )
                                    cute.arch.mbarrier_wait(
                                        down_empty + stage, (down_iteration - 1) % 2
                                    )
                                    with cute.arch.elect_one():
                                        cute.arch.mbarrier_arrive_and_expect_tx(
                                            down_full + stage, post_bytes
                                        )
                                    self._load_down_weight(
                                        gd,
                                        sd,
                                        stage,
                                        self._down_input_tile(tile, logical_split_rank),
                                        down_full,
                                        map_d,
                                        down_atom,
                                        down_tma,
                                        projection,
                                    )
                                    if cutlass.const_expr(self.joint_down_producer):
                                        # W and X share one empty wait and one credit.
                                        if cutlass.const_expr(
                                            self.down_x_cache_mode != "native"
                                        ):
                                            self._load_down_x(
                                                joint_gx,
                                                sx,
                                                stage,
                                                self._down_input_tile(
                                                    tile, logical_split_rank
                                                ),
                                                down_full,
                                                joint_map_x,
                                                x_atom,
                                                x_tma,
                                                down_token,
                                            )
                                        else:
                                            cute_ext.tma_load(
                                                joint_gx[
                                                    None,
                                                    None,
                                                    self._down_input_tile(
                                                        tile, logical_split_rank
                                                    ),
                                                ],
                                                sx[None, None, None, stage],
                                                (down_full + stage).value,
                                                cta_v_map=joint_map_x,
                                                tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                                                update_expect_tx=False,
                                            )
                                        if cutlass.const_expr(
                                            self.token_tile_batches == 2
                                        ):
                                            if pair_valid:
                                                if cutlass.const_expr(
                                                    self.down_x_cache_mode != "native"
                                                ):
                                                    self._load_down_x(
                                                        joint_gx_pair,
                                                        sx_b,
                                                        stage,
                                                        self._down_input_tile(
                                                            tile, logical_split_rank
                                                        ),
                                                        down_full,
                                                        joint_map_x,
                                                        x_atom,
                                                        x_tma,
                                                        down_token_pair,
                                                    )
                                                else:
                                                    cute_ext.tma_load(
                                                        joint_gx_pair[
                                                            None,
                                                            None,
                                                            self._down_input_tile(
                                                                tile, logical_split_rank
                                                            ),
                                                        ],
                                                        sx_b[None, None, None, stage],
                                                        (down_full + stage).value,
                                                        cta_v_map=joint_map_x,
                                                        tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                                                        update_expect_tx=False,
                                                    )
                if cutlass.const_expr(projection_step == self.projection_tiles - 1):
                    if cutlass.const_expr(self.reuse_down_storage):
                        if cutlass.const_expr(self.split_k == 1):
                            if has_projection:
                                # S1 projection epilogues read TMEM/global
                                # partials, so the operand arena is free here.
                                cute.arch.mbarrier_wait(down_done, projection_phase)
                            if pid * self.up_output_parts < self.hidden_tiles:
                                # Keep one weight stage ahead of local readiness.
                                # Later stages follow the selected READY policy.
                                self._load_up(
                                    up,
                                    su,
                                    up_full,
                                    map_u,
                                    pid * self.up_output_parts,
                                    tile_iteration * cta_up_steps,
                                    up_k_empty,
                                    0,
                                    1,
                                    up_atom,
                                    up_tma,
                                    k_offset,
                                )
                                if cutlass.const_expr(self.up_l2_prefetch_k > 0):
                                    self._prefetch_future_up_weights(
                                        up, pid * self.up_output_parts
                                    )
                        if has_projection:
                            cute.arch.mbarrier_wait(projection_ready, tile_phase)
                        if pid * self.up_output_parts < self.hidden_tiles:
                            self._load_up(
                                up,
                                su,
                                up_full,
                                map_u,
                                pid * self.up_output_parts,
                                tile_iteration * cta_up_steps,
                                up_k_empty,
                                1 if self.split_k == 1 else 0,
                                self.up_stages,
                                up_atom,
                                up_tma,
                                k_offset,
                            )
                    else:
                        if not has_projection:
                            if pid * self.up_output_parts < self.hidden_tiles:
                                self._load_up(
                                    up,
                                    su,
                                    up_full,
                                    map_u,
                                    pid * self.up_output_parts,
                                    tile_iteration * cta_up_steps,
                                    up_k_empty,
                                    0,
                                    self.up_stages,
                                    up_atom,
                                    up_tma,
                                    k_offset,
                                )
                    # Complete Down's loads before waiting for recycled Up
                    # slots: Up cannot consume until projection is published.
                    if pid * self.up_output_parts < self.hidden_tiles:
                        self._load_up(
                            up,
                            su,
                            up_full,
                            map_u,
                            pid * self.up_output_parts,
                            tile_iteration * cta_up_steps,
                            up_k_empty,
                            self.up_stages,
                            self.up_k_tiles,
                            up_atom,
                            up_tma,
                            k_offset,
                        )
                    for up_step in cutlass.range_constexpr(1, up_steps):
                        hidden_id, up_id = self._up_tile(pid, up_step)
                        if hidden_id < self.hidden_tiles:
                            up_iteration = tile_iteration * cta_up_steps + up_step
                            cute.arch.mbarrier_wait(up_empty, (up_iteration - 1) % 2)
                            map_u = cute_ext.get_cta_v_map_ab(
                                up,
                                up_tiler,
                                up_mma,
                                "B" if self.token_major_up else "A",
                            )
                            self._load_up(
                                up,
                                su,
                                up_full,
                                map_u,
                                up_id,
                                up_iteration,
                                up_k_empty,
                                0,
                                self.up_k_tiles,
                                up_atom,
                                up_tma,
                                k_offset,
                            )
            elif warp == self.mma_warp:
                if cutlass.const_expr(self.single_tile and projection_step == 0):
                    cute.arch.alloc_tmem(self.tmem_columns, tmem_base, is_two_cta=False)
                    cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
                    if cutlass.const_expr(self.pdl):
                        cute.arch.griddepcontrol_wait()
                        cute.arch.griddepcontrol_launch_dependents()
                if cutlass.const_expr(projection_step == 0):
                    ready_cluster = cutlass.Int32(0)
                    if has_projection:
                        ready_cluster = cluster
                        if cutlass.const_expr(self.reduction_groups > 1):
                            ready_cluster = group_cluster
                    generation = (
                        _load_epoch_acquire(
                            epochs.iterator + worker * self.clusters + ready_cluster
                        )
                        + 1
                    )
                if has_projection:
                    with cute.arch.elect_one():
                        if cutlass.const_expr(self.reduction_groups > 1):
                            partial_generation_storage[0] = (
                                (generation - 1) * self.projection_tiles
                                + projection_step
                                + 1
                            )
                        cute.arch.mbarrier_arrive(control_ready)
                    if cutlass.const_expr(down_partner != 2):
                        if cutlass.const_expr(self.token_tile_batches == 2):
                            down_acc = cute.make_tensor(
                                cute.arch.retrieve_tmem_ptr(
                                    cutlass.Float32, 16, tmem_base + 1
                                ),
                                down_acc_layout,
                            )
                            down_acc_pair = cute.make_tensor(
                                cute.arch.retrieve_tmem_ptr(
                                    cutlass.Float32, 16, tmem_base + 2
                                ),
                                down_acc_layout,
                            )
                        else:
                            down_acc = cute.make_tensor(
                                cute.arch.retrieve_tmem_ptr(
                                    cutlass.Float32, 16, tmem_base
                                ),
                                down_acc_layout,
                            )
                    if cutlass.const_expr(down_partner != 2):
                        down_instruction = Tcgen05InstrDesc.build(
                            sparse_id2=0,
                            sparse_flag=0,
                            saturate=0,
                            sparse_format=0,
                            c_dtype=cutlass.Float32,
                            a_dtype=dtype,
                            b_dtype=dtype,
                            a_negate=0,
                            b_negate=0,
                            a_major=0,
                            b_major=0,
                            n_dim=self.down_n,
                            m_dim=self.down_m,
                            k_dim=0,
                            max_shift=0,
                        )
                        down_a = sd[None, None, 0, 0]
                        down_b = sx[None, None, 0, 0]
                        desc_a = tcgen05.smem_descriptor_to_int(
                            tcgen05.make_umma_smem_desc(
                                down_a.iterator, down_a.layout, "k", next_src=None
                            )
                        )
                        desc_b = tcgen05.smem_descriptor_to_int(
                            tcgen05.make_umma_smem_desc(
                                down_b.iterator, down_b.layout, "k", next_src=None
                            )
                        )
                        if cutlass.const_expr(self.token_tile_batches == 2):
                            down_b_pair = sx_b[None, None, 0, 0]
                            desc_b_pair = tcgen05.smem_descriptor_to_int(
                                tcgen05.make_umma_smem_desc(
                                    down_b_pair.iterator,
                                    down_b_pair.layout,
                                    "k",
                                    next_src=None,
                                )
                            )
                        for tile in cutlass.range_constexpr(self.k_tiles):
                            stage = tile % self.down_stages
                            stage_uses = (
                                self.k_tiles + self.down_stages - 1 - stage
                            ) // self.down_stages
                            phase = (
                                projection_iteration * stage_uses
                                + tile // self.down_stages
                            ) % 2
                            cute.arch.mbarrier_wait(down_full + stage, phase)
                            with cute.arch.elect_one():
                                for kb in cutlass.range_constexpr(self.down_k // 16):
                                    idx = stage * (self.down_k // 16) + kb
                                    _mma_single_thread(
                                        down_acc[None, None, None, 0],
                                        desc_a
                                        + (idx // 4) * self.down_m * 8
                                        + (idx % 4) * 2,
                                        desc_b
                                        + (idx // 4) * self.down_n * 8
                                        + (idx % 4) * 2,
                                        down_instruction,
                                        tile != 0 or kb != 0,
                                    )
                                if cutlass.const_expr(self.token_tile_batches == 2):
                                    if pair_valid:
                                        for kb_pair in cutlass.range_constexpr(
                                            self.down_k // 16
                                        ):
                                            idx_pair = (
                                                stage * (self.down_k // 16) + kb_pair
                                            )
                                            _mma_single_thread(
                                                down_acc_pair[None, None, None, 0],
                                                desc_a
                                                + (idx_pair // 4) * self.down_m * 8
                                                + (idx_pair % 4) * 2,
                                                desc_b_pair
                                                + (idx_pair // 4) * self.down_n * 8
                                                + (idx_pair % 4) * 2,
                                                down_instruction,
                                                tile != 0 or kb_pair != 0,
                                            )
                                if cutlass.const_expr(self.k_tiles > self.down_stages):
                                    tcgen05.commit(down_empty + stage, None, self.group)
                    with cute.arch.elect_one():
                        tcgen05.commit(down_done, None, self.group)
                    cute.arch.mbarrier_wait(epi_done, epi_phase)
                if cutlass.const_expr(projection_step == self.projection_tiles - 1):
                    if has_projection:
                        if cutlass.const_expr(self.reduction_groups == 1):
                            if cutlass.const_expr(self.split_k > 1):
                                with cute.arch.elect_one():
                                    _cluster_arrive(cluster_done)
                                    if split_rank == 0:
                                        _cluster_wait(cluster_done, tile_phase)
                            if cutlass.const_expr(self.up_clusters > 0):
                                if cluster == 0 and split_rank == 0:
                                    self._wait_epochs(
                                        epochs,
                                        start_base + worker * self.up_clusters,
                                        self.up_clusters,
                                        generation,
                                    )
                            with cute.arch.elect_one():
                                if split_rank == 0:
                                    _store_epoch_release(
                                        epochs.iterator
                                        + worker * self.clusters
                                        + cluster,
                                        generation,
                                    )
                    else:
                        with cute.arch.elect_one():
                            _cluster_arrive(cluster_done)
                            if split_rank == 0:
                                _cluster_wait(cluster_done, tile_phase)
                                _store_epoch_release(
                                    epochs.iterator
                                    + start_base
                                    + worker * self.up_clusters
                                    + cluster
                                    - self.clusters,
                                    generation,
                                )
                    if cutlass.const_expr(not self.stream_up_activation):
                        self._wait_epochs(
                            epochs,
                            worker * self.clusters,
                            self.group_clusters,
                            generation,
                        )
                    # Own Down EPI ACK has retired every local operand reader.
                    # A streamed X producer acquires the needed projection flags.
                    cute.arch.sync_warp(mask=0xFFFFFFFF)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(projection_ready)
                    if cutlass.const_expr(self.pin_up_activation):
                        if pid * self.up_output_parts < self.hidden_tiles:
                            cute.arch.mbarrier_wait(up_activation_ready, tile_phase)
                    for up_step in cutlass.range_constexpr(up_steps):
                        hidden_id, up_id = self._up_tile(pid, up_step)
                        if hidden_id < self.hidden_tiles:
                            up_iteration = tile_iteration * cta_up_steps + up_step
                            up_phase = up_iteration % 2
                            up_acc = cute.make_tensor(
                                cute.arch.retrieve_tmem_ptr(
                                    cutlass.Float32, 16, tmem_base
                                ),
                                up_acc_layout,
                            )
                            up_instruction = Tcgen05InstrDesc.build(
                                sparse_id2=0,
                                sparse_flag=0,
                                saturate=0,
                                sparse_format=0,
                                c_dtype=cutlass.Float32,
                                a_dtype=dtype,
                                b_dtype=dtype,
                                a_negate=0,
                                b_negate=0,
                                a_major=0,
                                b_major=0,
                                n_dim=self.up_cols,
                                m_dim=self.up_rows,
                                k_dim=0,
                                max_shift=0,
                            )
                            # K64 group bases are 1024-byte aligned, preserving
                            # swizzle/base/layout fields. The 14-bit start field
                            # uses 16-byte units, with no carry here.
                            if cutlass.const_expr(self.token_major_up):
                                up_a = sa[None, None, 0, 0]
                                up_b = su[None, None, 0, 0]
                            else:
                                up_a = su[None, None, 0, 0]
                                up_b = sa[None, None, 0, 0]
                            desc_a = tcgen05.smem_descriptor_to_int(
                                tcgen05.make_umma_smem_desc(
                                    up_a.iterator,
                                    up_a.layout,
                                    "k",
                                    next_src=None,
                                )
                            )
                            desc_b = tcgen05.smem_descriptor_to_int(
                                tcgen05.make_umma_smem_desc(
                                    up_b.iterator,
                                    up_b.layout,
                                    "k",
                                    next_src=None,
                                )
                            )
                            for token_phase in cutlass.range_constexpr(
                                3 if self.up_token_tail_overlap else 1
                            ):
                                for tile in cutlass.range_constexpr(
                                    (
                                        0
                                        if token_phase == 0
                                        else self.up_k_tiles - self.up_tail_tiles
                                    ),
                                    (
                                        self.up_k_tiles - self.up_tail_tiles
                                        if token_phase == 0
                                        else self.up_k_tiles
                                    ),
                                ):
                                    stage = tile % self.up_stages
                                    uses = (
                                        self.up_k_tiles + self.up_stages - 1 - stage
                                    ) // self.up_stages
                                    phase = (
                                        up_iteration * uses + tile // self.up_stages
                                    ) % 2
                                    cute.arch.mbarrier_wait(up_full + stage, phase)
                                    if cutlass.const_expr(self.independent_up_stages):
                                        act_stage = tile % self.up_activation_stages
                                        act_uses = (
                                            self.up_k_tiles
                                            + self.up_activation_stages
                                            - 1
                                            - act_stage
                                        ) // self.up_activation_stages
                                        act_phase = (
                                            up_iteration * act_uses
                                            + tile // self.up_activation_stages
                                        ) % 2
                                        cute.arch.mbarrier_wait(
                                            act_full + act_stage, act_phase
                                        )
                                    with cute.arch.elect_one():
                                        for output_part in cutlass.range_constexpr(
                                            self.up_output_parts
                                        ):
                                            for token_part in cutlass.range_constexpr(
                                                self.up_token_parts
                                            ):
                                                if cutlass.const_expr(
                                                    not self.up_token_tail_overlap
                                                    or token_phase == 0
                                                    or token_part == token_phase - 1
                                                ):
                                                    for (
                                                        block
                                                    ) in cutlass.range_constexpr(
                                                        min(
                                                            self.up_k // 16,
                                                            (
                                                                self.ps
                                                                - tile * self.up_k
                                                                + 15
                                                            )
                                                            // 16,
                                                        )
                                                    ):
                                                        kb = (
                                                            stage * (self.up_k // 16)
                                                            + block
                                                        )
                                                        activation_kb = (
                                                            tile
                                                            if self.pin_up_activation
                                                            else tile
                                                            % self.up_activation_stages
                                                        ) * (self.up_k // 16) + block
                                                        if cutlass.const_expr(
                                                            self.token_major_up
                                                        ):
                                                            a_kb, b_kb = (
                                                                activation_kb,
                                                                kb,
                                                            )
                                                        else:
                                                            a_kb, b_kb = (
                                                                kb,
                                                                activation_kb,
                                                            )
                                                        _mma_single_thread(
                                                            up_acc[
                                                                None,
                                                                None,
                                                                None,
                                                                output_part
                                                                * self.up_token_parts
                                                                + token_part,
                                                            ],
                                                            desc_a
                                                            + token_part
                                                            * (
                                                                self.up_token_m
                                                                * self.up_k
                                                                * self.up_activation_stages
                                                                * 2
                                                                // 16
                                                            )
                                                            + output_part
                                                            * (
                                                                self.up_m
                                                                * self.up_k
                                                                * self.up_stages
                                                                * 2
                                                                // 16
                                                            )
                                                            + (a_kb // 4)
                                                            * self.up_rows
                                                            * 8
                                                            + (a_kb % 4) * 2,
                                                            desc_b
                                                            + (b_kb // 4)
                                                            * self.up_cols
                                                            * 8
                                                            + (b_kb % 4) * 2,
                                                            up_instruction,
                                                            tile != 0 or block != 0,
                                                        )
                                        if cutlass.const_expr(
                                            (self.up_k_tiles > self.up_stages)
                                            and (
                                                not self.up_token_tail_overlap
                                                or token_phase != 1
                                            )
                                        ):
                                            tcgen05.commit(
                                                up_k_empty + stage, None, self.group
                                            )
                                        if cutlass.const_expr(
                                            (
                                                self.independent_up_stages
                                                and self.up_k_tiles
                                                > self.up_activation_stages
                                            )
                                            and (
                                                not self.up_token_tail_overlap
                                                or token_phase != 1
                                            )
                                        ):
                                            tcgen05.commit(
                                                act_empty
                                                + tile % self.up_activation_stages,
                                                None,
                                                self.group,
                                            )
                                if cutlass.const_expr(
                                    self.up_token_tail_overlap and token_phase > 0
                                ):
                                    with cute.arch.elect_one():
                                        tcgen05.commit(
                                            token_ready + token_phase - 1,
                                            None,
                                            self.group,
                                        )
                            with cute.arch.elect_one():
                                if cutlass.const_expr(not self.up_token_tail_overlap):
                                    tcgen05.commit(up_done, None, self.group)
                                if cutlass.const_expr(self.up_steps > 1):
                                    tcgen05.commit(up_empty, None, self.group)
                            # Each phase acknowledges the preceding TMEM read
                            # before its storage is reused by the next step.
                            cute.arch.mbarrier_wait(
                                epi_done, (epi_phase + up_step + 1) % 2
                            )
                    if cutlass.const_expr(
                        self.local_pair_reduce and self.stream_up_activation
                    ):
                        # Up acquired each ACT tile before loading it. Join all
                        # cohorts, including inject-only work, before retiring
                        # this generation or releasing the shared TMEM arena.
                        # Actors without an Up tile must also take this edge.
                        self._wait_epochs(
                            epochs,
                            worker * self.clusters,
                            self.group_clusters,
                            generation,
                        )
                        cute.arch.sync_warp(mask=0xFFFFFFFF)
                    if cutlass.const_expr(self.single_tile):
                        tcgen05_fence("after_thread_sync")
                        cute.arch.dealloc_tmem(
                            cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, tmem_base),
                            self.tmem_columns,
                            is_two_cta=False,
                        )
            elif warp < self.epilogue_warps:
                epi_tid = tid
                if has_projection:
                    cute.arch.mbarrier_wait(control_ready, projection_phase)
                    if warp == 0 and cutlass.const_expr(
                        not self.joint_down_producer and down_partner != 2
                    ):
                        map_x = cute_ext.get_cta_v_map_ab(x, down_tiler, down_mma, "B")
                        gx = cute.local_tile(
                            x, (self.down_n, self.down_k), (down_token, None, 0)
                        )
                        if cutlass.const_expr(self.token_tile_batches == 2):
                            gx_pair = cute.local_tile(
                                x,
                                (self.down_n, self.down_k),
                                (down_token_pair, None, 0),
                            )
                            x_post_bytes = self.down_n * self.down_k * 2
                            if pair_valid:
                                x_post_bytes = (
                                    x_post_bytes + self.down_n * self.down_k * 2
                                )
                        else:
                            x_post_bytes = self.down_n * self.down_k * 2
                        for stage in cutlass.range_constexpr(self.down_stages):
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(
                                    down_full + stage, x_post_bytes
                                )
                            if cutlass.const_expr(self.down_x_cache_mode != "native"):
                                self._load_down_x(
                                    gx,
                                    sx,
                                    stage,
                                    self._down_input_tile(stage, logical_split_rank),
                                    down_full,
                                    map_x,
                                    x_atom,
                                    x_tma,
                                    down_token,
                                )
                            else:
                                cute_ext.tma_load(
                                    gx[
                                        None,
                                        None,
                                        self._down_input_tile(
                                            stage, logical_split_rank
                                        ),
                                    ],
                                    sx[None, None, None, stage],
                                    (down_full + stage).value,
                                    cta_v_map=map_x,
                                    tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                                    update_expect_tx=False,
                                )
                            if cutlass.const_expr(self.token_tile_batches == 2):
                                if pair_valid:
                                    if cutlass.const_expr(
                                        self.down_x_cache_mode != "native"
                                    ):
                                        self._load_down_x(
                                            gx_pair,
                                            sx_b,
                                            stage,
                                            self._down_input_tile(
                                                stage, logical_split_rank
                                            ),
                                            down_full,
                                            map_x,
                                            x_atom,
                                            x_tma,
                                            down_token_pair,
                                        )
                                    else:
                                        cute_ext.tma_load(
                                            gx_pair[
                                                None,
                                                None,
                                                self._down_input_tile(
                                                    stage, logical_split_rank
                                                ),
                                            ],
                                            sx_b[None, None, None, stage],
                                            (down_full + stage).value,
                                            cta_v_map=map_x,
                                            tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                                            update_expect_tx=False,
                                        )
                        for cycle in cutlass.range(
                            1,
                            (self.k_tiles + self.down_stages - 1) // self.down_stages,
                            unroll_full=False,
                        ):
                            for stage in cutlass.range_constexpr(self.down_stages):
                                tile = cycle * self.down_stages + stage
                                if tile < self.k_tiles:
                                    stage_uses = (
                                        self.k_tiles + self.down_stages - 1 - stage
                                    ) // self.down_stages
                                    down_iteration = (
                                        projection_iteration * stage_uses + cycle
                                    )
                                    cute.arch.mbarrier_wait(
                                        down_empty + stage, (down_iteration - 1) % 2
                                    )
                                    with cute.arch.elect_one():
                                        cute.arch.mbarrier_arrive_and_expect_tx(
                                            down_full + stage,
                                            x_post_bytes,
                                        )
                                    if cutlass.const_expr(
                                        self.down_x_cache_mode != "native"
                                    ):
                                        self._load_down_x(
                                            gx,
                                            sx,
                                            stage,
                                            self._down_input_tile(
                                                tile, logical_split_rank
                                            ),
                                            down_full,
                                            map_x,
                                            x_atom,
                                            x_tma,
                                            down_token,
                                        )
                                    else:
                                        cute_ext.tma_load(
                                            gx[
                                                None,
                                                None,
                                                self._down_input_tile(
                                                    tile, logical_split_rank
                                                ),
                                            ],
                                            sx[None, None, None, stage],
                                            (down_full + stage).value,
                                            cta_v_map=map_x,
                                            tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                                            update_expect_tx=False,
                                        )
                                    if cutlass.const_expr(self.token_tile_batches == 2):
                                        if pair_valid:
                                            if cutlass.const_expr(
                                                self.down_x_cache_mode != "native"
                                            ):
                                                self._load_down_x(
                                                    gx_pair,
                                                    sx_b,
                                                    stage,
                                                    self._down_input_tile(
                                                        tile, logical_split_rank
                                                    ),
                                                    down_full,
                                                    map_x,
                                                    x_atom,
                                                    x_tma,
                                                    down_token_pair,
                                                )
                                            else:
                                                cute_ext.tma_load(
                                                    gx_pair[
                                                        None,
                                                        None,
                                                        self._down_input_tile(
                                                            tile, logical_split_rank
                                                        ),
                                                    ],
                                                    sx_b[None, None, None, stage],
                                                    (down_full + stage).value,
                                                    cta_v_map=map_x,
                                                    tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                                                    update_expect_tx=False,
                                                )
                    cute.arch.mbarrier_wait(down_done, projection_phase)
                    if cutlass.const_expr(self.local_pair_reduce):
                        peer_done = down_done + (1 - 2 * pipeline) * (
                            self.pipeline_shared_stride // 8
                        )
                        cute.arch.mbarrier_wait(peer_done, projection_phase)
                        tcgen05_fence("after_thread_sync")
                        down_values, down_coords = self._paired_down_accumulator_tile(
                            down_acc_base,
                            down_acc_layout,
                            activation,
                            epi_tid,
                            pipeline,
                        )
                    else:
                        down_acc, down_thr, down_values, down_coords = (
                            self._accumulator_tile(
                                down_acc_base,
                                down_acc_layout,
                                self.down_m,
                                self.down_n,
                                activation,
                                epi_tid,
                                1,
                                0,
                                0,
                            )
                        )
                        cute_ext.partition_and_copy(down_thr, down_acc, down_values)
                        cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(self.split_k == 1):
                        # A single physical CTA already owns its complete K
                        # group's result; write TMEM values directly to L2.
                        for i in cutlass.range_constexpr(cute.size(down_values)):
                            local_col, row = down_coords[i]
                            col = projection * self.down_m + local_col
                            value = down_values[i]
                            if cutlass.const_expr(self.reduction_groups > 1):
                                if col < self.p and (
                                    cutlass.const_expr(not self.narrow_down_pair)
                                    or pipeline == 0
                                ):
                                    partials[
                                        worker * self.slot_rows + down_row + row,
                                        partial_group * self.partial_stride + col,
                                    ] = value
                            else:
                                self._store_projection(
                                    value,
                                    col,
                                    row,
                                    activation,
                                    inject,
                                    x,
                                    worker,
                                    down_row,
                                    down_token,
                                )
                    else:
                        # Every source scatters aligned four-column packets to S owners.
                        if epi_tid == 0:
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                reduce_ready, self.down_m * self.down_n * 4
                            )
                        for i in cutlass.range_constexpr(cute.size(down_values)):
                            col, row = down_coords[i]
                            partial_tile[row * (self.down_m + 4) + col] = down_values[i]
                        _epilogue_barrier(
                            self.epilogue_threads, cluster % self.pipelines_per_cta + 1
                        )
                        for packet in cutlass.range_constexpr(
                            self.down_m * self.down_n // (4 * self.epilogue_threads)
                        ):
                            item = epi_tid + packet * self.epilogue_threads
                            owner = item // (self.down_n * (self.owner_cols // 4))
                            row = item % self.down_n
                            pack = item // self.down_n % (self.owner_cols // 4)
                            offset = (
                                row * (self.down_m + 4)
                                + owner * self.owner_cols
                                + pack * 4
                            )
                            _remote_partial(
                                partial_tile[offset],
                                partial_tile[offset + 1],
                                partial_tile[offset + 2],
                                partial_tile[offset + 3],
                                mailbox.iterator
                                + split_rank * self.down_n * self.owner_cols
                                + row * self.owner_cols
                                + pack * 4,
                                reduce_ready,
                                owner,
                            )
                        cute.arch.mbarrier_wait(reduce_ready, projection_phase)
                        for part in cutlass.range_constexpr(
                            (self.down_n * self.owner_cols + self.epilogue_threads - 1)
                            // self.epilogue_threads
                        ):
                            item = epi_tid + part * self.epilogue_threads
                            row = item // self.owner_cols
                            col = (
                                projection * self.down_m
                                + split_rank * self.owner_cols
                                + item % self.owner_cols
                            )
                            if item < self.down_n * self.owner_cols:
                                value = mailbox[item]
                                for peer in cutlass.range_constexpr(1, self.split_k):
                                    value = (
                                        value
                                        + mailbox[
                                            peer * self.down_n * self.owner_cols + item
                                        ]
                                    )
                                if cutlass.const_expr(self.reduction_groups > 1):
                                    if col < self.p:
                                        partials[
                                            worker * self.slot_rows + down_row + row,
                                            reduction_group * self.partial_stride + col,
                                        ] = value
                                else:
                                    self._store_projection(
                                        value,
                                        col,
                                        row,
                                        activation,
                                        inject,
                                        x,
                                        worker,
                                        down_row,
                                        down_token,
                                    )
                    if cutlass.const_expr(self.reduction_groups > 1):
                        # All CTA stores precede arrival1. The acq_rel chain
                        # and acquire wait make every K partial visible to reducers.
                        _epilogue_barrier(self.down_merge_threads, down_barrier_id)
                        partial_generation = partial_generation_storage[0]
                        cohort_size = self.reduction_groups * self.split_k
                        counter_base = (partial_generation - 1) * (2 * cohort_size)
                        counter_index = (
                            partial_epoch_base
                            + worker * self.group_clusters
                            + group_cluster
                        )
                        if warp == 0:
                            if counter_leader:
                                with cute.arch.elect_one():
                                    _arrive_counter_step(
                                        epochs.iterator + counter_index,
                                        self.down_counter_step,
                                    )
                            self._wait_epochs(
                                epochs, counter_index, 1, counter_base + cohort_size
                            )
                        _epilogue_barrier(self.down_merge_threads, down_barrier_id)
                        # Groups consume disjoint ranges of the owner output.
                        # Each projection value is scaled/activated once.
                        value = cutlass.Float32(0)
                        items_per_group = (
                            self.down_n * self.owner_cols // self.down_merge_groups
                        )
                        merge_tid = epi_tid
                        if cutlass.const_expr(self.local_pair_reduce):
                            merge_tid = pipeline * self.epilogue_threads + epi_tid
                        if cutlass.const_expr(self.vector_down_merge):
                            for part in cutlass.range_constexpr(
                                (items_per_group + 4 * self.down_merge_threads - 1)
                                // (4 * self.down_merge_threads)
                            ):
                                item = (merge_tid + part * self.down_merge_threads) * 4
                                output_item = partial_group * items_per_group + item
                                row = output_item // self.owner_cols
                                col = (
                                    projection * self.down_m
                                    + split_rank * self.owner_cols
                                    + output_item % self.owner_cols
                                )
                                if item < items_per_group and col + 3 < self.p:
                                    partial_row = cutlass.Int64(
                                        worker * self.slot_rows + down_row + row
                                    )
                                    partial_ptr = (
                                        partials.iterator
                                        + partial_row * partials.stride[0]
                                        + col
                                    )
                                    initial = _load_partial_fp32x4(partial_ptr)
                                    sums = cute.make_rmem_tensor((4,), cutlass.Float32)
                                    for lane in cutlass.range_constexpr(4):
                                        sums[lane] = initial[lane]
                                    for peer_group in cutlass.range_constexpr(
                                        1, self.down_merge_groups
                                    ):
                                        peer_values = _load_partial_fp32x4(
                                            partial_ptr
                                            + peer_group * self.partial_stride
                                        )
                                        for lane in cutlass.range_constexpr(4):
                                            sums[lane] = sums[lane] + peer_values[lane]
                                    if cutlass.const_expr(self.packet_activation_store):
                                        if col < self.ps:
                                            self._store_projection_packet(
                                                sums,
                                                col,
                                                row,
                                                activation,
                                                worker,
                                                down_row,
                                            )
                                        else:
                                            for lane in cutlass.range_constexpr(4):
                                                self._store_projection(
                                                    sums[lane],
                                                    col + lane,
                                                    row,
                                                    activation,
                                                    inject,
                                                    x,
                                                    worker,
                                                    down_row,
                                                    down_token,
                                                )
                                    else:
                                        for lane in cutlass.range_constexpr(4):
                                            self._store_projection(
                                                sums[lane],
                                                col + lane,
                                                row,
                                                activation,
                                                inject,
                                                x,
                                                worker,
                                                down_row,
                                                down_token,
                                            )
                        else:
                            for part in cutlass.range_constexpr(
                                (items_per_group + self.down_merge_threads - 1)
                                // self.down_merge_threads
                            ):
                                item = merge_tid + part * self.down_merge_threads
                                output_item = partial_group * items_per_group + item
                                row = output_item // self.owner_cols
                                col = (
                                    projection * self.down_m
                                    + split_rank * self.owner_cols
                                    + output_item % self.owner_cols
                                )
                                if item < items_per_group and col < self.p:
                                    partial_row = cutlass.Int64(
                                        worker * self.slot_rows + down_row + row
                                    )
                                    partial_ptr = (
                                        partials.iterator
                                        + partial_row * partials.stride[0]
                                        + col
                                    )
                                    value = _load_partial_fp32(partial_ptr)
                                    for peer_group in cutlass.range_constexpr(
                                        1, self.down_merge_groups
                                    ):
                                        value = value + _load_partial_fp32(
                                            partial_ptr
                                            + peer_group * self.partial_stride
                                        )
                                    self._store_projection(
                                        value,
                                        col,
                                        row,
                                        activation,
                                        inject,
                                        x,
                                        worker,
                                        down_row,
                                        down_token,
                                    )
                        # Arrival2 releases all partial readers and activation stores.
                        # Never reset the counter: a fast member may already be
                        # sampling the next projection's monotonic threshold.
                        _publish_activation_proxy()
                        _epilogue_barrier(self.down_merge_threads, down_barrier_id)
                        if warp == 0:
                            if cutlass.const_expr(
                                self.up_clusters > 0
                                and projection_step == self.projection_tiles - 1
                            ):
                                if group_cluster == 0:
                                    # Up-only members sample cohort0's old generation
                                    # before its READY can advance.
                                    self._wait_epochs(
                                        epochs,
                                        start_base + worker * self.up_clusters,
                                        self.up_clusters,
                                        (partial_generation - 1)
                                        // self.projection_tiles
                                        + 1,
                                    )
                            if counter_leader:
                                with cute.arch.elect_one():
                                    previous = _arrive_counter_step(
                                        epochs.iterator + counter_index,
                                        self.down_counter_step,
                                    )
                                    if cutlass.const_expr(
                                        projection_step == self.projection_tiles - 1
                                    ):
                                        if (
                                            previous
                                            == counter_base
                                            + 2 * cohort_size
                                            - self.down_counter_step
                                        ):
                                            _store_epoch_release(
                                                epochs.iterator
                                                + worker * self.clusters
                                                + group_cluster,
                                                (partial_generation - 1)
                                                // self.projection_tiles
                                                + 1,
                                            )
                            if cutlass.const_expr(
                                not (
                                    self.local_pair_reduce and self.stream_up_activation
                                )
                            ):
                                self._wait_epochs(
                                    epochs,
                                    counter_index,
                                    1,
                                    counter_base + 2 * cohort_size,
                                )
                        _epilogue_barrier(self.down_merge_threads, down_barrier_id)
                    # The complete epilogue acknowledgement retires local Down
                    # readers before Up can reuse their operand/TMEM storage.
                    tcgen05_fence("before_thread_sync")
                    cute.arch.mbarrier_arrive(epi_done)
                if cutlass.const_expr(projection_step == self.projection_tiles - 1):
                    epi_tid = tid
                    cute.arch.mbarrier_wait(projection_ready, tile_phase)
                    if cutlass.const_expr(self.pin_up_activation):
                        if warp == 0 and pid * self.up_output_parts < self.hidden_tiles:
                            self._load_activation(
                                activation,
                                sa,
                                up_activation_ready,
                                up_k_empty,
                                up_mma,
                                worker,
                                tile_iteration,
                                epochs,
                                cutlass.Int64(0),
                                k_offset,
                            )
                    for up_step in cutlass.range_constexpr(up_steps):
                        hidden_id, up_id = self._up_tile(pid, up_step)
                        if hidden_id < self.hidden_tiles:
                            up_iteration = tile_iteration * cta_up_steps + up_step
                            up_phase = up_iteration % 2
                            if cutlass.const_expr(up_step > 0):
                                _epilogue_barrier(
                                    self.epilogue_threads,
                                    cluster % self.pipelines_per_cta + 1,
                                )
                            if cutlass.const_expr(not self.pin_up_activation):
                                if warp == 0:
                                    if cutlass.const_expr(up_step > 0):
                                        cute.arch.mbarrier_wait(
                                            up_empty, (up_iteration - 1) % 2
                                        )
                                    activation_generation = cutlass.Int64(0)
                                    if cutlass.const_expr(self.stream_up_activation):
                                        activation_generation = (
                                            partial_generation_storage[0]
                                        )
                                    self._load_activation(
                                        activation,
                                        sa,
                                        act_full,
                                        act_empty,
                                        up_mma,
                                        worker,
                                        up_iteration,
                                        epochs,
                                        activation_generation,
                                        k_offset,
                                    )
                            x_values = cute.make_tensor(
                                x.iterator,
                                cute.make_layout(
                                    (x.shape[0], self.wide), stride=(self.wide, 1)
                                ),
                            )
                            if cutlass.const_expr(self.wide_token_up):
                                self._token_branch_epilogue(
                                    tmem_base,
                                    up_acc_layout,
                                    activation,
                                    x_values,
                                    out,
                                    token,
                                    hidden_id,
                                    epi_tid,
                                    (
                                        token_ready
                                        if cutlass.const_expr(
                                            self.up_token_tail_overlap
                                        )
                                        else up_done
                                    ),
                                    up_phase,
                                    epi_done,
                                )
                            else:
                                for output_part in cutlass.range_constexpr(
                                    self.up_output_parts
                                ):
                                    part_hidden_id = hidden_id + output_part
                                    part_up_id = up_id + output_part
                                    if cutlass.const_expr(output_part > 0):
                                        _epilogue_barrier(
                                            self.epilogue_threads,
                                            cluster % self.pipelines_per_cta + 1,
                                        )
                                    for read_slice in cutlass.range_constexpr(
                                        self.up_read_slices
                                    ):
                                        gates, up_coords, x_regs = (
                                            self._read_up_with_prefetched_x(
                                                tmem_base,
                                                up_acc_layout,
                                                activation,
                                                epi_tid,
                                                up_done,
                                                x_values,
                                                part_up_id,
                                                token,
                                                up_phase,
                                                read_slice,
                                                output_part,
                                            )
                                        )
                                        # All epilogue readers complete tcgen05.wait::ld before
                                        # releasing this phase. Gate/output no longer use TMEM.
                                        if cutlass.const_expr(
                                            read_slice + 1 == self.up_read_slices
                                            and output_part + 1 == self.up_output_parts
                                        ):
                                            tcgen05_fence("before_thread_sync")
                                            cute.arch.mbarrier_arrive(epi_done)
                                        if cutlass.const_expr(self.alias_gate):
                                            # Generic gate stores now reuse the retired
                                            # async-proxy MMA operand arena.
                                            cute.arch.fence_proxy(
                                                "async.shared", space="cta"
                                            )
                                        if cutlass.const_expr(read_slice > 0):
                                            # Finish every previous fragment's gate read
                                            # before any warp overwrites the shared gate.
                                            _epilogue_barrier(
                                                self.epilogue_threads,
                                                cluster % self.pipelines_per_cta + 1,
                                            )
                                        for i in cutlass.range_constexpr(
                                            cute.size(gates)
                                        ):
                                            if cutlass.const_expr(self.token_major_up):
                                                row, col = up_coords[i]
                                                if row < min(
                                                    self.up_token_rows,
                                                    cutlass.Int32(x.shape[0])
                                                    - token * self.n,
                                                ):
                                                    gate[col, row] = x_regs[i].to(
                                                        cutlass.Float32
                                                    ) * self._up_gate_sigmoid(gates[i])
                                            else:
                                                gate[
                                                    up_coords[i][0],
                                                    up_coords[i][1]
                                                    - read_slice
                                                    * (self.n // self.up_read_slices),
                                                ] = x_regs[i].to(
                                                    cutlass.Float32
                                                ) * self._up_gate_sigmoid(
                                                    gates[i]
                                                )
                                        _epilogue_barrier(
                                            self.epilogue_threads,
                                            cluster % self.pipelines_per_cta + 1,
                                        )
                                        for part in cutlass.range_constexpr(
                                            self.final_tile
                                            * (
                                                self.up_token_rows
                                                // self.up_read_slices
                                            )
                                            // self.epilogue_threads
                                        ):
                                            elem = (
                                                epi_tid + part * self.epilogue_threads
                                            )
                                            j = elem % self.final_tile
                                            row = (
                                                elem // self.final_tile
                                                + read_slice
                                                * (self.n // self.up_read_slices)
                                            )
                                            value = cutlass.Float32(0)
                                            if cutlass.const_expr(
                                                up_step % self.branch_tiles > 0
                                            ):
                                                value = mix_accum[j, row]
                                            for branch in cutlass.range_constexpr(
                                                self.branch_tile
                                            ):
                                                value = (
                                                    value
                                                    + gate[
                                                        branch * self.final_tile + j,
                                                        row
                                                        - read_slice
                                                        * (
                                                            self.n
                                                            // self.up_read_slices
                                                        ),
                                                    ]
                                                )
                                            if cutlass.const_expr(
                                                up_step % self.branch_tiles + 1
                                                < self.branch_tiles
                                            ):
                                                mix_accum[j, row] = value
                                            elif cutlass.const_expr(self.full_tiles):
                                                out[
                                                    row + token * self.n,
                                                    part_hidden_id * self.final_tile
                                                    + j,
                                                ] = (value * (1.0 / self.hc_count)).to(
                                                    dtype
                                                )
                                            else:
                                                _store_tail(
                                                    out.iterator
                                                    + (
                                                        row
                                                        + cutlass.Int64(token) * self.n
                                                    )
                                                    * self.hidden_size
                                                    + part_hidden_id * self.final_tile
                                                    + j,
                                                    (value * (1.0 / self.hc_count)).to(
                                                        dtype
                                                    ),
                                                    row
                                                    < cutlass.Int32(x.shape[0])
                                                    - token * self.n,
                                                )
            if cutlass.const_expr(projection_step + 1 < self.projection_tiles):
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
        if cutlass.const_expr(
            self.rounds * self.batch_tiles * self.token_tile_batches > 1
        ):
            # Every local epilogue is finished before a worker can recycle its
            # activation slot. Publish CONSUMED separately from READY.
            if cutlass.const_expr(self.alias_gate):
                # Publish gate accesses before another round's TMA overwrites
                # the same shared arena through the async proxy.
                cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            if warp == self.mma_warp:
                with cute.arch.elect_one():
                    _cluster_arrive(consumed)
                    if split_rank == 0:
                        _cluster_wait(consumed, tile_phase)
                        _store_epoch_release(
                            epochs.iterator
                            + epoch_half
                            + worker * self.grid_clusters
                            + cluster,
                            generation,
                        )
                self._wait_epochs(
                    epochs,
                    epoch_half + worker * self.grid_clusters,
                    self.grid_clusters,
                    generation,
                )
            cute.arch.sync_threads()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
