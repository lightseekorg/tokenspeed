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

"""Expert routing, grouped GateUp/Down and merge routines for the layer."""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Uint32
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.engine._plan import (
    RoutineDecl,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._expert_gate_v2_kernel import (
    _make_operands,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._expert_grouped_entry import (
    _DOWN_SCALE_K,
    _GATE_SCALE_K,
    _OUTPUT_TILES_MAX,
    GROUPS,
    HIDDEN,
    INTERMEDIATE,
    TOKENS,
    LongcatExpertGroupedEntry,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._primitives import (
    atomic_add_acq_rel_gpu_u32,
    red_add_release_gpu_u32,
)

TOPK = 12
SLOTS = TOKENS * TOPK
ROUTE_THREADS = 128
ROUTE_IDS = 1024  # expert ids the route's first-slot table covers (> 896)
EPILOGUE_THREADS = 256
ROUTE_BAR = 8
EPILOGUE_BAR = 9
DOWN_DONE_BAR = 7
# SMEM reservations checked against the traced structs in ``host_prepare``.
# Down reserves the full SM; its stage heuristic may use less.
GATE_SMEM = 220160
DOWN_SMEM = 232448


@dataclass(frozen=True)
class MoeShape:
    """What the MoE program is specialized on.

    Args:
        experts: routed experts on this rank (the weights' leading dim).
        gate_clusters, down_clusters, group_splits: the expert kernel's
            ``<gate>x<down>x<splits>`` configuration.
        identity_scale: factor on the identity-expert term (``1 / tp``).
    """

    experts: int
    gate_clusters: int = 16
    down_clusters: int = 8
    group_splits: int = 2
    identity_scale: float = 1.0 / 8

    @property
    def gate_ctas(self) -> int:
        return 2 * self.gate_clusters

    @property
    def down_ctas(self) -> int:
        return 2 * self.down_clusters


def make_entry(shape: MoeShape) -> LongcatExpertGroupedEntry:
    """The expert kernel whose routines the program uses."""
    if shape.experts != 768:
        raise ValueError("the layer router requires 768 routed experts")
    if (shape.gate_clusters, shape.down_clusters, shape.group_splits) != (16, 8, 2):
        raise ValueError("the layer expert topology is 16x8x2")
    return LongcatExpertGroupedEntry(
        gate_clusters=shape.gate_clusters,
        down_clusters=shape.down_clusters,
        down_group_splits=shape.group_splits,
    )


def _ordered(ptr, shape):
    return cute.make_tensor(ptr, cute.make_ordered_layout(shape, order=(1, 0, 2)))


def _fp4():
    return cutlass.Float4E2M1FN


def _sf():
    return cutlass.Float8E4M3FN


def _route_adapter(shape: MoeShape):
    """Top-k -> distinct expert groups (``route_groups``), one CTA.

    A group per distinct live expert, numbered by the slot of its first
    occurrence.  Leaders come from an SMEM ``atomic_min`` over the expert id,
    group numbers from a warp ballot.  Every live slot then adds its weight
    to its (group, token) cell with an SMEM atomic: a router never picks an
    expert twice for one token, so a cell gets one term and the sum is the
    weight exactly.
    """

    @cute.jit
    def invoke(ctx):
        ids = cute.make_tensor(
            ctx.ptr("router_ids", Int32),
            cute.make_layout((TOKENS, TOPK), stride=(TOPK, 1)),
        )
        weights = cute.make_tensor(
            ctx.ptr("router_weights", Float32),
            cute.make_layout((TOKENS, TOPK), stride=(TOPK, 1)),
        )
        group_ids = cute.make_tensor(
            ctx.ptr("moe_group_ids", Int32), cute.make_layout((GROUPS,))
        )
        base = cute.recast_ptr(ctx.smem, dtype=Int32)
        s_warp = cute.make_tensor(base, cute.make_layout((4,)))
        s_first = cute.make_tensor(base + 4, cute.make_layout((ROUTE_IDS,)))
        s_table = cute.make_tensor(
            cute.recast_ptr(base + 4 + ROUTE_IDS, dtype=Float32),
            cute.make_layout((GROUPS * TOKENS,)),
        )
        table_flat = cute.make_tensor(
            ctx.ptr("moe_group_table", Float32), cute.make_layout((GROUPS * TOKENS,))
        )
        done = ctx.ticket("moe_groups")
        tidx = ctx.tidx
        threads = ctx.threads
        for e in cutlass.range(tidx, ROUTE_IDS, threads):
            s_first[e] = Int32(SLOTS)
        for e in cutlass.range(tidx, GROUPS * TOKENS, threads):
            s_table[e] = Float32(0.0)
        # A slot is live when its route weight is non-zero (identity-expert
        # slots arrive as id 0 / weight 0).
        my_id = Int32(-1)
        my_w = Float32(0.0)
        if tidx < SLOTS:
            my_w = Float32(weights[tidx // TOPK, tidx % TOPK])
            if my_w != Float32(0.0):
                my_id = Int32(ids[tidx // TOPK, tidx % TOPK])
        cute.arch.barrier(barrier_id=ROUTE_BAR, number_of_threads=threads)
        if my_id >= 0:
            cute.arch.atomic_min(s_first.iterator + my_id, tidx, scope="cta")
        cute.arch.barrier(barrier_id=ROUTE_BAR, number_of_threads=threads)
        # Leader: the first live slot carrying its id.
        leader = cutlass.Boolean(False)
        if my_id >= 0:
            leader = s_first[my_id] == tidx
        warp = tidx // 32
        mask = cute.arch.vote_ballot_sync(leader)
        if tidx % 32 == 0:
            s_warp[warp] = Int32(cute.arch.popc(Uint32(mask)))
        cute.arch.barrier(barrier_id=ROUTE_BAR, number_of_threads=threads)
        g = Int32(cute.arch.popc(Uint32(mask) & cute.arch.lanemask_lt()))
        count = Int32(0)
        for w_ in cutlass.range_constexpr(ROUTE_THREADS // 32):
            if w_ < warp:
                g = g + s_warp[w_]
            count = count + s_warp[w_]
        if tidx < GROUPS and tidx >= count:
            group_ids[tidx] = Int32(-1)
        if leader:
            group_ids[g] = my_id
            # Every read of the leader slot is behind the barrier above; the
            # entry now holds the expert's group.
            s_first[my_id] = g
        cute.arch.barrier(barrier_id=ROUTE_BAR, number_of_threads=threads)
        if my_id >= 0:
            cute.arch.atomic_add(
                s_table.iterator + s_first[my_id] * Int32(TOKENS) + tidx // TOPK,
                my_w,
                scope="cta",
            )
        cute.arch.barrier(barrier_id=ROUTE_BAR, number_of_threads=threads)
        for e in cutlass.range(tidx, count * Int32(TOKENS), threads):
            table_flat[e] = s_table[e]
        cute.arch.barrier(barrier_id=ROUTE_BAR, number_of_threads=threads)
        if tidx == 0:
            red_add_release_gpu_u32(done, Uint32(1))

    return invoke


def _gate_prepare(entry: LongcatExpertGroupedEntry, experts: int):
    gate = entry.gate_v2_up

    @cute.jit
    def prepare(ctx):
        weight, weight_sf = _make_operands(
            ctx.ptr("moe_gate_weight", _fp4()),
            ctx.ptr("moe_gate_scale", _sf()),
            experts,
        )
        prepared = gate.prepare(
            _ordered(ctx.ptr("moe_x_fp4", _fp4()), (TOKENS, HIDDEN, 1)),
            cute.make_tensor(
                ctx.ptr("moe_x_scale", _sf()), cute.make_layout((128 * _GATE_SCALE_K,))
            ),
            weight,
            weight_sf,
        )
        if cutlass.const_expr(gate.shared_storage.size_in_bytes() != GATE_SMEM):
            raise ValueError(
                f"MoE gate SMEM {gate.shared_storage.size_in_bytes()} != declared "
                f"{GATE_SMEM}"
            )
        return prepared

    return prepare


def _gate_adapter(entry: LongcatExpertGroupedEntry, experts: int):
    """One GateUp owner per CTA (warps 0-5)."""
    gate = entry.gate_v2_up

    @cute.jit
    def invoke(ctx):
        gate.gate_v2_device_routine(
            *ctx.prepared,
            cute.make_tensor(
                ctx.ptr("moe_group_ids", Int32), cute.make_layout((GROUPS,))
            ),
            cute.make_tensor(
                ctx.ptr("moe_gate_alpha", Float32), cute.make_layout((experts,))
            ),
            cute.make_tensor(
                ctx.ptr("moe_up_alpha", Float32), cute.make_layout((experts,))
            ),
            cute.make_tensor(ctx.ptr("moe_out_scale", Float32), cute.make_layout((1,))),
            cute.make_tensor(
                ctx.ptr("moe_gate_out", cutlass.Uint8),
                cute.make_layout((GROUPS * TOKENS * (INTERMEDIATE // 2),)),
            ),
            cute.make_tensor(
                ctx.ptr("moe_gate_out_scale", _sf()), cute.make_layout((GROUPS * 2048,))
            ),
            cute.make_tensor(
                ctx.ptr("moe_gate_arrivals", Uint32), cute.make_layout((GROUPS,))
            ),
            ctx.smem,
            ctx.tile,
        )

    return invoke


def _down_prepare(entry: LongcatExpertGroupedEntry, experts: int):
    down = entry.down

    @cute.jit
    def prepare(ctx):
        prepared = down.prepare_k_stage_device_routine(
            _ordered(ctx.ptr("moe_gate_out", _fp4()), (TOKENS, INTERMEDIATE, GROUPS)),
            cute.make_tensor(
                ctx.ptr("moe_gate_out_scale", _sf()),
                cute.make_ordered_layout(
                    (32, 4, 1, 4, _DOWN_SCALE_K // 4, GROUPS), order=(2, 1, 4, 0, 3, 5)
                ),
            ),
            _ordered(
                ctx.ptr("moe_down_weight", _fp4()), (HIDDEN, INTERMEDIATE, experts)
            ),
            cute.make_tensor(
                ctx.ptr("moe_down_scale", _sf()),
                cute.make_ordered_layout(
                    (32, 4, HIDDEN // 128, 4, _DOWN_SCALE_K // 4, experts),
                    order=(2, 1, 4, 0, 3, 5),
                ),
            ),
        )
        if cutlass.const_expr(down.shared_storage.size_in_bytes() > DOWN_SMEM):
            raise ValueError(
                f"MoE down SMEM {down.shared_storage.size_in_bytes()} exceeds reservation "
                f"{DOWN_SMEM}"
            )
        return prepared

    return prepare


def _down_adapter(entry: LongcatExpertGroupedEntry, experts: int, shape: MoeShape):
    """One hidden-slab Down owner per CTA; releases ``moe_down`` on return."""
    down = entry.down

    @cute.jit
    def invoke(ctx):
        done = ctx.ticket("moe_down")
        threads = ctx.threads
        down.down_hidden_slab_device_routine(
            *ctx.prepared,
            cute.make_tensor(
                ctx.ptr("moe_down_alpha", Float32), cute.make_layout((experts,))
            ),
            cute.make_tensor(
                ctx.ptr("moe_gate_arrivals", Uint32), cute.make_layout((GROUPS,))
            ),
            cute.make_tensor(
                ctx.ptr("moe_route_arrivals", Uint32),
                cute.make_layout((GROUPS * _OUTPUT_TILES_MAX,)),
            ),
            cute.make_tensor(
                ctx.ptr("moe_group_ids", Int32), cute.make_layout((GROUPS,))
            ),
            cute.make_tensor(
                ctx.ptr("moe_group_table", Float32),
                cute.make_layout((GROUPS * TOKENS,)),
            ),
            cute.make_tensor(
                ctx.ptr("moe_acc", Float32), cute.make_layout((TOKENS * HIDDEN,))
            ),
            cute.make_tensor(ctx.ptr("moe_group_done", Uint32), cute.make_layout((1,))),
            ctx.smem,
            ctx.tile,
            shape.down_ctas,
            GROUPS,
            shape.group_splits,
        )
        tidx = ctx.tidx
        cute.arch.barrier(barrier_id=DOWN_DONE_BAR, number_of_threads=threads)
        if tidx == 0:
            red_add_release_gpu_u32(done, Uint32(1))

    return invoke


def _epilogue_parts(shape: MoeShape) -> int:
    """Slices per token row: as many as the gate CTAs allow, up to four."""
    parts = min(4, shape.gate_ctas // TOKENS)
    if parts < 1:
        raise ValueError(f"the epilogue needs >= {TOKENS} gate CTAs")
    return parts


def _epilogue_adapter(shape: MoeShape):
    """``moe_local = bf16(acc + hidden * zero_weight * scale)``, one row slice.

    A tile is one ``1 / parts`` slice of a token's row. All loads precede
    the first store to avoid serializing potentially aliasing accesses.
    Re-zeroes the accumulator slice
    for the next step; the last of the ``TOKENS * parts`` CTAs bumps
    ``moe_ready`` (release) and resets the arrival counter.
    """
    scale = shape.identity_scale
    parts = _epilogue_parts(shape)
    per_thread = HIDDEN // (parts * EPILOGUE_THREADS)

    @cute.jit
    def invoke(ctx):
        rows = cute.make_layout((TOKENS, HIDDEN), stride=(HIDDEN, 1))
        acc = cute.make_tensor(ctx.ptr("moe_acc", Float32), rows)
        hidden = cute.make_tensor(ctx.ptr("moe_hidden", BFloat16), rows)
        out = cute.make_tensor(ctx.ptr("moe_local", BFloat16), rows)
        zero = cute.make_tensor(
            ctx.ptr("router_zero", Float32), cute.make_layout((TOKENS,))
        )
        arrivals = cute.make_tensor(
            ctx.ptr("moe_epi_done", Uint32), cute.make_layout((1,))
        )
        ready = cute.make_tensor(ctx.ptr("moe_ready", Uint32), cute.make_layout((1,)))
        token = ctx.tile // parts
        base = (ctx.tile % parts) * (HIDDEN // parts)
        tidx = ctx.tidx
        threads = ctx.threads
        zw = Float32(zero[token]) * Float32(scale)
        r_acc = cute.make_rmem_tensor((per_thread,), Float32)
        r_hid = cute.make_rmem_tensor((per_thread,), Float32)
        for i in cutlass.range_constexpr(per_thread):
            h = base + tidx + i * EPILOGUE_THREADS
            r_acc[i] = Float32(acc[token, h])
            r_hid[i] = Float32(hidden[token, h])
        for i in cutlass.range_constexpr(per_thread):
            h = base + tidx + i * EPILOGUE_THREADS
            out[token, h] = BFloat16(r_acc[i] + r_hid[i] * zw)
            acc[token, h] = Float32(0.0)
        cute.arch.barrier(barrier_id=EPILOGUE_BAR, number_of_threads=threads)
        if tidx == 0:
            prev = atomic_add_acq_rel_gpu_u32(arrivals.iterator, Uint32(1))
            if prev == Uint32(TOKENS * parts - 1):
                arrivals[0] = Uint32(0)
                red_add_release_gpu_u32(ready.iterator, Uint32(1))

    return invoke


OPERANDS = (
    "moe_x_fp4",
    "moe_x_scale",
    "moe_hidden",
    "router_ids",
    "router_weights",
    "router_zero",
    "moe_gate_weight",
    "moe_gate_scale",
    "moe_down_weight",
    "moe_down_scale",
    "moe_gate_alpha",
    "moe_up_alpha",
    "moe_out_scale",
    "moe_down_alpha",
    "moe_group_ids",
    "moe_group_table",
    "moe_gate_out",
    "moe_gate_out_scale",
    "moe_gate_arrivals",
    "moe_route_arrivals",
    "moe_acc",
    "moe_group_done",
    "moe_epi_done",
    "moe_local",
    "moe_ready",
)


def moe_routines(shape: MoeShape) -> tuple[RoutineDecl, ...]:
    """Route, GateUp, Down and epilogue declarations for the two MoE roles."""
    entry = make_entry(shape)
    experts = shape.experts
    route_decl = RoutineDecl(
        "moe_route",
        threads=ROUTE_THREADS,
        smem_bytes=(4 + ROUTE_IDS + GROUPS * TOKENS) * 4,
        named_barriers=(ROUTE_BAR,),
        invoke=_route_adapter(shape),
    )
    gate_decl = RoutineDecl(
        "moe_gate",
        threads=192,
        smem_bytes=GATE_SMEM,
        smem_align=1024,
        tmem_cols=512,
        named_barriers=(1, 2, 3),
        host_prepare=_gate_prepare(entry, experts),
        invoke=_gate_adapter(entry, experts),
    )
    down_decl = RoutineDecl(
        "moe_down",
        threads=192,
        smem_bytes=DOWN_SMEM,
        smem_align=1024,
        tmem_cols=512,
        named_barriers=(0, 1, 2, 3, DOWN_DONE_BAR),
        host_prepare=_down_prepare(entry, experts),
        invoke=_down_adapter(entry, experts, shape),
    )
    epilogue_decl = RoutineDecl(
        "moe_epilogue",
        threads=EPILOGUE_THREADS,
        smem_bytes=128,
        named_barriers=(EPILOGUE_BAR,),
        invoke=_epilogue_adapter(shape),
    )
    return route_decl, gate_decl, down_decl, epilogue_decl


def make_moe_buffers(device) -> dict[str, torch.Tensor]:
    """The program's intermediates, counters and outputs (zeroed).

    The caller adds the step inputs (``moe_x_fp4`` / ``moe_x_scale`` -- the
    NVFP4 activation --, ``moe_hidden``, the routing ``router_ids`` /
    ``router_weights`` / ``router_zero``), the gate/down weights and block
    scales, the three alphas plus ``moe_out_scale``
    and ``moe_ready``.
    """
    u32 = dict(device=device, dtype=torch.uint32)
    return {
        "moe_group_ids": torch.full((GROUPS,), -1, device=device, dtype=torch.int32),
        "moe_group_table": torch.zeros(
            GROUPS, TOKENS, device=device, dtype=torch.float32
        ),
        "moe_gate_out": torch.zeros(
            (GROUPS, TOKENS, INTERMEDIATE // 2), dtype=torch.uint8, device=device
        ),
        "moe_gate_out_scale": torch.zeros(
            (GROUPS, 128, _DOWN_SCALE_K), dtype=torch.float8_e4m3fn, device=device
        ),
        "moe_gate_arrivals": torch.zeros((GROUPS,), **u32),
        "moe_route_arrivals": torch.zeros((GROUPS * _OUTPUT_TILES_MAX,), **u32),
        "moe_acc": torch.zeros((TOKENS, HIDDEN), dtype=torch.float32, device=device),
        "moe_group_done": torch.zeros((1,), **u32),
        "moe_epi_done": torch.zeros(4, **u32)[:1],
        "moe_local": torch.zeros(TOKENS, HIDDEN, device=device, dtype=torch.bfloat16),
    }


__all__ = [
    "MoeShape",
    "OPERANDS",
    "make_entry",
    "make_moe_buffers",
    "moe_routines",
]
