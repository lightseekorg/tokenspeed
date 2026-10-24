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

"""Dense MLP and router routines bound to the layer's operands and tickets."""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.attention._o_proj_collective import (
    LongcatOProjCollective,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.engine._plan import (
    Prefetch,
    RoutineDecl,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.engine.attention_worker import (
    COMM_THREADS,
    NORM_BAR,
    clear_adapter,
    norm_adapter,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._collective_primitives import (
    invalidate_mbarrier,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._downreduce_kernel import (
    LongcatDownReduceKernel,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._expert_gate_v2_kernel import (
    LongcatExpertGateV2Kernel,
    _make_operands,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._expert_grouped_entry import (
    GATE_V2_AB_STAGES,
    GATE_V2_K_BLOCKS_PER_STAGE,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._primitives import (
    load_acquire_gpu_u32,
    red_add_release_gpu_u32,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.routines._router_tma import (
    LongcatRouterTma,
)

TOKENS = 8
HIDDEN = 8192
INTER = 1536  # per-rank dense intermediate
EXPERTS = 896  # router logits: routed experts plus the zero experts
TOPK = 12
STAGE_K = 256  # GateUp -> Down hand-off granularity
GATE_CTAS = 24
DOWN_CTAS = 64
# The router splits K four ways over 28 CTAs (7 tiles of 128 experts).
# Top-k sums the four partial-logit slices in order.
ROUTER_K_SPLITS = 4
ROUTER_CTAS = 7 * ROUTER_K_SPLITS
# Per-routine SMEM reservations, checked against the traced SharedStorage.
# Down reserves the full SM; its stage heuristic may use less.
GATE_V2_SMEM = 220160  # the MoE's gate v2 configuration, the same struct
DOWN_SMEM = 232448
ROUTER_SMEM = 224896  # one-CTA MMA: B per stage is not split
DOWN_DONE_BAR = 7  # the adapter's closing CTA barrier; the routine uses 0-5
# Down CTAs past the reduce-norm's consumers (one two-CTA cluster per token):
# they re-arm the retired transport slot while the consumers reduce.
CLEARERS = DOWN_CTAS - TOKENS * 2
ROUTER_TOPK_THREADS = 128


@dataclass(frozen=True)
class MlpShape:
    """TP rank, normalization epsilon and dense branch (0 or 1)."""

    rank: int
    eps: float
    index: int


def mlp_names(index: int) -> dict[str, str]:
    """Return logical-to-layer operand names for dense branch ``index`` (0 or 1).

    Dictionary order is the branch's kernel operand order.
    """
    fields = (
        "x_fp4",
        "x_scale",
        "gate_alpha",
        "gate_quant",
        "act",
        "act_scale",
        "gate_arrivals",
        "gate_epochs",
        "down_w",
        "down_ws",
        "down_alpha",
        "down_scratch",
        "bases",
        "rotation",
        "residual",
        "post_gamma",
        "residual_out",
        "norm_out",
    )
    fields += ("gate_w", "gate_ws", "gate_ids")
    if index == 0:
        fields += (
            "router_x",
            "router_w",
            "router_logits",
            "router_bias",
            "router_ids",
            "router_weights",
            "router_zero",
        )
        names = {
            n: (
                n
                if n in ("bases", "rotation") or n.startswith("router_")
                else "mlp0_" + n
            )
            for n in fields
        }
        names.update(post_gamma="mlp0_gamma", residual_out="residual", norm_out="x")
    elif index == 1:
        fields += ("local_add", "local_add_ready", "local_add_gen")
        names = dict(zip(fields, fields))
        names.update(
            x_fp4="quant_out",
            x_scale="quant_scale",
            residual="residual_out",
            post_gamma="mlp_gamma",
            residual_out="mlp_residual_out",
            norm_out="mlp_norm_out",
            local_add="moe_local",
            local_add_ready="moe_ready",
        )
    else:
        raise ValueError(f"dense MLP index must be 0 or 1, got {index}")
    return names


def _fp4():
    return cutlass.Float4E2M1FN


def _sf():
    return cutlass.Float8E4M3FN


def _ordered(ptr, shape):
    return cute.make_tensor(ptr, cute.make_ordered_layout(shape, order=(1, 0, 2)))


def _gate_v2_kernel() -> LongcatExpertGateV2Kernel:
    """GateUp with the weights as the MMA M operand and the 8 tokens as N.

    The dense MLP is one group spanning the intermediate width. Output
    layout, scales and per-stage arrivals match the DownReduce contract.
    """
    return LongcatExpertGateV2Kernel(
        num_workers=GATE_CTAS,
        k_blocks_per_stage=GATE_V2_K_BLOCKS_PER_STAGE,
        ab_stages=GATE_V2_AB_STAGES,
        intermediate=INTER,
        groups=1,
        stage_tiles=STAGE_K // 64,
    )


def _gate_v2_prepare(gate: LongcatExpertGateV2Kernel, names: dict[str, str]):
    """Prepare GateUp weight and activation transport and shared-memory layouts."""

    @cute.jit
    def prepare(ctx):
        weight, weight_sf = _make_operands(
            ctx.ptr(names["gate_w"], _fp4()),
            ctx.ptr(names["gate_ws"], _sf()),
            1,
            2 * INTER,
        )
        prepared = gate.prepare(
            _ordered(ctx.ptr(names["x_fp4"], _fp4()), (TOKENS, HIDDEN, 1)),
            cute.make_tensor(
                ctx.ptr(names["x_scale"], _sf()), cute.make_layout((128 * 512,))
            ),
            weight,
            weight_sf,
        )
        if cutlass.const_expr(gate.shared_storage.size_in_bytes() != GATE_V2_SMEM):
            raise ValueError(
                f"GateUp v2 SMEM {gate.shared_storage.size_in_bytes()} != declared "
                f"{GATE_V2_SMEM}"
            )
        return prepared

    return prepare


@cute.jit
def _retire_pipeline_barriers(ctx, kernel: cutlass.Constexpr):
    # The routine has drained its pipelines and synchronized its threads.
    # End the barrier lifetime before the next phase repurposes the arena.
    storage = kernel.shared_storage(ctx.smem)
    tidx = ctx.tidx
    if tidx == 0:
        for i in cutlass.range_constexpr(kernel.num_ab_stage):
            invalidate_mbarrier(storage.ab_full_mbar_ptr.data_ptr() + i)
            invalidate_mbarrier(storage.ab_empty_mbar_ptr.data_ptr() + i)
        for i in cutlass.range_constexpr(kernel.num_acc_stage):
            invalidate_mbarrier(storage.acc_full_mbar_ptr.data_ptr() + i)
            invalidate_mbarrier(storage.acc_empty_mbar_ptr.data_ptr() + i)


def _gate_v2_adapter(gate: LongcatExpertGateV2Kernel, names: dict[str, str]):
    """One GateUp task (a 128-row weight tile, 64 outputs) per CTA."""

    @cute.jit
    def invoke(ctx):
        words = lambda name, n: cute.make_tensor(  # noqa: E731
            ctx.ptr(names[name], Int32), cute.make_layout((n,))
        )
        alpha = cute.make_tensor(
            ctx.ptr(names["gate_alpha"], Float32), cute.make_layout((1,))
        )
        gate.gate_v2_device_routine(
            *ctx.prepared,
            words("gate_ids", 1),
            alpha,  # one dequant scale for the gate and the up rows
            alpha,
            cute.make_tensor(
                ctx.ptr(names["gate_quant"], Float32), cute.make_layout((1,))
            ),
            cute.make_tensor(
                ctx.ptr(names["act"], cutlass.Uint8),
                cute.make_layout((TOKENS * INTER // 2,)),
            ),
            cute.make_tensor(
                ctx.ptr(names["act_scale"], _sf()), cute.make_layout((8 * INTER,))
            ),
            cute.make_tensor(
                ctx.ptr(names["gate_arrivals"], Uint32),
                cute.make_layout((INTER // STAGE_K,)),
            ),
            ctx.smem,
            ctx.tile,
            gate=ctx.gate() if ctx.gated else None,
        )
        _retire_pipeline_barriers(ctx, gate)

    return invoke


def _down_prepare(down: LongcatDownReduceKernel, names: dict[str, str]):
    """DownReduce's TMA atoms and its transport views."""

    @cute.jit
    def prepare(ctx):
        # The transport's uint32[9] control words (publish-only reads 0-4).
        controls = cute.make_tensor(
            ctx.ptr(names["rotation"], Uint32), cute.make_layout((9,))
        )
        bases = cute.make_tensor(ctx.ptr(names["bases"], Int64), cute.make_layout((8,)))
        args = down.prepare_device_routine(
            _ordered(ctx.ptr(names["down_w"], _fp4()), (HIDDEN, INTER, 1)),
            _ordered(ctx.ptr(names["act"], _fp4()), (TOKENS, INTER, 1)),
            cute.make_tensor(
                ctx.ptr(names["down_ws"], _sf()),
                cute.make_layout((HIDDEN * INTER // 16,)),
            ),
            cute.make_tensor(
                ctx.ptr(names["act_scale"], _sf()), cute.make_layout((12288,))
            ),
            cute.make_tensor(
                ctx.ptr(names["down_alpha"], Float32), cute.make_layout((1,))
            ),
            cute.make_tensor(
                ctx.ptr(names["down_scratch"], BFloat16),
                cute.make_layout((HIDDEN, TOKENS, 1), stride=(1, HIDDEN, 65536)),
            ),
            bases,
            controls,
        )
        if cutlass.const_expr(down.shared_storage.size_in_bytes() > DOWN_SMEM):
            raise ValueError(
                f"DownReduce SMEM {down.shared_storage.size_in_bytes()} exceeds reservation "
                f"{DOWN_SMEM}"
            )
        return args

    return prepare


def _down_adapter(
    down: LongcatDownReduceKernel,
    names: dict[str, str],
    local_add: bool,
    done: str,
    after: str | None,
):
    """One Down worker per CTA; releases ``down_done`` once it returns.

    ``after`` is a ticket (one arrival per generation) the worker waits for
    before it enters the routine, which reads the transport slot there: an
    earlier collective's rotation in the same kernel.
    """

    @cute.jit
    def invoke(ctx):
        if cutlass.const_expr(after is not None):
            word, gen, tidx = ctx.ticket(after), ctx.gen, ctx.tidx
            if tidx == 0:
                ready = load_acquire_gpu_u32(word)
                while (ready - gen).to(Int32) < Int32(0):
                    ready = load_acquire_gpu_u32(word)
            cute.arch.barrier(barrier_id=DOWN_DONE_BAR, number_of_threads=ctx.threads)
        la = la_ready = la_gen = None
        if cutlass.const_expr(local_add):
            la = cute.make_tensor(
                ctx.ptr(names["local_add"], BFloat16),
                cute.make_layout((TOKENS, HIDDEN), stride=(HIDDEN, 1)),
            )
            la_ready = cute.make_tensor(
                ctx.ptr(names["local_add_ready"], Uint32), cute.make_layout((1,))
            )
            la_gen = cute.make_tensor(
                ctx.ptr(names["local_add_gen"], Uint32), cute.make_layout((1,))
            )
        done_word = ctx.ticket(done)
        threads = ctx.threads
        down.downreduce_device_routine(
            *ctx.prepared,
            shared_storage_ptr=ctx.smem,
            logical_worker=ctx.tile,
            gate_arrivals=cute.make_tensor(
                ctx.ptr(names["gate_arrivals"], Uint32),
                cute.make_layout((INTER // STAGE_K,)),
            ),
            gate_epochs=cute.make_tensor(
                ctx.ptr(names["gate_epochs"], Uint32), cute.make_layout((DOWN_CTAS,))
            ),
            local_add=la,
            local_add_ready=la_ready,
            local_add_gen=la_gen,
        )
        # Every thread's read of ``local_add_gen`` (at routine entry) and
        # every published fragment precede the release.
        tidx = ctx.tidx
        cute.arch.barrier(barrier_id=DOWN_DONE_BAR, number_of_threads=threads)
        _retire_pipeline_barriers(ctx, down)
        if tidx == 0:
            red_add_release_gpu_u32(done_word, Uint32(1))

    return invoke


def _router_prepare(router: LongcatRouterTma):
    @cute.jit
    def prepare(ctx):
        args = router.prepare_device_routine(
            cute.make_tensor(
                ctx.ptr("router_x", BFloat16),
                cute.make_layout((TOKENS, HIDDEN), stride=(HIDDEN, 1)),
            ),
            cute.make_tensor(
                ctx.ptr("router_w", BFloat16),
                cute.make_layout((EXPERTS, HIDDEN), stride=(HIDDEN, 1)),
            ),
            cute.make_tensor(
                ctx.ptr("router_logits", Float32),
                cute.make_layout((TOKENS, EXPERTS), stride=(EXPERTS, 1)),
            ),
        )
        if cutlass.const_expr(router.smem_bytes != ROUTER_SMEM):
            raise ValueError(
                f"router SMEM {router.smem_bytes} != declared {ROUTER_SMEM}"
            )
        return args

    return prepare


def _topk_prepare():
    """The top-k phase reads only the logits and writes the routing: no TMA."""

    @cute.jit
    def prepare(ctx):
        return (
            cute.make_tensor(
                ctx.ptr("router_logits", Float32),
                cute.make_layout((EXPERTS, TOKENS), stride=(1, EXPERTS)),
            ),
            (
                cute.make_tensor(
                    ctx.ptr("router_bias", Float32), cute.make_layout((EXPERTS,))
                ),
                cute.make_tensor(
                    ctx.ptr("router_ids", Int32),
                    cute.make_layout((TOKENS, TOPK), stride=(TOPK, 1)),
                ),
                cute.make_tensor(
                    ctx.ptr("router_weights", Float32),
                    cute.make_layout((TOKENS, TOPK), stride=(TOPK, 1)),
                ),
                cute.make_tensor(
                    ctx.ptr("router_zero", Float32), cute.make_layout((TOKENS,))
                ),
            ),
        )

    return prepare


def _router_adapter(router: LongcatRouterTma):
    """One two-CTA pair's 256 experts of logits; releases ``logits``."""

    @cute.jit
    def invoke(ctx):
        done = ctx.ticket("mlp0_logits")
        tidx = ctx.tidx
        router.router_device_routine(*ctx.prepared, ctx.tile, ctx.pair_rank, ctx.smem)
        # The routine's closing ``sync_threads`` ordered the logits stores.
        if tidx == 0:
            # The drained pipeline barriers retire before the arena is reused.
            storage = router.storage(ctx.smem)
            for i in cutlass.range_constexpr(2 * router.stages):
                invalidate_mbarrier(storage.ab.data_ptr() + i)
            for i in cutlass.range_constexpr(2):
                invalidate_mbarrier(storage.acc.data_ptr() + i)
            red_add_release_gpu_u32(done, Uint32(1))

    return invoke


def _router_topk_adapter(router: LongcatRouterTma, done: str | None):
    """One token's top-12 from the global logits (four warps).

    With ``done``, releases that ticket once the token's routing is stored.
    In an ``issue_before_gate`` phase it first runs one store-free round at
    the entry to warm the instruction cache, then waits for the logits
    and runs the pass that stores the routing.
    """

    @cute.jit
    def invoke(ctx):
        out, routing = ctx.prepared
        tidx = ctx.tidx
        if cutlass.const_expr(ctx.gated):
            # Nothing of ``ctx`` inside the runtime loop: it would be carried.
            gates = ctx.gates()
            tile = ctx.tile
            smem = ctx.smem
            for p in cutlass.range(2, unroll=1):
                if p == 1:
                    if tidx == 0:
                        for word, target in gates:
                            seen = load_acquire_gpu_u32(word)
                            while (seen - target).to(Int32) < Int32(0):
                                seen = load_acquire_gpu_u32(word)
                    cute.arch.barrier(
                        barrier_id=DOWN_DONE_BAR, number_of_threads=ROUTER_TOPK_THREADS
                    )
                router.router_topk_device_routine(
                    out, routing, tile, tidx, smem, 1 + 11 * p, p == 1
                )
        else:
            router.router_topk_device_routine(out, routing, ctx.tile, tidx, ctx.smem)
        if cutlass.const_expr(done is not None):
            word = ctx.ticket(done)
            cute.arch.barrier(
                barrier_id=DOWN_DONE_BAR, number_of_threads=ROUTER_TOPK_THREADS
            )
            if tidx == 0:
                red_add_release_gpu_u32(word, Uint32(1))

    return invoke


def mlp_routines(shape: MlpShape) -> tuple[RoutineDecl, ...]:
    """GateUp, Down, norm and clear routines for one dense branch."""
    names = mlp_names(shape.index)
    first = shape.index == 0
    prefix = "mlp0_" if first else ""
    done = prefix + "down_done"
    rotated = "mlp0_rotated" if first else None
    down = LongcatDownReduceKernel(rank=shape.rank)
    collective = LongcatOProjCollective(
        indexer_rank=None,
        cluster_size=2,
        norm_eps=shape.eps,
    )
    gate = _gate_v2_kernel()
    return (
        RoutineDecl(
            prefix + "gate_up",
            threads=192,
            smem_bytes=GATE_V2_SMEM,
            smem_align=1024,
            tmem_cols=512,
            named_barriers=(1, 2, 3, DOWN_DONE_BAR),
            host_prepare=_gate_v2_prepare(gate, names),
            invoke=_gate_v2_adapter(gate, names),
            gates_itself=True,
        ),
        RoutineDecl(
            prefix + "downreduce",
            threads=384,
            smem_bytes=DOWN_SMEM,
            smem_align=1024,
            tmem_cols=512,
            named_barriers=(0, 1, 2, 3, 4, 5, DOWN_DONE_BAR),
            host_prepare=_down_prepare(down, names),
            invoke=_down_adapter(
                down, names, not first, done, None if first else "attn_rotated"
            ),
        ),
        RoutineDecl(
            prefix + "reduce_norm",
            threads=COMM_THREADS,
            smem_bytes=128,
            cluster_barriers=2,
            ctas_per_tile=2,
            named_barriers=(NORM_BAR,),
            invoke=norm_adapter(
                collective,
                names=names,
                cluster=2,
                quant_outputs=0,
                bump=None if first else "local_add_gen",
                done="mlp0_norm_done" if first else None,
                clearers=CLEARERS,
                rotated=rotated,
            ),
        ),
        RoutineDecl(
            prefix + "reduce_clear",
            threads=COMM_THREADS,
            smem_bytes=128,
            named_barriers=(NORM_BAR,),
            invoke=clear_adapter(
                collective,
                clearers=CLEARERS,
                participants=TOKENS * 2 + CLEARERS,
                rotated=rotated,
            ),
        ),
    )


def mlp_prefetch(index: int, *, before_gate: bool) -> tuple[Prefetch, ...]:
    """GateUp and Down weights/scales in issue order."""
    names = mlp_names(index)
    return tuple(
        Prefetch(names[name], size, before_gate=before_gate)
        for name, size in (
            ("gate_w", 2 * INTER * HIDDEN // 2),
            ("gate_ws", 2 * INTER * HIDDEN // 16),
            ("down_w", HIDDEN * INTER // 2),
            ("down_ws", HIDDEN * INTER // 16),
        )
    )


def router_routines() -> tuple[RoutineDecl, RoutineDecl]:
    """Router logits and top-k, hosted on the MoE gate and Down roles."""
    router = LongcatRouterTma()
    return (
        RoutineDecl(
            "mlp0_router",
            threads=384,
            smem_bytes=ROUTER_SMEM,
            smem_align=128,
            tmem_cols=32,
            cluster_barriers=2,
            ctas_per_tile=2,
            named_barriers=(0, 2, 3),
            host_prepare=_router_prepare(router),
            invoke=_router_adapter(router),
        ),
        RoutineDecl(
            "mlp0_router_topk",
            threads=ROUTER_TOPK_THREADS,
            smem_bytes=ROUTER_SMEM,
            smem_align=128,
            named_barriers=(3, DOWN_DONE_BAR),
            gates_itself=True,
            host_prepare=_topk_prepare(),
            invoke=_router_topk_adapter(router, "routing"),
        ),
    )


def make_mlp_buffers(index: int, device) -> dict[str, torch.Tensor]:
    """Allocate dense scratch and outputs under their final layer names."""
    names = mlp_names(index)
    bf = dict(device=device, dtype=torch.bfloat16)
    out = {
        names["act"]: torch.zeros(TOKENS, INTER // 2, device=device, dtype=torch.uint8),
        names["act_scale"]: torch.zeros(
            12288, device=device, dtype=torch.float8_e4m3fn
        ),
        names["gate_arrivals"]: torch.zeros(
            INTER // STAGE_K, device=device, dtype=torch.uint32
        ),
        names["gate_epochs"]: torch.zeros(DOWN_CTAS, device=device, dtype=torch.uint32),
        names["down_scratch"]: torch.zeros(TOKENS, HIDDEN, **bf),
        names["residual_out"]: torch.zeros(TOKENS, HIDDEN, **bf),
    }
    if index == 1:
        out[names["norm_out"]] = torch.zeros(TOKENS, HIDDEN, **bf)
    else:
        # MLP0 writes the attention's row-major x buffer directly.
        out.update(
            router_logits=torch.zeros(
                ROUTER_K_SPLITS, TOKENS, EXPERTS, device=device, dtype=torch.float32
            ),
            router_ids=torch.zeros(TOKENS, TOPK, device=device, dtype=torch.int32),
            router_weights=torch.zeros(
                TOKENS, TOPK, device=device, dtype=torch.float32
            ),
            router_zero=torch.zeros(TOKENS, device=device, dtype=torch.float32),
        )
    return out
