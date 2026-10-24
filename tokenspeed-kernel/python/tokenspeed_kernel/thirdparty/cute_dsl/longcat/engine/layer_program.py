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

"""Final phases and per-layer buffers for the LongCat fused decoder."""

from __future__ import annotations

import torch
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.engine._plan import (
    EngineConfig,
    Phase,
    Prefetch,
    Program,
    Role,
    RoutineCall,
    TicketDecl,
    TicketWait,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.engine.attention_worker import (
    ATTENTION_OPERANDS,
    WorkerShape,
    attention_phases,
    make_worker_buffers,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.engine.mlp_program import (
    CLEARERS,
    DOWN_CTAS,
    EXPERTS,
    GATE_CTAS,
    HIDDEN,
    ROUTER_CTAS,
    TOKENS,
    MlpShape,
    make_mlp_buffers,
    mlp_names,
    mlp_prefetch,
    mlp_routines,
    router_routines,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.engine.moe_program import (
    OPERANDS as MOE_OPERANDS,
)
from tokenspeed_kernel.thirdparty.cute_dsl.longcat.engine.moe_program import (
    MoeShape,
    _epilogue_parts,
    moe_routines,
)


def layer_program(
    mlp0: MlpShape,
    attn: WorkerShape,
    mlp1: MlpShape,
    moe: MoeShape,
    *,
    pdl: bool,
) -> Program:
    """Build the final main, MoE GateUp and MoE Down phase sequences.

    Args:
        mlp0, mlp1: Dense branches 0 and 1, on the attention's TP rank.
        attn: Eight-head attention branch with twelve CTAs per token.
        moe: Expert shape with the 16x8x2 grouped topology.
        pdl: Warm the first-phase weights before the preceding A0 retires.

    Returns:
        One program with 96 main CTAs, 32 MoE GateUp CTAs and 16 MoE Down CTAs.
    """
    if (mlp0.index, mlp1.index) != (0, 1):
        raise ValueError("the layer requires dense branches 0 and 1")
    if mlp0.rank != attn.rank or mlp1.rank != attn.rank:
        raise ValueError("the dense and attention routines must share a TP rank")
    if GATE_CTAS + DOWN_CTAS > attn.ctas:
        raise ValueError("the dense routines exceed the main role")
    if ROUTER_CTAS > moe.gate_ctas or moe.down_ctas < TOKENS:
        raise ValueError("the MoE roles cannot host the router and top-k")
    g0, d0, n0, c0 = mlp_routines(mlp0)
    g1, d1, n1, c1 = mlp_routines(mlp1)
    router, topk = router_routines()
    route, gate, down, epilogue = moe_routines(moe)
    attention = attention_phases(attn, norm_prefetch=mlp_prefetch(1, before_gate=False))

    # MLP0 consumes on GateUp CTAs while the Down clearers retain down_done.
    head = (
        Phase(
            "mlp0_mlp",
            issue_before_gate=pdl,
            prefetch=mlp_prefetch(0, before_gate=True) if pdl else (),
            routines=(
                RoutineCall(g0, "mlp0_gate", tiles=GATE_CTAS),
                RoutineCall(
                    d0,
                    "mlp0_down",
                    tiles=DOWN_CTAS,
                    cta_offset=GATE_CTAS,
                    produces=(("mlp0_down_done", DOWN_CTAS),),
                ),
            ),
        ),
        Phase(
            "mlp0_reduce_norm",
            routines=(
                RoutineCall(
                    n0,
                    "mlp0_mlp_norm",
                    tiles=TOKENS,
                    produces=(("mlp0_norm_done", TOKENS * 2),),
                ),
            ),
        ),
        Phase(
            "mlp0_reduce_clear",
            consumes=(TicketWait("mlp0_down_done"),),
            routines=(
                RoutineCall(
                    c0,
                    "mlp0_mlp_norm",
                    tiles=CLEARERS,
                    cta_offset=GATE_CTAS + TOKENS * 2,
                    produces=(("mlp0_rotated", 1),),
                ),
            ),
        ),
    )
    tail = (
        Phase(
            "mlp",
            consumes=(TicketWait("attn_norm_done"),),
            issue_before_gate=True,
            routines=(
                RoutineCall(g1, "gate", tiles=GATE_CTAS),
                RoutineCall(
                    d1,
                    "down",
                    tiles=DOWN_CTAS,
                    cta_offset=GATE_CTAS,
                    produces=(("down_done", DOWN_CTAS),),
                ),
            ),
        ),
        Phase(
            "reduce_norm",
            consumes=(TicketWait("down_done"),),
            routines=(
                RoutineCall(n1, "mlp_norm", tiles=TOKENS, cta_offset=GATE_CTAS),
                RoutineCall(
                    c1, "mlp_norm", tiles=CLEARERS, cta_offset=GATE_CTAS + TOKENS * 2
                ),
            ),
        ),
    )
    gate_phases = (
        Phase(
            "mlp0_router",
            prefetch=(
                (Prefetch("router_w", EXPERTS * HIDDEN * 2, before_gate=True),)
                if pdl
                else ()
            ),
            routines=(
                RoutineCall(
                    router,
                    "mlp0_router",
                    tiles=ROUTER_CTAS // 2,
                    produces=(("mlp0_logits", ROUTER_CTAS),),
                ),
            ),
        ),
        Phase(
            "moe_route",
            consumes=(TicketWait("routing"),),
            routines=(
                RoutineCall(route, "moe_route", tiles=1, produces=(("moe_groups", 1),)),
            ),
        ),
        Phase(
            "moe_experts",
            consumes=(TicketWait("moe_groups"),),
            routines=(RoutineCall(gate, "moe_gate", tiles=moe.gate_ctas),),
        ),
        Phase(
            "moe_epilogue",
            consumes=(TicketWait("moe_down"),),
            routines=(
                RoutineCall(
                    epilogue, "moe_epilogue", tiles=TOKENS * _epilogue_parts(moe)
                ),
            ),
        ),
    )
    down_phases = (
        Phase(
            "mlp0_router_topk",
            consumes=(TicketWait("mlp0_logits"),),
            issue_before_gate=True,
            routines=(
                RoutineCall(
                    topk, "mlp0_topk", tiles=TOKENS, produces=(("routing", TOKENS),)
                ),
            ),
        ),
        Phase(
            "moe_experts",
            consumes=(TicketWait("moe_groups"),),
            routines=(
                RoutineCall(
                    down,
                    "moe_down",
                    tiles=moe.down_ctas,
                    produces=(("moe_down", moe.down_ctas),),
                ),
            ),
        ),
    )
    return Program(
        name=f"layer_s{attn.splits}_c{attn.cluster}_moe{moe.gate_clusters}x"
        f"{moe.down_clusters}x{moe.group_splits}",
        config=EngineConfig(
            grid=attn.ctas + moe.gate_ctas + moe.down_ctas, cluster=attn.cluster
        ),
        roles=(
            Role("main", attn.ctas, head + attention + tail),
            Role("moe_gate", moe.gate_ctas, gate_phases),
            Role("moe_down", moe.down_ctas, down_phases),
        ),
        phase_order=(
            "mlp0_router",
            "mlp0_router_topk",
            "mlp0_mlp",
            "mlp0_reduce_norm",
            "mlp0_reduce_clear",
            "moe_route",
            "moe_experts",
            "moe_epilogue",
            *(phase.name for phase in attention),
            "mlp",
            "reduce_norm",
        ),
        tickets=(
            TicketDecl("mlp0_down_done"),
            TicketDecl("mlp0_norm_done"),
            TicketDecl("mlp0_rotated"),
            TicketDecl("mlp0_logits"),
            TicketDecl("routing"),
            TicketDecl("comp_q"),
            TicketDecl("qkv_done"),
            TicketDecl("qb_done"),
            TicketDecl("qb_ready", width=attn.q_b_tiles),
            TicketDecl("qb_sub", width=attn.q_b_tiles),
            TicketDecl("abs_done"),
            TicketDecl("abs_ready", width=attn.abs_tiles),
            TicketDecl("prepare_done"),
            TicketDecl("head_ready", width=attn.heads),
            TicketDecl("attn_norm_done"),
            TicketDecl("attn_rotated"),
            TicketDecl("down_done"),
            TicketDecl("moe_groups"),
            TicketDecl("moe_down"),
        ),
        stream_budgets=(("sparse_mla", 0),),
        operands=tuple(
            dict.fromkeys(
                (
                    *mlp_names(0).values(),
                    *ATTENTION_OPERANDS,
                    *mlp_names(1).values(),
                    *MOE_OPERANDS,
                )
            )
        ),
    )


def make_layer_buffers(shape: WorkerShape, device) -> dict[str, torch.Tensor]:
    """Allocate A1 and both dense branches' scratch under final operand names.

    ``shape`` supplies A1's KV dtype and split geometry; ``device`` selects
    the allocation device. The returned name-to-tensor map lives with the
    workspace. Shared expert scratch and per-layer MoE ready/generation
    words are owned separately by the runtime.
    """
    return {
        **make_worker_buffers(shape, device),
        **make_mlp_buffers(1, device),
        **make_mlp_buffers(0, device),
    }
