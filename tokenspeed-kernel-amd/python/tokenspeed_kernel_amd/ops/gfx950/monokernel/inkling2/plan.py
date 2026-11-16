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

"""Stage plans of the Inkling2 decoder layer monokernels.

One launch runs one decoder layer's decode step on one rank of the attention
TP group. The residual state stays a full row on every rank, as on the regular
path, so the mixes, norms and routing run on every rank. The layer's two
all-reduces (attention output, expert fold) each become one exchange: every
rank stores its BF16 partial rows to every rank's peer region and raises one
flag, and every rank sums the ranks' partials in rank order in FP32 and rounds
once, as the regular path's rank-ordered fold does. Past 16 rows the
attention output's exchange scatters instead (``WO_SCATTER``): each rank
sends a column block's partial to the block's owner only, the owner folds it
in rank order and sends the folded rows to every rank, half the bytes.

Region sizes are 32-bit pairs per row. BF16 vectors pack two values a pair.
"""

from __future__ import annotations

from tokenspeed_kernel_amd.ops.gfx950.gemm.fp16.fp32_output import (
    projection_split_k,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Plan, Stage

# Per-rank model geometry at attention TP4 / experts EP4.
HIDDEN = 4608
Q_HEADS = 12
KV_HEADS = 2
HEAD_DIM = 128
REL_EXTENT = 1024
QKVR = 3792  # q 1536 | k 256 | v 256 | rel 192 | gate 1536 | sink 12 | pad 4
QKVR_PADDED = 3840  # the packed qkvr weight: 60 tasks of 64 rows
# A full attention layer's projection (3808 rows a rank): the local layer's
# 3792 rows with the K and V convolution taps it projects (a head's 4 taps,
# K's after the sink, then V's) and 4 zero rows. It projects three sources:
# the Q source's row holds q, rel, gate and sink at the local offsets; the K
# and V sources' rows hold only k and its taps (``kproj``) and v and its
# taps (``vproj``), the columns the regular path reads from them.
FULL_QKVR = 3808
TAPS_OFF = 3788  # the K taps' first row; the V taps' follow
KV_PROJ = 264  # k (or v) 256 | its taps 8
KV_PROJ_PADDED = 320  # 5 tasks of 64 rows
ROUTED = 384
TOP_K = 8
INTER = 2304  # expert intermediate width
LOCAL_SPLITS = 8  # KV splits of a 1023-key window
# KV splits of a full attention layer: the regular decode's count at the
# served context length (``_decode_split_count``: one a 2048 keys, at most 32).
FULL_SPLITS = 32
WORLD = 4
GRID = 256

# Projection tasks (``projection.py``): 16-row weight blocks per task, and tasks.
# SPLITS is the regular projection's K split, SLOTS the split chains a pass
# (the CTA's 8 waves own them: 8 / BLOCKS slot waves). The qkvr and router
# tasks follow from the row count (``projection_shape``).
QKVR_SPLITS = projection_split_k(QKVR, HIDDEN)  # 12
WO_BLOCKS = 2
WO_TASKS = HIDDEN // (16 * WO_BLOCKS)  # 144, also the attention reduce tasks
WO_SPLITS = projection_split_k(HIDDEN, Q_HEADS * HEAD_DIM)  # 3
WO_SLOTS = 4  # one pass, 1 empty slot
ROUTER_SPLITS = projection_split_k(ROUTED, HIDDEN)  # 18
# Query rows of a block set's partials (``AttentionResidual.set_rows``): the
# MLP queries of its 6 layers, the attention queries of the 4 mid local
# layers (1), of the full layer (3) and of the next block's first layer (1).
SET_QUERIES = 14

# Whether the attention output exchange is a reduce-scatter and an all-gather
# (projection.scatter_fold_stage), a row count: half the bytes over the links,
# one hop more. At 16 rows and fewer every rank sends its whole partial to
# every rank (fewer bytes there, so the hop costs more than it saves).
WO_SCATTER = {1: False, 8: False, 16: False, 32: True, 64: True}
FOLD_COLUMNS = 32
FOLD_TASKS = HIDDEN // FOLD_COLUMNS  # 144, also the expert reduce tasks

# Past 16 rows each projection task would load every row's activations for
# all of K: the tasks split K into chained groups (``projection.chain_stage``),
# each loading its splits' activations only, and take wider weight blocks,
# so that a task's activation and weight bytes stay those of 16 rows. A
# row count's ``(blocks a task, K split groups)``:
PROJECTION_SHAPES = {
    "qkvr": {1: (1, 1), 8: (1, 1), 16: (1, 1), 32: (2, 2), 64: (4, 4)},
    "kproj": {1: (1, 1), 8: (1, 1), 16: (1, 1), 32: (2, 2), 64: (4, 4)},
    "vproj": {1: (1, 1), 8: (1, 1), 16: (1, 1), 32: (2, 2), 64: (4, 4)},
    "router": {1: (1, 1), 8: (1, 1), 16: (1, 1), 32: (1, 2), 64: (1, 3)},
}

# Heads a head prologue task takes, a row count (head.py): ``(Q heads, K and
# V heads)`` of a row, a wave a head. Past the grid the per-head tasks run in
# rounds; grouped heads fit one round.
HEAD_SHAPES = {1: (1, 1), 8: (1, 1), 16: (1, 1), 32: (6, 4), 64: (6, 4)}


def head_tasks(rows: int) -> int:
    """Head prologue tasks of ``rows`` rows."""
    q, kv = HEAD_SHAPES[rows]
    return (Q_HEADS // q + 2 * KV_HEADS // kv) * rows


def _max_groups(name: str) -> int:
    return max(groups for _, groups in PROJECTION_SHAPES[name].values())


def projection_shape(name: str, rows: int) -> tuple[int, int]:
    """``(blocks a task, K split groups)`` of projection ``name`` (``qkvr``
    or ``router``) at ``rows`` rows."""
    return PROJECTION_SHAPES[name][rows]


def chain_stage(name: str, group: int) -> str:
    """The stage of K split group ``group`` of projection ``name`` (the last
    group is the projection's own stage)."""
    return f"{name}_k{group}"


def chain_run(name: str, group: int) -> str:
    """The region of the running sum after K split group ``group``."""
    return f"{name}_run{group}"


def _bf16_pairs(values: int) -> int:
    return values // 2


# Regions of the mid local layer, in pairs per row.
LOCAL_REGIONS = (
    ("x_attn", _bf16_pairs(HIDDEN)),  # normalized attention input (Q source)
    # A full layer's normalized K and V sources, and their projections.
    ("x_k", _bf16_pairs(HIDDEN)),
    ("x_v", _bf16_pairs(HIDDEN)),
    ("kproj", _bf16_pairs(KV_PROJ_PADDED)),
    ("vproj", _bf16_pairs(KV_PROJ_PADDED)),
    ("cand", _bf16_pairs(HIDDEN)),  # the block partial P, for the MLP mix
    ("qkvr", _bf16_pairs(QKVR_PADDED)),
    ("q", _bf16_pairs(Q_HEADS * HEAD_DIM)),
    ("kv", _bf16_pairs(2 * KV_HEADS * HEAD_DIM)),  # this token's K and V
    ("rel", _bf16_pairs(Q_HEADS * REL_EXTENT)),
    # FP32 split outputs and LSEs, room for a full layer's splits.
    ("attn_part", FULL_SPLITS * Q_HEADS * HEAD_DIM),
    ("attn_lse", FULL_SPLITS * Q_HEADS),
    # The gated attention output: plain words of BF16 pairs, each KV head's
    # published with a flag (the output projection loads every row's).
    ("o", _bf16_pairs(Q_HEADS * HEAD_DIM)),
    ("o_flag", KV_HEADS),
    # Peer regions of plain words, two BF16 values a word: [source rank][hidden].
    ("wo_part", WORLD * HIDDEN // 4),
    ("wo_flag", WORLD * WO_TASKS),  # row 0: [source rank][task]
    # The folded attention output, gathered from the owners of its columns
    # (projection.scatter_fold_stage): plain words, a flag a column block.
    ("wo_gather", HIDDEN // 4),
    ("wo_gather_flag", WO_TASKS),
    ("attn_sc", _bf16_pairs(HIDDEN)),  # reduced attention output after its conv
    ("x_mlp", _bf16_pairs(HIDDEN)),
    ("logits", ROUTED),  # FP32
    ("route", 2 * TOP_K),  # ids and weights
    # Plain words of the experts' quantized inputs (the BF16 values of their
    # codes, in pairs), each row's published with a flag: the GEMM tiles load
    # them as MFMA operands.
    ("xq", HIDDEN // 4),
    ("xq_flag", 1),
    ("xa", 1),  # FP32 amax of the input row (the NVFP4 outer factor)
    ("gate_up", TOP_K * _bf16_pairs(2 * INTER)),  # [route slot][w13 columns]
    ("hq", TOP_K * INTER // 4),  # words: [route slot][INTER / 2]
    ("hq_flag", TOP_K),
    ("ha", TOP_K),  # FP32 amax of each route's h (NVFP4)
    ("ex2", TOP_K * _bf16_pairs(HIDDEN)),  # [route slot][hidden]
    ("fold_part", WORLD * HIDDEN // 4),
    ("fold_flag", WORLD * FOLD_TASKS),
    ("mlp_sc", _bf16_pairs(HIDDEN)),  # reduced expert output after its conv
    # A boundary layer's MLP query's set partial (mix.sweep_stage): the FP32
    # accumulator (one value a pair), then (max, sum).
    ("set_acc", HIDDEN),
    ("set_ms", 2),
    # FP32 running sums of the projections' chained K split groups, one pair
    # a value (launches of many rows only).
    *((chain_run("qkvr", g), QKVR_PADDED) for g in range(_max_groups("qkvr") - 1)),
    *((chain_run("router", g), ROUTED) for g in range(_max_groups("router") - 1)),
    *((chain_run("kproj", g), KV_PROJ_PADDED) for g in range(_max_groups("kproj") - 1)),
    *((chain_run("vproj", g), KV_PROJ_PADDED) for g in range(_max_groups("vproj") - 1)),
)
LOCAL_PEERS = (
    "wo_part",
    "wo_flag",
    "wo_gather",
    "wo_gather_flag",
    "fold_part",
    "fold_flag",
)
LOCAL_FLAGS = ("wo_flag", "wo_gather_flag", "fold_flag")

# First CTA of each stage. A first placement, to tune from the stage timeline:
# weight-streaming stages start at CTA 0 and spread over the grid; per-row
# vector stages take CTAs the streaming stages reach last.
LOCAL_BASES = {
    "mix_attn": 192,
    "norm_attn": 192,
    "sweep": 128,
    "sweep_rest": 0,
    "qkvr": 0,
    # A full layer's K and V source projections, past the Q source's.
    "kproj": 238,
    "vproj": 255,
    "head": 128,
    "attn": 144,
    "merge": 128,
    "wo": 0,
    "fold_attn": 144,
    "red_attn": 128,
    "mix_mlp": 192,
    "router": 200,
    "topk": 248,
    "quant": 240,
    "ex1": 0,
    "swiglu": 128,
    "ex2": 0,
    "fold": 64,
    "red_moe": 128,
}


EXPERT_FORMATS = ("nvfp4", "mxfp8")


def _chained(
    name: str, n: int, rows: int, base: int, x: str, out: str
) -> tuple[Stage, ...]:
    """The stages of projection ``name`` (``n`` output columns) at ``rows``
    rows: its K split groups on consecutive CTAs from ``base``, the last
    group in the stage ``name`` that writes ``out``. A last task past ``n``
    reads the packed weight's zero rows (``QKVR_PADDED``) and writes into
    its region's padding."""
    blocks, groups = projection_shape(name, rows)
    tasks = (n + 16 * blocks - 1) // (16 * blocks)
    stages = []
    for g in range(groups):
        reads = (x,) if g == 0 else (x, chain_run(name, g - 1))
        last = g == groups - 1
        stages.append(
            Stage(
                name if last else chain_stage(name, g),
                tasks,
                (base + g * tasks) % GRID,
                reads,
                (out,) if last else (chain_run(name, g),),
            )
        )
    return tuple(stages)


def local_layer_plan(
    rows: int, chain: int, experts: str, boundary: bool, first: bool
) -> Plan:
    """The local (sliding-window) layer's plan for ``rows`` token rows,
    ``chain`` of them per request (8 for a DSPARK verify block), with
    ``experts`` (``"nvfp4"`` or ``"mxfp8"``) routed experts: a mid local
    layer, or with ``boundary`` the first layer of an attention-residual
    block (it stores the block, sweeps the new block set's partials and
    mixes its MLP input over them). ``first`` (a boundary layer) is layer
    0, whose block starts from the embedding with no block stored before
    it: its attention input is the embedding's RMSNorm (``norm_attn``), not
    a mix (``mix_attn``). The variants share every stage they have and
    every region; MXFP8 stages leave the row maxima (``xa``, ``ha``)
    unused, mid layers the set partial regions. The regions stay: launches
    of every variant share the scratch and alternate its peer copies, which
    needs the same offsets in every plan."""
    return _layer_plan(rows, chain, experts, boundary, first, None)


# The rows of a verify block a full layer's attention task takes
# (``attention.shared_attention_stage``: each split's keys read once for
# them), a row count: 8, or half a block where a block a task would leave
# the grid short of tasks; 0 is a row a task. The values come from cold
# timings of the attention and merge stages at 9k and 50k keys.
SHARED_ATTENTION = {1: 0, 8: 4, 16: 4, 32: 8, 64: 8}


def shared_attention(rows: int, chain: int) -> int:
    """The rows a full layer's attention task shares at ``rows`` rows: a
    verify block's 8 or half of them (``SHARED_ATTENTION``), or 0 for a row
    a task."""
    return SHARED_ATTENTION[rows] if chain == 8 else 0


def attention_tasks(rows: int, chain: int, full_splits: int | None) -> int:
    """Attention tasks of a launch: a local layer's (row, KV head, split);
    a full layer's (verify block, KV head, split) when it shares its tasks
    (:func:`shared_attention`), else (row, KV head, split)."""
    if full_splits is None:
        return KV_HEADS * LOCAL_SPLITS * rows
    if shared_attention(rows, chain):
        return KV_HEADS * full_splits * (rows // shared_attention(rows, chain))
    return KV_HEADS * full_splits * rows


def full_layer_plan(rows: int, chain: int, experts: str, splits: int) -> Plan:
    """A full attention layer's plan (the last layer of an
    attention-residual block), as :func:`local_layer_plan`'s mid layer
    plan: the attention mix takes three queries, normalizing the Q, K and V
    sources (``x_attn``, ``x_k``, ``x_v``); the projection projects each
    source (``qkvr``, ``kproj``, ``vproj``); the head stage convolves K and
    V with taps the projection gives; and the attention takes every key
    over ``splits`` splits (the regular decode's count at the served
    context length, at most ``FULL_SPLITS``). Its regions are every plan's."""
    if not 1 <= splits <= FULL_SPLITS:
        raise ValueError(
            f"a full layer's split count {splits} is not 1 .. {FULL_SPLITS}"
        )
    return _layer_plan(rows, chain, experts, False, False, splits)


def _layer_plan(
    rows: int,
    chain: int,
    experts: str,
    boundary: bool,
    first: bool,
    full_splits: int | None,
) -> Plan:
    """A local layer's plan, or with ``full_splits`` a full layer's."""
    if experts not in EXPERT_FORMATS:
        raise ValueError(f"no Inkling2 layer plan for {experts} experts")
    full = full_splits is not None
    if full and boundary:
        raise ValueError("a full layer is never a block boundary")
    if first and not boundary:
        raise ValueError("layer 0 starts an attention-residual block")
    nvfp4 = experts == "nvfp4"
    # A full layer's K and V source projections: on the CTAs past the Q
    # source's tasks when it fits the grid in one group; else (K split
    # groups) beside its first group, whose CTAs it frees first.
    kv_bases = (
        (LOCAL_BASES["kproj"], LOCAL_BASES["vproj"])
        if projection_shape("qkvr", rows)[1] == 1
        else (0, 4 * (KV_PROJ_PADDED // (16 * projection_shape("kproj", rows)[0])))
    )
    xa = ("xa",) if nvfp4 else ()
    ha = ("ha",) if nvfp4 else ()
    b = LOCAL_BASES
    stages = (
        Stage(
            "norm_attn" if first else "mix_attn",
            3 * rows if full else rows,
            b["norm_attn" if first else "mix_attn"],
            (),
            ("x_attn", "x_k", "x_v", "cand") if full else ("x_attn", "cand"),
        ),
        # The new block set's partials: the layer's MLP query first, for its
        # mix in this launch; the others for the following layers.
        *(
            (
                Stage("sweep", rows, b["sweep"], ("cand",), ("set_acc", "set_ms")),
                Stage(
                    "sweep_rest",
                    (SET_QUERIES - 1) * rows,
                    b["sweep_rest"],
                    ("cand",),
                    (),
                ),
            )
            if boundary
            else ()
        ),
        *_chained(
            "qkvr", FULL_QKVR if full else QKVR, rows, b["qkvr"], "x_attn", "qkvr"
        ),
        *(
            (
                *_chained("kproj", KV_PROJ, rows, kv_bases[0], "x_k", "kproj"),
                *_chained("vproj", KV_PROJ, rows, kv_bases[1], "x_v", "vproj"),
            )
            if full
            else ()
        ),
        Stage(
            "head",
            head_tasks(rows),
            b["head"],
            ("qkvr", "kproj", "vproj") if full else ("qkvr",),
            ("q", "kv", "rel"),
        ),
        Stage(
            "attn",
            attention_tasks(rows, chain, full_splits),
            b["attn"],
            ("q", "kv", "rel"),
            ("attn_part", "attn_lse"),
        ),
        Stage(
            "merge",
            KV_HEADS * rows,
            b["merge"],
            ("attn_part", "attn_lse", "qkvr"),
            ("o", "o_flag"),
        ),
        Stage("wo", WO_TASKS, b["wo"], ("o", "o_flag"), ("wo_part", "wo_flag")),
        # Scattered (WO_SCATTER): each rank folds the partials of the column
        # blocks it owns and gathers the folded rows to every rank. The
        # reduce tasks run the output conv of their channels (conv.py).
        *(
            (
                Stage(
                    "fold_attn",
                    WO_TASKS // WORLD,
                    b["fold_attn"],
                    ("wo_part", "wo_flag"),
                    ("wo_gather", "wo_gather_flag"),
                ),
            )
            if WO_SCATTER[rows]
            else ()
        ),
        Stage(
            "red_attn",
            WO_TASKS,
            b["red_attn"],
            (
                ("wo_gather", "wo_gather_flag")
                if WO_SCATTER[rows]
                else ("wo_part", "wo_flag")
            ),
            ("attn_sc",),
        ),
        Stage(
            "mix_mlp",
            rows,
            b["mix_mlp"],
            ("attn_sc", "set_acc", "set_ms") if boundary else ("cand", "attn_sc"),
            ("x_mlp",),
        ),
        *_chained("router", ROUTED, rows, b["router"], "x_mlp", "logits"),
        Stage("topk", rows, b["topk"], ("logits",), ("route",)),
        Stage("quant", rows, b["quant"], ("x_mlp",), ("xq", *xa, "xq_flag")),
        Stage("ex1", GRID, b["ex1"], ("route", "xq", *xa, "xq_flag"), ("gate_up",)),
        Stage(
            "swiglu",
            TOP_K * rows,
            b["swiglu"],
            ("route", "gate_up"),
            ("hq", *ha, "hq_flag"),
        ),
        Stage("ex2", GRID, b["ex2"], ("route", "hq", *ha, "hq_flag"), ("ex2",)),
        Stage(
            "fold", FOLD_TASKS, b["fold"], ("route", "ex2"), ("fold_part", "fold_flag")
        ),
        Stage(
            "red_moe", FOLD_TASKS, b["red_moe"], ("fold_part", "fold_flag"), ("mlp_sc",)
        ),
    )
    return Plan(
        grid=GRID,
        stages=stages,
        regions=LOCAL_REGIONS,
        rows=rows,
        chain=chain,
        world=WORLD,
        peers=LOCAL_PEERS,
        flags=LOCAL_FLAGS,
    )
