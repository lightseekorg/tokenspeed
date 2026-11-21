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

import re

from tokenspeed_kernel_amd.ops.gfx950.monokernel.inkling2.dims import Dims, tp4
from tokenspeed_kernel_amd.ops.gfx950.monokernel.plan import Plan, Stage

# The geometries the plans lay out and the stages implement: a rank at TP4 of
# Inkling 2 Small, and of the Inkling2 geometry of half its width.
GEOMETRIES = (
    tp4(hidden=4608, heads=48, kv_heads=8, routed=384, inter=2304),
    tp4(hidden=2304, heads=24, kv_heads=8, routed=128, inter=1152),
)
SERVED_DIMS = GEOMETRIES

REL_EXTENT = 1024
TOP_K = 8
LOCAL_SPLITS = 8  # KV splits of a 1023-key window
# KV splits of a full attention layer: the regular decode's count at the
# served context length (``_decode_split_count``: one a 2048 keys, at most 32).
FULL_SPLITS = 32
GRID = 256
# Query rows of a block set's partials (``AttentionResidual.set_rows``): the
# MLP queries of its 6 layers, the attention queries of the 4 mid local
# layers (1), of the full layer (3) and of the next block's first layer (1).
SET_QUERIES = 14
WO_BLOCKS = 2  # 16-row weight blocks of an output projection task
FOLD_COLUMNS = 32


def wo_tasks(dims: Dims) -> int:
    """Output projection tasks, also the attention reduce tasks."""
    return dims.hidden // (16 * WO_BLOCKS)


def fold_tasks(dims: Dims) -> int:
    """Column blocks of an exchange fold, also the expert reduce tasks."""
    return dims.hidden // FOLD_COLUMNS


# The output projection's split chains a pass: one pass, the CTA's 8 waves
# in 2 blocks of 4 slots.
WO_SLOTS = 4

# Whether the attention output exchange is a reduce-scatter and an all-gather
# (projection.scatter_fold_stage), a row count: half the bytes over the links,
# one hop more. At 16 rows and fewer every rank sends its whole partial to
# every rank (fewer bytes there, so the hop costs more than it saves).
WO_SCATTER = {1: False, 4: False, 8: False, 16: False, 32: True, 64: True}

# Past 16 rows each projection task would load every row's activations for
# all of K: the tasks split K into chained groups (``projection.chain_stage``),
# each loading its splits' activations only, and take wider weight blocks,
# so that a task's activation and weight bytes stay those of 16 rows. A
# model width's, then a row count's ``(blocks a task, K split groups)``; a
# group count divides the projection's K splits. The half-width geometry
# halves the groups of the width it halves (its K is half) where they divide
# its splits (untimed).
PROJECTION_SHAPES = {
    4608: {
        "qkvr": {1: (1, 1), 4: (1, 1), 8: (1, 1), 16: (1, 1), 32: (2, 2), 64: (4, 4)},
        "kproj": {1: (1, 1), 4: (1, 1), 8: (1, 1), 16: (1, 1), 32: (2, 2), 64: (4, 4)},
        "vproj": {1: (1, 1), 4: (1, 1), 8: (1, 1), 16: (1, 1), 32: (2, 2), 64: (4, 4)},
        "router": {1: (1, 1), 4: (1, 1), 8: (1, 1), 16: (1, 1), 32: (1, 2), 64: (1, 3)},
    },
    2304: {
        "qkvr": {1: (1, 1), 4: (1, 1), 8: (1, 1), 16: (1, 1), 32: (2, 1), 64: (4, 2)},
        "kproj": {1: (1, 1), 4: (1, 1), 8: (1, 1), 16: (1, 1), 32: (2, 1), 64: (4, 2)},
        "vproj": {1: (1, 1), 4: (1, 1), 8: (1, 1), 16: (1, 1), 32: (2, 1), 64: (4, 2)},
        "router": {1: (1, 1), 4: (1, 1), 8: (1, 1), 16: (1, 1), 32: (1, 1), 64: (1, 3)},
    },
}

# Heads a head prologue task takes, a model width's and a row count's (head.py):
# ``(Q heads, K and V heads)`` of a row, a wave a head. Past the grid the
# per-head tasks run in rounds; grouped heads (the Q heads of a KV head, the
# 4 K and V heads) fit one round.
HEAD_SHAPES = {
    4608: {1: (1, 1), 4: (1, 1), 8: (1, 1), 16: (1, 1), 32: (6, 4), 64: (6, 4)},
    2304: {1: (1, 1), 4: (1, 1), 8: (1, 1), 16: (1, 1), 32: (3, 4), 64: (3, 4)},
}


def head_tasks(dims: Dims, rows: int) -> int:
    """Head prologue tasks of ``rows`` rows."""
    q, kv = HEAD_SHAPES[dims.hidden][rows]
    return (dims.q_heads // q + 2 * dims.kv_heads // kv) * rows


def _max_groups(dims: Dims, name: str) -> int:
    return max(groups for _, groups in PROJECTION_SHAPES[dims.hidden][name].values())


def projection_shape(dims: Dims, name: str, rows: int) -> tuple[int, int]:
    """``(blocks a task, K split groups)`` of projection ``name`` (``qkvr``,
    ``kproj``, ``vproj`` or ``router``) at ``rows`` rows."""
    return PROJECTION_SHAPES[dims.hidden][name][rows]


def chain_stage(name: str, group: int) -> str:
    """The stage of K split group ``group`` of projection ``name`` (the last
    group is the projection's own stage)."""
    return f"{name}_k{group}"


def chain_run(name: str, group: int) -> str:
    """The region of the running sum after K split group ``group``."""
    return f"{name}_run{group}"


def _bf16_pairs(values: int) -> int:
    return values // 2


def regions(dims: Dims) -> tuple[tuple[str, int], ...]:
    """Every layer plan's regions at ``dims``, in pairs per row."""
    H = dims.hidden
    QD = dims.q_dim
    KV_PROJ_PADDED = dims.kv_proj_padded
    QKVR_PADDED = dims.qkvr_padded
    return (
        ("x_attn", _bf16_pairs(H)),  # normalized attention input (Q source)
        # A full layer's normalized K and V sources, and their projections.
        ("x_k", _bf16_pairs(H)),
        ("x_v", _bf16_pairs(H)),
        ("kproj", _bf16_pairs(KV_PROJ_PADDED)),
        ("vproj", _bf16_pairs(KV_PROJ_PADDED)),
        ("cand", _bf16_pairs(H)),  # the block partial P, for the MLP mix
        ("qkvr", _bf16_pairs(QKVR_PADDED)),
        ("q", _bf16_pairs(QD)),
        ("kv", _bf16_pairs(2 * dims.kv_dim)),  # this token's K and V
        ("rel", _bf16_pairs(dims.q_heads * REL_EXTENT)),
        # FP32 split outputs and LSEs, room for a full layer's splits.
        ("attn_part", FULL_SPLITS * QD),
        ("attn_lse", FULL_SPLITS * dims.q_heads),
        # The gated attention output: plain words of BF16 pairs, each KV head's
        # published with a flag (the output projection loads every row's).
        ("o", _bf16_pairs(QD)),
        ("o_flag", dims.kv_heads),
        # Peer regions of plain words, two BF16 values a word: [source rank][hidden].
        ("wo_part", dims.world * H // 4),
        ("wo_flag", dims.world * wo_tasks(dims)),  # row 0: [source rank][task]
        # The folded attention output, gathered from the owners of its columns
        # (projection.scatter_fold_stage): plain words, a flag a column block.
        ("wo_gather", H // 4),
        ("wo_gather_flag", wo_tasks(dims)),
        ("attn_sc", _bf16_pairs(H)),  # reduced attention output after its conv
        ("x_mlp", _bf16_pairs(H)),
        ("logits", dims.routed),  # FP32
        ("route", 2 * TOP_K),  # ids and weights
        # Plain words of the experts' quantized inputs (the BF16 values of their
        # codes, in pairs), each row's published with a flag: the GEMM tiles load
        # them as MFMA operands.
        ("xq", H // 4),
        ("xq_flag", 1),
        ("xa", 1),  # FP32 amax of the input row (the NVFP4 outer factor)
        ("gate_up", TOP_K * _bf16_pairs(2 * dims.inter)),  # [route slot][w13 columns]
        ("hq", TOP_K * dims.inter // 4),  # words: [route slot][inter / 2]
        ("hq_flag", TOP_K),
        ("ha", TOP_K),  # FP32 amax of each route's h (NVFP4)
        ("ex2", TOP_K * _bf16_pairs(H)),  # [route slot][hidden]
        ("fold_part", dims.world * H // 4),
        ("fold_flag", dims.world * fold_tasks(dims)),
        ("mlp_sc", _bf16_pairs(H)),  # reduced expert output after its conv
        # A boundary layer's MLP query's set partial (mix.sweep_stage): the FP32
        # accumulator (one value a pair), then (max, sum).
        ("set_acc", H),
        ("set_ms", 2),
        # FP32 running sums of the projections' chained K split groups, one pair
        # a value (launches of many rows only).
        *(
            (chain_run("qkvr", g), QKVR_PADDED)
            for g in range(_max_groups(dims, "qkvr") - 1)
        ),
        *(
            (chain_run("router", g), dims.routed)
            for g in range(_max_groups(dims, "router") - 1)
        ),
        *(
            (chain_run("kproj", g), KV_PROJ_PADDED)
            for g in range(_max_groups(dims, "kproj") - 1)
        ),
        *(
            (chain_run("vproj", g), KV_PROJ_PADDED)
            for g in range(_max_groups(dims, "vproj") - 1)
        ),
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
    "head": 128,
    "attn": 144,
    "merge": 128,
    "wo": 0,
    "fold_attn": 144,
    "red_attn": 128,
    "mix_mlp": 192,
    # A looped core's later passes: the inputs the regular path gated and
    # normalized (``later_pass_plan``).
    "in_attn": 192,
    "in_mlp": 192,
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
    dims: Dims, name: str, n: int, rows: int, base: int, x: str, out: str
) -> tuple[Stage, ...]:
    """The stages of projection ``name`` (``n`` output columns) at ``rows``
    rows: its K split groups on consecutive CTAs from ``base``, the last
    group in the stage ``name`` that writes ``out``. A last task past ``n``
    reads the packed weight's zero rows (``QKVR_PADDED``) and writes into
    its region's padding."""
    blocks, groups = projection_shape(dims, name, rows)
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
    dims: Dims, rows: int, chain: int, experts: str, boundary: bool, first: bool
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
    needs the same offsets in every plan (of one ``dims``)."""
    return _layer_plan(dims, rows, chain, experts, boundary, first, None, 1)


# The rows of a verify block a full layer's attention task takes
# (``attention.shared_attention_stage``: each split's keys read once for
# them), a row count: 8, or half a block where a block a task would leave
# the grid short of tasks; 0 is a row a task. The values come from cold
# timings of the attention and merge stages at 9k and 50k keys.
SHARED_ATTENTION = {1: 0, 4: 0, 8: 4, 16: 4, 32: 8, 64: 8}


def shared_attention(rows: int, chain: int) -> int:
    """The rows a full layer's attention task shares at ``rows`` rows: a
    verify block's 8 or half of them (``SHARED_ATTENTION``), or 0 for a row
    a task."""
    return SHARED_ATTENTION[rows] if chain == 8 else 0


def attention_tasks(dims: Dims, rows: int, chain: int, full_splits: int | None) -> int:
    """Attention tasks of a launch: a local layer's (row, KV head, split);
    a full layer's (verify block, KV head, split) when it shares its tasks
    (:func:`shared_attention`), else (row, KV head, split)."""
    if full_splits is None:
        return dims.kv_heads * LOCAL_SPLITS * rows
    if shared_attention(rows, chain):
        return dims.kv_heads * full_splits * (rows // shared_attention(rows, chain))
    return dims.kv_heads * full_splits * rows


def full_layer_plan(
    dims: Dims, rows: int, chain: int, experts: str, splits: int
) -> Plan:
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
    return _layer_plan(dims, rows, chain, experts, False, False, splits, 3)


def _layer_plan(
    dims: Dims,
    rows: int,
    chain: int,
    experts: str,
    boundary: bool,
    first: bool,
    full_splits: int | None,
    sources: int,
) -> Plan:
    """A local layer's plan, or with ``full_splits`` a full layer's.
    ``sources`` is 3 for a layer whose Q, K and V inputs differ (a full
    layer, and a local one in the later passes of a looped core), else 1."""
    if experts not in EXPERT_FORMATS:
        raise ValueError(f"no Inkling2 layer plan for {experts} experts")
    full = full_splits is not None
    if full and boundary:
        raise ValueError("a full layer is never a block boundary")
    if first and not boundary:
        raise ValueError("layer 0 starts an attention-residual block")
    if sources not in (1, 3) or (full and sources != 3):
        raise ValueError(f"a {'full' if full else 'local'} layer of {sources} sources")
    three = sources == 3
    # A local layer's K and V sources project its K and V rows only; a full
    # layer's add their conv taps.
    kv_n = dims.kv_proj if full else dims.kv_dim
    nvfp4 = experts == "nvfp4"
    # A full layer's K and V source projections: on the CTAs past the Q
    # source's tasks (the V source's past the K source's, the last CTA at
    # most) when it fits the grid in one group; else (K split groups)
    # beside its first group, whose CTAs it frees first.
    kv_proj_blocks = projection_shape(dims, "kproj", rows)[0]
    if projection_shape(dims, "qkvr", rows)[1] == 1:
        q_tasks = -(-dims.full_qkvr // (16 * projection_shape(dims, "qkvr", rows)[0]))
        kv_tasks = -(-dims.kv_proj // (16 * kv_proj_blocks))
        kv_bases = (q_tasks, min(q_tasks + kv_tasks, GRID - 1))
    else:
        kv_bases = (0, 4 * (dims.kv_proj_padded // (16 * kv_proj_blocks)))
    xa = ("xa",) if nvfp4 else ()
    ha = ("ha",) if nvfp4 else ()
    b = LOCAL_BASES
    stages = (
        Stage(
            "norm_attn" if first else "mix_attn",
            3 * rows if three else rows,
            b["norm_attn" if first else "mix_attn"],
            (),
            ("x_attn", "x_k", "x_v", "cand") if three else ("x_attn", "cand"),
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
            dims,
            "qkvr",
            dims.full_qkvr if full else dims.qkvr,
            rows,
            b["qkvr"],
            "x_attn",
            "qkvr",
        ),
        *(
            (
                *_chained(dims, "kproj", kv_n, rows, kv_bases[0], "x_k", "kproj"),
                *_chained(dims, "vproj", kv_n, rows, kv_bases[1], "x_v", "vproj"),
            )
            if three
            else ()
        ),
        Stage(
            "head",
            head_tasks(dims, rows),
            b["head"],
            ("qkvr", "kproj", "vproj") if three else ("qkvr",),
            ("q", "kv", "rel"),
        ),
        Stage(
            "attn",
            attention_tasks(dims, rows, chain, full_splits),
            b["attn"],
            ("q", "kv", "rel"),
            ("attn_part", "attn_lse"),
        ),
        Stage(
            "merge",
            dims.kv_heads * rows,
            b["merge"],
            ("attn_part", "attn_lse", "qkvr"),
            ("o", "o_flag"),
        ),
        Stage("wo", wo_tasks(dims), b["wo"], ("o", "o_flag"), ("wo_part", "wo_flag")),
        # Scattered (WO_SCATTER): each rank folds the partials of the column
        # blocks it owns and gathers the folded rows to every rank. The
        # reduce tasks run the output conv of their channels (conv.py).
        *(
            (
                Stage(
                    "fold_attn",
                    wo_tasks(dims) // dims.world,
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
            wo_tasks(dims),
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
        *_chained(dims, "router", dims.routed, rows, b["router"], "x_mlp", "logits"),
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
            "fold",
            fold_tasks(dims),
            b["fold"],
            ("route", "ex2"),
            ("fold_part", "fold_flag"),
        ),
        Stage(
            "red_moe",
            fold_tasks(dims),
            b["red_moe"],
            ("fold_part", "fold_flag"),
            ("mlp_sc",),
        ),
    )
    return Plan(
        grid=GRID,
        stages=stages,
        regions=regions(dims),
        rows=rows,
        chain=chain,
        world=dims.world,
        peers=LOCAL_PEERS,
        flags=LOCAL_FLAGS,
        dims=dims,
    )


# A looped core's later passes gate a message from the previous pass into a
# layer's Q, K and V inputs before their norms, and into its MoE input,
# which mixes the attention output, before its norm. The regular path
# computes those gated, normalized inputs between two launches of the
# layer's visit: the attention part takes the Q, K and V inputs (three
# sources, local layers too) and returns the attention output ``A``, the
# MoE part takes the MoE input and returns ``M``. Each part is a slice of
# the layer's three-source plan whose first stage reads its inputs from
# memory (``in_attn``, ``in_mlp``), with every plan's regions.
LATER_PARTS = ("attention", "mlp")
_ATTENTION_PART = (
    "qkvr",
    "kproj",
    "vproj",
    "head",
    "attn",
    "merge",
    "wo",
    "fold_attn",
    "red_attn",
)


def later_pass_plan(
    dims: Dims,
    rows: int,
    chain: int,
    experts: str,
    full_splits: int | None,
    part: str,
) -> Plan:
    """Part ``part`` (``LATER_PARTS``) of a later pass's launch of a local
    layer, or with ``full_splits`` a full layer (``experts`` as in
    :func:`local_layer_plan`; the attention part runs no expert stage)."""
    if part not in LATER_PARTS:
        raise ValueError(f"no later-pass part {part!r}")
    if full_splits is not None and not 1 <= full_splits <= FULL_SPLITS:
        raise ValueError(
            f"a full layer's split count {full_splits} is not 1 .. {FULL_SPLITS}"
        )
    layer = _layer_plan(dims, rows, chain, experts, False, False, full_splits, 3)
    b = LOCAL_BASES
    if part == "attention":
        first = Stage("in_attn", 3 * rows, b["in_attn"], (), ("x_attn", "x_k", "x_v"))
        # A projection's K split groups (``chain_stage``) go with it.
        kept = tuple(
            s for s in layer.stages if re.sub(r"_k\d+$", "", s.name) in _ATTENTION_PART
        )
    else:
        first = Stage("in_mlp", rows, b["in_mlp"], (), ("x_mlp",))
        start = [s.name for s in layer.stages].index("mix_mlp") + 1
        kept = layer.stages[start:]
    return Plan(
        grid=GRID,
        stages=(first, *kept),
        regions=layer.regions,
        rows=rows,
        chain=chain,
        world=dims.world,
        peers=LOCAL_PEERS,
        flags=LOCAL_FLAGS,
        dims=dims,
    )
