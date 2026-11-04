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

"""Offline gfx950 codegen of the native fused layer's Iris hop bodies.

The hop bodies of the standalone kernels run inside the 8-wave fused-layer
launch. These tests compile them at 8 waves (no device needed) and check the
ordering contract of each: pushes are whole-line system-scope stores with no
wait, polls stay in their loop with system-scope loads, and with the entry
clear (CLEAR False) a body stores no empty markers.
"""

import re

import pytest
from asm_checks import innermost_loops, instruction_lines

triton = pytest.importorskip("tokenspeed_triton")
from tokenspeed_kernel._triton import gl, gluon, tl  # noqa: E402
from tokenspeed_kernel.ops.communication._iris.all_gather import (  # noqa: E402
    iris_lamport_allgather_push,
    iris_lamport_allgather_wait,
    iris_rmsnorm_mxfp4_row,
)
from tokenspeed_kernel.ops.communication._iris.all_reduce import (  # noqa: E402
    _iris_lamport_push,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.moetail import (  # noqa: E402
    iris_muse_moetail_send,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.routed_moe import (  # noqa: E402
    iris_column_routes_combine,
    iris_column_routes_complete,
    iris_column_routes_push_words,
    iris_column_routes_wait,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.persistent import (  # noqa: E402
    persistent_compile_options,
)

_WORLD = 8
_STAGE = gl.constexpr(32 * ((2 * _WORLD - 1) * 2048 + _WORLD * _WORLD * 8))


def _compile(fn, signature, constexprs, num_warps=8):
    attrs = {}
    for index, name in enumerate(fn.arg_names):
        kind = signature.get(name)
        if isinstance(kind, tuple):
            attrs.update(
                {(index, p): [["tt.divisibility", 16]] for p in range(len(kind))}
            )
        elif isinstance(kind, str) and kind.startswith("*"):
            attrs[(index,)] = [["tt.divisibility", 16]]
    signature = dict(signature)
    signature.update({name: "constexpr" for name in constexprs})
    source = gluon._runtime.GluonASTSource(
        fn, signature, constexprs=constexprs, attrs=attrs
    )
    return triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": num_warps, **persistent_compile_options()},
    ).asm["amdgcn"]


@gluon.jit
def _route_owner_kernel(
    routes, norm_weight, region, regions, row, copy, RANK: gl.constexpr
):
    """ROUTE_PUSH push of one route slice, then ROUTE_OWNER: complete + ROUTE_AG, as in the layer."""
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [64, 1], [gl.num_warps(), 1], [1, 0]
    )
    lane = gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    word = gl.arange(0, 4, layout=gl.SliceLayout(0, layout))
    words = lane[:, None] * 4 + word[None, :]
    partial = gl.load(routes.to(gl.pointer_type(gl.uint32), bitcast=True) + words)
    iris_column_routes_push_words(
        partial, row, words, (RANK + 1) % 8, copy, words >= 0, regions, RANK, 8,
        32, _STAGE, ".wt"
    )  # fmt: skip
    iris_column_routes_complete(
        partial, copy, norm_weight, region, regions, row, RANK, 8, 32,
        _STAGE, 0.0, ".wt", False, None, None, 0,
    )  # fmt: skip


@gluon.jit
def _finisher_kernel(route_weights, norm_weight, region, out, token, owner, copy):
    """FINISHER for (token, owner): wait + routed norm + weighted sum."""
    loaded, squares = iris_column_routes_wait(
        region, copy, token, owner, 8, 32, _STAGE, False, None, None, 0
    )
    combined = iris_column_routes_combine(
        loaded, squares, route_weights, norm_weight, token, owner, 1e-6, 0.0
    )
    column = gl.arange(0, 512, layout=combined.type.layout)
    gl.store(out + column, combined)


def test_column_route_owner_compiles_at_8_waves():
    assembly = _compile(
        _route_owner_kernel,
        {"routes": "*bf16", "norm_weight": "*bf16", "region": "*i32",
         "regions": ("i64",) * 8, "row": "i32", "copy": "i32"},
        {"RANK": 3},
    )  # fmt: skip
    lines = instruction_lines(assembly)
    pushes = [
        l for l in lines if l.startswith("buffer_store") and l.endswith("sc0 sc1")
    ]
    assert pushes and all(
        "dwordx4" in l or "dwordx2" in l or "dword " in l for l in pushes
    )
    # With the entry clear, the owner stores no empty markers (0x80000000).
    assert "0x80000000" not in assembly
    polls = [
        l
        for l in innermost_loops(assembly)
        if re.search(r"buffer_load_dword\w* .*sc0 sc1", l)
    ]
    assert polls, "the peer slot loads left the poll loop"


def test_column_route_finisher_compiles_at_8_waves():
    assembly = _compile(
        _finisher_kernel,
        {"route_weights": "*fp32", "norm_weight": "*bf16", "region": "*i32",
         "out": "*fp32", "token": "i32", "owner": "i32", "copy": "i32"},
        {},
    )  # fmt: skip
    assert "0x80000000" not in assembly
    polls = [
        l
        for l in innermost_loops(assembly)
        if re.search(r"buffer_load_dword\w* .*sc0 sc1", l)
    ]
    assert polls


@gluon.jit
def _pooled_norm_kernel(local_input, weight, packed, scales, region, regions, row, copy,RANK: gl.constexpr):  # fmt: skip
    """POOLED_AG push of one row's 512 reduced columns, then POOLED_NORM: wait, RMSNorm, MXFP4."""
    STAGE_WORDS: gl.constexpr = 8 * 16384 // 2
    stages = (copy.to(gl.uint32),)
    local_words = tl.cast(local_input, gl.pointer_type(gl.uint32)) + row * 256
    iris_lamport_allgather_push(
        local_words, row * 256, stages, regions, RANK, 8, 512, STAGE_WORDS, 8192, ".wt"
    )
    words = iris_lamport_allgather_wait(
        region, local_words, row * 256, stages, RANK, 8, 512, 4096, STAGE_WORDS, 8192,
        False, True, None, None, 0,
    )  # fmt: skip
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 16], [1, 64], [gl.num_warps(), 1], [1, 0]
    )
    columns = gl.arange(0, 1024, layout=gl.SliceLayout(0, layout))
    norm_weights = ()
    for rep in gl.static_range(4):
        norm_weights += (gl.load(weight + rep * 1024 + columns).to(gl.float32),)
    iris_rmsnorm_mxfp4_row(
        words, row, norm_weights, packed, scales, 4096, 1e-6, 0.0, 128, 4096
    )


def test_pooled_ag_push_and_pooled_norm_compile_at_8_waves():
    assembly = _compile(
        _pooled_norm_kernel,
        {"local_input": "*bf16", "weight": "*bf16", "packed": "*u8", "scales": "*u8",
         "region": "*bf16", "regions": ("i64",) * 8, "row": "i32", "copy": "i32"},
        {"RANK": 3},
    )  # fmt: skip
    assert "0x80000000" not in assembly
    polls = [
        l
        for l in innermost_loops(assembly)
        if re.search(r"buffer_load_dword\w* .*sc0 sc1", l)
    ]
    assert polls


@gluon.jit
def _post_ag_kernel(values_ptr, regions, copy, RANK: gl.constexpr):
    """POST_AG: one POST_SENDER sender's 16 columns of 8 rows pushed to every rank."""
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 16], [64, 1], [gl.num_warps(), 1], [1, 0]
    )
    rows = gl.arange(0, 8, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, 16, layout=gl.SliceLayout(0, layout))
    values = gl.load(values_ptr + rows[:, None] * 16 + cols[None, :])
    iris_muse_moetail_send(
        values, 32, 5, copy.to(gl.uint32), regions, 8, _iris_lamport_push, RANK, 8,
        1024, 32768, 8, 16, gl.num_warps(), ".wt", ".wt",
    )  # fmt: skip


def test_post_ag_push_compiles_at_8_waves():
    assembly = _compile(
        _post_ag_kernel, {"values_ptr": "*bf16", "regions": ("i64",) * 8, "copy": "i32"},
        {"RANK": 3},
    )  # fmt: skip
    lines = instruction_lines(assembly)
    pushes = [
        l for l in lines if l.startswith("buffer_store") and l.endswith("sc0 sc1")
    ]
    assert len(pushes) >= 8  # 7 peers + the own copy
    first, last = lines.index(pushes[0]), lines.index(pushes[-1])
    assert not any("vmcnt" in l for l in lines[first:last])


from tokenspeed_kernel.ops.monokernel._gluon.muse_spark.iris import (  # noqa: E402
    MuseSparkIrisView,
    RoutePushArgs,
    ar_owner,
    ar_rs_sink,
    clear_next_copies,
    finisher,
    iris_fail_stop,
    pooled_ag_sink,
    pooled_norm_wait,
    post_ag_sink,
    post_ag_sums_wait,
    post_sender_sums_publish,
    reducer_sums,
    route_owner_complete,
    route_push_sink,
)
from tokenspeed_kernel_amd._scheduling import wave_uniform_i32  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.persistent import (  # noqa: E402
    Launch,
    vote,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.persistent.wave import (  # noqa: E402
    wave_scalar,
)

_VIEW = {
    "heaps": ("i64",) * 8, "rank": "i32", "box": "*i32", "ar": "*bf16", "ar_sums": "*i64",
    "pooled_ag": "*bf16", "route_push_own": "*i32", "routes": "*i32", "post_ag": "*bf16",
    "reducer_in": "*i64", "post_ag_sums": "*i64",
}  # fmt: skip


@gluon.jit
def _layer_hops_kernel(heaps, rank, box, ar, ar_sums, pooled_ag, route_push_own, routes, post_ag, reducer_in,post_ag_sums, errors,floats, words, copy, rows, ROLE: gl.constexpr):  # fmt: skip
    """Each fused-layer hop end on the view, at 8 waves, by ROLE."""
    view = MuseSparkIrisView(
        heaps,
        rank,
        box,
        ar,
        ar_sums,
        pooled_ag,
        route_push_own,
        routes,
        post_ag,
        reducer_in,
        post_ag_sums,
    )
    zero = gl.program_id(0) * 0
    launch = Launch(
        gl.program_id(0), zero, zero, copy.to(gl.int64) + 2, zero + 7, copy, zero,
        errors, errors + 32, zero.to(gl.int64) + 1000, errors,
    )  # fmt: skip
    if ROLE == "clear":
        clear_next_copies(view, launch, gl.program_id(0))
    elif ROLE == "ar":
        layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [8, 1], [1, 0])
        r = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
        c = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
        out = gl.load(floats + r[:, None] * 64 + c[None, :])
        ar_rs_sink(view, launch, out, 5, rows, 3, ".wt")
    elif ROLE == "ar_owner":
        ar_owner(view, launch, 2, rows, floats, words.to(gl.pointer_type(gl.bfloat16),
             bitcast=True), floats, floats, iris_fail_stop, 3, 0.0, True, ".wt")  # fmt: skip
    elif ROLE == "pooled_ag":
        layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [8, 1], [1, 0])
        r = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
        c = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
        pooled_ag_sink(view, launch, gl.load(floats + r[:, None] * 64 + c[None, :]), 4, rows,
                16, 3, ".wt")  # fmt: skip
        got = pooled_norm_wait(view, launch, 1, iris_fail_stop, 3)
        gl.store(
            words
            + gl.arange(0, 64, gl.SliceLayout(0, gl.SliceLayout(2, got.type.layout))),
            gl.sum(gl.sum(got, 2), 0),
        )
    elif ROLE == "route_push":
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [1, 8], [0, 1])
        r = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
        c = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
        y = gl.load(floats + r[:, None] * 128 + c[None, :])
        route = gl.load(words + r)[:, None]
        # Each wave's first column, as the MoE stream hands it (per wave).
        wave = wave_scalar(
            gl.arange(0, 512, gl.BlockedLayout([1], [64], [8], [0])) // 64
        )
        route_push_sink(
            RoutePushArgs(view, launch, 3),
            (y,),
            route,
            ((1024 + c)[None, :],),
            1024 + 16 * wave,
        )
    elif ROLE == "route_owner":
        route_owner_complete(view, launch, rows, words.to(gl.pointer_type(gl.bfloat16),
                      bitcast=True), iris_fail_stop, 3, 0.0, ".wt")  # fmt: skip
    elif ROLE == "finisher":
        combined = finisher(view, launch, 1, 2, floats, words.to(gl.pointer_type(gl.bfloat16),
                       bitcast=True), iris_fail_stop, 1e-6, 0.0)  # fmt: skip
        gl.store(floats + gl.arange(0, 512, layout=combined.type.layout), combined)
    elif ROLE == "reducer":
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [8, 1], [1, 0])
        r = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
        c = gl.arange(0, 4, layout=gl.SliceLayout(0, layout))
        post_sender_sums_publish(
            view, launch, gl.load(floats + r[:, None] * 4 + c[None, :]), 3, rows
        )
        reducer_sums(view, launch, rows, iris_fail_stop, 3, ".wt")
        sums = post_ag_sums_wait(view, launch, 1, iris_fail_stop)
        s4 = gl.arange(0, 4, layout=gl.SliceLayout(0, sums.type.layout))
        gl.store(floats + s4, gl.sum(sums, 0))
    else:
        layout: gl.constexpr = gl.BlockedLayout([1, 16], [64, 1], [8, 1], [1, 0])
        r = gl.arange(0, 8, layout=gl.SliceLayout(1, layout))
        c = gl.arange(0, 16, layout=gl.SliceLayout(0, layout))
        values = gl.load(floats + r[:, None] * 16 + c[None, :]).to(gl.bfloat16)
        post_ag_sink(view, launch, values, 32, rows, 8, 16, 3, ".wt")


@pytest.mark.parametrize(
    "role",
    [
        "clear",
        "ar",
        "ar_owner",
        "pooled_ag",
        "route_push",
        "route_owner",
        "finisher",
        "post_ag",
        "reducer",
    ],
)
def test_fused_layer_hop_compiles_at_8_waves(role):
    """Every hop end of the fused layer compiles at 8 waves, with no L2
    invalidate or write-back, no scratch, and system-scope peer stores."""
    signature = dict(
        _VIEW, errors="*i32", floats="*fp32", words="*i32", copy="i32", rows="i32"
    )
    assembly = _compile(_layer_hops_kernel, signature, {"ROLE": role})
    assert "buffer_inv" not in assembly and "buffer_wbl2" not in assembly
    assert re.search(r"\.private_segment_fixed_size:\s+0|ScratchSize: 0", assembly)
    if role in ("ar_owner", "pooled_ag", "route_owner", "finisher", "reducer"):
        # Every poll loop reads the clock (deadline) on its retry path, and
        # the failure path traps after storing the peers' abort words.
        from asm_checks import _clock_on_retry_path, _innermost_loop_lines

        raw = assembly.splitlines()
        polls = [
            set(lines)
            for lines in _innermost_loop_lines(assembly).values()
            if any(re.search(r"_load_dword\w* .*sc1", raw[i]) for i in lines)
        ]
        assert polls and all(_clock_on_retry_path(raw, loop) for loop in polls)
        assert "s_trap 2" in assembly
    if role in (
        "ar",
        "ar_owner",
        "pooled_ag",
        "route_push",
        "route_owner",
        "post_ag",
        "reducer",
    ):
        assert re.search(r"(buffer|global)_(store|atomic)\w* .*sc0 sc1", assembly)
    if role == "clear":
        lines = instruction_lines(assembly)
        stores = [l for l in lines if l.startswith("buffer_store")]
        assert stores and all(
            l.startswith("buffer_store_dwordx4") and l.endswith("sc0 sc1")
            for l in stores
        )
        assert not any("vmcnt" in l for l in lines[lines.index(stores[0]) :])


def test_every_cleared_area_splits_into_whole_sender_shares():
    """128 senders clear equal 16-byte-multiple shares of every area copy, and
    the area sizes follow the 32-row geometry the hop bodies index."""
    from tokenspeed_kernel.ops.monokernel._gluon.muse_spark import iris as f

    for area in f.CLEARED_AREAS:
        assert area.copy_words % (f.SENDERS * 4) == 0, area
    # Column-route mailbox: peers' slices, completed slices, sums of squares.
    assert f.ROUTES_WORDS == 32 * (15 * 2048 + 8 * 8 * 8)
    # About 5.6 MB per copy in all, ~44 KB cleared per sender per launch.
    total = sum(4 * area.copy_words for area in f.CLEARED_AREAS)
    assert 5 << 20 < total < 6 << 20


@gluon.jit
def _route_push_per_wave_kernel(heaps, rank, box, ar, ar_sums, pooled_ag, route_push_own, routes, post_ag, reducer_in,post_ag_sums, errors, floats, words, copy, LINES: gl.constexpr):  # fmt: skip
    """route_push_sink inside a loop whose trip count differs per wave, as in the
    MoE down units: a workgroup barrier inside would deadlock. ``LINES``:
    one call per down unit with its 4 strips (the stream's whole-line
    column map), else one call per strip."""
    view = MuseSparkIrisView(
        heaps,
        rank,
        box,
        ar,
        ar_sums,
        pooled_ag,
        route_push_own,
        routes,
        post_ag,
        reducer_in,
        post_ag_sums,
    )
    zero = gl.program_id(0) * 0
    launch = Launch(
        gl.program_id(0), zero, zero, copy.to(gl.int64) + 2, zero + 7, copy, zero,
        errors, errors + 32, zero.to(gl.int64) + 1000, errors,
    )  # fmt: skip
    # The MoE stream's sink layout (MOE_SINK_LAYOUT): wave w's [16, 16] tile,
    # two whole column pairs per lane.
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [1, 8], [0, 1])
    r = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    c = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    lane = gl.arange(0, 512, gl.BlockedLayout([1], [64], [8], [0]))
    # This wave's index, as a per-wave scalar: wave w runs w + 1 units.
    units = wave_uniform_i32(gl.max(lane // 64, 0)) + 1
    wave = wave_scalar(lane // 64)
    for unit in range(units):
        route = gl.load(words + unit * 16 + r)[:, None]
        if LINES:
            # Strip s of wave w's down unit: lane column group k, value j
            # holds column 64 w + 32 (k // 2) + 8 (k % 2) + 16 (s % 2)
            # + s // 2 + 2 j.
            k = c % 16 // 4
            local = 32 * (k // 2) + 8 * (k % 2) + 2 * (c % 4)
            first = unit * 512 + 64 * (c // 16)
            ys = ()
            for s in gl.static_range(4):
                ys += (gl.load(floats + (4 * unit + s) * 2048 + r[:, None] * 128 + c[None, :]),)  # fmt: skip
            cols = ((first + local)[None, :], (first + local + 16)[None, :])
            cols += (cols[0] + 1, cols[1] + 1)
            route_push_sink(
                RoutePushArgs(view, launch, 3), ys, route, cols, unit * 512 + 64 * wave
            )
        else:
            y = gl.load(floats + unit * 2048 + r[:, None] * 128 + c[None, :])
            route_push_sink(
                RoutePushArgs(view, launch, 3), (y,), route, ((unit * 128 + c)[None, :],),
                unit * 128 + 16 * wave,
            )  # fmt: skip


@pytest.mark.parametrize("lines", [0, 1])
def test_route_push_sink_moves_no_data_between_lanes(lines):
    """The sink runs in per-wave code: no workgroup barrier and no LDS
    traffic, so waves with different trip counts never wait for each other."""
    signature = dict(_VIEW, errors="*i32", floats="*fp32", words="*i32", copy="i32")
    assembly = _compile(_route_push_per_wave_kernel, signature, {"LINES": lines})
    # The loop that pushes (finding the wave index before it needs one
    # cross-wave reduction, which is the test's, not the sink's).
    loops = [l for l in innermost_loops(assembly) if "sc0 sc1" in l]
    assert loops
    for loop in loops:
        assert "s_barrier" not in loop
        assert not re.search(r"\bds_(write|read|bpermute)", loop)


@pytest.mark.parametrize(
    "lines, stores",
    [(0, ["buffer_store_dwordx2"] * 2), (1, ["buffer_store_dwordx4"] * 4)],
)
def test_route_push_sink_stores_each_lanes_words_once(lines, stores):
    """A wave's columns have one owner (``col0``), so each lane's words go
    out once, to the own slice or to that owner's mailbox, never as one
    masked store per possible owner: one 8-byte store per strip, or one
    16-byte line per strip pair with the down unit's 4 strips."""
    signature = dict(_VIEW, errors="*i32", floats="*fp32", words="*i32", copy="i32")
    assembly = _compile(_route_push_per_wave_kernel, signature, {"LINES": lines})
    loops = [l for l in innermost_loops(assembly) if "sc0 sc1" in l]
    assert loops
    for loop in loops:
        found = re.findall(r"\b(?:buffer|global)_store_\w+", loop)
        assert sorted(found) == stores


@gluon.jit
def _fail_stop_kernel(errors, flags, copy):
    """iris_fail_stop called from the waves whose flag is set (a per-wave
    poll's expiry), not from the whole workgroup."""
    zero = gl.program_id(0) * 0
    launch = Launch(
        gl.program_id(0), zero, zero, copy.to(gl.int64) + 2, zero + 7, copy, zero,
        errors, errors + 32, zero.to(gl.int64) + 1000, errors,
    )  # fmt: skip
    lane = gl.arange(0, 512, gl.BlockedLayout([1], [64], [8], [0]))
    if vote(gl.load(flags + lane)):
        iris_fail_stop(launch, 40, 7)


def test_iris_fail_stop_runs_from_any_subset_of_waves():
    """No workgroup barrier or LDS exchange on the failure path (the other
    waves may never get there); the peers' abort words are swapped at system
    scope before the trap."""
    assembly = _compile(
        _fail_stop_kernel, {"errors": "*i32", "flags": "*i32", "copy": "i32"}, {}
    )
    assert "s_barrier" not in assembly
    assert not re.search(r"\bds_(write|read|bpermute)", assembly)
    lines = instruction_lines(assembly)
    trap = next(i for i, l in enumerate(lines) if l.startswith("s_trap"))
    # On atomics, sc1 is the system scope (sc0 only asks for the old value).
    assert any(re.match(r"global_atomic_cmpswap\w* .*\bsc1$", l) for l in lines[:trap])
