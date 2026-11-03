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
    iris_muse_moetail_push,
)
from tokenspeed_kernel.ops.communication._iris.muse_spark.routed_moe import (  # noqa: E402
    iris_column_routes_combine,
    iris_column_routes_complete,
    iris_column_routes_push,
    iris_column_routes_wait,
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
        options={"num_warps": num_warps},
    ).asm["amdgcn"]


@gluon.jit
def _own4_kernel(routes, norm_weight, region, regions, row, copy, RANK: gl.constexpr):
    """H4 push of one route slice, then OWN4: complete + H5, as in the layer."""
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [64, 1], [gl.num_warps(), 1], [1, 0]
    )
    lane = gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    word = gl.arange(0, 4, layout=gl.SliceLayout(0, layout))
    words = lane[:, None] * 4 + word[None, :]
    partial = gl.load(routes.to(gl.pointer_type(gl.uint32), bitcast=True) + words)
    iris_column_routes_push(partial, row, 1, copy, regions, RANK, 8, 32, _STAGE, ".wt")
    iris_column_routes_complete(
        partial, copy, norm_weight, region, regions, row, RANK, 8, 32,
        _STAGE, 0.0, ".wt", False,
    )  # fmt: skip


@gluon.jit
def _fin_kernel(route_weights, norm_weight, region, out, token, owner, copy):
    """FIN for (token, owner): wait + routed norm + weighted sum."""
    loaded, squares = iris_column_routes_wait(
        region, copy, token, owner, 8, 32, _STAGE, False
    )
    combined = iris_column_routes_combine(
        loaded, squares, route_weights, norm_weight, token, owner, 1e-6, 0.0
    )
    column = gl.arange(0, 512, layout=combined.type.layout)
    gl.store(out + column, combined)


def test_column_route_owner_compiles_at_8_waves():
    assembly = _compile(
        _own4_kernel,
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
        _fin_kernel,
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
def _xn_kernel(local_input, weight, packed, scales, region, regions, row, copy,RANK: gl.constexpr):  # fmt: skip
    """H3 push of one row's 512 reduced columns, then XN: wait, RMSNorm, MXFP4."""
    STAGE_WORDS: gl.constexpr = 8 * 16384 // 2
    stages = (copy.to(gl.uint32),)
    local_words = tl.cast(local_input, gl.pointer_type(gl.uint32)) + row * 256
    iris_lamport_allgather_push(
        local_words, row * 256, stages, regions, RANK, 8, 512, STAGE_WORDS, 8192, ".wt"
    )
    words = iris_lamport_allgather_wait(
        region, local_words, row * 256, stages, RANK, 8, 512, 4096, STAGE_WORDS, 8192,
        False, True,
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


def test_h3_push_and_xn_compile_at_8_waves():
    assembly = _compile(
        _xn_kernel,
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
def _h6_kernel(values_ptr, regions, copy, RANK: gl.constexpr):
    """H6: one UNP sender's 16 columns of 8 rows pushed to every rank."""
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 16], [64, 1], [gl.num_warps(), 1], [1, 0]
    )
    rows = gl.arange(0, 8, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, 16, layout=gl.SliceLayout(0, layout))
    values = gl.load(values_ptr + rows[:, None] * 16 + cols[None, :])
    iris_muse_moetail_push(
        values, copy.to(gl.uint32), 32, 5, regions, 8, _iris_lamport_push, RANK, 8,
        1024, 32768, 8, 16, gl.num_warps(), ".wt", ".wt",
    )  # fmt: skip


def test_h6_push_compiles_at_8_waves():
    assembly = _compile(
        _h6_kernel, {"values_ptr": "*bf16", "regions": ("i64",) * 8, "copy": "i32"},
        {"RANK": 3},
    )  # fmt: skip
    lines = instruction_lines(assembly)
    pushes = [
        l for l in lines if l.startswith("buffer_store") and l.endswith("sc0 sc1")
    ]
    assert len(pushes) >= 8  # 7 peers + the own copy
    first, last = lines.index(pushes[0]), lines.index(pushes[-1])
    assert not any("vmcnt" in l for l in lines[first:last])
