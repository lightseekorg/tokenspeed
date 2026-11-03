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

"""Offline gfx950 codegen tests of the persistent sync helpers.

They compile for gfx950 without a device (GPUs may be hidden) and check the
instruction-level contract of each hand-off primitive: which waits, barriers
and cache bits it emits, and which it must not.
"""

import re

import pytest
from asm_checks import instruction_lines

triton = pytest.importorskip("tokenspeed_triton")
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.persistent._intrinsics import (  # noqa: E402
    persistent_compile_options,
    release_drain,
    wait_vmcnt,
)


def _compile(fn, signature, constexprs=None, num_warps=8):
    constexprs = constexprs or {}
    signature = dict(signature)
    signature.update({name: "constexpr" for name in constexprs})
    attrs = {
        (index,): [["tt.divisibility", 16]]
        for index, name in enumerate(fn.arg_names)
        if signature.get(name, "").startswith("*")
    }
    source = gluon._runtime.GluonASTSource(
        fn, signature, constexprs=constexprs, attrs=attrs
    )
    kernel = triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": num_warps, **persistent_compile_options()},
    )
    return kernel.asm["amdgcn"]


@gluon.jit(do_not_specialize=["tag"])
def _xcd_plain_producer(payload, flag, src, tag):
    lane = gl.arange(0, 512, gl.BlockedLayout([1], [64], [8], [0]))
    gl.amd.cdna4.buffer_store(gl.load(src + lane), payload, lane)
    release_drain()
    gl.barrier()
    gl.atomic_store(flag, tag, sem="relaxed", scope="cta")


def test_release_drain_orders_plain_payload_before_the_flag():
    assembly = _compile(
        _xcd_plain_producer,
        {"payload": "*i32", "flag": "*i32", "src": "*i32", "tag": "i32"},
    )
    lines = instruction_lines(assembly)
    store = next(i for i, l in enumerate(lines) if l.startswith("buffer_store"))
    assert not re.search(r"\bsc[01]\b|\bnt\b", lines[store])
    drain = lines.index("s_waitcnt vmcnt(0)", store)
    barrier = lines.index("s_barrier", drain)
    flag = next(i for i in range(barrier, len(lines)) if lines[i].startswith("global_store"))
    assert lines[flag].endswith(" sc0")
    assert "buffer_wbl2" not in assembly and "buffer_inv" not in assembly


@gluon.jit
def _paced_kernel(w_ptr, out, W: gl.constexpr, LOADS: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [64, 1], [8, 1], [1, 0])
    rows = gl.arange(0, 512, gl.SliceLayout(1, layout))
    cols = gl.arange(0, 4, gl.SliceLayout(0, layout))
    offs = rows[:, None] * 4 + cols[None, :]
    loaded = ()
    for i in gl.static_range(LOADS):
        if i >= W:
            wait_vmcnt(W - 1)
        loaded += (gl.amd.cdna4.buffer_load(w_ptr + i * 65536, offs, cache=".cg"),)
    total = loaded[0]
    for i in gl.static_range(1, LOADS):
        total += loaded[i]
    gl.amd.cdna4.buffer_store(total, out, offs)


def test_paced_issue_keeps_w_loads_in_flight():
    assembly = _compile(_paced_kernel, {"w_ptr": "*fp32", "out": "*fp32"}, {"W": 4, "LOADS": 12})
    lines = instruction_lines(assembly)
    loads = [i for i, l in enumerate(lines) if l.startswith("buffer_load")]
    assert len(loads) == 12
    for previous, current in zip(loads[3:], loads[4:]):
        assert "s_waitcnt vmcnt(3)" in lines[previous:current]
    assert not any("vmcnt(0)" in l for l in lines[: loads[-1]])
