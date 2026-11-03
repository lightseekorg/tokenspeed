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

"""Offline gfx950 compile of the native MoE stream: the ring contract.

Compiles the standalone launch for gfx950 without a GPU and checks, in each
plan variant, that the gate/up and down item loops keep their weight ring:
no workgroup barrier and no ``vmcnt(0)`` inside them, and after every wait
in them at least ``RING`` vector-memory operations of the wave still in
flight (``RING - 1`` in a down unit, whose A operand rides in the same
queue). Also: no scratch, 1 KB weight chunks as ``buffer_load_dwordx4 nt``,
and a group tag stored only after a wait that covers its wave's mids stores.
"""

import re

import pytest

pytest.importorskip("tokenspeed_kernel_amd")

from tokenspeed_kernel_amd._triton import gl, gluon, triton  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4 import native_stream  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.persistent import (  # noqa: E402
    persistent_compile_options,
)

_LOAD = re.compile(r"^(buffer|global)_load_\w+")
_STORE = re.compile(r"^(buffer|global)_store_\w+")
_DIVISIBLE = {
    "stride_w13",
    "stride_s13",
    "stride_s13nb",
    "stride_w2",
    "stride_s2",
    "stride_s2nb",
    "stride_xsnb",
}

VARIANTS = [
    dict(ROWS=8, RING=8, QUAD_MAP=0, DN_ORDER=0, ADDR_SOFFSET=0, SCALE_LANES=1),
    dict(ROWS=8, RING=8, QUAD_MAP=0, DN_ORDER=0, ADDR_SOFFSET=0, SCALE_LANES=0),
    dict(ROWS=16, RING=8, QUAD_MAP=1, DN_ORDER=1, ADDR_SOFFSET=1, SCALE_LANES=0),
    dict(ROWS=16, RING=16, QUAD_MAP=0, DN_ORDER=0, ADDR_SOFFSET=0, SCALE_LANES=0),
]


@gluon.jit
def _pair_word_sink(args, y, route, cols):
    """A Lamport-style sink like the layer's H4 push: whole bf16 pairs as
    32-bit words, one relaxed system-scope store each, all in the lane."""
    words_ptr = args[0].to(gl.pointer_type(gl.int32), bitcast=True)
    zeros = gl.zeros(y.shape, gl.int32, y.type.layout)
    low, high = gl.split(gl.reshape(y.to(gl.bfloat16), [16, 64, 2]))
    column, _ = gl.split(gl.reshape(cols + zeros, [16, 64, 2]))
    row, _ = gl.split(gl.reshape(route + zeros, [16, 64, 2]))
    words = low.to(gl.uint16, bitcast=True).to(gl.uint32) | (
        high.to(gl.uint16, bitcast=True).to(gl.uint32) << 16
    )
    gl.atomic_store(
        words_ptr + row * 2048 + column // 2,
        words.to(gl.int32, bitcast=True),
        mask=row >= 0,
        sem="relaxed",
        scope="sys",
    )


def _compile(fields, sink=native_stream.store_routes_sink):
    fn = native_stream.gluon_mxfp4_moe_native_stream_gfx950
    const = dict(
        SINK=sink,
        N_PHYS1=1024,
        N_PHYS2=4096,
        TOPK=8,
        TAG_GLC=1,
        TAIL_NUM=1,
        TAIL_DEN=2,
        SWIGLU_ALPHA=1.0,
        SWIGLU_LIMIT=0.0,
        SWIGLU_BETA=0.0,
        **fields,
    )
    types = dict(
        epochs="*i64",
        claims="*i64",
        errors="*i32",
        arena="*i64",
        ids="*i32",
        xq="*u8",
        xs="*u8",
        w13="*u8",
        s13_lanes="*i32",
        s13="*i32",
        w2="*u8",
        s2="*i32",
        mids="*u8",
        mids_s="*i32",
        tags="*i32",
        out="*bf16",
    )
    signature = {n: types.get(n, "i32") for n in fn.arg_names if n not in const}
    attrs = {
        (i,): [["tt.divisibility", 16]]
        for i, n in enumerate(fn.arg_names)
        if signature.get(n, "").startswith("*") or n in _DIVISIBLE
    }
    source = gluon._runtime.GluonASTSource(fn, signature, constexprs=const, attrs=attrs)
    target = triton.backends.compiler.GPUTarget("hip", "gfx950", 64)
    return triton.compile(
        source, target=target, options={"num_warps": 8, **persistent_compile_options()}
    ).asm["amdgcn"]


def _instructions(assembly):
    out = []
    for line in assembly.splitlines():
        text = line.strip()
        if (
            not text
            or text.startswith((";", ".loc"))
            or (text.startswith(".") and not text.startswith(".LBB"))
        ):
            continue
        marker = "after" if re.search(r";\s*after\b", text) else ""
        out.append((text.split(";")[0].strip(), marker))
    return out


def _item_loops(ins):
    """The smallest loop around each group of MFMAs: (start, end, mfma count)."""
    labels = {s[:-1]: i for i, (s, _) in enumerate(ins) if re.match(r"^\.LBB\w+:$", s)}
    loops = []
    for i, (s, _) in enumerate(ins):
        m = re.match(r"^s_(?:cbranch_\w+|branch)\s+(\.LBB\w+)", s)
        if m and m.group(1) in labels and labels[m.group(1)] < i:
            start = labels[m.group(1)]
            mfma = sum("v_mfma" in x for x, _ in ins[start:i])
            if mfma:
                loops.append((start, i, mfma))
    smallest = {}
    for start, end, mfma in loops:
        if mfma not in smallest or end - start < smallest[mfma][1] - smallest[mfma][0]:
            smallest[mfma] = (start, end)
    return smallest


def _walk(seg):
    """Two trips around a loop body (the second starts in the first's end
    state). Returns the second trip's waits as (vmcnt, outstanding before),
    its barrier and vmcnt(0) counts, and its tag stores as booleans: whether
    a wait since the wave's last data store covers that store."""
    outstanding, since_store, covered = 0, None, True
    waits, barriers, drains, tags = [], 0, 0, []
    for trip in range(2):
        tag_next = False
        for s, marker in seg:
            op = s.split()[0]
            if marker == "after":
                tag_next = True
            if _LOAD.match(op) or _STORE.match(op):
                outstanding += 1
                if since_store is not None:
                    since_store += 1
                if _STORE.match(op):
                    if tag_next:
                        if trip:
                            tags.append(covered)
                        tag_next = False
                    else:
                        since_store, covered = 0, False
            if op == "s_waitcnt" and (m := re.search(r"vmcnt\((\d+)\)", s)):
                n = int(m.group(1))
                if trip:
                    waits.append((n, outstanding))
                    drains += n == 0
                if since_store is not None and n < since_store:
                    covered = True
                outstanding = min(outstanding, n)
            if trip and op == "s_barrier":
                barriers += 1
    return waits, barriers, drains, tags


@pytest.mark.parametrize(
    "fields", VARIANTS, ids=lambda f: "-".join(f"{k}{v}" for k, v in f.items())
)
def test_item_loops_keep_the_ring(fields):
    assembly = _compile(fields)
    assert (
        int(re.search(r"\.private_segment_fixed_size:\s+(\d+)", assembly).group(1)) == 0
    )
    assert int(re.search(r"\.vgpr_spill_count:\s+(\d+)", assembly).group(1)) == 0
    ins = _instructions(assembly)
    loops = _item_loops(ins)
    ring = fields["RING"]
    for mfma, floor in ((32, ring), (16, ring - 1)):
        start, end = loops[mfma]
        waits, barriers, drains, tags = _walk(ins[start : end + 1])
        assert barriers == 0, f"{mfma}-MFMA loop holds a workgroup barrier"
        assert drains == 0, f"{mfma}-MFMA loop drains vmcnt"
        assert min(n for n, before in waits if n < before) >= floor, (mfma, waits)
        chunks = [
            s
            for s, _ in ins[start:end]
            if s.startswith("buffer_load_dwordx4") and s.endswith(" nt")
        ]
        assert (
            len(chunks) == mfma
        ), f"{mfma}-MFMA loop issues {len(chunks)} weight chunks"
        if mfma == 32:
            # S-MOE2: each wave's group tag follows a wait covering its mids stores.
            assert tags and all(tags), tags
            # Gate/up scales: two lane-major dwordx4 per strip, or one
            # swizzled dword per two steps.
            body = [s for s, _ in ins[start:end]]
            if fields["SCALE_LANES"]:
                scales = [
                    s
                    for s in body
                    if s.startswith("buffer_load_dwordx4") and not s.endswith(" nt")
                ]
                assert len(scales) == 2, scales
            else:
                assert sum(s.startswith("buffer_load_dword ") for s in body) == 16


def test_down_loop_sink_stays_in_the_wave():
    """A sink that packs column pairs works in the handed layout: the down
    loop (per-wave trip counts) gets no barrier and no LDS round trip."""
    fields = dict(ROWS=8, RING=8, QUAD_MAP=0, DN_ORDER=0, ADDR_SOFFSET=0, SCALE_LANES=0)
    plain = _compile(fields)
    paired = _compile(fields, _pair_word_sink)
    ins = _instructions(paired)
    start, end = _item_loops(ins)[16]
    body = [s for s, _ in ins[start : end + 1]]
    assert not any(s.startswith("s_barrier") for s in body)
    assert not any(s.startswith("ds_write") for s in body)
    vgprs = [
        int(re.search(r"\.vgpr_count:\s+(\d+)", a).group(1)) for a in (plain, paired)
    ]
    assert vgprs[1] <= vgprs[0] + 8, vgprs
