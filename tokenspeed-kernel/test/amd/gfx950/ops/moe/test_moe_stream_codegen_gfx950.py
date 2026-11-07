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


"""Offline gfx950 compile of the MXFP4 MoE stream kernel: the weight ring.

Compiles the kernel for gfx950 without a GPU, at both selected row counts
and both tail shares, and checks that the gate/up and down item loops keep
their weight ring: no workgroup barrier and no ``vmcnt(0)`` inside them, and
after every wait in them at least ``RING`` vector-memory operations of the
wave still in flight (``RING - 1`` in a down unit, whose A operand rides in
the same queue). Also: no scratch, a register budget that leaves one
8-wave workgroup per CU, 1 KB weight chunks as ``buffer_load_dwordx4 nt``,
and a group tag stored only after a wait that covers its wave's mids stores.
"""

import re

import pytest

pytest.importorskip("tokenspeed_kernel_amd", reason="AMD kernel package is optional")

from moe_stream_helpers import compile_moe_stream  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.moe.mxfp4 import moe_stream  # noqa: E402

_LOAD = re.compile(r"^(buffer|global)_load_\w+")
_STORE = re.compile(r"^(buffer|global)_store_\w+")

VARIANTS = [
    dict(ROWS=8, TAIL=moe_stream.TAIL_SHARE),
    dict(ROWS=16, TAIL=moe_stream.TAIL_SHARE),
    dict(ROWS=8, TAIL=(1, 1)),
]


def _compile(rows, tail):
    return compile_moe_stream(rows, tail).asm["amdgcn"]


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
    """The smallest loop around each group of MFMAs: (start, end) by count."""
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


def _field(assembly, name):
    return int(re.search(rf"\.{name}:\s+(\d+)", assembly).group(1))


@pytest.mark.parametrize(
    "variant",
    VARIANTS,
    ids=lambda v: f"rows{v['ROWS']}-tail{v['TAIL'][0]}of{v['TAIL'][1]}",
)
def test_item_loops_keep_the_ring(variant):
    assembly = _compile(variant["ROWS"], variant["TAIL"])
    assert _field(assembly, "private_segment_fixed_size") == 0
    assert _field(assembly, "vgpr_spill_count") == 0
    # 8 waves on 4 SIMDs: two waves per SIMD share its 512 registers.
    assert _field(assembly, "vgpr_count") + _field(assembly, "agpr_count") <= 256
    ins = _instructions(assembly)
    loops = _item_loops(ins)
    ring = moe_stream.RING
    for mfma, floor in ((32, ring), (16, ring - 1)):
        start, end = loops[mfma]
        waits, barriers, drains, tags = _walk(ins[start : end + 1])
        assert barriers == 0, f"{mfma}-MFMA loop holds a workgroup barrier"
        assert drains == 0, f"{mfma}-MFMA loop drains vmcnt"
        assert min(n for n, before in waits if n < before) >= floor, (mfma, waits)
        body = [s for s, _ in ins[start:end]]
        # Weight chunks are read once: every dwordx4 load in an item loop is
        # nontemporal except the system-scope mids reads of the down units.
        wide = [
            s
            for s in body
            if s.startswith("buffer_load_dwordx4") and not s.endswith(" sc1")
        ]
        assert all(s.endswith(" nt") for s in wide), wide
        assert len(wide) == mfma, len(wide)
        if mfma == 16:
            # Intermediates may come from another XCD, whose L2 does not snoop
            # this one's: their reads bypass every L2 (SC0 SC1).
            mids = [
                s
                for s in body
                if s.startswith("buffer_load_dwordx4") and s.endswith(" sc1")
            ]
            assert mids and all(s.endswith(" sc0 sc1") for s in mids), mids
        dwords = [s for s in body if s.startswith("buffer_load_dword ")]
        if mfma == 32:
            # Each wave's group tag follows a wait covering its mids stores.
            assert tags and all(tags), tags
            # Gate/up scales: one swizzled dword per two K steps.
            assert len(dwords) == 16, dwords
        else:
            # Down scales: strips 16 columns apart share their E8M0 dwords
            # (one per 32-column block and step pair).
            assert sum(not s.endswith(" sc1") for s in dwords) == 4, dwords
            # Each strip's bf16 results go out as 4 consecutive columns per
            # lane.
            stores = [s.split()[0] for s in body if _STORE.match(s.split()[0])]
            assert stores == ["buffer_store_dwordx2"] * 4, stores


def test_readiness_waits_use_scalar_loads_past_the_cache():
    """The down units poll their expert's 64 tags as 4 lines of 16 with
    scalar loads that bypass the scalar data cache (the tags are uncached
    memory written by other XCDs); no vector load polls them."""
    ins = [s for s, _ in _instructions(_compile(8, moe_stream.TAIL_SHARE))]
    polls = [s for s in ins if s.startswith("s_load_dwordx16")]
    assert polls and all(s.endswith(" glc") for s in polls), polls


@pytest.mark.parametrize(
    "variant",
    VARIANTS,
    ids=lambda v: f"rows{v['ROWS']}-tail{v['TAIL'][0]}of{v['TAIL'][1]}",
)
def test_shared_buffer_stores_use_system_scope(variant):
    """The mids, their scales and the tags (including the clear of the next
    launch's tag copy) are read by waves on other XCDs, so every buffer store
    to them is system scope (SC0 SC1); only the route rows, read after the
    launch, carry no scope bits. No buffer store is device scope alone."""
    ins = [s for s, _ in _instructions(_compile(variant["ROWS"], variant["TAIL"]))]
    stores = [s for s in ins if s.startswith("buffer_store")]
    assert stores
    assert not [s for s in stores if s.endswith(" sc1") and " sc0 " not in s], stores
    plain = [s for s in stores if not s.endswith((" sc1", " nt"))]
    assert plain and all(s.startswith("buffer_store_dwordx2") for s in plain), plain
