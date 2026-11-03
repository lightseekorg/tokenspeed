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
from asm_checks import (
    instruction_lines,
    persistent_codegen_problems,
    scalar_poll_vector_waits,
)

triton = pytest.importorskip("tokenspeed_triton")
from tokenspeed_kernel_amd._triton import gl, gluon  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.persistent import (  # noqa: E402
    enter_check,
    enter_issue,
    fail_stop,
    grid_join,
    persistent_compile_options,
    poll_pairs,
    poll_tags16,
    poll_until,
    poll_until_all,
    release_drain,
    store_pairs,
    wait_vmcnt,
)

_LOADS = 4


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


@gluon.jit
def _weights(w_ptr, LOADS: gl.constexpr):
    """A role's first loads: ``LOADS`` dwordx4 per lane."""
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [64, 1], [gl.num_warps(), 1], [1, 0]
    )
    rows = gl.arange(0, 64 * gl.num_warps(), gl.SliceLayout(1, layout))
    cols = gl.arange(0, 4, gl.SliceLayout(0, layout))
    offs = rows[:, None] * 4 + cols[None, :]
    total = gl.amd.cdna4.buffer_load(w_ptr, offs, cache=".cg")
    for i in gl.static_range(1, LOADS):
        total += gl.amd.cdna4.buffer_load(w_ptr + i * 65536, offs, cache=".cg")
    return total, offs


@gluon.jit(do_not_specialize=["deadline_ticks"])
def _entry_kernel(epochs, claims, errors, arena, w_ptr, out, deadline_ticks,LOADS: gl.constexpr):  # fmt: skip
    entry = enter_issue(epochs, claims)
    total, offs = _weights(w_ptr, LOADS)
    launch = enter_check(entry, errors, arena, deadline_ticks, 0, fail_stop)
    gl.amd.cdna4.buffer_store(total + launch.tag.to(gl.float32), out, offs)


_ENTRY_SIGNATURE = {
    "epochs": "*i64",
    "claims": "*i64",
    "errors": "*i32",
    "arena": "*i64",
    "w_ptr": "*fp32",
    "out": "*fp32",
    "deadline_ticks": "i64",
}


def test_entry_never_waits_for_the_roles_loads():
    """The role's first loads go out with nothing vector outstanding, and
    the entry's counters are scalar atomics waited for with lgkmcnt: the
    loads stay in flight until the role uses them."""
    assembly = _compile(_entry_kernel, _ENTRY_SIGNATURE, {"LOADS": _LOADS})
    assert persistent_codegen_problems(assembly, entry=True) == []
    lines = instruction_lines(assembly)
    first_load = next(i for i, l in enumerate(lines) if l.startswith("buffer_load"))
    assert not any("vmcnt" in l for l in lines[:first_load])
    counters = [i for i, l in enumerate(lines) if l.startswith("s_atomic_add_x2")]
    assert len(counters) == 2 and counters[0] > first_load
    # Nothing waits for the loads before the counters' compare.
    check = next(
        i for i in range(counters[1], len(lines)) if lines[i].startswith("s_cmp")
    )
    assert not any("vmcnt" in l for l in lines[first_load:check])


@gluon.jit
def _raw_weights(w_ptr):
    """Four dwordx4 loads per lane, held in registers (not yet used)."""
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [64, 1], [gl.num_warps(), 1], [1, 0]
    )
    rows = gl.arange(0, 64 * gl.num_warps(), gl.SliceLayout(1, layout))
    cols = gl.arange(0, 4, gl.SliceLayout(0, layout))
    offs = rows[:, None] * 4 + cols[None, :]
    loads = ()
    for i in gl.static_range(4):
        loads += (gl.amd.cdna4.buffer_load(w_ptr + i * 65536, offs, cache=".cg"),)
    return loads, offs


@gluon.jit(do_not_specialize=["deadline_ticks"])
def _roles_kernel(epochs, claims, errors, arena, w_ptr, kv_ptr, out, deadline_ticks):
    """A composed layer's shape: role branches by slot, each issuing its own
    weights (or attention pages) before the entry check, used after it."""
    entry = enter_issue(epochs, claims)
    if entry.slot < 16:
        loads, offs = _raw_weights(w_ptr)
    elif entry.slot < 24:
        loads, offs = _raw_weights(w_ptr + 8 * 65536)
    else:
        loads, offs = _raw_weights(kv_ptr)
    launch = enter_check(entry, errors, arena, deadline_ticks, 0, fail_stop)
    total = loads[0] + loads[1] + loads[2] + loads[3]
    gl.amd.cdna4.buffer_store(total + launch.tag.to(gl.float32), out, offs)


def test_composed_role_loads_issue_with_nothing_outstanding():
    """In every role branch the first load follows no vmcnt wait: the entry
    leaves nothing vector in flight for a register reuse to wait on."""
    signature = dict(_ENTRY_SIGNATURE, kv_ptr="*fp32")
    assembly = _compile(_roles_kernel, signature)
    assert persistent_codegen_problems(assembly, entry=True) == []
    lines = instruction_lines(assembly)
    last_load = max(i for i, l in enumerate(lines) if l.startswith("buffer_load"))
    assert not any("vmcnt" in l for l in lines[:last_load])


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
    flag = next(
        i for i in range(barrier, len(lines)) if lines[i].startswith("global_store")
    )
    assert lines[flag].endswith(" sc0")
    assert "buffer_wbl2" not in assembly and "buffer_inv" not in assembly


@gluon.jit(do_not_specialize=["expected", "deadline_ticks"])
def _poll_kernel(epochs, claims, errors, arena, word, w_ptr, out, expected,deadline_ticks, LOAD: gl.constexpr):  # fmt: skip
    entry = enter_issue(epochs, claims)
    total, offs = _weights(w_ptr, 2)
    launch = enter_check(entry, errors, arena, deadline_ticks, 0, fail_stop)
    if not poll_until(word, expected, launch, LOAD):
        fail_stop(launch, 16, 0)
    gl.amd.cdna4.buffer_store(total, out, offs)


_POLL_SIGNATURE = dict(_ENTRY_SIGNATURE, word="*i32", expected="i32")


@pytest.mark.parametrize(
    "load, bits", [("sc1", " sc1"), (".cv", " sc0 sc1"), ("smem", " glc")]
)
def test_poll_until_codegen(load, bits):
    assembly = _compile(_poll_kernel, _POLL_SIGNATURE, {"LOAD": load})
    assert persistent_codegen_problems(assembly, entry=True) == []
    assert re.search(rf"(global_load_dword|s_load_dword) .*{bits}\s*$", assembly, re.M)


def test_scalar_poll_never_waits_for_vector_loads():
    """The scalar poll and its every-64 deadline/abort check stay on lgkmcnt;
    a vector load of the abort word there drained the role's weights."""
    assembly = _compile(_poll_kernel, _POLL_SIGNATURE, {"LOAD": "smem"})
    assert scalar_poll_vector_waits(assembly) == []


@gluon.jit(do_not_specialize=["deadline_ticks"])
def _pairs_kernel(epochs, claims, errors, arena, box, src, out, deadline_ticks,SCOPE: gl.constexpr, PRODUCER: gl.constexpr):  # fmt: skip
    entry = enter_issue(epochs, claims)
    launch = enter_check(entry, errors, arena, deadline_ticks, 0, fail_stop)
    lane = gl.arange(0, 512, gl.BlockedLayout([1], [64], [8], [0]))
    if PRODUCER:
        store_pairs(box + lane, gl.load(src + lane), launch, lane >= 0, SCOPE)
    else:
        values, seen = poll_pairs(box + lane, lane < 500, launch, SCOPE)
        if not seen:
            fail_stop(launch, 16, 0)
        gl.store(out + lane, values)


_PAIRS_SIGNATURE = dict(
    {k: v for k, v in _ENTRY_SIGNATURE.items() if k not in ("w_ptr", "out")},
    box="*i64",
    src="*i32",
    out="*i32",
)


@pytest.mark.parametrize("scope, bits", [("cta", " sc0"), ("gpu", " sc1")])
def test_pair_store_is_one_8_byte_store_without_a_wait(scope, bits):
    assembly = _compile(
        _pairs_kernel, _PAIRS_SIGNATURE, {"SCOPE": scope, "PRODUCER": True}
    )
    lines = instruction_lines(assembly)
    # The main path ends at the first s_endpgm; the failure path follows it.
    lines = lines[: lines.index("s_endpgm")]
    stores = [i for i, l in enumerate(lines) if l.startswith("global_store")]
    assert stores and all(lines[i].startswith("global_store_dwordx2") for i in stores)
    assert all(lines[i].endswith(bits) for i in stores)
    assert not any("vmcnt" in l for l in lines[stores[-1] :])


@pytest.mark.parametrize("scope", ["gpu", "sys"])
def test_pair_poll_reads_value_and_tag_in_one_load(scope):
    assembly = _compile(
        _pairs_kernel, _PAIRS_SIGNATURE, {"SCOPE": scope, "PRODUCER": False}
    )
    assert persistent_codegen_problems(assembly, entry=True) == []
    bits = " sc1" if scope == "gpu" else " sc0 sc1"
    assert re.search(rf"global_load_dwordx2 .*{bits}\s*$", assembly, re.M)


@gluon.jit(do_not_specialize=["tag", "deadline_ticks"])
def _tags16_kernel(epochs, claims, errors, arena, tags, w_ptr, out, tag,deadline_ticks, GLC: gl.constexpr, BARRIER: gl.constexpr):  # fmt: skip
    entry = enter_issue(epochs, claims)
    total, offs = _weights(w_ptr, 2)
    launch = enter_check(entry, errors, arena, deadline_ticks, 0, fail_stop)
    if not poll_tags16(tags, tag, launch, GLC, BARRIER):
        fail_stop(launch, 16, 0)
    # What the tags guard, read after the poll.
    total += gl.amd.cdna4.buffer_load(w_ptr + 524288, offs, cache=".cg")
    gl.amd.cdna4.buffer_store(total, out, offs)


@pytest.mark.parametrize("glc", [True, False])
@pytest.mark.parametrize("barrier", [True, False])
def test_poll_tags16_is_scalar_only(glc, barrier):
    signature = dict(_ENTRY_SIGNATURE, tags="*i32", tag="i32")
    assembly = _compile(_tags16_kernel, signature, {"GLC": glc, "BARRIER": barrier})
    if barrier:
        assert persistent_codegen_problems(assembly, entry=True) == []
    assert scalar_poll_vector_waits(assembly) == []
    # The guarded load stays after the poll loop, barrier or not.
    lines = instruction_lines(assembly)
    poll = max(i for i, l in enumerate(lines) if l.startswith("s_load_dwordx16"))
    guarded = [i for i, l in enumerate(lines) if "buffer_load" in l and "s[" in l]
    assert max(guarded) > poll
    assert ("s_barrier" in "\n".join(lines[poll:])) == barrier
    assert re.search(
        r"s_load_dwordx16 s\[64:79\], s\[\d+:\d+\], 0x0" + (" glc" if glc else "$"),
        assembly,
        re.M,
    )
    assert ("s_dcache_inv" in assembly) != glc


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
    assembly = _compile(
        _paced_kernel, {"w_ptr": "*fp32", "out": "*fp32"}, {"W": 4, "LOADS": 12}
    )
    lines = instruction_lines(assembly)
    loads = [i for i, l in enumerate(lines) if l.startswith("buffer_load")]
    assert len(loads) == 12
    for previous, current in zip(loads[3:], loads[4:]):
        assert "s_waitcnt vmcnt(3)" in lines[previous:current]
    assert not any("vmcnt(0)" in l for l in lines[: loads[-1]])


@gluon.jit(do_not_specialize=["deadline_ticks"])
def _join_kernel(epochs, claims, errors, arena, joins, values, deadline_ticks):
    entry = enter_issue(epochs, claims)
    launch = enter_check(entry, errors, arena, deadline_ticks, 0, fail_stop)
    gl.store(values + launch.cta, launch.tag)
    grid_join(joins, launch, fail_stop, 0, 0)
    gl.store(
        values + 256 + launch.cta,
        gl.load(values + (launch.cta + 1) % 256, cache_modifier=".cv"),
    )


def test_grid_join_drains_before_arriving():
    signature = {k: v for k, v in _ENTRY_SIGNATURE.items() if k not in ("w_ptr", "out")}
    signature.update(joins="*i32", values="*i32")
    assembly = _compile(_join_kernel, signature)
    assert persistent_codegen_problems(assembly, entry=True) == []


def test_fail_stop_mirrors_the_record_before_the_trap():
    assembly = _compile(_entry_kernel, _ENTRY_SIGNATURE, {"LOADS": 1})
    lines = instruction_lines(assembly)
    trap = next(i for i, l in enumerate(lines) if l.startswith("s_trap"))
    before = lines[:trap]
    host = [
        i
        for i, l in enumerate(before)
        if l.startswith("global_store") and l.endswith("sc0 sc1")
    ]
    assert host, "the record is not written through"
    assert "s_waitcnt vmcnt(0)" in before[host[-1] :]


@gluon.jit(do_not_specialize=["deadline_ticks"])
def _tag_poll_kernel(epochs, claims, errors, arena, word, deadline_ticks):
    entry = enter_issue(epochs, claims)
    launch = enter_check(entry, errors, arena, deadline_ticks, 0, fail_stop)
    if not poll_until(word, launch.tag, launch, "smem"):
        fail_stop(launch, 16, 0)


def test_launch_tag_is_a_scalar():
    """The epoch comes out of a cross-wave reduction in a VGPR; read back to
    a scalar, a scalar poll on the tag stays a uniform loop (no exec masks,
    no lane reads per retry)."""
    signature = {k: v for k, v in _ENTRY_SIGNATURE.items() if k not in ("w_ptr", "out")}
    signature["word"] = "*i32"
    assembly = _compile(_tag_poll_kernel, signature)
    from asm_checks import innermost_loops

    polls = [l for l in innermost_loops(assembly) if "s_load_dword" in l and "glc" in l]
    assert polls
    for loop in polls:
        # The retry compares two scalars; a VGPR tag needs a lane read or a
        # vector compare on every retry.
        assert re.search(r"s_cmp_(eq|lg)_u32 s\d+, s\d+", loop), loop
        assert "v_readfirstlane" not in loop
