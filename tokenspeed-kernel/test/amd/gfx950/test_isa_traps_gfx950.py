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

"""gfx950 codegen traps that persistent weight-streaming kernels must avoid.

Each test compiles, without a GPU, one small kernel written the trapped way
and one written the approved way, and checks that the gate's assembly check
flags the first and passes the second. A failure means either the compiler
changed how it lowers the construct (re-check the rule) or the check no
longer recognizes it (fix the check before trusting the gate).
"""

import pytest

pytest.importorskip("tokenspeed_kernel_amd")

from asm_checks import (  # noqa: E402
    drains_of,
    innermost_loops,
    instruction_lines,
    isa_report,
    lds_read_widths,
    persistent_codegen_problems,
    register_problems,
    serialized_coherent_loads,
    split_poll_problems,
    store_loop_preheader_drains,
    stream_loop_problems,
    write_through_store_drains,
)
from tokenspeed_kernel_amd._triton import gl, gluon, triton  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.monokernel.persistent._intrinsics import (  # noqa: E402
    memrealtime,
    persistent_compile_options,
    scalar_load_glc,
    vote,
    wave_acquire,
)

# 4 int32 per lane: one dwordx4 per lane, 1 KB per wave.
_LANE4 = gl.constexpr(gl.BlockedLayout([4], [64], [1], [0]))
_LANE1 = gl.constexpr(gl.BlockedLayout([1], [64], [1], [0]))


def _asm(fn, signature, constexprs=None, num_warps=1):
    """Compile for gfx950 without a GPU; pointers 16-byte aligned, as launched."""
    attrs = {
        (index,): [["tt.divisibility", 16]]
        for index, name in enumerate(fn.arg_names)
        if signature.get(name, "").startswith("*")
    }
    source = gluon._runtime.GluonASTSource(
        fn, signature, constexprs=constexprs or {}, attrs=attrs
    )
    kernel = triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": num_warps},
    )
    return kernel.asm["amdgcn"]


def _stream_problems(assembly, min_in_flight):
    return [
        problem
        for loop in innermost_loops(assembly)
        if "v_mfma" in loop
        for problem in stream_loop_problems(loop, min_in_flight)
    ]


@gluon.jit
def _coherent_reads(src, out, BUFFER: gl.constexpr):
    offsets = gl.arange(0, 64, _LANE1) * 64
    total = gl.zeros([64], gl.int32, _LANE1)
    for k in gl.static_range(4):
        if BUFFER:
            total += gl.amd.cdna4.buffer_load(src, offsets + k, cache=".cv")
        else:
            total += gl.load(src + offsets + k, cache_modifier=".cv")
    gl.store(out + gl.arange(0, 64, _LANE1), total)


def test_coherent_payload_reads_use_buffer_loads():
    # gl.load(".cv") is lowered as a volatile load: vmcnt(0) after each one.
    signature = {"src": "*i32", "out": "*i32"}
    trapped = _asm(_coherent_reads, signature, {"BUFFER": False})
    approved = _asm(_coherent_reads, signature, {"BUFFER": True})
    assert serialized_coherent_loads(instruction_lines(trapped))
    assert not serialized_coherent_loads(instruction_lines(approved))


@gluon.jit
def _publish_after_weights(weights, out, BUFFER: gl.constexpr):
    offsets = gl.arange(0, 256, _LANE4)
    first = gl.amd.cdna4.buffer_load(weights, offsets, cache=".cg")
    second = gl.amd.cdna4.buffer_load(weights, offsets + 256, cache=".cg")
    if BUFFER:
        gl.amd.cdna4.buffer_store(offsets, out, offsets, cache=".wt")
    else:
        gl.store(out + offsets, offsets, cache_modifier=".wt")
    gl.store(out + 256 + offsets, first ^ second)


def test_write_through_publish_does_not_wait_for_weights():
    # gl.store(".wt") is volatile: the wait after it drains the weights.
    signature = {"weights": "*i32", "out": "*i32"}
    trapped = _asm(_publish_after_weights, signature, {"BUFFER": False})
    approved = _asm(_publish_after_weights, signature, {"BUFFER": True})
    assert write_through_store_drains(instruction_lines(trapped))
    assert not write_through_store_drains(instruction_lines(approved))


@gluon.jit
def _pair_poll(pairs, out, tag, WIDE: gl.constexpr):
    if WIDE:
        # Both (value, tag) pairs of a lane, indexed [lane, pair, word]: the
        # volatile 16-byte read is split at the pair dimension.
        layout: gl.constexpr = gl.BlockedLayout(
            [1, 2, 2], [64, 1, 1], [1, 1, 1], [2, 1, 0]
        )
        lane = gl.arange(0, 64, gl.SliceLayout(1, gl.SliceLayout(2, layout)))[
            :, None, None
        ]
        pair = gl.arange(0, 2, gl.SliceLayout(0, gl.SliceLayout(2, layout)))[
            None, :, None
        ]
        word = gl.arange(0, 2, gl.SliceLayout(0, gl.SliceLayout(1, layout)))[
            None, None, :
        ]
        offsets = lane * 4 + pair * 2 + word
        words = gl.load(pairs + offsets, cache_modifier=".cv", volatile=True)
        missing = gl.sum(
            gl.sum(
                gl.sum(gl.where((word == 1) & (words != tag), 1, 0), axis=2), axis=1
            ),
            axis=0,
        )
        while missing != 0:
            words = gl.load(pairs + offsets, cache_modifier=".cv", volatile=True)
            missing = gl.sum(
                gl.sum(
                    gl.sum(gl.where((word == 1) & (words != tag), 1, 0), axis=2), axis=1
                ),
                axis=0,
            )
        gl.store(out + offsets, words)
    else:
        # Poll only the tag dword of each pair.
        lanes = gl.arange(0, 64, _LANE1)
        words = gl.load(pairs + lanes * 4 + 1, cache_modifier=".cv", volatile=True)
        missing = gl.sum(gl.where(words != tag, 1, 0), axis=0)
        while missing != 0:
            words = gl.load(pairs + lanes * 4 + 1, cache_modifier=".cv", volatile=True)
            missing = gl.sum(gl.where(words != tag, 1, 0), axis=0)
        gl.store(out + lanes, words)


def test_pair_polls_read_one_tag_dword_per_retry():
    signature = {"pairs": "*i32", "out": "*i32", "tag": "i32"}
    assert split_poll_problems(_asm(_pair_poll, signature, {"WIDE": True}))
    assert not split_poll_problems(_asm(_pair_poll, signature, {"WIDE": False}))


@gluon.constexpr_function
def _mfma_layouts(warps=1):
    mfma = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[1, warps]
    )
    return (
        mfma,
        gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=8),
        gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=8),
    )


@gluon.jit
def _weight_tile(weights, chunk):
    """B operand [32 K, 16 cols] bf16: one dwordx4 per lane, 1 KB per wave."""
    lb: gl.constexpr = _mfma_layouts()[2]
    k = gl.arange(0, 32, gl.SliceLayout(1, lb))[:, None]
    n = gl.arange(0, 16, gl.SliceLayout(0, lb))[None, :]
    return gl.amd.cdna4.buffer_load(weights, chunk * 512 + n * 32 + k, cache=".cg")


@gluon.jit
def _ring(weights, x, out, chunks, PIN: gl.constexpr):
    """A 4-slot register ring over a runtime number of 1 KB chunks."""
    mfma: gl.constexpr = _mfma_layouts()[0]
    la: gl.constexpr = _mfma_layouts()[1]
    m = gl.arange(0, 16, gl.SliceLayout(1, la))[:, None]
    k = gl.arange(0, 32, gl.SliceLayout(0, la))[None, :]
    a = gl.load(x + m * 32 + k)
    acc = gl.zeros([16, 16], gl.float32, mfma)
    s0 = _weight_tile(weights, 0)
    if PIN:
        gl.amd.hint.sched_barrier()
    s1 = _weight_tile(weights, 1)
    if PIN:
        gl.amd.hint.sched_barrier()
    s2 = _weight_tile(weights, 2)
    if PIN:
        gl.amd.hint.sched_barrier()
    s3 = _weight_tile(weights, 3)
    if PIN:
        gl.amd.hint.sched_barrier()
    for c in range(0, chunks, 4):
        acc = gl.amd.cdna3.mfma(a, s0, acc)
        s0 = _weight_tile(weights, c + 4)
        if PIN:
            gl.amd.hint.sched_barrier()
        acc = gl.amd.cdna3.mfma(a, s1, acc)
        s1 = _weight_tile(weights, c + 5)
        if PIN:
            gl.amd.hint.sched_barrier()
        acc = gl.amd.cdna3.mfma(a, s2, acc)
        s2 = _weight_tile(weights, c + 6)
        if PIN:
            gl.amd.hint.sched_barrier()
        acc = gl.amd.cdna3.mfma(a, s3, acc)
        s3 = _weight_tile(weights, c + 7)
        if PIN:
            gl.amd.hint.sched_barrier()
    r = gl.arange(0, 16, gl.SliceLayout(1, mfma))[:, None]
    n = gl.arange(0, 16, gl.SliceLayout(0, mfma))[None, :]
    gl.store(out + r * 16 + n, acc)


def test_weight_ring_keeps_its_depth_only_with_a_pinned_issue_order():
    # Without the pin the scheduler reorders the ring's first loads, and the
    # loop-header merge then waits for all of them on every iteration.
    signature = {"weights": "*bf16", "x": "*bf16", "out": "*fp32", "chunks": "i32"}
    assert _stream_problems(_asm(_ring, signature, {"PIN": False}), 4)
    assert not _stream_problems(_asm(_ring, signature, {"PIN": True}), 4)


@gluon.jit
def _push_after_weights(weights, src, peer, out, n, UNROLLED: gl.constexpr):
    offsets = gl.arange(0, 256, _LANE4)
    value = gl.amd.cdna4.buffer_load(src, offsets)
    gl.amd.hint.sched_barrier()
    first = gl.amd.cdna4.buffer_load(weights, offsets, cache=".cg")
    second = gl.amd.cdna4.buffer_load(weights, offsets + 256, cache=".cg")
    if UNROLLED:
        for i in gl.static_range(8):
            gl.amd.cdna4.buffer_store(value + i, peer, offsets + i * 256, cache=".wt")
    else:
        for i in range(n):
            gl.amd.cdna4.buffer_store(value + i, peer, offsets + i * 256, cache=".wt")
    gl.amd.cdna4.buffer_store(first ^ second, out, offsets)


def test_pushes_after_weights_are_unrolled():
    # LLVM flushes vmcnt before a runtime store-only loop that reads a value
    # loaded earlier; that flush also waits for the weights.
    signature = {
        "weights": "*i32",
        "src": "*i32",
        "peer": "*i32",
        "out": "*i32",
        "n": "i32",
    }
    assert store_loop_preheader_drains(
        _asm(_push_after_weights, signature, {"UNROLLED": False})
    )
    assert not store_loop_preheader_drains(
        _asm(_push_after_weights, signature, {"UNROLLED": True})
    )


@gluon.jit
def _exchange(out, COLUMN_FASTEST: gl.constexpr):
    """8 waves' fp32 partials of 64 columns x 8 tokens, summed per lane over
    the waves; each lane then owns 8 consecutive columns of one token."""
    if COLUMN_FASTEST:
        shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[1, 2, 0])
    else:
        shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, order=[2, 1, 0])
    write: gl.constexpr = gl.BlockedLayout([1, 4, 1], [1, 8, 8], [8, 1, 1], [1, 2, 0])
    read: gl.constexpr = gl.BlockedLayout([8, 8, 1], [1, 8, 8], [1, 1, 8], [1, 2, 0])
    partials = gl.allocate_shared_memory(gl.float32, [8, 64, 8], shared)
    w = gl.arange(0, 8, gl.SliceLayout(1, gl.SliceLayout(2, write)))[:, None, None]
    c = gl.arange(0, 64, gl.SliceLayout(0, gl.SliceLayout(2, write)))[None, :, None]
    t = gl.arange(0, 8, gl.SliceLayout(0, gl.SliceLayout(1, write)))[None, None, :]
    partials.store((w * 512 + c * 8 + t).to(gl.float32))
    gl.barrier()
    total = gl.sum(partials.load(read), axis=0)
    sl: gl.constexpr = gl.SliceLayout(0, read)
    cc = gl.arange(0, 64, gl.SliceLayout(1, sl))[:, None]
    tt = gl.arange(0, 8, gl.SliceLayout(0, sl))[None, :]
    gl.store(out + tt * 64 + cc, total)


def test_exchange_partials_are_stored_column_fastest():
    # Token-fastest partials make every lane's 8 columns a strided read.
    signature = {"out": "*fp32"}
    strided = lds_read_widths(
        instruction_lines(_asm(_exchange, signature, {"COLUMN_FASTEST": False}, 8))
    )
    wide = lds_read_widths(
        instruction_lines(_asm(_exchange, signature, {"COLUMN_FASTEST": True}, 8))
    )
    assert strided.get("ds_read2_b32", 0) > 0
    assert wide.get("ds_read2_b32", 0) == 0 and wide.get("ds_read_b128", 0) > 0


@gluon.jit
def _resident_weights(weights, x, flag, out, TILES: gl.constexpr):
    """TILES 1 KB weight tiles per wave (8 waves, 16 columns each) issued, then
    a flag wait, then one MFMA chain over them: the weights stay resident
    across the wait, as in a role that issues them before its input lands."""
    mfma: gl.constexpr = _mfma_layouts(8)[0]
    la: gl.constexpr = _mfma_layouts(8)[1]
    lb: gl.constexpr = _mfma_layouts(8)[2]
    kb = gl.arange(0, 32 * TILES, gl.SliceLayout(1, lb))[:, None]
    nb = gl.arange(0, 128, gl.SliceLayout(0, lb))[None, :]
    resident = gl.amd.cdna4.buffer_load(weights, nb * (32 * TILES) + kb, cache=".cg")
    while gl.atomic_load(flag, sem="relaxed", scope="gpu") == 0:
        pass
    m = gl.arange(0, 16, gl.SliceLayout(1, la))[:, None]
    k = gl.arange(0, 32 * TILES, gl.SliceLayout(0, la))[None, :]
    a = gl.load(x + m * (32 * TILES) + k)
    acc = gl.amd.cdna3.mfma(a, resident, gl.zeros([16, 128], gl.float32, mfma))
    r = gl.arange(0, 16, gl.SliceLayout(1, mfma))[:, None]
    n = gl.arange(0, 128, gl.SliceLayout(0, mfma))[None, :]
    gl.store(out + r * 128 + n, acc)


def test_register_budget_flags_spills_that_drain_resident_weights():
    # 64 KB of weights per wave (256 VGPRs) cannot stay resident: the spills
    # are stores that first wait for the in-flight loads.
    signature = {"weights": "*bf16", "x": "*bf16", "flag": "*i32", "out": "*fp32"}
    over = _asm(_resident_weights, signature, {"TILES": 64}, num_warps=8)
    under = _asm(_resident_weights, signature, {"TILES": 8}, num_warps=8)
    assert register_problems(isa_report(over))
    assert not register_problems(isa_report(under))
    lines = instruction_lines(over)
    first_load = next(
        i for i, line in enumerate(lines) if " nt" in line and "_load_" in line
    )
    first_mfma = next(i for i, line in enumerate(lines) if line.startswith("v_mfma"))
    assert any(
        line.startswith("scratch_store") for line in lines[first_load:first_mfma]
    )
    assert drains_of(lines, first_load, first_mfma)


@gluon.jit
def _flag_then_payload(flag, tags, payload, out, tag, deadline, SCALAR: gl.constexpr):
    """Wait for a flag without a workgroup barrier, then read the payload."""
    lanes = gl.arange(0, 64, _LANE1)
    start = memrealtime()
    if SCALAR:
        # Every wave polls for itself.
        while (scalar_load_glc(flag) != tag) & (memrealtime() - start < deadline):
            pass
    else:
        # One lane polls; the wave's other lanes rely on its vote.
        owner = lanes == 0
        seen = gl.atomic_load(flag + lanes * 0, mask=owner, sem="relaxed", scope="gpu")
        while vote((owner & (seen != tag)).to(gl.int32)) & (
            memrealtime() - start < deadline
        ):
            seen = gl.atomic_load(
                flag + lanes * 0, mask=owner, sem="relaxed", scope="gpu"
            )
    wave_acquire()
    gl.store(out + lanes, gl.load(payload + lanes, cache_modifier=".cg"))


def test_only_vector_flag_polls_need_a_barrier():
    # A scalar poll runs in every wave, so its own loads may follow a
    # per-wave acquire; a vector poll answers for the whole workgroup.
    signature = {
        "flag": "*i32", "tags": "*i32", "payload": "*i32", "out": "*i32",
        "tag": "i32", "deadline": "i64",
    }  # fmt: skip
    problems = {}
    for scalar in (True, False):
        attrs = {(i,): [["tt.divisibility", 16]] for i in range(4)}
        source = gluon._runtime.GluonASTSource(
            _flag_then_payload, signature, constexprs={"SCALAR": scalar}, attrs=attrs
        )
        assembly = triton.compile(
            source,
            target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
            options={"num_warps": 1, **persistent_compile_options()},
        ).asm["amdgcn"]
        problems[scalar] = persistent_codegen_problems(assembly, entry=None)
    assert "a load follows a flag poll before a barrier" in problems[False]
    assert problems[True] == []
