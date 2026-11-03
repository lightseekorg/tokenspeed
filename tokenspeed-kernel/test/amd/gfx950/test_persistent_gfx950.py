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

"""Entry, joins and fail-stop deadlines of persistent gfx950 kernels."""

import os
import re
import subprocess
import sys
import textwrap
import time

import pytest
import torch
from tokenspeed_kernel._triton import gl, gluon, triton
from utils import is_cdna4, polling_cycles

if not is_cdna4():
    pytest.skip("Persistent helpers target gfx950", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950.persistent import (  # noqa: E402
    ERROR_WORDS,
    GRID,
    JOIN_WORDS,
    LINE_WORDS,
    SITE_CLAIM,
    SITE_JOIN,
    enter_launch,
    fail_stop,
    grid_join,
)

_DEADLINE_TICKS = 200_000_000  # 2 s
_GRID = gl.constexpr(GRID)


@gluon.jit
def _record_and_return(launch, site, detail):
    """Test handler: record the failure like ``fail_stop`` but return."""
    gl.store(launch.error_ptr, site, cache_modifier=".wt")
    gl.store(launch.error_ptr + 1, launch.cta, cache_modifier=".wt")
    gl.store(launch.error_ptr + 2, detail, cache_modifier=".wt")
    gl.store(launch.error_ptr + 3, launch.epoch, cache_modifier=".wt")


@gluon.jit(do_not_specialize=["deadline_ticks"])
def _join_rounds_kernel(
    epochs,
    claims,
    joins,
    errors,
    values,
    seen,
    placement,
    deadline_ticks,
    ROUNDS: gl.constexpr,
    LAUNCHES: gl.constexpr,
):
    """Every round each workgroup writes its value, joins, and reads its
    neighbor's; the next round overwrites the same words after another join.
    ``placement[epoch % LAUNCHES, pid]`` records ``cta << 8 | xcd``."""
    launch = enter_launch(epochs, claims, errors, deadline_ticks, 0, _record_and_return)
    row = launch.epoch % LAUNCHES
    gl.store(placement + row * _GRID + gl.program_id(0), (launch.cta << 8) | launch.xcd)
    for r in gl.static_range(ROUNDS):
        value = (launch.epoch * ROUNDS + r) * 1000 + launch.cta
        gl.store(values + launch.cta, value, cache_modifier=".wt")
        grid_join(joins, launch, _record_and_return)
        other = (launch.cta + 1 + r) % _GRID
        got = gl.load(values + other, cache_modifier=".cv")
        expect = (launch.epoch * ROUNDS + r) * 1000 + other
        gl.atomic_add(seen + launch.cta, (got == expect).to(gl.int32), sem="relaxed")
        grid_join(joins, launch, _record_and_return)


class _State:
    def __init__(self):
        self.epochs = torch.zeros(GRID * LINE_WORDS, dtype=torch.int32, device="cuda")
        self.claims = torch.zeros(
            GRID * LINE_WORDS // 2, dtype=torch.int64, device="cuda"
        )
        self.joins = torch.zeros(JOIN_WORDS, dtype=torch.int32, device="cuda")
        self.errors = torch.zeros(ERROR_WORDS, dtype=torch.int32, device="cuda")


def _launch_rounds(state, values, seen, placement, rounds, grid=GRID, deadline=None):
    _join_rounds_kernel[(grid,)](
        state.epochs,
        state.claims,
        state.joins,
        state.errors,
        values,
        seen,
        placement,
        _DEADLINE_TICKS if deadline is None else deadline,
        ROUNDS=rounds,
        LAUNCHES=placement.shape[0],
        num_warps=8,
    )


@gluon.jit
def _small_grid_kernel(out):
    gl.store(out + gl.program_id(0), gl.program_id(0))


def _rotate_dispatch(scratch, workgroups):
    """A small kernel moves the XCD that the next launch's dispatch starts on."""
    _small_grid_kernel[(workgroups,)](scratch)


def test_joins_order_writes_across_launches_and_graph_replays():
    state = _State()
    rounds, replays = 4, 500
    # Each replay runs two launches; placement keeps one row per launch.
    launches = 1 + 2 * replays
    values = torch.zeros(GRID, dtype=torch.int32, device="cuda")
    seen = torch.zeros(GRID, dtype=torch.int32, device="cuda")
    placement = torch.full((launches + 1, GRID), -1, dtype=torch.int32, device="cuda")
    scratch = torch.zeros(64, dtype=torch.int32, device="cuda")
    _launch_rounds(state, values, seen, placement, rounds)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # Odd grids before and between the persistent launches rotate the
        # XCD that their first workgroup lands on.
        _rotate_dispatch(scratch, 13)
        _launch_rounds(state, values, seen, placement, rounds)
        _rotate_dispatch(scratch, 5)
        _launch_rounds(state, values, seen, placement, rounds)
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()
    assert state.errors.count_nonzero().item() == 0
    # Every neighbor read saw this round's value, in every launch.
    assert torch.equal(
        seen.cpu(), torch.full((GRID,), rounds * launches, dtype=torch.int32)
    )
    # Each program id advanced its own epoch once per launch.
    assert torch.equal(
        state.epochs.view(GRID, LINE_WORDS)[:, 0].cpu(),
        torch.full((GRID,), launches, dtype=torch.int32),
    )
    rows = placement[1 : launches + 1].cpu()
    pid = torch.arange(GRID, dtype=torch.int32)
    xcd, cta = rows & 0xFF, rows >> 8
    # Every launch's logical ids are a permutation: one owner per role.
    assert torch.equal(cta.sort(dim=1).values, pid.expand(launches, GRID))
    # One rotation per launch: dispatch hands the XCDs workgroups in turn.
    rotation = (xcd - pid) % 8
    assert bool((rotation == rotation[:, :1]).all())
    # The odd grids did move the start, so the logical ids were not the pids.
    assert bool((rotation[:, 0] != 0).any())
    # 2 joins per round per launch: an even number of flips leaves every
    # counter where it started.
    assert state.joins.count_nonzero().item() == 0


def test_join_waits_for_a_missing_workgroup_until_its_deadline():
    state = _State()
    values = torch.zeros(GRID, dtype=torch.int32, device="cuda")
    seen = torch.zeros(GRID, dtype=torch.int32, device="cuda")
    placement = torch.zeros((2, GRID), dtype=torch.int32, device="cuda")
    # Eight workgroups short: no XCD completes its join. 10 ms deadline.
    _launch_rounds(state, values, seen, placement, 1, grid=GRID - 8, deadline=1_000_000)
    torch.cuda.synchronize()
    assert state.errors[0].item() == SITE_JOIN
    assert state.errors[3].item() == 1  # epoch of the failing launch


def test_duplicate_claim_is_reported():
    state = _State()
    values = torch.zeros(GRID, dtype=torch.int32, device="cuda")
    seen = torch.zeros(GRID, dtype=torch.int32, device="cuda")
    placement = torch.zeros((2, GRID), dtype=torch.int32, device="cuda")
    # Plant this launch's tag (epoch 1) in role (xcd 3, slot 5), as a second
    # workgroup taking that role would have left it.
    claims = state.claims.view(GRID, LINE_WORDS // 2)
    claims[3 * 32 + 5, 0] = (1 << 32) | 77
    _launch_rounds(state, values, seen, placement, 1)
    torch.cuda.synchronize()
    assert state.errors[0].item() == SITE_CLAIM
    assert state.errors[1].item() == 5 * 8 + 3  # the role's logical id
    assert state.errors[2].item() == 77  # the other claimant's program id


_TRAP_CHILD = textwrap.dedent("""
    import sys
    sys.path.insert(0, sys.argv[1])
    import torch
    import test_persistent_gfx950 as t
    state = t._State()
    t._fail_stop_kernel[(t.GRID - 8,)](
        state.epochs, state.claims, state.joins, state.errors, 1_000_000,
        num_warps=8,
    )
    torch.cuda.synchronize()
    print("NO TRAP")
    """)


@gluon.jit(do_not_specialize=["deadline_ticks"])
def _fail_stop_kernel(epochs, claims, joins, errors, deadline_ticks):
    launch = enter_launch(epochs, claims, errors, deadline_ticks, 0, fail_stop)
    grid_join(joins, launch, fail_stop)


def test_expired_wait_stops_the_queue():
    """``fail_stop`` raises a queue error: the process ends, it never hangs."""
    start = time.monotonic()
    child = subprocess.run(
        [sys.executable, "-c", _TRAP_CHILD, os.path.dirname(__file__)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert child.returncode != 0, child.stdout + child.stderr
    assert "NO TRAP" not in child.stdout
    assert time.monotonic() - start < 300


def _compile(fn, signature, constexprs, num_warps):
    signature = dict(signature)
    signature.update({name: "constexpr" for name in constexprs})
    source = gluon._runtime.GluonASTSource(
        fn, signature, constexprs=constexprs, attrs={}
    )
    return triton.compile(
        source,
        target=triton.backends.compiler.GPUTarget("hip", "gfx950", 64),
        options={"num_warps": num_warps},
    )


def entry_and_join_codegen_problems(assembly: str) -> list[str]:
    """Return the ordering rules a persistent kernel's assembly breaks.

    - Nothing invalidates or writes back L2.
    - Only the epoch store precedes the claim exchange, and the claim's
      compare-and-branch completes before any other store or atomic.
    - Every join arrival follows a full drain and a workgroup barrier.
    - Every poll reloads with ``sc1`` inside its loop, exits on one
      lane-mask vote, and holds no workgroup barrier.
    """
    problems = []
    if "buffer_inv" in assembly or "buffer_wbl2" in assembly:
        problems.append("L2 invalidate or write-back")
    lines = [
        line
        for line in assembly.splitlines()
        if not line.lstrip().startswith((";", ".loc"))
    ]
    writes = [
        index
        for index, line in enumerate(lines)
        if re.search(r"\b(global|buffer)_(store|atomic)\w*", line)
    ]
    swap = next(i for i in writes if "atomic_swap_x2" in lines[i])
    if sum(1 for i in writes if i < swap) != 1:
        problems.append("a store or atomic other than the epoch precedes the claim")
    after = next(i for i in writes if i > swap)
    between = "\n".join(lines[swap:after])
    if "s_waitcnt vmcnt(0)" not in between or "s_cbranch" not in between:
        problems.append(
            "the claim's compare-and-branch does not precede the next store"
        )
    for index in writes:
        if "global_atomic_add" in lines[index] and " sc0" in lines[index]:
            # A returning add is a join arrival.
            before = "\n".join(lines[max(0, index - 40) : index])
            if "s_waitcnt vmcnt(0)" not in before or "s_barrier" not in before:
                problems.append(f"arrival without drain and barrier: {lines[index]}")
    for cycle in polling_cycles("\n".join(lines)):
        if not any(re.search(r"global_load_dword .* sc1$", l) for l in cycle):
            continue
        text = "\n".join(cycle)
        if re.search(r"s_barrier|\bds_|_dpp", text):
            problems.append("workgroup barrier or LDS exchange inside a poll loop")
        # One lane-mask vote per retry. (The deadline and abort check every
        # 64 retries reads one lane of the abort word.)
        if len(re.findall(r"v_cmp_ne_u32_e64 s\[\d+:\d+\], 0, v\d+", text)) != 1:
            problems.append("poll loop without a one-instruction vote")
    return problems


def test_entry_and_join_codegen():
    kernel = _compile(
        _join_rounds_kernel,
        {
            "epochs": "*i32",
            "claims": "*i64",
            "joins": "*i32",
            "errors": "*i32",
            "values": "*i32",
            "seen": "*i32",
            "placement": "*i32",
            "deadline_ticks": "i64",
        },
        {"ROUNDS": 2, "LAUNCHES": 4},
        num_warps=8,
    )
    assert entry_and_join_codegen_problems(kernel.asm["amdgcn"]) == []
