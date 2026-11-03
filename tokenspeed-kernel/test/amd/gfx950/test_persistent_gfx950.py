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
import subprocess
import sys
import textwrap
import time

import pytest
import torch
from tokenspeed_kernel._triton import gl, gluon
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("Persistent helpers target gfx950", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950.persistent import (  # noqa: E402
    ENTRY_WORDS,
    ERROR_WORDS,
    GRID,
    JOIN_WORDS,
    LINE_WORDS,
    SITE_CLAIM,
    SITE_JOIN,
    enter_check,
    enter_issue,
    fail_stop,
    grid_join,
    persistent_compile_options,
)

_DEADLINE_TICKS = 200_000_000  # 2 s
_GRID = gl.constexpr(GRID)


@gluon.jit
def _record_and_return(launch, site, detail):
    """Test handler: record the failure like ``fail_stop`` but return."""
    gl.store(launch.error_ptr, site, cache_modifier=".wt")
    gl.store(launch.error_ptr + 1, launch.cta, cache_modifier=".wt")
    gl.store(launch.error_ptr + 2, detail, cache_modifier=".wt")
    gl.store(launch.error_ptr + 3, launch.epoch.to(gl.int32), cache_modifier=".wt")


@gluon.jit(do_not_specialize=["deadline_ticks"])
def _join_rounds_kernel(
    epochs,
    claims,
    joins,
    errors,
    values,
    seen,
    placement,
    tags,
    deadline_ticks,
    ROUNDS: gl.constexpr,
    LAUNCHES: gl.constexpr,
):
    """Every round each workgroup writes its value, joins, and reads its
    neighbor's; the next round overwrites the same words after another join.
    ``placement[epoch % LAUNCHES, pid]`` records ``cta << 8 | xcd`` and
    ``tags[epoch % LAUNCHES, pid]`` the launch's ``tag << 1 | copy``."""
    entry = enter_issue(epochs, claims)
    launch = enter_check(entry, errors, errors, deadline_ticks, 0, _record_and_return)
    row = (launch.epoch % LAUNCHES).to(gl.int32)
    gl.store(placement + row * _GRID + gl.program_id(0), (launch.cta << 8) | launch.xcd)
    gl.store(tags + row * _GRID + gl.program_id(0), (launch.tag << 1) | launch.copy)
    epoch = (launch.epoch % 1000).to(gl.int32)
    for r in gl.static_range(ROUNDS):
        value = (epoch * ROUNDS + r) * 1000 + launch.cta
        gl.store(values + launch.cta, value, cache_modifier=".wt")
        grid_join(joins, launch, _record_and_return, 0, 0)
        other = (launch.cta + 1 + r) % _GRID
        got = gl.load(values + other, cache_modifier=".cv")
        expect = (epoch * ROUNDS + r) * 1000 + other
        gl.atomic_add(seen + launch.cta, (got == expect).to(gl.int32), sem="relaxed")
        grid_join(joins, launch, _record_and_return, 1, 1)


class _State:
    def __init__(self):
        self.epochs = torch.zeros(ENTRY_WORDS, dtype=torch.int64, device="cuda")
        self.claims = torch.zeros(ENTRY_WORDS, dtype=torch.int64, device="cuda")
        self.joins = torch.zeros(JOIN_WORDS, dtype=torch.int32, device="cuda")
        self.errors = torch.zeros(ERROR_WORDS, dtype=torch.int32, device="cuda")


def _launch_rounds(
    state, values, seen, placement, rounds, grid=GRID, deadline=None, tags=None
):
    if tags is None:
        tags = torch.zeros_like(placement)
    _join_rounds_kernel[(grid,)](
        state.epochs,
        state.claims,
        state.joins,
        state.errors,
        values,
        seen,
        placement,
        tags,
        _DEADLINE_TICKS if deadline is None else deadline,
        ROUNDS=rounds,
        LAUNCHES=placement.shape[0],
        num_warps=8,
        **persistent_compile_options(),
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
        state.epochs.view(GRID, LINE_WORDS // 2)[:, 0].cpu(),
        torch.full((GRID,), launches, dtype=torch.int64),
    )
    # Each role was claimed once per launch.
    assert torch.equal(
        state.claims.view(GRID, LINE_WORDS // 2)[:, 0].cpu(),
        torch.full((GRID,), launches, dtype=torch.int64),
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
    # Advance the counter of role (xcd 3, slot 5) once, as a second
    # workgroup taking that role in this launch would have.
    claims = state.claims.view(GRID, LINE_WORDS // 2)
    claims[3 * 32 + 5, 0] = 1
    _launch_rounds(state, values, seen, placement, 1)
    torch.cuda.synchronize()
    assert state.errors[0].item() == SITE_CLAIM
    assert state.errors[1].item() == 5 * 8 + 3  # the role's logical id
    assert state.errors[2].item() == 5  # its slot


def test_launches_cross_the_32_bit_epoch_boundary():
    """64-bit counters: tags stay nonzero and distinct, copies alternate and
    no claim trips across 2**32."""
    state = _State()
    start = 2**32 - 3
    state.epochs.view(GRID, LINE_WORDS // 2)[:, 0] = start
    state.claims.view(GRID, LINE_WORDS // 2)[:, 0] = start
    values = torch.zeros(GRID, dtype=torch.int32, device="cuda")
    seen = torch.zeros(GRID, dtype=torch.int32, device="cuda")
    placement = torch.zeros((8, GRID), dtype=torch.int32, device="cuda")
    tags = torch.zeros_like(placement)
    for _ in range(6):
        _launch_rounds(state, values, seen, placement, 1, tags=tags)
    torch.cuda.synchronize()
    assert state.errors.count_nonzero().item() == 0
    rows = [(start + n) % 8 for n in range(1, 7)]
    tag, copy = tags[rows].cpu() >> 1, tags[rows].cpu() & 1
    assert bool((tag != 0).all()) and bool((tag == tag[:, :1]).all())
    assert len(set(tag[:, 0].tolist())) == 6
    assert copy[:, 0].tolist() == [(start + n) & 1 for n in range(1, 7)]


_TRAP_CHILD = textwrap.dedent("""
    import sys
    sys.path.insert(0, sys.argv[1])
    import torch
    import test_persistent_gfx950 as t
    state = t._State()
    t._fail_stop_kernel[(t.GRID - 8,)](
        state.epochs, state.claims, state.joins, state.errors, 1_000_000,
        num_warps=8, **t.persistent_compile_options(),
    )
    torch.cuda.synchronize()
    print("NO TRAP")
    """)


@gluon.jit(do_not_specialize=["deadline_ticks"])
def _fail_stop_kernel(epochs, claims, joins, errors, deadline_ticks):
    entry = enter_issue(epochs, claims)
    launch = enter_check(entry, errors, errors, deadline_ticks, 0, fail_stop)
    grid_join(joins, launch, fail_stop, 0, 0)


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
