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
import utils
from tokenspeed_kernel._triton import gl, gluon
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("Persistent helpers target gfx950", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950.persistent import (  # noqa: E402
    ENTRY_WORDS,
    ERROR_ABORT,
    ERROR_WORDS,
    GRID,
    JOIN_WORDS,
    LINE_WORDS,
    SITE_ABORTED,
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
_WORDS = LINE_WORDS // 2


def _claim_mark(epoch, pid):
    """The vector entry's claim word for ``pid`` in launch ``epoch``: its
    tag (``launch_tag(epoch, 32)``) high, the program id low."""
    return ((epoch % (2**32 - 1) + 1) << 32) | pid


def _check_counters(state, launches):
    """After ``launches`` launches from zeroed counters: every workgroup's
    epoch word advanced once per launch, every role was claimed once per
    launch by one program id, and no other word was touched."""
    epochs = state.epochs.view(GRID, _WORDS).cpu()
    claims = state.claims.view(GRID, _WORDS).cpu()
    assert bool((epochs[:, 0] == launches).all())
    assert bool((epochs[:, 1:] == 0).all())
    assert bool((claims[:, 0] >> 32 == _claim_mark(launches, 0) >> 32).all())
    # Each role was claimed by one program id.
    assert sorted((claims[:, 0] & 0xFFFFFFFF).tolist()) == list(range(GRID))
    assert bool((claims[:, 1:] == 0).all())


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


def _launch_rounds(state, values, seen, placement, rounds, grid=GRID, deadline=None, tags=None):  # fmt: skip
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
    _check_counters(state, launches)
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
    # Claim role (xcd 3, slot 5) for launch 1 as a second workgroup taking
    # it would have: its counter advanced once, or program id 77's mark.
    claims = state.claims.view(GRID, _WORDS)
    claims[3 * 32 + 5, 0] = _claim_mark(1, 77)
    _launch_rounds(state, values, seen, placement, 1)
    torch.cuda.synchronize()
    assert state.errors[0].item() == SITE_CLAIM
    assert state.errors[1].item() == 5 * 8 + 3  # the role's logical id
    # The record names the other claimer's program id.
    assert state.errors[2].item() == 77


def test_launches_cross_the_32_bit_epoch_boundary():
    """64-bit counters: tags stay nonzero and distinct, copies alternate and
    no claim trips across 2**32."""
    state = _State()
    start = 2**32 - 3
    state.epochs.view(GRID, _WORDS)[:, 0] = start
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


@gluon.jit(do_not_specialize=["deadline_ticks"])
def _wave_epochs_kernel(epochs, claims, errors, seen, deadline_ticks, LAUNCHES: gl.constexpr):  # fmt: skip
    """Each wave of each workgroup records the epoch it entered with in
    ``seen[epoch % LAUNCHES, pid, wave]`` (one lane per wave stores)."""
    entry = enter_issue(epochs, claims)
    launch = enter_check(entry, errors, errors, deadline_ticks, 0, _record_and_return)
    lanes: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    lane = gl.arange(0, 64 * gl.num_warps(), lanes)
    row = (launch.epoch % LAUNCHES).to(gl.int32)
    slot = (row * _GRID + gl.program_id(0)) * gl.num_warps() + lane // 64
    value = gl.zeros([64 * gl.num_warps()], gl.int64, lanes) + launch.epoch
    gl.store(seen + slot, value, mask=lane % 64 == 0)


def test_every_wave_enters_with_the_same_epoch_advancing_by_one():
    """In every launch all 8 waves of all workgroups see the same epoch
    (each wave reads the epoch word itself), it grows by exactly 1 per launch, eager or replayed, wherever
    dispatch starts, and afterwards the counters hold the launch count."""
    state = _State()
    launches = 6
    seen = torch.zeros((launches, GRID, 8), dtype=torch.int64, device="cuda")
    scratch = torch.zeros(1024, dtype=torch.int32, device="cuda")

    def launch():
        _wave_epochs_kernel[(GRID,)](
            state.epochs, state.claims, state.errors, seen, _DEADLINE_TICKS,
            LAUNCHES=launches, num_warps=8,
            **persistent_compile_options(),
        )  # fmt: skip

    for n in range(3):
        _rotate_dispatch(scratch, 1 + 5 * n)
        launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for n in range(3):
        _rotate_dispatch(scratch, 3 + 7 * n)
        graph.replay()
    torch.cuda.synchronize()
    assert state.errors.count_nonzero().item() == 0
    # Graph capture does not run the kernel: 6 launches, epochs 1..6.
    by_launch = seen.cpu()
    for epoch in range(1, launches + 1):
        assert bool((by_launch[epoch % launches] == epoch).all()), epoch
    _check_counters(state, launches)


_TRAP_CHILD = textwrap.dedent("""
    import os
    import sys
    sys.path[:0] = sys.argv[1].split(os.pathsep)
    import torch
    import test_persistent_gfx950 as t
    from tokenspeed_kernel_amd.ops.gfx950.persistent.mirror import FailureMirror
    state = t._State()
    mirror = FailureMirror(state.errors, sys.argv[2])
    # 8 workgroups are missing, so the join stalls. "aborted": a peer's
    # abort marker is already set, so the join's abort check stops it;
    # otherwise it runs to its deadline.
    if sys.argv[3] == "aborted":
        state.errors[t.ERROR_ABORT] = t._MARKER
    t._fail_stop_kernel[(t.GRID - 8,)](
        state.epochs, state.claims, state.joins, state.errors, 1_000_000,
        num_warps=8, **t.persistent_compile_options(),
    )
    print("LAUNCHED", flush=True)
    torch.cuda.synchronize()
    print("NO TRAP")
    """)


@gluon.jit(do_not_specialize=["deadline_ticks"])
def _fail_stop_kernel(epochs, claims, joins, errors, deadline_ticks):
    entry = enter_issue(epochs, claims)
    # The serving entry form.
    launch = enter_check(entry, errors, errors, deadline_ticks, 0, fail_stop)
    grid_join(joins, launch, fail_stop, 0, 0)


# This module's directory and the test root (``utils``) for the child.
_CHILD_PATH = os.pathsep.join(
    [os.path.dirname(__file__), os.path.dirname(utils.__file__)]
)

# A peer's abort marker: (origin rank + 1) << 16 | the origin's site.
_MARKER = (5 + 1) << 16 | 40


def _run_trap_child(tmp_path, mode):
    from tokenspeed_kernel_amd.ops.gfx950.persistent.mirror import FailureMirror

    path = str(tmp_path / f"failure-record-{mode}")
    start = time.monotonic()
    child = subprocess.run(
        [sys.executable, "-c", _TRAP_CHILD, _CHILD_PATH, path, mode],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert child.returncode != 0, child.stdout + child.stderr
    # The kernel compiled and launched (a child that fails before its launch
    # also exits nonzero, with only the zeroed record).
    assert "LAUNCHED" in child.stdout, child.stderr[-4000:]
    assert "NO TRAP" not in child.stdout
    assert time.monotonic() - start < 300
    # A child that died before its kernel ran left no record.
    assert os.path.exists(path), child.stderr[-4000:]
    return FailureMirror.read(path)


def test_expired_wait_stops_the_queue(tmp_path):
    """``fail_stop`` raises a queue error: the process ends, it never hangs,
    and its record survives in the host mirror for the parent to read."""
    site, cta, _, epoch, label = _run_trap_child(tmp_path, "expired")
    # The 8 missing workgroups stall the join: a join failure in launch 1.
    assert (site, epoch, label) == (SITE_JOIN, 1, 0)
    assert 0 <= cta < GRID


def test_aborted_wait_records_the_origin_marker(tmp_path):
    """A wait stopped by a peer's abort records ``SITE_ABORTED`` with the
    peer's marker as the detail (not the wait's own detail), so the host
    record names the rank and site that failed first."""
    site, cta, detail, epoch, _ = _run_trap_child(tmp_path, "aborted")
    assert (site, detail, epoch) == (SITE_ABORTED, _MARKER, 1)
    assert 0 <= cta < GRID
