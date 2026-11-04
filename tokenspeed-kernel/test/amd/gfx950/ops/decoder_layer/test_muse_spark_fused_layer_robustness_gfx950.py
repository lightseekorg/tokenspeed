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

"""The fused Muse Spark layer's launch state under the conditions serving
creates, on 8 gfx950 ranks.

One state serves every launch of a group: launches alternate request counts
and the two copies of every Lamport area, epochs grow without bound, and a
rank can be late or never come. Each case compares launches with the
regular layer's math (``muse_layer_reference``) or with an earlier launch
bit for bit.
"""

import ctypes
import os
import socket
import subprocess
import sys
import textwrap
import time
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

WORLD = 8
if not (
    torch.cuda.is_available()
    and torch.cuda.device_count() >= WORLD
    and current_platform().is_cdna4
):
    pytest.skip("needs 8 gfx950 GPUs", allow_module_level=True)

from muse_layer_builder import HIDDEN, WIDTH, build_layer_inputs  # noqa: E402
from muse_layer_reference import (  # noqa: E402
    SEEDS,
    RoutingTie,
    assert_matches,
    failure_records,
    regular_reference,
    run_fused,
)
from tokenspeed_kernel.ops.monokernel import (  # noqa: E402
    PlanKey,
    prepare_decoder_layer_state,
)
from tokenspeed_kernel.ops.monokernel._gluon.muse_spark.iris import (  # noqa: E402
    CLEARED_AREAS,
)
from tokenspeed_kernel.ops.monokernel._gluon.muse_spark.layer import (  # noqa: E402
    EX_IDS_TAGS,
    EX_S_TAGS,
    EX_XQ_TAGS,
    KPARTS,
    MAX_ROWS,
)
from tokenspeed_kernel.ops.monokernel._gluon.muse_spark.prepared import (  # noqa: E402
    DEFAULT_PLAN_FIELDS,
    MuseSparkFusedDecoderLayer,
)
from tokenspeed_kernel_amd.ops.gfx950.persistent import GRID  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.persistent.mirror import (  # noqa: E402
    FailureMirror,
    _hip,
)
from tokenspeed_kernel_amd.ops.gfx950.persistent.sync import (  # noqa: E402
    FIRST_KERNEL_SITE,
    SITE_ABORTED,
)

ONE = PlanKey(1, "sliding", "sliding")
TWO = PlanKey(2, "sliding", "sliding")
CONTEXT = {1: (1500,), 2: (1500, 2600)}
TICKS_PER_SECOND = 100_000_000  # the 100 MHz device clock
# int64 words per 128 B line: one epoch or claim counter per line.
LINE_INT64 = 16


def _tag(epoch: int) -> int:
    """``launch_tag(epoch, 32)`` as the int32 word the kernel stores."""
    tag = epoch % ((1 << 32) - 1) + 1
    return tag - (1 << 32) if tag >= 1 << 31 else tag


def _epoch(state) -> int:
    """The epoch of the group's last launch (program 0's counter)."""
    return int(state.epochs[0])


def _layer(state, weights, deadline_seconds: float):
    return MuseSparkFusedDecoderLayer(
        plan_keys=[ONE, TWO], weights=weights, state=state,
        plan_fields=DEFAULT_PLAN_FIELDS, stamps=False,
        deadline_ticks=int(deadline_seconds * TICKS_PER_SECOND),
    )  # fmt: skip


def _state(rank, record_dir):
    return prepare_decoder_layer_state(
        recipe="muse_spark", group=dist.group.WORLD, rank_in_group=rank,
        max_requests=2, failure_record_path=f"{record_dir}/rank{rank}.bin",
        device=torch.device("cuda", rank),
    )  # fmt: skip


def _steps(rank):
    """Both request counts' inputs and references; one weights dict (the
    builder draws the same weights for both, from the same seed)."""
    device = torch.device("cuda", rank)
    for seed in SEEDS:
        weights, one = build_layer_inputs(ONE, rank, seed, device, CONTEXT[1])
        _, two = build_layer_inputs(TWO, rank, seed, device, CONTEXT[2])
        steps = {ONE: one, TWO: two}
        try:
            references = {
                key: regular_reference(key, weights, steps[key]) for key in steps
            }
        except RoutingTie:
            continue
        return weights, steps, references
    raise AssertionError("every seed has a routing tie")


def _launch(fused, key, inputs, residual=None):
    residual = inputs["residual"].clone() if residual is None else residual
    actual = run_fused(fused, key, inputs, residual)
    torch.cuda.synchronize()
    return actual


def _assert_areas_empty(state):
    """After launch L the copy of every Lamport area that launch L + 1 uses
    is empty: the senders cleared it at L's entry, and peers write it only in
    L + 1 (every rank has finished L, none has started L + 1)."""
    dist.barrier()
    copy = (_epoch(state) + 1) & 1
    areas = dict(
        att_box=state.iris.box, h1=state.iris.h1, h3=state.iris.h3,
        h4_own=state.iris.h4_own, routes=state.iris.routes, h6=state.iris.h6,
    )  # fmt: skip
    for area in CLEARED_AREAS:
        words = areas[area.name].view(2, area.copy_words)[copy]
        sentinel = area.sentinel - (1 << 32)
        assert bool((words == sentinel).all()), (area.name, _epoch(state))
    dist.barrier()


def _write_exchange(state, word_offset: int, values: list[int]) -> None:
    """Store int32 ``values`` into the uncached exchange at ``word_offset``."""
    host = torch.tensor(values, dtype=torch.int32)
    status = _hip().hipMemcpy(
        ctypes.c_void_p(state.exchange.data_ptr() + 4 * word_offset),
        ctypes.c_void_p(host.data_ptr()),
        ctypes.c_size_t(host.numel() * 4),
        1,  # hipMemcpyHostToDevice
    )
    assert status == 0, status


def _alternation(rank, record_dir):
    """1 -> 2 -> 1 requests on one state; before the second 1-request
    launch, the next launch's tag sits in the exchange's per-row tag lines
    of rows 8-15 (ids, xq, every S part) and NaN in their S and h' rows. The
    launch must match the reference without reading rows 8-15, and leave a
    residual's rows 8-15 alone."""
    state = _state(rank, record_dir)
    weights, steps, references = _steps(rank)
    fused = _layer(state, weights, 2.0)
    for key in (ONE, TWO):
        assert_matches(key, _launch(fused, key, steps[key]), references[key])
        _assert_areas_empty(state)

    stale = _tag(_epoch(state) + 1)
    rows = range(WIDTH, 2 * WIDTH)
    _write_exchange(state, EX_IDS_TAGS + WIDTH, [stale] * WIDTH)
    _write_exchange(state, EX_XQ_TAGS + WIDTH, [stale] * WIDTH)
    for part in range(KPARTS):
        _write_exchange(state, EX_S_TAGS + part * MAX_ROWS + WIDTH, [stale] * WIDTH)
    state.s_rows[rows.start : rows.stop] = float("nan")
    state.hprime[rows.start : rows.stop] = float("nan")
    # A 16-row residual whose first 8 rows are the step's.
    residual = torch.randn((2 * WIDTH, HIDDEN), device=f"cuda:{rank}")
    residual[:WIDTH] = steps[ONE]["residual"]
    untouched = residual[WIDTH:].clone()
    actual = _launch(fused, ONE, steps[ONE], residual[:WIDTH])
    assert_matches(ONE, actual, references[ONE])
    assert torch.equal(residual[WIDTH:], untouched)
    _assert_areas_empty(state)


def _replays(rank, record_dir, replays: int):
    """Captured launches at both request counts, replayed alternately: each
    replay repeats the eager result bit for bit, and leaves the next copy of
    every Lamport area empty."""
    state = _state(rank, record_dir)
    weights, steps, _ = _steps(rank)
    fused = _layer(state, weights, 2.0)
    eager, graphs, outputs = {}, {}, {}
    for key in (ONE, TWO):
        eager[key] = _launch(fused, key, steps[key])
        graphs[key] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[key]):
            outputs[key] = run_fused(
                fused, key, steps[key], steps[key]["residual"].clone()
            )
    for _ in range(replays):
        for key in (ONE, TWO):
            graphs[key].replay()
            torch.cuda.synchronize()
            for name, value in eager[key].items():
                assert torch.equal(
                    value.view(torch.uint8), outputs[key][name].view(torch.uint8)
                ), (key, name)
            _assert_areas_empty(state)


def _epoch_wrap(rank, record_dir):
    """Every epoch and claim counter seeded at 2**32 - 3: six launches cross
    the 32-bit wrap of the tags (2**32 - 2, 2**32 - 1, 2**32, ...) and each
    matches the reference."""
    state = _state(rank, record_dir)
    for counters in (state.epochs, state.claims):
        counters.view(-1, LINE_INT64)[:GRID, 0] = (1 << 32) - 3
    weights, steps, references = _steps(rank)
    fused = _layer(state, weights, 2.0)
    for launch in range(6):
        key = (ONE, TWO)[launch % 2]
        assert_matches(key, _launch(fused, key, steps[key]), references[key])
        _assert_areas_empty(state)
    assert _epoch(state) == (1 << 32) + 3


def _late_rank(rank, record_dir):
    """One rank enters a launch 30 s after the others, within a 60 s
    deadline: every rank completes, bit for bit as an on-time launch."""
    state = _state(rank, record_dir)
    weights, steps, _ = _steps(rank)
    fused = _layer(state, weights, 60.0)
    on_time = _launch(fused, ONE, steps[ONE])
    dist.barrier()
    if rank == 3:
        time.sleep(30)
    late = _launch(fused, ONE, steps[ONE])
    for name, value in on_time.items():
        assert torch.equal(value.view(torch.uint8), late[name].view(torch.uint8))


def _worker(rank, port, record_dir, errors):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl", init_method=f"tcp://localhost:{port}", rank=rank, world_size=WORLD
        )
        _alternation(rank, record_dir)
        _replays(rank, record_dir, replays=4)
        _epoch_wrap(rank, record_dir)
        _late_rank(rank, record_dir)
    except Exception:
        errors[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def test_one_state_serves_alternation_replays_wrap_and_a_late_rank(tmp_path):
    errors = mp.Manager().dict()
    try:
        mp.spawn(
            _worker, args=(_free_port(), str(tmp_path), errors), nprocs=WORLD,
            join=True,
        )  # fmt: skip
    except Exception as error:
        raise AssertionError(failure_records(tmp_path, WORLD)) from error
    assert not errors, "\n".join(f"rank {r}: {e}" for r, e in errors.items())


# One rank of a withheld launch, in its own process: the launch traps.
_WITHHELD_CHILD = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, sys.argv[1])
    rank, port, record_dir, withheld = int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
    import torch, torch.distributed as dist
    import test_muse_spark_fused_layer_robustness_gfx950 as t
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://localhost:{port}", rank=rank, world_size=t.WORLD
    )
    state = t._state(rank, record_dir)
    weights, steps, _ = t._steps(rank)
    fused = t._layer(state, weights, 0.5)
    t._launch(fused, t.ONE, steps[t.ONE])
    dist.barrier()
    if rank == withheld:
        # Stay alive (peers push into this rank's areas) past the deadline.
        time.sleep(20)
        print("WITHHELD")
        sys.exit(0)
    t._launch(fused, t.ONE, steps[t.ONE])
    print("NO TRAP")
    """)


def test_a_withheld_launch_stops_every_peer_with_the_origin(tmp_path):
    """Rank 5 skips launch 2. Every other rank stops at its 0.5 s deadline
    (a queue error, not a hang); each record names its origin, a rank that
    timed out itself and its site, and the withheld rank records nothing."""
    withheld, port = 5, _free_port()
    here = os.path.dirname(os.path.abspath(__file__))
    start = time.monotonic()
    children = [
        subprocess.Popen(
            [sys.executable, "-c", _WITHHELD_CHILD, here, str(rank), str(port),
             str(tmp_path), str(withheld)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )  # fmt: skip
        for rank in range(WORLD)
    ]
    outputs = [child.communicate(timeout=600)[0] for child in children]
    assert time.monotonic() - start < 600
    records = [
        FailureMirror.read(f"{tmp_path}/rank{rank}.bin") for rank in range(WORLD)
    ]
    for rank, (child, output) in enumerate(zip(children, outputs)):
        if rank == withheld:
            assert child.returncode == 0 and "WITHHELD" in output, (
                output[-2000:] + "\n" + failure_records(tmp_path, WORLD)
            )
            assert not any(records[rank]), records[rank]
        else:
            assert child.returncode != 0 and "NO TRAP" not in output, output[-2000:]
    # Ranks that timed out themselves (several may, waiting on the same
    # missing rank) record their own site; the others record the abort
    # marker of one of them: (origin rank + 1) << 16 | origin site.
    own, named = {}, set()
    for rank, (site, _, detail, epoch, _) in enumerate(records):
        if rank == withheld:
            continue
        assert epoch == 2, (rank, records[rank])
        if site == SITE_ABORTED:
            named.add(((detail >> 16) - 1, detail & 0xFFFF))
        else:
            assert site >= FIRST_KERNEL_SITE, (rank, records[rank])
            own[rank] = site
    assert own, records
    assert named <= set(own.items()), (named, own)
