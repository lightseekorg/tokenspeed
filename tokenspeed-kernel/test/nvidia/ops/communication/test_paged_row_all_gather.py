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

"""The fused paged-row all-gather against NCCL's all-gather of the packed rows,
on two and four node-local GB300 GPUs: every byte value, both slots, owners
without rows, source offsets past 2**31, both modes, group-consistent
refusals and argument errors, and the in-kernel failure records."""

import json
import multiprocessing
import os
import sys
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp
from tokenspeed_kernel.ops.communication.cute_dsl import (
    create_paged_row_all_gather_state,
    cute_dsl_paged_row_all_gather,
)
from utils import kernel_supported, move_pages_past_int32

ROW_BYTES = 576
CAP_ROWS = 4096
PLANE_ROWS = 8192
# Failure cases: in-kernel waits end after 2 s (the first window loads the
# module on every rank inside this bound too).
SHORT_TIMEOUT_NS = 2_000_000_000
LONG_TIMEOUT_NS = 60_000_000_000


def _init(rank, size, rendezvous):
    from tokenspeed_kernel.ops.communication.fabric import gather_fabric_map

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=300),
        device_id=device,
    )
    gather_fabric_map()
    return device, dist.new_group(backend="gloo")


def _create(group, device, *, modes, eligible, timeout_ns):
    return create_paged_row_all_gather_state(
        group,
        device,
        row_bytes=ROW_BYTES,
        cap_bytes=CAP_ROWS * ROW_BYTES,
        timeout_ns=timeout_ns,
        modes=modes,
        eligible=eligible,
    )


def _agree(device, failure, what):
    """Fail on every rank when any rank failed (``failure``: this rank's
    reason, None when fine), so the next check's collectives still line up."""
    flag = torch.tensor([0 if failure else 1], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    assert int(flag) == 1, f"{what} failed on some rank; this rank: {failure or 'ok'}"


def _owned_rows(shapes, size):
    """Every rank's row count of a window, from ``shapes``, a generator seeded
    alike on every rank; at least one row in the window."""
    owned = torch.randint(0, CAP_ROWS + 1, (size,), generator=shapes).tolist()
    owned[0] = max(owned[0], 1)
    return owned


def _fetch(generator, owned, send_rows, device):
    """This rank's ``fetch_slots``: ``owned`` distinct random plane rows, then
    row 0 up to ``send_rows`` as the runtime's planner pads."""
    fetch = torch.zeros(send_rows, dtype=torch.int32)
    fetch[:owned] = torch.randperm(PLANE_ROWS, generator=generator)[:owned].to(
        torch.int32
    )
    return fetch.to(device)


def _expected(plane, fetch, size):
    """NCCL's all-gather of every rank's ``send_rows`` packed rows."""
    sent = plane.index_select(0, fetch.to(torch.int64))
    out = sent.new_empty((size * sent.shape[0], ROW_BYTES))
    dist.all_gather_into_tensor(out, sent)
    return out


def _compare(got, want, owned, send_rows):
    """Only the rows each rank owns are defined in the fused slot."""
    for peer, count in enumerate(owned):
        rows = slice(peer * send_rows, peer * send_rows + count)
        if not torch.equal(got[rows], want[rows]):
            return f"rank {peer}'s rows differ"
    return None


def _check_windows(state, device, size, plane, generators, epochs):
    """``epochs`` windows, each launched before the previous one is read, read
    on a consumer stream that orders each slot's reads before its reuse."""
    shapes, generator = generators
    rank = dist.get_rank()
    consumer = torch.cuda.Stream()
    consumer.wait_stream(torch.cuda.current_stream())
    windows, copies = [], []
    with torch.cuda.stream(consumer):
        for i in range(epochs + 1):
            if i < epochs:
                owned = _owned_rows(shapes, size)
                if i % 3 == 1:
                    # Half the ranks own no row of this window.
                    owned = [0 if p % 2 else count for p, count in enumerate(owned)]
                send_rows = max(owned)
                fetch = _fetch(generator, owned[rank], send_rows, device)
                epoch, done = cute_dsl_paged_row_all_gather(
                    state, plane, fetch, owned[rank], send_rows, i
                )
                windows.append((fetch, owned, send_rows, epoch, done))
            if i >= 1:
                _, owned, send_rows, epoch, done = windows[i - 1]
                consumer.wait_event(done)
                copies.append(state.receive_view(epoch)[: size * send_rows].clone())
    torch.cuda.current_stream().wait_stream(consumer)
    failure = None
    for (fetch, owned, send_rows, epoch, _), got in zip(windows, copies):
        mismatch = _compare(got, _expected(plane, fetch, size), owned, send_rows)
        if failure is None and mismatch is not None:
            failure = f"epoch {epoch}: {mismatch}"
    return failure


def _worker(rank, size, rendezvous):
    from tokenspeed_kernel.ops.communication._cute_dsl import paged_row_all_gather
    from tokenspeed_kernel.ops.communication.multimem import multimem_available

    device, group = _init(rank, size, rendezvous)
    # Window shapes alike on every rank; rows and their order per rank.
    shapes = torch.Generator().manual_seed(99)
    generator = torch.Generator().manual_seed(1234 + rank)
    # Every byte value, NaN and denormal patterns included.
    plane = torch.randint(
        0, 256, (PLANE_ROWS, ROW_BYTES), dtype=torch.uint8, generator=generator
    ).to(device)
    state = _create(
        group,
        device,
        modes=("multicast", "peer"),
        eligible=True,
        timeout_ns=LONG_TIMEOUT_NS,
    )
    _agree(device, None if state is not None else "no state", "the state")
    # Multicast where every rank supports multicast objects, else peer stores.
    with torch.cuda.device(device):
        multicast = torch.tensor(
            [int(multimem_available())], dtype=torch.int32, device=device
        )
    dist.all_reduce(multicast, op=dist.ReduceOp.MIN)
    expected = "multicast" if int(multicast) else "peer"
    _agree(
        device,
        None if state.mode == expected else f"mode {state.mode}",
        f"the {expected} mode inside a node",
    )
    # The first window waited under its own short bound; the later ones wait
    # up to the caller's (no public accessor: the kernel reads the bound).
    bound = int(state.params[paged_row_all_gather._PARAM_TIMEOUT])
    _agree(
        device,
        None if bound == LONG_TIMEOUT_NS else f"bound {bound}",
        "the wait bound after the first window",
    )
    compiles = patch.object(
        paged_row_all_gather,
        "compile_paged_row_all_gather",
        wraps=paged_row_all_gather.compile_paged_row_all_gather,
    )
    with compiles as compile_calls:
        failure = _check_windows(state, device, size, plane, (shapes, generator), 64)
        _agree(device, failure, f"{expected} windows")
        # The same rows past 2**31 bytes of the plane: int64 source offsets.
        (far,), far_slots = move_pages_past_int32(
            (plane,), torch.arange(PLANE_ROWS, dtype=torch.int32, device=device)
        )
        offset = int(far_slots[0])
        _agree(
            device,
            None if offset * ROW_BYTES > 2**31 else "no offset past 2**31",
            "the far plane",
        )
        owned = [max(count, 1) for count in _owned_rows(shapes, size)]
        send_rows = max(owned)
        fetch = _fetch(generator, owned[rank], send_rows, device)
        epoch, done = cute_dsl_paged_row_all_gather(
            state, far, fetch + offset, owned[rank], send_rows, 0
        )
        torch.cuda.current_stream().wait_event(done)
        got = state.receive_view(epoch)[: size * send_rows].clone()
        failure = _compare(got, _expected(plane, fetch, size), owned, send_rows)
        _agree(device, failure, "rows past 2**31")
        del far
    _agree(
        device,
        (
            None
            if compile_calls.call_count == 0
            else f"{compile_calls.call_count} compiles"
        ),
        "launches after the state's creation",
    )
    _agree(device, None if state.read_error() is None else "error record", "errors")
    state.close()

    peer = _create(
        group, device, modes=("peer",), eligible=True, timeout_ns=LONG_TIMEOUT_NS
    )
    _agree(
        device,
        None if peer is not None and peer.mode == "peer" else "no peer state",
        "the peer state",
    )
    failure = _check_windows(peer, device, size, plane, (shapes, generator), 16)
    _agree(device, failure, "peer windows")
    peer.close()

    # A refusal on one rank is every rank's: an allocation failure, then the
    # caller's own verdict. Nobody waits in the rendezvous.
    if rank == 1:
        with patch.object(symm_mem, "empty", side_effect=RuntimeError("no memory")):
            refused = _create(
                group,
                device,
                modes=("multicast", "peer"),
                eligible=True,
                timeout_ns=LONG_TIMEOUT_NS,
            )
    else:
        refused = _create(
            group,
            device,
            modes=("multicast", "peer"),
            eligible=True,
            timeout_ns=LONG_TIMEOUT_NS,
        )
    _agree(device, None if refused is None else "a state", "an allocation failure")
    refused = _create(
        group,
        device,
        modes=("multicast", "peer"),
        eligible=rank != 1,
        timeout_ns=LONG_TIMEOUT_NS,
    )
    _agree(device, None if refused is None else "a state", "an ineligible rank")
    # The same when the kernel module does not import on one rank.
    with patch.dict(
        sys.modules, {paged_row_all_gather.__name__: None} if rank == 1 else {}
    ):
        refused = _create(
            group,
            device,
            modes=("multicast", "peer"),
            eligible=True,
            timeout_ns=LONG_TIMEOUT_NS,
        )
    _agree(device, None if refused is None else "a state", "a failed import")
    # A bad argument on one rank raises there after the vote; the others get
    # None. Unequal arguments raise on every rank.
    try:
        refused = _create(
            group,
            device,
            modes=("bogus",) if rank == 1 else ("multicast", "peer"),
            eligible=True,
            timeout_ns=LONG_TIMEOUT_NS,
        )
        failure = "no error" if rank == 1 else (None if refused is None else "a state")
    except ValueError:
        failure = None if rank == 1 else "raised"
    _agree(device, failure, "bad modes on one rank")
    try:
        create_paged_row_all_gather_state(
            group,
            device,
            row_bytes=ROW_BYTES,
            cap_bytes=(CAP_ROWS + rank) * ROW_BYTES,
            timeout_ns=LONG_TIMEOUT_NS,
            modes=("multicast", "peer"),
            eligible=True,
        )
        failure = "no error"
    except ValueError:
        failure = None
    _agree(device, failure, "unequal capacities")
    dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 4], ids=["tp2", "tp4"])
def test_paged_row_all_gather(size, tmp_path):
    if torch.cuda.device_count() < size or not kernel_supported(
        "cute_dsl_paged_row_all_gather"
    ):
        pytest.skip(f"requires {size} node-local GB300 GPUs")
    mp.spawn(
        _worker,
        args=(size, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=size,
        join=True,
    )


def _failing_worker(rank, size, rendezvous, case, out):
    """Two good windows, then ``case`` on rank 1 in the third: ``header``
    launches it with another header, ``skip`` skips its epoch and launches
    the next one. Writes this rank's error record once the kernel trapped,
    with host-only operations, then exits."""
    device, group = _init(rank, size, rendezvous)
    plane = torch.zeros((PLANE_ROWS, ROW_BYTES), dtype=torch.uint8, device=device)
    state = _create(
        group,
        device,
        modes=("multicast", "peer"),
        eligible=True,
        timeout_ns=SHORT_TIMEOUT_NS,
    )
    fetch = torch.arange(64, dtype=torch.int32, device=device)
    record = None
    try:
        for window in range(3):
            header = window
            if rank == 1 and window == 2:
                if case == "skip":
                    # The declared epoch counter, moved past one window.
                    state.epoch += 1
                else:
                    header += 1
            _, done = cute_dsl_paged_row_all_gather(state, plane, fetch, 64, 64, header)
            torch.cuda.current_stream().wait_event(done)
        torch.cuda.synchronize(device)
    except Exception:  # noqa: BLE001 - the trap surfaces as a CUDA error
        record = state.read_error()
    with open(os.path.join(out, f"rank{rank}.json"), "w") as f:
        json.dump(record, f)
    os._exit(0)


@pytest.mark.parametrize("case", ["header", "skip"])
def test_paged_row_all_gather_failures(case, tmp_path):
    """Each rank in a process of its own (a trapped context cannot take part
    in a collective, and ``mp.spawn`` stops the others when one fails): the
    error records name the expected stage on every rank."""
    size = 4
    if torch.cuda.device_count() < size or not kernel_supported(
        "cute_dsl_paged_row_all_gather"
    ):
        pytest.skip(f"requires {size} node-local GB300 GPUs")
    # Imported once the platform is known to have CuTe DSL.
    from tokenspeed_kernel.ops.communication._cute_dsl.paged_row_all_gather import (
        STAGE_ENTRY_AHEAD,
        STAGE_ENTRY_TIMEOUT,
        STAGE_HEADER,
    )

    context = multiprocessing.get_context("spawn")
    rendezvous = f"file://{tmp_path / 'rendezvous'}"
    workers = [
        context.Process(
            target=_failing_worker, args=(rank, size, rendezvous, case, str(tmp_path))
        )
        for rank in range(size)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=300)
    stuck = [rank for rank, worker in enumerate(workers) if worker.is_alive()]
    for worker in workers:
        if worker.is_alive():
            worker.kill()
    assert not stuck, f"ranks {stuck} did not finish"
    records = []
    for rank in range(size):
        with open(tmp_path / f"rank{rank}.json") as f:
            records.append(json.load(f))
    assert all(record is not None for record in records), records
    # Several CTAs fail at once on a rank; the record is still written whole.
    assert all(record["complete"] for record in records), records
    stages = [record["stage_code"] for record in records]
    if case == "header":
        assert stages == [STAGE_HEADER] * size, records
    else:
        # The others see rank 1's credit ahead of their epoch; rank 1 waits
        # for credits that never come.
        assert stages == [
            STAGE_ENTRY_AHEAD,
            STAGE_ENTRY_TIMEOUT,
            STAGE_ENTRY_AHEAD,
            STAGE_ENTRY_AHEAD,
        ], records
        others = [record for rank, record in enumerate(records) if rank != 1]
        assert all(record["mask"] & 0b10 for record in others), records
