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
on node-local SM100 or SM103 GPUs: random windows of any row count, every byte
value, both slots, owners without rows, and both modes with each group size's
CTA count. The runtime's test of the KVP history reads through this kernel
(``test/runtime/distributed/test_kvp_history_fused_gather.py``), which CI runs
on two GPUs, checks the rest: no trace at launch, source offsets past 2**31,
the refusal votes and argument errors, and the in-kernel failure records."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.ops.communication.cute_dsl import (
    create_paged_row_all_gather_state,
    cute_dsl_paged_row_all_gather,
)
from utils import kernel_supported

ROW_BYTES = 576
CAP_ROWS = 4096
PLANE_ROWS = 8192
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
    failure = _check_windows(state, device, size, plane, (shapes, generator), 64)
    _agree(device, failure, f"{expected} windows")
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
    dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 4, 8], ids=["tp2", "tp4", "tp8"])
def test_paged_row_all_gather(size, tmp_path):
    if (
        torch.cuda.device_count() < size
        or not kernel_supported("cute_dsl_paged_row_all_gather")
        # The registered range admits 10.1 and 10.2 too.
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3))
    ):
        pytest.skip(f"requires {size} node-local SM100 or SM103 GPUs")
    mp.spawn(
        _worker,
        args=(size, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=size,
        join=True,
    )
