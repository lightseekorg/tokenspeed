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

"""Lamport all-reduce MAX of int64 words: exact against the maximum of the
gathered words for every int64 value, mailbox reuse across row counts and
in-place calls, interleaving with another collective on the group, and graph
replay under rank skew."""

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_cdna4, reason="the word max targets gfx950"
)

MAX_ROWS = 1024
# The extremes, zero, minus one, and words whose 16-bit pieces or 32-bit
# halves equal the halves of the mailbox's empty marker.
SPECIAL = (
    -(1 << 63),
    (1 << 63) - 1,
    0,
    -1,
    0x0000800000008000,
    0x8000000080000000 - (1 << 64),
    0x7FFF80007FFF8000,
)


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _worker(rank, world_size, port, error_dict):
    try:
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        _check_word_max(rank, world_size, device)
    except Exception:
        error_dict[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _group_words(rows, world_size, step):
    """Every rank's words of one call, the same on every rank: ``[world, rows]``."""
    generator = torch.Generator().manual_seed(1000 * step + rows)
    words = torch.randint(
        -(1 << 63), (1 << 63) - 1, (world_size, rows), generator=generator
    )
    special = torch.tensor(SPECIAL, dtype=torch.int64)
    for peer in range(world_size):
        # Each rank sends the special words at its own rows, and the row's
        # maximum is a special word wherever every other rank sends less.
        columns = (torch.arange(len(SPECIAL)) * world_size + peer) % rows
        words[peer, columns] = special
    # Ties across ranks.
    words[:, rows // 2] = words[0, rows // 2]
    return words


def _check_word_max(rank, world_size, device):
    from tokenspeed_kernel.ops.communication.iris import (
        create_iris_state,
        iris_all_reduce,
        iris_lamport_word_max,
        prepare_iris_lamport_word_max,
    )

    all_reduce = create_iris_state(
        group=dist.group.WORLD,
        rank_in_group=rank,
        staged_max_numel=64 * 1024,
        producer_direct_max_numel=8 * 8192,
        dtype=torch.bfloat16,
        heap_size=None,
        device=device,
    )
    state = prepare_iris_lamport_word_max(all_reduce, MAX_ROWS)
    mailbox = state.mailbox.data_ptr()
    # One row, a partial subgroup, whole and partial 256-row blocks, the
    # capacity, and a few rows again: every call flips the mailbox stage.
    rows_sequence = (1, 7, 64, 256, 257, MAX_ROWS, 3)
    other = torch.full((4, 2048), float(rank + 1), device=device).bfloat16()

    def inputs(step):
        return [
            _group_words(rows, world_size, step)[rank].to(device)
            for rows in rows_sequence
        ]

    def expected(step):
        return [
            _group_words(rows, world_size, step).amax(0).to(device)
            for rows in rows_sequence
        ]

    words = inputs(0)
    in_place = words[2].clone()

    def sequence():
        outputs = [iris_lamport_word_max(state, w) for w in words]
        # Another collective on the group in between, as the target's
        # reductions run between the walk's in one stream.
        outputs.append(iris_all_reduce(all_reduce, other.clone(), safe=False))
        outputs.append(iris_lamport_word_max(state, in_place, out=in_place))
        return outputs

    def check(outputs, step, in_place_expected):
        reduced_other = outputs.pop(len(rows_sequence))
        assert torch.all(reduced_other == world_size * (world_size + 1) / 2)
        for got, want in zip(outputs, expected(step) + [in_place_expected]):
            assert torch.equal(got, want)

    check(sequence(), 0, expected(0)[2])
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = sequence()
    for step in range(1, 5):
        for w, fresh in zip(words, inputs(step)):
            w.copy_(fresh)
        in_place.copy_(words[2])
        # A late rank finds its peers' pushes already waiting, and its own
        # pushes reach peers still polling.
        if rank == step % world_size:
            torch.cuda._sleep(20000)
        graph.replay()
        check(list(outputs), step, expected(step)[2])
    torch.cuda.synchronize()
    dist.barrier()

    # Replay and reuse never moved the captured mailboxes.
    assert state.mailbox.data_ptr() == mailbox
    # Every rank restored the slots it read: no piece is left behind.
    assert torch.all(state.mailbox.view(torch.int32) == -(1 << 31))
    dist.barrier()


@pytest.mark.parametrize("world_size", [4, 8])
def test_lamport_word_max_matches_the_gathered_maximum(world_size):
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    pytest.importorskip("iris")
    error_dict = mp.Manager().dict()
    mp.spawn(
        _worker,
        args=(world_size, _get_open_port(), error_dict),
        nprocs=world_size,
        join=True,
    )
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))
