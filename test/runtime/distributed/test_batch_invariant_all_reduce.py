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

"""The batch-invariant all-reduce on real groups: the route and the ordered
fold, end to end on several GPUs."""

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HIDDEN = 2048


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def ill_conditioned_payload(
    rank_in_group: int, world_size: int, rows: int, device
) -> torch.Tensor:
    """This rank's slice of a payload whose group sum exposes the order.

    Per element one rank holds ``+B``, another ``-B`` and the rest values whose
    low bits fall below fp32's resolution at ``B``, so almost any two
    association orders round the sum differently. Every rank draws the whole
    assignment from one seed and keeps its own slice.
    """
    gen = torch.Generator(device=device).manual_seed(7)
    holder = torch.rand(world_size, rows, HIDDEN, generator=gen, device=device)
    holder = holder.argsort(dim=0)
    big = 2.0 ** torch.randint(4, 12, (rows, HIDDEN), generator=gen, device=device)
    small = torch.randn(world_size, rows, HIDDEN, generator=gen, device=device)
    mine = small[rank_in_group] * big * 2.0**-18
    mine = torch.where(holder[0] == rank_in_group, big, mine)
    mine = torch.where(holder[1] == rank_in_group, -big, mine)
    return mine.to(torch.bfloat16)


def worker_fn(rank, world_size, port, attn_tp, error_dict):
    try:
        worker_main(rank, world_size, port, attn_tp)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def worker_main(rank: int, world_size: int, port: int, attn_tp: int) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    from tokenspeed.runtime.distributed.comm_backend import get_global_backend
    from tokenspeed.runtime.distributed.comm_backend.auto import (
        AutoBackend,
        Collective,
        Route,
    )
    from tokenspeed.runtime.distributed.comm_ops import all_reduce
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = Mapping(
        rank=rank,
        world_size=world_size,
        attn_tp_size=attn_tp,
        dense_tp_size=attn_tp,
    )
    pg_manager.init_distributed(
        mapping=mapping,
        distributed_init_method=f"tcp://localhost:{port}",
        backend="nccl",
    )
    for group in (
        mapping.world_group,
        mapping.attn.tp_group,
        mapping.dense.tp_group,
        mapping.moe.tp_ep_group,
    ):
        if len(group) > 1:
            pg_manager.init_process_group(group)

    rows = 64
    global_server_args_dict["mapping"] = mapping
    global_server_args_dict["chunked_prefill_size"] = rows * world_size
    global_server_args_dict["max_prefill_tokens"] = rows * world_size
    global_server_args_dict["max_model_len"] = 4096
    global_server_args_dict["max_num_seqs"] = rows
    global_server_args_dict["batch_invariant_collectives"] = True
    global_server_args_dict["force_deterministic_rsag"] = False
    try:
        backend = get_global_backend()
        assert isinstance(backend, AutoBackend)
        group = mapping.attn.tp_group
        world_pg = pg_manager.get_process_group("nccl", mapping.world_group)
        probe = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)
        for collective in (Collective.ALL_REDUCE, Collective.TOKEN_REDUCE_SCATTER):
            assert backend.route(collective, probe, group) is Route.ORDERED_FOLD

        # The public all-reduce every layer calls: in place, equal on every
        # rank, and the rank-ordered fp32 fold of the partials bit for bit.
        payload = ill_conditioned_payload(group.index(rank), len(group), rows, device)
        full = payload.clone()
        assert all_reduce(full, group) is full
        gathered = [torch.empty_like(full) for _ in mapping.world_group]
        dist.all_gather(gathered, full, group=world_pg)
        for peer in group:
            assert torch.equal(gathered[peer], full), f"rank {peer} disagrees"
        # Groups of the same kind reduce identical inputs (the payload is
        # seeded by rank in group) to identical bits, whichever GPUs they are.
        for peer in mapping.world_group:
            assert torch.equal(gathered[peer], full), f"group of {peer} disagrees"
        parts = [torch.empty_like(payload) for _ in mapping.world_group]
        dist.all_gather(parts, payload, group=world_pg)
        folded = torch.zeros(rows, HIDDEN, dtype=torch.float32, device=device)
        for peer in group:
            folded += parts[peer].float()
        assert torch.equal(full, folded.to(torch.bfloat16))

        # A row's bits depend on neither how many rows ride along nor where it
        # sits in the batch. The in-switch reduction fails the second on this
        # payload: its order follows the row's offset in the buffer.
        for row_count in (1, 5, rows // 2):
            part = payload[:row_count].clone()
            all_reduce(part, group)
            assert torch.equal(part, full[:row_count]), row_count
        for shift in (1, 3, 16):
            moved = torch.cat([payload[:shift].flip(0), payload[: rows - shift]])
            all_reduce(moved, group)
            assert torch.equal(moved[shift:], full[: rows - shift]), shift

        # The armed workspace folds the same bits (-0.0 as +0.0), chunked past capacity.
        big, capacity = 512, 200
        payload = ill_conditioned_payload(group.index(rank), len(group), big, device)
        payload[:, :8] = -0.0
        parts = [torch.empty_like(payload) for _ in mapping.world_group]
        dist.all_gather(parts, payload, group=world_pg)
        folded = torch.zeros(big, HIDDEN, dtype=torch.float32, device=device)
        for peer in group:
            folded += parts[peer].float()
        expected = folded.to(torch.bfloat16)
        if backend.trtllm_ar.configure_group(
            rank=group.index(rank),
            group=group,
            max_token_num=capacity,
            hidden_dim=HIDDEN,
        ):
            for row_count in (5, capacity, big):
                folded_rows = backend.trtllm_ar.ordered_all_reduce(
                    payload[:row_count].clone(), group
                )
                assert folded_rows is not None, row_count
                assert torch.equal(
                    folded_rows.view(torch.int16),
                    expected[:row_count].view(torch.int16),
                ), row_count
            folded_rows = all_reduce(payload.clone(), group)
            assert torch.equal(
                folded_rows.view(torch.int16), expected.view(torch.int16)
            )
    finally:
        dist.barrier()
        dist.destroy_process_group()


def run_test(world_size: int, attn_tp: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the multi-GPU all-reduce")
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    port = get_open_port()
    error_dict = mp.Manager().dict()
    mp.spawn(
        worker_fn,
        args=(world_size, port, attn_tp, error_dict),
        nprocs=world_size,
        join=True,
    )
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def test_batch_invariant_all_reduce_one_group_world4():
    run_test(world_size=4, attn_tp=4)


def test_batch_invariant_all_reduce_two_groups_world4():
    run_test(world_size=4, attn_tp=2)


def test_batch_invariant_all_reduce_one_group_world8():
    run_test(world_size=8, attn_tp=8)


def test_batch_invariant_all_reduce_two_groups_world8():
    run_test(world_size=8, attn_tp=4)
