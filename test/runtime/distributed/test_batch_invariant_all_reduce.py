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

"""The batch-invariant all-reduce on real groups: the route, the kernel and
the startup self-check, end to end on several GPUs."""

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

HIDDEN = 2048


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


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
    from tokenspeed.runtime.distributed.comm_backend.self_check import (
        verify_multimem_all_reduce,
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
        probe = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)
        assert backend.route(Collective.ALL_REDUCE, probe, group) is Route.MULTIMEM
        assert (
            backend.route(Collective.TOKEN_REDUCE_SCATTER, probe, group)
            is Route.ORDERED_FOLD
        )

        # The startup self-check on the deployment's own groups: with
        # attn_tp < world_size there are several attention TP groups, so the
        # one-function leg compares the switch's order across them and pins
        # the kind to the fold where the GPU sets reduce differently (measured
        # for groups of four on 8xH20; groups of two cannot differ).
        kinds = (
            ("attention TP", mapping.attn.tp_group),
            ("dense TP", mapping.dense.tp_group),
            ("MoE TP-EP", mapping.moe.tp_ep_group),
        )
        outcome = verify_multimem_all_reduce(
            backend,
            groups=kinds,
            world_group=mapping.world_group,
            rank=rank,
            hidden_size=HIDDEN,
            device=device,
            repetitions=8,
        )
        routes = dict(outcome)
        assert set(routes) == {kind for kind, _ in kinds}, outcome
        # One group of everyone has nothing to disagree with; two groups of
        # two cannot order two addends differently. Larger groups of a kind
        # may land either way, but the decision must be the route served.
        assert routes["MoE TP-EP"] is Route.MULTIMEM, outcome
        if attn_tp == world_size or attn_tp == 2:
            assert routes["attention TP"] is Route.MULTIMEM, outcome
        # The dense TP group is the attention TP group here: decided once.
        assert routes["dense TP"] is routes["attention TP"], outcome
        for kind, kind_group in kinds:
            assert (
                backend.route(Collective.ALL_REDUCE, probe, kind_group) is routes[kind]
            )

        # The public all-reduce every layer calls: in place, equal on every
        # rank, the sum up to the association order, the same bits for a row
        # whatever the batch around it -- and, the contract the self-check
        # exists for, the same bits on every group of the kind (the payload
        # is seeded by rank in group, so the groups reduce identical inputs).
        gen = torch.Generator(device=device).manual_seed(100 + group.index(rank))
        scale = 2.0 ** torch.randint(-6, 7, (rows, 1), generator=gen, device=device)
        payload = (torch.randn(rows, HIDDEN, generator=gen, device=device) * scale).to(
            torch.bfloat16
        )
        full = payload.clone()
        assert all_reduce(full, group) is full
        world_pg = pg_manager.get_process_group("nccl", mapping.world_group)
        gathered = [torch.empty_like(full) for _ in mapping.world_group]
        dist.all_gather(gathered, full, group=world_pg)
        for peer in mapping.world_group:
            assert torch.equal(gathered[peer], full), f"rank {peer} disagrees"
        parts = [torch.empty_like(payload) for _ in mapping.world_group]
        dist.all_gather(parts, payload, group=world_pg)
        folded = torch.zeros(rows, HIDDEN, dtype=torch.float32, device=device)
        for peer in group:
            folded += parts[peer].float()
        torch.testing.assert_close(
            full.float(), folded.to(torch.bfloat16).float(), rtol=2**-7, atol=1e-2
        )
        for row_count in (1, 5, rows // 2):
            part = payload[:row_count].clone()
            all_reduce(part, group)
            assert torch.equal(part, full[:row_count]), row_count
    finally:
        dist.barrier()
        dist.destroy_process_group()


def run_test(world_size: int, attn_tp: int) -> None:
    if not torch.cuda.is_available() or not current_platform().is_nvidia:
        pytest.skip("the in-switch (NVLS) all-reduce is NVIDIA-only")
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
