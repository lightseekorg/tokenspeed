"""CUDA correctness and graph-replay tests for TRT-LLM all-reduce."""

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="runtime-2gpu")


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _worker_main(rank, world_size, port):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg_manager,
        )

        group = tuple(range(world_size))
        pg_manager.init_process_group(group)
        ref_group = pg_manager.get_process_group("nccl", group)
        _exercise_allreduce(rank, world_size, device, group, ref_group)
    finally:
        dist.destroy_process_group()


def _worker_entry(rank, world_size, port, errors):
    try:
        _worker_main(rank, world_size, port)
    except Exception:
        import traceback

        errors[rank] = traceback.format_exc()


def _exercise_allreduce(rank, world_size, device, group, ref_group):
    from tokenspeed.runtime.distributed.comm_backend.nccl import NcclBackend
    from tokenspeed.runtime.distributed.comm_backend.trtllm_allreduce import (
        TrtllmAllReduceBackend,
    )

    hidden_dim = 512
    token_num = 16
    backend = TrtllmAllReduceBackend(fallback=NcclBackend())
    assert backend.configure_group(
        rank=rank,
        group=group,
        max_token_num=128,
        hidden_dim=hidden_dim,
    )

    try:
        torch.manual_seed(1000 + rank)
        eager_input = torch.randn(
            token_num, hidden_dim, dtype=torch.bfloat16, device=device
        )
        eager_expected = eager_input.clone()
        dist.all_reduce(eager_expected, group=ref_group)
        assert backend.can_run(eager_input, group)
        eager_result = backend.all_reduce(eager_input, group)
        torch.cuda.synchronize(device)
        torch.testing.assert_close(eager_result, eager_expected, atol=0.125, rtol=0.05)

        for fallback_shape in ((129, hidden_dim), (hidden_dim,)):
            fallback_input = torch.randn(
                fallback_shape,
                dtype=torch.bfloat16,
                device=device,
            )
            fallback_expected = fallback_input.clone()
            dist.all_reduce(fallback_expected, group=ref_group)
            assert not backend.can_run(fallback_input, group)
            fallback_result = backend.all_reduce(fallback_input, group)
            torch.testing.assert_close(fallback_result, fallback_expected)

        static_input = eager_input.clone()
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                backend.all_reduce(eager_input, group)
        torch.cuda.current_stream(device).wait_stream(capture_stream)
        torch.cuda.synchronize(device)
        dist.barrier(group=ref_group)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            graph_result = backend.all_reduce(static_input, group)
        torch.cuda.synchronize(device)
        dist.barrier(group=ref_group)

        for iteration in range(10):
            torch.manual_seed(2000 + iteration * world_size + rank)
            replay_input = torch.randn_like(static_input)
            replay_expected = replay_input.clone()
            dist.all_reduce(replay_expected, group=ref_group)
            static_input.copy_(replay_input)
            dist.barrier(group=ref_group)
            graph.replay()
            torch.cuda.synchronize(device)
            torch.testing.assert_close(
                graph_result, replay_expected, atol=0.125, rtol=0.05
            )
            dist.barrier(group=ref_group)
    finally:
        backend.close_group(group)


def _run(world_size):
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    port = _get_open_port()
    errors = mp.Manager().dict()
    mp.spawn(
        _worker_entry,
        args=(world_size, port, errors),
        nprocs=world_size,
        join=True,
    )

    if errors:
        raise RuntimeError(
            "\n".join(f"Rank {rank}: {error}" for rank, error in errors.items())
        )


@pytest.mark.parametrize(
    "world_size",
    (
        pytest.param(2, id="tp2-ci"),
        pytest.param(8, id="tp8-manual"),
    ),
)
def test_correctness_and_cuda_graph_replay(world_size):
    _run(world_size)
