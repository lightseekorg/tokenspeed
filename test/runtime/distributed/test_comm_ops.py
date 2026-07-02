"""Tests for comm_ops and comm_backend.

Spawns real distributed workers to test all_reduce, all_gather, reduce_scatter,
token_all_gather, token_reduce_scatter, fused ops, and backend registry.

Usage:
    python -m pytest test/runtime/distributed/test_comm_ops.py -v
"""

import socket
from typing import List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.distributed.comm_ops import all_to_all_single


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def worker_fn(rank, world_size, port, test_fn, error_dict):
    try:
        _worker_main(rank, world_size, port, test_fn)
    except Exception:
        import traceback

        error_dict[rank] = traceback.format_exc()


def _worker_main(rank, world_size, port, test_fn):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg_manager,
    )

    group = tuple(range(world_size))
    pg_manager.init_process_group(group)
    ref_group = pg_manager.get_process_group("nccl", group)

    _setup_runtime_globals(rank, world_size)

    test_fn(
        rank=rank,
        world_size=world_size,
        device=device,
        group=group,
        ref_group=ref_group,
    )

    dist.destroy_process_group()


def _setup_runtime_globals(rank, world_size):
    """Match the runtime's setup of global_server_args_dict.

    AutoBackend's 2-D last-dim all_gather and all token-aware ops route through
    TritonRSAGBackend, which sizes its persistent buffers from these globals.
    """
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.utils.env import global_server_args_dict

    mapping = Mapping(rank=rank, world_size=world_size, attn_tp_size=world_size)
    global_server_args_dict["mapping"] = mapping
    global_server_args_dict["chunked_prefill_size"] = 8192
    global_server_args_dict["max_prefill_tokens"] = 8192
    global_server_args_dict["max_model_len"] = 4096
    global_server_args_dict["force_deterministic_rsag"] = True


def _run(world_size, test_fn):
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")

    port = get_open_port()
    error_dict = mp.Manager().dict()

    mp.spawn(
        worker_fn,
        args=(world_size, port, test_fn, error_dict),
        nprocs=world_size,
        join=True,
    )

    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


# ---------------------------------------------------------------------------
# Test functions (run inside each worker)
# ---------------------------------------------------------------------------

TEST_SIZES = [512, 4096, 32768]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _test_all_reduce(rank, world_size, device, group, ref_group):
    from tokenspeed.runtime.distributed.comm_ops import all_reduce

    for sz in TEST_SIZES:
        for dtype in DTYPES:
            inp = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
            expected = inp.clone()
            dist.all_reduce(expected, group=ref_group)
            result = all_reduce(inp.clone(), group)
            torch.testing.assert_close(result, expected)

    # 2D
    for dtype in DTYPES:
        inp = torch.randint(1, 16, (8, 512), dtype=dtype, device=device)
        expected = inp.clone()
        dist.all_reduce(expected, group=ref_group)
        result = all_reduce(inp.clone(), group)
        torch.testing.assert_close(result, expected)


def _test_all_gather(rank, world_size, device, group, ref_group):
    from tokenspeed.runtime.distributed.comm_ops import all_gather

    for sz in TEST_SIZES:
        for dtype in DTYPES:
            inp = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
            output_list = [torch.empty_like(inp) for _ in range(world_size)]
            dist.all_gather(output_list, inp, group=ref_group)
            expected = torch.cat(output_list, dim=0)
            result = all_gather(inp, group, dim=0)
            torch.testing.assert_close(result, expected)

    # last dim
    for dtype in DTYPES:
        inp = torch.randint(1, 16, (4, 128), dtype=dtype, device=device)
        output_list = [torch.empty_like(inp) for _ in range(world_size)]
        dist.all_gather(output_list, inp, group=ref_group)
        expected = torch.cat(output_list, dim=-1)
        result = all_gather(inp, group, dim=-1)
        torch.testing.assert_close(result, expected)


def _test_all_gather_into_tensor(rank, world_size, device, group, ref_group):
    from tokenspeed.runtime.distributed.comm_ops import all_gather_into_tensor

    for sz in TEST_SIZES:
        for dtype in DTYPES:
            inp = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
            output = torch.empty(sz * world_size, dtype=dtype, device=device)
            expected = torch.empty_like(output)
            dist.all_gather_into_tensor(expected, inp, group=ref_group)
            all_gather_into_tensor(output, inp, group)
            torch.testing.assert_close(output, expected)

    # 2D
    inp = torch.randint(1, 16, (4, 128), dtype=torch.float32, device=device)
    output = torch.empty(4 * world_size, 128, dtype=torch.float32, device=device)
    expected = torch.empty_like(output)
    dist.all_gather_into_tensor(expected, inp, group=ref_group)
    all_gather_into_tensor(output, inp, group)
    torch.testing.assert_close(output, expected)


def _test_all_to_all_single(rank, world_size, device, group, ref_group):
    for sz in TEST_SIZES:
        for dtype in DTYPES:
            total = sz * world_size
            inp = torch.randint(1, 16, (total,), dtype=dtype, device=device)
            expected = torch.empty_like(inp)
            dist.all_to_all_single(expected, inp, group=ref_group)
            output = torch.empty_like(inp)
            all_to_all_single(output, inp, group)
            torch.testing.assert_close(output, expected)

    for dtype in DTYPES:
        rows_per_rank = 4
        total_rows = rows_per_rank * world_size
        inp = torch.randint(1, 16, (total_rows, 128), dtype=dtype, device=device)
        expected = torch.empty_like(inp)
        dist.all_to_all_single(expected, inp, group=ref_group)
        output = torch.empty_like(inp)
        all_to_all_single(output, inp, group)
        torch.testing.assert_close(output, expected)


def _test_reduce_scatter(rank, world_size, device, group, ref_group):
    from tokenspeed.runtime.distributed.comm_ops import reduce_scatter

    for sz in TEST_SIZES:
        for dtype in DTYPES:
            total_sz = sz * world_size
            inp = torch.randint(1, 16, (total_sz,), dtype=dtype, device=device)
            expected = torch.empty(sz, dtype=dtype, device=device)
            dist.reduce_scatter_tensor(expected, inp, group=ref_group)
            result = reduce_scatter(inp.clone(), group)
            torch.testing.assert_close(result, expected)

    # 2D
    for dtype in DTYPES:
        total_rows = 16 * world_size
        inp = torch.randint(1, 16, (total_rows, 128), dtype=dtype, device=device)
        expected = torch.empty(16, 128, dtype=dtype, device=device)
        dist.reduce_scatter_tensor(expected, inp, group=ref_group)
        result = reduce_scatter(inp.clone(), group)
        torch.testing.assert_close(result, expected)


def _test_token_ops(rank, world_size, device, group, ref_group):
    from tokenspeed.runtime.distributed.comm_ops import (
        token_all_gather,
        token_reduce_scatter,
    )

    hidden_size = 256

    # Even all_gather
    tokens_per_rank = 64
    scattered = [tokens_per_rank] * world_size
    inp = torch.randn(tokens_per_rank, hidden_size, dtype=torch.bfloat16, device=device)
    result = token_all_gather(inp, group, scattered_num_tokens=scattered)
    assert result.shape[0] == tokens_per_rank * world_size

    # Even reduce_scatter
    total_tokens = tokens_per_rank * world_size
    inp = torch.randn(total_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    result = token_reduce_scatter(inp, group, scattered_num_tokens=scattered)
    assert result.shape[0] == tokens_per_rank

    # Roundtrip: all_gather(reduce_scatter(x) / world_size) == x
    tokens_per_rank = 32
    total_tokens = tokens_per_rank * world_size
    scattered = [tokens_per_rank] * world_size
    torch.manual_seed(42)
    full = torch.randn(total_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    scattered_out = token_reduce_scatter(full, group, scattered_num_tokens=scattered)
    scattered_out = scattered_out / world_size
    gathered = token_all_gather(scattered_out, group, scattered_num_tokens=scattered)
    torch.testing.assert_close(gathered, full, atol=0.02, rtol=0.02)

    # Uneven distribution
    scattered = [1] * world_size
    scattered[0] = 100
    total_tokens = sum(scattered)
    my_tokens = scattered[rank]
    full = torch.randn(total_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    scattered_out = token_reduce_scatter(full, group, scattered_num_tokens=scattered)
    assert scattered_out.shape[0] == my_tokens
    gathered = token_all_gather(scattered_out, group, scattered_num_tokens=scattered)
    assert gathered.shape[0] == total_tokens


def _test_fused_ops(rank, world_size, device, group, ref_group):
    from tokenspeed.runtime.distributed.comm_ops import (
        FusionOp,
        FusionParams,
        fused_all_gather,
        fused_all_reduce,
        fused_reduce_scatter,
    )

    # fused_all_reduce with NONE
    inp = torch.randint(1, 16, (1024,), dtype=torch.float32, device=device)
    expected = inp.clone()
    dist.all_reduce(expected, group=ref_group)
    result = fused_all_reduce(inp.clone(), rank, group)
    torch.testing.assert_close(result, expected)
    result2 = fused_all_reduce(
        inp.clone(), rank, group, fusion_params=FusionParams(fusion_op=FusionOp.NONE)
    )
    torch.testing.assert_close(result2, expected)

    # fused_reduce_scatter with NONE
    total_sz = 512 * world_size
    inp = torch.randint(1, 16, (total_sz,), dtype=torch.float32, device=device)
    expected = torch.empty(512, dtype=torch.float32, device=device)
    dist.reduce_scatter_tensor(expected, inp, group=ref_group)
    result = fused_reduce_scatter(inp.clone(), rank, group)
    torch.testing.assert_close(result, expected)

    # fused_all_gather with NONE
    inp = torch.randint(1, 16, (256,), dtype=torch.float32, device=device)
    output_list = [torch.empty_like(inp) for _ in range(world_size)]
    dist.all_gather(output_list, inp, group=ref_group)
    expected = torch.cat(output_list, dim=0)
    result = fused_all_gather(inp, rank, group, dim=0)
    torch.testing.assert_close(result, expected)


def _test_backend_registry(rank, world_size, device, group, ref_group):
    from tokenspeed.runtime.distributed.comm_backend import get_global_backend

    backend = get_global_backend()
    assert backend is not None

    # Singleton
    b2 = get_global_backend()
    assert backend is b2

    # Auto-create resources on first use
    inp = torch.ones(4, device=device)
    result = backend.all_reduce(inp, group)
    assert result.shape == inp.shape


def _test_trtllm_allreduce_graph_replay(rank, world_size, device, group, ref_group):
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


# ---------------------------------------------------------------------------
# FusionParams (no GPU needed)
# ---------------------------------------------------------------------------


class TestFusionParams:
    def test_default_params(self):
        from tokenspeed.runtime.distributed.comm_ops import FusionOp, FusionParams

        params = FusionParams()
        assert params.fusion_op == FusionOp.NONE
        assert params.residual is None
        assert params.norm_weight is None

    def test_residual_rmsnorm_params(self):
        from tokenspeed.runtime.distributed.comm_ops import FusionOp, FusionParams

        weight = torch.ones(128)
        residual = torch.zeros(4, 128)
        params = FusionParams(
            fusion_op=FusionOp.RESIDUAL_RMS_NORM,
            norm_weight=weight,
            residual=residual,
            eps=1e-5,
        )
        assert params.fusion_op == FusionOp.RESIDUAL_RMS_NORM
        assert params.norm_weight is weight


# ---------------------------------------------------------------------------
# Multi-GPU test classes
# ---------------------------------------------------------------------------

WORLD_SIZES = [
    pytest.param(2, id="ws2"),
    pytest.param(4, id="ws4"),
]


class TestCommOps:

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_all_reduce(self, world_size):
        _run(world_size, _test_all_reduce)

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_all_gather(self, world_size):
        _run(world_size, _test_all_gather)

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_all_gather_into_tensor(self, world_size):
        _run(world_size, _test_all_gather_into_tensor)

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_all_to_all_single(self, world_size):
        _run(world_size, _test_all_to_all_single)

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_reduce_scatter(self, world_size):
        _run(world_size, _test_reduce_scatter)

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_token_ops(self, world_size):
        _run(world_size, _test_token_ops)

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_fused_ops(self, world_size):
        _run(world_size, _test_fused_ops)

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_backend_registry(self, world_size):
        _run(world_size, _test_backend_registry)


class TestTrtllmAllReduceBackend:

    def test_tp8_correctness_and_cuda_graph_replay(self):
        _run(8, _test_trtllm_allreduce_graph_replay)
