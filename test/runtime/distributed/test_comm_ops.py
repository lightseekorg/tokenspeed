"""Tests for comm_ops and comm_backend.

Spawns real distributed workers to test all_reduce, all_gather, reduce_scatter,
token_all_gather, token_reduce_scatter, fused ops, and backend registry.

Usage:
    python -m pytest test/runtime/distributed/test_comm_ops.py -v
"""

import socket
from types import SimpleNamespace
from typing import List
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.distributed.comm_ops import all_to_all_single


class TestAutoBackendTopology:
    @pytest.fixture
    def backend(self, monkeypatch):
        from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
        from tokenspeed.runtime.utils.env import global_server_args_dict

        monkeypatch.setitem(
            global_server_args_dict,
            "mapping",
            SimpleNamespace(nprocs_per_node=4),
        )
        backend = AutoBackend.__new__(AutoBackend)
        backend._nccl = Mock()
        backend._rsag = Mock()
        backend._triton_ar = Mock()
        backend._custom_ar = Mock()
        backend._trtllm_ar = Mock()
        return backend

    def test_group_spans_nodes(self, backend):
        assert not backend._group_spans_nodes((0, 1, 2, 3))
        assert not backend._group_spans_nodes((4, 5, 6, 7))
        assert backend._group_spans_nodes((0, 1, 4, 5))

    @pytest.mark.parametrize("method", ["token_all_gather", "token_reduce_scatter"])
    def test_cross_node_token_ops_fall_back_to_nccl(self, backend, method):
        tensor = Mock()
        scattered = [1] * 8
        getattr(backend._nccl, method).return_value = "nccl-result"

        result = getattr(backend, method)(tensor, tuple(range(8)), scattered)

        assert result == "nccl-result"
        getattr(backend._nccl, method).assert_called_once_with(
            tensor, tuple(range(8)), scattered
        )
        getattr(backend._rsag, method).assert_not_called()

    @pytest.mark.parametrize("method", ["token_all_gather", "token_reduce_scatter"])
    def test_node_local_token_ops_use_rsag(self, backend, method):
        tensor = Mock()
        scattered = [1] * 4
        getattr(backend._rsag, method).return_value = "rsag-result"

        result = getattr(backend, method)(tensor, (0, 1, 2, 3), scattered)

        assert result == "rsag-result"
        getattr(backend._rsag, method).assert_called_once_with(
            tensor, (0, 1, 2, 3), scattered
        )
        getattr(backend._nccl, method).assert_not_called()

    def test_cross_node_all_reduce_falls_back_to_nccl(self, backend):
        tensor = Mock()
        backend._nccl.all_reduce.return_value = "nccl-result"

        result = backend.all_reduce(tensor, tuple(range(8)))

        assert result == "nccl-result"
        backend._nccl.all_reduce.assert_called_once_with(
            tensor, tuple(range(8)), op=None
        )
        backend._custom_ar.has_custom_ar.assert_not_called()
        backend._trtllm_ar.has_trtllm_ar.assert_not_called()
        backend._triton_ar.can_run.assert_not_called()

    def test_cross_node_last_dim_all_gather_falls_back_to_nccl(self, backend):
        tensor = Mock()
        tensor.dim.return_value = 2
        backend._nccl.all_gather.return_value = "nccl-result"

        result = backend.all_gather(tensor, tuple(range(8)), dim=-1)

        assert result == "nccl-result"
        backend._nccl.all_gather.assert_called_once_with(tensor, tuple(range(8)), -1)
        backend._rsag.all_gather.assert_not_called()


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
    from tokenspeed.runtime.distributed.comm_ops import all_reduce, all_reduce_two

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

    # Independent production-sized Kimi shared and routed segments. AMD Iris
    # handles this with one kernel; other backends use the two-call fallback.
    first = torch.randint(1, 16, (1, 7168), dtype=torch.bfloat16, device=device)
    second = torch.randint(1, 16, (1, 3584), dtype=torch.bfloat16, device=device)
    expected_first = first.clone()
    expected_second = second.clone()
    dist.all_reduce(expected_first, group=ref_group)
    dist.all_reduce(expected_second, group=ref_group)
    result_first, result_second = all_reduce_two(
        first.clone(),
        second.clone(),
        group,
    )
    torch.testing.assert_close(result_first, expected_first)
    torch.testing.assert_close(result_second, expected_second)


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
    inp = torch.full(
        (tokens_per_rank, hidden_size), rank + 1, dtype=torch.bfloat16, device=device
    )
    result = token_all_gather(inp, group, scattered_num_tokens=scattered)
    expected = torch.cat(
        [torch.full_like(inp, peer + 1) for peer in range(world_size)], dim=0
    )
    torch.testing.assert_close(result, expected)

    # Even reduce_scatter
    total_tokens = tokens_per_rank * world_size
    inp = torch.full(
        (total_tokens, hidden_size), rank + 1, dtype=torch.bfloat16, device=device
    )
    result = token_reduce_scatter(inp, group, scattered_num_tokens=scattered)
    expected = torch.full_like(result, world_size * (world_size + 1) // 2)
    torch.testing.assert_close(result, expected)

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
    rows = torch.arange(total_tokens, dtype=torch.bfloat16, device=device)[:, None]
    full = (rows + rank + 1).expand(-1, hidden_size).contiguous()
    reduced = full.clone()
    dist.all_reduce(reduced, group=ref_group)
    scattered_out = token_reduce_scatter(full, group, scattered_num_tokens=scattered)
    offset = sum(scattered[:rank])
    torch.testing.assert_close(scattered_out, reduced[offset : offset + my_tokens])
    gathered = token_all_gather(scattered_out, group, scattered_num_tokens=scattered)
    torch.testing.assert_close(gathered, reduced)


def _test_token_ops_kimi_dp_idle_rank(rank, world_size, device, group, ref_group):
    """Exercise Kimi's largest prefill chunk with an empty DP rank."""
    if torch.version.hip is None:
        return

    from tokenspeed.runtime.distributed.comm_ops import (
        token_all_gather,
        token_reduce_scatter,
    )

    hidden_size = 7168
    distributions = (
        [8192] + [0] * (world_size - 1),
        [0] * (world_size - 1) + [8192],
    )
    for scattered in distributions:
        owner = scattered.index(8192)
        local_tokens = scattered[rank]
        gather_input = torch.full(
            (local_tokens, hidden_size),
            rank + 1,
            dtype=torch.bfloat16,
            device=device,
        )
        gathered = token_all_gather(
            gather_input,
            group,
            scattered_num_tokens=scattered,
        )
        assert gathered.shape == (sum(scattered), hidden_size)
        torch.testing.assert_close(
            gathered[::1024, ::512],
            torch.full_like(gathered[::1024, ::512], owner + 1),
        )

        # An idle rank still contributes its full symmetric input to the
        # reduction, but owns no output slice and therefore has a null output
        # data pointer. Running both endpoint distributions also proves that
        # the centralized barrier does not depend on rank zero owning payload.
        del gather_input, gathered
        scatter_input = torch.full(
            (sum(scattered), hidden_size),
            rank + 1,
            dtype=torch.bfloat16,
            device=device,
        )
        scattered_out = token_reduce_scatter(
            scatter_input,
            group,
            scattered_num_tokens=scattered,
        )
        assert scattered_out.shape == (local_tokens, hidden_size)
        if local_tokens > 0:
            expected = world_size * (world_size + 1) // 2
            torch.testing.assert_close(
                scattered_out[::1024, ::512],
                torch.full_like(scattered_out[::1024, ::512], expected),
            )
        else:
            assert scattered_out.numel() == 0


def _test_token_ops_cuda_graph(rank, world_size, device, group, ref_group):
    """Cycle graph streams over one AMD symmetric-memory workspace."""
    if torch.version.hip is None:
        return

    from tokenspeed.runtime.distributed.comm_ops import (
        token_all_gather,
        token_reduce_scatter,
    )

    hidden_size = 256
    graph_pool = torch.cuda.graph_pool_handle()
    captures = []
    distributions = (
        [16] * world_size,
        [31] + [3 + peer for peer in range(1, world_size)],
        [
            16 + peer if peer < max(1, world_size // 2) else 0
            for peer in range(world_size)
        ],
    )

    for scattered in distributions:
        total_tokens = sum(scattered)
        local_tokens = scattered[rank]
        gather_input = torch.empty(
            (local_tokens, hidden_size), dtype=torch.bfloat16, device=device
        )
        scatter_input = torch.empty(
            (total_tokens, hidden_size), dtype=torch.bfloat16, device=device
        )
        gather_input.fill_(rank + 1)
        scatter_input.fill_(rank + 1)

        # Model startup compiles and eagerly warms every shape before capture.
        token_all_gather(gather_input, group, scattered_num_tokens=scattered)
        token_reduce_scatter(scatter_input, group, scattered_num_tokens=scattered)
        torch.cuda.synchronize()
        dist.barrier()

        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=graph_pool, stream=stream):
            gathered = token_all_gather(
                gather_input, group, scattered_num_tokens=scattered
            )
            scattered_out = token_reduce_scatter(
                scatter_input, group, scattered_num_tokens=scattered
            )
        torch.cuda.synchronize()
        dist.barrier()
        captures.append(
            (
                graph,
                stream,
                scattered,
                gather_input,
                scatter_input,
                gathered,
                scattered_out,
            )
        )

    # Alternate the graphs so signal reuse, distinct capture streams, ragged
    # offsets, and replay-time input updates are all exercised.
    for iteration in range(20):
        capture = captures[iteration % len(captures)]
        (
            graph,
            _stream,
            scattered,
            gather_input,
            scatter_input,
            gathered,
            scattered_out,
        ) = capture
        total_tokens = sum(scattered)
        local_tokens = scattered[rank]
        gather_input.fill_(rank * 16 + iteration)
        rows = torch.arange(total_tokens, dtype=torch.bfloat16, device=device)[:, None]
        scatter_input.copy_((rows + rank + iteration).expand(-1, hidden_size))

        graph.replay()
        torch.cuda.synchronize()

        expected_gathered = torch.cat(
            [
                torch.full(
                    (peer_tokens, hidden_size),
                    peer * 16 + iteration,
                    dtype=torch.bfloat16,
                    device=device,
                )
                for peer, peer_tokens in enumerate(scattered)
            ],
            dim=0,
        )
        torch.testing.assert_close(gathered, expected_gathered)

        offset = sum(scattered[:rank])
        expected_scattered = (
            rows[offset : offset + local_tokens] * world_size
            + world_size * iteration
            + world_size * (world_size - 1) // 2
        ).expand(-1, hidden_size)
        torch.testing.assert_close(scattered_out, expected_scattered)


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

    def test_prepare_all_reduce_lane_uses_backend_capability(self):
        from tokenspeed.runtime.distributed.comm_ops import prepare_all_reduce_lane

        calls = []

        class Backend:
            def prepare_all_reduce_lane(self, group, hidden_dim):
                calls.append((group, hidden_dim))
                return True

        group = (0, 1)
        assert prepare_all_reduce_lane(group, 10752, backend=Backend())
        assert calls == [(group, 10752)]

    def test_prepare_all_reduce_fusion_hides_kernel_backend(self, monkeypatch):
        from tokenspeed.runtime.distributed import comm_ops

        process_group = type("ProcessGroup", (), {"rank": lambda self: 3})()
        calls = []
        monkeypatch.setattr(
            comm_ops,
            "_get_process_group",
            lambda group: process_group,
        )
        monkeypatch.setattr(
            comm_ops,
            "kernel_prepare_allreduce_fusion",
            lambda **kwargs: calls.append(kwargs) or True,
        )

        assert comm_ops.prepare_all_reduce_fusion((0, 1), 10752, 8)
        assert calls == [
            {
                "rank": 3,
                "group": process_group,
                "max_token_num": 8,
                "hidden_dim": 10752,
            }
        ]


# ---------------------------------------------------------------------------
# Multi-GPU test classes
# ---------------------------------------------------------------------------

WORLD_SIZES = [
    pytest.param(2, id="ws2"),
    pytest.param(4, id="ws4"),
]
TOKEN_WORLD_SIZES = [
    *WORLD_SIZES,
    pytest.param(8, id="ws8"),
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

    @pytest.mark.parametrize("world_size", TOKEN_WORLD_SIZES)
    def test_token_ops(self, world_size):
        _run(world_size, _test_token_ops)

    @pytest.mark.parametrize(
        "world_size",
        [pytest.param(2, id="ws2"), pytest.param(8, id="ws8")],
    )
    def test_token_ops_kimi_dp_idle_rank(self, world_size):
        _run(world_size, _test_token_ops_kimi_dp_idle_rank)

    @pytest.mark.parametrize("world_size", TOKEN_WORLD_SIZES)
    def test_token_ops_cuda_graph(self, world_size):
        _run(world_size, _test_token_ops_cuda_graph)

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_fused_ops(self, world_size):
        _run(world_size, _test_fused_ops)

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    def test_backend_registry(self, world_size):
        _run(world_size, _test_backend_registry)
