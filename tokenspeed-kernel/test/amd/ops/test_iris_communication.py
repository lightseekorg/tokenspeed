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


import math
import socket
import time
import traceback
from typing import List, Tuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_amd,
    reason="Iris communication tests require AMD ROCm",
)


@pytest.fixture(autouse=True)
def _require_iris():
    pytest.importorskip(
        "tokenspeed_kernel.ops.communication._iris", exc_type=ImportError
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _skip_if_unsupported(world_size: int, reason_prefix: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip(f"CUDA/ROCm is required for {reason_prefix}")
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    if not current_platform().is_amd:
        pytest.skip(f"{reason_prefix} only targets AMD ROCm")
    try:
        import tokenspeed_kernel.ops.communication._iris  # noqa: F401
    except ImportError:
        pytest.skip("iris is not installed")


def _spawn_and_collect(worker_fn, args, world_size: int) -> None:
    error_dict = mp.Manager().dict()
    mp.spawn(
        worker_fn,
        args=args + (error_dict,),
        nprocs=world_size,
        join=True,
    )

    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


@pytest.mark.parametrize(
    ("world_size", "dtype", "min_bytes"),
    [
        (4, torch.bfloat16, 160 << 10),
        (4, torch.float32, 160 << 10),
        (8, torch.bfloat16, 96 << 10),
        (8, torch.float32, 96 << 10),
    ],
)
def test_producer_direct_two_stage_threshold(world_size, dtype, min_bytes):
    try:
        from tokenspeed_kernel.ops.communication.iris import (
            _use_two_stage_producer_direct,
        )
    except ImportError:
        pytest.skip("iris is not installed")

    alignment = world_size * (8 // dtype.itemsize)
    min_numel = min_bytes // dtype.itemsize
    assert _use_two_stage_producer_direct(world_size, min_numel, dtype)
    assert not _use_two_stage_producer_direct(world_size, min_numel - alignment, dtype)
    assert not _use_two_stage_producer_direct(world_size, min_numel + 1, dtype)
    assert not _use_two_stage_producer_direct(2, min_numel, dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_plain_two_stage_admits_partitionable_shapes(dtype):
    try:
        from tokenspeed_kernel.ops.communication.iris import _use_two_stage_plain
    except ImportError:
        pytest.skip("iris is not installed")

    elements_per_word = 8 // dtype.itemsize
    # The plain path carries no minimum size -- two-stage measured faster than
    # one-shot at every shape down to 14 KB -- so the predicate is purely the
    # kernel's partitioning requirement.
    aligned = 8 * elements_per_word
    assert _use_two_stage_plain(8, aligned, dtype)
    assert _use_two_stage_plain(8, aligned * 4096, dtype)
    # every K3 decode width partitions evenly across 8 ranks
    for tokens in (1, 8, 16, 32, 64):
        assert _use_two_stage_plain(8, tokens * 7168, dtype)
    # a payload that does not split into whole words per rank stays on one-shot
    assert not _use_two_stage_plain(8, aligned + 1, dtype)
    # world sizes without a tuned partitioning fall back
    assert not _use_two_stage_plain(2, aligned, dtype)


def test_plain_two_stage_is_independent_of_producer_thresholds(monkeypatch):
    try:
        from tokenspeed_kernel.ops.communication import iris as iris_ops
    except ImportError:
        pytest.skip("iris is not installed")

    monkeypatch.setattr(iris_ops, "_PRODUCER_TWO_SHOT_MIN_BYTES", {})
    assert iris_ops._use_two_stage_plain(8, 8 * 7168, torch.bfloat16)
    assert not iris_ops._use_two_stage_producer_direct(8, 8 * 7168, torch.bfloat16)


# ---------------------------------------------------------------------------
# Suite 1: iris_all_reduce
# ---------------------------------------------------------------------------


def _ar_shape_cases() -> List[Tuple[int, ...]]:
    """Shapes covering small, vector, and 2-D cases."""
    return [
        (8,),
        (16, 64),
        (4, 7, 32),
    ]


def _ar_graph_shape_cases() -> List[Tuple[int, ...]]:
    return [(16, 4096), (64, 4096)]


def _ar_output_shape_cases() -> List[Tuple[Tuple[int, ...], ...]]:
    """Producer-direct collections spanning one, two, and three outputs."""
    return [
        ((16, 7168),),
        ((37, 7168),),
        ((8191, 7168),),
        ((1, 7168), (1, 3584)),
        ((2, 7168), (2, 3584)),
        ((4, 7168), (4, 3584)),
        ((8, 7168), (8, 3584)),
        ((16, 7168), (16, 3584)),
        ((513, 7168), (513, 3584)),
        ((3, 20), (2, 12)),
        ((3, 5), (1, 1)),
        ((2, 16),),
        ((3, 20), (2, 12), (4, 4)),
    ]


def _ar_worker_fn(rank, world_size, port, allocator_type, error_dict):
    try:
        _ar_worker_main(rank, world_size, port, allocator_type)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _ar_worker_main(
    rank: int, world_size: int, port: int, allocator_type: str | None
) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # Iris's example uses gloo because heap-base exchange is host-side; nccl
    # also works, but gloo avoids contending with the iris-managed device
    # memory and matches the upstream example.
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        # Importing inside the worker avoids pulling iris into the parent
        # process (which has no distributed context).
        from tokenspeed_kernel.ops.communication.iris import (
            configure_iris,
            create_iris_state,
        )

        configure_iris(heap_size=None, allocator_type=allocator_type)
        output_shape_cases = _ar_output_shape_cases()
        staged_max_numel = max(
            max(int(torch.tensor(shape).prod()) for shape in _ar_shape_cases()),
            max(int(torch.tensor(shape).prod()) for shape in _ar_graph_shape_cases()),
        )
        producer_direct_max_numel = max(
            sum(int(torch.tensor(shape).prod()) for shape in shapes)
            for shapes in output_shape_cases
        )
        attnres_max_rows = 16 if world_size == 8 else 0
        attnres_max_numel = attnres_max_rows * 7168
        staged_max_numel = max(staged_max_numel, attnres_max_numel)
        state = create_iris_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            staged_max_numel=staged_max_numel,
            producer_direct_max_numel=producer_direct_max_numel,
            dtype=torch.bfloat16,
            heap_size=None,
            device=device,
        )
        if world_size == 4 and current_platform().is_cdna4:
            assert state.tuned_staged.slots.data_ptr() != state.staged.slots.data_ptr()
            assert (
                state.tuned_staged.ready_flags.data_ptr()
                != state.staged.ready_flags.data_ptr()
            )
        k3 = None
        if attnres_max_numel:
            from tokenspeed_kernel.ops.communication._iris.kimi_k3 import (
                prepare_k3_communication,
            )

            k3 = prepare_k3_communication(
                group=dist.group.WORLD,
                rank_in_group=rank,
                staged_max_numel=staged_max_numel,
                producer_direct_max_numel=producer_direct_max_numel,
                attnres_max_numel=attnres_max_numel,
                attnres_max_rows=attnres_max_rows,
                enable_lamport=False,
                sharded_max_rows=0,
                dtype=torch.bfloat16,
                device=device,
                all_reduce=state,
            )
        for shape in _ar_shape_cases():
            _check_all_reduce(state, rank, world_size, shape, device)
        if state.staged_two_shot is not None:
            for safe in (False, True):
                _check_all_reduce_graph_replay(
                    state,
                    rank,
                    world_size,
                    (1024,),
                    device,
                    storage_offset=rank % 2,
                    safe=safe,
                )
        if world_size == 4:
            for shape in _ar_graph_shape_cases():
                verify_tuning = current_platform().is_cdna4
                if verify_tuning:
                    torch.cuda.synchronize()
                    dist.barrier()
                    tuned_before = state.tuned_staged.ready_flags.clone()
                    twoshot_before = state.staged_two_shot.ready_flags.clone()
                    torch.cuda.synchronize()
                    dist.barrier()
                _check_all_reduce_graph_replay(
                    state,
                    rank,
                    world_size,
                    shape,
                    device,
                    storage_offset=0,
                    safe=False,
                )
                if verify_tuning:
                    torch.cuda.synchronize()
                    dist.barrier()
                    tuned_changed = not torch.equal(
                        state.tuned_staged.ready_flags, tuned_before
                    )
                    twoshot_changed = not torch.equal(
                        state.staged_two_shot.ready_flags, twoshot_before
                    )
                    assert tuned_changed == (math.prod(shape) == 65536)
                    assert twoshot_changed == (math.prod(shape) != 65536)
                    dist.barrier()
            if current_platform().is_cdna4:
                _check_mixed_geometry_graph_replay(
                    state,
                    rank,
                    world_size,
                    device,
                )
        for shapes in output_shape_cases:
            _check_all_reduce_symmetric_outputs(
                state,
                rank,
                world_size,
                shapes,
                device,
            )

        # These dtype checks only use the small shape. Their states share the
        # fixed heap, so do not allocate another full BF16 prefill capacity.
        other_dtype_max_numel = 16 * (7168 + 3584)
        fp16_state = create_iris_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            staged_max_numel=0,
            producer_direct_max_numel=other_dtype_max_numel,
            dtype=torch.float16,
            heap_size=None,
            device=device,
        )
        _check_all_reduce_symmetric_outputs(
            fp16_state,
            rank,
            world_size,
            ((3, 5), (1, 1)),
            device,
        )
        if world_size > 2:
            _check_all_reduce_symmetric_outputs(
                fp16_state,
                rank,
                world_size,
                ((16, 7168), (16, 3584)),
                device,
            )

        fp32_state = create_iris_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            staged_max_numel=0,
            producer_direct_max_numel=other_dtype_max_numel,
            dtype=torch.float32,
            heap_size=None,
            device=device,
        )
        _check_all_reduce_symmetric_outputs(
            fp32_state,
            rank,
            world_size,
            ((3, 5), (1, 1)),
            device,
        )
        if world_size > 2:
            _check_all_reduce_symmetric_outputs(
                fp32_state,
                rank,
                world_size,
                ((16, 7168), (16, 3584)),
                device,
            )
        if world_size == 8:
            _check_all_reduce_residual_attnres(k3, rank, device)
    finally:
        dist.destroy_process_group()


def _check_all_reduce(state, rank: int, world_size: int, shape, device) -> None:
    from tokenspeed_kernel.ops.communication.iris import iris_all_reduce

    # Each rank contributes a tensor filled with ``rank + 1``; the reduction
    # is therefore ``sum(1..world_size) = world_size*(world_size+1)/2``.
    local = torch.full(shape, rank + 1, dtype=torch.bfloat16, device=device)

    result = iris_all_reduce(state, local)

    expected_value = world_size * (world_size + 1) // 2
    expected = torch.full(shape, expected_value, dtype=torch.bfloat16, device=device)

    assert (
        result.shape == expected.shape
    ), f"shape mismatch: {result.shape} vs {expected.shape}"
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


def _check_all_reduce_graph_replay(
    state,
    rank: int,
    world_size: int,
    shape,
    device,
    storage_offset: int,
    safe: bool,
) -> None:
    from tokenspeed_kernel.ops.communication.iris import iris_all_reduce

    # Odd ranks can start one BF16 element into storage; retain guard elements.
    numel = math.prod(shape)
    storage = torch.full(
        (numel + storage_offset + 1,), -1, dtype=torch.bfloat16, device=device
    )
    local = storage[storage_offset : storage_offset + numel].view(shape)
    assert local.is_contiguous() and local.data_ptr() % 8 == 2 * storage_offset
    local.fill_(rank + 1)
    result = iris_all_reduce(
        state, local, op=dist.ReduceOp.SUM, safe=safe, async_op=False
    )
    assert (result.data_ptr() == local.data_ptr()) == (not safe)
    expected_value = world_size * (world_size + 1) // 2
    torch.testing.assert_close(local, result, atol=0, rtol=0)
    torch.testing.assert_close(
        result,
        torch.full_like(result, expected_value),
        atol=0,
        rtol=0,
    )

    local.fill_(rank + 1)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_result = iris_all_reduce(
            state, local, op=dist.ReduceOp.SUM, safe=safe, async_op=False
        )
    assert (graph_result.data_ptr() == local.data_ptr()) == (not safe)

    for replay_index in range(1, 5):
        scale = replay_index + 1
        local.fill_(scale * (rank + 1))
        torch.cuda.synchronize()
        dist.barrier()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(local, graph_result, atol=0, rtol=0)
        torch.testing.assert_close(
            graph_result,
            torch.full_like(graph_result, scale * expected_value),
            atol=0,
            rtol=0,
        )
    assert storage[-1].item() == -1
    if storage_offset:
        assert storage[0].item() == -1


def _check_mixed_geometry_graph_replay(
    state,
    rank: int,
    world_size: int,
    device,
) -> None:
    from tokenspeed_kernel.ops.communication.iris import iris_all_reduce

    shapes = ((16, 4096), (65535,))
    inputs = [
        torch.empty(shape, dtype=torch.bfloat16, device=device) for shape in shapes
    ]
    graphs = []
    for local in inputs:
        local.fill_(rank + 1)
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            iris_all_reduce(state, local, safe=False)
        graphs.append(graph)

    expected_rank_sum = world_size * (world_size + 1) // 2
    snapshots = [[torch.empty_like(local) for _ in range(8)] for local in inputs]
    for replay_index in range(8):
        scale = replay_index + 1
        for local, graph, outputs in zip(inputs, graphs, snapshots, strict=True):
            local.fill_(scale * (rank + 1))
            if rank == 0:
                torch.cuda._sleep(1_000_000)
            graph.replay()
            outputs[replay_index].copy_(local)

    torch.cuda.synchronize()
    for outputs in snapshots:
        for replay_index, output in enumerate(outputs):
            expected = (replay_index + 1) * expected_rank_sum
            torch.testing.assert_close(
                output,
                torch.full_like(output, expected),
                atol=0,
                rtol=0,
            )


def _check_all_reduce_symmetric_outputs(
    state,
    rank: int,
    world_size: int,
    shapes,
    device,
) -> None:
    if not current_platform().is_cdna4:
        return

    from tokenspeed_kernel.ops.communication.iris import (
        iris_acquire_outputs,
        iris_all_reduce_symmetric,
    )

    outputs = iris_acquire_outputs(state, shapes)
    assert state.owns_outputs(outputs)
    assert not state.owns_outputs(tuple(torch.empty_like(output) for output in outputs))
    for index, output in enumerate(outputs, start=1):
        output.fill_(index * (rank + 1))
    results = iris_all_reduce_symmetric(state, outputs)
    result_bytes = sum(output.numel() for output in outputs) * state.dtype.itemsize
    assert all(result.untyped_storage().nbytes() == result_bytes for result in results)
    expected_value = world_size * (world_size + 1) // 2
    for index, (output, result) in enumerate(zip(outputs, results), start=1):
        torch.testing.assert_close(
            result,
            torch.full_like(output, index * expected_value),
            atol=0,
            rtol=0,
        )

    snapshots = []
    for scale in range(1, 5):
        for index, output in enumerate(outputs, start=1):
            output.fill_(scale * index * (rank + 1))
        results = iris_all_reduce_symmetric(state, outputs)
        snapshots.append(results)
    torch.cuda.synchronize()
    for scale, results in enumerate(snapshots, start=1):
        for index, (output, result) in enumerate(zip(outputs, results), start=1):
            torch.testing.assert_close(
                result,
                torch.full_like(output, scale * index * expected_value),
                atol=0,
                rtol=0,
            )

    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index, output in enumerate(outputs, start=1):
            output.fill_(index * (rank + 1))
        graph_results = iris_all_reduce_symmetric(state, outputs)
    dist.barrier()
    eager_results = []
    for scale in range(2, 6):
        for index, output in enumerate(outputs, start=1):
            output.fill_(scale * index * (rank + 1))
        eager_results.append(iris_all_reduce_symmetric(state, outputs))
        graph.replay()
    torch.cuda.synchronize()
    for index, (output, result) in enumerate(zip(outputs, graph_results), start=1):
        torch.testing.assert_close(
            result,
            torch.full_like(output, index * expected_value),
            atol=0,
            rtol=0,
        )
    for scale, results in enumerate(eager_results, start=2):
        for index, result in enumerate(results, start=1):
            torch.testing.assert_close(
                result,
                torch.full_like(result, scale * index * expected_value),
                atol=0,
                rtol=0,
            )


def _check_all_reduce_residual_attnres(k3_state, rank: int, device) -> None:
    from tokenspeed_kernel.ops.activation.triton import (
        attnres_combine,
        attnres_partial,
    )
    from tokenspeed_kernel.ops.communication.iris import (
        iris_all_reduce,
    )

    state = k3_state.all_reduce
    for num_tokens in (1, 2, 4, 8, 16):
        torch.manual_seed(101 + num_tokens)
        hidden = 7168
        blocks = (torch.randn(4, num_tokens, hidden, device=device) * 0.1).to(
            torch.bfloat16
        )
        score_weight = (torch.randn(hidden, device=device) * 0.02).to(torch.bfloat16)
        output_weight = (1.0 + torch.randn(hidden, device=device) * 0.02).to(
            torch.bfloat16
        )
        residual = (torch.randn(num_tokens, hidden, device=device) * 0.1).to(
            torch.bfloat16
        )
        local = (
            torch.randn(num_tokens, hidden, device=device) * 0.01 + (rank + 1) * 0.002
        ).to(torch.bfloat16)
        scratch = (
            torch.empty(num_tokens, device=device, dtype=torch.float32),
            torch.empty(num_tokens, device=device, dtype=torch.float32),
            torch.empty(num_tokens, hidden, device=device, dtype=torch.float32),
        )
        attnres_partial(blocks, score_weight, 1e-6, scratch)

        reduced = iris_all_reduce(state, local.clone(), safe=False)
        expected_residual = residual + reduced
        expected_hidden = attnres_combine(
            expected_residual,
            score_weight,
            output_weight,
            1e-6,
            scratch,
            torch.empty_like(residual),
        )
        assert k3_state.attnres_supported(
            local,
            residual,
            score_weight,
            output_weight,
            scratch,
        )
        for _ in range(4):
            actual_hidden, actual_residual = k3_state.attnres_combine(
                local,
                residual,
                score_weight,
                output_weight,
                scratch,
                eps=1e-6,
            )
            torch.testing.assert_close(
                actual_residual, expected_residual, atol=0, rtol=0
            )
            torch.testing.assert_close(
                actual_hidden, expected_hidden, atol=2e-2, rtol=2e-2
            )

        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with torch.cuda.graph(graph):
            graph_hidden, graph_residual = k3_state.attnres_combine(
                local,
                residual,
                score_weight,
                output_weight,
                scratch,
                eps=1e-6,
            )
        dist.barrier()
        expected_rank_sum = sum(range(1, 9))
        for replay, scale in enumerate((2, -3, 5)):
            local.fill_(scale * (rank + 1) / 128.0)
            expected_reduced = torch.full_like(
                local,
                scale * expected_rank_sum / 128.0,
            )
            expected_residual = (residual.float() + expected_reduced.float()).to(
                torch.bfloat16
            )
            expected_hidden = attnres_combine(
                expected_residual,
                score_weight,
                output_weight,
                1e-6,
                scratch,
                torch.empty_like(residual),
            )
            dist.barrier()
            if rank == replay:
                time.sleep(0.005)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                graph_residual,
                expected_residual,
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                graph_hidden,
                expected_hidden,
                atol=2e-2,
                rtol=2e-2,
            )


def _run_ar_test(world_size: int, allocator_type: str | None) -> None:
    _skip_if_unsupported(world_size, "Iris all-reduce tests")
    port = _get_open_port()
    _spawn_and_collect(_ar_worker_fn, (world_size, port, allocator_type), world_size)


@pytest.mark.parametrize("allocator_type", ["torch", "vmem"])
def test_iris_all_reduce_correctness_world2(allocator_type):
    _run_ar_test(world_size=2, allocator_type=allocator_type)


def test_iris_all_reduce_correctness_world4():
    _run_ar_test(world_size=4, allocator_type=None)


def test_iris_all_reduce_correctness_world8():
    _run_ar_test(world_size=8, allocator_type=None)


def _ar_epoch_worker_fn(rank, world_size, port, path, rows, error_dict):
    try:
        _ar_epoch_worker_main(rank, world_size, port, path, rows)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _ar_epoch_worker_main(rank, world_size, port, path, rows):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    from tokenspeed_kernel.ops.communication.iris import (
        create_iris_state,
        iris_acquire_outputs,
        iris_all_reduce,
        iris_all_reduce_symmetric,
    )

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        state = create_iris_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            staged_max_numel=16 * 7168,
            producer_direct_max_numel=16 * 7168,
            dtype=torch.bfloat16,
            heap_size=None,
            device=device,
        )
        producer_direct = path == "producer_direct"
        if producer_direct:
            local = iris_acquire_outputs(state, ((rows, 7168),))[0]
            flags = state.producer_pull.ready_flags
        else:
            local = torch.empty((rows, 7168), dtype=torch.bfloat16, device=device)
            staged = (
                state.staged_two_shot if path == "staged_two_shot" else state.staged
            )
            assert staged is not None
            flags = staged.ready_flags

        def reduce():
            if producer_direct:
                return iris_all_reduce_symmetric(state, (local,))[0]
            return iris_all_reduce(
                state, local, op=dist.ReduceOp.SUM, safe=True, async_op=False
            )

        local.fill_(rank + 1)
        reduce()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = reduce()
        if path == "staged_one_shot":
            # Guard against the dispatcher silently picking another path.
            before = flags.clone()
            graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            used = math.ceil(rows * 7168 / state.staged.block_size)
            assert torch.equal(flags[:used, rank], before[:used, rank] + 1)

        for replay in (False, True):
            # Include entry and intermediate-stage wraps, plus a normal epoch.
            # Consecutive calls from each preset cross the wrap, so a staging
            # slot reused too early or a wait that passes too early shows up.
            for index, initial_epoch in enumerate((17, 2**31 - 1, -1, 2**31 - 2, -2)):
                torch.cuda.synchronize()
                dist.barrier()
                flags.fill_(initial_epoch)
                local.zero_()
                torch.cuda.synchronize()
                dist.barrier()
                for call in range(3):
                    # A stale flag must not let peers read before this producer.
                    if rank == world_size - 1:
                        time.sleep(0.1)
                    scale = 3 * index + call + 2
                    local.fill_(scale * (rank + 1))
                    if replay:
                        graph.replay()
                        result = captured
                    else:
                        result = reduce()
                    torch.cuda.synchronize()
                    expected = scale * world_size * (world_size + 1) // 2
                    mismatches = [None] * world_size
                    dist.all_gather_object(
                        mismatches, int(torch.count_nonzero(result != expected).item())
                    )
                    assert not any(mismatches), (
                        f"{path=} {rows=} {initial_epoch=} {call=} {replay=}: "
                        f"mismatched elements per rank: {mismatches}"
                    )
                    dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size,path,rows",
    [
        (8, "producer_direct", 1),
        (8, "producer_direct", 16),
        (8, "staged_two_shot", 16),
        (2, "staged_one_shot", 1),
        (2, "staged_one_shot", 16),
    ],
)
def test_iris_all_reduce_epoch_rollover(world_size, path, rows):
    _skip_if_unsupported(world_size, "Iris epoch rollover tests")
    if path != "staged_one_shot" and not current_platform().is_cdna4:
        pytest.skip("Producer-direct Iris reductions require CDNA4")
    _spawn_and_collect(
        _ar_epoch_worker_fn,
        (world_size, _get_open_port(), path, rows),
        world_size,
    )


def _ar_subgroup_worker_fn(rank, world_size, port, error_dict):
    try:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )
        groups = (
            tuple(range(0, world_size, 2)),
            tuple(range(1, world_size, 2)),
        )
        process_groups = tuple(dist.new_group(ranks) for ranks in groups)
        group_index = rank % 2
        group = process_groups[group_index]
        group_rank = groups[group_index].index(rank)

        from tokenspeed_kernel.ops.communication.iris import create_iris_state

        state = create_iris_state(
            group=group,
            rank_in_group=group_rank,
            staged_max_numel=4 * 7,
            producer_direct_max_numel=8 * (7168 + 3584),
            dtype=torch.bfloat16,
            heap_size=None,
            device=device,
        )
        _check_all_reduce(
            state,
            group_rank,
            len(groups[group_index]),
            (4, 7),
            device,
        )
        _check_all_reduce_symmetric_outputs(
            state,
            group_rank,
            len(groups[group_index]),
            ((3, 5), (1, 1)),
            device,
        )
        _check_all_reduce_symmetric_outputs(
            state,
            group_rank,
            len(groups[group_index]),
            ((8, 7168), (8, 3584)),
            device,
        )
    except Exception:
        error_dict[rank] = traceback.format_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_iris_all_reduce_noncontiguous_subgroups():
    world_size = 8
    _skip_if_unsupported(world_size, "Iris subgroup all-reduce tests")
    port = _get_open_port()
    _spawn_and_collect(_ar_subgroup_worker_fn, (world_size, port), world_size)


# ---------------------------------------------------------------------------
# Suite 2: fused allreduce + residual + RMSNorm
# ---------------------------------------------------------------------------


# Token shapes spanning decode (1), short/long prefill (256, 1024), and
# the full ``max_token_num`` (8192) so we exercise both the small-M code
# path and the path that walks the full symmetric heap buffer. Hidden=2880
# is the gpt-oss-120b size we use elsewhere.
_ARRMS_TOKEN_CASES: List[int] = [1, 64, 256, 1024, 8192]
_ARRMS_HIDDEN_DIM = 2880
_ARRMS_EPS = 1e-6


def _arrms_worker_fn(rank, world_size, port, persistent, error_dict):
    try:
        _arrms_worker_main(rank, world_size, port, persistent)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _arrms_worker_main(rank: int, world_size: int, port: int, persistent: bool) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    # NCCL is fine here — iris's heap-base exchange is host-side and works
    # the same over any default group.
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )

    try:
        from tokenspeed_kernel.ops.communication.iris import (
            create_iris_ar_rmsnorm_state,
        )

        max_token_num = max(_ARRMS_TOKEN_CASES)
        state = create_iris_ar_rmsnorm_state(
            group=dist.group.WORLD,
            rank_in_group=rank,
            max_token_num=max_token_num,
            hidden_dim=_ARRMS_HIDDEN_DIM,
            dtype=torch.bfloat16,
            persistent=persistent,
        )

        # Use a fixed RMSNorm weight that is *not* identity, so a bug in
        # the weight load path would fail the test.
        weight = torch.linspace(
            0.5, 1.5, _ARRMS_HIDDEN_DIM, dtype=torch.bfloat16, device=device
        )

        for tokens in _ARRMS_TOKEN_CASES:
            _check_arrms_one(
                state,
                rank=rank,
                world_size=world_size,
                tokens=tokens,
                weight=weight,
                device=device,
            )
    finally:
        dist.destroy_process_group()


def _check_arrms_one(state, rank, world_size, tokens, weight, device) -> None:
    from tokenspeed_kernel.ops.communication.iris import (
        iris_allreduce_residual_rmsnorm,
    )

    # Each rank contributes ``rank + 1``; sum across ranks is therefore
    # ``world_size * (world_size + 1) / 2``. Residual is non-uniform
    # (linspace) so the kernel can't accidentally short-circuit it.
    x = torch.full(
        (tokens, _ARRMS_HIDDEN_DIM), rank + 1, dtype=torch.bfloat16, device=device
    )
    residual = (
        torch.arange(tokens * _ARRMS_HIDDEN_DIM, dtype=torch.float32, device=device)
        .reshape(tokens, _ARRMS_HIDDEN_DIM)
        .mul_(0.001)
        .to(torch.bfloat16)
    )

    norm_out, residual_out = iris_allreduce_residual_rmsnorm(
        state,
        input_tensor=x,
        residual=residual,
        weight=weight,
        eps=_ARRMS_EPS,
    )

    # Reference: do everything in fp32, mirroring the AMD test exactly so
    # tolerance differences only reflect implementation noise, not
    # reference noise.
    reduced = torch.full(
        (tokens, _ARRMS_HIDDEN_DIM),
        world_size * (world_size + 1) // 2,
        dtype=torch.float32,
        device=device,
    )
    ref_residual = reduced + residual.float()
    ref_norm = ref_residual * torch.rsqrt(
        ref_residual.pow(2).mean(dim=-1, keepdim=True) + _ARRMS_EPS
    )
    ref_norm = ref_norm * weight.float()

    torch.testing.assert_close(residual_out.float(), ref_residual, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(norm_out.float(), ref_norm, atol=2e-2, rtol=2e-2)


def _run_arrms_test(world_size: int, persistent: bool) -> None:
    _skip_if_unsupported(world_size, "Iris fused tests")
    port = _get_open_port()
    _spawn_and_collect(_arrms_worker_fn, (world_size, port, persistent), world_size)


@pytest.mark.parametrize("persistent", [False, True], ids=["per_row", "persistent"])
def test_iris_allreduce_residual_rmsnorm_world1(persistent: bool):
    # Single-rank smoke test: exercises the inline-barrier self-signal/wait
    # path (rank sends to itself) and the v1 device_barrier no-op case.
    _run_arrms_test(world_size=1, persistent=persistent)


@pytest.mark.parametrize("persistent", [False, True], ids=["per_row", "persistent"])
def test_iris_allreduce_residual_rmsnorm_world2(persistent: bool):
    _run_arrms_test(world_size=2, persistent=persistent)


@pytest.mark.parametrize("persistent", [False, True], ids=["per_row", "persistent"])
def test_iris_allreduce_residual_rmsnorm_world4(persistent: bool):
    _run_arrms_test(world_size=4, persistent=persistent)


@pytest.mark.parametrize("persistent", [False, True], ids=["per_row", "persistent"])
def test_iris_allreduce_residual_rmsnorm_world8(persistent: bool):
    _run_arrms_test(world_size=8, persistent=persistent)
