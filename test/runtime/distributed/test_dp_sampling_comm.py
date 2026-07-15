"""Tests for ``DpSamplingComm``."""

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.distributed.comm_backend import get_global_backend
from tokenspeed.runtime.distributed.dp_sampling_comm import (
    DpSamplingComm,
    _onesided_available,
    _resolve_backend,
)
from tokenspeed.runtime.distributed.dp_sampling_swap import swap_batch_vocab
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _worker_main(rank, world_size, port, test_fn, error_dict, args):
    try:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://localhost:{port}",
            rank=rank,
            world_size=world_size,
        )

        group = tuple(range(world_size))
        pg_manager.init_process_group(group)

        test_fn(rank=rank, world_size=world_size, device=device, group=group, **args)

        dist.destroy_process_group()
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _run(world_size, test_fn, **args):
    if world_size > torch.cuda.device_count():
        pytest.skip(f"Need {world_size} GPUs, have {torch.cuda.device_count()}")
    port = _get_open_port()
    error_dict = mp.Manager().dict()
    mp.spawn(
        _worker_main,
        args=(world_size, port, test_fn, error_dict, args),
        nprocs=world_size,
        join=True,
    )
    if error_dict:
        raise RuntimeError("\n".join(f"Rank {r}: {e}" for r, e in error_dict.items()))


def _onesided_available_for_test(group) -> bool:
    try:
        return _onesided_available(group)
    except Exception:
        return False


def test_backend_resolution_ignores_process_environment(monkeypatch):
    monkeypatch.setenv("TOKENSPEED_DP_SAMPLING_BACKEND", "bogus")
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.dp_sampling_comm._onesided_available",
        lambda group: False,
    )

    assert _resolve_backend("auto", (0, 1)) == "nccl"
    assert _resolve_backend("nccl", (0, 1)) == "nccl"


def test_backend_resolution_rejects_invalid_explicit_value():
    with pytest.raises(ValueError, match="backend='bogus'"):
        _resolve_backend("bogus", (0, 1))


def _build_comm(rank, world_size, group, *, pad_bs, n, vocab, dtype, backend):
    return DpSamplingComm(
        tp_size=world_size,
        rank=rank,
        group=group,
        max_pad_bs=pad_bs,
        num_tokens_per_req=n,
        vocab_size=vocab,
        logits_dtype=dtype,
        backend=backend,
    )


def _ground_truth_full_logits(pad_bs, n, vocab, *, dtype, device):
    return torch.arange(pad_bs * n * vocab, dtype=dtype, device=device).view(
        pad_bs * n, vocab
    )


def _test_swap_parity_with_free_function(
    rank, world_size, device, group, *, pad_bs, n, vocab, dtype, backend
):
    tp = world_size
    v_local = vocab // tp

    full = _ground_truth_full_logits(pad_bs, n, vocab, dtype=dtype, device=device)
    local_logits = full[:, rank * v_local : (rank + 1) * v_local].contiguous()

    comm = _build_comm(
        rank,
        world_size,
        group,
        pad_bs=pad_bs,
        n=n,
        vocab=vocab,
        dtype=dtype,
        backend=backend,
    )
    assert comm.backend == backend
    assert comm.fast_path_enabled is (backend == "onesided")

    out_class = comm.swap_batch_vocab(local_logits, pad_bs=pad_bs)

    out_free = swap_batch_vocab(
        local_logits,
        tp_size=tp,
        pad_bs=pad_bs,
        num_tokens_per_req=n,
        vocab_size=vocab,
        group=group,
    )

    assert out_class.shape == out_free.shape
    torch.testing.assert_close(out_class, out_free)


def _test_gather_verify_outputs_correctness(
    rank, world_size, device, group, *, pad_bs, n, backend
):
    tp = world_size
    reqs_per_rank = pad_bs // tp

    comm = _build_comm(
        rank,
        world_size,
        group,
        pad_bs=pad_bs,
        n=n,
        vocab=tp * 4,
        dtype=torch.bfloat16,
        backend=backend,
    )

    predict_local = torch.arange(
        rank * reqs_per_rank * n,
        (rank + 1) * reqs_per_rank * n,
        dtype=torch.int32,
        device=device,
    ).view(reqs_per_rank, n)
    accept_index_local = (predict_local * 2 + 1).contiguous()
    accept_length_local = torch.arange(
        rank * reqs_per_rank,
        (rank + 1) * reqs_per_rank,
        dtype=torch.int32,
        device=device,
    )

    predict_full, accept_index_full, accept_length_full = comm.gather_verify_outputs(
        predict_local,
        accept_index_local,
        accept_length_local,
        pad_bs=pad_bs,
    )

    expected_predict = torch.arange(
        0, pad_bs * n, dtype=torch.int32, device=device
    ).view(pad_bs, n)
    expected_accept_index = expected_predict * 2 + 1
    expected_accept_length = torch.arange(0, pad_bs, dtype=torch.int32, device=device)

    torch.testing.assert_close(predict_full, expected_predict)
    torch.testing.assert_close(accept_index_full, expected_accept_index)
    torch.testing.assert_close(accept_length_full, expected_accept_length)


def _test_gather_persistent_buffer_reuse(
    rank, world_size, device, group, *, pad_bs, n, backend
):
    tp = world_size
    reqs_per_rank = pad_bs // tp

    comm = _build_comm(
        rank,
        world_size,
        group,
        pad_bs=pad_bs,
        n=n,
        vocab=tp * 4,
        dtype=torch.bfloat16,
        backend=backend,
    )

    predict_local = torch.zeros(reqs_per_rank, n, dtype=torch.int32, device=device)
    accept_index_local = torch.zeros(reqs_per_rank, n, dtype=torch.int32, device=device)
    accept_length_local = torch.zeros(reqs_per_rank, dtype=torch.int32, device=device)

    p1, ai1, al1 = comm.gather_verify_outputs(
        predict_local, accept_index_local, accept_length_local, pad_bs=pad_bs
    )
    p2, ai2, al2 = comm.gather_verify_outputs(
        predict_local, accept_index_local, accept_length_local, pad_bs=pad_bs
    )

    assert p1.data_ptr() == p2.data_ptr()
    assert ai1.data_ptr() == ai2.data_ptr()
    assert al1.data_ptr() == al2.data_ptr()


def _test_gather_verify_logprobs_correctness(
    rank, world_size, device, group, *, pad_bs, n
):
    tp = world_size
    reqs_per_rank = pad_bs // tp

    comm = _build_comm(
        rank,
        world_size,
        group,
        pad_bs=pad_bs,
        n=n,
        vocab=tp * 4,
        dtype=torch.float32,
        backend="nccl",
    )

    logprobs_local = (
        torch.arange(
            rank * reqs_per_rank * n,
            (rank + 1) * reqs_per_rank * n,
            dtype=torch.float32,
            device=device,
        ).view(reqs_per_rank, n)
        / 100.0
    )

    logprobs_full = comm.gather_verify_logprobs(logprobs_local, pad_bs=pad_bs)
    expected = (
        torch.arange(0, pad_bs * n, dtype=torch.float32, device=device).view(pad_bs, n)
        / 100.0
    )
    torch.testing.assert_close(logprobs_full, expected)


def _test_swap_and_gather_cuda_graph_replay(
    rank, world_size, device, group, *, pad_bs, n, vocab, backend
):
    tp = world_size
    reqs_per_rank = pad_bs // tp
    v_local = vocab // tp

    comm = _build_comm(
        rank,
        world_size,
        group,
        pad_bs=pad_bs,
        n=n,
        vocab=vocab,
        dtype=torch.float32,
        backend=backend,
    )

    local_logits_buf = torch.empty(
        pad_bs * n, v_local, dtype=torch.float32, device=device
    )
    predict_local_buf = torch.empty(reqs_per_rank, n, dtype=torch.int32, device=device)
    accept_index_local_buf = torch.empty(
        reqs_per_rank, n, dtype=torch.int32, device=device
    )
    accept_length_local_buf = torch.empty(
        reqs_per_rank, dtype=torch.int32, device=device
    )

    def _fill_inputs(step: int):
        full = _ground_truth_full_logits(
            pad_bs, n, vocab, dtype=torch.float32, device=device
        )
        full = full + step * 1000.0
        local_logits_buf.copy_(
            full[:, rank * v_local : (rank + 1) * v_local].contiguous()
        )
        predict_local_buf.copy_(
            torch.arange(
                rank * reqs_per_rank * n,
                (rank + 1) * reqs_per_rank * n,
                dtype=torch.int32,
                device=device,
            ).view(reqs_per_rank, n)
            + step
        )
        accept_index_local_buf.copy_(predict_local_buf * 2)
        accept_length_local_buf.copy_(
            torch.arange(
                rank * reqs_per_rank,
                (rank + 1) * reqs_per_rank,
                dtype=torch.int32,
                device=device,
            )
            + step
        )

    def _run_one_step():
        swapped = comm.swap_batch_vocab(local_logits_buf, pad_bs=pad_bs)
        p, ai, al = comm.gather_verify_outputs(
            predict_local_buf,
            accept_index_local_buf,
            accept_length_local_buf,
            pad_bs=pad_bs,
        )
        return swapped, p, ai, al

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _fill_inputs(step=0)
        for _ in range(3):
            _run_one_step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize(device)
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    _fill_inputs(step=0)
    with torch.cuda.graph(graph, stream=side):
        swapped_captured, p_captured, ai_captured, al_captured = _run_one_step()
    torch.cuda.synchronize(device)
    dist.barrier()

    for step in range(1, 6):
        _fill_inputs(step=step)
        graph.replay()
        torch.cuda.synchronize(device)

        graph_swapped = swapped_captured.clone()
        graph_predict = p_captured.clone()
        graph_accept_index = ai_captured.clone()
        graph_accept_length = al_captured.clone()
        dist.barrier()

        _fill_inputs(step=step)
        ref_swapped, ref_predict, ref_accept_index, ref_accept_length = _run_one_step()
        torch.cuda.synchronize(device)

        torch.testing.assert_close(graph_swapped, ref_swapped)
        torch.testing.assert_close(graph_predict, ref_predict)
        torch.testing.assert_close(graph_accept_index, ref_accept_index)
        torch.testing.assert_close(graph_accept_length, ref_accept_length)
        dist.barrier()


class _CountingNcclBackend:
    def __init__(self, inner):
        self._inner = inner
        self.all_gather_calls = 0

    def all_gather_into_tensor(self, output, input, group):
        self.all_gather_calls += 1
        return self._inner.all_gather_into_tensor(output, input, group)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _test_nccl_single_allgather(rank, world_size, device, group, *, pad_bs, n):
    counter = _CountingNcclBackend(get_global_backend())

    comm = DpSamplingComm(
        tp_size=world_size,
        rank=rank,
        group=group,
        max_pad_bs=pad_bs,
        num_tokens_per_req=n,
        vocab_size=world_size * 4,
        logits_dtype=torch.bfloat16,
        backend="nccl",
        fallback_comm_backend=counter,
    )

    tp = world_size
    reqs_per_rank = pad_bs // tp
    predict_local = torch.arange(
        rank * reqs_per_rank * n,
        (rank + 1) * reqs_per_rank * n,
        dtype=torch.int32,
        device=device,
    ).view(reqs_per_rank, n)
    accept_index_local = (predict_local * 5 + 3).contiguous()
    accept_length_local = torch.arange(
        rank * reqs_per_rank,
        (rank + 1) * reqs_per_rank,
        dtype=torch.int32,
        device=device,
    )

    predict_full, accept_index_full, accept_length_full = comm.gather_verify_outputs(
        predict_local,
        accept_index_local,
        accept_length_local,
        pad_bs=pad_bs,
    )

    assert (
        counter.all_gather_calls == 1
    ), f"expected 1 all_gather call, got {counter.all_gather_calls}"

    expected_predict = torch.arange(
        0, pad_bs * n, dtype=torch.int32, device=device
    ).view(pad_bs, n)
    expected_accept_index = expected_predict * 5 + 3
    expected_accept_length = torch.arange(0, pad_bs, dtype=torch.int32, device=device)
    torch.testing.assert_close(predict_full, expected_predict)
    torch.testing.assert_close(accept_index_full, expected_accept_index)
    torch.testing.assert_close(accept_length_full, expected_accept_length)


def _test_onesided_matches_nccl(
    rank, world_size, device, group, *, pad_bs, n, vocab, dtype
):
    tp = world_size
    v_local = vocab // tp
    reqs_per_rank = pad_bs // tp

    full = _ground_truth_full_logits(pad_bs, n, vocab, dtype=dtype, device=device)
    local_logits = full[:, rank * v_local : (rank + 1) * v_local].contiguous()

    nccl_comm = _build_comm(
        rank,
        world_size,
        group,
        pad_bs=pad_bs,
        n=n,
        vocab=vocab,
        dtype=dtype,
        backend="nccl",
    )
    onesided_comm = _build_comm(
        rank,
        world_size,
        group,
        pad_bs=pad_bs,
        n=n,
        vocab=vocab,
        dtype=dtype,
        backend="onesided",
    )

    torch.testing.assert_close(
        onesided_comm.swap_batch_vocab(local_logits, pad_bs=pad_bs),
        nccl_comm.swap_batch_vocab(local_logits, pad_bs=pad_bs),
    )

    predict_local = torch.arange(
        rank * reqs_per_rank * n,
        (rank + 1) * reqs_per_rank * n,
        dtype=torch.int32,
        device=device,
    ).view(reqs_per_rank, n)
    accept_index_local = (predict_local * 3 + 7).contiguous()
    accept_length_local = torch.arange(
        rank * reqs_per_rank,
        (rank + 1) * reqs_per_rank,
        dtype=torch.int32,
        device=device,
    )

    onesided_outputs = onesided_comm.gather_verify_outputs(
        predict_local, accept_index_local, accept_length_local, pad_bs=pad_bs
    )
    nccl_outputs = nccl_comm.gather_verify_outputs(
        predict_local, accept_index_local, accept_length_local, pad_bs=pad_bs
    )
    for actual, expected in zip(onesided_outputs, nccl_outputs, strict=True):
        torch.testing.assert_close(actual, expected)


def _test_onesided_gather_lazy_init_without_swap(
    rank, world_size, device, group, *, pad_bs, n, dtype
):
    tp = world_size
    reqs_per_rank = pad_bs // tp

    nccl_comm = _build_comm(
        rank,
        world_size,
        group,
        pad_bs=pad_bs,
        n=n,
        vocab=tp * 4,
        dtype=dtype,
        backend="nccl",
    )
    onesided_comm = _build_comm(
        rank,
        world_size,
        group,
        pad_bs=pad_bs,
        n=n,
        vocab=tp * 4,
        dtype=None,
        backend="onesided",
    )
    assert onesided_comm.fast_path_enabled

    predict_local = torch.arange(
        rank * reqs_per_rank * n,
        (rank + 1) * reqs_per_rank * n,
        dtype=torch.int32,
        device=device,
    ).view(reqs_per_rank, n)
    accept_index_local = (predict_local * 7 + 11).contiguous()
    accept_length_local = torch.arange(
        rank * reqs_per_rank,
        (rank + 1) * reqs_per_rank,
        dtype=torch.int32,
        device=device,
    )

    onesided_comm.prepare_verify_outputs(dtype)
    onesided_outputs = onesided_comm.gather_verify_outputs(
        predict_local, accept_index_local, accept_length_local, pad_bs=pad_bs
    )
    nccl_outputs = nccl_comm.gather_verify_outputs(
        predict_local, accept_index_local, accept_length_local, pad_bs=pad_bs
    )
    for actual, expected in zip(onesided_outputs, nccl_outputs, strict=True):
        torch.testing.assert_close(actual, expected)


def _test_onesided_prepare_after_swap_keeps_comm_dtype(
    rank, world_size, device, group, *, pad_bs, n, vocab, dtype
):
    tp = world_size
    v_local = vocab // tp
    reqs_per_rank = pad_bs // tp

    full = _ground_truth_full_logits(pad_bs, n, vocab, dtype=dtype, device=device)
    local_logits = full[:, rank * v_local : (rank + 1) * v_local].contiguous()

    nccl_comm = _build_comm(
        rank,
        world_size,
        group,
        pad_bs=pad_bs,
        n=n,
        vocab=vocab,
        dtype=dtype,
        backend="nccl",
    )
    onesided_comm = _build_comm(
        rank,
        world_size,
        group,
        pad_bs=pad_bs,
        n=n,
        vocab=vocab,
        dtype=None,
        backend="onesided",
    )

    torch.testing.assert_close(
        onesided_comm.swap_batch_vocab(local_logits, pad_bs=pad_bs),
        nccl_comm.swap_batch_vocab(local_logits, pad_bs=pad_bs),
    )

    # LogitsProcessor converts sampled logits to fp32 after the DP swap. Verify
    # gather should reuse the existing one-sided state instead of re-preparing it
    # with that post-conversion dtype.
    onesided_comm.prepare_verify_outputs(torch.float32)

    predict_local = torch.arange(
        rank * reqs_per_rank * n,
        (rank + 1) * reqs_per_rank * n,
        dtype=torch.int32,
        device=device,
    ).view(reqs_per_rank, n)
    accept_index_local = (predict_local * 5 + 13).contiguous()
    accept_length_local = torch.arange(
        rank * reqs_per_rank,
        (rank + 1) * reqs_per_rank,
        dtype=torch.int32,
        device=device,
    )

    onesided_outputs = onesided_comm.gather_verify_outputs(
        predict_local, accept_index_local, accept_length_local, pad_bs=pad_bs
    )
    nccl_outputs = nccl_comm.gather_verify_outputs(
        predict_local, accept_index_local, accept_length_local, pad_bs=pad_bs
    )
    for actual, expected in zip(onesided_outputs, nccl_outputs, strict=True):
        torch.testing.assert_close(actual, expected)


WORLD_SIZES = [
    pytest.param(2, id="tp2"),
]

SHAPES = [
    pytest.param(8, 1, 64, id="sample_pad_bs8"),
    pytest.param(8, 4, 64, id="spec_pad_bs8_n4"),
]

DTYPES = [
    pytest.param(torch.float32, id="fp32"),
    pytest.param(torch.bfloat16, id="bf16"),
]

BACKENDS = [
    pytest.param("nccl", id="nccl"),
    pytest.param("onesided", id="onesided"),
]


class TestDpSamplingComm:

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("pad_bs,n,vocab", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_swap_parity_with_free_function(
        self, world_size, pad_bs, n, vocab, dtype, backend
    ):
        if pad_bs % world_size != 0 or vocab % world_size != 0:
            pytest.skip("shape not divisible by tp")
        if backend == "onesided" and not _onesided_available_for_test(
            tuple(range(world_size))
        ):
            pytest.skip("one-sided dp-sampling backend is not available")
        _run(
            world_size,
            _test_swap_parity_with_free_function,
            pad_bs=pad_bs,
            n=n,
            vocab=vocab,
            dtype=dtype,
            backend=backend,
        )

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("pad_bs,n", [(8, 1), (8, 4)])
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_gather_verify_outputs_correctness(self, world_size, pad_bs, n, backend):
        if pad_bs % world_size != 0:
            pytest.skip("pad_bs not divisible by tp")
        if backend == "onesided" and not _onesided_available_for_test(
            tuple(range(world_size))
        ):
            pytest.skip("one-sided dp-sampling backend is not available")
        _run(
            world_size,
            _test_gather_verify_outputs_correctness,
            pad_bs=pad_bs,
            n=n,
            backend=backend,
        )

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_gather_persistent_buffer_reuse(self, world_size, backend):
        if backend == "onesided" and not _onesided_available_for_test(
            tuple(range(world_size))
        ):
            pytest.skip("one-sided dp-sampling backend is not available")
        _run(
            world_size,
            _test_gather_persistent_buffer_reuse,
            pad_bs=8,
            n=2,
            backend=backend,
        )

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("pad_bs,n", [(8, 1), (8, 4)])
    def test_gather_verify_logprobs_correctness(self, world_size, pad_bs, n):
        if pad_bs % world_size != 0:
            pytest.skip("pad_bs not divisible by tp")
        _run(
            world_size,
            _test_gather_verify_logprobs_correctness,
            pad_bs=pad_bs,
            n=n,
        )

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_swap_and_gather_cuda_graph_replay(self, world_size, backend):
        if backend == "onesided" and not _onesided_available_for_test(
            tuple(range(world_size))
        ):
            pytest.skip("one-sided dp-sampling backend is not available")
        _run(
            world_size,
            _test_swap_and_gather_cuda_graph_replay,
            pad_bs=8,
            n=2,
            vocab=64,
            backend=backend,
        )

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("pad_bs,n", [(8, 1), (8, 4)])
    def test_nccl_single_allgather(self, world_size, pad_bs, n):
        if pad_bs % world_size != 0:
            pytest.skip("pad_bs not divisible by tp")
        _run(
            world_size,
            _test_nccl_single_allgather,
            pad_bs=pad_bs,
            n=n,
        )

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("pad_bs,n,vocab", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_onesided_matches_nccl(self, world_size, pad_bs, n, vocab, dtype):
        if pad_bs % world_size != 0 or vocab % world_size != 0:
            pytest.skip("shape not divisible by tp")
        if not _onesided_available_for_test(tuple(range(world_size))):
            pytest.skip("one-sided dp-sampling backend is not available")
        _run(
            world_size,
            _test_onesided_matches_nccl,
            pad_bs=pad_bs,
            n=n,
            vocab=vocab,
            dtype=dtype,
        )

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("pad_bs,n", [(8, 1), (8, 4)])
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_onesided_gather_lazy_init_without_swap(self, world_size, pad_bs, n, dtype):
        if pad_bs % world_size != 0:
            pytest.skip("pad_bs not divisible by tp")
        if not _onesided_available_for_test(tuple(range(world_size))):
            pytest.skip("one-sided dp-sampling backend is not available")
        _run(
            world_size,
            _test_onesided_gather_lazy_init_without_swap,
            pad_bs=pad_bs,
            n=n,
            dtype=dtype,
        )

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("pad_bs,n,vocab", SHAPES)
    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(torch.bfloat16, id="bf16"),
            pytest.param(torch.float16, id="fp16"),
        ],
    )
    def test_onesided_prepare_after_swap_keeps_comm_dtype(
        self, world_size, pad_bs, n, vocab, dtype
    ):
        if pad_bs % world_size != 0 or vocab % world_size != 0:
            pytest.skip("shape not divisible by tp")
        if not _onesided_available_for_test(tuple(range(world_size))):
            pytest.skip("one-sided dp-sampling backend is not available")
        _run(
            world_size,
            _test_onesided_prepare_after_swap_keeps_comm_dtype,
            pad_bs=pad_bs,
            n=n,
            vocab=vocab,
            dtype=dtype,
        )
