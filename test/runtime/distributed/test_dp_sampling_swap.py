"""Tests for ``swap_batch_vocab``."""

import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


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

        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg_manager,
        )

        group = tuple(range(world_size))
        pg_manager.init_process_group(group)

        test_fn(rank=rank, world_size=world_size, device=device, group=group, **args)

        dist.destroy_process_group()
    except Exception:
        import traceback

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


def _ground_truth_full(pad_bs: int, n: int, vocab: int, *, dtype, device):
    return torch.arange(pad_bs * n * vocab, dtype=dtype, device=device).view(
        pad_bs * n, vocab
    )


def _test_swap_matches_reference(
    rank, world_size, device, group, *, pad_bs, n, vocab, dtype
):
    from tokenspeed.runtime.distributed.dp_sampling_swap import swap_batch_vocab

    tp = world_size
    v_local = vocab // tp
    reqs_per_rank = pad_bs // tp

    full = _ground_truth_full(pad_bs, n, vocab, dtype=dtype, device=device)

    local_logits = full[:, rank * v_local : (rank + 1) * v_local].contiguous()

    out = swap_batch_vocab(
        local_logits,
        tp_size=tp,
        pad_bs=pad_bs,
        num_tokens_per_req=n,
        vocab_size=vocab,
        rank=rank,
        group=group,
    )

    expected = full[rank * reqs_per_rank * n : (rank + 1) * reqs_per_rank * n].contiguous()
    assert tuple(out.shape) == tuple(
        expected.shape
    ), f"shape mismatch: got {tuple(out.shape)} expected {tuple(expected.shape)}"
    torch.testing.assert_close(out, expected)


def _test_swap_chain_safety(
    rank, world_size, device, group, *, pad_bs, n, vocab, dtype
):
    from tokenspeed.runtime.distributed.dp_sampling_swap import swap_batch_vocab

    tp = world_size
    v_local = vocab // tp
    reqs_per_rank = pad_bs // tp

    full = torch.empty(pad_bs * n, vocab, dtype=dtype, device=device)
    for req in range(pad_bs):
        for d in range(n):
            base = req * 10_000 + d * 100
            full[req * n + d] = torch.arange(vocab, dtype=dtype, device=device) + base
    local_logits = full[:, rank * v_local : (rank + 1) * v_local].contiguous()

    out = swap_batch_vocab(
        local_logits,
        tp_size=tp,
        pad_bs=pad_bs,
        num_tokens_per_req=n,
        vocab_size=vocab,
        rank=rank,
        group=group,
    )

    for local_req in range(reqs_per_rank):
        global_req = rank * reqs_per_rank + local_req
        for d in range(n):
            row = out[local_req * n + d]
            expected_first = global_req * 10_000 + d * 100
            assert int(row[0].item()) == expected_first, (
                f"rank={rank} local_req={local_req} d={d} got row[0]={int(row[0].item())}"
                f" expected {expected_first}"
            )
            assert int(row[-1].item()) == expected_first + (vocab - 1)


WORLD_SIZES = [
    pytest.param(2, id="tp2"),
    pytest.param(4, id="tp4"),
]

SHAPES = [
    pytest.param(8, 1, 64, id="sample_pad_bs8"),
    pytest.param(8, 4, 64, id="spec_pad_bs8_n4"),
    pytest.param(12, 3, 96, id="spec_pad_bs12_n3"),
]

DTYPES = [
    pytest.param(torch.float32, id="fp32"),
    pytest.param(torch.bfloat16, id="bf16"),
]


class TestDPSamplingSwap:

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("pad_bs,n,vocab", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_swap_matches_reference(self, world_size, pad_bs, n, vocab, dtype):
        if pad_bs % world_size != 0:
            pytest.skip(f"pad_bs={pad_bs} not divisible by tp={world_size}")
        if vocab % world_size != 0:
            pytest.skip(f"vocab={vocab} not divisible by tp={world_size}")
        _run(
            world_size,
            _test_swap_matches_reference,
            pad_bs=pad_bs,
            n=n,
            vocab=vocab,
            dtype=dtype,
        )

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("pad_bs,n,vocab", SHAPES)
    def test_swap_chain_safety(self, world_size, pad_bs, n, vocab):
        if pad_bs % world_size != 0:
            pytest.skip(f"pad_bs={pad_bs} not divisible by tp={world_size}")
        if vocab % world_size != 0:
            pytest.skip(f"vocab={vocab} not divisible by tp={world_size}")
        _run(
            world_size,
            _test_swap_chain_safety,
            pad_bs=pad_bs,
            n=n,
            vocab=vocab,
            dtype=torch.float32,
        )
