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

"""FlashInfer verify parity for Batch-DP."""

import socket
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.sampling.dp_sampling_config import (
    DpSamplingRuntimeConfig,
    DpSamplingTopology,
    slice_dp_vocab_mask,
)
from tokenspeed.runtime.sampling.logits_layout import LogitsLayoutPlan
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo


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


def _make_logits(bs: int, n: int, vocab: int, *, dtype, device, seed: int):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.empty(bs * n, vocab, dtype=dtype, device=device).normal_(generator=g)


def _make_candidates(bs: int, n: int, vocab: int, *, device, seed: int):
    g = torch.Generator(device=device)
    g.manual_seed(seed + 1)
    return torch.randint(
        low=0, high=vocab, size=(bs, n), dtype=torch.int32, device=device, generator=g
    )


def _seed_pool_scalars(
    backend, *, bs: int, temperature: float, top_k: int, top_p: float
):
    backend._temperature_pool[: bs + 1].fill_(temperature)
    backend._top_k_pool[: bs + 1].fill_(top_k)
    backend._top_p_pool[: bs + 1].fill_(top_p)


def _seed_coins(backend, *, bs: int, n: int, seed: int):
    g = torch.Generator(device=backend._coins_buf.device)
    g.manual_seed(seed + 7)
    backend._coins_buf[:bs, :n].uniform_(1e-6, 1.0, generator=g)
    backend._final_coins_buf[:bs].uniform_(1e-6, 1.0, generator=g)


def _build_backend(
    *,
    max_bs: int,
    max_n: int,
    vocab: int,
    device,
    group,
    enable_output_logprobs: bool = False,
):
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.backends.flashinfer import (
        FlashInferSamplingBackend,
    )

    cfg = SamplingBackendConfig(
        enable_output_logprobs=enable_output_logprobs,
        max_bs=max_bs,
        max_draft_tokens_per_req=max_n,
        max_req_pool_size=max(max_bs, 4),
        vocab_size=vocab,
        device=device,
        random_seed=123,
        tp_group=group,
        enable_tp_sync=False,
    )
    return FlashInferSamplingBackend(cfg)


def _test_verify_dp_matches_today(
    rank,
    world_size,
    device,
    group,
    *,
    bs: int,
    n: int,
    vocab: int,
    is_all_greedy: bool,
    dtype,
    enable_output_logprobs: bool = False,
    forbid_global_logprob_writer: bool = False,
):
    tp_size = world_size
    pad_bs = ((bs + tp_size - 1) // tp_size) * tp_size
    reqs_per_rank = pad_bs // tp_size

    backend = _build_backend(
        max_bs=max(bs, pad_bs),
        max_n=max(n, 1),
        vocab=vocab,
        device=device,
        group=group,
        enable_output_logprobs=enable_output_logprobs,
    )
    backend.configure_dp_sampling(
        DpSamplingRuntimeConfig(
            enabled=True,
            vocab_size=vocab,
            max_bucket_bs=pad_bs,
            min_bs=1,
            num_tokens_per_req=n,
            topology=DpSamplingTopology(
                tp_rank=rank,
                tp_size=tp_size,
                tp_group=group,
                skip_all_gather=False,
            ),
            device=device,
        )
    )
    _seed_pool_scalars(backend, bs=bs, temperature=1.0, top_k=32, top_p=0.9)

    full_logits = _make_logits(bs, n, vocab, dtype=dtype, device=device, seed=2024)
    candidates = _make_candidates(bs, n, vocab, device=device, seed=2024)
    req_pool_indices = torch.arange(bs, dtype=torch.int64, device=device)

    sampling_info_full_batch = SamplingBatchInfo(
        is_all_greedy=is_all_greedy,
        vocab_size=vocab,
        req_pool_indices=req_pool_indices,
        device=str(device),
    )
    sampling_info_dp = SamplingBatchInfo(
        is_all_greedy=is_all_greedy,
        vocab_size=vocab,
        req_pool_indices=req_pool_indices,
        device=str(device),
    )

    class _StubOutput:
        pass

    _seed_coins(backend, bs=bs, n=n, seed=2024)
    full_batch_in = _StubOutput()
    full_batch_in.next_token_logits = full_logits.clone()
    predict_full, accept_length_full = backend.verify(
        full_batch_in, sampling_info_full_batch, candidates
    )
    predict_full = predict_full.clone()
    accept_length_full = accept_length_full.clone()
    logprobs_full = (
        full_batch_in.next_token_logprobs.clone() if enable_output_logprobs else None
    )

    _seed_coins(backend, bs=bs, n=n, seed=2024)
    full_logits_padded = torch.nn.functional.pad(
        full_logits.view(bs, n, vocab), (0, 0, 0, 0, 0, pad_bs - bs)
    ).view(pad_bs * n, vocab)
    local_logits = full_logits_padded[
        rank * reqs_per_rank * n : (rank + 1) * reqs_per_rank * n
    ].clone()
    dp_in = _StubOutput()
    dp_in.next_token_logits = local_logits
    dp_in.logits_layout_plan = LogitsLayoutPlan(
        effective_bs=bs,
        bucket_bs=pad_bs,
        tp_size=tp_size,
        num_tokens_per_req=n,
    )
    if forbid_global_logprob_writer:
        import tokenspeed.runtime.sampling.backends.flashinfer as flashinfer_backend

        original_writer = getattr(flashinfer_backend, "write_output_logprobs", None)
        if original_writer is not None:

            def _fail_global_writer(*args, **kwargs):
                raise AssertionError("DP verify must not call write_output_logprobs")

            flashinfer_backend.write_output_logprobs = _fail_global_writer
            try:
                predict_dp, accept_length_dp = backend.verify(
                    dp_in, sampling_info_dp, candidates
                )
            finally:
                flashinfer_backend.write_output_logprobs = original_writer
        else:
            predict_dp, accept_length_dp = backend.verify(
                dp_in, sampling_info_dp, candidates
            )
    else:
        predict_dp, accept_length_dp = backend.verify(
            dp_in, sampling_info_dp, candidates
        )

    torch.testing.assert_close(predict_dp, predict_full, rtol=0, atol=0)
    torch.testing.assert_close(accept_length_dp, accept_length_full, rtol=0, atol=0)
    if enable_output_logprobs:
        torch.testing.assert_close(
            dp_in.next_token_logprobs,
            logprobs_full,
            rtol=1e-5,
            atol=1e-5,
        )


def test_dp_vocab_mask_slices_by_request_shard():
    full_bs = 5
    pad_bs = 6
    n = 3
    mask_words = 4
    vocab_mask = torch.arange(full_bs * n * mask_words, dtype=torch.int32).view(
        full_bs * n, mask_words
    )

    rank0 = slice_dp_vocab_mask(
        vocab_mask,
        full_bs=full_bs,
        pad_bs=pad_bs,
        num_tokens_per_req=n,
        shard=slice(0, 3),
    )
    torch.testing.assert_close(rank0, vocab_mask[: 3 * n], rtol=0, atol=0)

    rank1 = slice_dp_vocab_mask(
        vocab_mask,
        full_bs=full_bs,
        pad_bs=pad_bs,
        num_tokens_per_req=n,
        shard=slice(3, 6),
    )
    expected_rank1 = torch.cat(
        [
            vocab_mask[3 * n :],
            torch.full((n, mask_words), -1, dtype=torch.int32),
        ]
    )
    torch.testing.assert_close(rank1, expected_rank1, rtol=0, atol=0)


WORLD_SIZES = [
    pytest.param(2, id="tp2"),
]

SHAPES = [
    pytest.param(8, 2, id="bs8_n2"),
    pytest.param(9, 4, id="bs9_n4"),
]


class TestFlashInferVerifyDP:
    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("bs,n", SHAPES)
    @pytest.mark.parametrize(
        "dtype",
        [pytest.param(torch.float32, id="fp32")],
    )
    def test_stochastic_path(self, world_size, bs, n, dtype):
        _run(
            world_size,
            _test_verify_dp_matches_today,
            bs=bs,
            n=n,
            vocab=256,
            is_all_greedy=False,
            dtype=dtype,
        )

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("bs,n", SHAPES)
    def test_greedy_path(self, world_size, bs, n):
        _run(
            world_size,
            _test_verify_dp_matches_today,
            bs=bs,
            n=n,
            vocab=256,
            is_all_greedy=True,
            dtype=torch.float32,
        )

    @pytest.mark.parametrize("world_size", [pytest.param(2, id="tp2")])
    @pytest.mark.parametrize("bs,n", [pytest.param(9, 2, id="bs9_n2")])
    def test_greedy_path_output_logprobs(self, world_size, bs, n):
        _run(
            world_size,
            _test_verify_dp_matches_today,
            bs=bs,
            n=n,
            vocab=256,
            is_all_greedy=True,
            dtype=torch.float32,
            enable_output_logprobs=True,
            forbid_global_logprob_writer=True,
        )
