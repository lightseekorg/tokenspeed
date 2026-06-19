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

"""Batch-DP logits-to-verify parity tests."""

from __future__ import annotations

import socket
import traceback
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from tokenspeed.runtime.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.flashinfer import FlashInferSamplingBackend
from tokenspeed.runtime.sampling.dp_sampling_config import (
    DpSamplingRuntimeConfig,
    DpSamplingTopology,
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


class _StubLMHead(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = weight


@dataclass
class _StubConfig:
    vocab_size: int
    final_logit_softcapping: float | None = None
    model_type: str = "test_dp_sampling_e2e"


def _make_hidden_states(bs: int, n: int, hidden: int, *, dtype, device, seed: int):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.empty(bs * n, hidden, dtype=dtype, device=device).normal_(generator=g)


def _make_lm_head_weight(vocab: int, hidden: int, *, dtype, device, seed: int):
    g = torch.Generator(device=device)
    g.manual_seed(seed + 11)
    return torch.empty(vocab, hidden, dtype=dtype, device=device).normal_(generator=g)


def _make_candidates(bs: int, n: int, vocab: int, *, device, seed: int):
    g = torch.Generator(device=device)
    g.manual_seed(seed + 23)
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
    g.manual_seed(seed + 47)
    backend._coins_buf[:bs, :n].uniform_(1e-6, 1.0, generator=g)
    backend._final_coins_buf[:bs].uniform_(1e-6, 1.0, generator=g)


def _build_backend(*, max_bs: int, max_n: int, vocab: int, device, group):
    cfg = SamplingBackendConfig(
        enable_output_logprobs=False,
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


def _build_processor(
    *,
    config: _StubConfig,
    tp_rank: int,
    tp_size: int,
    tp_group: tuple[int, ...],
):
    return LogitsProcessor(
        config=config,
        skip_all_gather=False,
        tp_rank=tp_rank,
        tp_size=tp_size,
        tp_group=tp_group,
    )


def _build_metadata():
    return LogitsMetadata(
        forward_mode=ForwardMode.DECODE,
        capture_hidden_mode=CaptureHiddenMode.NULL,
    )


def _test_dp_chain_matches_legacy(
    rank,
    world_size,
    device,
    group,
    *,
    bs: int,
    n: int,
    vocab: int,
    hidden: int,
    is_all_greedy: bool,
    dtype,
):
    tp_size = world_size
    pad_bs = ((bs + tp_size - 1) // tp_size) * tp_size
    assert vocab % tp_size == 0, "vocab must be divisible by tp for the test"
    v_local = vocab // tp_size

    full_weight = _make_lm_head_weight(
        vocab, hidden, dtype=dtype, device=device, seed=4096
    )
    weight_shard = full_weight[rank * v_local : (rank + 1) * v_local].clone()
    lm_head = _StubLMHead(weight_shard)

    config = _StubConfig(vocab_size=vocab)
    processor = _build_processor(
        config=config, tp_rank=rank, tp_size=tp_size, tp_group=group
    )
    processor.configure_dp_logits_layout(
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

    backend = _build_backend(
        max_bs=max(bs, pad_bs),
        max_n=max(n, 1),
        vocab=vocab,
        device=device,
        group=group,
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

    hidden_states = _make_hidden_states(
        bs, n, hidden, dtype=dtype, device=device, seed=2024
    )
    candidates = _make_candidates(bs, n, vocab, device=device, seed=2024)
    req_pool_indices = torch.arange(bs, dtype=torch.int64, device=device)

    legacy_meta = _build_metadata()
    legacy_logits = processor._get_logits(hidden_states.clone(), lm_head, legacy_meta)
    assert legacy_logits.shape == (
        bs * n,
        vocab,
    ), f"legacy logits {legacy_logits.shape}, expected {(bs*n, vocab)}"

    legacy_info = SamplingBatchInfo(
        is_all_greedy=is_all_greedy,
        vocab_size=vocab,
        req_pool_indices=req_pool_indices,
        device=str(device),
    )
    legacy_out = LogitsProcessorOutput(next_token_logits=legacy_logits)
    _seed_coins(backend, bs=bs, n=n, seed=2024)
    legacy_predict, legacy_accept_length = backend.verify(
        legacy_out, legacy_info, candidates
    )
    legacy_predict = legacy_predict.clone()
    legacy_accept_length = legacy_accept_length.clone()

    dp_meta = _build_metadata()
    dp_plan = LogitsLayoutPlan(
        effective_bs=bs,
        bucket_bs=pad_bs,
        tp_size=tp_size,
        num_tokens_per_req=n,
    )
    dp_logits = processor._get_logits(
        hidden_states.clone(), lm_head, dp_meta, plan=dp_plan
    )
    reqs_per_rank = pad_bs // tp_size
    assert dp_logits.shape == (
        reqs_per_rank * n,
        vocab,
    ), f"dp logits {dp_logits.shape}, expected {(reqs_per_rank*n, vocab)}"

    dp_info = SamplingBatchInfo(
        is_all_greedy=is_all_greedy,
        vocab_size=vocab,
        req_pool_indices=req_pool_indices,
        device=str(device),
    )
    dp_out = LogitsProcessorOutput(
        next_token_logits=dp_logits,
        logits_layout_plan=dp_plan,
    )
    _seed_coins(backend, bs=bs, n=n, seed=2024)
    dp_predict, dp_accept_length = backend.verify(dp_out, dp_info, candidates)

    # Phantom rows consume neutral pool values and are not part of the result.
    torch.testing.assert_close(
        dp_predict,
        legacy_predict,
        rtol=0,
        atol=0,
        msg="DP predict diverged from legacy",
    )
    torch.testing.assert_close(
        dp_accept_length,
        legacy_accept_length,
        rtol=0,
        atol=0,
        msg="DP accept_length diverged from legacy",
    )


WORLD_SIZES = [
    pytest.param(2, id="tp2"),
]

SHAPES = [
    pytest.param(2, 2, id="bs2_n2"),
    pytest.param(9, 2, id="bs9_n2"),
]


class TestDPSamplingLogitsVerify:
    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("bs,n", SHAPES)
    def test_stochastic(self, world_size, bs, n):
        _run(
            world_size,
            _test_dp_chain_matches_legacy,
            bs=bs,
            n=n,
            vocab=256,
            hidden=64,
            is_all_greedy=False,
            dtype=torch.bfloat16,
        )

    @pytest.mark.parametrize("world_size", WORLD_SIZES)
    @pytest.mark.parametrize("bs,n", SHAPES)
    def test_greedy(self, world_size, bs, n):
        _run(
            world_size,
            _test_dp_chain_matches_legacy,
            bs=bs,
            n=n,
            vocab=256,
            hidden=64,
            is_all_greedy=True,
            dtype=torch.bfloat16,
        )
