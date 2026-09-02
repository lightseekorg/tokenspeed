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

"""MIXED-batch sampling seams: sample-then-verify over shared buffers.

A MIXED round samples its prefill rows and verifies its decode rows through
the SAME backend, back to back (``ModelExecutor._run_sampling``). Two seams
this file pins:

* Both calls land outputs in the backend's packed output region, so the
  prefill results must be snapshotted before verify() overwrites the buffer
  prefix — the executor clones them; sharing the raw views corrupts the
  prefill rows' tokens and accept lengths.
* The per-step coin buffers are filled in the step's full
  prefill-then-decode batch order, but verify() receives only the decode
  suffix — it must read each row's own coins at ``batch_row_offset``, not
  the prefill rows' draws from the buffer head.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.sampling.sampling_params import SamplingParams

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

VOCAB = 256
POOL = 8
MAX_BS = 6
MAX_N = 2  # == n below: verify slices [:bs, :n] and the chain kernel
# requires contiguous coins, exactly like production (n == spec width).


def test_slice_accumulates_batch_row_offset():
    info = SamplingBatchInfo(
        req_pool_indices=torch.arange(6),
        device="cpu",
    )
    assert info.batch_row_offset == 0
    tail = info[2:]
    assert tail.batch_row_offset == 2
    # Nested slicing keeps accumulating from the original batch.
    assert tail[1:].batch_row_offset == 3
    # A no-start slice adds nothing.
    assert info[:4].batch_row_offset == 0


@requires_cuda
def test_verify_reads_decode_rows_own_coins():
    """verify() over a MIXED round's decode suffix must consume the decode
    rows' coin draws, not the prefill rows' (buffer head)."""
    from tokenspeed.runtime.sampling.backends.flashinfer import (
        FlashInferSamplingBackend,
    )

    config = SamplingBackendConfig(
        max_bs=MAX_BS,
        max_draft_tokens_per_req=MAX_N,
        max_req_pool_size=POOL,
        vocab_size=VOCAB,
        device="cuda",
        enable_output_logprobs=False,
        enable_nan_detection=False,
    )
    backend = FlashInferSamplingBackend(config)

    num_extends, num_decodes, n = 2, 2, 2
    bs = num_extends + num_decodes
    rids = [f"r{i}" for i in range(bs)]
    pool = list(range(bs))
    sp = [
        SamplingParams(temperature=0.8, top_k=50, top_p=0.9, seed=7 + i)
        for i in range(bs)
    ]
    for p in sp:
        p.verify(VOCAB)

    backend.prepare_step(
        request_ids=rids,
        request_pool_indices=pool,
        sampling_params_list=sp,
        num_tokens_per_req=n,
    )
    # The full-batch coin fill: rows 0..1 are prefill draws, rows 2..3 decode.
    decode_coins = backend._coins_buf[num_extends:bs, :n].clone()
    prefill_coins = backend._coins_buf[:num_decodes, :n].clone()
    assert not torch.equal(decode_coins, prefill_coins)

    info = SamplingBatchInfo(
        req_pool_indices=torch.tensor(pool, device="cuda"),
        valid_cache_lengths=torch.zeros(POOL + 1, dtype=torch.int32, device="cuda"),
        device="cuda",
    )
    decode_info = info[num_extends:]
    assert decode_info.batch_row_offset == num_extends

    seen = {}
    import tokenspeed.runtime.sampling.backends.flashinfer as fi

    original = fi.chain_speculative_sampling_target_only

    def probe(**kwargs):
        seen["coins"] = kwargs["uniform_samples"].clone()
        return original(**kwargs)

    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput

    logits = torch.randn(num_decodes * n, VOCAB, device="cuda")
    candidates = torch.randint(
        0, VOCAB, (num_decodes, n), device="cuda", dtype=torch.int64
    )
    fi.chain_speculative_sampling_target_only = probe
    try:
        backend.verify(
            LogitsProcessorOutput(next_token_logits=logits),
            decode_info,
            candidates,
        )
    finally:
        fi.chain_speculative_sampling_target_only = original

    torch.testing.assert_close(seen["coins"], decode_coins)


@requires_cuda
def test_mixed_round_preserves_prefill_outputs():
    """The executor's mixed arm snapshots sample() outputs before verify()
    reuses the packed output buffers."""
    from tokenspeed.runtime.execution.model_executor import ModelExecutor
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.backends.flashinfer import (
        FlashInferSamplingBackend,
    )

    config = SamplingBackendConfig(
        max_bs=MAX_BS,
        max_draft_tokens_per_req=MAX_N,
        max_req_pool_size=POOL,
        vocab_size=VOCAB,
        device="cuda",
        enable_output_logprobs=False,
        enable_nan_detection=False,
    )
    backend = FlashInferSamplingBackend(config)

    num_extends, num_decodes, n = 2, 2, 2
    bs = num_extends + num_decodes
    rids = [f"m{i}" for i in range(bs)]
    pool = list(range(bs))
    sp = [SamplingParams(temperature=0.0) for _ in range(bs)]
    for p in sp:
        p.verify(VOCAB)
    backend.prepare_step(
        request_ids=rids,
        request_pool_indices=pool,
        sampling_params_list=sp,
        num_tokens_per_req=n,
    )

    # Distinct argmax per row so corruption is visible.
    logits = torch.full((num_extends + num_decodes * n, VOCAB), -10.0, device="cuda")
    for row in range(logits.shape[0]):
        logits[row, row + 1] = 10.0

    info = SamplingBatchInfo(
        req_pool_indices=torch.tensor(pool, device="cuda"),
        valid_cache_lengths=torch.zeros(POOL + 1, dtype=torch.int32, device="cuda"),
        device="cuda",
    )
    candidates = torch.zeros(num_decodes, n, device="cuda", dtype=torch.int64)

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.num_extends = num_extends
    ctx.bs = bs
    ctx.decode_input_ids = None

    executor = ModelExecutor.__new__(ModelExecutor)
    executor.sampling_backend = backend
    executor._apply_force_single_token_verify = lambda accept, off, cnt, ids: accept

    out_tokens, out_accept = ModelExecutor._run_sampling(
        executor,
        LogitsProcessorOutput(next_token_logits=logits),
        info,
        ctx,
        candidates=candidates,
    )
    torch.cuda.synchronize()
    # Prefill rows keep their own argmaxes (rows 0..1 -> tokens 1, 2);
    # without the snapshot they would show the decode rows' verify output.
    assert out_tokens[:num_extends].tolist() == [1, 2]
    assert out_accept[:num_extends].tolist() == [1, 1]


def test_explicit_top_k_one_pins_seed():
    sp = SamplingParams(temperature=0.7, top_k=1, seed=None)
    sp.verify(VOCAB)
    assert sp.seed == 0

    greedy = SamplingParams(temperature=0.0)
    greedy.verify(VOCAB)
    assert greedy.seed == 0 and greedy.top_k == 1

    stochastic = SamplingParams(temperature=0.7, top_k=50)
    stochastic.verify(VOCAB)
    assert stochastic.seed != 0 or stochastic.seed is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
