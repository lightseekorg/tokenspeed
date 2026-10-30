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

"""TritonSamplingBackend.warmup_sample_kernels compiles every kernel sample()
can pick, so eager prefill sampling compiles nothing while serving."""

import contextlib
import os
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")

from tokenspeed_kernel.ops.sampling.triton import (  # noqa: E402
    generic,
    gumbel,
    topk_topp,
    topp,
)

from tokenspeed.runtime.layers.logits_processor import (  # noqa: E402
    LogitsProcessorOutput,
)
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.triton import (  # noqa: E402
    TritonSamplingBackend,
)
from tokenspeed.runtime.sampling.sampling_batch_info import (  # noqa: E402
    SamplingBatchInfo,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402

POOL = 8
MAX_BS = 4
SAMPLE_KERNELS = (
    generic._gumbel_sample_generic_pool_kernel,
    gumbel._gumbel_sample_pool_stage1_kernel,
    gumbel._gumbel_sample_stage2_kernel,
    gumbel._gumbel_sample_compact_pool_kernel,
    topk_topp._top_k_top_p_candidates_stage1_kernel,
    topk_topp._top_k_top_p_gumbel_stage2_kernel,
    topp._top_p_parallel_stage1_kernel,
    topp._top_p_parallel_stage2_kernel,
    topp._top_p_parallel_stage3_kernel,
    topp._top_p_parallel_stage4_kernel,
    topp._top_p_parallel_repair_kernel,
)
# (top_k, top_p) per request: no filter, top-p, top-k at both pads, top-k
# with top-p at both pads, and a mix (the generic route).
BATCHES = (
    [(-1, 1.0)],
    [(-1, 0.9)],
    [(20, 1.0)],
    [(64, 1.0), (1, 1.0)],
    [(100, 1.0)],
    [(20, 0.9), (50, 0.8), (64, 0.95)],
    [(200, 0.9)],
    [(20, 1.0), (-1, 0.9)],
)


def _params(top_k: int, top_p: float, seed: int) -> SamplingParams:
    sp = SamplingParams(temperature=0.8, top_k=top_k, top_p=top_p, seed=seed)
    sp.normalize(None)
    return sp


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("vocab", [1024, 40960])
def test_sample_compiles_nothing_after_warmup(vocab):
    backend = TritonSamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            sampling_stream="batch",
            logprob_order="torch",
            max_bs=MAX_BS,
            max_draft_tokens_per_req=4,
            max_req_pool_size=POOL,
            vocab_size=vocab,
            device="cuda",
        )
    )
    logits = torch.randn((MAX_BS, vocab), device="cuda")
    backend.warmup_sample_kernels(logits)

    with contextlib.ExitStack() as stack:
        compiles = [
            stack.enter_context(
                patch.object(kernel, "_do_compile", wraps=kernel._do_compile)
            )
            for kernel in SAMPLE_KERNELS
        ]
        for i, batch in enumerate(BATCHES):
            rows = len(batch)
            slots = list(range(1, rows + 1))
            backend.prepare_step(
                request_ids=[f"r{i}-{j}" for j in range(rows)],
                request_pool_indices=slots,
                sampling_params_list=[
                    _params(top_k, top_p, seed=i * 10 + j)
                    for j, (top_k, top_p) in enumerate(batch)
                ],
            )
            backend.sample(
                LogitsProcessorOutput(next_token_logits=logits[:rows].clone()),
                SamplingBatchInfo(
                    req_pool_indices=torch.tensor(
                        slots, dtype=torch.int32, device="cuda"
                    ),
                    valid_cache_lengths=torch.zeros(
                        (POOL + 1,), dtype=torch.int32, device="cuda"
                    ),
                    vocab_size=vocab,
                    device="cuda",
                ),
            )
        torch.cuda.synchronize()
    compiled = {
        kernel.fn.__name__: calls.call_count
        for kernel, calls in zip(SAMPLE_KERNELS, compiles, strict=True)
        if calls.call_count
    }
    assert not compiled, compiled
