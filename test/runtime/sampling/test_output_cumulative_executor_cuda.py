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

"""The executor's sampling step records cumulative probabilities.

Drives ``ModelExecutor._run_sampling`` -- the production call site -- with a
real sampling backend and real runtime states: the flag comes from the
backend's config, the layout from the forward context, and the diagnostic
scores the tokens the drafter's rewrite leaves.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires an NVIDIA CUDA GPU",
)

VOCAB = 64
POOL = 8


def _executor(*, enabled, drafter):
    from tokenspeed.runtime.execution.model_executor import ModelExecutor
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
    from tokenspeed.runtime.sampling.backends.greedy import GreedySamplingBackend

    executor = ModelExecutor.__new__(ModelExecutor)
    executor.sampling_backend = GreedySamplingBackend(
        SamplingBackendConfig(
            synthetic_acceptance_length=None,
            enable_speculative_sampling=False,
            enable_block_rejection=False,
            sampling_stream="per-request",
            logprob_order="torch",
            enable_output_logprobs=True,
            enable_output_cumulative_probs=enabled,
            max_bs=4,
            max_draft_tokens_per_req=1,
            max_req_pool_size=POOL,
            vocab_size=VOCAB,
            device="cuda",
        )
    )
    executor.tree_spec = None
    executor.drafter = drafter
    executor.runtime_states = RuntimeStates(POOL, VOCAB, 1, device="cuda")
    # The input buffers' rows this step reads.
    executor.input_buffers = SimpleNamespace(
        state_write_req_pool_indices_buf=torch.tensor([3, 5], device="cuda"),
        input_lengths_buf=torch.tensor([4, 6], dtype=torch.int32, device="cuda"),
        extend_prompt_lens_buf=torch.tensor([4, 6], dtype=torch.int32, device="cuda"),
        state_write_padding_pool_index=POOL,
    )
    return executor


class RewritingDrafter:
    """Rewrites its verify outputs in place, as the drafter hook allows."""

    def __init__(self, tokens):
        self.tokens = tokens

    def rewrite_verify_outputs(self, ctx, output_tokens, accept_lengths):
        output_tokens.copy_(self.tokens)


def _reference(logits, tokens):
    probabilities = logits.double().softmax(-1)
    order = torch.argsort(logits, descending=True, stable=True, dim=-1)
    ranked = probabilities.gather(-1, order)
    left = ranked.cumsum(-1) - ranked
    rank = (order == tokens.long()[:, None]).long().argmax(-1)
    return left.gather(-1, rank[:, None]).squeeze(-1)


@pytest.mark.parametrize("enabled", [False, True])
def test_the_sampling_step_scores_the_committed_tokens(enabled):
    from tokenspeed.runtime.execution.model_executor import ModelExecutor
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.logits_processor_output import (
        LogitsProcessorOutput,
    )
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo

    committed = torch.tensor([9, 17], dtype=torch.int32, device="cuda")
    executor = _executor(enabled=enabled, drafter=RewritingDrafter(committed))
    logits = torch.randn(
        2, VOCAB, generator=torch.Generator(device="cuda").manual_seed(5), device="cuda"
    )
    output = LogitsProcessorOutput(next_token_logits=logits.clone())
    info = SamplingBatchInfo(
        device="cuda", req_pool_indices=torch.tensor([3, 5], device="cuda")
    )
    ctx = SimpleNamespace(
        bs=2,
        num_extends=2,
        output_layout=ForwardOutputLayout(2, 2, 0, 1),
        sampled_top_k=0,
    )
    tokens, lengths = ModelExecutor._run_sampling(executor, output, info, ctx)
    torch.cuda.synchronize()
    assert torch.equal(tokens, committed)
    assert lengths.tolist() == [1, 1]
    if not enabled:
        assert output.next_token_cumulative_probs is None
        return
    torch.testing.assert_close(
        output.next_token_cumulative_probs.double(),
        _reference(logits, committed),
        rtol=1e-5,
        atol=1e-6,
    )
