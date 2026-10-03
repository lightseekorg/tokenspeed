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

"""``--logprob-order``: the log-softmax behind every returned logprob."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.sampling import vocab_parallel_logprobs

import tokenspeed.runtime.sampling.backends.flashinfer as flashinfer_module
from tokenspeed.runtime.configs.numerics import MEGATRON_VOCAB_BLOCK
from tokenspeed.runtime.execution.context import InputLogprobRows
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.flashinfer import FlashInferSamplingBackend
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.sampling.utils import (
    gather_token_logprobs,
    gather_token_logprobs_torch,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

VOCAB = 2 * MEGATRON_VOCAB_BLOCK
POOL = 4


def _logits(rows: int) -> torch.Tensor:
    return torch.randn(rows, VOCAB, generator=torch.Generator().manual_seed(1)) * 3


def test_gather_dispatches_on_the_order():
    logits = _logits(3)
    tokens = torch.tensor([5, 60000, 123], dtype=torch.int32)
    torch_order = gather_token_logprobs(logits, tokens, logprob_order="torch")
    megatron = gather_token_logprobs(logits, tokens, logprob_order="megatron")
    assert torch.equal(
        torch_order,
        torch.log_softmax(logits, -1).gather(1, tokens.long()[:, None]).squeeze(1),
    )
    assert torch.equal(
        megatron,
        vocab_parallel_logprobs(logits, tokens, vocab_block=MEGATRON_VOCAB_BLOCK),
    )
    torch.testing.assert_close(megatron, torch_order, rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="logprob_order"):
        gather_token_logprobs(logits, tokens, logprob_order="apex")


def test_flashinfer_backend_reports_megatron_logprobs(monkeypatch):
    def fake_gumbel(logits, req_pool_indices, *pools, min_p_pool=None, **_):
        out = pools[-1]
        out[: logits.shape[0]].copy_(torch.tensor([7, 60001], dtype=torch.int32))
        return out[: logits.shape[0]]

    def fake_gather(req_pool_indices, **pools):
        order = ("temperature", "top_k", "top_p", "min_p", "seed", "offsets")
        return tuple(
            None if pools.get(n) is None else pools[n][req_pool_indices] for n in order
        )

    monkeypatch.setattr(flashinfer_module, "_FUSED_TOPK_TOPP_AVAILABLE", False)
    monkeypatch.setattr(
        flashinfer_module, "gumbel_sample_from_pools_generic", fake_gumbel
    )
    monkeypatch.setattr(flashinfer_module, "gather_and_expand_scalars", fake_gather)
    backend = FlashInferSamplingBackend(
        SamplingBackendConfig(
            enable_speculative_sampling=False,
            sampling_stream="per-request",
            logprob_order="megatron",
            enable_output_logprobs=True,
            max_bs=2,
            max_draft_tokens_per_req=1,
            max_req_pool_size=POOL,
            vocab_size=VOCAB,
            device="cpu",
        )
    )
    params = []
    for rid in ("a", "b"):
        sp = SamplingParams(temperature=1.0, top_k=-1, top_p=1.0)
        sp.resolve_seed(rid)
        sp.normalize(None)
        params.append(sp)
    backend.prepare_step(["a", "b"], [1, 2], params)
    logits = _logits(2)
    output = LogitsProcessorOutput(next_token_logits=logits)
    sampled, _ = backend.sample(
        output,
        SamplingBatchInfo(
            req_pool_indices=torch.tensor([1, 2]),
            valid_cache_lengths=torch.zeros(POOL + 1, dtype=torch.int32),
            vocab_size=VOCAB,
            device="cpu",
        ),
    )
    assert sampled.tolist() == [7, 60001]
    assert torch.equal(
        output.next_token_logprobs,
        vocab_parallel_logprobs(logits, sampled, vocab_block=MEGATRON_VOCAB_BLOCK),
    )


def _processor(order: str, monkeypatch) -> LogitsProcessor:
    monkeypatch.setitem(global_server_args_dict, "logprob_order", order)
    return LogitsProcessor(
        SimpleNamespace(model_type="test", vocab_size=VOCAB, final_logit_softcapping=None)
    )


def _prompt_rows(rows: list[int], targets: list[int], *, num_input_rows: int):
    return InputLogprobRows(
        rows=torch.tensor(rows, dtype=torch.int64),
        targets=torch.tensor(targets, dtype=torch.int64),
        slots=torch.zeros(len(rows), dtype=torch.int64),
        num_input_rows=num_input_rows,
        chunk_tokens=2,
    )


def test_prompt_logprobs_follow_the_order(monkeypatch):
    # Prompt rows take the sampler's gather in the launch's order, so a
    # token's prompt and output logprobs are the same number under either.
    hidden = torch.randn(4, 8, generator=torch.Generator().manual_seed(2))
    weight = torch.randn(VOCAB, 8, generator=torch.Generator().manual_seed(3))
    lm_head = SimpleNamespace(weight=weight)
    rows, targets = [0, 1, 2], [3, 40000, 65535]
    logits = (hidden @ weight.T)[rows]
    ids = torch.tensor(targets)

    def run(order: str) -> torch.Tensor:
        metadata = LogitsMetadata(
            forward_mode=ForwardMode.EXTEND,
            gather_ids=torch.tensor([3]),
            input_logprob_rows=_prompt_rows(rows, targets, num_input_rows=4),
        )
        return _processor(order, monkeypatch)(
            input_ids=None,
            hidden_states=hidden,
            lm_head=lm_head,
            logits_metadata=metadata,
        ).input_token_logprobs

    torch_order = run("torch")
    assert torch.equal(torch_order, gather_token_logprobs_torch(logits, ids))
    megatron = run("megatron")
    assert torch.equal(
        megatron, vocab_parallel_logprobs(logits, ids, vocab_block=MEGATRON_VOCAB_BLOCK)
    )
    torch.testing.assert_close(megatron, torch_order, rtol=1e-5, atol=1e-5)


def test_processor_refuses_an_unknown_order(monkeypatch):
    with pytest.raises(ValueError, match="logprob_order"):
        _processor("apex", monkeypatch)
