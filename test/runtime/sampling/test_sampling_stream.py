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

"""``--sampling-stream per-request`` on the FlashInfer sampling backends.

flashinfer's ``*_sampling_from_probs`` kernels seed curand with the batch
row, so a request's T>0 draw depends on its co-batch. Under ``per-request``
the backends route sampled rows through the Gumbel-max pool kernel keyed by
``(seed_pool[pool_idx], valid_cache_lengths[pool_idx])``. The CPU tests pin
the route selection and the pool plumbing with the kernels mocked; the CUDA
test is the contract itself: the same request samples the same token alone
and inside a batch.
"""

from __future__ import annotations

import pytest
import torch

import tokenspeed.runtime.sampling.backends.flashinfer as flashinfer_module
import tokenspeed.runtime.sampling.backends.flashinfer_full as flashinfer_full_module
from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
from tokenspeed.runtime.sampling.backends.base import SamplingBackendConfig
from tokenspeed.runtime.sampling.backends.flashinfer import FlashInferSamplingBackend
from tokenspeed.runtime.sampling.backends.flashinfer_full import (
    FlashInferFullSamplingBackend,
)
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
from tokenspeed.runtime.sampling.sampling_params import SamplingParams

VOCAB = 256
POOL = 8
MAX_BS = 4


def _config(sampling_stream: str, device: str) -> SamplingBackendConfig:
    return SamplingBackendConfig(
        enable_speculative_sampling=False,
        sampling_stream=sampling_stream,
        logprob_order="torch",
        max_bs=MAX_BS,
        max_draft_tokens_per_req=2,
        max_req_pool_size=POOL,
        vocab_size=VOCAB,
        device=device,
    )


def _sp(rid: str, **overrides) -> SamplingParams:
    params = dict(temperature=1.0, top_k=-1, top_p=1.0, min_p=0.0)
    params.update(overrides)
    sp = SamplingParams(**params)
    sp.resolve_seed(rid)
    sp.normalize(None)
    return sp


def _info(pool_indices: list[int], device: str) -> SamplingBatchInfo:
    return SamplingBatchInfo(
        req_pool_indices=torch.tensor(pool_indices, dtype=torch.int64, device=device),
        valid_cache_lengths=torch.arange(
            100, 100 + POOL + 1, dtype=torch.int32, device=device
        ),
        vocab_size=VOCAB,
        device=device,
    )


@pytest.fixture
def cpu_kernels(monkeypatch: pytest.MonkeyPatch):
    """Mock every Triton/flashinfer kernel the sample() paths touch."""
    calls: dict[str, list] = {"gumbel": [], "flashinfer": [], "gather": []}

    def fake_gumbel(
        logits,
        req_pool_indices,
        temperature_pool,
        top_k_pool,
        top_p_pool,
        seed_pool,
        offsets_pool,
        out,
        *,
        min_p_pool=None,
        num_tokens_per_req=1,
    ):
        calls["gumbel"].append(
            dict(
                logits=logits,
                req_pool_indices=req_pool_indices,
                temperature_pool=temperature_pool,
                top_k_pool=top_k_pool,
                top_p_pool=top_p_pool,
                seed_pool=seed_pool,
                offsets_pool=offsets_pool,
                out=out,
                min_p_pool=min_p_pool,
            )
        )
        out[: logits.shape[0]].copy_(torch.arange(logits.shape[0], dtype=torch.int32))
        return out[: logits.shape[0]]

    def fake_gather(req_pool_indices, **pools):
        calls["gather"].append(req_pool_indices)
        order = ("temperature", "top_k", "top_p", "min_p", "seed", "offsets")
        outs = []
        for name in order:
            pool = pools.get(name)
            outs.append(None if pool is None else pool[req_pool_indices])
        return tuple(outs)

    def fake_flashinfer_sample(probs, *args, **kwargs):
        calls["flashinfer"].append(kwargs)
        return torch.zeros(probs.shape[0], dtype=torch.int32)

    for module in (flashinfer_module, flashinfer_full_module):
        monkeypatch.setattr(module, "_FUSED_TOPK_TOPP_AVAILABLE", False)
        monkeypatch.setattr(module, "gather_and_expand_scalars", fake_gather)
        monkeypatch.setattr(
            module, "softmax", lambda logits, temperature: logits.softmax(-1)
        )
    monkeypatch.setattr(
        flashinfer_module, "gumbel_sample_from_pools_generic", fake_gumbel
    )
    monkeypatch.setattr(
        flashinfer_module, "top_k_top_p_sampling_from_probs", fake_flashinfer_sample
    )
    monkeypatch.setattr(
        flashinfer_full_module, "min_p_sampling_from_probs", fake_flashinfer_sample
    )
    monkeypatch.setattr(
        flashinfer_full_module, "top_k_renorm_prob", lambda probs, top_ks: probs
    )
    monkeypatch.setattr(
        flashinfer_full_module,
        "top_p_renorm_prob",
        lambda probs, top_ps, is_deterministic: probs,
    )
    return calls


def test_per_request_route_reads_the_request_pools(cpu_kernels):
    backend = FlashInferSamplingBackend(_config("per-request", "cpu"))
    sp_a = _sp("a", temperature=0.7, top_k=40, top_p=0.9, seed=11)
    sp_b = _sp("b", temperature=1.3, top_k=-1, top_p=1.0, seed=22)
    backend.prepare_step(
        request_ids=["a", "b"],
        request_pool_indices=[3, 5],
        sampling_params_list=[sp_a, sp_b],
    )
    info = _info([3, 5], "cpu")
    logits = torch.randn(2, VOCAB)
    sampled, lengths = backend.sample(
        LogitsProcessorOutput(next_token_logits=logits), info
    )

    assert cpu_kernels["flashinfer"] == []
    # The per-request route reads the pools directly: the expanded per-row
    # scalars of the batch route are never built.
    assert cpu_kernels["gather"] == []
    (call,) = cpu_kernels["gumbel"]
    # Pool indices reach the kernel as int32, the pools as the backend's own
    # buffers, and the offsets as the step's pool-indexed cache lengths.
    assert call["req_pool_indices"].dtype == torch.int32
    assert call["req_pool_indices"].tolist() == [3, 5]
    assert call["seed_pool"] is backend._seed_pool
    assert call["seed_pool"][3].item() == 11 and call["seed_pool"][5].item() == 22
    assert call["temperature_pool"][3].item() == pytest.approx(0.7)
    assert call["top_k_pool"][3].item() == 40
    assert call["top_p_pool"][3].item() == pytest.approx(0.9)
    assert call["offsets_pool"] is info.valid_cache_lengths
    assert call["min_p_pool"] is None
    assert call["logits"] is logits
    assert sampled.tolist() == [0, 1]
    assert lengths.tolist() == [1, 1]


def test_per_request_route_without_cache_lengths_uses_zero_offsets(cpu_kernels):
    backend = FlashInferSamplingBackend(_config("per-request", "cpu"))
    backend.prepare_step(["a"], [1], [_sp("a")])
    info = SamplingBatchInfo(
        req_pool_indices=torch.tensor([1]), vocab_size=VOCAB, device="cpu"
    )
    backend.sample(LogitsProcessorOutput(next_token_logits=torch.randn(1, VOCAB)), info)
    (call,) = cpu_kernels["gumbel"]
    assert call["offsets_pool"] is backend._zero_offsets_pool


def test_per_request_route_feeds_the_greedy_overlay_each_rows_top_k(
    cpu_kernels, monkeypatch
):
    # Under a bitwise envelope greedy rows take the canonical lowest-index
    # argmax; the per-request route must still hand the overlay each row's
    # top-k (from the pool, not from the skipped scalar gather).
    monkeypatch.setitem(
        flashinfer_module.global_server_args_dict, "numerics", "rl-bitwise"
    )
    backend = FlashInferSamplingBackend(_config("per-request", "cpu"))
    greedy = _sp("g", temperature=0.0, top_k=1)
    backend.prepare_step(["g", "s"], [2, 6], [greedy, _sp("s", top_k=-1)])
    logits = torch.zeros(2, VOCAB)
    logits[:, 3] = logits[:, 9] = 5.0  # an exact tie; canonical is id 3
    sampled, _ = backend.sample(
        LogitsProcessorOutput(next_token_logits=logits), _info([2, 6], "cpu")
    )
    # The mocked pool kernel returned [0, 1]; only the greedy row is overlaid.
    assert sampled.tolist() == [3, 1]


def test_batch_stream_keeps_the_flashinfer_kernel(cpu_kernels):
    backend = FlashInferSamplingBackend(_config("batch", "cpu"))
    backend.prepare_step(["a"], [2], [_sp("a")])
    backend.sample(
        LogitsProcessorOutput(next_token_logits=torch.randn(1, VOCAB)),
        _info([2], "cpu"),
    )
    assert cpu_kernels["gumbel"] == []
    assert len(cpu_kernels["gather"]) == 1
    (call,) = cpu_kernels["flashinfer"]
    assert call["seed"].tolist() == [backend._seed_pool[2].item()]


def test_full_backend_passes_its_min_p_pool(cpu_kernels):
    backend = FlashInferFullSamplingBackend(_config("per-request", "cpu"))
    backend.prepare_step(["a"], [4], [_sp("a", min_p=0.05)])
    backend.sample(
        LogitsProcessorOutput(next_token_logits=torch.randn(1, VOCAB)),
        _info([4], "cpu"),
    )
    assert cpu_kernels["flashinfer"] == []
    (call,) = cpu_kernels["gumbel"]
    assert call["min_p_pool"] is backend._min_p_pool
    assert call["min_p_pool"][4].item() == pytest.approx(0.05)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("top_k", [-1, 50])
def test_per_request_stream_is_batch_invariant(top_k: int):
    """The same request draws the same token alone and inside a batch."""
    torch.manual_seed(0)
    device = "cuda"
    rid = "the_request"
    sp = _sp(rid, temperature=1.0, top_k=top_k, top_p=0.95, seed=1234)
    row = torch.randn(1, VOCAB, device=device)
    others = [_sp(f"other_{i}", seed=100 + i) for i in range(MAX_BS - 1)]
    other_rows = torch.randn(MAX_BS - 1, VOCAB, device=device)

    solo = FlashInferSamplingBackend(_config("per-request", device))
    solo.prepare_step([rid], [2], [sp])
    solo_token, _ = solo.sample(
        LogitsProcessorOutput(next_token_logits=row.clone()), _info([2], device)
    )

    packed = FlashInferSamplingBackend(_config("per-request", device))
    # Same request on the same pool slot, now second in a batch of four.
    packed.prepare_step(
        ["other_0", rid, "other_1", "other_2"],
        [0, 2, 1, 3],
        [others[0], sp, others[1], others[2]],
    )
    logits = torch.cat([other_rows[:1], row, other_rows[1:]], dim=0)
    packed_tokens, _ = packed.sample(
        LogitsProcessorOutput(next_token_logits=logits.clone()),
        _info([0, 2, 1, 3], device),
    )
    assert packed_tokens[1].item() == solo_token[0].item()
